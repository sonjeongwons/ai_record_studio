"""
AI Voice Studio - Backend Server
FastAPI 서버: 파일 업로드, RunPod GPU 학습/변환, 프로젝트 관리
"""

import os
import sys
import json
import time
import shutil
import hashlib
import sqlite3
import base64
import asyncio
import threading
import webbrowser
import uuid
import subprocess
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import uvicorn
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정 (PyInstaller .exe 모드 + 개발 모드 자동 감지)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FROZEN = getattr(sys, 'frozen', False)

if FROZEN:
    # .exe 실행: 정적 파일은 번들 내부, 사용자 데이터는 .exe 옆 폴더
    APP_DIR = Path(sys._MEIPASS)
    DATA_DIR = Path(sys.executable).parent / "VoiceStudio_Data"
else:
    # 개발 모드: 모든 파일이 소스 디렉토리에
    APP_DIR = Path(__file__).parent
    DATA_DIR = Path(__file__).parent

BASE_DIR = APP_DIR  # 하위 호환
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "output"
CHUNK_DIR = DATA_DIR / "chunks"           # 청크 업로드 임시 디렉토리
PREPROCESSED_DIR = DATA_DIR / "preprocessed"  # 전처리 결과 세그먼트 저장
DB_PATH = DATA_DIR / "studio.db"
CONFIG_PATH = DATA_DIR / "config.json"

# 청크 업로드 제한: 최대 2GB
MAX_CHUNK_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

DATA_DIR.mkdir(parents=True, exist_ok=True)
for d in [UPLOAD_DIR, MODEL_DIR, OUTPUT_DIR, CHUNK_DIR, PREPROCESSED_DIR]:
    d.mkdir(exist_ok=True)

# 청크 업로드 상태 추적 (메모리 내)
chunk_uploads: dict = {}

# 전처리 작업별 원본 파일 ID 추적 (완료 시 preprocessed=1 마킹용)
preprocess_file_map: dict[str, list[int]] = {}

# RunPod payload 제한: 오디오 데이터 최대 7MB (base64 → ~10MB JSON)
MAX_RUNPOD_AUDIO_BYTES = 7 * 1024 * 1024
SPLIT_SEGMENT_SECONDS = 180  # 3분 단위 분할


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FFmpeg 경로 (번들 → 시스템 PATH 순서로 탐색)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_ffmpeg_path() -> Optional[str]:
    """번들된 ffmpeg.exe 우선, 없으면 시스템 PATH 탐색. 없으면 None."""
    # 1) .exe 번들 내부 (PyInstaller)
    if FROZEN:
        bundled = Path(sys._MEIPASS) / "tools" / "ffmpeg.exe"
        if bundled.exists():
            return str(bundled)

    # 2) 개발 모드: tools/ 디렉토리
    dev_path = APP_DIR / "tools" / "ffmpeg.exe"
    if dev_path.exists():
        return str(dev_path)

    # 3) 시스템 PATH
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        return "ffmpeg"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


FFMPEG_PATH: Optional[str] = None  # 지연 초기화


def _get_ffmpeg() -> str:
    """FFmpeg 경로 반환 (캐시). 없으면 HTTPException."""
    global FFMPEG_PATH
    if FFMPEG_PATH is None:
        FFMPEG_PATH = _get_ffmpeg_path() or ""
    if not FFMPEG_PATH:
        raise HTTPException(
            400, "FFmpeg를 찾을 수 없습니다. tools/ 폴더에 ffmpeg.exe를 넣거나 시스템에 설치하세요.")
    return FFMPEG_PATH


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 대용량 오디오 분할 (로컬 전처리)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def split_large_audio(file_path: Path) -> list[Path]:
    """대용량 오디오를 3분 단위 MP3로 분할.
    7MB 이하 파일은 분할하지 않음. 번들된 FFmpeg 사용."""

    if file_path.stat().st_size <= MAX_RUNPOD_AUDIO_BYTES:
        return [file_path]

    ffmpeg = _get_ffmpeg()

    seg_dir = CHUNK_DIR / f"split_{uuid.uuid4().hex[:8]}"
    seg_dir.mkdir(exist_ok=True)

    result = subprocess.run([
        ffmpeg, '-i', str(file_path),
        '-f', 'segment',
        '-segment_time', str(SPLIT_SEGMENT_SECONDS),
        '-c:a', 'libmp3lame', '-b:a', '128k', '-ac', '1',  # 모노 128kbps MP3
        '-y', str(seg_dir / 'seg_%04d.mp3')
    ], capture_output=True, timeout=600)

    if result.returncode != 0:
        shutil.rmtree(seg_dir, ignore_errors=True)
        raise HTTPException(500, f"오디오 분할 실패: {result.stderr.decode(errors='replace')[:200]}")

    segments = sorted(seg_dir.glob('seg_*.mp3'))
    if not segments:
        shutil.rmtree(seg_dir, ignore_errors=True)
        return [file_path]

    return segments


def prepare_files_for_runpod(file_paths: list[str]) -> list[list[dict]]:
    """파일들을 RunPod payload 크기에 맞게 분할 + 인코딩.
    Returns: list of batches, 각 batch는 [{"filename": ..., "data_base64": ...}, ...]"""

    all_segments: list[Path] = []
    for p in file_paths:
        path = Path(p)
        segments = split_large_audio(path)
        all_segments.extend(segments)

    # 배치 분할: 각 배치의 base64 합계가 MAX_RUNPOD_AUDIO_BYTES 이내
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_size = 0

    for seg_path in all_segments:
        raw_size = seg_path.stat().st_size
        b64_size = int(raw_size * 1.37)  # base64 오버헤드

        if current_size + b64_size > MAX_RUNPOD_AUDIO_BYTES and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        with open(seg_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()

        current_batch.append({
            "filename": seg_path.name,
            "data_base64": data
        })
        current_size += b64_size

    if current_batch:
        batches.append(current_batch)

    return batches


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정 파일 관리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"runpod_api_key": "", "runpod_endpoint_id": ""}

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SQLite 데이터베이스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS training_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                file_size INTEGER,
                duration_seconds REAL,
                file_type TEXT,
                category TEXT DEFAULT 'vocal',
                preprocessed INTEGER DEFAULT 0,
                uploaded_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS voice_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                pth_path TEXT,
                index_path TEXT,
                epochs INTEGER,
                training_time_seconds REAL,
                quality_score REAL,
                status TEXT DEFAULT 'ready',
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS conversions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                input_file TEXT,
                output_file TEXT,
                output_name TEXT,
                pitch_shift INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                processing_time REAL,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (model_id) REFERENCES voice_models(id)
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                message TEXT DEFAULT '',
                result_json TEXT,
                runpod_job_id TEXT,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            );
        """)
        # 기존 DB 마이그레이션: preprocessed 컬럼 추가
        try:
            db.execute("ALTER TABLE training_files ADD COLUMN preprocessed INTEGER DEFAULT 0")
        except Exception:
            pass  # 이미 존재하면 무시


def cleanup_stale_jobs():
    """서버 시작 시 멈춰있는 작업 정리.
    running/submitting 상태로 1시간 이상 방치된 작업을 실패 처리."""
    with get_db() as db:
        stale = db.execute("""
            UPDATE jobs
            SET status='failed',
                message='서버 재시작으로 인한 자동 정리',
                updated_at=?
            WHERE status IN ('running', 'submitting')
              AND updated_at < datetime('now', 'localtime', '-1 hour')
        """, (datetime.now().isoformat(),))
        if stale.rowcount > 0:
            print(f"  정리된 멈춘 작업: {stale.rowcount}개")

        # 서버 재시작 시 모든 running 작업도 정리 (백그라운드 스레드가 사라지므로)
        active = db.execute("""
            UPDATE jobs
            SET status='failed',
                message='서버 재시작으로 인한 자동 정리',
                updated_at=?
            WHERE status IN ('running', 'submitting')
        """, (datetime.now().isoformat(),))
        if active.rowcount > 0:
            print(f"  정리된 진행 중 작업: {active.rowcount}개")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RunPod 에러 메시지 (한국어)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RUNPOD_ERROR_MESSAGES = {
    "timeout": "네트워크 연결 실패. 인터넷 연결을 확인하세요.",
    "auth": "RunPod API 키가 올바르지 않습니다.",
    "funds": "RunPod 잔액이 부족합니다.",
    "file_too_large": "파일 크기 제한 초과",
    "unknown": "RunPod API 오류가 발생했습니다: {detail}",
}


def classify_runpod_error(exc: Exception, response=None) -> str:
    """RunPod API 에러를 한국어 메시지로 분류"""
    if isinstance(exc, (requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError)):
        return RUNPOD_ERROR_MESSAGES["timeout"]

    if response is not None:
        status = response.status_code
        if status in (401, 403):
            return RUNPOD_ERROR_MESSAGES["auth"]
        if status == 402:
            return RUNPOD_ERROR_MESSAGES["funds"]
        try:
            body = response.text.lower()
            if "insufficient" in body or "balance" in body or "funds" in body:
                return RUNPOD_ERROR_MESSAGES["funds"]
        except Exception:
            pass
        return RUNPOD_ERROR_MESSAGES["unknown"].format(
            detail=f"HTTP {status}: {response.text[:200]}"
        )

    return RUNPOD_ERROR_MESSAGES["unknown"].format(detail=str(exc))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RunPod API 클라이언트 (지수 백오프 재시도 포함)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RunPodClient:
    MAX_RETRIES = 3
    BACKOFF_DELAYS = [2, 4, 8]

    def __init__(self):
        self.config = load_config()

    def reload_config(self):
        self.config = load_config()

    @property
    def api_key(self):
        return self.config.get("runpod_api_key", "")

    @property
    def endpoint_id(self):
        return self.config.get("runpod_endpoint_id", "")

    @property
    def base_url(self):
        return f"https://api.runpod.ai/v2/{self.endpoint_id}"

    @property
    def headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def is_configured(self):
        return bool(self.api_key and self.endpoint_id)

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """지수 백오프 재시도 HTTP 요청 (3회, 2/4/8초)"""
        last_exc = None
        last_response = None

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.request(method, url, **kwargs)
                if resp.status_code in (401, 402, 403):
                    error_msg = classify_runpod_error(Exception(), resp)
                    raise HTTPException(status_code=resp.status_code, detail=error_msg)
                resp.raise_for_status()
                return resp
            except HTTPException:
                raise
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                last_exc = e
                last_response = None
            except requests.exceptions.HTTPError as e:
                last_exc = e
                last_response = getattr(e, 'response', None)
            except Exception as e:
                last_exc = e
                last_response = None

            if attempt < self.MAX_RETRIES - 1:
                delay = self.BACKOFF_DELAYS[attempt]
                time.sleep(delay)

        error_msg = classify_runpod_error(last_exc, last_response)
        raise HTTPException(status_code=502, detail=error_msg)

    def submit_job(self, payload: dict) -> str:
        """비동기 작업 제출 → job_id 반환 (재시도 포함)"""
        resp = self._request_with_retry(
            "POST",
            f"{self.base_url}/run",
            headers=self.headers,
            json={"input": payload},
            timeout=120
        )
        return resp.json()["id"]

    def check_status(self, runpod_job_id: str) -> dict:
        """작업 상태 확인 (재시도 포함)"""
        resp = self._request_with_retry(
            "GET",
            f"{self.base_url}/status/{runpod_job_id}",
            headers=self.headers,
            timeout=30
        )
        return resp.json()

    def encode_files(self, file_paths: list) -> list:
        """파일들을 base64 인코딩"""
        encoded = []
        for path in file_paths:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            encoded.append({
                "filename": Path(path).name,
                "data_base64": data
            })
        return encoded


runpod_client = RunPodClient()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 백그라운드 작업 관리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def update_job(job_id: str, **kwargs):
    with get_db() as db:
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values())
        vals.append(datetime.now().isoformat())
        vals.append(job_id)
        db.execute(f"UPDATE jobs SET {sets}, updated_at=? WHERE id=?", vals)


def _is_job_cancelled(job_id: str) -> bool:
    """작업이 취소/실패 상태인지 확인 (폴링 루프 종료용)"""
    try:
        with get_db() as db:
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            return row and row["status"] == "failed"
    except Exception:
        return False


def poll_runpod_job(job_id: str, runpod_job_id: str, job_type: str):
    """RunPod 작업 완료까지 폴링"""
    start_time = time.time()
    while True:
        time.sleep(5)

        # 취소된 작업이면 폴링 중단
        if _is_job_cancelled(job_id):
            return

        try:
            result = runpod_client.check_status(runpod_job_id)
            status = result.get("status", "UNKNOWN")
            elapsed = int(time.time() - start_time)

            if status == "COMPLETED":
                output = result.get("output", {})
                # 결과 처리
                handle_job_result(job_id, job_type, output)
                return

            elif status == "FAILED":
                raw_error = result.get("error", "알 수 없는 오류")
                # RunPod 에러가 dict이면 error_message 추출
                if isinstance(raw_error, dict):
                    error = raw_error.get("error_message") or raw_error.get("message") or str(raw_error)
                else:
                    error = str(raw_error)
                    # JSON 문자열이면 파싱 시도
                    try:
                        import json as _json
                        parsed = _json.loads(error)
                        if isinstance(parsed, dict):
                            error = parsed.get("error_message") or parsed.get("message") or error
                    except (ValueError, TypeError):
                        pass
                update_job(job_id, status="failed", message=error)
                return

            elif status == "IN_QUEUE":
                update_job(job_id, status="running",
                          progress=min(10, elapsed // 6),
                          message=f"GPU 대기 중... ({elapsed}초)")

            elif status == "IN_PROGRESS":
                # 학습은 오래 걸리므로 예상 진행률 계산
                if job_type == "train":
                    est_total = 5 * 3600  # 약 5시간 예상
                    pct = min(90, int((elapsed / est_total) * 100))
                    mins = elapsed // 60
                    update_job(job_id, status="running", progress=pct,
                              message=f"학습 중... ({mins}분 경과)")
                elif job_type == "preprocess":
                    pct = min(90, int(elapsed / 3))
                    update_job(job_id, status="running", progress=pct,
                              message=f"전처리 중... ({elapsed}초)")
                else:
                    pct = min(90, int(elapsed / 0.6))
                    update_job(job_id, status="running", progress=pct,
                              message=f"변환 중... ({elapsed}초)")

            if elapsed > 10 * 3600:  # 10시간 타임아웃
                update_job(job_id, status="failed", message="시간 초과 (10시간)")
                return

        except Exception as e:
            update_job(job_id, status="running",
                      message=f"상태 확인 중... (재시도)")
            time.sleep(10)


def _save_preprocessed_segments(output: dict, job_id: str = "") -> dict:
    """전처리 결과의 세그먼트를 디스크에 저장하고 메타데이터만 반환.
    기존 세그먼트 유지 (append), 새 세그먼트만 추가."""
    segments = output.get("segments", [])
    total_duration = output.get("total_duration", 0.0)
    saved_files = []

    # 기존 파일과 이름 충돌 방지용 prefix
    existing = set(f.name for f in PREPROCESSED_DIR.iterdir() if f.is_file())
    prefix = uuid.uuid4().hex[:6]

    for i, seg in enumerate(segments):
        orig_name = seg.get("filename", f"seg_{i:04d}.wav")
        # 이름 충돌 방지
        fname = orig_name if orig_name not in existing else f"{prefix}_{orig_name}"
        seg_path = PREPROCESSED_DIR / fname
        if seg.get("data_base64"):
            with open(seg_path, "wb") as f:
                f.write(base64.b64decode(seg["data_base64"]))
            saved_files.append({
                "filename": fname,
                "duration_seconds": seg.get("duration_seconds", 0),
            })
            existing.add(fname)

    # 전처리 완료된 파일 마킹
    file_ids = preprocess_file_map.pop(job_id, [])
    if file_ids:
        with get_db() as db:
            placeholders = ",".join("?" * len(file_ids))
            db.execute(
                f"UPDATE training_files SET preprocessed=1 WHERE id IN ({placeholders})",
                file_ids
            )

    # 전체 전처리 세그먼트 수 (기존 + 새로 추가)
    all_files = sorted(PREPROCESSED_DIR.glob("*.wav"))

    return {
        "segment_count": len(all_files),
        "total_duration": round(total_duration, 2),
        "segment_files": [f.name for f in all_files],
    }


def handle_job_result(job_id: str, job_type: str, output: dict):
    """RunPod 작업 결과 처리"""
    try:
        if job_type == "train":
            # 모델 파일 저장
            model_name = output.get("model_name", f"model_{job_id[:8]}")
            model_dir = MODEL_DIR / model_name
            model_dir.mkdir(exist_ok=True)

            pth_path = None
            index_path = None

            if output.get("pth_data"):
                pth_filename = output.get("pth_filename", f"{model_name}.pth")
                pth_path = str(model_dir / pth_filename)
                with open(pth_path, "wb") as f:
                    f.write(base64.b64decode(output["pth_data"]))

            if output.get("index_data"):
                idx_filename = output.get("index_filename", f"{model_name}.index")
                index_path = str(model_dir / idx_filename)
                with open(index_path, "wb") as f:
                    f.write(base64.b64decode(output["index_data"]))

            with get_db() as db:
                db.execute("""
                    INSERT INTO voice_models (name, pth_path, index_path, epochs, 
                                            training_time_seconds, status)
                    VALUES (?, ?, ?, ?, ?, 'ready')
                """, (model_name, pth_path, index_path,
                      output.get("epochs_trained", 0),
                      output.get("training_time_seconds", 0)))

            update_job(job_id, status="completed", progress=100,
                      message="학습 완료!",
                      result_json=json.dumps({"model_name": model_name}))

        elif job_type == "convert":
            # 변환 파일 저장
            if output.get("converted_audio"):
                out_filename = output.get("filename", f"converted_{job_id[:8]}.wav")
                out_path = str(OUTPUT_DIR / out_filename)
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(output["converted_audio"]))

                update_job(job_id, status="completed", progress=100,
                          message="변환 완료!",
                          result_json=json.dumps({
                              "output_file": out_filename,
                              "processing_time": output.get("processing_time_seconds", 0)
                          }))
            else:
                update_job(job_id, status="failed", message="변환 결과 없음")

        elif job_type == "preprocess":
            saved = _save_preprocessed_segments(output, job_id)
            seg_count = saved["segment_count"]
            total_dur = saved["total_duration"]
            dur_str = f", 총 {total_dur:.1f}초" if total_dur > 0 else ""
            update_job(job_id, status="completed", progress=100,
                      message=f"전처리 완료! (세그먼트: {seg_count}개{dur_str})",
                      result_json=json.dumps(saved))

    except Exception as e:
        update_job(job_id, status="failed", message=f"결과 처리 오류: {str(e)}")


def _run_batched_preprocess(job_id: str, batches: list[list[dict]]):
    """대용량 파일의 다중 배치 전처리를 순차 실행.
    각 배치를 별도 RunPod 작업으로 제출하고 완료를 기다린 뒤 다음 배치로 진행."""
    total_batches = len(batches)
    all_results = []

    for idx, batch in enumerate(batches, 1):
        # 취소된 작업이면 중단
        if _is_job_cancelled(job_id):
            return

        batch_label = f"배치 {idx}/{total_batches}"
        pct_base = int((idx - 1) / total_batches * 90)

        update_job(job_id, status="running", progress=pct_base + 2,
                   message=f"{batch_label} 제출 중... ({len(batch)}개 세그먼트)")

        try:
            runpod_job_id = runpod_client.submit_job({
                "task_type": "preprocess",
                "audio_files": batch,
            })
        except Exception as e:
            error_msg = classify_runpod_error(e)
            update_job(job_id, status="failed",
                       message=f"{batch_label} 제출 실패: {error_msg}")
            return

        update_job(job_id, status="running", progress=pct_base + 5,
                   message=f"{batch_label} GPU 처리 중...",
                   runpod_job_id=runpod_job_id)

        # 이 배치의 RunPod 작업 완료까지 폴링
        start_time = time.time()
        while True:
            time.sleep(5)
            if _is_job_cancelled(job_id):
                return
            try:
                result = runpod_client.check_status(runpod_job_id)
                status = result.get("status", "UNKNOWN")
                elapsed = int(time.time() - start_time)

                if status == "COMPLETED":
                    output = result.get("output", {})
                    all_results.append(output)
                    pct = pct_base + int(90 / total_batches)
                    update_job(job_id, status="running", progress=pct,
                               message=f"{batch_label} 완료! (다음 배치 준비 중...)")
                    break

                elif status == "FAILED":
                    error = result.get("error", "알 수 없는 오류")
                    update_job(job_id, status="failed",
                               message=f"{batch_label} 실패: {error}")
                    return

                elif status in ("IN_QUEUE", "IN_PROGRESS"):
                    pct = pct_base + min(int(90 / total_batches) - 2, int(elapsed / 3))
                    state_msg = "GPU 대기 중..." if status == "IN_QUEUE" else "전처리 중..."
                    update_job(job_id, status="running", progress=pct,
                               message=f"{batch_label} {state_msg} ({elapsed}초)")

                if elapsed > 3600:  # 배치당 1시간 타임아웃
                    update_job(job_id, status="failed",
                               message=f"{batch_label} 시간 초과 (1시간)")
                    return

            except Exception:
                update_job(job_id, status="running",
                           message=f"{batch_label} 상태 확인 중... (재시도)")
                time.sleep(10)

    # 모든 배치 완료 — 결과 집계 + 세그먼트 디스크 저장
    merged_segments = []
    merged_duration = 0.0
    for r in all_results:
        merged_segments.extend(r.get("segments", []))
        merged_duration += r.get("total_duration", 0.0)

    saved = _save_preprocessed_segments({
        "segments": merged_segments,
        "total_duration": merged_duration,
    }, job_id)
    seg_count = saved["segment_count"]
    total_dur = saved["total_duration"]
    dur_str = f", 총 {total_dur:.1f}초" if total_dur > 0 else ""
    update_job(job_id, status="completed", progress=100,
               message=f"전처리 완료! (총 {seg_count}개 세그먼트{dur_str}, {total_batches}개 배치)",
               result_json=json.dumps(saved))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI 앱
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(title="AI Voice Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

@app.on_event("startup")
async def startup():
    init_db()
    cleanup_stale_jobs()


# ─── 메인 페이지 ───

@app.get("/")
async def root():
    return FileResponse(str(APP_DIR / "static" / "index.html"))


# ─── 설정 API ───

@app.get("/api/config")
async def get_config():
    config = load_config()
    return {
        "runpod_api_key": "***" + config.get("runpod_api_key", "")[-4:] if config.get("runpod_api_key") else "",
        "runpod_endpoint_id": config.get("runpod_endpoint_id", ""),
        "is_configured": bool(config.get("runpod_api_key") and config.get("runpod_endpoint_id"))
    }

@app.post("/api/config")
async def set_config(api_key: str = Form(""), endpoint_id: str = Form("")):
    config = load_config()
    if api_key and not api_key.startswith("***"):
        config["runpod_api_key"] = api_key
    if endpoint_id:
        config["runpod_endpoint_id"] = endpoint_id
    save_config(config)
    runpod_client.reload_config()
    return {"status": "ok"}


# ─── 파일 업로드 API ───

@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    category: str = Form("vocal")
):
    saved = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".mkv", ".webm"]:
            continue

        unique_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
        save_path = UPLOAD_DIR / unique_name

        with open(save_path, "wb") as fp:
            content = await f.read()
            fp.write(content)

        with get_db() as db:
            db.execute("""
                INSERT INTO training_files (filename, original_name, file_size, file_type, category)
                VALUES (?, ?, ?, ?, ?)
            """, (unique_name, f.filename, len(content), ext, category))
            file_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        saved.append({
            "id": file_id,
            "filename": unique_name,
            "original_name": f.filename,
            "size": len(content),
            "type": ext,
            "category": category
        })

    return {"files": saved, "count": len(saved)}


@app.get("/api/files")
async def list_files():
    with get_db() as db:
        rows = db.execute("SELECT * FROM training_files ORDER BY uploaded_at DESC").fetchall()
    return {"files": [dict(r) for r in rows]}


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: int):
    with get_db() as db:
        row = db.execute("SELECT filename FROM training_files WHERE id=?", (file_id,)).fetchone()
        if row:
            filepath = UPLOAD_DIR / row["filename"]
            if filepath.exists():
                filepath.unlink()
            db.execute("DELETE FROM training_files WHERE id=?", (file_id,))
    return {"status": "ok"}


# ─── 청크 업로드 API (대용량 파일 지원, 최대 2GB) ───

@app.post("/api/upload/chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    filename: str = Form(...),
    category: str = Form("vocal"),
    chunk: UploadFile = File(...)
):
    """청크 단위 업로드 — 대용량 파일을 분할 전송. 모든 청크 도착 시 자동 재조립."""
    if upload_id not in chunk_uploads:
        chunk_uploads[upload_id] = {
            "total_chunks": total_chunks,
            "received": set(),
            "filename": filename,
            "category": category,
        }

    session = chunk_uploads[upload_id]

    if chunk_index < 0 or chunk_index >= total_chunks:
        raise HTTPException(400, f"잘못된 청크 인덱스: {chunk_index}")
    if chunk_index in session["received"]:
        return {"status": "duplicate", "upload_id": upload_id,
                "received": len(session["received"]), "total": total_chunks}

    chunk_dir = CHUNK_DIR / upload_id
    chunk_dir.mkdir(exist_ok=True)
    chunk_path = chunk_dir / f"chunk_{chunk_index:06d}"

    chunk_data = await chunk.read()
    with open(chunk_path, "wb") as f:
        f.write(chunk_data)

    session["received"].add(chunk_index)

    if len(session["received"]) == total_chunks:
        ext = Path(filename).suffix.lower()
        if ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".mkv", ".webm"]:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            chunk_uploads.pop(upload_id, None)
            raise HTTPException(400, f"지원하지 않는 파일 형식: {ext}")

        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        save_path = UPLOAD_DIR / unique_name
        total_size = 0

        with open(save_path, "wb") as out_f:
            for i in range(total_chunks):
                cp = chunk_dir / f"chunk_{i:06d}"
                with open(cp, "rb") as in_f:
                    data = in_f.read()
                    total_size += len(data)
                    out_f.write(data)

        if total_size > MAX_CHUNK_FILE_SIZE:
            save_path.unlink(missing_ok=True)
            shutil.rmtree(chunk_dir, ignore_errors=True)
            chunk_uploads.pop(upload_id, None)
            raise HTTPException(413, RUNPOD_ERROR_MESSAGES["file_too_large"])

        shutil.rmtree(chunk_dir, ignore_errors=True)
        chunk_uploads.pop(upload_id, None)

        with get_db() as db:
            db.execute("""
                INSERT INTO training_files (filename, original_name, file_size, file_type, category)
                VALUES (?, ?, ?, ?, ?)
            """, (unique_name, filename, total_size, ext, category))
            file_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        return {
            "status": "completed", "upload_id": upload_id,
            "file": {"id": file_id, "filename": unique_name, "original_name": filename,
                     "size": total_size, "type": ext, "category": category}
        }

    return {"status": "receiving", "upload_id": upload_id,
            "received": len(session["received"]), "total": total_chunks}


# ─── 학습 API ───

@app.post("/api/train")
async def start_training(
    model_name: str = Form(...),
    epochs: int = Form(300),
    sample_rate: int = Form(44100),
    batch_size: int = Form(8),
    file_ids: str = Form("")  # comma-separated
):
    if not runpod_client.is_configured():
        raise HTTPException(400, "RunPod API 설정이 필요합니다. 설정 페이지에서 API Key와 Endpoint ID를 입력하세요.")

    # 학습 파일 수집
    with get_db() as db:
        if file_ids:
            ids = [int(x.strip()) for x in file_ids.split(",") if x.strip()]
            placeholders = ",".join("?" * len(ids))
            rows = db.execute(
                f"SELECT * FROM training_files WHERE id IN ({placeholders})", ids
            ).fetchall()
        else:
            rows = db.execute("SELECT * FROM training_files").fetchall()

    if not rows:
        raise HTTPException(400, "학습할 파일이 없습니다.")

    # 전처리된 세그먼트가 있으면 우선 사용 (용량 작음 → 10MB 이내)
    preprocessed_files = sorted(PREPROCESSED_DIR.glob("*.wav"))
    if preprocessed_files:
        file_paths = [str(p) for p in preprocessed_files]
    else:
        file_paths = [str(UPLOAD_DIR / r["filename"]) for r in rows]

    batches = prepare_files_for_runpod(file_paths)
    total_segments = sum(len(b) for b in batches)

    # Job 생성
    job_id = uuid.uuid4().hex[:12]
    with get_db() as db:
        db.execute("""
            INSERT INTO jobs (id, job_type, status, progress, message)
            VALUES (?, 'train', 'submitting', 0, '작업 제출 중...')
        """, (job_id,))

    try:
        # 배치가 여러 개여도 학습은 모든 데이터가 한 번에 필요
        # → 순차 업로드 불가, 반드시 단일 요청에 포함되어야 함
        if len(batches) > 1:
            raise HTTPException(
                400,
                f"학습 데이터가 너무 큽니다 ({total_segments}개 세그먼트, {len(batches)}개 배치). "
                f"전처리를 먼저 실행하여 데이터를 줄이거나, 파일 수를 줄여주세요."
            )

        runpod_job_id = runpod_client.submit_job({
            "task_type": "train",
            "model_name": model_name,
            "audio_files": batches[0],
            "sample_rate": sample_rate,
            "epochs": epochs,
            "batch_size": batch_size
        })

        msg = f"GPU에 작업 제출됨 ({total_segments}개 세그먼트)"
        update_job(job_id, status="running", progress=5,
                  message=msg, runpod_job_id=runpod_job_id)

        thread = threading.Thread(
            target=poll_runpod_job,
            args=(job_id, runpod_job_id, "train"),
            daemon=True
        )
        thread.start()

    except HTTPException:
        raise

    except Exception as e:
        error_msg = classify_runpod_error(e)
        update_job(job_id, status="failed", message=f"제출 실패: {error_msg}")

    return {"job_id": job_id, "segments": total_segments}


# ─── 전처리 API ───

@app.post("/api/preprocess")
async def start_preprocess(
    file_ids: str = Form(...),
):
    """전처리 시작 — 영상→오디오 추출, Demucs 보컬 분리, 화자 분리, 노이즈 제거, 세그먼트 분할"""
    if not runpod_client.is_configured():
        raise HTTPException(400, "RunPod API 설정이 필요합니다. 설정 페이지에서 API Key와 Endpoint ID를 입력하세요.")

    ids = [int(x.strip()) for x in file_ids.split(",") if x.strip()]
    if not ids:
        raise HTTPException(400, "전처리할 파일을 선택하세요.")

    with get_db() as db:
        placeholders = ",".join("?" * len(ids))
        rows = db.execute(
            f"SELECT * FROM training_files WHERE id IN ({placeholders})", ids
        ).fetchall()

    if not rows:
        raise HTTPException(400, "선택한 파일을 찾을 수 없습니다.")

    # 이미 전처리된 파일 제외
    unprocessed = [r for r in rows if not r["preprocessed"]]
    if not unprocessed:
        return {"job_id": None, "segments": 0, "batches": 0,
                "message": "모든 파일이 이미 전처리되었습니다."}

    # 전처리 대상 파일 ID 기록 (완료 시 마킹용)
    preprocess_file_ids = [r["id"] for r in unprocessed]

    # 대용량 파일 자동 분할 + 배치 인코딩
    file_paths = [str(UPLOAD_DIR / r["filename"]) for r in unprocessed]
    batches = prepare_files_for_runpod(file_paths)
    total_segments = sum(len(b) for b in batches)

    skipped = len(rows) - len(unprocessed)
    skip_msg = f" ({skipped}개 파일은 이미 전처리됨)" if skipped else ""

    job_id = uuid.uuid4().hex[:12]
    preprocess_file_map[job_id] = preprocess_file_ids
    with get_db() as db:
        db.execute("""
            INSERT INTO jobs (id, job_type, status, progress, message)
            VALUES (?, 'preprocess', 'submitting', 0, '전처리 작업 제출 중...')
        """, (job_id,))

    try:
        # 배치별로 RunPod 작업 제출 (대용량 파일은 여러 작업으로 분할)
        if len(batches) == 1:
            runpod_job_id = runpod_client.submit_job({
                "task_type": "preprocess",
                "audio_files": batches[0],
            })
            update_job(job_id, status="running", progress=5,
                      message=f"GPU에 전처리 작업 제출됨 ({len(unprocessed)}개 파일{skip_msg})",
                      runpod_job_id=runpod_job_id)

            thread = threading.Thread(
                target=poll_runpod_job,
                args=(job_id, runpod_job_id, "preprocess"),
                daemon=True
            )
            thread.start()
        else:
            # 다중 배치: 순차적으로 전처리 (각 배치를 별도 RunPod 작업으로)
            update_job(job_id, status="running", progress=2,
                      message=f"대용량 파일 분할 완료 ({total_segments}개 세그먼트, {len(batches)}개 배치)")

            thread = threading.Thread(
                target=_run_batched_preprocess,
                args=(job_id, batches),
                daemon=True
            )
            thread.start()

    except HTTPException:
        raise

    except Exception as e:
        error_msg = classify_runpod_error(e)
        update_job(job_id, status="failed", message=f"제출 실패: {error_msg}")

    return {"job_id": job_id, "segments": total_segments, "batches": len(batches),
            "skipped": skipped, "processing": len(unprocessed)}


@app.get("/api/preprocess/status")
async def preprocess_status():
    """전처리 상태 확인 — preprocessed/ 디렉토리에 세그먼트가 있는지 반환.
    세그먼트가 존재하면 DB의 preprocessed 플래그도 동기화."""
    files = sorted(PREPROCESSED_DIR.glob("*.wav"))
    if not files:
        # 세그먼트 없으면 DB도 리셋
        with get_db() as db:
            db.execute("UPDATE training_files SET preprocessed=0 WHERE preprocessed=1")
        return {"preprocessed": False, "segment_count": 0, "total_duration": 0}

    # 세그먼트가 존재하면 미처리 파일을 전처리 완료로 동기화
    with get_db() as db:
        unmarked = db.execute(
            "SELECT COUNT(*) FROM training_files WHERE preprocessed=0"
        ).fetchone()[0]
        if unmarked > 0:
            db.execute("UPDATE training_files SET preprocessed=1")

    total_dur = 0.0
    try:
        import wave
        for f in files:
            with wave.open(str(f), "rb") as wf:
                total_dur += wf.getnframes() / wf.getframerate()
    except Exception:
        pass

    return {
        "preprocessed": True,
        "segment_count": len(files),
        "total_duration": round(total_dur, 2),
    }


@app.delete("/api/preprocess")
async def clear_preprocess():
    """전처리 결과 삭제 — preprocessed/ 비우기 + DB preprocessed 플래그 리셋"""
    count = 0
    for f in PREPROCESSED_DIR.iterdir():
        if f.is_file():
            f.unlink()
            count += 1
    with get_db() as db:
        db.execute("UPDATE training_files SET preprocessed=0")
    return {"cleared": count}


# ─── 변환 API ───

@app.post("/api/convert")
async def start_conversion(
    model_id: int = Form(...),
    pitch_shift: int = Form(0),
    index_rate: float = Form(0.75),
    audio: UploadFile = File(...)
):
    if not runpod_client.is_configured():
        raise HTTPException(400, "RunPod API 설정이 필요합니다.")

    # 모델 조회
    with get_db() as db:
        model = db.execute("SELECT * FROM voice_models WHERE id=?", (model_id,)).fetchone()
    if not model:
        raise HTTPException(404, "모델을 찾을 수 없습니다.")

    # 입력 파일 저장
    temp_name = f"input_{uuid.uuid4().hex[:8]}_{audio.filename}"
    temp_path = UPLOAD_DIR / temp_name
    with open(temp_path, "wb") as f:
        content = await audio.read()
        f.write(content)

    # 모델 파일 인코딩
    pth_b64 = ""
    if model["pth_path"] and Path(model["pth_path"]).exists():
        with open(model["pth_path"], "rb") as f:
            pth_b64 = base64.b64encode(f.read()).decode()

    index_b64 = ""
    if model["index_path"] and Path(model["index_path"]).exists():
        with open(model["index_path"], "rb") as f:
            index_b64 = base64.b64encode(f.read()).decode()

    # Job 생성
    job_id = uuid.uuid4().hex[:12]
    with get_db() as db:
        db.execute("""
            INSERT INTO jobs (id, job_type, status, progress, message)
            VALUES (?, 'convert', 'submitting', 0, '작업 제출 중...')
        """, (job_id,))

    try:
        with open(temp_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        runpod_job_id = runpod_client.submit_job({
            "task_type": "convert",
            "pth_data": pth_b64,
            "index_data": index_b64,
            "audio_data": audio_b64,
            "audio_filename": audio.filename,
            "pitch_shift": pitch_shift,
            "index_rate": index_rate,
            "f0_method": "rmvpe"
        })

        update_job(job_id, status="running", progress=10,
                  message="GPU 변환 시작", runpod_job_id=runpod_job_id)

        # DB에 변환 기록
        with get_db() as db:
            db.execute("""
                INSERT INTO conversions (model_id, input_file, status)
                VALUES (?, ?, 'processing')
            """, (model_id, audio.filename))

        thread = threading.Thread(
            target=poll_runpod_job,
            args=(job_id, runpod_job_id, "convert"),
            daemon=True
        )
        thread.start()

    except HTTPException:
        raise

    except Exception as e:
        error_msg = classify_runpod_error(e)
        update_job(job_id, status="failed", message=f"제출 실패: {error_msg}")

    return {"job_id": job_id}


# ─── 작업 상태 API ───

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    with get_db() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    result = dict(row)
    if result.get("result_json"):
        result["result"] = json.loads(result["result_json"])
    return result


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """실행 중인 작업 취소 (DB 상태만 변경, 백그라운드 스레드는 다음 폴링 시 자동 종료)"""
    with get_db() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        if row["status"] in ("completed", "failed"):
            return {"status": "already_done"}
        db.execute(
            "UPDATE jobs SET status='failed', message='사용자가 취소함', updated_at=? WHERE id=?",
            (datetime.now().isoformat(), job_id)
        )
    return {"status": "cancelled"}


@app.post("/api/jobs/cleanup")
async def cleanup_all_stuck_jobs():
    """멈춘 작업 일괄 정리 — running/submitting 상태의 모든 작업을 실패 처리"""
    with get_db() as db:
        result = db.execute("""
            UPDATE jobs
            SET status='failed',
                message='수동 정리',
                updated_at=?
            WHERE status IN ('running', 'submitting')
        """, (datetime.now().isoformat(),))
    return {"cleaned": result.rowcount}


@app.get("/api/jobs")
async def list_jobs():
    with get_db() as db:
        rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50").fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("result_json"):
            d["result"] = json.loads(d["result_json"])
        results.append(d)
    return {"jobs": results}


# ─── 모델 API ───

@app.get("/api/models")
async def list_models():
    with get_db() as db:
        rows = db.execute("SELECT * FROM voice_models ORDER BY created_at DESC").fetchall()
    return {"models": [dict(r) for r in rows]}


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: int):
    with get_db() as db:
        row = db.execute("SELECT * FROM voice_models WHERE id=?", (model_id,)).fetchone()
        if row:
            for path_key in ["pth_path", "index_path"]:
                if row[path_key] and Path(row[path_key]).exists():
                    Path(row[path_key]).unlink()
            db.execute("DELETE FROM voice_models WHERE id=?", (model_id,))
    return {"status": "ok"}


@app.post("/api/models/{model_id}/rename")
async def rename_model(model_id: int, name: str = Form(...)):
    """모델 이름 변경"""
    if not name or not name.strip():
        raise HTTPException(400, "모델 이름을 입력하세요.")
    with get_db() as db:
        row = db.execute("SELECT * FROM voice_models WHERE id=?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(404, "모델을 찾을 수 없습니다.")
        db.execute("UPDATE voice_models SET name=? WHERE id=?", (name.strip(), model_id))
    return {"status": "ok", "name": name.strip()}


@app.post("/api/models/{model_id}/quality")
async def update_model_quality(model_id: int, quality_score: float = Form(...)):
    """테스트 변환 후 모델 품질 점수 업데이트"""
    if quality_score < 0.0 or quality_score > 5.0:
        raise HTTPException(400, "품질 점수는 0.0~5.0 범위여야 합니다.")
    with get_db() as db:
        row = db.execute("SELECT * FROM voice_models WHERE id=?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(404, "모델을 찾을 수 없습니다.")
        db.execute("UPDATE voice_models SET quality_score=? WHERE id=?", (quality_score, model_id))
    return {"status": "ok", "quality_score": quality_score}


# ─── 변환 결과 API ───

@app.get("/api/conversions")
async def list_conversions():
    with get_db() as db:
        rows = db.execute("""
            SELECT c.*, m.name as model_name 
            FROM conversions c 
            LEFT JOIN voice_models m ON c.model_id = m.id
            ORDER BY c.created_at DESC LIMIT 50
        """).fetchall()
    return {"conversions": [dict(r) for r in rows]}


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    return FileResponse(str(filepath), filename=filename)


# ─── 대시보드 통계 API ───

@app.get("/api/stats")
async def get_stats():
    with get_db() as db:
        file_count = db.execute("SELECT COUNT(*) FROM training_files").fetchone()[0]
        model_count = db.execute("SELECT COUNT(*) FROM voice_models WHERE status='ready'").fetchone()[0]
        conv_count = db.execute("SELECT COUNT(*) FROM conversions").fetchone()[0]
        total_size = db.execute("SELECT COALESCE(SUM(file_size),0) FROM training_files").fetchone()[0]
    return {
        "files": file_count,
        "models": model_count,
        "conversions": conv_count,
        "total_size_mb": round(total_size / 1024 / 1024, 1)
    }


# ─── 헬스 체크 API ───

@app.get("/api/health")
async def health_check():
    """서버 상태, RunPod 연결, 디스크 용량 확인"""
    health = {
        "status": "ok",
        "server": "running",
        "timestamp": datetime.now().isoformat(),
    }

    if runpod_client.is_configured():
        try:
            resp = requests.get(
                f"https://api.runpod.ai/v2/{runpod_client.endpoint_id}/health",
                headers=runpod_client.headers,
                timeout=10
            )
            if resp.status_code == 200:
                health["runpod"] = {"status": "connected", "detail": resp.json()}
            elif resp.status_code in (401, 403):
                health["runpod"] = {"status": "auth_error",
                                    "message": RUNPOD_ERROR_MESSAGES["auth"]}
            else:
                health["runpod"] = {"status": "error",
                                    "message": f"HTTP {resp.status_code}"}
        except requests.exceptions.Timeout:
            health["runpod"] = {"status": "timeout",
                                "message": RUNPOD_ERROR_MESSAGES["timeout"]}
        except Exception as e:
            health["runpod"] = {"status": "error", "message": str(e)}
    else:
        health["runpod"] = {"status": "not_configured",
                            "message": "RunPod API 설정이 필요합니다."}

    try:
        disk = shutil.disk_usage(str(DATA_DIR))
        health["disk"] = {
            "total_gb": round(disk.total / (1024 ** 3), 1),
            "used_gb": round(disk.used / (1024 ** 3), 1),
            "free_gb": round(disk.free / (1024 ** 3), 1),
            "usage_percent": round((disk.used / disk.total) * 100, 1),
        }
    except Exception as e:
        health["disk"] = {"status": "error", "message": str(e)}

    try:
        with get_db() as db:
            db.execute("SELECT 1").fetchone()
        health["database"] = {"status": "ok"}
    except Exception as e:
        health["database"] = {"status": "error", "message": str(e)}
        health["status"] = "degraded"

    return health


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("\n🎤 AI Voice Studio 시작 중...")
    print("   http://localhost:8000 에서 접속하세요\n")

    if not FROZEN:
        # 개발 모드: 브라우저 자동 열기 (.exe 모드에서는 pywebview가 처리)
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
