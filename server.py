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
from fastapi.responses import FileResponse, JSONResponse, Response
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
PREPROCESSED_DIR = DATA_DIR / "preprocessed"  # 전처리 결과 세그먼트 저장 (WAV 또는 MP3)
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

def _list_preprocessed_files() -> list[Path]:
    """전처리된 오디오 파일 목록 (WAV + MP3)."""
    files: list[Path] = []
    for ext in ("*.wav", "*.mp3"):
        files.extend(PREPROCESSED_DIR.glob(ext))
    return sorted(files)


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


def _get_audio_duration(file_path: Path) -> Optional[float]:
    """FFprobe로 오디오 길이(초) 측정. 실패 시 None."""
    try:
        ffmpeg = _get_ffmpeg_path()
        # ffprobe는 ffmpeg와 같은 디렉토리에 있음
        if ffmpeg and ffmpeg != "ffmpeg":
            ffprobe = str(Path(ffmpeg).parent / "ffprobe.exe")
            if not Path(ffprobe).exists():
                ffprobe = "ffprobe"
        else:
            ffprobe = "ffprobe"
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
            capture_output=True, text=True, timeout=30
        )
        return round(float(result.stdout.strip()), 2)
    except Exception:
        return None


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
                pth_url TEXT,
                index_url TEXT,
                epochs INTEGER,
                training_time_seconds REAL,
                quality_score REAL,
                training_files_json TEXT,
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
        # 기존 DB 마이그레이션
        for col, default in [("preprocessed", "INTEGER DEFAULT 0"),
                             ("file_hash", "TEXT"),
                             ("deleted", "INTEGER DEFAULT 0")]:
            try:
                db.execute(f"ALTER TABLE training_files ADD COLUMN {col} {default}")
            except Exception:
                pass  # 이미 존재하면 무시
        # voice_models에 R2 URL 컬럼 추가 (기존 DB 마이그레이션)
        for col in ("pth_url", "index_url", "training_files_json"):
            try:
                db.execute(f"ALTER TABLE voice_models ADD COLUMN {col} TEXT")
            except Exception:
                pass
        # conversions에 job_id 컬럼 추가 (정확한 작업-변환 매핑)
        try:
            db.execute("ALTER TABLE conversions ADD COLUMN job_id TEXT")
        except Exception:
            pass
        # file_hash 인덱스 (중복 체크 성능)
        try:
            db.execute("CREATE INDEX IF NOT EXISTS idx_training_files_hash ON training_files(file_hash)")
        except Exception:
            pass


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

# 학습 작업별 사용된 훈련 파일 목록 (job_id → [파일명, ...])
_training_file_map: dict[str, list[str]] = {}


def upload_to_r2(file_path: Path, key: str) -> str:
    """로컬 파일을 R2에 업로드하고 presigned URL을 반환합니다."""
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError:
        raise HTTPException(400,
            "boto3가 설치되지 않았습니다. 터미널에서 'pip install boto3'를 실행하세요.")

    config = load_config()
    endpoint_url = config.get("r2_endpoint_url")
    access_key = config.get("r2_access_key_id")
    secret_key = config.get("r2_secret_access_key")
    bucket = config.get("r2_bucket_name", "voice-studio")

    if not all([endpoint_url, access_key, secret_key]):
        raise HTTPException(400,
            "R2 스토리지가 설정되지 않았습니다. 설정 페이지에서 R2 정보를 입력하세요.")

    s3 = boto3.client("s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )

    try:
        s3.upload_file(str(file_path), bucket, key)
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=86400 * 7,  # 7일
        )
    except Exception as e:
        print(f"[R2 Upload Error] {type(e).__name__}: {e}")
        raise HTTPException(500, f"R2 업로드 실패: {type(e).__name__}: {e}")

    return presigned_url

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
                if not output:
                    update_job(job_id, status="failed",
                              message="RunPod 작업은 완료되었지만 결과 데이터가 비어있습니다. "
                                      "페이로드 크기 제한(413) 초과일 수 있습니다.")
                    if job_type == "convert":
                        try:
                            with get_db() as db:
                                db.execute("UPDATE conversions SET status='failed' WHERE job_id=?", (job_id,))
                        except Exception:
                            pass
                    return
                try:
                    handle_job_result(job_id, job_type, output)
                except Exception as e:
                    update_job(job_id, status="failed",
                              message=f"결과 처리 실패: {e}")
                    if job_type == "convert":
                        try:
                            with get_db() as db:
                                db.execute("UPDATE conversions SET status='failed' WHERE job_id=?", (job_id,))
                        except Exception:
                            pass
                return

            elif status in ("FAILED", "TIMED_OUT", "CANCELLED"):
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
                if status == "TIMED_OUT":
                    error = f"RunPod 시간 초과: {error}"
                elif status == "CANCELLED":
                    error = f"RunPod 작업 취소됨: {error}"
                update_job(job_id, status="failed", message=error)
                # 변환 실패 시 conversions 테이블도 업데이트
                if job_type == "convert":
                    try:
                        with get_db() as db:
                            db.execute("UPDATE conversions SET status='failed' WHERE job_id=?", (job_id,))
                    except Exception:
                        pass
                return

            elif status == "IN_QUEUE":
                update_job(job_id, status="running",
                          progress=min(10, elapsed // 6),
                          message=f"GPU 대기 중... ({elapsed}초)")

            elif status == "IN_PROGRESS":
                # RunPod progress_update 메시지에서 실제 진행률 추출
                pct = None
                msg = None

                # progress_update → output 필드 (IN_PROGRESS 상태)
                # 또는 stream 배열에서 최신 메시지 확인
                import re
                progress_text = ""
                output_field = result.get("output")
                if isinstance(output_field, str) and output_field:
                    progress_text = output_field
                elif isinstance(output_field, dict):
                    progress_text = output_field.get("progress", "")

                # stream 필드도 확인
                if not progress_text:
                    stream = result.get("stream") or []
                    if isinstance(stream, list):
                        for entry in reversed(stream):
                            text = entry.get("output", "") if isinstance(entry, dict) else str(entry)
                            if text:
                                progress_text = text
                                break

                if progress_text:
                    # "Training epoch 50/300 (16%)" 파싱
                    m = re.search(r'\((\d+)%\)', progress_text)
                    if m:
                        pct = int(m.group(1))
                    else:
                        m2 = re.search(r'(\d+)\s*/\s*(\d+)', progress_text)
                        if m2:
                            pct = min(95, int(int(m2.group(1)) / int(m2.group(2)) * 100))

                    # 한국어 메시지 변환
                    if "epoch" in progress_text.lower():
                        m3 = re.search(r'epoch\s+(\d+)\s*/\s*(\d+)', progress_text, re.IGNORECASE)
                        if m3:
                            msg = f"학습 중... 에폭 {m3.group(1)}/{m3.group(2)}"
                    elif "(" in progress_text and "/" in progress_text:
                        # "(3/5)" 스텝 형태
                        m4 = re.search(r'\((\d+)/(\d+)\)', progress_text)
                        if m4:
                            step, total_steps = int(m4.group(1)), int(m4.group(2))
                            pct = min(90, int(step / total_steps * 90))
                            msg = progress_text

                if pct is None:
                    # fallback: 경과 시간 기반 추정
                    if job_type == "train":
                        est_total = 900  # ~15분 예상
                        pct = min(90, int((elapsed / est_total) * 100))
                    elif job_type == "preprocess":
                        pct = min(90, int(elapsed / 3))
                    else:
                        pct = min(90, int(elapsed / 0.6))

                pct = min(95, max(5, pct))
                mins = elapsed // 60

                if job_type == "train":
                    update_job(job_id, status="running", progress=pct,
                              message=msg or f"학습 중... ({mins}분 경과)")
                elif job_type == "preprocess":
                    update_job(job_id, status="running", progress=pct,
                              message=msg or f"전처리 중... ({elapsed}초)")
                else:
                    update_job(job_id, status="running", progress=pct,
                              message=msg or f"변환 중... ({elapsed}초)")

            else:
                # Unknown status — log and treat as transient
                update_job(job_id, status="running",
                          message=f"상태: {status} ({elapsed}초)")

            if elapsed > 10 * 3600:  # 10시간 타임아웃
                update_job(job_id, status="failed", message="시간 초과 (10시간)")
                return

        except Exception as e:
            poll_errors = getattr(poll_runpod_job, '_errors', 0) + 1
            poll_runpod_job._errors = poll_errors
            if poll_errors > 30:  # 30 consecutive errors = ~5 min of failures
                update_job(job_id, status="failed",
                          message=f"RunPod 상태 확인 반복 실패: {e}")
                return
            update_job(job_id, status="running",
                      message=f"상태 확인 중... (재시도 {poll_errors})")
            time.sleep(10)


def _save_preprocessed_segments(output: dict, job_id: str = "") -> dict:
    """전처리 결과의 세그먼트 + 반주(MR) + 보컬 파일을 디스크에 저장하고 메타데이터 반환.
    기존 세그먼트 유지 (append), 새 세그먼트만 추가."""
    segments = output.get("segments", [])
    accomp_files = output.get("accompaniment_files", [])
    vocal_files = output.get("vocal_files", [])
    total_duration = output.get("total_duration", 0.0)

    # Diagnostic: log what we received from RunPod
    print(f"[Preprocess] Received output keys: {list(output.keys())}")
    print(f"[Preprocess] segment_count={output.get('segment_count')}, "
          f"total_duration={total_duration}, "
          f"segments={len(segments)}, accomp={len(accomp_files)}, "
          f"vocals={len(vocal_files)}")
    if not segments and output.get("segment_count", 0) > 0:
        print(f"[Preprocess] WARNING: Handler reported {output.get('segment_count')} segments "
              f"but segments array is empty! RunPod response may have been truncated "
              f"due to payload size limit.")
    saved_files = []

    # 기존 파일과 이름 충돌 방지용 prefix
    existing = set(f.name for f in PREPROCESSED_DIR.iterdir() if f.is_file())
    prefix = uuid.uuid4().hex[:6]

    def _save_file_list(file_list, default_prefix="seg"):
        """Save files to PREPROCESSED_DIR — supports R2 URL or inline base64."""
        saved = []
        for i, fobj in enumerate(file_list):
            orig_name = fobj.get("filename", f"{default_prefix}_{i:04d}.wav")
            fname = orig_name if orig_name not in existing else f"{prefix}_{orig_name}"
            fpath = PREPROCESSED_DIR / fname
            if fobj.get("url"):
                # Download from R2 presigned URL
                try:
                    resp = requests.get(fobj["url"], timeout=120)
                    resp.raise_for_status()
                    with open(fpath, "wb") as f:
                        f.write(resp.content)
                    saved.append({
                        "filename": fname,
                        "duration_seconds": fobj.get("duration_seconds", 0),
                    })
                    existing.add(fname)
                except Exception as e:
                    print(f"[Preprocess] Failed to download {orig_name}: {e}")
            elif fobj.get("data_base64"):
                # Legacy: inline base64
                with open(fpath, "wb") as f:
                    f.write(base64.b64decode(fobj["data_base64"]))
                saved.append({
                    "filename": fname,
                    "duration_seconds": fobj.get("duration_seconds", 0),
                })
                existing.add(fname)
        return saved

    # Save training segments
    saved_files = _save_file_list(segments, "seg")

    # Save accompaniment (MR) files for user download
    saved_accomp = _save_file_list(accomp_files, "mr")

    # Save full vocal files for user download
    saved_vocals = _save_file_list(vocal_files, "vocal")

    # 전처리 완료된 파일 마킹 + 세그먼트 매핑 기록
    file_ids = preprocess_file_map.pop(job_id, [])
    if file_ids:
        with get_db() as db:
            placeholders = ",".join("?" * len(file_ids))
            db.execute(
                f"UPDATE training_files SET preprocessed=1 WHERE id IN ({placeholders})",
                file_ids
            )

    # 전체 전처리 세그먼트 수 (학습용 세그먼트만 카운트, mr_/vocal_ 제외)
    all_files = _list_preprocessed_files()
    training_files = [f for f in all_files if not f.name.startswith(("mr_", "vocal_"))]

    # 정확한 총 길이를 메타데이터 파일에 저장 (재시작 시 정확한 값 복원용)
    # + 파일 ID → 세그먼트 매핑 저장 (선택적 학습 시 필터링용)
    meta_path = PREPROCESSED_DIR / "_metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    prev_dur = meta.get("total_duration", 0.0)
    merged_dur = prev_dur + total_duration
    meta["total_duration"] = round(merged_dur, 2)

    # file_id → segment filenames 매핑 업데이트
    seg_map = meta.get("file_segments", {})
    if file_ids and saved_files:
        seg_names = [s["filename"] for s in saved_files]
        for fid in file_ids:
            seg_map[str(fid)] = seg_map.get(str(fid), []) + seg_names
    meta["file_segments"] = seg_map
    meta_path.write_text(json.dumps(meta))

    return {
        "segment_count": len(training_files),
        "total_duration": round(merged_dur, 2),
        "segment_files": [f.name for f in training_files],
        "accompaniment_files": [s["filename"] for s in saved_accomp],
        "vocal_files": [s["filename"] for s in saved_vocals],
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

            # Download model from presigned URL (primary path)
            if output.get("pth_url"):
                pth_filename = output.get("pth_filename", f"{model_name}.pth")
                pth_path = str(model_dir / pth_filename)
                resp = requests.get(output["pth_url"], timeout=300)
                resp.raise_for_status()
                with open(pth_path, "wb") as f:
                    f.write(resp.content)
            # Legacy: inline base64 (backward compat for small models)
            elif output.get("pth_data"):
                pth_filename = output.get("pth_filename", f"{model_name}.pth")
                pth_path = str(model_dir / pth_filename)
                raw = base64.b64decode(output["pth_data"])
                if output.get("pth_compressed"):
                    import gzip
                    raw = gzip.decompress(raw)
                with open(pth_path, "wb") as f:
                    f.write(raw)

            # Download index from presigned URL (primary path)
            if output.get("index_url"):
                idx_filename = output.get("index_filename", f"{model_name}.index")
                index_path = str(model_dir / idx_filename)
                resp = requests.get(output["index_url"], timeout=120)
                resp.raise_for_status()
                with open(index_path, "wb") as f:
                    f.write(resp.content)
            # Legacy: inline base64
            elif output.get("index_data"):
                idx_filename = output.get("index_filename", f"{model_name}.index")
                index_path = str(model_dir / idx_filename)
                raw = base64.b64decode(output["index_data"])
                if output.get("index_compressed"):
                    import gzip
                    raw = gzip.decompress(raw)
                with open(index_path, "wb") as f:
                    f.write(raw)

            with get_db() as db:
                db.execute("""
                    INSERT INTO voice_models (name, pth_path, index_path, pth_url, index_url,
                                            epochs, training_time_seconds, training_files_json, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready')
                """, (model_name, pth_path, index_path,
                      output.get("pth_url"), output.get("index_url"),
                      output.get("epochs_trained", 0),
                      output.get("training_time_seconds", 0),
                      json.dumps(_training_file_map.pop(job_id, []), ensure_ascii=False)))

            update_job(job_id, status="completed", progress=100,
                      message="학습 완료!",
                      result_json=json.dumps({"model_name": model_name}))

        elif job_type == "convert":
            # 변환 파일 저장 (보컬 + 믹스) — R2 URL 또는 inline base64
            out_filename = output.get("filename", f"converted_{job_id[:8]}.wav")
            out_path = OUTPUT_DIR / out_filename
            has_vocals = False

            # R2 URL → 다운로드
            if output.get("converted_audio_url"):
                resp = requests.get(output["converted_audio_url"], timeout=300)
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                has_vocals = True
                print(f"[Convert] Downloaded vocals from R2: {len(resp.content):,} bytes")
            # inline base64
            elif output.get("converted_audio"):
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(output["converted_audio"]))
                has_vocals = True

            if has_vocals:
                result_data = {
                    "output_file": out_filename,
                    "processing_time": output.get("processing_time_seconds", 0),
                }

                # 믹스 파일도 저장 — R2 URL 또는 inline base64
                mixed_filename = output.get("mixed_filename", f"mixed_{job_id[:8]}.wav")
                mixed_path = OUTPUT_DIR / mixed_filename
                if output.get("mixed_audio_url"):
                    resp = requests.get(output["mixed_audio_url"], timeout=300)
                    resp.raise_for_status()
                    with open(mixed_path, "wb") as f:
                        f.write(resp.content)
                    result_data["mixed_file"] = mixed_filename
                    print(f"[Convert] Downloaded mixed from R2: {len(resp.content):,} bytes")
                elif output.get("mixed_audio"):
                    with open(mixed_path, "wb") as f:
                        f.write(base64.b64decode(output["mixed_audio"]))
                    result_data["mixed_file"] = mixed_filename

                # conversions 테이블에 결과 파일 업데이트 (job_id 기반으로 정확한 row 매칭)
                with get_db() as db:
                    db.execute("""
                        UPDATE conversions SET output_file=?, output_name=?,
                            processing_time=?, status='completed'
                        WHERE job_id=?
                    """, (out_filename, result_data.get("mixed_file", out_filename),
                          output.get("processing_time_seconds", 0), job_id))

                update_job(job_id, status="completed", progress=100,
                          message="변환 완료!",
                          result_json=json.dumps(result_data))
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
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

@app.on_event("startup")
async def startup():
    init_db()
    cleanup_stale_jobs()


# ─── 메인 페이지 ───

@app.get("/")
async def root():
    return FileResponse(str(APP_DIR / "static" / "index.html"))

@app.get("/favicon.ico")
async def favicon():
    favicon_path = APP_DIR / "static" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path), media_type="image/x-icon")
    # Return empty 204 to suppress browser 404 warnings
    return Response(status_code=204)


# ─── 설정 API ───

@app.get("/api/config")
async def get_config():
    config = load_config()
    r2_secret = config.get("r2_secret_access_key", "")
    return {
        "runpod_api_key": "***" + config.get("runpod_api_key", "")[-4:] if config.get("runpod_api_key") else "",
        "runpod_endpoint_id": config.get("runpod_endpoint_id", ""),
        "is_configured": bool(config.get("runpod_api_key") and config.get("runpod_endpoint_id")),
        "r2_endpoint_url": config.get("r2_endpoint_url", ""),
        "r2_access_key_id": config.get("r2_access_key_id", ""),
        "r2_secret_key_display": "***" + r2_secret[-4:] if r2_secret else "",
        "r2_bucket_name": config.get("r2_bucket_name", "voice-studio"),
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

@app.post("/api/config/r2")
async def set_r2_config(
    r2_endpoint_url: str = Form(""),
    r2_access_key_id: str = Form(""),
    r2_secret_access_key: str = Form(""),
    r2_bucket_name: str = Form("voice-studio"),
):
    config = load_config()
    if r2_endpoint_url:
        config["r2_endpoint_url"] = r2_endpoint_url.rstrip("/")
    if r2_access_key_id:
        config["r2_access_key_id"] = r2_access_key_id
    if r2_secret_access_key and not r2_secret_access_key.startswith("***"):
        config["r2_secret_access_key"] = r2_secret_access_key
    if r2_bucket_name:
        config["r2_bucket_name"] = r2_bucket_name
    save_config(config)
    return {"status": "ok"}


# ─── 파일 업로드 API ───

@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    category: str = Form("vocal")
):
    saved = []
    skipped = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".mkv", ".webm"]:
            continue

        content = await f.read()
        file_hash = hashlib.sha256(content).hexdigest()

        # 동일 파일 중복 체크 (활성 파일)
        with get_db() as db:
            live_dup = db.execute(
                "SELECT id, original_name FROM training_files WHERE file_hash=? AND deleted=0",
                (file_hash,)
            ).fetchone()
        if live_dup:
            skipped.append({"filename": f.filename,
                            "existing_name": live_dup["original_name"]})
            continue

        # 삭제된 파일 중 같은 해시의 전처리 상태 확인
        with get_db() as db:
            prev = db.execute(
                "SELECT preprocessed FROM training_files WHERE file_hash=? AND deleted=1 ORDER BY id DESC LIMIT 1",
                (file_hash,)
            ).fetchone()
        was_preprocessed = 1 if (prev and prev["preprocessed"]) else 0

        unique_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
        save_path = UPLOAD_DIR / unique_name

        with open(save_path, "wb") as fp:
            fp.write(content)

        # 오디오 길이 측정
        duration = _get_audio_duration(save_path)

        with get_db() as db:
            db.execute("""
                INSERT INTO training_files (filename, original_name, file_size, duration_seconds, file_type, category, file_hash, preprocessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (unique_name, f.filename, len(content), duration, ext, category, file_hash, was_preprocessed))
            file_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        saved.append({
            "id": file_id,
            "filename": unique_name,
            "original_name": f.filename,
            "size": len(content),
            "type": ext,
            "category": category,
            "preprocessed": was_preprocessed,
            "url": f"/uploads/{unique_name}",
        })

    return {"files": saved, "count": len(saved), "skipped": skipped}


@app.get("/api/files")
async def list_files():
    with get_db() as db:
        rows = db.execute("SELECT * FROM training_files WHERE deleted=0 ORDER BY uploaded_at DESC").fetchall()
    files = []
    for r in rows:
        d = dict(r)
        # 프론트엔드 필드명 통일 (DB: file_size/file_type → size/type)
        d["size"] = d.get("file_size") or 0
        d["type"] = d.get("file_type") or ""
        d["url"] = f"/uploads/{d['filename']}"
        files.append(d)
    return {"files": files}


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: int):
    with get_db() as db:
        row = db.execute("SELECT filename FROM training_files WHERE id=? AND deleted=0",
                         (file_id,)).fetchone()
        if row:
            filepath = UPLOAD_DIR / row["filename"]
            if filepath.exists():
                filepath.unlink()
            # Soft delete: 해시 기록을 보존하여 재업로드 시 전처리 상태 복원 가능
            db.execute("UPDATE training_files SET deleted=1 WHERE id=?", (file_id,))
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

        # 청크 병합 → 임시 파일
        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        save_path = UPLOAD_DIR / unique_name
        total_size = 0
        hash_obj = hashlib.sha256()

        with open(save_path, "wb") as out_f:
            for i in range(total_chunks):
                cp = chunk_dir / f"chunk_{i:06d}"
                with open(cp, "rb") as in_f:
                    data = in_f.read()
                    total_size += len(data)
                    hash_obj.update(data)
                    out_f.write(data)

        if total_size > MAX_CHUNK_FILE_SIZE:
            save_path.unlink(missing_ok=True)
            shutil.rmtree(chunk_dir, ignore_errors=True)
            chunk_uploads.pop(upload_id, None)
            raise HTTPException(413, RUNPOD_ERROR_MESSAGES["file_too_large"])

        file_hash = hash_obj.hexdigest()

        # 동일 파일 중복 체크
        with get_db() as db:
            dup = db.execute(
                "SELECT id, original_name FROM training_files WHERE file_hash=? AND deleted=0",
                (file_hash,)
            ).fetchone()
        if dup:
            save_path.unlink(missing_ok=True)
            shutil.rmtree(chunk_dir, ignore_errors=True)
            chunk_uploads.pop(upload_id, None)
            raise HTTPException(409,
                f"동일한 파일이 이미 존재합니다: '{dup['original_name']}'")

        shutil.rmtree(chunk_dir, ignore_errors=True)
        chunk_uploads.pop(upload_id, None)

        duration = _get_audio_duration(save_path)

        with get_db() as db:
            db.execute("""
                INSERT INTO training_files (filename, original_name, file_size, duration_seconds, file_type, category, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (unique_name, filename, total_size, duration, ext, category, file_hash))
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
    epochs: int = Form(500),
    sample_rate: int = Form(40000),  # 40k recommended for SVC quality
    batch_size: int = Form(0),  # 0 = GPU auto-detect (RTX 4090 → 24)
    f0_method: str = Form("rmvpe"),
    file_ids: str = Form("")  # comma-separated
):
    if not runpod_client.is_configured():
        raise HTTPException(400, "RunPod API 설정이 필요합니다. 설정 페이지에서 API Key와 Endpoint ID를 입력하세요.")

    # 모델 이름 중복 검사
    model_name = model_name.strip()
    if not model_name:
        raise HTTPException(400, "모델 이름을 입력해 주세요.")
    with get_db() as db:
        existing = db.execute(
            "SELECT id FROM voice_models WHERE name = ?", (model_name,)
        ).fetchone()
    if existing:
        raise HTTPException(400, f"'{model_name}' 이름의 모델이 이미 존재합니다. 다른 이름을 사용해 주세요.")

    # 학습 파일 수집
    with get_db() as db:
        if file_ids:
            ids = [int(x.strip()) for x in file_ids.split(",") if x.strip()]
            placeholders = ",".join("?" * len(ids))
            rows = db.execute(
                f"SELECT * FROM training_files WHERE id IN ({placeholders}) AND deleted=0", ids
            ).fetchall()
        else:
            rows = db.execute("SELECT * FROM training_files WHERE deleted=0").fetchall()

    if not rows:
        raise HTTPException(400, "학습할 파일이 없습니다.")

    # 전처리된 세그먼트가 있으면 우선 사용 (용량 작음 → 10MB 이내)
    # mr_/vocal_ 파일은 학습용이 아니므로 제외
    all_preprocessed = [f for f in _list_preprocessed_files()
                        if not f.name.startswith(("mr_", "vocal_"))]

    if all_preprocessed and file_ids:
        # 메타데이터의 file_id → segment 매핑으로 필터링
        meta_path = PREPROCESSED_DIR / "_metadata.json"
        seg_map = {}
        if meta_path.exists():
            try:
                seg_map = json.loads(meta_path.read_text()).get("file_segments", {})
            except Exception:
                pass

        ids = [int(x.strip()) for x in file_ids.split(",") if x.strip()]
        selected_seg_names = set()
        for fid in ids:
            selected_seg_names.update(seg_map.get(str(fid), []))

        if selected_seg_names:
            preprocessed_files = [
                f for f in all_preprocessed
                if f.name in selected_seg_names
            ]
            print(f"[Train] Segment filter: {len(selected_seg_names)} mapped → {len(preprocessed_files)} found on disk")
        else:
            # 매핑 없으면 (이전 전처리) 전체 사용
            print(f"[Train] No segment mapping found, using all {len(all_preprocessed)} segments")
            preprocessed_files = all_preprocessed
    else:
        preprocessed_files = all_preprocessed

    if preprocessed_files:
        file_paths = [str(p) for p in preprocessed_files]
    else:
        file_paths = [str(UPLOAD_DIR / r["filename"]) for r in rows]

    total_segments = len(file_paths)

    # 학습에 사용된 원본 파일명 기록
    training_file_names = [r["original_name"] for r in rows]

    # Job 생성
    job_id = uuid.uuid4().hex[:12]
    _training_file_map[job_id] = training_file_names
    with get_db() as db:
        db.execute("""
            INSERT INTO jobs (id, job_type, status, progress, message)
            VALUES (?, 'train', 'submitting', 0, '작업 제출 중...')
        """, (job_id,))

    try:
        # 학습 데이터를 R2에 업로드하여 10MB 페이로드 한도 회피
        audio_urls = []
        total_size = sum(Path(p).stat().st_size for p in file_paths)
        use_r2 = total_size > 5_000_000  # 5MB 이상이면 R2 사용

        if use_r2:
            print(f"[Train] Uploading {len(file_paths)} files ({total_size:,} bytes) to R2...")
            for i, fp in enumerate(file_paths):
                r2_key = f"voice-studio/train/{job_id}/{Path(fp).name}"
                try:
                    url = upload_to_r2(Path(fp), r2_key)
                    audio_urls.append({"filename": Path(fp).name, "url": url})
                except Exception as e:
                    print(f"[Train] R2 upload failed for {fp}: {e}")
                    update_job(job_id, status="failed",
                        message=f"학습 데이터 R2 업로드 실패: {e}")
                    return {"job_id": job_id}
            print(f"[Train] All {len(audio_urls)} files uploaded to R2")

        if audio_urls:
            # R2 URL 방식
            payload = {
                "task_type": "train",
                "model_name": model_name,
                "audio_urls": audio_urls,
                "sample_rate": sample_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "f0_method": f0_method,
            }
        else:
            # base64 인라인 (소용량)
            batches = prepare_files_for_runpod(file_paths)
            if len(batches) > 1:
                raise HTTPException(400,
                    f"학습 데이터가 너무 큽니다. R2 스토리지를 설정하거나 파일 수를 줄여주세요.")
            payload = {
                "task_type": "train",
                "model_name": model_name,
                "audio_files": batches[0],
                "sample_rate": sample_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "f0_method": f0_method,
            }

        payload_size = len(json.dumps(payload))
        print(f"[Train] Payload size: {payload_size:,} bytes, segments: {total_segments}")

        runpod_job_id = runpod_client.submit_job(payload)

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
        _training_file_map.pop(job_id, None)

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
        preprocess_file_map.pop(job_id, None)

    return {"job_id": job_id, "segments": total_segments, "batches": len(batches),
            "skipped": skipped, "processing": len(unprocessed)}


@app.get("/api/preprocess/status")
async def preprocess_status():
    """전처리 상태 확인 — 모든 파일이 전처리되었는지 DB + 디스크 기반으로 판단."""
    files = _list_preprocessed_files()

    # DB에서 전처리되지 않은 파일 수 확인
    with get_db() as db:
        total_files = db.execute("SELECT COUNT(*) FROM training_files WHERE deleted=0").fetchone()[0]
        unprocessed = db.execute(
            "SELECT COUNT(*) FROM training_files WHERE preprocessed=0 AND deleted=0"
        ).fetchone()[0]

    # Categorize files: training segments vs MR vs vocals
    training_segments = []
    accomp_files = []
    vocal_files = []
    for f in files:
        if f.name.startswith("mr_"):
            accomp_files.append({"filename": f.name, "size": f.stat().st_size})
        elif f.name.startswith("vocal_"):
            vocal_files.append({"filename": f.name, "size": f.stat().st_size})
        else:
            training_segments.append(f)

    has_segments = len(training_segments) > 0
    all_processed = total_files > 0 and unprocessed == 0
    processed_count = total_files - unprocessed

    # 메타데이터 파일에서 정확한 총 길이 읽기 (핸들러가 계산한 값)
    total_dur = 0.0
    meta_path = PREPROCESSED_DIR / "_metadata.json"
    if meta_path.exists():
        try:
            total_dur = json.loads(meta_path.read_text()).get("total_duration", 0.0)
        except Exception:
            pass

    # 메타데이터 없으면 WAV 파일에서 직접 계산 (폴백)
    if total_dur == 0.0 and has_segments:
        import wave
        for f in training_segments:
            try:
                if f.suffix.lower() == ".wav":
                    with wave.open(str(f), "rb") as wf:
                        total_dur += wf.getnframes() / wf.getframerate()
                elif f.suffix.lower() == ".mp3":
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_mp3(str(f))
                        total_dur += len(audio) / 1000.0
                    except Exception:
                        # MP3 192kbps ≈ 24000 bytes/sec
                        total_dur += f.stat().st_size / 24000.0
                else:
                    # Fallback for other formats: try ffprobe
                    dur = _get_audio_duration(f)
                    if dur:
                        total_dur += dur
            except Exception:
                pass

    return {
        "preprocessed": all_processed,
        "has_segments": has_segments,
        "segment_count": len(training_segments),
        "total_duration": round(total_dur, 2),
        "unprocessed_count": unprocessed,
        "processed_count": processed_count,
        "accompaniment_files": accomp_files,
        "vocal_files": vocal_files,
    }


@app.get("/api/preprocess/download/{filename}")
async def download_preprocessed_file(filename: str):
    """전처리된 파일 다운로드 (반주/보컬/세그먼트)."""
    # Path traversal 방지
    safe_name = Path(filename).name
    filepath = PREPROCESSED_DIR / safe_name
    if not filepath.exists():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    return FileResponse(str(filepath), filename=safe_name)


@app.delete("/api/preprocess")
async def clear_preprocess():
    """전처리 결과 삭제 — preprocessed/ 비우기 + DB preprocessed 플래그 리셋"""
    count = 0
    for f in PREPROCESSED_DIR.iterdir():
        if f.is_file():
            f.unlink()
            count += 1
    with get_db() as db:
        # 활성 파일만 리셋 (soft-deleted 파일은 유지 → 재업로드 시 전처리 상태 복원용)
        db.execute("UPDATE training_files SET preprocessed=0 WHERE deleted=0")
    return {"cleared": count}


@app.post("/api/preprocess/reset")
async def reset_preprocess_selected(file_ids: str = Form(...)):
    """선택한 파일의 전처리 상태만 초기화 — 해당 파일의 세그먼트 삭제 + DB 플래그 리셋"""
    ids = [int(x.strip()) for x in file_ids.split(",") if x.strip()]
    if not ids:
        raise HTTPException(400, "초기화할 파일을 선택하세요.")

    # 선택한 파일의 stem 목록 조회
    with get_db() as db:
        placeholders = ",".join("?" * len(ids))
        rows = db.execute(
            f"SELECT filename FROM training_files WHERE id IN ({placeholders}) AND deleted=0", ids
        ).fetchall()

    stems = set()
    for r in rows:
        stems.add(Path(r["filename"]).stem)

    # 해당 stem으로 시작하는 전처리 세그먼트 삭제
    removed = 0
    if stems:
        for f in PREPROCESSED_DIR.iterdir():
            if f.is_file() and f.name != "_metadata.json":
                if any(f.name.startswith(stem) for stem in stems):
                    f.unlink()
                    removed += 1

    # DB 플래그 리셋
    with get_db() as db:
        placeholders = ",".join("?" * len(ids))
        db.execute(
            f"UPDATE training_files SET preprocessed=0 WHERE id IN ({placeholders})", ids
        )

    # 메타데이터 갱신 (남은 세그먼트 기준으로 재계산)
    meta_path = PREPROCESSED_DIR / "_metadata.json"
    if meta_path.exists():
        meta_path.unlink()

    return {"cleared": removed, "reset_files": len(rows)}


# ─── 변환 API ───

@app.post("/api/convert")
async def start_conversion(
    model_id: int = Form(...),
    pitch_shift: int = Form(0),
    index_rate: float = Form(0.88),
    vocal_volume: float = Form(1.0),
    mr_volume: float = Form(1.0),
    clean_audio: bool = Form(True),
    clean_strength: float = Form(0.7),
    protect: float = Form(0.23),
    rms_mix_rate: float = Form(0.1),
    filter_radius: int = Form(3),
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

    # 모델 URL 또는 base64 준비
    pth_url = model["pth_url"] if model["pth_url"] else None
    index_url = model["index_url"] if model["index_url"] else None

    # Job 생성
    job_id = uuid.uuid4().hex[:12]
    with get_db() as db:
        db.execute("""
            INSERT INTO jobs (id, job_type, status, progress, message)
            VALUES (?, 'convert', 'submitting', 0, '작업 제출 중...')
        """, (job_id,))

    try:
        payload = {
            "task_type": "convert",
            "audio_filename": audio.filename,
            "pitch_shift": pitch_shift,
            "index_rate": index_rate,
            "f0_method": "rmvpe",
            "clean_audio": clean_audio,
            "clean_strength": clean_strength,
            "protect": protect,
            "rms_mix_rate": rms_mix_rate,
            "filter_radius": filter_radius,
            "separate_vocals": True,
            "vocal_volume": vocal_volume,
            "mr_volume": mr_volume,
        }

        # 오디오 파일을 R2에 업로드 (10 MB 페이로드 한도 회피)
        audio_r2_key = f"voice-studio/convert/{job_id}/{audio.filename}"
        try:
            audio_url = upload_to_r2(temp_path, audio_r2_key)
            payload["audio_url"] = audio_url
        except Exception as e:
            print(f"[Convert] R2 audio upload failed: {e}")
            file_size = temp_path.stat().st_size
            if file_size > 7_000_000:  # 7MB 이상이면 base64로도 10MB 초과
                update_job(job_id, status="failed",
                    message=f"R2 업로드 실패 (파일 {file_size//1_000_000}MB): {e}")
                return {"job_id": job_id}
            # 소용량 파일만 base64 폴백
            with open(temp_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            payload["audio_data"] = audio_b64

        # R2 URL로 모델 전달 (base64 인라인 대신)
        if pth_url:
            payload["pth_url"] = pth_url
        if index_url:
            payload["index_url"] = index_url

        # pth_url 없는 모델 (이전 학습 모델) → 로컬 파일을 R2에 업로드
        if not pth_url and model["pth_path"]:
            local_pth = Path(model["pth_path"])
            if not local_pth.is_absolute():
                local_pth = MODEL_DIR / local_pth
            if local_pth.exists():
                try:
                    pth_r2_key = f"voice-studio/convert/{job_id}/model.pth"
                    pth_url = upload_to_r2(local_pth, pth_r2_key)
                    payload["pth_url"] = pth_url
                    print(f"[Convert] Uploaded model pth to R2: {pth_r2_key}")
                except Exception as e:
                    print(f"[Convert] R2 model upload failed: {e}")
                    update_job(job_id, status="failed",
                        message=f"모델 R2 업로드 실패: {e}")
                    return {"job_id": job_id}

        if not index_url and model["index_path"]:
            local_idx = Path(model["index_path"])
            if not local_idx.is_absolute():
                local_idx = MODEL_DIR / local_idx
            if local_idx.exists():
                try:
                    idx_r2_key = f"voice-studio/convert/{job_id}/model.index"
                    index_url = upload_to_r2(local_idx, idx_r2_key)
                    payload["index_url"] = index_url
                    print(f"[Convert] Uploaded model index to R2: {idx_r2_key}")
                except Exception as e:
                    print(f"[Convert] R2 index upload failed: {e}")
                    # index는 선택사항이므로 계속 진행

        # 페이로드 크기 로깅 (디버깅용)
        payload_keys = {k: (len(v) if isinstance(v, str) and len(v) > 100 else v)
                        for k, v in payload.items()}
        payload_size = len(json.dumps(payload))
        print(f"[Convert] Payload size: {payload_size:,} bytes, keys: {payload_keys}")

        runpod_job_id = runpod_client.submit_job(payload)

        update_job(job_id, status="running", progress=10,
                  message="GPU 변환 시작", runpod_job_id=runpod_job_id)

        # DB에 변환 기록 (job_id로 정확한 매핑)
        with get_db() as db:
            db.execute("""
                INSERT INTO conversions (model_id, input_file, status, job_id)
                VALUES (?, ?, 'processing', ?)
            """, (model_id, audio.filename, job_id))

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
        if not row:
            raise HTTPException(404, "모델을 찾을 수 없습니다.")
        dirs_to_check = set()
        for path_key in ["pth_path", "index_path"]:
            if row[path_key] and Path(row[path_key]).exists():
                dirs_to_check.add(Path(row[path_key]).parent)
                Path(row[path_key]).unlink()
        # 빈 모델 디렉토리 정리
        for d in dirs_to_check:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        db.execute("DELETE FROM voice_models WHERE id=?", (model_id,))
    return {"status": "ok"}


@app.post("/api/models/{model_id}/rename")
async def rename_model(model_id: int, name: str = Form(...)):
    """모델 이름 변경"""
    if not name or not name.strip():
        raise HTTPException(400, "모델 이름을 입력하세요.")
    clean_name = name.strip()
    with get_db() as db:
        row = db.execute("SELECT * FROM voice_models WHERE id=?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(404, "모델을 찾을 수 없습니다.")
        dup = db.execute("SELECT id FROM voice_models WHERE name=? AND id!=?", (clean_name, model_id)).fetchone()
        if dup:
            raise HTTPException(400, f'이미 "{clean_name}" 이름의 모델이 존재합니다.')
        db.execute("UPDATE voice_models SET name=? WHERE id=?", (clean_name, model_id))
    return {"status": "ok", "name": clean_name}


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


@app.delete("/api/conversions/{conv_id}")
async def delete_conversion(conv_id: int):
    with get_db() as db:
        row = db.execute("SELECT * FROM conversions WHERE id=?", (conv_id,)).fetchone()
        if not row:
            raise HTTPException(404, "변환 기록을 찾을 수 없습니다.")
        for col in ("output_file", "output_name"):
            fname = row[col]
            if fname:
                fpath = OUTPUT_DIR / fname
                if fpath.exists():
                    fpath.unlink()
        db.execute("DELETE FROM conversions WHERE id=?", (conv_id,))
    return {"status": "ok"}


@app.get("/api/download/{filename:path}")
async def download_file(filename: str):
    # Path traversal 방지
    safe_name = Path(filename).name
    filepath = OUTPUT_DIR / safe_name
    if not filepath.exists():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    # application/octet-stream forces browser download (prevents in-browser audio playback)
    return FileResponse(str(filepath), filename=safe_name, media_type="application/octet-stream")


# ─── 대시보드 통계 API ───

@app.get("/api/stats")
async def get_stats():
    with get_db() as db:
        file_count = db.execute("SELECT COUNT(*) FROM training_files WHERE deleted=0").fetchone()[0]
        model_count = db.execute("SELECT COUNT(*) FROM voice_models WHERE status='ready'").fetchone()[0]
        conv_count = db.execute("SELECT COUNT(*) FROM conversions").fetchone()[0]
        total_size = db.execute("SELECT COALESCE(SUM(file_size),0) FROM training_files WHERE deleted=0").fetchone()[0]
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
    print("\nAI Voice Studio 시작 중...")
    print("   http://localhost:8000 에서 접속하세요\n")

    if not FROZEN:
        # 개발 모드: 브라우저 자동 열기 (.exe 모드에서는 pywebview가 처리)
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
