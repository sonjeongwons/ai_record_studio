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
import threading
import webbrowser
import uuid
import re
import subprocess
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import uvicorn
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

# 청크 업로드 상태 추적 (메모리 내, upload_id → {total, received, filename, category, started_at})
chunk_uploads: dict = {}
_chunk_lock = threading.Lock()  # 청크 업로드 동시성 보호

# 전처리 작업별 원본 파일 ID 추적 (완료 시 preprocessed=1 마킹용)
preprocess_file_map: dict[str, list[int]] = {}
_preprocess_lock = threading.Lock()  # preprocess_file_map 동시성 보호

# RunPod 폴링 에러 카운터 (job_id별 추적)
_poll_error_counts: dict[str, int] = {}


def _cleanup_stale_chunk_uploads():
    """24시간 이상 완료되지 않은 청크 업로드 정리 (메모리 + 임시 파일)"""
    now = time.time()
    with _chunk_lock:
        stale_ids = [uid for uid, sess in list(chunk_uploads.items())
                     if now - sess.get("started_at", now) > 86400]
        for uid in stale_ids:
            chunk_uploads.pop(uid, None)
    for uid in stale_ids:
        chunk_dir = CHUNK_DIR / uid
        shutil.rmtree(chunk_dir, ignore_errors=True)
    if stale_ids:
        print(f"[ChunkCleanup] {len(stale_ids)}개 미완료 업로드 정리")

def _list_preprocessed_files() -> list[Path]:
    """전처리된 오디오 파일 목록 (WAV + MP3 + FLAC)."""
    files: list[Path] = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
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

DEFAULT_CONFIG = {
    "runpod_api_key": "",
    "runpod_endpoint_id": "",
    "r2_endpoint_url": "",
    "r2_access_key_id": "",
    "r2_secret_access_key": "",
    "r2_bucket_name": "",
    "download_folder": str(Path.home() / "Downloads"),
}

def load_config():
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            cfg.update(saved)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # 손상된 config 백업 후 초기화
            backup = CONFIG_PATH.with_suffix(".json.bak")
            shutil.copy2(str(CONFIG_PATH), str(backup))
            print(f"[Warning] config.json 손상 → 백업: {backup}")
    return cfg

def save_config(config):
    # 원자적 쓰기: 임시 파일에 먼저 쓰고 rename (크래시 시 손상 방지)
    tmp_path = CONFIG_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    tmp_path.replace(CONFIG_PATH)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SQLite 데이터베이스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
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
        # jobs에 pause_state_json 컬럼 추가 (일시정지 재개용)
        try:
            db.execute("ALTER TABLE jobs ADD COLUMN pause_state_json TEXT")
        except Exception:
            pass
        # jobs에 started_at 컬럼 추가 (경과 시간 계산용)
        try:
            db.execute("ALTER TABLE jobs ADD COLUMN started_at TEXT")
        except Exception:
            pass
        # file_hash 인덱스 (중복 체크 성능)
        try:
            db.execute("CREATE INDEX IF NOT EXISTS idx_training_files_hash ON training_files(file_hash)")
        except Exception:
            pass
        # 성능 인덱스 추가
        db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_conversions_model_id ON conversions(model_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_conversions_job_id ON conversions(job_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_training_files_deleted ON training_files(deleted)")


def cleanup_stale_jobs_on_startup():
    """서버 시작 시 1회만 호출. 모든 running/submitting 작업을 실패 처리.
    서버 재시작 시 폴링 스레드가 사라지므로 진행 중 작업을 정리."""
    with get_db() as db:
        # RunPod 작업 ID 먼저 수집 → 실제 취소
        stale_rows = db.execute("""
            SELECT id, runpod_job_id FROM jobs
            WHERE status IN ('running', 'submitting') AND runpod_job_id IS NOT NULL
        """).fetchall()
        if stale_rows and runpod_client.is_configured():
            for row in stale_rows:
                try:
                    runpod_client.cancel_runpod_job(row["runpod_job_id"])
                except Exception:
                    pass
        now_iso = datetime.now().isoformat()
        active = db.execute("""
            UPDATE jobs
            SET status='failed',
                message='서버 재시작으로 인한 자동 정리',
                updated_at=?
            WHERE status IN ('running', 'submitting')
        """, (now_iso,))
        if active.rowcount > 0:
            print(f"  정리된 진행 중 작업: {active.rowcount}개")
        # conversions 테이블도 동기화 — orphan 'processing' 레코드 정리
        db.execute("""
            UPDATE conversions SET status='failed'
            WHERE status='processing'
            AND job_id NOT IN (
                SELECT id FROM jobs WHERE status NOT IN ('failed', 'cancelled', 'paused')
            )
        """)


def restore_paused_jobs_state():
    """서버 시작 시 paused 작업의 재개 상태를 메모리에 복원.
    서버 재시작 후에도 resume이 가능하도록 _active_job_states를 DB에서 채움."""
    with get_db() as db:
        rows = db.execute(
            "SELECT id, pause_state_json FROM jobs WHERE status='paused' AND pause_state_json IS NOT NULL"
        ).fetchall()
    count = 0
    for row in rows:
        try:
            state = json.loads(row["pause_state_json"])
            if state:
                with _job_states_lock:
                    _active_job_states[row["id"]] = state
                count += 1
        except Exception:
            pass
    if count > 0:
        print(f"  일시정지 작업 상태 복원: {count}개")


def cleanup_stale_jobs():
    """주기적 호출용. 1시간 이상 방치된 작업만 실패 처리.
    진행 중인 정상 작업은 건드리지 않음."""
    with get_db() as db:
        # RunPod 작업 ID 먼저 수집 → 실제 취소 (과금 중지)
        stale_rows = db.execute("""
            SELECT id, runpod_job_id FROM jobs
            WHERE status IN ('running', 'submitting')
              AND runpod_job_id IS NOT NULL
              AND updated_at < datetime('now', 'localtime', '-1 hour')
        """).fetchall()
        if stale_rows and runpod_client.is_configured():
            for row in stale_rows:
                try:
                    runpod_client.cancel_runpod_job(row["runpod_job_id"])
                except Exception:
                    pass
        stale = db.execute("""
            UPDATE jobs
            SET status='failed',
                message='1시간 이상 응답 없음 (자동 정리)',
                updated_at=?
            WHERE status IN ('running', 'submitting')
              AND updated_at < datetime('now', 'localtime', '-1 hour')
        """, (datetime.now().isoformat(),))
        if stale.rowcount > 0:
            print(f"  정리된 멈춘 작업: {stale.rowcount}개")


# 주기적 정리 스로틀 (30분 간격)
_last_cleanup_time: float = 0.0


def maybe_cleanup_stale_jobs():
    """30분 이상 경과 시 stale job 정리 (health 엔드포인트에서 호출).
    1시간 이상 방치된 작업만 정리 — 진행 중인 작업은 건드리지 않음."""
    global _last_cleanup_time
    now = time.time()
    if now - _last_cleanup_time > 1800:  # 30분
        _last_cleanup_time = now
        cleanup_stale_jobs()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RunPod 에러 메시지 (한국어)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RUNPOD_ERROR_MESSAGES = {
    "timeout": "네트워크 연결 실패. 인터넷 연결을 확인하세요.",
    "auth": "RunPod API 키가 올바르지 않습니다. 설정에서 API 키를 확인하세요.",
    "funds": "RunPod 잔액이 부족합니다. RunPod 대시보드에서 충전하세요.",
    "file_too_large": "파일 크기가 제한(2GB)을 초과합니다. 더 작은 파일을 사용하세요.",
    "unknown": "RunPod 서버 오류가 발생했습니다. 잠시 후 다시 시도하세요. (상세: {detail})",
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
        """비동기 작업 제출 → job_id 반환 (재시도 포함)
        실행 시간 제한은 RunPod 대시보드의 Execution Timeout 설정을 따름.
        """
        body = {"input": payload}
        resp = self._request_with_retry(
            "POST",
            f"{self.base_url}/run",
            headers=self.headers,
            json=body,
            timeout=120
        )
        job_id = resp.json().get("id")
        if not job_id:
            raise RuntimeError(f"RunPod 응답에 job ID가 없습니다: {resp.text[:200]}")
        return job_id

    def check_status(self, runpod_job_id: str) -> dict:
        """작업 상태 확인 (재시도 포함)"""
        resp = self._request_with_retry(
            "GET",
            f"{self.base_url}/status/{runpod_job_id}",
            headers=self.headers,
            timeout=30
        )
        return resp.json()

    def cancel_runpod_job(self, runpod_job_id: str) -> bool:
        """RunPod 서버리스 작업 실제 취소 — 과금 즉시 중지.
        재시도 로직 적용 (과금 중지가 중요하므로).
        실패해도 예외를 던지지 않고 False 반환."""
        if not runpod_job_id or not self.is_configured():
            return False
        try:
            resp = self._request_with_retry(
                "POST",
                f"{self.base_url}/cancel/{runpod_job_id}",
                headers=self.headers,
                timeout=30
            )
            print(f"[RunPod] Cancel {runpod_job_id}: HTTP {resp.status_code}")
            return resp.status_code in (200, 201, 202)
        except Exception as e:
            print(f"[RunPod] Cancel failed for {runpod_job_id}: {e}")
            return False

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
_training_lock = threading.Lock()  # _training_file_map 동시성 보호

# 활성 작업 상태 (일시정지/재개용 파라미터 저장, job_id → dict)
_active_job_states: dict[str, dict] = {}
_job_states_lock = threading.Lock()  # _active_job_states 동시성 보호

# RunPod 폴링 에러 카운터 동시성 보호
_poll_errors_lock = threading.Lock()

# 작업 유형별 타임아웃 (초)
_JOB_TIMEOUTS = {"train": 36000, "preprocess": 3600, "convert": 3600}  # convert: 30분→60분


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
            ExpiresIn=86400 * 30,  # 30일 (일시정지 후 재개, 장기 보관 지원)
        )
    except Exception as e:
        print(f"[R2 Upload Error] {type(e).__name__}: {e}")
        raise HTTPException(500, f"R2 업로드 실패: {type(e).__name__}: {e}")

    return presigned_url

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 백그라운드 작업 관리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_JOB_UPDATE_COLS = frozenset({"status", "progress", "message", "result_json",
                               "runpod_job_id", "pause_state_json", "started_at"})

def update_job(job_id: str, **kwargs):
    invalid = set(kwargs) - _JOB_UPDATE_COLS
    if invalid:
        raise ValueError(f"update_job: 허용되지 않은 컬럼 {invalid}")
    try:
        with get_db() as db:
            sets = ", ".join(f"{k}=?" for k in kwargs)
            vals = list(kwargs.values())
            # status가 running으로 전환될 때 started_at을 최초 1회 기록 (COALESCE로 덮어쓰기 방지)
            if kwargs.get("status") == "running" and "started_at" not in kwargs:
                sets += ", started_at=COALESCE(started_at, ?)"
                vals.append(datetime.now().isoformat())
            vals.append(datetime.now().isoformat())  # updated_at
            vals.append(job_id)
            # 터미널 상태 우선순위: completed > cancelled > failed
            # completed는 절대 덮어쓰지 않고, cancelled는 completed에 의해서만 덮어씀
            new_status = kwargs.get("status", "")
            if new_status == "completed":
                # completed는 cancelled/failed도 덮어쓸 수 있음 (정상 완료 우선)
                db.execute(f"UPDATE jobs SET {sets}, updated_at=? WHERE id=?", vals)
            elif new_status in ("cancelled", "failed"):
                # cancelled/failed는 completed를 덮어쓰지 않음
                db.execute(
                    f"UPDATE jobs SET {sets}, updated_at=? WHERE id=? "
                    f"AND status != 'completed'", vals)
            else:
                db.execute(
                    f"UPDATE jobs SET {sets}, updated_at=? WHERE id=? "
                    f"AND status NOT IN ('cancelled', 'failed', 'completed')", vals)
    except Exception as e:
        print(f"[update_job] DB 업데이트 실패 (job_id={job_id}): {e}")


def _is_job_cancelled(job_id: str) -> bool:
    """작업이 취소/실패/일시정지 상태인지 확인 (폴링 루프 종료용)"""
    try:
        with get_db() as db:
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            return row and row["status"] in ("failed", "cancelled", "paused")
    except Exception:
        return False


def poll_runpod_job(job_id: str, runpod_job_id: str, job_type: str):
    """RunPod 작업 완료까지 폴링"""
    start_time = time.time()
    try:  # 외부 try: 폴링 스레드 크래시 시 작업을 failed로 마킹 (무한 running 방지)
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
                with _poll_errors_lock:
                    _poll_error_counts.pop(job_id, None)
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
                    with _job_states_lock:
                        _active_job_states.pop(job_id, None)
                    with _poll_errors_lock:
                        _poll_error_counts.pop(job_id, None)
                    return
                try:
                    handle_job_result(job_id, job_type, output)
                    # handle_job_result의 finally 블록에서 _active_job_states 정리됨
                except Exception as e:
                    update_job(job_id, status="failed",
                              message=f"결과 처리 실패: {e}")
                    if job_type == "convert":
                        try:
                            with get_db() as db:
                                db.execute("UPDATE conversions SET status='failed' WHERE job_id=?", (job_id,))
                        except Exception:
                            pass
                    with _job_states_lock:
                        _active_job_states.pop(job_id, None)
                    with _poll_errors_lock:
                        _poll_error_counts.pop(job_id, None)
                return

            elif status in ("FAILED", "TIMED_OUT", "CANCELLED"):
                with _poll_errors_lock:
                    _poll_error_counts.pop(job_id, None)
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
                with _job_states_lock:
                    _active_job_states.pop(job_id, None)
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
                        if m2 and int(m2.group(2)) > 0:
                            pct = min(95, int(int(m2.group(1)) / int(m2.group(2)) * 100))

                    # 한국어 메시지 변환
                    if "epoch" in progress_text.lower():
                        m3 = re.search(r'epoch\s+(\d+)\s*/\s*(\d+)', progress_text, re.IGNORECASE)
                        if m3:
                            msg = f"학습 중... 에폭 {m3.group(1)}/{m3.group(2)}"
                    elif "(" in progress_text and "/" in progress_text:
                        # "(3/5)" 스텝 형태
                        m4 = re.search(r'\((\d+)/(\d+)\)', progress_text)
                        if m4 and int(m4.group(2)) > 0:
                            step, total_steps = int(m4.group(1)), int(m4.group(2))
                            pct = min(90, int(step / total_steps * 90))
                            msg = progress_text

                if pct is None:
                    # fallback: 경과 시간 기반 추정 (작업 유형별 예상 시간)
                    _EST_TIMES = {"train": 900, "preprocess": 600, "convert": 300}
                    est_total = _EST_TIMES.get(job_type, 600)
                    pct = min(90, int((elapsed / est_total) * 100))

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

            # 작업 타입별 타임아웃: train=10시간, preprocess=1시간, convert=30분
            job_timeout = _JOB_TIMEOUTS.get(job_type, 3600)
            if elapsed > job_timeout:
                with _poll_errors_lock:
                    _poll_error_counts.pop(job_id, None)
                with _job_states_lock:
                    _active_job_states.pop(job_id, None)
                timeout_label = f"{job_timeout // 3600}시간" if job_timeout >= 3600 else f"{job_timeout // 60}분"
                update_job(job_id, status="failed", message=f"시간 초과 ({timeout_label})")
                return

        except Exception as e:
            with _poll_errors_lock:
                poll_errors = _poll_error_counts.get(job_id, 0) + 1
                _poll_error_counts[job_id] = poll_errors
            if poll_errors > 60:  # 60회 연속 오류 = ~10분 장애 허용 (일시적 네트워크 불안정 대응)
                with _poll_errors_lock:
                    _poll_error_counts.pop(job_id, None)
                with _job_states_lock:
                    _active_job_states.pop(job_id, None)
                update_job(job_id, status="failed",
                          message=f"RunPod 상태 확인 반복 실패: {e}")
                return
            update_job(job_id, status="running",
                      message=f"상태 확인 중... (재시도 {poll_errors})")
            time.sleep(10)
    except Exception as _poll_fatal:
        # 폴링 스레드 예상치 못한 크래시 — 작업을 failed로 마킹하여 영구 running 상태 방지
        print(f"[CRITICAL] poll_runpod_job crashed for {job_id}: {_poll_fatal}")
        try:
            update_job(job_id, status="failed",
                      message=f"내부 오류로 작업이 중단되었습니다. 다시 시도해 주세요.")
            if job_type == "convert":
                with get_db() as db:
                    db.execute("UPDATE conversions SET status='failed' WHERE job_id=?", (job_id,))
        except Exception:
            pass
    finally:
        with _job_states_lock:
            _active_job_states.pop(job_id, None)
        with _poll_errors_lock:
            _poll_error_counts.pop(job_id, None)


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
                # Download from R2 presigned URL (재시도 포함)
                try:
                    _download_with_retry(fobj["url"], fpath, timeout=120)
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
    # pop 은 DB 업데이트 성공 후에 실행 (실패하면 재시도 가능하도록)
    with _preprocess_lock:
        file_ids = preprocess_file_map.get(job_id, [])
    if file_ids:
        try:
            with get_db() as db:
                placeholders = ",".join("?" * len(file_ids))
                db.execute(
                    f"UPDATE training_files SET preprocessed=1 WHERE id IN ({placeholders})",
                    file_ids
                )
        except Exception as e:
            print(f"[Preprocess] DB update failed for file_ids (will retry next poll): {e}")
        else:
            with _preprocess_lock:
                preprocess_file_map.pop(job_id, None)
    else:
        with _preprocess_lock:
            preprocess_file_map.pop(job_id, None)

    # 전체 전처리 세그먼트 수 (학습용 세그먼트만 카운트, mr_/vocal_ 제외)
    all_files = _list_preprocessed_files()
    training_files = [f for f in all_files if not f.name.startswith(("mr_", "vocal_"))]

    # 정확한 총 길이를 메타데이터 파일에 저장 (재시작 시 정확한 값 복원용)
    # + 파일 ID → 세그먼트 매핑 저장 (선택적 학습 시 필터링용)
    meta_path = PREPROCESSED_DIR / "_metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Warning] Failed to parse metadata: {e}")
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
    # 원자적 쓰기: 임시 파일에 먼저 쓰고 rename (크래시 시 손상 방지)
    meta_tmp = meta_path.with_suffix(".json.tmp")
    try:
        meta_tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        meta_tmp.replace(meta_path)
    except Exception as e:
        print(f"[Warning] metadata write failed: {e}")
        meta_tmp.unlink(missing_ok=True)

    return {
        "segment_count": len(training_files),
        "total_duration": round(merged_dur, 2),
        "segment_files": [f.name for f in training_files],
        "accompaniment_files": [s["filename"] for s in saved_accomp],
        "vocal_files": [s["filename"] for s in saved_vocals],
    }


def _download_with_retry(url: str, dest: Path, timeout: int = 300, retries: int = 3) -> None:
    """URL에서 파일 다운로드 (재시도 포함). GPU 결과물 손실 방지."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            return
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            if attempt < retries - 1:
                print(f"[Download] 재시도 {attempt + 1}/{retries}: {dest.name} ({e})")
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"다운로드 {retries}회 실패 ({dest.name}): {e}") from e


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
                _download_with_retry(output["pth_url"], Path(pth_path), timeout=300)
            # Fallback: base64 encoded model (when R2 upload failed)
            elif output.get("pth_base64"):
                pth_filename = output.get("pth_filename", f"{model_name}.pth")
                pth_path = str(model_dir / pth_filename)
                raw = base64.b64decode(output["pth_base64"])
                with open(pth_path, "wb") as f:
                    f.write(raw)
                print(f"[Train] Model saved via base64 fallback: {pth_filename}")
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

            # Check for upload failure (model too large for base64 + no R2)
            if output.get("upload_method") == "failed":
                error_msg = output.get("error", "모델 업로드 실패")
                shutil.rmtree(model_dir, ignore_errors=True)  # 빈 디렉토리 정리
                update_job(job_id, status="failed", message=error_msg)
                return

            # pth_path가 없으면 모델을 저장할 수 없음 → 실패 처리
            if pth_path is None:
                shutil.rmtree(model_dir, ignore_errors=True)
                update_job(job_id, status="failed",
                          message="모델 파일(.pth)을 받지 못했습니다. 학습 결과를 확인하거나 다시 학습해주세요.")
                return

            # Download index from presigned URL (primary path)
            if output.get("index_url"):
                idx_filename = output.get("index_filename", f"{model_name}.index")
                index_path = str(model_dir / idx_filename)
                _download_with_retry(output["index_url"], Path(index_path), timeout=120)
            # Fallback: base64 encoded index
            elif output.get("index_base64"):
                idx_filename = output.get("index_filename", f"{model_name}.index")
                index_path = str(model_dir / idx_filename)
                raw = base64.b64decode(output["index_base64"])
                with open(index_path, "wb") as f:
                    f.write(raw)
                print(f"[Train] Index saved via base64 fallback: {idx_filename}")
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

            try:
                with _training_lock:
                    _train_files_json = json.dumps(_training_file_map.pop(job_id, []), ensure_ascii=False)
                with get_db() as db:
                    db.execute("""
                        INSERT INTO voice_models (name, pth_path, index_path, pth_url, index_url,
                                                epochs, training_time_seconds, training_files_json, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready')
                    """, (model_name, pth_path, index_path,
                          output.get("pth_url"), output.get("index_url"),
                          output.get("epochs_trained", 0),
                          output.get("training_time_seconds", 0),
                          _train_files_json))
            except Exception as db_err:
                # DB 삽입 실패 시 다운로드된 모델 파일 정리 (고아 파일 방지)
                print(f"[Train] DB insert failed, cleaning up model files: {db_err}")
                shutil.rmtree(model_dir, ignore_errors=True)
                raise

            update_job(job_id, status="completed", progress=100,
                      message="학습 완료!",
                      result_json=json.dumps({"model_name": model_name}))

        elif job_type == "convert":
            # 변환 파일 저장 (보컬 + 믹스) — R2 URL 또는 inline base64
            out_filename = output.get("filename", f"converted_{job_id[:8]}.wav")
            out_path = OUTPUT_DIR / out_filename
            has_vocals = False

            # R2 URL → 다운로드 (재시도 포함 — GPU 결과물 손실 방지)
            if output.get("converted_audio_url"):
                _download_with_retry(output["converted_audio_url"], out_path, timeout=600)
                file_size = out_path.stat().st_size if out_path.exists() else 0
                print(f"[Convert] Downloaded vocals from R2: {file_size:,} bytes")
                if file_size < 50_000:
                    raise RuntimeError(f"다운로드된 보컬 파일이 너무 작습니다 ({file_size:,} bytes). 변환이 실패했을 수 있습니다.")
                has_vocals = True
            # inline base64
            elif output.get("converted_audio"):
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(output["converted_audio"]))
                file_size = out_path.stat().st_size
                if file_size < 50_000:
                    raise RuntimeError(f"inline base64 보컬 파일이 너무 작습니다 ({file_size:,} bytes).")
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
                    _download_with_retry(output["mixed_audio_url"], mixed_path, timeout=300)
                    result_data["mixed_file"] = mixed_filename
                    print(f"[Convert] Downloaded mixed from R2: {mixed_path.stat().st_size:,} bytes")
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
        # 변환 작업 실패 시 conversions 테이블도 업데이트
        if job_type == "convert":
            try:
                with get_db() as db:
                    db.execute("UPDATE conversions SET status='failed' WHERE job_id=?", (job_id,))
            except Exception:
                pass
    finally:
        # 완료/실패 시 인메모리 재개 상태 정리 (메모리 누수 방지)
        with _job_states_lock:
            _active_job_states.pop(job_id, None)
        with _poll_errors_lock:
            _poll_error_counts.pop(job_id, None)


def _run_batched_preprocess(job_id: str, batches: list[list[dict]]):
    """대용량 파일의 다중 배치 전처리를 순차 실행.
    각 배치를 별도 RunPod 작업으로 제출하고 완료를 기다린 뒤 다음 배치로 진행."""
    try:
        _run_batched_preprocess_inner(job_id, batches)
    except Exception as e:
        print(f"[BatchPreprocess] Unexpected crash for job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        update_job(job_id, status="failed",
                   message=f"배치 전처리 중 예기치 않은 오류: {e}")
    finally:
        with _job_states_lock:
            _active_job_states.pop(job_id, None)

def _run_batched_preprocess_inner(job_id: str, batches: list[list[dict]]):
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
            config = load_config()
            runpod_job_id = runpod_client.submit_job({
                "task_type": "preprocess",
                "audio_files": batch,
                "bucket_name": config.get("r2_bucket_name", ""),
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
        batch_poll_errors = 0
        while True:
            time.sleep(5)
            if _is_job_cancelled(job_id):
                runpod_client.cancel_runpod_job(runpod_job_id)
                return
            try:
                result = runpod_client.check_status(runpod_job_id)
                batch_poll_errors = 0  # 성공 시 에러 카운터 리셋
                status = result.get("status", "UNKNOWN")
                elapsed = int(time.time() - start_time)

                if status == "COMPLETED":
                    output = result.get("output", {})
                    all_results.append(output)
                    pct = pct_base + int(90 / total_batches)
                    update_job(job_id, status="running", progress=pct,
                               message=f"{batch_label} 완료! (다음 배치 준비 중...)")
                    break

                elif status in ("FAILED", "TIMED_OUT", "CANCELLED"):
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
                    # RunPod 작업도 취소하여 과금 중지
                    if runpod_job_id and runpod_client.is_configured():
                        try:
                            runpod_client.cancel_runpod_job(runpod_job_id)
                        except Exception:
                            pass
                    update_job(job_id, status="failed",
                               message=f"{batch_label} 시간 초과 (1시간)")
                    return

            except Exception as e:
                batch_poll_errors += 1
                if batch_poll_errors > 30:  # 30회 연속 오류 (~5분) → 작업 실패
                    update_job(job_id, status="failed",
                               message=f"{batch_label} 상태 확인 반복 실패: {e}")
                    return
                update_job(job_id, status="running",
                           message=f"{batch_label} 상태 확인 중... (재시도 {batch_poll_errors})")
                time.sleep(10)

    # 모든 배치 완료 — 결과 집계 + 세그먼트 디스크 저장
    merged_segments = []
    merged_duration = 0.0
    merged_accomp = []
    merged_vocals = []
    for r in all_results:
        merged_segments.extend(r.get("segments", []))
        merged_duration += r.get("total_duration", 0.0)
        merged_accomp.extend(r.get("accompaniment_files", []))
        merged_vocals.extend(r.get("vocal_files", []))

    saved = _save_preprocessed_segments({
        "segments": merged_segments,
        "total_duration": merged_duration,
        "accompaniment_files": merged_accomp,
        "vocal_files": merged_vocals,
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
    cleanup_stale_jobs_on_startup()
    restore_paused_jobs_state()
    # 30분마다 오래된 청크 업로드 세션 자동 정리
    def _chunk_cleanup_loop():
        while True:
            time.sleep(1800)
            try:
                _cleanup_stale_chunk_uploads()
            except Exception:
                pass
    threading.Thread(target=_chunk_cleanup_loop, daemon=True).start()


# ─── 메인 페이지 ───

@app.get("/")
async def root():
    return FileResponse(
        str(APP_DIR / "static" / "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"},
    )

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
        "r2_bucket_name": config.get("r2_bucket_name", ""),
        "download_folder": config.get("download_folder", str(Path.home() / "Downloads")),
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
    r2_bucket_name: str = Form(""),
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

@app.post("/api/config/download-folder")
async def set_download_folder(folder: str = Form(...)):
    folder_path = Path(folder)
    if not folder_path.exists():
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise HTTPException(400, f"폴더를 생성할 수 없습니다: {e}")
    if not folder_path.is_dir():
        raise HTTPException(400, "유효한 폴더 경로가 아닙니다.")
    config = load_config()
    config["download_folder"] = str(folder_path)
    save_config(config)
    return {"status": "ok", "folder": str(folder_path)}

@app.post("/api/save-to-folder")
async def save_to_folder(filename: str = Form(...)):
    """output 또는 preprocessed 디렉토리의 파일을 사용자 지정 다운로드 폴더에 복사"""
    safe_name = Path(filename).name
    # OUTPUT_DIR 먼저 확인, 없으면 PREPROCESSED_DIR 확인 (전처리 MR/보컬 파일)
    src = OUTPUT_DIR / safe_name
    if not src.exists():
        src = PREPROCESSED_DIR / safe_name
    if not src.exists():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")

    config = load_config()
    dl_folder = Path(config.get("download_folder", str(Path.home() / "Downloads")))
    if not dl_folder.exists():
        try:
            dl_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise HTTPException(500, f"다운로드 폴더 생성 실패: {e}")

    dst = dl_folder / safe_name

    # 파일이 이미 존재하고 잠겨있으면 번호 붙여서 저장 (예: file (2).wav)
    try:
        shutil.copy2(str(src), str(dst))
    except (PermissionError, OSError):
        # 파일이 잠겨 있을 때 자동 번호 부여
        stem = Path(safe_name).stem
        ext = Path(safe_name).suffix
        for i in range(2, 100):
            dst = dl_folder / f"{stem} ({i}){ext}"
            if not dst.exists():
                break
            try:
                # 존재하지만 잠기지 않은 파일이면 덮어쓰기
                shutil.copy2(str(src), str(dst))
                return {"status": "ok", "path": str(dst), "size": dst.stat().st_size}
            except (PermissionError, OSError):
                continue
        try:
            shutil.copy2(str(src), str(dst))
        except (PermissionError, OSError) as e:
            raise HTTPException(500, detail=f"파일 저장 실패 (모든 번호 시도 실패): {e}")

    return {"status": "ok", "path": str(dst), "size": dst.stat().st_size}


# ─── 파일 업로드 API ───

@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    category: str = Form("vocal")
):
    saved = []
    skipped = []

    # 디스크 공간 확인 (최소 500MB)
    disk = shutil.disk_usage(str(UPLOAD_DIR))
    if disk.free < 500_000_000:  # 500MB minimum
        raise HTTPException(507, "디스크 공간이 부족합니다. 최소 500MB 필요합니다.")

    for f in files:
        if not f.filename:
            continue
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
        duration = duration or 0.0

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
    if total_chunks < 1:
        raise HTTPException(400, f"잘못된 total_chunks: {total_chunks}")
    with _chunk_lock:
        if upload_id not in chunk_uploads:
            chunk_uploads[upload_id] = {
                "total_chunks": total_chunks,
                "received": set(),
                "filename": filename,
                "category": category,
                "started_at": time.time(),
            }

        session = chunk_uploads[upload_id]

        # total_chunks 일관성 검증
        if session["total_chunks"] != total_chunks:
            raise HTTPException(400,
                f"total_chunks 불일치: 세션={session['total_chunks']}, 요청={total_chunks}")

    if chunk_index < 0 or chunk_index >= session["total_chunks"]:
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

    with _chunk_lock:
        session["received"].add(chunk_index)
        is_complete = len(session["received"]) == session["total_chunks"]

    if is_complete:
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

        try:
            with open(save_path, "wb") as out_f:
                for i in range(total_chunks):
                    cp = chunk_dir / f"chunk_{i:06d}"
                    with open(cp, "rb") as in_f:
                        data = in_f.read()
                        total_size += len(data)
                        hash_obj.update(data)
                        out_f.write(data)
        except Exception as merge_err:
            save_path.unlink(missing_ok=True)
            shutil.rmtree(chunk_dir, ignore_errors=True)
            chunk_uploads.pop(upload_id, None)
            raise HTTPException(500, f"청크 병합 실패: {merge_err}")

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
        duration = duration or 0.0

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
    epochs: int = Form(800),           # 800: optimal for 5-15 min of training data
    sample_rate: int = Form(48000),    # 48k: better for studio/high-quality recordings
    batch_size: int = Form(0),         # 0 = GPU auto-detect (RTX 4090 → 24)
    f0_method: str = Form("rmvpe"),
    file_ids: str = Form("")  # comma-separated
):
    if not runpod_client.is_configured():
        raise HTTPException(400, "RunPod API 설정이 필요합니다. 설정 페이지에서 API Key와 Endpoint ID를 입력하세요.")

    # 모델 이름 유효성 검사
    model_name = model_name.strip()
    if not model_name:
        raise HTTPException(400, "모델 이름을 입력해 주세요.")
    if len(model_name) > 100:
        raise HTTPException(400, "모델 이름은 100자 이하여야 합니다.")
    if re.search(r'[/\\:*?"<>|]', model_name):
        raise HTTPException(400, "모델 이름에 사용할 수 없는 문자가 포함되어 있습니다: / \\ : * ? \" < > |")
    with get_db() as db:
        existing = db.execute(
            "SELECT id FROM voice_models WHERE name = ?", (model_name,)
        ).fetchone()
    if existing:
        raise HTTPException(400, f"'{model_name}' 이름의 모델이 이미 존재합니다. 다른 이름을 사용해 주세요.")

    # 파라미터 검증
    if not (10 <= epochs <= 10000):
        raise HTTPException(400, f"에포크는 10~10000 사이여야 합니다. (입력: {epochs})")
    if batch_size < 0 or batch_size > 64:
        raise HTTPException(400, f"배치 사이즈는 0~64 사이여야 합니다. (입력: {batch_size})")
    if sample_rate not in (32000, 40000, 48000):
        raise HTTPException(400, f"샘플레이트는 32000/40000/48000 중 하나여야 합니다. (입력: {sample_rate})")
    if f0_method not in ("rmvpe", "crepe", "crepe-tiny"):
        raise HTTPException(400, f"피치 추출 방식은 rmvpe/crepe/crepe-tiny 중 하나여야 합니다. (입력: {f0_method})")

    # 학습 파일 수집
    with get_db() as db:
        if file_ids:
            try:
                ids = [int(x.strip()) for x in file_ids.split(",") if x.strip()]
            except ValueError:
                raise HTTPException(400, "잘못된 파일 ID 형식입니다.")
            if not ids:
                raise HTTPException(400, "유효한 파일 ID가 없습니다.")
            else:
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
                seg_map = json.loads(meta_path.read_text(encoding="utf-8")).get("file_segments", {})
            except Exception as e:
                print(f"[Warning] Failed to parse metadata: {e}")

        try:
            ids = [int(x.strip()) for x in file_ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "잘못된 파일 ID 형식입니다.")
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
    with _training_lock:
        _training_file_map[job_id] = training_file_names
    # 재개용 상태 저장 (일시정지/재개 시 사용)
    with _job_states_lock:
        _active_job_states[job_id] = {
            "type": "train",
            "model_name": model_name,
            "epochs": epochs,
            "sample_rate": sample_rate,
            "batch_size": batch_size,
            "f0_method": f0_method,
            "file_ids": file_ids,
        }
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
                r2_key = f"train/{job_id}/{Path(fp).name}"
                try:
                    url = upload_to_r2(Path(fp), r2_key)
                    audio_urls.append({"filename": Path(fp).name, "url": url})
                except Exception as e:
                    print(f"[Train] R2 upload failed for {fp}: {e}")
                    update_job(job_id, status="failed",
                        message=f"학습 데이터 R2 업로드 실패: {e}")
                    raise HTTPException(500, f"학습 데이터 R2 업로드 실패: {e}")
            print(f"[Train] All {len(audio_urls)} files uploaded to R2")

        config = load_config()
        r2_bucket = config.get("r2_bucket_name", "")

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
                "bucket_name": r2_bucket,
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
                "bucket_name": r2_bucket,
            }

        payload_size = len(json.dumps(payload))
        print(f"[Train] Payload size: {payload_size:,} bytes, segments: {total_segments}")

        runpod_job_id = runpod_client.submit_job(payload)
        if not runpod_job_id:
            raise RuntimeError("RunPod 작업 ID를 받지 못했습니다. 잠시 후 다시 시도해주세요.")

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
        with _training_lock:
            _training_file_map.pop(job_id, None)
        with _job_states_lock:
            _active_job_states.pop(job_id, None)
        raise HTTPException(500, f"학습 제출 실패: {error_msg}")

    return {"job_id": job_id, "segments": total_segments}


# ─── 전처리 API ───

@app.post("/api/preprocess")
async def start_preprocess(
    file_ids: str = Form(...),
):
    """전처리 시작 — 영상→오디오 추출, Demucs 보컬 분리, 화자 분리, 노이즈 제거, 세그먼트 분할"""
    if not runpod_client.is_configured():
        raise HTTPException(400, "RunPod API 설정이 필요합니다. 설정 페이지에서 API Key와 Endpoint ID를 입력하세요.")

    try:
        ids = [int(x.strip()) for x in file_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "잘못된 파일 ID 형식입니다.")
    if not ids:
        raise HTTPException(400, "전처리할 파일을 선택하세요.")

    with get_db() as db:
        placeholders = ",".join("?" * len(ids))
        rows = db.execute(
            f"SELECT * FROM training_files WHERE id IN ({placeholders}) AND deleted=0", ids
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
    with _preprocess_lock:
        preprocess_file_map[job_id] = preprocess_file_ids
    # 재개용 상태 저장
    with _job_states_lock:
        _active_job_states[job_id] = {
            "type": "preprocess",
            "file_ids": preprocess_file_ids,
        }
    with get_db() as db:
        db.execute("""
            INSERT INTO jobs (id, job_type, status, progress, message)
            VALUES (?, 'preprocess', 'submitting', 0, '전처리 작업 제출 중...')
        """, (job_id,))

    try:
        # 배치별로 RunPod 작업 제출 (대용량 파일은 여러 작업으로 분할)
        if len(batches) == 1:
            config = load_config()
            runpod_job_id = runpod_client.submit_job({
                "task_type": "preprocess",
                "audio_files": batches[0],
                "bucket_name": config.get("r2_bucket_name", ""),
            })
            if not runpod_job_id:
                raise RuntimeError("RunPod 작업 ID를 받지 못했습니다. 잠시 후 다시 시도해주세요.")
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
        with _preprocess_lock:
            preprocess_file_map.pop(job_id, None)
        with _job_states_lock:
            _active_job_states.pop(job_id, None)
        raise HTTPException(500, f"전처리 제출 실패: {error_msg}")

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
            total_dur = json.loads(meta_path.read_text(encoding="utf-8")).get("total_duration", 0.0)
        except Exception as e:
            print(f"[Warning] Failed to parse metadata: {e}")

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
    try:
        ids = [int(x.strip()) for x in file_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "잘못된 파일 ID 형식입니다.")
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
    index_rate: float = Form(0.65),
    f0_method: str = Form("rmvpe"),
    vocal_volume: float = Form(1.0),
    mr_volume: float = Form(1.0),
    clean_audio: bool = Form(False),
    clean_strength: float = Form(0.7),
    protect: float = Form(0.35),
    rms_mix_rate: float = Form(0.15),
    filter_radius: int = Form(3),
    hop_length: int = Form(64),
    post_reverb: float = Form(0.05),
    harmonic_enhance: bool = Form(False),
    separate_vocals: bool = Form(True),
    audio: UploadFile = File(...)
):
    if not runpod_client.is_configured():
        raise HTTPException(400, "RunPod API 설정이 필요합니다.")

    # 파라미터 범위 검증
    if not (-24 <= pitch_shift <= 24):
        raise HTTPException(400, f"피치는 -24~24 사이여야 합니다. (입력: {pitch_shift})")
    if not (0.0 <= index_rate <= 1.0):
        raise HTTPException(400, f"인덱스 비율은 0.0~1.0 사이여야 합니다. (입력: {index_rate})")
    if not (0.0 <= protect <= 0.5):
        raise HTTPException(400, f"Protect는 0.0~0.5 사이여야 합니다. (입력: {protect})")
    if not (0.0 <= rms_mix_rate <= 1.0):
        raise HTTPException(400, f"RMS Mix는 0.0~1.0 사이여야 합니다. (입력: {rms_mix_rate})")
    if not (0 <= filter_radius <= 7):
        raise HTTPException(400, f"Filter Radius는 0~7 사이여야 합니다. (입력: {filter_radius})")
    if hop_length not in (64, 128, 256, 512):
        hop_length = 128  # 잘못된 값은 기본값으로
    if f0_method not in ("rmvpe", "crepe", "crepe-tiny", "harvest", "pm"):
        raise HTTPException(400, f"유효하지 않은 F0 방법입니다: {f0_method}")
    if not (0.0 <= vocal_volume <= 2.0):
        raise HTTPException(400, f"보컬 볼륨은 0.0~2.0 사이여야 합니다. (입력: {vocal_volume})")
    if not (0.0 <= mr_volume <= 2.0):
        raise HTTPException(400, f"MR 볼륨은 0.0~2.0 사이여야 합니다. (입력: {mr_volume})")
    if not (0.0 <= clean_strength <= 1.0):
        raise HTTPException(400, f"Clean Strength는 0.0~1.0 사이여야 합니다. (입력: {clean_strength})")
    if not (0.0 <= post_reverb <= 0.5):
        raise HTTPException(400, f"Post Reverb는 0.0~0.5 사이여야 합니다. (입력: {post_reverb})")

    # 모델 조회
    with get_db() as db:
        model = db.execute("SELECT * FROM voice_models WHERE id=?", (model_id,)).fetchone()
    if not model:
        raise HTTPException(404, "모델을 찾을 수 없습니다.")

    # 입력 파일 저장
    temp_name = f"input_{uuid.uuid4().hex[:8]}_{Path(audio.filename).name}"
    temp_path = UPLOAD_DIR / temp_name
    with open(temp_path, "wb") as f:
        content = await audio.read()
        f.write(content)

    # 모델 URL 또는 base64 준비
    pth_url = model["pth_url"] if model["pth_url"] else None
    index_url = model["index_url"] if model["index_url"] else None

    # Job + 변환 기록 동시 생성 (원자성: RunPod 제출 전에 모든 DB 레코드 삽입)
    job_id = uuid.uuid4().hex[:12]
    # 재개용 상태 기반 설정 (audio_url은 R2 업로드 후 업데이트)
    with _job_states_lock:
        _active_job_states[job_id] = {
            "type": "convert",
            "model_id": model_id,
            "audio_filename": audio.filename,
            "pitch_shift": pitch_shift,
            "index_rate": index_rate,
            "f0_method": f0_method,
            "vocal_volume": vocal_volume,
            "mr_volume": mr_volume,
            "clean_audio": clean_audio,
            "clean_strength": clean_strength,
            "protect": protect,
            "rms_mix_rate": rms_mix_rate,
            "filter_radius": filter_radius,
            "hop_length": hop_length,
            "post_reverb": post_reverb,
            "harmonic_enhance": harmonic_enhance,
            "separate_vocals": separate_vocals,
        }
    with get_db() as db:
        db.execute("""
            INSERT INTO jobs (id, job_type, status, progress, message)
            VALUES (?, 'convert', 'submitting', 0, '작업 제출 중...')
        """, (job_id,))
        db.execute("""
            INSERT INTO conversions (model_id, input_file, pitch_shift, status, job_id)
            VALUES (?, ?, ?, 'pending', ?)
        """, (model_id, audio.filename, pitch_shift, job_id))

    try:
        config = load_config()
        r2_bucket = config.get("r2_bucket_name", "")

        payload = {
            "task_type": "convert",
            "audio_filename": audio.filename,
            "pitch_shift": pitch_shift,
            "index_rate": index_rate,
            "f0_method": f0_method,
            "clean_audio": clean_audio,
            "clean_strength": clean_strength,
            "protect": protect,
            "rms_mix_rate": rms_mix_rate,
            "filter_radius": filter_radius,
            "hop_length": hop_length,
            "separate_vocals": separate_vocals,
            "vocal_volume": vocal_volume,
            "mr_volume": mr_volume,
            "post_reverb": post_reverb,
            "harmonic_enhance": harmonic_enhance,
            "bucket_name": r2_bucket,
        }

        # 오디오 파일을 R2에 업로드 (10 MB 페이로드 한도 회피)
        audio_r2_key = f"convert/{job_id}/{audio.filename}"
        try:
            audio_url = upload_to_r2(temp_path, audio_r2_key)
            payload["audio_url"] = audio_url
            # 재개용 상태에 R2 URL 저장 (temp 파일 삭제 후에도 재개 가능)
            with _job_states_lock:
                if job_id in _active_job_states:
                    _active_job_states[job_id]["audio_url"] = audio_url
        except Exception as e:
            print(f"[Convert] R2 audio upload failed: {e}")
            file_size = temp_path.stat().st_size
            if file_size > 7_000_000:  # 7MB 이상이면 base64로도 10MB 초과
                update_job(job_id, status="failed",
                    message=f"R2 업로드 실패 (파일 {file_size//1_000_000}MB): {e}")
                raise HTTPException(500, f"R2 업로드 실패 (파일 {file_size//1_000_000}MB): {e}")
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
                    pth_r2_key = f"convert/{job_id}/model.pth"
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
                    idx_r2_key = f"convert/{job_id}/model.index"
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

        # 임시 입력 파일 정리 — RunPod 제출 성공 후에만 삭제 (재개 시 R2 URL 사용)
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        if not runpod_job_id:
            raise RuntimeError("RunPod 작업 ID를 받지 못했습니다. 잠시 후 다시 시도해주세요.")

        update_job(job_id, status="running", progress=10,
                  message="GPU 변환 시작", runpod_job_id=runpod_job_id)

        # 변환 기록 상태 업데이트 (이미 'pending'으로 삽입됨 → 'processing'으로 전환)
        with get_db() as db:
            db.execute("UPDATE conversions SET status='processing' WHERE job_id=?", (job_id,))

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
        # conversions 테이블도 동기화 (pending→failed) — 정합성 유지
        try:
            with get_db() as db:
                db.execute("UPDATE conversions SET status='failed' WHERE job_id=?", (job_id,))
        except Exception:
            pass
        with _job_states_lock:
            _active_job_states.pop(job_id, None)
        # 임시 입력 파일 정리 (실패 시에도 누수 방지)
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        raise HTTPException(500, f"변환 제출 실패: {error_msg}")

    return {"job_id": job_id}


# ─── 작업 상태 API ───

@app.get("/api/jobs/active")
async def get_active_jobs():
    """페이지 로드 시 프론트엔드 상태 복원용.
    running/submitting/paused 상태의 최신 작업을 job_type별 1개씩 반환."""
    with get_db() as db:
        rows = db.execute("""
            SELECT id, job_type, status, progress, message, created_at, started_at
            FROM jobs
            WHERE status IN ('running', 'submitting', 'paused')
            ORDER BY created_at DESC
        """).fetchall()
    # job_type별 최신 1개만 반환 (중복 제거)
    seen: set = set()
    result = []
    for r in rows:
        jt = r["job_type"]
        if jt not in seen:
            seen.add(jt)
            result.append(dict(r))
    return {"active_jobs": result}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    with get_db() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    result = dict(row)
    if result.get("result_json"):
        try:
            result["result"] = json.loads(result["result_json"])
        except Exception:
            pass
    return result


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """실행 중인 작업 취소 — RunPod 작업도 실제로 취소하여 과금 즉시 중지"""
    # 1) DB 읽기
    with get_db() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        if row["status"] in ("completed", "cancelled", "failed"):
            return {"status": "already_done", "current_status": row["status"]}

    # 2) RunPod 취소 (네트워크 호출은 DB 컨텍스트 밖에서)
    runpod_job_id = row["runpod_job_id"] if row["runpod_job_id"] else None
    if runpod_job_id and runpod_client.is_configured():
        runpod_client.cancel_runpod_job(runpod_job_id)

    # 3) DB 업데이트
    with get_db() as db:
        db.execute(
            "UPDATE jobs SET status='cancelled', message='사용자가 취소함', updated_at=? "
            "WHERE id=? AND status NOT IN ('completed', 'failed', 'cancelled')",
            (datetime.now().isoformat(), job_id)
        )

    # 활성 상태 정리
    with _job_states_lock:
        _active_job_states.pop(job_id, None)
    with _poll_errors_lock:
        _poll_error_counts.pop(job_id, None)
    with _preprocess_lock:
        preprocess_file_map.pop(job_id, None)

    return {"status": "cancelled"}


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """실행 중인 작업 일시정지 — RunPod 작업 취소(과금 중지) + 재개 가능한 상태 저장"""
    # 1) DB 읽기
    with get_db() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        if row["status"] in ("completed", "cancelled", "paused", "failed"):
            return {"status": "already_done", "current_status": row["status"]}

    # 2) RunPod 취소 (네트워크 호출은 DB 컨텍스트 밖에서)
    runpod_job_id = row["runpod_job_id"] if row["runpod_job_id"] else None
    cancelled_runpod = False
    if runpod_job_id and runpod_client.is_configured():
        cancelled_runpod = runpod_client.cancel_runpod_job(runpod_job_id)

    # 3) 재개용 상태 저장 + DB 업데이트
    with _job_states_lock:
        pause_state = _active_job_states.get(job_id, {}).copy()
    pause_state_json = json.dumps(pause_state, ensure_ascii=False)

    with get_db() as db:
        db.execute(
            "UPDATE jobs SET status='paused', message='일시정지됨 (RunPod 과금 중지)', "
            "pause_state_json=?, updated_at=? "
            "WHERE id=? AND status NOT IN ('completed', 'failed', 'cancelled')",
            (pause_state_json, datetime.now().isoformat(), job_id)
        )

    with _poll_errors_lock:
        _poll_error_counts.pop(job_id, None)
    # _active_job_states는 보존 (재개 시 필요)

    return {
        "status": "paused",
        "runpod_cancelled": cancelled_runpod,
        "can_resume": bool(pause_state)
    }


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """일시정지된 작업 재개 — 저장된 파라미터로 RunPod에 재제출"""
    with get_db() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        if row["status"] != "paused":
            raise HTTPException(400, f"일시정지된 작업이 아닙니다. 현재 상태: {row['status']}")

        # pause_state 읽기
        pause_state_str = row["pause_state_json"] or ""
        pause_state: dict = {}
        if pause_state_str:
            try:
                pause_state = json.loads(pause_state_str)
            except Exception:
                pass

    if not pause_state:
        raise HTTPException(400, "재개 상태 정보가 없습니다. 처음부터 다시 시작해주세요.")

    if not runpod_client.is_configured():
        raise HTTPException(400, "RunPod API 설정이 필요합니다. 설정 페이지에서 확인하세요.")

    job_type = row["job_type"]
    if job_type not in ("preprocess", "train", "convert"):
        raise HTTPException(400, f"지원하지 않는 작업 유형입니다: {job_type}")

    try:
        if job_type == "preprocess":
            # ─── 전처리 재개 ───
            # DB에서 아직 전처리되지 않은 파일만 다시 처리
            file_ids = pause_state.get("file_ids", [])
            if not file_ids:
                raise HTTPException(400, "재개할 파일 정보가 없습니다.")

            with get_db() as db:
                placeholders = ",".join("?" * len(file_ids))
                rows = db.execute(
                    f"SELECT * FROM training_files WHERE id IN ({placeholders}) AND deleted=0", file_ids
                ).fetchall()

            unprocessed = [r for r in rows if not r["preprocessed"]]
            if not unprocessed:
                # 모든 파일 이미 완료
                update_job(job_id, status="completed", progress=100,
                          message="전처리 완료 (모든 파일 이미 처리됨)")
                return {"status": "already_completed"}

            # 파일 존재 여부 확인 (삭제된 파일 제외)
            existing_unprocessed = [r for r in unprocessed if (UPLOAD_DIR / r["filename"]).exists()]
            if not existing_unprocessed:
                update_job(job_id, status="failed",
                          message="재개할 파일이 없습니다. 원본 파일이 삭제된 것 같습니다.")
                return {"status": "failed"}
            file_paths = [str(UPLOAD_DIR / r["filename"]) for r in existing_unprocessed]
            batches = prepare_files_for_runpod(file_paths)

            with _preprocess_lock:
                preprocess_file_map[job_id] = [r["id"] for r in existing_unprocessed]
            with _job_states_lock:
                _active_job_states[job_id] = pause_state.copy()

            update_job(job_id, status="running", progress=3,
                      message=f"전처리 재개 중... ({len(unprocessed)}개 파일)")

            config = load_config()
            if len(batches) == 1:
                runpod_job_id = runpod_client.submit_job({
                    "task_type": "preprocess",
                    "audio_files": batches[0],
                    "bucket_name": config.get("r2_bucket_name", ""),
                })
                if not runpod_job_id:
                    raise RuntimeError("RunPod 작업 ID를 받지 못했습니다.")
                update_job(job_id, status="running", progress=8,
                          message="전처리 재개 — GPU 작업 제출됨",
                          runpod_job_id=runpod_job_id)
                threading.Thread(
                    target=poll_runpod_job,
                    args=(job_id, runpod_job_id, "preprocess"),
                    daemon=True
                ).start()
            else:
                update_job(job_id, status="running", progress=2,
                          message=f"전처리 재개 ({len(batches)}개 배치)")
                threading.Thread(
                    target=_run_batched_preprocess,
                    args=(job_id, batches),
                    daemon=True
                ).start()

        elif job_type == "train":
            # ─── 학습 재개 (처음부터 재시작) ───
            model_name = pause_state.get("model_name", "")
            epochs = pause_state.get("epochs", 800)
            sample_rate = pause_state.get("sample_rate", 48000)
            batch_size = pause_state.get("batch_size", 0)
            f0_method = pause_state.get("f0_method", "rmvpe")
            file_ids_str = pause_state.get("file_ids", "")

            if not model_name:
                raise HTTPException(400, "학습 재개 정보가 없습니다. 처음부터 다시 시작해주세요.")

            # 이름 중복 체크 — 기존 모델이 있으면 suffix 추가 (충돌 없을 때까지 루프)
            with get_db() as db:
                existing = db.execute(
                    "SELECT id FROM voice_models WHERE name=?", (model_name,)
                ).fetchone()
            if existing:
                base_name = model_name
                suffix = 1
                while True:
                    candidate = f"{base_name}_재개{suffix if suffix > 1 else ''}"
                    with get_db() as db:
                        if not db.execute(
                            "SELECT id FROM voice_models WHERE name=?", (candidate,)
                        ).fetchone():
                            model_name = candidate
                            break
                    suffix += 1
                    if suffix > 99:  # 무한루프 방지
                        model_name = f"{base_name}_{uuid.uuid4().hex[:6]}"
                        break

            update_job(job_id, status="running", progress=3, message="학습 재개 중...")

            try:
                ids = [int(x.strip()) for x in file_ids_str.split(",") if x.strip()] if file_ids_str else []
            except (ValueError, TypeError):
                ids = []
            with get_db() as db:
                if ids:
                    placeholders = ",".join("?" * len(ids))
                    rows = db.execute(
                        f"SELECT * FROM training_files WHERE id IN ({placeholders}) AND deleted=0", ids
                    ).fetchall()
                else:
                    rows = db.execute("SELECT * FROM training_files WHERE deleted=0").fetchall()

            all_preprocessed = [f for f in _list_preprocessed_files()
                                 if not f.name.startswith(("mr_", "vocal_"))]

            # file_ids 기반 세그먼트 필터링 (start_training과 동일 로직)
            preprocessed_files = all_preprocessed
            if all_preprocessed and ids:
                meta_path = PREPROCESSED_DIR / "_metadata.json"
                seg_map = {}
                if meta_path.exists():
                    try:
                        seg_map = json.loads(meta_path.read_text(encoding="utf-8")).get("file_segments", {})
                    except Exception:
                        pass
                selected_seg_names = set()
                for fid in ids:
                    selected_seg_names.update(seg_map.get(str(fid), []))
                if selected_seg_names:
                    preprocessed_files = [
                        f for f in all_preprocessed if f.name in selected_seg_names
                    ]

            file_paths = ([str(p) for p in preprocessed_files] if preprocessed_files
                          else [str(UPLOAD_DIR / r["filename"]) for r in rows])

            training_file_names = [r["original_name"] for r in rows]
            with _training_lock:
                _training_file_map[job_id] = training_file_names
            with _job_states_lock:
                _active_job_states[job_id] = pause_state.copy()
                _active_job_states[job_id]["model_name"] = model_name  # 업데이트된 이름

            config = load_config()
            r2_bucket = config.get("r2_bucket_name", "")
            audio_urls = []
            total_size = sum(Path(p).stat().st_size for p in file_paths if Path(p).exists())

            if total_size > 5_000_000:
                for fp in file_paths:
                    if not Path(fp).exists():
                        continue
                    r2_key = f"train/{job_id}_resume/{Path(fp).name}"
                    try:
                        url = upload_to_r2(Path(fp), r2_key)
                        audio_urls.append({"filename": Path(fp).name, "url": url})
                    except Exception as e:
                        update_job(job_id, status="failed", message=f"R2 업로드 실패: {e}")
                        raise HTTPException(500, f"R2 업로드 실패: {e}")

            if audio_urls:
                payload = {
                    "task_type": "train", "model_name": model_name,
                    "audio_urls": audio_urls, "sample_rate": sample_rate,
                    "epochs": epochs, "batch_size": batch_size,
                    "f0_method": f0_method, "bucket_name": r2_bucket,
                }
            else:
                batches = prepare_files_for_runpod(file_paths)
                payload = {
                    "task_type": "train", "model_name": model_name,
                    "audio_files": batches[0] if batches else [],
                    "sample_rate": sample_rate, "epochs": epochs,
                    "batch_size": batch_size, "f0_method": f0_method,
                    "bucket_name": r2_bucket,
                }

            runpod_job_id = runpod_client.submit_job(payload)
            if not runpod_job_id:
                raise RuntimeError("RunPod 작업 ID를 받지 못했습니다.")
            update_job(job_id, status="running", progress=5,
                      message=f"학습 재개 — GPU 작업 제출됨 (모델: {model_name})",
                      runpod_job_id=runpod_job_id)
            threading.Thread(
                target=poll_runpod_job,
                args=(job_id, runpod_job_id, "train"),
                daemon=True
            ).start()

        elif job_type == "convert":
            # ─── 변환 재개 ───
            model_id = pause_state.get("model_id")
            if not model_id:
                raise HTTPException(400, "변환 재개 정보가 없습니다. 파일을 다시 업로드해주세요.")

            with get_db() as db:
                model = db.execute("SELECT * FROM voice_models WHERE id=?", (model_id,)).fetchone()
            if not model:
                raise HTTPException(404, "모델을 찾을 수 없습니다.")

            update_job(job_id, status="running", progress=3, message="변환 재개 중...")
            with _job_states_lock:
                _active_job_states[job_id] = pause_state.copy()

            config = load_config()
            r2_bucket = config.get("r2_bucket_name", "")
            audio_filename = pause_state.get("audio_filename", "input.wav")
            payload = {
                "task_type": "convert",
                "audio_filename": audio_filename,
                "pitch_shift": pause_state.get("pitch_shift", 0),
                "index_rate": pause_state.get("index_rate", 0.65),
                "f0_method": pause_state.get("f0_method", "rmvpe"),
                "clean_audio": pause_state.get("clean_audio", False),
                "clean_strength": pause_state.get("clean_strength", 0.7),
                "protect": pause_state.get("protect", 0.35),
                "rms_mix_rate": pause_state.get("rms_mix_rate", 0.15),
                "filter_radius": pause_state.get("filter_radius", 3),
                "hop_length": pause_state.get("hop_length", 64),
                "separate_vocals": pause_state.get("separate_vocals", True),
                "vocal_volume": pause_state.get("vocal_volume", 1.0),
                "mr_volume": pause_state.get("mr_volume", 1.0),
                "post_reverb": pause_state.get("post_reverb", 0.05),
                "harmonic_enhance": pause_state.get("harmonic_enhance", False),
                "bucket_name": r2_bucket,
            }

            # 오디오 소스: R2 URL 우선 (temp 파일은 삭제되었으므로)
            saved_audio_url = pause_state.get("audio_url")
            if saved_audio_url:
                # R2에 이미 업로드된 URL 재사용 (추가 업로드 불필요)
                payload["audio_url"] = saved_audio_url
            else:
                # 폴백: 이전 방식 temp 파일 (하위 호환)
                input_file = pause_state.get("input_file", "")
                temp_path_resume = UPLOAD_DIR / input_file if input_file else None
                if temp_path_resume and temp_path_resume.exists():
                    audio_r2_key = f"convert/{job_id}_resume/{input_file}"
                    try:
                        audio_url = upload_to_r2(temp_path_resume, audio_r2_key)
                        payload["audio_url"] = audio_url
                        with _job_states_lock:
                            if job_id in _active_job_states:
                                _active_job_states[job_id]["audio_url"] = audio_url
                    except Exception as e:
                        file_size = temp_path_resume.stat().st_size
                        if file_size > 7_000_000:
                            update_job(job_id, status="failed", message=f"R2 업로드 실패: {e}")
                            raise HTTPException(500, f"R2 업로드 실패: {e}")
                        with open(temp_path_resume, "rb") as f:
                            payload["audio_data"] = base64.b64encode(f.read()).decode()
                else:
                    raise HTTPException(
                        400,
                        "변환할 오디오 파일을 찾을 수 없습니다. "
                        "파일을 다시 업로드하여 변환을 새로 시작해주세요."
                    )

            pth_url = model["pth_url"] if model["pth_url"] else None
            index_url = model["index_url"] if model["index_url"] else None
            if pth_url:
                payload["pth_url"] = pth_url
            if index_url:
                payload["index_url"] = index_url

            # pth_url 없는 모델 → 로컬 파일을 R2에 업로드 (presigned URL 만료 대비)
            if not pth_url and model["pth_path"]:
                local_pth = Path(model["pth_path"])
                if not local_pth.is_absolute():
                    local_pth = MODEL_DIR / local_pth
                if local_pth.exists():
                    try:
                        pth_r2_key = f"convert/{job_id}_resume/model.pth"
                        pth_url = upload_to_r2(local_pth, pth_r2_key)
                        payload["pth_url"] = pth_url
                    except Exception as e:
                        update_job(job_id, status="failed", message=f"모델 R2 업로드 실패: {e}")
                        raise HTTPException(500, f"R2 업로드 실패: {e}")

            if not index_url and model["index_path"]:
                local_idx = Path(model["index_path"])
                if not local_idx.is_absolute():
                    local_idx = MODEL_DIR / local_idx
                if local_idx.exists():
                    try:
                        idx_r2_key = f"convert/{job_id}_resume/model.index"
                        index_url = upload_to_r2(local_idx, idx_r2_key)
                        payload["index_url"] = index_url
                    except Exception:
                        pass  # index는 선택사항

            runpod_job_id = runpod_client.submit_job(payload)
            if not runpod_job_id:
                raise RuntimeError("RunPod 작업 ID를 받지 못했습니다.")
            update_job(job_id, status="running", progress=10,
                      message="변환 재개 — GPU 작업 제출됨",
                      runpod_job_id=runpod_job_id)

            with get_db() as db:
                db.execute("UPDATE conversions SET status='processing' WHERE job_id=?", (job_id,))

            threading.Thread(
                target=poll_runpod_job,
                args=(job_id, runpod_job_id, "convert"),
                daemon=True
            ).start()

        else:
            raise HTTPException(400, f"지원하지 않는 작업 유형: {job_type}")

        return {"status": "resumed", "job_id": job_id, "job_type": job_type}

    except HTTPException:
        raise
    except Exception as e:
        # 실패 시 메모리 맵 정리 (누수 방지)
        with _preprocess_lock:
            preprocess_file_map.pop(job_id, None)
        with _training_lock:
            _training_file_map.pop(job_id, None)
        update_job(job_id, status="failed", message=f"재개 실패: {e}")
        raise HTTPException(500, f"재개 실패: {e}")


@app.post("/api/jobs/cleanup")
async def cleanup_all_stuck_jobs():
    """멈춘 작업 일괄 정리 — running/submitting 상태의 모든 작업 취소 + RunPod 과금 중지"""
    with get_db() as db:
        # RunPod 작업 ID 수집 → 실제 취소 (과금 즉시 중지)
        active_rows = db.execute("""
            SELECT id, runpod_job_id FROM jobs
            WHERE status IN ('running', 'submitting') AND runpod_job_id IS NOT NULL
        """).fetchall()
        if active_rows and runpod_client.is_configured():
            for row in active_rows:
                try:
                    runpod_client.cancel_runpod_job(row["runpod_job_id"])
                except Exception:
                    pass
        result = db.execute("""
            UPDATE jobs
            SET status='cancelled',
                message='수동 정리',
                updated_at=?
            WHERE status IN ('running', 'submitting')
        """, (datetime.now().isoformat(),))
    # 메모리 상태 정리 (paused 작업의 상태는 보존 — 재개 시 필요)
    with _job_states_lock:
        paused_states = {}
        with get_db() as _db:
            _paused = _db.execute("SELECT id FROM jobs WHERE status='paused'").fetchall()
            paused_ids = {r["id"] for r in _paused}
        for jid in list(_active_job_states.keys()):
            if jid in paused_ids:
                paused_states[jid] = _active_job_states[jid]
        _active_job_states.clear()
        _active_job_states.update(paused_states)
    with _poll_errors_lock:
        _poll_error_counts.clear()
    with _preprocess_lock:
        preprocess_file_map.clear()
    return {"cleaned": result.rowcount}


@app.get("/api/jobs")
async def list_jobs():
    with get_db() as db:
        rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50").fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("result_json"):
            try:
                d["result"] = json.loads(d["result_json"])
            except Exception:
                pass
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
        # 실행 중 또는 일시정지된 변환 작업이 있으면 삭제 거부
        active = db.execute(
            "SELECT id FROM conversions WHERE model_id=? AND status IN ('pending','processing','paused')",
            (model_id,)
        ).fetchall()
        if active:
            raise HTTPException(409, f"이 모델을 사용 중인 변환 작업이 {len(active)}개 있습니다. 먼저 작업을 취소하거나 완료하세요.")
        dirs_to_check = set()
        for path_key in ["pth_path", "index_path"]:
            if row[path_key] and Path(row[path_key]).exists():
                dirs_to_check.add(Path(row[path_key]).parent)
                Path(row[path_key]).unlink()
        # 빈 모델 디렉토리 정리
        for d in dirs_to_check:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        # 연결된 변환 출력 파일 삭제 (고아 파일 방지)
        conv_rows = db.execute(
            "SELECT output_file, output_name FROM conversions WHERE model_id=?", (model_id,)
        ).fetchall()
        for conv in conv_rows:
            for col in ("output_file", "output_name"):
                fname = conv[col]
                if fname:
                    fpath = OUTPUT_DIR / fname
                    try:
                        fpath.unlink(missing_ok=True)
                    except Exception:
                        pass
        # 연결된 변환 기록도 삭제 (cascade)
        db.execute("DELETE FROM conversions WHERE model_id = ?", (model_id,))
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


_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
}

@app.get("/api/download/{filename:path}")
async def download_file(filename: str):
    # Path traversal 방지
    safe_name = Path(filename).name
    # 허용된 오디오 확장자만 서빙 (임의 파일 노출 방지)
    ext = Path(safe_name).suffix.lower()
    if ext not in _AUDIO_MIME:
        raise HTTPException(400, f"지원하지 않는 파일 형식입니다: {ext}")
    filepath = OUTPUT_DIR / safe_name
    if not filepath.exists():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    # 오디오 파일은 적절한 MIME 타입으로 서빙 (audio 태그 인라인 재생 + 다운로드 모두 지원)
    media_type = _AUDIO_MIME[ext]
    return FileResponse(str(filepath), filename=safe_name, media_type=media_type)


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
    # 주기적 stale job 정리 + 청크 업로드 정리 (30분 간격)
    maybe_cleanup_stale_jobs()
    _cleanup_stale_chunk_uploads()

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


@app.get("/api/system-info")
async def system_info():
    """시스템 요약 정보 (디스크, 모델/파일/변환 수)"""
    disk = shutil.disk_usage(str(DATA_DIR))
    with get_db() as db:
        models = db.execute("SELECT COUNT(*) FROM voice_models").fetchone()[0]
        files = db.execute("SELECT COUNT(*) FROM training_files WHERE deleted=0").fetchone()[0]
        conversions = db.execute("SELECT COUNT(*) FROM conversions").fetchone()[0]
    return {
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "disk_total_gb": round(disk.total / (1024**3), 1),
        "models_count": models,
        "training_files_count": files,
        "conversions_count": conversions,
    }


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
