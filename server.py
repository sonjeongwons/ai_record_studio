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
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import asyncio
import uvicorn
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
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
# 일반 업로드 제한: 최대 500MB (OOM 방지 — 대용량은 청크 업로드 사용)
MAX_UPLOAD_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 로깅 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logger = logging.getLogger("voice-studio")
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.DEBUG)
_console_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
))
logger.addHandler(_console_handler)

# 파일 핸들러 (DATA_DIR 확정 후 설정)
DATA_DIR.mkdir(parents=True, exist_ok=True)
_log_file = DATA_DIR / "server.log"
_file_handler = logging.handlers.RotatingFileHandler(
    _log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(_file_handler)

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
# 활성 폴링 스레드 추적 (job_id → Thread) — 중복 스레드 방지
_active_poll_threads: dict[str, threading.Thread] = {}
_poll_threads_lock = threading.Lock()


def _remove_chunk_session(upload_id: str):
    """chunk_uploads에서 세션 제거 (thread-safe)."""
    with _chunk_lock:
        chunk_uploads.pop(upload_id, None)


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
        logger.info("[ChunkCleanup] %d개 미완료 업로드 정리", len(stale_ids))

def _list_preprocessed_files() -> list[Path]:
    """전처리된 오디오 파일 목록 (WAV + MP3 + FLAC)."""
    files: list[Path] = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        files.extend(PREPROCESSED_DIR.glob(ext))
    return sorted(files)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸: FormData boolean 파싱 (JS FormData는 bool을 "true"/"false" 문자열로 전송)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_form_bool(value, default: bool = False) -> bool:
    """FormData에서 전송된 boolean 값을 안전하게 파싱.
    JS FormData.append(key, true/false)는 문자열 "true"/"false"로 전송됨.
    Python bool("false") == True 이므로 직접 파싱 필요."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return default


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
    except Exception as e:
        logger.debug("[FFprobe] 오디오 길이 측정 실패 (%s): %s", file_path.name if hasattr(file_path, 'name') else file_path, e)
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
            logger.warning("config.json 손상 → 백업: %s", backup)
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
        # 기존 DB 마이그레이션 — sqlite3.OperationalError만 캐치 (권한/손상 에러는 전파)
        for col, default in [("preprocessed", "INTEGER DEFAULT 0"),
                             ("file_hash", "TEXT"),
                             ("deleted", "INTEGER DEFAULT 0")]:
            try:
                db.execute(f"ALTER TABLE training_files ADD COLUMN {col} {default}")
            except sqlite3.OperationalError:
                pass  # 이미 존재하면 무시
        # voice_models에 R2 URL 컬럼 추가 (기존 DB 마이그레이션)
        for col in ("pth_url", "index_url", "training_files_json", "pretrained_model"):
            try:
                db.execute(f"ALTER TABLE voice_models ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass
        # conversions에 job_id 컬럼 추가 (정확한 작업-변환 매핑)
        try:
            db.execute("ALTER TABLE conversions ADD COLUMN job_id TEXT")
        except sqlite3.OperationalError:
            pass
        # jobs에 started_at 컬럼 추가 (경과 시간 계산용)
        try:
            db.execute("ALTER TABLE jobs ADD COLUMN started_at TEXT")
        except sqlite3.OperationalError:
            pass
        # file_hash 인덱스 (중복 체크 성능)
        try:
            db.execute("CREATE INDEX IF NOT EXISTS idx_training_files_hash ON training_files(file_hash)")
        except sqlite3.OperationalError:
            pass
        # 성능 인덱스 추가
        db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_conversions_model_id ON conversions(model_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_conversions_job_id ON conversions(job_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_training_files_deleted ON training_files(deleted)")
        # voice_models(name) UNIQUE 인덱스 — 동일 모델명 중복 방지
        try:
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_voice_models_name ON voice_models(name)")
        except sqlite3.OperationalError:
            pass  # 기존 DB에 중복 name이 있으면 인덱스 생성 실패 (무시)


def recover_orphan_jobs_on_startup():
    """서버 시작 시 1회만 호출. running/submitting 작업의 RunPod 상태를 확인하여
    아직 실행 중인 작업은 폴링 스레드를 재시작하고, 완료/실패된 작업만 정리."""
    with get_db() as db:
        stale_rows = db.execute("""
            SELECT id, job_type, runpod_job_id, status FROM jobs
            WHERE status IN ('running', 'submitting')
        """).fetchall()

    if not stale_rows:
        logger.info("[Startup] 복구할 진행 중 작업 없음")
        return

    logger.info("[Startup] 진행 중 작업 %d개 발견 — RunPod 상태 확인 중...", len(stale_rows))

    recovered = 0
    completed_on_runpod = 0
    failed_cleanup = 0

    for row in stale_rows:
        job_id = row["id"]
        job_type = row["job_type"]
        runpod_job_id = row["runpod_job_id"]

        # RunPod job ID가 없는 작업 (submit 전 크래시) → 실패 처리
        if not runpod_job_id:
            update_job(job_id, status="failed",
                      message="서버 재시작: RunPod 작업 ID 없음 (제출 전 중단)")
            failed_cleanup += 1
            continue

        # RunPod API 미설정 → 상태 확인 불가, 실패 처리
        if not runpod_client.is_configured():
            update_job(job_id, status="failed",
                      message="서버 재시작: RunPod API 미설정으로 상태 확인 불가")
            failed_cleanup += 1
            continue

        # RunPod에서 실제 상태 확인
        try:
            result = runpod_client.check_status(runpod_job_id)
            rp_status = result.get("status", "UNKNOWN")
        except Exception as e:
            logger.warning("[Startup] RunPod 상태 확인 실패 (%s): %s", job_id, e)
            # 네트워크 문제일 수 있으므로 폴링 스레드를 시작하여 나중에 재시도
            logger.info("[Startup] 폴링 스레드 재시작 (네트워크 복구 대기): %s", job_id)
            _spawn_polling_thread(job_id, runpod_job_id, job_type)
            recovered += 1
            continue

        if rp_status in ("IN_PROGRESS", "IN_QUEUE"):
            # 아직 실행 중 → 폴링 스레드 재시작
            logger.info("[Startup] RunPod 작업 진행 중 → 폴링 복구: %s (type=%s, rp=%s)", job_id, job_type, rp_status)
            update_job(job_id, status="running",
                      message=f"서버 재시작 후 자동 복구 (RunPod: {rp_status})")
            _spawn_polling_thread(job_id, runpod_job_id, job_type)
            recovered += 1

        elif rp_status == "COMPLETED":
            # RunPod에서 이미 완료 → 결과 수거
            logger.info("[Startup] RunPod 작업 완료됨 → 결과 수거: %s", job_id)
            output = result.get("output", {})
            if output:
                try:
                    handle_job_result(job_id, job_type, output)
                    completed_on_runpod += 1
                except Exception as e:
                    update_job(job_id, status="failed",
                              message=f"서버 재시작 후 결과 처리 실패: {e}")
                    failed_cleanup += 1
            else:
                update_job(job_id, status="failed",
                          message="서버 재시작 후 RunPod 결과가 비어있음")
                failed_cleanup += 1

        elif rp_status in ("FAILED", "TIMED_OUT", "CANCELLED"):
            # RunPod에서 실패/취소 → 정리
            raw_error = result.get("error", "알 수 없는 오류")
            if isinstance(raw_error, dict):
                error = raw_error.get("error_message") or raw_error.get("message") or str(raw_error)
            else:
                error = str(raw_error)
            update_job(job_id, status="failed",
                      message=f"서버 재시작 확인: RunPod {rp_status} — {error}")
            failed_cleanup += 1

        else:
            # 알 수 없는 상태 → 폴링 스레드 시작하여 추적 계속
            logger.info("[Startup] RunPod 상태 불명(%s) → 폴링 복구: %s", rp_status, job_id)
            _spawn_polling_thread(job_id, runpod_job_id, job_type)
            recovered += 1

    logger.info("[Startup] 작업 복구 완료: 폴링 재시작=%d, 결과 수거=%d, 실패 정리=%d", recovered, completed_on_runpod, failed_cleanup)

    # conversions 테이블 동기화 — orphan 'processing' 레코드 정리
    with get_db() as db:
        db.execute("""
            UPDATE conversions SET status='failed'
            WHERE status='processing'
            AND job_id NOT IN (
                SELECT id FROM jobs WHERE status NOT IN ('failed', 'cancelled')
            )
        """)



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
                    logger.debug("Silent exception in %s", __name__, exc_info=True)
        stale = db.execute("""
            UPDATE jobs
            SET status='failed',
                message='1시간 이상 응답 없음 (자동 정리)',
                updated_at=?
            WHERE status IN ('running', 'submitting')
              AND updated_at < datetime('now', 'localtime', '-1 hour')
        """, (datetime.now().isoformat(),))
        if stale.rowcount > 0:
            logger.info("정리된 멈춘 작업: %d개", stale.rowcount)


# 주기적 정리 스로틀 (30분 간격)
_last_cleanup_time: float = 0.0
_cleanup_lock = threading.Lock()


def maybe_cleanup_stale_jobs():
    """30분 이상 경과 시 stale job 정리 (health 엔드포인트에서 호출).
    1시간 이상 방치된 작업만 정리 — 진행 중인 작업은 건드리지 않음."""
    global _last_cleanup_time
    now = time.time()
    with _cleanup_lock:
        if now - _last_cleanup_time > 1800:  # 30분
            _last_cleanup_time = now
        else:
            return
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
            logger.debug("Silent exception in %s", __name__, exc_info=True)
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
            logger.info("[RunPod] Cancel %s: HTTP %d", runpod_job_id, resp.status_code)
            return resp.status_code in (200, 201, 202)
        except Exception as e:
            logger.error("[RunPod] Cancel failed for %s: %s", runpod_job_id, e)
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
# 학습 작업별 사전학습 모델 추적 (job_id → pretrained_model)
_training_pretrained_map: dict[str, str] = {}
_training_lock = threading.Lock()  # _training_file_map/_training_pretrained_map 동시성 보호

# RunPod 폴링 에러 카운터 동시성 보호
_poll_errors_lock = threading.Lock()

# 작업 유형별 타임아웃 (초)
_JOB_TIMEOUTS = {"train": 36000, "preprocess": 3600, "convert": 1800}  # convert: 30분


def _get_r2_client():
    """R2 boto3 클라이언트를 생성합니다. 캐시하여 재사용합니다."""
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
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
        region_name="auto",
    )
    return s3, bucket


def upload_to_r2(file_path: Path, key: str, s3_client=None, bucket_name=None) -> str:
    """로컬 파일을 R2에 업로드하고 presigned URL을 반환합니다."""
    if s3_client and bucket_name:
        s3, bucket = s3_client, bucket_name
    else:
        s3, bucket = _get_r2_client()

    try:
        s3.upload_file(str(file_path), bucket, key)
        # Cloudflare R2 presigned URL 최대 만료: 7일 (604800초)
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=604800,  # 7일 (R2 최대 허용)
        )
    except Exception as e:
        logger.error("R2 Upload Error: %s: %s", type(e).__name__, e)
        raise HTTPException(500, f"R2 업로드 실패: {type(e).__name__}: {e}")

    return presigned_url


def refresh_presigned_url(old_url: str) -> str:
    """만료된 R2 presigned URL에서 key를 추출하여 새 presigned URL을 생성합니다."""
    from urllib.parse import urlparse, unquote

    s3, bucket = _get_r2_client()

    # presigned URL에서 R2 key 추출: /{bucket}/{key}?X-Amz-...
    parsed = urlparse(old_url)
    path = unquote(parsed.path)  # URL 디코딩
    # path-style: /bucket/key 또는 /key
    if path.startswith(f"/{bucket}/"):
        key = path[len(f"/{bucket}/"):]
    elif path.startswith("/"):
        key = path[1:]
    else:
        key = path

    new_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=604800,
    )
    logger.info("[R2] Refreshed presigned URL for key: %s", key)
    return new_url


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 백그라운드 작업 관리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_JOB_UPDATE_COLS = frozenset({"status", "progress", "message", "result_json",
                               "runpod_job_id", "started_at"})

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
        logger.error("DB 업데이트 실패 (job_id=%s): %s", job_id, e)


def _is_job_cancelled(job_id: str) -> bool:
    """작업이 취소/실패 상태인지 확인 (폴링 루프 종료용)"""
    try:
        with get_db() as db:
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            return row and row["status"] in ("failed", "cancelled")
    except Exception as e:
        logger.warning("[Poll] _is_job_cancelled DB 오류 (job %s): %s — 안전하게 취소 처리", job_id, e)
        return True  # DB 오류 시 폴링 중단 (무한 폴링 방지)


def _extract_progress_info(result: dict, job_type: str, elapsed: int) -> tuple:
    """RunPod 응답에서 진행률(%)과 메시지를 추출.
    Returns: (pct: int 5-95, message: str)"""
    pct = None
    msg = None

    # output 필드에서 progress 텍스트 추출
    progress_text = ""
    output_field = result.get("output")
    if isinstance(output_field, str) and output_field:
        progress_text = output_field
    elif isinstance(output_field, dict):
        progress_text = output_field.get("progress", "")

    # stream 필드 폴백
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
            m4 = re.search(r'\((\d+)/(\d+)\)', progress_text)
            if m4 and int(m4.group(2)) > 0:
                step, total_steps = int(m4.group(1)), int(m4.group(2))
                pct = min(90, int(step / total_steps * 90))
                msg = progress_text

    if pct is None:
        # 경과 시간 기반 추정
        _EST_TIMES = {"train": 900, "preprocess": 600, "convert": 300}
        est_total = _EST_TIMES.get(job_type, 600)
        pct = min(90, int((elapsed / est_total) * 100))

    pct = min(95, max(5, pct))
    _TYPE_LABELS = {"train": "학습", "preprocess": "전처리", "convert": "변환"}
    label = _TYPE_LABELS.get(job_type, job_type)
    mins = elapsed // 60
    time_str = f"{mins}분 경과" if mins >= 1 else f"{elapsed}초"
    default_msg = f"{label} 중... ({time_str})"

    return pct, msg or default_msg


def _mark_job_failed_with_conversion(job_id: str, job_type: str, message: str):
    """작업 실패 마킹 + convert인 경우 conversions 테이블도 업데이트"""
    try:
        update_job(job_id, status="failed", message=message)
    except Exception:
        logger.error("Failed to mark job %s as failed", job_id)
    if job_type == "convert":
        try:
            with get_db() as db:
                db.execute("UPDATE conversions SET status='failed' WHERE job_id=?", (job_id,))
        except Exception:
            logger.debug("Silent exception in %s", __name__, exc_info=True)


def _spawn_polling_thread(job_id: str, runpod_job_id: str, job_type: str):
    """RunPod 폴링 데몬 스레드 생성 — start_training/conversion/preprocess 공통.
    동일 job_id에 대해 기존 스레드가 살아있으면 새 스레드를 생성하지 않음."""
    with _poll_threads_lock:
        existing = _active_poll_threads.get(job_id)
        if existing and existing.is_alive():
            logger.warning("[Poll] job %s 에 이미 활성 폴링 스레드 존재 — 중복 생성 방지", job_id)
            return existing
    t = threading.Thread(
        target=_poll_thread_wrapper,
        args=(job_id, runpod_job_id, job_type),
        daemon=True,
        name=f"poll-{job_type}-{job_id[:8]}"
    )
    with _poll_threads_lock:
        _active_poll_threads[job_id] = t
    t.start()
    return t


def _poll_thread_wrapper(job_id: str, runpod_job_id: str, job_type: str):
    """폴링 스레드 래퍼: 완료 시 _active_poll_threads에서 자동 제거."""
    try:
        poll_runpod_job(job_id, runpod_job_id, job_type)
    finally:
        with _poll_threads_lock:
            _active_poll_threads.pop(job_id, None)


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
                            logger.debug("Silent exception in %s", __name__, exc_info=True)
                    with _poll_errors_lock:
                        _poll_error_counts.pop(job_id, None)
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
                            logger.debug("Silent exception in %s", __name__, exc_info=True)
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
                        parsed = json.loads(error)
                        if isinstance(parsed, dict):
                            error = parsed.get("error_message") or parsed.get("message") or error
                    except (json.JSONDecodeError, TypeError):
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
                        logger.debug("Silent exception in %s", __name__, exc_info=True)
                return

            elif status == "IN_QUEUE":
                # RunPod 콜드 스타트: GPU 할당 후 Docker 이미지 풀링 + 컨테이너 시작 중
                # 이 단계에서 GPU 과금이 시작되지만, 핸들러 실행 전이므로 IN_QUEUE 유지됨
                # 대형 이미지(CUDA+Applio+Demucs)는 5~15분 소요 가능 → 타임아웃 없음 (전체 job 타임아웃이 보호)
                mins = elapsed // 60
                secs = elapsed % 60
                if mins >= 5:
                    queue_msg = f"GPU 콜드 스타트 중... ({mins}분 {secs}초) — Docker 이미지 로딩 중"
                elif mins >= 1:
                    queue_msg = f"GPU 할당 완료, 컨테이너 시작 중... ({mins}분 {secs}초)"
                else:
                    queue_msg = f"GPU 할당 대기 중... ({elapsed}초)"
                update_job(job_id, status="running",
                          progress=min(5, elapsed // 12),
                          message=queue_msg)

            elif status == "IN_PROGRESS":
                pct, msg = _extract_progress_info(result, job_type, elapsed)
                update_job(job_id, status="running", progress=pct, message=msg)

            else:
                # Unknown status — log and treat as transient
                update_job(job_id, status="running",
                          message=f"상태: {status} ({elapsed}초)")

            # 작업 타입별 타임아웃: train=10시간, preprocess=1시간, convert=30분
            job_timeout = _JOB_TIMEOUTS.get(job_type, 3600)
            if elapsed > job_timeout:
                with _poll_errors_lock:
                    _poll_error_counts.pop(job_id, None)
                timeout_label = f"{job_timeout // 3600}시간" if job_timeout >= 3600 else f"{job_timeout // 60}분"
                update_job(job_id, status="failed", message=f"시간 초과 ({timeout_label})")
                return

        except Exception as e:
            with _poll_errors_lock:
                poll_errors = _poll_error_counts.get(job_id, 0) + 1
                _poll_error_counts[job_id] = poll_errors

            # 점진적 백오프: 5→10→30→60초 (장시간 네트워크 장애에도 CPU 낭비 최소화)
            if poll_errors <= 12:       # ~1분: 5초 간격
                backoff = 5
            elif poll_errors <= 36:     # ~5분: 10초 간격
                backoff = 10
            elif poll_errors <= 96:     # ~30분: 30초 간격
                backoff = 30
            else:                       # 30분 이후: 60초 간격
                backoff = 60

            # 작업 유형별 최대 폴링 실패 허용: train=~11시간, preprocess=~5시간, convert=~2시간
            _MAX_POLL_ERRORS = {"train": 720, "preprocess": 360, "convert": 180}
            max_errors = _MAX_POLL_ERRORS.get(job_type, 360)

            if poll_errors > max_errors:
                with _poll_errors_lock:
                    _poll_error_counts.pop(job_id, None)
                update_job(job_id, status="failed",
                          message=f"네트워크 장애로 RunPod 상태 확인 불가 ({poll_errors}회 실패): {e}")
                return

            mins = (poll_errors * backoff) // 60
            update_job(job_id, status="running",
                      message=f"네트워크 연결 끊김 — 자동 재연결 대기 중 ({poll_errors}회, ~{mins}분)")
            time.sleep(backoff)
    except Exception as _poll_fatal:
        # 폴링 스레드 예상치 못한 크래시 — 작업을 failed로 마킹하여 영구 running 상태 방지
        logger.critical("poll_runpod_job crashed for %s: %s", job_id, _poll_fatal)
        _mark_job_failed_with_conversion(job_id, job_type, "내부 오류로 작업이 중단되었습니다. 다시 시도해 주세요.")
    finally:
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
    logger.info("[Preprocess] Received output keys: %s", list(output.keys()))
    logger.info("[Preprocess] segment_count=%s, total_duration=%s, segments=%d, accomp=%d, vocals=%d",
               output.get("segment_count"), total_duration, len(segments), len(accomp_files), len(vocal_files))
    if not segments and output.get("segment_count", 0) > 0:
        logger.warning("[Preprocess] Handler reported %s segments but segments array is empty! "
                       "RunPod response may have been truncated due to payload size limit.",
                       output.get("segment_count"))
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
                    logger.error("[Preprocess] Failed to download %s: %s", orig_name, e)
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
            logger.error("[Preprocess] DB update failed for file_ids (will retry next poll): %s", e)
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
            logger.warning("Failed to parse metadata: %s", e)
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
        logger.warning("metadata write failed: %s", e)
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
                logger.warning("[Download] 재시도 %d/%d: %s (%s)", attempt + 1, retries, dest.name, e)
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
                _download_with_retry(output["pth_url"], model_dir / pth_filename, timeout=300)
                pth_path = f"{model_name}/{pth_filename}"  # DB에 상대경로 저장 (이식성)
            # Fallback: base64 encoded model (when R2 upload failed)
            elif output.get("pth_base64"):
                pth_filename = output.get("pth_filename", f"{model_name}.pth")
                raw = base64.b64decode(output["pth_base64"])
                with open(model_dir / pth_filename, "wb") as f:
                    f.write(raw)
                pth_path = f"{model_name}/{pth_filename}"  # DB에 상대경로 저장
                logger.info("[Train] Model saved via base64 fallback: %s", pth_filename)

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
                _download_with_retry(output["index_url"], model_dir / idx_filename, timeout=120)
                index_path = f"{model_name}/{idx_filename}"  # DB에 상대경로 저장
            # Fallback: base64 encoded index
            elif output.get("index_base64"):
                idx_filename = output.get("index_filename", f"{model_name}.index")
                raw = base64.b64decode(output["index_base64"])
                with open(model_dir / idx_filename, "wb") as f:
                    f.write(raw)
                index_path = f"{model_name}/{idx_filename}"  # DB에 상대경로 저장
                logger.info("[Train] Index saved via base64 fallback: %s", idx_filename)

            try:
                with _training_lock:
                    _train_files_json = json.dumps(_training_file_map.pop(job_id, []), ensure_ascii=False)
                    _pretrained = _training_pretrained_map.pop(job_id, "klm49")
                with get_db() as db:
                    db.execute("""
                        INSERT INTO voice_models (name, pth_path, index_path, pth_url, index_url,
                                                epochs, training_time_seconds, training_files_json,
                                                pretrained_model, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready')
                    """, (model_name, pth_path, index_path,
                          output.get("pth_url"), output.get("index_url"),
                          output.get("epochs_trained", 0),
                          output.get("training_time_seconds", 0),
                          _train_files_json, _pretrained))
            except Exception as db_err:
                # DB 삽입 실패 시 다운로드된 모델 파일 정리 (고아 파일 방지)
                logger.error("[Train] DB insert failed, cleaning up model files: %s", db_err)
                shutil.rmtree(model_dir, ignore_errors=True)
                raise

            update_job(job_id, status="completed", progress=100,
                      message="학습 완료!",
                      result_json=json.dumps({"model_name": model_name}))

        elif job_type == "convert":
            # 변환 파일 저장 (보컬 + 믹스) — R2 URL 또는 inline base64
            out_filename = Path(output.get("filename", f"converted_{job_id[:8]}.wav")).name  # path traversal 방지
            out_path = OUTPUT_DIR / out_filename
            has_vocals = False

            # R2 URL → 다운로드 (재시도 포함 — GPU 결과물 손실 방지)
            if output.get("converted_audio_url"):
                _download_with_retry(output["converted_audio_url"], out_path, timeout=600)
                file_size = out_path.stat().st_size if out_path.exists() else 0
                logger.info("[Convert] Downloaded vocals from R2: %s bytes", f"{file_size:,}")
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
                mixed_filename = Path(output.get("mixed_filename", f"mixed_{job_id[:8]}.wav")).name  # path traversal 방지
                mixed_path = OUTPUT_DIR / mixed_filename
                if output.get("mixed_audio_url"):
                    _download_with_retry(output["mixed_audio_url"], mixed_path, timeout=300)
                    result_data["mixed_file"] = mixed_filename
                    logger.info("[Convert] Downloaded mixed from R2: %s bytes", f"{mixed_path.stat().st_size:,}")
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
                logger.debug("Silent exception in %s", __name__, exc_info=True)
    finally:
        with _poll_errors_lock:
            _poll_error_counts.pop(job_id, None)


def _run_batched_preprocess(job_id: str, batches: list[list[dict]]):
    """대용량 파일의 다중 배치 전처리를 순차 실행.
    각 배치를 별도 RunPod 작업으로 제출하고 완료를 기다린 뒤 다음 배치로 진행."""
    try:
        _run_batched_preprocess_inner(job_id, batches)
    except Exception as e:
        logger.error("[BatchPreprocess] Unexpected crash for job %s: %s", job_id, e)
        import traceback
        traceback.print_exc()
        update_job(job_id, status="failed",
                   message=f"배치 전처리 중 예기치 않은 오류: {e}")

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
                    pct = pct_base + max(0, min(int(90 / total_batches) - 2, int(elapsed / 3)))
                    state_msg = "GPU 대기 중..." if status == "IN_QUEUE" else "전처리 중..."
                    update_job(job_id, status="running", progress=pct,
                               message=f"{batch_label} {state_msg} ({elapsed}초)")

                if elapsed > 3600:  # 배치당 1시간 타임아웃
                    # RunPod 작업도 취소하여 과금 중지
                    if runpod_job_id and runpod_client.is_configured():
                        try:
                            runpod_client.cancel_runpod_job(runpod_job_id)
                        except Exception:
                            logger.debug("Silent exception in %s", __name__, exc_info=True)
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
    recover_orphan_jobs_on_startup()

    # Windows ProactorEventLoop에서 브라우저 연결 끊김 시 발생하는
    # ConnectionResetError [WinError 10054] 로그 억제
    loop = asyncio.get_event_loop()
    _original_handler = loop.get_exception_handler()

    def _suppress_connection_reset(loop, context):
        exc = context.get("exception")
        if isinstance(exc, ConnectionResetError):
            return  # 무시 — 브라우저가 연결을 먼저 닫은 것
        if _original_handler:
            _original_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_suppress_connection_reset)
    # 30분마다 오래된 청크 업로드 세션 자동 정리
    def _chunk_cleanup_loop():
        while True:
            time.sleep(1800)
            try:
                _cleanup_stale_chunk_uploads()
            except Exception:
                logger.debug("Silent exception in %s", __name__, exc_info=True)
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
        if len(content) > MAX_UPLOAD_FILE_SIZE:
            raise HTTPException(413,
                f"파일 '{f.filename}'이 너무 큽니다 ({len(content) / 1024 / 1024:.0f}MB). "
                f"최대 {MAX_UPLOAD_FILE_SIZE // 1024 // 1024}MB까지 지원합니다. 대용량 파일은 청크 업로드를 사용하세요.")
        file_hash = hashlib.sha256(content).hexdigest()

        # 동일 파일 중복 체크 + 이전 전처리 상태 확인 (단일 트랜잭션으로 레이스 컨디션 방지)
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
            prev = db.execute(
                "SELECT preprocessed FROM training_files WHERE file_hash=? AND deleted=1 ORDER BY id DESC LIMIT 1",
                (file_hash,)
            ).fetchone()
            was_preprocessed = 1 if (prev and prev["preprocessed"]) else 0

        safe_filename = Path(f.filename).name  # path traversal 방지
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_filename}"
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
            _remove_chunk_session(upload_id)
            raise HTTPException(400, f"지원하지 않는 파일 형식: {ext}")

        # 청크 병합 → 임시 파일 (path traversal 방지)
        safe_filename = Path(filename).name
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_filename}"
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
            _remove_chunk_session(upload_id)
            raise HTTPException(500, f"청크 병합 실패: {merge_err}")

        if total_size > MAX_CHUNK_FILE_SIZE:
            save_path.unlink(missing_ok=True)
            shutil.rmtree(chunk_dir, ignore_errors=True)
            _remove_chunk_session(upload_id)
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
            _remove_chunk_session(upload_id)
            raise HTTPException(409,
                f"동일한 파일이 이미 존재합니다: '{dup['original_name']}'")

        shutil.rmtree(chunk_dir, ignore_errors=True)
        _remove_chunk_session(upload_id)

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
    epochs: int = Form(150),           # v35: KLM49 + 한국어 커뮤니티: 50-150 epoch
    sample_rate: int = Form(40000),    # v32: 커뮤니티 표준 40k
    batch_size: int = Form(4),         # v35: 한국어 커뮤니티: batch 4 권장
    f0_method: str = Form("rmvpe"),
    pretrained_model: str = Form("klm49"),  # "klm49" 한국어 / "rin_e3" 다국어(팝송)
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
    if pretrained_model not in ("klm49", "rin_e3"):
        raise HTTPException(400, f"사전학습 모델은 klm49/rin_e3 중 하나여야 합니다. (입력: {pretrained_model})")

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
                logger.warning("Failed to parse metadata: %s", e)

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
            logger.info("[Train] Segment filter: %d mapped → %d found on disk", len(selected_seg_names), len(preprocessed_files))
        else:
            # 매핑 없으면 (이전 전처리) 전체 사용
            logger.info("[Train] No segment mapping found, using all %d segments", len(all_preprocessed))
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
        _training_pretrained_map[job_id] = pretrained_model
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
            import time as _time
            logger.info("[Train] Uploading %d files (%s bytes, %.1fMB) to R2...", len(file_paths), f"{total_size:,}", total_size/1024/1024)
            r2_client, r2_bucket_name = _get_r2_client()
            upload_start = _time.time()
            uploaded_bytes = 0
            for i, fp in enumerate(file_paths):
                r2_key = f"train/{job_id}/{Path(fp).name}"
                file_size = Path(fp).stat().st_size
                try:
                    url = upload_to_r2(Path(fp), r2_key, s3_client=r2_client, bucket_name=r2_bucket_name)
                    audio_urls.append({"filename": Path(fp).name, "url": url})
                    uploaded_bytes += file_size
                    # 10개마다 또는 마지막 파일일 때 진행률 출력
                    if (i + 1) % 10 == 0 or i == len(file_paths) - 1:
                        elapsed = _time.time() - upload_start
                        pct = (i + 1) / len(file_paths) * 100
                        speed = uploaded_bytes / 1024 / 1024 / max(elapsed, 0.1)
                        logger.info("[Train] R2 upload: %d/%d (%.0f%%) — %.1fMB, %.1fMB/s, %.0fs elapsed", i+1, len(file_paths), pct, uploaded_bytes/1024/1024, speed, elapsed)
                except Exception as e:
                    logger.error("[Train] R2 upload failed at file %d/%d (%s): %s", i+1, len(file_paths), Path(fp).name, e)
                    update_job(job_id, status="failed",
                        message=f"학습 데이터 R2 업로드 실패 ({i+1}/{len(file_paths)}): {e}")
                    raise HTTPException(500, f"학습 데이터 R2 업로드 실패: {e}")
            total_elapsed = _time.time() - upload_start
            logger.info("[Train] All %d files uploaded to R2 in %.1fs (%.1fMB, %.1fMB/s)", len(audio_urls), total_elapsed, total_size/1024/1024, total_size/1024/1024/max(total_elapsed,0.1))

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
                "pretrained_model": pretrained_model,
                "bucket_name": r2_bucket,
            }
        else:
            # base64 인라인 (소용량)
            batches = prepare_files_for_runpod(file_paths)
            if len(batches) > 1:
                raise HTTPException(400,
                    "학습 데이터가 너무 큽니다. R2 스토리지를 설정하거나 파일 수를 줄여주세요.")
            payload = {
                "task_type": "train",
                "model_name": model_name,
                "audio_files": batches[0],
                "sample_rate": sample_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "f0_method": f0_method,
                "pretrained_model": pretrained_model,
                "bucket_name": r2_bucket,
            }

        payload_size = len(json.dumps(payload))
        logger.info("[Train] Payload size: %s bytes, segments: %d", f"{payload_size:,}", total_segments)

        runpod_job_id = runpod_client.submit_job(payload)
        if not runpod_job_id:
            raise RuntimeError("RunPod 작업 ID를 받지 못했습니다. 잠시 후 다시 시도해주세요.")

        msg = f"GPU에 작업 제출됨 ({total_segments}개 세그먼트)"
        update_job(job_id, status="running", progress=5,
                  message=msg, runpod_job_id=runpod_job_id)

        _spawn_polling_thread(job_id, runpod_job_id, "train")

    except HTTPException:
        raise

    except Exception as e:
        error_msg = classify_runpod_error(e)
        update_job(job_id, status="failed", message=f"제출 실패: {error_msg}")
        with _training_lock:
            _training_file_map.pop(job_id, None)
            _training_pretrained_map.pop(job_id, None)
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

            _spawn_polling_thread(job_id, runpod_job_id, "preprocess")
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
        raise HTTPException(500, f"전처리 제출 실패: {error_msg}")

    return {"job_id": job_id, "segments": total_segments, "batches": len(batches),
            "skipped": skipped, "processing": len(unprocessed)}


def _preprocess_status_sync():
    """전처리 상태 확인 (동기) — asyncio.to_thread로 호출."""
    files = _list_preprocessed_files()

    with get_db() as db:
        total_files = db.execute("SELECT COUNT(*) FROM training_files WHERE deleted=0").fetchone()[0]
        unprocessed = db.execute(
            "SELECT COUNT(*) FROM training_files WHERE preprocessed=0 AND deleted=0"
        ).fetchone()[0]

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
    processed_count = total_files - unprocessed

    total_dur = 0.0
    meta_path = PREPROCESSED_DIR / "_metadata.json"
    if meta_path.exists():
        try:
            total_dur = json.loads(meta_path.read_text(encoding="utf-8")).get("total_duration", 0.0)
        except Exception as e:
            logger.warning("Failed to parse metadata: %s", e)

    # 메타데이터 없으면 파일 크기 기반으로 빠르게 추정 (수백 개 파일을 열지 않음)
    if total_dur == 0.0 and has_segments:
        for f in training_segments:
            try:
                size = f.stat().st_size
                ext = f.suffix.lower()
                if ext == ".wav":
                    total_dur += size / 96000.0
                elif ext == ".mp3":
                    total_dur += size / 24000.0
                else:
                    total_dur += size / 96000.0
            except Exception:
                logger.debug("Silent exception in %s", __name__, exc_info=True)

    return {
        "preprocessed": has_segments,  # 세그먼트가 있으면 학습 가능 (전체 완료 불필요)
        "has_segments": has_segments,
        "segment_count": len(training_segments),
        "total_duration": round(total_dur, 2),
        "unprocessed_count": unprocessed,
        "processed_count": processed_count,
        "accompaniment_files": accomp_files,
        "vocal_files": vocal_files,
    }

@app.get("/api/preprocess/status")
async def preprocess_status():
    """전처리 상태 확인 — 이벤트 루프 블로킹 방지를 위해 스레드에서 실행."""
    return await asyncio.to_thread(_preprocess_status_sync)


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
    index_rate: float = Form(0.40),   # v36: 0.35→0.40 (장홍권 음색 더 반영, 커뮤니티 0.3-0.5 권장)
    f0_method: str = Form("rmvpe"),
    vocal_volume: float = Form(1.0),
    mr_volume: float = Form(1.0),
    clean_audio: str = Form("false"),
    clean_strength: float = Form(0.7),
    protect: float = Form(0.35),        # v36: 0.40→0.35 (자음/숨소리 보호 + 자연스러운 전환)
    rms_mix_rate: float = Form(0.0),    # v36: 0.25→0.0 (원곡 다이나믹스 100% 보존 — 기계음 최대 원인)
    filter_radius: int = Form(3),
    hop_length: int = Form(64),
    post_reverb: float = Form(0.05),
    harmonic_enhance: str = Form("false"),
    high_note_mode: str = Form("false"),
    harmony_filter: float = Form(0.0),
    separate_vocals: str = Form("true"),
    vocal_pitch_pre_shift: int = Form(0),
    audio: UploadFile = File(...)
):
    if not runpod_client.is_configured():
        raise HTTPException(400, "RunPod API 설정이 필요합니다.")

    # 파라미터 범위 검증
    if not (-24 <= pitch_shift <= 24):
        raise HTTPException(400, f"피치는 -24~24 사이여야 합니다. (입력: {pitch_shift})")
    if not (0.0 <= index_rate <= 1.0):
        raise HTTPException(400, f"인덱스 비율은 0.0~1.0 사이여야 합니다. (입력: {index_rate})")
    # Protect 범위 초과 시 자동 클램프 (이전 버전 호환 + 슬라이더 오차 허용)
    if protect > 0.5:
        logger.info("[Convert] protect 값 %.2f → 0.50 으로 클램프", protect)
        protect = 0.5
    if protect < 0.0:
        protect = 0.0
    if not (0.0 <= rms_mix_rate <= 1.0):
        raise HTTPException(400, f"RMS Mix는 0.0~1.0 사이여야 합니다. (입력: {rms_mix_rate})")
    if not (0 <= filter_radius <= 12):
        raise HTTPException(400, f"Filter Radius는 0~12 사이여야 합니다. (입력: {filter_radius})")
    if hop_length not in (32, 64, 128, 256, 512):
        hop_length = 64  # 잘못된 값은 기본값으로
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

    # FormData boolean 파싱 (JS는 "true"/"false" 문자열 전송)
    clean_audio = _parse_form_bool(clean_audio, False)
    harmonic_enhance = _parse_form_bool(harmonic_enhance, False)
    high_note_mode = _parse_form_bool(high_note_mode, False)
    separate_vocals = _parse_form_bool(separate_vocals, True)

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

    # 모델 URL 준비 — presigned URL은 7일 후 만료되므로 항상 새로 생성
    pth_url = None
    index_url = None
    if model["pth_url"]:
        try:
            pth_url = refresh_presigned_url(model["pth_url"])
        except Exception as e:
            logger.error("[Convert] pth presigned URL 갱신 실패: %s", e)
            # job 생성 전이므로 update_job 없이 바로 HTTP 에러 반환
            raise HTTPException(500, f"모델 파일 URL 갱신 실패 (R2 설정 확인 필요): {e}")
    if model["index_url"]:
        try:
            index_url = refresh_presigned_url(model["index_url"])
        except Exception as e:
            logger.warning("[Convert] index presigned URL 갱신 실패: %s, index 없이 진행", e)
            # index는 선택사항이므로 없이 진행 (pth와 달리 치명적이지 않음)

    # Job + 변환 기록 동시 생성 (원자성: RunPod 제출 전에 모든 DB 레코드 삽입)
    job_id = uuid.uuid4().hex[:12]
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
            "audio_filename": Path(audio.filename).name,  # path traversal 방지
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
            "high_note_mode": high_note_mode,
            "harmony_filter": max(0.0, min(1.0, harmony_filter)),
            "vocal_pitch_pre_shift": max(-12, min(12, vocal_pitch_pre_shift)),
            "bucket_name": r2_bucket,
        }

        # 오디오 파일을 R2에 업로드 (10 MB 페이로드 한도 회피)
        audio_r2_key = f"convert/{job_id}/{Path(audio.filename).name}"
        try:
            audio_url = upload_to_r2(temp_path, audio_r2_key)
            payload["audio_url"] = audio_url
        except Exception as e:
            logger.error("[Convert] R2 audio upload failed: %s", e)
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
                    logger.info("[Convert] Uploaded model pth to R2: %s", pth_r2_key)
                except Exception as e:
                    logger.error("[Convert] R2 model upload failed: %s", e)
                    # 임시 파일 정리
                    if temp_path and Path(temp_path).exists():
                        Path(temp_path).unlink(missing_ok=True)
                    update_job(job_id, status="failed",
                        message=f"모델 R2 업로드 실패: {e}")
                    raise HTTPException(500, f"모델 R2 업로드 실패: {e}")

        if not index_url and model["index_path"]:
            local_idx = Path(model["index_path"])
            if not local_idx.is_absolute():
                local_idx = MODEL_DIR / local_idx
            if local_idx.exists():
                try:
                    idx_r2_key = f"convert/{job_id}/model.index"
                    index_url = upload_to_r2(local_idx, idx_r2_key)
                    payload["index_url"] = index_url
                    logger.info("[Convert] Uploaded model index to R2: %s", idx_r2_key)
                except Exception as e:
                    logger.error("[Convert] R2 index upload failed: %s", e)
                    # index는 선택사항이므로 계속 진행

        # 페이로드 크기 로깅 (디버깅용)
        payload_keys = {k: (len(v) if isinstance(v, str) and len(v) > 100 else v)
                        for k, v in payload.items()}
        payload_size = len(json.dumps(payload))
        logger.info("[Convert] Payload size: %s bytes, keys: %s", f"{payload_size:,}", payload_keys)

        runpod_job_id = runpod_client.submit_job(payload)

        if not runpod_job_id:
            raise RuntimeError("RunPod 작업 ID를 받지 못했습니다. 잠시 후 다시 시도해주세요.")

        # 임시 입력 파일 정리 — RunPod 제출 성공 후에만 삭제 (재개 시 R2 URL 사용)
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            logger.debug("Silent exception in %s", __name__, exc_info=True)

        update_job(job_id, status="running", progress=10,
                  message="GPU 변환 시작", runpod_job_id=runpod_job_id)

        # 변환 기록 상태 업데이트 (이미 'pending'으로 삽입됨 → 'processing'으로 전환)
        with get_db() as db:
            db.execute("UPDATE conversions SET status='processing' WHERE job_id=?", (job_id,))

        _spawn_polling_thread(job_id, runpod_job_id, "convert")

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
            logger.debug("Silent exception in %s", __name__, exc_info=True)
        # 임시 입력 파일 정리 (실패 시에도 누수 방지)
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            logger.debug("Silent exception in %s", __name__, exc_info=True)
        raise HTTPException(500, f"변환 제출 실패: {error_msg}")

    return {"job_id": job_id}


# ─── 작업 상태 API ───

@app.get("/api/jobs/active")
async def get_active_jobs():
    """페이지 로드 시 프론트엔드 상태 복원용.
    running/submitting 상태의 최신 작업을 job_type별 1개씩 반환."""
    with get_db() as db:
        rows = db.execute("""
            SELECT id, job_type, status, progress, message, created_at, started_at
            FROM jobs
            WHERE status IN ('running', 'submitting')
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
            logger.debug("Silent exception in %s", __name__, exc_info=True)
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

    # 활성 폴링 스레드 종료 대기 (최대 6초 — 폴링 주기 5초 + 여유 1초)
    with _poll_threads_lock:
        poll_thread = _active_poll_threads.get(job_id)
    if poll_thread and poll_thread.is_alive():
        poll_thread.join(timeout=6)
        if poll_thread.is_alive():
            logger.warning("[Cancel] 폴링 스레드 %s 가 6초 내 종료되지 않음", job_id)

    # 활성 상태 정리
    with _poll_errors_lock:
        _poll_error_counts.pop(job_id, None)
    with _poll_threads_lock:
        _active_poll_threads.pop(job_id, None)
    with _preprocess_lock:
        preprocess_file_map.pop(job_id, None)
    with _training_lock:
        _training_file_map.pop(job_id, None)
        _training_pretrained_map.pop(job_id, None)

    return {"status": "cancelled"}


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """[제거됨] 일시정지 기능은 제거되었습니다. 취소(cancel)를 사용하세요."""
    raise HTTPException(410, "일시정지 기능이 제거되었습니다. 취소를 사용하세요.")


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """[제거됨] 재개 기능은 제거되었습니다. 처음부터 다시 시작하세요."""
    raise HTTPException(410, "재개 기능이 제거되었습니다. 처음부터 다시 시작하세요.")


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
                    logger.debug("Silent exception in %s", __name__, exc_info=True)
        result = db.execute("""
            UPDATE jobs
            SET status='cancelled',
                message='수동 정리',
                updated_at=?
            WHERE status IN ('running', 'submitting')
        """, (datetime.now().isoformat(),))
    with _poll_errors_lock:
        _poll_error_counts.clear()
    with _preprocess_lock:
        preprocess_file_map.clear()
    with _training_lock:
        _training_file_map.clear()
        _training_pretrained_map.clear()
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
                logger.debug("Silent exception in %s", __name__, exc_info=True)
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
        # 실행 중인 변환 작업이 있으면 삭제 거부
        active = db.execute(
            "SELECT id FROM conversions WHERE model_id=? AND status IN ('pending','processing')",
            (model_id,)
        ).fetchall()
        if active:
            raise HTTPException(409, f"이 모델을 사용 중인 변환 작업이 {len(active)}개 있습니다. 먼저 작업을 취소하거나 완료하세요.")
        dirs_to_check = set()
        for path_key in ["pth_path", "index_path"]:
            if row[path_key]:
                p = Path(row[path_key])
                if not p.is_absolute():
                    p = MODEL_DIR / p  # 상대경로 → 절대경로
                if p.exists():
                    dirs_to_check.add(p.parent)
                    p.unlink()
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
                        logger.debug("Silent exception in %s", __name__, exc_info=True)
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
    await asyncio.to_thread(maybe_cleanup_stale_jobs)
    await asyncio.to_thread(_cleanup_stale_chunk_uploads)

    health = {
        "status": "ok",
        "server": "running",
        "timestamp": datetime.now().isoformat(),
    }

    if runpod_client.is_configured():
        try:
            resp = await asyncio.to_thread(
                requests.get,
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
# 클라우드 동기화 (PC 간 작업 이전)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SYNC_PREFIX = "sync/"  # R2 동기화 전용 prefix


def _r2_key_from_presigned(url: str, bucket: str) -> str:
    """presigned URL에서 R2 object key 추출."""
    from urllib.parse import urlparse, unquote
    parsed = urlparse(url)
    path = unquote(parsed.path)
    if path.startswith(f"/{bucket}/"):
        return path[len(f"/{bucket}/"):]
    return path.lstrip("/")


def _sync_upload_dir(s3, bucket: str, local_dir: Path, r2_prefix: str) -> int:
    """로컬 디렉토리의 모든 파일을 R2에 업로드. 업로드한 파일 수 반환."""
    count = 0
    if not local_dir.exists():
        return count
    for fpath in local_dir.rglob("*"):
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(local_dir).as_posix()
        key = f"{r2_prefix}{rel}"
        try:
            s3.upload_file(str(fpath), bucket, key)
            count += 1
        except Exception as e:
            logger.warning("[Sync] Upload skip %s: %s", rel, e)
    return count


def _sync_download_prefix(s3, bucket: str, r2_prefix: str, local_dir: Path) -> int:
    """R2 prefix 아래 모든 오브젝트를 로컬 디렉토리에 다운로드. 다운로드한 파일 수 반환."""
    count = 0
    local_dir.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=r2_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(r2_prefix):]
            if not rel:
                continue
            dest = local_dir / rel.replace("/", os.sep)
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(bucket, key, str(dest))
                count += 1
            except Exception as e:
                logger.warning("[Sync] Download skip %s: %s", rel, e)
    return count


def _sync_list_prefix(s3, bucket: str, r2_prefix: str) -> list[dict]:
    """R2 prefix 아래 오브젝트 목록."""
    items = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=r2_prefix):
        for obj in page.get("Contents", []):
            items.append({"key": obj["Key"], "size": obj["Size"],
                          "modified": obj["LastModified"].isoformat()})
    return items


@app.post("/api/sync/backup")
async def sync_backup():
    """현재 PC의 DB + 파일을 R2에 백업 (다른 PC에서 복원 가능)."""
    s3, bucket = _get_r2_client()
    result = {"uploaded": {}}

    # 1) studio.db 백업
    if DB_PATH.exists():
        # WAL 모드 대비: 별도 연결로 안전한 백업 생성
        backup_path = DATA_DIR / "studio_backup.db"
        src = sqlite3.connect(str(DB_PATH))
        try:
            dst = sqlite3.connect(str(backup_path))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        s3.upload_file(str(backup_path), bucket, f"{_SYNC_PREFIX}studio.db")
        backup_path.unlink(missing_ok=True)
        result["uploaded"]["database"] = True
        logger.info("[Sync] DB 백업 완료")

    # 2) uploads 디렉토리
    n = _sync_upload_dir(s3, bucket, UPLOAD_DIR, f"{_SYNC_PREFIX}uploads/")
    result["uploaded"]["uploads"] = n

    # 3) models 디렉토리
    n = _sync_upload_dir(s3, bucket, MODEL_DIR, f"{_SYNC_PREFIX}models/")
    result["uploaded"]["models"] = n

    # 4) output 디렉토리
    n = _sync_upload_dir(s3, bucket, OUTPUT_DIR, f"{_SYNC_PREFIX}output/")
    result["uploaded"]["output"] = n

    # 5) preprocessed 디렉토리
    n = _sync_upload_dir(s3, bucket, PREPROCESSED_DIR, f"{_SYNC_PREFIX}preprocessed/")
    result["uploaded"]["preprocessed"] = n

    # 6) config.json 백업 (API 키 등)
    if CONFIG_PATH.exists():
        s3.upload_file(str(CONFIG_PATH), bucket, f"{_SYNC_PREFIX}config.json")
        result["uploaded"]["config"] = True

    total = sum(v for v in result["uploaded"].values() if isinstance(v, int))
    logger.info("[Sync] 백업 완료: 총 %d개 파일", total)
    result["message"] = f"백업 완료: 총 {total}개 파일 업로드됨"
    return result


@app.post("/api/sync/restore")
async def sync_restore():
    """R2에서 DB + 파일을 다운로드하여 현재 PC에 복원."""
    s3, bucket = _get_r2_client()
    result = {"downloaded": {}}

    # 0) config.json 복원 (R2 설정은 이미 있으므로, 나머지 설정만 머지)
    try:
        local_cfg = load_config()
        tmp_cfg = DATA_DIR / "config_remote.json"
        s3.download_file(bucket, f"{_SYNC_PREFIX}config.json", str(tmp_cfg))
        with open(tmp_cfg, "r", encoding="utf-8") as f:
            remote_cfg = json.load(f)
        # R2 키는 현재 PC 값 유지, 나머지만 머지
        for k, v in remote_cfg.items():
            if k.startswith("r2_") or k == "runpod_api_key" or k == "runpod_endpoint_id":
                continue
            local_cfg[k] = v
        save_config(local_cfg)
        tmp_cfg.unlink(missing_ok=True)
        result["downloaded"]["config"] = True
    except Exception as e:
        logger.debug("[Sync] config 복원 스킵: %s", e)

    # 1) studio.db 복원
    try:
        db_tmp = DATA_DIR / "studio_remote.db"
        s3.download_file(bucket, f"{_SYNC_PREFIX}studio.db", str(db_tmp))
        # 기존 DB가 있으면 백업
        if DB_PATH.exists():
            bak = DATA_DIR / f"studio_before_sync_{int(time.time())}.db"
            shutil.copy2(str(DB_PATH), str(bak))
            logger.info("[Sync] 기존 DB 백업: %s", bak.name)
        shutil.move(str(db_tmp), str(DB_PATH))
        init_db()  # 마이그레이션 적용
        result["downloaded"]["database"] = True
        logger.info("[Sync] DB 복원 완료")
    except Exception as e:
        logger.warning("[Sync] DB 복원 실패: %s", e)
        result["downloaded"]["database"] = False

    # 2) uploads
    n = _sync_download_prefix(s3, bucket, f"{_SYNC_PREFIX}uploads/", UPLOAD_DIR)
    result["downloaded"]["uploads"] = n

    # 3) models
    n = _sync_download_prefix(s3, bucket, f"{_SYNC_PREFIX}models/", MODEL_DIR)
    result["downloaded"]["models"] = n

    # 4) output
    n = _sync_download_prefix(s3, bucket, f"{_SYNC_PREFIX}output/", OUTPUT_DIR)
    result["downloaded"]["output"] = n

    # 5) preprocessed
    n = _sync_download_prefix(s3, bucket, f"{_SYNC_PREFIX}preprocessed/", PREPROCESSED_DIR)
    result["downloaded"]["preprocessed"] = n

    # 6) DB의 presigned URL 갱신 (만료 대비)
    refreshed = 0
    try:
        with get_db() as db:
            models = db.execute("SELECT id, pth_url, index_url FROM voice_models").fetchall()
            for m in models:
                mid, pth_url, idx_url = m
                updates = {}
                if pth_url:
                    try:
                        updates["pth_url"] = refresh_presigned_url(pth_url)
                    except Exception as e:
                        logger.debug("[Sync] pth URL 갱신 실패 (model %s): %s", mid, e)
                if idx_url:
                    try:
                        updates["index_url"] = refresh_presigned_url(idx_url)
                    except Exception as e:
                        logger.debug("[Sync] index URL 갱신 실패 (model %s): %s", mid, e)
                if updates:
                    sets = ", ".join(f"{k}=?" for k in updates)
                    db.execute(f"UPDATE voice_models SET {sets} WHERE id=?",
                               [*updates.values(), mid])
                    refreshed += 1
    except Exception as e:
        logger.debug("[Sync] URL 갱신 중 오류: %s", e)
    result["downloaded"]["urls_refreshed"] = refreshed

    total = sum(v for v in result["downloaded"].values() if isinstance(v, int))
    logger.info("[Sync] 복원 완료: 총 %d개 파일", total)
    result["message"] = f"복원 완료: 총 {total}개 파일 다운로드됨"
    return result


@app.get("/api/sync/status")
async def sync_status():
    """R2 동기화 상태 조회 (클라우드에 백업된 데이터 현황)."""
    s3, bucket = _get_r2_client()

    categories = {
        "database": f"{_SYNC_PREFIX}studio.db",
        "config": f"{_SYNC_PREFIX}config.json",
        "uploads": f"{_SYNC_PREFIX}uploads/",
        "models": f"{_SYNC_PREFIX}models/",
        "output": f"{_SYNC_PREFIX}output/",
        "preprocessed": f"{_SYNC_PREFIX}preprocessed/",
    }
    result = {}

    for name, prefix in categories.items():
        if name in ("database", "config"):
            # 단일 파일: 존재 여부 + 최종 수정 시간
            try:
                resp = s3.head_object(Bucket=bucket, Key=prefix)
                result[name] = {
                    "exists": True,
                    "size": resp["ContentLength"],
                    "last_modified": resp["LastModified"].isoformat(),
                }
            except Exception as e:
                logger.debug("[Sync] head_object 실패 (%s): %s", prefix, e)
                result[name] = {"exists": False}
        else:
            # 디렉토리: 파일 수 + 총 크기
            total_size = 0
            count = 0
            latest = None
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    count += 1
                    total_size += obj["Size"]
                    mod = obj["LastModified"]
                    if latest is None or mod > latest:
                        latest = mod
            result[name] = {
                "count": count,
                "total_size_mb": round(total_size / (1024 * 1024), 1),
                "last_modified": latest.isoformat() if latest else None,
            }

    # 로컬 현황
    local = {}
    for name, local_dir in [("uploads", UPLOAD_DIR), ("models", MODEL_DIR),
                             ("output", OUTPUT_DIR), ("preprocessed", PREPROCESSED_DIR)]:
        if local_dir.exists():
            files = [f for f in local_dir.rglob("*") if f.is_file()]
            local[name] = {"count": len(files),
                           "total_size_mb": round(sum(f.stat().st_size for f in files) / (1024 * 1024), 1)}
        else:
            local[name] = {"count": 0, "total_size_mb": 0}
    local["database"] = {"exists": DB_PATH.exists()}

    return {"cloud": result, "local": local}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    logger.info("AI Voice Studio 시작 중...")
    logger.info("http://localhost:8000 에서 접속하세요")

    if not FROZEN:
        # 개발 모드: 브라우저 자동 열기 (.exe 모드에서는 pywebview가 처리)
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
