"""
pytest fixtures — 테스트용 격리 환경 설정
임시 디렉토리에서 DB/파일을 생성하여 실제 데이터에 영향 없음
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# server.py가 import 시점에 디렉토리를 생성하므로, import 전에 패치 필요
VOICE_STUDIO_DIR = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def tmp_data_dir():
    """세션 전체에서 공유하는 임시 데이터 디렉토리"""
    with tempfile.TemporaryDirectory(prefix="vs_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture()
def app_client(tmp_path):
    """각 테스트마다 격리된 FastAPI TestClient 생성"""
    # server.py 모듈-레벨 변수를 임시 경로로 패치
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    patches = {
        "server.DATA_DIR": data_dir,
        "server.UPLOAD_DIR": data_dir / "uploads",
        "server.MODEL_DIR": data_dir / "models",
        "server.OUTPUT_DIR": data_dir / "output",
        "server.CHUNK_DIR": data_dir / "chunks",
        "server.PREPROCESSED_DIR": data_dir / "preprocessed",
        "server.DB_PATH": data_dir / "test.db",
        "server.CONFIG_PATH": data_dir / "config.json",
    }

    # 디렉토리 생성
    for key, val in patches.items():
        if key.endswith("_DIR"):
            val.mkdir(parents=True, exist_ok=True)

    # 패치 적용
    with patch.multiple("server", **{k.split(".")[-1]: v for k, v in patches.items()}):
        # DB 초기화
        from server import init_db, app
        init_db()

        from fastapi.testclient import TestClient
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


@pytest.fixture()
def sample_audio_file(tmp_path):
    """테스트용 더미 오디오 파일 (실제 오디오 아님, API 흐름 테스트용)"""
    audio = tmp_path / "test_audio.mp3"
    # 최소한의 MP3 헤더 (실제 디코딩 불가하지만 업로드 테스트에는 충분)
    audio.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 1024)
    return audio
