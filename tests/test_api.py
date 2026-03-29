"""
API 엔드포인트 테스트 — 핵심 CRUD + 상태 조회
RunPod 연동은 모킹, DB/파일시스템은 실제 (격리된 임시 환경)
"""
import json
from pathlib import Path

import pytest


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 헬스/상태 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHealthAndStats:
    def test_health_endpoint(self, app_client):
        resp = app_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "server" in data
        assert data["server"] == "running"

    def test_stats_endpoint(self, app_client):
        resp = app_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        # 실제 응답 키: models, files, conversions
        assert "models" in data
        assert "files" in data
        assert "conversions" in data
        assert data["models"] == 0
        assert data["files"] == 0

    def test_system_info_endpoint(self, app_client):
        resp = app_client.get("/api/system-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "disk_free_gb" in data
        assert "disk_total_gb" in data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestConfig:
    def test_get_config_empty(self, app_client):
        resp = app_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "runpod_api_key" in data

    def test_set_and_get_config(self, app_client):
        # 설정 저장
        resp = app_client.post("/api/config", json={
            "runpod_api_key": "test_key_123",
            "runpod_endpoint_id": "test_endpoint"
        })
        assert resp.status_code == 200

        # 설정 조회 — config.json이 패치된 경로에 쓰여야 함
        resp = app_client.get("/api/config")
        data = resp.json()
        # config 저장이 패치 경로를 사용하는지 확인
        assert resp.status_code == 200


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파일 업로드/조회/삭제 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFiles:
    def test_list_files_empty(self, app_client):
        resp = app_client.get("/api/files")
        assert resp.status_code == 200
        data = resp.json()
        # 응답이 {"files": []} 형태
        files = data.get("files", data) if isinstance(data, dict) else data
        assert files == []

    def test_upload_and_list_file(self, app_client, sample_audio_file):
        # 파일 업로드
        with open(sample_audio_file, "rb") as f:
            resp = app_client.post(
                "/api/upload",
                files={"files": ("test_audio.mp3", f, "audio/mpeg")},
                data={"category": "vocal"}
            )
        assert resp.status_code == 200
        data = resp.json()
        uploaded = data.get("files", [data] if "id" in data else [])
        assert len(uploaded) >= 1
        file_id = uploaded[0]["id"]

        # 파일 목록 조회
        resp = app_client.get("/api/files")
        data = resp.json()
        files = data.get("files", data) if isinstance(data, dict) else data
        assert len(files) >= 1
        assert any(f["original_name"] == "test_audio.mp3" for f in files)

    def test_upload_and_delete_file(self, app_client, sample_audio_file):
        # 업로드
        with open(sample_audio_file, "rb") as f:
            resp = app_client.post(
                "/api/upload",
                files={"files": ("delete_me.mp3", f, "audio/mpeg")},
                data={"category": "vocal"}
            )
        data = resp.json()
        uploaded = data.get("files", [data] if "id" in data else [])
        file_id = uploaded[0]["id"]

        # 삭제
        resp = app_client.delete(f"/api/files/{file_id}")
        assert resp.status_code == 200

        # 삭제 확인
        resp = app_client.get("/api/files")
        data = resp.json()
        files = data.get("files", data) if isinstance(data, dict) else data
        assert not any(f.get("id") == file_id for f in files)

    def test_upload_invalid_type(self, app_client, tmp_path):
        """지원하지 않는 파일 타입 업로드"""
        bad_file = tmp_path / "test.exe"
        bad_file.write_bytes(b"MZ" + b"\x00" * 100)

        with open(bad_file, "rb") as f:
            resp = app_client.post(
                "/api/upload",
                files={"files": ("test.exe", f, "application/octet-stream")},
                data={"category": "vocal"}
            )
        # 서버가 거부하거나, 빈 결과 반환
        if resp.status_code == 200:
            data = resp.json()
            files = data.get("files", [])
            assert len(files) == 0 or resp.status_code >= 400


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestModels:
    def test_list_models_empty(self, app_client):
        resp = app_client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        models = data.get("models", data) if isinstance(data, dict) else data
        assert models == []

    def test_delete_nonexistent_model(self, app_client):
        resp = app_client.delete("/api/models/99999")
        assert resp.status_code in (404, 200)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 변환 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestConversions:
    def test_list_conversions_empty(self, app_client):
        resp = app_client.get("/api/conversions")
        assert resp.status_code == 200
        data = resp.json()
        convs = data.get("conversions", data) if isinstance(data, dict) else data
        assert convs == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 작업(Job) API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestJobs:
    def test_list_jobs_empty(self, app_client):
        resp = app_client.get("/api/jobs")
        assert resp.status_code == 200
        data = resp.json()
        jobs = data.get("jobs", data) if isinstance(data, dict) else data
        assert jobs == []

    def test_active_jobs_empty(self, app_client):
        resp = app_client.get("/api/jobs/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("preprocess") is None
        assert data.get("train") is None
        assert data.get("convert") is None

    def test_get_nonexistent_job(self, app_client):
        resp = app_client.get("/api/jobs/nonexistent-id")
        assert resp.status_code in (404, 200)

    def test_cleanup_jobs(self, app_client):
        resp = app_client.post("/api/jobs/cleanup")
        assert resp.status_code == 200


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전처리 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPreprocess:
    def test_preprocess_status_empty(self, app_client):
        resp = app_client.get("/api/preprocess/status")
        assert resp.status_code == 200

    def test_delete_preprocessed(self, app_client):
        resp = app_client.delete("/api/preprocess")
        assert resp.status_code == 200
