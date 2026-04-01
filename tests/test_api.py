"""
API 엔드포인트 테스트 — 핵심 CRUD + 상태 조회
RunPod 연동은 모킹, DB/파일시스템은 실제 (격리된 임시 환경)
"""


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
        # 설정 저장 (Form 데이터)
        resp = app_client.post("/api/config", data={
            "api_key": "test_key_123",
            "endpoint_id": "test_endpoint"
        })
        assert resp.status_code == 200

        # 설정 조회 — API 키는 마스킹됨, endpoint_id는 그대로
        resp = app_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runpod_api_key"] == "***_123"  # 마지막 4자 + 마스크
        assert data["runpod_endpoint_id"] == "test_endpoint"
        assert data["is_configured"] is True


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
        """지원하지 않는 파일 타입 업로드 → 200이지만 빈 파일 목록"""
        bad_file = tmp_path / "test.exe"
        bad_file.write_bytes(b"MZ" + b"\x00" * 100)

        with open(bad_file, "rb") as f:
            resp = app_client.post(
                "/api/upload",
                files={"files": ("test.exe", f, "application/octet-stream")},
                data={"category": "vocal"}
            )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data.get("files", [])) == 0


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
        assert resp.status_code == 404


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
        assert resp.status_code == 404

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


class TestConfigR2:
    def test_save_r2_config(self, app_client):
        """R2 설정 저장 테스트"""
        resp = app_client.post("/api/config/r2", data={
            "r2_endpoint_url": "https://test.r2.cloudflarestorage.com",
            "r2_access_key_id": "test_key",
            "r2_secret_access_key": "test_secret",
            "r2_bucket_name": "test-bucket",
        })
        assert resp.status_code == 200

    def test_save_download_folder(self, app_client, tmp_path):
        """다운로드 폴더 설정 테스트"""
        folder = str(tmp_path / "downloads")
        resp = app_client.post("/api/config/download-folder", data={"folder": folder})
        assert resp.status_code == 200
        assert resp.json()["folder"] == folder


class TestModelManagement:
    def test_rename_nonexistent_model(self, app_client):
        """존재하지 않는 모델 이름 변경 → 404"""
        resp = app_client.post("/api/models/99999/rename", data={"name": "new_name"})
        assert resp.status_code == 404

    def test_quality_nonexistent_model(self, app_client):
        """존재하지 않는 모델 품질 점수 → 404"""
        resp = app_client.post("/api/models/99999/quality", data={"quality_score": "4.5"})
        assert resp.status_code == 404

    def test_quality_out_of_range(self, app_client):
        """품질 점수 범위 초과 → 400"""
        resp = app_client.post("/api/models/1/quality", data={"quality_score": "6.0"})
        assert resp.status_code in (400, 404)


class TestTrainValidation:
    def test_train_without_config(self, app_client):
        """RunPod 미설정 시 학습 → 400"""
        resp = app_client.post("/api/train", data={
            "model_name": "test_model",
            "epochs": "150",
            "sample_rate": "40000",
            "batch_size": "4",
            "f0_method": "rmvpe",
            "pretrained_model": "klm49",
        })
        assert resp.status_code == 400

    def test_train_invalid_pretrained(self, app_client):
        """잘못된 pretrained_model 값 → 400"""
        resp = app_client.post("/api/train", data={
            "model_name": "test_model",
            "pretrained_model": "invalid_model",
        })
        assert resp.status_code == 400

    def test_train_invalid_epochs(self, app_client):
        """에포크 범위 초과 → 400"""
        resp = app_client.post("/api/train", data={
            "model_name": "test_model",
            "epochs": "99999",
        })
        assert resp.status_code == 400

    def test_train_invalid_model_name(self, app_client):
        """모델명 특수문자 → 400"""
        resp = app_client.post("/api/train", data={
            "model_name": "test/model:bad",
        })
        assert resp.status_code == 400


class TestConvertValidation:
    def test_convert_without_config(self, app_client, sample_audio_file):
        """RunPod 미설정 시 변환 → 400 (config 미설정) 또는 404 (모델 없음)"""
        with open(sample_audio_file, "rb") as f:
            resp = app_client.post("/api/convert", data={
                "model_id": "1",
                "pitch_shift": "0",
                "pretrained_model": "klm49",
            }, files={"audio": ("test.mp3", f, "audio/mpeg")})
        # 빈 DB에서 model_id=1이 없으므로 404, 또는 config 미설정 400
        assert resp.status_code in (400, 404)

    def test_convert_invalid_pitch(self, app_client, sample_audio_file):
        """피치 범위 초과 → 400"""
        with open(sample_audio_file, "rb") as f:
            resp = app_client.post("/api/convert", data={
                "model_id": "1",
                "pitch_shift": "99",
            }, files={"audio": ("test.mp3", f, "audio/mpeg")})
        assert resp.status_code == 400


class TestSyncEndpoints:
    def test_sync_status_without_r2(self, app_client):
        """R2 미설정 시 동기화 상태 → 400"""
        resp = app_client.get("/api/sync/status")
        assert resp.status_code == 400

    def test_sync_backup_without_r2(self, app_client):
        """R2 미설정 시 백업 → 400"""
        resp = app_client.post("/api/sync/backup")
        assert resp.status_code == 400


class TestConversionHistory:
    def test_delete_nonexistent_conversion(self, app_client):
        """존재하지 않는 변환 삭제 → 404"""
        resp = app_client.delete("/api/conversions/99999")
        assert resp.status_code == 404


class TestJobPauseRemoved:
    def test_pause_returns_410(self, app_client):
        """제거된 pause 엔드포인트 → 410 Gone"""
        resp = app_client.post("/api/jobs/test123/pause")
        assert resp.status_code == 410

    def test_resume_returns_410(self, app_client):
        """제거된 resume 엔드포인트 → 410 Gone"""
        resp = app_client.post("/api/jobs/test123/resume")
        assert resp.status_code == 410


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 작업 취소 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestJobCancel:
    def test_cancel_nonexistent_job(self, app_client):
        """존재하지 않는 작업 취소 → 404"""
        resp = app_client.post("/api/jobs/nonexistent-id/cancel")
        assert resp.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파일 다운로드 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDownload:
    def test_download_nonexistent_file(self, app_client):
        """존재하지 않는 파일 다운로드 → 404"""
        resp = app_client.get("/api/download/nonexistent.wav")
        assert resp.status_code == 404

    def test_download_invalid_extension(self, app_client):
        """지원하지 않는 확장자 다운로드 → 400"""
        resp = app_client.get("/api/download/test.exe")
        assert resp.status_code == 400

    def test_download_path_traversal(self, app_client):
        """경로 탐색 시도 → 404 (파일 없음)"""
        resp = app_client.get("/api/download/..%2F..%2Fetc%2Fpasswd.wav")
        assert resp.status_code == 404

    def test_preprocess_download_nonexistent(self, app_client):
        """전처리 파일 다운로드 — 존재하지 않는 파일 → 404"""
        resp = app_client.get("/api/preprocess/download/nonexistent.wav")
        assert resp.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# save-to-folder API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSaveToFolder:
    def test_save_nonexistent_file(self, app_client):
        """존재하지 않는 파일 저장 → 404"""
        resp = app_client.post("/api/save-to-folder", data={"filename": "nonexistent.wav"})
        assert resp.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전처리 리셋 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPreprocessReset:
    def test_reset_preprocess_empty(self, app_client):
        """빈 파일 ID로 전처리 리셋 → 400"""
        resp = app_client.post("/api/preprocess/reset", data={"file_ids": ""})
        assert resp.status_code == 400

    def test_reset_preprocess_nonexistent_ids(self, app_client):
        """존재하지 않는 파일 ID로 전처리 리셋"""
        resp = app_client.post("/api/preprocess/reset", data={"file_ids": "99999"})
        assert resp.status_code == 200


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 청크 업로드 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestChunkUpload:
    def test_chunk_upload_invalid_extension(self, app_client, tmp_path):
        """지원하지 않는 파일 형식 청크 업로드 → 400"""
        chunk = tmp_path / "chunk0"
        chunk.write_bytes(b"\x00" * 100)
        with open(chunk, "rb") as f:
            resp = app_client.post("/api/upload/chunk", data={
                "upload_id": "test-upload-123",
                "chunk_index": "0",
                "total_chunks": "1",
                "filename": "malware.exe",
                "total_size": "100",
            }, files={"chunk": ("chunk0", f, "application/octet-stream")})
        assert resp.status_code == 400
