#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_client.py — AI Voice Studio 변환 Python 클라이언트 (v72)

서버(server.py)가 실행 중일 때 이 스크립트로 변환을 제출할 수 있습니다.
웹 UI 없이 터미널에서 직접 변환을 제어하거나 자동화할 때 사용합니다.

사용법:
  python convert_client.py list-models          # 사용 가능한 모델 목록
  python convert_client.py presets              # 5곡 프리셋 상세 출력
  python convert_client.py convert gidarilge    # 기다릴게 변환
  python convert_client.py convert comethru     # comethru 변환
  python convert_client.py convert breaking     # Breaking Through 변환
  python convert_client.py convert monster      # Monster 변환
  python convert_client.py convert lovers       # Lovers rough 변환
  python convert_client.py all                  # 5곡 전체 순차 변환 (자동)
  python convert_client.py convert custom --file <경로> [--f0-method fcpe] ...
  python convert_client.py status <job_id>      # 작업 상태 조회
  python convert_client.py jobs                 # 최근 작업 목록
  python convert_client.py download <job_id>    # 결과 파일 다운로드

서버 주소 변경:
  SERVER_URL 환경변수 또는 --server 옵션 사용
  예) SERVER_URL=http://192.168.1.100:8000 python convert_client.py all
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

# Windows 터미널 UTF-8 출력 강제
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    import requests
except ImportError:
    print("requests 라이브러리가 필요합니다: pip install requests")
    sys.exit(1)

# ──────────────────────────────────────────────
# 기본 설정
# ──────────────────────────────────────────────
DEFAULT_SERVER = os.environ.get("SERVER_URL", "http://127.0.0.1:8000")
ROOT = Path(__file__).parent  # ai_record_studio 디렉토리

# ──────────────────────────────────────────────
# 공통 변환 파라미터 (v72 기준 — CLAUDE.md 동기화)
# ──────────────────────────────────────────────
_COMMON = {
    # ── RVC 핵심 파라미터 ──
    "index_rate":           0.50,    # FAISS index 반영 비율 (v55: 0.45→0.50)
    "rms_mix_rate":         0.15,    # 원본 RMS 유지 비율 (v55: 0.20→0.15)
    "protect":              0.40,    # 자음 보호 (v64: 0.50→0.40; 0.5=자음보호비활성 — AI Hub 공식)
    "filter_radius":        3,       # 미디언 F0 스무딩 (v62: 2→3; Applio 3.x에선 내부 무시)
    "hop_length":           128,     # 피치 추적 간격 (v49: 64→128; 64는 노이즈 추적→삑사리)
    # ── 오토튠 ──
    "f0_autotune":          "true",  # 피치 안정화 — 노래 변환 시 항상 ON (Applio 공식 권장)
    "f0_autotune_strength": 0.2,     # 오토튠 강도 (v57: 0.4→0.2; 피치 상방편향 +2~3반음 교정)
    # ── 바이패스 (v62-v70) ──
    "harmony_bypass":       "true",  # 화음/폴리포닉 구간 → 원본 보컬 유지
    "falsetto_bypass":      "true",  # 팔세토 불안정 구간(360Hz이상/instability>1.0st) → 원본 유지
    "noisy_bypass":         "true",  # 기계음/뭉개짐 구간(flatness>0.10) → 원본 유지
    "female_bypass":        "false", # 여성 보컬 감지 바이패스 — 곡별로 설정
    # ── 믹싱 ──
    "vocal_blend":          0.0,     # 원본 보컬 블렌딩 비율 (v45: 0% 고정; 더블링 원인)
    "vocal_volume":         1.0,
    "mr_volume":            1.0,
    # ── 전처리/후처리 ──
    "separate_vocals":      "true",  # 보컬/MR 분리 (BS-Roformer → Demucs 폴백)
    "embedder_model":       "contentvec",  # My Voice v50 학습 시 사용한 embedder
    "clean_audio":          "false",
    "clean_strength":       0.7,
    "post_reverb":          0.0,
    "harmonic_enhance":     "true",  # AI 금속성 질감 제거 (배음 강화)
    "high_note_mode":       "false",
}

# ──────────────────────────────────────────────
# 5곡 사전 정의 프리셋 (v72 기준)
# ──────────────────────────────────────────────
PRESETS = {
    "gidarilge": {
        "label": "플레이브 - 기다릴게",
        "file": ROOT / "플레이브 - 기다릴게.mp3",
        "params": {
            **_COMMON,
            "f0_method":       "fcpe",   # 팔세토 구간 多(520Hz대) — FCPE가 Applio 3.x 팔세토 권장
            "language":        "ko",     # 한국어 EQ (300/600Hz 감쇄 생략, 비음 포먼트 보존)
            "pitch_shift":     -3,       # 원키 대비 3반음 낮춤 — 고음 안정화
            "falsetto_bypass": "true",   # 20-30s(옥타브 에러) / 95-118s(instability>1.0st) 원본 대체
            "harmony_bypass":  "true",   # 1:43-52 화음 구간 원본 대체
            "noisy_bypass":    "true",
            "female_bypass":   "false",  # 남성곡 — 여성 바이패스 불필요
        },
        "notes": (
            "팔세토 구간(95-118s, 20-30s 옥타브 에러)은 falsetto_bypass가 원본으로 대체.\n"
            "화음 구간(103-112s)은 harmony_bypass가 원본 유지.\n"
            "pitch=-3은 MR 키 맞춤 — 원키 필요 시 0으로 변경.\n"
            "FCPE: Applio 3.x에서 팔세토/고음 최적 피치 추출 방식 (hybrid 대체)."
        ),
    },

    "comethru": {
        "label": "Jeremy Zucker - comethru ft. Bea Miller",
        "file": ROOT / "Jeremy Zucker - comethru ft. Bea Miller.mp3",
        "params": {
            **_COMMON,
            "f0_method":       "rmvpe",  # 안정적 남성 보컬 위주 — RMVPE 충분
            "language":        "en",     # 영어 EQ (300Hz -0.3 + 600Hz -0.3 + 7.5kHz -0.5 추가)
            "pitch_shift":     0,
            "female_bypass":   "true",   # ★ 핵심: 1:18-36 Bea Miller 여성 구간 원본 유지
            "harmony_bypass":  "true",
            "falsetto_bypass": "true",
            "noisy_bypass":    "true",
        },
        "notes": (
            "female_bypass=true 가 핵심.\n"
            "v62에서 female_f0_thresh 200→280Hz 수정: 남성 보컬(98-260Hz)은 모델 정상 적용.\n"
            "1:18-36 여성 보컬 구간은 원본 Bea Miller 목소리 유지."
        ),
    },

    "breaking": {
        "label": "Breaking Through (4824 Wave)",
        "file": ROOT / "01_Breaking Through (4824 Wave).wav",
        "params": {
            **_COMMON,
            "f0_method":       "fcpe",   # 전반적 기계음/가래 — FCPE로 안정성 확보
            "language":        "en",
            "pitch_shift":     0,
            "harmony_bypass":  "true",
            "noisy_bypass":    "true",   # 기계음/뭉개짐 구간(flatness>0.10) 원본 대체
            "falsetto_bypass": "true",
            "female_bypass":   "false",
        },
        "notes": (
            "전반적으로 기계음/가래가 심했던 곡.\n"
            "noisy_bypass(spectral flatness>0.10)가 뭉개짐 구간 자동 원본 대체.\n"
            "FCPE: RMVPE 대비 팔세토/고음 피치 추출 더 안정적."
        ),
    },

    "monster": {
        "label": "Monster",
        "file": ROOT / "Monster.wav",
        "params": {
            **_COMMON,
            "f0_method":       "fcpe",   # 팔세토 多(15-17s, 90-118s) — FCPE 권장
            "language":        "en",
            "pitch_shift":     0,
            "falsetto_bypass": "true",   # 15-17s(360Hz 무조건) / 90-92s / 99-118s 원본 대체
            "harmony_bypass":  "true",   # 1:30-32 더블링 구간
            "noisy_bypass":    "true",
            "female_bypass":   "false",
        },
        "notes": (
            "v65 bypass 임계치로 문제 구간 전부 포착:\n"
            "  15-17s: median 370Hz → 360Hz unconditional threshold\n"
            "  90-92s: instability_p95=1.475st > 1.0st threshold\n"
            "  99-118s: instability_p95=1.015st > 1.0st (아슬아슬)\n"
            "주의: Monster 원본 파일 없음 → MR 분리 불가 (separate_vocals=true 여도 MR 미분리)."
        ),
    },

    "lovers": {
        "label": "Lovers rough",
        "file": ROOT / "180427 Lovers rough.mp3",
        "params": {
            **_COMMON,
            "f0_method":       "rmvpe",  # 노이즈 구간 제외 시 RMVPE 충분
            "language":        "en",
            "pitch_shift":     0,
            "noisy_bypass":    "true",   # 0-17s: spectral flatness_p95=0.963 (순수 노이즈에 가까움)
            "harmony_bypass":  "true",
            "falsetto_bypass": "true",
            "female_bypass":   "false",
        },
        "notes": (
            "0-17s 구간: spectral flatness_p95=0.963 (1.0=순수 노이즈).\n"
            "noisy_bypass(flatness>0.10)가 해당 구간 자동 원본 대체.\n"
            "해당 구간 제외 시 RMVPE로 충분히 안정적."
        ),
    },
}


# ──────────────────────────────────────────────
# API 헬퍼
# ──────────────────────────────────────────────

class VoiceStudioClient:
    def __init__(self, server_url: str = DEFAULT_SERVER):
        self.base = server_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def health(self) -> dict:
        return self.session.get(self._url("/api/health"), timeout=10).json()

    def list_models(self) -> list:
        r = self.session.get(self._url("/api/models"), timeout=10)
        r.raise_for_status()
        return r.json()

    def list_jobs(self) -> list:
        r = self.session.get(self._url("/api/jobs"), timeout=10)
        r.raise_for_status()
        return r.json()

    def job_status(self, job_id: str) -> dict:
        r = self.session.get(self._url(f"/api/jobs/{job_id}"), timeout=10)
        r.raise_for_status()
        return r.json()

    def convert(self, audio_path: Path, model_id: int, params: dict) -> str:
        """변환 작업 제출. 반환값: job_id"""
        if not audio_path.exists():
            raise FileNotFoundError(f"파일 없음: {audio_path}")

        form = {k: str(v) for k, v in params.items() if k != "pitch_shift"}
        form["model_id"] = str(model_id)
        form["pitch_shift"] = str(params.get("pitch_shift", 0))

        # WAV 파일은 audio/wav, 나머지는 audio/mpeg
        mime = "audio/wav" if audio_path.suffix.lower() == ".wav" else "audio/mpeg"
        with open(audio_path, "rb") as f:
            files = {"audio": (audio_path.name, f, mime)}
            r = self.session.post(
                self._url("/api/convert"),
                data=form,
                files=files,
                timeout=120,
            )

        if r.status_code != 200:
            raise RuntimeError(f"변환 제출 실패 {r.status_code}: {r.text[:500]}")

        return r.json()["job_id"]

    def wait_for_job(self, job_id: str, poll_sec: float = 10.0, timeout_min: float = 90.0) -> dict:
        """작업 완료까지 폴링 대기. 반환값: 최종 작업 상태"""
        deadline = time.time() + timeout_min * 60
        while time.time() < deadline:
            status = self.job_status(job_id)
            state = status.get("status", "?")
            progress = status.get("progress", 0)
            message = status.get("message", "")
            print(f"  [{state:12s}] {progress:3d}%  {message[:50]}", end="\r", flush=True)

            if state in ("completed", "failed", "cancelled"):
                print()
                return status

            time.sleep(poll_sec)

        print()
        raise TimeoutError(f"작업 {job_id} 타임아웃 ({timeout_min}분 초과)")

    def download_result(self, job_id: str, out_dir: Path = None) -> tuple[Path, Path | None]:
        """완료된 작업의 결과 파일 다운로드. 반환값: (변환보컬경로, 믹스경로|None)"""
        status = self.job_status(job_id)
        if status.get("status") != "completed":
            raise RuntimeError(f"작업이 완료되지 않았습니다: {status.get('status')}")

        out_dir = out_dir or Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)

        def _dl(filename: str) -> Path:
            path = out_dir / filename
            r = self.session.get(self._url(f"/api/download/{filename}"), stream=True, timeout=300)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            return path

        vocal_file = status.get("output_filename")
        mixed_file = status.get("mixed_filename")

        if not vocal_file:
            raise RuntimeError("출력 파일명을 찾을 수 없습니다.")

        vocal_path = _dl(vocal_file)
        mixed_path = _dl(mixed_file) if mixed_file else None
        return vocal_path, mixed_path


# ──────────────────────────────────────────────
# CLI 명령어 구현
# ──────────────────────────────────────────────

def _get_model_id(client: VoiceStudioClient, model_id_arg: int | None) -> tuple[int, str]:
    """모델 ID를 결정하고 (id, name) 반환."""
    if model_id_arg:
        return model_id_arg, f"ID={model_id_arg}"
    models = client.list_models()
    if not models:
        print("오류: 등록된 모델이 없습니다. 먼저 학습을 완료하세요.")
        sys.exit(1)
    return models[0]["id"], models[0]["name"]


def _print_params(params: dict) -> None:
    """변환 파라미터 요약 출력."""
    print(f"  F0 방식:    {params.get('f0_method', 'rmvpe')}")
    print(f"  언어:       {params.get('language', 'auto')}")
    print(f"  피치:       {params.get('pitch_shift', 0):+d}반음")
    bypass_parts = [
        f"화음={'ON' if params.get('harmony_bypass') == 'true' else 'OFF'}",
        f"팔세토={'ON' if params.get('falsetto_bypass') == 'true' else 'OFF'}",
        f"노이즈={'ON' if params.get('noisy_bypass') == 'true' else 'OFF'}",
        f"여성={'ON' if params.get('female_bypass') == 'true' else 'OFF'}",
    ]
    print(f"  바이패스:   {' / '.join(bypass_parts)}")
    print(f"  배음강화:   {'ON' if params.get('harmonic_enhance') == 'true' else 'OFF'}")
    print(f"  protect={params.get('protect')} / index_rate={params.get('index_rate')} / "
          f"rms={params.get('rms_mix_rate')} / autotune_strength={params.get('f0_autotune_strength')}")


def cmd_list_models(client: VoiceStudioClient, _args):
    models = client.list_models()
    if not models:
        print("등록된 모델 없음")
        return
    print(f"{'ID':>4}  {'이름':<32}  {'에폭':>5}  {'SR':>6}  {'생성일'}")
    print("-" * 72)
    for m in models:
        print(f"{m['id']:>4}  {m['name']:<32}  {m.get('epochs', '?'):>5}  "
              f"{m.get('sample_rate', '?'):>6}  {m.get('created_at', '')[:10]}")


def _run_single_convert(client: VoiceStudioClient, preset_key: str,
                        model_id: int, model_name: str,
                        no_wait: bool, download_dir: str | None,
                        yes: bool) -> bool:
    """단일 프리셋 변환 실행. 성공 시 True 반환."""
    preset = PRESETS[preset_key]
    audio_path = preset["file"]
    params = dict(preset["params"])
    label = preset["label"]

    print(f"\n{'='*62}")
    print(f"  곡:    {label}")
    print(f"  파일:  {audio_path.name}")
    print(f"  모델:  [{model_id}] {model_name}")
    print(f"{'='*62}")
    if preset.get("notes"):
        print(f"\n[참고]\n{preset['notes']}\n")
    _print_params(params)

    if not audio_path.exists():
        print(f"\n오류: 파일이 존재하지 않습니다 — {audio_path}")
        return False

    if not yes:
        ans = input("\n변환을 제출하시겠습니까? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("건너뜁니다.")
            return False

    print("\n변환 제출 중...")
    job_id = client.convert(audio_path, model_id, params)
    print(f"  job_id: {job_id}")

    if no_wait:
        print(f"  나중에 확인: python convert_client.py status {job_id}")
        return True

    print("\n완료 대기 중 (최대 90분)...")
    final = client.wait_for_job(job_id, poll_sec=10, timeout_min=90)

    if final["status"] == "completed":
        print(f"\n변환 완료!")
        if download_dir:
            out_dir = Path(download_dir)
            vocal_path, mixed_path = client.download_result(job_id, out_dir)
            print(f"  변환 보컬: {vocal_path}")
            if mixed_path:
                print(f"  믹스:      {mixed_path}")
        else:
            print(f"  변환 파일: {final.get('output_filename', '?')}")
            if final.get("mixed_filename"):
                print(f"  믹스 파일: {final.get('mixed_filename')}")
            print(f"  다운로드:  python convert_client.py download {job_id}")
        return True
    else:
        print(f"\n변환 실패: {final.get('message', '?')}")
        return False


def cmd_convert(client: VoiceStudioClient, args):
    preset_key = args.preset

    if preset_key == "custom":
        if not args.file:
            print("오류: custom 변환 시 --file 옵션이 필요합니다.")
            sys.exit(1)
        audio_path = Path(args.file)
        params = dict(_COMMON)
        if args.f0_method:
            params["f0_method"] = args.f0_method
        if args.language:
            params["language"] = args.language
        if args.pitch is not None:
            params["pitch_shift"] = args.pitch
        if args.no_female_bypass:
            params["female_bypass"] = "false"
        elif args.female_bypass:
            params["female_bypass"] = "true"

        model_id, model_name = _get_model_id(client, args.model_id)
        print(f"\n{'='*62}")
        print(f"  곡:    {audio_path.stem}")
        print(f"  파일:  {audio_path.name}")
        print(f"  모델:  [{model_id}] {model_name}")
        print(f"{'='*62}")
        _print_params(params)

        if not audio_path.exists():
            print(f"\n오류: 파일이 존재하지 않습니다 — {audio_path}")
            sys.exit(1)

        if not args.yes:
            ans = input("\n변환을 제출하시겠습니까? [Y/n] ").strip().lower()
            if ans not in ("", "y", "yes"):
                print("취소되었습니다.")
                return

        print("\n변환 제출 중...")
        job_id = client.convert(audio_path, model_id, params)
        print(f"  job_id: {job_id}")

        if args.no_wait:
            print(f"  나중에 확인: python convert_client.py status {job_id}")
            return

        print("\n완료 대기 중 (최대 90분)...")
        final = client.wait_for_job(job_id, poll_sec=10, timeout_min=90)
        if final["status"] == "completed":
            print(f"\n변환 완료!")
            if args.download:
                vocal_path, mixed_path = client.download_result(job_id, Path(args.download))
                print(f"  변환 보컬: {vocal_path}")
                if mixed_path:
                    print(f"  믹스:      {mixed_path}")
            else:
                print(f"  변환 파일: {final.get('output_filename', '?')}")
                print(f"  다운로드:  python convert_client.py download {job_id}")
        else:
            print(f"\n변환 실패: {final.get('message', '?')}")
        return

    # 프리셋 변환
    if preset_key not in PRESETS:
        print(f"알 수 없는 프리셋: {preset_key}")
        print(f"사용 가능: {', '.join(PRESETS.keys())}, custom")
        sys.exit(1)

    model_id, model_name = _get_model_id(client, args.model_id)
    _run_single_convert(
        client, preset_key, model_id, model_name,
        no_wait=args.no_wait,
        download_dir=args.download,
        yes=args.yes,
    )


def cmd_all(client: VoiceStudioClient, args):
    """5곡 전체를 순차 변환."""
    model_id, model_name = _get_model_id(client, args.model_id)

    print(f"\n5곡 전체 순차 변환 시작")
    print(f"모델: [{model_id}] {model_name}")
    print(f"다운로드 경로: {args.download or '(저장 생략 — 수동 download 명령 사용)'}")

    results = {}
    for key in PRESETS:
        print(f"\n{'─'*62}")
        print(f"[{list(PRESETS.keys()).index(key)+1}/5] {PRESETS[key]['label']}")
        ok = _run_single_convert(
            client, key, model_id, model_name,
            no_wait=False,
            download_dir=args.download,
            yes=True,  # 일괄 변환은 확인 없이 진행
        )
        results[key] = "완료" if ok else "실패"

    print(f"\n{'='*62}")
    print("전체 변환 결과:")
    for key, result in results.items():
        label = PRESETS[key]["label"]
        icon = "✓" if result == "완료" else "✗"
        print(f"  {icon} {label:<40} {result}")


def cmd_status(client: VoiceStudioClient, args):
    status = client.job_status(args.job_id)
    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))


def cmd_jobs(client: VoiceStudioClient, _args):
    jobs = client.list_jobs()
    if not jobs:
        print("작업 없음")
        return
    print(f"{'ID':<14}  {'타입':<10}  {'상태':<12}  {'진행':>4}  메시지")
    print("-" * 72)
    for j in jobs[:20]:
        print(f"{j['id']:<14}  {j.get('job_type','?'):<10}  {j['status']:<12}  "
              f"{j.get('progress',0):>3}%  {j.get('message','')[:42]}")


def cmd_download(client: VoiceStudioClient, args):
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    print(f"다운로드 중 → {out_dir}")
    vocal_path, mixed_path = client.download_result(args.job_id, out_dir)
    print(f"저장 완료:")
    print(f"  변환 보컬: {vocal_path}")
    if mixed_path:
        print(f"  믹스:      {mixed_path}")


def cmd_show_presets(_client, _args):
    """사전 정의 프리셋 전체 상세 출력"""
    for key, preset in PRESETS.items():
        p = preset["params"]
        print(f"\n{'='*62}")
        print(f"  프리셋 키: {key}")
        print(f"  곡명:      {preset['label']}")
        print(f"  파일:      {preset['file'].name}")
        print(f"  {'─'*58}")
        print(f"  f0_method:           {p.get('f0_method')}")
        print(f"  language:            {p.get('language')}")
        print(f"  pitch_shift:         {p.get('pitch_shift', 0):+d}반음")
        print(f"  index_rate:          {p.get('index_rate')}")
        print(f"  rms_mix_rate:        {p.get('rms_mix_rate')}")
        print(f"  protect:             {p.get('protect')}")
        print(f"  filter_radius:       {p.get('filter_radius')}")
        print(f"  hop_length:          {p.get('hop_length')}")
        print(f"  f0_autotune:         {p.get('f0_autotune')}  (strength={p.get('f0_autotune_strength')})")
        print(f"  harmony_bypass:      {p.get('harmony_bypass')}")
        print(f"  falsetto_bypass:     {p.get('falsetto_bypass')}")
        print(f"  noisy_bypass:        {p.get('noisy_bypass')}")
        print(f"  female_bypass:       {p.get('female_bypass')}")
        print(f"  harmonic_enhance:    {p.get('harmonic_enhance')}")
        print(f"  vocal_blend:         {p.get('vocal_blend')}  (0=더블링 없음)")
        print(f"  embedder_model:      {p.get('embedder_model')}")
        print(f"  separate_vocals:     {p.get('separate_vocals')}")
        if preset.get("notes"):
            print(f"\n  [참고]\n  {preset['notes'].replace(chr(10), chr(10)+'  ')}")


# ──────────────────────────────────────────────
# CLI 파서
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Voice Studio 변환 Python 클라이언트 (v72)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python convert_client.py all -y --download ./results\n"
            "  python convert_client.py convert gidarilge -y\n"
            "  python convert_client.py convert custom --file song.mp3 --f0-method fcpe --pitch -2\n"
        ),
    )
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"서버 주소 (기본값: {DEFAULT_SERVER})")

    sub = parser.add_subparsers(dest="command")

    # list-models
    sub.add_parser("list-models", help="사용 가능한 모델 목록 출력")

    # presets
    sub.add_parser("presets", help="5곡 프리셋 상세 출력 (서버 연결 불필요)")

    # convert
    conv = sub.add_parser("convert", help="단일 곡 변환 제출")
    conv.add_argument("preset",
                      choices=list(PRESETS.keys()) + ["custom"],
                      help="사전 정의 프리셋 키 또는 'custom'")
    conv.add_argument("--model-id", type=int, default=None, dest="model_id",
                      help="사용할 모델 ID (미지정 시 첫 번째 모델 자동 선택)")
    conv.add_argument("--file", help="custom 프리셋 시 오디오 파일 경로")
    conv.add_argument("--f0-method", default=None, dest="f0_method",
                      choices=["rmvpe", "fcpe", "crepe", "crepe-tiny", "harvest", "pm"],
                      help="F0 피치 추출 방식 (기본: rmvpe / 팔세토 많은 곡: fcpe)")
    conv.add_argument("--language", choices=["ko", "en", "auto"], default=None,
                      help="언어 EQ 프리셋 (ko=한국어, en=영어, auto=자동)")
    conv.add_argument("--pitch", type=int, default=None,
                      help="피치 조정 (반음 단위, -24~+24, 기본 0)")
    conv.add_argument("--female-bypass", action="store_true", dest="female_bypass",
                      help="여성 보컬 구간 바이패스 활성화 (duet/ft. 곡에 사용)")
    conv.add_argument("--no-female-bypass", action="store_true", dest="no_female_bypass",
                      help="여성 보컬 바이패스 비활성화 (custom 기본값 재정의)")
    conv.add_argument("--no-wait", action="store_true", dest="no_wait",
                      help="제출 후 완료 대기 없이 즉시 반환")
    conv.add_argument("--download", metavar="DIR",
                      help="완료 후 결과를 해당 디렉토리에 저장")
    conv.add_argument("-y", "--yes", action="store_true",
                      help="확인 없이 즉시 제출")

    # all — 5곡 일괄 변환
    all_cmd = sub.add_parser("all", help="5곡 전체 순차 변환 (자동 yes)")
    all_cmd.add_argument("--model-id", type=int, default=None, dest="model_id",
                         help="사용할 모델 ID (미지정 시 첫 번째 모델 자동 선택)")
    all_cmd.add_argument("--download", metavar="DIR",
                         help="완료 후 결과를 해당 디렉토리에 저장")

    # status
    st = sub.add_parser("status", help="작업 상태 조회")
    st.add_argument("job_id")

    # jobs
    sub.add_parser("jobs", help="최근 작업 목록 (최대 20개)")

    # download
    dl = sub.add_parser("download", help="완료된 작업 결과 다운로드")
    dl.add_argument("job_id")
    dl.add_argument("--out-dir", default=None, dest="out_dir",
                    help="저장 디렉토리 (기본값: 현재 디렉토리)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # presets 명령은 서버 연결 불필요
    if args.command == "presets":
        cmd_show_presets(None, args)
        return

    client = VoiceStudioClient(args.server)
    try:
        client.health()
    except Exception as e:
        print(f"서버에 연결할 수 없습니다 ({args.server}): {e}")
        print("server.py가 실행 중인지 확인하세요: python server.py")
        sys.exit(1)

    dispatch = {
        "list-models": cmd_list_models,
        "convert":     cmd_convert,
        "all":         cmd_all,
        "status":      cmd_status,
        "jobs":        cmd_jobs,
        "download":    cmd_download,
    }
    dispatch[args.command](client, args)


if __name__ == "__main__":
    main()
