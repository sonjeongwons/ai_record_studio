#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_client.py — AI Voice Studio 변환 Python 클라이언트 (v65)

서버(server.py)가 실행 중일 때 이 스크립트로 변환을 제출할 수 있습니다.
웹 UI 없이 터미널에서 직접 변환을 제어하거나 자동화할 때 사용합니다.

사용법:
  python convert_client.py list-models          # 사용 가능한 모델 목록
  python convert_client.py convert gidarilge    # 기다릴게 변환 (사전 정의 프리셋)
  python convert_client.py convert comethru
  python convert_client.py convert breaking
  python convert_client.py convert monster
  python convert_client.py convert lovers
  python convert_client.py convert custom <파일경로> [--model-id N] [--pitch N] ...
  python convert_client.py status <job_id>      # 작업 상태 조회
  python convert_client.py jobs                 # 최근 작업 목록
  python convert_client.py download <job_id>    # 결과 파일 다운로드

서버 주소 변경:
  SERVER_URL 환경변수 또는 --server 옵션 사용
  예) SERVER_URL=http://192.168.1.100:8000 python convert_client.py convert gidarilge
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

# requests 없으면 안내 후 종료
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
# 공통 변환 파라미터 (v65 기준 — CLAUDE.md 동기화)
# ──────────────────────────────────────────────
_COMMON = {
    # ── RVC 핵심 파라미터 ──
    "index_rate":          0.50,   # v55: 0.45→0.50 (FAISS index 반영 비율)
    "rms_mix_rate":        0.15,   # v55: 0.20→0.15 (원본 RMS 유지 비율)
    "protect":             0.40,   # v64: 0.50→0.40 (AI Hub 공식: 0.5=자음보호 완전비활성, 0.4=재활성)
    "filter_radius":       3,      # v62: 2→3 (미디언 F0 스무딩 재활성화, 팔세토 안정화)
    "hop_length":          128,    # v49: 64→128 (커뮤니티 표준; 64는 노이즈 추적→삑사리)
    # ── 오토튠 ──
    "f0_autotune":         "true", # v49: 노래 변환 시 피치 안정화 (항상 ON)
    "f0_autotune_strength": 0.2,   # v57: 0.4→0.2 (피치 상방편향 +2~3반음 교정)
    # ── 바이패스 (v62-v65 신규) ──
    "harmony_bypass":      "true", # v62: 화음/폴리포닉 구간 → 원본 유지
    "falsetto_bypass":     "true", # v62: 고음 불안정 구간 → 원본 유지
    "noisy_bypass":        "true", # v64: 기계음/뭉개짐 구간 → 원본 유지
    "female_bypass":       "false",# v60: 여성 보컬 감지 바이패스 (곡별로 다름)
    # ── 기타 ──
    "vocal_blend":         0.0,    # v45: 0% 고정 (더블링 원인 제거)
    "separate_vocals":     "true", # 보컬/MR 분리 (BS-Roformer→Demucs 폴백)
    "embedder_model":      "contentvec",  # My Voice v50 학습 시 사용한 embedder
    "vocal_volume":        1.0,
    "mr_volume":           1.0,
    "clean_audio":         "false",
    "clean_strength":      0.7,
    "post_reverb":         0.0,
    "harmonic_enhance":    "true",   # AI 금속성 질감 제거 — UI 기본 권장값
    "high_note_mode":      "false",
    "harmony_filter":      0.0,
    "vocal_pitch_pre_shift": 0,
}

# ──────────────────────────────────────────────
# 5곡 사전 정의 프리셋 (v65 기준)
# ──────────────────────────────────────────────
# 키 → (레이블, 파일경로, 파라미터 오버라이드)
PRESETS = {
    "gidarilge": {
        "label": "플레이브 - 기다릴게",
        "file": ROOT / "플레이브 - 기다릴게.mp3",
        "params": {
            **_COMMON,
            # ── 곡별 설정 ──
            "f0_method":  "fcpe",  # 팔세토 구간 많음(520Hz대) → FCPE (Applio 3.x hybrid 대체)
            "language":   "ko",    # 한국어 EQ: HPF 80Hz + 800Hz -1.0 + 5kHz -0.7 + Air
                                   #            (영어 추가 커트 300/600/7.5kHz는 생략)
            "pitch_shift": -3,     # 원키보다 3반음 낮춤 — 고음 안정화 (기존 고음곡 프리셋)
            "falsetto_bypass": "true",  # 20-30s / 95-118s / 180-183s 팔세토 구간 원본 대체
            "harmony_bypass":  "true",  # 1:43-52 화음 구간 원본 대체 ("으악" 아티팩트 방지)
            "noisy_bypass":    "true",
            "female_bypass":   "false", # 남성 곡 — 여성 바이패스 불필요
        },
        "notes": (
            "팔세토 구간(95-118s, 20-30s 옥타브 에러)은 falsetto_bypass가 원본으로 대체.\n"
            "화음 구간(103-112s 추정)은 harmony_bypass가 원본 유지.\n"
            "pitch=-3 은 MR 키와 맞추기 위한 설정 — 필요 시 조정."
        ),
    },

    "comethru": {
        "label": "Jeremy Zucker - comethru ft. Bea Miller",
        "file": ROOT / "Jeremy Zucker - comethru ft. Bea Miller.mp3",
        "params": {
            **_COMMON,
            "f0_method":  "rmvpe",   # 안정적인 남성 보컬 위주
            "language":   "en",      # 영어 EQ: 공통 + 300Hz -0.3 + 600Hz -0.3 + 7.5kHz -0.5
            "pitch_shift": 0,
            "female_bypass":   "true",   # ★ 핵심: 1:18-36 Bea Miller 여성 구간 원본 유지
                                         #   (v62에서 감지 임계치 200Hz→280Hz 수정 완료)
                                         #   남성 보컬(98-260Hz)은 모델 정상 적용됨
            "harmony_bypass":  "true",
            "falsetto_bypass": "true",
            "noisy_bypass":    "true",
        },
        "notes": (
            "female_bypass=true 가 핵심. v61에서 남성 보컬도 바이패스되던 버그는\n"
            "v62에서 female_f0_thresh 200→280Hz 수정으로 해결됨.\n"
            "1:18-36 여성 보컬 구간은 원본 Bea Miller 목소리 유지."
        ),
    },

    "breaking": {
        "label": "Breaking Through (4824 Wave)",
        "file": ROOT / "01_Breaking Through (4824 Wave).wav",
        "params": {
            **_COMMON,
            "f0_method":  "fcpe",  # 전반 기계음/가래 → FCPE (Applio 3.x hybrid 대체)
            "language":   "en",
            "pitch_shift": 0,
            "harmony_bypass":  "true",
            "noisy_bypass":    "true",   # 전반적 기계음 구간 원본 대체
            "falsetto_bypass": "true",
            "female_bypass":   "false",
        },
        "notes": (
            "전반적으로 기계음/가래가 심했던 곡. noisy_bypass(flatness>0.10)가\n"
            "뭉개짐 구간을 원본으로 자동 대체."
        ),
    },

    "monster": {
        "label": "Monster",
        "file": ROOT / "Monster.wav",  # 원본 없음 → MR만 분리 안 됨, 보컬 직접 변환
        "params": {
            **_COMMON,
            "f0_method":  "fcpe",  # 15-17s / 90-118s 팔세토 구간 많음 → FCPE (Applio 3.x 권장)
            "language":   "en",
            "pitch_shift": 0,
            "falsetto_bypass": "true",  # 15-17s(370Hz→360Hz 임계 포착) / 90-92s(instab 1.475st>1.0) / 99-118s(instab 1.015st)
            "harmony_bypass":  "true",  # 1:30-32 더블링 구간
            "noisy_bypass":    "true",
            "female_bypass":   "false",
        },
        "notes": (
            "v65 임계치 강화로 모든 문제 구간 포착:\n"
            "  15-17s: median=370Hz → 360Hz unconditional threshold 포착\n"
            "  90-92s: instability_p95=1.475st → 1.0st threshold 포착\n"
            "  99-118s: instability_p95=1.015st → 1.0st threshold 포착 (아슬아슬)\n"
            "Monster는 원본 파일 없어 MR 분리 불가 → separate_vocals=true 여도 MR 미분리."
        ),
    },

    "lovers": {
        "label": "Lovers rough",
        "file": ROOT / "180427 Lovers rough.mp3",
        "params": {
            **_COMMON,
            "f0_method":  "rmvpe",
            "language":   "en",
            "pitch_shift": 0,
            "noisy_bypass":    "true",  # 0-17s: flatness_p95=0.963 (극심한 기계음) 원본 대체
            "harmony_bypass":  "true",
            "falsetto_bypass": "true",
            "female_bypass":   "false",
        },
        "notes": (
            "0-17s 구간: spectral flatness_p95=0.963 (1.0=순수 노이즈).\n"
            "noisy_bypass(flatness>0.10)가 해당 구간 자동 원본 대체."
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
        """서버 상태 확인"""
        return self.session.get(self._url("/api/health"), timeout=10).json()

    def list_models(self) -> list:
        """음성 모델 목록 반환"""
        r = self.session.get(self._url("/api/models"), timeout=10)
        r.raise_for_status()
        return r.json()

    def list_jobs(self) -> list:
        """전체 작업 목록 반환"""
        r = self.session.get(self._url("/api/jobs"), timeout=10)
        r.raise_for_status()
        return r.json()

    def job_status(self, job_id: str) -> dict:
        """특정 작업 상태 조회"""
        r = self.session.get(self._url(f"/api/jobs/{job_id}"), timeout=10)
        r.raise_for_status()
        return r.json()

    def convert(
        self,
        audio_path: Path,
        model_id: int,
        params: dict,
    ) -> str:
        """
        변환 작업 제출.
        audio_path: 원본 오디오 파일 경로
        model_id:   DB의 voice_models.id
        params:     _COMMON 또는 PRESETS[*]["params"] 딕셔너리
        반환값:     job_id (str)
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"파일 없음: {audio_path}")

        # pitch_shift는 별도 처리 (선택적)
        form = {k: str(v) for k, v in params.items() if k != "pitch_shift"}
        form["model_id"] = str(model_id)
        form["pitch_shift"] = str(params.get("pitch_shift", 0))

        with open(audio_path, "rb") as f:
            files = {"audio": (audio_path.name, f, "audio/mpeg")}
            r = self.session.post(
                self._url("/api/convert"),
                data=form,
                files=files,
                timeout=120,  # 대용량 파일 업로드 여유 시간
            )

        if r.status_code != 200:
            raise RuntimeError(f"변환 제출 실패 {r.status_code}: {r.text[:500]}")

        resp = r.json()
        return resp["job_id"]

    def wait_for_job(self, job_id: str, poll_sec: float = 10.0, timeout_min: float = 60.0) -> dict:
        """
        작업 완료까지 폴링 대기.
        반환값: 최종 작업 상태 딕셔너리
        """
        deadline = time.time() + timeout_min * 60
        while time.time() < deadline:
            status = self.job_status(job_id)
            state = status.get("status", "?")
            progress = status.get("progress", 0)
            message = status.get("message", "")
            print(f"  [{state:12s}] {progress:3d}%  {message}", end="\r", flush=True)

            if state in ("completed", "failed", "cancelled"):
                print()  # 줄바꿈
                return status

            time.sleep(poll_sec)

        print()
        raise TimeoutError(f"작업 {job_id} 타임아웃 ({timeout_min}분 초과)")

    def download_result(self, job_id: str, out_dir: Path = None) -> Path:
        """
        완료된 작업의 결과 파일 다운로드.
        반환값: 저장된 파일 경로
        """
        status = self.job_status(job_id)
        if status.get("status") != "completed":
            raise RuntimeError(f"작업이 완료되지 않았습니다: {status.get('status')}")

        filename = status.get("output_filename")
        if not filename:
            raise RuntimeError("출력 파일명을 찾을 수 없습니다.")

        out_dir = out_dir or Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

        r = self.session.get(self._url(f"/api/download/{filename}"), stream=True, timeout=300)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)

        return out_path


# ──────────────────────────────────────────────
# CLI 명령어 구현
# ──────────────────────────────────────────────

def cmd_list_models(client: VoiceStudioClient, _args):
    models = client.list_models()
    if not models:
        print("등록된 모델 없음")
        return
    print(f"{'ID':>4}  {'이름':<30}  {'에폭':>5}  {'SR':>6}  {'생성일'}")
    print("-" * 70)
    for m in models:
        print(f"{m['id']:>4}  {m['name']:<30}  {m.get('epochs', '?'):>5}  "
              f"{m.get('sample_rate', '?'):>6}  {m.get('created_at', '')[:10]}")


def cmd_convert(client: VoiceStudioClient, args):
    preset_key = args.preset

    if preset_key == "custom":
        # 커스텀 변환: 파일 경로 + 파라미터 수동 지정
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
        else:
            params["female_bypass"] = "true" if args.female_bypass else "false"
        label = audio_path.stem
    else:
        if preset_key not in PRESETS:
            print(f"알 수 없는 프리셋: {preset_key}")
            print(f"사용 가능: {', '.join(PRESETS.keys())}, custom")
            sys.exit(1)
        preset = PRESETS[preset_key]
        audio_path = preset["file"]
        params = dict(preset["params"])
        label = preset["label"]
        print(f"\n{'='*60}")
        print(f"  곡: {label}")
        print(f"  파일: {audio_path}")
        print(f"{'='*60}")
        if preset.get("notes"):
            print(f"\n[참고]\n{preset['notes']}\n")

    # 모델 ID 결정
    model_id = args.model_id
    if not model_id:
        models = client.list_models()
        if not models:
            print("오류: 등록된 모델이 없습니다. 먼저 학습을 완료하세요.")
            sys.exit(1)
        model_id = models[0]["id"]  # 첫 번째 모델 (보통 My Voice v50)
        print(f"  모델: [{model_id}] {models[0]['name']} (자동 선택)")
    else:
        print(f"  모델 ID: {model_id}")

    print(f"\n  F0 방식: {params.get('f0_method', 'rmvpe')}")
    print(f"  언어: {params.get('language', 'auto')}")
    print(f"  피치: {params.get('pitch_shift', 0):+d}반음")
    print(f"  바이패스: 화음={params['harmony_bypass']} / "
          f"팔세토={params['falsetto_bypass']} / "
          f"노이즈={params['noisy_bypass']} / "
          f"여성={params['female_bypass']}")
    print(f"  protect={params['protect']} / index_rate={params['index_rate']} / "
          f"rms={params['rms_mix_rate']} / filter={params['filter_radius']}")

    if not args.yes:
        ans = input("\n변환을 제출하시겠습니까? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("취소되었습니다.")
            return

    print("\n변환 제출 중...")
    job_id = client.convert(audio_path, model_id, params)
    print(f"  job_id: {job_id}")

    if args.no_wait:
        print("  --no-wait 옵션으로 대기 생략. 나중에 확인:")
        print(f"  python convert_client.py status {job_id}")
        return

    print("\n완료 대기 중 (최대 60분)...")
    final = client.wait_for_job(job_id, poll_sec=10, timeout_min=60)

    if final["status"] == "completed":
        print(f"\n변환 완료!")
        if args.download:
            out_dir = Path(args.download)
            path = client.download_result(job_id, out_dir)
            print(f"  저장: {path}")
        else:
            print(f"  결과 파일명: {final.get('output_filename', '?')}")
            print(f"  다운로드: python convert_client.py download {job_id}")
    else:
        print(f"\n변환 실패: {final.get('message', '?')}")


def cmd_status(client: VoiceStudioClient, args):
    status = client.job_status(args.job_id)
    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))


def cmd_jobs(client: VoiceStudioClient, _args):
    jobs = client.list_jobs()
    if not jobs:
        print("작업 없음")
        return
    print(f"{'ID':<14}  {'타입':<10}  {'상태':<12}  {'진행':>4}  메시지")
    print("-" * 70)
    for j in jobs[:20]:
        print(f"{j['id']:<14}  {j.get('job_type','?'):<10}  {j['status']:<12}  "
              f"{j.get('progress',0):>3}%  {j.get('message','')[:40]}")


def cmd_download(client: VoiceStudioClient, args):
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    print(f"다운로드 중 → {out_dir}")
    path = client.download_result(args.job_id, out_dir)
    print(f"저장 완료: {path}")


def cmd_show_presets(_client, _args):
    """사전 정의 프리셋 상세 출력"""
    for key, preset in PRESETS.items():
        p = preset["params"]
        print(f"\n{'='*60}")
        print(f"  프리셋 키: {key}")
        print(f"  곡명:      {preset['label']}")
        print(f"  파일:      {preset['file']}")
        print(f"  ─────────────────────────────────────────")
        print(f"  f0_method:         {p.get('f0_method')}")
        print(f"  language:          {p.get('language')}")
        print(f"  pitch_shift:       {p.get('pitch_shift', 0):+d}")
        print(f"  index_rate:        {p.get('index_rate')}")
        print(f"  rms_mix_rate:      {p.get('rms_mix_rate')}")
        print(f"  protect:           {p.get('protect')}")
        print(f"  filter_radius:     {p.get('filter_radius')}")
        print(f"  hop_length:        {p.get('hop_length')}")
        print(f"  f0_autotune:       {p.get('f0_autotune')} (strength={p.get('f0_autotune_strength')})")
        print(f"  harmony_bypass:    {p.get('harmony_bypass')}")
        print(f"  falsetto_bypass:   {p.get('falsetto_bypass')}")
        print(f"  noisy_bypass:      {p.get('noisy_bypass')}")
        print(f"  female_bypass:     {p.get('female_bypass')}")
        print(f"  vocal_blend:       {p.get('vocal_blend')} (0=더블링 없음)")
        print(f"  embedder_model:    {p.get('embedder_model')}")
        if preset.get("notes"):
            print(f"\n  [참고]\n  {preset['notes'].replace(chr(10), chr(10)+'  ')}")


# ──────────────────────────────────────────────
# CLI 파서
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Voice Studio 변환 Python 클라이언트 (v65)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"서버 주소 (기본값: {DEFAULT_SERVER})")

    sub = parser.add_subparsers(dest="command")

    # list-models
    sub.add_parser("list-models", help="사용 가능한 모델 목록 출력")

    # presets
    sub.add_parser("presets", help="사전 정의 프리셋 전체 상세 출력")

    # convert
    conv = sub.add_parser("convert", help="변환 제출")
    conv.add_argument("preset",
                      choices=list(PRESETS.keys()) + ["custom"],
                      help="사전 정의 프리셋 또는 'custom'")
    conv.add_argument("--model-id", type=int, default=None, dest="model_id",
                      help="사용할 모델 ID (미지정 시 첫 번째 모델 자동 선택)")
    conv.add_argument("--file", help="custom 프리셋 시 오디오 파일 경로")
    conv.add_argument("--f0-method", default=None, dest="f0_method",
                      help="F0 방식 (rmvpe / fcpe / hybrid[rmvpe+fcpe] 등)")
    conv.add_argument("--language", choices=["ko", "en", "auto"], default=None)
    conv.add_argument("--pitch", type=int, default=None,
                      help="피치 조정 (반음, -24~+24)")
    conv.add_argument("--female-bypass", action="store_true", dest="female_bypass",
                      help="여성 보컬 구간 바이패스 활성화")
    conv.add_argument("--no-female-bypass", action="store_true", dest="no_female_bypass",
                      help="여성 보컬 바이패스 비활성화 (custom 기본값 덮기)")
    conv.add_argument("--no-wait", action="store_true", dest="no_wait",
                      help="제출 후 완료 대기 없이 즉시 반환")
    conv.add_argument("--download", metavar="DIR",
                      help="완료 후 결과를 해당 디렉토리에 저장")
    conv.add_argument("-y", "--yes", action="store_true",
                      help="확인 없이 바로 제출")

    # status
    st = sub.add_parser("status", help="작업 상태 조회")
    st.add_argument("job_id")

    # jobs
    sub.add_parser("jobs", help="최근 작업 목록")

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

    client = VoiceStudioClient(args.server)

    # 서버 연결 확인 (convert/status/download 시)
    if args.command not in ("presets",):
        try:
            client.health()
        except Exception as e:
            print(f"서버에 연결할 수 없습니다 ({args.server}): {e}")
            print("server.py가 실행 중인지 확인하세요: python server.py")
            sys.exit(1)

    dispatch = {
        "list-models": cmd_list_models,
        "presets":     cmd_show_presets,
        "convert":     cmd_convert,
        "status":      cmd_status,
        "jobs":        cmd_jobs,
        "download":    cmd_download,
    }
    dispatch[args.command](client, args)


if __name__ == "__main__":
    main()
