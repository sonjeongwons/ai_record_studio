#!/usr/bin/env python3
"""RunPod 관리 CLI — 에이전트가 직접 RunPod를 제어하기 위한 도구.

사용법:
  python runpod_admin.py <command> [args]

명령어 (Serverless Endpoint):
  status              — 엔드포인트 헬스 + 워커 상태
  requests            — 현재 요청 목록 (큐/진행 중)
  logs <job_id>       — 특정 작업 상태 + 출력 조회
  cancel <job_id>     — 작업 취소
  purge               — 큐의 모든 대기 작업 제거
  run <json_payload>  — 동기 실행 (테스트용)
  runsync <json>      — 비동기 실행

관리 (runpod Python SDK 필요):
  pods                — GPU Pod 목록
  pod-create <args>   — GPU Pod 생성
  pod-stop <pod_id>   — Pod 중지
  pod-start <pod_id>  — Pod 시작
  pod-kill <pod_id>   — Pod 삭제
  pod-exec <pod_id> <command> — Pod에서 명령 실행 (RunPod CLI)
  endpoint-info       — 엔드포인트 상세 정보 (SDK)
"""
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"
def _load():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_cfg = _load()
API_KEY = _cfg.get("runpod_api_key", "")
ENDPOINT_ID = _cfg.get("runpod_endpoint_id", "")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"


def _rest(method: str, path: str, body: dict = None) -> dict:
    url = f"{BASE}/{path}" if not path.startswith("http") else path
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"_error": e.code, "_body": e.read().decode()[:500]}


def _sdk_call(func_name: str, *args, **kwargs):
    """runpod SDK 호출 (설치되어 있을 때만)."""
    try:
        import runpod
        runpod.api_key = API_KEY
        func = getattr(runpod, func_name, None)
        if not func:
            # Try nested
            parts = func_name.split(".")
            obj = runpod
            for p in parts:
                obj = getattr(obj, p)
            func = obj
        return func(*args, **kwargs)
    except ImportError:
        print("runpod SDK가 설치되지 않았습니다. pip install runpod")
        return None
    except Exception as e:
        print(f"SDK 에러: {e}")
        return None


def _pp(data):
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


# ── Serverless REST API 명령어 ──

def cmd_status():
    health = _rest("GET", "health")
    print("=== 엔드포인트 헬스 ===")
    jobs = health.get("jobs", {})
    workers = health.get("workers", {})
    print(f"작업: 완료={jobs.get('completed',0)} | 실패={jobs.get('failed',0)} | "
          f"진행중={jobs.get('inProgress',0)} | 큐={jobs.get('inQueue',0)} | 재시도={jobs.get('retried',0)}")
    print(f"워커: 실행={workers.get('running',0)} | 대기={workers.get('idle',0)} | "
          f"초기화={workers.get('initializing',0)} | 준비={workers.get('ready',0)} | "
          f"쓰로틀={workers.get('throttled',0)} | 비정상={workers.get('unhealthy',0)}")


def cmd_requests():
    data = _rest("GET", "requests")
    reqs = data.get("requests", [])
    if not reqs:
        print("대기/진행 중인 요청 없음")
        return
    for r in reqs:
        delay = r.get("delayTime", 0)
        delay_str = f"{delay/1000:.1f}초" if delay < 60000 else f"{delay/60000:.1f}분"
        print(f"[{r['id']}] {r['status']} (대기: {delay_str})")


def cmd_logs(job_id: str):
    data = _rest("GET", f"status/{job_id}")
    if data.get("_error"):
        print(f"에러 {data['_error']}: {data['_body']}")
        return
    print(f"Status: {data.get('status')}")
    if data.get("delayTime"):
        print(f"Delay: {data['delayTime']/1000:.1f}초")
    if data.get("executionTime"):
        print(f"Execution: {data['executionTime']/1000:.1f}초")
    if data.get("error"):
        print(f"Error: {data['error']}")
    output = data.get("output")
    if output:
        if isinstance(output, dict):
            # 큰 base64 데이터는 잘라서 표시
            for k, v in output.items():
                if isinstance(v, str) and len(v) > 200:
                    print(f"  {k}: [{len(v)} chars] {v[:100]}...")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"Output: {str(output)[:1000]}")
    stream = data.get("stream", [])
    if stream:
        print("\n--- Stream ---")
        for s in stream[-20:]:
            if isinstance(s, dict):
                print(f"  {s.get('output', s)}")
            else:
                print(f"  {s}")


def cmd_cancel(job_id: str):
    data = _rest("POST", f"cancel/{job_id}")
    _pp(data)


def cmd_purge():
    data = _rest("POST", "purge-queue")
    _pp(data)


def cmd_run(payload_str: str):
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        print("유효한 JSON이 아닙니다")
        return
    data = _rest("POST", "run", {"input": payload})
    _pp(data)


def cmd_runsync(payload_str: str):
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        print("유효한 JSON이 아닙니다")
        return
    data = _rest("POST", "runsync", {"input": payload})
    _pp(data)


# ── Pod 관리 (SDK) ──

def cmd_pods():
    result = _sdk_call("get_pods")
    if result is None:
        return
    if not result:
        print("활성 Pod 없음")
        return
    for p in result:
        gpu = p.get("machine", {}).get("gpuDisplayName", "?")
        status = p.get("desiredStatus", "?")
        cost = p.get("costPerHr", 0)
        print(f"[{p['id']}] {p.get('name','?')} | {status} | {gpu} | ${cost:.3f}/hr")


def cmd_pod_create():
    result = _sdk_call("create_pod",
        name="voice-studio-dev",
        image_name="ghcr.io/sonjeongwons/ai_record_studio:latest",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        volume_in_gb=50,
        container_disk_in_gb=20,
        ports="22/tcp,8888/http",
        start_ssh=True,
    )
    if result:
        _pp(result)


def cmd_pod_stop(pod_id: str):
    result = _sdk_call("stop_pod", pod_id)
    print(f"Pod {pod_id} 중지 요청: {result}")


def cmd_pod_start(pod_id: str):
    result = _sdk_call("resume_pod", pod_id, gpu_count=1)
    print(f"Pod {pod_id} 시작 요청: {result}")


def cmd_pod_kill(pod_id: str):
    result = _sdk_call("terminate_pod", pod_id)
    print(f"Pod {pod_id} 삭제: {result}")


def cmd_pod_exec(pod_id: str, *command_parts):
    """Pod에서 명령 실행 — runpodctl이 필요합니다."""
    import subprocess
    cmd = " ".join(command_parts)
    print(f"Pod {pod_id}에서 실행: {cmd}")
    try:
        result = subprocess.run(
            ["runpodctl", "exec", pod_id, "--", *command_parts],
            capture_output=True, text=True, timeout=120
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
    except FileNotFoundError:
        print("runpodctl이 설치되지 않았습니다.")
        print("설치: pip install runpodctl 또는 https://github.com/runpod/runpodctl")
        print(f"\n대안 — SSH로 직접 접속:")
        print(f"  1. RunPod 대시보드에서 Pod의 SSH 정보 확인")
        print(f"  2. ssh root@<ip> -p <port> '{cmd}'")


def cmd_endpoint_info():
    result = _sdk_call("get_endpoint", ENDPOINT_ID)
    if result:
        _pp(result)


# ── CLI ──

COMMANDS = {
    "status": (cmd_status, []),
    "requests": (cmd_requests, []),
    "logs": (cmd_logs, ["job_id"]),
    "cancel": (cmd_cancel, ["job_id"]),
    "purge": (cmd_purge, []),
    "run": (cmd_run, ["json_payload"]),
    "runsync": (cmd_runsync, ["json_payload"]),
    "pods": (cmd_pods, []),
    "pod-create": (cmd_pod_create, []),
    "pod-stop": (cmd_pod_stop, ["pod_id"]),
    "pod-start": (cmd_pod_start, ["pod_id"]),
    "pod-kill": (cmd_pod_kill, ["pod_id"]),
    "pod-exec": (cmd_pod_exec, ["pod_id", "command..."]),
    "endpoint-info": (cmd_endpoint_info, []),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        print("명령어 목록:")
        for name, (_, args) in COMMANDS.items():
            args_str = " ".join(f"<{a}>" for a in args)
            print(f"  {name:20s} {args_str}")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"알 수 없는 명령어: {cmd}")
        print(f"사용 가능: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    func, expected_args = COMMANDS[cmd]
    args = sys.argv[2:]

    # pod-exec는 가변 인자
    if cmd == "pod-exec" and len(args) < 2:
        print("사용법: pod-exec <pod_id> <command...>")
        sys.exit(1)

    try:
        func(*args) if args else func()
    except TypeError as e:
        print(f"인자 오류: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"에러: {e}")
        sys.exit(1)
