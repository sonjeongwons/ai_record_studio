"""
AI Voice Studio - Desktop Application
PyWebView로 FastAPI 서버를 네이티브 윈도우에 내장.
고객은 이 .exe 하나만 실행하면 됨 (Python 설치 불필요).
"""

import sys
import os
import threading
import time
import socket
import logging

# PyInstaller 환경에서 multiprocessing freeze 지원
if getattr(sys, 'frozen', False):
    import multiprocessing
    multiprocessing.freeze_support()


def find_free_port(start=8000, end=8100):
    """사용 가능한 포트 탐색 (8000~8100)"""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return start  # fallback


def wait_for_server(port, timeout=15):
    """서버가 준비될 때까지 대기"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('127.0.0.1', port))
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def start_server(port):
    """백그라운드에서 FastAPI 서버 시작"""
    import uvicorn
    from server import app

    # .exe 모드에서는 로그 최소화
    log_level = "warning" if getattr(sys, 'frozen', False) else "info"

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level=log_level,
        # .exe에서 reload 사용 불가
        reload=False,
    )


def main():
    import webview

    port = find_free_port()

    # 서버를 데몬 스레드로 시작 (윈도우 닫으면 자동 종료)
    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()

    # 서버 준비 대기
    if not wait_for_server(port):
        # 서버 시작 실패 시 에러 표시
        webview.create_window(
            'AI Voice Studio - Error',
            html='<h2 style="color:red;font-family:sans-serif;padding:40px;">'
                 '서버 시작에 실패했습니다. 프로그램을 다시 실행해주세요.</h2>',
            width=500, height=200
        )
        webview.start()
        return

    # 메인 윈도우 생성
    window = webview.create_window(
        'AI Voice Studio',
        url=f'http://127.0.0.1:{port}',
        width=1440,
        height=900,
        min_size=(1024, 600),
        text_select=True,
    )

    # 윈도우 시작 (블로킹 — 닫히면 프로세스 종료)
    webview.start(
        gui='edgechromium',  # Windows: Edge WebView2 (기본 내장)
    )


if __name__ == '__main__':
    main()
