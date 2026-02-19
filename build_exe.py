"""
AI Voice Studio - EXE 빌드 스크립트
실행: python build_exe.py
결과: dist/AI Voice Studio/ 폴더에 .exe 생성
"""

import PyInstaller.__main__
import shutil
from pathlib import Path

# 빌드 전 정리
for d in ['build', 'dist']:
    p = Path(d)
    if p.exists():
        shutil.rmtree(p)

PyInstaller.__main__.run([
    'app.py',

    # ── 기본 설정 ──
    '--name=AI Voice Studio',
    '--onedir',               # onefile보다 시작 속도 빠름
    '--windowed',             # 콘솔 창 숨김
    '--noconfirm',            # 기존 dist 덮어쓰기

    # ── 데이터 파일 번들 ──
    '--add-data=static;static',         # index.html 등 웹 UI
    '--add-data=server.py;.',           # FastAPI 서버 모듈

    # ── uvicorn 숨은 의존성 (PyInstaller가 자동 감지 못함) ──
    '--hidden-import=uvicorn',
    '--hidden-import=uvicorn.logging',
    '--hidden-import=uvicorn.loops',
    '--hidden-import=uvicorn.loops.auto',
    '--hidden-import=uvicorn.loops.asyncio',
    '--hidden-import=uvicorn.protocols',
    '--hidden-import=uvicorn.protocols.http',
    '--hidden-import=uvicorn.protocols.http.auto',
    '--hidden-import=uvicorn.protocols.http.h11_impl',
    '--hidden-import=uvicorn.protocols.http.httptools_impl',
    '--hidden-import=uvicorn.protocols.websockets',
    '--hidden-import=uvicorn.protocols.websockets.auto',
    '--hidden-import=uvicorn.protocols.websockets.wsproto_impl',
    '--hidden-import=uvicorn.protocols.websockets.websockets_impl',
    '--hidden-import=uvicorn.lifespan',
    '--hidden-import=uvicorn.lifespan.on',
    '--hidden-import=uvicorn.lifespan.off',

    # ── FastAPI / Starlette 숨은 의존성 ──
    '--hidden-import=fastapi',
    '--hidden-import=starlette',
    '--hidden-import=starlette.responses',
    '--hidden-import=starlette.routing',
    '--hidden-import=starlette.middleware',
    '--hidden-import=starlette.middleware.cors',
    '--hidden-import=starlette.staticfiles',
    '--hidden-import=multipart',
    '--hidden-import=multipart.multipart',

    # ── 기타 ──
    '--hidden-import=sqlite3',
    '--hidden-import=requests',

    # ── 최적화 ──
    '--strip',                # 디버그 심볼 제거 (용량 감소)

    # ── 아이콘 (나중에 추가 가능) ──
    # '--icon=static/icon.ico',
])

print()
print("=" * 50)
print("빌드 완료!")
print("=" * 50)
print()
print('결과물: dist/AI Voice Studio/')
print('실행:   dist/AI Voice Studio/AI Voice Studio.exe')
print()
print('고객 배포: "AI Voice Studio" 폴더를 통째로 zip 압축하여 전달')
print()
