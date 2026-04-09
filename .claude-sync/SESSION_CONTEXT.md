# Session Context — 2026-04-09 (v49 고음/치찰음/발음 전면 개선)

## 마지막 작업: v49 음질 최적화

### v49 변경사항 (12개)
1. **f0_autotune=True** (strength=0.6) — 노래 피치 안정화
2. **hop_length 64→128** — 커뮤니티 표준, 노이즈 추적→삑사리 방지
3. **filter_radius 2→3** (한국어 4) — 피치 스무딩→크래킹 감소
4. **protect 0.33→0.40** — 과도한 자음 보호 완화→인덱스 정확도↑
5. **index_rate 0.40→0.45** (한국어 0.55, 영어 0.35)
6. **3kHz presence boost 제거** — 치찰음 증폭 주범
7. **300Hz/600Hz EQ** — 한국어: 제거(비음 보호), 영어: -0.3만
8. **language 파라미터 신규** (ko/en/auto) — 한/영 EQ 분리
9. **8kHz 디에서 좁은 대역** (w=0.3) — 공기감 보존
10. **split_audio 300→180초** — 청크 경계 아티팩트 감소
11. **프리셋 12개 전면 재설계** — 한/영 분리, autotune 반영
12. **UI: 언어 드롭다운 추가** — 한국어/영어/자동

### 근거
- Applio 공식문서: f0_autotune "singing conversion에 권장"
- AI Hub 커뮤니티: hop_length 128 기본, 64는 노이즈 추적
- AI Hub: "robotic sibilances = dataset 짧거나 overfitted"
- 커뮤니티: index_rate 고품질 모델 0.6-0.75, 영어는 낮게

### 다음 할 일
1. v49 파라미터로 3곡 재변환 테스트
   - Breaking Through: 기본값 프리셋 (language=auto)
   - 기다릴게: 한국어 프리셋 (language=ko, index 0.55, filter 4)
   - comethru: 영어 프리셋 (language=en, index 0.35)
2. 변환 결과 비교 청취 (v48 vs v49)
3. 필요시 에폭 체크포인트 비교 (v41 모델 과적합 확인)

### 핵심 파라미터 (v49)
- index_rate: 0.45 (ko:0.55, en:0.35), rms: 0.0
- protect: 0.40, filter_radius: 3 (ko:4)
- hop_length: 128, f0_autotune: True (0.6)
- language: auto/ko/en
- vocal_blend: 0%, post_reverb: 0.0
- 테스트: 48/48 통과
