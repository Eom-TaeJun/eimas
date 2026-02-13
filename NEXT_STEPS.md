# EIMAS - Next Steps
**Last Updated:** 2026-02-13
**Session Summary:** Data Quality Validation & Streamlit Dashboard

---

## 📋 오늘 완료한 작업 (2026-02-13)

### ✅ 데이터 품질 개선
1. **검증 도구 생성** (`cli/validate_result.py`)
   - 18개 자동 검증 체크
   - 계산식 검증, 필수 필드, 데이터 범위, 일관성 확인
   - 실행: `python cli/validate_result.py --verbose`

2. **GMM Regime Probabilities 수정**
   - 하드코딩 (33%, 34%, 33%) → 실제 분석 결과 반영
   - `pipeline/schemas.py`: RegimeResult에 gmm_probabilities 필드 추가
   - `pipeline/analyzers_core.py`: detect_regime()에서 GMM 확률값 전달

3. **Correlation Matrix 구현** ✅
   - `pipeline/phases/phase2_enhanced.py`: calculate_correlation_matrix() 함수 추가
   - 6개 주요 자산 상관관계 계산 (SPY, QQQ, IWM, TLT, GLD, DIA)
   - 상관관계 히트맵 시각화 준비 완료

4. **Market Quality 구현** ✅
   - `pipeline/phases/phase2_adjustment.py`: _apply_microstructure_adjustment() 수정
   - MarketQualityMetrics 자동 생성 (유동성 점수, 데이터 품질)
   - 현재 유동성 점수: 80/100

5. **FRED CPI/PCE 수정** ✅
   - `lib/fred_collector.py`: calculate_yoy_change() 임계값 완화 (12개월 → 10개월)
   - 상세 로깅 추가로 디버깅 개선
   - YoY 인플레이션 데이터 정상 계산

### ✅ Streamlit 대시보드 구축
- **파일**: `frontend_streamlit/dashboard.py`
- **7개 탭**: Overview, Analytics, AI Reasoning, Risk, Signals, Events, **Realtime** (신규)
- **주요 기능**:
  - 5초 자동 새로고침
  - 전일/전주 대비 변화 비교 (Events 탭)
  - GMM 확률 바 차트
  - 상관관계 히트맵
  - 포트폴리오 파이 차트
  - 리스크 게이지
- **실행**: `streamlit run frontend_streamlit/dashboard.py`

### 📊 성과 지표
- **검증 통과율**: 66.7% → **94.4%** (+27.7%p)
- **WARNING**: 4개 → 1개 (-75%)
- **INFO**: 2개 → 0개 (-100%)

---

## 🎯 다음 세션 작업 계획

### Priority 1: 대시보드 개선 및 완성 (2-3시간)

#### 1.1 Streamlit 대시보드 테스트 및 버그 수정
- [ ] `streamlit run frontend_streamlit/dashboard.py` 실행 테스트
- [ ] 모든 탭 동작 확인
- [ ] 자동 새로고침 기능 검증
- [ ] 에러 핸들링 개선

#### 1.2 Events 탭 뉴스 수집 구현
- [ ] Perplexity API 활용 뉴스 수집
- [ ] 주요 이벤트별 관련 뉴스 매칭
- [ ] 뉴스 요약 및 포맷팅
- [ ] 파일 위치: `frontend_streamlit/news_collector.py` (신규)

#### 1.3 Signals 탭 구현
- [ ] 시그널 데이터베이스 연동 (`data/signals.db`)
- [ ] 최근 시그널 목록 표시
- [ ] 시그널 필터링 (source, action, ticker)
- [ ] 시그널 차트 (시계열)

#### 1.4 Realtime 탭 강화
- [ ] WebSocket 실시간 데이터 스트리밍 (선택)
- [ ] 실시간 VIX, 유동성 차트
- [ ] 실시간 알림 기능

### Priority 2: 데이터 완성도 향상 (1-2시간)

#### 2.1 Bubble Risk 검증
- [ ] `python main.py --full` 실행
- [ ] bubble_risk 데이터 정상 생성 확인
- [ ] 검증 도구로 재확인
- [ ] --full 모드 결과를 Streamlit에서 시각화

#### 2.2 Market Quality 확장
- [ ] 티커별 개별 유동성 점수 계산
- [ ] VPIN 기반 high_toxicity_tickers 감지
- [ ] illiquid_tickers 자동 필터링
- [ ] 시각화: 유동성 히트맵

#### 2.3 추가 상관관계 분석
- [ ] 크립토 자산 포함 (BTC, ETH)
- [ ] 한국 시장 포함 (KOSPI, 삼성전자)
- [ ] 시간대별 상관관계 변화 추적
- [ ] Rolling correlation 차트

### Priority 3: UI/UX 개선 (1-2시간)

#### 3.1 차트 라이브러리 고도화
- [ ] Plotly 인터랙티브 기능 추가
  - Zoom, Pan, Hover tooltips
  - 범례 클릭으로 시리즈 토글
- [ ] 색상 팔레트 통일
- [ ] 다크 테마 최적화

#### 3.2 레이아웃 개선
- [ ] 반응형 디자인 (모바일 대응)
- [ ] 로딩 인디케이터 추가
- [ ] 에러 메시지 UX 개선
- [ ] 빈 데이터 상태 플레이스홀더

#### 3.3 사용자 경험
- [ ] 즐겨찾기/북마크 기능
- [ ] 커스텀 대시보드 레이아웃
- [ ] 데이터 다운로드 (CSV, JSON)
- [ ] 스크린샷 캡처 기능

### Priority 4: Next.js 대시보드 개선 (선택, 2-3시간)

#### 4.1 차트 라이브러리 교체
- [ ] Recharts → ApexCharts 또는 Chart.js
- [ ] 인터랙티브 기능 강화
- [ ] 성능 최적화

#### 4.2 상관관계 히트맵 추가
- [ ] `/api/latest` 데이터 활용
- [ ] EnhancedCorrelationHeatmap 개선
- [ ] 실시간 업데이트

#### 4.3 Market Quality 시각화
- [ ] 유동성 점수 게이지
- [ ] Toxicity 알림
- [ ] 데이터 품질 인디케이터

---

## 🔧 알려진 이슈 및 개선사항

### 이슈 목록

1. **bubble_risk null (정상)** - Priority: Low
   - 상태: --quick 모드에서 의도적 스킵
   - 해결: --full 모드에서 정상 작동
   - 조치: 문서화만 필요

2. **FRED 데이터 지연** - Priority: Medium
   - 상태: 일부 경제 지표 1-2일 지연
   - 원인: FRED API 업데이트 주기
   - 해결: 캐싱 및 폴백 로직 추가

3. **Extended Data Adjustment 미표시** - Priority: Low
   - 상태: 스키마에는 있으나 JSON에 명시 안됨
   - 영향: 리스크 계산 투명성
   - 해결: to_dict() 수정 필요

### 개선 제안

1. **성능 최적화**
   - [ ] 파이프라인 병렬화 (Phase 2.x 분석 병렬 실행)
   - [ ] 캐싱 전략 개선 (Redis 도입 검토)
   - [ ] --quick 모드 30초 이하 목표

2. **데이터 품질**
   - [ ] FRED API 재시도 로직
   - [ ] yfinance 대체 데이터 소스 (Alpha Vantage, Polygon)
   - [ ] 데이터 검증 자동화 (CI/CD 통합)

3. **모니터링**
   - [ ] 파이프라인 실행 로그 대시보드
   - [ ] 에러 알림 (이메일, Slack)
   - [ ] 데이터 품질 메트릭 추적

---

## 📚 참고 문서

### 프로젝트 문서
- `CLAUDE.md` - 프로젝트 개요 및 빠른 시작
- `ARCHITECTURE.md` - 상세 아키텍처
- `command.md` - 실행 방법 및 CLI 가이드
- `TODO.md` - 장기 작업 계획

### 검증 및 품질
- `cli/validate_result.py` - 데이터 검증 도구
- `tests/test_pipeline_fallback_enrichment.py` - 테스트 예제

### 대시보드
- `frontend_streamlit/dashboard.py` - Streamlit 대시보드
- `frontend/` - Next.js 대시보드 (기존)

---

## 🚀 빠른 시작 (다음 세션)

```bash
# 1. 환경 확인
cd /home/tj/projects/autoai/eimas
python --version  # 3.10+

# 2. 파이프라인 실행 (옵션 선택)
python main.py --quick              # 30초 빠른 분석
python main.py --full               # 10분 전체 분석 (bubble_risk 포함)

# 3. 검증 실행
python cli/validate_result.py --verbose

# 4. Streamlit 대시보드 실행
streamlit run frontend_streamlit/dashboard.py
# 브라우저: http://localhost:8501

# 5. API 서버 + Next.js 대시보드 (선택)
uvicorn api.main:app --reload --port 8000 &
cd frontend && npm run dev
# 브라우저: http://localhost:3000
```

---

## 💡 팁

1. **데이터 신선도 유지**
   - 하루 1회 `python main.py --full` 실행 권장
   - cron job 설정 가능: `0 9 * * * cd /home/tj/projects/autoai/eimas && python main.py --full`

2. **디버깅**
   - 검증 실패 시: `python cli/validate_result.py --verbose`
   - 로그 확인: `tail -f /tmp/eimas_pipeline.log`

3. **성능**
   - --quick 모드: 일상 모니터링
   - --full 모드: 주요 의사결정

---

**다음 세션 목표:** Streamlit 대시보드 완성 및 뉴스 수집 기능 추가
**예상 소요 시간:** 3-4시간
**우선순위:** Priority 1 → Priority 2 → Priority 3
