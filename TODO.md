# EIMAS TODO List (2026-02-04)

## 🔥 우선순위 1: 백테스트 시스템 (이번 주)

- [x] **lib/backtest_engine.py 분석** (30분) ✅ 2026-02-04 완료
  - 530줄 단일 파일 확인
  - 종합 백테스트 엔진 (Sharpe, Sortino, Calmar, Omega 등 15개 지표)

- [x] **lib/backtest/ 패키지 설계** (2시간) ✅ 2026-02-04 완료
  - enums.py: RebalanceFrequency, BacktestMode (39 lines)
  - schemas.py: BacktestConfig, BacktestMetrics, BacktestResult (191 lines)
  - engine.py: BacktestEngine 클래스 (272 lines)
  - metrics.py: 15개 성과 지표 계산 함수 (329 lines)
  - utils.py: compare_strategies, generate_report (223 lines)
  - __init__.py: Safe exports (100 lines)
  - 총 1,154 lines (530 lines → 모듈화)
  - Commit: 628a13f

- [x] **BACKTEST_GUIDE.md 작성** (30분) ✅ 2026-02-04 완료
  - 사용법, 예제 코드, 경제학적 방법론 설명

- [x] **scripts/prepare_historical_data.py 생성** (30분) ✅ 2026-02-04 완료
  - FRED + 24 market tickers + 5 crypto/RWA 수집
  - Parquet/CSV 저장 지원

- [ ] **12개월 과거 데이터 수집 실행** (1시간)
  - 2025-02-04 ~ 2026-02-04
  - data/backtest_historical.parquet 생성
  - 주의: FRED API 호출 제한 고려

- [ ] **백테스트 실행 및 검증** (2시간)
  - Equal Weight, Risk Parity, HRP 전략 테스트
  - 목표: Sharpe > 1.0, Win Rate > 55%, Max DD < 20%

- [ ] **백테스트 보고서 생성** (1시간)
  - 레짐별 성과 비교
  - 월별 수익률
  - 최대 손실 구간 분석

**예상 소요 시간**: 6-7시간
**측정 가능한 결과**: Sharpe Ratio, Max Drawdown, Win Rate

---

## ⚡ 우선순위 2: 성능 최적화 (이번 주)

- [x] **데이터 수집 병렬화** (2시간) ✅ 2026-02-04 완료
  - lib/parallel_data_collector.py 생성 (430+ lines)
  - ParallelMarketCollector (10 workers)
  - ParallelCryptoCollector (5 workers)
  - ParallelFREDCollector (5 workers, API rate limit 고려)
  - benchmark_collection() 유틸리티
  - Commit: ac594f1
  - 목표: 75초 → 30초 (main.py 통합 후 검증 필요)
  
- [ ] **분석 모듈 캐싱** (3시간)
  - Redis 또는 파일 기반 캐싱
  - TTL: 1시간
  - 캐시 키: (date, ticker, module_name)
  - 목표: 120초 → 60초
  
- [ ] **AI 호출 최적화** (2시간)
  - async/await 패턴
  - asyncio.gather() 병렬 호출
  - 목표: 30초 → 15초
  
- [ ] **성능 벤치마크** (1시간)
  - 최적화 전/후 비교
  - 병목 지점 재확인
  - 목표: FULL 249초 → 120초

**예상 소요 시간**: 8시간
**측정 가능한 결과**: 실행 시간 50% 감소

---

## 📊 우선순위 3: 대시보드 개선 (다음 주)

- [ ] **차트 추가 (Recharts)** (2시간)
  - 포트폴리오 가중치 파이 차트
  - 리스크 점수 타임라인
  - 상관관계 히트맵
  - GMM 확률 분포 차트
  
- [ ] **시그널 테이블 통합** (1시간)
  - `/latest` 엔드포인트 기반
  - `integrated_signals` 활용
  
- [ ] **WebSocket 연결** (1시간)
  - Phase 4 결과 반영
  - 실시간 업데이트

**예상 소요 시간**: 4시간

---

## 🔔 우선순위 4: 알림 시스템 (다음 주)

- [ ] **Slack 연동** (1.5시간)
  - Webhook 설정
  - 알림 포맷 정의
  
- [ ] **알림 규칙 구현** (1.5시간)
  - 버블 DANGER level
  - 레짐 변화 (Bull ↔ Bear)
  - 리스크 급등 (50+ → 70+)
  - AI 합의 불일치

**예상 소요 시간**: 3시간

---

## 📚 우선순위 5: 문서화 (다음 주)

- [ ] **API_REFERENCE.md** (2시간)
  - FastAPI 엔드포인트
  - 요청/응답 예시
  
- [ ] **PACKAGE_GUIDE.md** (2시간)
  - 패키지별 사용법
  - 예제 코드

**예상 소요 시간**: 4시간

---

## 🔧 선택 사항: 추가 리팩토링

- [ ] trading_db.py → lib/db/trading/ (2-3h)
- [ ] event_db.py → lib/db/events/ (1-2h)
- [ ] json_to_html_converter.py → lib/converters/ (1h)
- [ ] json_to_md_converter.py → lib/converters/ (1h)
- [ ] market_indicators.py → lib/collectors/indicators/ (2h)
- [ ] data_collector.py → lib/collectors/market/ (2h)

**총 예상 시간**: 9-11시간

---

## 🐛 버그 수정

- [ ] **자산 배분 제약 위반** (P2)
  - RebalancingPolicy 강화
  - Failsafe 메커니즘
  
- [ ] **NVDA 버블 경고 대응** (P1)
  - 포지션 크기 제한 (최대 5%)
  - 방어적 헤지 전략
  
- [ ] **yfinance 401 오류** (P3)
  - Retry 로직 (최대 3회)
  - 대체 데이터 소스

---

## 📈 측정 지표 (완료 시 체크)

### 성능
- [ ] FULL 실행 시간: < 120초 (현재 249초)
- [ ] --quick 실행 시간: < 15초 (현재 30초)
- [ ] 메모리 사용: < 600MB (현재 850MB)

### 백테스트
- [ ] Sharpe Ratio: > 1.0
- [ ] Win Rate: > 55%
- [ ] Max Drawdown: < 20%

### 대시보드
- [ ] 차트: 4개 추가
- [ ] WebSocket 지연: < 100ms
- [ ] 알림 지연: < 5초

---

## 🎯 이번 세션 시작점

```bash
# 1. 백테스트 엔진 분석
cat lib/backtest_engine.py | head -100

# 2. 패키지 구조 설계
mkdir -p lib/backtest
touch lib/backtest/{__init__.py,enums.py,schemas.py,engine.py,simulator.py,metrics.py,report.py}

# 3. 과거 데이터 준비
python scripts/prepare_historical_data.py --start 2025-02-04 --end 2026-02-04
```

---

*Created: 2026-02-04 19:30 KST*
*Priority: 백테스트 > 성능 > 대시보드 > 알림 > 문서*
