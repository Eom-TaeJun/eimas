# EIMAS TODO List (2026-02-04)

## 🔥 우선순위 1: 백테스트 시스템 (이번 주)

- [ ] **lib/backtest_engine.py 분석** (30분)
  - 현재 구현 상태 확인
  - 필요한 기능 파악
  
- [ ] **lib/backtest/ 패키지 설계** (2시간)
  - enums.py: BacktestMode, MetricType
  - schemas.py: BacktestResult, PerformanceMetrics
  - engine.py: BacktestEngine
  - simulator.py: PortfolioSimulator
  - metrics.py: calculate_sharpe, max_drawdown, win_rate
  - report.py: BacktestReportGenerator
  
- [ ] **12개월 과거 데이터 준비** (1시간)
  - 2025-02-04 ~ 2026-02-04
  - FRED, 시장, 크립토 데이터
  - 일별 스냅샷 저장
  
- [ ] **백테스트 실행 및 검증** (2시간)
  - FULL mode 과거 데이터로 실행
  - 성과 지표 계산
  - 목표: Sharpe > 1.0, Win Rate > 55%
  
- [ ] **백테스트 보고서 생성** (1시간)
  - 레짐별 성과 비교
  - 월별 수익률
  - 최대 손실 구간 분석

**예상 소요 시간**: 6-7시간
**측정 가능한 결과**: Sharpe Ratio, Max Drawdown, Win Rate

---

## ⚡ 우선순위 2: 성능 최적화 (이번 주)

- [ ] **데이터 수집 병렬화** (2시간)
  - ThreadPoolExecutor 구현
  - 24 tickers 동시 수집
  - 목표: 75초 → 30초
  
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
