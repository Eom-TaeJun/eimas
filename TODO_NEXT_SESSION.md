# EIMAS - Next Session TODO
**작성일**: 2026-02-14
**마지막 세션 작업**: 웹 대시보드 수정 및 이벤트 시스템 강화

---

## ✅ 완료된 작업 (이번 세션)

### 1. Risk Score Breakdown 수정
- Extended Data Adjustment 필드 추가
- 공식 업데이트: `Final = Base + Micro + Bubble + Extended`
- 파일: `frontend/components/charts/RiskBreakdownChart.tsx`

### 2. Depeg Risk & Crypto Stress Test 표시 개선
- 빈 데이터일 때 "Data Not Available" 메시지 표시
- 파일: `frontend/components/charts/CryptoRiskGauge.tsx`, `frontend/components/CryptoStressTest.tsx`

### 3. Correlation Matrix 히트맵 구현
- Recharts 기반 인터랙티브 히트맵
- 색상 그라디언트, 통계, CSV 내보내기
- 파일: `frontend/components/charts/RechartsCorrelationHeatmap.tsx`

### 4. Events 시스템 강화
- 뉴스 이벤트 생성기 구현 (`lib/news_event_generator.py`)
- 거시 경제, 암호화폐 뉴스, 가격 급등/락, 섹터 시그널 자동 감지
- 파이프라인 통합 (`pipeline/analyzers_core.py`)
- 프론트엔드 표시 컴포넌트 (`frontend/components/SimpleEventFeed.tsx`)

---

## 🔴 우선순위 1 (다음 세션 즉시 처리)

### 1. Event Database 오류 수정
**문제**: `NOT NULL constraint failed: detected_events.event_id`
**위치**: Event DB 저장 시
**영향**: 이벤트가 데이터베이스에 저장되지 않음 (JSON 출력은 정상)
**해결 방안**:
- `lib/event_db.py` 확인
- `event_id` 필드 자동 생성 또는 스키마 수정

### 2. Depeg Risk & Crypto Stress Test 실제 계산 구현
**문제**: 현재 빈 객체 `{}` 반환
**영향**: 관련 차트가 "Data Not Available" 표시
**해결 방안**:
- Stablecoin depeg probability 계산 로직 구현
- VaR (Value at Risk) 계산
- 파일: `lib/genius_act_macro.py` 또는 새 모듈 생성

---

## 🟡 우선순위 2 (주요 기능 개선)

### 1. AI Debate Topic 기반 이벤트 확장
**현재 상태**: `--quick` 모드에서는 AI 토론 주제 기반 이벤트 생성 안됨
**목표**: AI 합의 결과에서 주요 토픽 추출 → 뉴스 검색 → 이벤트 생성
**구현 위치**: `lib/news_event_generator.py`의 `_generate_debate_topic_events()`
**필요 작업**:
- Phase 3 AI 토론 결과를 Phase 2 이벤트 감지로 전달
- 토픽 추출 로직 개선
- `--full` 모드 실행 시 자동 생성 확인

### 2. 이벤트 카테고리별 필터링
**목표**: 프론트엔드에서 이벤트 유형별 필터 (Macro, Crypto, Sector, Ticker)
**구현 위치**: `frontend/components/SimpleEventFeed.tsx`
**기능**:
- 카테고리별 탭 또는 드롭다운
- 중요도별 필터 (LOW/MEDIUM/HIGH)
- 날짜 범위 필터

### 3. 대시보드 차트 추가 구현
**누락된 차트**:
- Portfolio Allocation Pie Chart (현재 테이블만 있음)
- Risk Heatmap 개선
- Time Series Charts (가격, 유동성 추이)

---

## 🟢 우선순위 3 (최적화 및 개선)

### 1. Perplexity API 속도 개선
**문제**: 뉴스 이벤트 생성 시 API 호출로 인한 지연
**해결 방안**:
- 비동기 처리 (asyncio)
- 캐싱 (최근 1시간 내 동일 쿼리는 캐시 사용)
- Rate limit 관리

### 2. 이벤트 중복 제거
**문제**: 여러 실행에서 동일 이벤트 중복 생성 가능
**해결 방안**:
- 이벤트 해시 또는 고유 ID 생성
- 데이터베이스에서 중복 체크
- 시간 윈도우 기반 중복 제거 (같은 날 동일 ticker + event_type)

### 3. 프론트엔드 성능 최적화
**개선 사항**:
- 큰 correlation matrix에 대한 가상화 (virtualization)
- 차트 렌더링 최적화
- SWR 캐시 전략 개선

---

## 🔵 우선순위 4 (장기 개선)

### 1. 테스트 작성 (Task #5)
**목표**: Codex로 테스트 코드 생성
**범위**:
- `lib/news_event_generator.py` 단위 테스트
- Events 파이프라인 통합 테스트
- 프론트엔드 컴포넌트 테스트

### 2. Korea Exchange API 통합
**목표**: KOSPI 데이터 신뢰도 개선
**현재**: yfinance 의존 (제한적)
**필요**: KRX API 또는 대체 데이터 소스

### 3. 알림 시스템
**목표**: 중요 이벤트 발생 시 알림
**방법**:
- 이메일 알림
- Webhook (Slack, Discord)
- WebSocket을 통한 실시간 브라우저 알림

---

## 📝 기술 부채

### 1. 코드 정리
- [ ] `tracked_events` vs `events_detected` 필드명 통일
- [ ] 사용하지 않는 EventFeed.tsx 제거 또는 통합
- [ ] 타입 정의 업데이트 (events_detected 구조 명시)

### 2. 문서화
- [ ] `lib/news_event_generator.py` docstring 보완
- [ ] API 엔드포인트 문서 업데이트
- [ ] 프론트엔드 컴포넌트 README 작성

### 3. 설정 관리
- [ ] Perplexity API 키 검증 추가
- [ ] 이벤트 생성 관련 설정 config.yaml로 이동
- [ ] 환경 변수 문서화

---

## 🚀 빠른 시작 (다음 세션)

### 1단계: Event Database 오류 수정
```bash
cd /home/tj/projects/autoai/eimas
grep -n "event_id" lib/event_db.py
# 스키마 확인 및 수정
```

### 2단계: Depeg Risk 구현 시작
```bash
# 기존 genius_act_macro.py 확인
grep -n "depeg" lib/genius_act_macro.py
# 또는 새 모듈 생성
```

### 3단계: 테스트 실행
```bash
# 이벤트 생성 테스트
python -c "from lib.news_event_generator import generate_news_events; print(len(generate_news_events()))"

# 전체 파이프라인
python main.py --quick
```

---

## 📊 현재 시스템 상태

### 작동 중 ✅
- 데이터 수집 (FRED, Market, Crypto)
- 기본 분석 (Regime, Risk Score)
- AI 토론 (Full + Reference mode)
- 이벤트 생성 (Macro, Crypto, Price Shocks, Sector)
- 웹 대시보드 (모든 주요 컴포넌트)
- Risk Score Breakdown (4개 조정값)
- Correlation Matrix 히트맵

### 부분 작동 ⚠️
- Event Database 저장 (JSON은 정상, DB는 오류)
- AI Debate Topic 이벤트 (--full 모드에서만)

### 미구현 ❌
- Depeg Risk 계산
- Crypto Stress Test 계산
- Portfolio Pie Chart
- 이벤트 필터링/검색

---

## 🔗 관련 파일 참조

### Backend
- `lib/news_event_generator.py` - 이벤트 생성기
- `pipeline/analyzers_core.py` - 이벤트 파이프라인 통합
- `lib/event_db.py` - 이벤트 데이터베이스
- `lib/event_framework.py` - 이벤트 프레임워크

### Frontend
- `frontend/components/SimpleEventFeed.tsx` - 이벤트 표시
- `frontend/components/charts/RiskBreakdownChart.tsx` - Risk Score
- `frontend/components/charts/RechartsCorrelationHeatmap.tsx` - Correlation
- `frontend/components/TabbedDashboard.tsx` - 메인 대시보드

### Configuration
- `outputs/eimas_*.json` - 분석 결과
- `.env` - API 키 설정
- `requirements.txt` - Python 의존성

---

**마지막 업데이트**: 2026-02-14 01:45 KST
**다음 세션 추천**: Event Database 오류 수정부터 시작
