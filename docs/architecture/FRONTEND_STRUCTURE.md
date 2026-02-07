# Frontend 구조 설명

## 병합 완료 상태

`frontend_steps/step1-5`의 모든 파일이 `frontend/` 디렉토리에 병합되었습니다.

### 병합 스크립트 상태

`scripts/merge_frontend.sh`는 과거 1회성 병합용 스크립트였고,
현재는 입력 원본(`frontend_steps/`)이 제거되어 더 이상 사용하지 않습니다.
2026-02-07 기준으로 해당 스크립트는 삭제되었습니다.

---

## 각 Step에서 추가된 파일

### Step 1: Dashboard (기본 레이아웃)
- app/layout.tsx
- app/page.tsx (메인 대시보드)
- components/MetricsGrid.tsx
- components/Navbar.tsx
- components/RealTimeClock.tsx
- components/SignalsTable.tsx
- lib/api.ts
- lib/types.ts
- package.json

### Step 2: Analysis (분석 도구)
- app/analysis/page.tsx
- app/api/analyze/route.ts
- app/api/analyze/[id]/route.ts
- components/analysis/analysis-form.tsx
- components/analysis/historical-analyses.tsx
- components/analysis/results-display.tsx
- lib/types/analysis.ts

### Step 3: Portfolio (포트폴리오 + UI 컴포넌트)
- app/portfolio/page.tsx
- app/api/portfolio/route.ts
- app/api/portfolio/trades/route.ts
- app/api/paper-trade/route.ts
- components/paper-trade-form.tsx
- components/portfolio-summary.tsx
- components/positions-table.tsx
- components/trade-history.tsx
- **components/ui/** (shadcn/ui 컴포넌트 6개)
  - badge.tsx
  - form.tsx
  - label.tsx
  - skeleton.tsx
  - toast.tsx
  - toaster.tsx

### Step 4: Risk (리스크 분석)
- app/risk/page.tsx
- app/api/risk/route.ts
- app/api/correlation/route.ts
- components/risk/correlation-alerts.tsx
- components/risk/correlation-matrix.tsx
- components/risk/portfolio-composition.tsx
- components/risk/risk-metrics.tsx

### Step 5: Settings (설정)
- app/settings/page.tsx
- app/layout.tsx (업데이트)
- components/ui/toaster.tsx (업데이트)

---

## 실행 위치

### 올바른 실행 디렉토리: `frontend/`

```bash
cd /home/tj/projects/autoai/eimas/frontend
npm run dev -- --port 3002
```

### frontend_steps는 참고용

`frontend_steps/step1-5` 디렉토리는 v0 API로 생성된 원본입니다.
**실제 실행은 병합된 `frontend/` 디렉토리에서 합니다.**

---

## 파일 개수 비교

```
frontend_steps/step1: 9개 파일
frontend_steps/step2: 9개 파일
frontend_steps/step3: 16개 파일
frontend_steps/step4: 8개 파일
frontend_steps/step5: 4개 파일
----------------------------
총: 46개 파일

frontend/: 46개 파일 (병합 완료)
```

---

## 현재 실행 중인 서버

```bash
# 확인
ps aux | grep "next dev"

# 결과
node /home/tj/projects/autoai/eimas/frontend/node_modules/.bin/next dev --port 3002
```

✅ 현재 `frontend/` 디렉토리에서 실행 중
