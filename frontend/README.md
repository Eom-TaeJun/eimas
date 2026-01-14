# EIMAS Real-Time Dashboard

> Economic Intelligence Multi-Agent System - 실시간 대시보드

## 특징

- **5초 자동 폴링**: 최신 EIMAS 분석 결과를 5초마다 자동 업데이트
- **실시간 시각화**: 시장 레짐, 리스크 점수, AI 합의 결과 실시간 표시
- **다크 테마**: GitHub 스타일 다크 UI
- **v2.1.1 지원**: Market Quality & Bubble Risk 메트릭 포함
- **FRED 제외**: 정적 데이터 제외, 동적 시장 데이터만 표시

## 빠른 시작

### 1. 의존성 설치

```bash
cd /home/tj/projects/autoai/eimas/frontend
npm install
```

### 2. FastAPI 서버 실행

터미널 1:
```bash
cd /home/tj/projects/autoai/eimas
uvicorn api.main:app --reload --port 8000
```

### 3. EIMAS 분석 실행 (최소 1회 필요)

터미널 2:
```bash
cd /home/tj/projects/autoai/eimas
python main.py --quick
```

이렇게 하면 `outputs/integrated_YYYYMMDD_HHMMSS.json` 파일이 생성됩니다.

### 4. 프론트엔드 실행

터미널 3:
```bash
cd /home/tj/projects/autoai/eimas/frontend
npm run dev
```

브라우저에서 http://localhost:3000 접속

## 한 번에 실행하기

```bash
# 백그라운드에서 FastAPI 실행
cd /home/tj/projects/autoai/eimas
uvicorn api.main:app --reload --port 8000 &

# 프론트엔드 실행
cd frontend
npm run dev
```

## 화면 구성

### Main Status Banner
- **Final Recommendation**: BULLISH/BEARISH/NEUTRAL
- **Confidence**: 0-100% 진행바
- **Risk Level**: LOW/MEDIUM/HIGH + 리스크 점수

### Metrics Grid (4 cards)
1. **Market Regime**
   - Bull/Bear/Neutral
   - Trend, Volatility
   - Confidence

2. **AI Consensus**
   - Full Mode vs Reference Mode
   - 합의 여부 표시

3. **Data Collection**
   - Market Tickers 개수
   - Crypto Assets 개수

4. **Market Quality** (v2.1.1)
   - 평균 유동성 점수
   - 데이터 품질 상태

### Warnings
- 시스템 경고 표시 (있을 경우)

## 데이터 흐름

```
┌─────────────────┐
│  main.py --quick│
│  (EIMAS 분석)    │
└────────┬────────┘
         │ 생성
         ▼
┌─────────────────────────┐
│ outputs/integrated_*.json│
└────────┬────────────────┘
         │ 읽기
         ▼
┌─────────────────┐
│ FastAPI /latest │  ← 5초마다 폴링
│ (localhost:8000)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Next.js UI    │
│ (localhost:3000)│
└─────────────────┘
```

## API 엔드포인트

### GET /latest
최신 integrated JSON 파일 반환

**응답 예시:**
```json
{
  "timestamp": "2026-01-10T22:16:19",
  "final_recommendation": "BULLISH",
  "confidence": 0.65,
  "risk_level": "MEDIUM",
  "risk_score": 11.46,
  "regime": {
    "regime": "Bull (Low Vol)",
    "trend": "Weak Uptrend",
    "volatility": "Low",
    "confidence": 0.75
  },
  "full_mode_position": "BULLISH",
  "reference_mode_position": "BULLISH",
  "modes_agree": true,
  "market_data_count": 24,
  "crypto_data_count": 2,
  "_meta": {
    "source_file": "integrated_20260110_221638.json",
    "file_modified": "2026-01-10T22:16:38"
  }
}
```

## 폴링 간격 변경

`components/MetricsGrid.tsx`:
```typescript
const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
  refreshInterval: 5000, // 5초 → 원하는 밀리초로 변경
})
```

## 프로덕션 빌드

```bash
npm run build
npm start
```

## 문제 해결

### "Failed to load EIMAS analysis data"
- FastAPI 서버가 실행 중인지 확인: `curl http://localhost:8000/latest`
- outputs 디렉토리에 integrated JSON 파일이 있는지 확인

### "No integrated results found" (FastAPI 오류)
```bash
# EIMAS 분석을 먼저 실행하세요
python main.py --quick
```

### CORS 오류
- `next.config.js`에서 API 프록시 설정 확인
- FastAPI의 CORS 미들웨어 설정 확인

### 자동 새로고침이 안 됨
- 브라우저 개발자 도구에서 네트워크 탭 확인
- 5초마다 `/latest` 요청이 가는지 확인
- SWR 캐시 문제일 경우 페이지 새로고침

## 기술 스택

- **프론트엔드**
  - Next.js 16
  - React 19
  - TypeScript 5
  - Tailwind CSS 4
  - SWR (실시간 데이터 폴링)
  - Radix UI (컴포넌트)
  - Lucide React (아이콘)

- **백엔드**
  - FastAPI
  - Python 3.10+

## 커스터마이징

### 색상 변경
`components/MetricsGrid.tsx`에서:
```typescript
const getRecommendationColor = (rec: string) => {
  switch (rec) {
    case "BULLISH":
      return "bg-green-500/10 text-green-500 border-green-500/20" // 여기 수정
    // ...
  }
}
```

### 카드 추가
`components/MetricsGrid.tsx`의 Grid 섹션에 새 Card 추가:
```tsx
<Card className="bg-[#161b22] border-[#30363d]">
  <CardHeader>
    <CardTitle>새 메트릭</CardTitle>
  </CardHeader>
  <CardContent>
    {/* 내용 */}
  </CardContent>
</Card>
```

## 개발 팁

### Mock 데이터 테스트
`lib/api.ts`에서:
```typescript
export async function fetchLatestAnalysis() {
  // 실제 API 대신 mock 데이터 반환
  return {
    timestamp: new Date().toISOString(),
    final_recommendation: "BULLISH",
    // ...
  }
}
```

### 디버그 모드
브라우저 콘솔에서:
```javascript
// SWR 캐시 확인
localStorage.debug = 'swr:*'
```

## 라이선스

MIT

## 버전

- EIMAS: v2.1.1 (Real-World Agent Edition)
- Dashboard: v1.0.0
- Last Updated: 2026-01-11
