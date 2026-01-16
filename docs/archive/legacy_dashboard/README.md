# EIMAS Real-Time Dashboard

실시간 경제 인텔리전스 멀티에이전트 시스템 대시보드

## 기능

- **실시간 업데이트**: 5초마다 자동으로 최신 분석 결과 폴링
- **시장 레짐 표시**: Bull/Bear/Neutral 시장 상태
- **리스크 점수**: 0-100 스케일 시각화
- **AI 합의 표시**: Full Mode vs Reference Mode 비교
- **최종 권고**: BULLISH/BEARISH/NEUTRAL
- **신뢰도**: 퍼센트로 표시

## 설치

```bash
cd dashboard
npm install
```

## 실행

### 1. FastAPI 서버 시작 (백엔드)

터미널 1에서:
```bash
cd /home/tj/projects/autoai/eimas
uvicorn api.main:app --reload --port 8000
```

### 2. Next.js 개발 서버 시작 (프론트엔드)

터미널 2에서:
```bash
cd dashboard
npm run dev
```

브라우저에서 http://localhost:3000 접속

## 프로덕션 빌드

```bash
npm run build
npm start
```

## 데이터 소스

대시보드는 `outputs/integrated_YYYYMMDD_HHMMSS.json` 파일에서 최신 결과를 자동으로 읽어옵니다.

FastAPI 엔드포인트: `GET /api/latest`

## 아키텍처

```
┌─────────────────┐
│   Next.js UI    │ (localhost:3000)
│   (5초 폴링)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │ (localhost:8000)
│  GET /latest    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  outputs/*.json │ (최신 파일 자동 선택)
│  integrated_*   │
└─────────────────┘
```

## 화면 구성

1. **헤더**: 타이틀 + Live 상태 표시 + 마지막 업데이트 시간
2. **상태 배너**: 최종 권고 (BULLISH/BEARISH/NEUTRAL) + 신뢰도 + 리스크 레벨
3. **메트릭 그리드**:
   - Market Regime (Bull/Bear/Neutral)
   - Risk Score (원형 진행바)
   - AI Consensus (Full vs Reference 모드 비교)
4. **데이터 수집 상태**: Market tickers + Crypto assets
5. **경고**: 시스템 경고 표시 (있을 경우)

## 기술 스택

- **프론트엔드**: Next.js 15, React 19, TypeScript
- **스타일링**: Tailwind CSS (dark theme)
- **아이콘**: Lucide React
- **차트**: Recharts (원형 진행바)
- **백엔드**: FastAPI (Python)

## 환경 변수

필요 없음. 모든 설정은 `next.config.js`에서 처리됩니다.

## 문제 해결

### "No integrated results found" 오류

```bash
# EIMAS 분석을 먼저 실행하세요
cd /home/tj/projects/autoai/eimas
python main.py --quick
```

### 포트 충돌

```bash
# FastAPI 포트 변경
uvicorn api.main:app --reload --port 8001

# next.config.js에서 API 포트도 변경
```

### 자동 새로고침이 작동하지 않음

- 브라우저 콘솔에서 에러 확인
- FastAPI 서버가 실행 중인지 확인
- `/api/latest` 엔드포인트가 200 응답하는지 확인

## 개발 팁

### 폴링 간격 변경

`Dashboard.tsx`에서:
```typescript
const interval = setInterval(() => {
  fetchLatestData()
}, 5000) // 5초 → 원하는 밀리초로 변경
```

### 테마 변경

`Dashboard.tsx`에서 Tailwind 클래스 수정:
- `bg-slate-900` → 원하는 배경색
- `text-green-400` → 원하는 텍스트색

## 버전

- EIMAS: v2.1.1 (Real-World Agent Edition)
- Dashboard: v1.0.0
