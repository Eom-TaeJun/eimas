# EIMAS 실행 가이드

## 방법 1: 2개의 터미널 사용 (권장)

### Terminal 1 - Backend (FastAPI)
```bash
cd /home/tj/projects/autoai/eimas
uvicorn api.main:app --reload --port 8000
```

접속: http://localhost:8000
API 문서: http://localhost:8000/docs

### Terminal 2 - Frontend (Next.js)
```bash
cd /home/tj/projects/autoai/eimas/frontend
npm run dev -- --port 3002
```

접속: http://localhost:3002

---

## 방법 2: 1개의 터미널 사용 (백그라운드)

### 한 번에 실행
```bash
cd /home/tj/projects/autoai/eimas

# Backend 백그라운드 실행
uvicorn api.main:app --reload --port 8000 > logs/backend.log 2>&1 &
echo "Backend PID: $!"

# Frontend 실행
cd frontend
npm run dev -- --port 3002
```

### 종료
```bash
# Next.js 종료 (Ctrl+C)
# Backend 종료
pkill -f "uvicorn api.main:app"
```

---

## 방법 3: 자동화 스크립트 사용

### run_all.sh 사용
```bash
cd /home/tj/projects/autoai/eimas
./run_all.sh
```

### 종료
```bash
./stop_all.sh
```

---

## 프로세스 확인

### 실행 중인 프로세스 확인
```bash
# Backend 확인
ps aux | grep uvicorn

# Frontend 확인
ps aux | grep "next dev"

# 포트 확인
lsof -i :8000  # Backend
lsof -i :3002  # Frontend
```

### 강제 종료
```bash
# Backend 종료
pkill -9 -f "uvicorn api.main:app"

# Frontend 종료
pkill -9 -f "next dev"

# 또는 포트로 종료
kill -9 $(lsof -t -i:8000)
kill -9 $(lsof -t -i:3002)
```

---

## 초기 설정 (최초 1회만)

### Frontend 의존성 설치
```bash
cd /home/tj/projects/autoai/eimas/frontend
npm install
```

### Backend 환경 확인
```bash
cd /home/tj/projects/autoai/eimas
python -c "from api.main import app; print('Backend ready')"
```

---

## 주요 엔드포인트

### Backend API
- GET  /                      - API 홈
- GET  /docs                  - Swagger UI
- GET  /api/signals           - 최신 시그널
- GET  /api/portfolio         - 현재 포트폴리오
- GET  /api/risk              - 리스크 지표
- GET  /api/correlation       - 상관관계 행렬
- GET  /api/regime            - 시장 레짐
- POST /api/optimize          - 포트폴리오 최적화
- POST /api/paper-trade       - 페이퍼 트레이드

### Frontend Pages
- /                           - Dashboard
- /analysis                   - Analysis Tools
- /portfolio                  - Portfolio Management
- /risk                       - Risk Analysis
- /settings                   - Settings

---

## 트러블슈팅

### Backend 실행 안 됨
```bash
# Python 경로 확인
which python3

# 의존성 재설치
pip install -r requirements.txt
```

### Frontend 실행 안 됨
```bash
# Node.js 버전 확인
node --version  # v24.11.0 이상

# 캐시 삭제 후 재시작
rm -rf .next
rm -rf node_modules
npm install
npm run dev -- --port 3002
```

### Port already in use
```bash
# 8000번 포트 사용 중인 프로세스 종료
kill -9 $(lsof -t -i:8000)

# 3002번 포트 사용 중인 프로세스 종료
kill -9 $(lsof -t -i:3002)
```

---

## 빠른 시작 (Quick Start)

```bash
# Terminal 1
cd /home/tj/projects/autoai/eimas && uvicorn api.main:app --reload --port 8000

# Terminal 2
cd /home/tj/projects/autoai/eimas/frontend && npm run dev -- --port 3002

# 브라우저에서 접속
# http://localhost:3002
```
