# EIMAS 실행 가이드

## 1. 백엔드 실행 (FastAPI)

```bash
# eimas 디렉토리에서
cd /home/tj/projects/autoai/eimas

# Python 환경 활성화 (필요시)
# source venv/bin/activate

# 백엔드 서버 실행
uvicorn api.main:app --reload --port 8000

# 또는 간단하게
python -m uvicorn api.main:app --reload --port 8000
```

확인: http://localhost:8000/docs


## 2. 프론트엔드 실행 (Next.js)

```bash
# 프론트엔드 디렉토리로 이동
cd /home/tj/projects/autoai/eimas/frontend

# 의존성 설치 (최초 1회)
npm install

# 환경변수 설정
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# 개발 서버 실행
npm run dev
```

확인: http://localhost:3000


## 3. 동시 실행 (권장)

### Terminal 1 - 백엔드
```bash
cd /home/tj/projects/autoai/eimas
uvicorn api.main:app --reload --port 8000
```

### Terminal 2 - 프론트엔드
```bash
cd /home/tj/projects/autoai/eimas/frontend
npm run dev
```


## 4. 빠른 실행 스크립트

```bash
# eimas 디렉토리에서
./run_all.sh
```


## 트러블슈팅

### 백엔드 포트 충돌
```bash
# 8000 포트 사용 중인 프로세스 확인
lsof -i :8000

# 프로세스 종료
kill -9 <PID>
```

### 프론트엔드 포트 충돌
```bash
# 3000 포트 사용 중인 프로세스 확인
lsof -i :3000

# 다른 포트로 실행
npm run dev -- --port 3001
```

### 의존성 문제
```bash
# 백엔드
pip install -r requirements.txt

# 프론트엔드
cd frontend
rm -rf node_modules package-lock.json
npm install
```
