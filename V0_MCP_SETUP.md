# v0 MCP Server 설정 가이드 (EIMAS 프로젝트 전용)

## 1. MCP 서버 활성화 완료 ✅

`.claude.json`의 **eimas 프로젝트**에만 v0 MCP 서버가 추가되었습니다:

```json
"/home/tj/projects/autoai/eimas": {
  "mcpServers": {
    "v0": {
      "command": "node",
      "args": [
        "/home/tj/v0-mcp/dist/main.js"
      ]
    }
  }
}
```

## 2. v0 API 키 설정

### 2-1. v0 API 키 발급
1. https://v0.dev 방문
2. 계정 로그인
3. Settings → API Keys에서 새 API 키 생성

### 2-2. .env 파일에 키 추가
```bash
# v0-mcp 디렉토리로 이동
cd /home/tj/v0-mcp

# .env 파일 생성 (없으면)
cp .env.example .env

# .env 파일 편집
nano .env

# V0_API_KEY 값 변경
V0_API_KEY=your_actual_v0_api_key
```

## 3. Claude Code 재시작

MCP 서버 설정이 적용되려면 Claude Code를 완전히 재시작해야 합니다:

```bash
# Claude Code 종료 후 재시작
# 또는 /clear 명령어 후 새 대화 시작
```

## 4. 현재 디렉토리 확인

v0 MCP는 **eimas 프로젝트 내에서만** 작동합니다:

```bash
# eimas 디렉토리로 이동
cd /home/tj/projects/autoai/eimas

# Claude Code 시작
claude
```

## 5. v0 MCP 사용 방법

### 5-1. UI 컴포넌트 생성
```
Claude에게: "v0를 사용해서 대시보드 컴포넌트를 만들어줘"
```

### 5-2. 사용 가능한 도구
- `create_ui`: UI 컴포넌트 생성
- `iterate_ui`: 기존 UI 수정
- `get_ui_status`: UI 생성 상태 확인

## 6. 백엔드 & 프론트엔드 서버 실행

### 백엔드 (FastAPI) - Port 8000
```bash
cd /home/tj/projects/autoai/eimas
source venv/bin/activate
python api/main.py
# 또는
uvicorn api.main:app --reload --port 8000
```

**API 엔드포인트:**
- http://localhost:8000 - API 홈
- http://localhost:8000/docs - Swagger UI
- http://localhost:8000/dashboard - 실시간 대시보드

### 프론트엔드 (Next.js) - Port 3000 또는 3002
```bash
cd /home/tj/projects/autoai/eimas/frontend
npm run dev
# 또는
npm run dev -- -p 3002
```

**프론트엔드 URL:**
- http://localhost:3000 (기본)
- http://localhost:3002 (대체 포트)

## 7. 트러블슈팅

### v0 MCP 서버가 실행되지 않을 때
```bash
# v0-mcp 빌드 확인
cd /home/tj/v0-mcp
ls dist/main.js

# 없으면 빌드
npm run build

# MCP 서버 수동 테스트
node dist/main.js
```

### v0 MCP가 보이지 않을 때
- **원인**: eimas 프로젝트 외부에서 Claude Code 실행
- **해결**: `cd /home/tj/projects/autoai/eimas` 후 Claude Code 재시작

### API 키 오류
```bash
# .env 파일 확인
cat /home/tj/v0-mcp/.env

# V0_API_KEY가 'your_v0_api_key_here'이면 실제 키로 교체 필요
```

## 8. 참고

- v0 MCP 저장소: https://github.com/hellolucky/v0-mcp
- v0 API 문서: https://v0.dev/docs
- MCP 프로토콜: https://modelcontextprotocol.io

---

**적용 범위:** `/home/tj/projects/autoai/eimas` 프로젝트만
**마지막 업데이트:** 2026-01-11
