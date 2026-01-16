# EIMAS: Economic Intelligence Multi-Agent System

EIMAS는 거시경제 데이터, 시장 데이터, 실시간 암호화폐 데이터를 통합 분석하여 최적의 투자 전략을 도출하는 자율 AI 에이전트 시스템입니다.

## 🚀 최신 업데이트 (v2.1)
- **CLI 통합**: `main.py` 대신 `cli/eimas.py`를 통한 통합 관리
- **파이프라인 모듈화**: 수집, 분석, 신호, 리포팅 등 책임 단위 분리 (`pipeline/`)
- **Elicit Deep Research**: 웹 인터페이스를 통한 대화형 심층 리서치 기능 추가
- **문서 정리**: `docs/` 디렉토리로 문서 통합

## 📂 프로젝트 구조
```
eimas/
├── api/                # FastAPI 백엔드 서버
├── cli/                # CLI 명령어 도구
├── docs/               # 문서 (manuals, architecture, features)
├── frontend/           # Next.js 웹 프론트엔드
├── pipeline/           # 핵심 분석 파이프라인
│   ├── collection/     # 데이터 수집
│   ├── analysis/       # 시장 분석 (레짐, 리스크, 펀더멘털)
│   ├── signal/         # 멀티 에이전트 토론 및 신호 생성
│   ├── realtime/       # 실시간 모니터링
│   ├── reporting/      # 리포트 생성
│   └── storage/        # DB 저장
├── lib/                # 공용 라이브러리 및 유틸리티
└── main.py             # CLI 진입점
```

## 🛠️ 설치 및 실행

### 1. 환경 설정
```bash
# Python 의존성 설치
pip install -r requirements.txt

# Frontend 의존성 설치
cd frontend
npm install
cd ..
```

### 2. 통합 파이프라인 실행 (CLI)
가장 간편하게 시스템 전체를 실행하는 방법입니다.
```bash
# 기본 실행
python cli/eimas.py run

# 빠른 실행 (Quick Mode)
python cli/eimas.py run --quick

# 도움말 확인
python cli/eimas.py run --help
```

### 3. 웹 대시보드 및 리서치 (Web)
두 개의 터미널에서 각각 백엔드와 프론트엔드를 실행합니다.

**Terminal 1 (Backend)**
```bash
python api/main.py
```

**Terminal 2 (Frontend)**
```bash
cd frontend
npm run dev
```
브라우저에서 `http://localhost:3000` (대시보드) 또는 `http://localhost:3000/elicit` (심층 리서치) 접속.

## 📖 문서
- [사용자 가이드 (User Guide)](docs/manuals/USER_GUIDE.md): 상세 명령어 및 사용법
- [아키텍처 문서](docs/architecture/ARCHITECTURE.md): 시스템 설계 및 구조

## ⚠️ 주의사항
- `.env` 파일에 필요한 API 키(OpenAI, Anthropic, FRED 등)가 설정되어 있어야 합니다.
- 실시간 모니터링 기능은 바이낸스 웹소켓 연결이 필요합니다.