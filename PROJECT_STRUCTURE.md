# EIMAS 프로젝트 구조 및 파일 가이드

이 문서는 EIMAS(Economic Intelligence Multi-Agent System) 프로젝트의 디렉토리 구조와 주요 파일의 역할을 설명합니다.

## 📂 전체 디렉토리 트리

```text
eimas/
├── api/                # FastAPI 백엔드 서버 (프론트엔드 연동)
├── cli/                # CLI(명령줄 인터페이스) 도구
├── core/               # 시스템 핵심 설정 및 공통 스키마
├── data/               # 데이터베이스(SQLite) 및 수집된 데이터 저장소
├── docs/               # 문서 라이브러리 (매뉴얼, 아키텍처, 기능 명세)
├── frontend/           # Next.js 웹 프론트엔드 (V0 기반)
├── lib/                # 공용 라이브러리 및 유틸리티 (수집기, 분석기 등)
├── pipeline/           # 단계별 실행 파이프라인 (Modular Architecture)
├── outputs/            # 분석 결과물 (JSON, Markdown 리포트, HTML 대시보드)
├── scripts/            # 운영 및 관리용 유틸리티 스크립트
├── tests/              # 단위 테스트 및 통합 테스트 코드
└── main.py             # CLI 진입점 (실행기)
```

---

## 🔍 주요 디렉토리별 상세 설명

### 1. `pipeline/` (핵심 분석 흐름)
EIMAS의 가장 중요한 로직이 모듈별로 분리되어 있습니다. 각 폴더의 `runner.py`가 해당 단계의 실행을 담당합니다.
- **`collection/`**: 데이터 수집 (FRED, Yahoo Finance, Crypto 등).
- **`analysis/`**: 정량 분석 (레짐 탐지, 유동성 인과관계, 버블 리스크 등).
- **`signal/`**: 멀티 에이전트 토론 및 최종 매매 신호 생성.
- **`realtime/`**: 실시간 시장 미세구조(VPIN) 모니터링.
- **`storage/`**: 분석 결과를 DB 및 파일로 저장.
- **`reporting/`**: AI를 활용한 자연어 리포트 생성 및 품질 검증.
- **`runner.py`**: 위의 모든 단계를 조율하는 오케스트레이터.

### 2. `lib/` (비즈니스 로직 부품)
파이프라인이나 CLI에서 사용하는 실제 기능 구현체들입니다.
- `fred_collector.py`, `data_collector.py`: 데이터 수집 엔진.
- `regime_detector.py`, `bubble_detector.py`: 경제 분석 알고리즘.
- `trading_db.py`, `event_db.py`: 데이터베이스 인터페이스.
- `ai_report_generator.py`: LLM 기반 리포트 생성기.

### 3. `cli/` (사용자 인터페이스)
- `eimas.py`: `argparse` 기반의 명령줄 도구. 사용자는 이 파일을 통해 시스템을 조작합니다.

### 4. `api/` & `frontend/` (웹 인터페이스)
- **`api/main.py`**: FastAPI 서버. 실시간 데이터와 리서치 요청을 처리합니다.
- **`frontend/app/`**: Next.js (App Router) 기반 화면.
    - `/`: 메인 대시보드.
    - `/elicit`: AI 심층 리서치 및 팩트체크 페이지.

### 5. `docs/` (문서화)
- **`manuals/`**: 사용자를 위한 가이드 (`USER_GUIDE.md`).
- **`architecture/`**: 개발자를 위한 설계 문서.
- **`archive/`**: 레거시 코드 및 과거 문서 보관소.

---

## 🚀 핵심 실행 파일

| 파일명 | 역할 | 실행 명령어 |
|:---|:---|:---|
| `main.py` | 시스템 전체 진입점 (CLI) | `python main.py run` |
| `api/main.py` | 백엔드 서버 실행 | `python api/main.py` |
| `frontend/` | 웹 프론트엔드 실행 | `npm run dev` (폴더 진입 후) |
| `dashboard.py` | 간단한 Streamlit 대시보드 | `streamlit run dashboard.py` |

---

## 🛠️ 개발 지침
1. **모듈 추가**: 새로운 분석 로직은 `lib/`에 구현하고, `pipeline/analysis/`에 단계를 추가합니다.
2. **데이터 모델**: 공통으로 사용하는 데이터 구조는 `pipeline/schemas.py` 또는 `core/schemas.py`에 정의합니다.
3. **문서 업데이트**: 새로운 기능을 추가하면 반드시 `docs/` 내 관련 문서를 업데이트합니다.
