# 기능 요구사항 정의서

| 항목 | 내용 |
|------|------|
| 기능명 | 저축은행 건전성 지표 수집 및 파이프라인 통합 |
| 문서 번호 | EIMAS-PLAN-001 |
| 작성일 | 2026-02-23 |
| 버전 | v1.0 |
| 상태 | 구현 완료 |

---

## 1. 배경 및 현업 요구

**배경**
EIMAS는 글로벌 거시경제 지표(FRED, yfinance)와 암호화폐 데이터를 수집하지만,
국내 저축은행 건전성 데이터는 전혀 연동하지 않았다.
저축은행 여신 건전성 악화(NPL 급등, BIS 비율 하락)는 국내 신용 경색의 선행 지표이나
기존 파이프라인에서는 이를 포착할 수 없었다.

**현업 요구 (가상 시나리오: 리스크 관리 담당자)**
> "NPL 비율이 8%를 넘는 시점을 자동으로 감지해서,
>  거시 리스크 대시보드에 경고 신호를 띄워 주세요.
>  지금은 금감원 공시를 직접 찾아봐야 해서 대응이 늦어집니다."

---

## 2. AS-IS / TO-BE

### AS-IS
```
Phase 1 데이터 수집
├── FRED 지표 (금리, 실업률 등)
├── 주식 시장 데이터 (yfinance)
├── 암호화폐 데이터
└── 한국 자산 (KOSPI, KRW/USD)
     ↑ 저축은행 건전성 데이터 없음
```

분석 파이프라인이 저축은행 관련 리스크를 인지하지 못한 채
경기 국면 판단과 투자 권고를 산출함.

### TO-BE
```
Phase 1 데이터 수집
├── FRED 지표
├── 주식 시장 데이터
├── 암호화폐 데이터
├── 한국 자산
└── [NEW] Phase 1.5 — 저축은행 건전성 지표
         ├── NPL 비율 (고정이하여신비율)
         ├── BIS 자기자본비율
         └── ROA (총자산순이익률)
              ↓
         EIMASResult.korea_savings_bank 적재
              ↓
         DB 저장 (korea_savings_bank 테이블)
              ↓
         BusinessSummary 위험 요인 반영
```

---

## 3. 기능 명세

### 3.1 데이터 수집 (`lib/korea_savings_bank.py`)

| 항목 | 내용 |
|------|------|
| 수집 주기 | 파이프라인 실행 시 1회 (일 1회 cron 기준) |
| 데이터 소스 1 | FRED API (DDSI06KRA066NWDB, DSSB01KRA066NWDB) |
| 데이터 소스 2 | 금감원 FSS 공시 (Mock fallback: 2025Q3 실적치) |
| Fallback | API 실패 시 최신 공시 수치로 대체, `data_source="fss_mock"` 표기 |

**수집 지표**

| 지표 | FRED 코드 | 임계치 | 경고 기준 |
|------|-----------|--------|-----------|
| NPL 비율 (고정이하여신) | DDSI06KRA066NWDB | 8.0% | 초과 시 HIGH 경보 |
| BIS 자기자본비율 | DSSB01KRA066NWDB | 11.0% | 미달 시 경보 |
| ROA (총자산순이익률) | FSS 직접 수집 | 0.0% | 음수 시 경보 |

**출력 구조** (`KoreaSavingsBankIndicators`)
```python
@dataclass
class KoreaSavingsBankIndicators:
    timestamp: str
    npl_ratio: float        # 고정이하여신비율 (%)
    bis_capital_ratio: float # BIS 자기자본비율 (%)
    roa: float              # 총자산순이익률 (%)
    data_source: str        # "fred" | "fss_mock"
    signals: List[str]      # 경보 문자열 목록
    is_valid: bool
```

### 3.2 파이프라인 통합 (`pipeline/phases/phase1_collect.py`)

- Phase 1.5로 독립 단계 추가
- 환경변수 `EIMAS_SKIP_KOREA_SAVINGS_BANK=true` 설정 시 건너뜀
- 수집 결과를 `EIMASResult.korea_savings_bank` (Dict)에 저장

### 3.3 DB 저장 (`core/database.py`)

```sql
CREATE TABLE korea_savings_bank (
    date             TEXT PRIMARY KEY,
    npl_ratio        REAL,
    bis_capital_ratio REAL,
    roa              REAL,
    data_source      TEXT,
    signals_json     TEXT,
    note             TEXT,
    is_valid         INTEGER DEFAULT 1
);
```

### 3.4 비즈니스 요약 반영 (`lib/reports/business_summary.py`)

NPL 비율 ≥ 8.0% 시 `key_risks` 목록에 자동 추가:
```
"저축은행 NPL 비율 8.7% — 위험 임계치 초과"
```

---

## 4. 비기능 요건

| 구분 | 요건 |
|------|------|
| 성능 | FRED API 응답 포함 5초 이내 완료 |
| 안정성 | API 장애 시 Mock fallback으로 파이프라인 중단 없이 진행 |
| 확장성 | 저축은행별 개별 조회 지원 가능한 구조 |
| 운영 | `EIMAS_SKIP_KOREA_SAVINGS_BANK` 환경변수로 수집 비활성화 가능 |
| 추적성 | `data_source` 필드로 실제 데이터 vs Mock 구분 가능 |

---

## 5. 구현 일정

| 단계 | 내용 | 완료 |
|------|------|------|
| 분석 | 데이터 소스 조사 (FRED Korea 코드 확인) | ✅ 2026-02-22 |
| 설계 | 스키마 설계 + 파이프라인 통합 위치 결정 | ✅ 2026-02-22 |
| 개발 | `lib/korea_savings_bank.py` + Phase 1.5 구현 | ✅ 2026-02-22 |
| 통합 | `EIMASResult` 스키마 + DB 테이블 추가 | ✅ 2026-02-22 |
| 검증 | `python main.py --help` 통과 + Phase 1.5 실행 확인 | ✅ 2026-02-22 |

---

## 6. 관련 파일

| 파일 | 역할 |
|------|------|
| `lib/korea_savings_bank.py` | 수집 로직 + `KoreaSavingsBankIndicators` 정의 |
| `pipeline/phases/phase1_collect.py` | Phase 1.5 통합 |
| `pipeline/schemas.py` | `EIMASResult.korea_savings_bank` 필드 |
| `core/database.py` | `korea_savings_bank` 테이블 + 저장/조회 메서드 |
| `lib/reports/business_summary.py` | NPL 임계치 경보 반영 |
