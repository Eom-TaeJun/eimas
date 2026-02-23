# EIMAS IT 운영 절차서 (Runbook)

| 항목 | 내용 |
|------|------|
| 문서 번호 | EIMAS-OPS-001 |
| 버전 | v1.0 |
| 작성일 | 2026-02-23 |
| 대상 환경 | Linux / WSL2, Python 3.10+ |
| 관련 시스템 | EIMAS 파이프라인, SQLite DB, PostgreSQL(RA), 외부 API |

---

## 1. 시스템 구성 개요

```
[cron / 수동 실행]
       ↓
  main.py --full / --short
       ↓
  Phase 1: 데이터 수집 (FRED, yfinance, Crypto)
  Phase 2: 정량 분석 (레짐, 리스크, 포트폴리오)
  Phase 3: AI 에이전트 토론
  Phase 4.5: 운용 의사결정
  Phase 5: DB 저장 (data/eimas.db)
  Phase 7: 리포트 생성 (outputs/)
       ↓
  logs/failures.jsonl  ← 실패 이력 기록
  logs/cron.log        ← cron 실행 이력
```

**주요 경로**

| 항목 | 경로 |
|------|------|
| 프로젝트 루트 | `/home/tj/projects/autoai/eimas/` |
| 메인 실행 | `main.py` |
| DB 파일 | `data/eimas.db` |
| PostgreSQL 데이터 | `.pgdata_ra/` |
| 실패 로그 | `logs/failures.jsonl` |
| cron 로그 | `logs/cron.log` |
| 출력 결과 | `outputs/eimas_*.json` |
| 모니터 모듈 | `pipeline/monitor.py` |
| 복구 스크립트 | `scripts/run_with_recovery.sh` |

---

## 2. 정기 운영 절차

### 2.1 일일 파이프라인 실행 (cron)

**스케줄:** 매일 오전 07:00 KST

```bash
# cron 등록 (최초 1회)
crontab -e
# 아래 라인 추가:
0 22 * * * /home/tj/projects/autoai/eimas/scripts/run_with_recovery.sh >> /home/tj/projects/autoai/eimas/logs/cron.log 2>&1
# (UTC 22:00 = KST 07:00)
```

**자동 재시도:** 실패 시 2배수 대기(2초 → 4초) 후 최대 2회 재시도.
3회 모두 실패 시 `exit 1` 반환, cron 실패로 기록됨.

### 2.2 수동 실행

```bash
cd /home/tj/projects/autoai/eimas

# 전체 분석 (Phase 1~9)
python main.py --full

# 경량 분석 (Phase 1, 4, 4.5, 5)
python main.py --short

# 실행 가능 여부 사전 확인
python main.py --help
python -m compileall main.py pipeline/app
```

### 2.3 실행 결과 확인

```bash
# 최신 출력 JSON 확인
ls -lt outputs/eimas_*.json | head -3

# DB 행 수 확인
sqlite3 data/eimas.db "SELECT name, (SELECT COUNT(*) FROM name) FROM sqlite_master WHERE type='table';"

# 실패 이력 확인
cat logs/failures.jsonl | python3 -m json.tool | tail -50
```

---

## 3. 장애 유형별 대응 절차

### 3.1 외부 API 실패 (FRED / yfinance / AI API)

**감지 방법**
- `logs/failures.jsonl` 에서 `"stage": "phase1_*"` 레코드 확인
- Phase 1 완료 후 `result.market_data` 가 비어 있거나 일부 누락

**영향도**
| API | 실패 시 영향 | Fallback |
|-----|-------------|---------|
| FRED | 거시 지표 누락 | 이전 캐시 사용 |
| yfinance | 시장 데이터 누락 | 분석 스킵 처리 |
| Anthropic/OpenAI | AI 토론 불가 | Phase 3 결과 없이 진행 |
| Perplexity | 리서치 에이전트 불가 | 해당 에이전트 스킵 |

**대응 절차**
```bash
# 1. API 키 환경변수 확인
env | grep -E "ANTHROPIC|OPENAI|FRED|PERPLEXITY|GOOGLE"

# 2. 네트워크 연결 확인
curl -s https://api.stlouisfed.org/fred/series?api_key=$FRED_API_KEY&series_id=FEDFUNDS | head -c 200

# 3. API 키 만료 여부 확인 → .env 파일 업데이트 후 재실행
source .env
python main.py --short
```

---

### 3.2 DB 연결 실패 / 잠금 (SQLite)

**감지 방법**
- `sqlite3.OperationalError: database is locked` 에러
- `logs/failures.jsonl` 에서 `"stage": "phase5_*"` 레코드

**원인**
- 이전 파이프라인 프로세스가 종료되지 않고 DB를 점유 중

**대응 절차**
```bash
# 1. DB를 점유 중인 프로세스 확인
lsof data/eimas.db 2>/dev/null || fuser data/eimas.db 2>/dev/null

# 2. 해당 프로세스 종료 (PID 확인 후)
kill -9 <PID>

# 3. DB 무결성 확인
sqlite3 data/eimas.db "PRAGMA integrity_check;"
# 정상: "ok" 출력

# 4. 재실행
python main.py --short
```

---

### 3.3 PostgreSQL 연결 실패 (RA 모듈)

**감지 방법**
- `psycopg2.OperationalError: could not connect to server`
- RA 분석 결과 누락

**원인**
- 재부팅 후 PostgreSQL 서버 미기동

**대응 절차**
```bash
PG_BIN="/home/tj/projects/autoai/eimas/.conda_pg/bin"
PGDATA="/home/tj/projects/autoai/eimas/.pgdata_ra"
PG_LOG="/home/tj/projects/autoai/eimas/logs/pg_ra.log"

# 1. 상태 확인
$PG_BIN/pg_ctl status -D $PGDATA

# 2. 기동 (중단 상태일 때)
$PG_BIN/pg_ctl start -D $PGDATA -l $PG_LOG

# 3. 기동 확인
$PG_BIN/psql -p 55432 -d ra_fi -c "SELECT version();"

# ※ 상세 절차: docs/manuals/RA_POSTGRES_REBOOT_RUNBOOK.md 참조
```

---

### 3.4 메모리 부족 / 프로세스 강제 종료

**감지 방법**
- `MemoryError` 또는 `Killed` 메시지
- `outputs/eimas_*.json` 파일이 생성되지 않음

**대응 절차**
```bash
# 1. 현재 메모리 확인
free -h

# 2. 경량 모드로 재실행 (데이터 수집 범위 축소)
python main.py --short

# 3. 특정 Phase 건너뛰기 (환경변수)
EIMAS_SKIP_KOREA_SAVINGS_BANK=true python main.py --short
```

---

### 3.5 파이프라인 모듈 Import 오류

**감지 방법**
- `ImportError` / `ModuleNotFoundError` 로 즉시 종료
- `python main.py --help` 가 실패

**대응 절차**
```bash
# 1. 즉시 진단
python main.py --help 2>&1

# 2. 컴파일 오류 검사
python -m compileall pipeline/ lib/ core/ -q

# 3. 의존성 재설치
pip install -r requirements.txt

# 4. 특정 모듈 진단
python -c "from pipeline.phases.phase1_collect import run_phase1; print('OK')"
```

---

## 4. 상태 점검 명령어 (Quick Reference)

```bash
# ── 파이프라인 상태 ──
python -c "
from pipeline.monitor import PipelineMonitor
m = PipelineMonitor()
import json; print(json.dumps(m.check_health(), indent=2))
"

# ── 최근 실패 3건 ──
tail -n 3 logs/failures.jsonl | python3 -c "import sys,json; [print(json.dumps(json.loads(l), indent=2, ensure_ascii=False)) for l in sys.stdin]"

# ── DB 테이블별 행 수 ──
sqlite3 data/eimas.db ".tables"
sqlite3 data/eimas.db "SELECT 'signals', COUNT(*) FROM signals UNION ALL SELECT 'actions', COUNT(*) FROM actions UNION ALL SELECT 'market_regime', COUNT(*) FROM market_regime;"

# ── 최신 분석 결과 요약 ──
ls -lt outputs/eimas_*.json | head -1 | awk '{print $NF}' | xargs python3 -c "
import sys, json
d = json.load(open(sys.argv[1]))
print(f'추천: {d.get(\"final_recommendation\")}')
print(f'리스크: {d.get(\"risk_level\")} ({d.get(\"risk_score\",0):.1f}/100)')
print(f'레짐: {d.get(\"regime\",{}).get(\"regime\")}')
" 2>/dev/null || echo "출력 파일 없음"
```

---

## 5. 에스컬레이션 기준

| 조건 | 조치 |
|------|------|
| 3회 연속 파이프라인 실패 | 로그 확인 후 원인 분석, API 키 점검 |
| DB integrity_check 실패 | 백업에서 복구 (`data/eimas.db.bak`) |
| PostgreSQL 기동 불가 | `RA_POSTGRES_REBOOT_RUNBOOK.md` 절차 수행 |
| `outputs/` 에 24시간 이상 신규 파일 없음 | cron 스케줄 재확인, 수동 실행 |

---

## 6. 백업 및 복구

```bash
# DB 백업 (일일 권장)
cp data/eimas.db data/eimas.db.bak.$(date +%Y%m%d)

# 최신 분석 결과 보관
mkdir -p outputs/archive/$(date +%Y%m)
cp outputs/eimas_*.json outputs/archive/$(date +%Y%m)/ 2>/dev/null || true

# 복구 (백업으로 롤백)
cp data/eimas.db.bak.<날짜> data/eimas.db
```
