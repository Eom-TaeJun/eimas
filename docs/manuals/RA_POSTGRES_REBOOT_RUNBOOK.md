# RA PostgreSQL Reboot Runbook

## 목적

재부팅 이후 EIMAS + `financial_indicators`의 RA 기업분석 데이터(`fi_ra.company_fundamentals`)를
다시 정상 적재할 수 있도록 로컬 PostgreSQL 기동 절차를 표준화한다.

## 대상 경로

- EIMAS 루트: `/home/tj/projects/autoai/eimas`
- PostgreSQL binary: `/home/tj/projects/autoai/eimas/.conda_pg/bin`
- PostgreSQL data dir: `/home/tj/projects/autoai/eimas/.pgdata_ra`
- PostgreSQL log: `/home/tj/projects/autoai/eimas/logs/pg_ra.log`
- DB: `ra_fi`
- Schema/Table: `fi_ra.company_fundamentals`
- Port: `55432`

## 0) 1회성 설치 (이미 되어 있으면 생략)

```bash
CONDA_NO_PLUGINS=true conda create --solver classic -y -p /home/tj/projects/autoai/eimas/.conda_pg postgresql
```

## 1) 재부팅 후 SQL(PostgreSQL) 올리기

```bash
cd /home/tj/projects/autoai/eimas

PG_BIN="/home/tj/projects/autoai/eimas/.conda_pg/bin"
PGDATA="/home/tj/projects/autoai/eimas/.pgdata_ra"
LOGFILE="/home/tj/projects/autoai/eimas/logs/pg_ra.log"
PORT=55432

mkdir -p /home/tj/projects/autoai/eimas/logs

# 최초 1회만 initdb 수행
if [ ! -d "$PGDATA/base" ]; then
  "$PG_BIN/initdb" -D "$PGDATA" --username=postgres --auth=trust
fi

"$PG_BIN/pg_ctl" -D "$PGDATA" -l "$LOGFILE" -o "-p ${PORT}" start
"$PG_BIN/createdb" -h 127.0.0.1 -p ${PORT} -U postgres ra_fi || true
"$PG_BIN/pg_isready" -h 127.0.0.1 -p ${PORT} -U postgres
```

정상 상태면 아래와 유사한 메시지가 출력된다.

```text
127.0.0.1:55432 - accepting connections
```

## 2) 파이프라인 실행

`.env`에 다음 값이 있어야 한다.

```bash
FI_RA_POSTGRES_ENABLED=true
FI_PG_DSN=postgresql://postgres@127.0.0.1:55432/ra_fi
FI_PG_SCHEMA=fi_ra
FI_PG_TABLE=company_fundamentals
```

실행 예시:

```bash
cd /home/tj/projects/autoai/eimas
./run_full_auto.sh
```

또는 빠른 검증:

```bash
python main.py --short --cron-mode
```

## 3) 적재 확인 SQL

```bash
"/home/tj/projects/autoai/eimas/.conda_pg/bin/psql" \
  -h 127.0.0.1 -p 55432 -U postgres -d ra_fi \
  -c "SELECT COUNT(*) AS total_rows, COUNT(DISTINCT ticker) AS tickers, MIN(as_of_date) AS min_date, MAX(as_of_date) AS max_date FROM fi_ra.company_fundamentals;" \
  -c "SELECT as_of_date, ticker, sector, trailing_pe, revenue FROM fi_ra.company_fundamentals ORDER BY ticker LIMIT 10;"
```

## 4) 종료/재시작

종료:

```bash
"/home/tj/projects/autoai/eimas/.conda_pg/bin/pg_ctl" \
  -D "/home/tj/projects/autoai/eimas/.pgdata_ra" stop
```

재시작:

```bash
"/home/tj/projects/autoai/eimas/.conda_pg/bin/pg_ctl" \
  -D "/home/tj/projects/autoai/eimas/.pgdata_ra" -l "/home/tj/projects/autoai/eimas/logs/pg_ra.log" -o "-p 55432" restart
```

## 5) 장애 대응

- `pg_isready`가 실패하면:
  - 로그 확인: `tail -n 200 /home/tj/projects/autoai/eimas/logs/pg_ra.log`
  - 포트 충돌 확인: `ss -ltnp | rg 55432`
- `stored_rows=0`이면:
  - 실행 JSON의 `company_ra_analysis.postgresql` 필드에서 `dsn_configured`, `driver_available`, `error` 확인
  - Yahoo/FRED DNS 상태 확인 후 재실행
