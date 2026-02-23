#!/usr/bin/env bash
# EIMAS 파이프라인 cron 실행 래퍼
# 실패 시 자동 재시도 (최대 2회), 결과를 logs/cron.log 에 기록
#
# cron 등록 예시 (매일 오전 7시 실행):
#   0 7 * * * /home/tj/projects/autoai/eimas/scripts/run_with_recovery.sh >> /home/tj/projects/autoai/eimas/logs/cron.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
CRON_LOG="$LOG_DIR/cron.log"
MAX_RETRIES=2

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$CRON_LOG"
}

run_pipeline() {
    cd "$PROJECT_DIR"
    python main.py --full
}

log "=== EIMAS 파이프라인 시작 ==="

attempt=0
success=false

while [ $attempt -le $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    log "실행 시도 $attempt/$((MAX_RETRIES + 1))"

    if run_pipeline; then
        log "파이프라인 성공"
        success=true
        break
    else
        exit_code=$?
        log "실패 (exit code: $exit_code)"
        if [ $attempt -le $MAX_RETRIES ]; then
            sleep_sec=$((2 ** attempt))
            log "${sleep_sec}초 후 재시도..."
            sleep $sleep_sec
        fi
    fi
done

if [ "$success" = false ]; then
    log "=== 최대 재시도 초과, 파이프라인 중단 ==="
    exit 1
fi

log "=== 완료 ==="
