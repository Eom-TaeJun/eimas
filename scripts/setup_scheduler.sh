#!/bin/bash
# EIMAS Scheduler Setup Script
# ============================
# cron 또는 systemd timer 설정

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

echo "======================================"
echo "EIMAS Scheduler Setup"
echo "======================================"
echo ""
echo "Project: $PROJECT_DIR"
echo "Logs: $LOG_DIR"
echo ""

# cron 항목 생성
CRON_ENTRY="0 17 * * 1-5 cd $PROJECT_DIR && /usr/bin/python3 scripts/daily_collector.py >> $LOG_DIR/daily.log 2>&1"

echo "Option 1: Add to cron (runs at 5 PM EST, Mon-Fri)"
echo ""
echo "Run this command to add to crontab:"
echo ""
echo "  (crontab -l 2>/dev/null; echo \"$CRON_ENTRY\") | crontab -"
echo ""
echo "Or manually edit crontab with: crontab -e"
echo ""
echo "Add this line:"
echo "  $CRON_ENTRY"
echo ""
echo "======================================"
echo ""

# 현재 cron 확인
echo "Current cron jobs:"
crontab -l 2>/dev/null || echo "  (none)"
echo ""

# 테스트 실행 옵션
echo "To test the collector:"
echo "  cd $PROJECT_DIR && python3 scripts/daily_collector.py"
echo ""
echo "To run a specific task only:"
echo "  python3 scripts/daily_collector.py --task prices"
echo "  python3 scripts/daily_collector.py --task indicators"
echo "  python3 scripts/daily_collector.py --task fred"
echo "  python3 scripts/daily_collector.py --task ark"
echo ""
