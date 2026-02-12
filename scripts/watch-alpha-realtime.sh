#!/bin/bash
# Real-time monitoring of Alpha team activities

echo "🔍 Real-time Alpha Team Monitor"
echo "================================"
echo "Press Ctrl+C to stop"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

while true; do
    clear
    echo -e "${BLUE}🔍 Real-time Alpha Team Monitor${NC}"
    echo "================================"
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Check Alpha status
    echo -e "${YELLOW}📊 Alpha Team Status:${NC}"
    if [ -f ".claude/sync/status/alpha_status.json" ]; then
        echo -e "${GREEN}✅ Found${NC}"
        jq -r '"  Status: \(.status)\n  Current Task: \(.current_task // "none")\n  Last Update: \(.last_update)"' .claude/sync/status/alpha_status.json 2>/dev/null
    else
        echo -e "${RED}⏳ No status file yet${NC}"
    fi
    echo ""

    # Check pending tasks
    echo -e "${YELLOW}📋 Pending Tasks:${NC}"
    PENDING_COUNT=$(ls -1 .claude/sync/tasks/pending/ 2>/dev/null | wc -l)
    if [ "$PENDING_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✅ ${PENDING_COUNT} task(s) waiting${NC}"
        ls -1t .claude/sync/tasks/pending/ 2>/dev/null | head -5 | while read task; do
            echo "  📝 $task"
            jq -r '"    Type: \(.type) | Priority: \(.priority)"' ".claude/sync/tasks/pending/$task" 2>/dev/null
        done
    else
        echo "  📭 No tasks yet"
    fi
    echo ""

    # Check messages
    echo -e "${YELLOW}💬 Messages from Alpha:${NC}"
    MSG_COUNT=$(ls -1 .claude/sync/messages/alpha_to_beta/ 2>/dev/null | wc -l)
    if [ "$MSG_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✅ ${MSG_COUNT} message(s)${NC}"
        ls -1t .claude/sync/messages/alpha_to_beta/ 2>/dev/null | head -3 | while read msg; do
            echo "  📨 $msg"
        done
    else
        echo "  📭 No messages"
    fi
    echo ""

    # Check recent file changes
    echo -e "${YELLOW}📁 Recent File Changes (last 5 min):${NC}"
    RECENT_FILES=$(find . -type f -mmin -5 \( -name "*.py" -o -name "*.yaml" -o -name "*.json" \) 2>/dev/null | grep -v ".git\|node_modules\|__pycache__\|outputs" | head -10)
    if [ -n "$RECENT_FILES" ]; then
        echo "$RECENT_FILES" | while read file; do
            echo -e "  ${GREEN}✏️  $file${NC}"
        done
    else
        echo "  📭 No recent changes"
    fi
    echo ""

    # Check git status
    echo -e "${YELLOW}🔀 Git Changes:${NC}"
    GIT_CHANGES=$(git status --short 2>/dev/null | head -10)
    if [ -n "$GIT_CHANGES" ]; then
        echo "$GIT_CHANGES" | while read line; do
            echo -e "  ${YELLOW}$line${NC}"
        done
    else
        echo "  ✅ Working tree clean"
    fi
    echo ""

    echo "================================"
    echo "Refreshing in 3 seconds..."

    sleep 3
done
