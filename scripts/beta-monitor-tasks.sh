#!/bin/bash
# Team Beta - Monitor pending tasks from Team Alpha

SYNC_DIR=".claude/sync"

echo "🔍 Team Beta - Monitoring Tasks from Alpha"
echo "=========================================="
echo ""

# Check if sync directory exists
if [ ! -d "$SYNC_DIR" ]; then
    echo "❌ Sync directory not found: $SYNC_DIR"
    exit 1
fi

# Pending tasks
echo "📋 PENDING TASKS:"
PENDING_COUNT=$(ls -1 "$SYNC_DIR/tasks/pending/" 2>/dev/null | wc -l)
if [ "$PENDING_COUNT" -eq 0 ]; then
    echo "  ✅ No pending tasks - waiting for Team Alpha"
else
    ls -1 "$SYNC_DIR/tasks/pending/" 2>/dev/null | while read task; do
        if [ -f "$SYNC_DIR/tasks/pending/$task" ]; then
            echo "  📝 $task"
            jq -r '"\tType: \(.type) | Priority: \(.priority) | Files: \(.files | length)"' "$SYNC_DIR/tasks/pending/$task" 2>/dev/null
        fi
    done
fi

echo ""

# In progress tasks
echo "🔄 IN PROGRESS:"
IN_PROGRESS_COUNT=$(ls -1 "$SYNC_DIR/tasks/in_progress/" 2>/dev/null | wc -l)
if [ "$IN_PROGRESS_COUNT" -eq 0 ]; then
    echo "  ✅ No tasks in progress"
else
    ls -1 "$SYNC_DIR/tasks/in_progress/" 2>/dev/null | while read task; do
        if [ -f "$SYNC_DIR/tasks/in_progress/$task" ]; then
            echo "  ⚙️  $task"
            jq -r '"\tAssigned: \(.assigned_to) | Type: \(.type)"' "$SYNC_DIR/tasks/in_progress/$task" 2>/dev/null
        fi
    done
fi

echo ""

# Completed tasks (last 5)
echo "✅ COMPLETED (last 5):"
COMPLETED_COUNT=$(ls -1 "$SYNC_DIR/tasks/completed/" 2>/dev/null | wc -l)
if [ "$COMPLETED_COUNT" -eq 0 ]; then
    echo "  ⏳ No completed tasks yet"
else
    ls -1t "$SYNC_DIR/tasks/completed/" 2>/dev/null | head -5 | while read task; do
        echo "  ✓ $task"
    done
fi

echo ""

# Messages from Alpha
echo "💬 MESSAGES FROM ALPHA (last 3):"
MSG_COUNT=$(ls -1 "$SYNC_DIR/messages/alpha_to_beta/" 2>/dev/null | wc -l)
if [ "$MSG_COUNT" -eq 0 ]; then
    echo "  📭 No messages from Team Alpha"
else
    ls -1t "$SYNC_DIR/messages/alpha_to_beta/" 2>/dev/null | head -3 | while read msg; do
        echo "  📨 $msg"
    done
fi

echo ""
echo "=========================================="
echo "🤖 Team Beta Status: READY"
echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
