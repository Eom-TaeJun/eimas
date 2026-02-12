#!/bin/bash
# Watch for Team Alpha setup files

echo "👀 Watching for Team Alpha setup..."
echo "===================================="
echo ""

while true; do
    clear
    echo "🔍 Team Alpha Setup Monitor"
    echo "===================================="
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check for alpha status
    if [ -f ".claude/sync/status/alpha_status.json" ]; then
        echo "✅ Alpha Status: FOUND"
        echo "---"
        cat .claude/sync/status/alpha_status.json | jq '.'
        echo ""
    else
        echo "⏳ Alpha Status: WAITING..."
    fi
    
    echo ""
    
    # Check for alpha agents
    echo "📂 Alpha Agents:"
    if ls .claude/agents/alpha-* 1> /dev/null 2>&1; then
        ls -1 .claude/agents/alpha-* | while read f; do
            echo "  ✅ $(basename $f)"
        done
    else
        echo "  ⏳ No alpha agents yet..."
    fi
    
    echo ""
    
    # Check for messages
    echo "📬 Sync Directory Status:"
    echo "  Tasks Pending: $(ls -1 .claude/sync/tasks/pending/ 2>/dev/null | wc -l)"
    echo "  Messages from Alpha: $(ls -1 .claude/sync/messages/alpha_to_beta/ 2>/dev/null | wc -l)"
    
    echo ""
    echo "===================================="
    echo "Press Ctrl+C to stop watching"
    
    sleep 3
done
