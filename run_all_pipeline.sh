#!/bin/bash

echo "========================================================"
echo "   EIMAS: Full Analysis Pipeline"
echo "========================================================"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Step 1: Running full integrated pipeline..."
python main.py --full "$@"

echo "========================================================"
echo "✅ Full pipeline completed successfully"
echo "   Check 'outputs/' directory for results"
echo "========================================================"
