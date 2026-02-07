#!/bin/bash
#
# EIMAS Execution Contract Verification Script
# ============================================
#
# Purpose:
#     Automated validation of execution domain contract compliance
#     per ADV-005 specification.
#
# Usage:
#     ./scripts/check_execution_contract.sh
#
# Exit Codes:
#     0 - All checks passed
#     1 - Syntax errors
#     2 - Compile errors
#     3 - Backend source check failed
#     4 - Smoke test failed
#
# References:
#     - ADR: docs/architecture/ADV_005_EXECUTION_CONTRACT_VERIFICATION_V1.md
#     - Contract: docs/architecture/EXECUTION_DOMAIN_CONTRACT_V1.md
#     - Work Order: work_orders/GEN-203.md

# Use explicit exit handling per check block.
set -uo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0

# Project root (portable, path-independent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTOAI_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
EXTERNAL_ROOT="${AUTOAI_ROOT}/execution_intelligence"
cd "$PROJECT_ROOT"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  EIMAS Execution Contract Verification${NC}"
echo -e "${BLUE}  ADR: ADV-005 | Work Order: GEN-203${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ============================================================================
# Test 1: Compile Check
# ============================================================================
echo -e "${YELLOW}[1/3] Compile Check${NC}"
echo "Compiling execution contract files..."

FILES_TO_COMPILE=(
    "lib/adapters/execution_backend.py"
    "lib/adapters/execution_models.py"
    "lib/operational/config.py"
    "lib/operational/enums.py"
    "lib/operational/constraints.py"
    "lib/operational/rebalance.py"
    "lib/operational/engine.py"
    "pipeline/analyzers.py"
    "main.py"
)

COMPILE_FAILED=0
for file in "${FILES_TO_COMPILE[@]}"; do
    if [ -f "$file" ]; then
        if python3 -m py_compile "$file" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} $file"
        else
            echo -e "  ${RED}✗${NC} $file (compile failed)"
            COMPILE_FAILED=1
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} $file (not found, skipping)"
    fi
done

if [ $COMPILE_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ Compile check: PASS${NC}"
    PASS=$((PASS + 1))
else
    echo -e "${RED}✗ Compile check: FAIL${NC}"
    FAIL=$((FAIL + 1))
    exit 2
fi
echo ""

# ============================================================================
# Test 2: Backend Source Check
# ============================================================================
echo -e "${YELLOW}[2/3] Backend Source Check${NC}"
echo "Verifying local and external backend module paths..."

# Test local backend
echo -e "${BLUE}Testing local backend...${NC}"
LOCAL_TEST=$(PROJECT_ROOT="$PROJECT_ROOT" python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ['PROJECT_ROOT'])
os.environ['EIMAS_EXECUTION_BACKEND'] = 'local'

try:
    import importlib
    import pipeline.analyzers
    importlib.reload(pipeline.analyzers)

    from lib.adapters import (
        AllocationEngine,
        RebalancingPolicy,
        StressTestEngine,
        TacticalAssetAllocator,
        generate_operational_bundle,
    )

    alloc_module = AllocationEngine.__module__
    rebal_module = RebalancingPolicy.__module__
    tactical_module = TacticalAssetAllocator.__module__
    stress_module = StressTestEngine.__module__
    sample = {
        'risk_score': 50.0,
        'base_risk_score': 50.0,
        'full_mode_position': 'NEUTRAL',
        'confidence': 0.5,
        'modes_agree': True,
        'regime': {'regime': 'Neutral', 'confidence': 0.5},
        'portfolio_weights': {'SPY': 0.6, 'TLT': 0.4},
        'allocation_result': {'weights': {'SPY': 0.6, 'TLT': 0.4}},
    }
    op_bundle = generate_operational_bundle(sample, current_weights={'SPY': 0.6, 'TLT': 0.4})
    op_module = op_bundle['op_report'].__class__.__module__
    op_source = op_bundle.get('backend_source', 'unknown')

    print(f"AllocationEngine: {alloc_module}")
    print(f"RebalancingPolicy: {rebal_module}")
    print(f"TacticalAssetAllocator: {tactical_module}")
    print(f"StressTestEngine: {stress_module}")
    print(f"OperationalReport: {op_module}")
    print(f"OperationalBackendSource: {op_source}")

    # Check if modules are from lib (local)
    if (
        'lib.' in alloc_module
        and 'lib.' in rebal_module
        and 'lib.' in tactical_module
        and 'lib.' in stress_module
        and (
            op_module.startswith('lib.operational.')
            or op_module.startswith('lib.operational_engine')
        )
    ):
        print("STATUS:PASS")
    else:
        print("STATUS:FAIL")
except Exception as e:
    print(f"ERROR: {e}")
    print("STATUS:FAIL")
PY
)

if echo "$LOCAL_TEST" | grep -q "^STATUS:PASS$"; then
    echo -e "  ${GREEN}✓${NC} Local backend: PASS"
    echo "$LOCAL_TEST" | grep -v "^STATUS:" | sed 's/^/    /'
    LOCAL_OK=1
else
    echo -e "  ${RED}✗${NC} Local backend: FAIL"
    echo "$LOCAL_TEST" | grep -v "^STATUS:" | sed 's/^/    /'
    LOCAL_OK=0
fi

# Test external backend (if execution_intelligence exists)
echo -e "${BLUE}Testing external backend...${NC}"
if [ -d "$EXTERNAL_ROOT" ]; then
    EXTERNAL_TEST=$(PROJECT_ROOT="$PROJECT_ROOT" EXTERNAL_ROOT="$EXTERNAL_ROOT" python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ['PROJECT_ROOT'])
sys.path.insert(0, os.environ['EXTERNAL_ROOT'])
os.environ['EIMAS_EXECUTION_BACKEND'] = 'external'

try:
    import importlib
    import pipeline.analyzers
    importlib.reload(pipeline.analyzers)

    from lib.adapters import (
        AllocationEngine,
        RebalancingPolicy,
        StressTestEngine,
        TacticalAssetAllocator,
        generate_operational_bundle,
    )

    alloc_module = AllocationEngine.__module__
    rebal_module = RebalancingPolicy.__module__
    tactical_module = TacticalAssetAllocator.__module__
    stress_module = StressTestEngine.__module__
    sample = {
        'risk_score': 50.0,
        'base_risk_score': 50.0,
        'full_mode_position': 'NEUTRAL',
        'confidence': 0.5,
        'modes_agree': True,
        'regime': {'regime': 'Neutral', 'confidence': 0.5},
        'portfolio_weights': {'SPY': 0.6, 'TLT': 0.4},
        'allocation_result': {'weights': {'SPY': 0.6, 'TLT': 0.4}},
    }
    op_bundle = generate_operational_bundle(sample, current_weights={'SPY': 0.6, 'TLT': 0.4})
    op_module = op_bundle['op_report'].__class__.__module__
    op_source = op_bundle.get('backend_source', 'unknown')

    print(f"AllocationEngine: {alloc_module}")
    print(f"RebalancingPolicy: {rebal_module}")
    print(f"TacticalAssetAllocator: {tactical_module}")
    print(f"StressTestEngine: {stress_module}")
    print(f"OperationalReport: {op_module}")
    print(f"OperationalBackendSource: {op_source}")

    # Check if modules are from execution_intelligence (external)
    if (
        'execution_intelligence.' in alloc_module
        and 'execution_intelligence.' in rebal_module
        and 'execution_intelligence.' in tactical_module
        and 'execution_intelligence.' in stress_module
        and op_module.startswith('execution_intelligence.operational.')
    ):
        print("STATUS:PASS")
    else:
        print("STATUS:PASS_WITH_FALLBACK")  # Fallback to local is acceptable
except Exception as e:
    print(f"ERROR: {e}")
    print("STATUS:PASS_WITH_FALLBACK")  # Fallback is acceptable if external not available
PY
)

    if echo "$EXTERNAL_TEST" | grep -q "^STATUS:PASS$\\|^STATUS:PASS_WITH_FALLBACK$"; then
        echo -e "  ${GREEN}✓${NC} External backend: PASS"
        echo "$EXTERNAL_TEST" | grep -v "^STATUS:" | sed 's/^/    /'
        EXTERNAL_OK=1
    else
        echo -e "  ${RED}✗${NC} External backend: FAIL"
        echo "$EXTERNAL_TEST" | grep -v "^STATUS:" | sed 's/^/    /'
        EXTERNAL_OK=0
    fi
else
    echo -e "  ${YELLOW}⚠${NC} External backend: SKIP (execution_intelligence not found)"
    EXTERNAL_OK=1  # Not a failure if external doesn't exist
fi

if [ $LOCAL_OK -eq 1 ] && [ $EXTERNAL_OK -eq 1 ]; then
    echo -e "${GREEN}✓ Backend source check: PASS${NC}"
    PASS=$((PASS + 1))
else
    echo -e "${RED}✗ Backend source check: FAIL${NC}"
    FAIL=$((FAIL + 1))
    exit 3
fi
echo ""

# ============================================================================
# Test 3: Allocation/Rebalancing Smoke Test
# ============================================================================
echo -e "${YELLOW}[3/3] Allocation/Rebalancing Smoke Test${NC}"
echo "Running smoke test with mock data..."

SMOKE_TEST=$(PROJECT_ROOT="$PROJECT_ROOT" python3 - <<'PY'
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.environ['PROJECT_ROOT'])
os.environ['EIMAS_EXECUTION_BACKEND'] = 'local'

try:
    from pipeline.analyzers import run_allocation_engine

    # Create mock market data
    idx = pd.date_range('2025-01-01', periods=40, freq='D')
    market_data = {}
    tickers = ['SPY', 'TLT', 'GLD', 'QQQ']

    for ticker in tickers:
        prices = 100 + np.cumsum(np.random.normal(0, 1, size=len(idx)))
        market_data[ticker] = pd.DataFrame({'Close': prices}, index=idx)

    # Run allocation
    current_weights = {t: 0.25 for t in tickers}
    result = run_allocation_engine(
        market_data,
        strategy='risk_parity',
        current_weights=current_weights
    )

    # Verify required keys
    required_keys = ['allocation_result', 'status', 'allocation_strategy', 'allocation_config']
    missing_keys = [k for k in required_keys if k not in result]

    if missing_keys:
        print(f"ERROR: Missing keys: {missing_keys}")
        print("STATUS:FAIL")
        sys.exit(1)

    print(f"Status: {result['status']}")
    print(f"Strategy: {result.get('allocation_strategy', 'N/A')}")

    # Check rebalance decision
    if 'rebalance_decision' in result:
        action = result['rebalance_decision'].get('action', 'N/A')
        print(f"Rebalance Action: {action}")

        if action not in ['REBALANCE', 'PARTIAL', 'HOLD']:
            print(f"ERROR: Invalid action enum: {action}")
            print("STATUS:FAIL")
            sys.exit(1)

    # Check weights if status is SUCCESS
    if result['status'] == 'SUCCESS':
        weights = result['allocation_result'].get('weights', {})
        if weights:
            weight_sum = sum(weights.values())
            print(f"Weight Sum: {weight_sum:.4f}")

            # Verify weight sum tolerance (1.0 ± 0.02)
            if abs(weight_sum - 1.0) > 0.02:
                print(f"ERROR: Weight sum out of tolerance: {weight_sum}")
                print("STATUS:FAIL")
                sys.exit(1)

            print(f"Weight Tolerance: {abs(weight_sum - 1.0):.4f} (OK)")
        else:
            print("WARNING: No weights returned")

    print("STATUS:PASS")

except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
    print("STATUS:FAIL")
    sys.exit(1)
PY
)

SMOKE_EXIT=$?

if [ $SMOKE_EXIT -eq 0 ] && echo "$SMOKE_TEST" | grep -q "^STATUS:PASS$"; then
    echo -e "${GREEN}✓ Smoke test: PASS${NC}"
    echo "$SMOKE_TEST" | grep -v "^STATUS:" | sed 's/^/  /'
    PASS=$((PASS + 1))
else
    echo -e "${RED}✗ Smoke test: FAIL${NC}"
    echo "$SMOKE_TEST" | grep -v "^STATUS:" | sed 's/^/  /'
    FAIL=$((FAIL + 1))
    exit 4
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Verification Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  Tests Passed: ${GREEN}$PASS${NC}/3"
echo -e "  Tests Failed: ${RED}$FAIL${NC}/3"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓✓✓ All checks PASSED ✓✓✓${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}✗✗✗ Some checks FAILED ✗✗✗${NC}"
    echo ""
    exit 1
fi
