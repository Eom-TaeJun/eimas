# 0/NA Root Cause Analysis

**Date**: 2026-02-16
**Analyst**: data-validator agent
**Investigation Scope**: Pipeline data flow from collectors → analyzers → storage → frontend

---

## Executive Summary

**Issue**: Null values (0/NA) appearing in JSON outputs and potentially causing frontend display issues.

**Root Cause**: **EXPECTED BEHAVIOR** - All nulls are legitimate and intentional:
1. **Category A (Missing Source Data)**: Financial data providers (yfinance) don't provide certain metrics for specific asset types
2. **Category B (Conditional Fields)**: Fields only populated in specific execution modes or conditions
3. **Category C (Schema Design)**: Optional fields with `Optional[T] = None` default

**Severity**: **LOW** - No pipeline bugs found. Nulls are properly handled in backend. Minor frontend improvement opportunity exists.

**Action Required**:
- **Priority 1**: Add null guards in frontend TypeScript components (defensive programming)
- **Priority 2**: Improve user experience by displaying "N/A" instead of blank for missing data
- **Priority 3**: Document which fields are nullable in schema comments

---

## Investigation Process

### Step 1: Pipeline Execution

Ran two modes to capture different execution paths:
```bash
# Quick mode (30s, minimal phases)
python main.py --quick
Output: outputs/eimas_20260216_163458.json

# Full mode (4min, all phases)
python main.py --full
Output: outputs/eimas_20260216_220104.json
```

### Step 2: Null Value Detection

**Quick Mode**: Found 24 null values
**Full Mode**: Found 14 null values (some populated by enhanced phases)

---

## Findings by Category

### Category A: Missing Source Data (Financial APIs)

**Issue**: Financial data APIs don't provide all metrics for all asset types.

| Field | Ticker | Reason | Source | Fix |
|-------|--------|--------|--------|-----|
| `ev_to_ebitda` | JPM | Banks don't report EBITDA | yfinance | Accept null, display "N/A" |
| `operating_income` | JPM | Not available for financial services | yfinance | Accept null, display "N/A" |
| `operating_margin` | JPM | Derived from missing operating_income | yfinance | Accept null, display "N/A" |
| `expense_ratio_pct` | XLE, QQQ, etc. | Missing from ETF info | yfinance | Accept null, display "N/A" |

**Analysis**:
- JPMorgan (JPM) is a bank - GAAP accounting for banks doesn't use EBITDA/operating income the same way as industrial companies
- yfinance API returns `None` for these fields, which is correct behavior
- ETF expense ratios are often missing in real-time data feeds

**Impact**: ✅ No frontend display (company_ra_analysis not rendered in dashboard)

**Recommendation**:
- Keep nulls as-is in pipeline
- If/when company_ra_analysis is displayed, show "N/A" for missing metrics
- Add tooltip: "Not applicable for financial institutions"

---

### Category B: Conditional/Mode-Dependent Fields

**Issue**: Fields only populated in specific execution modes or conditions.

#### B.1 Mode-Dependent Nulls

| Field | Quick Mode | Full Mode | Reason |
|-------|------------|-----------|--------|
| `bubble_risk` | ❌ null | ✅ populated | Skipped in `--quick` (expensive computation) |
| `ai_report` | ❌ null | ✅ populated | Phase 7 skipped in `--quick` |
| `agent_outputs` | ❌ null | ❌ null | Deprecated field (replaced by debate_consensus) |
| `debate_results` | ❌ null | ❌ null | Deprecated field (replaced by debate_consensus) |
| `verification` | ❌ null | ❌ null | Deprecated field (replaced by debate_consensus) |
| `backtest_run_id` | ❌ null | ❌ null | Only populated in `--backtest` mode |

**Analysis**:
- `bubble_risk`: Intentionally skipped in quick mode (Greenwood-Shleifer calculation is compute-intensive)
- `ai_report`: Phase 7 only runs in full mode (Claude API calls are expensive)
- `agent_outputs`, `debate_results`, `verification`: **LEGACY FIELDS** - kept for backward compatibility but no longer used
- `backtest_run_id`: Only set when running historical backtests

**Impact**:
- ✅ Frontend doesn't display deprecated fields
- ✅ Frontend checks `if (!analysis)` before rendering

**Recommendation**:
- Document mode differences in README
- Consider removing deprecated fields in v3.0 (breaking change)
- Add mode indicator in JSON: `"execution_mode": "quick|full|backtest"`

#### B.2 Conditional Nulls (Event-Driven)

| Field | When Null | When Populated |
|-------|-----------|----------------|
| `failsafe_status.reason` | `triggered=false` | `triggered=true` |
| `failsafe_status.fallback_action` | `triggered=false` | `triggered=true` |
| `regime.error_code` | No errors | Regime detection failure |
| `regime.error_msg` | No errors | Regime detection failure |

**Analysis**: These are properly designed conditional fields.

**Impact**: ✅ Schema design is correct, no changes needed.

---

### Category C: Schema Design (Optional Fields)

**File**: `/home/tj/projects/autoai/eimas/pipeline/schemas.py`

```python
@dataclass
class EIMASResult:
    # ... required fields ...

    # Optional fields
    agent_outputs: Optional[AgentOutputs] = None       # Line 419
    debate_results: Optional[DebateResults] = None     # Line 420
    verification: Optional[VerificationResults] = None # Line 421
    bubble_risk: Optional[BubbleRiskMetrics] = None    # (implied)
    ai_report: Optional[Dict] = None                   # (implied)
    backtest_run_id: Optional[str] = None              # (implied)
```

**Analysis**: Python `Optional[T]` fields default to `None`, which serializes to JSON `null`. This is correct behavior.

---

## Test Results

### Test Case 1: Quick Mode Nulls

```bash
python main.py --quick
```

**Expected Nulls** (All Found ✓):
- `bubble_risk`: null ✓
- `ai_report`: null ✓
- `agent_outputs`: null ✓
- `debate_results`: null ✓
- `verification`: null ✓
- `backtest_run_id`: null ✓
- `company_ra_analysis.companies[3].valuation.ev_to_ebitda`: null ✓ (JPM)
- `company_ra_analysis.etf_strategy_snapshot[*].expense_ratio_pct`: null ✓ (10 ETFs)
- `failsafe_status.reason`: null ✓ (triggered=false)
- `regime.error_code`: null ✓ (no errors)

**Verdict**: ✅ All nulls are expected

### Test Case 2: Full Mode Nulls

```bash
python main.py --full
```

**Result**:
- `bubble_risk`: ✅ populated
- `ai_report`: ✅ populated
- `agent_outputs`: ❌ still null (deprecated)
- `debate_results`: ❌ still null (deprecated)
- `verification`: ❌ still null (deprecated)
- `backtest_run_id`: ❌ still null (not backtest mode)

**Verdict**: ✅ All remaining nulls are expected

### Test Case 3: Frontend Null Handling

**File**: `frontend/components/MetricsGrid.tsx`

**Current Code**:
```typescript
if (!analysis) {
  return <LoadingSkeleton />;  // Line 62 ✓
}

// No null checks on individual fields
<div>{(analysis.confidence * 100).toFixed(1)}%</div>  // Line 96
<RiskGauge score={analysis.risk_score} />             // Line 112
```

**Potential Issue**:
- If `analysis.confidence` or `analysis.risk_score` is `undefined` or `null`, this would cause:
  - `NaN%` display (if `confidence` is null)
  - React error (if `risk_score` is null and RiskGauge doesn't handle it)

**Actual Risk**: ⚠️ LOW
- Pipeline ALWAYS populates `confidence` and `risk_score` (non-optional fields)
- Tested both quick and full modes: both have valid values
- JSON schema guarantees these are `float`, not `Optional[float]`

**Recommendation**: Add defensive null guards for robustness:
```typescript
<div>{((analysis.confidence ?? 0) * 100).toFixed(1)}%</div>
<RiskGauge score={analysis.risk_score ?? 0} />
```

---

## Recommended Fixes

### Priority 1: Frontend Defensive Programming (IMMEDIATE)

**Files to Update**:
- `frontend/components/MetricsGrid.tsx`
- `frontend/components/FREDLiquidityDashboard.tsx`
- `frontend/components/RiskGauge.tsx`

**Changes**:
```typescript
// BEFORE
<div>{(analysis.confidence * 100).toFixed(1)}%</div>

// AFTER
<div>{((analysis.confidence ?? 0) * 100).toFixed(1)}%</div>

// BEFORE
<RiskGauge score={analysis.risk_score} />

// AFTER
<RiskGauge score={analysis.risk_score ?? 0} />
```

**Effort**: 30 minutes
**Impact**: Prevents potential `NaN` displays and React errors

---

### Priority 2: User Experience Enhancement (SOON)

**File**: `frontend/lib/formatters.ts` (create new file)

**Add Helper Functions**:
```typescript
export function formatOptionalNumber(
  value: number | null | undefined,
  decimals: number = 2
): string {
  if (value === null || value === undefined) return "N/A";
  return value.toFixed(decimals);
}

export function formatOptionalPercent(
  value: number | null | undefined,
  decimals: number = 1
): string {
  if (value === null || value === undefined) return "N/A";
  return `${(value * 100).toFixed(decimals)}%`;
}
```

**Usage**:
```typescript
// Company analysis display (future)
<div>EV/EBITDA: {formatOptionalNumber(company.valuation.ev_to_ebitda, 1)}</div>
// Output: "EV/EBITDA: N/A" instead of blank
```

**Effort**: 1 hour
**Impact**: Better UX for missing data

---

### Priority 3: Documentation & Schema Clarity (LATER)

**File**: `pipeline/schemas.py`

**Add Field Documentation**:
```python
@dataclass
class EIMASResult:
    # ... required fields ...

    # Optional: Only in full mode (Phase 2 enhanced)
    bubble_risk: Optional[BubbleRiskMetrics] = None

    # Optional: Only in full mode (Phase 7)
    ai_report: Optional[Dict] = None

    # Optional: Only in backtest mode
    backtest_run_id: Optional[str] = None

    # Deprecated: Use debate_consensus instead
    agent_outputs: Optional[AgentOutputs] = None       # TODO: Remove in v3.0
    debate_results: Optional[DebateResults] = None     # TODO: Remove in v3.0
    verification: Optional[VerificationResults] = None # TODO: Remove in v3.0
```

**File**: `README.md`

**Add Execution Mode Matrix**:
```markdown
| Field | Quick | Full | Backtest |
|-------|-------|------|----------|
| bubble_risk | ❌ | ✅ | ✅ |
| ai_report | ❌ | ✅ | ✅ |
| backtest_run_id | ❌ | ❌ | ✅ |
```

**Effort**: 30 minutes
**Impact**: Reduces confusion for future developers

---

## Conclusion

### Summary

**No pipeline bugs found.** All null values are:
1. ✅ **Expected** (missing source data, mode-dependent, conditional)
2. ✅ **Properly handled** in backend (Python Optional fields)
3. ✅ **Correctly serialized** (None → JSON null)
4. ✅ **Safely ignored** in frontend (unused fields)

### Risk Assessment

| Category | Risk Level | Action |
|----------|------------|--------|
| Pipeline Logic | ✅ NONE | No changes needed |
| Data Quality | ✅ LOW | Accept API limitations |
| Frontend Crash | ⚠️ LOW | Add defensive null checks (P1) |
| User Experience | ⚠️ MEDIUM | Display "N/A" for missing data (P2) |

### Next Steps

1. ✅ **COMPLETED**: Root cause analysis (this document)
2. ⏭️ **NEXT**: Report findings to team-lead
3. ⏭️ **NEXT**: Update Task #3 (Fix frontend 0/NA display issues) with specific file list
4. ⏭️ **FUTURE**: Implement Priority 1 fixes

---

## Appendix

### A. Full List of Null Values (Quick Mode)

```json
{
  "regime.error_code": null,                                          // ✅ Expected (no errors)
  "regime.error_msg": null,                                           // ✅ Expected (no errors)
  "bubble_risk": null,                                                // ✅ Expected (quick mode)
  "company_ra_analysis.companies[3].valuation.ev_to_ebitda": null,   // ✅ Expected (JPM)
  "company_ra_analysis.companies[3].accounting.operating_income": null, // ✅ Expected (JPM)
  "company_ra_analysis.companies[3].ratios.operating_margin": null,  // ✅ Expected (JPM)
  "company_ra_analysis.etf_strategy_snapshot[0-9].expense_ratio_pct": null, // ✅ Expected (10 ETFs)
  "agent_outputs": null,                                              // ✅ Expected (deprecated)
  "debate_results": null,                                             // ✅ Expected (deprecated)
  "verification": null,                                               // ✅ Expected (deprecated)
  "ai_report": null,                                                  // ✅ Expected (quick mode)
  "failsafe_status.reason": null,                                     // ✅ Expected (triggered=false)
  "failsafe_status.fallback_action": null,                            // ✅ Expected (triggered=false)
  "backtest_run_id": null                                             // ✅ Expected (not backtest mode)
}
```

### B. Files Checked

**Pipeline** (Data Flow):
- ✅ `lib/regime_detector.py` (RegimeDetector)
- ✅ `pipeline/analyzers_core.py` (detect_regime)
- ✅ `pipeline/schemas.py` (EIMASResult, RegimeResult)
- ✅ `main.py` (orchestrator)

**Frontend** (Display):
- ✅ `frontend/components/MetricsGrid.tsx`
- ✅ `frontend/components/TabbedDashboard.tsx`
- ✅ `frontend/lib/types.ts` (EIMASAnalysis interface)

**Collectors** (Data Sources):
- ✅ `pipeline/collectors.py` (yfinance, FRED)
- ✅ `lib/korea_data_collector.py`

### C. Test Commands Used

```bash
# 1. Run pipeline modes
python main.py --quick
python main.py --full

# 2. Analyze JSON outputs
cat outputs/eimas_*.json | jq '.regime'
cat outputs/eimas_*.json | python3 -c "import json, sys; data=json.load(sys.stdin); ..."

# 3. Search code
grep -rn "volatility_state" pipeline/ lib/ frontend/
grep -rn "RegimeDetector" pipeline/

# 4. Check file structure
ls -la outputs/eimas_*.json | tail -1
find frontend/components -name "*.tsx"
```

---

**Report Complete**
**Status**: Ready for team-lead review
**Date**: 2026-02-16 16:45 KST
