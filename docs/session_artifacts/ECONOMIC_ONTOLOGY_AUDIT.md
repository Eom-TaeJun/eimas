# EIMAS Economic Ontology Audit Report

## Audit Summary

| Category | Status | Notes |
|----------|--------|-------|
| **Graph Implementation** | ✅ Good | `networkx` DiGraph used |
| **Granger Causality** | ✅ Good | statsmodels + fallback |
| **Layer/Ontology** | ✅ Good | POLICY→LIQUIDITY→RISK→ASSET |
| **Shock Propagation** | ✅ Good | DAG-based path finding |
| **Explainability** | 🔶 Partial | Some narrative, needs more |
| **IS-LM Integration** | 🔴 Missing | No explicit macro theory |
| **Crypto Feedback Loop** | 🔴 Missing | USDT→Treasury not explicit |

---

## Detailed Review

### 1. Graph Implementation ✅

**Files reviewed:**
- `lib/shock_propagation_graph.py` (898 lines)
- `lib/causality_graph.py` (1100 lines)
- `lib/causal_network.py` (753 lines)

**Findings:**
- Uses `networkx.DiGraph` for directed causality graphs
- Nodes have explicit layer attributes (POLICY=1, LIQUIDITY=2, RISK_PREMIUM=3, ASSET_PRICE=4)
- Edges include: lag, strength, p-value, correlation, edge_type

### 2. Economic Layer Hierarchy ✅

```
POLICY (Fed Funds, ECB Rate)
    ↓
LIQUIDITY (RRP, TGA, M2, Stablecoins)
    ↓
RISK_PREMIUM (VIX, Credit Spread, HY Spread)
    ↓
ASSET_PRICE (SPY, QQQ, BTC, Gold)
```

**Transmission enforced:** `enforce_layer_order=True` prevents reverse edges

### 3. Granger Causality ✅

```python
# From shock_propagation_graph.py
grangercausalitytests(df, maxlag=10, verbose=False)
```

- Uses statsmodels for proper F-test
- Fallback to Lead-Lag correlation when statsmodels unavailable
- p-value thresholds: STRONG (<0.01), MODERATE (<0.05), WEAK (<0.10)

### 4. Explainability 🔶

**Existing:**
- `visualize_text()` generates text representation
- Edge annotations include lag, correlation, type

**Missing:**
- No natural language explanation generator
- No "mechanism story" for each path
- No economic theory citations

---

## Gaps Identified

### Gap 1: No IS-LM/QTM Integration 🔴

The code lacks explicit macroeconomic transmission mechanisms:

| Theory | Expected Path | Current Status |
|--------|---------------|----------------|
| IS-LM | M↑ → r↓ → I↑ → Y↑ | Not implemented |
| QTM | M↑ → P↑ (long-run) | Not implemented |
| Taylor Rule | π↑ → FFR↑ → SPY↓ | Partially (layer order) |

### Gap 2: Crypto Feedback Loop 🔴

No explicit modeling of:
```
Stablecoin Issuance → Treasury Purchase → Yield↓ → Risk-On → Crypto↑ → More Stablecoins
```

### Gap 3: Semantic Edge Labels 🔶

Current edges have:
- `lag`, `strength`, `p_value`

Missing:
- `sign` (+/-): Positive or negative causality
- `time_horizon`: Short/Medium/Long term
- `mechanism`: Economic theory reference

---

## Recommended Refactoring

### Priority 1: Add Edge Sign & Kinetic Attributes

```python
 @dataclass
class EconomicEdge:
    source: str
    target: str
    sign: str  # "+" or "-"
    lag: int
    time_horizon: str  # "short", "medium", "long"
    mechanism: str  # "monetary_transmission", "risk_premium", etc.
    theory_reference: str  # "IS-LM", "QTM", "Minsky"
```

### Priority 2: IS-LM Transmission Template

```python
ISLM_TEMPLATE = {
    ("M2", "DFF"): {"sign": "-", "mechanism": "liquidity_preference"},
    ("DFF", "I"): {"sign": "-", "mechanism": "investment_response"},
    ("I", "Y"): {"sign": "+", "mechanism": "aggregate_demand"},
}
```

### Priority 3: Narrative Generator

```python
def generate_narrative(path: ShockPath) -> str:
    """Generate economic explanation for shock path"""
    # Example output:
    # "Fed rate cut (DFF↓) → Increases liquidity (M2↑) → 
    #  Reduces risk premium (VIX↓) → Equity rally (SPY↑)"
```

---

## Verdict

**Overall: 7/10**

The codebase has strong technical foundations but lacks explicit economic theory integration. The refactoring should focus on:

1. Adding edge signs and kinetic attributes
2. Implementing IS-LM/QTM transmission templates
3. Creating explainability narratives
