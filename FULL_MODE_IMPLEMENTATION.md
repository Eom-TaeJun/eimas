# --full Mode Implementation Summary

> **Date**: 2026-01-14
> **Feature**: Integrated execution of 7 standalone scripts via `--full` CLI option

---

## Overview

Added `--full` mode to main.py that executes all 7 standalone scripts in addition to the regular EIMAS pipeline, providing comprehensive real-time data collection and analysis.

**Command**: `python main.py --full`

---

## Implementation Details

### 1. New Phase Added: Phase 8

**Location**: After Phase 7 (Whitening & Fact Check), before Summary section

**Executed Scripts (7)**:

| # | Script | Purpose | Output Field |
|---|--------|---------|--------------|
| 8.1 | `intraday_collector.py` | 장중 1분봉 데이터 수집 | `intraday_summary` |
| 8.2 | `crypto_collector.py` | 24/7 암호화폐 모니터링 + 이상 탐지 | `crypto_monitoring` |
| 8.3 | `market_data_pipeline.py` | 다중 API 데이터 수집 (Twelve Data, CryptoCompare) | `intraday_summary['market_pipeline_status']` |
| 8.4 | `event_predictor.py` | 경제 이벤트 예측 (NFP, CPI, FOMC) | `event_predictions` |
| 8.5 | `event_attribution.py` | 이벤트 원인 분석 (Perplexity 연동) | `event_attributions` |
| 8.6 | `event_backtester.py` | 역사적 이벤트 백테스트 | `event_backtest_results` |
| 8.7 | `news_correlator.py` | 이상-뉴스 자동 귀인 (24시간) | `news_correlations` |

### 2. Code Changes

#### 2.1 Imports Added (line 104-111)
```python
# 2026-01-14 독립 스크립트 (--full mode)
from lib.intraday_collector import IntradayCollector
from lib.crypto_collector import CryptoCollector
from lib.event_predictor import EventPredictor
from lib.event_attribution import EventAttributor
from lib.event_backtester import EventBacktester
from lib.news_correlator import NewsCorrelator
import subprocess
```

#### 2.2 EIMASResult Dataclass Extended (line 246-252)
```python
# Extended Standalone Scripts (--full mode only)
intraday_summary: Dict = field(default_factory=dict)
crypto_monitoring: Dict = field(default_factory=dict)
event_predictions: List[Dict] = field(default_factory=list)
event_attributions: List[Dict] = field(default_factory=list)
event_backtest_results: Dict = field(default_factory=dict)
news_correlations: List[Dict] = field(default_factory=list)
```

#### 2.3 Function Signatures Updated
- `run_integrated_pipeline()`: Added `full_mode: bool = False` parameter
- `run_full_pipeline()`: Added `full_mode: bool = False` parameter
- `main()`: Pass `args.full` to pipeline

#### 2.4 CLI Argument Added (line 2942-2947)
```python
parser.add_argument(
    '--full', '-f',
    action='store_true',
    help='Full mode: include standalone scripts (intraday, crypto, events, news)'
)
```

#### 2.5 Phase 8 Implementation (line 2812-2951)
- 140 lines of code
- Executes all 7 standalone scripts when `full_mode=True`
- Error handling for each script (try-except blocks)
- Results stored in EIMASResult fields

#### 2.6 Summary Output Enhanced (line 2997-3012)
- Added "🚀 STANDALONE SCRIPTS (--full mode)" section
- Displays summary of each script's results
- Only shown when `full_mode=True`

#### 2.7 Markdown Report Extended (line 727-812)
- Added "## 11. Standalone Scripts Results (--full mode)"
- 6 subsections (11.1-11.6) for each script category
- Tables for structured data (event predictions, news correlations)

---

## Usage Examples

### Basic Full Mode
```bash
python main.py --full
# Executes: Phase 1-8 (all features including 7 standalone scripts)
```

### Full Mode + Quick Analysis
```bash
python main.py --full --quick
# Phase 2.3-2.10 skipped, but Phase 8 still runs
```

### Full Mode + Realtime + Report
```bash
python main.py --full --realtime --report --duration 60
# Maximum feature set: All phases 1-8 + AI report + Whitening
```

### Cron Mode (Server Automation)
```bash
python main.py --full --cron --output /var/log/eimas
# Minimal output, full analysis including standalone scripts
```

---

## Execution Flow Comparison

### Standard Mode (`python main.py`)
```
Phase 1: Data Collection (FRED, Market, Crypto, RWA)
Phase 2: Analysis (Regime, Risk, Events, etc.)
Phase 3: Multi-Agent Debate
Phase 4: [SKIP] Real-time
Phase 5: Database Storage
Phase 6: [SKIP] AI Report
Phase 7: [SKIP] Whitening
Phase 8: [SKIP] Standalone Scripts
```

### Full Mode (`python main.py --full`)
```
Phase 1: Data Collection (FRED, Market, Crypto, RWA)
Phase 2: Analysis (Regime, Risk, Events, etc.)
Phase 3: Multi-Agent Debate
Phase 4: [SKIP] Real-time
Phase 5: Database Storage
Phase 6: [SKIP] AI Report
Phase 7: [SKIP] Whitening
Phase 8: ✅ Standalone Scripts (7 scripts executed)
  └─ 8.1 Intraday data collection
  └─ 8.2 Crypto monitoring
  └─ 8.3 Market data pipeline (subprocess)
  └─ 8.4 Event predictions
  └─ 8.5 Event attributions
  └─ 8.6 Event backtest
  └─ 7 News correlations
```

---

## Expected Output Format

### Console Output (Phase 8)
```
==================================================
PHASE 8: STANDALONE SCRIPTS EXECUTION
==================================================
Running 7 independent scripts for comprehensive data collection...

[8.1] Intraday data collection...
      ✓ Date: 2026-01-13
      ✓ Tickers: 20
      ✓ Anomalies: 3

[8.2] 24/7 Cryptocurrency monitoring...
      ✓ Symbols: 10
      ✓ Anomalies: 2
      ✓ Risk: MEDIUM

[8.3] Multi-source data pipeline...
      ✓ Data sources collected: 5

[8.4] Economic event predictions...
      ✓ Events predicted: 8
      ✓ Next event: NFP on 2026-01-17

[8.5] Event cause analysis...
      ✓ Events analyzed: 5
      ✓ Recent attribution: Fed rate decision linked to ...

[8.6] Historical event backtesting...
      ✓ Events tested: 12
      ✓ Avg accuracy: 78%

[8.7] Anomaly-news correlation analysis...
      ✓ Correlations found: 7
      ✓ Top correlation: NVDA - AI chip breakthrough...

==================================================
PHASE 8 COMPLETE: All standalone scripts executed
==================================================
```

### Summary Section
```
🚀 STANDALONE SCRIPTS (--full mode)
   Intraday: 20 tickers, 3 anomalies
   Crypto: 10 symbols, Risk: MEDIUM
   Events: 8 predictions
   Attributions: 5 analyzed
   Backtest: 12 events, 78% accuracy
   News: 7 correlations
```

### JSON Output Fields
```json
{
  "intraday_summary": {
    "date": "2026-01-13",
    "tickers_collected": 20,
    "anomalies_detected": 3,
    "top_anomalies": [...]
  },
  "crypto_monitoring": {
    "symbols_monitored": 10,
    "anomalies_detected": 2,
    "risk_level": "MEDIUM",
    "top_anomalies": [...]
  },
  "event_predictions": [...],
  "event_attributions": [...],
  "event_backtest_results": {...},
  "news_correlations": [...]
}
```

---

## Error Handling

Each script execution is wrapped in try-except blocks:
- **On success**: Results stored in EIMASResult fields
- **On failure**: Error message stored (e.g., `{'error': 'API timeout'}`)
- **Graceful degradation**: Other scripts continue even if one fails

---

## Performance Impact

**Estimated Execution Time**:
- Standard mode: ~40 seconds
- Full mode: ~70-90 seconds (추가 30-50초)

**Additional Resources**:
- Network I/O: 7 additional API calls/data fetches
- CPU: Anomaly detection algorithms, correlation analysis
- Memory: ~50MB additional for intraday 1-minute data

---

## Documentation Updates

| File | Section Updated | Change |
|------|-----------------|--------|
| `CLAUDE.md` | Quick Reference | Added `--full` flag examples |
| `lib/README.md` | Standalone Scripts | Referenced Phase 8 integration |
| `COMMANDS.md` | (Future) | Add `--full` mode section |
| `FEATURE_COVERAGE_REPORT.md` | (Future) | Update coverage to 51/95 = 53.7% |

---

## Testing Checklist

- [x] Syntax validation (`python -m py_compile main.py`)
- [ ] Dry run: `python main.py --full --quick` (reduced execution time)
- [ ] Full run: `python main.py --full` (complete test)
- [ ] Verify JSON output includes all 6 new fields
- [ ] Verify markdown report includes Section 11
- [ ] Test error handling: Disconnect network during Phase 8
- [ ] Test with `--cron` flag for server automation

---

## Known Limitations

1. **market_data_pipeline.py** runs via subprocess (CLI), not Python API
   - Reason: The script is designed as standalone CLI tool
   - Alternative: Refactor to importable module in future

2. **API Dependencies**:
   - `event_attribution.py` requires Perplexity API key
   - `news_correlator.py` may require news API access
   - Graceful fallback if APIs unavailable

3. **Intraday Data**:
   - Collects **yesterday's** data by default (not today's live data)
   - Rationale: Market hours constraint, backfill strategy

---

## Future Enhancements

1. **Parallel Execution**: Run 7 scripts concurrently using `asyncio.gather()`
2. **Selective Scripts**: `--full-only=intraday,crypto` to run specific scripts
3. **Schedule Integration**: Cron templates for daily/hourly execution
4. **Dashboard Integration**: Real-time Phase 8 results in frontend
5. **API Mode**: FastAPI endpoint `/analysis/full` for on-demand execution

---

## Commit Details

**Commit Message**:
```
Add --full mode: Execute all 7 standalone scripts in Phase 8

- Integrated IntradayCollector, CryptoCollector, EventPredictor,
  EventAttributor, EventBacktester, NewsCorrelator
- Added Phase 8 execution logic (140 lines)
- Extended EIMASResult dataclass with 6 new fields
- Updated summary and markdown report sections
- CLI flag: --full / -f

Result: Comprehensive real-time analysis mode combining
integrated pipeline (Phase 1-7) with standalone scripts (Phase 8)
```

**Files Modified**:
- `main.py` (+200 lines, 7 imports, Phase 8 implementation)

**Files Created**:
- `FULL_MODE_IMPLEMENTATION.md` (this document)

---

**Status**: ✅ Implementation Complete
**Next Step**: Testing & Documentation Updates
