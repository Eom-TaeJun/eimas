# EIMAS Command & Output Guide

This document maps every EIMAS execution command to its specific outputs, files generated, and visible results.

## 1. Quick Reference Table

| Command | Time | Key Features Active | Output Files |
| :--- | :--- | :--- | :--- |
| `python main_orchestrator.py` | ~40s | Data, Regime, Risk, Debate, DB | JSON, MD, HTML, DB |
| `python main_orchestrator.py --quick` | ~16s | Data, Regime, Risk (Fast) | JSON, MD, HTML |
| `python main_orchestrator.py --full` | ~90s | **All of above** + Standalone Scripts | JSON, MD, HTML, DB |
| `python main_orchestrator.py --report` | ~90s | **All of above** + AI Report | JSON, MD, HTML, DB, **AI Report (MD)** |
| `python main_orchestrator.py --realtime` | +30s | **All of above** + VPIN Monitor | JSON, MD, HTML, DB (includes VPIN logs) |

---

## 2. Detailed Command Analysis

### A. Standard Execution (Default)
**Command:**
```bash
python projects/autoai/eimas/main_orchestrator.py
```
**What happens:**
1.  **Data Collection:** Fetches FRED & Market data.
2.  **Analysis:** Runs GMM Regime, Granger Liquidity, Bubble Detection, Critical Path Risk.
3.  **Debate:** Runs Full vs Reference Agent debate.
4.  **Storage:** Saves to SQL databases.

**Visible Outputs:**
*   **Terminal:** Progress logs, final recommendation (Action/Confidence/Risk), top portfolio weights.
*   **Files Created (`outputs/`):**
    *   `integrated_YYYYMMDD_HHMMSS.json`: Full raw data dump.
    *   `integrated_YYYYMMDD_HHMMSS.md`: Structured report with 12 sections (Data, Regime, Risk, Debate, Portfolio...).
    *   `dashboard_YYYYMMDD_HHMMSS.html`: Interactive HTML dashboard for browser viewing.
*   **Databases (`data/`):**
    *   `events.db`: New market snapshot & detected events.
    *   `trading.db`: New portfolio candidates & signals.

---

### B. Quick Mode
**Command:**
```bash
python projects/autoai/eimas/main_orchestrator.py --quick
```
**What happens:**
*   Skips heavy computations: Liquidity Causality, ETF Flow, Shock Propagation, Detailed Microstructure.
*   Focuses on: Basic Regime (GMM), Basic Risk Score, Core Data.

**Visible Outputs:**
*   **Terminal:** Faster execution logs.
*   **Files:** Same JSON/MD/HTML structure, but "Advanced Analysis" sections in MD/HTML will show "N/A" or simplified data.

---

### C. Full Mode
**Command:**
```bash
python projects/autoai/eimas/main_orchestrator.py --full
```
**What happens:**
*   **Includes Standard Execution.**
*   **Runs Standalone Scripts (Phase 8):**
    *   Intraday Data Collector (1-min bars).
    *   Crypto Monitor (24/7 anomaly check).
    *   Event Predictor (Next FOMC/CPI dates).
    *   News Correlator (Links anomalies to news).

**Visible Outputs:**
*   **Terminal:** Additional logs for "Phase 8: Standalone Scripts".
*   **Files:**
    *   `integrated_*.md`: Includes a new section **"8. Standalone Scripts"** showing Intraday status, Crypto risks, and Event predictions.

---

### D. Report Mode (AI-Powered)
**Command:**
```bash
python projects/autoai/eimas/main_orchestrator.py --report
```
**What happens:**
*   **Includes Standard Execution.**
*   **Generates AI Report:** Uses Claude/Perplexity to write a natural language investment memo.
*   **Quality Assurance:** Runs Whitening (Economic explanation) & Fact Checking.

**Visible Outputs:**
*   **Terminal:** Logs for "Phase 6: AI Report" and "Phase 7: QA".
*   **Files:**
    *   `ai_report_YYYYMMDD_HHMMSS.md`: A separate, high-quality investment memo written by AI.
    *   `integrated_*.md`: Includes "Quality Assurance" section with Fact Check grade.

---

### E. Realtime Mode
**Command:**
```bash
python projects/autoai/eimas/main_orchestrator.py --realtime --duration 60
```
**What happens:**
*   **Includes Standard Execution.**
*   **Runs Phase 4:** Connects to Binance WebSocket for 60 seconds.
*   **Calculates VPIN:** Measures toxic flow in real-time.

**Visible Outputs:**
*   **Terminal:** Live updates of VPIN metrics for BTC/ETH.
*   **Files:**
    *   `integrated_*.md`: Includes **"Real-time Signals"** section with VPIN stats.
    *   `dashboard_*.html`: Real-time panel populated with latest VPIN data.

---

## 3. How to View Results

### Option 1: Terminal Summary (Immediate)
Look at the end of execution:
```
======================================================================
EIMAS PIPELINE COMPLETE in 42.1s
======================================================================
Action: BULLISH (85%)
Risk: 51.0/100
Regime: Bull
JSON: outputs/integrated_20260116_100000.json
MD:   outputs/integrated_20260116_100000.md
```

### Option 2: Markdown Report (Detailed)
Open the generated `.md` file in VS Code (Preview) or any Markdown viewer. It contains formatted tables and lists of all analysis.

### Option 3: HTML Dashboard (Visual)
Open the generated `.html` file in Chrome/Edge/Safari. It provides a visual summary with charts and cards.

### Option 4: Web Dashboard (Advanced)
If you want the full React UI:
1.  Run the backend: `uvicorn api.main:app --reload`
2.  Run the frontend: `cd frontend && npm run dev`
3.  Go to `http://localhost:3000`
4.  This UI pulls data from the `data/` databases populated by your command.
