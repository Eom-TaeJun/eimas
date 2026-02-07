# EIMAS (Economic Intelligence Multi-Agent System) Overview

This document integrates and summarizes the core functionalities of EIMAS, based on the existing documentation and the current codebase structure. It serves as a blueprint for refactoring the monolithic `main.py` into modular components.

## 1. Core Concept
EIMAS is an AI multi-agent system designed for macroeconomic analysis and investment decision-making. It integrates quantitative methodologies (based on 8 academic papers) with AI agent debates to provide objective market insights.

**Key Features:**
*   **Integrated Pipeline:** Automates data collection, analysis, debate, and reporting in a single workflow.
*   **Multi-Agent Debate:** Utilizes different AI agents (Full Mode vs. Reference Mode) to reach a consensus.
*   **Quantitative Framework:** Implements GMM, Granger Causality, HRP, Bubble Detection, etc.
*   **Real-time Capabilities:** Supports Binance WebSocket for VPIN monitoring.

## 2. Functional Architecture (8-Phase Pipeline)

The system operates through a sequential 8-phase pipeline, currently orchestrated within `main.py`.

### Phase 1: Data Collection
Collects diverse financial data from multiple sources.
*   **FRED Collector:** Fetches macroeconomic indicators like RRP, TGA, Net Liquidity, Fed Funds Rate.
*   **Market Data Manager:** Retrieves historical price data for ETFs (SPY, QQQ, TLT, etc.), Crypto (BTC, ETH), and RWA tokens.
*   **Extended Data:** Collects DeFi TVL and MENA market data.
*   **ARK Analysis:** Analyzes ARK ETF holdings for institutional flow insights.
*   **Correlation Matrix:** Calculates asset correlations.

### Phase 2: Quantitative Analysis
Applies mathematical models to the collected data.
*   **Regime Detection:** Uses GMM (Gaussian Mixture Model) and Entropy to classify market regimes (Bull/Bear/Neutral).
*   **Event Detection:** Identifies significant liquidity and market events based on thresholds.
*   **Liquidity Analysis:** Performs Granger Causality tests to see if liquidity changes predict market moves (e.g., Net Liquidity → SPY).
*   **Critical Path & Risk:** Aggregates risk scores from various paths; includes Microstructure risk (VPIN, Amihud) and Bubble risk (Greenwood-Shleifer).
*   **Advanced Modules:**
    *   **Genius Act Macro:** Analyzes Stablecoin supply vs. Fed liquidity.
    *   **Theme ETF:** Analyzes supply chain risks for specific themes (e.g., AI/Semiconductor).
    *   **Shock Propagation:** Maps how shocks travel through the asset network.
    *   **GC-HRP Portfolio:** Optimizes portfolio weights using Hierarchical Risk Parity with Graph Clustering.
    *   **Volume Anomaly:** Detects unusual volume patterns indicative of private information flow (Kyle's model).

### Phase 3: Multi-Agent Debate
Simulates a debate between AI agents with different perspectives to form a robust conclusion.
*   **Full Mode Agent:** Analyzes long-term trends (365 days), tending to be optimistic.
*   **Reference Mode Agent:** Analyzes short-term momentum (90 days), tending to be conservative.
*   **Consensus Engine:** Synthesizes the views of both agents into a final recommendation (Bullish/Bearish/Neutral) with a confidence score.
*   **Adaptive Agents:** (Optional) Simulates Aggressive, Balanced, and Conservative portfolios based on market conditions.

### Phase 4: Real-time Monitoring (Optional)
*   **Binance Streamer:** Connects to Binance WebSocket to calculate VPIN (Volume-Synchronized Probability of Informed Trading) in real-time for BTC/ETH.

### Phase 5: Database Storage
Persists analysis results for tracking and historical reference.
*   **Event DB:** Stores detected events and market snapshots.
*   **Signal DB:** Records real-time and integrated signals.
*   **Trading DB:** Saves portfolio candidates and generated trading signals.
*   **Predictions DB:** Logs event predictions for future verification.

### Phase 6: AI Reporting (Optional)
*   **Report Generator:** Uses LLMs (Claude/Perplexity) to write a comprehensive investment report in natural language, explaining the quantitative findings.

### Phase 7: Quality Assurance (Optional)
*   **Whitening Engine:** Provides economic interpretations for the generated portfolio allocations ("Why this weight?").
*   **Fact Checker:** Verifies the claims made in the AI report against data facts.

### Phase 8: Standalone Scripts (Full Mode)
Runs independent modules for broader market coverage.
*   **Intraday Collector:** Gathers 1-minute interval data.
*   **Crypto Monitor:** 24/7 Cryptocurrency anomaly detection.
*   **Event Predictor:** Forecasts upcoming economic events (NFP, CPI, FOMC).
*   **Event Attribution:** Explains *why* an event happened using news search.
*   **News Correlator:** Links volume anomalies to specific news headlines.

## 3. Refactoring Plan for `main.py`

The goal is to split `main.py` into smaller, focused scripts based on the phases above.

| Proposed Script Name | Functionality Description |
| :--- | :--- |
| `run_data_collection.py` | Handles Phase 1. Collects FRED, Market, Crypto, and ARK data. |
| `run_analysis.py` | Handles Phase 2. Runs Regime, Risk, Liquidity, and Portfolio models. |
| `run_debate.py` | Handles Phase 3. Orchestrates the multi-agent debate and consensus. |
| `run_realtime.py` | Handles Phase 4. Manages the Binance WebSocket connection. |
| `run_storage.py` | Handles Phase 5. Saves results to SQL/SQLite databases. |
| `run_reporting.py` | Handles Phase 6 & 7. Generates reports and performs QA. |
| `run_standalone.py` | Handles Phase 8. Executes independent utility scripts. |
| `main.py` (`run_integrated_pipeline`) | Canonical entry point. Orchestrates the above phases based on arguments (Quick/Full/Report modes). |

This structure will make the codebase more maintainable, testable, and easier to understand.
