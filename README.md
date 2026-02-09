# EIMAS: Economic Intelligence Multi-Agent System 🚀

**AI-Native Macroeconomic Risk Analysis & Portfolio Strategy System**

EIMAS is a next-generation financial research pipeline that integrates advanced econometrics with a Multi-Agent system to solve the "Black Box" problem in financial AI.

[![Version](https://img.shields.io/badge/version-2.2.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)]()
[![License](https://img.shields.io/badge/license-Educational-orange.svg)]()

---

## 🔍 Key Features

### 1. Multi-Agent Investment Committee
7 specialized AI agents work together:
- **Critical Path Analyst**: Quantitative risk scoring using 68 macro indicators
- **LASSO Forecaster**: Fed rate predictions via high-dimensional feature selection
- **Real-time Researcher**: Web-scale search via Perplexity API
- **HRP Strategist**: Hierarchical Risk Parity portfolio allocation
- **Academic Panel**: Monetarist, Keynesian, Austrian perspective debates
- **Methodology Auditor**: Optimal statistical model selection (VAR vs. GARCH)
- **Verification Agent**: Logic & fact-checking to eliminate hallucinations

### 2. Multi-LLM Consensus Engine
3-round debate between **Claude**, **GPT-4**, and **Gemini** for unbiased consensus.

### 3. Traceable Decision Making
Full "Reasoning Chain" with input, output, and confidence for every step.

### 4. Advanced Econometrics
GMM regime detection, GARCH volatility, VPIN microstructure, DTW similarity, HRP optimization.

---

## 🛠️ Quick Start

### Installation
```bash
git clone https://github.com/Eom-TaeJun/eimas.git
cd eimas
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

### Run Analysis
```bash
# Canonical pipeline entrypoint (single source of truth)
python main.py              # Default analysis
python main.py --short      # Quick mode
python main.py --full       # Full mode with Multi-LLM
python main.py --realtime   # Real-time streaming
python main.py --full --paper-auto --paper-account ra_auto  # Auto LIMIT paper execution
python main.py --paper-auto --paper-poll-only --paper-account ra_auto  # Poll pending paper orders
python scripts/auto_paper_execution.py --run-backtest  # Auto execution + backtest loop
```

`cli/eimas.py run` is a thin wrapper that forwards arguments to `main.py`.
All integrated pipeline run options are defined only in `main.py`.

### Run Web Dashboard
```bash
./run_all.sh
# Access at http://localhost:3002
```

---

## 📂 Project Structure

```
eimas/
├── main.py                 # 🚀 Primary CLI entry point
├── api/                    # FastAPI backend server
├── agents/                 # 🤖 7 AI agents + orchestrator
├── agent/                  # 🔬 Economic Insight Agent (causal analysis)
├── core/                   # ⚙️ Core infrastructure & debate protocol
├── pipeline/               # 🔄 Modular analysis pipeline
├── lib/                    # 📚 80+ analysis modules
│   ├── collectors/         #    Data collection (FRED, Market, Crypto)
│   ├── analyzers/          #    Analysis engines
│   └── ...                 #    LASSO, HRP, GARCH, VPIN, etc.
├── frontend/               # 🌐 Next.js dashboard
├── outputs/                # 📁 Analysis results
└── docs/                   # 📖 Extended documentation
```

---

## 📚 Documentation Guide

### Recommended Read Order (Start Here)

Importance tiers are defined in [`command.md`](./command.md) (`P0`~`P3`).

1. [`command.md`](./command.md) - `P0` source of truth for entrypoint/command policy
2. [`README.md`](./README.md) - `P1` project overview and quick start
3. [`ARCHITECTURE.md`](./ARCHITECTURE.md) - `P1` system architecture and boundaries
4. [`CLAUDE.md`](./CLAUDE.md) - `P1` AI-assisted operational context
5. [`CURRENT_STATUS.md`](./CURRENT_STATUS.md) - `P2` latest migration/refactor state
6. [`TODO.md`](./TODO.md) - `P2` active execution checklist
7. [`FULL_EXECUTION_PROCESS.md`](./FULL_EXECUTION_PROCESS.md) - `P2` full run flow and gates

| What You Need | Where to Find It |
|---------------|------------------|
| **Project Overview** | This file (`README.md`) |
| **Command Policy** | [`command.md`](./command.md) - single entrypoint and `main.py --abc` rules |
| **System Architecture** | [`ARCHITECTURE.md`](./ARCHITECTURE.md) - Components, data flow, design patterns |
| **Contribution Guidelines** | [`CONTRIBUTING.md`](./CONTRIBUTING.md) - Setup, code style, PR process |
| **Version History** | [`CHANGELOG.md`](./CHANGELOG.md) - All version changes |
| **Detailed Workflow** | [`WORKFLOW.md`](./WORKFLOW.md) - 796-line comprehensive guide |
| **Claude Code Guide** | [`CLAUDE.md`](./CLAUDE.md) - Quick reference for AI assistants |
| **Full Pipeline Process** | [`FULL_EXECUTION_PROCESS.md`](./FULL_EXECUTION_PROCESS.md) - canonical `main.py --full` flow & gates |
| **Current Refactor Status** | [`CURRENT_STATUS.md`](./CURRENT_STATUS.md) - latest migration state & next actions |
| **Refactor Task Board** | [`TODO.md`](./TODO.md) - active checklist for split/cleanup waves |
| **RA SQL Reboot Runbook** | [`docs/manuals/RA_POSTGRES_REBOOT_RUNBOOK.md`](./docs/manuals/RA_POSTGRES_REBOOT_RUNBOOK.md) - reboot 후 로컬 PostgreSQL 기동/검증 절차 |
| **Economic Insight Agent** | [`agent/README.md`](./agent/README.md) - Causal analysis module |
| **API Endpoints** | [`api/`](./api/) - FastAPI server documentation |
| **Frontend Components** | [`frontend/`](./frontend/) - React dashboard |
| **Backtest Methodology** | [`docs/BACKTEST_METHODOLOGY.md`](./docs/BACKTEST_METHODOLOGY.md) |

---

## 🚧 Unimplemented Features (Roadmap)

> Full details in [`TODO.md`](./TODO.md)

### 🔴 Not Started (0%)

| Feature | Priority | Est. Time |
|---------|----------|-----------|
| CNN Pattern Detection | ⭐ | 3-6 months |
| Smart Contract Deployment | ⭐⭐ | 1 month |
| WebSocket Real-time Dashboard | ⭐⭐⭐ | 4-5 hours |
| IRF (Impulse Response Function) | ⭐⭐⭐ | 1 week |
| Roll's Measure (Effective Spread) | ⭐⭐⭐ | 1 day |

### 🟡 Partially Implemented

| Feature | Current | Target | Priority |
|---------|---------|--------|----------|
| Frontend Charts (Pie, Heatmap) | 40% | 100% | ⭐⭐⭐ |
| Clustering Portfolio (K-means, DBSCAN) | 40% | 100% | ⭐⭐ |
| LLM Domain Fine-tuning | 30% | 100% | ⭐⭐ |
| Palantir Ontology Visualization | 50% | 100% | ⭐⭐ |
| Real-time VPIN | 80% | 100% | ⭐⭐⭐ |

### ✅ Recently Completed (v2.2.0)

- Archive consolidation & project restructuring
- `ARCHITECTURE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`
- lib/ submodule organization (collectors/, analyzers/, etc.)
- BaseCollector & BaseAnalyzer abstract interfaces
- Enhanced bilingual documentation

---

## 🔑 API Keys Required

```bash
# Required
ANTHROPIC_API_KEY="sk-ant-..."    # Claude
FRED_API_KEY="your-key"           # FRED Data

# Optional
OPENAI_API_KEY="sk-..."           # GPT-4
GOOGLE_API_KEY="..."              # Gemini
PERPLEXITY_API_KEY="pplx-..."     # Real-time Search
```

---

## 📊 Sample Output

```json
{
  "timestamp": "2026-01-30T12:00:00",
  "risk_score": 65.3,
  "regime": {"regime": "BULL", "confidence": 0.85},
  "final_recommendation": "HOLD",
  "confidence": 0.72,
  "reasoning_chain": [...]
}
```

---

## 🤝 Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for development setup and guidelines.

---

## 📄 License

This project is for educational and research purposes. Use at your own risk.

---

*Created by EIMAS Development Team (2026)*  
*Version 2.2.0 | Last Updated: 2026-01-30*
