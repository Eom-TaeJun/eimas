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
python main.py              # Default analysis
python main.py --short      # Quick mode
python main.py --full       # Full mode with Multi-LLM
python main.py --realtime   # Real-time streaming
```

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
├── docs/                   # 📖 Extended documentation
└── archive/                # 📦 Historical code & docs
```

---

## 📚 Documentation Guide

| What You Need | Where to Find It |
|---------------|------------------|
| **Project Overview** | This file (`README.md`) |
| **System Architecture** | [`ARCHITECTURE.md`](./ARCHITECTURE.md) - Components, data flow, design patterns |
| **Contribution Guidelines** | [`CONTRIBUTING.md`](./CONTRIBUTING.md) - Setup, code style, PR process |
| **Version History** | [`CHANGELOG.md`](./CHANGELOG.md) - All version changes |
| **Detailed Workflow** | [`WORKFLOW.md`](./WORKFLOW.md) - 796-line comprehensive guide |
| **Claude Code Guide** | [`CLAUDE.md`](./CLAUDE.md) - Quick reference for AI assistants |
| **Economic Insight Agent** | [`agent/README.md`](./agent/README.md) - Causal analysis module |
| **API Endpoints** | [`api/`](./api/) - FastAPI server documentation |
| **Frontend Components** | [`frontend/`](./frontend/) - React dashboard |
| **Backtest Methodology** | [`docs/BACKTEST_METHODOLOGY.md`](./docs/BACKTEST_METHODOLOGY.md) |
| **Historical Development** | [`archive/docs/`](./archive/docs/) - Phase reports, TODO lists |

---

## 🚧 Unimplemented Features (Roadmap)

> Full details in [`archive/docs/notcompleted.md`](./archive/docs/notcompleted.md)

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