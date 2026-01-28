# EIMAS: Economic Intelligence Multi-Agent System 🚀

**AI-Native Macroeconomic Risk Analysis & Portfolio Strategy System**

EIMAS is a next-generation financial research pipeline that integrates advanced econometrics with a Multi-Agent system to solve the "Black Box" problem in financial AI.

---

## 🔍 Key Features

### 1. Multi-Agent Investment Committee
Instead of a single biased model, EIMAS utilizes 7 specialized AI agents:
- **Critical Path Analyst**: Quantitative risk scoring using 68 macro indicators.
- **LASSO Forecaster**: High-dimensional feature selection for Fed rate predictions.
- **Real-time Researcher**: Web-scale search for live news & policy context (Perplexity API).
- **HRP Strategist**: Asset allocation using Hierarchical Risk Parity.
- **Academic Panel**: Debates between Monetarist, Keynesian, and Austrian perspectives.
- **Methodology Auditor**: Selects optimal statistical models (VAR vs. GARCH).
- **Verification Agent**: Logic & fact-checking to eliminate hallucinations.

### 2. Multi-LLM Consensus Engine
To ensure objective results, EIMAS orchestrates a 3-round debate between **Claude 3.5**, **GPT-4o**, and **Gemini 1.5 Pro**. This reduces bias and builds a robust consensus on market outlook.

### 3. Traceable Decision Making (Reasoning Chain)
Every recommendation includes a full "Reasoning Chain," detailing the input, output, and confidence score of every agent step. No more black-box signals.

### 4. Advanced Econometrics
- **Regime Detection**: GMM (Gaussian Mixture Model) classification.
- **Volatility Modeling**: GARCH(1,1) persistence analysis.
- **Market Microstructure**: VPIN (Volume-Synchronized Probability of Informed Trading) & Kyle's Lambda.
- **Similarity Mapping**: DTW (Dynamic Time Warping) & DBSCAN for asset outlier detection.

---

## 🛠️ Quick Start

### Installation
```bash
git clone https://github.com/Eom-TaeJun/eimas.git
cd eimas
pip install -r requirements.txt
cp .env.example .env # Add your API keys here
```

### Run Full Pipeline
```bash
python main.py
```
*Generates a comprehensive Markdown report in `outputs/` with IB-style memorandum.*

### Run Web Dashboard
```bash
./run_all.sh
```
*Access the interactive UI at http://localhost:3002.*

---

## 📂 Project Structure
- `agents/`: Implementation of the 7 specialized AI agents.
- `core/`: Multi-LLM debate engine and reasoning chain logic.
- `pipeline/`: Modular data collection and analysis stages.
- `lib/`: Quantitative models (LASSO, HRP, GARCH, VPIN).
- `frontend/`: Next.js dashboard for result visualization.

---

## 📄 License
This project is for educational and research purposes. Use at your own risk.

*Created by EIMAS Development Team (2026)*