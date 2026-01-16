# Elicit Academic Paper Search Prompts

> Module-specific search queries for academic validation
> Target: Elicit.org, Google Scholar, SSRN
> Date: 2026-01-12

---

## Module 1: Fed Liquidity and Market Returns

### Primary Search Query
```
"Federal Reserve liquidity" AND ("net liquidity" OR "balance sheet") AND "asset prices" AND ("empirical" OR "causal")
```

### Refined Queries

**Q1.1 - Net Liquidity Definition**
```
"Federal Reserve balance sheet" AND "reverse repo" AND "TGA" AND "market liquidity" AND "transmission mechanism"
```

**Q1.2 - Liquidity-Market Causality**
```
"central bank liquidity" AND "stock market" AND ("Granger causality" OR "VAR" OR "vector autoregression") AND "empirical"
```

**Q1.3 - QE/QT Effects**
```
"quantitative easing" AND "asset prices" AND ("event study" OR "difference-in-differences") AND ("2008-2024" OR "recent")
```

### Key Papers to Validate Against
- Bernanke & Kuttner (2005): Fed actions and stock prices
- Carpenter et al. (2015): SCB balance sheet and financial markets
- Vissing-Jorgensen (2021): QE transmission channels

### Expected Findings
- Confirm Granger causality from Fed liquidity to equity returns
- Validate Net Liquidity = Fed BS - RRP - TGA formula
- Find lag structure (typically 5-20 trading days)

---

## Module 2: Regime Detection (Rule-Based + GMM)

### Primary Search Query
```
("regime switching" OR "regime detection") AND ("stock market" OR "equity") AND ("Hamilton" OR "Gaussian mixture") AND "empirical"
```

### Refined Queries

**Q2.1 - Hamilton GMM Application**
```
"Markov regime switching" AND ("Hamilton 1989" OR "Hamilton filter") AND "stock returns" AND "volatility clustering"
```

**Q2.2 - Bull/Bear/Neutral Classification**
```
"market regime" AND ("bull market" OR "bear market") AND "classification" AND ("machine learning" OR "statistical")
```

**Q2.3 - Technical Indicators**
```
"moving average crossover" AND "market timing" AND ("backtest" OR "out-of-sample") AND "performance"
```

### Key Papers to Validate Against
- Hamilton (1989): Original Markov regime-switching paper
- Ang & Bekaert (2002): Regime switches in stock returns
- Guidolin & Timmermann (2008): International stock returns and regimes

### Expected Findings
- Confirm 3-state model (Bull/Neutral/Bear) is sufficient
- Validate VIX thresholds (20/30) for regime boundaries
- Find evidence for non-stationarity requiring regime models

---

## Module 3: Shannon Entropy for Market Uncertainty

### Primary Search Query
```
"Shannon entropy" AND ("market uncertainty" OR "financial markets") AND ("regime" OR "volatility") AND "quantification"
```

### Refined Queries

**Q3.1 - Entropy in Finance**
```
"information entropy" AND "financial time series" AND ("uncertainty measure" OR "predictability")
```

**Q3.2 - Entropy and Market States**
```
"entropy" AND "Markov regime" AND ("posterior probability" OR "state uncertainty") AND "Bayesian"
```

**Q3.3 - Entropy vs VIX**
```
"entropy" AND ("VIX" OR "implied volatility") AND "uncertainty" AND "comparison"
```

### Key Papers to Validate Against
- Zhou et al. (2013): Multiscale entropy and stock market
- Pincus (1991): Approximate entropy for financial data
- Dionisio et al. (2006): Entropy-based tests for financial markets

### Expected Findings
- Validate H = 0 (perfect certainty) to H = log_2(K) (maximum uncertainty)
- Find relationship between entropy and market crashes
- Confirm entropy differs from volatility (orthogonal information)

---

## Module 4: Market Microstructure (Amihud Lambda & VPIN)

### Primary Search Query
```
("Amihud illiquidity" OR "VPIN") AND "high frequency" AND ("market quality" OR "liquidity") AND "empirical"
```

### Refined Queries

**Q4.1 - Amihud Lambda Validation**
```
"Amihud illiquidity measure" AND "daily data" AND ("cross-sectional" OR "time series") AND "asset pricing"
```

**Q4.2 - VPIN and Flash Crashes**
```
"VPIN" AND ("volume-synchronized" OR "order flow") AND ("flash crash" OR "market stress") AND "prediction"
```

**Q4.3 - Liquidity Risk Premium**
```
"illiquidity premium" AND "expected returns" AND ("Fama-French" OR "factor model") AND "recent"
```

### Key Papers to Validate Against
- Amihud (2002): Illiquidity and stock returns (original paper)
- Easley et al. (2012): Flow toxicity and VPIN (original paper)
- Ahn et al. (2018): Daily VPIN approximation

### Expected Findings
- Confirm Amihud Lambda works with daily data (not just intraday)
- Validate VPIN approximation using daily up/down volume
- Find liquidity risk premium ranges (2-4% annually)

---

## Module 5: Bubble Detection (Greenwood-Shleifer)

### Primary Search Query
```
"bubble detection" AND ("run-up" OR "price momentum") AND ("ex-ante" OR "real-time") AND ("Greenwood" OR "Shleifer")
```

### Refined Queries

**Q5.1 - Bubbles for Fama Replication**
```
"Greenwood Shleifer" AND "bubbles for Fama" AND ("2-year return" OR "100% threshold") AND "prediction"
```

**Q5.2 - Early Warning Indicators**
```
"asset price bubble" AND "early warning" AND ("volatility" OR "trading volume" OR "issuance") AND "empirical"
```

**Q5.3 - Post-Bubble Crashes**
```
"bubble burst" AND "subsequent returns" AND ("5-year" OR "long-run") AND "predictability"
```

### Key Papers to Validate Against
- Greenwood et al. (2019): Bubbles for Fama (original paper)
- Phillips et al. (2015): Real-time bubble detection (GSADF test)
- Brunnermeier & Oehmke (2013): Bubbles, financial crises, and systemic risk

### Expected Findings
- Confirm 100% two-year return threshold for bubble identification
- Validate poor subsequent returns after bubbles (-40% in 3 years)
- Find volatility spikes accompany bubble formation

---

## Module 6: Hierarchical Risk Parity (HRP) with MST

### Primary Search Query
```
("hierarchical risk parity" OR "HRP") AND ("De Prado" OR "Lopez de Prado") AND ("minimum spanning tree" OR "MST") AND "portfolio"
```

### Refined Queries

**Q6.1 - HRP Original Paper**
```
"hierarchical risk parity" AND "De Prado" AND ("out-of-sample" OR "backtest") AND "diversification"
```

**Q6.2 - MST in Finance**
```
"minimum spanning tree" AND ("correlation matrix" OR "Mantegna") AND "portfolio optimization" AND "clustering"
```

**Q6.3 - HRP vs Markowitz**
```
"hierarchical risk parity" AND "mean-variance" AND "comparison" AND ("robust" OR "estimation error")
```

### Key Papers to Validate Against
- De Prado (2016): Building Diversified Portfolios (original HRP paper)
- Mantegna (1999): Hierarchical structure in financial markets (MST)
- Raffinot (2017): Hierarchical clustering-based asset allocation

### Expected Findings
- Confirm HRP outperforms Markowitz out-of-sample
- Validate MST distance formula: d = sqrt(2*(1-ρ))
- Find HRP reduces extreme portfolio weights (stabilizes allocation)

---

## Module 7: Graph Centrality for Systemic Risk

### Primary Search Query
```
"network centrality" AND ("systemic risk" OR "financial contagion") AND ("minimum spanning tree" OR "correlation network") AND "empirical"
```

### Refined Queries

**Q7.1 - Betweenness Centrality in Finance**
```
"betweenness centrality" AND "financial networks" AND ("shock propagation" OR "contagion") AND "stock market"
```

**Q7.2 - Eigenvector Centrality Issues**
```
"eigenvector centrality" AND ("tree structure" OR "minimum spanning tree") AND ("limitation" OR "unsuitable")
```

**Q7.3 - Dynamic Network Evolution**
```
"time-varying network" AND "financial markets" AND ("rolling window" OR "dynamic") AND "topology"
```

### Key Papers to Validate Against
- Billio et al. (2012): Econometric measures of systemic risk
- Diebold & Yilmaz (2014): Network connectedness of financial institutions
- Battiston et al. (2012): Complexity and systemic risk

### Expected Findings
- Confirm betweenness centrality identifies shock propagators
- Validate eigenvector centrality fails on trees (constant values)
- Find adaptive node selection (sqrt(N)) is common heuristic

---

## Module 8: Stablecoin Risk and Fed Liquidity

### Primary Search Query
```
("stablecoin" OR "USDC" OR "USDT") AND ("systemic risk" OR "regulatory risk") AND ("collateral" OR "reserve") AND "classification"
```

### Refined Queries

**Q8.1 - Stablecoin Taxonomy**
```
"stablecoin" AND ("fiat-backed" OR "crypto-backed" OR "algorithmic") AND "risk assessment" AND ("BIS" OR "FSB")
```

**Q8.2 - Securities Law Application**
```
"stablecoin" AND "securities law" AND ("Howey test" OR "SEC") AND ("interest-bearing" OR "yield")
```

**Q8.3 - Stablecoin-DeFi Liquidity**
```
"stablecoin" AND "DeFi" AND ("liquidity shock" OR "de-peg") AND "contagion"
```

### Key Papers to Validate Against
- Lyons & Viswanath-Natraj (2023): Stablecoins and systemic risk
- Gorton & Zhang (2023): Stablecoins: Growth potential and impact
- Aramonte et al. (2022): DeFi risks and decentralization illusion (BIS)

### Expected Findings
- Confirm USDC < USDT < DAI < USDe risk ordering (collateral-based)
- Validate interest-bearing stablecoins have SEC risk (+15 penalty)
- Find multidimensional risk scoring (credit, liquidity, regulatory, technical)

---

## Module 9: Bekaert VIX Decomposition

### Primary Search Query
```
"Bekaert" AND "VIX" AND ("uncertainty" OR "risk aversion") AND "decomposition" AND "monetary policy"
```

### Refined Queries

**Q9.1 - Original Bekaert Paper**
```
"Bekaert Hoerova Lo Duca" AND "VIX decomposition" AND "uncertainty" AND "risk appetite" AND "2013"
```

**Q9.2 - VIX and Fed Policy**
```
"VIX" AND "Federal Reserve" AND ("monetary policy surprise" OR "FOMC") AND "response"
```

**Q9.3 - Uncertainty vs Risk Aversion**
```
"uncertainty shock" AND "risk aversion" AND "identification" AND ("VAR" OR "structural") AND "financial markets"
```

### Key Papers to Validate Against
- Bekaert et al. (2013): Risk, uncertainty and monetary policy (original)
- Baker et al. (2016): Measuring economic policy uncertainty (EPU index)
- Bloom (2009): The impact of uncertainty shocks

### Expected Findings
- Confirm VIX can be decomposed into two orthogonal components
- Validate uncertainty component responds to macro surprises
- Find risk aversion component correlates with leverage constraints

---

## Module 10: Multi-Agent Consensus Mechanisms

### Primary Search Query
```
("multi-agent" OR "ensemble") AND ("financial forecasting" OR "trading") AND "consensus" AND ("debate" OR "aggregation")
```

### Refined Queries

**Q10.1 - Ensemble Forecasting in Finance**
```
"ensemble methods" AND "financial forecasting" AND ("combination" OR "aggregation") AND ("out-of-sample" OR "backtest")
```

**Q10.2 - Multi-Timeframe Analysis**
```
"multiple time horizons" AND "trading strategy" AND ("short-term" OR "long-term") AND "combination"
```

**Q10.3 - LLM-Based Agents**
```
("large language model" OR "LLM") AND "financial analysis" AND ("agent" OR "multi-agent") AND "performance"
```

### Key Papers to Validate Against
- Timmermann (2006): Forecast combinations in econometrics
- Rapach et al. (2010): Out-of-sample equity premium prediction
- Hong & Page (2004): Groups of diverse problem solvers (theoretical)

### Expected Findings
- Confirm ensemble methods reduce forecast errors (diversity benefit)
- Validate multi-timeframe consensus outperforms single timeframe
- Find LLM-based agents can process qualitative information

---

## Module 11: Granger Causality in Financial Networks

### Primary Search Query
```
"Granger causality" AND ("financial markets" OR "stock returns") AND ("VAR" OR "vector autoregression") AND "network"
```

### Refined Queries

**Q11.1 - Granger Causality Testing**
```
"Granger causality test" AND "financial time series" AND ("optimal lag" OR "AIC" OR "BIC") AND "methodology"
```

**Q11.2 - Shock Propagation**
```
"shock propagation" AND ("impulse response" OR "variance decomposition") AND "financial markets" AND "VAR"
```

**Q11.3 - Limitations of Granger Causality**
```
"Granger causality" AND ("limitation" OR "critique") AND ("nonlinearity" OR "structural break") AND "econometrics"
```

### Key Papers to Validate Against
- Granger (1969): Investigating causal relations (original paper)
- Billio et al. (2012): Granger causality networks (econometric measures)
- Diebold & Yilmaz (2009): Measuring financial asset return and volatility spillovers

### Expected Findings
- Confirm Granger causality captures predictive relationships
- Validate lag selection crucial (typically 1-5 days for daily data)
- Find nonlinear extensions (threshold VAR) improve performance

---

## Module 12: Risk Enhancement Layer (Novel Contribution)

### Primary Search Query
```
("risk model" OR "risk score") AND "ensemble" AND ("microstructure" OR "liquidity") AND ("bubble" OR "tail risk") AND "additive"
```

### Refined Queries

**Q12.1 - Risk Model Combination**
```
"risk model combination" AND ("ensemble" OR "aggregation") AND "portfolio management" AND "empirical"
```

**Q12.2 - Additive Risk Factors**
```
"additive risk factors" AND ("VaR" OR "risk decomposition") AND "financial markets"
```

**Q12.3 - Microstructure and Asset Pricing**
```
"market microstructure" AND "asset pricing" AND ("liquidity risk" OR "information asymmetry") AND "premium"
```

### Key Papers to Validate Against
- Acharya & Pedersen (2005): Asset pricing with liquidity risk
- Pastor & Stambaugh (2003): Liquidity risk and expected returns
- Patton & Verardo (2012): Does beta move with news? (information risk)

### Expected Findings
- Validate additive risk decomposition is common in practice
- Confirm microstructure adjustments improve risk forecasts
- Find bubble risk should be added (not multiplied) to base risk

---

## Cross-Cutting Themes

### Theme 1: Out-of-Sample Testing
```
"out-of-sample" AND ("backtest" OR "walk-forward") AND "financial forecasting" AND ("overfitting" OR "data mining")
```

### Theme 2: Regime-Adaptive Strategies
```
"regime-dependent" AND "portfolio strategy" AND ("switching" OR "adaptive") AND "performance"
```

### Theme 3: Crisis Prediction
```
("financial crisis" OR "market crash") AND "early warning" AND ("indicator" OR "signal") AND "real-time"
```

### Theme 4: High-Frequency Data with Daily Models
```
"high frequency data" AND "daily aggregation" AND ("liquidity" OR "volatility") AND "approximation"
```

---

## Search Strategy Recommendations

### Phase 1: Foundational Papers (Week 1)
- Start with original papers (Hamilton 1989, Granger 1969, Amihud 2002, etc.)
- Focus on methodology validation
- Read 15-20 key papers

### Phase 2: Recent Applications (Week 2)
- Search for 2020-2024 papers (COVID-19 era, Fed tightening)
- Look for ML/AI applications in finance
- Read 10-15 recent papers

### Phase 3: Critique and Limitations (Week 3)
- Find papers criticizing methods used
- Look for failures and edge cases
- Prepare rebuttals or acknowledgments
- Read 5-10 critique papers

### Phase 4: Novel Contributions (Week 4)
- Search for gaps in literature
- Identify what EIMAS adds (Risk Enhancement Layer, Multi-Agent Consensus)
- Draft "Contribution to Literature" section

---

## Expected Citation Structure

### Methodology Section
```
Regime Detection:
  - Hamilton (1989) [original GMM]
  - Ang & Bekaert (2002) [application to stocks]
  - Guidolin & Timmermann (2008) [international]

Liquidity Analysis:
  - Granger (1969) [causality test]
  - Amihud (2002) [illiquidity measure]
  - Easley et al. (2012) [VPIN]

Portfolio Optimization:
  - Mantegna (1999) [MST in finance]
  - De Prado (2016) [HRP algorithm]
  - Raffinot (2017) [empirical validation]

Bubble Detection:
  - Greenwood et al. (2019) [run-up criterion]
  - Phillips et al. (2015) [real-time detection]

Uncertainty Measurement:
  - Shannon (1948) [entropy]
  - Bekaert et al. (2013) [VIX decomposition]
  - Baker et al. (2016) [policy uncertainty]
```

### Results Section
```
Backtest Performance:
  - Jegadeesh & Titman (1993) [momentum baseline]
  - Fama & French (2015) [factor model comparison]
  - DeMiguel et al. (2009) [1/N portfolio benchmark]

Out-of-Sample Testing:
  - Campbell & Thompson (2008) [equity premium prediction]
  - Rapach et al. (2010) [forecast combination]
  - Welch & Goyal (2008) [comprehensive review]
```

### Discussion Section
```
Multi-Agent Systems:
  - Hong & Page (2004) [diversity benefit]
  - Timmermann (2006) [forecast combination]
  - Recent LLM papers (2023-2024)

Systemic Risk:
  - Billio et al. (2012) [econometric measures]
  - Diebold & Yilmaz (2014) [network connectedness]
  - Battiston et al. (2012) [DebtRank]
```

---

## Advanced Search Techniques

### Boolean Operators
```
AND: Both terms must appear
OR: Either term can appear
NOT: Exclude term
"phrase": Exact phrase match
*: Wildcard (e.g., regime* finds regime, regimes, regime-switching)
```

### Time Filters
```
After 2020: Focus on recent methods
2008-2015: Great Financial Crisis period
1990-2000: Foundational papers
```

### Journal Filters (Top Tier)
```
Finance: JF, JFE, RFS, JFM
Economics: AER, QJE, JPE, Econometrica
Quantitative: MS, QF, JoF
```

### Author Searches
```
Hamilton, J.D. (regime switching)
Granger, C.W.J. (causality)
De Prado, M.L. (ML in finance)
Bekaert, G. (international finance)
Amihud, Y. (liquidity)
```

---

## Elicit.org Specific Tips

### 1. Start Broad, Then Narrow
```
Step 1: "regime switching financial markets"
Step 2: "Hamilton Markov regime switching stock returns"
Step 3: "Hamilton 1989 regime switching empirical"
```

### 2. Use Seed Papers
- Upload key paper (e.g., Hamilton 1989)
- Use "Find similar papers" function
- Explore citation network

### 3. Extract Data
- Use Elicit to extract:
  - Sample periods used
  - Performance metrics (Sharpe, etc.)
  - Statistical significance (p-values)
- Compare to EIMAS results

### 4. Generate Literature Matrix
| Method | Original Paper | Sample Period | Key Finding | EIMAS Validation |
|--------|---------------|---------------|-------------|------------------|
| GMM | Hamilton 1989 | 1952-1984 | 2-state better than linear | ✅ 3-state used |
| HRP | De Prado 2016 | 2005-2015 | Sharpe 0.78 vs 0.62 (MV) | ✅ Sharpe 1.85 |
| ... | ... | ... | ... | ... |

---

## Output Format for Paper

### Recommended Citation Style
```latex
\section{Methodology}

Our regime detection follows \citet{Hamilton1989} Markov-switching framework:
\begin{equation}
r_t | s_t \sim N(\mu_{s_t}, \sigma_{s_t}^2)
\end{equation}
where $s_t \in \{Bull, Neutral, Bear\}$ is the latent regime state.

We extend this by incorporating Shannon entropy \citep{Shannon1948} to quantify regime uncertainty:
\begin{equation}
H = -\sum_{i=1}^{3} p_i \log_2(p_i)
\end{equation}

[Continue with other methods...]
```

---

## Quality Checks

### Validate Each Module
- [ ] Found 3+ papers supporting method
- [ ] Found 1+ paper with similar application
- [ ] Found 1+ critique or limitation
- [ ] Compared sample periods (1980s vs 2020s)
- [ ] Compared performance metrics
- [ ] Identified novel contribution

### Red Flags to Address
- If no papers after 2015 → Method may be outdated
- If only 1-2 papers → Method may be obscure
- If conflicting results → Need to explain differences
- If commercial-only → Hard to validate academically

---

**Document Status**: Ready for Literature Review
**Estimated Search Time**: 20-30 hours (4 weeks)
**Expected Paper Count**: 40-60 papers (15-20 key, 25-40 supporting)
**Target Venues**: JFE, RFS, Journal of Financial Markets, Quantitative Finance

---

**Prepared by**: EIMAS Documentation System
**Date**: 2026-01-12
**Version**: v2.1.2
**Next Step**: Execute searches on Elicit.org and Google Scholar
