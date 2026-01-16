# EIMAS Critical Paths Framework
# 시장 데이터 기반 경제 인사이트 추출 프레임워크

> **Core Thesis**: "All data comes from markets where money moves."
> 
> Market prices aggregate the knowledge of millions of participants, making them the most comprehensive real-time source of economic intelligence.

---

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Normal Market Paths (1-10)](#2-normal-market-paths)
3. [Crisis & Extreme Event Paths (11-17)](#3-crisis--extreme-event-paths)
4. [Implementation Guide](#4-implementation-guide)
5. [Korean Summary](#5-korean-summary-한국어-요약)

---

## 1. Theoretical Foundation

### 1.1 Why Market Data Contains "Everything"

This framework is grounded in **Friedrich Hayek's Price System Theory** and the **Efficient Market Hypothesis**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Hayek's Insight (1945):                                        │
│  "The price system is a mechanism for communicating             │
│   information... knowledge which is not given to anyone         │
│   in its totality."                                             │
├─────────────────────────────────────────────────────────────────┤
│  Translation to Our Framework:                                  │
│  • Gold/Silver prices → Inflation expectations                  │
│  • Credit spreads → Default risk & economic health              │
│  • Dollar index → Global capital flows                          │
│  • VIX → Uncertainty & risk appetite                            │
│  • Crypto → Speculative sentiment & fiat distrust               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 The Information Cascade

```
         Real Economy              Market Prices             AI Detection
              │                         │                         │
   Fed thinking ──────→ Bond yields ────────→ d_Exp_Rate changes
   Inflation fears ───→ Gold/Silver surge ──→ Ret_GLD, Ret_SLV
   Credit stress ─────→ HY spreads widen ───→ d_Spread_HighYield
   Risk appetite ─────→ VIX drops ──────────→ d_VIX
   Dollar confidence ─→ DXY moves ──────────→ Ret_Dollar_Idx
```

### 1.3 AI Advantage in Market Analysis

| Traditional Analysis | AI-Enabled Analysis |
|---------------------|---------------------|
| Monthly data releases | Real-time tick data |
| Single-variable focus | Multi-dimensional network |
| Backward-looking | Forward-looking (leading indicators) |
| Human cognitive limits | Process 1000s of signals simultaneously |
| Subjective interpretation | Evidence-based agent debate |

---

## 2. Normal Market Paths

### Path 1: Yield Curve → Recession Prediction

**Theoretical Basis**: Expectations Hypothesis + Term Premium Theory

The yield curve reflects market expectations of future short-term rates. When investors expect economic slowdown, they anticipate rate cuts, causing long-term yields to fall below short-term yields (inversion).

#### Information Flow
```
Current Fed Policy (short rates)
        │
        ▼
Market Expectations of Future Growth
        │
        ▼
Long-term Treasury Yields (10Y, 30Y)
        │
        ▼
Term Spread (10Y - 2Y)
        │
        ├── Positive & Steepening → Expansion expected
        ├── Flattening → Growth slowing
        └── Inverted → Recession signal (12-18 month lead)
```

#### What AI Should Monitor
- **Term Spread trajectory**: Not just the level, but the rate of change
- **Real yield curve** (TIPS): Separates growth expectations from inflation
- **Credit curve**: If corporate curves invert before Treasury, stress is building

#### Economic Insight Extracted
> "The market is pricing in X% probability of recession within 18 months based on yield curve dynamics"

---

### Path 2: Copper/Gold Ratio → Global Industrial Activity

**Theoretical Basis**: Dr. Copper Theory + Safe Haven Dynamics

Copper demand is driven by industrial production (construction, manufacturing, electronics). Gold demand is driven by uncertainty and inflation hedging. Their ratio reveals the market's assessment of global growth versus fear.

#### Information Flow
```
Global Manufacturing Activity
        │
        ▼
Copper Demand (Industrial Metal)
        │                              Uncertainty/Inflation Fear
        │                                      │
        ▼                                      ▼
   Copper Price ◄─────── Ratio ──────► Gold Price
        │
        ▼
   Copper/Gold Ratio
        │
        ├── Rising → Industrial optimism, growth expected
        ├── Stable → Equilibrium
        └── Falling → Fear dominates, defensive positioning
```

#### What AI Should Monitor
- **Copper/Gold ratio trend**: 20-day and 60-day moving averages
- **Divergence from equity markets**: If stocks rise but Cu/Au falls, warning signal
- **Comparison with PMI data**: Validate the signal against official data

#### Economic Insight Extracted
> "Global industrial sentiment is deteriorating/improving based on commodity price ratios, suggesting PMI will move toward X"

---

### Path 3: High Yield Spreads → Credit Cycle & Default Risk

**Theoretical Basis**: Merton's Structural Credit Model + Credit Cycle Theory

High yield bond spreads represent the market's real-time assessment of corporate default probability. Unlike credit ratings (backward-looking), spreads are forward-looking and adjust instantly.

#### Information Flow
```
Corporate Earnings Outlook
        │
        ├── Balance Sheet Strength
        ├── Cash Flow Stability
        └── Refinancing Risk
        │
        ▼
High Yield Bond Prices (HYG, JNK)
        │
        ▼
HY Spread over Treasuries
        │
        ├── Narrowing → Risk appetite, easy credit
        ├── Stable (300-400bp) → Normal conditions
        ├── Widening (500bp+) → Stress building
        └── Blow-out (800bp+) → Crisis conditions
```

#### Related Signals to Cross-Reference
- **Investment Grade spreads** (LQD): If IG widens before HY, flight to quality within credit
- **CCC-rated spreads**: Lowest quality bonds crack first
- **Leveraged loan prices**: Bank loan market stress

#### Economic Insight Extracted
> "Credit markets are signaling X% increase in default probability over the next 12 months, which historically precedes earnings downgrades by 2-3 quarters"

---

### Path 4: Dollar Smile → Global Risk Appetite Assessment

**Theoretical Basis**: Dollar Smile Theory (Stephen Jen)

The dollar strengthens in two scenarios: (1) US economy very strong (growth differentials), or (2) Global crisis (safe haven). It weakens when global growth is synchronized and risk appetite is high.

#### Information Flow
```
                    Dollar Strength
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Left Side          Bottom            Right Side
   of Smile          of Smile           of Smile
        │                 │                 │
   Global Crisis    Global Growth      US Outperformance
   Risk-Off Mode    Synchronized       Growth Differential
        │                 │                 │
        ▼                 ▼                 ▼
   Buy: Treasuries   Buy: EM, Europe   Buy: US Equities
   Sell: EM, Comm.   Sell: Dollar      Sell: Defensives
```

#### What AI Should Monitor
- **Dollar direction + VIX combination**: Strong dollar + high VIX = crisis mode
- **Dollar + equity correlation**: Positive = risk-off; Negative = growth differential
- **EM currency basket**: Confirms or contradicts dollar signal

#### Economic Insight Extracted
> "Dollar is strengthening due to [crisis fear / US outperformance], implying [risk-off allocation / overweight US vs. international]"

---

### Path 5: Sector Rotation → Business Cycle Phase Detection

**Theoretical Basis**: Business Cycle Investment Clock (Merrill Lynch)

Different sectors outperform at different phases of the economic cycle. By observing relative sector performance, we can infer where we are in the cycle.

#### Information Flow
```
Economic Cycle Phase
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Early Recovery    │  Mid-Cycle      │  Late Cycle       │
│  (XLF, XLI, XLY)   │  (XLK, XLC)     │  (XLE, Materials) │
│                    │                  │                   │
│  Expansion         │  Peak           │  Slowdown         │
│  begins            │                  │  begins           │
├───────────────────────────────────────────────────────────┤
│  Recession         │                                      │
│  (XLU, XLP, XLV)   │  ◄── Defensive sectors outperform   │
└───────────────────────────────────────────────────────────┘
```

#### What AI Should Monitor
- **Relative strength**: XLY/XLP ratio (Consumer Discretionary vs. Staples)
- **Financial sector**: Banks lead in recovery, lag in recession
- **Small cap vs. Large cap**: IWM/SPY ratio indicates risk appetite

#### Economic Insight Extracted
> "Sector rotation patterns suggest we are in [early/mid/late] cycle, with X months until probable regime change based on historical patterns"

---

### Path 6: Breakeven Inflation → Fed Policy Prediction

**Theoretical Basis**: Fisher Equation + Inflation Targeting Framework

Breakeven inflation (nominal Treasury yield minus TIPS yield) represents market expectations of future inflation. The Fed explicitly targets 2% inflation, so breakevens reveal whether markets expect Fed to tighten or ease.

#### Information Flow
```
Market Inflation Expectations
        │
        ▼
TIPS Yields vs. Nominal Treasury Yields
        │
        ▼
Breakeven Inflation Rate (5Y, 10Y)
        │
        ├── Rising above 2.5% → Fed likely to tighten
        ├── Stable around 2.0% → Fed on hold
        ├── Falling below 1.5% → Fed likely to ease
        └── Negative real yields → Extremely accommodative
```

#### Cross-Reference Signals
- **Commodity prices**: Oil, agricultural goods confirm or contradict
- **Wage growth proxies**: If breakevens rise without commodity push, wage-driven
- **Long vs. short breakevens**: 5Y vs. 10Y spread shows inflation persistence expectations

#### Economic Insight Extracted
> "Market expects inflation to average X% over the next 5 years, implying Fed will need to [raise/cut] rates by approximately Y basis points"

---

### Path 7: VIX Term Structure → Market Stress Forecast

**Theoretical Basis**: Volatility Clustering + Mean Reversion

The VIX term structure (front month vs. back months) reveals whether current volatility is expected to persist or revert. Backwardation (front > back) signals acute stress; Contango (front < back) signals complacency.

#### Information Flow
```
Current Market Uncertainty
        │
        ▼
VIX Spot Level
        │
        ▼
VIX Futures Term Structure
        │
        ├── Steep Contango (VIX < VIX3M < VIX6M)
        │   → Complacency, possible vol spike ahead
        │
        ├── Flat
        │   → Neutral, no strong expectation
        │
        └── Backwardation (VIX > VIX3M)
            → Crisis mode, but stress may peak soon
```

#### Additional Signals
- **VVIX** (volatility of VIX): Extreme readings signal turning points
- **Put/Call ratio**: Confirms sentiment
- **Skew index**: Tail risk perception

#### Economic Insight Extracted
> "Volatility structure indicates [complacency/acute stress/normalization], suggesting [prepare for vol spike / vol likely to decline / maintain current positioning]"

---

### Path 8: Emerging Market Flows → Global Liquidity Conditions

**Theoretical Basis**: Global Liquidity Cycle + Carry Trade Dynamics

When global liquidity is abundant (low rates, QE), capital flows to higher-yielding EM assets. When liquidity tightens, capital repatriates to developed markets. EM asset prices are thus a barometer of global liquidity.

#### Information Flow
```
Global Central Bank Policy
        │
        ▼
Global Liquidity Conditions
        │
        ▼
┌─────────────────────────────────────┐
│  EM Bonds (EMB)                     │
│  EM Equities (EEM, VWO)             │
│  EM Currencies (CEW)                │
└─────────────────────────────────────┘
        │
        ├── All rising → Liquidity abundant, risk-on
        ├── Bonds up, Equities down → Flight to EM yield
        ├── All falling → Liquidity tightening
        └── Currencies crashing → Dollar funding stress
```

#### What AI Should Monitor
- **EM vs. DM relative performance**: EEM/SPY ratio
- **Local currency vs. dollar bonds**: FX risk appetite
- **China proxies**: FXI, copper, AUD

#### Economic Insight Extracted
> "Global liquidity conditions are [tightening/easing] based on EM flows, which historically leads DM equity volatility by 2-4 weeks"

---

### Path 9: Gold/Silver Ratio → Inflation Type Detection

**Theoretical Basis**: Monetary Metal vs. Industrial Metal Dynamics

Gold is purely a monetary/store-of-value asset. Silver has both monetary AND industrial demand. The ratio reveals whether "inflation fear" is monetary (debasement) or demand-driven (growth).

#### Information Flow
```
Gold/Silver Ratio
        │
        ├── Rising (Gold outperforms, ratio > 80)
        │   → Monetary fear, deflation risk, safe haven demand
        │   → Likely environment: Crisis, stagflation
        │
        ├── Stable (ratio 60-80)
        │   → Balanced environment
        │
        └── Falling (Silver outperforms, ratio < 60)
            → Industrial demand strong, growth-driven inflation
            → Likely environment: Economic expansion
```

#### Cross-Reference
- **Platinum/Gold ratio**: Platinum = industrial; similar logic
- **Copper correlation**: If silver and copper both rally, industrial confirmation

#### Economic Insight Extracted
> "Precious metals are signaling [monetary/growth] inflation, suggesting [defensive/cyclical] positioning"

---

### Path 10: Bank Stocks → Credit Availability & Economic Health

**Theoretical Basis**: Financial Accelerator Theory (Bernanke)

Banks are the transmission mechanism of monetary policy. Their stock prices reflect: (1) loan demand, (2) credit quality, (3) net interest margin, (4) regulatory environment. Bank stock performance leads broader economic activity.

#### Information Flow
```
Economic Outlook
        │
        ├── Loan Demand (business investment)
        ├── Net Interest Margin (yield curve)
        ├── Credit Losses (default expectations)
        └── Capital Requirements (regulatory)
        │
        ▼
Bank Stock Performance (XLF, KRE, KBE)
        │
        ├── Banks leading market up → Credit expansion ahead
        ├── Banks lagging market → Credit tightening
        ├── Banks crashing → Financial stress, possible crisis
        └── Regional banks diverging → Localized stress
```

#### What AI Should Monitor
- **XLF vs. SPY relative strength**: Banking sector leadership
- **KRE (Regional) vs. KBE (Large)**: Size-based divergence
- **Bank stock implied volatility**: Stress in the financial system

#### Economic Insight Extracted
> "Banking sector performance suggests credit conditions will [expand/contract] over the next 2 quarters, implying [accelerating/slowing] economic growth"

---

## 3. Crisis & Extreme Event Paths

> **Key Principle**: In extreme situations—bubble collapses, market crashes, melt-ups—the **sequence** of how different assets crack or surge provides the most actionable intelligence.

---

### Path 11: The Crack Sequence — How Bubbles Unwind

**Theoretical Basis**: Minsky's Financial Instability Hypothesis + Liquidity Hierarchy

During bubble collapse, the most speculative and illiquid assets crack first. Capital flees in a predictable sequence based on liquidity and risk profile.

#### The Collapse Sequence (Historical Pattern)

```
STAGE 1: Cracks in the Periphery (2-6 months before main event)
┌─────────────────────────────────────────────────────────────────┐
│  First to Fall:                                                  │
│  • Meme stocks, SPACs (extreme speculation)                     │
│  • Crypto altcoins (lowest quality)                             │
│  • Unprofitable tech (no earnings support)                      │
│  • IPO index (new, unproven companies)                          │
│                                                                  │
│  Signal: ARKK, IPO ETF, small altcoins down 20-30%              │
│          while S&P 500 still near highs                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
STAGE 2: Small Caps Diverge (1-3 months before)
┌─────────────────────────────────────────────────────────────────┐
│  Next to Fall:                                                   │
│  • Russell 2000 (IWM) breaks down                               │
│  • Small cap growth vs. value divergence                        │
│  • Regional banks start underperforming                         │
│  • Biotech (speculative healthcare)                             │
│                                                                  │
│  Signal: IWM/SPY ratio falling while SPY makes new highs        │
│          "Narrowing breadth" — fewer stocks participating       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
STAGE 3: Credit Markets Warn (2-6 weeks before)
┌─────────────────────────────────────────────────────────────────┐
│  Credit Stress Emerges:                                          │
│  • High yield spreads start widening                            │
│  • Leveraged loan prices drop                                   │
│  • CCC-rated bonds crack                                        │
│  • Investment grade starts underperforming                      │
│                                                                  │
│  Signal: HYG down, LQD weak, but SPY still resilient            │
│          ★ This is the CRITICAL warning zone ★                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
STAGE 4: Large Cap Generals Fall (The Main Event)
┌─────────────────────────────────────────────────────────────────┐
│  Finally:                                                        │
│  • Mega-cap tech breaks down (AAPL, MSFT, NVDA)                 │
│  • S&P 500, Nasdaq break key support                            │
│  • Correlation spike — everything sells together                │
│  • VIX explosion above 30-40                                    │
│                                                                  │
│  By this point: Most damage already done to speculative assets  │
└─────────────────────────────────────────────────────────────────┘
```

#### Daily Monitoring Dashboard

| Indicator | Normal | Warning | Crisis |
|-----------|--------|---------|--------|
| ARKK vs SPY (30d) | Within ±5% | ARKK -10% vs SPY | ARKK -20%+ vs SPY |
| IWM/SPY Ratio | Stable or rising | Declining | Breaking down |
| % Stocks > 200MA | >60% | 40-60% | <40% |
| HY Spread Change (5d) | <20bp | 20-50bp | >50bp |
| New Highs - New Lows | Positive | Near zero | Deeply negative |

#### Economic Insight
> "Speculative assets have declined X% while large caps remain elevated. Historical pattern suggests Y% probability of broader correction within Z weeks."

---

### Path 12: Liquidity Stress Cascade — The Plumbing Breaks

**Theoretical Basis**: Brunnermeier's Liquidity Spiral + Funding Liquidity vs. Market Liquidity

Market crashes aren't just about valuation—they're about liquidity evaporating.

#### The Liquidity Warning Sequence

```
STAGE 1: Funding Markets Tighten
┌─────────────────────────────────────────────────────────────────┐
│  Early Liquidity Stress:                                         │
│  • SOFR-Fed Funds spread widens                                 │
│  • Commercial paper rates spike                                 │
│  • Repo market stress (overnight rates volatile)                │
│  • TED spread rising                                            │
│                                                                  │
│  What it means: Banks/institutions paying more for short-term   │
│                 funding — they're hoarding cash                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
STAGE 2: Market Liquidity Deteriorates
┌─────────────────────────────────────────────────────────────────┐
│  Trading Conditions Worsen:                                      │
│  • Bid-ask spreads widen (even in liquid assets)                │
│  • Market depth decreases (thin order books)                    │
│  • Flash crashes in individual names                            │
│  • ETF discounts to NAV appear                                  │
│                                                                  │
│  Signal: Bond ETFs (LQD, HYG, TLT) trading at discount to NAV   │
│          ★ This is a MAJOR warning sign ★                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
STAGE 3: Forced Selling Begins
┌─────────────────────────────────────────────────────────────────┐
│  Cascade Effects:                                                │
│  • Margin calls trigger selling                                 │
│  • Correlation goes to 1 (everything sells)                     │
│  • Even "safe" assets drop (Treasuries, Gold)                   │
│  • Dollar spikes (cash hoarding)                                │
│                                                                  │
│  The "sell everything" phase — only cash is safety              │
└─────────────────────────────────────────────────────────────────┘
```

#### Critical Liquidity Indicators

| Indicator | Source | What It Shows |
|-----------|--------|---------------|
| MOVE Index | Bond market vol | Treasury liquidity stress |
| FRA-OIS Spread | Interbank | Bank funding stress |
| Cross-currency basis | FX swaps | Dollar funding shortage |
| ETF Premium/Discount | Price vs NAV | Market maker stress |
| Bid-Ask on SPY | Level 2 data | Equity market liquidity |

#### Economic Insight
> "Funding markets showing stress with X spread widening. Historically, this precedes equity volatility spike by 3-10 trading days."

---

### Path 13: The Melt-Up Detection — Euphoria Before the Fall

**Theoretical Basis**: Kindleberger's Mania Phase + Greater Fool Theory

Bubbles don't die quietly—they often accelerate into a "blow-off top" with parabolic price action before collapsing.

#### Melt-Up Warning Signs

```
EUPHORIA INDICATORS
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Price Action:                                                   │
│  • Parabolic moves (>2 standard deviation daily gains)          │
│  • Gap-ups that don't fill                                      │
│  • Acceleration in trend (slope increasing)                     │
│  • Price far above all moving averages                          │
│                                                                  │
│  Volume & Flow:                                                  │
│  • Retail trading volume explodes                               │
│  • Options volume (esp. calls) at record                        │
│  • Crypto volumes spike                                         │
│  • Margin debt at all-time highs                                │
│                                                                  │
│  Sentiment:                                                      │
│  • AAII Bull/Bear spread extremely high                         │
│  • Put/Call ratio extremely low (<0.5)                          │
│  • VIX paradoxically low during rapid gains                     │
│  • "This time is different" narratives                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### The Blow-Off Top Pattern

```
                                    ╱╲
                                   ╱  ╲   ← Final spike
                                  ╱    ╲     (days to weeks)
                                 ╱      ╲
                           _____╱        ╲
                      ____╱               ╲
                 ____╱                     ╲
            ____╱                           ╲____
       ____╱                                     ╲____
  ____╱                                               ╲
 ╱                                                     
│                                                      
│  Phase 1:      Phase 2:         Phase 3:    Phase 4:
│  Stealth       Awareness        Mania       Blow-off
│                                                      
│  Signal: When Phase 3 accelerates into near-vertical
│          price action, the end is typically 2-8 weeks away
```

#### Quantitative Melt-Up Signals

| Metric | Normal | Elevated | Extreme (Melt-up) |
|--------|--------|----------|-------------------|
| 14-day RSI | 40-60 | 60-70 | >70 sustained |
| Price vs 50MA | ±5% | 5-10% | >15% above |
| Weekly % gains | 1-2% | 3-4% | >5% multiple weeks |
| Call/Put ratio | 0.8-1.0 | 1.0-1.5 | >2.0 |
| Crypto Fear/Greed | 40-60 | 70-80 | >90 |

#### Economic Insight
> "Market showing X of Y melt-up indicators. Historical blow-off tops lasted average of Z weeks after reaching this level of euphoria."

---

### Path 14: Correlation Breakdown → Regime Change Signal

**Theoretical Basis**: Correlation Asymmetry + Regime Switching Models

In normal times, assets have stable correlations based on fundamentals. During crises, correlations spike (everything moves together). Before crises, unusual correlation breakdowns occur.

#### Correlation Warning Signals

```
NORMAL REGIME
┌─────────────────────────────────────────────────────────────────┐
│  Expected Correlations:                                          │
│  • Stocks ↔ Bonds: Negative (flight to safety works)            │
│  • Gold ↔ Dollar: Negative (inverse relationship)               │
│  • VIX ↔ SPY: Strongly negative                                 │
│  • EM ↔ DM: Positive but <0.8                                   │
└─────────────────────────────────────────────────────────────────┘

CRISIS WARNING (Correlations Break)
┌─────────────────────────────────────────────────────────────────┐
│  Abnormal Patterns:                                              │
│  • Stocks AND Bonds both falling (2022 pattern)                 │
│  • Gold AND Dollar both rising (extreme fear)                   │
│  • VIX rising but not spiking (slow stress build)               │
│  • All correlations going to 1 (everything sells)               │
│                                                                  │
│  CRITICAL: When traditional hedges stop working,                │
│            it signals regime change imminent                    │
└─────────────────────────────────────────────────────────────────┘
```

#### Correlation Monitoring Table

| Correlation Pair | Normal | Warning | Crisis |
|------------------|--------|---------|--------|
| SPY-TLT | -0.3 to 0 | 0 to +0.3 | >+0.5 (both falling) |
| GLD-UUP | -0.2 to -0.5 | Near 0 | Positive (both rising) |
| EEM-SPY | +0.6 to +0.8 | >+0.9 | =1.0 (no diversification) |
| VIX-SPY | -0.7 to -0.8 | -0.5 to -0.6 | <-0.9 (panic) |

#### Economic Insight
> "Stock-bond correlation has turned positive for X consecutive days. This breakdown in traditional hedging suggests portfolio rebalancing stress and possible regime shift."

---

### Path 15: Capitulation Detection — When to Buy the Blood

**Theoretical Basis**: Behavioral Finance: Pain Threshold + Washout Theory

Markets bottom not on good news, but on exhaustion of sellers. Identifying capitulation (forced, panic selling) helps detect potential bottoms.

#### Capitulation Indicators

```
PANIC SELLING SIGNALS
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Volume & Breadth:                                               │
│  • Volume spike (3-5x average)                                  │
│  • >90% down volume day                                         │
│  • New lows explode (>1000 on NYSE)                             │
│  • Advance/Decline deeply negative multiple days                │
│                                                                  │
│  Volatility:                                                     │
│  • VIX spike above 40-50                                        │
│  • VIX term structure inverted (backwardation)                  │
│  • VVIX (vol of vol) extreme                                    │
│  • Put/Call ratio spikes above 1.5                              │
│                                                                  │
│  Credit:                                                         │
│  • HY spreads gap wider                                         │
│  • Investment grade selling too                                 │
│  • "Good" assets being sold (forced liquidation)                │
│                                                                  │
│  Sentiment:                                                      │
│  • AAII Bears >50%                                              │
│  • CNN Fear & Greed at "Extreme Fear"                           │
│  • "End of world" headlines                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

POTENTIAL BOTTOM SIGNALS
┌─────────────────────────────────────────────────────────────────┐
│  After capitulation, watch for:                                  │
│                                                                  │
│  • VIX declining from peak while price stabilizes               │
│  • Volume declining (sellers exhausted)                         │
│  • Positive divergence (price lower, RSI higher)                │
│  • Credit spreads stabilizing before equities                   │
│  • "Bad news, no new lows" pattern                              │
│  • Defensive sectors (XLU, XLP) start underperforming           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Capitulation Checklist

| Signal | Threshold | Weight |
|--------|-----------|--------|
| VIX > 40 | Yes/No | High |
| VIX Backwardation | Yes/No | High |
| Down Volume > 90% | Yes/No | Medium |
| Put/Call > 1.3 | Yes/No | Medium |
| AAII Bears > 50% | Yes/No | Medium |
| HY Spread +100bp in week | Yes/No | High |
| New Lows > 500 | Yes/No | Medium |

**Rule**: When 5+ signals trigger simultaneously → High probability capitulation

#### Economic Insight
> "X of Y capitulation indicators triggered. Historical analysis suggests Y% probability this represents a washout low, with average bounce of Z% in following 10 days."

---

### Path 16: Contagion Mapping — Where Fire Spreads

**Theoretical Basis**: Network Theory + Financial Contagion Models

Crises spread through specific channels: credit, liquidity, confidence. By monitoring the contagion path, we can anticipate which assets will be affected next.

#### Contagion Sequence Maps

```
TYPE A: CREDIT CONTAGION (2008 Style)
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Subprime/Weak Credit                                           │
│         │                                                        │
│         ▼                                                        │
│  Banks & Financials ────→ Counterparty fear                     │
│         │                                                        │
│         ▼                                                        │
│  Investment Grade Credit ────→ Funding markets                  │
│         │                                                        │
│         ▼                                                        │
│  Equities (all) ────→ Real economy                              │
│         │                                                        │
│         ▼                                                        │
│  Emerging Markets ────→ Commodity producers                     │
│                                                                  │
│  Timeline: Weeks to months                                       │
└─────────────────────────────────────────────────────────────────┘

TYPE B: LIQUIDITY CONTAGION (March 2020 Style)
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Risk Assets Sell-off                                           │
│         │                                                        │
│         ▼                                                        │
│  Margin Calls ────→ Forced selling                              │
│         │                                                        │
│         ▼                                                        │
│  "Safe" Assets Sold (Treasuries, Gold)                          │
│         │                                                        │
│         ▼                                                        │
│  Cash Hoarding ────→ Dollar spike                               │
│         │                                                        │
│         ▼                                                        │
│  Credit Freeze ────→ Fed intervention                           │
│                                                                  │
│  Timeline: Days to weeks (very fast)                            │
└─────────────────────────────────────────────────────────────────┘

TYPE C: CONFIDENCE CONTAGION (Sector Specific)
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Single Company Failure (e.g., SVB)                             │
│         │                                                        │
│         ▼                                                        │
│  Sector Peers (Regional Banks)                                  │
│         │                                                        │
│         ▼                                                        │
│  Related Sectors (Financials broadly)                           │
│         │                                                        │
│         ▼                                                        │
│  If contained: Stops here                                       │
│  If not: Spreads to credit → equities → economy                 │
│                                                                  │
│  Timeline: Days (rapid), containment critical                   │
└─────────────────────────────────────────────────────────────────┘
```

#### Contagion Early Warning Dashboard

| Asset | Role | What Breaking Means |
|-------|------|---------------------|
| KRE (Regional Banks) | Canary | Local credit stress |
| XLF (Financials) | Transmission | Systemic risk rising |
| HYG (High Yield) | Credit barometer | Funding conditions |
| EMB (EM Bonds) | Global liquidity | Dollar shortage |
| BTC | Speculative sentiment | Risk appetite collapse |
| TLT (Long Treasury) | Safe haven | If falls WITH stocks = liquidity crisis |

#### Economic Insight
> "Contagion pattern resembles Type [A/B/C]. Based on current spread, expect [sector/asset class] to be affected within [timeframe]. Recommended hedge: [specific action]."

---

### Path 17: Divergence Warnings — When Markets Disagree

**Theoretical Basis**: Intermarket Analysis + Confirmation Theory

Markets should tell a consistent story. When they diverge (one says "risk on" while another says "risk off"), it signals instability and imminent resolution—usually violent.

#### Critical Divergences to Monitor

```
DIVERGENCE TYPE 1: Equity vs Credit
┌─────────────────────────────────────────────────────────────────┐
│  Stocks Rising + Credit Spreads Widening                        │
│                                                                  │
│  • Equity: "Everything is fine"                                 │
│  • Credit: "Default risk is rising"                             │
│                                                                  │
│  Resolution: Usually credit is right                            │
│  Historical lag: Equity follows credit down in 2-8 weeks        │
│                                                                  │
│  Monitor: SPY trend vs HYG/TLT ratio                            │
└─────────────────────────────────────────────────────────────────┘

DIVERGENCE TYPE 2: Large Cap vs Small Cap
┌─────────────────────────────────────────────────────────────────┐
│  SPY New Highs + IWM Breaking Down                              │
│                                                                  │
│  • Large cap: Mega-tech driven rally                            │
│  • Small cap: Broader economy weakening                         │
│                                                                  │
│  Resolution: Either small catches up (bullish) or               │
│              large caps eventually fall (bearish)               │
│  Signal: Divergence >10% over 3 months is critical              │
│                                                                  │
│  Monitor: IWM/SPY ratio with 50-day MA                          │
└─────────────────────────────────────────────────────────────────┘

DIVERGENCE TYPE 3: Equity vs Volatility
┌─────────────────────────────────────────────────────────────────┐
│  SPY Making New Highs + VIX Not Making New Lows                 │
│                                                                  │
│  • Price: Optimism                                              │
│  • VIX: Hedging demand not decreasing                           │
│                                                                  │
│  Resolution: "Smart money" buying protection                    │
│  Warning: Often precedes sharp correction                       │
│                                                                  │
│  Monitor: SPY highs vs VIX relative level                       │
└─────────────────────────────────────────────────────────────────┘

DIVERGENCE TYPE 4: US vs International
┌─────────────────────────────────────────────────────────────────┐
│  US Rallying + Europe/EM Falling                                │
│                                                                  │
│  • US: Domestic strength or capital flight to US                │
│  • International: Global growth concerns                        │
│                                                                  │
│  Resolution: Either US is "safe haven" (continues) or           │
│              Global weakness eventually hits US                 │
│  Timeline: US typically lags global slowdown by 3-6 months      │
│                                                                  │
│  Monitor: SPY vs VEU (ex-US) ratio                              │
└─────────────────────────────────────────────────────────────────┘
```

#### Divergence Scoring System

| Divergence | Current State | Severity (1-5) | Historical Resolution |
|------------|---------------|----------------|----------------------|
| SPY vs HYG | Aligned / Diverging | - | Credit usually leads |
| SPY vs IWM | Aligned / Diverging | - | Breadth matters |
| SPY vs VIX | Aligned / Diverging | - | VIX warns |
| US vs World | Aligned / Diverging | - | Global matters |

**Total Divergence Score** (Max 20):
- 0-5: Markets aligned, trend likely continues
- 6-10: Minor divergences, monitor closely
- 11-15: Significant divergences, reduce risk
- 16-20: Major divergences, expect volatility

---

## 4. Implementation Guide

### 4.1 Data Sources Required

| Data Type | Source | Frequency | Variables |
|-----------|--------|-----------|-----------|
| Equities | Yahoo Finance | Daily/Real-time | SPY, QQQ, IWM, Sectors (XL*) |
| Fixed Income | Yahoo + FRED | Daily | TLT, HYG, LQD, Treasury yields |
| Commodities | Yahoo Finance | Daily | GLD, SLV, USO, DBA |
| Volatility | CBOE | Daily/Real-time | VIX, VIX3M, VVIX, Skew |
| Credit Spreads | FRED | Daily | BAA-AAA, HY OAS |
| Currency | Yahoo Finance | Daily | UUP, FXY, FXE, DXY |
| Crypto | Various | Real-time | BTC, ETH, Total market cap |
| Breadth | Exchange | Daily | A/D line, New Highs/Lows |
| Sentiment | Various | Daily/Weekly | AAII, Put/Call, Fear & Greed |

### 4.2 Signal Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  1. DATA COLLECTION                                              │
│     • Real-time price feeds                                     │
│     • Daily fundamental data                                    │
│     • Sentiment indicators                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. FEATURE ENGINEERING                                          │
│     • Returns (Ret_*)                                           │
│     • Changes (d_*)                                             │
│     • Ratios (Cu/Au, Au/Ag, IWM/SPY...)                        │
│     • Rolling correlations                                      │
│     • Technical indicators                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. PATH ANALYSIS                                                │
│     • Check each of 17 paths                                    │
│     • Calculate signal strength                                 │
│     • Identify regime                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. MULTI-AGENT INTERPRETATION                                   │
│     • Monetarist view                                           │
│     • Keynesian view                                            │
│     • Austrian view                                             │
│     • Technical view                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. CONSENSUS & OUTPUT                                           │
│     • Aggregate signals                                         │
│     • Generate recommendations                                  │
│     • Risk alerts                                               │
│     • Dashboard visualization                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Alert Priority Framework

| Priority | Condition | Action |
|----------|-----------|--------|
| **CRITICAL** | Capitulation signals + Contagion spreading | Immediate notification |
| **HIGH** | Credit markets diverging from equities | Daily monitoring |
| **MEDIUM** | Correlation breakdown detected | Weekly review |
| **LOW** | Single path signaling | Note in report |

---

## 5. Korean Summary (한국어 요약)

### 전체 프레임워크 개요

**핵심 철학**: "모든 데이터는 돈이 움직이는 시장에서 나온다"

시장 가격은 수백만 참여자의 지식을 집약하므로, 경제 상황을 파악하는 가장 포괄적인 실시간 소스입니다.

---

### 일반 시장 경로 (Path 1-10) 요약

| # | 경로명 | 선행 지표 | 예측 대상 |
|---|--------|----------|----------|
| 1 | 수익률 곡선 | 장단기 금리차 | 경기침체 (12-18개월 선행) |
| 2 | 구리/금 비율 | Cu/Au Ratio | 글로벌 산업 활동 |
| 3 | 하이일드 스프레드 | HY Spread | 부도 사이클, 기업 실적 |
| 4 | 달러 스마일 | DXY + VIX | 글로벌 리스크 체제 |
| 5 | 섹터 로테이션 | XLY/XLP, XLF | 경기 사이클 위치 |
| 6 | 손익분기 인플레이션 | 5Y Breakeven | Fed 정책 방향 |
| 7 | VIX 기간구조 | VIX Term Structure | 변동성 체제 |
| 8 | 신흥국 자금흐름 | EEM, EMB, CEW | 글로벌 유동성 |
| 9 | 금/은 비율 | Au/Ag Ratio | 인플레이션 유형 |
| 10 | 은행주 | XLF, KRE | 신용 가용성 |

---

### 위기 & 극단적 이벤트 경로 (Path 11-17) 요약

| # | 경로명 | 시간 범위 | 핵심 신호 |
|---|--------|----------|----------|
| 11 | 붕괴 순서 | 주-월 | ARKK, IWM, HYG 순서 |
| 12 | 유동성 캐스케이드 | 일-주 | 펀딩 스프레드, ETF 할인 |
| 13 | 멜트업 감지 | 일-주 | RSI, Call/Put, 변동성 압축 |
| 14 | 상관관계 붕괴 | 일 | 주식-채권 상관관계 전환 |
| 15 | 투매 감지 | 시간-일 | VIX 급등, 거래량, 폭 |
| 16 | 전염 매핑 | 일-주 | 섹터 간 전파 |
| 17 | 괴리 경고 | 주 | 시장 간 불일치 |

---

### 버블 붕괴의 순서 (가장 중요)

```
1단계: 가장 투기적 자산 먼저 붕괴 (2-6개월 전)
   → 밈주식, SPAC, 알트코인, 비수익 기술주
   → S&P 500은 아직 고점 근처

2단계: 소형주 이탈 (1-3개월 전)
   → Russell 2000 하락, 지방은행 약세
   → 시장 폭(breadth) 축소

3단계: 신용시장 경고 (2-6주 전)
   → 하이일드 스프레드 확대
   → ★ 이 시점이 가장 중요한 경고 구간 ★

4단계: 대형주 붕괴 (본 이벤트)
   → 메가캡 기술주 하락
   → 모든 자산 동시 매도
```

---

### 투매(Capitulation) 감지 체크리스트

| 신호 | 기준 | 비중 |
|------|------|------|
| VIX > 40 | Yes/No | 높음 |
| VIX 백워데이션 | Yes/No | 높음 |
| 하락 거래량 > 90% | Yes/No | 중간 |
| Put/Call > 1.3 | Yes/No | 중간 |
| AAII 약세 > 50% | Yes/No | 중간 |
| HY 스프레드 주간 +100bp | Yes/No | 높음 |
| 신저점 > 500개 | Yes/No | 중간 |

**규칙**: 5개 이상 신호 동시 발생 → 투매 고확률 → 바닥 근접 가능성

---

### 핵심 원칙

> **"크리티컬 패스의 순서를 아는 것이 타이밍의 핵심"**

- 투기 자산이 먼저 무너지고
- 소형주가 뒤따르며
- 신용시장이 경고하고
- 마지막으로 대형주가 붕괴

**AI가 이 순서를 실시간 모니터링하면, 본격적인 하락 전에 2-8주의 리드타임을 확보할 수 있습니다.**

---

## Appendix: Quick Reference Tables

### A. All 17 Paths Summary

| # | Path | Type | Lead Time | Primary Indicator |
|---|------|------|-----------|-------------------|
| 1 | Yield Curve | Normal | 12-18 months | Term Spread |
| 2 | Copper/Gold | Normal | 1-3 months | Cu/Au Ratio |
| 3 | HY Spreads | Normal | 2-3 quarters | HY OAS |
| 4 | Dollar Smile | Normal | Concurrent | DXY + VIX |
| 5 | Sector Rotation | Normal | 1-3 months | XLY/XLP |
| 6 | Breakevens | Normal | 1-6 months | 5Y Breakeven |
| 7 | VIX Structure | Normal | Days-weeks | VIX/VIX3M |
| 8 | EM Flows | Normal | 2-4 weeks | EEM, EMB |
| 9 | Gold/Silver | Normal | Concurrent | Au/Ag Ratio |
| 10 | Bank Stocks | Normal | 2 quarters | XLF, KRE |
| 11 | Crack Sequence | Crisis | 2-6 months | ARKK, IWM, HYG |
| 12 | Liquidity Cascade | Crisis | Days-weeks | SOFR, ETF discount |
| 13 | Melt-Up | Crisis | Days-weeks | RSI, Call/Put |
| 14 | Correlation Break | Crisis | Days | SPY-TLT correl |
| 15 | Capitulation | Crisis | Hours-days | VIX, Volume |
| 16 | Contagion | Crisis | Days-weeks | Sector sequence |
| 17 | Divergences | Crisis | Weeks | Cross-market |

### B. Key ETFs & Tickers

| Category | Tickers | Purpose |
|----------|---------|---------|
| Equities | SPY, QQQ, IWM, DIA | Market breadth |
| Sectors | XLF, XLE, XLK, XLV, XLY, XLP, XLU, XLI | Rotation |
| Bonds | TLT, IEF, LQD, HYG, TIP | Credit & rates |
| Commodities | GLD, SLV, USO, DBA, COPX | Inflation & growth |
| International | EEM, VEU, EFA, FXI | Global flows |
| Volatility | VIX, VIX3M, VVIX | Risk sentiment |
| Speculative | ARKK, IPO, BITW | Bubble detection |

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-25 | v1.0 | Initial framework documentation |

---

*This document is part of the EIMAS (Economic Intelligence Multi-Agent System) project.*
