# Step 4: EIMAS Risk Analytics Page

Create risk analytics dashboard with metrics and correlations.

## Route: /risk

### Components

1. **Risk Metrics Grid** (6 cards in 3x2)
   - VaR 95%: -$5,000
   - VaR 99%: -$8,000
   - CVaR: -$9,500
   - Max Drawdown: -12.5%
   - Volatility: 15.2%
   - Sharpe Ratio: 1.45

2. **Portfolio Composition Pie Chart**
   - Show holdings by ticker
   - Percentage allocations
   - Use Recharts

3. **Correlation Matrix Heatmap**
   - Assets: SPY, TLT, GLD, QQQ (customizable)
   - Color scale: -1 (red) to +1 (green)
   - Show correlation values in cells
   - Use Recharts or custom grid

4. **Correlation Alerts**
   - List of unusual correlations
   - Alert type, message, severity (Low/Medium/High)

### API Integration

```typescript
// GET /api/risk
{
  "var_95": -5000,
  "var_99": -8000,
  "cvar": -9500,
  "max_drawdown": -12.5,
  "volatility": 15.2,
  "sharpe_ratio": 1.45,
  "composition": {
    "SPY": 45,
    "TLT": 25,
    "GLD": 20,
    "QQQ": 10
  }
}

// GET /api/correlation?assets=SPY,TLT,GLD,QQQ
{
  "matrix": [
    [1.0, -0.5, 0.3, 0.8],
    [-0.5, 1.0, 0.1, -0.4],
    [0.3, 0.1, 1.0, 0.2],
    [0.8, -0.4, 0.2, 1.0]
  ],
  "assets": ["SPY", "TLT", "GLD", "QQQ"],
  "alerts": [
    {
      "type": "High Correlation",
      "message": "SPY and QQQ correlation: 0.8",
      "severity": "Medium"
    }
  ]
}
```

### Tech Stack
- Recharts for charts
- shadcn/ui: Card, Badge, Table
- Color gradient for heatmap

Generate with proper data visualization and responsive design.
