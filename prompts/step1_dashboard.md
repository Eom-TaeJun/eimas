# Step 1: EIMAS Dashboard Page

Create a Next.js 14 dashboard page for EIMAS (Economic Intelligence Multi-Agent System).

## Requirements

### Layout
- Dark theme (#0d1117 background, #161b22 cards)
- Top navbar: "EIMAS" logo, navigation links (Dashboard, Analysis, Portfolio)
- Responsive grid layout, max-width 1400px

### Components

1. **Metrics Grid (4 cards in 2x2)**
   - Portfolio Value: total value, P&L, P&L %
   - Market Regime: regime type, trend, volatility, confidence %
   - Consensus Signal: action (BUY/SELL/HOLD), conviction %
   - Risk Level: LOW/MEDIUM/HIGH with color (green/yellow/red)

2. **Live Signals Table**
   - Columns: Source, Action, Ticker, Conviction, Timestamp
   - Color-coded: BUY=green, SELL=red, HOLD=yellow
   - Show last 10 signals
   - Auto-refresh every 60 seconds

3. **Real-time Clock**
   - Display current time in header
   - Update every second

### API Integration (FastAPI backend at localhost:8000)

```typescript
// GET /api/portfolio
{
  "cash": 100000,
  "positions_value": 50000,
  "total_value": 150000,
  "total_pnl": 5000,
  "total_pnl_pct": 3.45
}

// GET /api/regime?ticker=SPY
{
  "regime": "Bull (Low Vol)",
  "trend": "Uptrend",
  "volatility": "Low",
  "confidence": 75
}

// GET /api/signals?limit=10
[
  {
    "source": "FULL_MODE",
    "action": "BUY",
    "ticker": "SPY",
    "conviction": 65,
    "timestamp": "2026-01-10T12:00:00"
  }
]

// GET /api/risk
{
  "risk_score": 45.2,
  "risk_level": "MEDIUM"
}
```

### Tech Stack
- Next.js 14 App Router
- TypeScript
- Tailwind CSS
- shadcn/ui components (Card, Badge, Table)
- SWR for data fetching with 60s refresh
- date-fns for time formatting

### File Structure
```
app/
  layout.tsx      # Root layout with navbar
  page.tsx        # Dashboard page
components/
  ui/             # shadcn/ui components
  MetricsGrid.tsx
  SignalsTable.tsx
lib/
  api.ts          # API client
  types.ts        # TypeScript types
```

Generate complete, production-ready code with loading states and error handling.
