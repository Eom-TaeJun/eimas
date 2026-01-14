# Step 3: EIMAS Portfolio Page

Create portfolio management page for paper trading.

## Route: /portfolio

### Components

1. **Portfolio Summary Card**
   - Cash balance: $100,000
   - Positions value: $50,000
   - Total value: $150,000
   - Total P&L: $5,000 (+3.45%) with color (green/red)

2. **Positions Table**
   - Columns: Ticker, Quantity, Avg Cost, Current Price, Market Value, Unrealized P&L, P&L %
   - Color-coded P&L (green positive, red negative)
   - Sortable columns
   - Click row for details

3. **Trade History**
   - Date range filter: 7/30/90 days
   - Columns: Timestamp, Ticker, Side (BUY/SELL), Quantity, Price, Realized P&L
   - Color-coded side: BUY=green, SELL=red

4. **Paper Trade Form**
   - Input: Ticker (uppercase)
   - Select: Side (BUY/SELL)
   - Input: Quantity (number)
   - Submit button
   - Success/error toast notifications

### API Integration

```typescript
// GET /api/portfolio
{
  "cash": 100000,
  "positions_value": 50000,
  "total_value": 150000,
  "total_pnl": 5000,
  "total_pnl_pct": 3.45,
  "positions": [
    {
      "ticker": "SPY",
      "quantity": 100,
      "avg_cost": 450.0,
      "current_price": 455.0,
      "market_value": 45500,
      "unrealized_pnl": 500,
      "unrealized_pnl_pct": 1.11
    }
  ]
}

// GET /api/portfolio/trades?days=30
[
  {
    "timestamp": "2026-01-10T12:00:00",
    "ticker": "SPY",
    "side": "BUY",
    "quantity": 100,
    "price": 450.0,
    "realized_pnl": 0
  }
]

// POST /api/paper-trade
{
  "ticker": "SPY",
  "side": "BUY",
  "quantity": 10
}
```

### Tech Stack
- shadcn/ui: Table, Card, Input, Select, Button, Toast
- React Hook Form
- SWR for data fetching

Generate with form validation and proper error handling.
