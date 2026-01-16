# EIMAS Web Application - v0 Prompt

## Project Overview
Create a modern, full-stack web application for **EIMAS (Economic Intelligence Multi-Agent System)** - an AI-powered economic analysis and market intelligence platform.

## Tech Stack
- **Frontend**: Next.js 14+ (App Router), TypeScript, Tailwind CSS, shadcn/ui
- **State Management**: React hooks, SWR for data fetching
- **Charts**: Recharts or Chart.js
- **API Integration**: REST API endpoints (already built with FastAPI)

## Application Structure

### 1. Main Dashboard (Home Page)
**Route**: `/`

**Features**:
- **Header**: App logo, navigation menu, real-time clock
- **Key Metrics Grid** (4 cards):
  - Portfolio Value (total value, P&L, P&L %)
  - Market Regime (regime type, trend, volatility, confidence %)
  - Consensus Signal (action: BUY/SELL/HOLD, conviction %)
  - Risk Level (LOW/MEDIUM/HIGH with color coding)
- **Live Signals Panel**:
  - Table of recent signals from multiple sources
  - Columns: Source, Action, Ticker, Conviction, Timestamp
  - Color-coded actions (green=BUY, red=SELL, yellow=HOLD)
- **Sector Rotation Chart**:
  - Economic cycle indicator
  - Overweight/Underweight sectors
  - Sector momentum visualization
- **Auto-refresh**: Update data every 60 seconds

**API Endpoints**:
- `GET /api/portfolio` - Portfolio summary
- `GET /api/regime?ticker=SPY` - Market regime
- `GET /api/signals?limit=10` - Recent signals
- `GET /api/risk` - Risk metrics
- `GET /api/sectors` - Sector rotation analysis

### 2. Analysis Report Viewer
**Route**: `/analysis`

**Features**:
- **Analysis Request Form**:
  - Text input for research question
  - Dropdown: Analysis level (Geopolitics, Monetary, Sector, Individual)
  - Dropdown: Research goal (Variable Selection, Forecasting, Causal Inference)
  - Checkbox: Use mock data (for testing)
  - Submit button → POST /analyze
- **Results Display**:
  - Analysis ID, status, timestamp
  - Final stance (BULLISH/BEARISH/NEUTRAL) with large badge
  - Confidence score (progress bar)
  - Executive summary (markdown formatted)
  - Top-Down analysis summary
  - Regime context (if available)
  - Stages completed (chips/badges)
  - Total duration
- **Historical Analysis List**:
  - List of past analyses with IDs
  - Click to view full result (GET /analyze/{analysis_id})

**API Endpoints**:
- `POST /analyze` - Run full pipeline analysis
- `GET /analyze/{analysis_id}` - Get analysis result

### 3. Portfolio Management
**Route**: `/portfolio`

**Features**:
- **Portfolio Summary** (top section):
  - Cash balance
  - Positions value
  - Total value
  - Total P&L (absolute + percentage)
- **Positions Table**:
  - Columns: Ticker, Quantity, Avg Cost, Current Price, Market Value, Unrealized P&L, P&L %
  - Color-coded P&L (green/red)
  - Click row to see details
- **Trade History**:
  - Recent trades table
  - Columns: Timestamp, Ticker, Side (BUY/SELL), Quantity, Price, Realized P&L
  - Date range filter (7/30/90 days)
- **Paper Trade Form**:
  - Input: Ticker, Side (BUY/SELL), Quantity
  - Submit button → POST /api/paper-trade
  - Success/error toast notification

**API Endpoints**:
- `GET /api/portfolio` - Portfolio summary
- `GET /api/portfolio/trades?days=30` - Trade history
- `POST /api/paper-trade` - Execute paper trade

### 4. Risk Analytics
**Route**: `/risk`

**Features**:
- **Risk Metrics Cards**:
  - VaR 95% / VaR 99%
  - CVaR (Conditional VaR)
  - Max Drawdown
  - Volatility
  - Sharpe Ratio
  - Beta
- **Portfolio Composition**:
  - Pie chart showing holdings by ticker
  - Percentage allocations
- **Correlation Matrix**:
  - Heatmap of asset correlations
  - Assets: SPY, TLT, GLD, QQQ (customizable)
  - Color scale: -1 (red) to +1 (green)
- **Correlation Alerts**:
  - List of unusual correlations
  - Alert type, message, severity

**API Endpoints**:
- `GET /api/risk` - Portfolio risk metrics
- `GET /api/correlation?assets=SPY,TLT,GLD,QQQ` - Correlation analysis

### 5. Portfolio Optimizer
**Route**: `/optimize`

**Features**:
- **Optimization Settings Form**:
  - Multi-select: Asset tickers (default: SPY, TLT, GLD, QQQ, IWM)
  - Radio buttons: Optimization method
    - Sharpe Ratio (default)
    - Minimum Variance
    - Risk Parity
  - Submit button → POST /api/optimize
- **Optimization Results**:
  - Table: Asset → Optimal Weight (%)
  - Expected Return (annualized)
  - Expected Volatility (annualized)
  - Sharpe Ratio
  - Convergence status
- **Weight Distribution Chart**:
  - Bar chart showing optimal weights per asset

**API Endpoints**:
- `POST /api/optimize` - Optimize portfolio

### 6. Settings & Configuration
**Route**: `/settings`

**Features**:
- **Account Settings**:
  - Account name (default: "default")
  - Initial cash balance
- **Analysis Settings**:
  - Default analysis level
  - Default research goal
  - Skip stages (checkboxes)
- **Data Refresh Settings**:
  - Auto-refresh interval (30/60/120 seconds)
  - Toggle auto-refresh on/off
- **Theme Toggle**:
  - Light/Dark mode switch

## Design Guidelines

### Color Scheme
- **Background**: Dark mode (#0d1117 dark, #ffffff light)
- **Cards**: #161b22 (dark), #f6f8fa (light)
- **Primary**: #58a6ff (blue)
- **Success**: #3fb950 (green)
- **Warning**: #d29922 (yellow)
- **Danger**: #f85149 (red)

### Components (shadcn/ui)
- Button, Card, Input, Select, Table
- Badge, Progress, Tabs, Dialog
- Toast (for notifications)
- Skeleton (for loading states)

### Typography
- Font: System fonts (-apple-system, BlinkMacSystemFont, 'Segoe UI')
- Headers: Bold, larger size
- Monospace: Code, IDs, numbers

### Layout
- Responsive grid (mobile-first)
- Navigation: Top navbar with links
- Max width: 1400px, centered
- Spacing: Consistent padding/margins

## Additional Requirements

1. **Loading States**: Show skeleton loaders while fetching data
2. **Error Handling**: Display error messages with toast notifications
3. **Empty States**: Show meaningful messages when no data available
4. **Accessibility**: ARIA labels, keyboard navigation
5. **Mobile Responsive**: All pages work on mobile devices
6. **Real-time Updates**: Use SWR with refresh intervals
7. **Type Safety**: Full TypeScript coverage
8. **Code Quality**: ESLint, Prettier configured

## File Structure (Next.js App Router)
```
app/
├── layout.tsx              # Root layout with navigation
├── page.tsx                # Dashboard (home)
├── analysis/
│   └── page.tsx            # Analysis report viewer
├── portfolio/
│   └── page.tsx            # Portfolio management
├── risk/
│   └── page.tsx            # Risk analytics
├── optimize/
│   └── page.tsx            # Portfolio optimizer
└── settings/
    └── page.tsx            # Settings & config

components/
├── ui/                     # shadcn/ui components
├── dashboard/
│   ├── MetricsGrid.tsx
│   ├── SignalsPanel.tsx
│   └── SectorRotation.tsx
├── portfolio/
│   ├── PositionsTable.tsx
│   ├── TradeHistory.tsx
│   └── TradeForm.tsx
└── shared/
    ├── Navbar.tsx
    ├── LoadingState.tsx
    └── ErrorState.tsx

lib/
├── api.ts                  # API client functions
├── types.ts                # TypeScript types
└── utils.ts                # Utility functions

hooks/
├── usePortfolio.ts
├── useSignals.ts
├── useRisk.ts
└── useAnalysis.ts
```

## API Base URL
Configure API base URL as environment variable:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Implementation Notes

1. **API Client**: Create a centralized API client in `lib/api.ts` with typed functions for each endpoint
2. **Data Fetching**: Use SWR hooks for automatic caching and revalidation
3. **Charts**: Use Recharts for all data visualizations
4. **Forms**: Use React Hook Form for form management
5. **Styling**: Use Tailwind CSS utility classes consistently
6. **Icons**: Use Lucide React icons

## Success Criteria

- All 6 pages fully functional
- Real-time data updates working
- All API endpoints integrated
- Mobile responsive design
- Dark/light mode support
- Error handling implemented
- Loading states for all async operations
- TypeScript with no type errors

---

**Generate a complete, production-ready Next.js application following the specifications above.**
