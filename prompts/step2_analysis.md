# Step 2: EIMAS Analysis Page

Create an analysis page for EIMAS to run and view economic analysis reports.

## Route: /analysis

### Components

1. **Analysis Request Form**
   - Text input: Research question (multiline)
   - Dropdown: Analysis level (Geopolitics, Monetary, Sector, Individual)
   - Dropdown: Research goal (Variable Selection, Forecasting, Causal Inference)
   - Checkbox: Use mock data
   - Submit button (loading state during analysis)

2. **Results Display**
   - Analysis ID (monospace)
   - Status badge (PENDING/RUNNING/COMPLETED)
   - Final stance: Large badge (BULLISH=green, BEARISH=red, NEUTRAL=gray)
   - Confidence: Progress bar (0-100%)
   - Executive summary: Markdown formatted
   - Top-Down summary: Collapsible section
   - Regime context: If available
   - Stages completed: Chips/badges
   - Duration: Time elapsed

3. **Historical Analyses**
   - List of past analyses
   - Click to load result
   - Show: ID, timestamp, stance, confidence

### API Integration

```typescript
// POST /analyze
{
  "question": "What is the market outlook?",
  "analysis_level": "Monetary",
  "research_goal": "Forecasting",
  "use_mock": false
}

// Response
{
  "analysis_id": "abc123",
  "status": "completed",
  "final_stance": "BULLISH",
  "confidence": 65,
  "executive_summary": "Market shows...",
  "top_down_summary": "...",
  "regime_context": {...},
  "stages_completed": ["data", "analysis", "debate"],
  "duration": 45.2
}

// GET /analyze/{analysis_id}
// Returns same structure as above
```

### Tech Stack
- Next.js 14 App Router
- React Hook Form for form management
- Markdown renderer (react-markdown)
- shadcn/ui: Form, Input, Select, Checkbox, Button, Badge, Progress, Collapsible

Generate page with proper loading states and error handling.
