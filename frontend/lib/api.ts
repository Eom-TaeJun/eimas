// API client for EIMAS FastAPI backend
const API_BASE_URL = "http://localhost:8000"

// Fetch the latest integrated EIMAS analysis result
export async function fetchLatestAnalysis() {
  const response = await fetch(`${API_BASE_URL}/api/latest`)
  if (!response.ok) throw new Error("Failed to fetch latest analysis")
  return response.json()
}

// Legacy API functions (kept for backward compatibility)
export async function fetchPortfolio() {
  const response = await fetch(`${API_BASE_URL}/api/portfolio`)
  if (!response.ok) throw new Error("Failed to fetch portfolio")
  return response.json()
}

export async function fetchMarketRegime(ticker = "SPY") {
  const response = await fetch(`${API_BASE_URL}/api/regime?ticker=${ticker}`)
  if (!response.ok) throw new Error("Failed to fetch market regime")
  return response.json()
}

export async function fetchSignals(limit = 10) {
  const response = await fetch(`${API_BASE_URL}/api/signals?limit=${limit}`)
  if (!response.ok) throw new Error("Failed to fetch signals")
  const data = await response.json()
  return data.signals || []
}

export async function fetchRisk() {
  const response = await fetch(`${API_BASE_URL}/api/risk`)
  if (!response.ok) throw new Error("Failed to fetch risk")
  return response.json()
}
