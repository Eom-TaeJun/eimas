// API client for EIMAS FastAPI backend
const API_BASE_URL = "http://localhost:8000"

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
  return response.json()
}

export async function fetchRisk() {
  const response = await fetch(`${API_BASE_URL}/api/risk`)
  if (!response.ok) throw new Error("Failed to fetch risk")
  return response.json()
}
