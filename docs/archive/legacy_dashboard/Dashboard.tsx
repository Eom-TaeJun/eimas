'use client'

import { useState, useEffect } from 'react'
import { TrendingUp, TrendingDown, AlertCircle, Activity, Shield, Target } from 'lucide-react'

interface EIMASData {
  timestamp: string
  final_recommendation: string
  confidence: number
  risk_level: string
  risk_score: number
  regime: {
    regime: string
    trend: string
    volatility: string
  }
  full_mode_position: string
  reference_mode_position: string
  modes_agree: boolean
  market_data_count: number
  crypto_data_count: number
  warnings: string[]
}

export default function Dashboard() {
  const [data, setData] = useState<EIMASData | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const [isLive, setIsLive] = useState(true)

  useEffect(() => {
    // Initial fetch
    fetchLatestData()

    // Auto-refresh every 5 seconds
    const interval = setInterval(() => {
      fetchLatestData()
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const fetchLatestData = async () => {
    try {
      const response = await fetch('/api/latest')
      const jsonData = await response.json()
      setData(jsonData)
      setLastUpdate(new Date())
      setIsLive(true)
    } catch (error) {
      console.error('Failed to fetch data:', error)
      setIsLive(false)
    }
  }

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case 'BULLISH': return 'text-green-400 bg-green-500/10 border-green-500/20'
      case 'BEARISH': return 'text-red-400 bg-red-500/10 border-red-500/20'
      default: return 'text-gray-400 bg-gray-500/10 border-gray-500/20'
    }
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'LOW': return 'bg-green-500'
      case 'MEDIUM': return 'bg-yellow-500'
      case 'HIGH': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getRegimeIcon = (regime: string) => {
    if (regime.includes('Bull')) return <TrendingUp className="w-6 h-6 text-green-400" />
    if (regime.includes('Bear')) return <TrendingDown className="w-6 h-6 text-red-400" />
    return <Activity className="w-6 h-6 text-gray-400" />
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading EIMAS Dashboard...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold mb-2">EIMAS Real-Time Dashboard</h1>
            <p className="text-slate-400">Economic Intelligence Multi-Agent System</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isLive ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
              <span className="text-sm text-slate-400">{isLive ? 'Live' : 'Offline'}</span>
            </div>
            <div className="text-right">
              <div className="text-xs text-slate-500">Last Update</div>
              <div className="text-sm text-slate-300">{lastUpdate.toLocaleTimeString()}</div>
            </div>
          </div>
        </div>
      </header>

      {/* Status Banner */}
      <div className={`rounded-lg border-2 p-8 mb-8 ${getRecommendationColor(data.final_recommendation)}`}>
        <div className="grid grid-cols-3 gap-8">
          <div>
            <div className="text-sm text-slate-400 mb-2">Final Recommendation</div>
            <div className="text-5xl font-bold">{data.final_recommendation}</div>
          </div>
          <div>
            <div className="text-sm text-slate-400 mb-2">Confidence</div>
            <div className="text-3xl font-bold mb-2">{(data.confidence * 100).toFixed(1)}%</div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
                style={{ width: `${data.confidence * 100}%` }}
              ></div>
            </div>
          </div>
          <div>
            <div className="text-sm text-slate-400 mb-2">Risk Level</div>
            <div className="flex items-center gap-3">
              <span className={`px-4 py-2 rounded-lg text-2xl font-bold ${getRiskColor(data.risk_level)} text-white`}>
                {data.risk_level}
              </span>
              <span className="text-3xl font-bold text-slate-300">{data.risk_score.toFixed(1)}/100</span>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-3 gap-6 mb-8">
        {/* Market Regime */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center gap-3 mb-4">
            {getRegimeIcon(data.regime.regime)}
            <h3 className="text-lg font-semibold">Market Regime</h3>
          </div>
          <div className="space-y-2">
            <div className="text-2xl font-bold">{data.regime.regime}</div>
            <div className="text-sm text-slate-400">Trend: {data.regime.trend}</div>
            <div className="text-sm text-slate-400">Vol: {data.regime.volatility}</div>
          </div>
        </div>

        {/* Risk Score */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-6 h-6 text-blue-400" />
            <h3 className="text-lg font-semibold">Risk Score</h3>
          </div>
          <div className="relative pt-4">
            <svg className="w-32 h-32 mx-auto" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="40"
                fill="none"
                stroke="#334155"
                strokeWidth="8"
              />
              <circle
                cx="50"
                cy="50"
                r="40"
                fill="none"
                stroke="#3b82f6"
                strokeWidth="8"
                strokeDasharray={`${data.risk_score * 2.51} 251`}
                strokeLinecap="round"
                transform="rotate(-90 50 50)"
              />
              <text
                x="50"
                y="50"
                textAnchor="middle"
                dy="7"
                className="text-2xl font-bold fill-white"
              >
                {data.risk_score.toFixed(0)}
              </text>
            </svg>
          </div>
        </div>

        {/* AI Agreement */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center gap-3 mb-4">
            <Target className="w-6 h-6 text-purple-400" />
            <h3 className="text-lg font-semibold">AI Consensus</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-400">Full Mode</span>
              <span className={`font-bold ${getRecommendationColor(data.full_mode_position).split(' ')[0]}`}>
                {data.full_mode_position}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-400">Reference Mode</span>
              <span className={`font-bold ${getRecommendationColor(data.reference_mode_position).split(' ')[0]}`}>
                {data.reference_mode_position}
              </span>
            </div>
            <div className="pt-3 border-t border-slate-700">
              {data.modes_agree ? (
                <div className="flex items-center gap-2 text-green-400">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span className="text-sm font-semibold">Modes Agree</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-yellow-400">
                  <AlertCircle className="w-4 h-4" />
                  <span className="text-sm font-semibold">Dissent Detected</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Market Data Summary */}
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 mb-8">
        <h3 className="text-lg font-semibold mb-4">Data Collection Status</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-700/50 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Market Tickers</div>
            <div className="text-3xl font-bold">{data.market_data_count}</div>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Crypto Assets</div>
            <div className="text-3xl font-bold">{data.crypto_data_count}</div>
          </div>
        </div>
      </div>

      {/* Warnings */}
      {data.warnings && data.warnings.length > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-3">
            <AlertCircle className="w-5 h-5 text-yellow-400" />
            <h3 className="text-lg font-semibold text-yellow-400">Warnings</h3>
          </div>
          <ul className="space-y-2">
            {data.warnings.map((warning, idx) => (
              <li key={idx} className="text-sm text-slate-300">• {warning}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Footer */}
      <footer className="mt-8 text-center text-sm text-slate-500">
        <p>EIMAS v2.1.1 - Real-World Agent Edition</p>
        <p className="mt-1">Auto-refresh every 5 seconds</p>
      </footer>
    </div>
  )
}
