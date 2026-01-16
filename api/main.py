#!/usr/bin/env python3
"""
EIMAS API Server
=================
FastAPI 기반 실시간 대시보드 서버

Usage:
    uvicorn api.main:app --reload --port 8000

    또는:
    python api/main.py

Endpoints:
    GET  /                      - API 상태
    GET  /api/signals           - 최신 시그널
    GET  /api/portfolio         - 현재 포트폴리오
    GET  /api/risk              - 리스크 지표
    GET  /api/correlation       - 상관관계 행렬
    GET  /api/sectors           - 섹터 로테이션
    GET  /api/optimize          - 포트폴리오 최적화
    POST /api/paper-trade       - 페이퍼 트레이드 실행
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import json
import asyncio

# EIMAS 모듈
from lib.signal_pipeline import SignalPipeline, PortfolioGenerator
from lib.risk_manager import RiskManager
from lib.correlation_monitor import CorrelationMonitor, quick_correlation_check
from lib.sector_rotation import SectorRotationModel
from lib.portfolio_optimizer import PortfolioOptimizer
from lib.paper_trader import PaperTrader
from lib.regime_detector import RegimeDetector
from lib.trading_db import TradingDB

# Standard imports
import os
import glob


# ============================================================================
# App Configuration
# ============================================================================

app = FastAPI(
    title="EIMAS API",
    description="Economic Intelligence Multi-Agent System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from api.routes import analysis_router

# Include routers
app.include_router(analysis_router, prefix="")


# ============================================================================
# Request/Response Models
# ============================================================================

class TradeRequest(BaseModel):
    ticker: str
    side: str  # 'buy' or 'sell'
    quantity: float
    account: str = "default"


class OptimizeRequest(BaseModel):
    assets: List[str] = ["SPY", "TLT", "GLD", "QQQ", "IWM"]
    method: str = "sharpe"  # 'sharpe', 'min_var', 'risk_parity'


class SignalResponse(BaseModel):
    timestamp: str
    signals_count: int
    consensus_action: str
    consensus_conviction: float
    signals: List[Dict[str, Any]]


class PortfolioResponse(BaseModel):
    timestamp: str
    cash: float
    positions_value: float
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    positions: Dict[str, Any]


class ResearchRequest(BaseModel):
    query: str
    quick_mode: bool = True


# ============================================================================
# API Routes
# ============================================================================

@app.post("/api/research")
async def run_research(request: ResearchRequest):
    """
    Elicit 심층 리서치 실행
    
    사용자의 쿼리를 분석하고 멀티 에이전트 토론을 수행하여 결과를 반환합니다.
    (실제로는 통합 파이프라인을 실행하고 관련 섹션을 추출합니다)
    """
    from pipeline.runner import run_integrated_pipeline
    
    try:
        # 통합 파이프라인 실행
        # 주: 실제 구현에서는 query를 pipeline에 전달하는 매커니즘이 필요함.
        # 현재 구조상 query는 내부적으로 하드코딩 되어 있거나 ("Analyze current market...")
        # pipeline/signal/runner.py를 수정해야 함.
        # 우선은 파이프라인을 실행하고 결과 포맷을 반환.
        
        result = await run_integrated_pipeline(
            quick_mode=request.quick_mode,
            enable_realtime=False
        )
        
        return {
            "timestamp": result.timestamp,
            "query": request.query,
            "recommendation": result.final_recommendation,
            "confidence": result.confidence,
            "full_mode_position": result.full_mode_position,
            "reference_mode_position": result.reference_mode_position,
            "agreement": result.modes_agree,
            "devil_advocate": result.devils_advocate_arguments,
            "risk_score": result.risk_score,
            "regime": result.regime
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def root():
    """API 홈"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EIMAS API</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   background: #0d1117; color: #c9d1d9; padding: 40px; }
            h1 { color: #58a6ff; }
            a { color: #58a6ff; }
            .endpoint { background: #161b22; padding: 15px; margin: 10px 0; border-radius: 8px; }
            code { background: #21262d; padding: 2px 6px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>EIMAS API</h1>
        <p>Economic Intelligence Multi-Agent System</p>

        <h2>Endpoints</h2>
        <div class="endpoint">
            <strong>GET</strong> <code>/api/signals</code> - 최신 시그널 조회
        </div>
        <div class="endpoint">
            <strong>GET</strong> <code>/api/portfolio</code> - 현재 포트폴리오
        </div>
        <div class="endpoint">
            <strong>GET</strong> <code>/api/risk</code> - 리스크 지표
        </div>
        <div class="endpoint">
            <strong>GET</strong> <code>/api/correlation</code> - 상관관계 행렬
        </div>
        <div class="endpoint">
            <strong>GET</strong> <code>/api/sectors</code> - 섹터 로테이션
        </div>
        <div class="endpoint">
            <strong>GET</strong> <code>/api/regime</code> - 시장 레짐
        </div>
        <div class="endpoint">
            <strong>POST</strong> <code>/api/optimize</code> - 포트폴리오 최적화
        </div>
        <div class="endpoint">
            <strong>POST</strong> <code>/api/paper-trade</code> - 페이퍼 트레이드
        </div>

        <p><a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a></p>
    </body>
    </html>
    """
    return html


@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


# ============================================================================
# Signal Endpoints
# ============================================================================

@app.get("/api/signals")
async def get_signals(
    limit: int = Query(10, ge=1, le=100),
    source: Optional[str] = None,
):
    """시그널 조회"""
    try:
        pipeline = SignalPipeline()
        signals = pipeline.run()
        consensus = pipeline.get_consensus()

        signal_list = []
        for s in signals[:limit]:
            if source and s.source.value != source:
                continue
            signal_list.append({
                "id": s.id,
                "source": s.source.value,
                "action": s.action.value,
                "ticker": s.ticker,
                "conviction": s.conviction,
                "reasoning": s.reasoning,
                "timestamp": s.timestamp.isoformat(),
            })

        return {
            "timestamp": datetime.now().isoformat(),
            "signals_count": len(signals),
            "consensus_action": consensus["action"],
            "consensus_conviction": consensus["conviction"],
            "signals": signal_list,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals/history")
async def get_signal_history(days: int = Query(7, ge=1, le=90)):
    """시그널 히스토리"""
    try:
        db = TradingDB()
        signals = db.get_recent_signals(days=days)

        return {
            "timestamp": datetime.now().isoformat(),
            "days": days,
            "count": len(signals),
            "signals": [
                {
                    "id": s.id,
                    "source": s.source.value,
                    "action": s.action.value,
                    "ticker": s.ticker,
                    "conviction": s.conviction,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in signals
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Portfolio Endpoints
# ============================================================================

@app.get("/api/portfolio")
async def get_portfolio(account: str = Query("default")):
    """포트폴리오 조회"""
    try:
        trader = PaperTrader(account_name=account)
        summary = trader.get_portfolio_summary()

        positions = {}
        for ticker, pos in summary.positions.items():
            positions[ticker] = {
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "account": account,
            "cash": summary.cash,
            "positions_value": summary.positions_value,
            "total_value": summary.total_value,
            "total_pnl": summary.total_pnl,
            "total_pnl_pct": summary.total_pnl_pct,
            "positions": positions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/trades")
async def get_trades(
    account: str = Query("default"),
    days: int = Query(30, ge=1, le=365),
):
    """거래 내역"""
    try:
        trader = PaperTrader(account_name=account)
        trades = trader.get_trade_history(days=days)

        return {
            "timestamp": datetime.now().isoformat(),
            "account": account,
            "count": len(trades),
            "trades": [
                {
                    "id": t.id,
                    "ticker": t.ticker,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "realized_pnl": t.realized_pnl,
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in trades
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/paper-trade")
async def execute_paper_trade(request: TradeRequest):
    """페이퍼 트레이드 실행"""
    try:
        trader = PaperTrader(account_name=request.account)
        order = trader.execute_order(
            ticker=request.ticker,
            side=request.side,
            quantity=request.quantity,
        )

        return {
            "success": order.status.value == "filled",
            "order": {
                "ticker": order.ticker,
                "side": order.side.value,
                "quantity": order.quantity,
                "status": order.status.value,
                "filled_price": order.filled_price,
                "commission": order.commission,
            },
            "account_summary": {
                "cash": trader.cash,
                "positions": trader.positions,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Risk Endpoints
# ============================================================================

@app.get("/api/risk")
async def get_risk(account: str = Query("default")):
    """리스크 지표"""
    try:
        trader = PaperTrader(account_name=account)
        summary = trader.get_portfolio_summary()

        if not summary.positions:
            return {
                "timestamp": datetime.now().isoformat(),
                "risk_level": "LOW",
                "risk_score": 0.0,
                "message": "No positions to analyze",
            }

        # 포지션 비중 계산
        holdings = {}
        for ticker, pos in summary.positions.items():
            holdings[ticker] = pos.market_value / summary.total_value

        rm = RiskManager()
        risk = rm.calculate_portfolio_risk(holdings, summary.total_value)

        # Calculate risk_score (0-100 scale)
        # Based on annual volatility: 0% vol = 0 score, 100% vol = 100 score
        risk_score = min(100, max(0, risk.annual_vol * 100))

        return {
            "timestamp": datetime.now().isoformat(),
            "risk_level": risk.risk_level.value,
            "risk_score": round(risk_score, 1),
            "var_95": risk.var_95,
            "var_99": risk.var_99,
            "cvar_95": risk.cvar_95,
            "max_drawdown": risk.max_drawdown,
            "volatility": risk.annual_vol,
            "sharpe_ratio": risk.sharpe_estimate,
            "beta": risk.beta,
            "holdings": holdings,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analysis Endpoints
# ============================================================================

@app.get("/api/correlation")
async def get_correlation(
    assets: str = Query("SPY,TLT,GLD,QQQ"),
):
    """상관관계 분석"""
    try:
        asset_list = [a.strip() for a in assets.split(",")]
        cm = CorrelationMonitor(asset_list)
        result = cm.analyze()

        return {
            "timestamp": datetime.now().isoformat(),
            "regime": result.regime.value,
            "diversification": {
                "average_correlation": result.diversification.average_correlation,
                "max_correlation": result.diversification.max_correlation,
                "min_correlation": result.diversification.min_correlation,
                "diversification_ratio": result.diversification.diversification_ratio,
            },
            "pairs": [
                {
                    "asset1": p.asset1,
                    "asset2": p.asset2,
                    "name": p.name,
                    "current": p.current,
                    "normal": p.normal,
                    "state": p.state.value,
                }
                for p in result.pair_correlations
            ],
            "alerts": [
                {
                    "type": a.alert_type,
                    "message": a.message,
                    "severity": a.severity,
                }
                for a in result.alerts
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sectors")
async def get_sectors():
    """섹터 로테이션 분석"""
    try:
        model = SectorRotationModel()
        result = model.analyze()

        return {
            "timestamp": datetime.now().isoformat(),
            "economic_cycle": result.cycle.current_cycle.value,
            "cycle_confidence": result.cycle.confidence,
            "top_sectors": result.top_sectors,
            "bottom_sectors": result.bottom_sectors,
            "rotation_signal": {
                "overweight": result.rotation_signal.overweight,
                "underweight": result.rotation_signal.underweight,
            },
            "recommended_weights": result.recommended_weights,
            "sectors": [
                {
                    "ticker": s.ticker,
                    "name": s.name,
                    "momentum_1m": s.momentum_1m,
                    "momentum_3m": s.momentum_3m,
                    "relative_strength": s.relative_strength,
                    "rank": s.rank,
                    "signal": s.signal.value,
                }
                for s in result.sector_stats
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/regime")
async def get_regime(ticker: str = Query("SPY")):
    """시장 레짐 감지"""
    try:
        detector = RegimeDetector(ticker)
        result = detector.detect()

        return {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "regime": result.regime.value,
            "trend": result.trend_state.value,
            "volatility": result.volatility_state.value,
            "confidence": result.confidence,
            "metrics": result.indicators.to_dict() if hasattr(result.indicators, 'to_dict') else {},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize")
async def optimize_portfolio(request: OptimizeRequest):
    """포트폴리오 최적화"""
    try:
        optimizer = PortfolioOptimizer(request.assets)
        optimizer.fetch_data()

        if request.method == "sharpe":
            result = optimizer.optimize_sharpe()
        elif request.method == "min_var":
            result = optimizer.optimize_min_variance()
        elif request.method == "risk_parity":
            result = optimizer.optimize_risk_parity()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")

        return {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "assets": request.assets,
            "weights": result.weights,
            "expected_return": result.expected_return,
            "expected_volatility": result.expected_volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "converged": result.converged,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dashboard Endpoint
# ============================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """실시간 대시보드"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EIMAS Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0d1117;
                color: #c9d1d9;
                padding: 20px;
            }
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            h1 { color: #58a6ff; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .card {
                background: #161b22;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #30363d;
            }
            .card h2 { color: #8b949e; font-size: 14px; margin-bottom: 10px; text-transform: uppercase; }
            .card .value { font-size: 28px; font-weight: bold; }
            .green { color: #3fb950; }
            .red { color: #f85149; }
            .yellow { color: #d29922; }
            .signal-list { list-style: none; }
            .signal-list li { padding: 8px 0; border-bottom: 1px solid #21262d; }
            .loading { text-align: center; padding: 40px; color: #8b949e; }
            #refresh-btn {
                background: #238636;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
            }
            #refresh-btn:hover { background: #2ea043; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>EIMAS Dashboard</h1>
            <button id="refresh-btn" onclick="refreshData()">Refresh</button>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Portfolio Value</h2>
                <div class="value" id="portfolio-value">Loading...</div>
                <div id="portfolio-pnl"></div>
            </div>

            <div class="card">
                <h2>Market Regime</h2>
                <div class="value" id="regime">Loading...</div>
                <div id="regime-detail"></div>
            </div>

            <div class="card">
                <h2>Consensus Signal</h2>
                <div class="value" id="consensus">Loading...</div>
                <div id="conviction"></div>
            </div>

            <div class="card">
                <h2>Risk Level</h2>
                <div class="value" id="risk-level">Loading...</div>
                <div id="risk-metrics"></div>
            </div>

            <div class="card" style="grid-column: span 2;">
                <h2>Recent Signals</h2>
                <ul class="signal-list" id="signals-list">
                    <li class="loading">Loading...</li>
                </ul>
            </div>

            <div class="card" style="grid-column: span 2;">
                <h2>Sector Rotation</h2>
                <div id="sectors-info">Loading...</div>
            </div>
        </div>

        <script>
            async function fetchData(url) {
                try {
                    const response = await fetch(url);
                    return await response.json();
                } catch (e) {
                    console.error('Error fetching:', url, e);
                    return null;
                }
            }

            async function refreshData() {
                // Portfolio
                const portfolio = await fetchData('/api/portfolio');
                if (portfolio) {
                    document.getElementById('portfolio-value').textContent =
                        '$' + portfolio.total_value.toLocaleString(undefined, {minimumFractionDigits: 2});
                    const pnlClass = portfolio.total_pnl >= 0 ? 'green' : 'red';
                    document.getElementById('portfolio-pnl').innerHTML =
                        `<span class="${pnlClass}">$${portfolio.total_pnl.toFixed(2)} (${portfolio.total_pnl_pct.toFixed(2)}%)</span>`;
                }

                // Regime
                const regime = await fetchData('/api/regime');
                if (regime) {
                    document.getElementById('regime').textContent = regime.regime;
                    document.getElementById('regime-detail').textContent =
                        `${regime.trend} | ${regime.volatility} (${(regime.confidence * 100).toFixed(0)}%)`;
                }

                // Signals
                const signals = await fetchData('/api/signals');
                if (signals) {
                    document.getElementById('consensus').textContent = signals.consensus_action.toUpperCase();
                    document.getElementById('conviction').textContent =
                        `Conviction: ${(signals.consensus_conviction * 100).toFixed(0)}%`;

                    const signalsList = document.getElementById('signals-list');
                    signalsList.innerHTML = signals.signals.slice(0, 5).map(s =>
                        `<li><strong>${s.source}</strong>: ${s.action.toUpperCase()} ${s.ticker} (${s.conviction}%)</li>`
                    ).join('');
                }

                // Risk
                const risk = await fetchData('/api/risk');
                if (risk) {
                    const riskClass = risk.risk_level === 'low' ? 'green' :
                                     risk.risk_level === 'high' ? 'red' : 'yellow';
                    document.getElementById('risk-level').innerHTML =
                        `<span class="${riskClass}">${risk.risk_level.toUpperCase()}</span>`;
                    if (risk.var_95) {
                        document.getElementById('risk-metrics').textContent =
                            `VaR 95: ${(risk.var_95 * 100).toFixed(1)}% | Sharpe: ${risk.sharpe_ratio?.toFixed(2) || 'N/A'}`;
                    }
                }

                // Sectors
                const sectors = await fetchData('/api/sectors');
                if (sectors) {
                    document.getElementById('sectors-info').innerHTML = `
                        <div>Cycle: <strong>${sectors.economic_cycle}</strong> (${(sectors.cycle_confidence * 100).toFixed(0)}%)</div>
                        <div style="margin-top: 10px;">
                            <span class="green">Overweight: ${sectors.rotation_signal.overweight.join(', ')}</span><br>
                            <span class="red">Underweight: ${sectors.rotation_signal.underweight.join(', ')}</span>
                        </div>
                    `;
                }
            }

            // Initial load
            refreshData();

            // Auto refresh every 60 seconds
            setInterval(refreshData, 60000);
        </script>
    </body>
    </html>
    """
    return html


# ============================================================================
# WebSocket Endpoint for Real-time Updates
# ============================================================================

class ConnectionManager:
    """WebSocket 연결 관리자"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """모든 연결된 클라이언트에게 메시지 전송"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")


manager = ConnectionManager()


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """
    실시간 WebSocket 엔드포인트

    클라이언트는 이 엔드포인트에 연결하면 5초마다 최신 분석 결과를 수신합니다.

    메시지 형식:
    {
        "type": "update",
        "timestamp": "2026-01-12T12:00:00",
        "data": {
            "regime": {...},
            "signals": [...],
            "portfolio": {...},
            "risk": {...}
        }
    }
    """
    await manager.connect(websocket)

    try:
        # 연결 직후 환영 메시지
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to EIMAS real-time stream",
            "timestamp": datetime.now().isoformat()
        })

        # 주기적 업데이트 태스크
        async def send_updates():
            while True:
                try:
                    # 최신 데이터 수집
                    update_data = {}

                    # Regime
                    try:
                        detector = RegimeDetector("SPY")
                        regime_result = detector.detect()
                        update_data["regime"] = {
                            "regime": regime_result.regime.value,
                            "trend": regime_result.trend_state.value,
                            "volatility": regime_result.volatility_state.value,
                            "confidence": regime_result.confidence
                        }
                    except Exception as e:
                        print(f"Error fetching regime: {e}")
                        update_data["regime"] = None

                    # Signals
                    try:
                        pipeline = SignalPipeline()
                        signals = pipeline.run()
                        consensus = pipeline.get_consensus()
                        update_data["signals"] = {
                            "count": len(signals),
                            "consensus_action": consensus["action"],
                            "consensus_conviction": consensus["conviction"],
                            "recent": [
                                {
                                    "source": s.source.value,
                                    "action": s.action.value,
                                    "ticker": s.ticker,
                                    "conviction": s.conviction
                                }
                                for s in signals[:5]
                            ]
                        }
                    except Exception as e:
                        print(f"Error fetching signals: {e}")
                        update_data["signals"] = None

                    # Portfolio
                    try:
                        trader = PaperTrader()
                        portfolio_summary = trader.get_portfolio_summary()
                        update_data["portfolio"] = {
                            "total_value": portfolio_summary.total_value,
                            "cash": portfolio_summary.cash,
                            "pnl": portfolio_summary.total_pnl,
                            "pnl_pct": portfolio_summary.total_pnl_pct
                        }
                    except Exception as e:
                        print(f"Error fetching portfolio: {e}")
                        update_data["portfolio"] = None

                    # Risk
                    try:
                        trader = PaperTrader()
                        summary = trader.get_portfolio_summary()
                        if summary.positions:
                            holdings = {t: p.market_value / summary.total_value
                                      for t, p in summary.positions.items()}
                            rm = RiskManager()
                            risk = rm.calculate_portfolio_risk(holdings, summary.total_value)
                            update_data["risk"] = {
                                "level": risk.risk_level.value,
                                "var_95": risk.var_95,
                                "volatility": risk.annual_vol
                            }
                        else:
                            update_data["risk"] = None
                    except Exception as e:
                        print(f"Error fetching risk: {e}")
                        update_data["risk"] = None

                    # 업데이트 전송
                    await websocket.send_json({
                        "type": "update",
                        "timestamp": datetime.now().isoformat(),
                        "data": update_data
                    })

                    # 5초 대기
                    await asyncio.sleep(5)

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"Error in send_updates: {e}")
                    await asyncio.sleep(5)

        # 업데이트 태스크 시작
        await send_updates()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting EIMAS API Server...")
    print("Dashboard: http://localhost:8000/dashboard")
    print("API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
