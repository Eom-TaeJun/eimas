#!/usr/bin/env python3
"""
EIMAS Dashboard
===============
Streamlit-based web dashboard for signal monitoring and analysis.

Run with:
    streamlit run dashboard.py
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Page config
st.set_page_config(
    page_title="EIMAS Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .buy-signal { color: #28a745; font-weight: bold; }
    .sell-signal { color: #dc3545; font-weight: bold; }
    .hold-signal { color: #6c757d; font-weight: bold; }
    .reduce-signal { color: #fd7e14; font-weight: bold; }
    .hedge-signal { color: #17a2b8; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Data Loading Functions
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_signals():
    """Load current signals from pipeline"""
    try:
        from lib.signal_pipeline import SignalPipeline
        pipeline = SignalPipeline()
        signals = pipeline.run()
        consensus = pipeline.get_consensus()

        signal_list = []
        for s in signals:
            signal_list.append({
                'source': s.source.value,
                'action': s.action.value,
                'ticker': s.ticker,
                'conviction': s.conviction,
                'reasoning': s.reasoning,
                'timestamp': datetime.now()
            })

        return signal_list, consensus
    except Exception as e:
        st.error(f"Error loading signals: {e}")
        return [], {}


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_backtest(start_date: str, end_date: str):
    """Load backtest results"""
    try:
        from lib.backtest import BacktestEngine
        engine = BacktestEngine(initial_capital=100000)
        metrics = engine.run(
            ticker="SPY",
            start_date=start_date,
            end_date=end_date,
            holding_period=5,
            position_size=0.2
        )
        trades_df = engine.get_trades_df()
        equity_curve = engine.get_equity_curve_data()
        return metrics, trades_df, equity_curve
    except Exception as e:
        st.error(f"Error running backtest: {e}")
        return None, pd.DataFrame(), {}


@st.cache_data(ttl=300)
def load_market_data():
    """Load current market indicators"""
    try:
        from lib.market_indicators import MarketIndicatorsCollector
        collector = MarketIndicatorsCollector()
        result = collector.collect_all()
        return result
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        return None


@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_multi_asset():
    """Load multi-asset analysis"""
    try:
        from lib.multi_asset import MultiAssetAnalyzer
        analyzer = MultiAssetAnalyzer(include_stocks=True)
        report = analyzer.analyze_universe()
        df = analyzer.to_dataframe(report)
        return report, df
    except Exception as e:
        st.error(f"Error loading multi-asset data: {e}")
        return None, pd.DataFrame()


@st.cache_data(ttl=600)
def load_etf_analysis():
    """Load ETF analysis with detailed info"""
    try:
        from lib.etf_flow_analyzer import ETFFlowAnalyzer
        analyzer = ETFFlowAnalyzer(lookback_days=90)
        result = analyzer.run_full_analysis()
        return result
    except Exception as e:
        st.error(f"Error loading ETF data: {e}")
        return None


@st.cache_data(ttl=300)
def load_global_markets():
    """Load global market data"""
    try:
        import yfinance as yf

        symbols = {
            'DXY': 'DX-Y.NYB',
            'DAX': '^GDAXI',
            'FTSE': '^FTSE',
            'Nikkei': '^N225',
            'Shanghai': '000001.SS',
            'KOSPI': '^KS11',
            'Gold': 'GC=F',
            'WTI Oil': 'CL=F',
            'Copper': 'HG=F',
        }

        data = []
        for name, symbol in symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    data.append({
                        'Name': name,
                        'Symbol': symbol,
                        'Price': current,
                        'Change (%)': change
                    })
            except:
                pass

        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading global markets: {e}")
        return pd.DataFrame()


def load_latest_report():
    """Load the latest AI report"""
    import os
    import json
    from pathlib import Path

    report_dir = Path("outputs")
    if not report_dir.exists():
        return None, None

    # Find latest JSON report
    json_files = sorted(report_dir.glob("ai_report_*.json"), reverse=True)
    md_files = sorted(report_dir.glob("ai_report_*.md"), reverse=True)

    if not json_files or not md_files:
        return None, None

    try:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        with open(md_files[0], 'r', encoding='utf-8') as f:
            md_content = f.read()
        return json_data, md_content
    except:
        return None, None


def load_portfolio():
    """Load portfolio from paper trader"""
    try:
        from lib.paper_trader import PaperTrader
        trader = PaperTrader()
        positions = trader.get_all_positions()
        summary = trader.get_portfolio_summary()
        return trader, positions, summary
    except Exception as e:
        st.error(f"Error loading portfolio: {e}")
        return None, [], {}


@st.cache_data(ttl=600)
def load_sentiment_analysis():
    """Load sentiment and options analysis"""
    try:
        from lib.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer(['SPY', 'QQQ'])
        result = analyzer.analyze(include_options=True)
        return result
    except Exception as e:
        st.error(f"Error loading sentiment data: {e}")
        return None


# ============================================================================
# Dashboard Components
# ============================================================================

def render_header():
    """Render dashboard header"""
    st.markdown('<div class="main-header">📊 EIMAS Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Economic Intelligence Multi-Agent System**")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.divider()


def render_consensus_card(consensus: Dict):
    """Render consensus recommendation card"""
    action = consensus.get('action', 'N/A').upper()
    conviction = consensus.get('conviction', 0)

    # Color based on action
    color_map = {
        'BUY': '🟢',
        'SELL': '🔴',
        'HOLD': '⚪',
        'REDUCE': '🟠',
        'HEDGE': '🔵'
    }
    emoji = color_map.get(action, '⚪')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Consensus Action",
            value=f"{emoji} {action}",
            delta=None
        )

    with col2:
        st.metric(
            label="Conviction",
            value=f"{conviction:.0%}",
            delta=None
        )

    with col3:
        st.metric(
            label="Signal Count",
            value=consensus.get('signal_count', 0),
            delta=None
        )


def render_signals_table(signals: List[Dict]):
    """Render signals table"""
    if not signals:
        st.info("No signals available. Click 'Refresh Signals' to load.")
        return

    df = pd.DataFrame(signals)

    # Style the action column
    def style_action(val):
        colors = {
            'buy': 'background-color: #d4edda',
            'sell': 'background-color: #f8d7da',
            'hold': 'background-color: #e2e3e5',
            'reduce': 'background-color: #fff3cd',
            'hedge': 'background-color: #d1ecf1'
        }
        return colors.get(val, '')

    st.dataframe(
        df[['source', 'action', 'ticker', 'conviction', 'reasoning']],
        column_config={
            'source': st.column_config.TextColumn('Source', width=120),
            'action': st.column_config.TextColumn('Action', width=80),
            'ticker': st.column_config.TextColumn('Ticker', width=80),
            'conviction': st.column_config.ProgressColumn(
                'Conviction',
                min_value=0,
                max_value=1,
                format="%.0f%%"
            ),
            'reasoning': st.column_config.TextColumn('Reasoning', width=400)
        },
        hide_index=True,
        use_container_width=True
    )


def render_signal_distribution(signals: List[Dict]):
    """Render signal distribution chart"""
    if not signals:
        return

    df = pd.DataFrame(signals)
    action_counts = df['action'].value_counts()

    fig = px.pie(
        values=action_counts.values,
        names=action_counts.index,
        title="Signal Distribution",
        color=action_counts.index,
        color_discrete_map={
            'buy': '#28a745',
            'sell': '#dc3545',
            'hold': '#6c757d',
            'reduce': '#fd7e14',
            'hedge': '#17a2b8'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)


def render_conviction_chart(signals: List[Dict]):
    """Render conviction by source chart"""
    if not signals:
        return

    df = pd.DataFrame(signals)

    fig = px.bar(
        df,
        x='source',
        y='conviction',
        color='action',
        title="Conviction by Source",
        color_discrete_map={
            'buy': '#28a745',
            'sell': '#dc3545',
            'hold': '#6c757d',
            'reduce': '#fd7e14',
            'hedge': '#17a2b8'
        }
    )
    fig.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)


def render_backtest_results(metrics, trades_df, equity_curve):
    """Render backtest results"""
    if metrics is None:
        st.warning("No backtest data available")
        return

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{metrics.total_return:.1%}")
    with col2:
        st.metric("Win Rate", f"{metrics.win_rate:.1%}")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{metrics.max_drawdown:.1%}")

    # Equity curve
    if equity_curve and equity_curve.get('values'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=equity_curve['values'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_hline(
            y=equity_curve['initial'],
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital"
        )
        fig.update_layout(
            title="Equity Curve",
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Trade #"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Signal performance breakdown
    if metrics.metrics_by_signal:
        st.subheader("Performance by Signal Type")
        perf_data = []
        for signal, data in metrics.metrics_by_signal.items():
            perf_data.append({
                'Signal': signal.upper(),
                'Count': data['count'],
                'Win Rate': f"{data['win_rate']:.1%}",
                'Avg Return': f"{data['avg_return']:.2%}"
            })
        st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)


def render_market_indicators(market_data):
    """Render market indicators"""
    if market_data is None:
        st.warning("No market data available")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("VIX")
        if market_data.vix:
            st.metric("VIX Spot", f"{market_data.vix.vix:.1f}")
            st.metric("Structure", market_data.vix.structure.value if market_data.vix.structure else "N/A")
            st.metric("Percentile", f"{market_data.vix.vix_percentile:.0f}%")

    with col2:
        st.subheader("Crypto")
        if market_data.crypto:
            st.metric("Fear & Greed", market_data.crypto.fear_greed_value)
            st.metric("Level", market_data.crypto.fear_greed_level)
            st.metric("BTC 30d Change", f"{market_data.crypto.btc_30d_change:.1%}" if market_data.crypto.btc_30d_change else "N/A")

    with col3:
        st.subheader("Credit")
        if market_data.credit:
            st.metric("IG Spread", f"{market_data.credit.ig_spread:.0f} bps" if market_data.credit.ig_spread else "N/A")
            st.metric("HY Spread", f"{market_data.credit.hy_spread:.0f} bps" if market_data.credit.hy_spread else "N/A")


def render_multi_asset(report, df):
    """Render multi-asset analysis"""
    if report is None:
        st.warning("No multi-asset data available. Click 'Refresh' to load.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        regime_emoji = "🟢" if "Risk-On" in report.market_regime else ("🔴" if "Risk-Off" in report.market_regime else "⚪")
        st.metric("Market Regime", f"{regime_emoji} {report.market_regime}")

    with col2:
        st.metric("Risk Score", f"{report.risk_on_score:+.2f}")

    with col3:
        st.metric("Top Buys", len(report.top_buys))

    with col4:
        st.metric("Top Sells", len(report.top_sells))

    st.divider()

    # Sector Analysis
    st.subheader("Sector Rankings")

    if report.sector_signals:
        sector_data = []
        for s in report.sector_signals:
            sector_data.append({
                'Sector': s.sector,
                'ETF': s.etf,
                'Signal': s.signal.value.replace('_', ' ').title(),
                'Score': s.score,
                'Rotation': s.rotation_signal
            })

        sector_df = pd.DataFrame(sector_data)

        # Sector bar chart
        fig = px.bar(
            sector_df,
            x='Sector',
            y='Score',
            color='Signal',
            title="Sector Scores",
            color_discrete_map={
                'Strong Buy': '#28a745',
                'Buy': '#90EE90',
                'Neutral': '#6c757d',
                'Sell': '#FFB6C1',
                'Strong Sell': '#dc3545'
            }
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(sector_df, hide_index=True, use_container_width=True)

    st.divider()

    # Stock Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Stock Picks")
        if report.stock_signals:
            top_stocks = [s for s in report.stock_signals if s.score > 0][:10]
            if top_stocks:
                stock_data = []
                for s in top_stocks:
                    stock_data.append({
                        'Ticker': s.ticker,
                        'Name': s.name,
                        'Signal': s.signal.value.replace('_', ' ').title(),
                        'Score': s.score,
                        'RSI': s.rsi,
                        '20d %': s.change_20d
                    })
                st.dataframe(
                    pd.DataFrame(stock_data),
                    column_config={
                        'Score': st.column_config.NumberColumn(format="%.2f"),
                        'RSI': st.column_config.NumberColumn(format="%.0f"),
                        '20d %': st.column_config.NumberColumn(format="%.1%")
                    },
                    hide_index=True
                )

    with col2:
        st.subheader("Stocks to Avoid")
        if report.stock_signals:
            bottom_stocks = [s for s in report.stock_signals if s.score < 0][-10:]
            if bottom_stocks:
                stock_data = []
                for s in bottom_stocks:
                    stock_data.append({
                        'Ticker': s.ticker,
                        'Name': s.name,
                        'Signal': s.signal.value.replace('_', ' ').title(),
                        'Score': s.score,
                        'RSI': s.rsi,
                        '20d %': s.change_20d
                    })
                st.dataframe(
                    pd.DataFrame(stock_data),
                    column_config={
                        'Score': st.column_config.NumberColumn(format="%.2f"),
                        'RSI': st.column_config.NumberColumn(format="%.0f"),
                        '20d %': st.column_config.NumberColumn(format="%.1%")
                    },
                    hide_index=True
                )

    st.divider()

    # Other Assets
    st.subheader("Other Assets (Bonds, Commodities, Currencies)")
    if report.other_signals:
        other_data = []
        for s in report.other_signals:
            other_data.append({
                'Ticker': s.ticker,
                'Name': s.name,
                'Class': s.asset_class.value.title(),
                'Signal': s.signal.value.replace('_', ' ').title(),
                'Score': s.score,
                'RSI': s.rsi
            })
        st.dataframe(pd.DataFrame(other_data), hide_index=True, use_container_width=True)

    # Summary
    st.divider()
    st.subheader("Summary")
    st.info(report.summary)


def render_etf_details(etf_result):
    """Render ETF details with expense ratio, yield, etc."""
    if etf_result is None:
        st.warning("No ETF data available")
        return

    # Market Regime Summary
    regime = etf_result.get('market_regime', {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sentiment_emoji = "🟢" if regime.get('sentiment') == 'risk_on' else "🔴" if regime.get('sentiment') == 'risk_off' else "⚪"
        st.metric("Market Sentiment", f"{sentiment_emoji} {regime.get('sentiment', 'N/A').upper()}")

    with col2:
        st.metric("Style Rotation", regime.get('style_rotation', 'N/A').replace('_', ' ').title())

    with col3:
        st.metric("Risk Appetite", f"{regime.get('risk_appetite_score', 0):.0f}/100")

    with col4:
        st.metric("Market Breadth", f"{regime.get('breadth_score', 0):.0f}%")

    st.divider()

    # ETF Details Table
    st.subheader("📊 ETF Details")
    etf_details = etf_result.get('etf_details', [])

    if etf_details:
        df = pd.DataFrame(etf_details)

        # Format columns
        display_cols = ['ticker', 'name', 'category', 'current_price', 'expense_ratio',
                        'dividend_yield', 'total_assets', 'pe_ratio', 'beta', 'change_20d', 'rsi']

        available_cols = [c for c in display_cols if c in df.columns]
        df_display = df[available_cols].copy()

        # Rename for display
        df_display.columns = ['Ticker', 'Name', 'Category', 'Price', 'Expense %',
                             'Yield %', 'AUM ($B)', 'P/E', 'Beta', '20D %', 'RSI'][:len(available_cols)]

        st.dataframe(
            df_display,
            column_config={
                'Price': st.column_config.NumberColumn(format="$%.2f"),
                'Expense %': st.column_config.NumberColumn(format="%.2f%%"),
                'Yield %': st.column_config.NumberColumn(format="%.2f%%"),
                'AUM ($B)': st.column_config.NumberColumn(format="$%.1fB"),
                'P/E': st.column_config.NumberColumn(format="%.1f"),
                'Beta': st.column_config.NumberColumn(format="%.2f"),
                '20D %': st.column_config.NumberColumn(format="%.2f%%"),
                'RSI': st.column_config.NumberColumn(format="%.0f")
            },
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    # Sector Rotation
    st.subheader("🔄 Sector Rotation")
    sector = etf_result.get('sector_rotation', {})

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Leading Sectors:**")
        for s in sector.get('leading_sectors', []):
            st.write(f"  🟢 {s}")

    with col2:
        st.write("**Lagging Sectors:**")
        for s in sector.get('lagging_sectors', []):
            st.write(f"  🔴 {s}")

    st.info(f"**Cycle Phase:** {sector.get('cycle_phase', 'N/A').replace('_', ' ').title()}")

    # Key Spreads
    st.subheader("📈 Key Spreads (20D)")
    spread_col1, spread_col2, spread_col3, spread_col4 = st.columns(4)

    with spread_col1:
        gv = regime.get('growth_value_spread', 0)
        st.metric("Growth vs Value", f"{gv:+.2f}%",
                  delta="Growth Leading" if gv > 0 else "Value Leading" if gv < 0 else "Neutral")

    with spread_col2:
        ls = regime.get('large_small_spread', 0)
        st.metric("Large vs Small", f"{ls:+.2f}%",
                  delta="Large Cap" if ls > 0 else "Small Cap" if ls < 0 else "Neutral")

    with spread_col3:
        eb = regime.get('equity_bond_spread', 0)
        st.metric("Equity vs Bond", f"{eb:+.2f}%",
                  delta="Risk On" if eb > 0 else "Risk Off" if eb < 0 else "Neutral")

    with spread_col4:
        hy = regime.get('hy_treasury_spread', 0)
        st.metric("HY vs Treasury", f"{hy:+.2f}%",
                  delta="Credit Risk On" if hy > 0 else "Flight to Safety" if hy < 0 else "Neutral")


def render_global_markets(df):
    """Render global markets overview"""
    if df.empty:
        st.warning("No global market data available")
        return

    st.subheader("🌍 Global Markets Overview")

    # Create columns for different market categories
    indices = df[df['Name'].isin(['DAX', 'FTSE', 'Nikkei', 'Shanghai', 'KOSPI'])]
    commodities = df[df['Name'].isin(['Gold', 'WTI Oil', 'Copper'])]
    currencies = df[df['Name'].isin(['DXY'])]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**📊 Global Indices**")
        for _, row in indices.iterrows():
            change_color = "🟢" if row['Change (%)'] > 0 else "🔴" if row['Change (%)'] < 0 else "⚪"
            st.metric(
                row['Name'],
                f"{row['Price']:,.2f}",
                f"{row['Change (%)']:+.2f}%"
            )

    with col2:
        st.write("**⛏️ Commodities**")
        for _, row in commodities.iterrows():
            st.metric(
                row['Name'],
                f"${row['Price']:,.2f}",
                f"{row['Change (%)']:+.2f}%"
            )

    with col3:
        st.write("**💵 Dollar Index**")
        for _, row in currencies.iterrows():
            st.metric(
                row['Name'],
                f"{row['Price']:.2f}",
                f"{row['Change (%)']:+.2f}%"
            )

    # Chart
    st.divider()
    fig = px.bar(
        df,
        x='Name',
        y='Change (%)',
        color='Change (%)',
        color_continuous_scale=['red', 'gray', 'green'],
        color_continuous_midpoint=0,
        title="Daily Change (%)"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_sentiment_analysis(result):
    """Render sentiment and options analysis"""
    if result is None:
        st.warning("Sentiment data not available")
        return

    # Composite Sentiment
    st.subheader("📊 종합 센티먼트")

    composite = result.composite
    level = composite.level.value.replace('_', ' ').title()

    # Emoji based on level
    level_emoji = {
        'extreme_fear': '😱',
        'fear': '😰',
        'neutral': '😐',
        'greed': '😊',
        'extreme_greed': '🤑'
    }.get(composite.level.value, '😐')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("종합 점수", f"{level_emoji} {composite.score:.0f}")

    with col2:
        st.metric("센티먼트", level)

    with col3:
        if composite.fear_greed:
            st.metric("Fear & Greed", f"{composite.fear_greed.value}")

    with col4:
        st.metric("역발상 시그널", composite.contrarian_signal.upper())

    st.divider()

    # Options Analysis
    context = result.market_context

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 VIX 분석")
        if 'vix_structure' in context:
            vs = context['vix_structure']
            structure_emoji = "🟢" if vs.structure.value == 'contango' else "🔴" if vs.structure.value == 'backwardation' else "⚪"

            st.metric("VIX Spot", f"{vs.vix_spot:.1f}")
            st.metric("VIX 3M", f"{vs.vix_3m:.1f}")
            st.metric("구조", f"{structure_emoji} {vs.structure.value.title()}")
            st.metric("Contango Ratio", f"{vs.contango_ratio:+.1%}")
            st.metric("VIX Percentile", f"{vs.percentile:.0f}%")
            st.info(vs.interpretation)
        elif 'vix' in context:
            st.metric("VIX", f"{context['vix']:.1f}")

    with col2:
        st.subheader("📊 Put/Call Ratio")
        if 'put_call_ratio' in context:
            pcr = context['put_call_ratio']
            signal_emoji = "🟢" if pcr.signal == "BULLISH" else "🔴" if pcr.signal == "BEARISH" else "⚪"

            st.metric("P/C Ratio (Volume)", f"{signal_emoji} {pcr.put_call_ratio:.2f}")
            st.metric("P/C Ratio (OI)", f"{pcr.put_call_oi_ratio:.2f}")
            st.metric("Put Volume", f"{pcr.put_volume:,}")
            st.metric("Call Volume", f"{pcr.call_volume:,}")
            st.info(pcr.interpretation)
        else:
            st.info("옵션 데이터 없음")

    st.divider()

    # IV Percentile
    st.subheader("📉 IV Percentile")
    if 'iv_percentile' in context:
        iv = context['iv_percentile']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current IV", f"{iv.current_iv:.1f}%")
        with col2:
            iv_emoji = "🔴" if iv.iv_percentile > 80 else "🟢" if iv.iv_percentile < 20 else "⚪"
            st.metric("IV Percentile", f"{iv_emoji} {iv.iv_percentile:.0f}%")
        with col3:
            st.metric("HV 20D", f"{iv.hv_20:.1f}%")
        with col4:
            st.metric("IV/HV Ratio", f"{iv.iv_hv_ratio:.2f}")

        st.info(iv.interpretation)
    else:
        st.info("IV 데이터 없음")

    st.divider()

    # Signals
    st.subheader("📡 신호")
    if result.signals:
        for sig in result.signals:
            st.write(f"• {sig}")
    else:
        st.info("현재 특이 신호 없음")

    # Fear & Greed History
    st.divider()
    st.subheader("📈 Fear & Greed 추이")
    if composite.fear_greed:
        fg = composite.fear_greed
        fg_data = {
            '현재': fg.value,
            '전일': fg.previous_close,
            '1주전': fg.week_ago,
            '1달전': fg.month_ago,
            '1년전': fg.year_ago
        }

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(fg_data.keys()),
            y=list(fg_data.values()),
            marker_color=['#1f77b4', '#7f7f7f', '#7f7f7f', '#7f7f7f', '#7f7f7f']
        ))
        fig.add_hline(y=25, line_dash="dash", line_color="red", annotation_text="Extreme Fear")
        fig.add_hline(y=75, line_dash="dash", line_color="green", annotation_text="Extreme Greed")
        fig.update_layout(title="Fear & Greed Index", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)


def render_portfolio(trader, positions, summary):
    """Render portfolio tracking view"""
    if trader is None:
        st.warning("Portfolio tracker not available")
        return

    # Portfolio Summary
    st.subheader("💰 포트폴리오 요약")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_value = summary.get('total_equity', 0)
        st.metric("총 자산", f"${total_value:,.2f}")

    with col2:
        cash = summary.get('cash', 0)
        st.metric("현금", f"${cash:,.2f}")

    with col3:
        total_pnl = summary.get('unrealized_pnl', 0)
        st.metric("미실현 손익", f"${total_pnl:+,.2f}",
                  delta=f"{summary.get('unrealized_pnl_pct', 0):+.2f}%")

    with col4:
        realized_pnl = summary.get('realized_pnl', 0)
        st.metric("실현 손익", f"${realized_pnl:+,.2f}")

    st.divider()

    # Positions Table
    st.subheader("📊 보유 포지션")

    if positions:
        positions_data = []
        for pos in positions:
            pnl_pct = pos.unrealized_pnl_pct if hasattr(pos, 'unrealized_pnl_pct') else 0
            positions_data.append({
                'Ticker': pos.ticker,
                'Quantity': pos.quantity,
                'Avg Cost': pos.avg_cost,
                'Current Price': pos.current_price,
                'Market Value': pos.market_value,
                'P&L': pos.unrealized_pnl,
                'P&L %': pnl_pct
            })

        df = pd.DataFrame(positions_data)

        st.dataframe(
            df,
            column_config={
                'Avg Cost': st.column_config.NumberColumn(format="$%.2f"),
                'Current Price': st.column_config.NumberColumn(format="$%.2f"),
                'Market Value': st.column_config.NumberColumn(format="$%.2f"),
                'P&L': st.column_config.NumberColumn(format="$%.2f"),
                'P&L %': st.column_config.NumberColumn(format="%.2f%%")
            },
            hide_index=True,
            use_container_width=True
        )

        # Pie Chart - Portfolio Composition
        if len(positions_data) > 0:
            fig = px.pie(
                df,
                values='Market Value',
                names='Ticker',
                title="포트폴리오 구성"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("보유 중인 포지션이 없습니다.")

    st.divider()

    # Add Position Form
    st.subheader("➕ 포지션 추가/수정")

    col1, col2, col3 = st.columns(3)

    with col1:
        ticker = st.text_input("티커 (예: SPY)", key="add_ticker").upper()

    with col2:
        action = st.selectbox("주문 유형", ["매수", "매도"], key="add_action")

    with col3:
        quantity = st.number_input("수량", min_value=0.0, step=1.0, key="add_quantity")

    if st.button("주문 실행", key="execute_order"):
        if ticker and quantity > 0:
            try:
                side = "buy" if action == "매수" else "sell"
                result = trader.execute_order(ticker, side, quantity)
                if result:
                    st.success(f"✅ {action} 완료: {ticker} {quantity}주")
                    st.rerun()
                else:
                    st.error("주문 실행 실패")
            except Exception as e:
                st.error(f"주문 실패: {e}")
        else:
            st.warning("티커와 수량을 입력해주세요.")

    st.divider()

    # Trade History
    st.subheader("📜 거래 내역")
    try:
        trades = trader.get_trade_history(limit=20)
        if trades:
            trades_data = []
            for trade in trades:
                trades_data.append({
                    'Date': trade.get('filled_at', 'N/A'),
                    'Ticker': trade.get('ticker', 'N/A'),
                    'Side': trade.get('side', 'N/A').upper(),
                    'Quantity': trade.get('filled_quantity', 0),
                    'Price': trade.get('filled_price', 0)
                })
            st.dataframe(
                pd.DataFrame(trades_data),
                column_config={
                    'Price': st.column_config.NumberColumn(format="$%.2f")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("거래 내역이 없습니다.")
    except:
        st.info("거래 내역을 불러올 수 없습니다.")


def render_ai_report_summary(json_data, md_content):
    """Render AI report with historical comparison"""
    if json_data is None:
        st.warning("No AI report available. Generate one using the sidebar button.")
        return

    # Historical Comparison (if available)
    hist_comp = json_data.get('historical_comparison')
    if hist_comp and hist_comp.get('previous_timestamp'):
        st.subheader("📊 이전 리포트 대비 변화")

        significance = hist_comp.get('change_significance', 'MINOR')
        sig_emoji = {"MAJOR": "🔴", "MODERATE": "🟡", "MINOR": "🟢"}.get(significance, "⚪")

        st.info(f"**비교 대상:** {hist_comp.get('previous_timestamp', 'N/A')}  |  **변화 수준:** {sig_emoji} {significance}")

        # Key Changes
        key_changes = hist_comp.get('key_changes', [])
        if key_changes:
            st.write("**🔔 주요 변화:**")
            for change in key_changes:
                st.write(f"  - {change}")

        # Comparison Table
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "레짐",
                hist_comp.get('current_regime', 'N/A'),
                hist_comp.get('regime_change_direction', '')
            )
        with col2:
            st.metric(
                "신뢰도",
                f"{hist_comp.get('current_confidence', 0):.0f}%",
                f"{hist_comp.get('confidence_delta', 0):+.0f}%p"
            )
        with col3:
            st.metric(
                "리스크 점수",
                f"{hist_comp.get('current_risk_score', 0):.1f}",
                f"{hist_comp.get('risk_score_delta', 0):+.1f}"
            )
        with col4:
            st.metric(
                "VIX",
                f"{hist_comp.get('current_vix', 0):.1f}",
                f"{hist_comp.get('vix_delta', 0):+.1f}"
            )
        with col5:
            rec_changed = "🔄" if hist_comp.get('recommendation_changed') else "➡️"
            st.metric(
                "투자 권고",
                hist_comp.get('current_recommendation', 'N/A'),
                rec_changed
            )

        st.divider()

    # Report Content
    st.subheader("📝 Full Report")
    if md_content:
        st.markdown(md_content)
    else:
        st.info("Report content not available")


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.title("Controls")

    # Refresh button
    if st.sidebar.button("🔄 Refresh Signals", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()

    # Backtest settings
    st.sidebar.subheader("Backtest Settings")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

    run_backtest = st.sidebar.button("📈 Run Backtest", use_container_width=True)

    st.sidebar.divider()

    # Report generation
    st.sidebar.subheader("Reports")
    if st.sidebar.button("📝 Generate Report", use_container_width=True):
        with st.spinner("Generating report..."):
            try:
                from lib.debate_agent import run_full_pipeline
                filepath = run_full_pipeline()
                st.sidebar.success(f"Report saved: {filepath}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    return {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'run_backtest': run_backtest
    }


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application"""
    render_header()

    # Sidebar
    sidebar_state = render_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Signals",
        "📈 Backtest",
        "🌡️ Market Data",
        "🏢 Multi-Asset",
        "💹 ETF Analysis",
        "🌍 Global Markets",
        "🎯 Sentiment",
        "💼 Portfolio",
        "📋 AI Reports"
    ])

    # Tab 1: Signals
    with tab1:
        st.header("Current Signals")

        with st.spinner("Loading signals..."):
            signals, consensus = load_signals()

        if consensus:
            render_consensus_card(consensus)

        st.divider()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Signal Details")
            render_signals_table(signals)

        with col2:
            render_signal_distribution(signals)

        st.divider()
        render_conviction_chart(signals)

    # Tab 2: Backtest
    with tab2:
        st.header("Backtest Results")

        if sidebar_state['run_backtest']:
            with st.spinner("Running backtest..."):
                metrics, trades_df, equity_curve = load_backtest(
                    sidebar_state['start_date'],
                    sidebar_state['end_date']
                )
            render_backtest_results(metrics, trades_df, equity_curve)

            # Show trades table
            if not trades_df.empty:
                st.subheader("Trade History")
                st.dataframe(trades_df, hide_index=True, use_container_width=True)
        else:
            st.info("👈 Click 'Run Backtest' in the sidebar to see results")

    # Tab 3: Market Data
    with tab3:
        st.header("Market Indicators")

        with st.spinner("Loading market data..."):
            market_data = load_market_data()

        render_market_indicators(market_data)

    # Tab 4: Multi-Asset
    with tab4:
        st.header("Multi-Asset Analysis")

        if st.button("🔄 Refresh Multi-Asset Data"):
            st.cache_data.clear()

        with st.spinner("Loading multi-asset data..."):
            report, df = load_multi_asset()

        render_multi_asset(report, df)

    # Tab 5: ETF Analysis
    with tab5:
        st.header("ETF Analysis")

        if st.button("🔄 Refresh ETF Data", key="refresh_etf"):
            st.cache_data.clear()

        with st.spinner("Loading ETF data..."):
            etf_result = load_etf_analysis()

        render_etf_details(etf_result)

    # Tab 6: Global Markets
    with tab6:
        st.header("Global Markets")

        if st.button("🔄 Refresh Global Data", key="refresh_global"):
            st.cache_data.clear()

        with st.spinner("Loading global market data..."):
            global_df = load_global_markets()

        render_global_markets(global_df)

    # Tab 7: Sentiment
    with tab7:
        st.header("Sentiment & Options Analysis")

        if st.button("🔄 Refresh Sentiment Data", key="refresh_sentiment"):
            st.cache_data.clear()

        with st.spinner("Loading sentiment data..."):
            sentiment_result = load_sentiment_analysis()

        render_sentiment_analysis(sentiment_result)

    # Tab 8: Portfolio
    with tab8:
        st.header("Portfolio Tracker")

        trader, positions, summary = load_portfolio()
        render_portfolio(trader, positions, summary)

    # Tab 9: AI Reports
    with tab9:
        st.header("AI Generated Reports")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Refresh", key="refresh_report"):
                st.cache_data.clear()

        # Load latest report
        json_data, md_content = load_latest_report()
        render_ai_report_summary(json_data, md_content)

        # Report Archive
        st.divider()
        st.subheader("📚 Report Archive")

        import os
        report_dir = "outputs"

        if os.path.exists(report_dir):
            reports = sorted(
                [f for f in os.listdir(report_dir) if f.endswith('.md')],
                reverse=True
            )

            if reports:
                selected_report = st.selectbox("Select archived report", reports)

                if selected_report:
                    with st.expander("View archived report", expanded=False):
                        with open(os.path.join(report_dir, selected_report), 'r') as f:
                            content = f.read()
                        st.markdown(content)
            else:
                st.info("No archived reports found.")
        else:
            st.info("No reports directory found.")


if __name__ == "__main__":
    main()
