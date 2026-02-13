#!/usr/bin/env python3
"""
EIMAS Streamlit Dashboard
==========================
실시간 경제 인텔리전스 대시보드

실행:
    streamlit run frontend_streamlit/dashboard.py
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import time

# 페이지 설정
st.set_page_config(
    page_title="EIMAS Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .status-bullish { color: #3fb950; font-weight: bold; }
    .status-bearish { color: #f85149; font-weight: bold; }
    .status-neutral { color: #d29922; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data(ttl=5)  # 5초 캐시
def load_latest_analysis():
    """최신 EIMAS 분석 결과 로드"""
    try:
        outputs_dir = Path("outputs")
        files = list(outputs_dir.glob("eimas_*.json"))
        if not files:
            return None

        latest_file = max(files, key=lambda f: f.stat().st_mtime)

        with open(latest_file, 'r') as f:
            data = json.load(f)

        # Add metadata
        data['_file_name'] = latest_file.name
        data['_file_time'] = datetime.fromtimestamp(latest_file.stat().st_mtime)

        return data
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None


@st.cache_data(ttl=60)  # 1분 캐시
def load_historical_results(days=7):
    """과거 분석 결과 로드 (전일/전주 비교용)"""
    try:
        outputs_dir = Path("outputs")
        files = sorted(outputs_dir.glob("eimas_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

        results = []
        cutoff = datetime.now() - timedelta(days=days)

        for file in files[:50]:  # 최대 50개
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            if mtime < cutoff:
                continue

            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    data['_file_time'] = mtime
                    results.append(data)
            except:
                continue

        return results
    except Exception as e:
        st.error(f"히스토리 로드 실패: {e}")
        return []


# =============================================================================
# Helper Functions
# =============================================================================

def get_status_color(recommendation):
    """추천에 따른 색상 반환"""
    if "BULLISH" in recommendation or "BUY" in recommendation:
        return "green"
    elif "BEARISH" in recommendation or "SELL" in recommendation:
        return "red"
    else:
        return "orange"


def format_change(current, previous):
    """변화량 포맷 (전일/전주 대비)"""
    if previous is None or previous == 0:
        return "N/A"

    change = current - previous
    pct = (change / previous) * 100

    color = "green" if change > 0 else "red" if change < 0 else "gray"
    arrow = "↑" if change > 0 else "↓" if change < 0 else "→"

    return f":{color}[{arrow} {abs(pct):.1f}%]"


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("🎛️ EIMAS Control")

    # 자동 새로고침
    auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=False)

    if auto_refresh:
        st.info("자동 새로고침 활성화")
        time.sleep(5)
        st.rerun()

    # 수동 새로고침
    if st.button("🔃 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # 파이프라인 실행
    st.subheader("🚀 Pipeline")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Quick Run"):
            st.info("실행 중... (터미널에서 확인)")
            # Note: Streamlit에서 백그라운드 실행은 제한적

    with col2:
        if st.button("Full Run"):
            st.info("실행 중... (터미널에서 확인)")

    st.divider()

    # 시스템 정보
    st.caption("📊 EIMAS v2.2.5")
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# Main Dashboard
# =============================================================================

# Load data
data = load_latest_analysis()

if data is None:
    st.error("❌ 데이터를 찾을 수 없습니다. 먼저 `python main.py --quick`를 실행하세요.")
    st.stop()

# Header
st.title("📊 EIMAS Dashboard")
st.caption(f"Last updated: {data.get('timestamp', 'N/A')} | File: {data.get('_file_name', 'N/A')}")

# Quick metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    rec = data.get('final_recommendation', 'NEUTRAL')
    conf = data.get('confidence', 0.5) * 100
    st.metric(
        "📈 Recommendation",
        rec,
        f"{conf:.0f}% confidence",
        delta_color="off"
    )

with col2:
    risk_score = data.get('risk_score', 0)
    risk_level = data.get('risk_level', 'MEDIUM')
    st.metric(
        "⚠️ Risk Score",
        f"{risk_score:.1f}/100",
        risk_level,
        delta_color="inverse"
    )

with col3:
    regime = data.get('regime', {}).get('regime', 'Unknown')
    regime_conf = data.get('regime', {}).get('confidence', 0) * 100
    st.metric(
        "🌡️ Market Regime",
        regime,
        f"{regime_conf:.0f}% conf"
    )

with col4:
    modes_agree = data.get('modes_agree', False)
    full_pos = data.get('full_mode_position', 'N/A')
    st.metric(
        "🤝 AI Consensus",
        "Agree" if modes_agree else "Diverge",
        full_pos,
        delta_color="normal" if modes_agree else "inverse"
    )

st.divider()

# =============================================================================
# Tabs
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "📈 Analytics",
    "🤖 AI Reasoning",
    "⚠️ Risk",
    "📡 Signals",
    "📰 Events",
    "⚡ Realtime"
])

# --- TAB 1: Overview ---
with tab1:
    st.header("Market Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("FRED Summary")
        fred = data.get('fred_summary', {})

        st.write(f"**Net Liquidity**: ${fred.get('net_liquidity', 0):.1f}B")
        st.write(f"**Fed Funds Rate**: {fred.get('fed_funds', 0):.2f}%")
        st.write(f"**10Y Treasury**: {fred.get('treasury_10y', 0):.2f}%")
        st.write(f"**Unemployment**: {fred.get('unemployment', 0):.1f}%")
        st.write(f"**RRP**: ${fred.get('rrp', 0):.1f}B")

        st.caption("**Signals:**")
        for sig in fred.get('signals', [])[:3]:
            st.success(f"✓ {sig}")

        st.caption("**Warnings:**")
        for warn in fred.get('warnings', [])[:3]:
            st.warning(f"⚠️ {warn}")

    with col2:
        st.subheader("Portfolio Allocation")
        weights = data.get('portfolio_weights', {})

        if weights:
            fig = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.4
            )])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("포트폴리오 데이터 없음")

# --- TAB 2: Analytics ---
with tab2:
    st.header("Analytics & Charts")

    # GMM Probabilities
    st.subheader("GMM Regime Probabilities")
    probs = data.get('regime', {}).get('gmm_probabilities', {})

    if probs:
        fig = go.Figure(data=[
            go.Bar(
                x=list(probs.keys()),
                y=[v*100 for v in probs.values()],
                marker_color=['#3fb950', '#d29922', '#f85149']
            )
        ])
        fig.update_layout(
            yaxis_title="Probability (%)",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("GMM 확률 데이터 없음")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr_tickers = data.get('correlation_tickers', [])
    corr_matrix = data.get('correlation_matrix', [])

    if corr_tickers and corr_matrix:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_tickers,
            y=corr_tickers,
            colorscale='RdYlGn',
            zmid=0,
            text=[[f"{val:.2f}" for val in row] for row in corr_matrix],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("상관관계 데이터 없음")

    # Risk Breakdown
    st.subheader("Risk Score Breakdown")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Base Risk", f"{data.get('base_risk_score', 0):.1f}")
    col2.metric("Micro Adj", f"{data.get('microstructure_adjustment', 0):.1f}")
    col3.metric("Bubble Adj", f"{data.get('bubble_risk_adjustment', 0):.1f}")
    col4.metric("Final Risk", f"{data.get('risk_score', 0):.1f}")

# --- TAB 3: AI Reasoning ---
with tab3:
    st.header("AI Agent Reasoning")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Full Mode Analysis")
        st.metric("Position", data.get('full_mode_position', 'N/A'))

    with col2:
        st.subheader("Reference Mode Analysis")
        st.metric("Position", data.get('reference_mode_position', 'N/A'))

    st.divider()

    agree = data.get('modes_agree', False)
    if agree:
        st.success("✅ **Modes Agree** - Strong consensus")
    else:
        st.warning("⚠️ **Modes Diverge** - Conflicting signals")

    st.subheader("Devil's Advocate Arguments")
    devils = data.get('devils_advocate_arguments', [])
    if devils:
        for i, arg in enumerate(devils[:5], 1):
            st.warning(f"{i}. {arg}")
    else:
        st.info("No contrarian arguments")

# --- TAB 4: Risk ---
with tab4:
    st.header("Risk Analysis")

    # Risk Level
    risk_level = data.get('risk_level', 'MEDIUM')
    risk_score = data.get('risk_score', 50)

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Level: {risk_level}"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if risk_score > 70 else "orange" if risk_score > 30 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ]
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Warnings
    st.subheader("⚠️ Warnings")
    warnings = data.get('warnings', [])
    if warnings:
        for warn in warnings:
            st.warning(warn)
    else:
        st.success("No warnings")

# --- TAB 5: Signals ---
with tab5:
    st.header("Market Signals")

    if not result:
        st.warning("No analysis data available")
    else:
        # Section 1: Liquidity Analysis
        st.subheader("1. Liquidity Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            liquidity_signal = result.get('liquidity_signal', 'NEUTRAL')
            signal_color = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}
            st.metric(
                "Liquidity Signal",
                f"{signal_color.get(liquidity_signal, '⚪')} {liquidity_signal}",
                delta=None
            )

        with col2:
            fred_summary = result.get('fred_summary', {})
            net_liq = fred_summary.get('net_liquidity', 0)
            st.metric(
                "Net Liquidity (Fed)",
                f"${net_liq:,.1f}B" if net_liq else "N/A",
                delta=None
            )

        with col3:
            liq_regime = fred_summary.get('liquidity_regime', 'Unknown')
            st.metric("Liquidity Regime", liq_regime)

        # FRED Liquidity Details
        if fred_summary:
            st.markdown("**FRED Data Details:**")
            liquidity_data = {
                "RRP": f"${fred_summary.get('rrp', 0):.1f}B (Δ {fred_summary.get('rrp_delta', 0):.1f}B)",
                "TGA": f"${fred_summary.get('tga', 0):.1f}B (Δ {fred_summary.get('tga_delta', 0):.1f}B)",
                "Fed Assets": f"${fred_summary.get('fed_assets', 0):.2f}T (Δ {fred_summary.get('fed_assets_delta', 0):.1f}B)",
            }
            st.dataframe(pd.DataFrame(liquidity_data.items(), columns=["Metric", "Value"]), use_container_width=True, hide_index=True)

        st.divider()

        # Section 2: Genius Act Macro
        st.subheader("2. Genius Act Macro")
        col1, col2 = st.columns(2)

        with col1:
            genius_regime = result.get('genius_act_regime', 'NEUTRAL')
            regime_color = {"expansion": "🟢", "contraction": "🔴", "NEUTRAL": "🟡"}
            st.metric(
                "Digital M2 Regime",
                f"{regime_color.get(genius_regime, '⚪')} {genius_regime.upper()}",
                delta=None
            )

        with col2:
            extended_data = result.get('extended_data', {})
            digital_liq = extended_data.get('digital_liquidity', {})
            if digital_liq:
                st.metric(
                    "Stablecoin Market Cap",
                    f"${digital_liq.get('total_mcap', 0) / 1e9:,.2f}B" if digital_liq.get('total_mcap') else "N/A",
                    delta=None
                )

        # Genius Act Signals
        genius_signals = result.get('genius_act_signals', [])
        if genius_signals:
            st.markdown("**Genius Act Signals:**")
            for signal in genius_signals[:5]:
                st.write(f"• {signal}")

        st.divider()

        # Section 3: ETF Flow & Sector Rotation
        st.subheader("3. ETF Flow & Sector Rotation")
        etf_flow = result.get('etf_flow_result', {})

        if etf_flow:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rotation Signal", etf_flow.get('rotation_signal', 'N/A'))
            with col2:
                st.metric("Style Signal", etf_flow.get('style_signal', 'N/A'))

            # ETF Flow Details
            if 'details' in etf_flow and etf_flow['details']:
                st.markdown("**ETF Flow Details:**")
                st.json(etf_flow['details'])
        else:
            st.info("No ETF flow data available in this analysis run")

        st.divider()

        # Section 4: Volume Anomalies
        st.subheader("4. Volume Anomalies")
        volume_anomalies = result.get('volume_anomalies', [])

        if volume_anomalies and len(volume_anomalies) > 0:
            st.metric("Anomalies Detected", len(volume_anomalies))

            # Create DataFrame from volume anomalies
            anomaly_df = pd.DataFrame(volume_anomalies)
            display_cols = ['ticker', 'timestamp', 'volume_ratio', 'z_score', 'price_change_1d', 'anomaly_type', 'severity']
            available_cols = [col for col in display_cols if col in anomaly_df.columns]

            if available_cols:
                st.dataframe(
                    anomaly_df[available_cols].head(10),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No volume anomalies detected")

        st.divider()

        # Section 5: Extended Market Metrics
        st.subheader("5. Extended Market Metrics")

        if extended_data:
            # Crypto Fear & Greed
            col1, col2, col3 = st.columns(3)

            with col1:
                crypto_fng = extended_data.get('crypto_fng', {})
                if crypto_fng:
                    fng_value = crypto_fng.get('value', 50)
                    fng_class = crypto_fng.get('classification', 'Neutral')
                    st.metric(
                        "Crypto Fear & Greed",
                        f"{fng_value}",
                        delta=fng_class,
                        delta_color="off"
                    )

            with col2:
                defi_tvl = extended_data.get('defi_tvl', {})
                if defi_tvl:
                    total_tvl = defi_tvl.get('total_tvl', 0)
                    st.metric(
                        "DeFi TVL",
                        f"${total_tvl / 1e9:,.2f}B" if total_tvl else "N/A"
                    )

            with col3:
                news_sent = extended_data.get('news_sentiment', {})
                if news_sent:
                    sent_label = news_sent.get('label', 'Neutral')
                    sent_score = news_sent.get('score', 0)
                    st.metric(
                        "News Sentiment",
                        f"{sent_label}",
                        delta=f"Score: {sent_score:.2f}",
                        delta_color="off"
                    )

            # Market Depth (Short Interest & Institutional Holdings)
            market_depth = extended_data.get('market_depth', {})
            if market_depth:
                st.markdown("**Market Depth (Short Interest & Institutional Holdings):**")
                depth_data = []
                for key, value in market_depth.items():
                    if value is not None and value != 0:
                        ticker = key.split('_')[0]
                        metric_type = '_'.join(key.split('_')[1:])
                        depth_data.append({
                            "Ticker": ticker,
                            "Metric": metric_type.replace('_', ' ').title(),
                            "Value": f"{value * 100:.2f}%" if isinstance(value, (int, float)) and value < 1 else f"{value:.2f}"
                        })

                if depth_data:
                    st.dataframe(pd.DataFrame(depth_data), use_container_width=True, hide_index=True)

            # Put/Call Ratio
            put_call = extended_data.get('put_call_ratio', {})
            if put_call:
                st.markdown("**Put/Call Ratio:**")
                st.json(put_call)
        else:
            st.info("No extended market metrics available")

# --- TAB 6: Events ---
with tab6:
    st.header("Market Events & Changes")

    # Load historical data for comparison
    history = load_historical_results(days=7)

    if len(history) >= 2:
        current = history[0]
        prev_day = history[1] if len(history) > 1 else None
        prev_week = history[-1] if len(history) > 6 else None

        st.subheader("📊 Key Metrics Changes")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Current**")
            st.metric("Risk Score", f"{current.get('risk_score', 0):.1f}")
            st.metric("Confidence", f"{current.get('confidence', 0)*100:.0f}%")

        with col2:
            st.write("**vs Yesterday**")
            if prev_day:
                prev_risk = prev_day.get('risk_score', 0)
                prev_conf = prev_day.get('confidence', 0)
                st.metric("Risk", f"{current.get('risk_score', 0):.1f}",
                         f"{current.get('risk_score', 0) - prev_risk:.1f}")
                st.metric("Confidence", f"{current.get('confidence', 0)*100:.0f}%",
                         f"{(current.get('confidence', 0) - prev_conf)*100:.1f}%")
            else:
                st.info("No data")

        with col3:
            st.write("**vs Last Week**")
            if prev_week:
                prev_risk = prev_week.get('risk_score', 0)
                prev_conf = prev_week.get('confidence', 0)
                st.metric("Risk", f"{current.get('risk_score', 0):.1f}",
                         f"{current.get('risk_score', 0) - prev_risk:.1f}")
                st.metric("Confidence", f"{current.get('confidence', 0)*100:.0f}%",
                         f"{(current.get('confidence', 0) - prev_conf)*100:.1f}%")
            else:
                st.info("No data")

    st.divider()

    # Detected Events
    st.subheader("🎯 Detected Events")
    events = data.get('events_detected', [])

    if events:
        for event in events[:10]:
            event_type = event.get('type', 'Unknown')
            importance = event.get('importance', 'MEDIUM')
            description = event.get('description', 'No description')

            if importance == 'HIGH':
                st.error(f"🔴 **{event_type}**: {description}")
            elif importance == 'MEDIUM':
                st.warning(f"🟡 **{event_type}**: {description}")
            else:
                st.info(f"🔵 **{event_type}**: {description}")
    else:
        st.success("✓ No significant events detected")

    st.divider()

    # News Section (placeholder)
    st.subheader("📰 Market News")
    st.info("🚧 뉴스 수집 기능은 곧 추가됩니다...")

    # Sample news structure
    with st.expander("📰 Sample News"):
        st.write("""
        **[2026-02-13] Fed 금리 동결**
        - FOMC 회의에서 현 수준 유지 결정
        - 인플레이션 목표치 근접

        **[2026-02-12] 테크 주식 급등**
        - NASDAQ 2.3% 상승
        - AI 관련 종목 강세
        """)

# --- TAB 7: Realtime ---
with tab7:
    st.header("⚡ Realtime Monitor")

    st.info("실시간 데이터 스트리밍 (5초 갱신)")

    # Current time
    st.metric("Current Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Latest data timestamp
    data_time = data.get('timestamp', 'N/A')
    st.metric("Latest Data", data_time)

    # Time since last update
    try:
        data_dt = datetime.fromisoformat(data_time)
        age = (datetime.now() - data_dt).total_seconds() / 60
        st.metric("Data Age", f"{age:.1f} minutes")
    except:
        pass

    st.divider()

    # Live metrics
    st.subheader("📊 Live Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("VIX", f"{data.get('market_indicators', {}).get('vix_current', 0):.1f}")

    with col2:
        st.metric("Fed Liquidity", f"${data.get('fred_summary', {}).get('net_liquidity', 0):.1f}B")

    with col3:
        regime = data.get('regime', {}).get('regime', 'Unknown')
        st.metric("Regime", regime)

    st.divider()

    # Realtime signals (placeholder)
    st.subheader("⚡ Realtime Signals")
    realtime_signals = data.get('realtime_signals', [])

    if realtime_signals:
        for signal in realtime_signals[:5]:
            st.write(signal)
    else:
        st.info("실시간 신호 데이터 없음 (백그라운드 스트리밍 필요)")

    # Auto-refresh hint
    if not auto_refresh:
        st.info("💡 사이드바에서 'Auto-refresh'를 활성화하면 5초마다 자동 갱신됩니다.")

# =============================================================================
# Footer
# =============================================================================

st.divider()
st.caption("EIMAS Dashboard v1.0 | Economic Intelligence Multi-Agent System")
