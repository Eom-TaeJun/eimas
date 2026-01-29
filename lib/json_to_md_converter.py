#!/usr/bin/env python3
"""
EIMAS JSON to Markdown Converter
=================================
eimas_*.json 파일을 읽기 쉬운 마크다운 리포트로 변환

Usage:
    python lib/json_to_md_converter.py                    # 최신 파일 변환
    python lib/json_to_md_converter.py eimas_20260129.json  # 특정 파일 변환
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class EIMASMarkdownConverter:
    """EIMAS JSON을 가독성 높은 마크다운으로 변환"""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.timestamp = data.get('timestamp', datetime.now().isoformat())
    
    def convert(self) -> str:
        """전체 마크다운 생성"""
        sections = [
            self._header(),
            self._executive_summary(),
            self._macro_indicators(),
            self._market_regime(),
            self._ai_debate_results(),
            self._ark_analysis(),
            self._technical_analysis(),
            self._sentiment_analysis(),
            self._recommendation(),
            self._footer()
        ]
        return "\n\n---\n\n".join(filter(None, sections))
    
    def _header(self) -> str:
        ts = self.timestamp[:19].replace('T', ' ')
        return f"""# 📊 EIMAS 분석 리포트

**생성 시간:** {ts}  
**시스템:** EIMAS (Economic Intelligence Multi-Agent System)"""

    def _executive_summary(self) -> str:
        regime = self.data.get('regime', {})
        risk = self.data.get('risk_score', 0)
        rec = self.data.get('final_recommendation', 'N/A')
        conf = self.data.get('confidence', 0) * 100
        
        regime_name = regime.get('regime', 'Unknown')
        regime_desc = regime.get('description', '')
        
        return f"""## 📋 Executive Summary

**시장 레짐:** {regime_name}  
**리스크 점수:** {risk:.1f}/100  
**최종 권고:** {rec} (신뢰도: {conf:.0f}%)  

> {regime_desc}"""

    def _macro_indicators(self) -> str:
        fred = self.data.get('fred_summary', {})
        if not fred:
            return ""
        
        lines = [
            "## 🏛️ 거시경제 지표 (FRED)",
            "",
            "### 금리",
            f"- **연준 기준금리:** {fred.get('fed_funds', 'N/A')}%",
            f"- **2년물 국채:** {fred.get('treasury_2y', 'N/A')}%",
            f"- **10년물 국채:** {fred.get('treasury_10y', 'N/A')}%",
            f"- **10Y-2Y 스프레드:** {fred.get('spread_10y2y', 'N/A')}%",
            f"- **수익률 곡선:** {fred.get('curve_status', 'N/A')}",
            "",
            "### 인플레이션",
            f"- **CPI (YoY):** {fred.get('cpi_yoy', 'N/A')}%",
            f"- **Core PCE:** {fred.get('core_pce_yoy', 'N/A')}%",
            f"- **5년 손익분기 인플레이션:** {fred.get('breakeven_5y', 'N/A')}%",
            "",
            "### 유동성",
            f"- **순 유동성:** ${fred.get('net_liquidity', 0):.1f}T",
            f"- **유동성 레짐:** {fred.get('liquidity_regime', 'N/A')}",
            f"- **RRP:** ${fred.get('rrp', 0):.1f}T ({fred.get('rrp_delta_pct', 0):+.1f}%)",
            f"- **TGA:** ${fred.get('tga', 0):.1f}B ({fred.get('tga_delta', 0):+.1f}B)",
        ]
        
        signals = fred.get('signals', [])
        warnings = fred.get('warnings', [])
        
        if signals:
            lines.append("\n### 📈 시그널")
            for s in signals:
                lines.append(f"- ✓ {s}")
        
        if warnings:
            lines.append("\n### ⚠️ 경고")
            for w in warnings:
                lines.append(f"- ⚠️ {w}")
        
        return "\n".join(lines)

    def _market_regime(self) -> str:
        regime = self.data.get('regime', {})
        if not regime:
            return ""
        
        return f"""## 📈 시장 레짐 분석

**레짐:** {regime.get('regime', 'Unknown')}  
**추세:** {regime.get('trend', 'N/A')}  
**변동성:** {regime.get('volatility', 'N/A')}  
**신뢰도:** {regime.get('confidence', 0) * 100:.0f}%  

**투자 전략:** {regime.get('strategy', 'N/A')}"""

    def _ai_debate_results(self) -> str:
        debate = self.data.get('debate_consensus', {})
        enhanced = debate.get('enhanced', {})
        interp = enhanced.get('interpretation', {})
        metadata = debate.get('metadata', {})
        
        if not interp:
            return ""
        
        lines = [
            "## 🤖 AI 에이전트 토론 결과",
            "",
            f"**참여 에이전트:** {metadata.get('num_agents', 'N/A')}개",
            f"**권고 방향:** {interp.get('recommended_action', 'N/A')}",
            f"**평균 신뢰도:** {metadata.get('avg_confidence', 0) * 100:.0f}%",
            "",
            "### 합의 사항"
        ]
        
        for point in interp.get('consensus_points', []):
            lines.append(f"- ✓ {point}")
        
        lines.append("\n### 이견 사항")
        for point in interp.get('divergence_points', []):
            lines.append(f"- ⚠️ {point}")
        
        # 학파별 해석
        schools = interp.get('school_interpretations', [])
        if schools:
            lines.append("\n### 경제학파별 해석")
            for school in schools:
                lines.append(f"\n**{school.get('school', 'Unknown')}** ({school.get('stance', 'N/A')})")
                for reason in school.get('reasoning', [])[:2]:
                    lines.append(f"> {reason[:200]}{'...' if len(reason) > 200 else ''}")
        
        # Reasoning Chain
        chain = self.data.get('reasoning_chain', [])
        if chain:
            lines.append("\n### 추론 과정 (Reasoning Chain)")
            for step in chain:
                lines.append(f"\n**Step {step.get('step', '?')}: {step.get('agent', 'Unknown')}**")
                lines.append(f"- Output: {step.get('output', 'N/A')}")
                lines.append(f"- 신뢰도: {step.get('confidence', 0):.0f}%")
        
        return "\n".join(lines)

    def _ark_analysis(self) -> str:
        ark = self.data.get('ark_analysis', {})
        if not ark or ark.get('timestamp') is None:
            return ""
        
        lines = [
            "## 🚀 ARK Invest 분석",
            "",
            f"**분석 시점:** {ark.get('timestamp', '')[:19]}",
            "",
            "### 컨센서스 매수",
        ]
        
        for ticker in ark.get('consensus_buys', []):
            lines.append(f"- 📈 **{ticker}**")
        
        lines.append("\n### 컨센서스 매도")
        for ticker in ark.get('consensus_sells', []):
            lines.append(f"- 📉 **{ticker}**")
        
        lines.append("\n### 신규 편입")
        for ticker in ark.get('new_positions', []):
            lines.append(f"- 🆕 **{ticker}**")
        
        lines.append("\n### 주요 시그널")
        for sig in ark.get('signals', [])[:5]:
            lines.append(f"- {sig}")
        
        return "\n".join(lines)

    def _technical_analysis(self) -> str:
        lines = ["## 📐 기술적 분석"]
        
        # HFT Microstructure
        hft = self.data.get('hft_microstructure', {})
        if hft:
            tick = hft.get('tick_rule', {})
            lines.append("\n### HFT 미시구조")
            lines.append(f"- **매수 압력:** {tick.get('buy_ratio', 0) * 100:.1f}%")
            lines.append(f"- **매도 압력:** {tick.get('sell_ratio', 0) * 100:.1f}%")
            lines.append(f"- **해석:** {tick.get('interpretation', 'N/A')}")
        
        # GARCH
        garch = self.data.get('garch_volatility', {})
        if garch:
            lines.append("\n### GARCH 변동성")
            lines.append(f"- **현재 변동성:** {garch.get('current_volatility', 0) * 100:.1f}%")
            lines.append(f"- **10일 평균 예측:** {garch.get('forecast_avg_volatility', 0) * 100:.1f}%")
        
        # Proof of Index
        poi = self.data.get('proof_of_index', {})
        if poi:
            mr = poi.get('mean_reversion_signal', {})
            lines.append("\n### Proof-of-Index")
            lines.append(f"- **지수 값:** {poi.get('index_value', 0):.2f}")
            lines.append(f"- **Z-Score:** {mr.get('z_score', 0):.2f}")
            lines.append(f"- **신호:** {mr.get('signal', 'N/A')}")
        
        # DTW Similarity
        dtw = self.data.get('dtw_similarity', {})
        if dtw:
            sim = dtw.get('most_similar_pair', {})
            lead = dtw.get('lead_lag_spy_qqq', {})
            lines.append("\n### DTW 시계열 유사도")
            lines.append(f"- **가장 유사:** {sim.get('asset1', '')} ↔ {sim.get('asset2', '')}")
            lines.append(f"- **선후행:** {lead.get('interpretation', 'N/A')}")
        
        # DBSCAN
        dbscan = self.data.get('dbscan_outliers', {})
        if dbscan:
            lines.append("\n### DBSCAN 이상치 탐지")
            lines.append(f"- **이상치 비율:** {dbscan.get('outlier_ratio', 0) * 100:.1f}%")
            outliers = dbscan.get('outlier_tickers', [])
            if outliers:
                lines.append(f"- **이상 자산:** {', '.join(outliers)}")
        
        return "\n".join(lines)

    def _sentiment_analysis(self) -> str:
        sent = self.data.get('sentiment_analysis', {})
        if not sent:
            return ""
        
        fg = sent.get('fear_greed', {})
        vix = sent.get('vix_structure', {})
        news = sent.get('news_sentiment', {})
        
        lines = [
            "## 😊 센티먼트 분석",
            "",
            "### Fear & Greed Index",
            f"- **현재:** {fg.get('value', 'N/A')} ({fg.get('level', 'N/A')})",
            f"- **직전 종가:** {fg.get('previous_close', 'N/A')}",
            f"- **1주 전:** {fg.get('week_ago', 'N/A')}",
            "",
            "### VIX 구조",
            f"- **VIX Spot:** {vix.get('vix_spot', 'N/A')}",
            f"- **구조:** {vix.get('structure', 'N/A')}",
            f"- **신호:** {vix.get('signal', 'N/A')}",
            "",
            "### 뉴스 센티먼트",
            f"- **평균 점수:** {news.get('avg_score', 0):.2f}",
            f"- **분석 건수:** {news.get('count', 0)}건",
            f"- **전체:** {news.get('overall', 'N/A')}"
        ]
        
        # Extended Data
        ext = self.data.get('extended_data', {})
        if ext:
            pc = ext.get('put_call_ratio', {})
            fund = ext.get('fundamentals', {})
            credit = ext.get('credit_spreads', {})
            
            lines.append("\n### 확장 지표")
            if pc:
                lines.append(f"- **Put/Call Ratio:** {pc.get('ratio', 0):.2f} ({pc.get('sentiment', 'N/A')})")
            if fund:
                lines.append(f"- **S&P 500 P/E:** {fund.get('pe_ratio', 0):.1f}x")
                lines.append(f"- **어닝 일드:** {fund.get('earnings_yield', 0):.2f}%")
            if credit:
                lines.append(f"- **신용 스프레드 해석:** {credit.get('interpretation', 'N/A')}")
        
        return "\n".join(lines)

    def _recommendation(self) -> str:
        rec = self.data.get('final_recommendation', 'N/A')
        conf = self.data.get('confidence', 0) * 100
        risk_level = self.data.get('risk_level', 'N/A')
        
        adaptive = self.data.get('adaptive_portfolios', {})
        
        lines = [
            "## 💡 최종 권고",
            "",
            f"### 투자 포지션: **{rec}**",
            "",
            f"- **신뢰도:** {conf:.0f}%",
            f"- **리스크 레벨:** {risk_level}",
        ]
        
        if adaptive:
            lines.append("\n### 투자자 성향별 권고")
            lines.append(f"- **적극형:** {adaptive.get('aggressive', 'N/A')}")
            lines.append(f"- **균형형:** {adaptive.get('balanced', 'N/A')}")
            lines.append(f"- **보수형:** {adaptive.get('conservative', 'N/A')}")
        
        # AI Report highlights
        ai_report = self.data.get('ai_report', {})
        highlights = ai_report.get('highlights', {})
        notable = highlights.get('notable_stocks', [])
        
        if notable:
            lines.append("\n### 주목할 종목")
            for stock in notable:
                lines.append(f"- **{stock.get('ticker', '')}:** {stock.get('reason', '')}")
        
        return "\n".join(lines)

    def _footer(self) -> str:
        return """## ⚠️ Disclaimer

본 리포트는 EIMAS 시스템에 의해 자동 생성되었으며, 투자 권유가 아닙니다.
모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.

---
*Generated by EIMAS (Economic Intelligence Multi-Agent System)*"""


def convert_json_to_md(json_path: Path) -> Path:
    """JSON 파일을 MD로 변환"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converter = EIMASMarkdownConverter(data)
    md_content = converter.convert()
    
    # 출력 파일명 생성
    md_path = json_path.with_suffix('.md')
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✓ Converted: {json_path.name} → {md_path.name}")
    return md_path


def main():
    output_dir = Path(__file__).parent.parent / "outputs"
    
    if len(sys.argv) > 1:
        # 특정 파일 지정
        json_path = output_dir / sys.argv[1]
    else:
        # 최신 eimas_*.json 찾기
        json_files = sorted(output_dir.glob("eimas_*.json"), reverse=True)
        if not json_files:
            print("No eimas_*.json files found in outputs/")
            return
        json_path = json_files[0]
    
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return
    
    convert_json_to_md(json_path)


if __name__ == "__main__":
    main()
