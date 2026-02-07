#!/usr/bin/env python3
"""
AI Report Generator
====================
JSON 분석 결과를 바탕으로 AI API들을 활용해 최종 제안서 생성

사용 API:
- Claude: 종합 분석 및 제안서 작성
- Perplexity: 최신 뉴스/이벤트 검색
- GPT: 특정 종목 심층 분석

Usage:
    generator = AIReportGenerator()
    report = await generator.generate(json_result, market_data)
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from core.config import APIConfig, AGENT_CONFIG

logger = logging.getLogger('eimas.ai_report')


@dataclass
class StockAnalysis:
    """개별 종목 분석 결과"""
    ticker: str
    change_1d: float = 0.0
    change_5d: float = 0.0
    change_20d: float = 0.0
    volatility: float = 0.0
    is_notable: bool = False
    notable_reason: str = ""
    news_summary: str = ""
    deep_analysis: str = ""


@dataclass
class TechnicalIndicators:
    """기술적 지표"""
    vix: float = 0.0
    vix_change: float = 0.0
    rsi_14: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    ma_50: float = 0.0
    ma_200: float = 0.0
    current_price: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0


@dataclass
class ScenarioCase:
    """시나리오 케이스"""
    name: str
    probability: float
    expected_return: str
    sp500_target: str
    strategy: str
    key_triggers: List[str] = field(default_factory=list)


@dataclass
class GlobalMarketData:
    """국제 시장 데이터"""
    # 달러 인덱스
    dxy: float = 0.0
    dxy_change: float = 0.0

    # 주요 지수
    dax: float = 0.0  # 독일
    dax_change: float = 0.0
    ftse: float = 0.0  # 영국
    ftse_change: float = 0.0
    nikkei: float = 0.0  # 일본
    nikkei_change: float = 0.0
    shanghai: float = 0.0  # 중국
    shanghai_change: float = 0.0
    kospi: float = 0.0  # 한국
    kospi_change: float = 0.0

    # 원자재
    gold: float = 0.0
    gold_change: float = 0.0
    wti: float = 0.0  # 원유
    wti_change: float = 0.0
    copper: float = 0.0  # 구리
    copper_change: float = 0.0

    # 분석
    global_sentiment: str = "NEUTRAL"  # RISK_ON, RISK_OFF, NEUTRAL
    correlation_with_us: str = ""
    key_risks: List[str] = field(default_factory=list)


@dataclass
class ReportComparison:
    """이전 리포트와의 비교"""
    previous_timestamp: str = ""

    # 레짐 변화
    regime_changed: bool = False
    previous_regime: str = ""
    current_regime: str = ""
    regime_change_direction: str = ""  # "UPGRADE", "DOWNGRADE", "SAME"

    # 신뢰도 변화
    confidence_delta: float = 0.0
    previous_confidence: float = 0.0
    current_confidence: float = 0.0

    # 리스크 점수 변화
    risk_score_delta: float = 0.0
    previous_risk_score: float = 0.0
    current_risk_score: float = 0.0

    # VIX 변화
    vix_delta: float = 0.0
    previous_vix: float = 0.0
    current_vix: float = 0.0

    # 투자 권고 변화
    recommendation_changed: bool = False
    previous_recommendation: str = ""
    current_recommendation: str = ""

    # 주요 변화 요약
    key_changes: List[str] = field(default_factory=list)
    change_significance: str = "MINOR"  # "MAJOR", "MODERATE", "MINOR"


@dataclass
class EntryExitStrategy:
    """진입/청산 전략"""
    # 현재 가격 기준
    current_price: float = 0.0

    # 진입 전략
    entry_levels: List[Dict[str, Any]] = field(default_factory=list)  # [{"price": 680, "ratio": 30, "condition": "1차 진입"}]
    entry_ratios: str = ""  # "30%-30%-40%"

    # 청산 전략
    take_profit_levels: List[Dict[str, Any]] = field(default_factory=list)  # [{"price": 720, "ratio": 50, "target": "+5%"}]
    stop_loss_level: float = 0.0
    stop_loss_percent: float = 0.0
    trailing_stop: str = ""

    # 시나리오별 전략
    bull_strategy: str = ""
    bear_strategy: str = ""

    # 리밸런싱
    rebalancing_trigger: str = ""
    position_sizing: str = ""


@dataclass
class FinalReport:
    """최종 제안서"""
    timestamp: str

    # 시장 요약
    market_summary: str = ""
    regime_analysis: str = ""
    risk_assessment: str = ""

    # 기술적 지표 (NEW)
    technical_indicators: Optional[TechnicalIndicators] = None

    # 종목별 분석
    notable_stocks: List[StockAnalysis] = field(default_factory=list)
    notable_stocks_reason: str = ""  # 종목이 없는 경우 이유 설명

    # 시나리오 분석 (NEW)
    scenarios: List[ScenarioCase] = field(default_factory=list)

    # 국제 시장 분석 (NEW)
    global_market: Optional[GlobalMarketData] = None

    # 진입/청산 전략 (NEW)
    entry_exit_strategy: Optional[EntryExitStrategy] = None

    # AI 분석
    perplexity_news: str = ""
    claude_analysis: str = ""
    gpt_recommendations: str = ""

    # 섹터/산업군 추천
    sector_recommendations: Dict[str, Any] = field(default_factory=dict)

    # 최종 권고
    final_recommendation: str = ""
    action_items: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)

    # 신뢰도 분석 (NEW)
    confidence_analysis: str = ""

    # 참고문헌 및 면책조항 (NEW)
    references: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    disclaimer: str = ""

    # 히스토리컬 비교 (NEW)
    historical_comparison: Optional[ReportComparison] = None

    # 백테스팅 섹션 (NEW)
    backtest_section: str = ""

    # 옵션/센티먼트 분석 (NEW)
    options_analysis: Optional[Dict[str, Any]] = None
    sentiment_analysis: Optional[Dict[str, Any]] = None

    # 멀티 LLM 인사이트 (NEW)
    multi_llm_insights: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_markdown(self) -> str:
        """마크다운 형식 리포트 생성"""
        md = []
        md.append("# EIMAS 투자 제안서")
        md.append(f"**생성일시**: {self.timestamp}")
        md.append("")

        # 히스토리컬 비교 섹션 (있는 경우)
        if self.historical_comparison and self.historical_comparison.previous_timestamp:
            hc = self.historical_comparison
            md.append("---")
            md.append("## 📊 이전 리포트 대비 변화")
            md.append(f"**비교 대상**: {hc.previous_timestamp}")
            md.append("")

            # 변화 중요도 표시
            significance_emoji = {"MAJOR": "🔴", "MODERATE": "🟡", "MINOR": "🟢"}.get(hc.change_significance, "⚪")
            md.append(f"**변화 수준**: {significance_emoji} {hc.change_significance}")
            md.append("")

            # 주요 변화 요약
            if hc.key_changes:
                md.append("### 🔔 주요 변화")
                for change in hc.key_changes:
                    md.append(f"- {change}")
                md.append("")

            # 상세 비교 테이블
            md.append("### 📈 지표 비교")
            md.append("| 항목 | 이전 | 현재 | 변화 |")
            md.append("|------|------|------|------|")

            # 레짐
            regime_emoji = "⬆️" if hc.regime_change_direction == "UPGRADE" else "⬇️" if hc.regime_change_direction == "DOWNGRADE" else "➡️"
            md.append(f"| 레짐 | {hc.previous_regime} | {hc.current_regime} | {regime_emoji} {hc.regime_change_direction} |")

            # 신뢰도
            conf_emoji = "⬆️" if hc.confidence_delta > 0 else "⬇️" if hc.confidence_delta < 0 else "➡️"
            md.append(f"| 신뢰도 | {hc.previous_confidence:.0f}% | {hc.current_confidence:.0f}% | {conf_emoji} {hc.confidence_delta:+.0f}%p |")

            # 리스크 점수
            risk_emoji = "⬇️" if hc.risk_score_delta < 0 else "⬆️" if hc.risk_score_delta > 0 else "➡️"  # 리스크는 낮아지면 좋음
            md.append(f"| 리스크 점수 | {hc.previous_risk_score:.1f} | {hc.current_risk_score:.1f} | {risk_emoji} {hc.risk_score_delta:+.1f} |")

            # VIX
            vix_emoji = "⬇️" if hc.vix_delta < 0 else "⬆️" if hc.vix_delta > 0 else "➡️"
            md.append(f"| VIX | {hc.previous_vix:.1f} | {hc.current_vix:.1f} | {vix_emoji} {hc.vix_delta:+.1f} |")

            # 투자 권고
            rec_emoji = "🔄" if hc.recommendation_changed else "➡️"
            md.append(f"| 투자 권고 | {hc.previous_recommendation} | {hc.current_recommendation} | {rec_emoji} |")
            md.append("")
            md.append("---")
            md.append("")

        # Section 1: 시장 요약
        md.append("## 1. 시장 요약")
        md.append(self.market_summary)
        md.append("")

        # Section 2: 레짐 분석
        md.append("## 2. 레짐 분석")
        md.append(self.regime_analysis)
        if self.confidence_analysis:
            md.append("")
            md.append("### 📊 신뢰도 분석")
            md.append(self.confidence_analysis)
        md.append("")

        # Section 2.5: 백테스팅 (유사 레짐 분석)
        if self.backtest_section:
            md.append(self.backtest_section)
            md.append("")

        # Section 3: 기술적 지표
        md.append("## 3. 기술적 지표")
        if self.technical_indicators:
            ti = self.technical_indicators
            md.append(f"### 📈 주요 지수")
            # SPY는 S&P 500의 약 1/10 가격으로 거래됨
            sp500_approx = ti.current_price * 10
            md.append(f"- **SPY**: ${ti.current_price:,.2f} (S&P 500 ≈ {sp500_approx:,.0f})")
            md.append(f"- **VIX**: {ti.vix:.2f} ({ti.vix_change:+.2f}%)")
            md.append("")
            md.append(f"### 📉 모멘텀 지표")
            md.append(f"- **RSI (14일)**: {ti.rsi_14:.1f}")
            rsi_signal = "과매수" if ti.rsi_14 > 70 else "과매도" if ti.rsi_14 < 30 else "중립"
            md.append(f"  - 해석: {rsi_signal} 구간")
            md.append(f"- **MACD**: {ti.macd:.2f}")
            md.append(f"- **MACD Signal**: {ti.macd_signal:.2f}")
            macd_signal = "매수 신호" if ti.macd > ti.macd_signal else "매도 신호"
            md.append(f"  - 해석: {macd_signal}")
            md.append("")
            md.append(f"### 📊 이동평균선")
            md.append(f"- **50일 이동평균**: {ti.ma_50:,.2f}")
            md.append(f"- **200일 이동평균**: {ti.ma_200:,.2f}")
            if ti.ma_50 > ti.ma_200:
                md.append("  - 해석: 골든 크로스 상태 (상승 추세)")
            else:
                md.append("  - 해석: 데드 크로스 상태 (하락 추세)")
            md.append("")
            md.append(f"### 🎯 지지/저항선")
            md.append(f"- **지지선**: {ti.support_level:,.2f}")
            md.append(f"- **저항선**: {ti.resistance_level:,.2f}")
        else:
            md.append("기술적 지표 데이터 없음")
        md.append("")

        # Section 4: 국제 시장 분석 (NEW)
        md.append("## 4. 국제 시장 분석")
        if self.global_market:
            gm = self.global_market
            md.append("### 💵 달러 인덱스")
            md.append(f"- **DXY**: {gm.dxy:.2f} ({gm.dxy_change:+.2f}%)")
            md.append("")

            md.append("### 🌍 글로벌 지수")
            md.append(f"- **DAX (독일)**: {gm.dax:,.2f} ({gm.dax_change:+.2f}%)")
            md.append(f"- **FTSE 100 (영국)**: {gm.ftse:,.2f} ({gm.ftse_change:+.2f}%)")
            md.append(f"- **Nikkei 225 (일본)**: {gm.nikkei:,.2f} ({gm.nikkei_change:+.2f}%)")
            md.append(f"- **Shanghai Composite (중국)**: {gm.shanghai:,.2f} ({gm.shanghai_change:+.2f}%)")
            md.append(f"- **KOSPI (한국)**: {gm.kospi:,.2f} ({gm.kospi_change:+.2f}%)")
            md.append("")

            md.append("### ⛏️ 원자재")
            md.append(f"- **Gold**: ${gm.gold:,.2f} ({gm.gold_change:+.2f}%)")
            md.append(f"- **WTI 원유**: ${gm.wti:.2f} ({gm.wti_change:+.2f}%)")
            md.append(f"- **Copper**: ${gm.copper:.2f} ({gm.copper_change:+.2f}%)")
            md.append("")

            md.append("### 📊 글로벌 시장 심리")
            sentiment_emoji = "🟢" if gm.global_sentiment == "RISK_ON" else "🔴" if gm.global_sentiment == "RISK_OFF" else "🟡"
            md.append(f"- **글로벌 심리**: {sentiment_emoji} {gm.global_sentiment}")
            if gm.correlation_with_us:
                md.append(f"- **미국 시장 연동성**: {gm.correlation_with_us}")
            if gm.key_risks:
                md.append("- **주요 리스크**:")
                for risk in gm.key_risks:
                    md.append(f"  - {risk}")
        else:
            md.append("국제 시장 데이터 없음")
        md.append("")

        # Section 5: 리스크 평가 (was 4)
        md.append("## 5. 리스크 평가")
        md.append(self.risk_assessment)
        md.append("")

        # Section 6: 시나리오 분석 (was 5)
        md.append("## 6. 시나리오 분석")
        if self.scenarios:
            for scenario in self.scenarios:
                emoji = "🐂" if "Bull" in scenario.name else "🐻" if "Bear" in scenario.name else "📊"
                md.append(f"### {emoji} {scenario.name}")
                md.append(f"- **확률**: {scenario.probability:.0f}%")
                md.append(f"- **예상 수익률**: {scenario.expected_return}")
                md.append(f"- **S&P 500 목표**: {scenario.sp500_target}")
                md.append(f"- **전략**: {scenario.strategy}")
                if scenario.key_triggers:
                    md.append("- **주요 트리거**:")
                    for trigger in scenario.key_triggers:
                        md.append(f"  - {trigger}")
                md.append("")
        else:
            md.append("시나리오 분석 데이터 없음")
            md.append("")

        # Section 7: 주목할 종목 (was 6)
        md.append("## 7. 주목할 종목")
        if self.notable_stocks:
            for stock in self.notable_stocks:
                md.append(f"### {stock.ticker}")
                md.append(f"- 1일 변화: {stock.change_1d:+.2f}%")
                md.append(f"- 5일 변화: {stock.change_5d:+.2f}%")
                md.append(f"- 20일 변화: {stock.change_20d:+.2f}%")
                md.append(f"- 변동성: {stock.volatility:.2f}%")
                if stock.notable_reason:
                    md.append(f"- **주목 이유**: {stock.notable_reason}")
                if stock.deep_analysis:
                    md.append(f"\n{stock.deep_analysis}")
                md.append("")
        else:
            md.append("### 분석 결과")
            if self.notable_stocks_reason:
                md.append(self.notable_stocks_reason)
            else:
                md.append("현재 분석 기준(1일 ±3%, 5일 ±7%, 변동성 3% 이상)을 충족하는 특이 종목이 없습니다.")
                md.append("이는 시장이 안정적인 상태임을 나타낼 수 있습니다.")
            md.append("")

        # Section 8: 최신 뉴스 및 이벤트 (was 7)
        md.append("## 8. 최신 뉴스 및 이벤트")
        md.append(self.perplexity_news if self.perplexity_news else "뉴스 정보 없음")
        md.append("")

        # Section 9: AI 종합 분석 (was 8)
        md.append("## 9. AI 종합 분석")
        md.append(self.claude_analysis if self.claude_analysis else "분석 정보 없음")
        md.append("")

        # Section 10: 투자 권고 (was 9)
        md.append("## 10. 투자 권고")
        md.append(self.gpt_recommendations if self.gpt_recommendations else "권고 정보 없음")
        md.append("")

        # Section 11: 진입/청산 전략 (NEW)
        md.append("## 11. 진입/청산 전략")
        if self.entry_exit_strategy:
            ees = self.entry_exit_strategy
            md.append(f"### 📍 현재 가격: ${ees.current_price:,.2f}")
            md.append("")

            if ees.entry_levels:
                md.append("### 📥 진입 전략")
                md.append(f"**분할 매수 비율**: {ees.entry_ratios}")
                md.append("")
                md.append("| 구분 | 진입가 | 비율 | 조건 |")
                md.append("|------|--------|------|------|")
                for level in ees.entry_levels:
                    md.append(f"| {level.get('name', 'N/A')} | ${level.get('price', 0):,.2f} | {level.get('ratio', 0)}% | {level.get('condition', 'N/A')} |")
                md.append("")

            if ees.take_profit_levels:
                md.append("### 📤 청산 전략")
                md.append("| 구분 | 목표가 | 비율 | 예상 수익 |")
                md.append("|------|--------|------|----------|")
                for level in ees.take_profit_levels:
                    md.append(f"| {level.get('name', 'N/A')} | ${level.get('price', 0):,.2f} | {level.get('ratio', 0)}% | {level.get('target', 'N/A')} |")
                md.append("")

            md.append("### 🛑 손절 전략")
            md.append(f"- **손절가**: ${ees.stop_loss_level:,.2f} ({ees.stop_loss_percent:+.1f}%)")
            if ees.trailing_stop:
                md.append(f"- **트레일링 스탑**: {ees.trailing_stop}")
            md.append("")

            if ees.bull_strategy or ees.bear_strategy:
                md.append("### 📊 시나리오별 전략")
                if ees.bull_strategy:
                    md.append(f"- **상승장**: {ees.bull_strategy}")
                if ees.bear_strategy:
                    md.append(f"- **하락장**: {ees.bear_strategy}")
                md.append("")

            if ees.rebalancing_trigger or ees.position_sizing:
                md.append("### ⚖️ 포지션 관리")
                if ees.position_sizing:
                    md.append(f"- **포지션 사이징**: {ees.position_sizing}")
                if ees.rebalancing_trigger:
                    md.append(f"- **리밸런싱 조건**: {ees.rebalancing_trigger}")
                md.append("")
        else:
            md.append("진입/청산 전략 데이터 없음")
        md.append("")

        # Section 12: 추천 섹터 및 산업군 (was 10)
        md.append("## 12. 추천 섹터 및 산업군")
        if self.sector_recommendations:
            sectors = self.sector_recommendations

            # Bullish 섹터
            if sectors.get('bullish_sectors'):
                md.append("### 📈 강세 예상 섹터")
                for sector in sectors['bullish_sectors']:
                    md.append(f"**{sector.get('name', 'N/A')}**")
                    md.append(f"- 투자 의견: {sector.get('rating', 'N/A')}")
                    md.append(f"- 근거: {sector.get('rationale', 'N/A')}")
                    if sector.get('etfs'):
                        md.append(f"- 관련 ETF: {', '.join(sector['etfs'])}")
                    if sector.get('expense_ratio'):
                        md.append(f"- 비용비율: {sector['expense_ratio']}")
                    md.append("")

            # Neutral 섹터
            if sectors.get('neutral_sectors'):
                md.append("### ➡️ 중립 섹터")
                for sector in sectors['neutral_sectors']:
                    md.append(f"**{sector.get('name', 'N/A')}**")
                    md.append(f"- 투자 의견: {sector.get('rating', 'N/A')}")
                    md.append(f"- 근거: {sector.get('rationale', 'N/A')}")
                    md.append("")

            # Bearish 섹터
            if sectors.get('bearish_sectors'):
                md.append("### 📉 약세 예상 섹터")
                for sector in sectors['bearish_sectors']:
                    md.append(f"**{sector.get('name', 'N/A')}**")
                    md.append(f"- 투자 의견: {sector.get('rating', 'N/A')}")
                    md.append(f"- 근거: {sector.get('rationale', 'N/A')}")
                    md.append("")

            # 주목할 산업군
            if sectors.get('hot_industries'):
                md.append("### 🔥 주목할 산업군")
                for industry in sectors['hot_industries']:
                    md.append(f"- **{industry.get('name', 'N/A')}**: {industry.get('description', 'N/A')}")
                md.append("")

            # AI 기반 섹터 분석
            if sectors.get('ai_analysis'):
                md.append("### 🤖 AI 섹터 분석")
                md.append(sectors['ai_analysis'])
                md.append("")
        else:
            md.append("섹터 분석 정보 없음")
            md.append("")

        # Section 13: 최종 제안 (was 11)
        md.append("## 13. 최종 제안")
        md.append(self.final_recommendation)
        md.append("")

        if self.action_items:
            md.append("### 액션 아이템")
            for item in self.action_items:
                md.append(f"- {item}")
            md.append("")

        if self.risk_warnings:
            md.append("### 리스크 경고")
            for warning in self.risk_warnings:
                md.append(f"- ⚠️ {warning}")
            md.append("")

        # Section 14: 참고문헌 및 데이터 소스 (was 12)
        md.append("## 14. 참고문헌 및 데이터 소스")
        md.append("### 📚 데이터 소스")
        if self.data_sources:
            for source in self.data_sources:
                md.append(f"- {source}")
        else:
            md.append("- Yahoo Finance (시장 데이터)")
            md.append("- FRED (Federal Reserve Economic Data)")
            md.append("- Perplexity AI (뉴스 검색)")
            md.append("- OpenAI GPT-4 (분석 및 권고)")
            md.append("- Anthropic Claude (종합 분석)")
        md.append("")

        if self.references:
            md.append("### 📰 참고 뉴스")
            for ref in self.references:
                md.append(f"- {ref}")
            md.append("")

        # Section 15: 옵션/센티먼트 분석 (NEW)
        if self.options_analysis or self.sentiment_analysis:
            md.append("## 15. 옵션 & 센티먼트 분석")
            md.append("")

            if self.sentiment_analysis:
                sa = self.sentiment_analysis
                md.append("### 😨 Fear & Greed Index")
                if 'fear_greed_index' in sa:
                    fg = sa['fear_greed_index']
                    md.append(f"- **현재**: {fg.get('value', 'N/A')} ({fg.get('classification', 'N/A')})")
                    if 'previous_close' in fg:
                        md.append(f"- **전일**: {fg.get('previous_close', 'N/A')}")
                md.append("")

            if self.options_analysis:
                oa = self.options_analysis
                md.append("### 📊 VIX 기간 구조")
                if 'vix_term_structure' in oa:
                    vts = oa['vix_term_structure']
                    md.append(f"- **구조**: {vts.get('structure', 'N/A')}")
                    md.append(f"- **VIX Spot**: {vts.get('vix_spot', 'N/A'):.2f}")
                    md.append(f"- **VIX 3M**: {vts.get('vix_3m', 'N/A'):.2f}")
                    md.append(f"- **스프레드**: {vts.get('spread', 0):.2f}%")
                    md.append(f"- **시그널**: {vts.get('signal', 'N/A')}")
                md.append("")

                md.append("### 📈 Put/Call Ratio")
                if 'put_call_ratio' in oa:
                    pcr = oa['put_call_ratio']
                    md.append(f"- **P/C Ratio**: {pcr.get('ratio', 'N/A'):.2f}")
                    md.append(f"- **레벨**: {pcr.get('level', 'N/A')}")
                    md.append(f"- **역발상 시그널**: {pcr.get('contrarian_signal', 'N/A')}")
                md.append("")

                md.append("### 💹 IV Percentile")
                if 'iv_percentile' in oa:
                    ivp = oa['iv_percentile']
                    md.append(f"- **IV Percentile**: {ivp.get('percentile', 'N/A'):.1f}%")
                    md.append(f"- **현재 IV**: {ivp.get('current_iv', 'N/A'):.1f}%")
                    md.append(f"- **레벨**: {ivp.get('level', 'N/A')}")
                md.append("")

        # Section 16: 멀티 LLM 인사이트 (NEW)
        if self.multi_llm_insights:
            md.append("## 16. Multi-LLM 인사이트")
            md.append("")
            mli = self.multi_llm_insights
            if 'consensus_points' in mli:
                md.append("### ✅ 합의 포인트")
                for point in mli['consensus_points']:
                    md.append(f"- {point}")
                md.append("")
            if 'divergence_points' in mli:
                md.append("### ⚠️ 의견 차이")
                for point in mli['divergence_points']:
                    md.append(f"- {point}")
                md.append("")
            if 'actionable_items' in mli:
                md.append("### 📋 실행 가능 항목")
                for item in mli['actionable_items']:
                    md.append(f"- {item}")
                md.append("")

        # Section 17: 면책조항
        md.append("## 17. 면책조항")
        if self.disclaimer:
            md.append(self.disclaimer)
        else:
            md.append("""
⚠️ **투자 위험 고지**

본 리포트는 정보 제공 목적으로만 작성되었으며, 투자 권유나 매매 추천을 구성하지 않습니다.

**주요 한계점:**
- AI 모델의 분석은 과거 데이터에 기반하며, 미래 수익을 보장하지 않습니다
- 시장 데이터는 15-20분 지연될 수 있습니다
- 레짐 탐지 모델은 급격한 시장 변화에 후행할 수 있습니다
- 뉴스 분석은 실시간이 아닐 수 있습니다

**투자자 유의사항:**
- 모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다
- 투자 전 전문 금융 상담사와 상담하시기 바랍니다
- 과거 수익률이 미래 수익을 보장하지 않습니다
- 레버리지 상품은 원금 손실 위험이 있습니다
""")
        md.append("")

        return "\n".join(md)


class AIReportGenerator:
    """AI 기반 리포트 생성기"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._validate_apis()

    def _validate_apis(self):
        """API 키 검증"""
        status = APIConfig.validate()
        self.has_claude = status.get('anthropic', False)
        self.has_perplexity = status.get('perplexity', False)
        self.has_gpt = status.get('openai', False)

        if self.verbose:
            print(f"[AIReportGenerator] APIs: Claude={self.has_claude}, Perplexity={self.has_perplexity}, GPT={self.has_gpt}")

    def _log(self, msg: str):
        if self.verbose:
            print(f"[AIReportGenerator] {msg}")

    def _load_previous_report(self, output_dir: str = "outputs") -> Optional[Dict]:
        """이전 리포트 JSON 로드"""
        output_path = Path(output_dir)

        if not output_path.exists():
            return None

        # ai_report_*.json 파일 검색 (최신순 정렬)
        json_files = sorted(output_path.glob("ai_report_*.json"), reverse=True)

        if len(json_files) < 1:
            return None

        # 가장 최신 파일 로드 (현재 생성 전이므로 이게 이전 리포트)
        try:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self._log(f"Failed to load previous report: {e}")
            return None

    def _compare_with_previous(
        self,
        current_result: Dict,
        current_report: FinalReport,
        previous_data: Dict
    ) -> ReportComparison:
        """이전 리포트와 비교"""
        comparison = ReportComparison()

        # 이전 타임스탬프
        comparison.previous_timestamp = previous_data.get('timestamp', 'Unknown')

        # 현재 값들
        current_regime = current_result.get('regime', {}).get('regime', 'Unknown')
        current_conf = current_result.get('confidence', 0.5) * 100
        current_risk = current_result.get('risk_score', 50)
        current_rec = current_result.get('final_recommendation', 'NEUTRAL')
        current_vix = 0.0
        if current_report.technical_indicators:
            current_vix = current_report.technical_indicators.vix

        # 이전 값들 추출
        prev_regime = previous_data.get('regime_analysis', '')
        # 레짐 파싱 (마크다운에서 추출)
        if '**현재 레짐**:' in prev_regime:
            try:
                prev_regime = prev_regime.split('**현재 레짐**:')[1].split('\n')[0].strip()
            except:
                prev_regime = 'Unknown'
        else:
            prev_regime = 'Unknown'

        # 이전 신뢰도 (confidence_analysis에서 추출)
        prev_conf_str = previous_data.get('confidence_analysis', '')
        prev_conf = 50.0
        if '신뢰도' in prev_conf_str:
            try:
                import re
                match = re.search(r'(\d+)%', prev_conf_str)
                if match:
                    prev_conf = float(match.group(1))
            except:
                pass

        # 이전 리스크 점수 (risk_assessment에서 추출)
        prev_risk_str = previous_data.get('risk_assessment', '')
        prev_risk = 50.0
        try:
            import re
            match = re.search(r'리스크 점수\*\*:\s*(\d+\.?\d*)/100', prev_risk_str)
            if match:
                prev_risk = float(match.group(1))
        except:
            pass

        # 이전 투자 권고 (final_recommendation에서 추출)
        prev_rec_str = previous_data.get('final_recommendation', '')
        prev_rec = 'NEUTRAL'
        if '적극적 매수' in prev_rec_str or 'BULLISH' in prev_rec_str.upper():
            prev_rec = 'BULLISH'
        elif '방어적' in prev_rec_str or 'BEARISH' in prev_rec_str.upper():
            prev_rec = 'BEARISH'

        # 이전 VIX (technical_indicators에서)
        prev_vix = 0.0
        prev_ti = previous_data.get('technical_indicators', {})
        if prev_ti:
            prev_vix = prev_ti.get('vix', 0.0)

        # 비교 결과 저장
        comparison.current_regime = current_regime
        comparison.previous_regime = prev_regime
        comparison.regime_changed = (current_regime != prev_regime)

        comparison.current_confidence = current_conf
        comparison.previous_confidence = prev_conf
        comparison.confidence_delta = current_conf - prev_conf

        comparison.current_risk_score = current_risk
        comparison.previous_risk_score = prev_risk
        comparison.risk_score_delta = current_risk - prev_risk

        comparison.current_vix = current_vix
        comparison.previous_vix = prev_vix
        comparison.vix_delta = current_vix - prev_vix

        comparison.current_recommendation = current_rec
        comparison.previous_recommendation = prev_rec
        comparison.recommendation_changed = (current_rec != prev_rec)

        # 레짐 변화 방향 결정
        regime_order = ['Bear', 'Neutral', 'Bull']
        def get_regime_level(r: str) -> int:
            for i, level in enumerate(regime_order):
                if level.lower() in r.lower():
                    return i
            return 1  # Neutral

        current_level = get_regime_level(current_regime)
        prev_level = get_regime_level(prev_regime)

        if current_level > prev_level:
            comparison.regime_change_direction = "UPGRADE"
        elif current_level < prev_level:
            comparison.regime_change_direction = "DOWNGRADE"
        else:
            comparison.regime_change_direction = "SAME"

        # 주요 변화 식별
        key_changes = []

        if comparison.regime_changed:
            direction_text = "상향" if comparison.regime_change_direction == "UPGRADE" else "하향" if comparison.regime_change_direction == "DOWNGRADE" else "변경"
            key_changes.append(f"🔄 레짐 {direction_text}: {prev_regime} → {current_regime}")

        if comparison.recommendation_changed:
            key_changes.append(f"📋 투자 권고 변경: {prev_rec} → {current_rec}")

        if abs(comparison.confidence_delta) >= 10:
            direction = "상승" if comparison.confidence_delta > 0 else "하락"
            key_changes.append(f"📊 신뢰도 {abs(comparison.confidence_delta):.0f}%p {direction}")

        if abs(comparison.risk_score_delta) >= 10:
            direction = "증가" if comparison.risk_score_delta > 0 else "감소"
            key_changes.append(f"⚠️ 리스크 점수 {abs(comparison.risk_score_delta):.1f}점 {direction}")

        if abs(comparison.vix_delta) >= 3:
            direction = "상승 (공포 증가)" if comparison.vix_delta > 0 else "하락 (안정화)"
            key_changes.append(f"📉 VIX {abs(comparison.vix_delta):.1f}p {direction}")

        comparison.key_changes = key_changes

        # 변화 중요도 결정
        if comparison.regime_changed or comparison.recommendation_changed:
            comparison.change_significance = "MAJOR"
        elif abs(comparison.confidence_delta) >= 10 or abs(comparison.risk_score_delta) >= 10:
            comparison.change_significance = "MODERATE"
        else:
            comparison.change_significance = "MINOR"

        return comparison

    async def generate(
        self,
        analysis_result: Dict[str, Any],
        market_data: Dict[str, Any] = None
    ) -> FinalReport:
        """최종 제안서 생성"""
        self._log("Starting report generation...")

        report = FinalReport(timestamp=datetime.now().isoformat())

        # 1. 기본 요약 생성
        report.market_summary = self._create_market_summary(analysis_result)
        report.regime_analysis = self._create_regime_analysis(analysis_result)
        report.risk_assessment = self._create_risk_assessment(analysis_result)

        # 2. 신뢰도 분석 (NEW)
        report.confidence_analysis = self._create_confidence_analysis(analysis_result)

        # 3. 기술적 지표 계산 (NEW)
        if market_data:
            self._log("Calculating technical indicators...")
            report.technical_indicators = self._calculate_technical_indicators(market_data, analysis_result)

        # 4. 주목할 종목 분석
        if market_data:
            report.notable_stocks = self._find_notable_stocks(market_data)
            self._log(f"Found {len(report.notable_stocks)} notable stocks")
            if not report.notable_stocks:
                report.notable_stocks_reason = self._explain_no_notable_stocks(market_data, analysis_result)
        else:
            report.notable_stocks_reason = "시장 데이터가 제공되지 않아 개별 종목 분석을 수행할 수 없습니다."

        # 5. 시나리오 분석 (NEW)
        self._log("Generating scenario analysis...")
        report.scenarios = self._generate_scenarios(analysis_result, report)

        # 5.5. 국제 시장 데이터 수집 (NEW)
        self._log("Fetching global market data...")
        report.global_market = await self._fetch_global_markets()

        # 6. Perplexity로 최신 뉴스 검색
        if self.has_perplexity:
            self._log("Fetching latest news with Perplexity...")
            report.perplexity_news = await self._search_news(analysis_result, report.notable_stocks)

        # 7. GPT로 특정 종목 심층 분석
        if self.has_gpt and report.notable_stocks:
            self._log("Running deep analysis with GPT...")
            await self._deep_analyze_stocks(report.notable_stocks, analysis_result)

        # 8. Claude로 종합 분석 및 제안서 작성
        if self.has_claude:
            self._log("Generating comprehensive analysis with Claude...")
            report.claude_analysis = await self._claude_analysis(analysis_result, report)

        # 9. GPT로 최종 권고 생성
        if self.has_gpt:
            self._log("Generating recommendations with GPT...")
            report.gpt_recommendations = await self._gpt_recommendations(analysis_result, report)

        # 10. 섹터/산업군 추천 생성
        if self.has_gpt:
            self._log("Generating sector recommendations with GPT...")
            report.sector_recommendations = await self._generate_sector_recommendations(analysis_result, report)

        # 11. 최종 제안 종합
        report.final_recommendation = self._synthesize_final_recommendation(analysis_result, report)
        report.action_items = self._generate_action_items(analysis_result, report)
        report.risk_warnings = self._generate_risk_warnings(analysis_result, report)

        # 11.5. 진입/청산 전략 생성 (NEW)
        self._log("Generating entry/exit strategy...")
        report.entry_exit_strategy = self._generate_entry_exit_strategy(analysis_result, report, market_data)

        # 12. 히스토리컬 비교 (NEW)
        self._log("Comparing with previous report...")
        previous_report = self._load_previous_report()
        if previous_report:
            report.historical_comparison = self._compare_with_previous(
                analysis_result, report, previous_report
            )
            if report.historical_comparison.change_significance == "MAJOR":
                self._log(f"⚠️ Major change detected: {report.historical_comparison.key_changes}")
        else:
            self._log("No previous report found for comparison")

        # 13. 백테스팅 섹션 (유사 레짐 분석)
        self._log("Generating backtest section (similar regime analysis)...")
        try:
            # Legacy module was removed during cleanup. Keep this optional.
            from lib.regime_history import add_backtest_section_to_report
            report.backtest_section = add_backtest_section_to_report(report.to_dict())
            self._log("Backtest section generated successfully")
        except Exception:
            self._log("Backtest section skipped (legacy regime-history module unavailable)")
            report.backtest_section = ""

        # 14. 옵션/센티먼트 분석 (NEW)
        self._log("Analyzing options and sentiment...")
        try:
            report.options_analysis, report.sentiment_analysis = await self._analyze_options_sentiment()
            self._log("Options/Sentiment analysis completed")
        except Exception as e:
            self._log(f"Options/Sentiment analysis failed: {e}")
            report.options_analysis = None
            report.sentiment_analysis = None

        # 15. 데이터 소스 설정
        report.data_sources = [
            f"Yahoo Finance (시장 데이터, {datetime.now().strftime('%Y-%m-%d %H:%M')} 기준)",
            "FRED - Federal Reserve Economic Data (유동성/금리 데이터)",
            "Perplexity AI (실시간 뉴스 검색)",
            "OpenAI GPT-4o (투자 분석 및 권고)",
            "Anthropic Claude Sonnet (종합 분석)"
        ]

        self._log("Report generation complete!")
        return report

    async def generate_ib_report(
        self,
        analysis_result: Dict[str, Any],
        market_data: Dict[str, Any] = None
    ) -> str:
        """Investment Banking Style Report (Memorandum) 생성"""
        self._log("Generating IB-style memorandum...")
        
        # 1. 데이터 추출
        explanation = analysis_result.get('market_explanation', {})
        shap_narrative = explanation.get('narrative', '데이터 부족으로 분석 불가')
        drivers = explanation.get('drivers', [])
        
        # 2. SHAP 설명 포맷팅
        shap_text = self._format_causal_explanation(explanation)
        
        # 3. 프롬프트 생성
        prompt = self._build_ib_prompt(analysis_result, shap_text, drivers)
        
        # 4. LLM 호출
        report_content = ""
        if self.has_claude:
            report_content = await self._call_claude_ib(prompt)
        elif self.has_gpt:
            report_content = await self._call_gpt_ib(prompt)
        else:
            report_content = "LLM API가 설정되지 않아 IB 리포트를 생성할 수 없습니다."
            
        return report_content

    async def save_ib_report(self, content: str) -> str:
        """IB 리포트 파일 저장"""
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"ib_memorandum_{timestamp_str}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
            
        self._log(f"IB Report saved to {filename}")
        return str(filename)

    def _format_causal_explanation(self, explanation: Dict) -> str:
        """SHAP 설명을 자연어로 변환"""
        if not explanation or "error" in explanation:
            return "시장 데이터 부족으로 인과관계 분석을 수행할 수 없습니다."
            
        drivers = explanation.get('drivers', [])
        prediction = explanation.get('prediction', 0.0)
        
        lines = []
        direction = "상승" if prediction > 0 else "하락"
        lines.append(f"AI 모델은 익일 시장의 {direction}({prediction:+.2f}%) 가능성을 예측하고 있습니다.")
        lines.append("주요 원인은 다음과 같습니다:")
        
        for d in drivers:
            impact = d.get('impact', 0)
            desc = d.get('description', d.get('name', 'Unknown'))
            lines.append(f"- **{desc}**: {impact:+.2f}% 기여")
            
        return "\n".join(lines)

    def _format_new_metrics(self, result: Dict) -> str:
        """새로운 분석 지표(PoI, DTW, HFT) 요약"""
        analyses = result.get('analyses', {})
        lines = []

        # 1. Proof-of-Index
        poi = analyses.get('proof_of_index', {})
        if poi.get('completed'):
            summary = poi.get('summary', {})
            # summary가 문자열이면 eval 시도, 아니면 딕셔너리로 간주
            if isinstance(summary, str):
                try:
                    # 안전하지 않을 수 있으나 내부 데이터라 가정
                    pass 
                except:
                    pass
            # 실제 데이터 구조에 따라 파싱 (legacy 구조 포함)
            # JSON 로드 시 딕셔너리로 들어옴
            
            # PoI 상세 데이터가 analyses['proof_of_index'] 자체에 있을 수도 있음 (구조 확인 필요)
            # legacy 출력은 'summary' 키에 문자열 요약을 넣거나, 전체 데이터를 넣음.
            # 여기서는 전체 데이터를 가정하고 접근
            pass 

        # legacy 결과 구조:
        # results['analyses']['proof_of_index'] = {...}
        
        # 1. HFT Microstructure
        hft = analyses.get('hft_microstructure', {})
        if hft:
            lines.append("**HFT 미세구조:**")
            if 'tick_rule' in hft:
                buy_ratio = hft['tick_rule'].get('buy_ratio', 0.5)
                lines.append(f"- 매수 압력: {buy_ratio:.1%} ({'매수 우위' if buy_ratio > 0.55 else '매도 우위' if buy_ratio < 0.45 else '중립'})")
            if 'kyles_lambda' in hft:
                k_lambda = hft['kyles_lambda'].get('lambda', 0)
                lines.append(f"- 시장 충격(Kyle's λ): {k_lambda:.6f}")

        # 2. Information Flow
        info = analyses.get('information_flow', {})
        if info:
            lines.append("\n**정보 플로우:**")
            if 'abnormal_volume' in info:
                ab_vol = info['abnormal_volume']
                lines.append(f"- 이상 거래일: {ab_vol.get('total_abnormal_days', 0)}일 ({ab_vol.get('interpretation', '')})")
            
            # CAPM Alpha (QQQ 예시)
            capm = info.get('capm_QQQ', {})
            if capm:
                alpha = capm.get('alpha', 0) * 252
                lines.append(f"- QQQ Alpha: {alpha:+.1%}/yr (정보 우위 추정)")

        # 3. Proof-of-Index
        poi = analyses.get('proof_of_index', {})
        if poi:
            lines.append("\n**Proof-of-Index (투명성):**")
            snapshot = poi.get('index_snapshot', {})
            if snapshot:
                lines.append(f"- 인덱스 가치: {snapshot.get('index_value', 0):.2f}")
            verify = poi.get('verification', {})
            if verify:
                lines.append(f"- 블록체인 검증: {'✅ PASS' if verify.get('is_valid') else '❌ FAIL'}")
            signal = poi.get('mean_reversion_signal', {})
            if signal:
                lines.append(f"- 전략 신호: {signal.get('signal', 'N/A')} (Z={signal.get('z_score', 0):.2f})")

        # 4. DTW Similarity
        dtw = analyses.get('dtw_similarity', {})
        if dtw:
            lines.append("\n**시계열 유사도 (DTW):**")
            lead_lag = dtw.get('lead_lag_spy_qqq', {})
            if lead_lag:
                lines.append(f"- 리드-래그: {lead_lag.get('interpretation', 'N/A')}")
            sim_pair = dtw.get('most_similar_pair', {})
            if sim_pair:
                lines.append(f"- 최다 유사 쌍: {sim_pair.get('asset1')} ↔ {sim_pair.get('asset2')}")

        # 5. ARK Invest
        ark = result.get('ark_analysis', {})
        if ark:
            lines.append("\n**ARK Invest (Smart Money Flow):**")
            if ark.get('consensus_buys'):
                lines.append(f"- Consensus BUY: {', '.join(ark['consensus_buys'])} (다수 ETF 매수)")
            if ark.get('consensus_sells'):
                lines.append(f"- Consensus SELL: {', '.join(ark['consensus_sells'])} (다수 ETF 매도)")
            if ark.get('new_positions'):
                lines.append(f"- 신규 편입: {', '.join(ark['new_positions'])}")

        # 6. Extended Metrics
        ext = result.get('extended_data', {})
        if ext:
            lines.append("\n**Extended Market Metrics (Valuation & Sentiment):**")
            pcr = ext.get('put_call_ratio', {})
            if pcr: lines.append(f"- Put/Call Ratio: {pcr.get('ratio', 0.0):.2f} ({pcr.get('sentiment')})")
            
            fund = ext.get('fundamentals', {})
            if fund: lines.append(f"- SP500 Earnings Yield: {fund.get('earnings_yield', 0.0):.2f}%")
            
            stable = ext.get('digital_liquidity', {})
            if stable: lines.append(f"- Stablecoin Market Cap: ${stable.get('total_mcap', 0)/1e9:.1f}B")

        return "\n".join(lines)

    def _build_ib_prompt(self, result: Dict, shap_text: str, drivers: List) -> str:
        """IB 스타일 프롬프트 구성"""
        regime = result.get('regime', {})
        risk_score = result.get('risk_score', 50)
        fred = result.get('fred_summary', {})
        
        # 새로운 지표 포맷팅
        new_metrics_text = self._format_new_metrics(result)
        
        prompt = f"""
당신은 골드만삭스나 모건스탠리의 수석 전략가입니다.
기관 투자자를 위한 "Daily Investment Memorandum"을 작성해야 합니다.

다음 시장 데이터를 바탕으로 전문적인 보고서를 작성하십시오.

## 1. 시장 상황 데이터
- 레짐: {regime.get('regime', 'Unknown')} (신뢰도 {regime.get('confidence', 0)*100:.0f}%)
- 리스크 점수: {risk_score:.1f}/100
- 금리: Fed Funds {fred.get('fed_funds', 0):.2f}%, 10Y {fred.get('treasury_10y', 0):.2f}%
- 유동성: Net Liquidity ${fred.get('net_liquidity', 0):.0f}B ({fred.get('liquidity_regime', 'Unknown')})

## 2. 심층 정량 분석 (New Metrics)
{new_metrics_text}

## 3. AI 인과관계 분석 (Why-Based)
{shap_text}

## 4. 작성 지침
보고서는 다음 목차를 엄격히 준수하여 작성하십시오:

# EIMAS Daily Investment Memorandum

## 1. Investment Highlights (The "Alpha")
- 단순히 시장 방향을 나열하지 말고, **"Why"**에 집중하십시오.
- "심층 정량 분석" 섹션의 **DTW 리드-래그**, **HFT 매수 압력**, **PoI 신호**를 반드시 인용하여 분석 깊이를 더하십시오.
- 예: "SPY가 QQQ를 1일 선행한다는 DTW 분석 결과는 현재 기술주 주도의 장세가..."
- 예: "HFT 매수 압력이 57%로 우위를 점하며 단기 수급이 견조함을 시사..."

## 2. Key Risk Factors (Quantitative)
- 리스크 점수와 레짐 신뢰도를 언급하십시오.
- **정보 플로우(Information Flow)** 분석의 이상 거래일 여부를 언급하여 내부자 거래/정보 비대칭 리스크를 평가하십시오.
- Kyle's Lambda 값을 인용하여 시장 충격 비용(유동성 리스크)을 언급하십시오.

## 3. Valuation & Liquidity Logic
- **Proof-of-Index**의 밸류에이션(Mean Reversion Z-score)을 기반으로 현재 가격의 적정성을 논하십시오.
- Net Liquidity 및 Digital M2 관점과 결합하십시오.

## 4. Strategic Recommendation
- 기관 투자자를 위한 구체적인 액션 플랜(Overweight/Underweight)을 제시하십시오.
- 단순 매수/매도가 아닌, "조정 시 매수", "변동성 매도" 등 구조적 전략을 제안하십시오.

**톤앤매너:**
- 매우 전문적이고 드라이한 IB(Investment Banking) 스타일
- 명확한 근거 제시 (Data-Driven)
- 불필요한 미사여구 제거
"""
        return prompt

    async def _call_claude_ib(self, prompt: str) -> str:
        try:
            client = APIConfig.get_client('anthropic')
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            self._log(f"Claude IB generation failed: {e}")
            return f"Error generating IB report: {e}"

    async def _call_gpt_ib(self, prompt: str) -> str:
        try:
            client = APIConfig.get_client('openai')
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Chief Market Strategist at a top-tier investment bank."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            self._log(f"GPT IB generation failed: {e}")
            return f"Error generating IB report: {e}"

    def _create_market_summary(self, result: Dict) -> str:
        """시장 요약 생성"""
        fred = result.get('fred_summary', {})
        regime = result.get('regime', {})

        summary = f"""
현재 시장은 **{regime.get('regime', 'Unknown')}** 레짐에 있으며,
추세는 **{regime.get('trend', 'Unknown')}**, 변동성은 **{regime.get('volatility', 'Unknown')}** 수준입니다.

**유동성 현황:**
- RRP: ${fred.get('rrp', 0):.1f}B (Δ{fred.get('rrp_delta', 0):+.1f}B)
- TGA: ${fred.get('tga', 0):.1f}B (Δ{fred.get('tga_delta', 0):+.1f}B)
- Net Liquidity: ${fred.get('net_liquidity', 0):.1f}B ({fred.get('liquidity_regime', 'Unknown')})

**금리 환경:**
- Fed Funds: {fred.get('fed_funds', 0):.2f}%
- 10Y Treasury: {fred.get('treasury_10y', 0):.2f}%
- 10Y-2Y Spread: {fred.get('spread_10y2y', 0):.2f}% ({fred.get('curve_status', 'Unknown')})
"""
        return summary.strip()

    def _create_regime_analysis(self, result: Dict) -> str:
        """레짐 분석"""
        regime = result.get('regime', {})
        conf = regime.get('confidence', 0)
        if isinstance(conf, float) and conf <= 1:
            conf *= 100

        return f"""
**현재 레짐**: {regime.get('regime', 'Unknown')}
**신뢰도**: {conf:.0f}%
**설명**: {regime.get('description', 'N/A')}
**권장 전략**: {regime.get('strategy', 'N/A')}

두 분석 모드(FULL/REFERENCE) 결과:
- FULL Mode: {result.get('full_mode_position', 'NEUTRAL')}
- REFERENCE Mode: {result.get('reference_mode_position', 'NEUTRAL')}
- 모드 일치: {'예' if result.get('modes_agree', False) else '아니오'}
"""

    def _create_risk_assessment(self, result: Dict) -> str:
        """리스크 평가"""
        risk_score = result.get('risk_score', 0)
        warnings = result.get('warnings', [])

        if risk_score < 20:
            risk_level = "매우 낮음"
            risk_color = "🟢"
        elif risk_score < 40:
            risk_level = "낮음"
            risk_color = "🟢"
        elif risk_score < 60:
            risk_level = "보통"
            risk_color = "🟡"
        elif risk_score < 80:
            risk_level = "높음"
            risk_color = "🟠"
        else:
            risk_level = "매우 높음"
            risk_color = "🔴"

        assessment = f"""
{risk_color} **리스크 점수**: {risk_score:.1f}/100 ({risk_level})

**최종 권고**: {result.get('final_recommendation', 'NEUTRAL')}
**신뢰도**: {result.get('confidence', 0.5)*100:.0f}%
"""

        if warnings:
            assessment += "\n**경고:**\n"
            for w in warnings:
                assessment += f"- ⚠️ {w}\n"

        return assessment

    def _create_confidence_analysis(self, result: Dict) -> str:
        """신뢰도 불일치 분석"""
        regime_conf = result.get('regime', {}).get('confidence', 0)
        if isinstance(regime_conf, float) and regime_conf <= 1:
            regime_conf *= 100

        final_conf = result.get('confidence', 0.5) * 100
        risk_score = result.get('risk_score', 50)

        # 신뢰도 차이 계산
        conf_diff = regime_conf - final_conf

        analysis_parts = []

        if abs(conf_diff) > 5:
            analysis_parts.append(f"레짐 신뢰도({regime_conf:.0f}%)와 최종 권고 신뢰도({final_conf:.0f}%)에 **{abs(conf_diff):.0f}%p 차이**가 있습니다.")

            # 차이 원인 분석
            reasons = []

            # 1. 모드 불일치
            if not result.get('modes_agree', True):
                reasons.append("FULL/REFERENCE 모드 간 의견 불일치로 인한 신뢰도 감소")

            # 2. 리스크 점수 영향
            if risk_score > 40:
                reasons.append(f"리스크 점수({risk_score:.1f}/100)가 상승하여 신뢰도 조정")

            # 3. 반대 의견 존재
            if result.get('has_strong_dissent', False):
                reasons.append("에이전트 간 강한 반대의견 존재")

            # 4. 유동성 신호
            liquidity_signal = result.get('liquidity_signal', 'NEUTRAL')
            if liquidity_signal != 'NEUTRAL':
                reasons.append(f"유동성 신호({liquidity_signal})가 레짐 분석과 상충")

            # 5. 최종 신뢰도는 여러 요소의 가중평균
            if not reasons:
                # 설명이 없을 때 기본 설명 제공
                reasons.append("최종 신뢰도는 레짐 신뢰도, 에이전트 합의도, 시장 변동성 등을 종합하여 산출")
                reasons.append(f"레짐 탐지 신뢰도: {regime_conf:.0f}%")
                if result.get('modes_agree', True):
                    reasons.append("FULL/REFERENCE 모드 일치 (+신뢰도)")
                reasons.append(f"리스크 점수 {risk_score:.1f}/100 반영")

            analysis_parts.append("\n**신뢰도 산출 요인:**")
            for reason in reasons:
                analysis_parts.append(f"- {reason}")
        else:
            analysis_parts.append(f"레짐 신뢰도({regime_conf:.0f}%)와 최종 권고 신뢰도({final_conf:.0f}%)가 일관성 있게 유지되고 있습니다.")

        # 신뢰도 해석
        analysis_parts.append("")
        if final_conf >= 70:
            analysis_parts.append("✅ **높은 신뢰도**: 분석 결과에 대한 확신이 높습니다.")
        elif final_conf >= 50:
            analysis_parts.append("⚠️ **중간 신뢰도**: 신중한 접근이 권장됩니다. 시장 지표를 지속 모니터링하세요.")
        else:
            analysis_parts.append("❗ **낮은 신뢰도**: 추가 확인 후 의사결정을 권장합니다.")

        return "\n".join(analysis_parts)

    def _calculate_technical_indicators(
        self,
        market_data: Dict,
        result: Dict
    ) -> Optional[TechnicalIndicators]:
        """기술적 지표 계산"""
        try:
            import numpy as np
            import pandas as pd

            # SPY 데이터 사용 (S&P 500 대용)
            spy_data = market_data.get('SPY')
            if spy_data is None or (isinstance(spy_data, pd.DataFrame) and spy_data.empty):
                spy_data = market_data.get('^GSPC')

            vix_data = market_data.get('^VIX')

            # DataFrame 체크
            if spy_data is None:
                return None
            if isinstance(spy_data, pd.DataFrame) and spy_data.empty:
                return None
            if not hasattr(spy_data, 'iloc'):
                return None

            close = spy_data['Close']
            current_price = close.iloc[-1]

            # VIX
            vix = 0.0
            vix_change = 0.0
            if vix_data is not None:
                if isinstance(vix_data, pd.DataFrame) and not vix_data.empty and 'Close' in vix_data.columns:
                    vix_close = vix_data['Close']
                    vix = float(vix_close.iloc[-1])
                    if len(vix_close) >= 2:
                        vix_change = ((vix_close.iloc[-1] / vix_close.iloc[-2]) - 1) * 100

            # RSI (14일)
            rsi_14 = 50.0
            if len(close) >= 15:
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, np.inf)
                rsi = 100 - (100 / (1 + rs))
                rsi_14 = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0

            # MACD (12, 26, 9)
            macd = 0.0
            macd_signal = 0.0
            if len(close) >= 35:
                ema_12 = close.ewm(span=12, adjust=False).mean()
                ema_26 = close.ewm(span=26, adjust=False).mean()
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                macd = macd_line.iloc[-1]
                macd_signal = signal_line.iloc[-1]

            # 이동평균
            ma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else current_price
            ma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else current_price

            # 지지/저항선 (20일 최저/최고)
            support_level = close.tail(20).min() if len(close) >= 20 else current_price * 0.95
            resistance_level = close.tail(20).max() if len(close) >= 20 else current_price * 1.05

            return TechnicalIndicators(
                vix=vix,
                vix_change=vix_change,
                rsi_14=rsi_14,
                macd=macd,
                macd_signal=macd_signal,
                ma_50=ma_50,
                ma_200=ma_200,
                current_price=current_price,
                support_level=support_level,
                resistance_level=resistance_level
            )

        except Exception as e:
            logger.warning(f"Technical indicators calculation failed: {e}")
            return None

    def _generate_scenarios(
        self,
        result: Dict,
        report: FinalReport
    ) -> List[ScenarioCase]:
        """시나리오 분석 생성"""
        regime = result.get('regime', {}).get('regime', 'Unknown')
        risk_score = result.get('risk_score', 50)
        position = result.get('final_recommendation', 'NEUTRAL')

        scenarios = []

        # Base Case
        if 'Bull' in regime:
            base_prob = 55
            base_return = "+8% ~ +12%"
            base_target = "7,200 ~ 7,400"
            base_strategy = "현재 포지션 유지, 조정 시 추가 매수"
        elif 'Bear' in regime:
            base_prob = 50
            base_return = "-5% ~ +2%"
            base_target = "6,200 ~ 6,600"
            base_strategy = "현금 비중 확대, 방어주 선호"
        else:
            base_prob = 50
            base_return = "+3% ~ +7%"
            base_target = "6,800 ~ 7,100"
            base_strategy = "분산 투자, 점진적 리밸런싱"

        scenarios.append(ScenarioCase(
            name="Base Case (기본 시나리오)",
            probability=base_prob,
            expected_return=base_return,
            sp500_target=base_target,
            strategy=base_strategy,
            key_triggers=[
                "현재 경제 지표 추세 유지",
                "Fed 정책 예상대로 진행",
                "기업 실적 컨센서스 부합"
            ]
        ))

        # Bull Case
        if 'Bull' in regime:
            bull_prob = 30
            bull_return = "+15% ~ +20%"
            bull_target = "7,600 ~ 8,000"
        else:
            bull_prob = 25
            bull_return = "+12% ~ +18%"
            bull_target = "7,400 ~ 7,800"

        scenarios.append(ScenarioCase(
            name="Bull Case (강세 시나리오)",
            probability=bull_prob,
            expected_return=bull_return,
            sp500_target=bull_target,
            strategy="주식 비중 최대 확대, 성장주/소형주 집중, 레버리지 ETF 활용",
            key_triggers=[
                "인플레이션 예상보다 빠른 안정화",
                "Fed 금리 인하 가속화",
                "AI 생산성 향상 가시화",
                "중국 경기 부양책 효과"
            ]
        ))

        # Bear Case
        if 'Bear' in regime:
            bear_prob = 30
            bear_return = "-15% ~ -25%"
            bear_target = "5,400 ~ 5,800"
        else:
            bear_prob = 15
            bear_return = "-10% ~ -15%"
            bear_target = "5,800 ~ 6,200"

        scenarios.append(ScenarioCase(
            name="Bear Case (약세 시나리오)",
            probability=bear_prob,
            expected_return=bear_return,
            sp500_target=bear_target,
            strategy="주식 비중 최소화, 현금/채권 확대, 인버스 ETF 헤지",
            key_triggers=[
                "인플레이션 재상승",
                "Fed 긴축 재개",
                "경기 침체 진입",
                "지정학적 리스크 확대",
                "신용 위기 발생"
            ]
        ))

        return scenarios

    def _explain_no_notable_stocks(
        self,
        market_data: Dict,
        result: Dict
    ) -> str:
        """주목할 종목이 없는 이유 설명"""
        regime = result.get('regime', {}).get('regime', 'Unknown')
        volatility = result.get('regime', {}).get('volatility', 'Unknown')

        explanations = []

        # 레짐 기반 설명
        if 'Low Vol' in regime or volatility == 'Low':
            explanations.append("현재 저변동성 레짐으로 개별 종목의 급격한 움직임이 제한적입니다.")

        # 시장 안정성
        explanations.append("\n**분석 기준:**")
        explanations.append("- 1일 변동률 ±3% 이상")
        explanations.append("- 5일 변동률 ±7% 이상")
        explanations.append("- 일일 변동성 3% 이상")
        explanations.append("- 추세 전환 신호 (20일/5일 반대 방향)")

        # 해석
        explanations.append("\n**해석:**")
        explanations.append("위 기준을 충족하는 종목이 없다는 것은 시장이 전반적으로 안정적인 상태임을 나타냅니다.")
        explanations.append("이는 Bull (Low Vol) 레짐의 특성과 일치하며, 급격한 가격 변동보다는")
        explanations.append("점진적인 상승 추세가 유지되고 있음을 시사합니다.")

        return "\n".join(explanations)

    def _find_notable_stocks(self, market_data: Dict) -> List[StockAnalysis]:
        """주목할 종목 찾기"""
        notable = []

        for ticker, df in market_data.items():
            if not hasattr(df, 'iloc') or len(df) < 20:
                continue

            try:
                close = df['Close']

                # 변화율 계산
                change_1d = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100 if len(close) >= 2 else 0
                change_5d = ((close.iloc[-1] / close.iloc[-5]) - 1) * 100 if len(close) >= 5 else 0
                change_20d = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100 if len(close) >= 20 else 0

                # 변동성 (20일 표준편차)
                returns = close.pct_change().dropna()
                volatility = returns.tail(20).std() * 100 if len(returns) >= 20 else 0

                # 주목할만한 변화 판단
                is_notable = False
                notable_reason = ""

                # 1일 급등/급락 (3% 이상)
                if abs(change_1d) >= 3:
                    is_notable = True
                    direction = "급등" if change_1d > 0 else "급락"
                    notable_reason = f"1일 {direction} ({change_1d:+.1f}%)"

                # 5일 큰 변화 (7% 이상)
                elif abs(change_5d) >= 7:
                    is_notable = True
                    direction = "상승" if change_5d > 0 else "하락"
                    notable_reason = f"5일간 큰 {direction} ({change_5d:+.1f}%)"

                # 높은 변동성
                elif volatility >= 3:
                    is_notable = True
                    notable_reason = f"높은 변동성 (일일 {volatility:.1f}%)"

                # 추세 전환 (20일 대비 5일이 반대 방향)
                elif change_20d * change_5d < 0 and abs(change_5d) >= 3:
                    is_notable = True
                    notable_reason = "추세 전환 가능성"

                if is_notable:
                    notable.append(StockAnalysis(
                        ticker=ticker,
                        change_1d=change_1d,
                        change_5d=change_5d,
                        change_20d=change_20d,
                        volatility=volatility,
                        is_notable=True,
                        notable_reason=notable_reason
                    ))

            except Exception as e:
                logger.warning(f"Error analyzing {ticker}: {e}")

        # 변화율 절대값 기준 정렬
        notable.sort(key=lambda x: abs(x.change_1d), reverse=True)
        return notable[:5]  # 상위 5개만

    async def _search_news(
        self,
        result: Dict,
        notable_stocks: List[StockAnalysis]
    ) -> str:
        """Perplexity로 최신 뉴스 검색"""
        try:
            client = APIConfig.get_client('perplexity')

            # 검색 쿼리 구성
            tickers = [s.ticker for s in notable_stocks[:3]]
            regime = result.get('regime', {}).get('regime', 'Unknown')

            query = f"""
다음 주제에 대한 최신 뉴스와 시장 이벤트를 검색해주세요:

1. 현재 미국 주식시장 상황 및 전망 (현재 {regime} 레짐)
2. Fed 통화정책 및 금리 전망
3. 주요 경제 지표 발표 일정
"""
            if tickers:
                query += f"\n4. 다음 종목들의 최근 뉴스: {', '.join(tickers)}"

            response = client.chat.completions.create(
                model="sonar-pro",
                messages=[
                    {"role": "system", "content": "You are a financial news analyst. Provide concise, relevant market news in Korean."},
                    {"role": "user", "content": query}
                ],
                max_tokens=2000,
                temperature=0.1
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Perplexity search failed: {e}")
            return f"뉴스 검색 실패: {e}"

    async def _deep_analyze_stocks(
        self,
        notable_stocks: List[StockAnalysis],
        result: Dict
    ):
        """GPT로 특정 종목 심층 분석"""
        try:
            client = APIConfig.get_client('openai')

            for stock in notable_stocks[:3]:  # 상위 3개만
                prompt = f"""
다음 종목에 대해 심층 분석해주세요:

종목: {stock.ticker}
1일 변화: {stock.change_1d:+.2f}%
5일 변화: {stock.change_5d:+.2f}%
20일 변화: {stock.change_20d:+.2f}%
변동성: {stock.volatility:.2f}%
주목 이유: {stock.notable_reason}

현재 시장 레짐: {result.get('regime', {}).get('regime', 'Unknown')}
리스크 점수: {result.get('risk_score', 0):.1f}/100

이 종목의 최근 움직임에 대한 가능한 원인과 향후 전망을 간략히 분석해주세요.
한국어로 3-4문장으로 답변해주세요.
"""

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a professional stock analyst. Provide concise analysis in Korean."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )

                stock.deep_analysis = response.choices[0].message.content

        except Exception as e:
            logger.error(f"GPT deep analysis failed: {e}")

    async def _claude_analysis(
        self,
        result: Dict,
        report: FinalReport
    ) -> str:
        """Claude로 종합 분석"""
        try:
            client = APIConfig.get_client('anthropic')

            # 분석 컨텍스트 구성
            context = f"""
## 분석 데이터

### 시장 요약
{report.market_summary}

### 레짐 분석
{report.regime_analysis}

### 리스크 평가
{report.risk_assessment}

### 주목할 종목
"""
            for stock in report.notable_stocks:
                context += f"- {stock.ticker}: {stock.notable_reason}\n"

            if report.perplexity_news:
                context += f"\n### 최신 뉴스\n{report.perplexity_news[:1500]}"

            prompt = f"""
위 데이터를 바탕으로 종합적인 시장 분석을 제공해주세요.

다음 내용을 포함해주세요:
1. 현재 시장 상황 해석
2. 주요 리스크 요인
3. 기회 요인
4. 섹터/자산군별 전망
5. 투자자 유형별 권고사항

한국어로 작성하고, 전문적이면서도 이해하기 쉽게 설명해주세요.

중요: 섹션 제목은 ### (3개)를 사용하세요. ## (2개)는 사용하지 마세요.
예시: ### 1. 현재 시장 상황 해석
"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                messages=[
                    {"role": "user", "content": context + "\n\n" + prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            return f"Claude 분석 실패: {e}"

    async def _gpt_recommendations(
        self,
        result: Dict,
        report: FinalReport
    ) -> str:
        """GPT로 투자 권고 생성"""
        try:
            client = APIConfig.get_client('openai')

            position = result.get('final_recommendation', 'NEUTRAL')
            confidence = result.get('confidence', 0.5) * 100
            risk_score = result.get('risk_score', 50)

            prompt = f"""
현재 시장 상황:
- 최종 포지션: {position}
- 신뢰도: {confidence:.0f}%
- 리스크 점수: {risk_score:.1f}/100
- 레짐: {result.get('regime', {}).get('regime', 'Unknown')}

위 분석 결과를 바탕으로 구체적인 투자 권고를 제시해주세요:

1. 자산배분 권고 (주식/채권/현금 비율)
2. 섹터 선호도 (Overweight/Neutral/Underweight)
3. 구체적인 ETF 또는 종목 제안 (있다면)
4. 진입/청산 전략
5. 리스크 관리 방안

한국어로 간결하게 작성해주세요.
"""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional investment advisor. Provide actionable recommendations in Korean."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"GPT recommendations failed: {e}")
            return f"GPT 권고 생성 실패: {e}"

    async def _generate_sector_recommendations(
        self,
        result: Dict,
        report: FinalReport
    ) -> Dict[str, Any]:
        """섹터/산업군 추천 생성"""
        try:
            client = APIConfig.get_client('openai')

            position = result.get('final_recommendation', 'NEUTRAL')
            regime = result.get('regime', {}).get('regime', 'Unknown')
            risk_score = result.get('risk_score', 50)
            confidence = result.get('confidence', 0.5) * 100
            fred = result.get('fred_summary', {})

            # 뉴스 컨텍스트 추가
            news_context = ""
            if report.perplexity_news:
                news_context = f"\n최신 뉴스 요약:\n{report.perplexity_news[:1000]}"

            prompt = f"""
현재 시장 상황 분석 결과를 바탕으로 섹터 및 산업군 추천을 JSON 형식으로 작성해주세요.

## 시장 상황
- 최종 포지션: {position}
- 시장 레짐: {regime}
- 리스크 점수: {risk_score:.1f}/100
- 신뢰도: {confidence:.0f}%
- 금리 환경: Fed Funds {fred.get('fed_funds', 0):.2f}%, 10Y {fred.get('treasury_10y', 0):.2f}%
- 유동성: Net Liquidity ${fred.get('net_liquidity', 0):.1f}B ({fred.get('liquidity_regime', 'Unknown')})
{news_context}

## 요청 형식 (반드시 JSON으로 응답)
{{
  "bullish_sectors": [
    {{"name": "섹터명", "rating": "Overweight", "rationale": "근거", "etfs": ["ETF1", "ETF2"]}}
  ],
  "neutral_sectors": [
    {{"name": "섹터명", "rating": "Neutral", "rationale": "근거"}}
  ],
  "bearish_sectors": [
    {{"name": "섹터명", "rating": "Underweight", "rationale": "근거"}}
  ],
  "hot_industries": [
    {{"name": "산업군명", "description": "주목 이유"}}
  ],
  "ai_analysis": "현재 시장 상황에서 섹터 전략에 대한 종합적인 분석 (3-5문장)"
}}

주요 섹터: 기술(XLK), 헬스케어(XLV), 금융(XLF), 에너지(XLE), 소비재(XLY), 필수소비재(XLP), 유틸리티(XLU), 산업재(XLI), 소재(XLB), 부동산(XLRE), 통신(XLC)
각 카테고리에 2-3개 섹터를 포함하고 현재 시장 상황에 맞는 분석을 제공해주세요.
"""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional sector analyst. Always respond with valid JSON only, no markdown formatting. Use Korean for descriptions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )

            # JSON 파싱
            response_text = response.choices[0].message.content
            # JSON 블록 추출 (```json ... ``` 형식인 경우)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            import json
            sector_data = json.loads(response_text.strip())

            return sector_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            # 기본 섹터 추천 반환
            return self._get_default_sector_recommendations(result)
        except Exception as e:
            logger.error(f"Sector recommendations failed: {e}")
            return self._get_default_sector_recommendations(result)

    def _get_default_sector_recommendations(self, result: Dict) -> Dict[str, Any]:
        """기본 섹터 추천 (API 실패 시)"""
        position = result.get('final_recommendation', 'NEUTRAL')
        regime = result.get('regime', {}).get('regime', 'Unknown')

        if position == 'BULLISH' or 'Bull' in regime:
            return {
                "bullish_sectors": [
                    {"name": "기술 (Technology)", "rating": "Overweight", "rationale": "Bull 레짐에서 성장주 선호", "etfs": ["XLK", "QQQ"]},
                    {"name": "소비재 (Consumer Discretionary)", "rating": "Overweight", "rationale": "경기 확장기 수혜", "etfs": ["XLY"]},
                    {"name": "산업재 (Industrials)", "rating": "Overweight", "rationale": "경제 성장 수혜", "etfs": ["XLI"]}
                ],
                "neutral_sectors": [
                    {"name": "금융 (Financials)", "rating": "Neutral", "rationale": "금리 환경 변화 주시 필요"},
                    {"name": "헬스케어 (Healthcare)", "rating": "Neutral", "rationale": "방어적 성격으로 상대적 저조"}
                ],
                "bearish_sectors": [
                    {"name": "유틸리티 (Utilities)", "rating": "Underweight", "rationale": "Bull 레짐에서 선호도 하락"},
                    {"name": "필수소비재 (Consumer Staples)", "rating": "Underweight", "rationale": "방어주 상대적 저조"}
                ],
                "hot_industries": [
                    {"name": "AI/반도체", "description": "생성형 AI 투자 확대 지속"},
                    {"name": "클라우드 컴퓨팅", "description": "기업 디지털 전환 가속화"},
                    {"name": "청정에너지", "description": "에너지 전환 정책 수혜"}
                ],
                "ai_analysis": f"현재 {regime} 레짐에서는 리스크 자산 선호가 적절합니다. 기술주와 성장주 중심의 포트폴리오가 유리하며, 방어주는 상대적으로 비중을 낮추는 것이 권장됩니다."
            }
        elif position == 'BEARISH' or 'Bear' in regime:
            return {
                "bullish_sectors": [
                    {"name": "유틸리티 (Utilities)", "rating": "Overweight", "rationale": "방어적 성격으로 약세장 선호", "etfs": ["XLU"]},
                    {"name": "필수소비재 (Consumer Staples)", "rating": "Overweight", "rationale": "경기 방어적 특성", "etfs": ["XLP"]},
                    {"name": "헬스케어 (Healthcare)", "rating": "Overweight", "rationale": "비경기 민감 섹터", "etfs": ["XLV"]}
                ],
                "neutral_sectors": [
                    {"name": "통신 (Communication Services)", "rating": "Neutral", "rationale": "배당 수익 + 성장 혼합"}
                ],
                "bearish_sectors": [
                    {"name": "기술 (Technology)", "rating": "Underweight", "rationale": "성장주 밸류에이션 부담"},
                    {"name": "소비재 (Consumer Discretionary)", "rating": "Underweight", "rationale": "경기 둔화 취약"},
                    {"name": "에너지 (Energy)", "rating": "Underweight", "rationale": "경기 민감 섹터"}
                ],
                "hot_industries": [
                    {"name": "헬스케어 방어주", "description": "경기 침체 방어"},
                    {"name": "배당주", "description": "안정적 수익 추구"},
                    {"name": "금", "description": "안전자산 선호"}
                ],
                "ai_analysis": f"현재 {regime} 레짐에서는 방어적 포지션이 권장됩니다. 유틸리티, 필수소비재 등 비경기 민감 섹터 중심으로 포트폴리오를 구성하고, 현금 비중을 높이는 것이 적절합니다."
            }
        else:
            return {
                "bullish_sectors": [
                    {"name": "헬스케어 (Healthcare)", "rating": "Overweight", "rationale": "방어적 성장 섹터", "etfs": ["XLV"]},
                    {"name": "배당 성장주", "rating": "Overweight", "rationale": "안정적 수익 + 성장", "etfs": ["VIG", "SCHD"]}
                ],
                "neutral_sectors": [
                    {"name": "기술 (Technology)", "rating": "Neutral", "rationale": "선별적 접근 필요"},
                    {"name": "금융 (Financials)", "rating": "Neutral", "rationale": "금리 방향성 주시"},
                    {"name": "산업재 (Industrials)", "rating": "Neutral", "rationale": "경기 지표 확인 필요"}
                ],
                "bearish_sectors": [
                    {"name": "고베타주", "rating": "Underweight", "rationale": "불확실성 기간 리스크 관리"}
                ],
                "hot_industries": [
                    {"name": "AI/반도체", "description": "장기 구조적 성장 테마"},
                    {"name": "바이오테크", "description": "혁신 의료 기술"},
                    {"name": "인프라", "description": "정부 투자 수혜"}
                ],
                "ai_analysis": f"현재 {regime} 레짐에서는 균형 잡힌 접근이 필요합니다. 성장과 방어 섹터를 적절히 혼합하고, 시장 방향성이 확인될 때까지 현금 비중을 유지하는 것이 권장됩니다."
            }

    def _synthesize_final_recommendation(
        self,
        result: Dict,
        report: FinalReport
    ) -> str:
        """최종 제안 종합"""
        position = result.get('final_recommendation', 'NEUTRAL')
        confidence = result.get('confidence', 0.5) * 100
        regime = result.get('regime', {}).get('regime', 'Unknown')
        risk_score = result.get('risk_score', 50)

        if position == 'BULLISH':
            stance = "적극적 매수"
            emoji = "📈"
        elif position == 'BEARISH':
            stance = "방어적 포지션"
            emoji = "📉"
        else:
            stance = "중립 유지"
            emoji = "➡️"

        return f"""
{emoji} **{stance}** (신뢰도: {confidence:.0f}%)

현재 시장은 {regime} 레짐에서 리스크 점수 {risk_score:.1f}/100 수준입니다.
두 분석 모드(FULL/REFERENCE)가 {'일치' if result.get('modes_agree') else '불일치'}하여
{'신호의 신뢰성이 높습니다.' if result.get('modes_agree') else '추가적인 확인이 필요합니다.'}
"""

    def _generate_action_items(
        self,
        result: Dict,
        report: FinalReport
    ) -> List[str]:
        """액션 아이템 생성"""
        items = []

        position = result.get('final_recommendation', 'NEUTRAL')
        risk_score = result.get('risk_score', 50)

        if position == 'BULLISH':
            items.append("주식 비중 확대 고려")
            items.append("성장주/소형주 비중 점검")
            if risk_score < 30:
                items.append("레버리지 ETF 검토 가능")
        elif position == 'BEARISH':
            items.append("현금 비중 확대")
            items.append("방어주/채권 비중 확대 검토")
            items.append("손절 라인 재점검")
        else:
            items.append("현 포지션 유지")
            items.append("시장 모니터링 강화")

        # 주목할 종목 관련
        for stock in report.notable_stocks[:2]:
            if stock.change_1d > 5:
                items.append(f"{stock.ticker}: 급등 후 조정 가능성 모니터링")
            elif stock.change_1d < -5:
                items.append(f"{stock.ticker}: 반등 기회 모니터링")

        return items

    def _generate_risk_warnings(
        self,
        result: Dict,
        report: FinalReport
    ) -> List[str]:
        """리스크 경고 생성"""
        warnings = list(result.get('warnings', []))

        risk_score = result.get('risk_score', 50)
        if risk_score > 60:
            warnings.append(f"⚠️ 높은 리스크 점수 ({risk_score:.1f}/100) - 포지션 축소 고려")

        if not result.get('modes_agree', True):
            warnings.append("⚠️ FULL/REFERENCE 모드 불일치 - 레짐 변화 가능성")

        if result.get('has_strong_dissent', False):
            warnings.append("⚠️ 에이전트 간 강한 반대의견 존재")

        # 유동성 경고
        fred = result.get('fred_summary', {})
        if fred.get('rrp_delta', 0) > 50:
            warnings.append("💧 RRP 급증 - 유동성 회수 가능성")
        if fred.get('tga_delta', 0) > 100:
            warnings.append("💧 TGA 급증 - 유동성 축소 가능성")

        # 기술적 지표 경고
        if report.technical_indicators:
            ti = report.technical_indicators
            if ti.rsi_14 and ti.rsi_14 > 70:
                warnings.append(f"📈 RSI 과매수 구간 ({ti.rsi_14:.1f}) - 단기 조정 가능성")
            elif ti.rsi_14 and ti.rsi_14 < 30:
                warnings.append(f"📉 RSI 과매도 구간 ({ti.rsi_14:.1f}) - 반등 기회 또는 추가 하락")
            if ti.vix and ti.vix > 25:
                warnings.append(f"😰 높은 변동성 (VIX: {ti.vix:.1f}) - 위험 관리 강화 필요")
            if ti.current_price and ti.resistance_level:
                if ti.current_price > ti.resistance_level * 0.98:
                    warnings.append("📊 저항선 근접 - 돌파 실패 시 조정 가능")

        # 글로벌 시장 경고
        if report.global_market:
            gm = report.global_market
            if gm.dxy_change and gm.dxy_change > 1.0:
                warnings.append(f"💵 달러 강세 ({gm.dxy_change:+.1f}%) - 신흥시장/원자재 압박")
            if gm.wti_change and gm.wti_change < -5.0:
                warnings.append(f"🛢️ 유가 급락 ({gm.wti_change:.1f}%) - 경기 둔화 시그널 가능")
            if gm.gold_change and gm.gold_change > 3.0:
                warnings.append(f"🥇 금 급등 ({gm.gold_change:.1f}%) - 안전자산 선호 증가")

        # 시나리오 기반 경고
        for scenario in report.scenarios:
            if scenario.name == "Bear Case (약세 시나리오)" and scenario.probability > 20:
                warnings.append(f"🐻 Bear Case 확률 상승 ({scenario.probability}%) - 헤지 전략 고려")

        return warnings

    async def _analyze_options_sentiment(self) -> tuple:
        """옵션 및 센티먼트 분석"""
        options_data = {}
        sentiment_data = {}

        try:
            from lib.sentiment_analyzer import SentimentAnalyzer
            analyzer = SentimentAnalyzer()

            # VIX Term Structure
            vts_result = analyzer.analyze_vix_term_structure()
            if vts_result:
                options_data['vix_term_structure'] = {
                    'structure': vts_result.structure.value,
                    'vix_spot': vts_result.vix_spot,
                    'vix_3m': vts_result.vix_3m,
                    'spread': getattr(vts_result, 'spread', getattr(vts_result, 'spread_pct', 0)),
                    'signal': getattr(vts_result, 'signal', getattr(vts_result, 'market_signal', 'NEUTRAL'))
                }

            # Put/Call Ratio
            pcr_result = analyzer.analyze_put_call_ratio("SPY")
            if pcr_result:
                options_data['put_call_ratio'] = {
                    'ratio': getattr(pcr_result, 'put_call_ratio', getattr(pcr_result, 'ratio', 0)),
                    'level': getattr(pcr_result, 'signal', getattr(pcr_result, 'level', 'NEUTRAL')),
                    'contrarian_signal': getattr(pcr_result, 'interpretation', getattr(pcr_result, 'contrarian_signal', ''))
                }

            # IV Percentile
            ivp_result = analyzer.calculate_iv_percentile("SPY")
            if ivp_result:
                options_data['iv_percentile'] = {
                    'percentile': ivp_result.iv_percentile,  # 올바른 속성명: iv_percentile
                    'current_iv': ivp_result.current_iv,
                    'level': ivp_result.signal  # 올바른 속성명: signal
                }

            # Fear & Greed Index
            full_analysis = analyzer.analyze()
            if full_analysis and full_analysis.composite:
                fg = full_analysis.composite.fear_greed
                sentiment_data['fear_greed_index'] = {
                    'value': fg.value if fg else 50,
                    'classification': fg.level.value if fg else 'neutral'
                }

        except Exception as e:
            self._log(f"Options/Sentiment analysis error: {e}")

        return options_data, sentiment_data

    async def _fetch_global_markets(self) -> GlobalMarketData:
        """국제 시장 데이터 수집"""
        import yfinance as yf

        gm = GlobalMarketData()

        try:
            # 심볼 정의
            symbols = {
                'dxy': 'DX-Y.NYB',        # 달러 인덱스
                'dax': '^GDAXI',          # 독일 DAX
                'ftse': '^FTSE',          # 영국 FTSE 100
                'nikkei': '^N225',        # 일본 Nikkei 225
                'shanghai': '000001.SS',   # 상하이 종합
                'kospi': '^KS11',          # 한국 KOSPI
                'gold': 'GC=F',            # 금
                'wti': 'CL=F',             # WTI 원유
                'copper': 'HG=F',          # 구리
            }

            # 데이터 가져오기
            for name, symbol in symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d')

                    if len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2]
                        change = ((current - previous) / previous) * 100

                        setattr(gm, name, current)
                        setattr(gm, f'{name}_change', change)
                except Exception as e:
                    self._log(f"Failed to fetch {name}: {e}")

            # 글로벌 심리 분석
            risk_on_count = 0
            risk_off_count = 0

            # 지수 상승 = Risk On
            for idx in ['dax_change', 'ftse_change', 'nikkei_change', 'kospi_change']:
                val = getattr(gm, idx, 0)
                if val > 0.5:
                    risk_on_count += 1
                elif val < -0.5:
                    risk_off_count += 1

            # 달러 하락 = Risk On
            if gm.dxy_change < -0.3:
                risk_on_count += 1
            elif gm.dxy_change > 0.3:
                risk_off_count += 1

            # 금 상승 = Risk Off
            if gm.gold_change > 0.5:
                risk_off_count += 1

            # 구리 상승 = Risk On (경기 민감)
            if gm.copper_change > 0.5:
                risk_on_count += 1

            if risk_on_count >= 4:
                gm.global_sentiment = "RISK_ON"
            elif risk_off_count >= 4:
                gm.global_sentiment = "RISK_OFF"
            else:
                gm.global_sentiment = "NEUTRAL"

            # 미국 시장 연동성 분석
            if gm.dax_change * gm.kospi_change > 0 and gm.dax_change * gm.nikkei_change > 0:
                gm.correlation_with_us = "글로벌 지수 동조화 진행 중"
            elif abs(gm.dax_change) < 0.2 and abs(gm.kospi_change) < 0.2:
                gm.correlation_with_us = "관망세, 미국 시장 대기 중"
            else:
                gm.correlation_with_us = "지역별 차별화"

            # 주요 리스크 식별
            risks = []
            if gm.dxy_change > 1:
                risks.append("달러 강세로 인한 신흥국 압박 가능성")
            if gm.wti_change > 3:
                risks.append("유가 급등 - 인플레이션 압력")
            elif gm.wti_change < -3:
                risks.append("유가 급락 - 경기 둔화 우려")
            if gm.shanghai_change < -1:
                risks.append("중국 시장 약세 - 글로벌 수요 둔화 우려")
            if gm.gold_change > 2:
                risks.append("안전자산 선호 증가 - 위험 회피 심리")

            gm.key_risks = risks

        except Exception as e:
            self._log(f"Error fetching global markets: {e}")

        return gm

    def _generate_entry_exit_strategy(
        self,
        result: Dict,
        report: FinalReport,
        market_data: Dict = None
    ) -> EntryExitStrategy:
        """진입/청산 전략 생성"""
        ees = EntryExitStrategy()

        # 현재 가격 (SPY 기준)
        if report.technical_indicators:
            ees.current_price = report.technical_indicators.current_price

        if ees.current_price == 0 and market_data:
            spy_data = market_data.get('SPY', {})
            ees.current_price = spy_data.get('current', 0)

        if ees.current_price == 0:
            return ees

        position = result.get('final_recommendation', 'NEUTRAL')
        confidence = result.get('confidence', 0.5)
        risk_score = result.get('risk_score', 50)

        # 지지/저항선 참조
        support = report.technical_indicators.support_level if report.technical_indicators else ees.current_price * 0.95
        resistance = report.technical_indicators.resistance_level if report.technical_indicators else ees.current_price * 1.05

        if position == 'BULLISH':
            # 상승 전망: 공격적 진입 전략
            ees.entry_ratios = "30%-30%-40%"
            ees.entry_levels = [
                {"name": "1차 진입", "price": ees.current_price, "ratio": 30, "condition": "즉시 진입"},
                {"name": "2차 진입", "price": round(support * 1.01, 2), "ratio": 30, "condition": "지지선 확인 후"},
                {"name": "3차 진입", "price": round(support, 2), "ratio": 40, "condition": "지지선 터치 시"},
            ]

            ees.take_profit_levels = [
                {"name": "1차 청산", "price": round(resistance, 2), "ratio": 50, "target": f"+{((resistance/ees.current_price)-1)*100:.1f}%"},
                {"name": "2차 청산", "price": round(resistance * 1.03, 2), "ratio": 30, "target": f"+{((resistance*1.03/ees.current_price)-1)*100:.1f}%"},
                {"name": "3차 청산", "price": round(resistance * 1.05, 2), "ratio": 20, "target": f"+{((resistance*1.05/ees.current_price)-1)*100:.1f}%"},
            ]

            ees.stop_loss_level = round(support * 0.97, 2)
            ees.stop_loss_percent = ((ees.stop_loss_level / ees.current_price) - 1) * 100
            ees.trailing_stop = "고점 대비 -5% 하락 시"

            ees.bull_strategy = "레버리지 ETF (SSO) 비중 확대 고려"
            ees.bear_strategy = "현금 비중 50%로 확대, 방어주 중심"

            ees.position_sizing = f"총 자산의 {min(30 + int(confidence * 20), 50)}% 배분"
            ees.rebalancing_trigger = "저항선 돌파 시 추가 매수, RSI 70 이상 시 일부 청산"

        elif position == 'BEARISH':
            # 하락 전망: 방어적 진입 전략
            ees.entry_ratios = "20%-30%-50%"
            ees.entry_levels = [
                {"name": "소규모 진입", "price": round(support, 2), "ratio": 20, "condition": "지지선 확인 시"},
                {"name": "중규모 진입", "price": round(support * 0.97, 2), "ratio": 30, "condition": "지지선 이탈 후 반등 시"},
                {"name": "대규모 진입", "price": round(support * 0.95, 2), "ratio": 50, "condition": "패닉 매도 시"},
            ]

            ees.take_profit_levels = [
                {"name": "1차 청산", "price": round(ees.current_price * 0.98, 2), "ratio": 30, "target": "-2% (손절 최소화)"},
                {"name": "2차 청산", "price": round(support * 1.02, 2), "ratio": 40, "target": "지지선 회복 시"},
                {"name": "3차 청산", "price": round(ees.current_price, 2), "ratio": 30, "target": "본전"},
            ]

            ees.stop_loss_level = round(support * 0.92, 2)
            ees.stop_loss_percent = ((ees.stop_loss_level / ees.current_price) - 1) * 100
            ees.trailing_stop = "반등 고점 대비 -3% 하락 시"

            ees.bull_strategy = "방어주(XLU, XLP) 및 채권(TLT) 중심"
            ees.bear_strategy = "인버스 ETF(SH) 소규모 헤지, 현금 비중 60%"

            ees.position_sizing = f"총 자산의 {max(30 - int(risk_score * 0.2), 10)}% 배분"
            ees.rebalancing_trigger = "VIX 30 이상 시 추가 매도, 지지선 회복 시 비중 확대"

        else:
            # 중립: 보수적 진입 전략
            ees.entry_ratios = "25%-25%-50%"
            ees.entry_levels = [
                {"name": "관망 진입", "price": round(support * 1.01, 2), "ratio": 25, "condition": "지지선 확인 시"},
                {"name": "추가 진입", "price": round(support * 0.98, 2), "ratio": 25, "condition": "조정 시"},
                {"name": "기회 진입", "price": round(support * 0.95, 2), "ratio": 50, "condition": "급락 시"},
            ]

            ees.take_profit_levels = [
                {"name": "1차 청산", "price": round(ees.current_price * 1.03, 2), "ratio": 40, "target": "+3%"},
                {"name": "2차 청산", "price": round(resistance, 2), "ratio": 40, "target": f"+{((resistance/ees.current_price)-1)*100:.1f}%"},
                {"name": "3차 청산", "price": round(resistance * 1.02, 2), "ratio": 20, "target": f"+{((resistance*1.02/ees.current_price)-1)*100:.1f}%"},
            ]

            ees.stop_loss_level = round(support * 0.95, 2)
            ees.stop_loss_percent = ((ees.stop_loss_level / ees.current_price) - 1) * 100
            ees.trailing_stop = "고점 대비 -4% 하락 시"

            ees.bull_strategy = "균형 포트폴리오 유지, 성장주 소폭 확대"
            ees.bear_strategy = "방어주 비중 확대, 현금 40%"

            ees.position_sizing = "총 자산의 30% 배분"
            ees.rebalancing_trigger = "레짐 변화 시 재평가, 월 1회 정기 리밸런싱"

        return ees

    async def save_report(
        self,
        report: FinalReport,
        output_dir: str = "outputs"
    ) -> str:
        """리포트 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 저장
        json_file = output_path / f"ai_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # Markdown 저장
        md_file = output_path / f"ai_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(report.to_markdown())

        self._log(f"Report saved: {md_file}")
        return str(md_file)


async def generate_report_from_json(json_path: str, market_data: Dict = None) -> FinalReport:
    """JSON 파일에서 리포트 생성"""
    with open(json_path, 'r') as f:
        result = json.load(f)

    generator = AIReportGenerator()
    report = await generator.generate(result, market_data)
    await generator.save_report(report)

    return report


if __name__ == "__main__":
    async def main():
        # 가장 최근 JSON 파일 찾기
        output_dir = Path(__file__).parent.parent / "outputs"
        json_files = sorted(output_dir.glob("integrated_*.json"), reverse=True)

        if not json_files:
            print("No analysis JSON files found!")
            return

        latest = json_files[0]
        print(f"Using: {latest}")

        report = await generate_report_from_json(str(latest))
        print("\n" + "=" * 60)
        print(report.to_markdown())

    asyncio.run(main())
