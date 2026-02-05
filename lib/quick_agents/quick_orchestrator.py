"""
Quick Orchestrator
==================
Quick 모드 AI 에이전트 조율자

Purpose:
- Full 모드 결과를 Quick 모드 에이전트들로 검증
- 5개 전문 에이전트 조율:
  1. PortfolioValidator - 포트폴리오 이론 검증
  2. AllocationReasoner - 자산배분 논리 분석
  3. MarketSentimentAgent - 시장 정서 (KOSPI + SPX)
  4. AlternativeAssetAgent - 대체자산 판단
  5. FinalValidator - 최종 종합 검증

Usage:
    from lib.quick_agents import QuickOrchestrator

    orchestrator = QuickOrchestrator()
    result = orchestrator.run_quick_validation(
        full_json_path="outputs/eimas_20260204_120000.json"
    )
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from .portfolio_validator import PortfolioValidator
from .allocation_reasoner import AllocationReasoner
from .market_sentiment_agent import MarketSentimentAgent
from .alternative_asset_agent import AlternativeAssetAgent
from .final_validator import FinalValidator

logger = logging.getLogger(__name__)


class QuickOrchestrator:
    """Quick 모드 에이전트 조율자"""

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        perplexity_api_key: Optional[str] = None
    ):
        """
        Args:
            anthropic_api_key: Anthropic API key
            perplexity_api_key: Perplexity API key
        """
        logger.info("Initializing Quick Mode Orchestrator...")

        # Initialize agents
        self.portfolio_validator = PortfolioValidator(api_key=anthropic_api_key)
        self.allocation_reasoner = AllocationReasoner(api_key=perplexity_api_key)
        self.market_sentiment_agent = MarketSentimentAgent(api_key=anthropic_api_key)
        self.alternative_asset_agent = AlternativeAssetAgent(api_key=perplexity_api_key)
        self.final_validator = FinalValidator(api_key=anthropic_api_key)

        logger.info("✓ All Quick mode agents initialized")

    def run_quick_validation(
        self,
        full_json_path: Optional[str] = None,
        allocation_result: Optional[Dict] = None,
        market_data: Optional[Dict] = None
    ) -> Dict:
        """
        Quick 모드 검증 실행

        Args:
            full_json_path: Full 모드 JSON 결과 경로
            allocation_result: 자산배분 결과 (선택)
            market_data: 시장 데이터 (선택)

        Returns:
            {
                'timestamp': str,
                'portfolio_validation': {...},
                'allocation_reasoning': {...},
                'market_sentiment': {...},
                'alternative_assets': {...},
                'final_validation': {...},
                'execution_time_seconds': float
            }
        """
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("Quick Mode Validation Started")
        logger.info("=" * 80)

        # Step 1: Load Full mode results
        full_mode_result = self._load_full_mode_result(full_json_path)
        if not full_mode_result:
            logger.error("Failed to load Full mode results")
            return {'error': 'Failed to load Full mode results'}

        # Step 2: Extract data for agents
        extracted_data = self._extract_agent_inputs(full_mode_result, allocation_result, market_data)

        # Step 3: Run agents in sequence
        results = {}

        # Agent 1/5: Portfolio Validator (Claude)
        logger.info("\n[Agent 1/5] Running Portfolio Validator...")
        try:
            results['portfolio_validation'] = self.portfolio_validator.validate_portfolio(
                allocation_result=extracted_data['allocation_result'],
                market_context=extracted_data['market_context'],
                constraints=extracted_data.get('constraints')
            )
            logger.info(f"✓ Portfolio Validation: {results['portfolio_validation'].get('validation_result', 'N/A')}")
        except Exception as e:
            logger.error(f"Portfolio Validator failed: {e}")
            results['portfolio_validation'] = {
                'validation_result': 'SKIPPED', 'confidence': 0,
                'error': str(e), 'degraded': True
            }

        # Agent 2/5: Allocation Reasoner (Perplexity)
        logger.info("\n[Agent 2/5] Running Allocation Reasoner...")
        try:
            results['allocation_reasoning'] = self.allocation_reasoner.analyze_reasoning(
                allocation_decision=extracted_data['allocation_decision'],
                calculation_details=extracted_data['calculation_details'],
                market_context=extracted_data['market_context']
            )
            logger.info(f"✓ Reasoning Quality: {results['allocation_reasoning'].get('reasoning_quality', 'N/A')}")
        except Exception as e:
            logger.error(f"Allocation Reasoner failed: {e}")
            results['allocation_reasoning'] = {
                'reasoning_quality': 'SKIPPED', 'academic_support': {},
                'error': str(e), 'degraded': True
            }

        # Agent 3/5: Market Sentiment Agent (Claude)
        logger.info("\n[Agent 3/5] Running Market Sentiment Agent...")
        try:
            results['market_sentiment'] = self.market_sentiment_agent.analyze_market_sentiment(
                kospi_data=extracted_data['kospi_data'],
                spx_data=extracted_data['spx_data'],
                market_context=extracted_data['market_context']
            )
            kospi_sent = results['market_sentiment'].get('kospi_sentiment', {}).get('sentiment', 'N/A')
            spx_sent = results['market_sentiment'].get('spx_sentiment', {}).get('sentiment', 'N/A')
            logger.info(f"✓ KOSPI Sentiment: {kospi_sent}, SPX Sentiment: {spx_sent}")
        except Exception as e:
            logger.error(f"Market Sentiment Agent failed: {e}")
            results['market_sentiment'] = {
                'kospi_sentiment': {'sentiment': 'SKIPPED', 'confidence': 0},
                'spx_sentiment': {'sentiment': 'SKIPPED', 'confidence': 0},
                'error': str(e), 'degraded': True
            }

        # Agent 4/5: Alternative Asset Agent (Perplexity)
        logger.info("\n[Agent 4/5] Running Alternative Asset Agent...")
        try:
            results['alternative_assets'] = self.alternative_asset_agent.analyze_alternative_assets(
                crypto_data=extracted_data['crypto_data'],
                commodity_data=extracted_data['commodity_data'],
                market_context=extracted_data['market_context']
            )
            crypto_rec = results['alternative_assets'].get('crypto_assessment', {}).get('recommendation', 'N/A')
            logger.info(f"✓ Crypto Recommendation: {crypto_rec}")
        except Exception as e:
            logger.error(f"Alternative Asset Agent failed: {e}")
            results['alternative_assets'] = {
                'crypto_assessment': {'recommendation': 'SKIPPED'},
                'commodity_assessment': {'recommendation': 'SKIPPED'},
                'error': str(e), 'degraded': True
            }

        # Step 4: Final Validation & Synthesis
        degraded_agents = [k for k in results if results[k].get('degraded')]
        if degraded_agents:
            logger.warning(f"⚠ Degraded agents (skipped): {degraded_agents}")

        logger.info("\n[Agent 5/5] Running Final Validator...")
        try:
            results['final_validation'] = self.final_validator.validate_and_synthesize(
                full_mode_result=full_mode_result,
                portfolio_validation=results.get('portfolio_validation', {}),
                allocation_reasoning=results.get('allocation_reasoning', {}),
                market_sentiment=results.get('market_sentiment', {}),
                alternative_assets=results.get('alternative_assets', {})
            )
            final_rec = results['final_validation'].get('final_recommendation', 'N/A')
            confidence = results['final_validation'].get('confidence', 0) * 100
            logger.info(f"✓ Final Recommendation: {final_rec} (Confidence: {confidence:.0f}%)")
        except Exception as e:
            logger.error(f"Final Validator failed: {e}")
            results['final_validation'] = {'error': str(e)}

        # Add metadata
        end_time = datetime.now()
        results['timestamp'] = datetime.now().isoformat()
        results['execution_time_seconds'] = (end_time - start_time).total_seconds()
        results['degraded_agents'] = [k for k in ('portfolio_validation', 'allocation_reasoning', 'market_sentiment', 'alternative_assets') if results.get(k, {}).get('degraded')]
        results['success_rate'] = 1.0 - len(results['degraded_agents']) / 4.0

        logger.info("\n" + "=" * 80)
        logger.info(f"Quick Mode Validation Completed in {results['execution_time_seconds']:.1f}s")
        logger.info("=" * 80)

        return results

    def _load_full_mode_result(self, json_path: Optional[str] = None) -> Dict:
        """Full 모드 결과 로드"""
        if json_path:
            # Use provided path
            path = Path(json_path)
            if not path.exists():
                logger.error(f"Full mode JSON not found: {json_path}")
                return {}
        else:
            # Find latest eimas_*.json in outputs
            outputs_dir = Path("outputs")
            if not outputs_dir.exists():
                logger.error("outputs/ directory not found")
                return {}

            json_files = sorted(
                outputs_dir.glob("eimas_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if not json_files:
                logger.error("No Full mode JSON files found in outputs/")
                return {}

            path = json_files[0]
            logger.info(f"Loading latest Full mode result: {path.name}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✓ Loaded Full mode result ({len(json.dumps(data))} bytes)")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            return {}

    def _extract_agent_inputs(
        self,
        full_result: Dict,
        allocation_result: Optional[Dict],
        market_data: Optional[Dict]
    ) -> Dict:
        """에이전트 입력 데이터 추출"""
        logger.info("Extracting agent inputs from Full mode result...")

        # Market Context (공통)
        market_context = {
            'regime': full_result.get('regime', {}).get('regime', 'Unknown'),
            'regime_confidence': full_result.get('regime', {}).get('confidence', 0.0),
            'risk_score': full_result.get('risk_score', 50.0),
            'risk_level': full_result.get('risk_level', 'MEDIUM'),
            'volatility': full_result.get('regime', {}).get('volatility', 0.16),
            'bubble_status': full_result.get('bubble_risk', {}).get('overall_status', 'NONE'),
            'liquidity_signal': full_result.get('liquidity_signal', 'NEUTRAL')
        }

        # VIX and Fear & Greed (if available)
        if 'vix' in full_result:
            market_context['vix'] = full_result['vix']
        if 'fear_greed_index' in full_result:
            market_context['fear_greed_index'] = full_result['fear_greed_index']

        # Allocation Result (for Portfolio Validator)
        if allocation_result:
            alloc_result = allocation_result
        else:
            # Extract from full_result
            alloc_result = {
                'stock': full_result.get('stock_bond_allocation', {}).get('tactical_allocation', {}).get('stock', 0.6),
                'bond': full_result.get('stock_bond_allocation', {}).get('tactical_allocation', {}).get('bond', 0.4),
                'expected_return': full_result.get('stock_bond_allocation', {}).get('portfolio_metrics', {}).get('expected_return', 0.08),
                'volatility': full_result.get('stock_bond_allocation', {}).get('portfolio_metrics', {}).get('volatility', 0.10),
                'sharpe_ratio': full_result.get('stock_bond_allocation', {}).get('portfolio_metrics', {}).get('sharpe_ratio', 0.8)
            }

        # Allocation Decision (for Allocation Reasoner)
        allocation_decision = alloc_result.copy()

        # Calculation Details (for Allocation Reasoner)
        calculation_details = []
        if 'calculation_evidence' in full_result:
            calculation_details = full_result['calculation_evidence']
        else:
            calculation_details = [
                f"Risk Tolerance: {full_result.get('stock_bond_allocation', {}).get('risk_tolerance', 'moderate')}",
                f"Market Regime: {market_context['regime']}",
                f"Risk Score: {market_context['risk_score']:.1f}/100"
            ]

        # KOSPI Data (for Market Sentiment Agent)
        kospi_data = self._extract_kospi_data(full_result, market_data)

        # SPX Data (for Market Sentiment Agent)
        spx_data = self._extract_spx_data(full_result, market_data)

        # Crypto Data (for Alternative Asset Agent)
        crypto_data = self._extract_crypto_data(full_result, market_data)

        # Commodity Data (for Alternative Asset Agent)
        commodity_data = self._extract_commodity_data(full_result, market_data)

        return {
            'market_context': market_context,
            'allocation_result': alloc_result,
            'allocation_decision': allocation_decision,
            'calculation_details': calculation_details,
            'kospi_data': kospi_data,
            'spx_data': spx_data,
            'crypto_data': crypto_data,
            'commodity_data': commodity_data,
            'constraints': None  # Can add constraints if needed
        }

    def _extract_kospi_data(self, full_result: Dict, market_data: Optional[Dict]) -> Dict:
        """KOSPI 데이터 추출"""
        kospi = {}

        # From Korea data in full_result
        if 'korea_data' in full_result:
            korea_summary = full_result['korea_data'].get('summary', {})
            kospi_stats = korea_summary.get('kospi_stats', {})
            kospi['current_price'] = kospi_stats.get('current_price', 0)
            kospi['ytd_return'] = kospi_stats.get('ytd_return', 0)
            kospi['volatility'] = kospi_stats.get('volatility', 0)
            kospi['sharpe'] = kospi_stats.get('sharpe', 0)

        # Fair Value gap (if available)
        if 'fair_value_results' in full_result and 'kospi' in full_result['fair_value_results']:
            kospi_fv = full_result['fair_value_results']['kospi']
            if 'consensus' in kospi_fv:
                kospi['fair_value_gap'] = kospi_fv['consensus'].get('valuation_gap_pct', 0)
            elif 'fed_model' in kospi_fv:
                kospi['fair_value_gap'] = kospi_fv['fed_model'].get('valuation_gap_pct', 0)

        return kospi

    def _extract_spx_data(self, full_result: Dict, market_data: Optional[Dict]) -> Dict:
        """SPX 데이터 추출"""
        spx = {}

        # From market_data or full_result
        if market_data and 'SPY' in market_data:
            spy_df = market_data['SPY']
            if not spy_df.empty:
                spx['current_price'] = float(spy_df['Close'].iloc[-1])
                spx['ytd_return'] = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[0] - 1) * 100

        # Fair Value gap (if available)
        if 'fair_value_results' in full_result and 'spx' in full_result['fair_value_results']:
            spx_fv = full_result['fair_value_results']['spx']
            if 'consensus' in spx_fv:
                spx['fair_value_gap'] = spx_fv['consensus'].get('valuation_gap_pct', 0)
            elif 'fed_model' in spx_fv:
                spx['fair_value_gap'] = spx_fv['fed_model'].get('valuation_gap_pct', 0)

        # Volatility from regime
        if 'regime' in full_result:
            spx['volatility'] = full_result['regime'].get('volatility', 0.16) * 100

        return spx

    def _extract_crypto_data(self, full_result: Dict, market_data: Optional[Dict]) -> Dict:
        """크립토 데이터 추출"""
        crypto = {}

        # From crypto_data in full_result
        if 'crypto_data' in full_result and full_result['crypto_data']:
            crypto_summary = full_result['crypto_data']

            # BTC
            if 'BTC-USD' in crypto_summary:
                btc = crypto_summary['BTC-USD']
                crypto['btc_price'] = btc.get('current_price', 0)
                crypto['btc_ytd_return'] = btc.get('ytd_return', 0)

            # ETH
            if 'ETH-USD' in crypto_summary:
                eth = crypto_summary['ETH-USD']
                crypto['eth_price'] = eth.get('current_price', 0)
                crypto['eth_ytd_return'] = eth.get('ytd_return', 0)

        # Stablecoin supply (from Genius Act if available)
        if 'genius_act_regime' in full_result:
            # Proxy: expansion regime → stablecoin supply increasing
            regime = full_result['genius_act_regime']
            if regime == 'expansion':
                crypto['stablecoin_supply_change'] = 5.0  # Proxy
            elif regime == 'contraction':
                crypto['stablecoin_supply_change'] = -3.0
            else:
                crypto['stablecoin_supply_change'] = 0.0

        return crypto

    def _extract_commodity_data(self, full_result: Dict, market_data: Optional[Dict]) -> Dict:
        """원자재 데이터 추출"""
        commodity = {}

        # From market_data
        if market_data and 'GLD' in market_data:
            gld_df = market_data['GLD']
            if not gld_df.empty:
                # GLD ETF as proxy for gold price
                commodity['gold_price'] = float(gld_df['Close'].iloc[-1]) * 10  # Proxy: ~1/100 of spot
                commodity['gold_ytd_return'] = (gld_df['Close'].iloc[-1] / gld_df['Close'].iloc[0] - 1) * 100

        if market_data and 'USO' in market_data:
            uso_df = market_data['USO']
            if not uso_df.empty:
                commodity['oil_price'] = float(uso_df['Close'].iloc[-1])

        return commodity


if __name__ == "__main__":
    # Test orchestrator
    orchestrator = QuickOrchestrator()

    # Run with latest Full mode result
    result = orchestrator.run_quick_validation()

    print("\n" + "=" * 80)
    print("QUICK MODE VALIDATION RESULT")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Save result
    output_path = Path("outputs") / f"quick_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Result saved to: {output_path}")
