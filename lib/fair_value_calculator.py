#!/usr/bin/env python3
"""
Fair Value Calculator
=====================
주식 시장 Fair Value 산출 (Fed Model, Shiller PE, Equity Risk Premium)

Purpose:
- Fair KOSPI / Fair SPX 산출
- 과대/과소 평가 판단
- 전략적 자산배분 입력으로 활용

Economic Foundation:
- Fed Model: Yardeni (1997), Asness (2003)
- Shiller CAPE: Shiller (2000), Campbell & Shiller (1998)
- Equity Risk Premium: Damodaran (2023), Ilmanen (2011)
- Gordon Growth Model: Gordon & Shapiro (1956)

Models:
1. Fed Model: E/P = 10Y Bond Yield
2. Shiller CAPE: Cyclically Adjusted P/E Ratio
3. ERP Model: Expected Return = Earnings Yield + Growth - Risk Free Rate
4. Gordon Model: P = D / (r - g)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FairValueCalculator:
    """Fair Value 계산기"""

    def __init__(self):
        """Initialize Fair Value Calculator"""
        # Historical averages (can be updated with real data)
        self.HISTORICAL_AVERAGES = {
            'spx': {
                'pe_ratio': 16.5,          # Historical P/E
                'cape_ratio': 17.0,        # Shiller CAPE
                'earnings_growth': 0.07,   # 7% 장기 성장률
                'dividend_yield': 0.02,    # 2% 배당수익률
            },
            'kospi': {
                'pe_ratio': 12.0,          # Historical P/E (lower than SPX)
                'cape_ratio': 13.5,        # Shiller CAPE
                'earnings_growth': 0.05,   # 5% 장기 성장률 (emerging market)
                'dividend_yield': 0.025,   # 2.5% 배당수익률
            }
        }

    def calculate_fed_model_fair_value(
        self,
        current_price: float,
        earnings_per_share: float,
        bond_yield_10y: float,
        market: str = 'spx'
    ) -> Dict:
        """
        Fed Model Fair Value 계산

        Fed Model: E/P = Bond Yield
        → Fair P/E = 1 / Bond Yield
        → Fair Price = EPS / Bond Yield

        Args:
            current_price: 현재 지수 가격
            earnings_per_share: 주당순이익 (EPS)
            bond_yield_10y: 10년 국채 수익률 (예: 0.04 = 4%)
            market: 'spx' or 'kospi'

        Returns:
            Fair value analysis
        """
        if bond_yield_10y <= 0 or earnings_per_share <= 0:
            logger.warning("Invalid inputs for Fed Model")
            return {}

        # Current metrics
        current_pe = current_price / earnings_per_share
        earnings_yield = earnings_per_share / current_price  # E/P

        # Fed Model fair value
        fair_pe = 1 / bond_yield_10y
        fair_price = earnings_per_share / bond_yield_10y

        # Over/Under valuation
        valuation_gap = (current_price / fair_price - 1) * 100
        pe_premium = (current_pe / fair_pe - 1) * 100

        result = {
            'model': 'Fed Model',
            'current_price': current_price,
            'fair_price': fair_price,
            'valuation_gap_pct': valuation_gap,
            'current_pe': current_pe,
            'fair_pe': fair_pe,
            'pe_premium_pct': pe_premium,
            'earnings_yield': earnings_yield * 100,
            'bond_yield': bond_yield_10y * 100,
            'signal': self._get_valuation_signal(valuation_gap)
        }

        return result

    def calculate_shiller_cape_fair_value(
        self,
        current_price: float,
        average_earnings_10y: float,
        market: str = 'spx'
    ) -> Dict:
        """
        Shiller CAPE Fair Value 계산

        CAPE = Price / Average(Real Earnings, 10Y)
        Fair Price = Historical CAPE * Current Average Earnings

        Args:
            current_price: 현재 지수 가격
            average_earnings_10y: 10년 평균 실질 이익
            market: 'spx' or 'kospi'

        Returns:
            Fair value analysis
        """
        if average_earnings_10y <= 0:
            logger.warning("Invalid average earnings")
            return {}

        # Current CAPE
        current_cape = current_price / average_earnings_10y

        # Historical average CAPE
        historical_cape = self.HISTORICAL_AVERAGES[market]['cape_ratio']

        # Fair price based on historical CAPE
        fair_price = historical_cape * average_earnings_10y

        # Valuation gap
        valuation_gap = (current_price / fair_price - 1) * 100
        cape_premium = (current_cape / historical_cape - 1) * 100

        result = {
            'model': 'Shiller CAPE',
            'current_price': current_price,
            'fair_price': fair_price,
            'valuation_gap_pct': valuation_gap,
            'current_cape': current_cape,
            'historical_cape': historical_cape,
            'cape_premium_pct': cape_premium,
            'signal': self._get_valuation_signal(valuation_gap),
            'interpretation': self._interpret_cape(current_cape, historical_cape)
        }

        return result

    def calculate_equity_risk_premium(
        self,
        earnings_yield: float,
        earnings_growth: float,
        risk_free_rate: float,
        market: str = 'spx'
    ) -> Dict:
        """
        Equity Risk Premium (ERP) 계산

        ERP = Earnings Yield + Growth - Risk Free Rate

        Expected Equity Return = Earnings Yield + Growth
        → ERP = Expected Return - Risk Free Rate

        Args:
            earnings_yield: 이익수익률 (E/P)
            earnings_growth: 예상 이익 성장률
            risk_free_rate: 무위험 수익률 (10Y 국채)
            market: 'spx' or 'kospi'

        Returns:
            ERP analysis
        """
        # Expected equity return
        expected_return = earnings_yield + earnings_growth

        # Equity Risk Premium
        erp = expected_return - risk_free_rate

        # Historical ERP (typical: 4-6% for US, 6-8% for EM)
        historical_erp = 0.05 if market == 'spx' else 0.07

        # ERP spread
        erp_spread = (erp - historical_erp) * 100

        result = {
            'model': 'Equity Risk Premium',
            'expected_return_pct': expected_return * 100,
            'risk_free_rate_pct': risk_free_rate * 100,
            'erp_pct': erp * 100,
            'historical_erp_pct': historical_erp * 100,
            'erp_spread_bps': erp_spread,
            'signal': self._get_erp_signal(erp, historical_erp),
            'interpretation': f"ERP {erp*100:.2f}% vs Historical {historical_erp*100:.2f}%"
        }

        return result

    def calculate_gordon_growth_fair_value(
        self,
        current_dividend: float,
        required_return: float,
        growth_rate: float,
        current_price: float
    ) -> Dict:
        """
        Gordon Growth Model Fair Value

        Fair Price = D / (r - g)

        Args:
            current_dividend: 현재 배당 (per share)
            required_return: 요구수익률
            growth_rate: 배당 성장률
            current_price: 현재 가격

        Returns:
            Fair value analysis
        """
        if required_return <= growth_rate:
            logger.warning("Invalid Gordon Model: r <= g")
            return {}

        # Fair price
        fair_price = current_dividend / (required_return - growth_rate)

        # Valuation gap
        valuation_gap = (current_price / fair_price - 1) * 100

        result = {
            'model': 'Gordon Growth',
            'current_price': current_price,
            'fair_price': fair_price,
            'valuation_gap_pct': valuation_gap,
            'dividend': current_dividend,
            'required_return_pct': required_return * 100,
            'growth_rate_pct': growth_rate * 100,
            'dividend_yield_implied': (current_dividend / fair_price) * 100,
            'signal': self._get_valuation_signal(valuation_gap)
        }

        return result

    def calculate_comprehensive_fair_value(
        self,
        current_price: float,
        eps: float,
        average_earnings_10y: float,
        dividend: float,
        bond_yield: float,
        earnings_growth: float,
        market: str = 'spx'
    ) -> Dict:
        """
        종합 Fair Value 분석

        4가지 모델 통합:
        1. Fed Model
        2. Shiller CAPE
        3. Equity Risk Premium
        4. Gordon Growth

        Returns:
            Comprehensive analysis
        """
        results = {}

        # 1. Fed Model
        fed_result = self.calculate_fed_model_fair_value(
            current_price, eps, bond_yield, market
        )
        if fed_result:
            results['fed_model'] = fed_result

        # 2. Shiller CAPE
        cape_result = self.calculate_shiller_cape_fair_value(
            current_price, average_earnings_10y, market
        )
        if cape_result:
            results['shiller_cape'] = cape_result

        # 3. Equity Risk Premium
        earnings_yield = eps / current_price
        erp_result = self.calculate_equity_risk_premium(
            earnings_yield, earnings_growth, bond_yield, market
        )
        if erp_result:
            results['erp'] = erp_result

        # 4. Gordon Growth (if dividend available)
        if dividend > 0:
            required_return = bond_yield + 0.05  # Risk premium
            gordon_result = self.calculate_gordon_growth_fair_value(
                dividend, required_return, earnings_growth * 0.7, current_price
            )
            if gordon_result:
                results['gordon_growth'] = gordon_result

        # Consensus fair value (average)
        fair_prices = []
        if 'fed_model' in results:
            fair_prices.append(results['fed_model']['fair_price'])
        if 'shiller_cape' in results:
            fair_prices.append(results['shiller_cape']['fair_price'])
        if 'gordon_growth' in results:
            fair_prices.append(results['gordon_growth']['fair_price'])

        if fair_prices:
            consensus_fair_price = np.mean(fair_prices)
            consensus_gap = (current_price / consensus_fair_price - 1) * 100

            results['consensus'] = {
                'fair_price': consensus_fair_price,
                'valuation_gap_pct': consensus_gap,
                'signal': self._get_valuation_signal(consensus_gap),
                'num_models': len(fair_prices)
            }

        results['market'] = market
        results['current_price'] = current_price
        results['timestamp'] = datetime.now().isoformat()

        return results

    def _get_valuation_signal(self, gap_pct: float) -> str:
        """
        Valuation signal 판단

        Args:
            gap_pct: Valuation gap (%)

        Returns:
            'OVERVALUED', 'FAIR', 'UNDERVALUED'
        """
        if gap_pct > 15:
            return 'OVERVALUED'
        elif gap_pct < -15:
            return 'UNDERVALUED'
        else:
            return 'FAIR'

    def _get_erp_signal(self, current_erp: float, historical_erp: float) -> str:
        """ERP signal 판단"""
        if current_erp > historical_erp * 1.2:
            return 'ATTRACTIVE'  # High ERP = stocks cheap
        elif current_erp < historical_erp * 0.8:
            return 'UNATTRACTIVE'  # Low ERP = stocks expensive
        else:
            return 'NEUTRAL'

    def _interpret_cape(self, current_cape: float, historical_cape: float) -> str:
        """CAPE 해석"""
        ratio = current_cape / historical_cape

        if ratio > 1.3:
            return "Significantly overvalued vs history"
        elif ratio > 1.1:
            return "Moderately overvalued"
        elif ratio < 0.8:
            return "Significantly undervalued"
        elif ratio < 0.9:
            return "Moderately undervalued"
        else:
            return "Fair valuation"


def compare_fair_values(
    spx_analysis: Dict,
    kospi_analysis: Dict
) -> Dict:
    """
    SPX vs KOSPI Fair Value 비교

    Args:
        spx_analysis: SPX comprehensive analysis
        kospi_analysis: KOSPI comprehensive analysis

    Returns:
        Comparison summary
    """
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'spx': {},
        'kospi': {},
        'relative_valuation': {}
    }

    # Extract consensus
    if 'consensus' in spx_analysis:
        comparison['spx'] = {
            'current_price': spx_analysis['current_price'],
            'fair_price': spx_analysis['consensus']['fair_price'],
            'gap_pct': spx_analysis['consensus']['valuation_gap_pct'],
            'signal': spx_analysis['consensus']['signal']
        }

    if 'consensus' in kospi_analysis:
        comparison['kospi'] = {
            'current_price': kospi_analysis['current_price'],
            'fair_price': kospi_analysis['consensus']['fair_price'],
            'gap_pct': kospi_analysis['consensus']['valuation_gap_pct'],
            'signal': kospi_analysis['consensus']['signal']
        }

    # Relative valuation
    if comparison['spx'] and comparison['kospi']:
        spx_gap = comparison['spx']['gap_pct']
        kospi_gap = comparison['kospi']['gap_pct']

        comparison['relative_valuation'] = {
            'spx_vs_kospi_gap': spx_gap - kospi_gap,
            'recommendation': (
                'Favor SPX' if spx_gap < kospi_gap - 10 else
                'Favor KOSPI' if kospi_gap < spx_gap - 10 else
                'Neutral'
            )
        }

    return comparison


if __name__ == "__main__":
    # Example: SPX Fair Value
    calculator = FairValueCalculator()

    spx_analysis = calculator.calculate_comprehensive_fair_value(
        current_price=4800.0,
        eps=220.0,
        average_earnings_10y=200.0,
        dividend=70.0,
        bond_yield=0.042,
        earnings_growth=0.07,
        market='spx'
    )

    print("\n" + "="*80)
    print("SPX FAIR VALUE ANALYSIS")
    print("="*80)
    for model, result in spx_analysis.items():
        if isinstance(result, dict) and 'fair_price' in result:
            print(f"\n{model.upper()}:")
            print(f"  Fair Price: {result['fair_price']:.2f}")
            print(f"  Gap: {result['valuation_gap_pct']:+.2f}%")
            print(f"  Signal: {result.get('signal', 'N/A')}")
