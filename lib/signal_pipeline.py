#!/usr/bin/env python3
"""
EIMAS Signal Pipeline
=====================
Connects existing analyzers to generate signals and store in DB

Usage:
    from lib.signal_pipeline import SignalPipeline

    pipeline = SignalPipeline()
    signals = pipeline.run()
    pipeline.print_summary()
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from lib.trading_db import (
    TradingDB, Signal, SignalSource, SignalAction,
    PortfolioCandidate, InvestorProfile
)


# ============================================================================
# Investor Profile Definitions
# ============================================================================

INVESTOR_PROFILES = {
    InvestorProfile.CONSERVATIVE: {
        "name": "Conservative",
        "risk_tolerance": 0.3,
        "max_equity": 0.40,
        "min_cash": 0.20,
        "max_single_position": 0.15,
        "preferred_actions": ["hold", "reduce"],
        "signal_threshold": 0.7,  # Only act on high conviction
    },
    InvestorProfile.MODERATE: {
        "name": "Moderate",
        "risk_tolerance": 0.5,
        "max_equity": 0.60,
        "min_cash": 0.10,
        "max_single_position": 0.20,
        "preferred_actions": ["buy", "hold", "sell"],
        "signal_threshold": 0.5,
    },
    InvestorProfile.AGGRESSIVE: {
        "name": "Aggressive",
        "risk_tolerance": 0.7,
        "max_equity": 0.90,
        "min_cash": 0.05,
        "max_single_position": 0.30,
        "preferred_actions": ["buy", "sell"],
        "signal_threshold": 0.4,
    },
    InvestorProfile.TACTICAL: {
        "name": "Tactical",
        "risk_tolerance": 0.6,
        "max_equity": 0.75,
        "min_cash": 0.10,
        "max_single_position": 0.25,
        "preferred_actions": ["buy", "sell", "hedge"],
        "signal_threshold": 0.5,
    },
}


# ============================================================================
# Signal Pipeline
# ============================================================================

class SignalPipeline:
    """Signal collection and storage pipeline"""

    def __init__(self, db: TradingDB = None):
        self.db = db or TradingDB()
        self.signals: List[Signal] = []

    def collect_regime_signals(self) -> List[Signal]:
        """Collect signals from Regime Detector"""
        signals = []

        try:
            from lib.regime_detector import RegimeDetector, MarketRegime

            detector = RegimeDetector()
            result = detector.detect()

            # Regime-to-action mapping
            regime_actions = {
                MarketRegime.BULL_LOW_VOL: (SignalAction.BUY, 0.8, "Bull + Low Vol: Aggressive buying opportunity"),
                MarketRegime.BULL_HIGH_VOL: (SignalAction.HOLD, 0.5, "Bull + High Vol: Hold positions, avoid new entries"),
                MarketRegime.BEAR_LOW_VOL: (SignalAction.HOLD, 0.4, "Bear + Low Vol: Wait and watch for bottom"),
                MarketRegime.BEAR_HIGH_VOL: (SignalAction.REDUCE, 0.8, "Bear + High Vol: Reduce risk exposure"),
                MarketRegime.TRANSITION: (SignalAction.HOLD, 0.3, "Transition: Wait for direction confirmation"),
            }

            action, conviction, reasoning = regime_actions.get(
                result.regime,
                (SignalAction.HOLD, 0.3, "Unknown regime")
            )

            # confidence를 0-1 범위로 변환 (0-100에서)
            regime_confidence = result.confidence / 100 if result.confidence > 1 else result.confidence
            signal = Signal(
                source=SignalSource.REGIME_DETECTOR,
                action=action,
                ticker="SPY",
                conviction=conviction * regime_confidence,
                reasoning=f"{reasoning} (Confidence: {regime_confidence:.0%})",
                metadata={
                    "regime": result.regime.value,
                    "trend": getattr(result, 'trend_state', result.regime).value if hasattr(getattr(result, 'trend_state', result.regime), 'value') else str(result.regime),
                    "volatility": getattr(result, 'volatility_state', result.regime).value if hasattr(getattr(result, 'volatility_state', result.regime), 'value') else str(result.regime),
                    "confidence": result.confidence,
                }
            )
            signals.append(signal)
            print(f"  ✓ Regime: {result.regime.value} → {action.value}")

        except Exception as e:
            print(f"  ✗ Regime Detector error: {e}")

        return signals

    def collect_critical_path_signals(self) -> List[Signal]:
        """Collect signals from Critical Path Monitor"""
        signals = []

        try:
            from lib.critical_path_monitor import CriticalPathMonitor, SignalLevel, PathStatus

            monitor = CriticalPathMonitor()
            result = monitor.analyze()

            # Generate signals for active paths
            for path_signal in result.signals:
                # Action based on signal level
                if path_signal.level == SignalLevel.CRITICAL:
                    action = SignalAction.REDUCE
                    conviction = 0.9
                elif path_signal.level == SignalLevel.WARNING:
                    action = SignalAction.HEDGE
                    conviction = 0.7
                else:
                    action = SignalAction.HOLD
                    conviction = 0.5

                # Only generate signals for active paths
                if path_signal.status != PathStatus.INACTIVE:
                    signal = Signal(
                        source=SignalSource.CRITICAL_PATH,
                        action=action,
                        ticker="SPY",
                        conviction=conviction,
                        reasoning=f"[Path {path_signal.path_id}] {path_signal.message}",
                        metadata={
                            "path_id": path_signal.path_id,
                            "path_name": path_signal.path_name,
                            "level": path_signal.level.value,
                            "current_value": path_signal.value,
                            "threshold": path_signal.threshold,
                        }
                    )
                    signals.append(signal)

            if signals:
                print(f"  ✓ Critical Paths: {len(signals)} active ({result.market_regime})")
            else:
                print(f"  ✓ Critical Paths: No active signals ({result.market_regime})")

        except Exception as e:
            print(f"  ✗ Critical Path error: {e}")

        return signals

    def collect_fear_greed_signals(self) -> List[Signal]:
        """Collect signals from Fear & Greed Index"""
        signals = []

        try:
            from lib.market_indicators import MarketIndicatorsCollector, FearGreedLevel

            collector = MarketIndicatorsCollector()
            result = collector.collect_all()

            if result.crypto and result.crypto.fear_greed_value:
                fg_value = result.crypto.fear_greed_value
                fg_level = result.crypto.fear_greed_level.lower()

                # Contrarian strategy
                if 'extreme fear' in fg_level:
                    action = SignalAction.BUY
                    conviction = 0.8
                    reasoning = f"Extreme Fear ({fg_value}): Contrarian buy opportunity"
                elif 'fear' in fg_level:
                    action = SignalAction.BUY
                    conviction = 0.6
                    reasoning = f"Fear ({fg_value}): Consider gradual accumulation"
                elif 'extreme greed' in fg_level:
                    action = SignalAction.REDUCE
                    conviction = 0.7
                    reasoning = f"Extreme Greed ({fg_value}): Overheated, consider taking profits"
                elif 'greed' in fg_level:
                    action = SignalAction.HOLD
                    conviction = 0.5
                    reasoning = f"Greed ({fg_value}): Avoid new entries"
                else:
                    action = SignalAction.HOLD
                    conviction = 0.3
                    reasoning = f"Neutral ({fg_value})"

                signal = Signal(
                    source=SignalSource.FEAR_GREED,
                    action=action,
                    ticker="BTC-USD",  # 암호화폐 기반 지표
                    conviction=conviction,
                    reasoning=reasoning,
                    metadata={
                        "value": fg_value,
                        "level": result.crypto.fear_greed_level,
                    }
                )
                signals.append(signal)
                print(f"  ✓ Fear & Greed: {fg_value} ({result.crypto.fear_greed_level}) → {action.value}")

        except Exception as e:
            print(f"  ✗ Fear & Greed error: {e}")

        return signals

    def collect_vix_signals(self) -> List[Signal]:
        """Collect signals from VIX structure"""
        signals = []

        try:
            from lib.market_indicators import MarketIndicatorsCollector, VIXStructure

            collector = MarketIndicatorsCollector()
            result = collector.collect_all()

            if result.vix:
                vix_spot = result.vix.spot
                structure = result.vix.structure

                # VIX-based signals
                if vix_spot > 30:
                    action = SignalAction.BUY  # Extreme fear = contrarian buy
                    conviction = 0.7
                    reasoning = f"VIX Spike ({vix_spot:.1f}): Post-panic bounce expected"
                elif vix_spot < 15:
                    action = SignalAction.REDUCE  # Complacency = caution
                    conviction = 0.5
                    reasoning = f"VIX Low ({vix_spot:.1f}): Complacency warning"
                elif structure == VIXStructure.BACKWARDATION:
                    action = SignalAction.HEDGE
                    conviction = 0.6
                    reasoning = f"VIX Backwardation: Near-term stress, consider hedging"
                else:
                    action = SignalAction.HOLD
                    conviction = 0.3
                    reasoning = f"VIX Normal ({vix_spot:.1f})"

                signal = Signal(
                    source=SignalSource.VIX_STRUCTURE,
                    action=action,
                    ticker="VIX",
                    conviction=conviction,
                    reasoning=reasoning,
                    metadata={
                        "spot": vix_spot,
                        "structure": structure.value if structure else "unknown",
                        "percentile": result.vix.vix_percentile,
                    }
                )
                signals.append(signal)
                print(f"  ✓ VIX: {vix_spot:.1f} ({structure.value if structure else 'N/A'}) → {action.value}")

        except Exception as e:
            print(f"  ✗ VIX error: {e}")

        return signals

    def run(self) -> List[Signal]:
        """Run the full signal collection pipeline"""
        print("\n" + "=" * 60)
        print("EIMAS Signal Pipeline")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("\nCollecting signals...")

        self.signals = []

        # Collect signals from each source
        self.signals.extend(self.collect_regime_signals())
        self.signals.extend(self.collect_critical_path_signals())
        self.signals.extend(self.collect_fear_greed_signals())
        self.signals.extend(self.collect_vix_signals())

        # Save to DB
        print(f"\nSaving {len(self.signals)} signals to DB...")
        signal_ids = []
        for signal in self.signals:
            signal_id = self.db.save_signal(signal)
            signal.id = signal_id
            signal_ids.append(signal_id)

        print(f"✅ Saved signal IDs: {signal_ids}")

        return self.signals

    def get_consensus(self) -> Dict[str, Any]:
        """Derive signal consensus"""
        if not self.signals:
            return {"action": "hold", "conviction": 0.0, "reasoning": "No signals"}

        # Weighted voting by action
        action_scores = {
            SignalAction.BUY: 0,
            SignalAction.SELL: 0,
            SignalAction.HOLD: 0,
            SignalAction.REDUCE: 0,
            SignalAction.HEDGE: 0,
        }

        for signal in self.signals:
            action_scores[signal.action] += signal.conviction

        # Highest scoring action
        best_action = max(action_scores, key=action_scores.get)
        total_conviction = sum(action_scores.values())
        consensus_conviction = action_scores[best_action] / total_conviction if total_conviction > 0 else 0

        # Collect reasons
        reasons = [s.reasoning for s in self.signals if s.action == best_action]

        return {
            "action": best_action.value,
            "conviction": round(consensus_conviction, 2),
            "action_scores": {k.value: round(v, 2) for k, v in action_scores.items()},
            "reasoning": "; ".join(reasons[:3]),
            "signal_count": len(self.signals),
        }

    def print_summary(self):
        """Print summary"""
        print("\n" + "=" * 60)
        print("Signal Summary")
        print("=" * 60)

        if not self.signals:
            print("No signals collected")
            return

        # Group by source
        by_source = {}
        for s in self.signals:
            source = s.source.value
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(s)

        for source, signals in by_source.items():
            print(f"\n[{source}]")
            for s in signals:
                print(f"  {s.action.value.upper():8} | Conv: {s.conviction:.0%} | {s.reasoning[:50]}")

        # Consensus
        consensus = self.get_consensus()
        print("\n" + "-" * 60)
        print("CONSENSUS")
        print("-" * 60)
        print(f"Action:     {consensus['action'].upper()}")
        print(f"Conviction: {consensus['conviction']:.0%}")
        print(f"Reasoning:  {consensus['reasoning'][:80]}")
        print("=" * 60)


# ============================================================================
# Portfolio Generator
# ============================================================================

class PortfolioGenerator:
    """Generate portfolios based on investor profiles"""

    def __init__(self, db: TradingDB = None):
        self.db = db or TradingDB()

    def generate(
        self,
        signals: List[Signal],
        profile: InvestorProfile = InvestorProfile.MODERATE
    ) -> PortfolioCandidate:
        """Generate portfolio based on signals"""
        profile_config = INVESTOR_PROFILES[profile]

        # Calculate consensus
        consensus = self._get_consensus(signals)
        action = consensus['action']
        conviction = consensus['conviction']

        # Base allocation
        base_allocation = self._get_base_allocation(profile)

        # Adjust based on signals
        adjusted = self._adjust_allocation(
            base_allocation,
            action,
            conviction,
            profile_config
        )

        # Calculate expected return/risk
        expected_return, expected_risk = self._estimate_risk_return(adjusted, action)
        expected_sharpe = expected_return / expected_risk if expected_risk > 0 else 0

        portfolio = PortfolioCandidate(
            profile=profile,
            allocations=adjusted,
            expected_return=expected_return,
            expected_risk=expected_risk,
            expected_sharpe=expected_sharpe,
            signals_used=[s.id for s in signals if s.id],
            reasoning=f"{profile_config['name']} profile: {consensus['reasoning'][:100]}",
            rank=1,
        )

        return portfolio

    def _get_consensus(self, signals: List[Signal]) -> Dict:
        """Get signal consensus"""
        if not signals:
            return {"action": "hold", "conviction": 0.3, "reasoning": "No signals"}

        action_scores = {}
        for s in signals:
            key = s.action.value
            if key not in action_scores:
                action_scores[key] = 0
            action_scores[key] += s.conviction

        best_action = max(action_scores, key=action_scores.get)
        total = sum(action_scores.values())

        return {
            "action": best_action,
            "conviction": action_scores[best_action] / total if total > 0 else 0,
            "reasoning": "; ".join([s.reasoning for s in signals if s.action.value == best_action][:2])
        }

    def _get_base_allocation(self, profile: InvestorProfile) -> Dict[str, float]:
        """Get base allocation for profile"""
        allocations = {
            InvestorProfile.CONSERVATIVE: {
                "SPY": 0.25, "TLT": 0.30, "GLD": 0.15, "CASH": 0.30
            },
            InvestorProfile.MODERATE: {
                "SPY": 0.40, "QQQ": 0.15, "TLT": 0.20, "GLD": 0.10, "CASH": 0.15
            },
            InvestorProfile.AGGRESSIVE: {
                "SPY": 0.35, "QQQ": 0.30, "ARKK": 0.15, "TLT": 0.10, "CASH": 0.10
            },
            InvestorProfile.TACTICAL: {
                "SPY": 0.30, "QQQ": 0.20, "TLT": 0.20, "GLD": 0.15, "CASH": 0.15
            },
        }
        return allocations.get(profile, allocations[InvestorProfile.MODERATE])

    def _adjust_allocation(
        self,
        base: Dict[str, float],
        action: str,
        conviction: float,
        config: Dict
    ) -> Dict[str, float]:
        """Adjust allocation based on signals"""
        adjusted = base.copy()
        adjustment = conviction * 0.2  # Max 20% adjustment

        if action == "buy":
            # Increase equity exposure
            if "SPY" in adjusted:
                adjusted["SPY"] = min(adjusted["SPY"] + adjustment, config['max_single_position'] * 2)
            if "CASH" in adjusted:
                adjusted["CASH"] = max(adjusted["CASH"] - adjustment, config['min_cash'])

        elif action in ["sell", "reduce"]:
            # Decrease equity, increase cash
            if "SPY" in adjusted:
                adjusted["SPY"] = max(adjusted["SPY"] - adjustment, 0.1)
            if "CASH" in adjusted:
                adjusted["CASH"] = min(adjusted["CASH"] + adjustment, 0.5)

        elif action == "hedge":
            # Increase gold and bonds
            if "GLD" in adjusted:
                adjusted["GLD"] = min(adjusted["GLD"] + adjustment * 0.5, 0.25)
            if "TLT" in adjusted:
                adjusted["TLT"] = min(adjusted["TLT"] + adjustment * 0.5, 0.35)
            if "SPY" in adjusted:
                adjusted["SPY"] = max(adjusted["SPY"] - adjustment, 0.15)

        # Normalize (sum = 1.0)
        total = sum(adjusted.values())
        return {k: round(v / total, 3) for k, v in adjusted.items()}

    def _estimate_risk_return(
        self,
        allocation: Dict[str, float],
        action: str
    ) -> Tuple[float, float]:
        """Estimate expected return/risk (simplified)"""
        # Asset-level expected return/volatility (annualized)
        asset_params = {
            "SPY": (0.10, 0.18),   # 10% return, 18% vol
            "QQQ": (0.12, 0.22),
            "ARKK": (0.15, 0.35),
            "TLT": (0.04, 0.12),
            "GLD": (0.05, 0.15),
            "CASH": (0.04, 0.01),
        }

        exp_return = 0
        exp_risk_sq = 0

        for asset, weight in allocation.items():
            params = asset_params.get(asset, (0.05, 0.10))
            exp_return += weight * params[0]
            exp_risk_sq += (weight * params[1]) ** 2  # Simplified: ignoring correlations

        return exp_return, exp_risk_sq ** 0.5

    def generate_all_profiles(self, signals: List[Signal]) -> List[PortfolioCandidate]:
        """Generate portfolios for all profiles"""
        portfolios = []

        for profile in InvestorProfile:
            portfolio = self.generate(signals, profile)
            portfolio_id = self.db.save_portfolio(portfolio)
            portfolio.id = portfolio_id
            portfolios.append(portfolio)

        return portfolios

    def print_portfolios(self, portfolios: List[PortfolioCandidate]):
        """Print portfolios"""
        print("\n" + "=" * 70)
        print("Generated Portfolios")
        print("=" * 70)

        for p in portfolios:
            print(f"\n[{p.profile.value.upper()}] ID={p.id}")
            print(f"  Expected Return: {p.expected_return:.1%}")
            print(f"  Expected Risk:   {p.expected_risk:.1%}")
            print(f"  Sharpe Ratio:    {p.expected_sharpe:.2f}")
            print(f"  Allocations:")
            for asset, weight in sorted(p.allocations.items(), key=lambda x: -x[1]):
                bar = "█" * int(weight * 30)
                print(f"    {asset:6} {weight:5.1%} {bar}")

        print("=" * 70)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EIMAS Signal Pipeline Test")
    print("=" * 70)

    # Collect signals
    pipeline = SignalPipeline()
    signals = pipeline.run()
    pipeline.print_summary()

    # Generate portfolios
    generator = PortfolioGenerator(pipeline.db)
    portfolios = generator.generate_all_profiles(signals)
    generator.print_portfolios(portfolios)

    # DB summary
    pipeline.db.print_summary()

    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
