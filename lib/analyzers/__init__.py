"""
EIMAS Analyzers - 분석 엔진 모듈
================================
Market analysis engine modules.

Modules:
    - etf: ETF Flow Analyzer
"""

# Import with fallback for missing modules
def _safe_import(module_name, class_names):
    """Safely import classes from a module, returning None if not available"""
    try:
        module = __import__(module_name, fromlist=class_names)
        return {name: getattr(module, name, None) for name in class_names}
    except (ImportError, AttributeError):
        return {name: None for name in class_names}

# Safe imports
_regime = _safe_import('lib.regime_detector', ['RegimeDetector'])
_regime_gmm = _safe_import('lib.regime_analyzer', ['GMMRegimeAnalyzer'])
_liquidity = _safe_import('lib.liquidity_analysis', ['LiquidityMarketAnalyzer', 'DynamicLagAnalyzer'])
_micro = _safe_import('lib.microstructure', ['MicrostructureAnalyzer', 'RealtimeMicrostructureAnalyzer', 'DailyMicrostructureAnalyzer'])
_sentiment = _safe_import('lib.sentiment_analyzer', ['SentimentAnalyzer'])
_causal = _safe_import('lib.causal_network', ['GrangerCausalityAnalyzer', 'CausalNetworkAnalyzer'])
_volume = _safe_import('lib.volume_analyzer', ['VolumeAnalyzer'])
_bubble = _safe_import('lib.bubble', ['BubbleDetector'])
_dual = _safe_import('lib.dual_mode_analyzer', ['DualModeAnalyzer'])
_macro = _safe_import('lib.macro_analyzer', ['MacroLiquidityAnalyzer'])
_info = _safe_import('lib.information_flow', ['InformationFlowAnalyzer'])
_ark = _safe_import('lib.ark_holdings_analyzer', ['ARKHoldingsAnalyzer'])

# ETF subpackage
try:
    from .etf import ETFFlowAnalyzer
except ImportError:
    # Fallback to old location
    try:
        from lib.etf_flow_analyzer import ETFFlowAnalyzer
    except ImportError:
        ETFFlowAnalyzer = None

# Collect all available exports
RegimeDetector = _regime.get('RegimeDetector')
GMMRegimeAnalyzer = _regime_gmm.get('GMMRegimeAnalyzer')
LiquidityMarketAnalyzer = _liquidity.get('LiquidityMarketAnalyzer')
DynamicLagAnalyzer = _liquidity.get('DynamicLagAnalyzer')
MicrostructureAnalyzer = _micro.get('MicrostructureAnalyzer')
RealtimeMicrostructureAnalyzer = _micro.get('RealtimeMicrostructureAnalyzer')
DailyMicrostructureAnalyzer = _micro.get('DailyMicrostructureAnalyzer')
SentimentAnalyzer = _sentiment.get('SentimentAnalyzer')
BubbleDetector = _bubble.get('BubbleDetector')
GrangerCausalityAnalyzer = _causal.get('GrangerCausalityAnalyzer')
CausalNetworkAnalyzer = _causal.get('CausalNetworkAnalyzer')
InformationFlowAnalyzer = _info.get('InformationFlowAnalyzer')
VolumeAnalyzer = _volume.get('VolumeAnalyzer')
DualModeAnalyzer = _dual.get('DualModeAnalyzer')
MacroLiquidityAnalyzer = _macro.get('MacroLiquidityAnalyzer')
ARKHoldingsAnalyzer = _ark.get('ARKHoldingsAnalyzer')

__all__ = [
    'RegimeDetector',
    'GMMRegimeAnalyzer',
    'LiquidityMarketAnalyzer',
    'DynamicLagAnalyzer',
    'MicrostructureAnalyzer',
    'RealtimeMicrostructureAnalyzer',
    'DailyMicrostructureAnalyzer',
    'SentimentAnalyzer',
    'BubbleDetector',
    'GrangerCausalityAnalyzer',
    'CausalNetworkAnalyzer',
    'InformationFlowAnalyzer',
    'VolumeAnalyzer',
    'DualModeAnalyzer',
    'MacroLiquidityAnalyzer',
    'ETFFlowAnalyzer',
    'ARKHoldingsAnalyzer',
]
