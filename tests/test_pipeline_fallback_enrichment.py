import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.phases.phase7_report import _apply_ai_report_fallback_enrichment
from pipeline.phases.phase8_validation import run_ai_validation_phase
from pipeline.analyzers_governance import _validation_input_fingerprint
from pipeline.risk_utils import derive_risk_level
from pipeline.schemas import EIMASResult


def test_derive_risk_level_thresholds():
    assert derive_risk_level(0) == "LOW"
    assert derive_risk_level(29.99) == "LOW"
    assert derive_risk_level(30) == "MEDIUM"
    assert derive_risk_level(69.99) == "MEDIUM"
    assert derive_risk_level(70) == "HIGH"
    assert derive_risk_level(100) == "HIGH"


def test_ai_report_fallback_populates_ark_and_news_when_empty():
    result = EIMASResult(timestamp="2026-02-08T00:00:00")
    payload = {
        "report_data": {
            "notable_stocks": [
                {"ticker": "IWM", "notable_reason": "Strong momentum", "change_1d": 3.2, "change_5d": 4.5},
                {"ticker": "GLD", "notable_reason": "Safe-haven demand", "change_1d": -2.5, "change_5d": -1.2},
            ],
            "perplexity_news": "- **IWM**: Rotation into small caps.\n- **GLD**: Gold demand increased.",
        }
    }

    meta = _apply_ai_report_fallback_enrichment(result, payload)

    assert meta["ark_analysis_source"] == "ai_report.notable_stocks"
    assert meta["news_source"] == "ai_report.perplexity_news"
    assert result.ark_analysis.get("derived") is True
    assert len(result.ark_analysis.get("signals", [])) == 2
    assert len(result.news_correlations) == 2
    assert result.news_correlations[0]["source"] == "ai_report.perplexity_news"


def test_ai_validation_phase_marks_skipped_when_not_full_mode(tmp_path: Path):
    result = EIMASResult(timestamp="2026-02-08T00:00:00")
    run_ai_validation_phase(result, full_mode=False, output_dir=tmp_path, output_file="")

    assert result.validation_loop_result["skipped"] is True
    assert result.validation_loop_result["reason"] == "full_mode_disabled"


def test_validation_fingerprint_changes_with_core_inputs():
    base = {
        "final_recommendation": "BULLISH",
        "confidence": 0.66,
        "risk_level": "LOW",
        "risk_score": 12.34,
        "regime": {"regime": "Bull (Low Vol)", "confidence": 0.75},
    }
    same = dict(base)
    changed = dict(base)
    changed["risk_score"] = 18.0

    assert _validation_input_fingerprint(base) == _validation_input_fingerprint(same)
    assert _validation_input_fingerprint(base) != _validation_input_fingerprint(changed)
