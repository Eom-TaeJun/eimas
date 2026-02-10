"""
Pipeline execution profiles for phase-level runtime policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PipelineProfile:
    """Phase-level execution policy for a single pipeline run."""

    name: str
    description: str

    # Phase 2 policies
    run_sentiment_bubble: bool = True
    skip_bubble_analysis: bool = False
    run_institutional_frameworks: bool = True
    run_adaptive_portfolio: bool = True

    # Debate / validation
    run_debate: bool = True
    run_phase8_ai_validation: bool = True
    run_phase85_quick_validation: bool = True

    # Optional portfolio modules
    run_backtest: bool = True
    run_attribution: bool = True
    run_stress_test: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_LEGACY_PROFILE = PipelineProfile(
    name="legacy",
    description="Full legacy flow (all phases available by CLI flags).",
)


_US_TRADER_V1_PROFILE = PipelineProfile(
    name="us-trader-v1",
    description=(
        "US institutional trader baseline: keep execution/explainability path, "
        "defer heavy research-only phases."
    ),
    run_sentiment_bubble=True,
    skip_bubble_analysis=True,
    run_institutional_frameworks=False,
    run_adaptive_portfolio=True,
    run_debate=True,
    run_phase8_ai_validation=False,
    run_phase85_quick_validation=False,
    run_backtest=False,
    run_attribution=False,
    run_stress_test=False,
)


_PROFILE_ALIASES = {
    "legacy": "legacy",
    "default": "legacy",
    "us-trader-v1": "us-trader-v1",
    "us_trader_v1": "us-trader-v1",
    "trader": "us-trader-v1",
}


_PROFILES = {
    "legacy": _LEGACY_PROFILE,
    "us-trader-v1": _US_TRADER_V1_PROFILE,
}


def pipeline_profile_choices() -> tuple[str, ...]:
    """Canonical profile names for argparse choices."""
    return tuple(_PROFILES.keys())


def resolve_pipeline_profile(name: str | None) -> PipelineProfile:
    """Resolve canonical pipeline profile from name/alias."""
    raw = (name or "legacy").strip().lower()
    canonical = _PROFILE_ALIASES.get(raw)
    if canonical is None:
        supported = ", ".join(pipeline_profile_choices())
        raise ValueError(f"Unsupported profile '{name}'. Supported: {supported}")
    return _PROFILES[canonical]
