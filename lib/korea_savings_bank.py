#!/usr/bin/env python3
"""
Korea Savings Bank Indicators
==============================
한국 저축은행 건전성 지표 수집 모듈.

Real data sources:
  - NPL Ratio (고정이하여신비율):
      FRED series: DDSI06KRA066NWDB
        (Korea, Nonperforming Loans to Total Gross Loans — IMF FSI)
      Note: covers all deposit-taking institutions, not savings banks specifically.
      Savings-bank-specific data: FSS FISIS (http://fisis.fss.or.kr)

  - BIS Capital Ratio (BIS 자기자본비율):
      FRED series: DSSB01KRA066NWDB
        (Korea, Regulatory Capital to Risk-Weighted Assets — IMF FSI)
      한국은행 ECOS: 7.3.3 저축은행 자기자본비율

  - ROA (총자산순이익률):
      한국은행 ECOS API: 통계코드 102Y008 (저축은행 경영지표)
      No FRED equivalent for savings-bank-specific ROA.

Mock values (fallback) reflect FSS 2025 Q3 저축은행 영업실적 추정치.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import os

# KoreaSavingsBankIndicators is defined here as a self-contained dataclass.
# pipeline/schemas.py imports it from this module.


@dataclass
class KoreaSavingsBankIndicators:
    """한국 저축은행 건전성 지표"""
    timestamp: str
    # NPL ratio = 고정이하여신비율 (%)
    npl_ratio: float = 0.0
    # BIS capital adequacy ratio = BIS 자기자본비율 (%)
    bis_capital_ratio: float = 0.0
    # Return on assets = 총자산순이익률 (%)
    roa: float = 0.0
    data_source: str = "fss_mock"
    note: str = ""
    signals: List[str] = field(default_factory=list)
    is_valid: bool = True
    error_msg: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collect_korea_savings_bank_indicators() -> KoreaSavingsBankIndicators:
    """
    한국 저축은행 건전성 지표 수집.

    수집 순서:
    1. FRED API (NPL, BIS 프록시 — 전 은행권 기준)
    2. Mock fallback (FSS 2025Q3 저축은행 영업실적 추정치)

    Returns:
        KoreaSavingsBankIndicators with npl_ratio, bis_capital_ratio, roa.
    """
    timestamp = datetime.now().isoformat()
    npl_ratio: Optional[float] = None
    bis_capital_ratio: Optional[float] = None

    fred_api_key = os.getenv("FRED_API_KEY", "").strip()
    if fred_api_key:
        try:
            import fredapi  # type: ignore[import]
            fred = fredapi.Fred(api_key=fred_api_key)

            # FRED: Korea Nonperforming Loans (all deposit-taking, IMF FSI)
            npl_series = fred.get_series("DDSI06KRA066NWDB")
            if npl_series is not None and not npl_series.dropna().empty:
                npl_ratio = float(npl_series.dropna().iloc[-1])

            # FRED: Korea Regulatory Capital to Risk-Weighted Assets (IMF FSI)
            bis_series = fred.get_series("DSSB01KRA066NWDB")
            if bis_series is not None and not bis_series.dropna().empty:
                bis_capital_ratio = float(bis_series.dropna().iloc[-1])

        except Exception:
            pass  # Fall through to mock values

    # -----------------------------------------------------------------------
    # Mock fallback: FSS 2025Q3 저축은행 영업실적 기준 추정치
    # Sources:
    #   NPL:  금융감독원 보도자료 2025Q3 — 저축은행 고정이하여신비율 약 8.7%
    #   BIS:  금융감독원 2025Q3 — 저축은행 BIS비율 약 14.2% (권고 8% 상회)
    #   ROA:  금융감독원 2025Q3 — 저축은행 ROA 약 -0.3% (부실채권 대손 부담)
    # -----------------------------------------------------------------------
    final_npl = npl_ratio if npl_ratio is not None else 8.7
    final_bis = bis_capital_ratio if bis_capital_ratio is not None else 14.2
    roa_value = -0.3  # No FRED equivalent; FSS estimate used

    source_parts = []
    if npl_ratio is not None or bis_capital_ratio is not None:
        source_parts.append("fred")
    source_parts.append("fss_mock")
    data_source = "+".join(source_parts)

    signals: List[str] = []
    if final_npl > 8.0:
        signals.append(f"NPL 비율 {final_npl:.1f}% — 위험 임계치(8%) 초과")
    if final_bis < 11.0:
        signals.append(f"BIS 비율 {final_bis:.1f}% — 권고치 미만 위험")
    elif final_bis < 14.0:
        signals.append(f"BIS 비율 {final_bis:.1f}% — 적정 범위 하단 주의")
    if roa_value < 0:
        signals.append(f"ROA {roa_value:.2f}% — 수익성 적자")

    return KoreaSavingsBankIndicators(
        timestamp=timestamp,
        npl_ratio=final_npl,
        bis_capital_ratio=final_bis,
        roa=roa_value,
        data_source=data_source,
        note=(
            "NPL/BIS: FRED 수집 시도 후 FSS 2025Q3 저축은행 영업실적 모의값 사용. "
            "ROA: 한국은행 ECOS 또는 FSS FISIS 연동 필요. "
            "저축은행 전용 API: FSS FISIS (http://fisis.fss.or.kr)"
        ),
        signals=signals,
    )
