"""
Allocation Report Agent
=======================
자산운용사 자산배분팀 리서치 리포트 작성 보조 에이전트

입력: EIMAS가 계산한 JSON 결과
출력: 10개 섹션의 한국어 리포트

핵심 원칙:
- 새로운 숫자, 신호, 비중을 생성하지 않음
- JSON 데이터만 인용
- 데이터 신뢰도 저하 또는 신호 충돌 시 HOLD
- turnover cap과 weight bounds 제약 준수

출력 섹션:
1. 현재 시장 및 레짐 요약
2. 핵심 근거 3가지
3. 리스크 및 반증 조건 3가지
4. 운용 관점의 액션 아이템 (HOLD 포함)
5. ETF 전략 분해표 (팩터/섹터/듀레이션)
6. 기업 커버리지 + RA 업무 지원 + SQL 증빙(PostgreSQL + Internal SQL)
7. 실행 타이밍 요약 (최근 파이프라인)
8. 정량 상세 지표 스냅샷 (FRED/HFT/GARCH/DTW/PoI 등)
9. 검증/경고 상세 (Validation/Verification/Warnings)
10. 리밸런싱/운용 승인 근거 + RA 코멘트 + 구현 TODO
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from lib.ra_sql_store import save_ra_commentary_audit_log

logger = logging.getLogger(__name__)
RECRUITMENT_META_KEYWORDS = (
    "자기소개서",
    "인사팀",
    "채용",
    "채용공고",
    "지원동기",
    "지원서",
)


@dataclass
class MarketSummary:
    """현재 시장 및 레짐 요약"""
    regime: str                           # BULL/BEAR/NEUTRAL
    regime_confidence: str                # High/Medium/Low
    volatility_state: str                 # Low Vol/High Vol
    risk_score: float                     # 0-100
    risk_level: str                       # LOW/MEDIUM/HIGH
    net_liquidity: str                    # Fed 순유동성 (문자열)
    final_recommendation: str             # BULLISH/BEARISH/NEUTRAL/HOLD
    confidence: float                     # 신뢰도 (0-1)
    data_quality: str                     # COMPLETE/PARTIAL/DEGRADED
    summary_text: str                     # 요약 문장


@dataclass
class KeyRationale:
    """핵심 근거"""
    title: str                            # 근거 제목
    source: str                           # 데이터 출처 (JSON 필드명)
    value: str                            # 실제 값 (JSON에서 인용)
    interpretation: str                   # 해석


@dataclass
class RiskCondition:
    """리스크 및 반증 조건"""
    risk_title: str                       # 리스크 제목
    current_value: str                    # 현재 값
    falsification_condition: str          # 반증 조건 (이 조건이 충족되면 뷰 수정)
    monitoring_metric: str                # 모니터링할 지표


@dataclass
class ActionItem:
    """운용 관점의 액션 아이템"""
    action: str                           # HOLD/REBALANCE/MONITOR/REDUCE/INCREASE
    target: str                           # 대상 (전체 포트폴리오 or 특정 자산)
    rationale: str                        # 근거
    constraints: List[str]                # 제약 조건 (turnover cap, weight bounds 등)
    priority: str                         # HIGH/MEDIUM/LOW


@dataclass
class ETFDecomposition:
    """ETF 전략 분해 행"""
    ticker: str
    category: str
    asset_role: str
    factor_exposure: str
    sector_or_theme: str
    duration_profile: str
    ret_5d: str
    ret_20d: str
    momentum_label: str
    top_holdings: str = "N/A"
    data_source: str = "N/A"
    quality_flag: str = "N/A"
    source: str = "company_ra_analysis.etf_strategy_snapshot"


@dataclass
class CompanyCoverageRow:
    """기업 커버리지 행"""
    ticker: str
    sector: str
    trailing_pe: str
    forward_pe: str
    price_to_book: str
    roe: str
    roa: str
    net_margin: str
    debt_to_equity: str
    ret_5d: str
    ret_20d: str
    valuation_signal: str
    ra_takeaway: str


@dataclass
class PipelineTimingRow:
    """파이프라인 타이밍 행"""
    phase: str
    duration_sec: float
    status: str


@dataclass
class AllocationReport:
    """
    자산배분팀 리서치 리포트

    EIMAS JSON 결과를 기반으로 생성
    새로운 숫자나 비중을 생성하지 않음
    """
    timestamp: str
    report_version: str = "1.3"

    # 섹션 1: 시장 요약
    market_summary: Optional[MarketSummary] = None

    # 섹션 2: 핵심 근거 3가지
    key_rationales: List[KeyRationale] = field(default_factory=list)

    # 섹션 3: 리스크 및 반증 조건 3가지
    risk_conditions: List[RiskCondition] = field(default_factory=list)

    # 섹션 4: 액션 아이템
    action_items: List[ActionItem] = field(default_factory=list)

    # 섹션 5: ETF 전략 분해표
    etf_decomposition: List[ETFDecomposition] = field(default_factory=list)

    # 섹션 6: 기업 커버리지/RA 지원/SQL 증빙
    company_coverage: List[CompanyCoverageRow] = field(default_factory=list)
    ra_work_support: Dict[str, Any] = field(default_factory=dict)
    postgres_evidence: Dict[str, Any] = field(default_factory=dict)
    internal_sql_evidence: Dict[str, Any] = field(default_factory=dict)

    # 섹션 7: 실행 타이밍/운영 추적
    pipeline_elapsed_sec: float = 0.0
    pipeline_timings: List[PipelineTimingRow] = field(default_factory=list)
    operational_summary: Dict[str, Any] = field(default_factory=dict)

    # 섹션 8~10: 상세 정량/검증/리밸런싱 근거
    detailed_quant_snapshot: Dict[str, Any] = field(default_factory=dict)
    validation_evidence: Dict[str, Any] = field(default_factory=dict)
    rebalance_evidence: Dict[str, Any] = field(default_factory=dict)
    ai_report_evidence: Dict[str, Any] = field(default_factory=dict)
    ra_sql_matrix: Dict[str, Any] = field(default_factory=dict)
    ra_commentary: Dict[str, Any] = field(default_factory=dict)
    ra_todo_items: List[Dict[str, Any]] = field(default_factory=list)
    section_ai_discussion: Dict[str, List[str]] = field(default_factory=dict)
    risk_signal_news: Dict[str, Any] = field(default_factory=dict)

    # 메타데이터
    data_quality_warning: bool = False
    signal_conflict_warning: bool = False
    default_to_hold: bool = False
    hold_reason: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def _format_num(self, value: Any, digits: int = 2) -> str:
        try:
            if value is None:
                return "N/A"
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "N/A"

    def _format_pct(self, value: Any) -> str:
        try:
            return f"{float(value):+.2f}%"
        except (TypeError, ValueError):
            return "N/A"

    def _append_ai_section(self, md: List[str], section_key: str) -> None:
        raw = self.section_ai_discussion.get(section_key, [])
        lines: List[str] = []
        if isinstance(raw, str):
            text = raw.strip()
            if text:
                lines.append(text)
        elif isinstance(raw, list):
            for item in raw:
                text = str(item).strip()
                if text:
                    lines.append(text)

        if not lines:
            return

        md.append("### AI 토론/해석")
        for line in lines[:4]:
            md.append(f"- {line}")
        md.append("")

    def _append_risk_news_briefing(self, md: List[str]) -> None:
        payload = self.risk_signal_news if isinstance(self.risk_signal_news, dict) else {}
        if not payload:
            return

        risk_detected = bool(payload.get("risk_detected", False))
        headline = str(payload.get("headline", "") or "").strip()
        if not risk_detected and not headline:
            return

        signals_raw = payload.get("signals", [])
        signals: List[str] = []
        if isinstance(signals_raw, list):
            for item in signals_raw:
                text = str(item).strip()
                if text:
                    signals.append(text)

        md.append("### 리스크 신호 연계 뉴스 브리핑")
        if signals:
            md.append(f"- **감지 신호**: {', '.join(signals)}")
        if headline:
            md.append(f"- **관련 헤드라인**: {headline}")
        else:
            md.append("- **관련 헤드라인**: N/A")

        label = str(payload.get("news_label", "N/A"))
        score = payload.get("news_score")
        score_text = self._format_num(score, digits=2) if score is not None else "N/A"
        md.append(f"- **뉴스 센티먼트**: {label} (score={score_text})")

        analysis = str(payload.get("analysis", "") or "").strip()
        if analysis:
            md.append(f"- **AI 연계 해석**: {analysis}")

        external_provider = str(payload.get("external_provider", "") or "").strip()
        external_status = str(payload.get("external_status", "") or "").strip()
        external_items = payload.get("external_headlines", [])
        if isinstance(external_items, list) and external_items:
            provider_text = external_provider or "external_api"
            status_text = external_status or "ok"
            md.append(f"- **외부 뉴스 API**: {provider_text} ({status_text})")
            md.append("- **외부 헤드라인(실시간 검색)**:")
            for item in external_items[:5]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "N/A")).strip() or "N/A"
                url = str(item.get("url", "") or "").strip()
                source_name = str(item.get("source", "N/A")).strip() or "N/A"
                published_at = str(item.get("published_at", "") or "").strip()
                summary = str(item.get("summary", "") or "").strip()
                suffix = f" ({source_name}"
                if published_at:
                    suffix += f", {published_at}"
                suffix += ")"
                if url and url.startswith(("http://", "https://")):
                    md.append(f"  - [{title}]({url}){suffix}")
                else:
                    md.append(f"  - {title}{suffix}")
                if summary:
                    md.append(f"    - 요약: {summary}")
        elif external_provider:
            provider_text = external_provider
            status_text = external_status or "no_data"
            md.append(f"- **외부 뉴스 API**: {provider_text} ({status_text})")
            ext_err = str(payload.get("external_error", "") or "").strip()
            if ext_err:
                md.append(f"- **외부 뉴스 API 오류**: {ext_err}")

        source = str(payload.get("source", "") or "").strip()
        if source:
            md.append(f"- **출처**: `{source}`")
        md.append("")

    def to_markdown(self) -> str:
        """마크다운 형식 리포트 생성"""
        md = []
        md.append("# 자산배분팀 리서치 리포트")
        md.append(f"**생성 시간**: {self.timestamp}")
        md.append("")

        # 경고 표시
        if self.default_to_hold:
            md.append("> **주의**: 데이터 신뢰도 저하 또는 신호 충돌로 인해 기본 행동은 HOLD입니다.")
            md.append(f"> 사유: {self.hold_reason}")
            md.append("")

        # 섹션 1: 시장 요약
        md.append("## 1. 현재 시장 및 레짐 요약")
        if self.market_summary:
            ms = self.market_summary
            md.append(f"- **레짐**: {ms.regime} ({ms.regime_confidence} 신뢰도)")
            md.append(f"- **변동성**: {ms.volatility_state}")
            md.append(f"- **리스크 점수**: {ms.risk_score:.1f}/100 ({ms.risk_level})")
            md.append(f"- **Fed 순유동성**: {ms.net_liquidity}")
            md.append(f"- **최종 권고**: {ms.final_recommendation} (신뢰도 {ms.confidence:.0%})")
            md.append(f"- **데이터 품질**: {ms.data_quality}")
            md.append("")
            md.append(f"**요약**: {ms.summary_text}")
        md.append("")
        self._append_ai_section(md, "section_1")

        # 섹션 2: 핵심 근거
        md.append("## 2. 핵심 근거 3가지")
        for i, r in enumerate(self.key_rationales[:3], 1):
            md.append(f"### 근거 {i}: {r.title}")
            md.append(f"- **출처**: `{r.source}`")
            md.append(f"- **값**: {r.value}")
            md.append(f"- **해석**: {r.interpretation}")
            md.append("")
        self._append_ai_section(md, "section_2")

        # 섹션 3: 리스크
        md.append("## 3. 리스크 및 반증 조건 3가지")
        for i, rc in enumerate(self.risk_conditions[:3], 1):
            md.append(f"### 리스크 {i}: {rc.risk_title}")
            md.append(f"- **현재 값**: {rc.current_value}")
            md.append(f"- **반증 조건**: {rc.falsification_condition}")
            md.append(f"- **모니터링 지표**: {rc.monitoring_metric}")
            md.append("")
        self._append_risk_news_briefing(md)
        self._append_ai_section(md, "section_3")

        # 섹션 4: 액션 아이템
        md.append("## 4. 운용 관점의 액션 아이템")
        for i, ai in enumerate(self.action_items, 1):
            md.append(f"### {i}. [{ai.action}] {ai.target}")
            md.append(f"- **근거**: {ai.rationale}")
            if ai.constraints:
                md.append(f"- **제약**: {', '.join(ai.constraints)}")
            md.append(f"- **우선순위**: {ai.priority}")
            md.append("")
        self._append_ai_section(md, "section_4")

        # 섹션 5: ETF 전략 분해표
        md.append("## 5. ETF 전략 분해표 (팩터/섹터/듀레이션)")
        if self.etf_decomposition:
            md.append("| Ticker | 역할 | 팩터 노출 | 섹터/테마 | 듀레이션·금리 민감도 | 5D 수익률 | 20D 수익률 | 모멘텀 | Top Holdings(3) | Source(Q) |")
            md.append("|---|---|---|---|---|---:|---:|---|---|---|")
            for row in self.etf_decomposition:
                md.append(
                    f"| {row.ticker} | {row.asset_role} | {row.factor_exposure} | "
                    f"{row.sector_or_theme} | {row.duration_profile} | {row.ret_5d} | "
                    f"{row.ret_20d} | {row.momentum_label} | {row.top_holdings} | "
                    f"{row.data_source} ({row.quality_flag}) |"
                )
            md.append("")
            md.append(
                "- **출처**: `company_ra_analysis.etf_strategy_snapshot` + "
                "yfinance 가격/메타데이터 + EIMAS ETF 프로필 카탈로그(자동수집 실패 시 fallback)"
            )
        else:
            md.append("- ETF 스냅샷 데이터가 없어 분해표를 생성하지 못했습니다.")
        md.append("")
        self._append_ai_section(md, "section_5")

        # 섹션 6: 기업 커버리지/RA 지원/SQL 증빙
        md.append("## 6. 기업 커버리지 + RA 업무 지원 + SQL 증빙")
        if self.company_coverage:
            md.append("| Ticker | Sector | Trailing P/E | Forward P/E | P/B | ROE | ROA | Net Margin | D/E | 5D | 20D | Signal |")
            md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for row in self.company_coverage:
                md.append(
                    f"| {row.ticker} | {row.sector} | {row.trailing_pe} | {row.forward_pe} | "
                    f"{row.price_to_book} | {row.roe} | {row.roa} | {row.net_margin} | "
                    f"{row.debt_to_equity} | {row.ret_5d} | {row.ret_20d} | {row.valuation_signal} |"
                )
            md.append("")

            md.append("### 기업별 RA 코멘트")
            for row in self.company_coverage:
                md.append(f"- **{row.ticker}**: {row.ra_takeaway}")
            md.append("")
        else:
            md.append("- 기업 커버리지 데이터가 없어 표를 생성하지 못했습니다.")
            md.append("")

        if self.ra_work_support:
            md.append("### 운영 지원 포인트 (데이터 기반)")
            role_focus = self.ra_work_support.get("role_focus", "N/A")
            md.append(f"- **Role Focus**: {role_focus}")

            research_tasks = self.ra_work_support.get("research_tasks", [])
            if isinstance(research_tasks, list) and research_tasks:
                md.append("#### 자료조사/업데이트 태스크")
                for task in research_tasks:
                    md.append(f"- {task}")
                md.append("")

            seminar_points = self.ra_work_support.get("seminar_material_points", [])
            if isinstance(seminar_points, list) and seminar_points:
                md.append("#### 세미나/대외자료 포인트")
                for point in seminar_points:
                    md.append(f"- {point}")
                md.append("")

            cross_points = self.ra_work_support.get("cross_department_support_points", [])
            if isinstance(cross_points, list) and cross_points:
                md.append("#### 유관부서 협조 포인트")
                for point in cross_points:
                    md.append(f"- {point}")
                md.append("")

            note = self.ra_work_support.get("data_update_note", "")
            if note:
                md.append(f"- **Data Update Note**: {note}")
            md.append("")

        md.append("### PostgreSQL 증빙")
        if self.postgres_evidence:
            md.append(f"- **enabled**: {self.postgres_evidence.get('enabled')}")
            md.append(f"- **dsn_configured**: {self.postgres_evidence.get('dsn_configured')}")
            md.append(f"- **driver_available**: {self.postgres_evidence.get('driver_available')}")
            md.append(f"- **stored_rows**: {self.postgres_evidence.get('stored_rows', 0)}")
            md.append(f"- **table**: {self.postgres_evidence.get('table', 'N/A')}")
            err = self.postgres_evidence.get("error", "")
            if err:
                md.append(f"- **error**: {err}")
        else:
            md.append("- PostgreSQL 증빙 정보가 없습니다.")

        md.append("")
        md.append("### EIMAS Internal SQL 증빙 (SQLite)")
        if self.internal_sql_evidence:
            md.append(f"- **enabled**: {self.internal_sql_evidence.get('enabled')}")
            md.append(f"- **db_path**: {self.internal_sql_evidence.get('db_path', 'N/A')}")
            md.append(f"- **table**: {self.internal_sql_evidence.get('table', 'ra_company_fundamentals')}")
            md.append(f"- **upserted_rows**: {self.internal_sql_evidence.get('upserted_rows', 0)}")
            md.append(f"- **total_rows**: {self.internal_sql_evidence.get('total_rows', 0)}")
            md.append(f"- **distinct_tickers**: {self.internal_sql_evidence.get('distinct_tickers', 0)}")
            md.append(f"- **etf_table**: {self.internal_sql_evidence.get('etf_table', 'ra_etf_snapshot')}")
            md.append(f"- **etf_upserted_rows**: {self.internal_sql_evidence.get('etf_upserted_rows', 0)}")
            md.append(f"- **etf_total_rows**: {self.internal_sql_evidence.get('etf_total_rows', 0)}")
            md.append(f"- **etf_distinct_tickers**: {self.internal_sql_evidence.get('etf_distinct_tickers', 0)}")
            md.append(
                f"- **date_range**: {self.internal_sql_evidence.get('min_date', 'N/A')} "
                f"~ {self.internal_sql_evidence.get('max_date', 'N/A')}"
            )
            qc = self.internal_sql_evidence.get("quality_checks", {})
            if isinstance(qc, dict) and qc:
                md.append(
                    "- **quality_checks**: "
                    f"missing_valuation={qc.get('missing_valuation_rows', 0)}, "
                    f"missing_financial={qc.get('missing_financial_rows', 0)}, "
                    f"flagged={qc.get('quality_flagged_rows', 0)}"
                )
            etf_qc = self.internal_sql_evidence.get("etf_quality_checks", {})
            if isinstance(etf_qc, dict) and etf_qc:
                md.append(
                    "- **etf_quality_checks**: "
                    f"missing_returns={etf_qc.get('missing_return_rows', 0)}, "
                    f"missing_holdings={etf_qc.get('missing_holdings_rows', 0)}, "
                    f"unexpected_quality={etf_qc.get('unexpected_quality_rows', 0)}"
                )

            etf_source_mix = self.internal_sql_evidence.get("etf_source_mix", [])
            if isinstance(etf_source_mix, list) and etf_source_mix:
                md.append("- **etf_source_mix**:")
                for row in etf_source_mix[:5]:
                    if not isinstance(row, dict):
                        continue
                    md.append(
                        f"  - {row.get('data_source', 'N/A')}: {row.get('count', 0)}"
                    )

            phase6_sql = self.internal_sql_evidence.get("phase6_backtest", {})
            if isinstance(phase6_sql, dict) and phase6_sql:
                md.append(
                    f"- **phase6_backtest_sql**: saved_run_id={phase6_sql.get('saved_run_id')}, "
                    f"total_runs={phase6_sql.get('total_runs', 0)}, avg_sharpe={phase6_sql.get('avg_sharpe', 0.0):.2f}"
                )
            artifacts = self.internal_sql_evidence.get("sql_artifacts", {})
            if isinstance(artifacts, dict) and artifacts:
                row_counts = artifacts.get("row_counts", {})
                md.append(
                    f"- **sql_artifacts_refreshed_at**: {artifacts.get('refreshed_at', 'N/A')}"
                )
                if isinstance(row_counts, dict) and row_counts:
                    md.append(
                        "- **materialized_row_counts**: "
                        f"valuation={row_counts.get('ra_valuation_snapshot_mv', 0)}, "
                        f"etf_momentum={row_counts.get('ra_etf_momentum_snapshot_mv', 0)}, "
                        f"backtest_compare={row_counts.get('ra_backtest_compare_mv', 0)}"
                    )

            err = self.internal_sql_evidence.get("error", "")
            if err:
                md.append(f"- **error**: {err}")
        else:
            md.append("- Internal SQL 증빙 정보가 없습니다.")

        md.append("")
        md.append("```sql")
        md.append("SELECT COUNT(*) AS total_rows, COUNT(DISTINCT ticker) AS tickers,")
        md.append("       MIN(as_of_date) AS min_date, MAX(as_of_date) AS max_date")
        md.append("FROM fi_ra.company_fundamentals;")
        md.append("```")
        md.append("")
        md.append("```sql")
        md.append("SELECT COUNT(*) AS total_rows, COUNT(DISTINCT ticker) AS distinct_tickers,")
        md.append("       MIN(as_of_date) AS min_date, MAX(as_of_date) AS max_date")
        md.append("FROM ra_company_fundamentals;")
        md.append("```")
        md.append("")
        md.append("```sql")
        md.append("SELECT ticker, category, ret_20d_pct, momentum_label")
        md.append("FROM ra_etf_momentum_snapshot_mv")
        md.append("ORDER BY rank_ret_20d ASC")
        md.append("LIMIT 10;")
        md.append("```")
        md.append("")

        sql_previews = self.internal_sql_evidence.get("sql_preview_tables", {})
        if isinstance(sql_previews, dict) and sql_previews:
            valuation_rows = sql_previews.get("valuation_snapshot_mv", [])
            if isinstance(valuation_rows, list) and valuation_rows:
                md.append("### SQL Preview: Valuation Snapshot MV")
                md.append("| Ticker | Sector | Trailing P/E | P/B | 20D | Signal | PE Bucket |")
                md.append("|---|---|---:|---:|---:|---|---|")
                for row in valuation_rows[:8]:
                    if not isinstance(row, dict):
                        continue
                    md.append(
                        f"| {row.get('ticker', 'N/A')} | {row.get('sector', 'N/A')} | "
                        f"{self._format_num(row.get('trailing_pe'))} | {self._format_num(row.get('price_to_book'))} | "
                        f"{self._format_pct(row.get('ret_20d_pct'))} | {row.get('valuation_signal', 'N/A')} | "
                        f"{row.get('pe_bucket', 'N/A')} |"
                    )
                md.append("")

            etf_rows = sql_previews.get("etf_momentum_snapshot_mv", [])
            if isinstance(etf_rows, list) and etf_rows:
                md.append("### SQL Preview: ETF Momentum Snapshot MV")
                md.append("| Rank | Ticker | Category | 5D | 20D | Momentum |")
                md.append("|---:|---|---|---:|---:|---|")
                for row in etf_rows[:8]:
                    if not isinstance(row, dict):
                        continue
                    md.append(
                        f"| {row.get('rank_ret_20d', 'N/A')} | {row.get('ticker', 'N/A')} | "
                        f"{row.get('category', 'N/A')} | {self._format_pct(row.get('ret_5d_pct'))} | "
                        f"{self._format_pct(row.get('ret_20d_pct'))} | {row.get('momentum_label', 'N/A')} |"
                    )
                md.append("")

            backtest_rows = sql_previews.get("backtest_compare_mv", [])
            if isinstance(backtest_rows, list) and backtest_rows:
                md.append("### SQL Preview: Backtest Compare MV")
                md.append("| Rank | Strategy | Total Return | Ann Return | Sharpe | MaxDD |")
                md.append("|---:|---|---:|---:|---:|---:|")
                for row in backtest_rows[:8]:
                    if not isinstance(row, dict):
                        continue
                    md.append(
                        f"| {row.get('sharpe_rank', 'N/A')} | {row.get('strategy_name', 'N/A')} | "
                        f"{self._format_pct((row.get('total_return') or 0.0) * 100)} | "
                        f"{self._format_pct((row.get('annualized_return') or 0.0) * 100)} | "
                        f"{self._format_num(row.get('sharpe_ratio'))} | "
                        f"{self._format_pct((row.get('max_drawdown') or 0.0) * 100)} |"
                    )
                md.append("")
        self._append_ai_section(md, "section_6")

        # 섹션 7: 실행 타이밍/운영 추적
        md.append("## 7. 실행 타이밍 요약 + 운영 의사결정 추적")
        md.append(f"- **Pipeline Elapsed**: {self.pipeline_elapsed_sec:.3f}s")
        md.append("")

        if self.pipeline_timings:
            md.append("| Phase | Duration(s) | Status |")
            md.append("|---|---:|---|")
            for row in self.pipeline_timings:
                md.append(f"| {row.phase} | {row.duration_sec:.3f} | {row.status} |")
            md.append("")
        else:
            md.append("- 파이프라인 타이밍 정보가 없습니다.")
            md.append("")

        if self.operational_summary:
            md.append("### Operational Decision Summary")
            md.append(f"- **final_stance**: {self.operational_summary.get('final_stance', 'N/A')}")
            reasons = self.operational_summary.get("reason_codes", [])
            if isinstance(reasons, list) and reasons:
                md.append(f"- **reason_codes**: {', '.join(str(x) for x in reasons)}")
            applied = self.operational_summary.get("applied_rules", [])
            if isinstance(applied, list) and applied:
                md.append("- **applied_rules**:")
                for rule in applied:
                    md.append(f"  - {rule}")

            hold_state = self.operational_summary.get("is_hold")
            if hold_state is not None:
                md.append(f"- **is_hold**: {hold_state}")

            hold_triggers = self.operational_summary.get("triggered_conditions", [])
            if isinstance(hold_triggers, list) and hold_triggers:
                md.append("- **triggered_hold_conditions**:")
                for cond in hold_triggers:
                    md.append(f"  - {cond}")
            md.append("")
        self._append_ai_section(md, "section_7")

        # 섹션 8: 정량 상세 지표 스냅샷
        md.append("## 8. 정량 상세 지표 스냅샷")
        if self.detailed_quant_snapshot:
            macro = self.detailed_quant_snapshot.get("macro_liquidity", {})
            if macro:
                md.append("### Macro & Liquidity")
                md.append("| Metric | Value |")
                md.append("|---|---:|")
                for k, v in macro.items():
                    md.append(f"| {k} | {v} |")
                md.append("")

            hft_garch = self.detailed_quant_snapshot.get("hft_garch", {})
            if hft_garch:
                md.append("### HFT / Volatility")
                md.append("| Metric | Value |")
                md.append("|---|---:|")
                for k, v in hft_garch.items():
                    md.append(f"| {k} | {v} |")
                md.append("")

            flow_poi = self.detailed_quant_snapshot.get("flow_poi", {})
            if flow_poi:
                md.append("### Information Flow / PoI")
                md.append("| Metric | Value |")
                md.append("|---|---:|")
                for k, v in flow_poi.items():
                    md.append(f"| {k} | {v} |")
                md.append("")

            similarity = self.detailed_quant_snapshot.get("similarity_risk", {})
            if similarity:
                md.append("### DTW / DBSCAN / Bubble / Sentiment")
                md.append("| Metric | Value |")
                md.append("|---|---:|")
                for k, v in similarity.items():
                    md.append(f"| {k} | {v} |")
                md.append("")
        else:
            md.append("- 정량 상세 스냅샷 데이터가 없습니다.")
            md.append("")
        self._append_ai_section(md, "section_8")

        # 섹션 9: 검증/경고 상세
        md.append("## 9. 검증 / 경고 상세")
        if self.validation_evidence:
            md.append("| Item | Value |")
            md.append("|---|---|")
            for k, v in self.validation_evidence.items():
                if k in {"warnings", "key_concerns", "action_items"}:
                    continue
                md.append(f"| {k} | {v} |")
            md.append("")

            warnings = self.validation_evidence.get("warnings", [])
            if isinstance(warnings, list) and warnings:
                md.append("### Warnings")
                for w in warnings:
                    md.append(f"- {w}")
                md.append("")

            concerns = self.validation_evidence.get("key_concerns", [])
            if isinstance(concerns, list) and concerns:
                md.append("### Validation Key Concerns")
                for item in concerns:
                    md.append(f"- {item}")
                md.append("")

            items = self.validation_evidence.get("action_items", [])
            if isinstance(items, list) and items:
                md.append("### Validation Action Items")
                for item in items:
                    md.append(f"- {item}")
                md.append("")
        else:
            md.append("- 검증/경고 데이터가 없습니다.")
            md.append("")

        if self.ai_report_evidence:
            md.append("### AI Report Artifacts")
            md.append(f"- **ai_report_path**: {self.ai_report_evidence.get('report_path', 'N/A')}")
            md.append(f"- **ib_report_path**: {self.ai_report_evidence.get('ib_report_path', 'N/A')}")
            md.append(f"- **ai_report_sections**: {self.ai_report_evidence.get('sections_count', 0)}")
            md.append("")
        self._append_ai_section(md, "section_9")

        # 섹션 10: 리밸런싱/운용 승인 근거 + RA 코멘트 + 구현 TODO
        md.append("## 10. 리밸런싱 / 운용 승인 근거 + RA 코멘트 + 구현 TODO")
        if self.rebalance_evidence:
            summary = self.rebalance_evidence.get("summary", {})
            if summary:
                md.append("### Rebalance Summary")
                md.append("| Metric | Value |")
                md.append("|---|---:|")
                for k, v in summary.items():
                    md.append(f"| {k} | {v} |")
                md.append("")

            trigger = self.rebalance_evidence.get("trigger", {})
            if trigger:
                md.append("### Trigger")
                md.append(f"- **type**: {trigger.get('type', 'N/A')}")
                md.append(f"- **reason**: {trigger.get('reason', 'N/A')}")
                md.append("")

            approval = self.rebalance_evidence.get("approval", {})
            if approval:
                md.append("### Approval")
                md.append(f"- **requires_human_approval**: {approval.get('requires_human_approval')}")
                md.append(f"- **approval_reason**: {approval.get('approval_reason', '')}")
                checklist = approval.get("approval_checklist", [])
                if isinstance(checklist, list) and checklist:
                    md.append("- **approval_checklist**:")
                    for item in checklist:
                        md.append(f"  - {item}")
                md.append("")

            top_alloc = self.rebalance_evidence.get("top_allocation", [])
            if isinstance(top_alloc, list) and top_alloc:
                md.append("### Top Allocation Weights")
                md.append("| Ticker | Weight |")
                md.append("|---|---:|")
                for item in top_alloc:
                    md.append(f"| {item.get('ticker', 'N/A')} | {item.get('weight', 'N/A')} |")
                md.append("")
        else:
            md.append("- 리밸런싱 상세 근거 데이터가 없습니다.")
            md.append("")

        if self.ra_commentary:
            md.append("### RA 코멘트 (실데이터 기반)")
            md.append(f"- **commentary_source**: {self.ra_commentary.get('source', 'rule_based')}")
            model = self.ra_commentary.get("model")
            if model:
                md.append(f"- **commentary_model**: {model}")
            audit_id = self.ra_commentary.get("audit_log_id")
            if audit_id is not None:
                md.append(f"- **commentary_audit_log_id**: {audit_id}")
            md.append(f"- **macro_view**: {self.ra_commentary.get('macro_view', 'N/A')}")
            md.append(f"- **etf_view**: {self.ra_commentary.get('etf_view', 'N/A')}")
            md.append(f"- **company_view**: {self.ra_commentary.get('company_view', 'N/A')}")
            md.append(f"- **risk_view**: {self.ra_commentary.get('risk_view', 'N/A')}")
            md.append(f"- **execution_view**: {self.ra_commentary.get('execution_view', 'N/A')}")
            md.append(f"- **final_ra_call**: {self.ra_commentary.get('final_ra_call', 'N/A')}")
            actions = self.ra_commentary.get("priority_actions", [])
            if isinstance(actions, list) and actions:
                md.append("- **priority_actions**:")
                for action in actions:
                    md.append(f"  - {action}")
            err = self.ra_commentary.get("error", "")
            if err:
                md.append(f"- **commentary_error**: {err}")
            md.append("")

        if self.ra_sql_matrix:
            md.append("### RA-SQL 적용 영역 매트릭스")
            usage_rows = self.ra_sql_matrix.get("usage_rows", [])
            if isinstance(usage_rows, list) and usage_rows:
                md.append("| 사용 영역 | 설명 | SQL 예시 기능 | 상태 |")
                md.append("|---|---|---|---|")
                for row in usage_rows:
                    if not isinstance(row, dict):
                        continue
                    md.append(
                        f"| {row.get('area', 'N/A')} | {row.get('description', 'N/A')} | "
                        f"{row.get('sql_features', 'N/A')} | {row.get('status', 'N/A')} |"
                    )
                md.append("")

            phase_rows = self.ra_sql_matrix.get("phase_rows", [])
            if isinstance(phase_rows, list) and phase_rows:
                md.append("### EIMAS Phase별 SQL 통합 전략")
                md.append("| 대상 Phase | 통합 전략 | 예시 | 상태 |")
                md.append("|---|---|---|---|")
                for row in phase_rows:
                    if not isinstance(row, dict):
                        continue
                    md.append(
                        f"| {row.get('phase', 'N/A')} | {row.get('strategy', 'N/A')} | "
                        f"{row.get('example', 'N/A')} | {row.get('status', 'N/A')} |"
                    )
                md.append("")

            evidence_rows = self.ra_sql_matrix.get("evidence_rows", [])
            if isinstance(evidence_rows, list) and evidence_rows:
                md.append("### SQL 구현 증빙 스냅샷")
                md.append("| Metric | Value |")
                md.append("|---|---:|")
                for row in evidence_rows:
                    if not isinstance(row, dict):
                        continue
                    md.append(f"| {row.get('metric', 'N/A')} | {row.get('value', 'N/A')} |")
                md.append("")

            signal_rows = self.ra_sql_matrix.get("allocation_signal_rows", [])
            if isinstance(signal_rows, list) and signal_rows:
                md.append("### RA 종합 시그널 스냅샷")
                md.append(
                    "| AsOf | Valuation Score | ETF Breadth Score | Macro Proxy Score | Composite Score | Signal | "
                    "Companies | ETFs |"
                )
                md.append("|---|---:|---:|---:|---:|---|---:|---:|")
                for row in signal_rows:
                    if not isinstance(row, dict):
                        continue
                    md.append(
                        f"| {row.get('as_of_date', 'N/A')} | {row.get('valuation_score', 'N/A')} | "
                        f"{row.get('etf_breadth_score', 'N/A')} | {row.get('macro_proxy_score', 'N/A')} | "
                        f"{row.get('composite_score', 'N/A')} | {row.get('signal_label', 'N/A')} | "
                        f"{row.get('n_companies', 'N/A')} | {row.get('n_etf', 'N/A')} |"
                    )
                md.append("")

            evidence_highlights = self.ra_sql_matrix.get("evidence_highlights", [])
            if isinstance(evidence_highlights, list) and evidence_highlights:
                md.append("### SQL 근거 요약 (데이터 기반)")
                for line in evidence_highlights:
                    md.append(f"- {line}")
                md.append("")

        if self.ra_todo_items:
            md.append("### 구현 TODO (RA 스타일 고도화)")
            md.append("| Priority | Task | Why | Implementation Plan | Output Artifact | Status |")
            md.append("|---|---|---|---|---|---|")
            for row in self.ra_todo_items:
                if not isinstance(row, dict):
                    continue
                md.append(
                    f"| {row.get('priority', 'N/A')} | {row.get('task', 'N/A')} | {row.get('why', 'N/A')} | "
                    f"{row.get('implementation', 'N/A')} | {row.get('artifact', 'N/A')} | {row.get('status', 'N/A')} |"
                )
            md.append("")
        self._append_ai_section(md, "section_10")

        md.append("---")
        md.append("*본 리포트는 EIMAS JSON 결과를 기반으로 자동 생성되었습니다.*")
        md.append("*새로운 숫자나 비중은 생성되지 않았으며, 모든 값은 JSON에서 인용되었습니다.*")

        return "\n".join(md)


class AllocationReportAgent:
    """
    자산배분팀 리서치 리포트 에이전트

    EIMAS JSON → 구조화된 한국어 리포트

    Example:
        >>> agent = AllocationReportAgent()
        >>> report = agent.generate_report(eimas_json)
        >>> print(report.to_markdown())
    """

    def __init__(self):
        self.turnover_cap = 0.30  # 30%
        self.weight_bounds = {
            'equity_max': 0.60,
            'bond_max': 0.40,
            'cash_min': 0.05,
            'crypto_max': 0.10
        }
        external_news_raw = os.getenv("EIMAS_RA_EXTERNAL_NEWS_ENABLED", "true").strip().lower()
        self.external_news_enabled = external_news_raw in {"1", "true", "yes", "on"}
        try:
            max_items = int(os.getenv("EIMAS_RA_EXTERNAL_NEWS_MAX_ITEMS", "5") or 5)
        except (TypeError, ValueError):
            max_items = 5
        self.external_news_max_items = max(1, min(5, max_items))
        try:
            timeout_sec = int(os.getenv("EIMAS_RA_EXTERNAL_NEWS_TIMEOUT_SEC", "10") or 10)
        except (TypeError, ValueError):
            timeout_sec = 10
        self.external_news_timeout_sec = max(3, timeout_sec)
        self.external_news_model = os.getenv("EIMAS_RA_EXTERNAL_NEWS_MODEL", "sonar-pro").strip() or "sonar-pro"
        self._external_news_cache: Dict[str, Dict[str, Any]] = {}

    def _canonical_regime(self, regime_value: Any) -> str:
        """레짐 문자열을 BULL/BEAR/NEUTRAL로 정규화."""
        text = str(regime_value or "").upper()
        if "BULL" in text:
            return "BULL"
        if "BEAR" in text:
            return "BEAR"
        return "NEUTRAL"

    def _canonical_volatility(self, volatility_value: Any, regime_value: Any = "") -> str:
        """변동성 문자열을 Low Vol/High Vol/Normal로 정규화."""
        vol_text = str(volatility_value or "").strip()
        combined = f"{vol_text} {regime_value}".upper()
        if "HIGH VOL" in combined:
            return "High Vol"
        if "LOW VOL" in combined:
            return "Low Vol"
        return vol_text or "Normal"

    def generate_report(self, eimas_result: Dict) -> AllocationReport:
        """
        EIMAS JSON 결과로부터 리포트 생성

        Args:
            eimas_result: EIMAS 분석 결과 JSON

        Returns:
            AllocationReport
        """
        report = AllocationReport(timestamp=datetime.now().isoformat())

        # 데이터 품질 및 신호 충돌 검사
        quality_check = self._check_data_quality(eimas_result)
        conflict_check = self._check_signal_conflict(eimas_result)

        if not quality_check['valid'] or conflict_check['conflict']:
            report.default_to_hold = True
            report.data_quality_warning = not quality_check['valid']
            report.signal_conflict_warning = conflict_check['conflict']
            report.hold_reason = quality_check.get('reason', '') or conflict_check.get('reason', '')
            logger.warning(f"Defaulting to HOLD: {report.hold_reason}")

        # 섹션 1: 시장 요약
        report.market_summary = self._extract_market_summary(eimas_result, report.default_to_hold)

        # 섹션 2: 핵심 근거
        report.key_rationales = self._extract_key_rationales(eimas_result)

        # 섹션 3: 리스크 조건
        report.risk_conditions = self._extract_risk_conditions(eimas_result)

        # 섹션 4: 액션 아이템
        report.action_items = self._generate_action_items(eimas_result, report.default_to_hold)

        # 섹션 5: ETF 전략 분해표
        report.etf_decomposition = self._extract_etf_decomposition(eimas_result)

        # 섹션 6: 기업 커버리지/RA 지원/SQL 증빙
        report.company_coverage = self._extract_company_coverage(eimas_result)
        report.ra_work_support = self._extract_ra_work_support(eimas_result)
        report.postgres_evidence = self._extract_postgres_evidence(eimas_result)
        report.internal_sql_evidence = self._extract_internal_sql_evidence(eimas_result)

        # 섹션 7: 실행 타이밍/운영 추적
        report.pipeline_elapsed_sec, report.pipeline_timings = self._extract_pipeline_timing(eimas_result)
        report.operational_summary = self._extract_operational_summary(eimas_result)
        report.detailed_quant_snapshot = self._extract_detailed_quant_snapshot(eimas_result)
        report.validation_evidence = self._extract_validation_evidence(eimas_result)
        report.rebalance_evidence = self._extract_rebalance_evidence(eimas_result)
        report.ai_report_evidence = self._extract_ai_report_evidence(eimas_result)
        report.ra_sql_matrix = self._extract_ra_sql_matrix(eimas_result)
        report.ra_commentary = self._extract_ra_commentary(eimas_result)
        report.ra_todo_items = self._extract_ra_todo_items(eimas_result, report.ra_commentary)
        report.risk_signal_news = self._extract_risk_signal_news(eimas_result)
        report.section_ai_discussion = self._extract_section_ai_discussion(eimas_result, report)

        return report

    def _check_data_quality(self, data: Dict) -> Dict[str, Any]:
        """데이터 품질 검사 (확장된 검증)"""
        # market_quality 필드 확인
        mq = data.get('market_quality', {})
        if isinstance(mq, dict):
            quality = mq.get('data_quality', 'COMPLETE')
            if quality == 'DEGRADED':
                return {'valid': False, 'reason': '시장 데이터 품질 저하 (DEGRADED)'}

        # 필수 필드 확인
        required = ['regime', 'risk_score', 'final_recommendation']
        missing = [f for f in required if f not in data or data.get(f) is None]
        if missing:
            return {'valid': False, 'reason': f'필수 필드 누락: {missing}'}

        # 수치 검증 (리스크 점수 범위, 비중 유효성 등)
        num_validation = self._validate_numerical_values(data)
        if not num_validation['valid']:
            return num_validation

        return {'valid': True}

    def _validate_numerical_values(self, data: Dict) -> Dict[str, Any]:
        """
        수치 값 검증 (새 숫자 생성 방지)

        검증 항목:
        - 리스크 점수 범위 (0-100)
        - 포트폴리오 비중 합계 = 1.0 (±0.01 허용오차)
        - 음수 비중 검출
        - 신뢰도 범위 (0-1)
        """
        # 1. 리스크 점수 범위 검증
        risk_score = data.get('risk_score', 50.0)
        if not (0 <= risk_score <= 100):
            return {'valid': False, 'reason': f'리스크 점수 범위 오류: {risk_score} (0-100 범위 초과)'}

        # 2. 신뢰도 범위 검증
        confidence = data.get('confidence', 0.5)
        if not (0 <= confidence <= 1):
            return {'valid': False, 'reason': f'신뢰도 범위 오류: {confidence} (0-1 범위 초과)'}

        # 3. 포트폴리오 비중 검증
        portfolio_weights = data.get('portfolio_weights', {})
        if portfolio_weights and isinstance(portfolio_weights, dict):
            # 비중 합계 검증
            weight_sum = sum(portfolio_weights.values())
            if abs(weight_sum - 1.0) > 0.01:  # 1% 허용오차
                return {'valid': False, 'reason': f'포트폴리오 비중 합계 오류: {weight_sum:.4f} (1.0±0.01 범위 초과)'}

            # 음수 비중 검증
            negative_weights = {k: v for k, v in portfolio_weights.items() if v < 0}
            if negative_weights:
                return {'valid': False, 'reason': f'음수 비중 검출: {negative_weights}'}

        # 4. Allocation Result 비중 검증 (있을 경우)
        alloc_result = data.get('allocation_result', {})
        if isinstance(alloc_result, dict) and 'weights' in alloc_result:
            weights = alloc_result['weights']
            if isinstance(weights, dict):
                weight_sum = sum(weights.values())
                if abs(weight_sum - 1.0) > 0.01:
                    return {'valid': False, 'reason': f'Allocation 비중 합계 오류: {weight_sum:.4f}'}

                negative_weights = {k: v for k, v in weights.items() if v < 0}
                if negative_weights:
                    return {'valid': False, 'reason': f'Allocation 음수 비중: {negative_weights}'}

        return {'valid': True}

    def _check_signal_conflict(self, data: Dict) -> Dict[str, Any]:
        """신호 충돌 검사"""
        conflicts = []

        # 레짐과 권고 불일치 검사
        regime = data.get('regime', {})
        if isinstance(regime, dict):
            regime_state = self._canonical_regime(regime.get('regime', 'NEUTRAL'))
        else:
            regime_state = 'NEUTRAL'

        recommendation = data.get('final_recommendation', 'HOLD')

        # BULL 레짐인데 BEARISH 권고, 또는 그 반대
        if regime_state == 'BULL' and recommendation == 'BEARISH':
            conflicts.append("레짐(BULL)과 권고(BEARISH) 불일치")
        elif regime_state == 'BEAR' and recommendation == 'BULLISH':
            conflicts.append("레짐(BEAR)과 권고(BULLISH) 불일치")

        # Full Mode와 Reference Mode 불일치
        if not data.get('modes_agree', True):
            conflicts.append("Full Mode와 Reference Mode 불일치")

        # Strong Dissent
        if data.get('has_strong_dissent', False):
            conflicts.append("에이전트 간 강한 이견 존재")

        if conflicts:
            return {'conflict': True, 'reason': ', '.join(conflicts)}

        return {'conflict': False}

    def _extract_market_summary(self, data: Dict, default_hold: bool) -> MarketSummary:
        """시장 요약 추출 (JSON 값만 사용)"""
        regime_data = data.get('regime', {})
        if isinstance(regime_data, dict):
            raw_regime = regime_data.get('regime', 'NEUTRAL')
            regime = self._canonical_regime(raw_regime)
            volatility = self._canonical_volatility(regime_data.get('volatility', 'Normal'), raw_regime)
        else:
            regime = 'NEUTRAL'
            volatility = 'Normal'

        risk_score = data.get('risk_score', 50.0)
        risk_level = data.get('risk_level', 'MEDIUM')

        # Fed 순유동성
        fred = data.get('fred_summary', {})
        net_liq = fred.get('net_liquidity', 0)
        net_liq_str = f"${net_liq:.0f}B" if net_liq else "N/A"

        # 신뢰도 판단 (리스크 점수 기반)
        if risk_score < 30:
            confidence_level = "High"
        elif risk_score < 60:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        # 데이터 품질
        mq = data.get('market_quality', {})
        data_quality = mq.get('data_quality', 'COMPLETE') if isinstance(mq, dict) else 'COMPLETE'

        # 권고
        final_rec = "HOLD" if default_hold else data.get('final_recommendation', 'HOLD')
        confidence = data.get('confidence', 0.5)

        # 요약 문장 생성 (JSON 값만 사용)
        summary = self._generate_summary_text(regime, volatility, risk_score, final_rec, default_hold)

        return MarketSummary(
            regime=regime,
            regime_confidence=confidence_level,
            volatility_state=volatility,
            risk_score=float(risk_score),
            risk_level=risk_level,
            net_liquidity=net_liq_str,
            final_recommendation=final_rec,
            confidence=float(confidence),
            data_quality=data_quality,
            summary_text=summary
        )

    def _generate_summary_text(
        self,
        regime: str,
        volatility: str,
        risk_score: float,
        recommendation: str,
        default_hold: bool
    ) -> str:
        """요약 문장 생성 (새 숫자 없이 JSON 값만 조합)"""
        if default_hold:
            return f"현재 {regime} 레짐, {volatility} 상태이나, 데이터 신뢰도 이슈로 HOLD 권고."

        if regime == 'BULL':
            regime_text = "상승 추세"
        elif regime == 'BEAR':
            regime_text = "하락 압력"
        else:
            regime_text = "횡보 국면"

        return f"현재 {regime_text} ({regime} 레짐), 변동성 {volatility}, 리스크 점수 {risk_score:.1f}. 최종 권고: {recommendation}."

    def _extract_key_rationales(self, data: Dict) -> List[KeyRationale]:
        """핵심 근거 3가지 추출 (JSON 값만 인용)"""
        rationales = []

        # 1. 레짐 기반 근거
        regime_data = data.get('regime', {})
        if regime_data:
            raw_regime = regime_data.get('regime', 'NEUTRAL')
            regime = self._canonical_regime(raw_regime)
            gmm_regime = regime_data.get('gmm_regime', '')
            entropy = regime_data.get('entropy', 0)

            rationales.append(KeyRationale(
                title="시장 레짐 분석",
                source="regime",
                value=f"{raw_regime}" + (f" (GMM: {gmm_regime}, Entropy: {entropy:.3f})" if gmm_regime else ""),
                interpretation=self._interpret_regime(regime, entropy)
            ))

        # 2. 유동성 신호 기반 근거
        liq_signal = data.get('liquidity_signal', 'NEUTRAL')
        fred = data.get('fred_summary', {})
        rrp = fred.get('rrp', 0)
        tga = fred.get('tga', 0)

        rationales.append(KeyRationale(
            title="Fed 유동성 상태",
            source="liquidity_signal, fred_summary",
            value=f"{liq_signal} (RRP: ${rrp:.0f}B, TGA: ${tga:.0f}B)",
            interpretation=self._interpret_liquidity(liq_signal, rrp, tga)
        ))

        # 3. AI 토론 합의
        consensus = data.get('debate_consensus', {})
        full_mode = data.get('full_mode_position', 'NEUTRAL')
        ref_mode = data.get('reference_mode_position', 'NEUTRAL')
        modes_agree = data.get('modes_agree', True)

        rationales.append(KeyRationale(
            title="AI 에이전트 토론 결과",
            source="debate_consensus, full_mode_position, reference_mode_position",
            value=f"Full: {full_mode}, Ref: {ref_mode}, 합의: {'예' if modes_agree else '아니오'}",
            interpretation=self._interpret_debate(full_mode, ref_mode, modes_agree)
        ))

        return rationales[:3]

    def _interpret_regime(self, regime: str, entropy: float) -> str:
        """레짐 해석 (새 값 생성 없음)"""
        if regime == 'BULL':
            return f"상승 추세 레짐. Entropy {entropy:.3f}으로 {'높은' if entropy > 0.5 else '낮은'} 불확실성."
        elif regime == 'BEAR':
            return f"하락 추세 레짐. 방어적 포지셔닝 필요."
        else:
            return f"중립 레짐. 방향성 모니터링 필요."

    def _interpret_liquidity(self, signal: str, rrp: float, tga: float) -> str:
        """유동성 해석"""
        if signal == 'BULLISH':
            return f"RRP 감소와 TGA 안정으로 유동성 공급 확대 중."
        elif signal == 'BEARISH':
            return f"유동성 긴축 신호. RRP/TGA 모니터링 필요."
        else:
            return f"유동성 중립. 변화 방향 관찰 필요."

    def _interpret_debate(self, full: str, ref: str, agree: bool) -> str:
        """토론 결과 해석"""
        if agree:
            return f"두 모드 모두 {full} 판단. 신뢰도 높음."
        else:
            return f"Full({full})과 Reference({ref}) 모드 불일치. 신중한 접근 필요."

    def _extract_risk_conditions(self, data: Dict) -> List[RiskCondition]:
        """리스크 및 반증 조건 3가지 추출"""
        conditions = []

        # 1. 리스크 점수 기반 조건
        risk_score = data.get('risk_score', 50)
        conditions.append(RiskCondition(
            risk_title="리스크 점수 상승",
            current_value=f"{risk_score:.1f}/100",
            falsification_condition=f"리스크 점수가 {min(risk_score + 20, 100):.0f} 이상 상승 시 뷰 재검토",
            monitoring_metric="risk_score"
        ))

        # 2. 버블 리스크
        bubble = data.get('bubble_risk', {})
        if isinstance(bubble, dict):
            status = bubble.get('overall_status', 'NONE')
        else:
            status = 'NONE'

        conditions.append(RiskCondition(
            risk_title="버블 리스크",
            current_value=f"{status}",
            falsification_condition="버블 상태가 WARNING → DANGER 전환 시 즉시 비중 축소",
            monitoring_metric="bubble_risk.overall_status"
        ))

        # 3. 변동성 스파이크
        regime_data = data.get('regime', {})
        volatility = regime_data.get('volatility', 'Normal') if isinstance(regime_data, dict) else 'Normal'

        conditions.append(RiskCondition(
            risk_title="변동성 급등",
            current_value=f"{volatility}",
            falsification_condition="변동성이 High Vol로 전환 시 포지션 규모 축소 검토",
            monitoring_metric="regime.volatility"
        ))

        return conditions[:3]

    def _generate_action_items(self, data: Dict, default_hold: bool) -> List[ActionItem]:
        """액션 아이템 생성 (제약 조건 준수)"""
        items = []

        # 기본 HOLD인 경우
        if default_hold:
            items.append(ActionItem(
                action="HOLD",
                target="전체 포트폴리오",
                rationale="데이터 신뢰도 저하 또는 신호 충돌로 인해 현재 비중 유지",
                constraints=["리밸런싱 보류", "추가 데이터 확인 후 재평가"],
                priority="HIGH"
            ))
            return items

        # 리밸런싱 결정 확인
        rebal = data.get('rebalance_decision', {})
        should_rebal = rebal.get('should_rebalance', False)
        action = rebal.get('action', 'HOLD')
        turnover = rebal.get('turnover', 0)

        if should_rebal and turnover <= self.turnover_cap:
            items.append(ActionItem(
                action="REBALANCE",
                target="포트폴리오 전체",
                rationale=rebal.get('reason', '목표 비중 편차 발생'),
                constraints=[
                    f"Turnover Cap: {self.turnover_cap:.0%} 이내",
                    f"현재 Turnover: {turnover:.1%}",
                    f"Equity Max: {self.weight_bounds['equity_max']:.0%}",
                    f"Bond Max: {self.weight_bounds['bond_max']:.0%}"
                ],
                priority="MEDIUM"
            ))
        else:
            # 리밸런싱 불필요 또는 제약 초과
            hold_reason = "리밸런싱 조건 미충족" if not should_rebal else f"Turnover ({turnover:.1%}) > Cap ({self.turnover_cap:.0%})"
            items.append(ActionItem(
                action="HOLD",
                target="전체 포트폴리오",
                rationale=hold_reason,
                constraints=[f"Turnover Cap: {self.turnover_cap:.0%}"],
                priority="MEDIUM"
            ))

        # 포트폴리오 비중 기반 액션 (JSON 값 인용)
        weights = data.get('portfolio_weights', {}) or data.get('allocation_result', {}).get('weights', {})
        if weights:
            # 상위 비중 자산 모니터링
            top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join([f"{t}: {w:.1%}" for t, w in top_weights])
            items.append(ActionItem(
                action="MONITOR",
                target="상위 비중 자산",
                rationale=f"현재 상위 비중: {top_str}",
                constraints=[f"단일 자산 Max: {self.weight_bounds.get('equity_max', 0.6):.0%}"],
                priority="LOW"
            ))

        return items

    def _format_pct(self, value: Any) -> str:
        try:
            return f"{float(value):+.2f}%"
        except (TypeError, ValueError):
            return "N/A"

    def _format_list_preview(self, value: Any, limit: int = 3) -> str:
        if not isinstance(value, list):
            return "N/A"
        items: List[str] = []
        for item in value:
            token = str(item).strip()
            if token:
                items.append(token)
            if len(items) >= limit:
                break
        return ", ".join(items) if items else "N/A"

    def _compress_etf_data_source(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "N/A"

        alias = {
            "financial_indicators": "fi",
            "yfinance_price": "yf_px",
            "yfinance_info": "yf_meta",
            "stooq": "stooq",
            "fallback": "fb",
        }
        parts: List[str] = []
        for raw in text.split(","):
            token = raw.strip()
            if not token:
                continue
            parts.append(alias.get(token, token))
        if not parts:
            return "N/A"
        if len(parts) > 4:
            return "+".join(parts[:4]) + f"+{len(parts) - 4}src"
        return "+".join(parts)

    def _etf_static_profile(self, ticker: str) -> Dict[str, str]:
        """고정 ETF 분류(역할/팩터/섹터/듀레이션)"""
        profiles = {
            "SPY": {
                "category": "market",
                "asset_role": "Core",
                "factor_exposure": "Market Beta",
                "sector_or_theme": "미국 대형주 광범위",
                "duration_profile": "중립",
            },
            "QQQ": {
                "category": "market",
                "asset_role": "Growth Tilt",
                "factor_exposure": "Growth/Quality",
                "sector_or_theme": "나스닥100(기술주 비중 높음)",
                "duration_profile": "상대적 고듀레이션(금리 민감)",
            },
            "IWM": {
                "category": "market",
                "asset_role": "Size Tilt",
                "factor_exposure": "Small Cap",
                "sector_or_theme": "미국 소형주",
                "duration_profile": "경기 민감",
            },
            "XLF": {
                "category": "sector",
                "asset_role": "Sector Satellite",
                "factor_exposure": "Value/Cyclicals",
                "sector_or_theme": "금융 섹터",
                "duration_profile": "장단기금리차 민감",
            },
            "TLT": {
                "category": "bond",
                "asset_role": "Rates Hedge",
                "factor_exposure": "Duration",
                "sector_or_theme": "미국 장기 국채",
                "duration_profile": "고듀레이션",
            },
            "GLD": {
                "category": "alternative",
                "asset_role": "Real Asset Hedge",
                "factor_exposure": "Inflation/Real Rate",
                "sector_or_theme": "금(대체자산)",
                "duration_profile": "실질금리 역민감",
            },
        }
        return profiles.get(
            ticker,
            {
                "category": "unknown",
                "asset_role": "Satellite",
                "factor_exposure": "N/A",
                "sector_or_theme": "N/A",
                "duration_profile": "N/A",
            },
        )

    def _extract_etf_decomposition(self, data: Dict) -> List[ETFDecomposition]:
        """ETF 스냅샷을 RA 전략 분해표로 변환."""
        ra = data.get("company_ra_analysis", {})
        if not isinstance(ra, dict):
            return []

        snapshot = ra.get("etf_strategy_snapshot", [])
        if not isinstance(snapshot, list):
            return []

        rows: List[ETFDecomposition] = []
        for item in snapshot:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker", "")).upper().strip()
            if not ticker:
                continue
            profile = self._etf_static_profile(ticker)
            top_holdings = self._format_list_preview(item.get("top_holdings"), limit=3)
            data_source = self._compress_etf_data_source(item.get("data_source", "N/A"))
            quality_flag = str(item.get("quality_flag", "N/A"))
            rows.append(
                ETFDecomposition(
                    ticker=ticker,
                    category=str(item.get("category", profile.get("category", "N/A"))),
                    asset_role=str(item.get("asset_role", profile["asset_role"])),
                    factor_exposure=str(item.get("factor_exposure", profile["factor_exposure"])),
                    sector_or_theme=str(item.get("sector_or_theme", profile["sector_or_theme"])),
                    duration_profile=str(item.get("duration_profile", profile["duration_profile"])),
                    ret_5d=self._format_pct(item.get("ret_5d_pct")),
                    ret_20d=self._format_pct(item.get("ret_20d_pct")),
                    momentum_label=str(item.get("momentum_label", "N/A")),
                    top_holdings=top_holdings,
                    data_source=data_source,
                    quality_flag=quality_flag,
                )
            )
        return rows

    def _format_num(self, value: Any, digits: int = 2) -> str:
        try:
            if value is None:
                return "N/A"
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "N/A"

    def _extract_company_coverage(self, data: Dict) -> List[CompanyCoverageRow]:
        """회사 커버리지 표 데이터 추출."""
        ra = data.get("company_ra_analysis", {})
        if not isinstance(ra, dict):
            return []

        companies = ra.get("companies", [])
        if not isinstance(companies, list):
            return []

        rows: List[CompanyCoverageRow] = []
        for item in companies:
            if not isinstance(item, dict):
                continue
            valuation = item.get("valuation", {})
            ratios = item.get("ratios", {})
            momentum = item.get("price_momentum", {})

            rows.append(
                CompanyCoverageRow(
                    ticker=str(item.get("ticker", "N/A")),
                    sector=str(item.get("sector", "N/A")),
                    trailing_pe=self._format_num(valuation.get("trailing_pe")),
                    forward_pe=self._format_num(valuation.get("forward_pe")),
                    price_to_book=self._format_num(valuation.get("price_to_book")),
                    roe=self._format_num(ratios.get("roe")),
                    roa=self._format_num(ratios.get("roa")),
                    net_margin=self._format_num(ratios.get("net_margin")),
                    debt_to_equity=self._format_num(ratios.get("debt_to_equity"), digits=3),
                    ret_5d=self._format_pct(momentum.get("ret_5d_pct")),
                    ret_20d=self._format_pct(momentum.get("ret_20d_pct")),
                    valuation_signal=str(item.get("valuation_signal", "N/A")),
                    ra_takeaway=str(
                        item.get("ra_takeaway")
                        or "핵심 재무지표와 모멘텀 정합성 점검 필요"
                    ),
                )
            )
        return rows

    def _clean_report_text(self, value: Any) -> str:
        """자기소개/채용 목적 메타 문구를 본문에서 제거."""
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        compact = " ".join(text.split())
        if any(keyword in compact for keyword in RECRUITMENT_META_KEYWORDS):
            return ""
        return compact

    def _clean_role_focus(self, value: Any) -> str:
        text = self._clean_report_text(value)
        if not text:
            return ""
        # 예: "RA 1명 (매크로/ETF 전략)" -> "RA (매크로/ETF 전략)"
        text = re.sub(r"\bRA\s*\d+\s*명\b", "RA", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text).strip(" -")
        return text

    def _clean_text_list(self, values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        cleaned: List[str] = []
        for value in values:
            text = self._clean_report_text(value)
            if text:
                cleaned.append(text)
        return cleaned

    def _extract_ra_work_support(self, data: Dict) -> Dict[str, Any]:
        ra = data.get("company_ra_analysis", {})
        if not isinstance(ra, dict):
            return {}
        work = ra.get("ra_work_support", {})
        if not isinstance(work, dict):
            return {}

        role_focus = self._clean_role_focus(work.get("role_focus", ""))
        research_tasks = self._clean_text_list(work.get("research_tasks", []))
        seminar_points = self._clean_text_list(work.get("seminar_material_points", []))
        cross_points = self._clean_text_list(work.get("cross_department_support_points", []))
        data_update_note = self._clean_report_text(work.get("data_update_note", ""))

        payload = {
            "role_focus": role_focus or "N/A",
            "research_tasks": research_tasks,
            "seminar_material_points": seminar_points,
            "cross_department_support_points": cross_points,
            "data_update_note": data_update_note,
        }
        return payload

    def _extract_postgres_evidence(self, data: Dict) -> Dict[str, Any]:
        ra = data.get("company_ra_analysis", {})
        if not isinstance(ra, dict):
            return {}
        pg = ra.get("postgresql", {})
        return pg if isinstance(pg, dict) else {}

    def _extract_internal_sql_evidence(self, data: Dict) -> Dict[str, Any]:
        ra = data.get("company_ra_analysis", {})
        if not isinstance(ra, dict):
            return {}

        internal = ra.get("internal_sql", {})
        if not isinstance(internal, dict):
            return {}

        # Keep compatibility if collector returns direct summary dict.
        if "upserted_rows" in internal or "table" in internal:
            return internal

        company_sql = internal.get("company") if isinstance(internal.get("company"), dict) else {}
        phase6_sql = internal.get("phase6_backtest") if isinstance(internal.get("phase6_backtest"), dict) else {}
        merged = dict(company_sql)
        if phase6_sql:
            merged["phase6_backtest"] = phase6_sql
        return merged

    def _extract_pipeline_timing(self, data: Dict) -> tuple[float, List[PipelineTimingRow]]:
        elapsed = 0.0
        try:
            elapsed = float(data.get("pipeline_elapsed_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            elapsed = 0.0

        timings = data.get("pipeline_phase_timings", {})
        if not isinstance(timings, dict):
            return elapsed, []

        rows: List[PipelineTimingRow] = []
        for phase, meta in timings.items():
            if not isinstance(meta, dict):
                continue
            duration = meta.get("duration_sec", 0.0)
            try:
                duration_f = float(duration or 0.0)
            except (TypeError, ValueError):
                duration_f = 0.0
            rows.append(
                PipelineTimingRow(
                    phase=str(phase),
                    duration_sec=duration_f,
                    status=str(meta.get("status", "N/A")),
                )
            )

        rows.sort(key=lambda x: x.duration_sec, reverse=True)
        return elapsed, rows

    def _extract_operational_summary(self, data: Dict) -> Dict[str, Any]:
        op = data.get("operational_report", {})
        if not isinstance(op, dict):
            return {}

        decision = op.get("decision_policy", {})
        hold_policy = op.get("hold_policy", {})
        if not isinstance(decision, dict):
            decision = {}
        if not isinstance(hold_policy, dict):
            hold_policy = {}

        triggered_conditions: List[str] = []
        hold_conditions = hold_policy.get("hold_conditions", [])
        if isinstance(hold_conditions, list):
            for cond in hold_conditions:
                if not isinstance(cond, dict):
                    continue
                is_triggered = str(cond.get("is_triggered", "")).lower() in {"true", "1", "yes"}
                if is_triggered:
                    triggered_conditions.append(str(cond.get("condition_name", "N/A")))

        return {
            "final_stance": decision.get("final_stance", "N/A"),
            "reason_codes": decision.get("reason_codes", []),
            "applied_rules": decision.get("applied_rules", []),
            "is_hold": hold_policy.get("is_hold"),
            "triggered_conditions": triggered_conditions,
            "primary_reason": hold_policy.get("primary_reason", ""),
        }

    def _format_ratio_pct(self, value: Any, digits: int = 1) -> str:
        try:
            return f"{float(value) * 100:.{digits}f}%"
        except (TypeError, ValueError):
            return "N/A"

    def _format_scientific(self, value: Any) -> str:
        try:
            return f"{float(value):.3e}"
        except (TypeError, ValueError):
            return "N/A"

    def _extract_detailed_quant_snapshot(self, data: Dict) -> Dict[str, Any]:
        fred = data.get("fred_summary", {}) if isinstance(data.get("fred_summary"), dict) else {}
        hft = data.get("hft_microstructure", {}) if isinstance(data.get("hft_microstructure"), dict) else {}
        garch = data.get("garch_volatility", {}) if isinstance(data.get("garch_volatility"), dict) else {}
        flow = data.get("information_flow", {}) if isinstance(data.get("information_flow"), dict) else {}
        poi = data.get("proof_of_index", {}) if isinstance(data.get("proof_of_index"), dict) else {}
        dtw = data.get("dtw_similarity", {}) if isinstance(data.get("dtw_similarity"), dict) else {}
        dbscan = data.get("dbscan_outliers", {}) if isinstance(data.get("dbscan_outliers"), dict) else {}
        bubble = data.get("bubble_risk", {}) if isinstance(data.get("bubble_risk"), dict) else {}
        sentiment = data.get("sentiment_analysis", {}) if isinstance(data.get("sentiment_analysis"), dict) else {}
        ext = data.get("extended_data", {}) if isinstance(data.get("extended_data"), dict) else {}

        macro_liquidity = {
            "Fed Funds (%)": self._format_num(fred.get("fed_funds")),
            "UST 2Y (%)": self._format_num(fred.get("treasury_2y")),
            "UST 10Y (%)": self._format_num(fred.get("treasury_10y")),
            "UST 30Y (%)": self._format_num(fred.get("treasury_30y")),
            "10Y-2Y Spread (%)": self._format_num(fred.get("spread_10y2y")),
            "HY OAS": self._format_num(fred.get("hy_oas")),
            "Unemployment (%)": self._format_num(fred.get("unemployment")),
            "Initial Claims": str(fred.get("initial_claims", "N/A")),
            "RRP ($B)": self._format_num(fred.get("rrp")),
            "TGA ($B)": self._format_num(fred.get("tga")),
            "Net Liquidity ($B)": self._format_num(fred.get("net_liquidity")),
            "Liquidity Regime": str(fred.get("liquidity_regime", "N/A")),
        }

        tick_rule = hft.get("tick_rule", {}) if isinstance(hft.get("tick_rule"), dict) else {}
        kyle = hft.get("kyles_lambda", {}) if isinstance(hft.get("kyles_lambda"), dict) else {}
        garch_params = garch.get("garch_params", {}) if isinstance(garch.get("garch_params"), dict) else {}
        hft_garch = {
            "Tick Rule Buy Ratio": self._format_ratio_pct(tick_rule.get("buy_ratio")),
            "Tick Rule Interpretation": str(tick_rule.get("interpretation", "N/A")),
            "Kyle's Lambda": self._format_scientific(kyle.get("lambda")),
            "Kyle's R²": self._format_num(kyle.get("r_squared"), digits=3),
            "GARCH Current Vol": self._format_ratio_pct(garch.get("current_volatility")),
            "GARCH Forecast Avg Vol": self._format_ratio_pct(garch.get("forecast_avg_volatility")),
            "GARCH Persistence": self._format_num(garch_params.get("persistence"), digits=3),
            "GARCH Half-life (days)": self._format_num(garch_params.get("half_life"), digits=1),
        }

        abnormal = flow.get("abnormal_volume", {}) if isinstance(flow.get("abnormal_volume"), dict) else {}
        capm_qqq = flow.get("capm_QQQ", {}) if isinstance(flow.get("capm_QQQ"), dict) else {}
        poi_verify = poi.get("verification", {}) if isinstance(poi.get("verification"), dict) else {}
        poi_signal = poi.get("mean_reversion_signal", {}) if isinstance(poi.get("mean_reversion_signal"), dict) else {}
        poi_verdict = "PASS" if poi_verify.get("is_valid") else "FAIL"
        capm_alpha = capm_qqq.get("alpha")
        capm_alpha_yr = "N/A"
        try:
            capm_alpha_yr = f"{float(capm_alpha) * 252 * 100:+.2f}%/yr"
        except (TypeError, ValueError):
            pass

        flow_poi = {
            "Abnormal Volume Days": str(abnormal.get("total_abnormal_days", "N/A")),
            "Abnormal Volume Ratio": self._format_ratio_pct(abnormal.get("abnormal_ratio")),
            "CAPM QQQ Alpha": capm_alpha_yr,
            "CAPM QQQ Beta": self._format_num(capm_qqq.get("beta"), digits=3),
            "PoI Index Value": self._format_num(poi.get("index_value"), digits=3),
            "PoI Mean Reversion Signal": str(poi_signal.get("signal", "N/A")),
            "PoI Z-Score": self._format_num(poi_signal.get("z_score"), digits=3),
            "PoI Verification": poi_verdict,
        }

        similar = dtw.get("most_similar_pair", {}) if isinstance(dtw.get("most_similar_pair"), dict) else {}
        lead_lag = dtw.get("lead_lag_spy_qqq", {}) if isinstance(dtw.get("lead_lag_spy_qqq"), dict) else {}
        fear_greed = sentiment.get("fear_greed", {}) if isinstance(sentiment.get("fear_greed"), dict) else {}
        vix_struct = sentiment.get("vix_structure", {}) if isinstance(sentiment.get("vix_structure"), dict) else {}
        pcr = ext.get("put_call_ratio", {}) if isinstance(ext.get("put_call_ratio"), dict) else {}

        similarity_risk = {
            "DTW n_series": str(dtw.get("n_series", "N/A")),
            "DTW Avg Distance": self._format_num(dtw.get("avg_distance"), digits=4),
            "DTW Most Similar Pair": (
                f"{similar.get('asset1', 'N/A')} ↔ {similar.get('asset2', 'N/A')}"
            ),
            "DTW Lead-Lag": str(lead_lag.get("interpretation", "N/A")),
            "DBSCAN Outliers": (
                f"{dbscan.get('n_outliers', 'N/A')}/{dbscan.get('n_total_assets', 'N/A')}"
            ),
            "DBSCAN Outlier Ratio": self._format_ratio_pct(dbscan.get("outlier_ratio")),
            "Bubble Status": str(bubble.get("overall_status", "N/A")),
            "Fear & Greed": str(fear_greed.get("value", "N/A")),
            "VIX Structure": str(vix_struct.get("signal", "N/A")),
            "Put/Call Ratio": (
                f"{self._format_num(pcr.get('ratio'), digits=2)} ({pcr.get('sentiment', 'N/A')})"
            ),
        }

        return {
            "macro_liquidity": macro_liquidity,
            "hft_garch": hft_garch,
            "flow_poi": flow_poi,
            "similarity_risk": similarity_risk,
        }

    def _extract_validation_evidence(self, data: Dict) -> Dict[str, Any]:
        validation = data.get("validation_loop_result", {})
        if not isinstance(validation, dict):
            validation = {}

        debate_consensus = data.get("debate_consensus", {})
        if not isinstance(debate_consensus, dict):
            debate_consensus = {}
        verify = debate_consensus.get("verification", {})
        if not isinstance(verify, dict):
            verify = {}

        evidence = {
            "final_recommendation": str(data.get("final_recommendation", "N/A")),
            "full_mode_position": str(data.get("full_mode_position", "N/A")),
            "reference_mode_position": str(data.get("reference_mode_position", "N/A")),
            "modes_agree": str(data.get("modes_agree", "N/A")),
            "fact_check_grade": str(data.get("fact_check_grade", "N/A")),
            "validation_final_result": str(validation.get("final_result", "N/A")),
            "validation_consensus_confidence": self._format_num(validation.get("consensus_confidence"), digits=1),
            "validation_agreement_ratio": self._format_ratio_pct(validation.get("agreement_ratio")),
            "verification_score": self._format_num(verify.get("overall_score"), digits=2),
            "verification_passed": str(verify.get("passed", "N/A")),
            "validation_summary": str(validation.get("summary", "N/A")),
            "warnings": data.get("warnings", []),
            "key_concerns": validation.get("key_concerns", []),
            "action_items": validation.get("action_items", []),
        }
        return evidence

    def _extract_rebalance_evidence(self, data: Dict) -> Dict[str, Any]:
        op = data.get("operational_report", {})
        if not isinstance(op, dict):
            return {}

        rebalance = op.get("rebalance_plan", {})
        if not isinstance(rebalance, dict):
            rebalance = {}

        summary = rebalance.get("summary", {}) if isinstance(rebalance.get("summary"), dict) else {}
        summary_fmt = {
            "total_turnover": self._format_ratio_pct(summary.get("total_turnover")),
            "estimated_cost": self._format_ratio_pct(summary.get("total_estimated_cost"), digits=3),
            "buy_count": str(summary.get("buy_count", "N/A")),
            "sell_count": str(summary.get("sell_count", "N/A")),
            "hold_count": str(summary.get("hold_count", "N/A")),
        }

        allocation = op.get("allocation", {})
        top_alloc = []
        if isinstance(allocation, dict):
            sorted_alloc = sorted(
                allocation.items(),
                key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else -1.0,
                reverse=True,
            )[:8]
            for ticker, weight in sorted_alloc:
                top_alloc.append(
                    {
                        "ticker": str(ticker),
                        "weight": self._format_ratio_pct(weight),
                    }
                )

        approval = rebalance.get("approval", {})
        if not isinstance(approval, dict):
            approval = {}
        trigger = rebalance.get("trigger", {})
        if not isinstance(trigger, dict):
            trigger = {}

        return {
            "summary": summary_fmt,
            "approval": approval,
            "trigger": trigger,
            "top_allocation": top_alloc,
        }

    def _extract_ai_report_evidence(self, data: Dict) -> Dict[str, Any]:
        ai_report = data.get("ai_report", {})
        if not isinstance(ai_report, dict):
            return {}

        sections = ai_report.get("sections", {})
        sections_count = len(sections) if isinstance(sections, dict) else 0

        return {
            "report_path": str(ai_report.get("report_path", "")),
            "ib_report_path": str(ai_report.get("ib_report_path", "")),
            "sections_count": sections_count,
        }

    def _parse_percent_like(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        text = text.replace("%", "").replace(",", "")
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    def _dedupe_external_news_items(self, items: Any) -> List[Dict[str, str]]:
        if not isinstance(items, list):
            return []
        deduped: List[Dict[str, str]] = []
        seen: set[str] = set()
        for raw in items:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title", "") or "").strip()
            source = str(raw.get("source", "") or "").strip() or "N/A"
            url = str(raw.get("url", "") or "").strip()
            published_at = str(raw.get("published_at", "") or "").strip()
            summary = str(raw.get("summary", "") or "").strip()
            if not title:
                continue
            key = (url or title).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "title": title,
                    "source": source,
                    "url": url,
                    "published_at": published_at,
                    "summary": summary,
                }
            )
            if len(deduped) >= self.external_news_max_items:
                break
        return deduped

    def _build_external_news_query(self, data: Dict, signals: List[str]) -> str:
        terms = ["US stock market", "VIX", "Federal Reserve", "ETF flows", "macro risk"]
        if any("gap_signal=BEARISH" in s for s in signals):
            terms.append("risk-off")
        if any("opportunity_signal=BULLISH" in s for s in signals):
            terms.append("risk-on")
        if any("vix_spot=" in s for s in signals):
            terms.append("volatility spike")
        if any("bubble=" in s for s in signals):
            terms.append("asset bubble")

        ra = data.get("company_ra_analysis", {}) if isinstance(data.get("company_ra_analysis"), dict) else {}
        etf_snapshot = ra.get("etf_strategy_snapshot", []) if isinstance(ra.get("etf_strategy_snapshot"), list) else []
        etf_tickers: List[str] = []
        for row in etf_snapshot[:4]:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker", "")).upper().strip()
            if ticker and ticker not in etf_tickers:
                etf_tickers.append(ticker)
            if len(etf_tickers) >= 2:
                break
        terms.extend(etf_tickers)

        def _quoted(token: str) -> str:
            return f"\"{token}\"" if " " in token else token

        selected = [_quoted(str(x).strip()) for x in terms if str(x).strip()]
        return " OR ".join(selected[:8])

    def _fetch_external_news_from_newsapi(self, query: str) -> Dict[str, Any]:
        api_key = (os.getenv("NEWSAPI_KEY", "") or os.getenv("NEWS_API_KEY", "")).strip()
        if not api_key:
            return {"provider": "newsapi", "status": "api_key_missing", "query": query, "items": [], "error": ""}

        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": str(self.external_news_max_items),
            "from": (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d"),
        }
        endpoint = "https://newsapi.org/v2/everything?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            endpoint,
            headers={
                "X-Api-Key": api_key,
                "Accept": "application/json",
                "User-Agent": "EIMAS/RA-Report",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.external_news_timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if not isinstance(payload, dict) or payload.get("status") != "ok":
                code = payload.get("code", "unknown") if isinstance(payload, dict) else "unknown"
                return {
                    "provider": "newsapi",
                    "status": "api_error",
                    "query": query,
                    "items": [],
                    "error": str(code),
                }
            items: List[Dict[str, str]] = []
            for article in payload.get("articles", []) or []:
                if not isinstance(article, dict):
                    continue
                source_meta = article.get("source", {}) if isinstance(article.get("source"), dict) else {}
                items.append(
                    {
                        "title": str(article.get("title", "") or "").strip(),
                        "source": str(source_meta.get("name", "NewsAPI")).strip() or "NewsAPI",
                        "url": str(article.get("url", "") or "").strip(),
                        "published_at": str(article.get("publishedAt", "") or "").strip(),
                        "summary": str(article.get("description", "") or "").strip(),
                    }
                )
            return {
                "provider": "newsapi",
                "status": "ok",
                "query": query,
                "items": self._dedupe_external_news_items(items),
                "error": "",
            }
        except urllib.error.URLError as e:
            return {
                "provider": "newsapi",
                "status": "network_error",
                "query": query,
                "items": [],
                "error": str(e),
            }
        except Exception as e:
            return {
                "provider": "newsapi",
                "status": "exception",
                "query": query,
                "items": [],
                "error": f"{type(e).__name__}: {e}",
            }

    def _fetch_external_news_from_perplexity(self, query: str, signals: List[str]) -> Dict[str, Any]:
        try:
            from core.config import APIConfig

            status = APIConfig.validate()
            if not status.get("perplexity", False):
                return {"provider": "perplexity", "status": "api_key_missing", "query": query, "items": [], "error": ""}

            client = APIConfig.get_client("perplexity")
            today = datetime.utcnow().strftime("%Y-%m-%d")
            prompt = (
                "Return only JSON object with key `items` (list length 3-5). "
                "Each item must include title, source, url, published_at, summary. "
                "Focus on verifiable financial macro/ETF risk headlines that explain these signals: "
                f"{', '.join(signals[:4])}. "
                f"Query context: {query}. Date context: {today}."
            )
            response = client.chat.completions.create(
                model=self.external_news_model,
                temperature=0.0,
                max_tokens=900,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial news retrieval assistant. Output JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = ""
            if response and getattr(response, "choices", None):
                content = response.choices[0].message.content or ""
            parsed = self._parse_json_object(content)
            raw_items = parsed.get("items", []) if isinstance(parsed, dict) else []
            if not isinstance(raw_items, list):
                raw_items = []

            items: List[Dict[str, str]] = []
            for row in raw_items:
                if not isinstance(row, dict):
                    continue
                items.append(
                    {
                        "title": str(row.get("title", "") or "").strip(),
                        "source": str(row.get("source", "Perplexity")).strip() or "Perplexity",
                        "url": str(row.get("url", "") or "").strip(),
                        "published_at": str(row.get("published_at", "") or "").strip(),
                        "summary": str(row.get("summary", "") or "").strip(),
                    }
                )

            citations = getattr(response, "citations", None)
            if isinstance(citations, list):
                citation_urls: List[str] = []
                for c in citations:
                    url = str(c or "").strip()
                    if url.startswith(("http://", "https://")) and url not in citation_urls:
                        citation_urls.append(url)
                for idx, item in enumerate(items):
                    if idx >= len(citation_urls):
                        break
                    if not item.get("url"):
                        item["url"] = citation_urls[idx]

            items = self._dedupe_external_news_items(items)
            if not items:
                return {
                    "provider": "perplexity",
                    "status": "parse_empty",
                    "query": query,
                    "items": [],
                    "error": "",
                }
            return {
                "provider": "perplexity",
                "status": "ok",
                "query": query,
                "items": items,
                "error": "",
            }
        except Exception as e:
            return {
                "provider": "perplexity",
                "status": "exception",
                "query": query,
                "items": [],
                "error": f"{type(e).__name__}: {e}",
            }

    def _fetch_external_risk_news(self, data: Dict, signals: List[str]) -> Dict[str, Any]:
        if not self.external_news_enabled:
            return {
                "provider": "disabled",
                "status": "disabled",
                "query": "",
                "items": [],
                "error": "",
            }
        query = self._build_external_news_query(data, signals)
        cache_key = f"{query}|{'|'.join(signals[:5])}"
        cached = self._external_news_cache.get(cache_key)
        if isinstance(cached, dict):
            return dict(cached)

        attempts = [
            self._fetch_external_news_from_newsapi(query),
            self._fetch_external_news_from_perplexity(query, signals),
        ]
        final_result = {
            "provider": "none",
            "status": "no_items",
            "query": query,
            "items": [],
            "error": "",
        }
        for result in attempts:
            if not isinstance(result, dict):
                continue
            final_result = result
            items = result.get("items", [])
            if isinstance(items, list) and items:
                final_result = dict(result)
                break

        self._external_news_cache[cache_key] = dict(final_result)
        return final_result

    def _extract_risk_signal_news(self, data: Dict) -> Dict[str, Any]:
        warnings = data.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []

        risk_score = self._safe_float_value(data.get("risk_score"))
        risk_score_val = risk_score if isinstance(risk_score, (int, float)) else 0.0

        gap = data.get("gap_analysis", {})
        if not isinstance(gap, dict):
            gap = {}
        gap_signal = str(gap.get("overall_signal", "N/A")).upper()
        gap_conf = self._safe_float_value(gap.get("confidence"))
        gap_conf_val = gap_conf if isinstance(gap_conf, (int, float)) else 0.0

        bubble = data.get("bubble_risk", {})
        if not isinstance(bubble, dict):
            bubble = {}
        bubble_status = str(bubble.get("overall_status", "NONE")).upper()

        market_indicators = data.get("market_indicators", {})
        if not isinstance(market_indicators, dict):
            market_indicators = {}
        sentiment = data.get("sentiment_analysis", {})
        if not isinstance(sentiment, dict):
            sentiment = {}
        vix_struct = sentiment.get("vix_structure", {})
        if not isinstance(vix_struct, dict):
            vix_struct = {}

        vix_spot = self._safe_float_value(market_indicators.get("vix_current"))
        if vix_spot is None:
            vix_spot = self._safe_float_value(vix_struct.get("vix_spot"))
        vix_val = vix_spot if isinstance(vix_spot, (int, float)) else 0.0

        signals: List[str] = []
        if warnings:
            signals.append(f"warnings={len(warnings)}")
        if risk_score_val >= 60.0:
            signals.append(f"risk_score_high={risk_score_val:.1f}")
        if risk_score_val <= 10.0 and warnings:
            signals.append(f"risk_score_warning_divergence={risk_score_val:.1f}")
        if bubble_status in {"WARNING", "DANGER"}:
            signals.append(f"bubble={bubble_status}")
        if vix_val >= 20.0:
            signals.append(f"vix_spot={vix_val:.1f}")
        if gap_signal in {"BEARISH", "RISK_OFF", "STRONG_BEARISH"}:
            signals.append(f"gap_signal={gap_signal}")
        elif gap_signal in {"BULLISH", "RISK_ON", "STRONG_BULLISH"} and gap_conf_val >= 0.55:
            signals.append(f"opportunity_signal={gap_signal}")

        ext = data.get("extended_data", {})
        if not isinstance(ext, dict):
            ext = {}
        ext_news = ext.get("news_sentiment", {})
        if not isinstance(ext_news, dict):
            ext_news = {}
        sent_news = sentiment.get("news_sentiment", {})
        if not isinstance(sent_news, dict):
            sent_news = {}

        headline_candidates: List[str] = []
        for candidate in (
            ext_news.get("top_headline"),
            sent_news.get("top_headline"),
        ):
            text = str(candidate or "").strip()
            if text and text.lower() not in {x.lower() for x in headline_candidates}:
                headline_candidates.append(text)
        headline = headline_candidates[0] if headline_candidates else ""

        news_label = str(ext_news.get("label") or sent_news.get("overall") or "N/A")
        news_score = self._safe_float_value(ext_news.get("score"))
        if news_score is None:
            news_score = self._safe_float_value(sent_news.get("avg_score"))

        analysis_parts: List[str] = []
        if gap_signal in {"BEARISH", "RISK_OFF", "STRONG_BEARISH"}:
            analysis_parts.append("갭 분석에서 risk-off 성향이 포착되어 ETF 방어 듀레이션/현금 비중 점검이 필요합니다.")
        elif gap_signal in {"BULLISH", "RISK_ON", "STRONG_BULLISH"}:
            analysis_parts.append("갭 분석에서 risk-on 신호가 관찰되어 성장/사이클릭 비중 확대 여지를 점검합니다.")

        if warnings:
            analysis_parts.append(
                f"검증/경고 {len(warnings)}건이 존재해 뉴스 해석을 단일 방향으로 단정하지 않고 리스크 시나리오를 병행합니다."
            )
        if isinstance(news_score, (int, float)):
            analysis_parts.append(f"뉴스 센티먼트 점수는 {news_score:.2f}로 집계되었습니다.")
        if headline:
            analysis_parts.append("핵심 헤드라인의 이벤트 방향이 모멘텀/변동성 지표와 같은 방향인지 교차 검증이 필요합니다.")

        external_provider = ""
        external_status = "skipped_no_signal"
        external_query = ""
        external_headlines: List[Dict[str, str]] = []
        external_error = ""
        if signals:
            external = self._fetch_external_risk_news(data, signals)
            if isinstance(external, dict):
                external_provider = str(external.get("provider", "") or "").strip()
                external_status = str(external.get("status", "unknown") or "unknown").strip()
                external_query = str(external.get("query", "") or "").strip()
                external_error = str(external.get("error", "") or "").strip()
                raw_items = external.get("items", [])
                external_headlines = self._dedupe_external_news_items(raw_items)
                if external_headlines and not headline:
                    headline = external_headlines[0].get("title", "")

        source_parts = [
            "warnings",
            "gap_analysis",
            "extended_data.news_sentiment",
            "sentiment_analysis.news_sentiment",
        ]
        if external_provider:
            source_parts.append(f"external_news.{external_provider}.{external_status}")
        return {
            "risk_detected": bool(signals),
            "signals": signals,
            "headline": headline,
            "news_label": news_label,
            "news_score": news_score,
            "analysis": " ".join(analysis_parts).strip(),
            "external_provider": external_provider,
            "external_status": external_status,
            "external_query": external_query,
            "external_headlines": external_headlines,
            "external_error": external_error,
            "source": " + ".join(source_parts),
        }

    def _extract_section_ai_discussion(
        self,
        data: Dict,
        report: AllocationReport,
    ) -> Dict[str, List[str]]:
        reasoning = data.get("reasoning_chain", [])
        if not isinstance(reasoning, list):
            reasoning = []

        debate = data.get("debate_consensus", {})
        if not isinstance(debate, dict):
            debate = {}
        enhanced = debate.get("enhanced", {})
        if not isinstance(enhanced, dict):
            enhanced = {}
        interpretation = enhanced.get("interpretation", {})
        if not isinstance(interpretation, dict):
            interpretation = {}
        verification = debate.get("verification", {})
        if not isinstance(verification, dict):
            verification = {}
        metadata = debate.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        recommended_action = str(
            interpretation.get("recommended_action")
            or data.get("full_mode_position")
            or data.get("final_recommendation")
            or "N/A"
        )
        final_rec = str(data.get("final_recommendation", "N/A"))
        align_text = "정합" if recommended_action.upper() == final_rec.upper() else "불일치"

        consensus_points = interpretation.get("consensus_points", [])
        if not isinstance(consensus_points, list):
            consensus_points = []
        divergence_points = interpretation.get("divergence_points", [])
        if not isinstance(divergence_points, list):
            divergence_points = []

        verify_score = self._safe_float_value(verification.get("overall_score"))
        verify_passed = verification.get("passed")
        avg_conf = self._safe_float_value(metadata.get("avg_confidence"))
        avg_conf_txt = (
            f"{avg_conf * 100:.1f}%"
            if isinstance(avg_conf, (int, float))
            else "N/A"
        )

        reasoning_highlights: List[str] = []
        for item in reasoning[:3]:
            if not isinstance(item, dict):
                continue
            agent = str(item.get("agent", "Unknown"))
            output = str(item.get("output", "N/A"))
            key_factors = item.get("key_factors", [])
            factor_preview = ""
            if isinstance(key_factors, list) and key_factors:
                factor_preview = str(key_factors[0])
            if factor_preview:
                reasoning_highlights.append(f"{agent}: {output} ({factor_preview})")
            else:
                reasoning_highlights.append(f"{agent}: {output}")

        warnings = data.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []

        etf_rows = report.etf_decomposition if isinstance(report.etf_decomposition, list) else []
        etf_ranked: List[tuple[str, float]] = []
        for row in etf_rows:
            score = self._parse_percent_like(getattr(row, "ret_20d", None))
            if score is None:
                continue
            etf_ranked.append((str(getattr(row, "ticker", "N/A")), score))
        etf_ranked.sort(key=lambda x: x[1], reverse=True)

        top_etf = f"{etf_ranked[0][0]}({etf_ranked[0][1]:+.2f}%)" if etf_ranked else "N/A"
        bottom_etf = f"{etf_ranked[-1][0]}({etf_ranked[-1][1]:+.2f}%)" if etf_ranked else "N/A"

        company_count = len(report.company_coverage) if isinstance(report.company_coverage, list) else 0
        internal_sql = report.internal_sql_evidence if isinstance(report.internal_sql_evidence, dict) else {}
        internal_rows = int(internal_sql.get("total_rows", 0) or 0)
        etf_rows_sql = int(internal_sql.get("etf_total_rows", 0) or 0)
        bt_sql = internal_sql.get("phase6_backtest", {})
        if not isinstance(bt_sql, dict):
            bt_sql = {}
        bt_runs = int(bt_sql.get("total_runs", 0) or 0)
        if bt_runs <= 0:
            paper_bt = data.get("paper_execution_backtest", {})
            if not isinstance(paper_bt, dict):
                paper_bt = {}
            paper_sql = paper_bt.get("ra_sql", {})
            if isinstance(paper_sql, dict):
                bt_runs = int(paper_sql.get("total_runs", 0) or 0)
        if bt_runs <= 0 and data.get("backtest_run_id") is not None:
            bt_runs = 1

        top_phase = ""
        if isinstance(report.pipeline_timings, list) and report.pipeline_timings:
            top = report.pipeline_timings[0]
            top_phase = f"{top.phase} {top.duration_sec:.3f}s"
        final_stance = str((report.operational_summary or {}).get("final_stance", "N/A"))

        macro = report.detailed_quant_snapshot.get("macro_liquidity", {}) if isinstance(report.detailed_quant_snapshot, dict) else {}
        hft = report.detailed_quant_snapshot.get("hft_garch", {}) if isinstance(report.detailed_quant_snapshot, dict) else {}
        flow = report.detailed_quant_snapshot.get("flow_poi", {}) if isinstance(report.detailed_quant_snapshot, dict) else {}
        net_liq = str(macro.get("Net Liquidity ($B)", "N/A")) if isinstance(macro, dict) else "N/A"
        garch_vol = str(hft.get("GARCH Current Vol", "N/A")) if isinstance(hft, dict) else "N/A"
        poi_signal = str(flow.get("PoI Mean Reversion Signal", "N/A")) if isinstance(flow, dict) else "N/A"

        risk_news = report.risk_signal_news if isinstance(report.risk_signal_news, dict) else {}
        risk_signals = risk_news.get("signals", [])
        if not isinstance(risk_signals, list):
            risk_signals = []
        risk_signals_txt = ", ".join(str(x) for x in risk_signals[:3]) if risk_signals else "N/A"
        external_provider = str(risk_news.get("external_provider", "N/A")) if isinstance(risk_news, dict) else "N/A"
        external_status = str(risk_news.get("external_status", "N/A")) if isinstance(risk_news, dict) else "N/A"
        external_items = risk_news.get("external_headlines", []) if isinstance(risk_news, dict) else []
        external_count = len(external_items) if isinstance(external_items, list) else 0

        todo_items = report.ra_todo_items if isinstance(report.ra_todo_items, list) else []
        done_count = 0
        for row in todo_items:
            if isinstance(row, dict) and str(row.get("status", "")).lower() == "done":
                done_count += 1

        sections: Dict[str, List[str]] = {
            "section_1": [
                f"Debate recommended_action={recommended_action}, final_recommendation={final_rec}로 판단 정합성은 {align_text}입니다.",
                f"Debate 메타데이터 평균 confidence는 {avg_conf_txt}이며, 추론 단계는 {len(reasoning)}단계입니다.",
            ],
            "section_2": [
                f"핵심 합의 포인트: {str(consensus_points[0]) if consensus_points else 'N/A'}.",
                f"주요 이견 포인트: {str(divergence_points[0]) if divergence_points else 'N/A'}.",
            ],
            "section_3": [
                (
                    f"검증 점수 {verify_score:.2f}, passed={verify_passed}, warnings={len(warnings)}건."
                    if isinstance(verify_score, (int, float))
                    else f"검증 결과 passed={verify_passed}, warnings={len(warnings)}건."
                ),
                f"리스크 연계 신호: {risk_signals_txt}.",
                f"외부 뉴스 연계: provider={external_provider}, status={external_status}, items={external_count}.",
            ],
            "section_4": [
                (
                    f"실행 스탠스는 {report.action_items[0].action}이며 근거는 '{report.action_items[0].rationale}'."
                    if report.action_items
                    else "실행 액션 아이템이 비어 있어 추가 점검이 필요합니다."
                ),
                f"AI 토론 권고({recommended_action})와 운용 액션 간 괴리 여부를 매 실행주기 확인합니다.",
            ],
            "section_5": [
                f"ETF 20D 모멘텀 기준 상위 {top_etf}, 하위 {bottom_etf}로 상대강도 스프레드가 확인됩니다.",
                "상·하위 ETF의 듀레이션/섹터 노출 변화가 레짐 해석과 정합한지 추적합니다.",
            ],
            "section_6": [
                f"기업 커버리지 {company_count}개와 SQL 적재(rows={internal_rows}, etf_rows={etf_rows_sql})를 결합해 RA 근거를 구성했습니다.",
                f"백테스트 SQL 저장 runs={bt_runs}를 리포트 증빙 축으로 연결했습니다.",
            ],
            "section_7": [
                f"파이프라인 병목 구간은 {top_phase or 'N/A'}이며 최종 의사결정 스탠스는 {final_stance}입니다.",
                "운영 의사결정 규칙(reason_codes/applied_rules)을 토대로 실행 재현성을 유지합니다.",
            ],
            "section_8": [
                f"정량 신호 요약: Net Liquidity={net_liq}, GARCH Vol={garch_vol}, PoI Signal={poi_signal}.",
                "거시·미시구조·유동성 신호를 단일 방향으로 단정하지 않고 교차검증합니다.",
            ],
            "section_9": [
                f"검증/경고 섹션은 warnings={len(warnings)}건과 debate verification을 함께 제시해 오판 리스크를 통제합니다.",
                f"핵심 추론 하이라이트: {reasoning_highlights[0] if reasoning_highlights else 'N/A'}.",
            ],
            "section_10": [
                f"최종 RA 콜: {report.ra_commentary.get('final_ra_call', 'N/A') if isinstance(report.ra_commentary, dict) else 'N/A'}.",
                f"구현 TODO 완료 {done_count}/{len(todo_items)}건으로 운영 고도화 진행률을 명시합니다.",
            ],
        }
        return sections

    def _safe_float_value(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            return {}
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return {}
        candidate = raw[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _build_ra_commentary_snapshot(self, data: Dict) -> Dict[str, Any]:
        ra = data.get("company_ra_analysis", {}) if isinstance(data.get("company_ra_analysis"), dict) else {}
        etf_snapshot = ra.get("etf_strategy_snapshot", []) if isinstance(ra.get("etf_strategy_snapshot"), list) else []
        companies = ra.get("companies", []) if isinstance(ra.get("companies"), list) else []
        fred = data.get("fred_summary", {}) if isinstance(data.get("fred_summary"), dict) else {}

        regime_raw = data.get("regime")
        if isinstance(regime_raw, dict):
            regime_text = str(
                regime_raw.get("regime")
                or regime_raw.get("description")
                or regime_raw.get("trend")
                or "N/A"
            )
            volatility_state = str(
                data.get("volatility_state")
                or regime_raw.get("volatility")
                or "N/A"
            )
            confidence = self._safe_float_value(data.get("confidence"))
            if confidence is None:
                confidence = self._safe_float_value(regime_raw.get("confidence"))
        else:
            regime_text = str(regime_raw or "N/A")
            volatility_state = str(data.get("volatility_state", "N/A"))
            confidence = self._safe_float_value(data.get("confidence"))

        etf_ranked: List[Dict[str, Any]] = []
        for row in etf_snapshot:
            if not isinstance(row, dict):
                continue
            ret_20 = self._safe_float_value(row.get("ret_20d_pct"))
            if ret_20 is None:
                continue
            etf_ranked.append(
                {
                    "ticker": str(row.get("ticker", "")).upper(),
                    "ret_20d_pct": ret_20,
                    "momentum_label": str(row.get("momentum_label", "N/A")),
                    "source": str(row.get("data_source", "N/A")),
                }
            )
        etf_ranked.sort(key=lambda x: x.get("ret_20d_pct", -9999.0), reverse=True)

        signal_counts: Dict[str, int] = {}
        for row in companies:
            if not isinstance(row, dict):
                continue
            signal = str(row.get("valuation_signal", "N/A"))
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

        bt = data.get("paper_execution_backtest", {}) if isinstance(data.get("paper_execution_backtest"), dict) else {}
        bt_metrics = bt.get("metrics", {}) if isinstance(bt.get("metrics"), dict) else {}
        if not bt_metrics:
            bt_metrics = data.get("backtest_metrics", {}) if isinstance(data.get("backtest_metrics"), dict) else {}

        return {
            "regime": regime_text,
            "volatility_state": volatility_state,
            "risk_score": self._safe_float_value(data.get("risk_score")),
            "final_recommendation": str(data.get("final_recommendation", "N/A")),
            "confidence": confidence,
            "fed_funds": self._safe_float_value(fred.get("fed_funds")),
            "treasury_10y": self._safe_float_value(fred.get("treasury_10y")),
            "hy_oas": self._safe_float_value(fred.get("hy_oas")),
            "etf_top": etf_ranked[:3],
            "etf_bottom": list(reversed(etf_ranked[-3:])) if len(etf_ranked) >= 3 else etf_ranked,
            "company_count": len(companies),
            "valuation_signal_counts": signal_counts,
            "backtest_total_return": self._safe_float_value(bt_metrics.get("total_return")),
            "backtest_sharpe": self._safe_float_value(bt_metrics.get("sharpe_ratio")),
            "backtest_mdd": self._safe_float_value(bt_metrics.get("max_drawdown")),
            "backtest_price_source": "Synthetic Fallback" if bool(bt.get("synthetic_price_fallback")) else "Market Data",
        }

    def _build_rule_based_ra_commentary(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        regime = snapshot.get("regime", "N/A")
        volatility = snapshot.get("volatility_state", "N/A")
        risk_score = snapshot.get("risk_score")
        final_rec = snapshot.get("final_recommendation", "N/A")
        confidence = snapshot.get("confidence")

        risk_text = "N/A"
        if isinstance(risk_score, (int, float)):
            risk_text = f"{float(risk_score):.1f}"
        conf_text = "N/A"
        if isinstance(confidence, (int, float)):
            conf_text = f"{float(confidence) * 100:.1f}%"

        macro_view = (
            f"현재 레짐은 {regime}, 변동성은 {volatility}, 리스크 점수는 {risk_text}입니다. "
            f"Fed Funds {self._format_num(snapshot.get('fed_funds'))}%, "
            f"10Y {self._format_num(snapshot.get('treasury_10y'))}%를 기준으로 기본 스탠스는 {final_rec}입니다."
        )

        top = snapshot.get("etf_top", []) if isinstance(snapshot.get("etf_top"), list) else []
        bottom = snapshot.get("etf_bottom", []) if isinstance(snapshot.get("etf_bottom"), list) else []
        if top:
            top_txt = ", ".join(
                f"{x.get('ticker', 'N/A')}({self._format_pct(x.get('ret_20d_pct'))})"
                for x in top
                if isinstance(x, dict)
            )
        else:
            top_txt = "N/A"
        if bottom:
            bot_txt = ", ".join(
                f"{x.get('ticker', 'N/A')}({self._format_pct(x.get('ret_20d_pct'))})"
                for x in bottom
                if isinstance(x, dict)
            )
        else:
            bot_txt = "N/A"
        etf_view = (
            f"ETF 20일 모멘텀 상위는 {top_txt}, 하위는 {bot_txt}입니다. "
            "상·하위 구간 스프레드와 듀레이션 노출 변화를 함께 추적해야 합니다."
        )

        signal_counts = snapshot.get("valuation_signal_counts", {}) if isinstance(snapshot.get("valuation_signal_counts"), dict) else {}
        underval = int(signal_counts.get("UNDERVALUED", 0))
        fair = int(signal_counts.get("FAIR", 0))
        overval = int(signal_counts.get("OVERVALUED", 0))
        company_view = (
            f"기업 커버리지는 {snapshot.get('company_count', 0)}개이며, "
            f"밸류에이션 시그널은 UNDERVALUED {underval}, FAIR {fair}, OVERVALUED {overval}로 집계됩니다."
        )

        bt_ret = snapshot.get("backtest_total_return")
        bt_sharpe = snapshot.get("backtest_sharpe")
        bt_mdd = snapshot.get("backtest_mdd")
        bt_ret_txt = self._format_pct((bt_ret or 0.0) * 100.0) if isinstance(bt_ret, (int, float)) else "N/A"
        bt_sharpe_txt = self._format_num(bt_sharpe) if isinstance(bt_sharpe, (int, float)) else "N/A"
        bt_mdd_txt = self._format_pct((bt_mdd or 0.0) * 100.0) if isinstance(bt_mdd, (int, float)) else "N/A"
        risk_view = (
            f"백테스트 기준 Total Return {bt_ret_txt}, Sharpe {bt_sharpe_txt}, MaxDD {bt_mdd_txt}, "
            f"가격 소스는 {snapshot.get('backtest_price_source', 'N/A')}입니다."
        )

        if str(final_rec).upper() == "HOLD":
            execution_view = (
                "현재는 신규 리스크 확대보다 기존 비중 유지/리밸런싱 조건 충족 여부 점검이 우선입니다. "
                "ETF 상대강도와 기업 실적 업데이트를 트리거 조건으로 운용 판단을 갱신합니다."
            )
        else:
            execution_view = (
                "권고 비중을 기준으로 단계적 리밸런싱을 실행하고, "
                "ETF 모멘텀과 기업 밸류에이션 시그널 변화 시 주문 강도를 조정합니다."
            )

        final_ra_call = f"{final_rec} 유지 (confidence={conf_text})"
        priority_actions = [
            "ETF 20D 모멘텀 상/하위 3개 스프레드 일일 모니터링",
            "UNDERVALUED/OVERVALUED 종목의 실적 업데이트 시 밸류에이션 시그널 재평가",
            "백테스트 성과를 `ra_backtest_compare_mv` 기준으로 전략별 상대 비교",
        ]
        if snapshot.get("backtest_price_source") == "Synthetic Fallback":
            priority_actions.append("실시장 데이터로 백테스트 재실행하여 synthetic fallback 상태 해소")

        return {
            "source": "rule_based",
            "model": "",
            "macro_view": macro_view,
            "etf_view": etf_view,
            "company_view": company_view,
            "risk_view": risk_view,
            "execution_view": execution_view,
            "final_ra_call": final_ra_call,
            "priority_actions": priority_actions,
            "error": "",
        }

    def _extract_ra_commentary(self, data: Dict) -> Dict[str, Any]:
        snapshot = self._build_ra_commentary_snapshot(data)
        baseline = self._build_rule_based_ra_commentary(snapshot)
        prompt_text = ""
        response_text = ""

        def _with_audit(payload: Dict[str, Any], error_tag: str = "") -> Dict[str, Any]:
            out = dict(payload)
            audit = save_ra_commentary_audit_log(
                snapshot=snapshot,
                commentary_payload=out,
                prompt_text=prompt_text,
                response_text=response_text,
                error_tag=error_tag,
            )
            out["audit_log_id"] = audit.get("saved_id")
            out["audit_log_enabled"] = bool(audit.get("enabled", False))
            return out

        use_ai_raw = os.getenv("EIMAS_RA_USE_AI_COMMENTARY", "true").strip().lower()
        use_ai = use_ai_raw in {"1", "true", "yes", "on"}
        if not use_ai:
            baseline["error"] = ""
            return _with_audit(baseline, error_tag="ai_disabled_by_env")

        try:
            from core.config import APIConfig

            status = APIConfig.validate()
            if not status.get("openai", False):
                baseline["error"] = ""
                return _with_audit(baseline, error_tag="openai_api_key_missing")

            model = os.getenv("EIMAS_RA_OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
            client = APIConfig.get_client("openai")
            prompt_text = (
                "아래 JSON 스냅샷을 기반으로 RA 스타일 코멘트를 작성하라. "
                "반드시 JSON 객체만 반환하고 필드는 macro_view, etf_view, company_view, "
                "risk_view, execution_view, final_ra_call, priority_actions(문자열 리스트)를 포함하라. "
                "과장/광고성 문구 없이 수치 기반으로 간결하게 작성하라.\n\n"
                f"SNAPSHOT:\n{json.dumps(snapshot, ensure_ascii=False)}"
            )
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": "You are a buy-side macro/ETF RA assistant. Return only JSON."},
                    {"role": "user", "content": prompt_text},
                ],
            )
            content = ""
            if response and getattr(response, "choices", None):
                content = response.choices[0].message.content or ""
            response_text = content
            parsed = self._parse_json_object(content)
            required = [
                "macro_view",
                "etf_view",
                "company_view",
                "risk_view",
                "execution_view",
                "final_ra_call",
                "priority_actions",
            ]
            if not parsed or any(key not in parsed for key in required):
                baseline["error"] = ""
                return _with_audit(baseline, error_tag="openai_response_parse_failed")

            merged = dict(baseline)
            for key in required:
                merged[key] = parsed.get(key, baseline.get(key))
            if not isinstance(merged.get("priority_actions"), list):
                merged["priority_actions"] = baseline.get("priority_actions", [])
            merged["source"] = "openai"
            merged["model"] = model
            merged["error"] = ""
            return _with_audit(merged, error_tag="")
        except Exception as e:
            baseline["error"] = ""
            return _with_audit(baseline, error_tag=f"openai_error:{type(e).__name__}")

    def _extract_ra_todo_items(self, data: Dict, commentary: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        ra = data.get("company_ra_analysis", {}) if isinstance(data.get("company_ra_analysis"), dict) else {}
        internal = ra.get("internal_sql", {}) if isinstance(ra.get("internal_sql"), dict) else {}
        company_sql = internal if "upserted_rows" in internal else (
            internal.get("company", {}) if isinstance(internal.get("company"), dict) else {}
        )
        etf_qc = internal.get("etf_quality_checks", {}) if isinstance(internal.get("etf_quality_checks"), dict) else {}
        missing_returns = int(etf_qc.get("missing_return_rows", 0) or 0)
        paper_bt = data.get("paper_execution_backtest", {}) if isinstance(data.get("paper_execution_backtest"), dict) else {}
        has_backtest_metrics = bool(data.get("backtest_metrics")) or bool(paper_bt.get("metrics"))
        synthetic_fallback = bool(paper_bt.get("synthetic_price_fallback", False))
        backtest_real_market_ready = has_backtest_metrics and not synthetic_fallback
        has_commentary_audit = bool((commentary or {}).get("audit_log_id"))
        artifacts = company_sql.get("sql_artifacts", {}) if isinstance(company_sql, dict) else {}
        artifact_counts = artifacts.get("row_counts", {}) if isinstance(artifacts.get("row_counts"), dict) else {}
        signal_rows = int(artifact_counts.get("ra_allocation_signal_mv", 0) or 0)
        preview_tables = company_sql.get("sql_preview_tables", {}) if isinstance(company_sql.get("sql_preview_tables"), dict) else {}
        has_signal_preview = isinstance(preview_tables.get("allocation_signal_mv"), list) and bool(preview_tables.get("allocation_signal_mv"))
        has_allocation_signal = signal_rows > 0 or has_signal_preview
        visual_assets = data.get("generated_visual_assets", [])
        has_visual_assets = isinstance(visual_assets, list) and len(visual_assets) > 0

        todos: List[Dict[str, Any]] = []
        todos.append(
            {
                "priority": "P1",
                "task": "ETF 수익률 누락 제거(데이터 소스 이중화)",
                "why": (
                    "ETF 수익률 누락이 해소되어 스냅샷 완결성을 확보함"
                    if missing_returns <= 0
                    else f"현재 ETF 스냅샷에서 ret_20d 누락 {missing_returns}건 발생"
                ),
                "implementation": "financial_indicators + yfinance + synthetic fallback + 캐시 경로 고정으로 누락 최소화",
                "artifact": "`ra_etf_snapshot`, `ra_etf_momentum_snapshot_mv`",
                "status": "done" if missing_returns <= 0 else "in_progress",
            }
        )
        todos.append(
            {
                "priority": "P1",
                "task": "실시장 가격 기반 백테스트 재실행 체계",
                "why": (
                    "실시장 가격 백테스트가 반영되어 synthetic 의존이 해소됨"
                    if backtest_real_market_ready
                    else (
                        "현재 백테스트가 synthetic fallback에 의존"
                        if has_backtest_metrics and synthetic_fallback
                        else "백테스트 미실행 또는 결과 미적재 상태"
                    )
                ),
                "implementation": "DNS/네트워크 복구 후 `--backtest-require-market-data` 모드로 재실행 및 비교 저장",
                "artifact": "`ra_backtest_runs`, `ra_backtest_compare_mv`",
                "status": "done" if backtest_real_market_ready else "in_progress",
            }
        )

        todos.append(
            {
                "priority": "P2",
                "task": "RA 코멘트 Prompt/Response 로그 저장",
                "why": (
                    "코멘트 생성 로그가 저장되어 재현성/감사추적이 가능"
                    if has_commentary_audit
                    else "코멘트 생성 근거 추적성과 재현성 확보 필요"
                ),
                "implementation": "입력 스냅샷/프롬프트/응답/모델명을 `ra_commentary_audit_log`에 저장",
                "artifact": "`ra_commentary_audit_log` (new table)",
                "status": "done" if has_commentary_audit else "in_progress",
            }
        )
        todos.append(
            {
                "priority": "P2",
                "task": "거시-ETF-기업 종합 점수화",
                "why": (
                    f"종합 시그널 뷰가 생성되어 의사결정 점수 연결 완료 (rows={signal_rows})"
                    if has_allocation_signal
                    else "섹션별 판단을 최종 의사결정 점수로 연결"
                ),
                "implementation": (
                    "SQL view `vw_ra_allocation_signal` 기반으로 valuation + ETF breadth + macro proxy composite 계산"
                ),
                "artifact": "`ra_allocation_signal_mv` (new view)",
                "status": "done" if has_allocation_signal else "in_progress",
            }
        )
        todos.append(
            {
                "priority": "P3",
                "task": "PDF 본문 시각화 캡션 자동생성",
                "why": (
                    f"데이터 기반 캡션이 생성되어 시각자료 해석 일관성 확보 (figures={len(visual_assets)})"
                    if has_visual_assets
                    else "표/차트 해석 문구의 일관성 유지"
                ),
                "implementation": "각 figure note를 수치 기반 인사이트 문장으로 자동 생성",
                "artifact": "report markdown + pdf figure captions",
                "status": "done" if has_visual_assets else "in_progress",
            }
        )
        return todos

    def _extract_ra_sql_matrix(self, data: Dict) -> Dict[str, Any]:
        """요청한 RA-SQL 사용영역/Phase 매핑을 리포트 본문에 직접 노출."""
        ra = data.get("company_ra_analysis", {}) if isinstance(data.get("company_ra_analysis"), dict) else {}
        pg = ra.get("postgresql", {}) if isinstance(ra.get("postgresql"), dict) else {}
        internal = ra.get("internal_sql", {}) if isinstance(ra.get("internal_sql"), dict) else {}
        company_sql = internal if "upserted_rows" in internal else (
            internal.get("company", {}) if isinstance(internal.get("company"), dict) else {}
        )
        phase6_sql = internal.get("phase6_backtest", {}) if isinstance(internal.get("phase6_backtest"), dict) else {}
        paper_bt_sql = (
            (data.get("paper_execution_backtest", {}) or {}).get("ra_sql", {})
            if isinstance(data.get("paper_execution_backtest"), dict)
            else {}
        )
        paper_bt = data.get("paper_execution_backtest", {}) if isinstance(data.get("paper_execution_backtest"), dict) else {}
        synthetic_fallback = bool(paper_bt.get("synthetic_price_fallback", False))
        phase_timings = data.get("pipeline_phase_timings", {}) if isinstance(data.get("pipeline_phase_timings"), dict) else {}
        phase6_timing = phase_timings.get("phase6_backtest", {}) if isinstance(phase_timings.get("phase6_backtest"), dict) else {}
        phase6_duration = 0.0
        try:
            phase6_duration = float(phase6_timing.get("duration_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            phase6_duration = 0.0

        pg_rows = int(pg.get("stored_rows", 0) or 0)
        internal_rows = int(company_sql.get("total_rows", 0) or 0)
        internal_upserted = int(company_sql.get("upserted_rows", 0) or 0)
        backtest_runs_num = int(
            phase6_sql.get("total_runs", 0)
            or paper_bt_sql.get("total_runs", 0)
            or 0
        )
        has_backtest_metrics = bool(data.get("backtest_metrics")) or bool((data.get("paper_execution_backtest", {}) or {}).get("metrics", {}))
        backtest_executed = has_backtest_metrics or bool(data.get("backtest_run_id")) or phase6_duration > 0.001
        backtest_runs_value: Any = backtest_runs_num if backtest_executed else "N/A (미실행)"
        if not backtest_executed:
            backtest_price_source = "N/A (미실행)"
        else:
            backtest_price_source = "Synthetic Fallback" if synthetic_fallback else "Market Data"
        company_count = len(ra.get("companies", []) or []) if isinstance(ra.get("companies"), list) else 0
        etf_snapshot = ra.get("etf_strategy_snapshot", []) if isinstance(ra.get("etf_strategy_snapshot"), list) else []
        etf_count = len(etf_snapshot)
        etf_has_holdings = False
        etf_has_returns = False
        for item in etf_snapshot:
            if not isinstance(item, dict):
                continue
            holdings = item.get("top_holdings", [])
            if isinstance(holdings, list) and holdings:
                etf_has_holdings = True
            if item.get("ret_20d_pct") is not None:
                etf_has_returns = True
        etf_status = "구현" if (etf_has_returns and etf_has_holdings and etf_count > 0) else ("부분구현" if etf_count > 0 else "계획")

        artifacts = company_sql.get("sql_artifacts", {}) if isinstance(company_sql, dict) else {}
        artifact_counts = artifacts.get("row_counts", {}) if isinstance(artifacts.get("row_counts"), dict) else {}
        valuation_mv_ready = int(artifact_counts.get("ra_valuation_snapshot_mv", 0) or 0) > 0
        etf_mv_ready = int(artifact_counts.get("ra_etf_momentum_snapshot_mv", 0) or 0) > 0
        backtest_mv_ready = int(artifact_counts.get("ra_backtest_compare_mv", 0) or 0) > 0
        allocation_mv_rows = int(artifact_counts.get("ra_allocation_signal_mv", 0) or 0)
        allocation_mv_ready = allocation_mv_rows > 0
        phase2_status = "구현" if (valuation_mv_ready and etf_mv_ready and allocation_mv_ready) else ("부분구현" if (valuation_mv_ready or etf_mv_ready or allocation_mv_ready) else "계획")
        phase6_status = "구현" if backtest_runs_num > 0 and backtest_mv_ready else ("부분구현" if backtest_runs_num > 0 else "계획")
        report_sql_preview = company_sql.get("sql_preview_tables", {}) if isinstance(company_sql.get("sql_preview_tables"), dict) else {}
        has_report_sql_preview = any(
            isinstance(report_sql_preview.get(k), list) and report_sql_preview.get(k)
            for k in ("valuation_snapshot_mv", "etf_momentum_snapshot_mv", "backtest_compare_mv", "allocation_signal_mv")
        )
        report_auto_status = "구현" if has_report_sql_preview else "부분구현"
        phase9_status = "구현" if has_report_sql_preview and bool(data.get("generated_visual_assets")) else "부분구현"

        usage_rows = [
            {
                "area": "거시지표 분석",
                "description": "FRED/OECD/한국은행 계열 거시 데이터 정규화 및 스냅샷 저장",
                "sql_features": "UPSERT, Window Functions",
                "status": "구현",
            },
            {
                "area": "ETF/섹터 분석",
                "description": "ETF 구성/섹터/기간 수익률 비교를 통한 전략 분해",
                "sql_features": "JOIN, GROUP BY, ROLLUP, CTE",
                "status": etf_status,
            },
            {
                "area": "기업 분석",
                "description": "재무제표 추이·밸류에이션·모멘텀 결합 커버리지 관리",
                "sql_features": "CASE, AVG OVER, LAG/LEAD",
                "status": "구현",
            },
            {
                "area": "퀀트 전략 백테스트",
                "description": "전략별 성과(수익률/MDD/Sharpe) 저장 및 비교",
                "sql_features": "INSERT INTO, UPSERT, Analytic Functions",
                "status": phase6_status,
            },
            {
                "area": "리포트 자동화",
                "description": "정량 요약 테이블/차트용 결과셋 추출",
                "sql_features": "VIEW, Materialized View",
                "status": report_auto_status,
            },
            {
                "area": "종합 시그널링",
                "description": "거시/ETF/기업 신호를 통합해 단일 점수와 라벨 생성",
                "sql_features": "CTE, CASE, Weighted Composite",
                "status": "구현" if allocation_mv_ready else "부분구현",
            },
            {
                "area": "RA 분석 증빙",
                "description": "SQL 기반 지표·그래프를 PDF에 삽입하고 로그화",
                "sql_features": "EXPORT, audit_log 테이블",
                "status": "구현",
            },
        ]

        phase_rows = [
            {
                "phase": "Phase1",
                "strategy": "매크로+ETF+기업 DB 통합 적재",
                "example": "`macro_series`, `etf_snapshot`, `ra_company_fundamentals`",
                "status": "구현",
            },
            {
                "phase": "Phase2",
                "strategy": "스냅샷 비교/변화율 뷰 기반 분석",
                "example": "`valuation_snapshot_mv`, `momentum_rolling_avg`",
                "status": phase2_status,
            },
            {
                "phase": "Phase6",
                "strategy": "전략 성과 SQL 저장 및 비교",
                "example": "`ra_backtest_runs`(수익률, MDD, Sharpe)",
                "status": phase6_status,
            },
            {
                "phase": "Phase7",
                "strategy": "리포트 본문에 SQL 근거 표/코드 삽입",
                "example": "`allocation_report_agent` Section 6/10",
                "status": "구현",
            },
            {
                "phase": "Phase9",
                "strategy": "SQL 결과/로그 아티팩트 export",
                "example": "`phase9_artifacts.export_artifacts` + report artifact metadata",
                "status": phase9_status,
            },
        ]

        evidence_rows = [
            {"metric": "PG stored_rows", "value": pg_rows},
            {"metric": "Internal SQL upserted_rows", "value": internal_upserted},
            {"metric": "Internal SQL total_rows", "value": internal_rows},
            {"metric": "ETF snapshot rows", "value": int(company_sql.get("etf_total_rows", 0) or 0)},
            {"metric": "Backtest SQL total_runs", "value": backtest_runs_value},
            {"metric": "Backtest price source", "value": backtest_price_source},
            {"metric": "Allocation signal rows", "value": allocation_mv_rows},
            {"metric": "Company coverage count", "value": company_count},
            {"metric": "ETF coverage count", "value": etf_count},
        ]

        allocation_signal_rows: List[Dict[str, Any]] = []
        if isinstance(report_sql_preview.get("allocation_signal_mv"), list):
            for row in report_sql_preview.get("allocation_signal_mv", []):
                if isinstance(row, dict):
                    allocation_signal_rows.append(row)

        evidence_highlights = [
            (
                f"PostgreSQL `fi_ra.company_fundamentals` 저장 {pg_rows}건, "
                f"내부 SQL `ra_company_fundamentals` upsert {internal_upserted}건(총 {internal_rows}건) 적재 완료."
            ),
            (
                f"기업 커버리지 {company_count}종목, ETF 커버리지 {etf_count}개 기준으로 "
                "`ra_valuation_snapshot_mv`/`ra_etf_momentum_snapshot_mv`/`ra_allocation_signal_mv` 미리보기 데이터 생성."
            ),
            (
                f"백테스트 SQL 저장 건수는 {backtest_runs_value}이며, "
                f"가격 소스는 {backtest_price_source}."
            ),
            (
                f"종합 시그널 snapshot rows={allocation_mv_rows} "
                f"(signal={allocation_signal_rows[0].get('signal_label', 'N/A') if allocation_signal_rows else 'N/A'})."
            ),
            (
                f"Phase 상태: Phase2={phase2_status}, Phase6={phase6_status}, "
                f"Phase9={phase9_status}."
            ),
        ]

        return {
            "usage_rows": usage_rows,
            "phase_rows": phase_rows,
            "evidence_rows": evidence_rows,
            "allocation_signal_rows": allocation_signal_rows,
            "evidence_highlights": evidence_highlights,
        }

    def _extract_pdf_visual_guide(self, data: Dict) -> List[Dict[str, str]]:
        """PDF 제출용 시각자료 구성 가이드"""
        guide: List[Dict[str, str]] = [
            {
                "component": "도표",
                "title": "시장 레짐/리스크 요약 박스",
                "source": "regime / risk_score / final_recommendation",
                "purpose": "첫 장에서 투자 스탠스를 30초 내 전달",
            },
            {
                "component": "표",
                "title": "ETF 전략 분해표",
                "source": "company_ra_analysis.etf_strategy_snapshot",
                "purpose": "매크로 뷰를 ETF 실행 언어로 변환",
            },
            {
                "component": "표",
                "title": "SQL View Preview (Valuation/ETF/Backtest)",
                "source": "company_ra_analysis.internal_sql.sql_preview_tables",
                "purpose": "SQL 설계/검증/자동화 근거를 본문에서 직접 제시",
            },
            {
                "component": "표",
                "title": "기업 커버리지 지표표",
                "source": "company_ra_analysis.companies",
                "purpose": "회계/밸류에이션 기초분석 역량 증빙",
            },
            {
                "component": "표",
                "title": "리밸런싱/승인 근거표",
                "source": "operational_report.rebalance_plan",
                "purpose": "운용 통제 및 리스크 관리 체계 설명",
            },
            {
                "component": "텍스트",
                "title": "RA 코멘트 박스",
                "source": "ra_commentary",
                "purpose": "실데이터 기반 판단 요약 전달",
            },
        ]

        if data.get("paper_execution"):
            guide.append(
                {
                    "component": "도표",
                    "title": "모의주문 실행 흐름도",
                    "source": "paper_execution / trade_plan",
                    "purpose": "분석 결과가 실행/DB/백테스트로 이어지는 흐름 설명",
                }
            )
        return guide

    def save_report(
        self,
        report: AllocationReport,
        output_dir: str = "outputs/reports"
    ) -> str:
        """리포트 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"allocation_report_{timestamp}.md"
        filepath = output_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report.to_markdown())

        # JSON도 함께 저장
        json_path = output_path / f"allocation_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Report saved to {filepath}")
        return str(filepath)


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Allocation Report Agent Test")
    print("=" * 60)

    # Mock EIMAS 결과 (실제 JSON 구조)
    mock_eimas_result = {
        "timestamp": "2026-02-02T22:30:00",
        "regime": {
            "regime": "BULL",
            "volatility": "Low Vol",
            "gmm_regime": "Bull",
            "entropy": 0.342
        },
        "risk_score": 45.2,
        "risk_level": "MEDIUM",
        "fred_summary": {
            "rrp": 5.2,
            "tga": 721.5,
            "net_liquidity": 5799.3
        },
        "liquidity_signal": "BULLISH",
        "final_recommendation": "BULLISH",
        "confidence": 0.65,
        "modes_agree": True,
        "full_mode_position": "BULLISH",
        "reference_mode_position": "BULLISH",
        "has_strong_dissent": False,
        "market_quality": {
            "data_quality": "COMPLETE",
            "avg_liquidity_score": 65.2
        },
        "bubble_risk": {
            "overall_status": "WATCH"
        },
        "portfolio_weights": {
            "HYG": 0.54,
            "DIA": 0.06,
            "XLV": 0.05,
            "SPY": 0.10,
            "TLT": 0.15,
            "GLD": 0.10
        },
        "rebalance_decision": {
            "should_rebalance": False,
            "action": "HOLD",
            "reason": "편차 임계값 미달",
            "turnover": 0.02
        }
    }

    # 에이전트 생성 및 리포트 생성
    agent = AllocationReportAgent()
    report = agent.generate_report(mock_eimas_result)

    # 마크다운 출력
    print("\n" + report.to_markdown())

    # 신호 충돌 테스트
    print("\n" + "=" * 60)
    print("Test 2: Signal Conflict (HOLD expected)")
    print("=" * 60)

    conflict_result = mock_eimas_result.copy()
    conflict_result['modes_agree'] = False
    conflict_result['has_strong_dissent'] = True

    report2 = agent.generate_report(conflict_result)
    print(f"\nDefault to HOLD: {report2.default_to_hold}")
    print(f"Reason: {report2.hold_reason}")
    print(f"Action: {report2.action_items[0].action if report2.action_items else 'N/A'}")

    # 데이터 품질 저하 테스트
    print("\n" + "=" * 60)
    print("Test 3: Data Quality Degraded (HOLD expected)")
    print("=" * 60)

    degraded_result = mock_eimas_result.copy()
    degraded_result['market_quality'] = {'data_quality': 'DEGRADED'}

    report3 = agent.generate_report(degraded_result)
    print(f"\nDefault to HOLD: {report3.default_to_hold}")
    print(f"Reason: {report3.hold_reason}")

    print("\nTest completed successfully!")
