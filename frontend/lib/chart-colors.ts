/**
 * EIMAS Chart Color Constants
 * Based on: autoai/design-system/web-design-system.md
 *
 * 사용법:
 *   import { COLORS } from "@/lib/chart-colors"
 *   <Line stroke={COLORS.primary} />
 *   <CartesianGrid stroke={COLORS.grid} />
 */

export const COLORS = {
  // ── Design System Tokens ─────────────────────────────────
  primary:       '#2563EB',   // Indigo Blue — primary CTA, 강조
  primaryLight:  '#60A5FA',   // primary-400  — 보조 강조
  secondary:     '#10B981',   // Emerald      — 상승, 긍정
  secondaryLight:'#34D399',   // secondary-400 — 보조 긍정

  // ── Semantic States ───────────────────────────────────────
  danger:        '#ef4444',   // red-500  — 하락, 위험
  dangerDark:    '#dc2626',   // red-600  — 심각 위험
  warning:       '#d97706',   // amber-600 — 경고
  warningLight:  '#fbbf24',   // amber-400 — 경고 보조

  // ── Surface / Chrome ─────────────────────────────────────
  surface:       '#0d1117',   // 차트 배경
  surfaceCard:   '#161b22',   // 패널 배경
  grid:          '#21262d',   // 그리드 라인
  border:        '#30363d',   // 차트 테두리
  textMuted:     '#8b949e',   // 축 레이블, 보조 텍스트
  white:         '#ffffff',

  // ── Chart Data Series (순서대로 사용) ─────────────────────
  series: [
    '#2563EB',  // primary     — 시리즈 1
    '#10B981',  // secondary   — 시리즈 2
    '#f59e0b',  // amber-500   — 시리즈 3
    '#8b5cf6',  // violet-500  — 시리즈 4
    '#ec4899',  // pink-500    — 시리즈 5
    '#06b6d4',  // cyan-500    — 시리즈 6
    '#f97316',  // orange-500  — 시리즈 7
    '#84cc16',  // lime-500    — 시리즈 8
  ],
} as const

// ── Recharts 공통 props ───────────────────────────────────────
export const CHART_DEFAULTS = {
  cartesianGrid: {
    strokeDasharray: '3 3',
    stroke: COLORS.grid,
  },
  xAxis: {
    tick: { fill: COLORS.textMuted, fontSize: 12 },
    axisLine: { stroke: COLORS.border },
    tickLine: false,
  },
  yAxis: {
    tick: { fill: COLORS.textMuted, fontSize: 12 },
    axisLine: false,
    tickLine: false,
  },
  tooltip: {
    contentStyle: {
      backgroundColor: COLORS.surfaceCard,
      border: `1px solid ${COLORS.border}`,
      borderRadius: '8px',
      color: COLORS.white,
    },
  },
  legend: {
    wrapperStyle: { color: COLORS.textMuted },
  },
} as const
