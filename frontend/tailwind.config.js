// tailwind.config.js — EIMAS Design System
// Based on: autoai/design-system/web-design-system.md
// Theme: Dark (GitHub-inspired) + Design System color tokens

/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      // ─── Design System: Primary / Secondary (교체 가능) ──────
      colors: {
        primary: {
          DEFAULT: '#2563EB',
          50:  '#EFF6FF',
          100: '#DBEAFE',
          400: '#60A5FA',
          500: '#2563EB',
          600: '#1D4ED8',
          700: '#1E40AF',
        },
        secondary: {
          DEFAULT: '#10B981',
          50:  '#ECFDF5',
          100: '#D1FAE5',
          400: '#34D399',
          500: '#10B981',
          600: '#059669',
          700: '#047857',
        },
        // ─── EIMAS Dark Surface (하드코딩 대체 토큰) ────────────
        surface: {
          DEFAULT: '#0d1117',  // bg-[#0d1117] → bg-surface
          card:    '#161b22',  // bg-[#161b22] → bg-surface-card
          overlay: '#21262d',  // 중간 레벨 배경
        },
        border: {
          DEFAULT: '#30363d',  // border-[#30363d] → border-DEFAULT
          subtle:  '#21262d',
        },
      },
      // ─── 폰트 (Design System 표준) ────────────────────────────
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      // ─── 간격 스케일 (4px 기반) ───────────────────────────────
      // Tailwind 기본값과 동일하므로 별도 override 불필요
      // ─── 최대 너비 ────────────────────────────────────────────
      maxWidth: {
        'content': '1400px',  // EIMAS 기존 max-w-[1400px] 표준화
      },
      borderRadius: {
        'sm':  '6px',
        'md':  '8px',
        'lg':  '12px',
        'xl':  '16px',
      },
    },
  },
  plugins: [
    require('tailwindcss-animate'),
  ],
}
