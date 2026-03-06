# Web Design System — EIMAS
> Version: 1.0.0 | 마스터: `autoai/design-system/web-design-system.md`
> **EIMAS 커스터마이징**: 다크 테마 고정, surface 토큰 추가

---

## 역할 정의

당신은 전문 프론트엔드 개발자이자 UI/UX 디자이너입니다.
기술 스택: **Next.js + Tailwind CSS + shadcn/ui**.
이 문서의 규칙을 어기는 코드는 작성하지 마세요.

---

## 1. 컬러 팔레트 (EIMAS 다크 테마)

```
PRIMARY:   #2563EB   (Indigo Blue)  — CTA 버튼, 링크, 활성 상태, 로고 강조
SECONDARY: #10B981   (Emerald)      — 긍정 지표, 상승, 성공, 완료 상태, 액션 버튼

SURFACE (다크 배경 레이어):
  surface:         #0d1117   → bg-surface        (페이지 최상위 배경)
  surface-card:    #161b22   → bg-surface-card   (카드, 패널 배경)
  surface-overlay: #21262d   → bg-surface-overlay (호버, 입력 배경)

BORDER:
  border:          #30363d   → border-border      (모든 구분선, 카드 테두리)
  border-subtle:   #21262d   → border-border-subtle

TEXT:
  기본:   #e6edf3   (gray-200 상당)
  보조:   #8b949e   (gray-400 상당)
```

**절대 금지:**
- 위 3가지 계열 외 색상 임의 추가 금지
- 그라디언트 남용 금지 (primary→secondary 그라디언트 1곳만 허용)
- 색상으로만 정보 전달 금지 (반드시 아이콘/텍스트 병행)

### tailwind.config.js 설정 (필수)
```js
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#2563EB',
          50:  '#EFF6FF',
          100: '#DBEAFE',
          500: '#2563EB',
          600: '#1D4ED8',
          700: '#1E40AF',
        },
        secondary: {
          DEFAULT: '#10B981',
          50:  '#ECFDF5',
          100: '#D1FAE5',
          500: '#10B981',
          600: '#059669',
          700: '#047857',
        },
      },
    },
  },
}
```

---

## 2. 타이포그래피

```
폰트: Inter (기본) — next/font/google로 로드
크기 스케일 (엄수):
  - 페이지 제목:     text-3xl font-bold      (30px)
  - 섹션 제목:       text-xl  font-semibold   (20px)
  - 카드 제목:       text-base font-semibold  (16px)
  - 본문:            text-sm  font-normal     (14px)
  - 보조 텍스트:     text-xs  font-normal     (12px)  text-gray-500

행간: leading-relaxed (본문) / leading-tight (제목)
자간: tracking-tight (제목만)
```

**절대 금지:**
- text-lg, text-2xl 등 스케일 임의 변형 금지
- font-black, font-thin 등 극단적 굵기 금지

---

## 3. 간격 시스템

**4px 기반 8단계 스케일만 사용:**

```
gap / padding / margin:
  xs:  4px   → p-1,  gap-1
  sm:  8px   → p-2,  gap-2
  md:  16px  → p-4,  gap-4   ← 카드 내부 기본값
  lg:  24px  → p-6,  gap-6   ← 섹션 간 기본값
  xl:  32px  → p-8,  gap-8
  2xl: 48px  → p-12, gap-12
  3xl: 64px  → p-16, gap-16
  4xl: 96px  → p-24, gap-24
```

**레이아웃 기본값:**
- 페이지 좌우 패딩: `px-4 md:px-8 lg:px-16`
- 최대 너비: `max-w-7xl mx-auto`
- 카드 내부: `p-4` (소) / `p-6` (중) / `p-8` (대)
- 섹션 간 세로 간격: `py-12` (모바일) / `py-20` (데스크톱)

---

## 4. 컴포넌트 규칙

### 사용 가능
- **shadcn/ui 전용** — Button, Card, Input, Badge, Dialog, Table, Tabs 등
- Lucide React — 아이콘 (shadcn/ui 기본 아이콘 세트)
- Recharts — 차트/그래프 (데이터 시각화)

### 절대 금지
- MUI (Material UI)
- Ant Design
- Chakra UI
- 외부 MCP/플러그인 호출로 컴포넌트 생성

### Button 규칙
```tsx
// Primary CTA — 페이지당 1개 원칙
<Button className="bg-primary hover:bg-primary-600 text-white">
  행동 유도 문구
</Button>

// Secondary
<Button variant="outline" className="border-primary text-primary">
  보조 행동
</Button>

// Destructive / Danger
<Button variant="destructive">삭제</Button>

// 크기: sm (보조) / default (기본) / lg (히어로 CTA)
```

### Card 규칙
```tsx
// 기본 카드
<Card className="bg-white border border-gray-200 rounded-xl shadow-sm">
  <CardHeader className="p-6 pb-2">
    <CardTitle className="text-base font-semibold text-gray-900">제목</CardTitle>
  </CardHeader>
  <CardContent className="p-6 pt-2">내용</CardContent>
</Card>

// 강조 카드 (primary 테두리)
<Card className="border-primary border-2">
```

### Badge 규칙
```tsx
// 상태 배지
상승/긍정: <Badge className="bg-secondary-50 text-secondary-700">+2.4%</Badge>
하락/부정: <Badge className="bg-red-50 text-red-700">-1.2%</Badge>
중립:      <Badge className="bg-gray-100 text-gray-600">중립</Badge>
```

---

## 5. 레이아웃 패턴

### 대시보드 레이아웃
```
┌─────────────────────────────────────────┐
│  Sidebar (240px)  │  Main Content Area  │
│  - 네비게이션     │  - 페이지 헤더      │
│  - 메뉴 아이템    │  - 콘텐츠 그리드    │
│  - 하단 유저 정보 │                     │
└─────────────────────────────────────────┘
```

### 그리드 시스템
```
통계 카드: grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4
메인 콘텐츠: grid-cols-1 lg:grid-cols-3 gap-6 (2:1 비율)
전체 너비: col-span-full
```

### 반응형 브레이크포인트
```
모바일:  <640px   → 단일 컬럼, 사이드바 숨김
태블릿:  640-1024px → 2컬럼, 사이드바 오버레이
데스크톱: >1024px → 풀 레이아웃
```

---

## 6. 상태 표현 규칙

```
로딩:   Skeleton 컴포넌트 (shadcn/ui) — spinner 금지
에러:   red-50 배경 + red-600 텍스트 + AlertCircle 아이콘
성공:   secondary-50 배경 + secondary-700 텍스트 + CheckCircle 아이콘
경고:   amber-50 배경 + amber-700 텍스트 + AlertTriangle 아이콘
비어있음: gray-50 배경 + gray-400 텍스트 + 관련 아이콘 + 안내 문구
```

---

## 7. 다크모드 (EIMAS: 다크 고정)

- **EIMAS는 다크 테마 고정** — `html` 태그에 `class="dark"` 하드코딩
- `light:` 모드 클래스 추가 금지
- 배경 기본값: `bg-surface` (페이지) / `bg-surface-card` (카드)
- 텍스트 기본값: `text-white` (강조) / `text-gray-400` (보조)

---

## 8. 작업 방식 제한

1. **MCP 디자인 도구 호출 금지** — 토큰 낭비 + 일관성 파괴
2. **이 문서만을 디자인 판단 기준으로 사용**
3. UI 생성 전 반드시 확인:
   - [ ] 컬러가 팔레트 안에 있는가?
   - [ ] shadcn/ui 컴포넌트를 쓰고 있는가?
   - [ ] 간격이 4px 스케일을 따르는가?
   - [ ] 타이포 스케일을 벗어나지 않는가?

---

## 9. 레퍼런스 스타일 (EIMAS)

```
"A dark, data-dense financial intelligence dashboard.
 Inspired by Bloomberg Terminal, GitHub, and Linear dark mode aesthetics.
 Surface hierarchy: #0d1117 (page) → #161b22 (card) → #21262d (overlay).
 Indigo blue for primary actions. Emerald green for positive signals.
 Prioritize information density over decoration.
 Charts use Recharts with muted, professional color palettes."
```

---

## 프로젝트별 커스터마이징 방법

새 프로젝트 시작 시:
1. 이 파일을 `{프로젝트}/.claude/skills/web-design-system.md`로 복사
2. **섹션 1 (컬러)만** 프로젝트 색상으로 교체
3. **섹션 9 (레퍼런스)만** 프로젝트 성격에 맞게 교체
4. 나머지 규칙은 그대로 유지
