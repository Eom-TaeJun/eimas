---
name: eimas-output-compare
description: 두 EIMAS 실행 결과 JSON을 비교해 권고·리스크·레짐 변화를 요약. $ARGUMENTS로 파일 경로 1~2개 전달 가능.
argument-hint: "[file1.json] [file2.json] — 생략 시 최신 2개 자동 선택"
allowed-tools: Bash, Read, Glob
user-invocable: true
---

# EIMAS 결과 비교

## 1. 파일 선택

`$ARGUMENTS`가 있으면 전달된 경로 사용. 없으면:

```bash
ls -t /home/tj/projects/autoai/eimas/outputs/eimas_*.json 2>/dev/null | head -2
```

첫 번째(newer) = **현재**, 두 번째(older) = **이전**

## 2. 핵심 지표 추출 (두 파일 모두)

```bash
jq '{
  ts: .timestamp,
  rec: .final_recommendation,
  risk: .risk_score,
  regime: .regime_result.regime,
  confidence: .debate_result.confidence,
  weights: (.portfolio_weights // .allocation_result.weights // null)
}' <파일경로>
```

## 3. 출력 형식

```
📊 EIMAS 결과 비교
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           이전              현재
시각:    <ts_old>      →  <ts_new>
권고:    <rec_old>     →  <rec_new>   [변화: ✅/⚠️/🔴]
리스크:  <risk_old>    →  <risk_new>  [Δ <diff>]
레짐:    <reg_old>     →  <reg_new>
확신도:  <conf_old>%   →  <conf_new>%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
포트폴리오 배분 변화:
  <asset>: <old>% → <new>% (Δ <diff>%)
  ...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
변화 해석: <1줄 요약>
```

변화 아이콘 기준:
- ✅ 동일 방향 유지
- ⚠️ 강도 변화 (BULLISH→NEUTRAL 등)
- 🔴 방향 전환 (BULLISH→BEARISH)
