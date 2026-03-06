---
name: eimas-output-check
description: 최신 EIMAS 실행 결과 JSON을 읽어 핵심 지표를 요약 출력
argument-hint: [json-file or latest]
allowed-tools: Bash, Read, Glob
---

# EIMAS 결과 확인

다음 순서로 최신 EIMAS 결과를 확인한다.

## 1. 최신 결과 파일 찾기

```bash
ls -t /home/tj/projects/autoai/eimas/outputs/eimas_*.json 2>/dev/null | head -3
```

인수로 파일 경로가 전달된 경우: `$ARGUMENTS` 사용

## 2. 핵심 지표 추출

```bash
jq '{
  timestamp: .timestamp,
  recommendation: .final_recommendation,
  risk_score: .risk_score,
  regime: .regime_result.regime,
  confidence: .debate_result.confidence
}' <파일경로>
```

## 3. 포트폴리오 배분 확인

```bash
jq '.portfolio_weights // .allocation_result.weights // empty' <파일경로>
```

## 4. 출력 형식

결과를 아래 형식으로 요약:

```
📊 EIMAS 분석 결과 — <timestamp>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
권고:       <final_recommendation>
리스크 점수: <risk_score>/100
시장 레짐:  <regime> (<confidence>% 확신)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
포트폴리오 배분:
  <asset>: <weight>%
  ...
```

파일이 없으면 "outputs/에 실행 결과 없음. `python main.py --short` 실행 필요" 안내.
