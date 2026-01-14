# EIMAS - Economic Intelligence Multi-Agent System

**Multi-Agent 기반 경제 분석 및 예측 시스템**

## 개요
LASSO 예측 모델과 Critical Path 분석을 Multi-Agent 아키텍처로 통합한 경제 인텔리전스 시스템

## 아키텍처
```
User Query → MetaOrchestrator
                ↓
    ┌───────────┼───────────┐
    ↓           ↓           ↓
Analysis    Forecast    Strategy
Agent       Agent       Agent
    └───────────┼───────────┘
                ↓
        Debate Protocol
                ↓
        Consensus Report
```

## 디렉토리 구조
```
eimas/
├── agents/           # 에이전트 모듈
│   ├── base_agent.py        # 추상 베이스 클래스
│   ├── analysis_agent.py    # Critical Path 분석
│   ├── forecast_agent.py    # LASSO 예측
│   ├── research_agent.py    # Perplexity 리서치
│   ├── strategy_agent.py    # 전략 권고
│   └── orchestrator.py      # 워크플로우 조정
├── core/             # 핵심 프레임워크
│   ├── schemas.py           # 데이터 스키마
│   ├── debate.py            # 토론 프로토콜
│   └── config.py            # API 설정
├── lib/              # 외부 라이브러리 래핑
│   ├── critical_path.py     # Bekaert VIX 분해
│   ├── data_collector.py    # yfinance/FRED 수집
│   └── lasso_forecaster.py  # LASSO 모델
├── tests/            # 테스트
├── configs/          # 설정 파일
├── outputs/          # 결과 저장
└── main.py           # 메인 실행 파일
```

## 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# API 키 설정
export ANTHROPIC_API_KEY='your-key'
export OPENAI_API_KEY='your-key'
export PERPLEXITY_API_KEY='your-key'
export GEMINI_API_KEY='your-key'
export FRED_API_KEY='your-key'
```

## 사용법

```bash
python main.py
```

## 경제학적 방법론

1. **LASSO 변수 선택**: L1 정규화로 핵심 변수만 선택
2. **Simultaneity 회피**: Treasury 변수 제외
3. **Horizon 분리**: 초단기(≤30일), 단기(31-90일), 장기(≥180일)
4. **Bekaert VIX 분해**: Uncertainty vs Risk Appetite
5. **Debate Protocol**: Rule-based 합의 도출

## 라이선스
MIT
