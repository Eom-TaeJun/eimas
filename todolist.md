# EIMAS 프로젝트 TODO LIST

> 8개 DOCX 파일 분석 결과를 주제별로 정리한 구현 작업 목록입니다.
> 마지막 업데이트: 2026-01-24

---

## 📋 개요

DOCX 파일 분석 결과, **5개 핵심 주제**로 분류됩니다:

1. **포트폴리오 최적화 및 자산 배분** (eco1, eco2)
2. **시장 미세구조 및 거래 메커니즘** (eco3)
3. **블록체인 기반 인덱스 & 스마트 거래** (eco4)
4. **AI/ML 기술 기초** (eco5, eco6)
5. **경제학 통합 및 인과관계 분석** (금융경제정리)

각 섹션에는 **구현 배경**, **구현 내용**, **주요 함수**, **키워드**가 포함되어 있습니다.

---

## 🎯 1. 포트폴리오 최적화 및 자산 배분

### 1.1 Hierarchical Risk Parity (HRP) 고도화

**출처:** eco1.docx

**구현 배경:**
- **Markowitz Curse**: 전통 Mean-Variance Optimization (MVO)의 수치 불안정성
  - 공분산 역행렬의 조건수(condition number)가 높으면 최적화 실패
  - 60/40 포트폴리오에서 리스크의 90%가 주식에 집중되는 문제
- **무한 자산 시대**: 토큰화로 인해 자산 개수가 급증 → 기존 MVO 방식 한계

**구현 내용:**
1. **Correlation-Distance 변환**
   ```python
   D[i,j] = sqrt(0.5 * (1 - rho[i,j]))
   ```

2. **Systemic Similarity 계산**
   - 단순 correlation을 넘어 자산 간 상호작용 방식 파악
   ```python
   D_bar[i,j] = sqrt(sum((D[k,i] - D[k,j])**2))
   ```

3. **Hierarchical Clustering**
   - Dendrogram 구성 (계층적 트리 구조)

4. **Matrix Seriation**
   - 유사 자산을 행렬에서 인접하게 배치

5. **Recursive Bisection**
   - Top-down 방식 가중치 할당
   ```python
   Var_Cluster = w_cluster.T @ Cov_Cluster @ w_cluster
   alpha = 1 - (Var_Left / (Var_Left + Var_Right))
   ```

**주요 함수:**
- `correlation_to_distance(corr_matrix)` - Distance 행렬 생성
- `compute_systemic_similarity(distance_matrix)` - D_bar 계산
- `hierarchical_clustering(D_bar)` - Dendrogram 구성
- `matrix_seriation(dendrogram)` - 재정렬
- `recursive_bisection(cov_matrix, clusters)` - 가중치 할당
- `validate_hrp_weights(weights)` - 가중치 합=1.0 검증

**검증 메트릭:**
- HRP 분산이 CLA (Critical Line Algorithm) 대비 **42% 낮음**
- Condition number 감소 확인
- Correlation 변화에 대한 강건성 테스트

**키워드:** HRP, Markowitz, MVO, Dendrogram, Recursive Bisection, 비용행렬, Seriation, Inverse-Variance Weighting

**EIMAS 통합:**
- `lib/graph_clustered_portfolio.py`에 Systemic Similarity 로직 추가
- 기존 MST 방식과 비교 벤치마킹

**우선순위:** ⭐⭐⭐ (단기 구현, 1-2주)

---

### 1.2 머신러닝 클러스터링 기반 포트폴리오 최적화

**출처:** eco2.docx

**구현 배경:**
- MVO 확장성 문제 (고차원 자산 시 계산 불가)
- Expected return 및 covariance 추정 오차 (측정 오류에 민감)
- 대규모 자산(100개+) 처리 필요

**구현 내용:**

1. **K-means 클러스터링**
   - 자산을 k개 군집으로 분할
   - Elbow Method로 최적 k 선택

2. **Hierarchical Clustering**
   - 다층 관계 파악 (Dendrogram)

3. **DBSCAN**
   - 이상치 탐지 (밀도 기반)
   - 비정상 자산 자동 제거

4. **Minimum Spanning Tree (MST)**
   - 최강 엣지만 유지
   - 거리 공식: `distance = sqrt(2*(1-correlation))`

5. **Dynamic Time Warping (DTW)**
   - 시계열 리드-래그 관계 파악

6. **LASSO 기반 공분산 추정**
   - Sparse covariance matrix 생성
   - 노이즈 제거

**주요 함수:**
- `kmeans_clustering(returns, n_clusters)` - K-means 실행
- `hierarchical_clustering(returns, method='ward')` - 계층적 클러스터링
- `dbscan_outlier_detection(returns, eps, min_samples)` - 이상치 탐지
- `build_mst(correlation_matrix)` - MST 구성
- `dtw_distance(series1, series2)` - DTW 거리 계산
- `lasso_covariance(returns, alpha)` - LASSO 공분산 추정

**검증 메트릭:**
- **Silhouette Score**: -1.0 ~ 1.0 (>0.7: 강한 구조, <0.25: 약한 구조)
- **Davies-Bouldin Index**: 낮을수록 좋음 (군집 분리도)
- **Calinski-Harabasz Score**: 높을수록 좋음 (분산 비율)

**알고리즘 플로우:**
```
1. 거리 행렬 계산: distance = sqrt(2*(1-correlation))
2. MST 또는 계층적 클러스터링 실행
3. 군집별 중심성 계산 (Degree, Betweenness, Eigenvector)
4. 포트폴리오 구성 (클러스터 간 분산화)
```

**키워드:** 클러스터링, K-means, MST, Silhouette, Davies-Bouldin, LASSO, DTW, DBSCAN

**EIMAS 통합:**
- `lib/regime_analyzer.py`에 GMM 클러스터링 이미 구현됨
- `lib/graph_clustered_portfolio.py`에 MST 로직 추가
- 섹터 클러스터링 + 포트폴리오 최적화 통합

**우선순위:** ⭐⭐ (중기 구현, 1개월)

---

## 🔬 2. 시장 미세구조 및 거래 메커니즘

### 2.1 High-Frequency Trading (HFT) 환경의 미세구조 지표

**출처:** eco3.docx

**구현 배경:**
- 시간 기준(Time bars) vs. 시간에 무관한 거래 분류 실패
- 알고리즘 거래자의 의도 숨김 (주문 분할)
- 유동성 충격/Liquidity crash 예측 불가
- 2010년 Flash Crash 등 극단적 시장 이벤트 대응

**구현 내용:**

1. **Tick Rule** (거래 분류)
   ```python
   if p[t] > p[t-1]: b[t] = 1  # Buy
   elif p[t] < p[t-1]: b[t] = -1  # Sell
   else: b[t] = b[t-1]  # Hold (이전 방향 유지)
   ```

2. **Roll's Measure** (Effective Spread)
   ```python
   Spread = 2 * sqrt(max(0, -cov(delta_p, delta_p_lag)))
   ```

3. **Kyle's Lambda** (Market Impact)
   ```python
   delta_p[t] = Lambda * (b[t] * V[t]) + error[t]
   # OLS 회귀로 Lambda 추정
   ```

4. **Amihud's Illiquidity**
   ```python
   Illiquidity = abs(Return[t]) / Volume[t]
   # 높을수록 비유동적
   ```

5. **VPIN** (Volume-Synchronized Probability of Informed Trading)
   ```python
   VPIN = sum(abs(V_buy[τ] - V_sell[τ])) / (n * V_bucket_size)
   # Volume Clock 기반 샘플링
   ```

**주요 함수:**
- `tick_rule_classification(prices)` - 거래 방향 분류
- `rolls_measure(price_changes)` - Effective Spread 계산
- `kyles_lambda(price_changes, signed_volume)` - Market Impact 추정
- `amihud_illiquidity(returns, volume)` - 비유동성 측정
- `vpin_indicator(prices, volumes, n_buckets)` - 정보거래확률 계산
- `volume_clock_sampling(df, volume_bucket)` - Volume 기준 샘플링
- `detect_silicon_traders(volumes)` - 알고리즘 거래자 식별

**검증 메트릭:**
- VPIN: Liquidity crash 1시간 전 급상승 (2010년 Flash Crash)
- Round Number Check: 100주 거래 vs. 99주 거래 비율 16.8:1
- Quote Stuffing 탐지: 주문 취소율 > 90%

**키워드:** Tick Rule, Kyle's Lambda, VPIN, Roll's Spread, 미세구조, HFT, 유동성, Volume Clock, Silicon Traders, Quote Stuffing

**EIMAS 통합:**
- `lib/microstructure.py`에 이미 구현됨 (Phase 2.4.1)
- **강화 필요**: Tick Rule, Roll's Measure 추가
- Volume Clock 샘플링 로직 구현
- VPIN 계산 정확도 개선 (현재는 일별 데이터 근사)

**우선순위:** ⭐⭐⭐⭐ (즉시 구현, 1주 이내)

---

## ⛓️ 3. 블록체인 기반 인덱스 & 스마트 거래

### 3.1 Proof-of-Index (PoI) 및 온체인 퀀트 전략

**출처:** eco4.docx

**구현 배경:**
- 기존 금융지수: 계산 블랙박스, 정산 지연 (T+2)
- 신흥국 통화 불안정성, 글로벌 유동성 접근 제한
- 탈중앙화 금융(DeFi) 시대의 투명성 요구

**구현 내용:**

1. **Proof-of-Index (PoI)**
   ```python
   I_t = sum(P_i_t * Q_i_t) / D_t
   hash = SHA-256(weights_dict)
   # On-chain 검증 가능
   ```

2. **Mean Reversion Signal** (Quant Strategy)
   ```python
   Z = (P_t - mean(P_window)) / std(P_window)
   # Buy if Z < -threshold (예: -2)
   # Sell if Z > threshold (예: +2)
   ```

3. **Smart Contract 기반 검증**
   - Off-chain 계산 → Hash → On-chain 검증
   - SHA-256 기반 데이터 무결성

4. **Stablecoin 활용**
   - **USDC**: 100% Treasury 담보 (규제 준수, 낮은 리스크)
   - **USDe**: Delta-Neutral Hedging 기반 수익 창출 (높은 리스크)

**주요 함수:**
- `calculate_proof_of_index(prices, quantities, divisor)` - 인덱스 계산
- `hash_index_weights(weights_dict)` - SHA-256 해시 생성
- `verify_on_chain(hash_value)` - Smart Contract 검증
- `mean_reversion_signal(prices, window, threshold)` - 평균회귀 신호
- `evaluate_stablecoin_risk(coin_type)` - 스테이블코인 리스크 평가

**알고리즘 플로우:**
```
1. 자산 가격(P), 공급량(Q) 수집 (Chainlink/Pyth 오라클)
2. 시가총액(MC) 계산 및 가중치(W) 산출
3. 인덱스 계산 및 Hash 생성
4. Smart Contract 검증
5. Mean Reversion 신호 생성
6. 거래 실행 (ZK-Rollup Layer 2)
```

**주요 기술:**
- **ZK-Rollup**: Layer 2 확장 (거래량 증가, 수수료 감소)
- **HFT 전략**: 밀리초 단위 실행
- **Arbitrage**: 거래소 간 가격 차익

**키워드:** Proof-of-Index, Smart Contract, Stablecoin, USDC, USDe, ZK-Rollup, Arbitrage, Mean Reversion, Chainlink

**EIMAS 통합:**
- `lib/genius_act_macro.py`에 Stablecoin 리스크 평가 이미 구현됨 (v2.1.1)
- **추가 구현**: Proof-of-Index 모듈 신규 생성
- Mean Reversion 전략 백테스팅

**우선순위:** ⭐⭐ (중기 구현, 1-2개월)

---

## 🤖 4. AI/ML 기술 기초

### 4.1 Convolution 기반 시계열 패턴 탐지

**출처:** eco5.docx

**구현 배경:**
- 주식 가격 heatmap에서 패턴 탐지 필요
- 기술적 지표 자동 추출
- CNN (Convolutional Neural Network) 기반 시계열 분석의 초석

**구현 내용:**

1. **2D Convolution 연산**
   ```python
   output_map[r, c] = sum(input_grid[r+i, c+j] * filter[i, j])
   # i, j는 필터 크기 (예: 3×3)
   ```

2. **알고리즘:**
   ```
   1. 3×3 필터 초기화 (Edge Detection, Momentum 등)
   2. Heatmap 좌상단에 필터 배치
   3. Element-wise 곱셈 → 합산 → Feature 값 저장
   4. Stride=1로 우향/하향 슬라이딩
   5. 최종 Feature Map 생성 (입력보다 작음)
   ```

3. **검증:**
   - 좌상단 값 검증
   - 우하단 값 검증
   - 출력 크기 = (입력 - 필터 + 1) / Stride

**주요 함수:**
- `conv2d(input_grid, filter, stride=1)` - 2D Convolution
- `generate_heatmap(prices, window)` - 가격 Heatmap 생성
- `edge_detection_filter()` - 엣지 탐지 필터
- `momentum_filter()` - 모멘텀 패턴 필터
- `validate_output_size(input, filter, stride)` - 출력 크기 검증

**적용 사례:**
- 기술적 지표 패턴 추출 (헤드앤숄더, 삼각 수렴 등)
- 가격 급등/급락 예측
- 섹터 간 상관관계 시각화

**키워드:** Convolution, Filter, Stride, Feature Map, CNN, 패턴 탐지, Heatmap, Edge Detection

**EIMAS 통합:**
- **신규 모듈**: `lib/cnn_pattern_detector.py` 생성
- 시계열 데이터를 2D 이미지로 변환 (가격×시간)
- `integrated_strategy.py`와 연동

**우선순위:** ⭐ (장기 구현, 3-6개월)

---

### 4.2 Large Language Model (LLM) 도메인 특화

**출처:** eco6.docx

**구현 배경:**
- N-gram, RNN/LSTM의 장거리 의존성 처리 실패
- 문맥 및 복잡한 언어 관계 표현 불가
- 경제학 도메인 특화 LLM 필요 (일반 LLM은 금융 용어 오류 多)

**구현 내용:**

1. **Transformer 아키텍처**
   - Self-Attention: 단어 간 가중치 학습
   - Multi-head Attention: 다양한 각도에서 관계 파악
   - Feed-forward Network: 비선형 변환

2. **LLM 개발 파이프라인**
   ```
   1. Pre-training: 대규모 말뭉치로 일반 패턴 학습
   2. Fine-tuning: 특정 작업(경제학, 금융)에 맞춤
   3. Multimodal Integration: 텍스트 + 차트/그래프
   ```

3. **주요 모델 벤치마크**
   - GPT-3: 1,750억 파라미터
   - LLaMA 2: 4,050억 파라미터
   - GPT-4o: 멀티모달 (텍스트 + 이미지)

**주요 함수:**
- `pretrain_transformer(corpus, vocab_size)` - Pre-training
- `finetune_on_economics(model, dataset)` - Fine-tuning
- `multimodal_inference(text, chart_image)` - 멀티모달 추론
- `evaluate_context_handling(model, long_sequence)` - 문맥 유지 평가
- `detect_bias(model_output)` - 편향 탐지

**알고리즘 플로우:**
```
1. 아키텍처 선택: Transformer
2. Pre-training (BPE 토크나이즈, masked language modeling)
3. 스케일링 (파라미터 수 증가)
4. Fine-tuning (Supervised Fine-Tuning)
5. 멀티모달 통합 (Vision Transformer, CLIP 등)
6. 배포 (편향 완화, 윤리적 AI)
```

**검증 메트릭:**
- Context handling: 긴 시퀀스에서 문맥 유지
- Reasoning: 복잡한 문제 해결
- Ethical compliance: 편향 감시

**키워드:** Transformer, Self-Attention, Pre-training, Fine-tuning, Multimodal, BERT, GPT, LLaMA

**EIMAS 통합:**
- `agents/orchestrator.py`에 Claude/Perplexity API 이미 사용 중
- **강화 방향**: 경제학 Fine-tuning 데이터셋 구축
- 멀티모달: 차트 이미지 → LLM 해석 기능 추가

**우선순위:** ⭐⭐ (중기 구현, 2-3개월)

---

## 📊 5. 경제학 통합 및 인과관계 분석

### 5.1 Causality vs. Correlation: 경제학적 인과관계 네트워크

**출처:** 금융경제정리.docx

**구현 배경:**
- **Causality vs. Correlation**: 경제학은 인과관계, ML은 상관관계
  - "거래량 증가 → 가격 상승" (Causality) vs. "거래량과 가격의 상관관계 0.7" (Correlation)
- **Whitening (Explainability)**: 블랙박스 모델 해석 필요
- **동질적 기대 vs. 이질적 기대** (Rational vs. Heterogenous Expectations)

**구현 내용:**

1. **정보 플로우 분석**
   ```python
   # 거래량 이상 탐지
   if volume[t] > MA(volume, 20) * 5:
       flag = "Abnormal"

   # Private Information Extraction Score
   score = (volume_buy - volume_sell) / total_volume
   ```

2. **포트폴리오 이론 (CAPM)**
   ```python
   E[R_i] = Alpha + Beta * E[R_m]
   Weight = (Sigma^-1 * a) / (a' * Sigma^-1 * a)
   ```

3. **ARCH/GARCH** (시변 위험)
   ```python
   # Autoregressive Conditional Heteroskedasticity
   sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
   ```

4. **섹터 로테이션**
   ```python
   # GMM (Gaussian Mixture Model)으로 상태 식별
   # Index 영향력 (~80%), Factor 영향력 (~20%)
   ```

5. **시장 미세구조 신호**
   - 거래량 증가 = 기대 불일치
   - 정보 우위자의 선제 행동 (가격 아님, 거래량으로 식별)

6. **거시 정책 효과 (Palantir Ontology)**
   ```
   M 증가 → R 감소 → C, I 증가 → Y 증가 (단기)
   M 증가 → P 증가 → Y 불변 (장기, 통화 중립성)
   ```

**주요 함수:**
- `detect_information_flow(volume, prices)` - 정보 플로우 탐지
- `calculate_private_info_score(buy_volume, sell_volume)` - 정보 비대칭 점수
- `capm_regression(returns, market_returns)` - CAPM Alpha/Beta 추정
- `garch_model(returns, p=1, q=1)` - GARCH 모델링
- `sector_rotation_gmm(sector_returns)` - GMM 기반 섹터 로테이션
- `build_causality_graph(variables, edges)` - Palantir Ontology 그래프

**주요 개념:**

- **Market Neutral**: 가격 상승/하락 중간값 → 은행 이상 초과 수익
- **Passive vs. Active**:
  - Passive: Index 추종
  - Active: Index 제외 Alpha 탐색 (섹터, RWA 등)

- **Real World Assets (RWA)**:
  - 금 채굴권, 희토류 채굴권
  - 블록체인 기반 토큰화 (미래)

- **무한 자산 시대**:
  - HRP (Hierarchical Risk Parity) 필수
  - Sigma^-1이 NaN이 되는 문제 해결

- **정책 효과 분석** (Palantir Ontology):
  - Node: 경제 변수 (M, R, C, I, P, Y)
  - Edge: 영향 관계 (numeric weight)
  - Shock Response Function = 충격반응함수

**키워드:** Causality, CAPM, ARCH/GARCH, 섹터 로테이션, GMM, 정보 플로우, Palantir Ontology, RWA, HRP, Market Neutral

**EIMAS 통합:**
- `lib/autonomous_agent.py`에 Whitening 로직 이미 구현됨
- `lib/causality_graph.py`에 인과관계 Narrative 이미 구현됨 (2026-01-08)
- **강화 필요**:
  - GARCH 모델 추가
  - Palantir Ontology 시각화
  - 정보 플로우 탐지 모듈 신규 생성

**우선순위:** ⭐⭐⭐ (단기 구현, 1-2주)

---

### 5.2 Whitening (Explainability) 강화

**출처:** 금융경제정리.docx

**구현 배경:**
- AI 블랙박스 모델의 설명 가능성 부족
- 규제 요구사항 (EU AI Act, 금융 감독기관)
- 투자자에게 "왜 이 포지션인가?"를 설명해야 함

**구현 내용:**

1. **경제학적 해석 레이어**
   - ML 예측 → 경제학 이론 매핑
   - 예: "LASSO가 M2를 선택" → "통화량 증가가 금리 하락 유도"

2. **인과관계 체인 추적**
   ```
   Input: "Net Liquidity 증가"
   Whitening: "RRP 감소 → 은행 유동성 증가 → Risk-On → 주가 상승"
   ```

3. **팩트체킹 통합**
   - `autonomous_agent.py`의 AutonomousFactChecker와 연동
   - AI 출력 검증 (A-F 등급)

**주요 함수:**
- `whitening_explain(model_output)` - 경제학적 해석 생성
- `trace_causality_chain(event, graph)` - 인과관계 체인 추출
- `fact_check_integration(explanation)` - 팩트체킹 통합

**키워드:** Whitening, Explainability, 인과관계, 팩트체킹, 규제 준수

**EIMAS 통합:**
- `lib/whitening_engine.py` 이미 구현됨 (Phase 7.1)
- **강화 필요**:
  - 인과관계 그래프 시각화
  - 실시간 Whitening (스트리밍 데이터)

**우선순위:** ⭐⭐⭐ (단기 구현, 1주 이내)

---

## 🔗 통합 작업 (Cross-Cutting)

### 6.1 무한 자산 시대 대응 (RWA 확장)

**출처:** Ai 시스템을 만들고 보니 경제학 활용이 많음.docx, 금융경제정리.docx

**배경:**
- Asset이 infinite (토큰화로 인한 자산 급증)
- 기존 시스템 (MVO, CAPM)의 한계 극복 필요

**구현 내용:**
1. **RWA 자산 확장**
   - 금 채굴권, 희토류 채굴권, 부동산 토큰 등
   - `lib/data_loader.py`에 이미 ONDO, PAXG, COIN 추가됨 (v2.1.0)

2. **HRP 적용**
   - 공분산 역행렬 NaN 문제 해결
   - Recursive Bisection으로 무한 자산 처리

3. **Stablecoin 리스크 관리**
   - 다차원 리스크 평가 (신용, 유동성, 규제, 기술)
   - `genius_act_macro.py`에 이미 구현됨 (v2.1.1)

**우선순위:** ⭐⭐ (중기 확장, 지속적 업데이트)

---

### 6.2 Palantir Ontology 기반 인과관계 네트워크 구축

**출처:** Ai 시스템을 만들고 보니 경제학 활용이 많음.docx, 금융경제정리.docx

**배경:**
- 정책 효과 분석 필요 (통화정책, 재정정책)
- 충격 전파 경로 시각화 (M → R → Y)

**구현 내용:**
1. **노드 정의**
   - M (통화량), R (금리), C (소비), I (투자), P (물가), Y (GDP)

2. **엣지 가중치**
   - Granger Causality 테스트로 추정
   - `shock_propagation_graph.py`에 이미 구현됨

3. **충격반응함수 (IRF)**
   - Shock Response Function 계산
   - 시간에 따른 효과 추적 (단기 vs. 장기)

4. **시각화**
   - NetworkX + Graphviz
   - 동적 그래프 (시간 축)

**주요 함수:**
- `define_ontology_nodes()` - 노드 정의
- `granger_causality_edges(data)` - 엣지 가중치 추정
- `impulse_response_function(shock, horizon)` - IRF 계산
- `visualize_ontology_graph(nodes, edges)` - 시각화

**키워드:** Palantir Ontology, 인과관계, Granger Causality, IRF, 충격반응함수

**EIMAS 통합:**
- `lib/shock_propagation_graph.py` 이미 구현됨 (Phase 2.8)
- **추가 작업**:
  - Ontology 시각화 개선
  - 실시간 업데이트

**우선순위:** ⭐⭐ (중기 구현, 2-3개월)

---

## 📈 우선순위 매트릭스

| 작업 | 우선순위 | 예상 기간 | EIMAS 통합 상태 |
|------|---------|----------|---------------|
| **1.1 HRP 고도화** | ⭐⭐⭐ | 1-2주 | 부분 구현 (MST만) |
| **1.2 클러스터링 포트폴리오** | ⭐⭐ | 1개월 | 부분 구현 (GMM만) |
| **2.1 HFT 미세구조** | ⭐⭐⭐⭐ | 1주 | 부분 구현 (VPIN 근사) |
| **3.1 Proof-of-Index** | ⭐⭐ | 1-2개월 | 미구현 |
| **4.1 CNN 패턴 탐지** | ⭐ | 3-6개월 | 미구현 |
| **4.2 LLM 도메인 특화** | ⭐⭐ | 2-3개월 | 부분 구현 (API만) |
| **5.1 인과관계 네트워크** | ⭐⭐⭐ | 1-2주 | 부분 구현 (Narrative만) |
| **5.2 Whitening 강화** | ⭐⭐⭐ | 1주 | 이미 구현됨 |
| **6.1 RWA 확장** | ⭐⭐ | 지속적 | 이미 구현됨 |
| **6.2 Palantir Ontology** | ⭐⭐ | 2-3개월 | 부분 구현 (Graph만) |

---

## 🚀 다음 단계 (Next Steps)

### 즉시 실행 (1주 이내)
1. ✅ **Tick Rule + Roll's Measure** 추가 (`lib/microstructure.py`)
2. ✅ **Whitening 강화** - 인과관계 그래프 시각화
3. ✅ **HRP Systemic Similarity** - `graph_clustered_portfolio.py` 개선

### 단기 (1-2주)
4. **GARCH 모델** 추가 (`lib/regime_analyzer.py`)
5. **정보 플로우 탐지** 신규 모듈 (`lib/information_flow.py`)
6. **HRP 벤치마킹** - CLA, IVP 비교

### 중기 (1-2개월)
7. **Proof-of-Index** 모듈 신규 생성
8. **클러스터링 포트폴리오** - K-means, DBSCAN 통합
9. **LLM Fine-tuning** - 경제학 데이터셋 구축

### 장기 (3-6개월)
10. **CNN 패턴 탐지** - 시계열 → 이미지 변환
11. **Palantir Ontology** 시각화 개선
12. **RWA 자산 확장** - 지속적 업데이트

---

## 📚 참고 문헌 및 키워드 인덱스

### 포트폴리오 최적화
- HRP, Markowitz, MVO, Dendrogram, Recursive Bisection, Seriation
- K-means, Hierarchical Clustering, DBSCAN, MST, DTW
- Silhouette, Davies-Bouldin, Calinski-Harabasz

### 시장 미세구조
- Tick Rule, Kyle's Lambda, Roll's Spread, Amihud, VPIN
- Volume Clock, Silicon Traders, Quote Stuffing, HFT

### 블록체인 & 암호화폐
- Proof-of-Index, Smart Contract, Stablecoin, USDC, USDe
- Tokenization, RWA, ZK-Rollup, Arbitrage

### 경제학 & 금융
- Causality, Whitening, Explainability, CAPM, ARCH/GARCH
- Sector Rotation, GMM, Market Neutral, Palantir Ontology

### AI/ML
- Transformer, Self-Attention, Pre-training, Fine-tuning
- Convolution, CNN, LLM, Multimodal, BERT, GPT

---

**마지막 업데이트:** 2026-01-24 23:00 KST
**작성자:** Claude Code (Explore 에이전트 분석 기반)
**소스:** /home/tj/projects/autoai/eimas/docx/*.docx (총 8개 파일)
