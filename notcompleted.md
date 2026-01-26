# EIMAS 미구현 기능 목록

> todolist.md 기반 미구현/부분구현 항목 정리
> 마지막 업데이트: 2026-01-25

---

## 1. 포트폴리오 최적화 및 자산 배분

### 1.1 HRP 고도화 (부분 구현)
**상태:** 70% → 95% (Systemic Similarity 추가됨)

**미완료 항목:**
- [ ] CLA (Critical Line Algorithm) 벤치마킹 비교
- [ ] Condition number 검증 테스트
- [ ] Correlation 변화에 대한 강건성 테스트
- [ ] HRP 분산이 CLA 대비 42% 낮음 검증

### 1.2 클러스터링 포트폴리오 (부분 구현)
**상태:** 40%

**미완료 항목:**
- [ ] K-means 클러스터링 (`kmeans_clustering()`)
- [ ] Elbow Method 최적 k 선택
- [ ] DBSCAN 이상치 탐지 (`dbscan_outlier_detection()`)
- [ ] DTW (Dynamic Time Warping) 리드-래그 분석
- [ ] LASSO 기반 공분산 추정 (`lasso_covariance()`)
- [ ] Silhouette Score, Davies-Bouldin, Calinski-Harabasz 검증

**우선순위:** ⭐⭐ (중기)

---

## 2. 시장 미세구조 및 거래 메커니즘

### 2.1 HFT 미세구조 (대부분 구현)
**상태:** 40% → 90%

**미완료 항목:**
- [ ] Roll's Measure (Effective Spread) 구현
  ```python
  Spread = 2 * sqrt(max(0, -cov(delta_p, delta_p_lag)))
  ```
- [ ] 실시간 VPIN (현재는 일별 데이터 근사)
- [ ] Round Number Check (100주 vs 99주 거래 비율)
- [ ] Silicon Traders 탐지 로직 고도화

**우선순위:** ⭐⭐⭐ (단기)

---

## 3. 블록체인 기반 인덱스 & 스마트 거래

### 3.1 Proof-of-Index (대부분 구현)
**상태:** 0% → 85%

**미완료 항목:**
- [ ] 실제 Smart Contract 배포 (현재 시뮬레이션)
- [ ] ZK-Rollup Layer 2 연동
- [ ] Chainlink/Pyth 오라클 연동
- [ ] HFT 밀리초 단위 실행
- [ ] 거래소 간 Arbitrage 자동화

**우선순위:** ⭐⭐ (중기)

---

## 4. AI/ML 기술 기초

### 4.1 CNN 패턴 탐지 (미구현)
**상태:** 0%

**미완료 항목:**
- [ ] `lib/cnn_pattern_detector.py` 신규 생성
- [ ] 2D Convolution 연산 구현
- [ ] 시계열 → 2D 이미지 변환 (가격×시간)
- [ ] Edge Detection, Momentum 필터
- [ ] Feature Map 생성
- [ ] 기술적 패턴 탐지 (헤드앤숄더, 삼각 수렴 등)

**우선순위:** ⭐ (장기, 3-6개월)

### 4.2 LLM 도메인 특화 (부분 구현)
**상태:** 30% (API 연동만 완료)

**미완료 항목:**
- [ ] 경제학 Fine-tuning 데이터셋 구축
- [ ] 멀티모달: 차트 이미지 → LLM 해석
- [ ] 경제학 용어 오류 감소
- [ ] Context handling 긴 시퀀스 테스트
- [ ] 편향(Bias) 탐지 로직

**우선순위:** ⭐⭐ (중기, 2-3개월)

---

## 5. 경제학 통합 및 인과관계 분석

### 5.1 인과관계 네트워크 (부분 구현)
**상태:** 65% → 90%

**미완료 항목:**
- [ ] Impulse Response Function (IRF) 계산
- [ ] 시간에 따른 효과 추적 (단기 vs 장기)
- [ ] Palantir Ontology 시각화 개선
- [ ] NetworkX + Graphviz 동적 그래프
- [ ] 충격반응함수 시계열 분석

**우선순위:** ⭐⭐⭐ (단기)

### 5.2 Whitening 강화 (구현 완료)
**상태:** 90%

**미완료 항목:**
- [ ] 인과관계 그래프 시각화 (프론트엔드)
- [ ] 실시간 Whitening (스트리밍 데이터)

**우선순위:** ⭐⭐⭐ (단기)

---

## 6. 통합 작업 (Cross-Cutting)

### 6.1 RWA 확장 (구현 완료)
**상태:** 90%

**미완료 항목:**
- [ ] 금 채굴권 토큰 추가
- [ ] 희토류 채굴권 토큰 추가
- [ ] 부동산 토큰 추가 (REIT 이외)

**우선순위:** ⭐⭐ (지속적 업데이트)

### 6.2 Palantir Ontology 구축 (부분 구현)
**상태:** 50%

**미완료 항목:**
- [ ] 노드 정의 확장 (M, R, C, I, P, Y 외)
- [ ] 엣지 가중치 Granger Causality 자동화
- [ ] IRF (Impulse Response Function) 계산
- [ ] 시각화 대시보드 연동

**우선순위:** ⭐⭐ (중기, 2-3개월)

---

## 7. 대시보드 및 시각화 (v2.1.2)

### 7.1 프론트엔드 시각화 (부분 구현)
**상태:** 40%

**미완료 항목:**
- [ ] 포트폴리오 가중치 파이 차트 (Recharts)
- [ ] 상관관계 히트맵 (24개 자산)
- [ ] 리스크 점수 타임라인 차트
- [ ] GMM 확률 분포 차트
- [ ] 섹터 로테이션 바 차트
- [ ] MST (Minimum Spanning Tree) 네트워크 그래프
- [ ] Systemic Similarity D̄ matrix 히트맵

### 7.2 실시간 기능 (미구현)
**상태:** 0%

**미완료 항목:**
- [ ] WebSocket 연결 (`useWebSocket` hook)
- [ ] Phase 4 (--realtime) 결과 실시간 업데이트
- [ ] 실시간 차트 애니메이션
- [ ] BinanceStreamer 데이터 시각화

**우선순위:** ⭐⭐⭐ (단기, 4-5시간)

---

## 8. 테스트 및 검증

### 8.1 통합 테스트 (부분 구현)
**상태:** 60%

**미완료 항목:**
- [ ] `tests/test_full_workflow.py` 완성
- [ ] DataManager → MetaOrchestrator → 보고서 생성 E2E 테스트
- [ ] CI/CD 파이프라인 연동
- [ ] 성능 벤치마크 (실행 시간, 메모리)

---

## 우선순위 매트릭스 (미완료 항목)

| 작업 | 우선순위 | 예상 기간 | 현재 구현도 |
|------|---------|----------|------------|
| Roll's Measure | ⭐⭐⭐ | 1일 | 0% |
| DTW 시계열 유사도 | ⭐⭐ | 3-5일 | 0% |
| IRF 충격반응함수 | ⭐⭐⭐ | 1주 | 0% |
| 프론트엔드 차트 | ⭐⭐⭐ | 2-3일 | 40% |
| WebSocket 실시간 | ⭐⭐⭐ | 4-5시간 | 0% |
| CNN 패턴 탐지 | ⭐ | 3-6개월 | 0% |
| LLM Fine-tuning | ⭐⭐ | 2-3개월 | 30% |
| Smart Contract 배포 | ⭐⭐ | 1개월 | 0% |

---

## 다음 작업 권장 순서

### 즉시 실행 (1-3일)
1. **Roll's Measure** 추가 (`lib/microstructure.py`)
2. **프론트엔드 파이 차트** 추가 (`frontend/components/`)
3. **WebSocket 연결** 구현

### 단기 (1-2주)
4. **DTW 시계열 유사도** 신규 모듈
5. **IRF 충격반응함수** 계산
6. **통합 테스트** 완성

### 중기 (1-2개월)
7. **K-means/DBSCAN 클러스터링**
8. **Smart Contract 배포**
9. **LLM Fine-tuning 데이터셋**

### 장기 (3-6개월)
10. **CNN 패턴 탐지**
11. **Palantir Ontology 고도화**

---

**마지막 업데이트:** 2026-01-25 05:00 KST
**참조:** todolist.md, COMPLETION_REPORT.md, INTEGRATION_REPORT.md
