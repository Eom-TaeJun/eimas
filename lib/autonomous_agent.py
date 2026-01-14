"""
Autonomous Fact-Checking Agent
==============================

터미널 기반 자율 팩트체킹 에이전트:

1. AI 결과에 대한 실시간 검증
2. 수치 데이터 크로스체크
3. 출처 신뢰도 평가
4. 자동 수정 제안

API 사용:
- Perplexity: 실시간 검색 및 사실 확인
- 외부 데이터소스: FRED, Yahoo Finance, CoinGecko 등
"""

import asyncio
import json
import os
import re
import httpx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod


class VerificationStatus(Enum):
    """검증 상태"""
    VERIFIED = "verified"           # 확인됨
    PARTIALLY_VERIFIED = "partial"  # 부분 확인
    UNVERIFIED = "unverified"       # 미확인
    CONTRADICTED = "contradicted"   # 반박됨
    OUTDATED = "outdated"           # 오래된 정보
    UNABLE_TO_VERIFY = "unable"     # 검증 불가


class ClaimType(Enum):
    """주장 타입"""
    NUMERIC = "numeric"             # 수치 데이터
    TREND = "trend"                 # 추세 주장
    CAUSAL = "causal"               # 인과관계
    PREDICTION = "prediction"       # 예측
    FACT = "fact"                   # 사실 진술
    OPINION = "opinion"             # 의견


@dataclass
class Claim:
    """검증할 주장"""
    text: str
    claim_type: ClaimType
    source: str = "AI_GENERATED"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class VerificationResult:
    """검증 결과"""
    claim: Claim
    status: VerificationStatus
    confidence: float  # 0-1
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    correction: Optional[str] = None
    details: Dict = field(default_factory=dict)


@dataclass
class AgentState:
    """에이전트 상태"""
    current_task: Optional[str] = None
    claims_verified: int = 0
    claims_contradicted: int = 0
    is_running: bool = False
    last_activity: datetime = field(default_factory=datetime.now)


# =============================================================================
# 검증 도구 (Tools)
# =============================================================================

class VerificationTool(ABC):
    """검증 도구 추상 클래스"""

    @abstractmethod
    async def verify(self, claim: Claim) -> VerificationResult:
        """주장 검증"""
        pass


class NumericVerifier(VerificationTool):
    """수치 데이터 검증"""

    # 알려진 수치 데이터 (캐시/하드코딩)
    KNOWN_VALUES = {
        "fed_funds_rate": {"value": 5.25, "unit": "%", "date": "2024-01"},
        "sp500_level": {"value": 4800, "unit": "points", "date": "2024-01"},
        "bitcoin_ath": {"value": 73000, "unit": "USD", "date": "2024-03"},
        "us_debt": {"value": 34, "unit": "trillion USD", "date": "2024-01"},
        "m2_supply": {"value": 20.8, "unit": "trillion USD", "date": "2024-01"},
        "cpi_yoy": {"value": 3.1, "unit": "%", "date": "2024-01"},
    }

    def extract_numbers(self, text: str) -> List[Tuple[float, str]]:
        """텍스트에서 숫자 추출"""
        patterns = [
            r'(\d+\.?\d*)\s*(%|percent|퍼센트)',
            r'(\d+\.?\d*)\s*(trillion|조|T)',
            r'(\d+\.?\d*)\s*(billion|억|B)',
            r'(\d+\.?\d*)\s*(million|백만|M)',
            r'\$(\d+,?\d*\.?\d*)',
            r'(\d+,?\d*\.?\d*)\s*(달러|USD|원|KRW)',
        ]

        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    num = float(match[0].replace(',', ''))
                    unit = match[1] if len(match) > 1 else ""
                    numbers.append((num, unit))
                except:
                    pass

        return numbers

    async def verify(self, claim: Claim) -> VerificationResult:
        """수치 검증"""
        text = claim.text.lower()
        evidence = []
        status = VerificationStatus.UNABLE_TO_VERIFY
        confidence = 0.3

        # 키워드 매칭
        for key, data in self.KNOWN_VALUES.items():
            key_variants = key.replace('_', ' ').split()
            if any(v in text for v in key_variants):
                numbers = self.extract_numbers(claim.text)
                for num, unit in numbers:
                    # 오차 범위 체크 (10% 허용)
                    if abs(num - data["value"]) / data["value"] < 0.1:
                        status = VerificationStatus.VERIFIED
                        confidence = 0.8
                        evidence.append(f"{key}: {data['value']}{data['unit']} (as of {data['date']})")
                    elif abs(num - data["value"]) / data["value"] < 0.3:
                        status = VerificationStatus.PARTIALLY_VERIFIED
                        confidence = 0.5
                        evidence.append(f"근접값 발견: {key}={data['value']}, 주장값={num}")
                    else:
                        status = VerificationStatus.CONTRADICTED
                        confidence = 0.7
                        evidence.append(f"불일치: {key} 실제값={data['value']}, 주장값={num}")

        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence,
            sources=["Internal Database"],
            details={"extracted_numbers": self.extract_numbers(claim.text)}
        )


class TrendVerifier(VerificationTool):
    """추세 주장 검증"""

    TREND_KEYWORDS = {
        "상승": ["상승", "증가", "오름", "rising", "increasing", "up", "higher"],
        "하락": ["하락", "감소", "내림", "falling", "decreasing", "down", "lower"],
        "횡보": ["횡보", "보합", "flat", "sideways", "unchanged"],
    }

    async def verify(self, claim: Claim) -> VerificationResult:
        """추세 검증 (실제 구현 시 외부 API 연동)"""
        text = claim.text.lower()

        detected_trend = None
        for trend, keywords in self.TREND_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                detected_trend = trend
                break

        if detected_trend:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.PARTIALLY_VERIFIED,
                confidence=0.5,
                evidence=[f"추세 키워드 감지: {detected_trend}"],
                sources=["Keyword Analysis"],
                details={"detected_trend": detected_trend}
            )

        return VerificationResult(
            claim=claim,
            status=VerificationStatus.UNABLE_TO_VERIFY,
            confidence=0.3,
            evidence=["추세 키워드 미발견"],
            sources=[]
        )


class CausalVerifier(VerificationTool):
    """인과관계 검증"""

    # 알려진 인과관계 (경제학적)
    KNOWN_CAUSAL = {
        ("금리", "주가"): {"direction": "inverse", "confidence": 0.7},
        ("금리", "채권"): {"direction": "inverse", "confidence": 0.9},
        ("유동성", "주가"): {"direction": "positive", "confidence": 0.75},
        ("인플레이션", "금리"): {"direction": "positive", "confidence": 0.8},
        ("달러", "금"): {"direction": "inverse", "confidence": 0.65},
        ("달러", "신흥국"): {"direction": "inverse", "confidence": 0.7},
        ("vix", "주가"): {"direction": "inverse", "confidence": 0.85},
    }

    CAUSAL_PATTERNS = [
        r'(.+)(?:이|가)\s+(.+)(?:에|를|을)\s+(?:영향|유발|초래)',
        r'(.+)\s*(?:→|->|때문에|으로 인해)\s*(.+)',
        r'(.+)\s+(?:leads to|causes|results in)\s+(.+)',
    ]

    def extract_causal_pair(self, text: str) -> Optional[Tuple[str, str]]:
        """인과관계 쌍 추출"""
        for pattern in self.CAUSAL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return (match.group(1).strip(), match.group(2).strip())
        return None

    async def verify(self, claim: Claim) -> VerificationResult:
        """인과관계 검증"""
        pair = self.extract_causal_pair(claim.text)

        if not pair:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNABLE_TO_VERIFY,
                confidence=0.3,
                evidence=["인과관계 패턴 미감지"],
                sources=[]
            )

        cause, effect = pair

        # 알려진 인과관계와 매칭
        for (known_cause, known_effect), data in self.KNOWN_CAUSAL.items():
            if known_cause in cause.lower() or known_effect in effect.lower():
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.PARTIALLY_VERIFIED,
                    confidence=data["confidence"],
                    evidence=[f"알려진 인과관계: {known_cause} → {known_effect} ({data['direction']})"],
                    sources=["Economic Theory"],
                    details={"detected_pair": pair, "known_relation": data}
                )

        return VerificationResult(
            claim=claim,
            status=VerificationStatus.UNVERIFIED,
            confidence=0.4,
            evidence=[f"감지된 인과관계: {cause} → {effect} (미검증)"],
            sources=["Pattern Extraction"],
            details={"detected_pair": pair}
        )


class PerplexityVerifier(VerificationTool):
    """Perplexity API 기반 실시간 검증"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.base_url = "https://api.perplexity.ai/chat/completions"

    async def verify(self, claim: Claim) -> VerificationResult:
        """Perplexity로 실시간 검증"""
        if not self.api_key:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNABLE_TO_VERIFY,
                confidence=0.0,
                evidence=["Perplexity API 키 없음"],
                sources=[]
            )

        prompt = f"""다음 주장의 사실 여부를 검증해주세요:

"{claim.text}"

1. 이 주장이 사실인지 아닌지 판단
2. 근거 제시 (출처 포함)
3. 수정이 필요하면 올바른 정보 제공

JSON 형식으로 응답:
{{
    "is_accurate": true/false/partial,
    "confidence": 0.0-1.0,
    "evidence": ["근거1", "근거2"],
    "sources": ["출처1", "출처2"],
    "correction": "수정 내용 (필요시)"
}}"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-sonar-small-128k-online",
                        "messages": [
                            {"role": "system", "content": "You are a fact-checking assistant. Verify claims with current data."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]

                    # JSON 파싱 시도
                    try:
                        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                        if json_match:
                            parsed = json.loads(json_match.group())

                            accuracy = parsed.get("is_accurate", "partial")
                            if accuracy == True or accuracy == "true":
                                status = VerificationStatus.VERIFIED
                            elif accuracy == False or accuracy == "false":
                                status = VerificationStatus.CONTRADICTED
                            else:
                                status = VerificationStatus.PARTIALLY_VERIFIED

                            return VerificationResult(
                                claim=claim,
                                status=status,
                                confidence=parsed.get("confidence", 0.6),
                                evidence=parsed.get("evidence", []),
                                sources=parsed.get("sources", ["Perplexity AI"]),
                                correction=parsed.get("correction"),
                                details={"raw_response": content[:500]}
                            )
                    except json.JSONDecodeError:
                        pass

                    # JSON 파싱 실패 시 텍스트 분석
                    return VerificationResult(
                        claim=claim,
                        status=VerificationStatus.PARTIALLY_VERIFIED,
                        confidence=0.5,
                        evidence=[content[:300]],
                        sources=["Perplexity AI"]
                    )

                else:
                    return VerificationResult(
                        claim=claim,
                        status=VerificationStatus.UNABLE_TO_VERIFY,
                        confidence=0.0,
                        evidence=[f"API Error: {response.status_code}"],
                        sources=[]
                    )

        except Exception as e:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNABLE_TO_VERIFY,
                confidence=0.0,
                evidence=[f"Error: {str(e)}"],
                sources=[]
            )


# =============================================================================
# 자율 에이전트
# =============================================================================

class AutonomousFactChecker:
    """자율 팩트체킹 에이전트"""

    def __init__(
        self,
        use_perplexity: bool = True,
        verbose: bool = True
    ):
        self.state = AgentState()
        self.verbose = verbose
        self.verification_history: List[VerificationResult] = []

        # 검증 도구 초기화
        self.tools = {
            ClaimType.NUMERIC: NumericVerifier(),
            ClaimType.TREND: TrendVerifier(),
            ClaimType.CAUSAL: CausalVerifier(),
        }

        if use_perplexity:
            self.perplexity = PerplexityVerifier()
        else:
            self.perplexity = None

    def log(self, message: str):
        """로깅"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def classify_claim(self, text: str) -> ClaimType:
        """주장 유형 분류"""
        text_lower = text.lower()

        # 수치 패턴
        if re.search(r'\d+\.?\d*\s*(%|조|억|trillion|billion)', text_lower):
            return ClaimType.NUMERIC

        # 인과관계 패턴
        if any(kw in text_lower for kw in ['때문', '영향', '→', 'leads to', 'causes']):
            return ClaimType.CAUSAL

        # 추세 패턴
        if any(kw in text_lower for kw in ['상승', '하락', '증가', '감소', 'rising', 'falling']):
            return ClaimType.TREND

        # 예측 패턴
        if any(kw in text_lower for kw in ['전망', '예상', '예측', 'forecast', 'expect']):
            return ClaimType.PREDICTION

        # 기본값
        return ClaimType.FACT

    def extract_claims(self, text: str) -> List[Claim]:
        """텍스트에서 검증 가능한 주장 추출"""
        claims = []

        # 문장 분리
        sentences = re.split(r'[.!?。]\s*', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # 주장 유형 분류
            claim_type = self.classify_claim(sentence)

            # 검증 가능한 유형만 추출
            if claim_type in [ClaimType.NUMERIC, ClaimType.TREND, ClaimType.CAUSAL]:
                claims.append(Claim(
                    text=sentence,
                    claim_type=claim_type,
                    source="TEXT_EXTRACTION"
                ))

        return claims

    async def verify_claim(self, claim: Claim) -> VerificationResult:
        """단일 주장 검증"""
        self.state.current_task = f"Verifying: {claim.text[:50]}..."
        self.log(f"🔍 검증 중: {claim.text[:60]}...")

        # 1. 전문 도구로 검증
        tool = self.tools.get(claim.claim_type)
        if tool:
            result = await tool.verify(claim)
            if result.status == VerificationStatus.VERIFIED:
                self.state.claims_verified += 1
                self.log(f"✅ 검증됨 (신뢰도: {result.confidence:.0%})")
                self.verification_history.append(result)
                return result

        # 2. Perplexity로 추가 검증
        if self.perplexity and result.status in [
            VerificationStatus.UNVERIFIED,
            VerificationStatus.PARTIALLY_VERIFIED,
            VerificationStatus.UNABLE_TO_VERIFY
        ]:
            self.log("🌐 Perplexity로 실시간 검증 중...")
            perplexity_result = await self.perplexity.verify(claim)

            # 결과 병합
            if perplexity_result.status == VerificationStatus.VERIFIED:
                result = perplexity_result
            elif perplexity_result.confidence > result.confidence:
                result = perplexity_result

        # 상태 업데이트
        if result.status == VerificationStatus.CONTRADICTED:
            self.state.claims_contradicted += 1
            self.log(f"❌ 반박됨: {result.correction or result.evidence[0] if result.evidence else 'N/A'}")
        elif result.status == VerificationStatus.VERIFIED:
            self.state.claims_verified += 1
            self.log(f"✅ 검증됨")
        else:
            self.log(f"⚠️ {result.status.value}")

        self.verification_history.append(result)
        return result

    async def verify_document(
        self,
        document: str,
        max_claims: int = 10
    ) -> Dict:
        """문서 전체 검증"""
        self.state.is_running = True
        self.log("=" * 50)
        self.log("📋 문서 검증 시작")
        self.log("=" * 50)

        # 주장 추출
        claims = self.extract_claims(document)
        self.log(f"📌 검증 대상: {len(claims)}개 주장 발견")

        if len(claims) > max_claims:
            claims = claims[:max_claims]
            self.log(f"⚠️ 최대 {max_claims}개로 제한")

        # 검증 실행
        results = []
        for i, claim in enumerate(claims, 1):
            self.log(f"\n[{i}/{len(claims)}] {claim.claim_type.value}")
            result = await self.verify_claim(claim)
            results.append(result)

        # 요약 생성
        summary = self._generate_summary(results)

        self.state.is_running = False
        self.state.current_task = None

        return {
            "total_claims": len(claims),
            "results": [self._result_to_dict(r) for r in results],
            "summary": summary
        }

    def _result_to_dict(self, result: VerificationResult) -> Dict:
        """결과를 딕셔너리로 변환"""
        return {
            "claim": result.claim.text,
            "type": result.claim.claim_type.value,
            "status": result.status.value,
            "confidence": f"{result.confidence:.0%}",
            "evidence": result.evidence[:3],
            "sources": result.sources,
            "correction": result.correction
        }

    def _generate_summary(self, results: List[VerificationResult]) -> Dict:
        """검증 요약 생성"""
        total = len(results)
        verified = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
        partial = sum(1 for r in results if r.status == VerificationStatus.PARTIALLY_VERIFIED)
        contradicted = sum(1 for r in results if r.status == VerificationStatus.CONTRADICTED)
        unable = sum(1 for r in results if r.status == VerificationStatus.UNABLE_TO_VERIFY)

        # 신뢰도 점수
        avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0

        # 수정 필요 항목
        corrections_needed = [r for r in results if r.correction]

        # 등급 결정
        if verified / total >= 0.8 and contradicted == 0:
            grade = "A"
            grade_desc = "높은 신뢰도"
        elif verified / total >= 0.6 and contradicted / total < 0.1:
            grade = "B"
            grade_desc = "양호"
        elif contradicted / total >= 0.3:
            grade = "D"
            grade_desc = "신뢰 주의"
        else:
            grade = "C"
            grade_desc = "추가 검증 필요"

        return {
            "total_claims": total,
            "verified": verified,
            "partially_verified": partial,
            "contradicted": contradicted,
            "unable_to_verify": unable,
            "average_confidence": f"{avg_confidence:.0%}",
            "grade": grade,
            "grade_description": grade_desc,
            "corrections_needed": len(corrections_needed),
            "key_issues": [
                {"claim": r.claim.text[:100], "correction": r.correction}
                for r in corrections_needed[:5]
            ]
        }

    async def interactive_mode(self):
        """대화형 모드"""
        print("\n" + "=" * 60)
        print("🤖 Autonomous Fact-Checker Agent")
        print("=" * 60)
        print("명령어: verify <텍스트>, check <주장>, status, exit")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    print("👋 종료합니다.")
                    break

                elif user_input.lower() == "status":
                    print(f"\n📊 상태:")
                    print(f"   검증된 주장: {self.state.claims_verified}")
                    print(f"   반박된 주장: {self.state.claims_contradicted}")
                    print(f"   실행 중: {self.state.is_running}")

                elif user_input.lower().startswith("verify "):
                    text = user_input[7:]
                    result = await self.verify_document(text)
                    print(f"\n📋 결과 요약:")
                    for key, value in result["summary"].items():
                        print(f"   {key}: {value}")

                elif user_input.lower().startswith("check "):
                    text = user_input[6:]
                    claim = Claim(
                        text=text,
                        claim_type=self.classify_claim(text)
                    )
                    result = await self.verify_claim(claim)
                    print(f"\n결과: {result.status.value}")
                    print(f"신뢰도: {result.confidence:.0%}")
                    if result.evidence:
                        print(f"근거: {result.evidence[0]}")
                    if result.correction:
                        print(f"수정: {result.correction}")

                else:
                    print("알 수 없는 명령어. 'verify', 'check', 'status', 'exit' 사용")

            except KeyboardInterrupt:
                print("\n👋 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")


# =============================================================================
# AI 결과 검증 특화
# =============================================================================

class AIOutputVerifier:
    """AI 출력 결과 전문 검증기"""

    def __init__(self):
        self.fact_checker = AutonomousFactChecker(verbose=False)

    async def verify_json_output(self, json_data: Dict) -> Dict:
        """JSON 형태의 AI 출력 검증"""
        claims = []

        # JSON에서 검증 가능한 필드 추출
        self._extract_from_json(json_data, claims)

        # 검증 실행
        results = []
        for claim in claims[:10]:  # 최대 10개
            result = await self.fact_checker.verify_claim(claim)
            results.append(result)

        return {
            "total_claims": len(claims),
            "verified": sum(1 for r in results if r.status == VerificationStatus.VERIFIED),
            "issues": [
                self.fact_checker._result_to_dict(r)
                for r in results
                if r.status in [VerificationStatus.CONTRADICTED, VerificationStatus.PARTIALLY_VERIFIED]
            ]
        }

    def _extract_from_json(self, data: Any, claims: List[Claim], path: str = ""):
        """JSON에서 검증 가능한 데이터 추출"""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._extract_from_json(value, claims, new_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._extract_from_json(item, claims, f"{path}[{i}]")

        elif isinstance(data, str):
            # 문자열에서 수치 주장 찾기
            if re.search(r'\d+\.?\d*\s*(%|조|억)', str(data)):
                claims.append(Claim(
                    text=str(data),
                    claim_type=ClaimType.NUMERIC,
                    source=f"JSON:{path}",
                    metadata={"json_path": path}
                ))


# =============================================================================
# 테스트
# =============================================================================

async def main():
    print("=" * 60)
    print("Autonomous Fact-Checker Agent Test")
    print("=" * 60)

    agent = AutonomousFactChecker(use_perplexity=False, verbose=True)

    # 테스트 문서
    test_doc = """
    현재 미국 연방기금금리는 약 5.25%입니다.
    금리 인상으로 인해 주식시장이 하락하고 있습니다.
    비트코인은 2024년 3월 73,000달러로 사상 최고가를 기록했습니다.
    미국 국가부채는 34조 달러를 넘어섰습니다.
    유동성 증가가 자산가격 상승을 유발합니다.
    현재 M2 통화량은 약 20조 달러입니다.
    달러 강세로 인해 신흥국 시장이 압박받고 있습니다.
    """

    print("\n📝 테스트 문서:")
    print(test_doc)

    # 검증 실행
    result = await agent.verify_document(test_doc)

    print("\n" + "=" * 60)
    print("📋 검증 결과 요약")
    print("=" * 60)

    summary = result["summary"]
    print(f"\n총 주장: {summary['total_claims']}")
    print(f"검증됨: {summary['verified']}")
    print(f"부분 검증: {summary['partially_verified']}")
    print(f"반박됨: {summary['contradicted']}")
    print(f"검증 불가: {summary['unable_to_verify']}")
    print(f"\n평균 신뢰도: {summary['average_confidence']}")
    print(f"등급: {summary['grade']} ({summary['grade_description']})")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
