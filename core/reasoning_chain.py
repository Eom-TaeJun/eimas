from typing import List, Dict, Any
from datetime import datetime
import json

class ReasoningChain:
    """
    AI 의사결정 과정 추적

    목적:
    - 최종 권고가 어떤 근거로 도출되었는지 명시
    - 각 에이전트의 기여도 추적
    - 감사(Audit) 가능한 AI 시스템 구현
    """

    def __init__(self):
        self.chain = []

    def add_step(
        self,
        agent: str,
        input_summary: str,
        output_summary: str,
        confidence: float,
        key_factors: List[str]
    ):
        """추론 단계 추가"""
        self.chain.append({
            'step': len(self.chain) + 1,
            'agent': agent,
            'input': input_summary,
            'output': output_summary,
            'confidence': confidence,
            'key_factors': key_factors,
            'timestamp': datetime.now().isoformat()
        })

    def get_summary(self) -> str:
        """추론 과정 요약"""
        summary_lines = ["## Reasoning Chain\n"]

        for step in self.chain:
            summary_lines.append(f"### Step {step['step']}: {step['agent']}")
            summary_lines.append(f"- **Input:** {step['input']}")
            summary_lines.append(f"- **Output:** {step['output']}")
            summary_lines.append(f"- **Confidence:** {step['confidence']}")
            summary_lines.append(f"- **Key Factors:**")
            for factor in step['key_factors']:
                summary_lines.append(f"  - {factor}")
            summary_lines.append("")

        return "\n".join(summary_lines)

    def to_dict(self) -> List[Dict]:
        """JSON 직렬화용"""
        return self.chain
