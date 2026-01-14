#!/usr/bin/env python3
"""
Signal-Action Mapping Framework
================================
신호에서 액션으로의 매핑 프레임워크

ADDITIONAL_CONSIDERATIONS.md 기반 구현:
- 신호 강도별 액션 결정
- 사용자 맞춤형 리스크 프로파일
- 신호 충돌 해결
- 액션 로그 및 피드백

경제학적 배경:
- 신호의 통계적 유의성 → 액션의 크기
- 리스크 관리 신호 > 수익 신호 (비대칭 손실)
- 의심스러우면 하지 않는다 (No action is an action)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Literal
from datetime import datetime
from enum import Enum
import uuid


# ============================================================================
# Enums
# ============================================================================

class SignalStrength(str, Enum):
    """
    신호 강도 분류

    ADDITIONAL_CONSIDERATIONS.md 기준:
    - WEAK (50-60%): 관망
    - MODERATE (60-70%): 소규모 진입
    - STRONG (70-80%): 중간 규모 진입
    - VERY_STRONG (80-90%): 대규모 진입
    - EXTREME (>90%): 최대 진입 (주의: 과신 경계)
    """
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    EXTREME = "extreme"


class ActionType(str, Enum):
    """액션 유형"""
    NEW_POSITION = "new_position"       # 신규 진입
    ADD_POSITION = "add_position"       # 포지션 확대
    REDUCE_POSITION = "reduce_position" # 포지션 축소
    CLOSE_POSITION = "close_position"   # 전량 청산
    HEDGE = "hedge"                     # 헤징
    NO_ACTION = "no_action"             # 관망


class PositionDirection(str, Enum):
    """포지션 방향"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class RiskProfileType(str, Enum):
    """리스크 프로파일 유형"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class MarketRegime(str, Enum):
    """시장 레짐 (동적 리스크 조절용)"""
    CALM = "calm"           # VIX < 15
    NORMAL = "normal"       # 15 <= VIX < 25
    ELEVATED = "elevated"   # 25 <= VIX < 35
    CRISIS = "crisis"       # VIX >= 35


class ConflictType(str, Enum):
    """신호 충돌 유형"""
    DIRECTIONAL = "directional"     # 방향성 충돌 (상승 vs 하락)
    MAGNITUDE = "magnitude"         # 강도 충돌 (신뢰도 차이 > 0.3)
    TEMPORAL = "temporal"           # 시점 충돌 (단기 vs 장기)
    MODEL_VALUE = "model_value"     # 모델 vs 내재가치 충돌


# ============================================================================
# Risk Profile
# ============================================================================

@dataclass
class RiskProfile:
    """
    사용자 맞춤형 리스크 프로파일

    ADDITIONAL_CONSIDERATIONS.md 2.2절 기반:
    - Conservative: 자본 보존 우선
    - Moderate: 수익과 위험의 균형
    - Aggressive: 수익 극대화
    """
    profile_type: RiskProfileType
    max_position_size: float          # 최대 포지션 크기 (0-1)
    max_single_asset: float           # 단일 자산 최대 비중 (0-1)
    stop_loss: float                  # Stop-loss 비율 (음수, 예: -0.07)
    take_profit: float                # Take-profit 비율 (양수, 예: 0.15)
    max_leverage: float               # 최대 레버리지 (1.0 = 없음)
    min_cash_ratio: float             # 현금 최소 비중 (0-1)
    rebalance_frequency: str          # 리밸런싱 주기 ("daily", "weekly", "monthly")
    signal_threshold: float           # 진입 신호 임계값 (0-1)

    # 동적 조절 설정
    auto_risk_adjust: bool = True     # VIX 기반 자동 조절
    weekend_reduce: bool = False      # 주말 포지션 자동 축소
    earnings_risk_up: bool = False    # 실적 시즌 리스크 강화
    vix_auto_hedge: bool = False      # VIX 급등 시 자동 헤징

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['profile_type'] = self.profile_type.value
        return data

    @classmethod
    def conservative(cls) -> 'RiskProfile':
        """보수적 프로파일"""
        return cls(
            profile_type=RiskProfileType.CONSERVATIVE,
            max_position_size=0.50,
            max_single_asset=0.10,
            stop_loss=-0.03,
            take_profit=0.05,
            max_leverage=1.0,
            min_cash_ratio=0.30,
            rebalance_frequency="monthly",
            signal_threshold=0.80,
            auto_risk_adjust=True,
            weekend_reduce=True,
            earnings_risk_up=True,
            vix_auto_hedge=True
        )

    @classmethod
    def moderate(cls) -> 'RiskProfile':
        """중립적 프로파일"""
        return cls(
            profile_type=RiskProfileType.MODERATE,
            max_position_size=0.80,
            max_single_asset=0.20,
            stop_loss=-0.07,
            take_profit=0.15,
            max_leverage=1.5,
            min_cash_ratio=0.10,
            rebalance_frequency="weekly",
            signal_threshold=0.65,
            auto_risk_adjust=True,
            weekend_reduce=False,
            earnings_risk_up=True,
            vix_auto_hedge=True
        )

    @classmethod
    def aggressive(cls) -> 'RiskProfile':
        """공격적 프로파일"""
        return cls(
            profile_type=RiskProfileType.AGGRESSIVE,
            max_position_size=1.00,
            max_single_asset=0.40,
            stop_loss=-0.15,
            take_profit=0.30,
            max_leverage=3.0,
            min_cash_ratio=0.0,
            rebalance_frequency="daily",
            signal_threshold=0.55,
            auto_risk_adjust=True,
            weekend_reduce=False,
            earnings_risk_up=False,
            vix_auto_hedge=False
        )


# ============================================================================
# Enhanced Signal
# ============================================================================

@dataclass
class EnhancedSignal:
    """
    확장된 신호 스키마

    기존 Signal 클래스를 확장하여 신호-액션 매핑에 필요한 추가 정보 포함
    """
    signal_id: str                    # 고유 ID
    type: str                         # "statistical", "theoretical", "leading"
    ticker: str                       # 티커
    name: str                         # 자산명
    indicator: str                    # 지표명
    value: float                      # 현재 값
    threshold: float                  # 임계값
    z_score: float                    # Z-score
    level: str                        # "NORMAL", "WARNING", "ALERT", "CRITICAL"
    description: str                  # 설명

    # 확장 필드
    confidence: float = 0.5           # 신뢰도 (0-1)
    direction: PositionDirection = PositionDirection.NEUTRAL  # 방향
    horizon: str = "short"            # "ultra_short", "short", "long"
    source: str = "market_anomaly"    # 신호 출처
    intrinsic_value_aligned: Optional[bool] = None  # 내재가치와 일치 여부
    regime_aligned: Optional[bool] = None           # 현재 레짐과 일치 여부

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = f"SIG-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"

    def get_strength(self) -> SignalStrength:
        """신뢰도 기반 신호 강도 반환"""
        if self.confidence >= 0.90:
            return SignalStrength.EXTREME
        elif self.confidence >= 0.80:
            return SignalStrength.VERY_STRONG
        elif self.confidence >= 0.70:
            return SignalStrength.STRONG
        elif self.confidence >= 0.60:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['direction'] = self.direction.value
        return data

    @classmethod
    def from_legacy_signal(cls, signal: Any, confidence: float = 0.5) -> 'EnhancedSignal':
        """기존 Signal 객체에서 변환"""
        # level에서 direction 추론
        direction = PositionDirection.NEUTRAL
        if hasattr(signal, 'z_score'):
            if signal.z_score > 0:
                direction = PositionDirection.LONG
            elif signal.z_score < 0:
                direction = PositionDirection.SHORT

        return cls(
            signal_id="",
            type=getattr(signal, 'type', 'unknown'),
            ticker=getattr(signal, 'ticker', 'UNKNOWN'),
            name=getattr(signal, 'name', 'Unknown'),
            indicator=getattr(signal, 'indicator', 'unknown'),
            value=getattr(signal, 'value', 0.0),
            threshold=getattr(signal, 'threshold', 0.0),
            z_score=getattr(signal, 'z_score', 0.0),
            level=getattr(signal, 'level', 'NORMAL'),
            description=getattr(signal, 'description', ''),
            confidence=confidence,
            direction=direction
        )


# ============================================================================
# Action
# ============================================================================

@dataclass
class Action:
    """
    실행할 액션

    ADDITIONAL_CONSIDERATIONS.md 4.4절 기반:
    - 신규 진입: 분할 진입 권장 (3회에 나눠서)
    - 포지션 확대: 기존 포지션이 수익일 때만
    - 포지션 축소: 부분 익절, 트레일링 Stop
    - 전량 청산: Stop-loss 도달 시 즉시
    - 헤징: 불확실성 증가, 주말 전, 이벤트 전
    """
    action_id: str
    action_type: ActionType
    ticker: str
    direction: PositionDirection

    # 포지션 크기
    position_size: float              # 포지션 크기 (0-1, 포트폴리오 대비)
    entry_price: Optional[float] = None

    # 리스크 관리
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None

    # 실행 전략
    execution_strategy: str = "immediate"  # "immediate", "split", "limit", "wait"
    split_count: int = 1              # 분할 매매 횟수
    limit_price: Optional[float] = None

    # 신호 근거
    signal_ids: List[str] = field(default_factory=list)
    total_confidence: float = 0.5
    risk_reward_ratio: float = 1.0

    # 메타데이터
    rationale: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: Optional[str] = None  # 만료 시간

    def __post_init__(self):
        if not self.action_id:
            self.action_id = f"ACT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['action_type'] = self.action_type.value
        data['direction'] = self.direction.value
        return data


# ============================================================================
# Action Log
# ============================================================================

@dataclass
class ActionLog:
    """
    액션 실행 로그 및 피드백

    ADDITIONAL_CONSIDERATIONS.md 4.5절 기반:
    - 신호 근거 기록
    - 실행 상세 기록
    - 후속 추적
    - 사후 분석
    """
    log_id: str
    action: Action

    # 실행 정보
    executed_at: Optional[str] = None
    executed_price: Optional[float] = None
    executed_size: float = 0.0
    execution_status: str = "pending"  # "pending", "executed", "partial", "cancelled", "failed"

    # 후속 추적
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    signal_still_valid: bool = True
    expected_holding_period: str = ""  # "1-2days", "1-2weeks", "1-3months"

    # 사후 분석 (청산 후)
    closed_at: Optional[str] = None
    closed_price: Optional[float] = None
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    signal_accuracy: Optional[bool] = None  # 신호가 맞았는지
    lessons_learned: str = ""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.log_id:
            self.log_id = f"LOG-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"

    def update_tracking(self, current_price: float):
        """현재가 기반 추적 업데이트"""
        self.current_price = current_price
        if self.executed_price and self.executed_price > 0:
            if self.action.direction == PositionDirection.LONG:
                self.unrealized_pnl_pct = (current_price - self.executed_price) / self.executed_price
            elif self.action.direction == PositionDirection.SHORT:
                self.unrealized_pnl_pct = (self.executed_price - current_price) / self.executed_price
            self.unrealized_pnl = self.unrealized_pnl_pct * self.executed_size

    def close_position(self, close_price: float, lessons: str = ""):
        """포지션 청산 기록"""
        self.closed_at = datetime.now().isoformat()
        self.closed_price = close_price
        if self.executed_price and self.executed_price > 0:
            if self.action.direction == PositionDirection.LONG:
                self.realized_pnl_pct = (close_price - self.executed_price) / self.executed_price
            elif self.action.direction == PositionDirection.SHORT:
                self.realized_pnl_pct = (self.executed_price - close_price) / self.executed_price
            self.realized_pnl = self.realized_pnl_pct * self.executed_size

        # 신호 정확성 판단
        if self.action.direction == PositionDirection.LONG:
            self.signal_accuracy = self.realized_pnl_pct > 0
        elif self.action.direction == PositionDirection.SHORT:
            self.signal_accuracy = self.realized_pnl_pct > 0

        self.lessons_learned = lessons
        self.execution_status = "closed"

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['action'] = self.action.to_dict()
        return data


# ============================================================================
# Signal Conflict
# ============================================================================

@dataclass
class SignalConflict:
    """
    신호 충돌 정보

    ADDITIONAL_CONSIDERATIONS.md 4.3절 기반:
    - 같은 자산, 다른 시간 지평
    - 다른 자산 클래스 신호 충돌
    - 모델 vs 내재가치 충돌
    """
    conflict_id: str
    conflict_type: ConflictType
    signals: List[EnhancedSignal]
    description: str
    resolution: str = ""              # 해결 방안
    resolved: bool = False

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.conflict_id:
            self.conflict_id = f"CONF-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"

    def to_dict(self) -> Dict:
        return {
            'conflict_id': self.conflict_id,
            'conflict_type': self.conflict_type.value,
            'signals': [s.to_dict() for s in self.signals],
            'description': self.description,
            'resolution': self.resolution,
            'resolved': self.resolved,
            'timestamp': self.timestamp
        }


# ============================================================================
# Signal-Action Mapper
# ============================================================================

class SignalActionMapper:
    """
    신호-액션 매핑 엔진

    ADDITIONAL_CONSIDERATIONS.md 4절 전체 구현:
    - 신호 검증
    - 컨텍스트 확인
    - 포지션 크기 결정
    - 실행 전략 결정
    - 충돌 해결
    """

    # 신호 강도별 포지션 크기 매트릭스 (기본 프로파일의 한도 대비 비율)
    POSITION_SIZE_MATRIX = {
        SignalStrength.WEAK: 0.0,         # 관망
        SignalStrength.MODERATE: 0.25,    # 25% of 한도
        SignalStrength.STRONG: 0.50,      # 50% of 한도
        SignalStrength.VERY_STRONG: 0.75, # 75% of 한도
        SignalStrength.EXTREME: 1.0       # 100% of 한도
    }

    # 레짐별 포지션 조절 계수
    REGIME_ADJUSTMENT = {
        MarketRegime.CALM: 1.2,      # 안정적 시장 → 약간 확대
        MarketRegime.NORMAL: 1.0,    # 기본
        MarketRegime.ELEVATED: 0.6,  # 불안정 → 축소
        MarketRegime.CRISIS: 0.3     # 위기 → 대폭 축소
    }

    def __init__(self, risk_profile: RiskProfile):
        """
        Args:
            risk_profile: 사용자 리스크 프로파일
        """
        self.risk_profile = risk_profile
        self.action_logs: List[ActionLog] = []
        self.conflicts: List[SignalConflict] = []
        self.current_regime = MarketRegime.NORMAL

    def set_regime(self, vix: float):
        """VIX 기반 레짐 설정"""
        if vix < 15:
            self.current_regime = MarketRegime.CALM
        elif vix < 25:
            self.current_regime = MarketRegime.NORMAL
        elif vix < 35:
            self.current_regime = MarketRegime.ELEVATED
        else:
            self.current_regime = MarketRegime.CRISIS

    def validate_signal(self, signal: EnhancedSignal) -> Tuple[bool, str]:
        """
        Step 1: 신호 검증

        Returns:
            (valid, reason)
        """
        # 신뢰도 임계값 체크
        if signal.confidence < self.risk_profile.signal_threshold:
            return False, f"신뢰도 {signal.confidence:.1%} < 임계값 {self.risk_profile.signal_threshold:.1%}"

        # 레짐 일치 확인
        if signal.regime_aligned is False:
            return False, "현재 레짐과 신호 방향 불일치"

        # 내재가치 일치 확인 (보수적 접근)
        if signal.intrinsic_value_aligned is False and self.risk_profile.profile_type == RiskProfileType.CONSERVATIVE:
            return False, "내재가치와 신호 방향 불일치 (보수적 프로파일)"

        return True, "검증 통과"

    def check_context(self, signal: EnhancedSignal, existing_signals: List[EnhancedSignal]) -> Tuple[bool, str, Optional[SignalConflict]]:
        """
        Step 2: 컨텍스트 확인 (다른 신호와 충돌 체크)

        Returns:
            (ok, reason, conflict)
        """
        for existing in existing_signals:
            # 같은 자산, 다른 방향
            if existing.ticker == signal.ticker and existing.direction != signal.direction:
                if existing.horizon != signal.horizon:
                    # 시간 지평 충돌 → 분리 운용 가능
                    conflict = SignalConflict(
                        conflict_id="",
                        conflict_type=ConflictType.TEMPORAL,
                        signals=[signal, existing],
                        description=f"{signal.ticker}: 단기({signal.horizon}) vs 장기({existing.horizon}) 신호 충돌",
                        resolution="시간 지평별 분리 운용"
                    )
                    self.conflicts.append(conflict)
                    return True, "시간 지평 충돌 (분리 운용)", conflict
                else:
                    # 같은 시간 지평에서 방향 충돌 → 심각
                    conflict = SignalConflict(
                        conflict_id="",
                        conflict_type=ConflictType.DIRECTIONAL,
                        signals=[signal, existing],
                        description=f"{signal.ticker}: 동일 시간 지평에서 방향 충돌",
                        resolution="포지션 축소 또는 관망 권고"
                    )
                    self.conflicts.append(conflict)
                    return False, "방향성 충돌", conflict

        return True, "컨텍스트 확인 완료", None

    def calculate_position_size(self, signal: EnhancedSignal) -> float:
        """
        Step 3: 포지션 크기 결정

        신호 강도 × 프로파일 한도 × 레짐 조절
        """
        strength = signal.get_strength()
        base_ratio = self.POSITION_SIZE_MATRIX.get(strength, 0.0)

        # 프로파일 한도 적용
        max_size = self.risk_profile.max_position_size

        # 레짐 조절
        regime_factor = self.REGIME_ADJUSTMENT.get(self.current_regime, 1.0)
        if not self.risk_profile.auto_risk_adjust:
            regime_factor = 1.0

        # 최종 포지션 크기
        position_size = base_ratio * max_size * regime_factor

        # 단일 자산 한도 적용
        position_size = min(position_size, self.risk_profile.max_single_asset)

        return round(position_size, 4)

    def determine_execution_strategy(self, signal: EnhancedSignal, position_size: float) -> Dict[str, Any]:
        """
        Step 4: 실행 전략 결정
        """
        strength = signal.get_strength()

        if strength in [SignalStrength.WEAK]:
            return {
                "strategy": "wait",
                "split_count": 0,
                "rationale": "신호 강도 미흡, 추가 확인 필요"
            }

        elif strength in [SignalStrength.MODERATE, SignalStrength.STRONG]:
            # 분할 매매 권장
            return {
                "strategy": "split",
                "split_count": 3,
                "split_ratios": [0.33, 0.33, 0.34],
                "rationale": "분할 진입으로 리스크 분산"
            }

        elif strength in [SignalStrength.VERY_STRONG, SignalStrength.EXTREME]:
            # 즉시 진입 가능 (단, 여전히 분할 권장)
            return {
                "strategy": "split",
                "split_count": 2,
                "split_ratios": [0.50, 0.50],
                "rationale": "강한 신호, 2회 분할 진입"
            }

        return {"strategy": "immediate", "split_count": 1, "rationale": "기본 진입"}

    def map_signal_to_action(
        self,
        signal: EnhancedSignal,
        existing_signals: List[EnhancedSignal] = None,
        current_position: Optional[Dict] = None
    ) -> Optional[Action]:
        """
        신호 → 액션 매핑 메인 함수

        Args:
            signal: 분석할 신호
            existing_signals: 기존 신호 목록 (충돌 확인용)
            current_position: 현재 해당 자산의 포지션 정보

        Returns:
            Action 또는 None (액션 없음)
        """
        existing_signals = existing_signals or []
        current_position = current_position or {}

        # Step 1: 신호 검증
        valid, reason = self.validate_signal(signal)
        if not valid:
            return Action(
                action_id="",
                action_type=ActionType.NO_ACTION,
                ticker=signal.ticker,
                direction=PositionDirection.NEUTRAL,
                position_size=0.0,
                rationale=f"신호 검증 실패: {reason}"
            )

        # Step 2: 컨텍스트 확인
        context_ok, context_reason, conflict = self.check_context(signal, existing_signals)
        if not context_ok:
            return Action(
                action_id="",
                action_type=ActionType.NO_ACTION,
                ticker=signal.ticker,
                direction=PositionDirection.NEUTRAL,
                position_size=0.0,
                rationale=f"컨텍스트 확인 실패: {context_reason}"
            )

        # Step 3: 포지션 크기 결정
        position_size = self.calculate_position_size(signal)
        if position_size == 0:
            return Action(
                action_id="",
                action_type=ActionType.NO_ACTION,
                ticker=signal.ticker,
                direction=PositionDirection.NEUTRAL,
                position_size=0.0,
                rationale="포지션 크기가 0으로 계산됨"
            )

        # Step 4: 실행 전략 결정
        exec_strategy = self.determine_execution_strategy(signal, position_size)

        # 액션 유형 결정
        has_position = current_position.get('size', 0) > 0
        same_direction = current_position.get('direction') == signal.direction.value
        in_profit = current_position.get('pnl_pct', 0) > 0

        if not has_position:
            action_type = ActionType.NEW_POSITION
        elif same_direction and in_profit:
            action_type = ActionType.ADD_POSITION
        elif not same_direction:
            action_type = ActionType.REDUCE_POSITION  # 또는 CLOSE_POSITION
        else:
            action_type = ActionType.NO_ACTION

        # Risk:Reward 계산
        if signal.direction == PositionDirection.LONG:
            rr_ratio = abs(self.risk_profile.take_profit) / abs(self.risk_profile.stop_loss)
        else:
            rr_ratio = abs(self.risk_profile.take_profit) / abs(self.risk_profile.stop_loss)

        return Action(
            action_id="",
            action_type=action_type,
            ticker=signal.ticker,
            direction=signal.direction,
            position_size=position_size,
            stop_loss=self.risk_profile.stop_loss,
            take_profit=self.risk_profile.take_profit,
            execution_strategy=exec_strategy['strategy'],
            split_count=exec_strategy.get('split_count', 1),
            signal_ids=[signal.signal_id],
            total_confidence=signal.confidence,
            risk_reward_ratio=rr_ratio,
            rationale=exec_strategy['rationale']
        )

    def process_signals(
        self,
        signals: List[EnhancedSignal],
        current_positions: Dict[str, Dict] = None
    ) -> List[Action]:
        """
        여러 신호를 일괄 처리

        Args:
            signals: 신호 목록
            current_positions: {ticker: position_info} 현재 포지션 정보

        Returns:
            액션 목록
        """
        current_positions = current_positions or {}
        processed_signals = []
        actions = []

        # 신뢰도 높은 순으로 정렬
        sorted_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)

        for signal in sorted_signals:
            action = self.map_signal_to_action(
                signal=signal,
                existing_signals=processed_signals,
                current_position=current_positions.get(signal.ticker)
            )

            if action and action.action_type != ActionType.NO_ACTION:
                actions.append(action)

                # 액션 로그 생성
                log = ActionLog(log_id="", action=action)
                self.action_logs.append(log)

            processed_signals.append(signal)

        return actions

    def get_summary(self) -> Dict[str, Any]:
        """매핑 결과 요약"""
        return {
            "risk_profile": self.risk_profile.profile_type.value,
            "current_regime": self.current_regime.value,
            "total_actions": len(self.action_logs),
            "conflicts_detected": len(self.conflicts),
            "actions_by_type": self._count_actions_by_type(),
            "avg_position_size": self._avg_position_size()
        }

    def _count_actions_by_type(self) -> Dict[str, int]:
        counts = {}
        for log in self.action_logs:
            action_type = log.action.action_type.value
            counts[action_type] = counts.get(action_type, 0) + 1
        return counts

    def _avg_position_size(self) -> float:
        if not self.action_logs:
            return 0.0
        total = sum(log.action.position_size for log in self.action_logs)
        return round(total / len(self.action_logs), 4)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Signal-Action Mapping Framework Test")
    print("=" * 60)

    # 1. 리스크 프로파일 생성
    profile = RiskProfile.moderate()
    print(f"\n[1] Risk Profile: {profile.profile_type.value}")
    print(f"    Max Position: {profile.max_position_size:.0%}")
    print(f"    Signal Threshold: {profile.signal_threshold:.0%}")

    # 2. 매퍼 생성
    mapper = SignalActionMapper(risk_profile=profile)
    mapper.set_regime(vix=22.5)  # NORMAL regime
    print(f"\n[2] Current Regime: {mapper.current_regime.value}")

    # 3. 테스트 신호 생성
    signals = [
        EnhancedSignal(
            signal_id="",
            type="statistical",
            ticker="SPY",
            name="S&P 500 ETF",
            indicator="price_z",
            value=450.0,
            threshold=2.0,
            z_score=2.5,
            level="ALERT",
            description="SPY 가격 급등",
            confidence=0.75,
            direction=PositionDirection.LONG,
            horizon="short"
        ),
        EnhancedSignal(
            signal_id="",
            type="statistical",
            ticker="GLD",
            name="Gold ETF",
            indicator="rsi",
            value=83.6,
            threshold=70,
            z_score=2.1,
            level="ALERT",
            description="GLD RSI 과매수",
            confidence=0.68,
            direction=PositionDirection.SHORT,  # 과매수 → 매도 신호
            horizon="short"
        ),
        EnhancedSignal(
            signal_id="",
            type="statistical",
            ticker="BTC",
            name="Bitcoin",
            indicator="return_z",
            value=-5.2,
            threshold=3.0,
            z_score=-3.2,
            level="CRITICAL",
            description="BTC 일일 수익률 급락",
            confidence=0.52,  # 임계값 미달
            direction=PositionDirection.SHORT,
            horizon="ultra_short"
        )
    ]

    print(f"\n[3] Processing {len(signals)} signals...")

    # 4. 신호 처리
    actions = mapper.process_signals(signals)

    # 5. 결과 출력
    print(f"\n[4] Results:")
    print(f"    Actions Generated: {len(actions)}")
    print(f"    Conflicts: {len(mapper.conflicts)}")

    for i, action in enumerate(actions, 1):
        print(f"\n    Action {i}:")
        print(f"      Ticker: {action.ticker}")
        print(f"      Type: {action.action_type.value}")
        print(f"      Direction: {action.direction.value}")
        print(f"      Position Size: {action.position_size:.2%}")
        print(f"      Stop-Loss: {action.stop_loss:.1%}")
        print(f"      Strategy: {action.execution_strategy}")
        print(f"      Rationale: {action.rationale}")

    # 6. 요약
    summary = mapper.get_summary()
    print(f"\n[5] Summary:")
    for key, value in summary.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 60)
    print("Test Complete!")
