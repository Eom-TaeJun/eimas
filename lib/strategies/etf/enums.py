#!/usr/bin/env python3
"""ETF Strategy - Enums"""
from enum import Enum

class ThemeCategory(Enum):
    """투자 테마 카테고리"""
    AI_SEMICONDUCTOR = "ai_semiconductor"       # AI/반도체
    ELECTRIC_VEHICLE = "electric_vehicle"       # 전기차
    CLEAN_ENERGY = "clean_energy"               # 클린에너지
    DEFENSE = "defense"                          # 방산
    BIOTECH = "biotech"                          # 바이오텍
    CYBERSECURITY = "cybersecurity"             # 사이버보안
    FINTECH = "fintech"                          # 핀테크
    SPACE = "space"                              # 우주산업
    BLOCKCHAIN = "blockchain"                    # 블록체인
    CLOUD_SAAS = "cloud_saas"                   # 클라우드/SaaS
    CUSTOM = "custom"                            # 사용자 정의


class SupplyChainLayer(Enum):
    """공급망 레이어"""
    RAW_MATERIAL = "raw_material"   # 원자재
    COMPONENT = "component"          # 부품
    EQUIPMENT = "equipment"          # 장비
    MANUFACTURER = "manufacturer"    # 제조
    INTEGRATOR = "integrator"        # 통합
    DISTRIBUTION = "distribution"    # 유통
    END_USER = "end_user"            # 최종 사용자

