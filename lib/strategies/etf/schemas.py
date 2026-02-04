#!/usr/bin/env python3
"""ETF Strategy - Data Schemas"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from .enums import ThemeCategory, SupplyChainLayer

@dataclass
class ThemeStock:
    """테마 내 개별 종목"""
    ticker: str
    name: str
    layer: SupplyChainLayer
    weight_base: float = 0.0
    suppliers: List[str] = field(default_factory=list)
    customers: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ThemeETF:
    """테마 ETF 정의"""
    name: str
    category: ThemeCategory
    description: str
    stocks: List[ThemeStock]
    target_weight: Dict[str, float] = field(default_factory=dict)
    supply_chain_graph: Optional[nx.DiGraph] = None


# =============================================================================
# 사전 정의된 테마 ETF 템플릿
# =============================================================================

THEME_TEMPLATES: Dict[ThemeCategory, Dict] = {
    ThemeCategory.AI_SEMICONDUCTOR: {
        "name": "AI & Semiconductor Revolution",
        "description": "AI 인프라와 반도체 밸류체인",
        "stocks": [
            # Raw Material / Component
            ThemeStock("AMAT", "Applied Materials", SupplyChainLayer.EQUIPMENT, 0.08,
                      customers=["TSM", "INTC", "NVDA"]),
            ThemeStock("ASML", "ASML Holding", SupplyChainLayer.EQUIPMENT, 0.10,
                      customers=["TSM", "INTC", "SSNLF"]),
            ThemeStock("LRCX", "Lam Research", SupplyChainLayer.EQUIPMENT, 0.06,
                      customers=["TSM", "INTC"]),
            ThemeStock("KLAC", "KLA Corp", SupplyChainLayer.EQUIPMENT, 0.05,
                      customers=["TSM", "INTC"]),
            # Manufacturer
            ThemeStock("TSM", "Taiwan Semiconductor", SupplyChainLayer.MANUFACTURER, 0.12,
                      suppliers=["ASML", "AMAT"], customers=["NVDA", "AMD", "AAPL"]),
            ThemeStock("INTC", "Intel", SupplyChainLayer.MANUFACTURER, 0.06,
                      suppliers=["ASML", "AMAT"]),
            # Integrator (Chip Designer)
            ThemeStock("NVDA", "NVIDIA", SupplyChainLayer.INTEGRATOR, 0.15,
                      suppliers=["TSM"], customers=["MSFT", "GOOGL", "AMZN"]),
            ThemeStock("AMD", "AMD", SupplyChainLayer.INTEGRATOR, 0.08,
                      suppliers=["TSM"], customers=["MSFT", "GOOGL"]),
            ThemeStock("AVGO", "Broadcom", SupplyChainLayer.INTEGRATOR, 0.08,
                      suppliers=["TSM"]),
            ThemeStock("MRVL", "Marvell", SupplyChainLayer.INTEGRATOR, 0.04,
                      suppliers=["TSM"]),
            # End User (AI Hyperscalers)
            ThemeStock("MSFT", "Microsoft", SupplyChainLayer.END_USER, 0.08,
                      suppliers=["NVDA", "AMD"]),
            ThemeStock("GOOGL", "Alphabet", SupplyChainLayer.END_USER, 0.06,
                      suppliers=["NVDA", "AMD"]),
            ThemeStock("AMZN", "Amazon", SupplyChainLayer.END_USER, 0.04,
                      suppliers=["NVDA"]),
        ]
    },

    ThemeCategory.ELECTRIC_VEHICLE: {
        "name": "Electric Vehicle Ecosystem",
        "description": "전기차 밸류체인 (배터리 → 차량 → 충전)",
        "stocks": [
            # Raw Material
            ThemeStock("ALB", "Albemarle", SupplyChainLayer.RAW_MATERIAL, 0.06,
                      customers=["PCRFY", "LG에너지"]),
            ThemeStock("SQM", "SQM", SupplyChainLayer.RAW_MATERIAL, 0.05,
                      customers=["PCRFY"]),
            ThemeStock("LAC", "Lithium Americas", SupplyChainLayer.RAW_MATERIAL, 0.03),
            # Component (Battery)
            ThemeStock("PCRFY", "Panasonic", SupplyChainLayer.COMPONENT, 0.08,
                      suppliers=["ALB"], customers=["TSLA"]),
            ThemeStock("QS", "QuantumScape", SupplyChainLayer.COMPONENT, 0.03),
            # Manufacturer (EV Makers)
            ThemeStock("TSLA", "Tesla", SupplyChainLayer.MANUFACTURER, 0.20,
                      suppliers=["PCRFY"], customers=["CHPT"]),
            ThemeStock("RIVN", "Rivian", SupplyChainLayer.MANUFACTURER, 0.06),
            ThemeStock("LCID", "Lucid", SupplyChainLayer.MANUFACTURER, 0.04),
            ThemeStock("NIO", "NIO", SupplyChainLayer.MANUFACTURER, 0.05,
                      customers=["XPEV"]),
            ThemeStock("XPEV", "XPeng", SupplyChainLayer.MANUFACTURER, 0.04),
            ThemeStock("LI", "Li Auto", SupplyChainLayer.MANUFACTURER, 0.04),
            ThemeStock("F", "Ford", SupplyChainLayer.MANUFACTURER, 0.06),
            ThemeStock("GM", "General Motors", SupplyChainLayer.MANUFACTURER, 0.06),
            # Infrastructure (Charging)
            ThemeStock("CHPT", "ChargePoint", SupplyChainLayer.DISTRIBUTION, 0.05,
                      suppliers=["TSLA"]),
            ThemeStock("BLNK", "Blink Charging", SupplyChainLayer.DISTRIBUTION, 0.03),
            ThemeStock("EVGO", "EVgo", SupplyChainLayer.DISTRIBUTION, 0.02),
        ]
    },

    ThemeCategory.CLEAN_ENERGY: {
        "name": "Clean Energy Transition",
        "description": "재생에너지 + ESS + 그리드",
        "stocks": [
            # Equipment
            ThemeStock("FSLR", "First Solar", SupplyChainLayer.EQUIPMENT, 0.12,
                      customers=["NEE"]),
            ThemeStock("ENPH", "Enphase", SupplyChainLayer.EQUIPMENT, 0.10,
                      customers=["RUN"]),
            ThemeStock("SEDG", "SolarEdge", SupplyChainLayer.EQUIPMENT, 0.06),
            ThemeStock("CSIQ", "Canadian Solar", SupplyChainLayer.EQUIPMENT, 0.05),
            # Storage
            ThemeStock("FLUENCE", "Fluence Energy", SupplyChainLayer.COMPONENT, 0.05),
            # Wind
            ThemeStock("VWDRY", "Vestas Wind", SupplyChainLayer.EQUIPMENT, 0.06),
            # Utility
            ThemeStock("NEE", "NextEra Energy", SupplyChainLayer.END_USER, 0.15,
                      suppliers=["FSLR"]),
            ThemeStock("AES", "AES Corp", SupplyChainLayer.END_USER, 0.06),
            ThemeStock("BEP", "Brookfield Renewable", SupplyChainLayer.END_USER, 0.08),
            # Installer/Service
            ThemeStock("RUN", "Sunrun", SupplyChainLayer.DISTRIBUTION, 0.08,
                      suppliers=["ENPH"]),
            ThemeStock("NOVA", "Sunnova", SupplyChainLayer.DISTRIBUTION, 0.04),
            # Hydrogen
            ThemeStock("PLUG", "Plug Power", SupplyChainLayer.COMPONENT, 0.05),
            ThemeStock("BE", "Bloom Energy", SupplyChainLayer.COMPONENT, 0.05),
        ]
    },

    ThemeCategory.DEFENSE: {
        "name": "Defense & Aerospace",
        "description": "방산/항공우주 밸류체인",
        "stocks": [
            ThemeStock("LMT", "Lockheed Martin", SupplyChainLayer.INTEGRATOR, 0.18),
            ThemeStock("RTX", "RTX Corp", SupplyChainLayer.INTEGRATOR, 0.15),
            ThemeStock("NOC", "Northrop Grumman", SupplyChainLayer.INTEGRATOR, 0.12),
            ThemeStock("GD", "General Dynamics", SupplyChainLayer.INTEGRATOR, 0.12),
            ThemeStock("BA", "Boeing", SupplyChainLayer.MANUFACTURER, 0.10),
            ThemeStock("LHX", "L3Harris", SupplyChainLayer.COMPONENT, 0.08),
            ThemeStock("HII", "Huntington Ingalls", SupplyChainLayer.MANUFACTURER, 0.06),
            ThemeStock("KTOS", "Kratos Defense", SupplyChainLayer.COMPONENT, 0.05),
            ThemeStock("PLTR", "Palantir", SupplyChainLayer.END_USER, 0.08),
            ThemeStock("LDOS", "Leidos", SupplyChainLayer.END_USER, 0.06),
        ]
    },

    ThemeCategory.CYBERSECURITY: {
        "name": "Cybersecurity Shield",
        "description": "사이버보안 생태계",
        "stocks": [
            ThemeStock("CRWD", "CrowdStrike", SupplyChainLayer.END_USER, 0.15),
            ThemeStock("PANW", "Palo Alto Networks", SupplyChainLayer.END_USER, 0.15),
            ThemeStock("FTNT", "Fortinet", SupplyChainLayer.INTEGRATOR, 0.12),
            ThemeStock("ZS", "Zscaler", SupplyChainLayer.END_USER, 0.10),
            ThemeStock("OKTA", "Okta", SupplyChainLayer.COMPONENT, 0.08),
            ThemeStock("NET", "Cloudflare", SupplyChainLayer.COMPONENT, 0.10),
            ThemeStock("S", "SentinelOne", SupplyChainLayer.END_USER, 0.08),
            ThemeStock("CYBR", "CyberArk", SupplyChainLayer.COMPONENT, 0.07),
            ThemeStock("TENB", "Tenable", SupplyChainLayer.COMPONENT, 0.05),
            ThemeStock("QLYS", "Qualys", SupplyChainLayer.COMPONENT, 0.05),
            ThemeStock("RPD", "Rapid7", SupplyChainLayer.COMPONENT, 0.05),
        ]
    },

    ThemeCategory.BLOCKCHAIN: {
        "name": "Blockchain & Digital Assets",
        "description": "블록체인 인프라 및 관련 기업",
        "stocks": [
            ThemeStock("COIN", "Coinbase", SupplyChainLayer.DISTRIBUTION, 0.15),
            ThemeStock("MSTR", "MicroStrategy", SupplyChainLayer.END_USER, 0.12),
            ThemeStock("SQ", "Block Inc", SupplyChainLayer.INTEGRATOR, 0.12),
            ThemeStock("MARA", "Marathon Digital", SupplyChainLayer.MANUFACTURER, 0.08),
            ThemeStock("RIOT", "Riot Platforms", SupplyChainLayer.MANUFACTURER, 0.08),
            ThemeStock("CLSK", "CleanSpark", SupplyChainLayer.MANUFACTURER, 0.06),
            ThemeStock("HUT", "Hut 8 Mining", SupplyChainLayer.MANUFACTURER, 0.05),
            ThemeStock("GLXY", "Galaxy Digital", SupplyChainLayer.INTEGRATOR, 0.08),
            ThemeStock("SI", "Silvergate", SupplyChainLayer.DISTRIBUTION, 0.05),
            ThemeStock("HOOD", "Robinhood", SupplyChainLayer.DISTRIBUTION, 0.08),
            # Add crypto exposure
            ThemeStock("BTC-USD", "Bitcoin", SupplyChainLayer.RAW_MATERIAL, 0.08),
            ThemeStock("ETH-USD", "Ethereum", SupplyChainLayer.RAW_MATERIAL, 0.05),
        ]
    },
}

