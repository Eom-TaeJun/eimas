from __future__ import annotations
#!/usr/bin/env python3
"""
Graph Portfolio - Representative Selection
============================================================

클러스터별 대표 자산 선정

Economic Foundation:
    - Liquidity-weighted selection
    - Centrality-based selection (most connected asset)
    - Sharpe ratio maximization

Class:
    - RepresentativeSelector: 대표 자산 선정 엔진
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import networkx as nx
import logging

from .enums import RepresentativeMethod

logger = logging.getLogger(__name__)


class RepresentativeSelector:
    """
    클러스터 대표 자산 선정

    경제학적 의미:
    - 각 클러스터를 대표하는 자산 1-3개 선정
    - N=10,000 → N'=50~100으로 차원 축소
    """

    def __init__(
        self,
        method: RepresentativeMethod = RepresentativeMethod.CENTRALITY,
        max_representatives: int = 3
    ):
        self.method = method
        self.max_representatives = max_representatives

    def select(
        self,
        clusters: Dict[int, List[str]],
        network: CorrelationNetwork,
        volumes: Optional[pd.DataFrame] = None
    ) -> Dict[int, List[str]]:
        """
        각 클러스터에서 대표 자산 선정

        Returns:
            {cluster_id: [representative_assets]}
        """
        representatives = {}
        centrality = network.calculate_centrality()

        for cluster_id, assets in clusters.items():
            if len(assets) == 1:
                representatives[cluster_id] = assets
                continue

            # 자산별 점수 계산
            scores = {}
            for asset in assets:
                if self.method == RepresentativeMethod.CENTRALITY:
                    scores[asset] = centrality.get(asset, {}).get('eigenvector', 0)

                elif self.method == RepresentativeMethod.VOLUME:
                    if volumes is not None and asset in volumes.columns:
                        scores[asset] = volumes[asset].mean()
                    else:
                        scores[asset] = network.graph.nodes[asset].get('avg_volume', 0)

                elif self.method == RepresentativeMethod.SHARPE:
                    scores[asset] = network.graph.nodes[asset].get('sharpe', 0)

                elif self.method == RepresentativeMethod.LIQUIDITY:
                    # 유동성 = 거래량 / 변동성
                    vol = network.graph.nodes[asset].get('avg_volume', 1)
                    volatility = network.graph.nodes[asset].get('volatility', 1)
                    scores[asset] = vol / (volatility + 1e-10)

            # 상위 자산 선정
            sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            n_reps = min(self.max_representatives, len(sorted_assets))
            representatives[cluster_id] = [a[0] for a in sorted_assets[:n_reps]]

        return representatives


