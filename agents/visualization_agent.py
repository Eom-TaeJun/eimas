#!/usr/bin/env python3
"""
Visualization Agent - 대시보드 생성 에이전트
=============================================
분석 결과를 HTML 대시보드로 시각화하는 에이전트.

역할:
- 다른 에이전트 결과 수집
- dashboard_generator 호출
- HTML 파일 저장 및 경로 반환
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# eimas 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schemas import (
    AgentRequest, 
    AgentResponse, 
    AgentOpinion, 
    AgentRole, 
    OpinionStrength,
    DashboardConfig,
    ForecastResult,
    Consensus,
    Conflict
)
from agents.base_agent import BaseAgent, AgentConfig

# 로거 설정
logger = logging.getLogger('eimas.visualization_agent')


class VisualizationAgent(BaseAgent):
    """
    대시보드 생성 전용 에이전트
    
    다른 에이전트들의 분석 결과를 수집하여 
    인터랙티브 HTML 대시보드로 시각화.
    
    Features:
        - 자산군별 위험 현황
        - 레짐 분석 (BULL/BEAR/TRANSITION/CRISIS)
        - LASSO 예측 결과
        - 멀티에이전트 토론 결과
        - Critical Path 분석
    
    Example:
        >>> agent = VisualizationAgent()
        >>> request = AgentRequest(
        ...     task_id="viz_001",
        ...     role=AgentRole.STRATEGY,
        ...     instruction="Generate dashboard",
        ...     context={
        ...         'signals': [...],
        ...         'regime_data': {...},
        ...         'forecast_results': [...],
        ...         'agent_opinions': [...],
        ...         'consensus': {...}
        ...     }
        ... )
        >>> response = await agent.execute(request)
        >>> print(response.result['dashboard_path'])
    """
    
    def __init__(
        self, 
        config: Optional[AgentConfig] = None,
        dashboard_config: Optional[DashboardConfig] = None
    ):
        """
        VisualizationAgent 초기화
        
        Args:
            config: 에이전트 설정. None이면 기본값 사용.
            dashboard_config: 대시보드 설정. None이면 기본값 사용.
        """
        if config is None:
            config = AgentConfig(
                name="VisualizationAgent",
                role=AgentRole.STRATEGY,  # 전략 역할로 분류
                timeout=120,
                verbose=True
            )
        super().__init__(config)
        
        # 대시보드 설정
        self.dashboard_config = dashboard_config or DashboardConfig()
        
        # 출력 디렉토리 확인/생성
        self.output_dir = Path(self.dashboard_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VisualizationAgent initialized, output_dir: {self.output_dir}")
    
    async def _execute(self, request: AgentRequest) -> Dict[str, Any]:
        """
        대시보드 생성 실행
        
        Args:
            request: AgentRequest with context containing:
                - signals: List[Dict] (이상 신호)
                - regime_data: Dict (레짐 정보)
                - risk_metrics: Dict (위험 메트릭)
                - macro_indicators: Dict (거시경제 지표)
                - agent_opinions: List[AgentOpinion] (에이전트 의견)
                - consensus: Consensus (합의 결과)
                - conflicts: List[Conflict] (충돌 목록)
                - forecast_results: List[ForecastResult] (LASSO 결과)
                - timestamp: str
                - project_id: str
        
        Returns:
            Dict containing:
                - dashboard_path: str (저장된 HTML 경로)
                - dashboard_size: int (바이트)
                - sections_generated: List[str]
        """
        context = request.context
        
        # 1. 에이전트 결과 수집
        collected_data = self._collect_agent_results(context)
        
        self.logger.info(f"Collected data: {list(collected_data.keys())}")
        
        # 2. 기본 대시보드 HTML 생성
        html_content = self._generate_base_dashboard(collected_data)
        
        sections_generated = ['header', 'summary']
        
        # 3. 조건부 섹션 추가
        if self.dashboard_config.include_regime and collected_data.get('regime_data'):
            regime_section = self._generate_regime_section(collected_data['regime_data'])
            html_content = self._insert_section(html_content, regime_section, 'regime')
            sections_generated.append('regime')
        
        if self.dashboard_config.include_lasso_results:
            # 결과가 없어도 진단 정보 표시
            lasso_section = self._generate_lasso_section_html(
                collected_data.get('forecast_results', []),
                collected_data.get('forecast_diagnostics', {})
            )
            html_content = self._insert_section(html_content, lasso_section, 'lasso')
            sections_generated.append('lasso')
        
        if self.dashboard_config.include_agent_debate and collected_data.get('agent_opinions'):
            agent_section = self._generate_agent_section_html(
                collected_data.get('agent_opinions', []),
                collected_data.get('consensus'),
                collected_data.get('conflicts', [])
            )
            html_content = self._insert_section(html_content, agent_section, 'agents')
            sections_generated.append('agents')
        
        if self.dashboard_config.include_risk_metrics and collected_data.get('risk_metrics'):
            risk_section = self._generate_risk_section(collected_data['risk_metrics'])
            html_content = self._insert_section(html_content, risk_section, 'risk')
            sections_generated.append('risk')
        
        # 4. 푸터 추가
        html_content = self._add_footer(html_content)
        
        # 5. 파일 저장
        output_path = self._generate_output_path(context)
        self._save_dashboard(html_content, output_path)
        
        dashboard_size = len(html_content.encode('utf-8'))
        
        self.logger.info(
            f"Dashboard generated: {output_path} "
            f"({dashboard_size / 1024:.1f} KB, {len(sections_generated)} sections)"
        )
        
        return {
            'dashboard_path': str(output_path),
            'dashboard_size': dashboard_size,
            'sections_generated': sections_generated,
            'confidence': 1.0,
            'reasoning': f"Dashboard generated with {len(sections_generated)} sections"
        }
    
    async def form_opinion(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> AgentOpinion:
        """
        VisualizationAgent는 분석보다 시각화에 집중하므로
        의견 형성은 제한적.
        """
        return AgentOpinion(
            agent_role=self.config.role,
            topic=topic,
            position="Visualization focus - no analytical opinion",
            strength=OpinionStrength.NEUTRAL,
            confidence=0.5,
            evidence=["This agent focuses on visualization, not analysis"]
        )
    
    def _collect_agent_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        컨텍스트에서 필요한 데이터 수집 및 정리
        """
        return {
            'signals': context.get('signals', []),
            'regime_data': context.get('regime_data', {}),
            'risk_metrics': context.get('risk_metrics', {}),
            'macro_indicators': context.get('macro_indicators', {}),
            'agent_opinions': context.get('agent_opinions', []),
            'consensus': context.get('consensus'),
            'conflicts': context.get('conflicts', []),
            'forecast_results': context.get('forecast_results', []),
            'forecast_diagnostics': context.get('forecast_diagnostics', {}),  # 진단 정보 추가
            'timestamp': context.get('timestamp', datetime.now().isoformat()),
            'project_id': context.get('project_id', 'default')
        }
    
    def _generate_output_path(self, context: Dict[str, Any]) -> Path:
        """출력 경로 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        project_id = context.get('project_id', 'default')
        filename = f"dashboard_{timestamp}_{project_id}.html"
        return self.output_dir / filename
    
    def _save_dashboard(self, html: str, path: Path) -> None:
        """HTML 파일 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def _generate_base_dashboard(self, data: Dict[str, Any]) -> str:
        """기본 대시보드 HTML 생성"""
        
        theme = self.dashboard_config.theme
        bg_color = '#1a1a2e' if theme == 'dark' else '#ffffff'
        text_color = '#e0e0e0' if theme == 'dark' else '#333333'
        card_bg = '#16213e' if theme == 'dark' else '#f5f5f5'
        accent_color = '#4a90d9'
        
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        html = f'''<!DOCTYPE html>
<html lang="{self.dashboard_config.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EIMAS Dashboard - {timestamp[:10]}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: {bg_color};
            color: {text_color};
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid {accent_color};
            margin-bottom: 30px;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            color: {accent_color};
            margin-bottom: 10px;
        }}
        
        header .timestamp {{
            color: #888;
            font-size: 0.9rem;
        }}
        
        .section {{
            background: {card_bg};
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        .section-title {{
            font-size: 1.4rem;
            color: {accent_color};
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid {accent_color};
        }}
        
        .card-title {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        
        .card-value {{
            font-size: 1.8rem;
            font-weight: bold;
        }}
        
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        .neutral {{ color: #f59e0b; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        th {{
            background: rgba(74, 144, 217, 0.2);
            font-weight: 600;
        }}
        
        .agent-cards {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .agent-card {{
            flex: 1;
            min-width: 200px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        
        .agent-name {{
            font-weight: bold;
            margin-bottom: 8px;
        }}
        
        .agent-position {{
            font-size: 1.2rem;
            margin-bottom: 5px;
        }}
        
        .agent-confidence {{
            font-size: 0.85rem;
            color: #888;
        }}
        
        .consensus-box {{
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid #22c55e;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .conflict-list {{
            list-style: none;
            margin-top: 15px;
        }}
        
        .conflict-list li {{
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 20px;
        }}
        
        footer {{
            text-align: center;
            padding: 30px 0;
            color: #666;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 40px;
        }}
        
        /* Sections placeholder */
        #section-regime {{ }}
        #section-lasso {{ }}
        #section-agents {{ }}
        #section-risk {{ }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 EIMAS Multi-Agent Dashboard</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>
        
        <!-- Summary Section -->
        <div class="section">
            <h2 class="section-title">📊 Summary</h2>
            <div class="card-grid">
                <div class="card">
                    <div class="card-title">Total Signals</div>
                    <div class="card-value">{len(data.get('signals', []))}</div>
                </div>
                <div class="card">
                    <div class="card-title">Forecast Horizons</div>
                    <div class="card-value">{len(data.get('forecast_results', []))}</div>
                </div>
                <div class="card">
                    <div class="card-title">Agent Opinions</div>
                    <div class="card-value">{len(data.get('agent_opinions', []))}</div>
                </div>
            </div>
        </div>
        
        <!-- Dynamic sections will be inserted here -->
        <div id="section-regime"></div>
        <div id="section-lasso"></div>
        <div id="section-agents"></div>
        <div id="section-risk"></div>
        
    </div>
</body>
</html>'''
        
        return html
    
    def _insert_section(self, html: str, section_html: str, section_id: str) -> str:
        """섹션을 HTML에 삽입"""
        placeholder = f'<div id="section-{section_id}"></div>'
        return html.replace(placeholder, section_html)
    
    def _generate_regime_section(self, regime_data: Dict) -> str:
        """레짐 분석 섹션 생성"""
        current_regime = regime_data.get('current_regime', 'UNKNOWN')
        probability = regime_data.get('probability', 0.0)
        
        regime_colors = {
            'BULL': '#22c55e',
            'BEAR': '#ef4444',
            'TRANSITION': '#f59e0b',
            'CRISIS': '#dc2626',
            'UNKNOWN': '#888888'
        }
        
        color = regime_colors.get(current_regime, '#888888')
        
        return f'''
        <div class="section">
            <h2 class="section-title">📈 Regime Analysis</h2>
            <div class="card-grid">
                <div class="card" style="border-left-color: {color};">
                    <div class="card-title">Current Regime</div>
                    <div class="card-value" style="color: {color};">{current_regime}</div>
                </div>
                <div class="card">
                    <div class="card-title">Confidence</div>
                    <div class="card-value">{probability:.1%}</div>
                </div>
            </div>
        </div>
        '''
    
    def _generate_lasso_section_html(self, results: List, diagnostics: Dict = None) -> str:
        """LASSO 결과 섹션 생성"""
        diagnostics = diagnostics or {}
        
        # 문제 진단 섹션 생성
        issues_html = ""
        issues = []
        
        # 결과가 없거나 모두 n_observations가 0인 경우
        if not results:
            issues.append("❌ 분석 결과 없음: LASSO 모델이 실행되지 않았습니다.")
        else:
            total_obs = 0
            for result in results:
                if isinstance(result, dict):
                    n_obs = result.get('n_observations', 0)
                else:
                    n_obs = getattr(result, 'n_observations', 0)
                total_obs += n_obs
            
            if total_obs == 0:
                issues.append("❌ 관측치 없음: 모든 horizon에서 n_observations = 0")
        
        # diagnostics에서 문제 추출
        if diagnostics:
            if diagnostics.get('common_dates', 0) == 0:
                issues.append("❌ 공통 날짜 없음: CME 데이터와 시장 데이터의 날짜가 겹치지 않음")
            elif diagnostics.get('common_dates', 0) < 30:
                issues.append(f"⚠️ 공통 날짜 부족: {diagnostics.get('common_dates')}개 (최소 30개 권장)")
            
            if not diagnostics.get('has_d_exp_rate', False):
                issues.append("❌ 종속변수 누락: d_Exp_Rate가 데이터에 없음")
            
            if diagnostics.get('feature_count', 0) < 5:
                issues.append(f"⚠️ 설명변수 부족: {diagnostics.get('feature_count', 0)}개 (최소 5개 권장)")
            
            if diagnostics.get('days_to_meeting_missing', False):
                issues.append("❌ days_to_meeting 누락: FOMC 일정 데이터 없음")
            
            if diagnostics.get('cme_data_rows', 0) == 0:
                issues.append("❌ CME 데이터 없음: CME 패널 데이터 로드 실패")
            
            if diagnostics.get('market_data_rows', 0) == 0:
                issues.append("❌ 시장 데이터 없음: market_data가 비어있음")
            
            # 날짜 범위 정보 (디버그 용)
            if diagnostics.get('market_date_range'):
                issues.append(f"📅 시장 데이터 기간: {diagnostics['market_date_range']}")
            if diagnostics.get('cme_date_range'):
                issues.append(f"📅 CME 데이터 기간: {diagnostics['cme_date_range']}")
        
        if issues:
            issues_items = ''.join([f'<li style="margin: 5px 0;">{issue}</li>' for issue in issues])
            issues_html = f'''
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border: 1px solid #e74c3c;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            ">
                <h3 style="color: #e74c3c; margin-top: 0; margin-bottom: 10px;">🔍 문제 진단 (Diagnostics)</h3>
                <ul style="color: #e0e0e0; margin: 0; padding-left: 20px; list-style: none;">
                    {issues_items}
                </ul>
            </div>
            '''
        
        # 테이블 행 생성
        rows = ""
        for result in results:
            if isinstance(result, dict):
                horizon = result.get('horizon', 'Unknown')
                r_squared = result.get('r_squared', 0.0)
                n_obs = result.get('n_observations', 0)
                n_selected = len(result.get('selected_variables', []))
                top_vars = result.get('selected_variables', [])[:3]
            else:
                horizon = result.horizon
                r_squared = result.r_squared
                n_obs = getattr(result, 'n_observations', 0)
                n_selected = len(result.selected_variables)
                top_vars = result.selected_variables[:3]
            
            top_vars_str = ', '.join(top_vars) if top_vars else 'None'
            
            # n_obs 색상
            if n_obs == 0:
                n_obs_style = 'color: #ef4444;'
                n_obs_warn = ' ⚠️'
            elif n_obs < 30:
                n_obs_style = 'color: #f59e0b;'
                n_obs_warn = ''
            else:
                n_obs_style = 'color: #22c55e;'
                n_obs_warn = ''
            
            rows += f'''
                <tr>
                    <td>{horizon}</td>
                    <td style="{n_obs_style}">{n_obs}{n_obs_warn}</td>
                    <td>{r_squared:.4f}</td>
                    <td>{n_selected}</td>
                    <td>{top_vars_str}...</td>
                </tr>
            '''
        
        # Long horizon 차트 데이터 준비
        long_result = results[2] if len(results) > 2 else (results[-1] if results else None)
        chart_script = ""
        
        if long_result:
            if isinstance(long_result, dict):
                coefficients = long_result.get('coefficients', {})
            else:
                coefficients = long_result.coefficients
            
            if coefficients:
                sorted_coefs = sorted(
                    coefficients.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]
                
                labels = [c[0] for c in sorted_coefs]
                values = [c[1] for c in sorted_coefs]
                colors = ['#22c55e' if v > 0 else '#ef4444' for v in values]
                
                chart_script = f'''
                <div class="chart-container">
                    <canvas id="lassoChart"></canvas>
                </div>
                <script>
                    new Chart(document.getElementById('lassoChart'), {{
                        type: 'bar',
                        data: {{
                            labels: {labels},
                            datasets: [{{
                                label: 'Coefficient',
                                data: {values},
                                backgroundColor: {colors}
                            }}]
                        }},
                        options: {{
                            indexAxis: 'y',
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ display: false }},
                                title: {{
                                    display: true,
                                    text: 'Top 10 LASSO Coefficients (Long Horizon)',
                                    color: '#e0e0e0'
                                }}
                            }},
                            scales: {{
                                x: {{
                                    grid: {{ color: 'rgba(255,255,255,0.1)' }},
                                    ticks: {{ color: '#888' }}
                                }},
                                y: {{
                                    grid: {{ display: false }},
                                    ticks: {{ color: '#e0e0e0' }}
                                }}
                            }}
                        }}
                    }});
                </script>
                '''
        
        return f'''
        <div class="section">
            <h2 class="section-title">📈 LASSO Fed Rate Forecast</h2>
            {issues_html}
            <table>
                <thead>
                    <tr>
                        <th>Horizon</th>
                        <th>Obs</th>
                        <th>R²</th>
                        <th>Selected</th>
                        <th>Top Variables</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            {chart_script}
        </div>
        '''
    
    def _generate_agent_section_html(
        self, 
        opinions: List, 
        consensus: Optional[Any],
        conflicts: List
    ) -> str:
        """멀티에이전트 섹션 생성"""
        
        # 에이전트 카드 생성
        cards = ""
        for opinion in opinions:
            if isinstance(opinion, dict):
                agent_id = opinion.get('agent_role', 'Unknown')
                position = opinion.get('position', 'N/A')
                confidence = opinion.get('confidence', 0.0)
            else:
                agent_id = opinion.agent_role.value if hasattr(opinion.agent_role, 'value') else str(opinion.agent_role)
                position = opinion.position
                confidence = opinion.confidence
            
            # 포지션에 따른 색상
            if 'UP' in str(position).upper() or 'HIKE' in str(position).upper():
                border_color = '#22c55e'
            elif 'DOWN' in str(position).upper() or 'CUT' in str(position).upper():
                border_color = '#ef4444'
            else:
                border_color = '#f59e0b'
            
            cards += f'''
                <div class="agent-card" style="border-left: 4px solid {border_color};">
                    <div class="agent-name">{agent_id}</div>
                    <div class="agent-position" style="color: {border_color};">{position}</div>
                    <div class="agent-confidence">conf: {confidence:.2f}</div>
                </div>
            '''
        
        # 합의 박스
        consensus_html = ""
        if consensus:
            if isinstance(consensus, dict):
                final_position = consensus.get('final_position', 'N/A')
                agreement = consensus.get('confidence', 0.0)
            else:
                final_position = consensus.final_position
                agreement = consensus.confidence
            
            consensus_html = f'''
            <div class="consensus-box">
                <span>📊</span>
                <span>Consensus: <strong>{final_position}</strong> (Agreement: {agreement:.0%})</span>
            </div>
            '''
        
        # 충돌 목록
        conflicts_html = ""
        if conflicts:
            conflicts_html = '<ul class="conflict-list">'
            for conflict in conflicts:
                if isinstance(conflict, dict):
                    topic = conflict.get('topic', 'Unknown')
                    agents = conflict.get('agents', [])
                else:
                    topic = conflict.topic
                    agents = [a.value if hasattr(a, 'value') else str(a) for a in conflict.agents]
                
                conflicts_html += f'<li>⚠️ {topic}: {" vs ".join(map(str, agents))}</li>'
            conflicts_html += '</ul>'
        
        return f'''
        <div class="section">
            <h2 class="section-title">🤖 Multi-Agent Analysis</h2>
            <div class="agent-cards">
                {cards}
            </div>
            {consensus_html}
            {conflicts_html if conflicts_html else ''}
        </div>
        '''
    
    def _generate_risk_section(self, risk_metrics: Dict) -> str:
        """위험 메트릭 섹션 생성"""
        
        cards = ""
        for metric_name, value in risk_metrics.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                
                # 값에 따른 색상
                if 'sharpe' in metric_name.lower():
                    color_class = 'positive' if value > 1 else 'negative' if value < 0 else 'neutral'
                elif 'drawdown' in metric_name.lower():
                    color_class = 'negative' if value < -0.1 else 'neutral'
                else:
                    color_class = 'neutral'
                
                cards += f'''
                    <div class="card">
                        <div class="card-title">{metric_name}</div>
                        <div class="card-value {color_class}">{formatted_value}</div>
                    </div>
                '''
        
        return f'''
        <div class="section">
            <h2 class="section-title">⚠️ Risk Metrics</h2>
            <div class="card-grid">
                {cards}
            </div>
        </div>
        '''
    
    def _add_footer(self, html: str) -> str:
        """푸터 추가"""
        footer = '''
        <footer>
            <p>Generated by EIMAS (Economic Intelligence Multi-Agent System)</p>
            <p>© 2025 - Dashboard v1.0</p>
        </footer>
    </div>
</body>
</html>'''
        
        # 기존 닫는 태그 교체
        return html.replace('</div>\n</body>\n</html>', footer)


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        """VisualizationAgent 테스트"""
        print("=" * 60)
        print("VisualizationAgent Test")
        print("=" * 60)
        
        # 에이전트 초기화
        config = DashboardConfig(
            theme='dark',
            language='ko',
            output_dir='outputs/dashboards'
        )
        agent = VisualizationAgent(dashboard_config=config)
        print(f"\n1. Initialized: {agent}")
        
        # Mock 데이터
        mock_context = {
            'signals': [
                {'type': 'anomaly', 'asset': 'SPY', 'severity': 0.8},
                {'type': 'anomaly', 'asset': 'BTC', 'severity': 0.6}
            ],
            'regime_data': {
                'current_regime': 'TRANSITION',
                'probability': 0.72
            },
            'risk_metrics': {
                'sharpe_ratio': 1.25,
                'max_drawdown': -0.15,
                'volatility': 0.18
            },
            'forecast_results': [
                {
                    'horizon': 'VeryShort',
                    'r_squared': 0.02,
                    'selected_variables': ['d_Breakeven5Y'],
                    'coefficients': {'d_Breakeven5Y': 0.15}
                },
                {
                    'horizon': 'Short',
                    'r_squared': 0.35,
                    'selected_variables': ['d_Spread_Baa', 'd_HighYield_Rate', 'Ret_VIX'],
                    'coefficients': {'d_Spread_Baa': -0.42, 'd_HighYield_Rate': 0.35, 'Ret_VIX': 0.28}
                },
                {
                    'horizon': 'Long',
                    'r_squared': 0.64,
                    'selected_variables': ['d_Baa_Yield', 'd_Spread_Baa', 'Ret_Dollar_Idx', 'd_Breakeven5Y'],
                    'coefficients': {
                        'd_Baa_Yield': 2.09,
                        'd_Spread_Baa': -1.66,
                        'Ret_Dollar_Idx': 1.04,
                        'd_Breakeven5Y': 0.85
                    }
                }
            ],
            'agent_opinions': [
                {
                    'agent_role': 'analysis',
                    'position': 'BEARISH',
                    'confidence': 0.75
                },
                {
                    'agent_role': 'forecast',
                    'position': 'HOLD',
                    'confidence': 0.68
                },
                {
                    'agent_role': 'strategy',
                    'position': 'CAUTIOUS',
                    'confidence': 0.72
                }
            ],
            'consensus': {
                'final_position': 'CAUTIOUS HOLD',
                'confidence': 0.78
            },
            'conflicts': [
                {
                    'topic': 'rate_magnitude',
                    'agents': ['forecast', 'strategy']
                }
            ],
            'timestamp': datetime.now().isoformat(),
            'project_id': 'test'
        }
        
        # 요청 생성
        request = AgentRequest(
            task_id="viz_test_001",
            role=AgentRole.STRATEGY,
            instruction="Generate test dashboard",
            context=mock_context
        )
        
        # 실행
        print("\n2. Generating dashboard...")
        try:
            result = await agent._execute(request)
            print(f"   ✓ Dashboard path: {result['dashboard_path']}")
            print(f"   ✓ Size: {result['dashboard_size'] / 1024:.1f} KB")
            print(f"   ✓ Sections: {result['sections_generated']}")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
    
    asyncio.run(test())

