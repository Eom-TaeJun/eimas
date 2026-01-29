import json
import sys
import os
from datetime import datetime

def generate_agent_dashboard(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract Data
    debate_enhanced = data.get('debate_consensus', {}).get('enhanced', {})
    interpretation = debate_enhanced.get('interpretation', {})
    methodology = debate_enhanced.get('methodology', {})
    reasoning_chain = data.get('reasoning_chain', [])
    
    # CSS
    css = """
    <style>
        :root {
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-purple: #a855f7;
            --border: #334155;
        }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text-main);
            margin: 0;
            padding: 40px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        
        h1, h2, h3 { color: var(--text-main); margin-top: 0; }
        h1 { border-bottom: 1px solid var(--border); padding-bottom: 20px; font-size: 2.5rem; }
        
        /* Agent Cards */
        .agent-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border);
        }
        .role-badge {
            background: var(--accent-blue);
            color: white;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        .stance-bullish { color: var(--accent-green); }
        .stance-bearish { color: var(--accent-red); }
        .stance-neutral { color: var(--text-sub); }
        
        /* Reasoning Chain */
        .timeline {
            position: relative;
            padding-left: 30px;
            border-left: 2px solid var(--border);
        }
        .step {
            position: relative;
            margin-bottom: 30px;
            background: var(--bg-card);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        .step::before {
            content: '';
            position: absolute;
            left: -36px;
            top: 25px;
            width: 10px;
            height: 10px;
            background: var(--accent-purple);
            border-radius: 50%;
            border: 2px solid var(--bg-dark);
        }
        .step-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .step-agent { font-weight: bold; color: var(--accent-purple); font-size: 1.1rem; }
        .step-conf { color: var(--text-sub); font-size: 0.9rem; }
        
        .code-block {
            background: #0f172a;
            padding: 10px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.9rem;
            color: #cbd5e1;
            margin-top: 10px;
        }
    </style>
    """.format()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>EIMAS Agent Intelligence Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        {css}
    </head>
    <body>
        <div class="container">
            <h1>🧠 EIMAS Agent Intelligence</h1>
            <p style="color:#94a3b8; margin-bottom:40px">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

            <!-- Section 1: The Council -->
            <h2>1. The Council (Multi-LLM Debate)</h2>
            <p style="margin-bottom:20px; color:#94a3b8">Three specialized AI agents debated the market interpretation.</p>
            <div class="agent-grid">
    """.format()

    # Agent Cards
    schools = interpretation.get('school_interpretations', [])
    for school in schools:
        role = school.get('school', 'Agent')
        stance = school.get('stance', 'NEUTRAL').upper()
        stance_class = f"stance-{stance.lower()}"
        reasoning_list = school.get('reasoning', [])
        if isinstance(reasoning_list, str): reasoning_list = [reasoning_list]

        html += f"""
        <div class="card">
            <div class="card-header">
                <span class="role-badge">{role}</span>
                <span style="font-weight:bold" class="{stance_class}">{stance}</span>
            </div>
            <ul style="padding-left:20px; color:#cbd5e1">
        """.format()
        for point in reasoning_list[:3]: # Limit to 3 points
            html += f"<li>{point}</li>"
        html += "</ul></div>"

    html += """
            </div>

            <!-- Section 2: Consensus -->
            <div class="card" style="border-left: 4px solid var(--accent-green)">
                <h2>🏛️ Final Consensus</h2>
                <div style="display:flex; gap:40px">
                    <div>
                        <h3 style="color:var(--text-sub); font-size:0.9rem">VERDICT</h3>
                        <div style="font-size:2rem; font-weight:bold; color:var(--accent-green)">
                            {verdict}
                        </div>
                    </div>
                    <div>
                        <h3 style="color:var(--text-sub); font-size:0.9rem">CONFIDENCE</h3>
                        <div style="font-size:2rem; font-weight:bold; color:white">
                            {conf:.0%}
                        </div>
                    </div>
                </div>
                <div style="margin-top:20px">
                    <h3 style="color:var(--text-sub); font-size:0.9rem">CONSENSUS POINTS</h3>
                    <ul style="padding-left:20px; color:#cbd5e1">
    """.format(
        verdict=interpretation.get('recommended_action', 'NEUTRAL'),
        conf=interpretation.get('confidence', 0.5)
    )

    for point in interpretation.get('consensus_points', []):
        html += f"<li>{point}</li>"
    
    html += """
                    </ul>
                </div>
            </div>

            <!-- Section 3: Reasoning Chain -->
            <h2 style="margin-top:60px">2. Reasoning Chain (Traceability)</h2>
            <div class="timeline">
    """.format()

    for step in reasoning_chain:
        html += f"""
        <div class="step">
            <div class="step-header">
                <span class="step-agent">{step.get('agent', 'Unknown')}</span>
                <span class="step-conf">Confidence: {step.get('confidence', 0):.1f}%</span>
            </div>
            <div style="color:#e2e8f0; margin-bottom:10px">
                <strong>Input:</strong> {step.get('input')}
            </div>
            <div style="color:#94a3b8; font-size:0.95rem">
                <strong>Output:</strong> {step.get('output')}
            </div>
        """.format()
        if step.get('key_factors'):
            html += "<div class='code-block'>Key Factors:\n"
            for factor in step['key_factors']:
                html += f"• {factor}\n"
            html += "</div>"
        
        html += "</div>"

    html += """
            </div>
        </div>
    </body>
    </html>
    """.format()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Agent Dashboard saved to: {output_path}")

if __name__ == "__main__":
    input_file = "outputs/integrated_20260128_184114.json"
    output_file = "outputs/reports/agent_dashboard_20260128.html"
    
    if os.path.exists(input_file):
        generate_agent_dashboard(input_file, output_file)
    else:
        print(f"Input file not found: {input_file}")
