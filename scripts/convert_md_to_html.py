import re
import sys
import os

def convert_md_to_html(md_content, output_path):
    css = """
    <style>
        :root { --bg-primary: #0d1117; --bg-secondary: #161b22; --text-primary: #c9d1d9; --text-secondary: #8b949e; --accent-green: #3fb950; --accent-red: #f85149; --accent-yellow: #d29922; --accent-blue: #58a6ff; --border: #30363d; }
        body { font-family: sans-serif; background: var(--bg-primary); color: var(--text-primary); padding: 40px; line-height: 1.6; }
        .container { max-width: 1000px; margin: 0 auto; }
        h1 { border-bottom: 1px solid var(--border); padding-bottom: 20px; }
        h2 { color: var(--accent-blue); border-bottom: 1px solid var(--border); margin-top: 40px; }
        .card { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        li { border-bottom: 1px solid var(--border); padding: 8px 0; }
        .badge { padding: 2px 8px; border-radius: 12px; font-weight: bold; font-size: 0.85em; margin-left: 8px; }
        .badge-bullish { background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }
        .badge-bearish { background: rgba(248, 81, 73, 0.2); color: var(--accent-red); }
        .badge-neutral { background: rgba(210, 153, 34, 0.2); color: var(--accent-yellow); }
        .reasoning-step { background: #21262d; border-left: 4px solid #bc8cff; padding: 15px; margin: 15px 0; }
        .reasoning-agent { color: #bc8cff; font-weight: bold; display: block; margin-bottom: 5px; }
    </style>
    """

    html_parts = ["<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Report</title>" + css + "</head><body><div class='container'>"]

    lines = md_content.splitlines()
    current_section = None
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith('# '):
            html_parts.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith('**Generated**'):
            html_parts.append(f"<p style='color:#8b949e'>{line}</p>")
        elif line.startswith('## '):
            if in_list: html_parts.append("</ul>"); in_list = False
            if current_section: html_parts.append("</div>")
            section_title = line[3:]
            html_parts.append(f"<div class='card'><h2>{section_title}</h2>")
            current_section = section_title
        elif line.startswith('### '):
            if in_list: html_parts.append("</ul>"); in_list = False
            html_parts.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith('- '):
            if not in_list: html_parts.append("<ul>"); in_list = True
            content = line[2:]
            if "BULLISH" in content: content = content.replace("BULLISH", "<span class='badge badge-bullish'>BULLISH</span>")
            if "BEARISH" in content: content = content.replace("BEARISH", "<span class='badge badge-bearish'>BEARISH</span>")
            if "NEUTRAL" in content: content = content.replace("NEUTRAL", "<span class='badge badge-neutral'>NEUTRAL</span>")
            
            if current_section and "Reasoning Chain" in current_section:
                match = re.match(r'\*\*(.*?)\*\*\s*\((.*?)\): (.*)', content)
                if match:
                    agent, conf, text = match.groups()
                    html_parts.append(f"<div class='reasoning-step'><span class='reasoning-agent'>{agent} ({conf})</span>{text}</div>")
                    continue
            
            html_parts.append(f"<li>{content}</li>")
        else:
            if in_list: html_parts.append("</ul>"); in_list = False
            html_parts.append(f"<p>{line}</p>")

    if in_list: html_parts.append("</ul>")
    if current_section: html_parts.append("</div>")
    html_parts.append("</div></body></html>")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_parts))
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Convert Markdown report to HTML')
    parser.add_argument('--input', '-i', help='Input Markdown file path')
    parser.add_argument('--output', '-o', help='Output HTML file path')
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    # If no input specified, find the latest md file in outputs
    if not input_path:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_dir = os.path.join(base_dir, "outputs")
        md_files = glob.glob(os.path.join(outputs_dir, "*.md"))
        if md_files:
            input_path = max(md_files, key=os.path.getctime)
            print(f"Auto-detected latest input: {input_path}")
    
    if input_path and os.path.exists(input_path):
        if not output_path:
            # Default output name based on input name
            output_path = input_path.replace('.md', '.html')
            
        with open(input_path, 'r', encoding='utf-8') as f:
            convert_md_to_html(f.read(), output_path)
    else:
        print("Input file not found or not specified")
