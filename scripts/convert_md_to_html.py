import re
import os
import html


def _inline_format(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r"<a href='\2' target='_blank' rel='noopener noreferrer'>\1</a>",
        escaped,
    )
    escaped = escaped.replace(
        "BULLISH", "<span class='badge badge-bullish'>BULLISH</span>"
    ).replace(
        "BEARISH", "<span class='badge badge-bearish'>BEARISH</span>"
    ).replace(
        "NEUTRAL", "<span class='badge badge-neutral'>NEUTRAL</span>"
    )
    return escaped


def _parse_table_row(line: str) -> list[str]:
    row = line.strip().strip("|")
    return [cell.strip() for cell in row.split("|")]


def _is_table_separator(line: str) -> bool:
    stripped = line.strip().strip("|").replace(" ", "")
    return bool(stripped) and all(ch in "-:|" for ch in stripped)


def convert_md_to_html(md_content, output_path):
    css = """
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-blue: #58a6ff;
            --border: #30363d;
            --code-bg: #11161f;
        }
        body {
            font-family: 'Noto Sans KR', 'Segoe UI', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 36px;
            line-height: 1.65;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 20px; }
        h2 { color: var(--accent-blue); border-bottom: 1px solid var(--border); margin-top: 30px; padding-bottom: 8px; }
        h3 { margin-top: 18px; }
        p { margin: 10px 0; }
        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px;
            margin-bottom: 18px;
        }
        ul { margin: 8px 0 8px 20px; }
        li { margin: 5px 0; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0 16px 0;
            font-size: 0.95rem;
            table-layout: auto;
        }
        th, td {
            border: 1px solid var(--border);
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
            white-space: normal;
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        th { background: #1e2631; color: #dce6f2; }
        table.table-compact {
            font-size: 0.84rem;
        }
        table.table-compact th,
        table.table-compact td {
            padding: 6px 7px;
        }
        table.table-wide {
            table-layout: fixed;
        }
        code {
            background: var(--code-bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 1px 5px;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            font-size: 0.9em;
        }
        pre {
            background: var(--code-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            overflow-x: auto;
        }
        pre code {
            border: none;
            background: transparent;
            padding: 0;
        }
        blockquote {
            margin: 10px 0;
            padding: 8px 12px;
            border-left: 4px solid var(--accent-blue);
            background: #1a2430;
            color: #d7e6fa;
        }
        figure {
            margin: 12px 0 18px 0;
            padding: 10px;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: #131a22;
        }
        figure img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            display: block;
            margin: 0 auto;
        }
        figure figcaption {
            margin-top: 8px;
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-align: center;
        }
        a { color: var(--accent-blue); }
        hr { border: 0; border-top: 1px solid var(--border); margin: 18px 0; }
        .badge {
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.82em;
            margin-left: 3px;
        }
        .badge-bullish { background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }
        .badge-bearish { background: rgba(248, 81, 73, 0.2); color: var(--accent-red); }
        .badge-neutral { background: rgba(210, 153, 34, 0.2); color: var(--accent-yellow); }
    </style>
    """

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Report</title>"
        + css
        + "</head><body><div class='container'>"
    ]

    lines = md_content.splitlines()
    i = 0
    in_list = False
    in_olist = False
    in_code = False
    in_blockquote = False
    current_section = False

    def close_list():
        nonlocal in_list
        if in_list:
            html_parts.append("</ul>")
            in_list = False

    def close_olist():
        nonlocal in_olist
        if in_olist:
            html_parts.append("</ol>")
            in_olist = False

    def close_blockquote():
        nonlocal in_blockquote
        if in_blockquote:
            html_parts.append("</blockquote>")
            in_blockquote = False

    def close_section():
        nonlocal current_section
        if current_section:
            html_parts.append("</div>")
            current_section = False

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if in_code:
            if stripped.startswith("```"):
                html_parts.append("</code></pre>")
                in_code = False
            else:
                html_parts.append(html.escape(raw))
            i += 1
            continue

        if not stripped:
            close_list()
            close_olist()
            close_blockquote()
            i += 1
            continue

        if stripped.startswith("```"):
            close_list()
            close_olist()
            close_blockquote()
            html_parts.append("<pre><code>")
            in_code = True
            i += 1
            continue

        if stripped.startswith("|"):
            close_list()
            close_olist()
            close_blockquote()
            table_lines = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                table_lines.append(lines[j].strip())
                j += 1

            if table_lines:
                header = _parse_table_row(table_lines[0])
                data_start = 1
                if len(table_lines) > 1 and _is_table_separator(table_lines[1]):
                    data_start = 2

                classes = []
                if len(header) >= 8:
                    classes.append("table-compact")
                if len(header) >= 10:
                    classes.append("table-wide")
                class_attr = f" class='{' '.join(classes)}'" if classes else ""

                html_parts.append(f"<table{class_attr}><thead><tr>")
                for cell in header:
                    html_parts.append(f"<th>{_inline_format(cell)}</th>")
                html_parts.append("</tr></thead><tbody>")

                for row_line in table_lines[data_start:]:
                    row_cells = _parse_table_row(row_line)
                    html_parts.append("<tr>")
                    for idx, cell in enumerate(row_cells):
                        tag = "th" if idx == 0 and len(row_cells) == 1 else "td"
                        html_parts.append(f"<{tag}>{_inline_format(cell)}</{tag}>")
                    html_parts.append("</tr>")

                html_parts.append("</tbody></table>")
            i = j
            continue

        if stripped.startswith("# "):
            close_list()
            close_olist()
            close_blockquote()
            close_section()
            html_parts.append(f"<h1>{_inline_format(stripped[2:])}</h1>")
            i += 1
            continue

        if stripped.startswith("## "):
            close_list()
            close_olist()
            close_blockquote()
            close_section()
            html_parts.append(f"<div class='card'><h2>{_inline_format(stripped[3:])}</h2>")
            current_section = True
            i += 1
            continue

        if stripped.startswith("### "):
            close_list()
            close_olist()
            close_blockquote()
            html_parts.append(f"<h3>{_inline_format(stripped[4:])}</h3>")
            i += 1
            continue

        if stripped == "---":
            close_list()
            close_olist()
            close_blockquote()
            html_parts.append("<hr>")
            i += 1
            continue

        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            close_list()
            close_olist()
            close_blockquote()
            alt = html.escape(image_match.group(1).strip())
            src = html.escape(image_match.group(2).strip())
            html_parts.append(
                "<figure>"
                f"<img src='{src}' alt='{alt}'>"
                f"<figcaption>{alt or src}</figcaption>"
                "</figure>"
            )
            i += 1
            continue

        if stripped.startswith(">"):
            close_list()
            close_olist()
            if not in_blockquote:
                html_parts.append("<blockquote>")
                in_blockquote = True
            quote_text = stripped[1:].strip()
            html_parts.append(f"<p>{_inline_format(quote_text)}</p>")
            i += 1
            continue
        close_blockquote()

        if stripped.startswith("- "):
            close_olist()
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            content = stripped[2:].strip()
            html_parts.append(f"<li>{_inline_format(content)}</li>")
            i += 1
            continue

        ordered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ordered_match:
            close_list()
            if not in_olist:
                html_parts.append("<ol>")
                in_olist = True
            content = ordered_match.group(1).strip()
            html_parts.append(f"<li>{_inline_format(content)}</li>")
            i += 1
            continue

        close_list()
        close_olist()
        html_parts.append(f"<p>{_inline_format(stripped)}</p>")
        i += 1

    close_list()
    close_olist()
    close_blockquote()
    close_section()
    html_parts.append("</div></body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
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
