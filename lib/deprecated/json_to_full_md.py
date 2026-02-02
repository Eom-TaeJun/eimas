#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from datetime import datetime

def json_to_markdown(data, level=1):
    md = ""
    indent = "  " * (level - 1)
    
    if isinstance(data, dict):
        for key, value in data.items():
            header_prefix = "#" * level
            if isinstance(value, (dict, list)):
                md += f"\n{header_prefix} {key}\n"
                md += json_to_markdown(value, level + 1)
            else:
                md += f"{indent}- **{key}**: {value}\n"
                
                # Check if value is a file path and embed it
                if isinstance(value, str) and (key.endswith("_path") or value.endswith(".md")):
                    # Try to resolve relative to current dir or outputs dir
                    potential_paths = [Path(value), Path("outputs") / Path(value).name]
                    found_path = None
                    for p in potential_paths:
                        if p.exists() and p.is_file():
                            found_path = p
                            break
                    
                    if found_path:
                        try:
                            with open(found_path, "r", encoding="utf-8") as f:
                                embedded_content = f.read()
                            md += f"\n{indent}  > **Embedded Content ({found_path.name}):**\n"
                            # Indent the embedded content for blockquote style
                            quoted_content = "\n".join([f"{indent}  > {line}" for line in embedded_content.splitlines()])
                            md += f"{quoted_content}\n\n"
                        except Exception as e:
                            md += f"{indent}  > *Error reading file: {e}*\n"

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                md += json_to_markdown(item, level)
                md += "\n"
            else:
                md += f"{indent}- {item}\n"
    else:
        md += f"{indent}- {data}\n"
    
    return md

def main():
    if len(sys.argv) < 2:
        print("Usage: python lib/json_to_full_md.py <filename.json>")
        return

    json_filename = sys.argv[1]
    output_dir = Path("outputs")
    json_path = output_dir / json_filename
    
    if not json_path.exists():
        # Try relative path if not in outputs
        json_path = Path(json_filename)
        if not json_path.exists():
            print(f"File not found: {json_filename}")
            return

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

    md_content = f"# EIMAS Full Data Report: {json_filename}\n"
    md_content += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += json_to_markdown(data)
    
    md_filename = json_path.stem + "_full.md"
    md_path = json_path.parent / md_filename
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"✓ Full report saved to: {md_path}")

if __name__ == "__main__":
    main()
