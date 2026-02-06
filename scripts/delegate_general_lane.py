#!/usr/bin/env python3
"""
Dispatch a General Lane WORK_ORDER to Claude Code (non-interactive).

Usage:
    python3 scripts/delegate_general_lane.py --work-order work_orders/GEN-101.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv


SYSTEM_PROMPT = """You are the General Lane execution worker for EIMAS.

Rules:
1. Execute only explicit instructions in [WORK_ORDER].
2. Do not change architecture, contracts, or design decisions.
3. Do not edit files outside scope_files unless required by imports/tests and explicitly report them.
4. Run listed validation commands and report exact outcomes.
5. If design ambiguity appears, stop and report BLOCKED with the reason.

Output format:
- changed_files: list
- summary: bullet list
- validation_results: command + pass/fail + key output
- blockers: list
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch WORK_ORDER to Claude Code")
    parser.add_argument(
        "--work-order",
        required=True,
        help="Path to WORK_ORDER markdown file",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CLAUDE_GENERAL_MODEL", "claude-sonnet-4-5-20250929"),
        help="Claude model name",
    )
    parser.add_argument(
        "--permission-mode",
        default=os.getenv("CLAUDE_GENERAL_PERMISSION_MODE", "acceptEdits"),
        choices=["acceptEdits", "bypassPermissions", "default", "delegate", "dontAsk", "plan"],
        help="Claude Code permission mode",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=float(os.getenv("CLAUDE_GENERAL_MAX_BUDGET_USD", "0") or 0),
        help="Optional API spend cap for one run (0 disables cap)",
    )
    parser.add_argument(
        "--allowed-tools",
        default=os.getenv("CLAUDE_GENERAL_ALLOWED_TOOLS", ""),
        help='Optional allowed tools string, e.g. "Bash(git:*) Edit"',
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/claude_general",
        help="Directory for request/response artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command and prompt path without calling Claude",
    )
    return parser.parse_args()


def load_env(repo_root: Path) -> None:
    env_file = repo_root / ".env"
    load_dotenv(env_file if env_file.exists() else None)


def read_work_order(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"WORK_ORDER not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_work_id(work_order: str) -> str:
    match = re.search(r"^id:\s*([A-Za-z0-9._-]+)\s*$", work_order, flags=re.MULTILINE)
    return match.group(1) if match else "GEN-UNKNOWN"


def extract_scope_files(work_order: str) -> List[str]:
    lines = work_order.splitlines()
    scope: List[str] = []
    in_scope = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "scope_files:":
            in_scope = True
            continue
        if in_scope and re.match(r"^[a-z_]+:\s*$", stripped):
            break
        if in_scope and stripped.startswith("- "):
            scope.append(stripped[2:].strip())
    return scope


def build_prompt(repo_root: Path, work_order_text: str, scope_files: List[str]) -> str:
    scope_block = "\n".join(f"- {f}" for f in scope_files) if scope_files else "- (not specified)"
    return f"""Repository root: {repo_root}

You are executing a General Lane task in WSL. Follow the WORK_ORDER exactly.

Scope files:
{scope_block}

WORK_ORDER:
{work_order_text}
"""


def build_command(
    model: str,
    permission_mode: str,
    max_budget_usd: float,
    allowed_tools: str,
    repo_root: Path,
    prompt: str,
) -> List[str]:
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
        "--permission-mode",
        permission_mode,
        "--add-dir",
        str(repo_root),
        "--append-system-prompt",
        SYSTEM_PROMPT,
    ]
    if max_budget_usd > 0:
        cmd.extend(["--max-budget-usd", str(max_budget_usd)])
    if allowed_tools:
        cmd.extend(["--allowed-tools", allowed_tools])
    cmd.append(prompt)
    return cmd


def run_claude(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def ensure_api_key() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in WSL shell or define it in .env."
        )


def save_artifacts(
    base_dir: Path,
    work_id: str,
    prompt: str,
    cmd: List[str],
    return_code: int,
    stdout: str,
    stderr: str,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{ts}_{work_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "request_prompt.md").write_text(prompt, encoding="utf-8")
    (out_dir / "response_stdout.json").write_text(stdout or "", encoding="utf-8")
    (out_dir / "response_stderr.log").write_text(stderr or "", encoding="utf-8")

    meta = {
        "timestamp": ts,
        "work_id": work_id,
        "command": cmd,
        "return_code": return_code,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    load_env(repo_root)

    work_order_path = Path(args.work_order).resolve()
    work_order_text = read_work_order(work_order_path)
    work_id = extract_work_id(work_order_text)
    scope_files = extract_scope_files(work_order_text)
    prompt = build_prompt(repo_root, work_order_text, scope_files)

    cmd = build_command(
        model=args.model,
        permission_mode=args.permission_mode,
        max_budget_usd=args.max_budget_usd,
        allowed_tools=args.allowed_tools,
        repo_root=repo_root,
        prompt=prompt,
    )

    if args.dry_run:
        print("DRY RUN")
        print("WORK_ORDER:", work_order_path)
        print("MODEL:", args.model)
        print(
            "COMMAND:",
            "claude --print --output-format json",
            f"--model {args.model}",
            f"--permission-mode {args.permission_mode}",
            f"--add-dir {repo_root}",
            "... <system/prompt omitted>",
        )
        return 0

    ensure_api_key()

    return_code, stdout, stderr = run_claude(cmd)
    output_dir = save_artifacts(
        base_dir=(repo_root / args.output_dir),
        work_id=work_id,
        prompt=prompt,
        cmd=cmd,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
    )

    print(f"WORK_ORDER: {work_order_path}")
    print(f"MODEL: {args.model}")
    print(f"RETURN_CODE: {return_code}")
    print(f"ARTIFACTS: {output_dir}")

    if return_code != 0:
        print("Claude run failed. Check response_stderr.log for details.", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
