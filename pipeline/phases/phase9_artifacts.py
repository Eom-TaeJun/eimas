#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 9: Artifact Export (JSON -> HTML/PDF)

Purpose:
    Convert unified JSON output into shareable artifacts and validate
    conversion quality using file-size comparisons.

Input:
    - output_file: str (canonical JSON artifact path)
    - output_dir: Path (base output directory)
    - enable_pdf: bool (default True)

Output:
    - Dict[str, Any]: conversion summary and size-check results
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return 0


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def export_artifacts(
    output_file: str,
    output_dir: Path,
    enable_pdf: bool = True,
) -> Dict[str, Any]:
    """
    [Phase 9] Export HTML/PDF artifacts from canonical JSON and validate by size.
    This function is best-effort and never raises to avoid blocking the pipeline.
    """
    print("\n[Phase 9] Exporting Artifacts (JSON → HTML/PDF)...")

    if not _env_bool("EIMAS_ENABLE_ARTIFACT_EXPORT", True):
        print("      - Skipped (EIMAS_ENABLE_ARTIFACT_EXPORT=false)")
        return {
            "enabled": False,
            "skipped": True,
            "reason": "artifact_export_disabled_by_env",
        }

    if not output_file:
        print("      - Skipped (no output_file)")
        return {
            "enabled": True,
            "skipped": True,
            "reason": "missing_output_file",
        }

    json_path = Path(output_file).expanduser()
    if not json_path.is_absolute():
        candidate_as_is = json_path
        candidate_under_output = Path(output_dir).expanduser() / json_path
        json_path = candidate_as_is if candidate_as_is.exists() else candidate_under_output
    json_path = json_path.resolve()

    if not json_path.exists():
        print(f"      ⚠️ JSON not found: {json_path}")
        return {
            "enabled": True,
            "skipped": True,
            "reason": "json_not_found",
            "json_path": str(json_path),
        }

    json_bytes = _file_size(json_path)
    summary: Dict[str, Any] = {
        "enabled": True,
        "json_path": str(json_path),
        "json_bytes": json_bytes,
        "html_path": "",
        "html_bytes": 0,
        "html_json_ratio": 0.0,
        "html_checks": {},
        "html_conversion_ok": False,
        "pdf_requested": bool(enable_pdf and _env_bool("EIMAS_ENABLE_PDF_EXPORT", True)),
        "pdf_path": "",
        "pdf_bytes": 0,
        "pdf_html_ratio": 0.0,
        "pdf_checks": {},
        "pdf_conversion_ok": False,
        "overall_ok": False,
    }

    html_path: Path | None = None
    try:
        from lib.json_to_html_converter import convert_json_to_html

        html_path = convert_json_to_html(json_path)
    except Exception as exc:
        print(f"      ⚠️ JSON→HTML conversion failed: {type(exc).__name__}: {exc}")

    if html_path and html_path.exists():
        html_bytes = _file_size(html_path)
        html_ratio = _safe_ratio(html_bytes, json_bytes)
        html_checks = {
            "non_empty": html_bytes >= 1024,
            "ratio_reasonable": 0.2 <= html_ratio <= 4.0,
        }
        html_ok = all(html_checks.values())

        summary.update(
            {
                "html_path": str(html_path),
                "html_bytes": html_bytes,
                "html_json_ratio": round(html_ratio, 4),
                "html_checks": html_checks,
                "html_conversion_ok": html_ok,
            }
        )

        print(
            "      ✓ HTML size check: "
            f"json={json_bytes:,}B, html={html_bytes:,}B, ratio={html_ratio:.2f}"
        )
    else:
        print("      ⚠️ HTML artifact missing after conversion")

    pdf_requested = summary["pdf_requested"]
    if pdf_requested and html_path and html_path.exists():
        wkhtmltopdf = shutil.which("wkhtmltopdf")
        if not wkhtmltopdf:
            print("      ⚠️ PDF skipped: wkhtmltopdf not installed")
            summary["pdf_checks"] = {"wkhtmltopdf_installed": False}
        else:
            pdf_path = html_path.with_suffix(".pdf")
            try:
                result = subprocess.run(
                    [
                        wkhtmltopdf,
                        "--enable-local-file-access",
                        "--encoding",
                        "utf-8",
                        "--page-size",
                        "A4",
                        str(html_path),
                        str(pdf_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0 and not pdf_path.exists():
                    print(
                        "      ⚠️ HTML→PDF conversion failed: "
                        f"exit={result.returncode}, stderr={result.stderr.strip()[:200]}"
                    )
                else:
                    pdf_bytes = _file_size(pdf_path)
                    html_bytes = summary["html_bytes"]
                    pdf_ratio = _safe_ratio(pdf_bytes, html_bytes)
                    pdf_checks = {
                        "non_empty": pdf_bytes >= 30 * 1024,
                        "ratio_reasonable": 0.2 <= pdf_ratio <= 20.0,
                        "wkhtmltopdf_installed": True,
                    }
                    pdf_ok = all(pdf_checks.values())
                    summary.update(
                        {
                            "pdf_path": str(pdf_path),
                            "pdf_bytes": pdf_bytes,
                            "pdf_html_ratio": round(pdf_ratio, 4),
                            "pdf_checks": pdf_checks,
                            "pdf_conversion_ok": pdf_ok,
                        }
                    )
                    print(
                        "      ✓ PDF size check: "
                        f"html={html_bytes:,}B, pdf={pdf_bytes:,}B, ratio={pdf_ratio:.2f}"
                    )
            except Exception as exc:
                print(f"      ⚠️ HTML→PDF conversion exception: {type(exc).__name__}: {exc}")

    summary["overall_ok"] = bool(
        summary.get("html_conversion_ok")
        and (
            not summary.get("pdf_requested")
            or summary.get("pdf_conversion_ok")
        )
    )

    if summary["overall_ok"]:
        print("      ✓ Artifact export complete")
    else:
        print("      ⚠️ Artifact export completed with warnings")

    return summary
