#!/usr/bin/env python3
"""
EIMAS Result Validator
======================
Validates EIMAS analysis results for data quality issues.

Usage:
    python cli/validate_result.py
    python cli/validate_result.py --file outputs/eimas_20260213_034124.json
    python cli/validate_result.py --verbose
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import argparse


@dataclass
class ValidationIssue:
    """Single validation issue"""
    severity: str  # "ERROR", "WARNING", "INFO"
    category: str  # "calculation", "missing_data", "range", "consistency"
    field: str
    message: str
    expected: Any = None
    actual: Any = None
    fix_suggestion: Optional[str] = None

    def __str__(self):
        symbol = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}.get(self.severity, "•")
        msg = f"{symbol} [{self.category}] {self.field}: {self.message}"
        if self.expected is not None and self.actual is not None:
            msg += f"\n    Expected: {self.expected}, Actual: {self.actual}"
        if self.fix_suggestion:
            msg += f"\n    💡 Fix: {self.fix_suggestion}"
        return msg


@dataclass
class ValidationReport:
    """Validation report"""
    timestamp: str
    file_path: str
    total_checks: int = 0
    passed_checks: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, severity: str, category: str, field: str, message: str,
                  expected=None, actual=None, fix_suggestion=None):
        """Add validation issue"""
        self.issues.append(ValidationIssue(
            severity=severity,
            category=category,
            field=field,
            message=message,
            expected=expected,
            actual=actual,
            fix_suggestion=fix_suggestion
        ))

    def get_summary(self) -> Dict[str, int]:
        """Get issue summary"""
        summary = {"ERROR": 0, "WARNING": 0, "INFO": 0}
        for issue in self.issues:
            summary[issue.severity] += 1
        return summary

    def print_report(self, verbose: bool = False):
        """Print validation report"""
        print("\n" + "=" * 80)
        print("EIMAS RESULT VALIDATION REPORT")
        print("=" * 80)
        print(f"File: {self.file_path}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Checks: {self.total_checks}")
        print(f"Passed: {self.passed_checks} ({self.passed_checks/self.total_checks*100:.1f}%)")

        summary = self.get_summary()
        print(f"\nIssues Found: {len(self.issues)}")
        print(f"  - Errors: {summary['ERROR']}")
        print(f"  - Warnings: {summary['WARNING']}")
        print(f"  - Info: {summary['INFO']}")

        if self.issues:
            print("\n" + "-" * 80)
            print("ISSUES DETAILS:")
            print("-" * 80)

            # Group by category
            by_category = {}
            for issue in self.issues:
                if issue.category not in by_category:
                    by_category[issue.category] = []
                by_category[issue.category].append(issue)

            for category, issues in sorted(by_category.items()):
                print(f"\n[{category.upper()}]")
                for issue in issues:
                    if verbose or issue.severity == "ERROR":
                        print(f"\n{issue}")
                    else:
                        print(f"  {issue.severity[0]} {issue.field}: {issue.message}")
        else:
            print("\n✅ All validation checks passed!")

        print("\n" + "=" * 80)
        return summary


class EIMASResultValidator:
    """EIMAS result validator"""

    def __init__(self, result_path: str):
        self.result_path = Path(result_path)
        self.data = self._load_result()
        self.report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            file_path=str(self.result_path)
        )

    def _load_result(self) -> Dict:
        """Load result JSON"""
        try:
            with open(self.result_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load result: {e}")
            sys.exit(1)

    def _get_field(self, path: str, default=None):
        """Get nested field value"""
        keys = path.split('.')
        value = self.data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value

    def validate_all(self) -> ValidationReport:
        """Run all validation checks"""
        print("Running validation checks...")

        self.validate_risk_calculation()
        self.validate_required_fields()
        self.validate_data_ranges()
        self.validate_consistency()
        self.validate_gmm_probabilities()
        self.validate_fred_data()

        return self.report

    def validate_risk_calculation(self):
        """Validate risk score calculation"""
        print("  [1/6] Risk calculation...")

        base = self._get_field('base_risk_score', 0)
        micro = self._get_field('microstructure_adjustment', 0)
        bubble = self._get_field('bubble_risk_adjustment', 0)
        extended = self._get_field('extended_data_adjustment', 0)
        final = self._get_field('risk_score', 0)

        self.report.total_checks += 1

        # Calculate expected
        expected = base + micro + bubble + extended
        expected = max(1.0, min(100.0, expected))  # Apply same bounds

        # Check if calculation matches
        diff = abs(final - expected)
        if diff > 0.1:  # Allow small floating point errors
            self.report.add_issue(
                severity="ERROR",
                category="calculation",
                field="risk_score",
                message=f"Risk score calculation mismatch (diff: {diff:.2f})",
                expected=f"base({base:.1f}) + micro({micro:.1f}) + bubble({bubble:.1f}) + extended({extended:.1f}) = {expected:.1f}",
                actual=final,
                fix_suggestion="Check phase2_adjustment.py for cumulative calculation bugs"
            )
        else:
            self.report.passed_checks += 1

        # Check for missing extended_data_adjustment field
        self.report.total_checks += 1
        if extended == 0 and self._get_field('extended_data_adjustment') is None:
            self.report.add_issue(
                severity="WARNING",
                category="missing_data",
                field="extended_data_adjustment",
                message="Extended data adjustment field missing from result",
                fix_suggestion="Add extended_data_adjustment to EIMASResult schema"
            )
        else:
            self.report.passed_checks += 1

    def validate_required_fields(self):
        """Validate required fields exist"""
        print("  [2/6] Required fields...")

        required_fields = {
            'correlation_matrix': 'Correlation matrix for heatmap visualization',
            'correlation_tickers': 'Tickers for correlation analysis',
            'bubble_risk.overall_status': 'Bubble risk status (NONE/WATCH/WARNING/DANGER)',
            'market_quality.avg_liquidity_score': 'Average liquidity score',
            'regime.gmm_probabilities': 'GMM regime probabilities',
        }

        for field_path, description in required_fields.items():
            self.report.total_checks += 1
            value = self._get_field(field_path)

            if value is None or (isinstance(value, (list, dict)) and len(value) == 0):
                self.report.add_issue(
                    severity="WARNING",
                    category="missing_data",
                    field=field_path,
                    message=f"Missing or empty: {description}",
                    fix_suggestion=f"Check if {field_path.split('.')[0]} analysis is running correctly"
                )
            else:
                self.report.passed_checks += 1

    def validate_data_ranges(self):
        """Validate data ranges"""
        print("  [3/6] Data ranges...")

        range_checks = {
            'risk_score': (1.0, 100.0, 'Risk score'),
            'confidence': (0.0, 1.0, 'Confidence'),
            'base_risk_score': (0.0, 100.0, 'Base risk score'),
            'microstructure_adjustment': (-50.0, 50.0, 'Microstructure adjustment'),
            'bubble_risk_adjustment': (0.0, 50.0, 'Bubble risk adjustment'),
        }

        for field, (min_val, max_val, description) in range_checks.items():
            self.report.total_checks += 1
            value = self._get_field(field)

            if value is not None:
                if not (min_val <= value <= max_val):
                    self.report.add_issue(
                        severity="WARNING",
                        category="range",
                        field=field,
                        message=f"{description} out of expected range [{min_val}, {max_val}]",
                        actual=value,
                        fix_suggestion=f"Verify {field} calculation logic"
                    )
                else:
                    self.report.passed_checks += 1
            else:
                self.report.passed_checks += 1  # Skip if missing (caught by required_fields)

    def validate_consistency(self):
        """Validate internal consistency"""
        print("  [4/6] Internal consistency...")

        # Check: modes_agree vs positions
        self.report.total_checks += 1
        full = self._get_field('full_mode_position')
        ref = self._get_field('reference_mode_position')
        agree = self._get_field('modes_agree')

        if full and ref and agree is not None:
            expected_agree = (full == ref)
            if expected_agree != agree:
                self.report.add_issue(
                    severity="ERROR",
                    category="consistency",
                    field="modes_agree",
                    message="modes_agree inconsistent with mode positions",
                    expected=expected_agree,
                    actual=agree,
                    fix_suggestion="Check debate consensus logic"
                )
            else:
                self.report.passed_checks += 1
        else:
            self.report.passed_checks += 1

    def validate_gmm_probabilities(self):
        """Validate GMM probabilities"""
        print("  [5/6] GMM probabilities...")

        probs = self._get_field('regime.gmm_probabilities')
        if probs and isinstance(probs, dict):
            # Check sum to 1.0
            self.report.total_checks += 1
            total = sum(probs.values())
            if abs(total - 1.0) > 0.01:
                self.report.add_issue(
                    severity="WARNING",
                    category="consistency",
                    field="regime.gmm_probabilities",
                    message=f"GMM probabilities don't sum to 1.0 (sum={total:.3f})",
                    fix_suggestion="Normalize probabilities in GMM analysis"
                )
            else:
                self.report.passed_checks += 1

            # Check for hardcoded values
            self.report.total_checks += 1
            if abs(probs.get('Bull', 0) - 0.33) < 0.01 and abs(probs.get('Neutral', 0) - 0.34) < 0.01:
                self.report.add_issue(
                    severity="ERROR",
                    category="consistency",
                    field="regime.gmm_probabilities",
                    message="GMM probabilities appear to be hardcoded (33%, 34%, 33%)",
                    fix_suggestion="Ensure actual GMM analysis results are being used"
                )
            else:
                self.report.passed_checks += 1
        else:
            self.report.total_checks += 2

    def validate_fred_data(self):
        """Validate FRED data"""
        print("  [6/6] FRED data...")

        fred_fields = {
            'fred_summary.cpi_yoy': 'CPI Year-over-Year',
            'fred_summary.core_pce_yoy': 'Core PCE Year-over-Year',
            'fred_summary.fed_funds': 'Fed Funds Rate',
        }

        for field, description in fred_fields.items():
            self.report.total_checks += 1
            value = self._get_field(field)

            if value == 0 or value is None:
                self.report.add_issue(
                    severity="INFO",
                    category="missing_data",
                    field=field,
                    message=f"{description} is zero or missing (may be stale FRED data)",
                    fix_suggestion="Check FRED API connection and data freshness"
                )
            else:
                self.report.passed_checks += 1


def main():
    parser = argparse.ArgumentParser(description="Validate EIMAS analysis results")
    parser.add_argument('--file', '-f', help='Path to result JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all issues including warnings')
    args = parser.parse_args()

    # Find latest result if not specified
    if args.file:
        result_path = args.file
    else:
        outputs_dir = Path(__file__).parent.parent / "outputs"
        files = list(outputs_dir.glob("eimas_*.json"))
        if not files:
            print("❌ No EIMAS result files found in outputs/")
            sys.exit(1)
        result_path = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Using latest result: {result_path.name}")

    # Run validation
    validator = EIMASResultValidator(result_path)
    report = validator.validate_all()

    # Print report
    summary = report.print_report(verbose=args.verbose)

    # Exit code based on errors
    sys.exit(1 if summary['ERROR'] > 0 else 0)


if __name__ == "__main__":
    main()
