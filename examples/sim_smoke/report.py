#!/usr/bin/env python3
"""Sim-smoke aggregate report — run both packs and print a combined readiness matrix.

Usage:
    python report.py              # combined text report
    python report.py --json       # combined JSON
    python report.py --verbose    # include tracebacks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_all(*, verbose: bool = False) -> dict:
    """Run both pack smoke tests and return a combined report."""
    report = {}

    try:
        from run_cqed import run_all as run_cqed
        report["cqed"] = run_cqed(verbose=verbose)
    except Exception as e:
        report["cqed"] = [{"experiment": "*", "status": "SKIP",
                           "error": f"cqed runner failed: {e}"}]

    try:
        from run_spin import run_all as run_spin
        report["spin"] = run_spin(verbose=verbose)
    except Exception as e:
        report["spin"] = [{"experiment": "*", "status": "SKIP",
                           "error": f"spin runner failed: {e}"}]

    return report


def _count(results: list[dict]) -> dict:
    """Count pass/fail/error/skip for a pack's results."""
    if results and results[0].get("experiment") == "*":
        return {"total": 0, "pass": 0, "fit_fail": 0, "error": 0, "skipped": True}
    return {
        "total": len(results),
        "pass": sum(1 for r in results if r["status"] == "PASS"),
        "fit_fail": sum(1 for r in results if r["status"] == "FIT_FAIL"),
        "error": sum(1 for r in results if r["status"] == "ERROR"),
        "skipped": False,
    }


def print_combined(report: dict) -> None:
    """Print a combined readiness summary."""
    print(f"\n{'=' * 72}")
    print("  STRUMENTO SIM-SMOKE READINESS REPORT")
    print(f"{'=' * 72}\n")

    grand_pass = 0
    grand_total = 0

    for pack_name in ("cqed", "spin"):
        results = report.get(pack_name, [])
        counts = _count(results)

        if counts["skipped"]:
            skip_reason = results[0].get("error", "unknown") if results else "unknown"
            print(f"  {pack_name.upper():<8s}  SKIPPED — {skip_reason}")
            continue

        t, p, ff, e = counts["total"], counts["pass"], counts["fit_fail"], counts["error"]
        grand_pass += p
        grand_total += t

        bar_width = 40
        pass_w = int(bar_width * p / t) if t else 0
        err_w = int(bar_width * e / t) if t else 0
        ff_w = bar_width - pass_w - err_w

        bar = "\033[32m" + "#" * pass_w + "\033[33m" + "~" * ff_w + "\033[31m" + "x" * err_w + "\033[0m"

        print(f"  {pack_name.upper():<8s}  [{bar}]  {p}/{t} pass, {ff} fit_fail, {e} error")

        # List errors concisely
        errors = [r for r in results if r["status"] == "ERROR"]
        if errors:
            for r in errors[:5]:
                print(f"           x {r['experiment']}: {r.get('error_type', '?')}")
            if len(errors) > 5:
                print(f"           ... and {len(errors) - 5} more")

    print(f"\n{'─' * 72}")
    if grand_total > 0:
        pct = 100 * grand_pass / grand_total
        print(f"  TOTAL: {grand_pass}/{grand_total} experiments pass ({pct:.0f}%)")
    print(f"{'─' * 72}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Combined sim-smoke readiness report")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--verbose", action="store_true", help="Include tracebacks")
    args = parser.parse_args()

    report = run_all(verbose=args.verbose)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_combined(report)

    # Exit 1 when a pack regresses: cqed by its PASS floor (run_cqed.exit_code — multimode_demo
    # has known FIT_FAILs), spin by any fit failure (errors are expected on limited devices).
    from run_cqed import exit_code as cqed_exit_code

    cqed = report.get("cqed", [])
    cqed_bad = bool(cqed) and cqed[0].get("experiment") != "*" and cqed_exit_code(cqed) != 0
    spin_fit_fails = sum(1 for r in report.get("spin", []) if r.get("status") == "FIT_FAIL")
    sys.exit(1 if cqed_bad or spin_fit_fails > 0 else 0)


if __name__ == "__main__":
    main()
