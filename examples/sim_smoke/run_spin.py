#!/usr/bin/env python3
"""Sim-smoke: run all registered spin experiments through SimulatorSoc.

Loads the spin_demo device, constructs a SimulatorSoc, and attempts to
build + compile + acquire + fit every registered spin experiment. Reports
a readiness matrix.

NOTE: The spin_demo attaches a fabricated soccfg (soccfg_v2_spin.json) that
may not match a real board. This test verifies the SOFTWARE pipeline
(construction, compilation, data flow) — not the physics.

spin_demo is a single-dot device (dot q0, plunger p0, barrier b0), so each
experiment's element-name arguments are bound to those elements; two-dot
experiments get q0 as both dots (a pipeline check, not a two-qubit gate).

Usage:
    python run_spin.py              # text report
    python run_spin.py --json       # machine-readable JSON
    python run_spin.py --verbose    # show tracebacks on failure
"""

from __future__ import annotations

import inspect
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

#: spin_demo's element for each element-name argument a spin experiment may take.
SPIN_DEMO_ELEMENTS = {
    "dot_name": "q0",
    "control_dot": "q0",
    "target_dot": "q0",
    "barrier_name": "b0",
    "plunger_name": "p0",
}


def _element_kwargs(cls: type) -> dict[str, str]:
    """The spin_demo element bindings for the element-name parameters ``cls`` accepts."""
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in SPIN_DEMO_ELEMENTS.items() if k in params}


def run_all(*, verbose: bool = False, fits: dict | None = None) -> list[dict]:
    """Run every registered spin experiment through SimulatorSoc. Returns a list of result dicts.

    Pass a dict as ``fits`` to also collect each experiment's ``FitResult`` (name → fit) — the
    fit-contract cross-check uses it (#174).
    """
    from strumento import Device
    from strumento.core.simulator import SimulatorSoc

    dev_path = str(REPO / "devices" / "spin_demo" / "device.yaml")

    try:
        dev = Device.load(dev_path)
    except Exception as e:
        print(f"WARNING: Could not load spin_demo: {e}")
        print("Spin sim-smoke requires a v2 soccfg snapshot. Skipping.")
        return [{"experiment": "*", "status": "SKIP",
                 "error": f"Device load failed: {e}"}]

    try:
        soc = SimulatorSoc.from_device(dev, truth={
            "qubit_frequency": 5000.0,
            "T1": 50.0,
            "T2": 20.0,
            "pi_gain": 0.4,
        })
    except Exception as e:
        print(f"WARNING: Could not create SimulatorSoc for spin_demo: {e}")
        return [{"experiment": "*", "status": "SKIP",
                 "error": f"SimulatorSoc failed: {e}"}]

    from strumento.packs.spin import EXPERIMENTS

    results = []
    for name, cls in sorted(EXPERIMENTS.items()):
        entry = {"experiment": name, "class": cls.__name__}
        t0 = time.monotonic()

        try:
            exp = cls(dev, **_element_kwargs(cls))
            entry["construct"] = "OK"

            outcome = exp.go(soc, reps=10, analyze=True, display=False)
            entry["acquire"] = "OK"

            if outcome.fit is not None:
                if fits is not None:
                    fits[name] = outcome.fit
                entry["fit"] = "OK" if outcome.fit.success else "FAIL"
                entry["derived"] = list(outcome.fit.derived.keys())
            else:
                entry["fit"] = "SKIP"
                entry["derived"] = []

            entry["status"] = "PASS" if entry["fit"] in ("OK", "SKIP") else "FIT_FAIL"

        except Exception as e:
            entry["status"] = "ERROR"
            entry["error_type"] = type(e).__name__
            entry["error"] = str(e)[:200]
            if "construct" not in entry:
                entry["construct"] = "FAIL"
            elif "acquire" not in entry:
                entry["acquire"] = "FAIL"
            else:
                entry["fit"] = "FAIL"

            if verbose:
                entry["traceback"] = traceback.format_exc()

        entry["elapsed_s"] = round(time.monotonic() - t0, 3)
        results.append(entry)

    return results


def print_report(results: list[dict]) -> None:
    """Print a human-readable readiness matrix."""
    if results and results[0].get("experiment") == "*":
        print(f"\n  spin sim-smoke: SKIP — {results[0].get('error', 'unknown')}\n")
        return

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    fit_fail = sum(1 for r in results if r["status"] == "FIT_FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"\n{'=' * 72}")
    print(f"  spin sim-smoke: {passed}/{total} PASS, {fit_fail} FIT_FAIL, {errors} ERROR")
    print("  Device: spin_demo")
    print(f"{'=' * 72}\n")

    for status_label, status_code in [("PASS", "PASS"), ("FIT_FAIL", "FIT_FAIL"), ("ERROR", "ERROR")]:
        group = [r for r in results if r["status"] == status_code]
        if not group:
            continue
        print(f"  [{status_label}] ({len(group)})")
        for r in group:
            line = f"    {r['experiment']:<35s}"
            if status_code == "PASS":
                derived = ", ".join(r.get("derived", []))
                if derived:
                    line += f"  derived: {derived}"
            elif status_code == "ERROR":
                line += f"  {r.get('error_type', '?')}: {r.get('error', '?')[:80]}"
            print(line)
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="spin sim-smoke test")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--verbose", action="store_true", help="Include tracebacks on failure")
    args = parser.parse_args()

    results = run_all(verbose=args.verbose)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

    fit_fails = sum(1 for r in results if r["status"] == "FIT_FAIL")
    sys.exit(1 if fit_fails > 0 else 0)


if __name__ == "__main__":
    main()
