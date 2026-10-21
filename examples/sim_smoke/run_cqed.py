#!/usr/bin/env python3
"""Sim-smoke: run all registered cqed experiments through SimulatorSoc.

Loads the cqed exemplar, ``devices/multimode_demo`` (its fabricated v2 snapshot
``devices/_fixtures/soccfg_v2_cqed.json`` has the separate drive, probe, cavity and coupler lines
the cqed experiments need), constructs a SimulatorSoc, and attempts to build + compile + acquire
+ fit every registered cqed experiment with its defaults (plus the swap pair the swap
experiments must be given on this device). Reports a readiness matrix: which
experiments are green (full pipeline works), which fail at construction (missing
wiring/calibration for this device instance), and which fail at acquire or fit — each failure
with its reason.

``devices/loopback_demo`` is not used here: it is the single-channel v2 testbench instance (one
generator, one readout), which assembles real v2 programs but cannot run the multi-line cqed
experiments.

Statuses:
    PASS      construct + acquire + fit all succeeded (fit.success is True)
    FIT_FAIL  the pipeline ran but the fit did not succeed
    ERROR     an exception at construct, acquire or fit (the phase and message are reported)
    SKIP      the analysis returned no fit at all — never counted as a pass

Exit code:
    0  PASS count >= the floor (PASS_FLOOR, or --min-pass) and nothing was SKIPped
    1  the PASS count fell below the floor, or an experiment was SKIPped

Usage:
    python run_cqed.py                  # text report
    python run_cqed.py --json           # machine-readable JSON
    python run_cqed.py --verbose        # include tracebacks on failure
    python run_cqed.py --min-pass 50    # demand a different PASS floor
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_DEVICE = REPO / "devices" / "multimode_demo" / "device.yaml"

#: The PASS count measured on multimode_demo when this smoke moved onto it (#131), with the
#: settings below (reps=10, the truth dict in run_all). A regression drops below it.
PASS_FLOOR = 46

REPS = 10

#: The targets an experiment cannot infer on multimode_demo: it calibrates two coupler swaps, so a
#: swap experiment names the one it plays. Every other experiment runs on its defaults.
TARGET_KWARGS = {
    "swap_rabi": {"pair": "cavity-S1"},
    "swap_spectroscopy": {"pair": "cavity-S1"},
    "storage_t1": {"mode_name": "S1"},
}


def _one_line(text: str, limit: int = 200) -> str:
    return " ".join(str(text).split())[:limit]


def run_all(*, verbose: bool = False, device: str | Path = DEFAULT_DEVICE,
            fits: dict | None = None) -> list[dict]:
    """Run every registered cqed experiment through SimulatorSoc. Returns a list of result dicts.

    Every non-PASS entry carries a one-line ``reason``. Pass a dict as ``fits`` to also collect
    each experiment's ``FitResult`` (name → fit) — the fit-contract cross-check uses it (#174).
    """
    from strumento import Device
    from strumento.core.simulator import SimulatorSoc

    dev = Device.load(str(device))
    soc = SimulatorSoc.from_device(dev, truth={
        "qubit_frequency": dev.calib.qubit.f_ge,
        "T1": dev.calib.qubit.T1,
        "T2": 60.0,       # synthetic T2
        "pi_gain": 8192,   # from calibration
        "readout_frequency": dev.calib.readout.frequency,
    })

    from strumento.packs.cqed import EXPERIMENTS

    results = []
    for name, cls in sorted(EXPERIMENTS.items()):
        entry = {"experiment": name, "class": cls.__name__}
        t0 = time.monotonic()

        try:
            # Stage 1: construct
            exp = cls(dev, **TARGET_KWARGS.get(name, {}))
            entry["construct"] = "OK"

            # Stage 2: build + compile + acquire
            outcome = exp.go(soc, reps=REPS, analyze=True, display=False, progress=False)
            entry["acquire"] = "OK"

            # Stage 3: fit
            if outcome.fit is None:
                entry["fit"] = "SKIP"
                entry["derived"] = []
                entry["status"] = "SKIP"
                entry["reason"] = "the analysis returned no fit"
            else:
                if fits is not None:
                    fits[name] = outcome.fit
                entry["fit"] = "OK" if outcome.fit.success else "FAIL"
                entry["derived"] = list(outcome.fit.derived.keys())
                if outcome.fit.success:
                    entry["status"] = "PASS"
                else:
                    entry["status"] = "FIT_FAIL"
                    entry["reason"] = "fit.success is False on the simulated data"

        except Exception as e:
            entry["status"] = "ERROR"
            entry["error_type"] = type(e).__name__
            entry["error"] = str(e)[:200]
            # Classify the error phase
            if "construct" not in entry:
                entry["construct"] = "FAIL"
                phase = "construct"
            elif "acquire" not in entry:
                entry["acquire"] = "FAIL"
                phase = "acquire"
            else:
                entry["fit"] = "FAIL"
                phase = "fit"
            entry["reason"] = _one_line(f"{phase}: {type(e).__name__}: {e}")

            if verbose:
                entry["traceback"] = traceback.format_exc()

        entry["elapsed_s"] = round(time.monotonic() - t0, 3)
        results.append(entry)

    return results


def exit_code(results: list[dict], *, min_pass: int = PASS_FLOOR) -> int:
    """0 when the PASS count holds ``min_pass`` and nothing was SKIPped, else 1."""
    passed = sum(1 for r in results if r["status"] == "PASS")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    return 0 if passed >= min_pass and skipped == 0 else 1


def print_report(results: list[dict], *, device: str | Path = DEFAULT_DEVICE,
                 min_pass: int = PASS_FLOOR) -> None:
    """Print a human-readable readiness matrix."""
    total = len(results)
    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in ("PASS", "FIT_FAIL", "ERROR", "SKIP")}

    print(f"\n{'=' * 72}")
    print(f"  cqed sim-smoke: {counts['PASS']}/{total} PASS, {counts['FIT_FAIL']} FIT_FAIL, "
          f"{counts['ERROR']} ERROR, {counts['SKIP']} SKIP  (floor: {min_pass} PASS)")
    print(f"  Device: {Path(device).parent.name} ({Path(device).name})")
    print(f"{'=' * 72}\n")

    for status in ("PASS", "FIT_FAIL", "ERROR", "SKIP"):
        group = [r for r in results if r["status"] == status]
        if not group:
            continue
        print(f"  [{status}] ({len(group)})")
        for r in group:
            line = f"    {r['experiment']:<35s}"
            if status == "PASS":
                derived = ", ".join(r.get("derived", []))
                if derived:
                    line += f"  derived: {derived}"
            else:
                line += f"  {r.get('reason', '?')}"
            print(line)
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="cqed sim-smoke test")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--verbose", action="store_true", help="Include tracebacks on failure")
    parser.add_argument("--min-pass", type=int, default=PASS_FLOOR,
                        help=f"PASS floor for a zero exit code (default {PASS_FLOOR})")
    args = parser.parse_args()

    results = run_all(verbose=args.verbose)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results, min_pass=args.min_pass)

    sys.exit(exit_code(results, min_pass=args.min_pass))


if __name__ == "__main__":
    main()
