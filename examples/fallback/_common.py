"""Shared helpers for fallback diagnostic scripts.

These scripts bypass strumento entirely — they import qick directly and build
bare AveragerProgramV2 programs. This module provides the loading, soc
construction, and plotting utilities they all share.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
FIXTURES = _HERE.parent.parent / "devices" / "_fixtures"
DEFAULT_SOCCFG = FIXTURES / "soccfg_v2_testbench.json"

# ---------------------------------------------------------------------------
# soccfg loading
# ---------------------------------------------------------------------------


def load_soccfg(path: str | Path | None = None) -> dict:
    """Load a soccfg dict from a ``dump_cfg()`` JSON snapshot.

    Defaults to the shipped v2 testbench fixture.
    """
    p = Path(path) if path else DEFAULT_SOCCFG
    if not p.exists():
        raise FileNotFoundError(f"soccfg snapshot not found: {p}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# soc construction
# ---------------------------------------------------------------------------


def make_soc(soccfg_dict: dict):
    """Build an offline soc for testing.

    Prefers strumento's ``MockQickSocV2`` (supports acquire with zero-valued IQ);
    falls back to ``qick.QickConfig`` (build + compile only, no acquire).
    Returns ``(soc, can_acquire)``.
    """
    try:
        from strumento.core.mock import MockQickSocV2

        return MockQickSocV2(soccfg_dict), True
    except ImportError:
        from qick import QickConfig

        return QickConfig(soccfg_dict), False


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def base_parser(description: str) -> argparse.ArgumentParser:
    """An argparse parser with the flags every fallback script shares."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--soccfg",
        type=str,
        default=str(DEFAULT_SOCCFG),
        help="Path to a dump_cfg() JSON snapshot (default: testbench fixture)",
    )
    p.add_argument("--gen-ch", type=int, default=0, help="Generator channel (default: 0)")
    p.add_argument("--ro-ch", type=int, default=0, help="Readout channel (default: 0)")
    p.add_argument("--reps", type=int, default=100, help="Hardware averages (default: 100)")
    p.add_argument("--plot", action="store_true", help="Show a matplotlib figure")
    return p


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_iq(
    xpts,
    i_data,
    q_data,
    *,
    xlabel: str = "",
    ylabel: str = "IQ (arb.)",
    title: str = "",
    show: bool = True,
):
    """Plot I and Q vs a swept axis.  Falls back to a text summary if matplotlib is absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _text_summary(xpts, i_data, q_data, title=title)
        return None

    fig, (ax_i, ax_q) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax_i.plot(np.asarray(xpts), np.asarray(i_data), ".-", markersize=3, label="I")
    ax_i.set_ylabel("I (arb.)")
    ax_i.set_title(title)
    ax_i.grid(True, alpha=0.3)
    ax_i.legend(loc="upper right")

    ax_q.plot(np.asarray(xpts), np.asarray(q_data), ".-", markersize=3, color="C1", label="Q")
    ax_q.set_xlabel(xlabel)
    ax_q.set_ylabel("Q (arb.)")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="upper right")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_mag(
    xpts,
    mag,
    *,
    xlabel: str = "",
    ylabel: str = "|S| (arb.)",
    title: str = "",
    show: bool = True,
):
    """Plot magnitude vs a swept axis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n{title}")
        print(f"  x: [{min(xpts):.4f}, {max(xpts):.4f}]")
        print(f"  y: [{min(mag):.4f}, {max(mag):.4f}]")
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.asarray(xpts), np.asarray(mag), ".-", markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _text_summary(xpts, i_data, q_data, *, title: str = ""):
    """Print a text summary when matplotlib is not available."""
    i_arr, q_arr = np.asarray(i_data), np.asarray(q_data)
    mag = np.hypot(i_arr, q_arr)
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Points:    {len(i_arr)}")
    print(f"  I range:   [{i_arr.min():.6f}, {i_arr.max():.6f}]")
    print(f"  Q range:   [{q_arr.min():.6f}, {q_arr.max():.6f}]")
    print(f"  |S| range: [{mag.min():.6f}, {mag.max():.6f}]")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------


def report_single(i_val: float, q_val: float, *, label: str = "ADC readout"):
    """Print a single-point IQ result."""
    mag = np.hypot(i_val, q_val)
    phase = np.degrees(np.arctan2(q_val, i_val))
    print(f"\n  {label}")
    print(f"  I = {i_val:.6f},  Q = {q_val:.6f}")
    print(f"  |S| = {mag:.6f},  phase = {phase:.1f} deg")
    if mag == 0.0:
        print("  (all zeros — expected for mock soc; on a real board, check the input)")
