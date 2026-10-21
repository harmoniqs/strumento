"""spin plot functions — registered display for all 21 experiments (D13 display layer).

Same pattern as cqed: shape functions + ``displays()`` registrations. matplotlib lazily imported.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from strumento.core.display import displays
from strumento.packs.spin.experiments.allxy import AllXY
from strumento.packs.spin.experiments.charge_noise_spectroscopy import ChargeNoiseSpectroscopy
from strumento.packs.spin.experiments.composite_pulse_calibration import CompositePulseCalibration
from strumento.packs.spin.experiments.cpmg import CPMG
from strumento.packs.spin.experiments.edsr_spectroscopy import EDSRSpectroscopy
from strumento.packs.spin.experiments.elzerman_optimize import ElzermanOptimize
from strumento.packs.spin.experiments.exchange_gate_calibration import ExchangeGateCalibration
from strumento.packs.spin.experiments.exchange_oscillation import ExchangeOscillation
from strumento.packs.spin.experiments.g_factor_map import GFactorMap
from strumento.packs.spin.experiments.gradient_calibration import GradientCalibration
from strumento.packs.spin.experiments.hahn_echo import HahnEcho
from strumento.packs.spin.experiments.heralded_init import HeraldedInit
from strumento.packs.spin.experiments.interleaved_rb import InterleavedRB
from strumento.packs.spin.experiments.leakage_benchmark import LeakageBenchmark
from strumento.packs.spin.experiments.psb_calibration import PSBCalibration
from strumento.packs.spin.experiments.randomized_benchmarking import RandomizedBenchmarking
from strumento.packs.spin.experiments.spin_rabi import SpinRabi
from strumento.packs.spin.experiments.spin_ramsey import SpinRamsey
from strumento.packs.spin.experiments.spin_t1 import SpinT1
from strumento.packs.spin.experiments.sweet_spot_tune import SweetSpotTune
from strumento.packs.spin.experiments.tunnel_rate_tune import TunnelRateTune

# =============================================================================
# Lazy matplotlib
# =============================================================================

def _try_import_mpl():
    try:
        import matplotlib.pyplot as plt
        return plt, None
    except ImportError:
        return None, "matplotlib not installed"


# =============================================================================
# Shape functions
# =============================================================================

def _plot_swept(result, fit, *, xlabel="Sweep", ylabel="P_up", fit_key=""):
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    xpts = np.asarray(result.xpts, dtype=float) if result.xpts is not None else np.arange(len(np.asarray(result.avgi).ravel()))
    signal = np.asarray(result.avgi, dtype=float).ravel()
    ax.plot(xpts[:len(signal)], signal, "o", markersize=3, alpha=0.7, label="data")
    if fit_key and fit is not None and fit.derived.get(fit_key) is not None:
        ax.set_title(f"{fit_key} = {fit.derived[fit_key]:.4g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_multiread(result, fit, *, xlabel="Index", ylabel="P_up", fit_key=""):
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    signal = np.asarray(result.avgi, dtype=float).ravel()
    ax.plot(range(len(signal)), signal, "o-", markersize=4)
    if fit_key and fit is not None and fit.derived.get(fit_key) is not None:
        ax.set_title(f"{fit_key} = {fit.derived[fit_key]:.4g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def _plot_scalar(result, fit, *, title=""):
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis("off")
    lines = [title] if title else []
    if fit is not None and fit.derived:
        for k, v in fit.derived.items():
            lines.append(f"{k} = {v:.4g}" if isinstance(v, (int, float)) else f"{k} = {v}")
    ax.text(0.5, 0.5, "\n".join(lines) if lines else "(no derived quantities)",
            ha="center", va="center", fontsize=12, transform=ax.transAxes)
    fig.tight_layout()
    return fig


# =============================================================================
# Per-shape display functions
# =============================================================================

def _display_exp_decay(r, f):
    return _plot_swept(r, f, xlabel="Delay (us)", ylabel="P_up", fit_key="tau")

def _display_rabi(r, f):
    return _plot_swept(r, f, xlabel="Duration (us)", ylabel="P_up", fit_key="period")

def _display_damped_osc(r, f):
    return _plot_swept(r, f, xlabel="Delay (us)", ylabel="P_up", fit_key="tau")

def _display_edsr(r, f):
    return _plot_swept(r, f, xlabel="Frequency (MHz)", ylabel="P_up", fit_key="f_qubit_mhz")

def _display_composite(r, f):
    return _plot_swept(r, f, xlabel="Amplitude scale", ylabel="Fidelity", fit_key="fidelity")

def _display_histogram(r, f):
    return _plot_swept(r, f, xlabel="Sweep", ylabel="Signal", fit_key="threshold")

_ALLXY_IDEAL = np.array([0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)

def _display_allxy(r, f):
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    signal = np.asarray(r.avgi, dtype=float).ravel()
    n = min(len(signal), 21)
    ax.plot(range(n), signal[:n], "o-", markersize=5, label="measured")
    ax.plot(range(len(_ALLXY_IDEAL)), _ALLXY_IDEAL, "s--", markersize=4, alpha=0.5, label="ideal")
    ax.set_xlabel("Pair index")
    ax.set_ylabel("P_up")
    ax.set_title("AllXY")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def _display_rb(r, f):
    return _plot_multiread(r, f, xlabel="Clifford depth", ylabel="Survival", fit_key="fidelity")

def _display_irb(r, f):
    return _plot_multiread(r, f, xlabel="Clifford depth", ylabel="Survival", fit_key="gate_fidelity")

def _display_leakage(r, f):
    return _plot_multiread(r, f, xlabel="Gate count", ylabel="Survival", fit_key="leakage_per_gate")

def _display_heralded(r, f):
    return _plot_scalar(r, f, title="Heralded Init")


# =============================================================================
# Registrations — all 21 registered experiments
# =============================================================================

# Exponential decay
for _cls in cast("tuple[type, ...]", (SpinT1, HahnEcho, CPMG, TunnelRateTune)):
    displays(_cls)(_display_exp_decay)

# Rabi oscillation
for _cls in (SpinRabi, ExchangeOscillation, ExchangeGateCalibration):
    displays(_cls)(_display_rabi)

# Decaying sinusoid
for _cls in (SpinRamsey, GradientCalibration, GFactorMap, SweetSpotTune, ChargeNoiseSpectroscopy):
    displays(_cls)(_display_damped_osc)

# Spectroscopy
displays(EDSRSpectroscopy)(_display_edsr)

# Composite
displays(CompositePulseCalibration)(_display_composite)

# Histogram
for _cls in (ElzermanOptimize, PSBCalibration):
    displays(_cls)(_display_histogram)

# AllXY staircase
displays(AllXY)(_display_allxy)

# Multi-read decay
displays(RandomizedBenchmarking)(_display_rb)
displays(InterleavedRB)(_display_irb)
displays(LeakageBenchmark)(_display_leakage)

# Scalar
displays(HeraldedInit)(_display_heralded)
