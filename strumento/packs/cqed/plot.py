"""cqed plot functions — registered display for every registered cqed experiment (D13 display layer).

Each shape function builds a matplotlib figure from ``(MeasurementResult, FitResult)``.
No side effects, no physics extraction. matplotlib imported lazily.

Registration iterates one table, ``DISPLAY_TABLE`` (group, display function, experiment
classes), through ``displays(ExperimentClass)`` from ``core.display`` — the same MRO-walk
pattern as response models. Pack import triggers registration.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from strumento.core.display import displays
from strumento.packs.cqed.experiments.active_reset import ActiveReset
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi
from strumento.packs.cqed.experiments.cavity_spectroscopy import CavitySpectroscopy
from strumento.packs.cqed.experiments.cavity_t1 import CavityT1
from strumento.packs.cqed.experiments.cavity_t2_ramsey import CavityT2Ramsey
from strumento.packs.cqed.experiments.cross_kerr import CrossKerrMeasurement
from strumento.packs.cqed.experiments.dispersive_shift import DispersiveShift
from strumento.packs.cqed.experiments.displacement_calibration import DisplacementCalibration
from strumento.packs.cqed.experiments.drag_calibration import DRAGCalibration
from strumento.packs.cqed.experiments.drag_pair_calibration import DRAGPairCalibration
from strumento.packs.cqed.experiments.ef_rabi import EFRabi
from strumento.packs.cqed.experiments.ef_spectroscopy import EFSpectroscopy
from strumento.packs.cqed.experiments.eft1 import EFT1
from strumento.packs.cqed.experiments.eft2_echo import EFT2Echo
from strumento.packs.cqed.experiments.eft2_ramsey import EFT2Ramsey
from strumento.packs.cqed.experiments.error_amplification import ErrorAmplification
from strumento.packs.cqed.experiments.fock_cavity_t1 import FockCavityT1
from strumento.packs.cqed.experiments.fock_cavity_t2_ramsey import FockCavityT2Ramsey
from strumento.packs.cqed.experiments.frequency_rabi import FrequencyRabi
from strumento.packs.cqed.experiments.gain_ladder_spectroscopy import GainLadderSpectroscopy
from strumento.packs.cqed.experiments.length_rabi import LengthRabi
from strumento.packs.cqed.experiments.multiphoton_ladder import MultiphotonLadder
from strumento.packs.cqed.experiments.number_splitting import NumberSplitting
from strumento.packs.cqed.experiments.parity_fringe import ParityFringe
from strumento.packs.cqed.experiments.photon_number_distribution import PhotonNumberDistribution
from strumento.packs.cqed.experiments.power_sweep import PowerSweep
from strumento.packs.cqed.experiments.punchout import Punchout
from strumento.packs.cqed.experiments.qnd_readout import QNDReadout
from strumento.packs.cqed.experiments.qubit_spectroscopy import QubitSpectroscopy
from strumento.packs.cqed.experiments.ramsey_detuning_pair import RamseyDetuningPair
from strumento.packs.cqed.experiments.readout_optimization import ReadoutOptimization
from strumento.packs.cqed.experiments.readout_state import ReadoutState
from strumento.packs.cqed.experiments.resonator_power_sweep import ResonatorPowerSweep
from strumento.packs.cqed.experiments.resonator_spectroscopy import ResonatorSpectroscopy
from strumento.packs.cqed.experiments.revival_time_calibration import RevivalTimeCalibration
from strumento.packs.cqed.experiments.sideband_rabi import SidebandRabi
from strumento.packs.cqed.experiments.sideband_spectroscopy import SidebandSpectroscopy
from strumento.packs.cqed.experiments.single_shot_readout import SingleShotReadout
from strumento.packs.cqed.experiments.state_prep_fidelity import StatePrepFidelity
from strumento.packs.cqed.experiments.storage_t1 import StorageT1
from strumento.packs.cqed.experiments.swap_rabi import SwapRabi
from strumento.packs.cqed.experiments.swap_spectroscopy import SwapSpectroscopy
from strumento.packs.cqed.experiments.t1 import T1
from strumento.packs.cqed.experiments.t2_echo import T2Echo
from strumento.packs.cqed.experiments.t2_ramsey import T2Ramsey
from strumento.packs.cqed.experiments.tls_scan import TlsScan
from strumento.packs.cqed.experiments.transmon_allxy import ALLXY_PAIRS, TransmonAllXY
from strumento.packs.cqed.experiments.transmon_rb import TransmonRB
from strumento.packs.cqed.experiments.validation_check import ValidationCheck
from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography

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

def _plot_swept(
    result,
    fit,
    *,
    xlabel="Sweep",
    ylabel="Signal",
    fit_key="",
    magnitude=False,
    readout=None,
    fit_curve=None,
    fit_label="fit",
):
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    xpts = np.asarray(result.xpts, dtype=float) if result.xpts is not None else np.arange(len(np.asarray(result.avgi).ravel()))
    if magnitude:
        signal = np.hypot(
            np.asarray(result.avgi, dtype=float), np.asarray(result.avgq, dtype=float)
        ).ravel()
    elif readout is not None and hasattr(readout, "project"):
        # Plot the SAME discrimination-axis signal the analyzer fit — readout.project (+ data
        # read), via readout_signal — not raw I. With a nonzero readout angle the two live on
        # different axes and can even anti-correlate, which left armed T1/T2 plots showing data
        # rising while the fitted curve fell (points off the curve). See tests/test_plot.py's
        # projection guard.
        from strumento.packs.cqed.experiments._signal import readout_signal

        signal = np.asarray(readout_signal(result, readout), dtype=float).ravel()
    else:
        signal = np.asarray(result.avgi, dtype=float).ravel()
    ax.plot(xpts[:len(signal)], signal, "o", markersize=3, alpha=0.7, label="data")
    if fit_curve is not None and fit is not None and fit.success and len(xpts) > 1:
        xfit = np.linspace(float(np.min(xpts)), float(np.max(xpts)), 1001)
        yfit = fit_curve(xfit, fit)
        if yfit is not None:
            ax.plot(xfit, yfit, "-", linewidth=2, color="tab:red", label=fit_label)
    if fit_key and fit is not None and fit.derived.get(fit_key) is not None:
        ax.set_title(f"{fit_key} = {fit.derived[fit_key]:.4g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_multiread(result, fit, *, xlabel="Index", ylabel="Signal", fit_key=""):
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


def _plot_bar(result, fit, *, xlabel="n", ylabel="Signal"):
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    signal = np.asarray(result.avgi, dtype=float).ravel()
    ax.bar(range(len(signal)), signal, alpha=0.7)
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


def plot_2d_map(result, fit=None, *, xlabel="X", ylabel="Y", title=""):
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    grid = np.asarray(result.avgi, dtype=float)
    xpts = np.asarray(result.xpts, dtype=float) if result.xpts is not None else np.arange(grid.shape[-1])
    ypts = np.asarray(result.meta.get("ypts", np.arange(grid.shape[0])), dtype=float)
    im = ax.pcolormesh(xpts, ypts, grid, shading="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# =============================================================================
# Per-shape display functions (named, not lambda — E731)
# =============================================================================

def _lorentzian_curve(freq, fit):
    required = ("y0", "amp", "f0", "hwhm")
    if not all(key in fit.params for key in required):
        return None
    y0, amp, f0, hwhm = (float(fit.params[key]) for key in required)
    if hwhm == 0.0:
        return None
    return y0 + amp / (1.0 + ((freq - f0) / hwhm) ** 2)


def _display_lorentzian(r, f):
    return _plot_swept(
        r,
        f,
        xlabel="Frequency (MHz)",
        ylabel="|S|",
        fit_key="f0",
        magnitude=True,
        fit_curve=_lorentzian_curve,
        fit_label="Lorentzian fit",
    )

def _display_rabi(r, f, readout=None):
    return _plot_swept(r, f, xlabel="Drive", ylabel="Projected IQ (ADC units)", fit_key="pi_gain",
                       readout=readout)

def _display_length_rabi(r, f, readout=None):
    return _plot_swept(r, f, xlabel="Duration (us)", ylabel="Population", fit_key="rabi_freq",
                       readout=readout)


def _exp_decay_curve(delay, fit):
    required = ("y0", "amp", "tau")
    if not all(key in fit.params for key in required):
        return None
    y0, amp, tau = (float(fit.params[key]) for key in required)
    if tau <= 0.0:
        return None
    return y0 + amp * np.exp(-delay / tau)


def _display_exp_decay(r, f, readout=None):
    return _plot_swept(
        r,
        f,
        xlabel="Delay (us)",
        ylabel="Signal",
        fit_key="T1",
        readout=readout,
        fit_curve=_exp_decay_curve,
        fit_label="Exponential fit",
    )

def _display_t2_echo(r, f, readout=None):
    return _plot_swept(r, f, xlabel="Half-delay (us)", ylabel="Signal", fit_key="T2_echo",
                       readout=readout, fit_curve=_exp_decay_curve, fit_label="Exponential fit")


def _damped_cosine_curve(delay, fit):
    required = ("offset", "amp", "freq", "phase", "tau")
    if not all(key in fit.params for key in required):
        return None
    offset, amp, freq, phase, tau = (float(fit.params[key]) for key in required)
    if tau <= 0.0:
        return None
    return offset + amp * np.exp(-delay / tau) * np.cos(2.0 * np.pi * freq * delay + phase)


def _display_damped_osc(r, f, readout=None):
    return _plot_swept(
        r,
        f,
        xlabel="Delay (us)",
        ylabel="Signal",
        fit_key="freq",
        readout=readout,
        fit_curve=_damped_cosine_curve,
        fit_label="Damped cosine fit",
    )

def _display_parity(r, f, readout=None):
    return _plot_swept(r, f, xlabel="Phase (deg)", ylabel="Parity", fit_key="contrast",
                       readout=readout)

def _display_displacement(r, f, readout=None):
    return _plot_swept(r, f, xlabel="Gain", ylabel="Parity", fit_key="alpha_per_gain",
                       readout=readout)
def _display_power(r, f, readout=None):
    return _plot_swept(r, f, xlabel="Gain", ylabel="Signal", fit_key="optimal_gain",
                       readout=readout)


def _display_ro_opt(r, f, readout=None):
    return _plot_swept(r, f, xlabel="Length (us)", ylabel="Signal", fit_key="optimal_length",
                       readout=readout)


def _display_resonator_power_sweep(r, f):
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    frequency = np.asarray(r.xpts, dtype=float)
    gain_codes = np.asarray(r.meta.get("gain_codes", ()), dtype=float)
    magnitude = np.hypot(np.asarray(r.avgi, dtype=float), np.asarray(r.avgq, dtype=float))
    if magnitude.shape != (len(gain_codes), len(frequency)):
        return None
    fig, axis = plt.subplots(figsize=(9, 6))
    image = axis.pcolormesh(frequency, gain_codes, magnitude, shading="nearest", cmap="viridis")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("|S| (ADC units)")
    axis.set_xlabel("Frequency (MHz)")
    axis.set_ylabel("Probe gain (QICK code)")
    axis.set_title("Resonator power sweep")
    fig.tight_layout()
    return fig

def _display_wigner(r, f, readout=None):
    return _plot_swept(r, f, xlabel="Re(beta)", ylabel="W(beta)", fit_key="w_max",
                       readout=readout)

# Ideal AllXY staircase, derived from the single source of truth (ALLXY_PAIRS) so the plotted
# reference always matches the experiment's pairs — 5 in |g>, 12 on the equator, 4 in |e>.
_ALLXY_IDEAL = np.array([pair[2] for pair in ALLXY_PAIRS], dtype=float)

def _display_allxy(r, f):
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    # Prefer the analyzer's normalized [0,1] staircase (population units); fall back to raw I.
    if f is not None and f.derived.get("populations") is not None:
        signal = np.asarray(f.derived["populations"], dtype=float).ravel()
    else:
        signal = np.asarray(r.avgi, dtype=float).ravel()
    n = min(len(signal), len(_ALLXY_IDEAL))
    ax.plot(range(n), signal[:n], "o-", markersize=5, label="measured")
    ax.plot(range(len(_ALLXY_IDEAL)), _ALLXY_IDEAL, "s--", markersize=4, alpha=0.5, label="ideal")
    ax.set_xlabel("Pair index")
    ax.set_ylabel("Population")
    ax.set_title("AllXY")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _display_drag(r, f):
    """DRAG scan: P_e vs DRAG coefficient α, with the equator crossing marked.

    The error signal is a population in [0, 1] that crosses the equator (0.5) at the optimal α.
    Reads the α axis and the population signal from the analyzer's ``derived`` (``alphas`` /
    ``signal``) so the plot shows the SAME quantity the fit consumed; falls back to the raw
    averaged-I over point index when those are absent (a legacy result that predates the fix).
    """
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))

    derived = {} if f is None else (f.derived or {})
    alphas = derived.get("alphas")
    signal = derived.get("signal")

    if alphas is not None and signal is not None and not derived.get("population_anchored", 1.0):
        # No g/e blob centres: the signal is raw projected IQ, not a population — so no equator, no
        # P_e label, and no optimum (the analyzer refuses to produce one).
        ax.plot(np.asarray(alphas, dtype=float), np.asarray(signal, dtype=float), "o-",
                markersize=5, color="C0", label="projected IQ")
        ax.set_xlabel(r"DRAG coefficient $\alpha$")
        ax.set_ylabel("Projected IQ (ADC units)")
        ax.set_title("DRAG calibration — no g/e blob centres: cannot anchor $P_e$")
    elif alphas is not None and signal is not None:
        x = np.asarray(alphas, dtype=float)
        y = np.asarray(signal, dtype=float)
        ax.plot(x, y, "o-", markersize=5, color="C0", label="measured $P_e$")
        # The equator the crossing is anchored to.
        ax.axhline(0.5, color="0.5", lw=1, ls="--", label="equator ($P_e = 0.5$)")
        alpha_opt = derived.get("alpha_opt")
        slope, intercept = derived.get("slope"), derived.get("intercept")
        lo, hi = derived.get("fit_alpha_min"), derived.get("fit_alpha_max")
        if slope is not None and intercept is not None and lo is not None and hi is not None:
            # The local line the crossing was fitted on, drawn only over the fitted window.
            xf = np.linspace(float(lo), float(hi), 50)
            ax.plot(xf, float(slope) * xf + float(intercept), "-", lw=2, color="C1",
                    label=f"linear fit ({int(derived.get('n_fit', 0))} pts)")
        if alpha_opt is not None and (f is None or f.success):
            se = derived.get("alpha_opt_stderr")
            pm = "" if se is None else fr" \pm {float(se):.2g}"
            ax.axvline(float(alpha_opt), color="C3", lw=1.5, ls="-",
                       label=fr"$\alpha_{{opt}}$ = ${float(alpha_opt):.3g}{pm}$")
            if se is not None:
                ax.axvspan(float(alpha_opt) - float(se), float(alpha_opt) + float(se),
                           color="C3", alpha=0.15, lw=0)
            ax.plot([float(alpha_opt)], [0.5], marker="*", markersize=14, color="C3",
                    zorder=5, linestyle="none")
        ax.set_xlabel(r"DRAG coefficient $\alpha$")
        ax.set_ylabel(r"Excited population $P_e$")
        success = True if f is None else bool(f.success)
        ax.set_title("DRAG calibration" + ("" if success else " — no resolved equator crossing"))
    else:
        # Legacy fallback: no stored axis/population — show raw averaged-I over point index.
        y = np.asarray(r.avgi, dtype=float).ravel()
        ax.plot(range(len(y)), y, "o-", markersize=4, color="C0")
        if f is not None and derived.get("drag_alpha") is not None:
            ax.set_title(f"drag_alpha = {derived['drag_alpha']:.4g}")
        ax.set_xlabel("Point index")
        ax.set_ylabel("Projected IQ (ADC units)")

    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def _display_ramsey_pair(r, f):
    """Ramsey pair: the +Δa and −Δa fringes vs delay, titled with the resolved signed detuning."""
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    derived = {} if f is None else (f.derived or {})
    t = derived.get("delays")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if t is None:
        y = np.asarray(r.avgi, dtype=float).ravel()
        ax.plot(range(len(y)), y, "o-", markersize=3)
        ax.set_xlabel("Read index")
        ax.set_ylabel("Projected IQ (ADC units)")
        fig.tight_layout()
        return fig
    t = np.asarray(t, dtype=float)
    da = derived.get("ramsey_freq_mhz", float("nan"))
    fp, fm = derived.get("fringe_plus_mhz"), derived.get("fringe_minus_mhz")
    ax.plot(t, derived["signal_plus"], "o-", ms=3, lw=0.8, color="C0",
            label=fr"$f_{{ge}}+\Delta_a$: fringe {1e3 * fp:.1f} kHz" if fp is not None else "+Δa")
    ax.plot(t, derived["signal_minus"], "s-", ms=3, lw=0.8, color="C3",
            label=fr"$f_{{ge}}-\Delta_a$: fringe {1e3 * fm:.1f} kHz" if fm is not None else "−Δa")
    delta = derived.get("qubit_detuning_mhz")
    if delta is not None and (f is None or f.success):
        se = derived.get("qubit_detuning_stderr")
        pm = "" if se is None or not np.isfinite(se) else fr" \pm {1e3 * float(se):.1f}"
        ax.set_title(fr"Ramsey pair ($\Delta_a$ = {1e3 * da:.0f} kHz) — $f_q - f_{{ge}}$ = "
                     fr"${1e3 * float(delta):+.1f}{pm}$ kHz ({derived.get('branch')}), "
                     fr"$T_2^*$ = {derived.get('T2', float('nan')):.0f} µs")
    else:
        ax.set_title("Ramsey pair — no consistent branch (sign unresolved)")
    ax.set_xlabel("Delay (µs)")
    ax.set_ylabel("Projected signal")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _display_drag_pair(r, f):
    """DRAG pair: P_A, P_B vs α (top) and their difference with the fitted zero crossing (bottom).

    The top title carries the common-mode equator offset — the quantity that biases a single
    X/2·Y/2 scan and cancels in the difference.
    """
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    derived = {} if f is None else (f.derived or {})
    alphas = derived.get("alphas")
    if alphas is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        y = np.asarray(r.avgi, dtype=float).ravel()
        ax.plot(range(len(y)), y, "o-", markersize=4)
        ax.set_xlabel("Read index")
        ax.set_ylabel("Projected IQ (ADC units)")
        fig.tight_layout()
        return fig

    x = np.asarray(alphas, dtype=float)
    pa = np.asarray(derived["signal_a"], dtype=float)
    pb = np.asarray(derived["signal_b"], dtype=float)
    d = np.asarray(derived["difference"], dtype=float)
    anchored = bool(derived.get("population_anchored", 1.0))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1]})
    top.plot(x, pa, "o-", ms=4, color="C0", label=r"A: $X_{\pi/2}\,Y_\pi$")
    top.plot(x, pb, "s-", ms=4, color="C2", label=r"B: $Y_{\pi/2}\,X_\pi$")
    if anchored:
        top.axhline(0.5, color="0.5", lw=1, ls="--", label="equator")
        top.set_ylabel(r"Excited population $P_e$")
        off, off_se = derived.get("equator_offset"), derived.get("equator_offset_stderr")
        if off is not None:
            se_txt = "" if off_se is None or not np.isfinite(off_se) else fr" \pm {off_se:.2g}"
            top.set_title(fr"DRAG pair — equator offset $(P_A+P_B)/2-\frac{{1}}{{2}}$ = "
                          fr"${off:+.3f}{se_txt}$")
    else:
        top.set_ylabel("Projected IQ (ADC units)")
        top.set_title("DRAG pair — no g/e blob centres: cannot anchor $P_e$")
    top.legend(fontsize=8, loc="best")
    top.grid(alpha=0.3)

    bottom.plot(x, d, "o-", ms=4, color="C0", label=r"$P_A - P_B$")
    bottom.axhline(0.0, color="0.5", lw=1, ls="--")
    slope, intercept = derived.get("slope"), derived.get("intercept")
    lo, hi = derived.get("fit_alpha_min"), derived.get("fit_alpha_max")
    if slope is not None and lo is not None and anchored:
        xf = np.linspace(float(lo), float(hi), 50)
        bottom.plot(xf, float(slope) * xf + float(intercept), "-", lw=2, color="C1",
                    label=f"linear fit ({int(derived.get('n_fit', 0))} pts)")
    a0 = derived.get("alpha_opt")
    if a0 is not None and (f is None or f.success):
        se = derived.get("alpha_opt_stderr")
        pm = "" if se is None else fr" \pm {float(se):.2g}"
        bottom.axvline(float(a0), color="C3", lw=1.5,
                       label=fr"$\alpha_{{opt}}$ = ${float(a0):.3g}{pm}$")
        if se is not None:
            bottom.axvspan(float(a0) - float(se), float(a0) + float(se), color="C3", alpha=0.15, lw=0)
    elif f is not None and not f.success and anchored:
        bottom.set_title("no resolved zero crossing")
    bottom.set_xlabel(r"DRAG coefficient $\alpha$")
    bottom.set_ylabel(r"$P_A - P_B$")
    bottom.legend(fontsize=8, loc="best")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _display_error_amplification(r, f, readout=None):
    return _plot_swept(r, f, xlabel="Drive gain (QICK code)", ylabel="Projected IQ (ADC units)",
                       fit_key="gain_corrected", readout=readout)

def _display_gain_ladder(r, f):
    # AC Stark characterization: the payload is the fitted coefficients (f_spec, stark_slope,
    # p_delta, ...), and the raw shape is a per-rung stack of spectroscopy traces — a scalar
    # panel renders the coefficients robustly without assuming that stacked shape.
    return _plot_scalar(r, f, title="AC Stark gain-ladder spectroscopy")

def _display_fock_bar(r, f):
    return _plot_bar(r, f, xlabel="Fock number n", ylabel="Signal")

def _display_rb(r, f):
    return _plot_multiread(r, f, xlabel="Clifford depth", ylabel="Survival", fit_key="fidelity")

def _display_active_reset(r, f):
    r"""Reset progress across rounds — a bar per round plus the verified residual.

    ``corrected_round_i`` is the fraction of shots that fired the corrective :math:`\pi` in round
    ``i``, i.e. the measured :math:`P(|e\rangle)` the reset *saw entering* that round; the final
    ``verify`` bar is the residual :math:`P(|e\rangle)` after the last round (``excited_population``,
    with its binomial error bar). Together they tell the reset story — a large drop after round 1
    then a plateau at the readout-fidelity floor. Falls back to the scalar card for a single round
    (no trend to show) or when there is no per-shot fit (an averaged/mock acquire).
    """
    if f is None or not f.derived:
        return _plot_scalar(r, f, title="Active Reset")
    corrected = sorted(
        (int(k.rsplit("_", 1)[-1]), float(v))
        for k, v in f.derived.items()
        if k.startswith("corrected_round_")
    )
    excited = f.derived.get("excited_population")
    if len(corrected) < 2 or excited is None:
        return _plot_scalar(r, f, title="Active Reset")   # rounds=1: no trend, keep the scalar card

    plt, err = _try_import_mpl()
    if plt is None:
        return None
    heights = [frac for _, frac in corrected]
    x = np.arange(len(heights) + 1)                       # one bar per round + the verify bar
    excited_err = (f.stderr or {}).get("excited_population")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x[:-1], heights, color="tab:blue", alpha=0.8,
           label=r"corrected: $P(|e\rangle)$ entering each round")
    ax.bar(x[-1:], [excited], yerr=([excited_err] if excited_err is not None else None),
           color="tab:red", alpha=0.8, capsize=4, label=r"residual $P(|e\rangle)$ (verified)")

    all_h = heights + [excited]
    top = max([h for h in all_h if np.isfinite(h)] + [0.0])
    for xi, h in zip(x, all_h):
        if np.isfinite(h):
            ax.annotate(f"{h:.3g}", (xi, h), ha="center", va="bottom", fontsize=8,
                        xytext=(0, 2), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels([f"round {i}" for i, _ in corrected] + ["verify"])
    ax.set_ylim(0.0, top * 1.25 if top > 0 else 1.0)
    ax.set_xlabel("read")
    ax.set_ylabel(r"$P(|e\rangle)$")
    ground = f.derived.get("ground_population")
    ax.set_title("Active Reset" + (f" — ground = {ground:.4g}" if ground is not None else ""))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def _display_punchout(r, f, readout=None):
    """Punchout: the on-resonance signal vs probe gain, the Lorentzian depth profile the
    analyzer fit, and the two gains it derived — the knee (``max_power``, the punched-out
    ceiling) and the operating point (``power``) below it."""
    fig = _plot_swept(r, f, xlabel="Probe gain (QICK code)", ylabel="Projected IQ (ADC units)",
                      fit_key="max_power", readout=readout, fit_curve=_lorentzian_curve,
                      fit_label="Lorentzian fit")
    if fig is None or f is None or not f.success:
        return fig
    ax = fig.axes[0]
    for key, label, style in (("max_power", "knee (max_power)", "-"),
                              ("power", "operating point (power)", "--")):
        value = f.derived.get(key)
        if value is not None:
            ax.axvline(float(value), color="tab:green", lw=1.5, ls=style, label=label)
    ax.legend(loc="best", fontsize=8)
    return fig


def _display_readout_state(r, f, readout=None):
    """Readout state: the per-shot projected histogram of each preparation (ground, then after a
    π) with the fitted discrimination threshold. Without per-shot data (an averaged acquire)
    there is no cloud to histogram, so it falls back to the scalar card of derived values."""
    shots = getattr(r, "shots", None)
    if shots is None:
        return _plot_scalar(r, f, title="Readout State")
    plt, err = _try_import_mpl()
    if plt is None:
        return None
    shots = np.asarray(shots, dtype=float)              # (2 reads, shots, 2 IQ)
    if readout is not None and hasattr(readout, "project"):
        clouds = [np.asarray(readout.project(shots[k]), dtype=float).ravel() for k in (0, 1)]
    else:
        clouds = [shots[k][..., 0].ravel() for k in (0, 1)]
    combined = np.concatenate(clouds)
    lo, hi = float(combined.min()), float(combined.max())
    bins = np.linspace(lo, hi if hi > lo else lo + 1.0, 41)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(clouds[0], bins=bins, alpha=0.6, color="tab:blue", label="ground prep")
    ax.hist(clouds[1], bins=bins, alpha=0.6, color="tab:red", label="excited prep")
    derived = {} if f is None else (f.derived or {})
    if derived.get("threshold") is not None:
        ax.axvline(float(derived["threshold"]), color="k", lw=1.5, ls="--", label="threshold")
    if derived.get("assignment_fidelity") is not None:
        ax.set_title(f"assignment_fidelity = {derived['assignment_fidelity']:.4g}")
    ax.set_xlabel("Projected IQ (ADC units)")
    ax.set_ylabel("Counts")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _display_tls_scan(r, f):
    """TLS scan: |S| vs qubit drive frequency with the main-line Lorentzian fit; the strongest
    residual dip the analyzer counted as a TLS line is marked, and the title gives the count."""
    fig = _plot_swept(r, f, xlabel="Qubit drive frequency (MHz)", ylabel="|S| (ADC units)",
                      fit_key="f0", magnitude=True, fit_curve=_lorentzian_curve,
                      fit_label="Lorentzian fit")
    if fig is None or f is None or f.derived.get("tls_lines") is None:
        return fig
    ax = fig.axes[0]
    count = int(f.derived["tls_lines"])
    f0 = f.derived.get("f0")
    if count > 0 and f0 is not None:
        offset = float(f.derived.get("tls_offset_mhz", 0.0))
        ax.axvline(float(f0) + offset, color="tab:purple", lw=1.5, ls="--",
                   label=f"strongest TLS line ({offset:+.3g} MHz)")
        ax.legend(loc="best", fontsize=8)
    head = f"f0 = {float(f0):.4g} — " if f0 is not None else ""
    ax.set_title(f"{head}{count} TLS line{'' if count == 1 else 's'}")
    return fig


def _display_validation_check(r, f, readout=None):
    """Validation check: the Ramsey fringe (the data read) with its damped-cosine fit, against
    the two readout preparations of the same shot (read 0 ground, read 1 excited) as reference
    traces; the title carries the two gated numbers, the detuning and the assignment fidelity."""
    fig = _plot_swept(r, f, xlabel="Delay (us)", ylabel="Projected IQ (ADC units)",
                      fit_key="detuning", readout=readout, fit_curve=_damped_cosine_curve,
                      fit_label="Damped cosine fit")
    if fig is None:
        return fig
    ax = fig.axes[0]
    avgi = np.asarray(r.avgi, dtype=float)
    if avgi.ndim == 2 and avgi.shape[0] >= 3:
        if readout is not None and hasattr(readout, "project"):
            iq = np.stack([avgi, np.asarray(r.avgq, dtype=float)], axis=-1)
            rows = np.asarray(readout.project(iq), dtype=float)
        else:
            rows = avgi
        xpts = np.asarray(r.xpts, dtype=float)
        for k, label, color in ((0, "ground prep (read 0)", "tab:blue"),
                                (1, "excited prep (read 1)", "tab:orange")):
            ax.plot(xpts[:rows.shape[1]], rows[k], "-", lw=1, alpha=0.6, color=color, label=label)
        ax.legend(loc="best", fontsize=8)
    derived = {} if f is None else (f.derived or {})
    parts = [f"{key} = {derived[key]:.4g}" for key in ("detuning", "assignment_fidelity")
             if derived.get(key) is not None]
    if parts:
        ax.set_title(", ".join(parts))
    return fig


def _display_single_shot(r, f):
    return _plot_scalar(r, f, title="Single-Shot Readout")

def _display_qnd(r, f):
    return _plot_scalar(r, f, title="QND Readout")


# =============================================================================
# Registrations — every registered cqed experiment, from one table
# =============================================================================

#: ``(group, display function, experiment classes)`` — the single source of the cqed display
#: registrations. Group sizes are whatever each row lists; nothing here is counted by hand.
#: ``tests/packs/cqed/test_display_coverage.py`` fails if a registered experiment has no row.
DISPLAY_TABLE: tuple[tuple[str, Callable, tuple[type, ...]], ...] = (
    ("Lorentzian peak", _display_lorentzian,
     (ResonatorSpectroscopy, QubitSpectroscopy, EFSpectroscopy, CavitySpectroscopy,
      DispersiveShift, NumberSplitting, FrequencyRabi, SidebandSpectroscopy, SwapSpectroscopy)),
    ("Rabi arch", _display_rabi, (AmplitudeRabi, EFRabi, SidebandRabi, SwapRabi)),
    ("Rabi arch", _display_length_rabi, (LengthRabi,)),
    ("Exponential decay", _display_exp_decay, (T1, CavityT1, FockCavityT1, StorageT1, EFT1)),
    ("Exponential decay", _display_t2_echo, (T2Echo, EFT2Echo)),
    ("Damped oscillation", _display_damped_osc,
     (T2Ramsey, CavityT2Ramsey, FockCavityT2Ramsey, RevivalTimeCalibration,
      CrossKerrMeasurement, EFT2Ramsey)),
    ("Cosine fringe", _display_parity, (ParityFringe,)),
    ("Cosine fringe", _display_displacement, (DisplacementCalibration,)),
    ("Swept argmax", _display_power, (PowerSweep,)),
    ("Swept argmax", _display_ro_opt, (ReadoutOptimization,)),
    ("Swept argmax", _display_punchout, (Punchout,)),
    ("Frequency-by-gain map", _display_resonator_power_sweep, (ResonatorPowerSweep,)),
    ("Wigner slice", _display_wigner, (WignerTomography, StatePrepFidelity)),
    ("AllXY staircase", _display_allxy, (TransmonAllXY,)),
    ("DRAG zero-crossing", _display_drag, (DRAGCalibration,)),
    ("DRAG zero-crossing", _display_drag_pair, (DRAGPairCalibration,)),
    ("Ramsey pair", _display_ramsey_pair, (RamseyDetuningPair,)),
    ("Fine calibration / AC Stark", _display_error_amplification, (ErrorAmplification,)),
    ("Fine calibration / AC Stark", _display_gain_ladder, (GainLadderSpectroscopy,)),
    ("Fock-level bar chart", _display_fock_bar, (MultiphotonLadder, PhotonNumberDistribution)),
    ("RB decay", _display_rb, (TransmonRB,)),
    ("Shot statistics", _display_active_reset, (ActiveReset,)),
    ("Shot statistics", _display_single_shot, (SingleShotReadout,)),
    ("Shot statistics", _display_qnd, (QNDReadout,)),
    ("Shot statistics", _display_readout_state, (ReadoutState,)),
    ("Spectroscopy with defect lines", _display_tls_scan, (TlsScan,)),
    ("Ramsey with readout references", _display_validation_check, (ValidationCheck,)),
)

for _group, _display_fn, _classes in DISPLAY_TABLE:
    for _cls in _classes:
        displays(_cls)(_display_fn)
