"""Spin pack response models — offline simulator physics for spin experiments (D32).

Each response model defines what signal the simulator produces for a given experiment
when armed with a ground-truth parameter set. The model is the closed-form inverse of
the fitter: `fit_exp` recovers T1 from an exponential; this module *generates* that
exponential from the armed T1.

Registration uses the core `@responds` decorator — the `SimulatorSoc` resolves the
model by experiment type at arming time (MRO walk, same as cqed).

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from strumento.core.response import ShotMachine, SimContext, SimResponse, responds
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
# Default truth parameters (simulator-side defaults for quantities the device
# calibration may not carry). Overridden by device belief, then by armed truth.
# =============================================================================

DEFAULT_TRUTH: dict[str, Any] = {
    # Coherence times (microseconds)
    "T1": 50.0,
    "T2_star": 2.0,
    "T2_echo": 15.0,
    "T2_DD": 30.0,
    "T_Rabi": 5.0,
    "T_J": 3.0,
    # Qubit / exchange parameters
    "omega_rabi_mhz": 20.0,       # Rabi frequency (MHz) for drive-duration Rabi
    "J_mhz": 80.0,                # exchange frequency (MHz)
    "ramsey_detuning": 0.5,       # detuning from resonance (MHz)
    # Readout model
    "p_thermal": 0.02,            # thermal excited-state population
    "readout_contrast": 0.96,     # readout visibility C (accounts for SPAM errors)
    "blob0": (0.0, 0.0),          # IQ blob center for |0⟩ (ground / singlet)
    "blob1": (1.0, 0.0),          # IQ blob center for |1⟩ (excited / triplet)
    "sigma": 0.15,                # per-shot IQ scatter (blob width)
    # RB / gate benchmarking
    "rb_depolarizing_p": 0.98,        # RB survival per Clifford (p parameter)
    "irb_depolarizing_p": 0.96,       # interleaved RB survival
    # Spectroscopy
    "f_qubit_mhz": 5000.0,           # qubit resonance frequency
    "linewidth_mhz": 0.5,            # EDSR linewidth
    # Leakage / composite
    "leakage_per_gate": 0.005,        # leakage rate per gate
    "composite_fidelity": 0.998,      # composite pulse fidelity
    # Noise
    "noise_amplitude": 1.0,           # charge noise amplitude (arb)
    "noise_exponent": 1.0,            # 1/f^alpha exponent
    # Tunnel rate
    "gamma_in_khz": 50.0,            # tunnel-in rate
    "gamma_out_khz": 30.0,           # tunnel-out rate
    # Readout (Elzerman / PSB)
    "readout_fidelity": 0.95,         # Elzerman/PSB readout fidelity
    "psb_visibility": 0.90,           # PSB singlet-triplet visibility
    "t_read_us": 100.0,              # Elzerman readout integration window (µs)
    # Dispersive gate readout
    "gate_sensor_angle_rad": 0.35,    # IQ discrimination angle (rad) from quantum capacitance
    "gate_sensor_separation": 0.8,    # blob separation (arb) — from spin-dependent C_Q shift
}


def truth_from_device(device=None, **overrides) -> dict[str, Any]:
    """Build truth params from device calibration, falling back to DEFAULT_TRUTH.

    Resolution: DEFAULT_TRUTH → device calibration → explicit overrides.
    """
    p = dict(DEFAULT_TRUTH)
    if device is not None and hasattr(device, "calib"):
        calib = device.calib
        # Pull coherence times from the first dot's calibration if available
        for dot_cal in calib.dots.values():
            if dot_cal.t1_us is not None:
                p["T1"] = dot_cal.t1_us
            if dot_cal.t2_star_us is not None:
                p["T2_star"] = dot_cal.t2_star_us
            if dot_cal.t2_echo_us is not None:
                p["T2_echo"] = dot_cal.t2_echo_us
            if dot_cal.exchange_j_mhz is not None:
                p["J_mhz"] = dot_cal.exchange_j_mhz
            break  # first dot only
    p.update(overrides)
    return p


def params(ctx: SimContext) -> dict[str, Any]:
    """Resolve the full truth for a simulation context.

    Three layers: DEFAULT_TRUTH → device calibration → armed truth dict.
    """
    return {**truth_from_device(ctx.device), **(ctx.truth or {})}


def _populations(pops, p: dict) -> SimResponse:
    """Wrap a population array in a SimResponse with the readout-model geometry."""
    return SimResponse(
        populations=np.asarray(pops, dtype=float),
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["sigma"]),
    )


# =============================================================================
# Response models — one per experiment type that has a standard fitter
# =============================================================================


@responds(SpinT1)
def spin_t1_response(ctx: SimContext) -> SimResponse:
    r"""Excited population vs wait time: :math:`p_e(t) = p_{th} + C \cdot e^{-t/T_1}`.

    ``fit_exp`` reports ``tau`` = T1 directly.
    """
    p = params(ctx)
    t = ctx.x()
    decay = np.exp(-np.abs(t) / float(p["T1"]))
    return _populations(p["p_thermal"] + p["readout_contrast"] * decay, p)


@responds(SpinRabi)
def spin_rabi_response(ctx: SimContext) -> SimResponse:
    r"""Excited population vs drive duration: a Rabi oscillation.

    .. math:: p_e(t) = p_{th} + C \left[\frac{1}{2} - \frac{1}{2} e^{-t/T_R}
              \cos(2\pi \Omega t)\right]

    ``fit_rabi`` reports ``period`` = :math:`1/\Omega` and derives
    ``pi_gain`` = ``period / 2`` (the duration for a π rotation).
    """
    p = params(ctx)
    t = ctx.x()
    omega = float(p["omega_rabi_mhz"])  # MHz — period in µs
    t_decay = float(p["T_Rabi"])
    osc = 0.5 - 0.5 * np.exp(-np.abs(t) / t_decay) * np.cos(2.0 * np.pi * omega * t)
    return _populations(p["p_thermal"] + p["readout_contrast"] * osc, p)


@responds(SpinRamsey)
def spin_ramsey_response(ctx: SimContext) -> SimResponse:
    r"""Ramsey fringe vs free evolution time.

    .. math:: p_e(\tau) = p_{th} + C \left[\frac{1}{2} + \frac{1}{2} e^{-\tau/T_2^*}
              \cos(2\pi \delta \tau)\right]

    ``fit_decaysin_detuned`` reports ``freq`` = detuning δ (MHz) and ``tau`` = T2*.
    """
    p = params(ctx)
    tau = ctx.x()
    detuning = float(p["ramsey_detuning"])
    t2_star = float(p["T2_star"])
    fringe = 0.5 + 0.5 * np.exp(-np.abs(tau) / t2_star) * np.cos(
        2.0 * np.pi * detuning * tau
    )
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


@responds(HahnEcho)
def hahn_echo_response(ctx: SimContext) -> SimResponse:
    r"""Echo decay vs half-evolution time τ: :math:`p_e(2\tau) = p_{th} + C \cdot e^{-2\tau/T_{2e}}`.

    ``fit_exp`` reports ``tau`` = T2_echo (the total evolution time at 1/e).
    """
    p = params(ctx)
    tau = ctx.x()  # half-evolution time
    t2_echo = float(p["T2_echo"])
    # Total evolution is 2*tau; decay constant is T2_echo
    decay = np.exp(-2.0 * np.abs(tau) / t2_echo)
    return _populations(p["p_thermal"] + p["readout_contrast"] * decay, p)


@responds(CPMG)
def cpmg_response(ctx: SimContext) -> SimResponse:
    r"""CPMG decay vs inter-pulse delay tau, accounting for N refocusing pulses.

    The total evolution time is :math:`T = 2 N \tau`, and coherence decays as
    :math:`p_e(\tau) = p_{th} + C \cdot e^{-2 N \tau / T_{2,\mathrm{DD}}}`.

    ``fit_exp`` on the tau axis reports :math:`\tau_{\mathrm{eff}} = T_{2,\mathrm{DD}} / (2N)`;
    the analyzer multiplies by :math:`2N` to recover :math:`T_{2,\mathrm{DD}}`.
    """
    p = params(ctx)
    tau = ctx.x()
    t2_dd = float(p["T2_DD"])
    n_pi = int(getattr(ctx.experiment, "n_pi", 1))
    # Total evolution time = 2 * N_pi * tau; decay depends on total time
    total_time = 2.0 * n_pi * np.abs(tau)
    decay = np.exp(-total_time / t2_dd)
    return _populations(p["p_thermal"] + p["readout_contrast"] * decay, p)


@responds(ExchangeOscillation)
def exchange_oscillation_response(ctx: SimContext) -> SimResponse:
    r"""Exchange oscillation: singlet probability vs exchange pulse duration.

    .. math:: P_S(t) = p_{th} + C \left[\frac{1}{2} - \frac{1}{2} e^{-t/T_J}
              \cos(2\pi J t)\right]

    Same shape as Rabi but in the exchange basis. ``fit_rabi`` reports ``period`` = 1/J
    and derives ``pi_gain`` = period/2 (the exchange π time).
    """
    p = params(ctx)
    t = ctx.x()
    j_mhz = float(p["J_mhz"])
    t_decay = float(p["T_J"])
    osc = 0.5 - 0.5 * np.exp(-np.abs(t) / t_decay) * np.cos(2.0 * np.pi * j_mhz * t)
    return _populations(p["p_thermal"] + p["readout_contrast"] * osc, p)


# =============================================================================
# Additional response models
# =============================================================================


@responds(EDSRSpectroscopy)
def edsr_spectroscopy_response(ctx: SimContext) -> SimResponse:
    r"""EDSR spectroscopy: Lorentzian dip at qubit resonance frequency.

    .. math:: p_e(f) = p_{th} + C \left[1 - \frac{\gamma^2}{(f - f_0)^2 + \gamma^2}\right]

    The analyzer finds ``argmin`` → ``f_qubit_mhz`` and FWHM → ``linewidth_mhz``.
    """
    p = params(ctx)
    f = ctx.x()
    f0 = float(p["f_qubit_mhz"])
    gamma = float(p["linewidth_mhz"]) / 2.0  # HWHM from FWHM
    lorentzian = gamma**2 / ((f - f0) ** 2 + gamma**2)
    signal = 1.0 - lorentzian
    return _populations(p["p_thermal"] + p["readout_contrast"] * signal, p)


@responds(AllXY)
def allxy_response(ctx: SimContext) -> SimResponse:
    r"""AllXY: 21 ideal populations (one per gate pair, shaped for multi-read).

    Returns the ideal staircase perturbed by a small amplitude error from the truth,
    shaped ``(1, 21)`` so the simulator generates one value per read in a 21-read shot.
    """
    from strumento.packs.spin.experiments.allxy import ALLXY_PAIRS

    p = params(ctx)
    ideal = np.array([pair[2] for pair in ALLXY_PAIRS], dtype=float)
    # Perturb by a small amplitude error (shifts the equator points)
    amp_err = 0.02 * (1.0 - float(p["readout_contrast"]))
    perturbed = np.clip(ideal + amp_err, 0.0, 1.0)
    return SimResponse(
        populations=perturbed.reshape(1, -1),
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["sigma"]),
    )


@responds(ExchangeGateCalibration)
def exchange_gate_calibration_response(ctx: SimContext) -> SimResponse:
    r"""Exchange gate calibration: same Rabi-like oscillation as ExchangeOscillation.

    Uses ``fit_rabi`` to extract period → SWAP time. Physics is identical to
    :func:`exchange_oscillation_response`.
    """
    p = params(ctx)
    t = ctx.x()
    j_mhz = float(p["J_mhz"])
    t_decay = float(p["T_J"])
    osc = 0.5 - 0.5 * np.exp(-np.abs(t) / t_decay) * np.cos(2.0 * np.pi * j_mhz * t)
    return _populations(p["p_thermal"] + p["readout_contrast"] * osc, p)


@responds(GradientCalibration)
def gradient_calibration_response(ctx: SimContext) -> SimResponse:
    r"""Gradient calibration: Ramsey fringe at the magnetic gradient frequency.

    .. math:: p_e(\tau) = \frac{1}{2} + \frac{1}{2} e^{-\tau/T_2^*}
              \cos(2\pi \,\Delta B_z \,\tau)

    Uses ``fit_decaysin_detuned`` to extract ``freq`` = ΔBz (MHz) and ``tau`` = T2*.
    Same physics as Ramsey with ``ramsey_detuning`` as the gradient splitting.
    """
    p = params(ctx)
    tau = ctx.x()
    delta_bz = float(p["ramsey_detuning"])  # gradient splitting (MHz)
    t2_star = float(p["T2_star"])
    fringe = 0.5 + 0.5 * np.exp(-np.abs(tau) / t2_star) * np.cos(
        2.0 * np.pi * delta_bz * tau
    )
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


@responds(GFactorMap)
def g_factor_map_response(ctx: SimContext) -> SimResponse:
    r"""g-factor map: Ramsey fringe — same physics as SpinRamsey.

    The g-factor is extracted from the Ramsey detuning as a function of applied
    field; the response model is identical to :func:`spin_ramsey_response`.
    """
    p = params(ctx)
    tau = ctx.x()
    detuning = float(p["ramsey_detuning"])
    t2_star = float(p["T2_star"])
    fringe = 0.5 + 0.5 * np.exp(-np.abs(tau) / t2_star) * np.cos(
        2.0 * np.pi * detuning * tau
    )
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


@responds(SweetSpotTune)
def sweet_spot_tune_response(ctx: SimContext) -> SimResponse:
    r"""Sweet-spot tune: Ramsey fringe — same physics as SpinRamsey.

    At the sweet spot, T2* is maximized and the detuning is stable. The analyzer
    sweeps a gate voltage and fits the Ramsey fringe at each point.
    """
    p = params(ctx)
    tau = ctx.x()
    detuning = float(p["ramsey_detuning"])
    t2_star = float(p["T2_star"])
    fringe = 0.5 + 0.5 * np.exp(-np.abs(tau) / t2_star) * np.cos(
        2.0 * np.pi * detuning * tau
    )
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


@responds(RandomizedBenchmarking)
def randomized_benchmarking_response(ctx: SimContext) -> SimResponse:
    r"""Randomized benchmarking: exponential decay of survival probability vs sequence depth.

    Returns ``(1, points)`` populations — one per depth block (multi-read, no sweep axis).
    The decay is :math:`p_{surv}(m) = A \cdot p^m + B` where :math:`p` is the depolarizing
    parameter.
    """
    p = params(ctx)
    exp = ctx.experiment
    depths = exp._depths.astype(float)
    rb_p = float(p["rb_depolarizing_p"])
    decay = rb_p ** depths
    pops = p["p_thermal"] + p["readout_contrast"] * decay
    return SimResponse(
        populations=np.asarray(pops, dtype=float).reshape(1, -1),
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["sigma"]),
    )


@responds(InterleavedRB)
def interleaved_rb_response(ctx: SimContext) -> SimResponse:
    r"""Interleaved RB: exponential decay with interleaved gate depolarizing (multi-read).

    Returns ``(1, points)`` populations — one per depth block. The interleaved fidelity
    is extracted from the ratio :math:`r_{gate} = p_{irb}/p_{rb}`.
    """
    p = params(ctx)
    exp = ctx.experiment
    depths = exp._depths.astype(float)
    irb_p = float(p["irb_depolarizing_p"])
    decay = irb_p ** depths
    pops = p["p_thermal"] + p["readout_contrast"] * decay
    return SimResponse(
        populations=np.asarray(pops, dtype=float).reshape(1, -1),
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["sigma"]),
    )


@responds(LeakageBenchmark)
def leakage_benchmark_response(ctx: SimContext) -> SimResponse:
    r"""Leakage benchmark: computational-subspace survival decays with gate count (multi-read).

    Returns ``(1, points)`` populations — one per gate-count block.
    :math:`p_{comp}(n) = A \cdot (1 - L)^n` where L is the leakage per gate.
    """
    p = params(ctx)
    exp = ctx.experiment
    gate_counts = exp._gate_counts.astype(float)
    leakage = float(p["leakage_per_gate"])
    decay = (1.0 - leakage) ** gate_counts
    pops = p["p_thermal"] + p["readout_contrast"] * decay
    return SimResponse(
        populations=np.asarray(pops, dtype=float).reshape(1, -1),
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["sigma"]),
    )


@responds(CompositePulseCalibration)
def composite_pulse_calibration_response(ctx: SimContext) -> SimResponse:
    r"""Composite pulse calibration: gentle exponential decay of fidelity vs error.

    .. math:: p(x) = p_{th} + C \cdot e^{-x \cdot (1 - F_{comp})}

    The composite pulse compensates for systematic errors; the decay rate is
    proportional to the residual infidelity :math:`1 - F_{comp}`.
    """
    p = params(ctx)
    x = ctx.x()  # error parameter / sweep axis
    fid = float(p["composite_fidelity"])
    decay = np.exp(-np.abs(x) * (1.0 - fid))
    return _populations(p["p_thermal"] + p["readout_contrast"] * decay, p)


@responds(ChargeNoiseSpectroscopy)
def charge_noise_spectroscopy_response(ctx: SimContext) -> SimResponse:
    r"""Charge noise spectroscopy: coherence decay vs inter-pulse delay.

    .. math:: p(\tau) = p_{th} + C \cdot e^{-\tau / T_2^*}

    Uses ``fit_exp`` (simple exponential). The stretched exponent :math:`\alpha`
    from 1/f noise is a refinement beyond the fitter's model.
    """
    p = params(ctx)
    tau = ctx.x()
    t2_star = float(p["T2_star"])
    decay = np.exp(-np.abs(tau) / t2_star)
    return _populations(p["p_thermal"] + p["readout_contrast"] * decay, p)


@responds(TunnelRateTune)
def tunnel_rate_tune_response(ctx: SimContext) -> SimResponse:
    r"""Tunnel rate tune: exponential decay of sensor signal vs wait time.

    .. math:: p(t) = p_{th} + C \cdot e^{-\gamma_{out} \cdot t}

    where :math:`\gamma_{out}` is in kHz → converted to 1/µs for the time axis.
    ``fit_exp`` reports ``tau`` = :math:`1/\gamma_{out}` (µs).
    """
    p = params(ctx)
    t = ctx.x()  # wait time (µs)
    gamma_out = float(p["gamma_out_khz"]) * 1e-3  # kHz → MHz (1/µs)
    decay = np.exp(-gamma_out * np.abs(t))
    return _populations(p["p_thermal"] + p["readout_contrast"] * decay, p)


@responds(ElzermanOptimize)
def elzerman_optimize_response(ctx: SimContext) -> SimResponse:
    r"""Elzerman readout optimization with tunnel-rate-dependent blip detection.

    The Elzerman scheme detects spin state via energy-selective tunneling: an excited
    spin tunnels out (producing a charge-sensor "blip"), while a ground spin cannot.
    The blip detection probability depends on the tunnel-out rate and readout window:

    .. math:: P_\text{detect}(\uparrow) = 1 - e^{-\Gamma_\text{out} \cdot t_\text{read}}

    For ground-state shots, the false-positive rate is set by thermal excitation.
    First half of sweep points are ground-prepared (low signal), second half excited.
    """
    p = params(ctx)
    x = ctx.x()
    n = len(x)

    gamma_out_khz = float(p.get("gamma_out_khz", 30.0))
    t_read_us = float(p.get("t_read_us", 100.0))
    p_thermal = float(p["p_thermal"])

    # Blip detection probability for excited state
    # P_detect = 1 - exp(-gamma_out * t_read), both in compatible units (kHz * µs = 1e-3)
    p_detect = 1.0 - np.exp(-gamma_out_khz * 1e-3 * t_read_us)

    pops = np.empty(n, dtype=float)
    half = n // 2
    # Ground-prepared: false positive from thermal excitation
    pops[:half] = p_thermal * p_detect
    # Excited-prepared: true positive from blip detection
    pops[half:] = p_detect
    return _populations(pops, p)


@responds(PSBCalibration)
def psb_calibration_response(ctx: SimContext) -> SimResponse:
    r"""PSB (Pauli spin blockade) calibration: bimodal singlet/triplet distribution.

    Same structure as Elzerman — alternating singlet (low) and triplet (high)
    preparations. ``fit_double_gaussian`` finds the threshold. Visibility is
    set by ``psb_visibility``.
    """
    p = params(ctx)
    x = ctx.x()
    n = len(x)
    vis = float(p["psb_visibility"])
    pops = np.empty(n, dtype=float)
    half = n // 2
    # First half: singlet preparation → low blockade signal
    pops[:half] = 0.5 * (1.0 - vis)
    # Second half: triplet preparation → high blockade signal
    pops[half:] = 0.5 * (1.0 + vis)
    return _populations(pops, p)


# =============================================================================
# HeraldedInit — per-shot conditional (ShotMachine)
# =============================================================================


@responds(HeraldedInit)
def heralded_init_response(ctx: SimContext) -> SimResponse:
    r"""Heralded initialization — per-shot conditional reset (ShotMachine).

    Like the cqed active reset: each shot starts with probability ``p_thermal`` of being
    excited. Each reset round measures (with readout error ``reset_readout_error``) and
    conditionally flips (with π error ``reset_pi_error``). The corrective π is identified
    by its ``RESET_PI_LABEL``.
    """
    from strumento.packs.spin.library.composite import RESET_PI_LABEL

    p = params(ctx)
    eps = float(p.get("reset_readout_error", 0.05))
    delta = float(p.get("reset_pi_error", 0.01))

    def measure(state: int, rng) -> int:
        return int(state) ^ int(rng.random() < eps)

    def apply_pulse(state: int, pulse, rng) -> int:
        if getattr(pulse, "label", None) != RESET_PI_LABEL:
            return state
        return state if rng.random() < delta else 1 - int(state)

    machine = ShotMachine(
        thermal_population=float(p["p_thermal"]),
        measure=measure,
        apply_pulse=apply_pulse,
    )
    return SimResponse(
        machine=machine,
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["sigma"]),
    )
