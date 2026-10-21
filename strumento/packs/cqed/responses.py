"""cqed Tier-1 response models (D32) — the *physics* half of the offline simulator.

:mod:`strumento.core.response` owns the mechanism (registry, shot sampling, IQ placement) and
:mod:`strumento.core.simulator` owns the soc. Neither knows what a Rabi oscillation is — that
belongs to a pack, and this is it: five closed-form responses covering the characterization
experiments this pack ships, registered into the core registry at pack import.

**Ground truth is data.** Every model reads its parameters from an ordinary dict (hard rule 4 /
D10) resolved in three layers:

1. :data:`DEFAULT_TRUTH` — simulator-side defaults for quantities a device instance holds no
   belief about (readout blob geometry, thermal population, resonator Qs). These are *simulator*
   values, not device calibration, which is why they live in code rather than an instance YAML.
2. the loaded device's calibration — the lab's current **belief** (π gain, T1, readout frequency).
3. the armed ``SimContext.truth`` — the simulator's **ground truth**, which overrides the belief
   key by key. A partial dict is enough: ``SimulatorSoc(truth={"pi_gain": 8600})`` perturbs the π
   gain and leaves everything else following the instance.

That layering is what makes a closed-loop demo non-trivial: the fit recovers the *truth*, which is
deliberately not what the device believes, and a calibration write-back closes the gap.

Models are **phenomenological**, not a Hamiltonian rollout: each is the closed form its matching
fitter fits, so a fit-recovery gate measures the *plumbing* (IR → compiler → v2 program → acquire
→ readout → fitter), not the solver. Tier 2 (a rollout response consuming
``dev.to_operator_model(...)``) is a specified, deferred seam.

Pure: no ``qick`` import — packs are imported by ``Device.load``, which runs in the pure lane.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from strumento.core.response import ShotMachine, SimContext, SimResponse, responds
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
from strumento.packs.cqed.experiments.sideband_spectroscopy import (
    SidebandSpectroscopy,
)
from strumento.packs.cqed.experiments.single_shot_readout import SingleShotReadout
from strumento.packs.cqed.experiments.state_prep_fidelity import StatePrepFidelity
from strumento.packs.cqed.experiments.storage_t1 import StorageT1
from strumento.packs.cqed.experiments.swap_rabi import SwapRabi
from strumento.packs.cqed.experiments.swap_spectroscopy import SwapSpectroscopy
from strumento.packs.cqed.experiments.t1 import T1
from strumento.packs.cqed.experiments.t2_echo import T2Echo
from strumento.packs.cqed.experiments.t2_ramsey import T2Ramsey
from strumento.packs.cqed.experiments.tls_scan import TlsScan
from strumento.packs.cqed.experiments.transmon_allxy import TransmonAllXY
from strumento.packs.cqed.experiments.transmon_rb import TransmonRB
from strumento.packs.cqed.experiments.validation_check import ValidationCheck
from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography
from strumento.packs.cqed.library.composite import RESET_PI_LABEL

#: Simulator-side defaults (NOT device calibration — see the module docstring). Every value is
#: overridable through the armed truth dict; the device's own belief fills in the rest.
DEFAULT_TRUTH: dict[str, Any] = {
    # -- qubit ------------------------------------------------------------------------------
    "pi_gain": 8192.0,          # lab-native int code at which a ge rotation reaches π (D23)
    "rabi_gain_decay": 8.0,     # amplitude-Rabi coherence envelope, in units of pi_gain
    "T1": 100.0,                # µs
    "T2": 50.0,                 # µs (T2*)
    "ramsey_detuning": 0.15,    # MHz — the qubit–drive detuning a Ramsey fringe reports
    "p_thermal": 0.02,          # residual excited population (equilibrium + prep error)
    # -- active reset: the two errors that limit a measure-and-correct round ------
    "reset_readout_error": 0.002,  # mid-circuit assignment error (the branch acts on this read)
    "reset_pi_error": 0.01,        # probability the corrective pi fails to flip the qubit
    # -- readout ----------------------------------------------------------------------------
    "readout_contrast": 0.96,   # |g>/|e> assignment contrast
    "blob0": (0.0, 0.0),        # IQ centre of the |g> blob (readout-model geometry)
    "blob1": (1.0, 0.0),        # IQ centre of the |e> blob
    "blob_sigma": 0.15,         # per-shot Gaussian scatter (blob width)
    # -- readout resonator (notch / hanger geometry, matching the circle fitter's default) ---
    "resonator_f0": 750.0,      # MHz
    "resonator_Q_loaded": 3000.0,
    "resonator_Q_coupling": 4500.0,
    "resonator_phi": 0.10,      # impedance-mismatch / asymmetry angle (rad)
    "resonator_delay": 0.02,    # cable delay in µs — the inverse of the MHz frequency axis
    "resonator_gain": 1.0,      # environmental gain of the off-resonant continuum
    "resonator_rotation": 0.30, # environmental IQ rotation (rad)
    "resonator_sigma": 0.004,   # per-shot IQ scatter on the probe response
    # -- parity -----------------------------------------------------------------------------
    "parity_phase": 45.0,       # fringe phase offset (deg) — what the parity fit reports
    "parity_contrast": 0.90,    # fringe visibility
    # -- error amplification (issue #82): the coherent parameter error already IN the belief
    # (belief = ideal + delta, so the correction a fit should recover is exactly -delta), and
    # the per-pulse phase sensitivity (cycles of amplified fringe per unit parameter deviation,
    # AFTER ONE pulse — n_pulses multiplies it, which is the tightening-fringe mechanism). Sized
    # so n_pulses=1 puts exactly one cycle across ErrorAmplification's own default window (0.4
    # MHz / 400 gain-code span) — see error_amplification_response's docstring.
    "err_amp_freq_delta": 0.0,      # MHz
    "err_amp_freq_sens": 2.5,       # cycles/MHz per pulse (x 0.4 MHz default span = 1 cycle)
    "err_amp_gain_delta": 0.0,      # lab-native int code
    "err_amp_gain_sens": 0.0025,    # cycles/int-code per pulse (x 400-code default span = 1 cycle)
    # -- qubit spectroscopy -----------------------------------------------------------------
    "f_ge": 5000.0,             # MHz — g-e transition frequency
    "spectroscopy_linewidth": 2.0,  # MHz (HWHM) — qubit spectroscopy linewidth
    # -- ef transition ---------------------------------------------------------------------
    "ef_pi_gain": 7500.0,       # lab-native int code for ef pi
    "f_ef": 3800.0,             # MHz — e-f transition frequency
    "T1_ef": 40.0,              # µs — e-f transition relaxation time (issue #90)
    "T2_ef": 20.0,              # µs — e-f transition Ramsey dephasing time, T2*_ef (issue #90)
    "ef_ramsey_detuning": 0.30, # MHz — ef-drive detuning an ef-Ramsey fringe reports; larger
                                 # than the ge Ramsey's 0.15 MHz default (issue #90 Notes: an
                                 # e-f Ramsey conventionally runs at a larger detuning)
    "T2_echo_ef": 40.0,         # µs — e-f Hahn-echo coherence time (EFT2Echo); between T2_ef and
                                 # 2·T1_ef, scaled like the ge echo/T2* ratio (80/50 ≈ 1.6·T2_ef)
    # -- cavity / bosonic modes ------------------------------------------------------------
    "cavity_T1": 2000.0,        # µs — photon lifetime
    "cavity_T2": 1000.0,        # µs — cavity dephasing time
    "cavity_detuning": 0.05,    # MHz — cavity Ramsey detuning
    "cavity_f0": 9000.0,        # MHz — cavity bare frequency
    "cavity_Q": 5000.0,         # loaded Q of cavity mode
    "chi": 1.5,                 # MHz — dispersive shift (qubit-cavity)
    "alpha_per_gain": 2.0e-4,   # coherent-state amplitude per unit displacement gain (Wigner ruler)
    "nbar": 2.0,                # mean photon number of the number-splitting coherent state
    "number_splitting_linewidth": 0.3,  # MHz (HWHM) per comb peak — narrower than chi to resolve
    # -- sideband / multiphoton ------------------------------------------------------------
    "sideband_f0g1": 500.0,     # MHz — f0-g1 transition frequency
    "sideband_pi_gain": 6000.0, # lab-native int code for sideband pi
    # -- coupler / swaps -------------------------------------------------------------------
    "swap_frequency": 200.0,    # MHz — beamsplitter swap pump frequency
    "swap_period": 0.8,         # µs — full swap time (period of swap Rabi)
    "storage_T1": 5000.0,       # µs — storage mode lifetime
    # -- qubit extras ----------------------------------------------------------------------
    "T2_echo": 80.0,            # µs — Hahn echo T2
    "rabi_freq": 5.0,           # MHz — time-domain Rabi frequency (for length Rabi)
    "drag_alpha_opt": 0.015,    # optimal DRAG Q/I peak ratio — first-order λe^{-1/2}/(σ|Δ|),
                                # λ≈0.5–1, σ≈30–35 ns, |Δ|/2π≈140–180 MHz → ~0.01–0.02
    "drag_equator_offset": 0.0,  # common-mode P_e offset at the equator (readout/thermal/T1);
                                # biases X/2·Y/2, cancels in the DRAG pair difference
    # -- power sweep -----------------------------------------------------------------------
    "g_optimal": 10000.0,       # gain at which readout signal saturates
    # -- readout optimization --------------------------------------------------------------
    "readout_optimal_length": 3.0,  # µs — optimal readout pulse length
    # -- RB / advanced ---------------------------------------------------------------------
    "rb_decay": 50.0,           # depth constant for RB exponential decay
    "qnd_fidelity": 0.95,       # QND readout fidelity
    # -- bring-up nodes (issue #56) ----------------------------------------------------------
    # the resonator decay rate κ = f0/Q_loaded of the DEFAULT resonator (750/3000) — the
    # punchout's knee phenomenology (g_knee ∝ √κ) and its gate-only estimate's truth
    "resonator_kappa_mhz": 0.25,
    "punchout_depth": 0.5,      # the low-power on-resonance notch depth
    "gain_ladder_p_delta": 1.0,   # the two-level Stark exponent (Δ ∝ s)
    "gain_ladder_p_gamma": 0.5,   # the two-level power-broadening exponent (Γ ∝ √(1+s))
    "gain_ladder_stark_mhz": 0.5, # the Stark shift per unit relative drive
    "gain_ladder_depth": 0.6,     # the lowest rung's line depth
    "gain_ladder_impostor": "",   # "" | "shift" | "saturate" — the impostor signature to arm
    "tls_line_offsets_mhz": [],   # offsets of TLS defect lines from f_ge (none by default)
    "tls_linewidth_mhz": 0.15,    # HWHM of a TLS defect line
    "tls_power_broadening": 1.5,  # the main line's broadening factor at the scan's gain
}


def truth_from_device(device=None, **overrides) -> dict[str, Any]:
    """The ground-truth dict a device's own belief implies, with ``overrides`` applied.

    Called with no overrides this is the "simulator tells the truth" case: every fit recovers the
    value the device already believes. Perturb a key (``truth_from_device(dev, pi_gain=8600)``)
    and the same experiments now measure something the device does *not* know — the setup every
    closed-loop calibration demo needs.
    """
    truth = dict(DEFAULT_TRUTH)
    calib = getattr(device, "calib", None)
    if calib is not None:
        qubit = getattr(calib, "qubit", None)
        readout = getattr(calib, "readout", None)
        if qubit is not None:
            if getattr(qubit, "f_ge", None):
                truth["f_ge"] = float(qubit.f_ge)
            if getattr(qubit, "f_ef", None):
                truth["f_ef"] = float(qubit.f_ef)
            pi_ge = (getattr(qubit, "pulses", None) or {}).get("pi_ge")
            if pi_ge is not None:
                truth["pi_gain"] = float(pi_ge.gain)
            if getattr(qubit, "T1", None):
                truth["T1"] = float(qubit.T1)
                # a conservative stand-in when the instance holds no T2 belief (T2 <= 2 T1)
                truth["T2"] = float(getattr(qubit, "T2", None) or 0.5 * float(qubit.T1))
            elif getattr(qubit, "T2", None):
                truth["T2"] = float(qubit.T2)
        if readout is not None and getattr(readout, "frequency", None):
            truth["resonator_f0"] = float(readout.frequency)
    truth.update(overrides)
    return truth


def params(ctx: SimContext) -> dict[str, Any]:
    """Resolve a model's parameters: pack defaults → device belief → armed truth (last wins)."""
    return {**truth_from_device(ctx.device), **(ctx.truth or {})}


def _populations(pops, p: dict[str, Any]) -> SimResponse:
    """Wrap populations with the readout model's blob geometry (physics stays out of core)."""
    return SimResponse(
        populations=np.clip(np.asarray(pops, dtype=float), 0.0, 1.0),
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["blob_sigma"]),
    )


# --------------------------------------------------------------------------------------
# Tier 1: closed-form responses
# --------------------------------------------------------------------------------------


@responds(AmplitudeRabi)
def amplitude_rabi_response(ctx: SimContext) -> SimResponse:
    r"""Excited population vs drive amplitude: a cosine arch decaying toward :math:`1/2`.

    .. math:: p_e(g) = p_\mathrm{th} + C\left[\tfrac12 - \tfrac12 e^{-g/g_d}
              \cos\!\left(\pi g / g_\pi\right)\right]

    A π rotation lands at ``g = pi_gain``, so ``fit_rabi``'s period is ``2·pi_gain`` and its
    derived ``pi_gain`` is the truth. ``g_d`` (``rabi_gain_decay`` × ``pi_gain``) is the
    amplitude-domain coherence envelope the same fitter carries as ``tau``.
    """
    p = params(ctx)
    g = ctx.x()
    g_pi = float(p["pi_gain"])
    g_decay = float(p["rabi_gain_decay"]) * g_pi
    arch = 0.5 - 0.5 * np.exp(-np.abs(g) / g_decay) * np.cos(np.pi * g / g_pi)
    return _populations(p["p_thermal"] + p["readout_contrast"] * arch, p)


@responds(T1)
def t1_response(ctx: SimContext) -> SimResponse:
    r"""Excited population vs idle delay: :math:`p_e(t) = p_\mathrm{th} + C e^{-t/T_1}`.

    ``fit_exp`` reports ``tau`` = T1 directly; the thermal floor is the fit's ``y0``.
    """
    p = params(ctx)
    t = ctx.x()
    decay = np.exp(-np.abs(t) / float(p["T1"]))
    return _populations(p["p_thermal"] + p["readout_contrast"] * decay, p)


@responds(EFT1)
def eft1_response(ctx: SimContext) -> SimResponse:
    r"""Excited population vs idle delay, on the e-f transition after a ge-pi shelve.

    Mirrors ``t1_response`` exactly but reads the ef-specific truth key, so the e-f relaxation
    time is independent of the g-e one: :math:`p_f(t) = p_\mathrm{th} + C e^{-t/T_{1,ef}}`.
    ``fit_exp`` reports ``tau`` = T1_ef directly, same as the ge model.
    """
    p = params(ctx)
    t = ctx.x()
    decay = np.exp(-np.abs(t) / float(p["T1_ef"]))
    return _populations(p["p_thermal"] + p["readout_contrast"] * decay, p)


@responds(T2Ramsey)
def t2_ramsey_response(ctx: SimContext) -> SimResponse:
    r"""Ramsey fringe vs idle delay — a detuned cosine under a :math:`T_2^*` envelope.

    .. math:: p_e(t) = p_\mathrm{th} + C\left[\tfrac12 + \tfrac12 e^{-t/T_2}
              \cos\!\left(2\pi\delta t + \varphi\right)\right]

    ``fit_decaysin_detuned`` reports ``freq`` = the detuning ``δ`` (MHz for ``t`` in µs) and
    ``tau`` = T2*. ``φ`` comes from the experiment's own ``advance_phase_deg`` (the second
    half-pi's phase), so a phase-flipped echo-style scan is simulated consistently.
    """
    p = params(ctx)
    t = ctx.x()
    # The observed fringe is the intrinsic qubit-drive mismatch (ramsey_detuning, truth) plus the
    # experiment's artificial detuning (ramsey_freq_mhz, applied to both half-pi drives).
    detuning = float(p["ramsey_detuning"]) + float(
        getattr(ctx.experiment, "ramsey_freq_mhz", 0.0) or 0.0
    )
    phase = np.deg2rad(float(getattr(ctx.experiment, "advance_phase_deg", 0.0) or 0.0))
    fringe = 0.5 + 0.5 * np.exp(-np.abs(t) / float(p["T2"])) * np.cos(
        2.0 * np.pi * detuning * t + phase
    )
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


@responds(RamseyDetuningPair)
def ramsey_detuning_pair_response(ctx: SimContext) -> SimResponse:
    r"""Ramsey pair at ``f_ge ± Δa``, interleaved per delay (+ then −), shaped ``(1, 2N)``.

    Same truth convention as :func:`t2_ramsey_response`: the + series' fringe is
    ``ramsey_detuning + Δa``, i.e. ``ramsey_detuning = f_ge(belief) − f_q``. So the qubit offset is
    ``δ = f_q − f_ge = −ramsey_detuning`` and the two fringes are ``|Δa − δ|`` and ``|Δa + δ|``.
    """
    p = params(ctx)
    exp = ctx.experiment
    t = np.asarray(exp._delays, dtype=float)
    delta = -float(p["ramsey_detuning"])
    da = float(exp.ramsey_freq_mhz)
    env = np.exp(-np.abs(t) / float(p["T2"]))
    pops = np.empty(2 * len(t))
    for k, f in ((0, abs(da - delta)), (1, abs(da + delta))):
        pops[k::2] = p["p_thermal"] + p["readout_contrast"] * (0.5 + 0.5 * env * np.cos(2.0 * np.pi * f * t))
    return _populations(pops.reshape(1, -1), p)


@responds(EFT2Ramsey)
def eft2ramsey_response(ctx: SimContext) -> SimResponse:
    r"""Ramsey fringe on the e-f transition, after a ge-pi shelve.

    Mirrors ``t2_ramsey_response`` exactly but reads the ef-specific truth keys, so the e-f
    Ramsey fringe is independent of the g-e one's detuning and T2*:

    .. math:: p_f(t) = p_\mathrm{th} + C\left[\tfrac12 + \tfrac12 e^{-t/T_{2,ef}}
              \cos\!\left(2\pi\delta_{ef} t + \varphi\right)\right]

    ``fit_decaysin_detuned`` reports ``freq`` = the ef detuning ``δ_ef`` and ``tau`` = T2*_ef,
    same as the ge model. ``φ`` comes from the experiment's own ``advance_phase_deg``.
    """
    p = params(ctx)
    t = ctx.x()
    # The observed fringe is the intrinsic ef-drive mismatch (ef_ramsey_detuning, truth) plus the
    # experiment's artificial detuning (ramsey_freq_mhz, now wired into both ef half-pi drives),
    # mirroring t2_ramsey_response for the ge transition.
    detuning = float(p["ef_ramsey_detuning"]) + float(
        getattr(ctx.experiment, "ramsey_freq_mhz", 0.0) or 0.0
    )
    phase = np.deg2rad(float(getattr(ctx.experiment, "advance_phase_deg", 0.0) or 0.0))
    fringe = 0.5 + 0.5 * np.exp(-np.abs(t) / float(p["T2_ef"])) * np.cos(
        2.0 * np.pi * detuning * t + phase
    )
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


@responds(EFT2Echo)
def eft2echo_response(ctx: SimContext) -> SimResponse:
    r"""Hahn echo on the e-f transition, after a ge-pi shelve (no oscillation).

    Mirrors ``t2_echo_response`` exactly but reads the ef-specific truth key, so the e-f echo
    coherence time is independent of the g-e one, over half-delay :math:`\tau`:

    .. math:: p_f(\tau) = p_\mathrm{th} + C e^{-2\tau/T_{2,\mathrm{echo},ef}}

    ``fit_exp`` reports ``tau`` (the half-delay decay constant) and the experiment derives
    ``T2_echo = 2·tau``, same as the ge model.
    """
    p = params(ctx)
    tau = ctx.x()
    decay = np.exp(-2.0 * tau / float(p["T2_echo_ef"]))
    return _populations(p["p_thermal"] + p["readout_contrast"] * decay, p)


@responds(ResonatorSpectroscopy)
def resonator_response(ctx: SimContext) -> SimResponse:
    r"""Complex probe transmission across the readout resonator (notch / hanger geometry).

    .. math:: S_{21}(f) = a e^{i\alpha} e^{-2\pi i f \tau}
              \left[1 - \frac{(Q_l/Q_c) e^{i\phi}}{1 + 2 i Q_l (f - f_0)/f_0}\right]

    The three real-world non-idealities the circle fitter exists to undo are all present and all
    truth-driven: environmental gain/rotation (``a``, ``α``), cable delay (``τ``, in µs against
    the MHz axis), and the impedance-mismatch/Fano angle ``φ``. A magnitude Lorentzian fit sees
    the same trace; the circle fit is what recovers ``Q_loaded`` (and ``Q_internal``).
    """
    p = params(ctx)
    f = ctx.x()
    q_l = float(p["resonator_Q_loaded"])
    q_c = float(p["resonator_Q_coupling"])
    f0 = float(p["resonator_f0"])
    resonance = 1.0 - (q_l / q_c) * np.exp(1j * float(p["resonator_phi"])) / (
        1.0 + 2j * q_l * (f - f0) / f0
    )
    env = float(p["resonator_gain"]) * np.exp(
        1j * (float(p["resonator_rotation"]) - 2.0 * np.pi * f * float(p["resonator_delay"]))
    )
    return SimResponse(iq=env * resonance, sigma=float(p["resonator_sigma"]))


@responds(ResonatorPowerSweep)
def resonator_power_sweep_response(ctx: SimContext) -> SimResponse:
    """The resonator response scaled by the host-selected probe-gain row.

    The calibrated row retains the ordinary resonator model used by the validation fit. The
    zero-gain row has no coherent stimulus, leaving only the simulator's receiver noise.
    """
    base = resonator_response(ctx)
    exp = ctx.experiment
    scale = float(exp.active_gain_code) / float(exp.reference_gain_code)
    return SimResponse(iq=np.asarray(base.iq) * scale, sigma=base.sigma)


@responds(ActiveReset)
def active_reset_response(ctx: SimContext) -> SimResponse:
    r"""Active reset — the one Tier-1 response that is **per-shot conditional**, not a curve.

    There is no population function of a swept variable to return: what a shot ends up in depends
    on what *that shot* read. So this model hands back a :class:`~strumento.core.response.ShotMachine`
    — the mechanism walks the sequence's ``Branch``/``Repeat`` structure per shot and this pack
    supplies the three physical questions the walk asks:

    * how a shot **starts** — excited with probability ``p_thermal`` (equilibrium + prep error);
    * what a **readout** returns — the true state, flipped with probability
      ``reset_readout_error`` (the mid-circuit assignment error the *branch* then acts on, which is
      why a false positive genuinely excites a ground shot);
    * what the **corrective π** does — flips the state unless it fails, with probability
      ``reset_pi_error``. Only the composite's labelled π is a correction; every other pulse in the
      sequence is inert to this model (it models the reset, not the experiment around it).

    One round therefore leaves an excited population of

    .. math:: p_e' = p_1\left[(1-\varepsilon)\delta + \varepsilon\right] + (1-p_1)\varepsilon(1-\delta)

    for thermal :math:`p_1`, readout error :math:`\varepsilon` and π error :math:`\delta` — the
    familiar result that reset trades a thermal population for a *readout-fidelity-limited* one.
    """
    p = params(ctx)
    eps = float(p["reset_readout_error"])
    delta = float(p["reset_pi_error"])

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
        sigma=float(p["blob_sigma"]),
    )


@responds(ParityFringe)
def parity_fringe_response(ctx: SimContext) -> SimResponse:
    r"""Parity fringe vs the second half-pi's phase (degrees).

    .. math:: p_e(\varphi) = p_\mathrm{th} + C\left[\tfrac12 + \tfrac12 P
              \cos\!\left(\varphi + \varphi_0\right)\right]

    ``P`` is the fringe visibility (the parity signal) and ``φ₀`` the fringe phase offset —
    exactly the two quantities ``fit_parity`` reports as ``contrast`` and ``phase_deg``.
    """
    p = params(ctx)
    phi = np.deg2rad(ctx.x())
    fringe = 0.5 + 0.5 * float(p["parity_contrast"]) * np.cos(
        phi + np.deg2rad(float(p["parity_phase"]))
    )
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


@responds(ErrorAmplification)
def error_amplification_response(ctx: SimContext) -> SimResponse:
    r"""Error-amplified fringe vs the swept frequency/gain (issue #82).

    A per-pulse rotation-angle error, linear in the deviation of the swept parameter ``x`` from
    the TRUE zero-error value :math:`x_\mathrm{ideal}`, compounds over ``n_pulses`` repeats into
    a population fringe whose cycles-per-``x`` scale with ``n_pulses`` — tighter fringes for more
    repeats, the defining error-amplification signature:

    .. math:: p_e(x) = p_\mathrm{th} + C\left[\tfrac12 + \tfrac12
              \cos\!\left(2\pi\, n\, \kappa\, (x - x_\mathrm{ideal})\right)\right]

    :math:`\kappa` (``err_amp_freq_sens``/``err_amp_gain_sens``) is the per-pulse phase
    sensitivity and :math:`n` is ``n_pulses``. :math:`x_\mathrm{ideal} = x_\mathrm{belief} -
    \delta`, where :math:`\delta` (``err_amp_freq_delta``/``err_amp_gain_delta``) is the
    coherent error already present in the belief (``f_ge``/``pi_gain``) — so the correction a
    fit should recover, :math:`x_\mathrm{ideal} - x_\mathrm{belief}`, is exactly :math:`-\delta`.
    This is precisely ``fit_parity``'s cosine model fit against the swept parameter instead of a
    phase, which is why :class:`~strumento.packs.cqed.experiments.error_amplification.ErrorAmplification`
    reuses it rather than adding a new fitter.
    """
    p = params(ctx)
    exp = ctx.experiment
    x = ctx.x()
    n = int(getattr(exp, "n_pulses", 1))
    parameter = getattr(exp, "parameter", "frequency")
    if parameter == "gain":
        belief = float(p["pi_gain"])
        delta = float(p["err_amp_gain_delta"])
        kappa = float(p["err_amp_gain_sens"])
    else:
        belief = float(p["f_ge"])
        delta = float(p["err_amp_freq_delta"])
        kappa = float(p["err_amp_freq_sens"])
    x_ideal = belief - delta
    fringe = 0.5 + 0.5 * np.cos(2.0 * np.pi * n * kappa * (x - x_ideal))
    period = getattr(exp, "fringe_period", None)
    if period:
        # Fixed-period experiments state the physical fringe period (e.g. 2·g_π/N for N π pulses)
        # and sweep around THEIR pulse's belief (pi_ge or hpi_ge) — the window centre — so the sim
        # honours both rather than the generic sensitivity + pi_gain belief.
        window = exp.window
        center = 0.5 * (float(window.start) + float(window.stop))
        x_ideal = (center if parameter == "gain" else belief) - delta
        fringe = 0.5 + 0.5 * np.cos(2.0 * np.pi * (x - x_ideal) / float(period))
        # The fixed-period analyzer reads a blob-anchored POPULATION, so place the shots on the
        # device's recorded centres (as the DRAG responses do) — default blobs would read garbage.
        return _fock_readout_response(p["p_thermal"] + p["readout_contrast"] * fringe, ctx, p)
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


# --------------------------------------------------------------------------------------
# Helper: Lorentzian lineshape
# --------------------------------------------------------------------------------------


def _lorentzian(f: np.ndarray, f0: float, hwhm: float) -> np.ndarray:
    """Normalised Lorentzian (peak = 1 at f = f0)."""
    return 1.0 / (1.0 + ((f - f0) / hwhm) ** 2)


# --------------------------------------------------------------------------------------
# Spectroscopy
# --------------------------------------------------------------------------------------


@responds(QubitSpectroscopy)
def qubit_spectroscopy_response(ctx: SimContext) -> SimResponse:
    r"""Two-tone spectroscopy: Lorentzian dip in excited population at :math:`f_{ge}`.

    .. math:: p_e(f) = p_\mathrm{th} + C\left[1 - L(f; f_{ge}, \gamma)\right]
    """
    p = params(ctx)
    f = ctx.x()
    dip = 1.0 - _lorentzian(f, float(p["f_ge"]), float(p["spectroscopy_linewidth"]))
    return _populations(p["p_thermal"] + p["readout_contrast"] * dip, p)


@responds(EFSpectroscopy)
def ef_spectroscopy_response(ctx: SimContext) -> SimResponse:
    r"""e-f spectroscopy: Lorentzian at :math:`f_{ef}` (qubit already prepared in |e⟩)."""
    p = params(ctx)
    f = ctx.x()
    dip = 1.0 - _lorentzian(f, float(p["f_ef"]), float(p["spectroscopy_linewidth"]))
    return _populations(p["readout_contrast"] * dip, p)


@responds(CavitySpectroscopy)
def cavity_spectroscopy_response(ctx: SimContext) -> SimResponse:
    r"""Complex transmission through the cavity mode — a simple Lorentzian notch.

    .. math:: S_{21}(f) = 1 - \frac{A}{1 + 2iQ(f - f_0)/f_0}
    """
    p = params(ctx)
    f = ctx.x()
    f0 = float(p["cavity_f0"])
    Q = float(p["cavity_Q"])
    iq = 1.0 - 1.0 / (1.0 + 2j * Q * (f - f0) / f0)
    return SimResponse(iq=iq, sigma=float(p["resonator_sigma"]))


@responds(DispersiveShift)
def dispersive_shift_response(ctx: SimContext) -> SimResponse:
    r"""Readout resonator transmission shifted by :math:`\chi` (qubit in |e⟩).

    Same notch model as resonator_response but with ``f0 → f0 + χ``.
    """
    p = params(ctx)
    f = ctx.x()
    q_l = float(p["resonator_Q_loaded"])
    q_c = float(p["resonator_Q_coupling"])
    f0 = float(p["resonator_f0"]) + float(p["chi"])
    resonance = 1.0 - (q_l / q_c) * np.exp(1j * float(p["resonator_phi"])) / (
        1.0 + 2j * q_l * (f - f0) / f0
    )
    env = float(p["resonator_gain"]) * np.exp(
        1j * (float(p["resonator_rotation"]) - 2.0 * np.pi * f * float(p["resonator_delay"]))
    )
    return SimResponse(iq=env * resonance, sigma=float(p["resonator_sigma"]))


@responds(PowerSweep)
def power_sweep_response(ctx: SimContext) -> SimResponse:
    r"""Signal vs readout power (gain): saturating exponential.

    .. math:: p_e(g) = C\left[1 - e^{-g/g_\mathrm{opt}}\right]
    """
    p = params(ctx)
    g = ctx.x()
    signal = 1.0 - np.exp(-np.abs(g) / float(p["g_optimal"]))
    return _populations(p["readout_contrast"] * signal, p)


@responds(NumberSplitting)
def number_splitting_response(ctx: SimContext) -> SimResponse:
    r"""Number-split qubit spectroscopy: a Poisson-weighted comb spaced by :math:`\chi`.

    .. math:: p_e(f) = p_\mathrm{th} + C \sum_{n} P(n; \bar n)\, L(f; f_{ge} - n\chi, \gamma)

    Displacing the cavity to a coherent state populates Fock ``n`` with Poisson weight
    ``P(n; n̄)``; each shifts the qubit by ``n·χ``, so the spectrum is a comb of peaks below
    ``f_ge`` spaced by ``χ`` — the quantity ``fit_number_splitting_comb`` recovers from the
    spacing (not one peak's offset).
    """
    p = params(ctx)
    f = ctx.x()
    chi = abs(float(p["chi"]))
    nbar = float(p["nbar"])
    gamma = float(p["number_splitting_linewidth"])
    f_ge = float(p["f_ge"])
    nmax = 6
    comb = np.zeros_like(f, dtype=float)
    for n in range(nmax + 1):
        weight = nbar ** n * math.exp(-nbar) / math.factorial(n)
        comb = comb + weight * _lorentzian(f, f_ge - n * chi, gamma)
    return _populations(p["p_thermal"] + p["readout_contrast"] * comb, p)


# --------------------------------------------------------------------------------------
# Transmon calibration
# --------------------------------------------------------------------------------------


@responds(LengthRabi)
def length_rabi_response(ctx: SimContext) -> SimResponse:
    r"""Damped Rabi oscillation vs pulse length.

    .. math:: p_e(t) = \tfrac12 + \tfrac12 e^{-t/T_1}\cos(2\pi f_R t)
    """
    p = params(ctx)
    t = ctx.x()
    osc = 0.5 + 0.5 * np.exp(-np.abs(t) / float(p["T1"])) * np.cos(
        2.0 * np.pi * float(p["rabi_freq"]) * t
    )
    return _populations(p["p_thermal"] + p["readout_contrast"] * osc, p)


@responds(T2Echo)
def t2_echo_response(ctx: SimContext) -> SimResponse:
    r"""Hahn echo decay over half-delay :math:`\tau` (no oscillation).

    .. math:: p_e(\tau) = p_\mathrm{th} + C e^{-2\tau/T_{2,\mathrm{echo}}}
    """
    p = params(ctx)
    tau = ctx.x()
    decay = np.exp(-2.0 * tau / float(p["T2_echo"]))
    return _populations(p["p_thermal"] + p["readout_contrast"] * decay, p)


@responds(DRAGCalibration)
def drag_calibration_response(ctx: SimContext) -> SimResponse:
    r"""DRAG coefficient scan: accumulated phase error vs DRAG alpha.

    Multi-read response (one read per alpha value, no loop). Near the optimal alpha the signal is
    linear (the dominant error is first-order in :math:`\alpha - \alpha_\mathrm{opt}`); away from it
    a ``tanh`` saturates it inside population bounds, monotonically, so any sweep sees exactly one
    equator crossing. The width is a few times the optimum's own scale (α ~ 0.01–0.02 for a
    σ ≈ 35 ns pulse), so the default ±0.1 scan sees both the linear region and the saturation.

    The zero-crossing at :math:`\alpha_\mathrm{opt}` is what the analyzer extracts.
    """
    p = params(ctx)
    exp = ctx.experiment
    alphas = np.asarray(getattr(exp, "_alphas", np.linspace(-0.1, 0.1, 21)), dtype=float)
    alpha_opt = float(p["drag_alpha_opt"])
    delta_alpha = alphas - alpha_opt

    # Phenomenological (D32: plumbing, not physics): 0.5 + A·tanh(Δα / w).
    amplitude = 0.45
    width = 0.04
    signal = 0.5 + amplitude * np.tanh(delta_alpha / width)
    pops = np.clip(signal, 0.0, 1.0)
    # Place the shots on the recorded g/e blob centres when calibrated, so the sim IQ lands where
    # the analyzer's population conversion (``DispersiveIQ.to_population``, anchored to those same
    # centres) expects it — the same shape a hardware capture emits. Without this the sim IQ sits at
    # the default (0,0)/(1,0) blobs while the calibrated readout projects onto the rotated g→e axis,
    # so ``to_population`` would collapse the scan and the equator crossing would vanish.
    # Falls back to the default blobs when no centres are recorded.
    return _fock_readout_response(pops.reshape(1, -1), ctx, p)


@responds(DRAGPairCalibration)
def drag_pair_calibration_response(ctx: SimContext) -> SimResponse:
    r"""DRAG pair: A = X(π/2)·Y(π) and B = Y(π/2)·X(π), interleaved per α (A then B).

    Phenomenological (D32): the DRAG error tilts the two blocks with opposite signs,
    :math:`P_{A,B} = \tfrac12 + \epsilon \pm A\tanh\big(s(\alpha-\alpha_\mathrm{opt})/A\big)` with the
    3-level model's per-block slope ``s = 1.56`` and ``A = 0.45``. ``drag_equator_offset`` (ε) is
    common to both — it shifts a single X/2·Y/2 crossing and cancels in ``P_A − P_B``.
    """
    p = params(ctx)
    exp = ctx.experiment
    alphas = np.asarray(getattr(exp, "_alphas", np.linspace(-0.25, 0.25, 41)), dtype=float)
    tilt = 0.45 * np.tanh(1.56 * (alphas - float(p["drag_alpha_opt"])) / 0.45)
    offset = float(p.get("drag_equator_offset", 0.0))
    pops = np.empty(2 * len(alphas))
    pops[0::2] = 0.5 + offset + tilt
    pops[1::2] = 0.5 + offset - tilt
    return _fock_readout_response(np.clip(pops, 0.0, 1.0).reshape(1, -1), ctx, p)


@responds(TransmonAllXY)
def allxy_response(ctx: SimContext) -> SimResponse:
    r"""AllXY: 21 ideal populations (one per gate pair, shaped for multi-read).

    Returns the ideal staircase perturbed by a small amplitude error from the truth,
    shaped ``(1, 21)`` so the simulator generates one value per read in a 21-read shot.
    """
    p = params(ctx)
    from strumento.packs.cqed.experiments.transmon_allxy import ALLXY_PAIRS
    ideal = np.array([pair[2] for pair in ALLXY_PAIRS], dtype=float)
    # Perturb by a small amplitude error (shifts the equator points)
    amp_err = 0.02 * (1.0 - float(p["readout_contrast"]))
    perturbed = np.clip(ideal + amp_err, 0.0, 1.0)
    return SimResponse(
        populations=perturbed.reshape(1, -1),
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["blob_sigma"]),
    )


@responds(EFRabi)
def ef_rabi_response(ctx: SimContext) -> SimResponse:
    r"""Amplitude Rabi on the e-f transition. Same arch shape, period = ``2·ef_pi_gain``.

    .. math:: p_e(g) = p_\mathrm{th} + C\left[\tfrac12 - \tfrac12
              e^{-g/g_d}\cos(\pi g / g_{\pi,ef})\right]
    """
    p = params(ctx)
    g = ctx.x()
    g_pi = float(p["ef_pi_gain"])
    g_decay = float(p["rabi_gain_decay"]) * g_pi
    arch = 0.5 - 0.5 * np.exp(-np.abs(g) / g_decay) * np.cos(np.pi * g / g_pi)
    return _populations(p["p_thermal"] + p["readout_contrast"] * arch, p)


@responds(FrequencyRabi)
def frequency_rabi_response(ctx: SimContext) -> SimResponse:
    r"""Rabi contrast vs drive frequency: Lorentzian envelope centred at :math:`f_{ge}`.

    .. math:: p_e(f) = p_\mathrm{th} + C \cdot L(f; f_{ge}, \gamma)
    """
    p = params(ctx)
    f = ctx.x()
    envelope = _lorentzian(f, float(p["f_ge"]), float(p["spectroscopy_linewidth"]))
    return _populations(p["p_thermal"] + p["readout_contrast"] * envelope, p)


@responds(SingleShotReadout)
def single_shot_readout_response(ctx: SimContext) -> SimResponse:
    r"""Single-shot readout: per-shot conditional via ShotMachine.

    Two reads per shot: read 0 = ground state, read 1 = after ge_pi (excited). The
    ShotMachine walks the sequence per shot, so the two reads within one shot are
    correlated (same qubit) — matching hardware behavior and enabling ``fit_double_gaussian``
    on per-shot data.
    """
    p = params(ctx)
    eps = float(p.get("readout_error", 0.02))
    delta = float(p.get("pi_error", 0.005))

    def measure(state: int, rng) -> int:
        return int(state) ^ int(rng.random() < eps)

    def apply_pulse(state: int, pulse, rng) -> int:
        # The ge_pi flips the state (with small error)
        if getattr(pulse, "label", None) in ("pi_ge", None):
            return state if rng.random() < delta else 1 - int(state)
        return state

    machine = ShotMachine(
        thermal_population=float(p["p_thermal"]),
        measure=measure,
        apply_pulse=apply_pulse,
    )
    return SimResponse(
        machine=machine,
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["blob_sigma"]),
    )


@responds(ReadoutOptimization)
def readout_optimization_response(ctx: SimContext) -> SimResponse:
    r"""Signal vs readout length: saturating exponential.

    .. math:: p_e(L) = C\left[1 - e^{-L/L_\mathrm{opt}}\right]
    """
    p = params(ctx)
    length = ctx.x()
    signal = 1.0 - np.exp(-np.abs(length) / float(p["readout_optimal_length"]))
    return _populations(p["readout_contrast"] * signal, p)


# --------------------------------------------------------------------------------------
# Sideband / multiphoton
# --------------------------------------------------------------------------------------


@responds(SidebandSpectroscopy)
def sideband_spectroscopy_response(ctx: SimContext) -> SimResponse:
    r"""f0-g1 sideband spectroscopy: Lorentzian peak at the sideband frequency.

    .. math:: p_e(f) = C \cdot L(f; f_{0g1}, \gamma)
    """
    p = params(ctx)
    f = ctx.x()
    peak = _lorentzian(f, float(p["sideband_f0g1"]), float(p["spectroscopy_linewidth"]))
    return _populations(p["readout_contrast"] * peak, p)


@responds(SidebandRabi)
def sideband_rabi_response(ctx: SimContext) -> SimResponse:
    r"""Mapped sideband Rabi vs gain. Period = ``2·sideband_pi_gain``.

    :class:`SidebandRabi` closes with an ef pi: no sideband transfer leaves f mapped to bright e,
    while a pi f0-g1 transfer leaves g,1 dark on the ge readout. The simulator therefore returns
    the complement of the transfer arch.
    """
    p = params(ctx)
    g = ctx.x()
    g_pi = float(p["sideband_pi_gain"])
    g_decay = float(p["rabi_gain_decay"]) * g_pi
    arch = 0.5 - 0.5 * np.exp(-np.abs(g) / g_decay) * np.cos(np.pi * g / g_pi)
    exp = ctx.experiment
    measured_population = 1.0 - arch if getattr(exp, "post_map_ef", True) else arch
    return _populations(p["p_thermal"] + p["readout_contrast"] * measured_population, p)


@responds(MultiphotonLadder)
def multiphoton_ladder_response(ctx: SimContext) -> SimResponse:
    r"""Multiphoton ladder: alternating parity for Fock(0), Fock(1), ..., Fock(n_max).

    Multi-read response: even Fock → population >0.5 (positive parity),
    odd Fock → population <0.5 (negative parity).
    """
    p = params(ctx)
    exp = ctx.experiment
    fock_numbers = getattr(exp, "_fock_numbers", [0, 1, 2])
    contrast = float(p["parity_contrast"])
    # Even n: population = 0.5 + contrast/2; odd n: population = 0.5 - contrast/2
    pops = np.array([
        0.5 + 0.5 * contrast if n % 2 == 0 else 0.5 - 0.5 * contrast
        for n in fock_numbers
    ], dtype=float)
    return SimResponse(
        populations=pops.reshape(1, -1),
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["blob_sigma"]),
    )


# --------------------------------------------------------------------------------------
# Cavity operations
# --------------------------------------------------------------------------------------


@responds(DisplacementCalibration)
def displacement_calibration_response(ctx: SimContext) -> SimResponse:
    r"""Parity of a displaced vacuum vs displacement gain — the vacuum Wigner Gaussian.

    .. math:: p_e(g) = \tfrac12 + \tfrac12 C\,e^{-2\,(k\,g)^2}

    with ``k`` = ``alpha_per_gain`` (``α = k·g``). Displaced-vacuum parity is
    ``⟨Π⟩ = e^{-2|α|^2}`` — a monotonic Gaussian in gain, not an oscillation — so
    ``fit_gaussian_wigner`` recovers ``k`` from the Gaussian width.
    """
    p = params(ctx)
    g = ctx.x()
    k = float(p["alpha_per_gain"])
    parity = np.exp(-2.0 * (k * g) ** 2)
    fringe = 0.5 + 0.5 * float(p["parity_contrast"]) * parity
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


@responds(CavityT1)
def cavity_t1_response(ctx: SimContext) -> SimResponse:
    r"""Cavity photon decay: parity contrast decays exponentially at cavity :math:`T_1`.

    .. math:: p_e(t) = \tfrac12 + \tfrac12 C e^{-t/T_{1,\mathrm{cav}}}
    """
    p = params(ctx)
    t = ctx.x()
    fringe = 0.5 + 0.5 * float(p["parity_contrast"]) * np.exp(
        -np.abs(t) / float(p["cavity_T1"])
    )
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


@responds(FockCavityT1)
def fock_cavity_t1_response(ctx: SimContext) -> SimResponse:
    r"""Direct single-photon survival, mapped to bright qubit population after retrieval."""
    p = params(ctx)
    t = ctx.x()
    populations = p["p_thermal"] + p["readout_contrast"] * np.exp(
        -np.abs(t) / float(p["cavity_T1"])
    )
    return _fock_readout_response(populations, ctx, p)


def _fock_readout_response(populations, ctx: SimContext, p: dict[str, Any]) -> SimResponse:
    """Place a direct Fock-mode population response on recorded hardware blob centers when present."""
    readout = getattr(getattr(ctx.device, "calib", None), "readout", None)
    g_center = getattr(readout, "g_center_iq", None)
    e_center = getattr(readout, "e_center_iq", None)
    scale = float(ctx.readout_output_scale)
    calibrated_blobs = g_center is not None and e_center is not None
    return SimResponse(
        populations=np.clip(np.asarray(populations, dtype=float), 0.0, 1.0),
        blob0=(
            tuple(np.asarray(g_center, dtype=float) / scale)
            if calibrated_blobs
            else tuple(p["blob0"])
        ),
        blob1=(
            tuple(np.asarray(e_center, dtype=float) / scale)
            if calibrated_blobs
            else tuple(p["blob1"])
        ),
        sigma=float(p["blob_sigma"]) / scale if calibrated_blobs else float(p["blob_sigma"]),
    )


@responds(FockCavityT2Ramsey)
def fock_cavity_t2_ramsey_response(ctx: SimContext) -> SimResponse:
    r"""Single-rail Ramsey coherence under a virtual return-sideband phase reference."""
    p = params(ctx)
    t = ctx.x()
    exp = ctx.experiment
    frequency = float(exp.reference_detuning_mhz) + float(p["cavity_detuning"])
    fringe = 0.5 + 0.5 * np.exp(-np.abs(t) / float(p["cavity_T2"])) * np.cos(
        2.0 * np.pi * frequency * t
    )
    populations = p["p_thermal"] + p["readout_contrast"] * fringe
    return _fock_readout_response(populations, ctx, p)


@responds(CavityT2Ramsey)
def cavity_t2_ramsey_response(ctx: SimContext) -> SimResponse:
    r"""Cavity Ramsey: damped cosine at the cavity detuning frequency.

    .. math:: p_e(t) = \tfrac12 + \tfrac12 C e^{-t/T_{2,\mathrm{cav}}}
              \cos(2\pi\delta_c t)
    """
    p = params(ctx)
    t = ctx.x()
    fringe = 0.5 + 0.5 * float(p["parity_contrast"]) * np.exp(
        -np.abs(t) / float(p["cavity_T2"])
    ) * np.cos(2.0 * np.pi * float(p["cavity_detuning"]) * t)
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


@responds(RevivalTimeCalibration)
def revival_time_calibration_response(ctx: SimContext) -> SimResponse:
    r"""Qubit Ramsey with photon-dependent shift: oscillation at :math:`\chi`.

    .. math:: p_e(t) = \tfrac12 + \tfrac12 C e^{-t/T_2}\cos(2\pi\chi t)
    """
    p = params(ctx)
    t = ctx.x()
    fringe = 0.5 + 0.5 * float(p["readout_contrast"]) * np.exp(
        -np.abs(t) / float(p["T2"])
    ) * np.cos(2.0 * np.pi * float(p["chi"]) * t)
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


@responds(PhotonNumberDistribution)
def photon_number_distribution_response(ctx: SimContext) -> SimResponse:
    r"""Photon number distribution: Poisson-distributed selective-pi response for vacuum.

    Multi-read response (one read per photon number). For a coherent state |alpha⟩ the
    probability of n photons is Poisson: P(n) = |alpha|^{2n} e^{-|alpha|^2} / n!.
    For vacuum (alpha=0): P(0)=1, P(n>0)=0.
    """
    p = params(ctx)
    exp = ctx.experiment
    photon_numbers = np.array(getattr(exp, "_photon_numbers", list(range(7))), dtype=float)
    # Vacuum state: only n=0 has population
    pops = np.zeros_like(photon_numbers)
    pops[0] = float(p["readout_contrast"])
    return SimResponse(
        populations=pops.reshape(1, -1),
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["blob_sigma"]),
    )


@responds(WignerTomography)
def wigner_tomography_response(ctx: SimContext) -> SimResponse:
    r"""Wigner sampling of a prepared coherent state along the tomography ray.

    The mode is prepared in ``|init_alpha⟩`` (real axis) and sampled at displacement
    ``β = |β|·e^{i·beta_phase_deg}`` along the swept magnitude axis:

    .. math:: p_e(\beta) = \tfrac12 + \tfrac12 C\, e^{-2\,|\beta - \alpha_0|^2}

    Vacuum (``init_alpha = 0``, ``beta_phase_deg = 0``) recovers the plain ``e^{-2|\beta|^2}``
    Gaussian, so the default-case response is unchanged. (Phenomenological: the free Kerr wait does
    not evolve the sim state — ``--sim`` proves the plumbing, not the physics.)
    """
    p = params(ctx)
    exp = ctx.experiment
    beta = ctx.x()
    a0 = float(getattr(exp, "init_alpha", 0.0))
    phi = np.radians(float(getattr(exp, "beta_phase_deg", 0.0)))
    bx = beta * np.cos(phi)
    by = beta * np.sin(phi)
    d2 = (bx - a0) ** 2 + by ** 2
    wigner = 0.5 + 0.5 * float(p["parity_contrast"]) * np.exp(-2.0 * d2)
    return _populations(p["p_thermal"] + p["readout_contrast"] * wigner, p)


# --------------------------------------------------------------------------------------
# Coupler / swaps
# --------------------------------------------------------------------------------------


@responds(SwapSpectroscopy)
def swap_spectroscopy_response(ctx: SimContext) -> SimResponse:
    r"""Swap spectroscopy: parity dips at the swap pump frequency.

    .. math:: p_e(f) = \tfrac12 - 0.3\,L(f; f_\mathrm{swap}, \gamma)
    """
    p = params(ctx)
    f = ctx.x()
    dip = 0.5 - 0.3 * _lorentzian(f, float(p["swap_frequency"]), float(p["spectroscopy_linewidth"]))
    return _populations(dip, p)


@responds(SwapRabi)
def swap_rabi_response(ctx: SimContext) -> SimResponse:
    r"""Swap Rabi: parity oscillation vs pump duration.

    .. math:: p_e(t) = \tfrac12 + \tfrac12 C\cos(2\pi t / T_\mathrm{swap})
    """
    p = params(ctx)
    t = ctx.x()
    fringe = 0.5 + 0.5 * float(p["parity_contrast"]) * np.cos(
        2.0 * np.pi * t / float(p["swap_period"])
    )
    return _populations(fringe, p)


@responds(StorageT1)
def storage_t1_response(ctx: SimContext) -> SimResponse:
    r"""Storage mode photon decay: parity contrast decays at storage :math:`T_1`.

    .. math:: p_e(t) = \tfrac12 + \tfrac12 C e^{-t/T_{1,\mathrm{stor}}}
    """
    p = params(ctx)
    t = ctx.x()
    fringe = 0.5 + 0.5 * float(p["parity_contrast"]) * np.exp(
        -np.abs(t) / float(p["storage_T1"])
    )
    return _populations(fringe, p)


@responds(CrossKerrMeasurement)
def cross_kerr_response(ctx: SimContext) -> SimResponse:
    r"""Cross-Kerr: qubit Ramsey fringe at :math:`\chi` (photon-number-dependent dephasing).

    .. math:: p_e(t) = \tfrac12 + \tfrac12 C e^{-t/T_2}\cos(2\pi\chi t)
    """
    p = params(ctx)
    t = ctx.x()
    fringe = 0.5 + 0.5 * float(p["readout_contrast"]) * np.exp(
        -np.abs(t) / float(p["T2"])
    ) * np.cos(2.0 * np.pi * float(p["chi"]) * t)
    return _populations(p["p_thermal"] + p["readout_contrast"] * fringe, p)


# --------------------------------------------------------------------------------------
# QEC / advanced
# --------------------------------------------------------------------------------------


@responds(QNDReadout)
def qnd_readout_response(ctx: SimContext) -> SimResponse:
    r"""QND readout: repeated measurement of |e⟩ with gradual T1 decay.

    Multi-read response (one read per repetition). Each read has a slightly lower
    excited-state population due to T1 decay during the readout window. A perfect
    QND readout would show no decay; the ``qnd_fidelity`` truth sets the per-read
    retention probability.
    """
    p = params(ctx)
    exp = ctx.experiment
    n_repeats = getattr(exp, "n_repeats", 5)
    fidelity = float(p["qnd_fidelity"])
    # Model: population decays slightly with each read due to measurement-induced T1
    # P(read_k still excited) = fidelity^k
    pops = np.array([fidelity ** k for k in range(n_repeats)], dtype=float)
    return SimResponse(
        populations=pops.reshape(1, -1),
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["blob_sigma"]),
    )


@responds(StatePrepFidelity)
def state_prep_fidelity_response(ctx: SimContext) -> SimResponse:
    r"""State preparation fidelity (vacuum target): Gaussian Wigner.

    .. math:: p_e(\beta) = \tfrac12 + \tfrac12 C e^{-2|\beta|^2}
    """
    p = params(ctx)
    beta = ctx.x()
    wigner = 0.5 + 0.5 * float(p["parity_contrast"]) * np.exp(-2.0 * beta ** 2)
    return _populations(p["p_thermal"] + p["readout_contrast"] * wigner, p)


@responds(TransmonRB)
def transmon_rb_response(ctx: SimContext) -> SimResponse:
    r"""Randomised benchmarking: exponential decay of survival probability vs depth.

    Multi-read response (one read per depth point, no loop). The depths are read from
    the experiment's ``_depths`` attribute; the decay follows :math:`e^{-m/m_d}`.
    """
    p = params(ctx)
    exp = ctx.experiment
    depths = np.asarray(getattr(exp, "_depths", [1, 10, 50, 100]), dtype=float)
    decay = np.exp(-depths / float(p["rb_decay"]))
    pops = float(p["p_thermal"]) + float(p["readout_contrast"]) * decay
    return SimResponse(
        populations=np.clip(pops, 0.0, 1.0).reshape(1, -1),
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["blob_sigma"]),
    )


# --------------------------------------------------------------------------------------
# Bring-up graph nodes (issue #56) — the thin Seq experiments' response models
# --------------------------------------------------------------------------------------


@responds(Punchout)
def punchout_response(ctx: SimContext) -> SimResponse:
    r"""On-resonance transmission vs probe gain: the dip shallows past the critical photon
    number — the punchout.

    Phenomenology: the normalized on-resonance transmission is
    :math:`V(g) = 1 - d_0 / (1 + (g/g_\mathrm{knee})^2)`, with the knee tied to the resonator
    decay rate by :math:`g_\mathrm{knee} = K \sqrt{\kappa}` — a lossier (larger-κ) resonator
    holds fewer photons per unit drive, so it punches out at a higher probe gain. :func:`fit_lorentzian`
    on the gain axis inverts the same closed form (the depth profile is a Lorentzian centered
    at zero gain), so a fit recovers the armed truth's κ.
    """
    p = params(ctx)
    g = np.abs(ctx.x())
    from strumento.packs.cqed.experiments.punchout import KNEE_GAIN_PER_SQRT_KAPPA

    knee = KNEE_GAIN_PER_SQRT_KAPPA * float(np.sqrt(max(float(p["resonator_kappa_mhz"]), 1e-9)))
    depth0 = float(p["punchout_depth"])
    signal = 1.0 - depth0 / (1.0 + (g / knee) ** 2)
    return _populations(np.clip(signal, 0.0, 1.0), p)


@responds(GainLadderSpectroscopy)
def gain_ladder_spectroscopy_response(ctx: SimContext) -> SimResponse:
    r"""The gain-ladder qubit line: one read per rung, the two-level frame across them.

    Per rung :math:`r` (relative drive :math:`s_r = g_r/g_1 - 1`) the line sits at

    .. math:: f_r = f_{ge} + \Delta\, s_r^{p_\delta}, \qquad
               \gamma_r = \gamma_0 (1+s_r)^{p_\gamma},

    the two-level model's :math:`p_\delta \approx 1`, :math:`p_\gamma \approx 0.5`, with the
    line depth growing as :math:`(1+s)^{p_\gamma}` (SNR with drive). An armed impostor
    signature overrides the model: ``"shift"`` grows the shift as :math:`s^2` (an off-model
    exponent); ``"saturate"`` collapses the line height after the lowest rung (premature
    saturation). Populations are shaped ``(n_points, n_rungs)`` — read *r* is rung *r*.
    """
    p = params(ctx)
    f = ctx.x()
    exp = ctx.experiment
    gains = np.asarray(getattr(exp, "gains"), dtype=float)
    s = gains / gains[0] - 1.0

    p_delta = float(p["gain_ladder_p_delta"])
    p_gamma = float(p["gain_ladder_p_gamma"])
    impostor = str(p.get("gain_ladder_impostor", "") or "")
    f_ge = float(p["f_ge"])
    gamma0 = float(p["spectroscopy_linewidth"])
    stark = float(p["gain_ladder_stark_mhz"])
    depth0 = float(p["gain_ladder_depth"])

    if impostor == "shift":
        shifts = 0.1 * s**2.0  # a two-photon-like line: the shift grows as s², off-model
    else:
        shifts = stark * s**p_delta
    depths = np.minimum(depth0 * (1.0 + s) ** p_gamma, 0.9)
    if impostor == "saturate":
        depths = np.where(s > 0, depth0 * 0.1, depth0)
    widths = gamma0 * (1.0 + s) ** p_gamma

    rows = [
        float(p["p_thermal"])
        + float(p["readout_contrast"])
        * (1.0 - float(depths[r]) * _lorentzian(f, f_ge + float(shifts[r]), float(widths[r])))
        for r in range(len(s))
    ]
    pops = np.stack(rows, axis=-1)  # (n_points, n_rungs)
    return _populations(np.clip(pops, 0.0, 1.0), p)


@responds(ReadoutState)
def readout_state_response(ctx: SimContext) -> SimResponse:
    r"""Readout state: per-shot conditional via :class:`ShotMachine` — the same two-read
    shape the single-shot experiment uses (read 0 = ground, read 1 = after ge_pi), so the
    per-shot projected histogram is the bimodal blob pair the vetted double-Gaussian fitter
    resolves into a threshold, a separation and an assignment fidelity."""
    p = params(ctx)
    eps = float(p.get("readout_error", 0.02))
    delta = float(p.get("pi_error", 0.005))

    def measure(state: int, rng) -> int:
        return int(state) ^ int(rng.random() < eps)

    def apply_pulse(state: int, pulse, rng) -> int:
        if getattr(pulse, "label", None) in ("pi_ge", None):
            return state if rng.random() < delta else 1 - int(state)
        return state

    machine = ShotMachine(
        thermal_population=float(p["p_thermal"]),
        measure=measure,
        apply_pulse=apply_pulse,
    )
    return SimResponse(
        machine=machine,
        blob0=tuple(p["blob0"]),
        blob1=tuple(p["blob1"]),
        sigma=float(p["blob_sigma"]),
    )


@responds(TlsScan)
def tls_scan_response(ctx: SimContext) -> SimResponse:
    r"""High-power TLS scan: the power-broadened main qubit line plus any armed narrow TLS
    defect lines (offsets from the truth's ``tls_line_offsets_mhz``) — the evidence the
    escalation-only node reports."""
    p = params(ctx)
    f = ctx.x()
    f_ge = float(p["f_ge"])
    gamma = float(p["spectroscopy_linewidth"]) * float(p["tls_power_broadening"])
    dip = 1.0 - _lorentzian(f, f_ge, gamma)
    for offset in p.get("tls_line_offsets_mhz", ()) or ():
        dip -= _lorentzian(f, f_ge + float(offset), float(p["tls_linewidth_mhz"]))
    pops = float(p["p_thermal"]) + float(p["readout_contrast"]) * np.clip(dip, 0.0, 1.0)
    return _populations(np.clip(pops, 0.0, 1.0), p)


@responds(ValidationCheck)
def validation_check_response(ctx: SimContext) -> SimResponse:
    r"""Terminal validation: three reads on one delay axis — ground, excited (after ge_pi),
    and the Ramsey fringe. Populations shaped ``(n_points, 3)``: read *r* is the r-th read."""
    p = params(ctx)
    t = ctx.x()
    p_th = float(p["p_thermal"])
    contrast = float(p["readout_contrast"])
    detuning = float(p["ramsey_detuning"])
    phase = np.deg2rad(float(getattr(ctx.experiment, "advance_phase_deg", 0.0) or 0.0))

    ground = np.full_like(t, p_th)
    excited = np.full_like(t, p_th + contrast * (1.0 - p_th))
    fringe = 0.5 + 0.5 * np.exp(-np.abs(t) / float(p["T2"])) * np.cos(
        2.0 * np.pi * detuning * t + phase
    )
    ramsey = p_th + contrast * fringe
    pops = np.stack([ground, excited, ramsey], axis=-1)
    return _populations(np.clip(pops, 0.0, 1.0), p)
