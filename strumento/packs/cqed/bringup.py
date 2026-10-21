"""The cqed bring-up graph — pack data (issue #56).

One transmon + readout, cold start to calibrated, authored against the core's typed plan
vocabulary (``BringupPlan`` / ``BringupStep`` / ``Provides`` / ``Requirement``). The core
never names this graph; this module is where it lives as data.

Chain (the graph, edge for edge)::

    resonator find-coarse (entry — force propose; wide span, minimum power)
      -> resonator find (targeted; refine)
      -> punchout (power sweep; banks readout.max_power [the RF envelope, bound-declared]
                   AND readout.power [the operating point]; the kappa estimate is GATE-ONLY)
      -> resonator refind (same writer, kind="refine": the dressed frequency at operating power)
      -> qubit spec (the gain ladder; frame check: fitted exponents p_delta ≈ 1, p_gamma ≈ 0.5
                     + premature-saturation detection + the 0.5× lowest-rung centroid cap)
      -> amplitude Rabi (prior: rabi_contrast > 0.7; chevron escalation on a failed fit)
      -> readout length (gated against the PACK kappa prior window [1/κ] — the punchout's
                         kappa wrote nothing, so this gate consumes the pack prior)
      -> readout state (threshold/angle; prior: assignment_fidelity >= the pack threshold —
                        the SAME threshold object the terminal validation references)
      -> T1 (banks qubit.T1)
      -> Ramsey (compound writeback in ONE call: qubit.freq_correction set + qubit.f_ge refine
                 at the final Rabi gain + qubit.T2 set; on_fail[1] -> TLS scan, [2] -> park)
      -> validation (terminal: Ramsey detuning < 50 kHz AND readout fidelity >= the pack
                     threshold AND qubit.T1 AND qubit.T2 recorded) == "calibrated"

Escalation ladders (rung counts, realized within the core's law that every
ladder's max_retries rung parks needs_human):

- **resonator miss (3 rungs)** — widen ×4 (self-retry: the factory re-parameterizes by
  attempt count), then the ``spurs-check`` node (an escalation-only wide scan), then the
  terminal park. The ~20 dB drop is the third attempt's re-parameterization, after the
  spurs detour returns control to the coarse find.
- **punchout no-show (2 rungs)** — re-enter the targeted find (re-identify the dip), then
  the terminal park.
- **no contrast (4 rungs)** — the SERIALIZED readout-chain revisit: frequency
  (``resonator-refind``), then power (``punchout``), then threshold (``readout-state``),
  each an existing node, terminating in the needs_human park. Each rung re-enters the chain
  at that node and the chain flows forward from there; the joint 2D revisit is the named
  waiver (``waived``) and no v1 edge routes to it.
- **Ramsey dead (2 rungs)** — the TLS scan node (escalation-only, banks nothing: gate-only
  evidence for the park decision), then the needs_human park.

The TLS scan and the spurs check are escalation-only: no ``on_pass`` edge routes to them.

**Gate verdicts.** The runner routes on the returned outcome's verdict map. A plain
``ExperimentOutcome`` carries ``{"fit": fit.success}``; a step declaring further gates needs
the outcome to grow its map — :class:`BringupNode` is that growth: it wraps a real
experiment (the inner one rides the full scaffold, so the vetted emitter writes the
``fit``/``fit-recovery`` events into the child record), delegates sequence/analyze, applies
the node's fit post-processing (Ramsey's compound derived), and returns a
:class:`NodeOutcome` whose ``gates`` are computed by this module's vetted gate emitters from
the fit and the **pack priors** below.

Pure at import: no ``qick``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from strumento.core.bringup import BringupPlan, BringupStep, Provides, Requirement
from strumento.core.experiment import Experiment, ExperimentOutcome
from strumento.core.fitters import FitResult
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi
from strumento.packs.cqed.experiments.gain_ladder_spectroscopy import GainLadderSpectroscopy
from strumento.packs.cqed.experiments.punchout import Punchout
from strumento.packs.cqed.experiments.readout_optimization import ReadoutOptimization
from strumento.packs.cqed.experiments.readout_state import ReadoutState
from strumento.packs.cqed.experiments.resonator_spectroscopy import ResonatorSpectroscopy
from strumento.packs.cqed.experiments.t1 import T1
from strumento.packs.cqed.experiments.t2_ramsey import T2Ramsey
from strumento.packs.cqed.experiments.tls_scan import TlsScan
from strumento.packs.cqed.experiments.validation_check import ValidationCheck
from strumento.packs.cqed.readout import DispersiveIQ

# ======================================================================================
# the hardware-mode physical priors — PACK DATA
# ======================================================================================

#: Resonator decay rate window (MHz) — consumed by the punchout's kappa-window gate.
KAPPA_MHZ: tuple[float, float] = (0.1, 10.0)

#: Transmon anharmonicity window (MHz) — the physical scale the gain-ladder frame check's
#: Stark-slope sanity consumes (a line shifting hundreds of MHz per unit drive is no qubit).
ANHARMONICITY_MHZ: tuple[float, float] = (100.0, 400.0)

#: Rabi-rate window (MHz) — the implied rate of the calibrated π pulse.
RABI_RATE_MHZ: tuple[float, float] = (0.1, 50.0)

#: The Rabi-contrast prior (the arch amplitude a genuine oscillation must exceed).
RABI_CONTRAST_MIN: float = 0.7

#: The assignment-fidelity pack threshold — the SAME object the readout-state node's gate
#: and the terminal validation's readout-fidelity gate consume.
ASSIGNMENT_FIDELITY_MIN: float = 0.80

#: The terminal's Ramsey-detuning window (kHz).
RAMSEY_DETUNING_MAX_KHZ: float = 50.0

#: The prior object as data — the gate emitters consume exactly this mapping.
PHYSICAL_PRIORS: dict[str, Any] = {
    "kappa_mhz": KAPPA_MHZ,
    "anharmonicity_mhz": ANHARMONICITY_MHZ,
    "rabi_rate_mhz": RABI_RATE_MHZ,
    "rabi_contrast_min": RABI_CONTRAST_MIN,
    "assignment_fidelity_min": ASSIGNMENT_FIDELITY_MIN,
    "ramsey_detuning_max_khz": RAMSEY_DETUNING_MAX_KHZ,
}

#: The gain-ladder frame check's fitted-exponent windows (the two-level model: ≈1 and ≈0.5).
FRAME_P_DELTA_WINDOW: tuple[float, float] = (0.5, 1.5)
FRAME_P_GAMMA_WINDOW: tuple[float, float] = (0.15, 0.9)


# ======================================================================================
# the node wrapper: the outcome's verdict map, grown per the runner's verdict vocabulary
# ======================================================================================


@dataclass
class NodeOutcome(ExperimentOutcome):
    """A bring-up node's outcome: the scaffold's result/fit plus the vetted verdict map the
    graph's gates and frame checks route on."""

    gates: dict[str, bool] = field(default_factory=dict)


class BringupNode(Experiment):
    """A bring-up step's sanctioned runnable: a real experiment plus the node's verdict map.

    The inner experiment rides the FULL scaffold path — ``go(soc, record=child)`` emits the
    vetted emitter's ``fit`` and (with a truth armed) ``fit-recovery`` gate events into the
    child record — and this wrapper delegates ``sequence``/``axes``/``analyze`` to it. The
    returned :class:`NodeOutcome` grows the verdict map: ``{"fit": fit.success}`` plus the
    node's gate emitter's verdicts (the pack priors, the frame check). ``fit_fn`` post-processes
    the fit before both uses (Ramsey's compound derived quantities).
    """

    def __init__(self, device, *, inner: Experiment, gates_fn=None, fit_fn=None) -> None:
        self.device = device
        self.inner = inner
        self.runner = inner.runner
        self._gates_fn = gates_fn
        self._fit_fn = fit_fn

    def sequence(self):
        return self.inner.sequence()

    def axes(self) -> tuple:
        return self.inner.axes()

    def recovery_map(self) -> dict[str, str]:
        return self.inner.recovery_map()

    def _post_fit(self, fit: FitResult | None) -> FitResult | None:
        if self._fit_fn is not None and fit is not None:
            return self._fit_fn(fit, self.device)
        return fit

    def analyze(self, result):
        return self._post_fit(self.inner.analyze(result))

    def go(self, soc, **kw):
        outcome = self.inner.go(soc, **kw)
        fit = self._post_fit(outcome.fit)
        gates = {"fit": bool(fit.success) if fit is not None else False}
        if self._gates_fn is not None:
            gates.update(self._gates_fn(NodeOutcome(result=outcome.result, fit=fit), self.device))
        return NodeOutcome(result=outcome.result, fit=fit, gates=gates)


# ======================================================================================
# the vetted gate emitters — the verdicts the graph routes on, from the fit + the priors
# ======================================================================================


def _fit_of(outcome: Any) -> FitResult | None:
    return getattr(outcome, "fit", None)


def punchout_gates(outcome, device) -> dict[str, bool]:
    """The punchout's gates: the fit plus the kappa-window check against the pack prior.
    The kappa estimate itself banks nothing (see the step's provides)."""
    fit = _fit_of(outcome)
    gates = {"fit": bool(fit.success) if fit is not None else False}
    kappa = fit.derived.get("resonator_kappa_mhz") if fit is not None else None
    gates["kappa-window"] = kappa is not None and KAPPA_MHZ[0] <= float(kappa) <= KAPPA_MHZ[1]
    return gates


def gain_ladder_frame_gates(outcome, device) -> dict[str, bool]:
    """The qubit-spec frame check: the fitted exponents in the two-level windows, no
    premature saturation, the lowest-rung centroid within the 0.5× cap, and the fitted
    Stark slope within the anharmonicity-scale physical window."""
    fit = _fit_of(outcome)
    gates = {"fit": bool(fit.success) if fit is not None else False}
    if fit is None:
        gates["gain-ladder-frame"] = False
        return gates
    d = fit.derived
    exponents_ok = (
        FRAME_P_DELTA_WINDOW[0] <= d["p_delta"] <= FRAME_P_DELTA_WINDOW[1]
        and FRAME_P_GAMMA_WINDOW[0] <= d["p_gamma"] <= FRAME_P_GAMMA_WINDOW[1]
    )
    slope_ok = d["stark_slope_mhz"] == d["stark_slope_mhz"] and (
        d["stark_slope_mhz"] <= ANHARMONICITY_MHZ[1]
    )
    gates["gain-ladder-frame"] = bool(
        exponents_ok and slope_ok and d["saturated"] == 0.0 and d["centroid_cap_ok"] == 1.0
    )
    return gates


def _implied_rabi_rate_mhz(device) -> float | None:
    """The calibrated π pulse's implied Rabi rate (MHz), the pack's convention: a Gaussian
    π pulse of σ µs has peak rate π/(σ·√(2π)); a flat-top of length L has rate π/L."""
    pc = device.calib.qubit.pulses.get("pi_ge")
    if pc is None:
        return None
    if pc.type == "flat_top" and pc.length > 0:
        return math.pi / pc.length
    if pc.sigma > 0:
        return math.pi / (math.sqrt(2.0 * math.pi) * pc.sigma)
    return None


def rabi_gates(outcome, device) -> dict[str, bool]:
    """The amplitude-Rabi node's gates: the fit, the contrast prior (rabi_contrast > 0.7,
    from the fitted arch amplitude) and the Rabi-rate window on the calibrated π pulse."""
    fit = _fit_of(outcome)
    gates = {"fit": bool(fit.success) if fit is not None else False}
    amp = abs(float(fit.params.get("amp", 0.0))) if fit is not None else 0.0
    gates["rabi-contrast"] = 2.0 * amp > RABI_CONTRAST_MIN
    rate = _implied_rabi_rate_mhz(device)
    gates["rabi-rate"] = rate is not None and RABI_RATE_MHZ[0] <= rate <= RABI_RATE_MHZ[1]
    return gates


def readout_length_gates(outcome, device) -> dict[str, bool]:
    """The readout-length node's window gate: the optimal integration must sit inside the
    PACK kappa prior's [1/κ] window — the punchout's kappa wrote nothing, so no banked path
    feeds this gate (the pack prior is the source)."""
    fit = _fit_of(outcome)
    gates = {"fit": bool(fit.success) if fit is not None else False}
    length = fit.derived.get("optimal_length") if fit is not None else None
    lo, hi = 1.0 / KAPPA_MHZ[1], 1.0 / KAPPA_MHZ[0]
    gates["kappa-window"] = length is not None and lo <= float(length) <= hi
    return gates


def readout_state_gates(outcome, device) -> dict[str, bool]:
    """The readout-state node's gate: the assignment fidelity against the pack threshold —
    the SAME threshold object the terminal validation references."""
    fit = _fit_of(outcome)
    gates = {"fit": bool(fit.success) if fit is not None else False}
    fidelity = fit.derived.get("assignment_fidelity") if fit is not None else None
    gates["assignment-fidelity"] = fidelity is not None and float(fidelity) >= (
        ASSIGNMENT_FIDELITY_MIN
    )
    return gates


def validation_gates(outcome, device) -> dict[str, bool]:
    """The terminal validation's measured gates: |Ramsey detuning| < 50 kHz and the readout
    fidelity ≥ the pack threshold. (The T1/T2-recorded half is the step's requires.)"""
    fit = _fit_of(outcome)
    gates = {"fit": bool(fit.success) if fit is not None else False}
    detuning = fit.derived.get("detuning") if fit is not None else None
    gates["detuning-window"] = detuning is not None and abs(float(detuning)) <= (
        RAMSEY_DETUNING_MAX_KHZ / 1000.0
    )
    fidelity = fit.derived.get("assignment_fidelity") if fit is not None else None
    gates["readout-fidelity"] = fidelity is not None and float(fidelity) >= ASSIGNMENT_FIDELITY_MIN
    return gates


# ======================================================================================
# the node factories — attempt-count-driven escalation parameterization
# ======================================================================================


def _dispersive_readout(device) -> DispersiveIQ:
    """The bring-up nodes' discriminator: the device's recorded readout, centres included (#196)."""
    return DispersiveIQ.from_calib(device.calib.readout)


def _find_coarse_factory(device, ctx):
    """Wide span at minimum power; retry 1 widens ×4, retry 2 (after the spurs detour)
    drops the probe ~20 dB — the resonator-miss ladder's re-parameterizations."""
    ro = device.calib.readout
    failures = ctx.state.attempts.get("resonator-find-coarse", 0)
    span = 40.0 * (4.0 ** min(failures, 1))            # widen ×4 on the first retry
    gain = 0.02 * (0.1 ** min(max(failures - 1, 0), 1))  # ~20 dB down on the third attempt
    return ResonatorSpectroscopy(
        device,
        freqs=Sweep(start=ro.frequency - span, stop=ro.frequency + span, on="f"),
        gain=gain,
        points=161,
    )


def _find_factory(device, ctx):
    """Targeted find using the coarse result; retry 1 widens ×4 around the prior."""
    ro = device.calib.readout
    failures = ctx.state.attempts.get("resonator-find", 0)
    span = 5.0 * (4.0 ** min(failures, 1))
    return ResonatorSpectroscopy(
        device,
        freqs=Sweep(start=ro.frequency - span, stop=ro.frequency + span, on="f"),
        points=161,
    )


def _punchout_factory(device, ctx):
    inner = Punchout(device, readout=_dispersive_readout(device))
    return BringupNode(device, inner=inner, gates_fn=punchout_gates)


def _refind_factory(device, ctx):
    """Same writer as the targeted find, kind="refine": the dressed frequency at the
    operating power."""
    ro = device.calib.readout
    failures = ctx.state.attempts.get("resonator-refind", 0)
    span = 3.0 * (2.0 ** min(failures, 1))
    return ResonatorSpectroscopy(
        device,
        freqs=Sweep(start=ro.frequency - span, stop=ro.frequency + span, on="f"),
        points=161,
    )


def _spurs_factory(device, ctx):
    """The spurs check: a wide, higher-dynamic-range scan. Escalation-only; banks nothing."""
    ro = device.calib.readout
    return ResonatorSpectroscopy(
        device,
        freqs=Sweep(start=ro.frequency - 100.0, stop=ro.frequency + 100.0, on="f"),
        gain=0.2,
        points=161,
    )


def _qubit_spec_factory(device, ctx):
    inner = GainLadderSpectroscopy(device, points=81, readout=_dispersive_readout(device))
    return BringupNode(device, inner=inner, gates_fn=gain_ladder_frame_gates)


def _rabi_factory(device, ctx):
    """Amplitude Rabi; the chevron escalation widens the sweep on a failed fit."""
    failures = ctx.state.attempts.get("amplitude-rabi", 0)
    # 12000 (0.366 frac) baseline, 16000 (0.488) escalated -- both under
    # MAX_QUBIT_DRIVE_GAIN=0.5. This ladder previously escalated to
    # 20000/30000 (0.61/0.92 frac), over the ceiling even on the FIRST, non-escalated
    # attempt -- inconsistent with safety.py's own docstring, which says 0.5 already
    # *includes* headroom for calibration sweeps beyond normal gate operation (~0.3), not
    # that calibration sweeps get to go past it. If a real device's pi-gain genuinely sits
    # beyond 16000, that's a signal to raise the qubit's max_drive_gain calibration override
    # deliberately, not something this escalation ladder should reach for silently.
    stop = 16000 if failures else 12000
    inner = AmplitudeRabi(
        device,
        gains=Sweep(start=0, stop=stop, on="amp"),
        points=51,
        readout=_dispersive_readout(device),
    )
    return BringupNode(device, inner=inner, gates_fn=rabi_gates)


def _readout_length_factory(device, ctx):
    inner = ReadoutOptimization(device, readout=_dispersive_readout(device))
    return BringupNode(device, inner=inner, gates_fn=readout_length_gates)


def _readout_state_factory(device, ctx):
    inner = ReadoutState(device, readout=_dispersive_readout(device))
    return BringupNode(device, inner=inner, gates_fn=readout_state_gates)


def _t1_factory(device, ctx):
    return T1(device, readout=_dispersive_readout(device))


def _ramsey_compound_fit(fit: FitResult, device) -> FitResult:
    """Ramsey's compound derived: the detuning correction and the refine value — f_spec plus
    Δ_Ramsey, computed at the device's current (final-Rabi-gain) belief."""
    detuning = float(fit.derived["detuning"])
    f_ge = float(device.calib.qubit.f_ge)
    derived = dict(fit.derived)
    derived["freq_correction"] = detuning
    derived["freq_refine"] = f_ge + detuning
    return FitResult(
        params=dict(fit.params), stderr=dict(fit.stderr), derived=derived, success=fit.success
    )


def _ramsey_factory(device, ctx):
    inner = T2Ramsey(device, readout=_dispersive_readout(device))
    return BringupNode(device, inner=inner, fit_fn=_ramsey_compound_fit)


def _tls_factory(device, ctx):
    return TlsScan(device, readout=_dispersive_readout(device))


def _validation_factory(device, ctx):
    inner = ValidationCheck(device, readout=_dispersive_readout(device))
    return BringupNode(device, inner=inner, gates_fn=validation_gates)


# ======================================================================================
# the graph
# ======================================================================================


def cqed_bringup_plan() -> BringupPlan:
    """The cqed bring-up graph, exactly as the pack ships it (see the module docstring)."""
    return BringupPlan(
        steps=(
            BringupStep(
                name="resonator-find-coarse",
                experiment=_find_coarse_factory,
                requires=(),
                provides=(Provides(path="readout", mapping={"f0": "frequency"}),),
                gates=("fit",),
                on_pass="resonator-find",
                on_fail={
                    1: "resonator-find-coarse",
                    2: "spurs-check",
                    3: "park:needs_human",
                },
                max_retries=3,
                envelope={"readout": 4000.0},
                poisons=("readout.frequency",),
                force_propose=True,
            ),
            BringupStep(
                name="resonator-find",
                experiment=_find_factory,
                requires=(
                    Requirement(path="readout.frequency", max_age_s=3600.0, drift_sensitive=True),
                ),
                provides=(Provides(path="readout", mapping={"f0": "frequency"}, kind="refine"),),
                gates=("fit",),
                on_pass="punchout",
                on_fail={1: "resonator-find-coarse", 2: "park:needs_human"},
                max_retries=2,
                envelope={"readout": 8000.0},
                poisons=("readout.frequency",),
            ),
            BringupStep(
                name="punchout",
                experiment=_punchout_factory,
                # NOTE: deliberately no requirement on readout.frequency. The refind
                # re-writes the dressed frequency AFTER the punchout (the at-operating-power
                # refine), and the punchout's operating point transitively
                # depending on the frequency would make that refine value-poison the
                # punchout's power on every pass — the freshness engine's refresh detour
                # would never terminate. The find chain owns the frequency's freshness; the
                # punchout's own kappa-window gate covers its sanity.
                requires=(),
                provides=(
                    Provides(path="readout", mapping={"max_power": "max_power"}, max_abs=32766.0),
                    Provides(path="readout", mapping={"power": "power"}),
                ),
                gates=("fit", "kappa-window"),
                on_pass="resonator-refind",
                on_fail={1: "resonator-find", 2: "park:needs_human"},
                max_retries=2,
                envelope={"readout": 32766.0},
                poisons=("readout.max_power", "readout.power"),
            ),
            BringupStep(
                name="resonator-refind",
                experiment=_refind_factory,
                requires=(
                    Requirement(path="readout.power", max_age_s=3600.0, drift_sensitive=True),
                ),
                provides=(Provides(path="readout", mapping={"f0": "frequency"}, kind="refine"),),
                gates=("fit",),
                on_pass="qubit-spec",
                on_fail={1: "resonator-refind", 2: "park:needs_human"},
                max_retries=2,
                envelope={"readout": 8000.0},
                poisons=("readout.frequency",),
            ),
            BringupStep(
                name="qubit-spec",
                experiment=_qubit_spec_factory,
                requires=(
                    Requirement(path="readout.frequency", max_age_s=3600.0, drift_sensitive=True),
                ),
                provides=(Provides(path="qubit", mapping={"f_spec": "f_ge"}),),
                gates=("fit",),
                frame_check="gain-ladder-frame",
                on_pass="amplitude-rabi",
                on_fail={1: "qubit-spec", 2: "park:needs_human"},
                max_retries=2,
                envelope={"readout": 8000.0, "qubit": 20000.0},
                poisons=("qubit.f_ge",),
            ),
            BringupStep(
                name="amplitude-rabi",
                experiment=_rabi_factory,
                # NOTE: deliberately no requirement on qubit.f_ge. Ramsey refines qubit.f_ge
                # LATER in the chain (the final-gain refine), and this step's
                # π-gain write transitively depending on f_ge would make that refine
                # value-poison the π gain on every pass — the t1 step (which requires the
                # π gain) would detour to this step forever. The drive rides the device's
                # current f_ge belief; the qubit-spec writeback lands in the store for the
                # supervisor's device refresh.
                requires=(),
                provides=(Provides(path="qubit.pulses.pi_ge", mapping={"pi_gain": "gain"}),),
                gates=("fit", "rabi-contrast", "rabi-rate"),
                on_pass="readout-length",
                on_fail={
                    1: "resonator-refind",
                    2: "punchout",
                    3: "readout-state",
                    4: "park:needs_human",
                },
                max_retries=4,
                # Matches _rabi_factory's escalated ceiling (16000, under
                # MAX_QUBIT_DRIVE_GAIN=0.5) -- was 30000, stale once the actual sweep was
                # lowered; an envelope wider than what a step can ever really apply
                # is over-permissive bookkeeping, not itself unsafe, but worth tightening to
                # match rather than leaving a declared authorization nothing uses.
                envelope={"qubit": 16000.0},
                poisons=("qubit.pulses.pi_ge.gain",),
            ),
            BringupStep(
                name="readout-length",
                experiment=_readout_length_factory,
                requires=(
                    Requirement(path="readout.power", max_age_s=3600.0, drift_sensitive=True),
                ),
                provides=(
                    Provides(path="readout", mapping={"optimal_length": "readout_length"}),
                ),
                gates=("fit", "kappa-window"),
                on_pass="readout-state",
                on_fail={1: "readout-length", 2: "park:needs_human"},
                max_retries=2,
                envelope={"readout": 32766.0},
                poisons=("readout.readout_length",),
            ),
            BringupStep(
                name="readout-state",
                experiment=_readout_state_factory,
                requires=(
                    Requirement(path="readout.frequency", max_age_s=3600.0, drift_sensitive=True),
                    Requirement(path="readout.power", max_age_s=3600.0, drift_sensitive=True),
                ),
                provides=(
                    Provides(path="readout", mapping={"threshold": "threshold"}),
                    Provides(path="readout", mapping={"angle": "angle"}),
                ),
                gates=("fit", "assignment-fidelity"),
                on_pass="t1",
                on_fail={1: "readout-state", 2: "park:needs_human"},
                max_retries=2,
                envelope={"readout": 8000.0, "qubit": 30000.0},
                poisons=("readout.threshold", "readout.angle"),
            ),
            BringupStep(
                name="t1",
                experiment=_t1_factory,
                requires=(Requirement(path="qubit.pulses.pi_ge.gain"),),
                provides=(Provides(path="qubit", mapping={"T1": "T1"}),),
                gates=("fit",),
                on_pass="ramsey",
                on_fail={1: "t1", 2: "park:needs_human"},
                max_retries=2,
                envelope={"qubit": 30000.0, "readout": 8000.0},
                poisons=("qubit.T1",),
            ),
            BringupStep(
                name="ramsey",
                experiment=_ramsey_factory,
                requires=(
                    Requirement(path="qubit.pulses.pi_ge.gain"),
                    Requirement(path="readout.threshold"),
                ),
                provides=(
                    Provides(path="qubit", mapping={"freq_correction": "freq_correction"}),
                    Provides(path="qubit", mapping={"freq_refine": "f_ge"}, kind="refine"),
                    Provides(path="qubit", mapping={"T2": "T2"}),
                ),
                gates=("fit",),
                on_pass="validation",
                on_fail={1: "tls-scan", 2: "park:needs_human"},
                max_retries=2,
                envelope={"qubit": 30000.0, "readout": 8000.0},
                poisons=("qubit.freq_correction", "qubit.f_ge", "qubit.T2"),
            ),
            BringupStep(
                name="tls-scan",
                experiment=_tls_factory,
                requires=(),
                provides=(),  # banks nothing — gate-only evidence for the park decision
                gates=("fit",),
                on_pass="ramsey",
                on_fail={1: "park:needs_human"},
                max_retries=1,
                envelope={"qubit": 32766.0},
            ),
            BringupStep(
                name="spurs-check",
                experiment=_spurs_factory,
                requires=(),
                provides=(),
                gates=("fit",),
                on_pass="resonator-find-coarse",
                on_fail={1: "park:needs_human"},
                max_retries=1,
                envelope={"readout": 32766.0},
            ),
            BringupStep(
                name="validation",
                experiment=_validation_factory,
                requires=(
                    Requirement(path="qubit.T1"),
                    Requirement(path="qubit.T2"),
                    Requirement(path="readout.threshold"),
                ),
                provides=(),
                gates=("fit", "detuning-window", "readout-fidelity"),
                on_pass="calibrated",
                on_fail={1: "validation", 2: "park:needs_human"},
                max_retries=2,
                envelope={"qubit": 30000.0, "readout": 8000.0},
            ),
        ),
        entry="resonator-find-coarse",
        waived=frozenset({"joint-2d-revisit"}),
    )
