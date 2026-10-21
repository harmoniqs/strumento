"""Response-model registry + simulation **mechanism** (D32) — the pure half of the simulator.

A *response model* answers one question: given what is being run (`SimContext`), what does the
readout return? That answer is **physics**, so it lives in packs; this module owns only the
seam and the machinery:

* the **registry** — ``register_response(experiment_cls, fn)`` / ``resolve_response(cls)``,
  type-keyed with an MRO walk, exactly mirroring the envelope-lowering registry's pattern in
  :mod:`strumento.core.compiler` (which is likewise a no-qick module). Packs register at pack
  import, so pack import stays pure and the CI ``pure`` job is unaffected.
* :class:`SimContext` — the frozen arming payload: the experiment, the device, the swept axis
  in lab-native units, the simulator's ground-truth calibration dict, and an optional explicit
  model. :class:`SimResponse` is the matching return type when a model wants to say *how* its
  signal reaches the readout (blob centers, per-shot width) instead of just what it is.
* the **mechanism** helpers — shot sampling from populations, IQ blob placement with Gaussian
  scatter, shot-axis averaging, and the per-shot conditional state machine
  (:class:`ShotMachine`) that walks a sequence's ``Branch``/``Repeat`` structure so
  measurement-conditional dynamics (active reset, feed-forward) are *demonstrable* offline.
  :meth:`ShotMachine.outcomes` is its emission form — ``(n_reads, shots)`` outcomes that
  :func:`place_shots` turns into IQ indistinguishable from any other shot data, with the
  herald/verification correlation *within* a shot preserved.

Every stochastic helper takes an **injected** :class:`numpy.random.Generator`: no module-level
random state, so a test that pins a seed pins the data. Nothing here knows any physics — a
response model that wants T1 decay or a conditional π supplies it through the machine's hooks.

Resolution order (the simulator's arming path, D32): an explicit model on the context, else the
experiment's own ``response_model(device, xpts)`` hook, else the registry by type, else a typed
:class:`ResponseError` naming the class. **Never** silent zeros — the old zero-data behavior
stays available only in ``MockQickSocV2``.

This module is **pure**: it never imports ``qick``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strumento.core.pulses import Branch, Measure, Pulse, Repeat, Segment, Seq

#: A response model: ``fn(ctx) -> readout response`` over ``ctx.xpts``. What the response *is*
#: (excited-state populations, a complex transmission trace, per-shot outcomes) is the calling
#: experiment's convention — core carries it, it does not interpret it.
ResponseModel = Callable[["SimContext"], Any]


class ResponseError(RuntimeError):
    """No usable response model, or a sequence the per-shot machine cannot interpret.

    Typed so the experiment path can surface "this experiment has no simulated response" as a
    clear failure instead of returning zeros that look like data.
    """


# --------------------------------------------------------------------------------------
# The arming payload
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SimContext:
    """What the simulator needs to know *before* acquire (the arming seam, D32).

    Attributes:
        experiment: the experiment instance being run (the registry key is its type).
        device: the loaded :class:`~strumento.core.device.Device` (calibrated belief).
        xpts: the swept axis in **lab-native** units — the same values the experiment pins on
            its result. Coerced to a tuple so the context stays immutable.
        model: an explicit response model, overriding hook and registry lookup.
        truth: the simulator's **ground truth** — an ordinary calibration dict (config-as-data,
            hard rule 4 / D10), held by the soc and merged in at arming time. It is deliberately
            *separate* from ``device`` (the calibrated belief): a response model reads truth
            values with the belief as fallback, so a perturbed truth is exactly what makes a
            closed-loop demo non-trivial. Core never interprets the keys — that is pack physics.
        seq: the sequence being run. Only a *conditional* response needs it — a measurement-driven
            response is a per-shot walk of the ``Branch``/``Repeat`` structure
            (:class:`ShotMachine`), and the structure lives nowhere else. Closed-form responses
            over ``xpts`` ignore it.
        readout_output_scale: conversion from response-model IQ units to the averaged IQ returned
            by acquisition. Hardware runners leave this at ``1``; the simulator derives it from
            its ADC scale and the realized integration length so a response can align model-space
            blobs with device-recorded centers when needed.
    """

    experiment: Any
    device: Any = None
    xpts: tuple = ()
    model: ResponseModel | None = None
    truth: dict = field(default_factory=dict)
    seq: Any = None
    readout_output_scale: float = 1.0

    def __post_init__(self) -> None:
        xpts = self.xpts
        if xpts is None:
            object.__setattr__(self, "xpts", ())
        elif not isinstance(xpts, tuple):
            object.__setattr__(self, "xpts", tuple(np.asarray(xpts).ravel().tolist()))
        output_scale = float(self.readout_output_scale)
        if not np.isfinite(output_scale) or output_scale <= 0.0:
            raise ValueError(
                "readout_output_scale must be finite and positive; "
                f"got {self.readout_output_scale}"
            )
        object.__setattr__(self, "readout_output_scale", output_scale)

    def x(self) -> np.ndarray:
        """``xpts`` as a float array — what a response model actually evaluates over."""
        return np.asarray(self.xpts, dtype=float)


@dataclass(frozen=True)
class SimResponse:
    """A response model's **full** return: the readout signal plus the readout-model knobs (D32).

    A model may return a bare array — real values are excited-state populations, complex values a
    transmission trace — and the simulator applies its unit-scale defaults. Returning this instead
    is how a pack supplies the *readout* half of the picture, which is physics and therefore never
    a core constant: where the two discrimination blobs sit and how wide the per-shot scatter is.

    Exactly one of ``populations`` / ``iq`` / ``machine`` is set:

    * ``populations`` — excited-state probability per sweep point, in ``[0, 1]``; each shot is a
      draw placed at ``blob0`` or ``blob1`` (see :func:`populations_to_iq`).
    * ``iq`` — the **complex** mean readout response per sweep point (e.g. resonator transmission);
      each shot is that point scattered by ``sigma``.
    * ``machine`` — a :class:`ShotMachine`: the response is *measurement-conditional* (active
      reset, feed-forward), so there is no population curve to draw from. The simulator walks the
      armed sequence per shot and places the resulting per-read outcomes at the blobs, which is
      what makes the herald read and the verification read of the same shot **correlated** — the
      entire content of a conditional experiment.

    The array forms may be shaped ``(n_points,)`` (broadcast over the reads in a shot) or
    ``(n_points, n_reads)`` when the reads differ — a herald read followed by a data read.
    """

    populations: Any | None = None
    iq: Any | None = None
    machine: ShotMachine | None = None
    blob0: tuple[float, float] = (0.0, 0.0)
    blob1: tuple[float, float] = (1.0, 0.0)
    sigma: float = 0.0

    def __post_init__(self) -> None:
        given = [f for f in ("populations", "iq", "machine") if getattr(self, f) is not None]
        if len(given) != 1:
            raise ResponseError(
                "a SimResponse carries exactly one of populations= (real, in [0, 1]), "
                "iq= (complex mean readout response) or machine= (a per-shot ShotMachine); "
                f"got {given or 'none of them'}"
            )


# --------------------------------------------------------------------------------------
# Registry (the envelope-registry pattern, D22 → D32)
# --------------------------------------------------------------------------------------

_RESPONSES: dict[type, ResponseModel] = {}


def register_response(experiment_type: type, model: ResponseModel) -> None:
    """Register ``model`` as the simulated response for ``experiment_type`` (and its subclasses).

    Packs call this at import. Registering a type twice replaces the entry — the same
    last-registration-wins rule the envelope registry uses, so an instance-specific override can
    shadow a pack default.
    """
    _RESPONSES[experiment_type] = model


def responds(experiment_type: type):
    """Decorator form of :func:`register_response` (mirrors ``@lowers`` in the compiler)."""

    def deco(fn: ResponseModel) -> ResponseModel:
        register_response(experiment_type, fn)
        return fn

    return deco


def resolve_response(experiment: type | Any) -> ResponseModel:
    """The registered response model for an experiment type (or instance), walking the MRO.

    The nearest registration wins, so a pack can register a family default on a base experiment
    and a specific subclass can refine it. Raises :class:`ResponseError` naming the class when
    nothing is registered — never a default, never zeros.
    """
    cls = experiment if isinstance(experiment, type) else type(experiment)
    for base in cls.__mro__:
        model = _RESPONSES.get(base)
        if model is not None:
            return model
    raise ResponseError(
        f"no response model registered for {cls.__name__!r}: a pack must register_response() it "
        f"(or the experiment must define response_model(device, xpts)) before it can be simulated"
    )


def response_for(ctx: SimContext) -> ResponseModel:
    """Resolve the model for an armed context: explicit → experiment hook → registry (D32)."""
    if ctx.model is not None:
        return ctx.model
    hook = getattr(ctx.experiment, "response_model", None)
    if callable(hook):
        model = hook(ctx.device, ctx.xpts)
        if model is not None:
            return model
    return resolve_response(ctx.experiment)


def registered_responses() -> tuple[type, ...]:
    """The experiment types with a registered response (introspection for error messages/CLI)."""
    return tuple(_RESPONSES)


# --------------------------------------------------------------------------------------
# Mechanism: shots and IQ
# --------------------------------------------------------------------------------------


def sample_shots(populations, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Draw per-shot outcomes (0/1) from excited-state ``populations``.

    ``populations`` may be a scalar or any array shape; the returned array appends the shot axis
    **last** (``populations.shape + (shots,)``) — the ``MeasurementResult.shots`` convention
    (leading axes index sweep points, the last axis is shots).
    """
    pops = np.asarray(populations, dtype=float)
    if pops.size and (np.nanmin(pops) < 0.0 or np.nanmax(pops) > 1.0):
        raise ValueError(f"population must lie in [0, 1]; got range [{pops.min()}, {pops.max()}]")
    draws = rng.random(pops.shape + (int(shots),))
    return (draws < pops[..., None]).astype(np.int8)


def populations_to_iq(
    populations,
    blob0,
    blob1,
    sigma: float,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place per-shot IQ points around the two readout blobs (the mechanism a readout kind inverts).

    Each shot is drawn from ``populations`` (see :func:`sample_shots`), placed at ``blob0``
    (outcome 0) or ``blob1`` (outcome 1), then scattered by isotropic Gaussian noise of width
    ``sigma``. Blob centers/width come from the *readout* model — a pack concern — so they are
    arguments, not constants.

    Returns an array of shape ``populations.shape + (shots, 2)``, last axis ``(I, Q)``: exactly
    what ``DispersiveIQ.discriminate``/``project`` consume, so post-selection and thresholding
    run on simulated data the same way they run on the board's.
    """
    return place_shots(sample_shots(populations, shots, rng), blob0, blob1, sigma, rng)


def place_shots(outcomes, blob0, blob1, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Place **already-decided** per-shot outcomes at the two readout blobs, scattered by ``sigma``.

    The second half of :func:`populations_to_iq`, separated because a conditional response
    (:class:`ShotMachine`) decides its outcomes by *walking the sequence* rather than by drawing
    from a population — but the IQ it emits must be indistinguishable from any other shot data.
    Returns ``outcomes.shape + (2,)``, last axis ``(I, Q)``.
    """
    draws = np.asarray(outcomes)
    b0 = np.asarray(blob0, dtype=float)
    b1 = np.asarray(blob1, dtype=float)
    if b0.shape != (2,) or b1.shape != (2,):
        raise ValueError("blob0/blob1 must each be an (I, Q) pair")
    centers = b0 + draws[..., None] * (b1 - b0)
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative; got {sigma}")
    if sigma == 0.0:
        return centers
    return centers + rng.normal(0.0, float(sigma), size=centers.shape)


def average_iq(iq) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-shot IQ ``(..., shots, 2)`` to ``(avgi, avgq)`` — the averaged-acquire form."""
    arr = np.asarray(iq, dtype=float)
    if arr.ndim < 2 or arr.shape[-1] != 2:
        raise ValueError("expected per-shot IQ with a trailing (I, Q) axis")
    return arr[..., 0].mean(axis=-1), arr[..., 1].mean(axis=-1)


# --------------------------------------------------------------------------------------
# Mechanism: the per-shot conditional state machine
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotRecord:
    """One shot's outcome: the final state and every measurement it produced, in order."""

    state: int
    measurements: tuple[int, ...] = ()


@dataclass
class _ShotState:
    state: int
    measurements: list[int] = field(default_factory=list)
    last: int | None = None


def _keep_state(state: int, op: Any, rng: np.random.Generator) -> int:
    """Default pulse action: none. Physics (what a π does) is a pack's response model."""
    return state


def _perfect_readout(state: int, rng: np.random.Generator) -> int:
    """Default readout: reads the state faithfully. Misassignment is a pack-supplied hook."""
    return state


def _outcome_as_value(outcome: int, branch: Branch) -> float:
    """Default branch value: the discriminated outcome itself (0/1).

    On hardware a ``Branch`` compares an ADC accumulator value against ``branch.threshold`` on
    ``branch.component``. Offline there is no accumulator, so the default convention treats the
    discriminated outcome as the value (thresholds between 0 and 1 select on it). A caller that
    knows its blob centers injects ``branch_value`` to compare in real readout units.
    """
    return float(outcome)


@dataclass(frozen=True)
class ShotMachine:
    """Walk a sequence's conditional structure **per shot** (mechanism only, D32).

    The machine understands *control flow* — ``Measure`` records an outcome, ``Branch`` takes an
    arm based on the last outcome, ``Repeat`` re-runs its body — and delegates every physical
    question to injected hooks:

    * ``apply_pulse(state, pulse, rng) -> state`` — what a pulse does to the state.
    * ``measure(state, rng) -> outcome`` — the readout (add misassignment here).
    * ``branch_value(outcome, branch) -> float`` — the value the branch test compares.

    ``thermal_population`` seeds each shot: the probability the shot starts in the excited state,
    which is what makes active reset a *measurable* improvement rather than a no-op.

    ``Delay`` and ``Raw`` are inert here: idle decay is physics, so a response model that wants
    it folds it into the populations it returns (or into ``apply_pulse``).
    """

    thermal_population: float = 0.0
    apply_pulse: Callable[[int, Any, np.random.Generator], int] = _keep_state
    measure: Callable[[int, np.random.Generator], int] = _perfect_readout
    branch_value: Callable[[int, Branch], float] = _outcome_as_value

    def __post_init__(self) -> None:
        p = float(self.thermal_population)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"thermal_population must lie in [0, 1]; got {p}")

    def initial_state(self, rng: np.random.Generator) -> int:
        """Draw a shot's starting state from the thermal population."""
        return int(rng.random() < float(self.thermal_population))

    def run(self, seq: Seq, rng: np.random.Generator, *, state: int | None = None) -> ShotRecord:
        """Run one shot through ``seq``; ``state`` overrides the thermal draw."""
        st = _ShotState(state=self.initial_state(rng) if state is None else int(state))
        self._walk(seq, st, rng)
        return ShotRecord(state=st.state, measurements=tuple(st.measurements))

    def run_shots(
        self, seq: Seq, shots: int, rng: np.random.Generator, *, state: int | None = None
    ) -> tuple[ShotRecord, ...]:
        """Run ``shots`` independent shots through ``seq``."""
        return tuple(self.run(seq, rng, state=state) for _ in range(int(shots)))

    def outcomes(
        self, seq: Seq, shots: int, rng: np.random.Generator, *, state: int | None = None
    ) -> np.ndarray:
        """Per-read outcomes for ``shots`` shots, shaped ``(n_reads, shots)`` — the emission form.

        The read axis leads and the shot axis is last, matching the acquisition's
        ``reads_per_shot`` layout (and the pack convention that the **last** read is the data read,
        earlier reads being heralds). Rows are the *same* shot across reads, which is the
        correlation a conditional experiment measures.

        A sequence whose shots produce differing numbers of measurements (a ``Measure`` inside a
        branch arm) has no rectangular emission form; that is a typed :class:`ResponseError` rather
        than a padded array.
        """
        records = self.run_shots(seq, shots, rng, state=state)
        if not records:
            return np.zeros((0, 0), dtype=np.int8)
        counts = {len(rec.measurements) for rec in records}
        if len(counts) > 1:
            raise ResponseError(
                f"shots produced different numbers of measurements ({sorted(counts)}): a "
                f"conditional sequence can only be emitted when every shot reads the same number "
                f"of times (keep Measure ops out of Branch arms)"
            )
        return np.array([rec.measurements for rec in records], dtype=np.int8).T

    # -- internals ---------------------------------------------------------------------

    def _walk(self, ops: Seq | Iterable[Any], st: _ShotState, rng: np.random.Generator) -> None:
        for op in ops:
            if isinstance(op, Measure):
                outcome = int(self.measure(st.state, rng))
                st.measurements.append(outcome)
                st.last = outcome
            elif isinstance(op, Branch):
                arm = self._arm(op, st)
                if arm is not None:
                    self._walk(arm, st, rng)
            elif isinstance(op, Repeat):
                for _ in range(int(op.count)):
                    self._walk(op.body, st, rng)
            elif isinstance(op, Segment):
                for pulse in op.pulses:
                    st.state = int(self.apply_pulse(st.state, pulse, rng))
            elif isinstance(op, Pulse):
                st.state = int(self.apply_pulse(st.state, op, rng))
            # Delay / Raw: no per-shot state action in the mechanism layer (see the class doc).

    def _arm(self, branch: Branch, st: _ShotState) -> Seq | None:
        if st.last is None:
            raise ResponseError(
                "Branch has no preceding Measure in this shot: the per-shot machine tests the "
                "last measurement outcome, so a conditional must follow a measurement"
            )
        value = float(self.branch_value(st.last, branch))
        if branch.test == ">=":
            passed = value >= float(branch.threshold)
        elif branch.test == "<":
            passed = value < float(branch.threshold)
        else:
            raise ResponseError(
                f"Branch test {branch.test!r} is not simulable; tProc v2 provides '>=' and '<'"
            )
        return branch.then if passed else branch.else_
