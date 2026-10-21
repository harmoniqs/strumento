"""Experiment data-lifecycle scaffold (D13).

Register-level experiment frameworks often conflate acquire / fit / plot in one `go`. Here the high-level contract
stays (`go`) but the internals are **separated collaborators**:

* **Runner** — builds a ``StrumentoProgram`` and acquires → :class:`MeasurementResult`. The only
  qick-facing step; imported lazily so this module stays pure at import and the Runner is
  injectable (a fake Runner drives the orchestration in tests without a board).
* **Analyzer** — the subclass's :meth:`Experiment.analyze` (fitters → :class:`FitResult`); **no
  plotting or printing**.
* **Display** — the subclass's :meth:`Experiment.display` (a figure from result + fit); default
  no-op so ``core`` needs no plotting dependency.

A concrete experiment subclasses :class:`Experiment`, implementing :meth:`sequence` (what to run)
and :meth:`analyze` (how to fit). ``go`` orchestrates and returns a typed
:class:`ExperimentOutcome` (raw result + fit + optional figure) — never stdout.

**Task records and the gate emitter.** Both :meth:`Runner.run` and :meth:`Experiment.go` take an
explicit ``record=`` keyword (a :class:`~strumento.core.record.TaskRecord`). With one supplied, a
standard experiment gets task integration for free — ``progress`` for the acquisition and for the
fit, ``artifact`` for a saved figure — and, more importantly, this is where **gate verdicts are
computed**: see :meth:`Experiment.go` for the trust boundary and its honest limits.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, SupportsFloat, SupportsIndex, TypeGuard

from strumento.core.environment import stamp_environment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.soc import ArmableSoc, ShotSource

#: Default ``fit-recovery`` tolerance: the relative error a fitted quantity may have against the
#: simulator's ground truth. 5 % is the *loosest* of the shipped offline-recovery gates (the
#: envelope-parameter fits — T1, T2, a resonator Q; see ``tests/e2e/test_fit_recovery.py``, whose
#: module docstring derives each axis's error budget). A caller with a better-conditioned scan
#: should tighten it per experiment rather than rely on this number.
DEFAULT_RECOVERY_TOLERANCE = 0.05

#: The reason an experiment that needs per-shot data gives for refusing a result without it
#: (``derived["refused"]``, with ``success=False``). The refusal depends only on the result in hand
#: (``result.shots is None``) — never on which soc produced it — so a caller that attaches per-shot
#: data itself is analysed as usual. See :attr:`Experiment.needs_shots`.
SHOTS_REFUSAL = (
    "needs per-shot data; this result carries averages only (the soc returned no shots)"
)


def shots_refusal(kept: dict | None = None, withheld: tuple[str, ...] = ()) -> FitResult:
    """The fit a ``needs_shots`` experiment returns for a result without per-shot data.

    ``success=False`` and ``derived["refused"] = SHOTS_REFUSAL``. A *partial* refusal also passes
    ``kept`` — derived quantities that are honest from averages alone — and must then name every
    quantity it withholds in ``withheld`` (``derived["refused_parts"]``), so a reader can tell a
    refused quantity from one the experiment never reports. No quantity that needs shots is kept.
    """
    derived: dict[str, Any] = dict(kept or {})
    if derived and not withheld:
        raise ValueError("a partial shots refusal must name the quantities it withholds")
    derived["refused"] = SHOTS_REFUSAL
    if withheld:
        derived["refused_parts"] = list(withheld)
    return FitResult(params={}, stderr={}, derived=derived, success=False)


@dataclass
class CompilationError(Exception):
    """A program failed to compile — structured context for actionable diagnostics.

    Raised by :meth:`Runner.run` when ``StrumentoProgram`` construction or compilation
    fails. Wraps the original exception with context the user can act on: which experiment,
    what went wrong, and (when available) which calibration path to check.

    The message is a human-readable one-liner; the fields carry the structured data for
    programmatic consumers (e.g. a bringup runner logging the failure). The original
    exception is chained (``raise CompilationError(...) from original``) so the full
    traceback is still available via ``__cause__``.

    This class is **backend-agnostic** by design. When QUA or Qblox compilers land, they
    raise the same ``CompilationError`` with the same fields — the consumer (Runner,
    bringup runner, agent) handles one type regardless of backend.
    """

    experiment: str
    """The experiment class name (e.g. ``"AmplitudeRabi"``)."""

    phase: str
    """Where in the lifecycle the error occurred: ``"build"`` (program construction /
    declaration / compilation) or ``"acquire"`` (execution on the soc)."""

    constraint: str
    """A concise description of what went wrong, extracted from the original exception."""

    original_type: str
    """The original exception's type name (e.g. ``"RuntimeError"``, ``"KeyError"``)."""

    suggestion: str = ""
    """An optional hint about what to fix (e.g. a calibration path to check)."""

    details: dict = field(default_factory=dict)
    """Optional structured details (pulse label, channel, wiring line, etc.)."""

    def __str__(self) -> str:
        parts = [f"{self.experiment}: compilation failed ({self.phase})."]
        parts.append(f"  {self.constraint}")
        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"  Details: {detail_str}")
        return "\n".join(parts)

    def __post_init__(self):
        # Exception.__init__ needs to be called with the message
        Exception.__init__(self, str(self))


@dataclass
class ExperimentOutcome:
    """What an experiment run produces: the raw data, the fit, and an optional figure."""

    result: MeasurementResult
    fit: FitResult | None = None
    figure: Any = None


class Runner:
    """Builds a program from a sequence + sweeps and acquires it on a soc.

    The ``StrumentoProgram`` import is local so this module (and any experiment) imports without
    ``qick``; only an actual :meth:`run` touches it.

    **Arming (D32).** A soc only ever sees ``start_readout``/``poll_data``, so a soc that generates
    its own data has to be told what is being run by the component that knows. That is the Runner:
    it holds the device, it built the program, and (since :meth:`Experiment.go` passes it) it knows
    the experiment. If the soc satisfies :class:`~strumento.core.soc.ArmableSoc` it is handed a
    :class:`~strumento.core.response.SimContext` before acquire; if it satisfies
    :class:`~strumento.core.soc.ShotSource` the per-shot data behind the acquisition is pinned on
    the result. Both checks are structural, so the mock and hardware paths are untouched — and
    ``core`` still imports no simulator (only the protocols).

    **Environment stamping (D34/D24).** The Runner is also where ``meta["environment"]`` is
    written, because acquire time is the only moment the reading is true. Configure it once
    (``Runner(environment=<source>)``) or per run (``go(..., environment=...)``); with neither, the
    null source still stamps, so an unstamped result cannot come out of this path. See
    :mod:`strumento.core.environment` — read-only telemetry, and a failed read never costs data.
    """

    def __init__(self, environment=None) -> None:
        self.environment = environment

    def run(self, device, seq: Seq, sweeps, soc, *, axes=(), experiment=None, reps: int = 100, final_delay=None, environment=None, record=None, **acquire_kw) -> MeasurementResult:
        """Build the program, acquire, attach shots, stamp the environment — one acquisition.

        ``axes`` is the experiment's declared loop axes (D36), forwarded to the program.

        ``record`` is an **explicit** keyword, not something ``**acquire_kw`` absorbs: a misspelled
        ``recrod=`` must not silently mean "no record" (it travels on into ``acquire`` and raises
        there). With a record supplied, one ``progress`` event is emitted when the acquisition
        completes — the acquire itself is a single call into the board, so that is the only honest
        granularity available at this layer; a per-point series would be invented.

        Raises :class:`CompilationError` when the program fails to build or compile — wrapping the
        original exception with the experiment name, the failure phase, and (where extractable) a
        suggestion about which calibration parameter to check.
        """
        from strumento.core.program import StrumentoProgram

        exp_name = type(experiment).__name__ if experiment is not None else "unknown"
        try:
            prog = StrumentoProgram(device, seq=seq, sweeps=sweeps, axes=axes, reps=reps, final_delay=final_delay, soc=soc)
        except Exception as exc:
            raise CompilationError(
                experiment=exp_name,
                phase="build",
                constraint=str(exc),
                original_type=type(exc).__name__,
                suggestion=_suggest_fix(exc),
                details=_extract_details(exc, seq),
            ) from exc
        self._arm(soc, prog, device, experiment, seq)
        result = prog.acquire(soc, **acquire_kw)
        result = self._attach_shots(soc, result)
        source = environment if environment is not None else self.environment
        result = stamp_environment(result, source, device=device, soc=soc)
        if record is not None:
            _emit_acquired(record, result, experiment=experiment, reps=reps)
        return result

    @staticmethod
    def _arm(soc, prog, device, experiment, seq: Seq) -> None:
        """Hand an armable soc the run context (experiment, device, swept axis, sequence).

        The sequence rides along because a *conditional* response (active reset, feed-forward) is
        a per-shot walk of the ``Branch``/``Repeat`` structure, which only the sequence carries.
        """
        if experiment is None or not isinstance(soc, ArmableSoc):
            return
        from strumento.core.response import SimContext

        adc_scale = getattr(soc, "adc_scale", None)
        readout_scale = 1.0
        readout_channels = getattr(prog, "ro_chs", {})
        if adc_scale is not None and readout_channels:
            first_readout = next(iter(readout_channels.values()))
            length = (
                first_readout.get("length")
                if isinstance(first_readout, dict)
                else getattr(first_readout, "length", None)
            )
            if length is not None and float(length) > 0.0:
                candidate = float(adc_scale) / float(length)
                if math.isfinite(candidate) and candidate > 0.0:
                    readout_scale = candidate
        soc.arm(
            SimContext(
                experiment=experiment,
                device=device,
                xpts=prog.xpts(),
                seq=seq,
                readout_output_scale=readout_scale,
            )
        )

    @staticmethod
    def _attach_shots(soc, result: MeasurementResult) -> MeasurementResult:
        """Pin a shot-resolved soc's per-shot data on the result (its per-shot ``shots`` slot)."""
        if result.shots is None and isinstance(soc, ShotSource):
            shots = soc.collect_shots()
            if shots is not None:
                result.shots = shots
        return result


class Experiment(ABC):
    """Base for a characterization experiment (Rabi, T1, T2, …).

    Subclasses implement :meth:`sequence` and :meth:`analyze`; :meth:`display` is optional.

    **Per-shot data (#199).** An experiment whose analysis needs per-shot data (shot statistics,
    a single-shot histogram, a confusion matrix) sets :attr:`needs_shots` to ``True``. On a
    result without shots its :meth:`analyze` refuses with :func:`shots_refusal` — ``success=False``
    and ``derived["refused"] = SHOTS_REFUSAL`` — rather than a bare failure or a shot quantity
    computed from averages. Only the result decides: a result that carries ``shots`` is analysed
    whatever soc produced it. Per-shot data reaches a result through the
    :class:`~strumento.core.soc.ShotSource` protocol (the simulator implements it; a board soc
    does not yet), or from a caller that attaches it.
    """

    #: Whether :meth:`analyze` needs per-shot data (``result.shots``). Read by
    #: ``list_experiments`` and the registry guard; ``False`` for an averaged experiment.
    needs_shots: ClassVar[bool] = False

    def __init__(self, device, *, runner: Runner | None = None) -> None:
        self.device = device
        self.runner = runner if runner is not None else Runner()

    @abstractmethod
    def sequence(self) -> tuple[Seq, tuple]:
        """Return ``(seq, sweeps)`` to run."""

    def axes(self) -> tuple:
        """The declared loop axes (D36) for this experiment's sweeps. Default: none — a legacy
        unbound sweep auto-declares its own axis (deprecated); override to declare axes and bind
        sweeps to them (``Sweep(start, stop, on=...)``)."""
        return ()

    @abstractmethod
    def analyze(self, result: MeasurementResult) -> FitResult:
        """Fit the result → a typed :class:`FitResult` (no plotting, no printing)."""

    def display(self, result: MeasurementResult, fit: FitResult | None):
        """Build a figure from result + fit.

        Default: looks up a registered plot function for this experiment type (the same
        MRO-walk registry pattern as response models). Returns ``None`` if no plot function
        is registered or if the plot function returns ``None`` (e.g. matplotlib not installed).

        Subclasses may override for custom rendering; the registry is the zero-boilerplate path.
        """
        from strumento.core.display import display_for
        return display_for(self, result, fit)

    def recovery_map(self) -> dict[str, str]:
        """Extra ``{derived key: truth key}`` pairs for the ``fit-recovery`` gate.

        The gate pairs a fit's ``derived`` quantities with the simulator's ground truth **by name** —
        ``derived["T1"]`` against ``truth["T1"]`` — which is a naming convention, not physics, so
        ``core`` can apply it without interpreting a single key (hard rule 2). Where a pack names the
        same quantity differently on the two sides (``derived["phase_deg"]`` vs
        ``truth["parity_phase"]``), the experiment says so here. Default: nothing to add.
        """
        return {}

    def acquire(self, soc, *, record=None, **run_kw) -> MeasurementResult:
        """Acquire this experiment's result before analysis.

        Standard experiments make one program through the injected runner. Host-batched
        experiments may override this narrow seam to merge several ordinary acquisitions while
        leaving :meth:`go` as the shared analysis, display, and gate lifecycle.
        """
        seq, sweeps = self.sequence()
        return self.runner.run(
            self.device,
            seq,
            sweeps,
            soc,
            axes=self.axes(),
            experiment=self,
            record=record,
            **run_kw,
        )

    def go(
        self,
        soc,
        *,
        analyze: bool = True,
        display: bool = False,
        record=None,
        recovery_tolerance: float = DEFAULT_RECOVERY_TOLERANCE,
        **run_kw,
    ) -> ExperimentOutcome:
        """Run → (optionally) fit → (optionally) draw, as a typed :class:`ExperimentOutcome`.

        The experiment is passed **into** the run call: the swept axis in lab-native units lives on
        the experiment (``AmplitudeRabi.gains``; the compiled sweep travels inside the pulse), so
        the Runner needs it to arm a simulator (D32). A custom Runner therefore takes
        ``experiment=`` (or absorbs it in ``**kw``).

        ``run_kw`` reaches the Runner, so ``go(soc, reps=..., environment=<source>)`` supplies the
        acquisition's telemetry stamp (D34) alongside the acquire options.

        **``record=`` — task integration, and the gate emitter (the point of this method).**
        With a :class:`~strumento.core.record.TaskRecord` supplied, a standard experiment emits its
        own record events: ``progress`` when the acquisition completes (from the Runner) and again
        after the fit, carrying every derived quantity as a metric; ``artifact`` for a figure that
        was actually written to the record's ``artifacts/``; and **``gate`` events computed here**:

        * ``fit`` — always, when analysis ran: did the fitter converge?
        * ``fit-recovery`` — additionally, when the soc carries an armed **ground truth** (a
          simulator): does each fitted quantity match the truth within ``recovery_tolerance``? One
          event per run, ``pass`` = every compared quantity inside tolerance, with the worst
          offender in ``params`` and the full per-quantity map alongside. One event because a reader
          takes the *last* gate event per name — several events under one name would silently
          discard all but the last verdict.

        *The trust boundary.* Gate verdicts feed a learned routing table that decides which model
        gets which work; a table fed self-graded homework is worse than no table. So the comparison
        happens **here**, in shipped and reviewed code, from the fit and the armed truth — and a task
        script's job is to *route* verdicts (``rec.gate(...)``), never to author them. Be honest
        about how strong that is:

        * *Structural, for a shipped experiment:* the comparison is between the pack's fitter output
          and the armed truth, and a script that merely *runs* ``AmplitudeRabi.go(...)`` cannot reach
          into it. It chooses whether the comparison happens; it does not choose the answer.
        * *Not structural:* two ways out, both real. :meth:`~strumento.core.record.TaskRecord.gate`
          is a public method, so a script can write ``rec.gate("fit-recovery", True)`` with no fit
          behind it; and :meth:`analyze` is a subclass's method, so a script that defines its *own*
          :class:`Experiment` can feed this emitter a fabricated :class:`FitResult` and get a
          genuine-looking pass. Vetted-code-computes-the-verdict holds for the **shipped**
          experiments, not for arbitrary agent-authored subclasses. This is a contract, not a
          sandbox.
        * *Where the teeth are:* supervisor-side — verdicts are re-derived from the record before
          being forwarded, only gates the manifest declared are forwarded at all, and forwarding
          requires a non-empty ``authored_by.model``. Enforcement belongs on the side that consumes
          the number, not the side that emits it.
        """
        result = self.acquire(soc, record=record, **run_kw)
        fit = self.analyze(result) if analyze else None
        figure = self.display(result, fit) if display else None
        if record is not None:
            if fit is not None:
                _emit_fit(record, self, fit)
                _emit_gates(record, self, soc, fit, tolerance=recovery_tolerance)
            if figure is not None:
                _emit_figure(record, self, figure)
        return ExperimentOutcome(result=result, fit=fit, figure=figure)


# --------------------------------------------------------------------------------------
# record emission (the vetted half — see Experiment.go's docstring for the trust boundary)
# --------------------------------------------------------------------------------------


def _n_points(xpts) -> int:
    """How many sweep points an acquisition covered (a single point has no axis)."""
    try:
        return int(len(xpts))
    except TypeError:
        return 1


def _emit_acquired(record, result: MeasurementResult, *, experiment=None, reps: int) -> None:
    """The acquisition completed: one ``progress`` event, stage 1 of a run."""
    points = int(result.meta.get("acquisition_points", _n_points(result.xpts)))
    name = type(experiment).__name__ if experiment is not None else "acquisition"
    record.progress(1, label=f"{name}: acquired {points} point(s)", points=points, reps=reps)


def _emit_fit(record, experiment, fit: FitResult) -> None:
    """The fit landed: one ``progress`` event carrying every derived quantity as a metric.

    Derived quantities are the series a monitor actually wants plotted (``t1_us`` over a night,
    ``infidelity`` over a calibration loop), and ``metrics`` is an open string→number map precisely
    so no schema changes when a new experiment reports a new quantity.
    """
    metrics = {key: value for key, value in fit.derived.items() if _is_number(value)}
    record.progress(2, label=f"{type(experiment).__name__}: fit", **metrics)


def _emit_figure(record, experiment, figure) -> None:
    """Write a figure into the record's ``artifacts/`` and record it — if it can be written.

    Duck-typed on ``savefig`` so ``core`` still needs no plotting dependency (hard rule: the Display
    layer is optional). A figure object that cannot save itself produces **no** artifact event: the
    contract says an ``artifact`` path resolves to a real file, and inventing one would break that.
    """
    savefig = getattr(figure, "savefig", None)
    if not callable(savefig):
        return
    stem = type(experiment).__name__.lower()
    index = len(list(record.artifacts.glob(f"{stem}-*.png")))
    path = record.artifacts / f"{stem}-{index:03d}.png"
    savefig(path)
    record.artifact(path, kind="figure")


def _emit_gates(record, experiment, soc, fit: FitResult, *, tolerance: float) -> None:
    """Compute and record this run's gate verdicts. **Vetted code only** — see :meth:`Experiment.go`."""
    record.gate(
        "fit",
        bool(fit.success),
        experiment=type(experiment).__name__,
        derived=sorted(fit.derived),
    )
    truth = _armed_truth(soc)
    if not truth:
        return  # no ground truth armed: there is nothing to recover, so no verdict to give
    compared = _recovery_errors(fit, truth, experiment.recovery_map())
    if not compared:
        return  # a truth is armed but shares no quantity with this fit — silence beats a fake pass
    worst = max(compared, key=lambda key: compared[key])
    record.gate(
        "fit-recovery",
        all(error <= tolerance for error in compared.values()),
        experiment=type(experiment).__name__,
        metric=f"{worst}_rel_err",
        value=compared[worst],
        tolerance=float(tolerance),
        compared=compared,
    )


def _armed_truth(soc) -> dict:
    """The ground-truth calibration a simulator was armed with, or ``{}`` for real hardware.

    Structural, like every other soc capability in this module: the armed context is asked first
    (that is what actually produced the data), the soc's own default second. A board has neither and
    contributes nothing, so the hardware path emits no ``fit-recovery`` verdict at all — which is
    correct. On hardware there is no truth to recover, and *physics agreement* is a different gate
    with a different meaning.
    """
    context = getattr(soc, "context", None)
    for candidate in (getattr(context, "truth", None), getattr(soc, "truth", None)):
        if isinstance(candidate, dict) and candidate:
            return dict(candidate)
    return {}


def _recovery_errors(fit: FitResult, truth: dict, extra_map: dict) -> dict[str, float]:
    """Relative error per quantity the fit and the truth both name.

    Paired by name (plus the experiment's ``recovery_map`` for the cases a name match cannot see).
    A truth value of zero is skipped rather than fudged: a relative error against zero is not a
    number, and an absolute one would silently change what the tolerance means.
    """
    errors: dict[str, float] = {}
    for key, fitted in fit.derived.items():
        truth_key = extra_map.get(key, key)
        expected = truth.get(truth_key)
        if not (_is_number(fitted) and _is_number(expected)) or float(expected) == 0.0:
            continue
        errors[key] = abs(float(fitted) / float(expected) - 1.0)
    return errors


# True means float(value) succeeds, so a caller may pass the value to float()
def _is_number(value) -> TypeGuard[str | SupportsFloat | SupportsIndex]:
    if isinstance(value, bool) or value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------------------
# compilation error helpers (extract actionable context from raw exceptions)
# --------------------------------------------------------------------------------------

#: patterns in exception messages → suggested calibration paths to check.
_SUGGESTION_PATTERNS: list[tuple[str, str]] = [
    ("pulse length", "Check the pulse sigma/length in calibration (too short for the DAC fabric clock)."),
    ("shorter than the min", "Increase the envelope sigma or length — the DAC needs a minimum number of samples."),
    ("no line", "The device wiring has no line matching the experiment's pulse. Check device.yaml wiring.lines."),
    ("no role", "The wiring line exists but has no generator with the required role. Check device.yaml generators."),
    ("no generator for", "No generator covers the requested frequency band. Check the generator's band_hz or Nyquist zone."),
    ("frequency sweep", "A frequency sweep crosses a generator band boundary. Narrow the sweep or check the wiring."),
    ("band boundary", "The swept frequency range spans two different generators. Narrow the sweep range."),
    ("no lowering registered", "The pulse uses an envelope type the compiler doesn't know. Check the pack's envelope registration."),
    ("undeclared axis", "A sweep is bound to a loop axis that was never declared. Add it to axes=[] on the experiment."),
    ("ambiguous wiring", "Multiple generators claim the same frequency on the same line. Check for overlapping band_hz."),
    ("readout", "Check the readout wiring: adc_chs, readout_length, or the readout frequency."),
    ("soccfg", "The device's soccfg snapshot may not match the board or may be missing. Check device.yaml soccfg_snapshot."),
]


def _suggest_fix(exc: Exception) -> str:
    """Match the exception message against known patterns and return a suggestion."""
    msg = str(exc).lower()
    for pattern, suggestion in _SUGGESTION_PATTERNS:
        if pattern in msg:
            return suggestion
    return ""


def _extract_details(exc: Exception, seq: Seq | None) -> dict:
    """Extract structured details from the exception and sequence context."""
    details: dict[str, Any] = {}
    msg = str(exc)

    # Try to extract line/role info from KeyError messages like "no line 'qubit' in wiring"
    if "line" in msg.lower() and "'" in msg:
        parts = msg.split("'")
        if len(parts) >= 2:
            details["line"] = parts[1]

    # Count ops in the sequence for context
    if seq is not None and hasattr(seq, "ops"):
        from strumento.core.pulses import Measure, Pulse
        details["n_ops"] = len(seq.ops)
        details["n_pulses"] = sum(1 for op in seq.ops if isinstance(op, Pulse))
        details["has_readout"] = any(isinstance(op, Measure) for op in seq.ops)

    return details
