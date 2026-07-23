"""Pulse intermediate representation + fluent sequence builder (spec §6.1, D22).

The IR is a small set of plain dataclasses. **Envelopes** are a *typed union* of **frozen**
dataclasses — ``Gaussian`` / ``FlatTop`` / ``Const`` / ``Arb`` — each carrying its own
parameters (no flat ``Shape`` enum, no ``None``-field conventions). Frozen ⇒ hashable, which
is exactly what lets the compiler (Task 8) dedup envelopes on ``(generator, envelope)`` for
free (spec R6). The union is **open**: a pack registers additional envelope types with the
compiler's lowering registry (the ``tweezer`` pack adds ``PolyChirp`` there, not here).

``Pulse`` / ``Delay`` / ``Raw`` / ``Segment`` / ``Measure`` / ``Seq`` are kept mutable and
Pydantic-free: this is the hot path, and the builder mutates the op list in place.

``Measure`` is the readout op that ``Seq.measure()`` emits. It is not in the spec's original
§6.1 IR list (it was added in the round-2 review); it exists so a sequence can express "read
out here" without the caller hand-assembling a trigger.

This module is **pure**: it never imports ``qick``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # annotation-only: keeps the IR module dependency-light
    from strumento.core.sweeps import Sweep

from strumento.core.wiring import LineRef

# --------------------------------------------------------------------------------------
# Envelopes — frozen (hashable) typed union (D22)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gaussian:
    """A Gaussian envelope, parameterized by its standard deviation (µs)."""

    sigma_us: float


@dataclass(frozen=True)
class FlatTop:
    """A flat-top pulse: Gaussian ramps of width ``ramp_sigma_us`` around a flat ``length_us``."""

    ramp_sigma_us: float
    length_us: float


@dataclass(frozen=True)
class Const:
    """A constant (square) envelope of the given length (µs)."""

    length_us: float


@dataclass(frozen=True)
class Arb:
    """An arbitrary sampled envelope (e.g. an imported optimal-control solution).

    Samples are coerced to tuples in ``__post_init__`` so the dataclass stays *frozen and
    hashable* — the same dedup-key invariant the analytic envelopes satisfy. The compiler
    (Task 15) reads them back as arrays via ``numpy.asarray``.
    """

    idata: tuple[float, ...]
    qdata: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "idata", tuple(float(x) for x in self.idata))
        if self.qdata is not None:
            object.__setattr__(self, "qdata", tuple(float(x) for x in self.qdata))


Envelope = Gaussian | FlatTop | Const | Arb


# --------------------------------------------------------------------------------------
# IR ops
# --------------------------------------------------------------------------------------


@dataclass
class Pulse:
    """A single pulse: which line/role to play on, at what frequency, gain, phase, and shape.

    ``gain`` is the normalized v2 fraction by the time a pulse reaches the compiler (the cqed
    factories convert lab-native int codes at the pack boundary, D23). ``start_us`` is an
    optional per-pulse offset used inside a :class:`Segment` for staggered simultaneity.
    """

    line: LineRef
    freq_mhz: float
    gain: float
    envelope: Envelope
    phase_deg: float = 0.0
    start_us: float | None = None
    label: str | None = None


@dataclass
class Delay:
    """Idle for ``duration_us`` — a scalar or a :class:`~strumento.core.sweeps.Sweep` (a swept
    idle is the T1 / T2-Ramsey primitive; the compiler lowers it to a swept ``delay``)."""

    duration_us: float | Sweep


@dataclass
class Raw:
    """An escape hatch: call ``fn(program)`` at this point in the sequence."""

    fn: Callable


@dataclass
class Segment:
    """A set of pulses played simultaneously (same start), with optional per-pulse offsets."""

    pulses: list[Pulse]


@dataclass
class Measure:
    """A readout op. ``kind`` names a readout kind (Task 16); ``params`` holds extra options."""

    kind: str | None = None
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class Branch:
    """Measurement-conditional control flow — the QEC/feed-forward primitive (spec R1(i)).

    Reads readout input ``adc`` mid-sequence and runs ``then`` when the discriminator test
    passes on that input, ``else_`` otherwise. Lowers to tProc-v2 ``read_and_jump`` + ``label`` +
    ``jump`` (real hardware flow control, not host post-selection). ``component`` is ``"I"`` or
    ``"Q"``; ``test`` is one of ``">="`` / ``"<"`` (the two comparisons v2 provides — invert the
    branches for the others) against an integer ``threshold`` in ADC-accumulator units (a float
    is realized to the nearest int at lowering).

    This is the in-sequence branch (active reset, mid-circuit feed-forward). The *result-layer*
    herald/post-selection path is separate (``MeasurementResult.postselect``) and carries no
    hardware-feedback dependency.
    """

    adc: int
    threshold: float
    then: Seq
    else_: Seq | None = None
    component: str = "I"
    test: str = "<"


@dataclass
class Repeat:
    """Run ``body`` ``count`` times as a bounded hardware loop (tProc-v2 ``open_loop``/``close_loop``).

    The building block for repeated syndrome extraction / dynamical-decoupling trains — a fixed
    repeat that stays on the tProc, distinct from a swept :class:`~strumento.core.sweeps.Sweep`
    (which is a *measured* axis with an ``add_loop``).
    """

    body: Seq
    count: int


Op = Pulse | Delay | Raw | Segment | Measure | Branch | Repeat


# --------------------------------------------------------------------------------------
# Fluent builder
# --------------------------------------------------------------------------------------


class Seq:
    """A fluent builder over an ordered op list: ``Seq().play(p).wait(0.5).measure()``.

    Every mutating method returns ``self`` for chaining; ``+`` concatenates two sequences into
    a new one without mutating either operand.
    """

    def __init__(self, ops: list[Op] | None = None) -> None:
        self.ops: list[Op] = list(ops) if ops is not None else []
        self.meta: dict = {}  # provenance & friends (e.g. stamped by from_solution, §6.5)

    # -- accumulator + context ergonomics ------------------------------------------------
    # Long sequences read (and diff) better line-by-line than as one chained expression, so
    # ``s += op`` appends in place (a Seq operand extends), and ``with Seq() as s:`` scopes
    # the construction block. Chaining still works everywhere; ``a + b`` (below) remains the
    # non-mutating concatenation. Deliberately NO ``seq + pulse`` sugar — mixing + and +=
    # semantics across operand types invites confusion.

    def __iadd__(self, other: Op | Seq) -> Seq:
        if isinstance(other, Seq):
            self.ops.extend(other.ops)
        else:
            self.ops.append(other)
        return self

    def __enter__(self) -> Seq:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def play(self, item: Pulse | Segment | Seq) -> Seq:
        """Append a pulse or segment; a ``Seq`` is *flattened* in (its ops are spliced)."""
        if isinstance(item, Seq):
            self.ops.extend(item.ops)
        elif isinstance(item, (Pulse, Segment)):
            self.ops.append(item)
        else:
            raise TypeError(f"play() expects Pulse | Segment | Seq, got {type(item).__name__}")
        return self

    def play_parallel(self, items: Iterable[Pulse] | Segment | Seq) -> Seq:
        """Append a :class:`Segment` playing ``items`` simultaneously."""
        if isinstance(items, Segment):
            seg = items
        elif isinstance(items, Seq):
            seg = Segment(pulses=[op for op in items.ops if isinstance(op, Pulse)])
        else:
            seg = Segment(pulses=list(items))
        self.ops.append(seg)
        return self

    def wait(self, duration_us: float) -> Seq:
        self.ops.append(Delay(duration_us))
        return self

    def raw(self, fn: Callable) -> Seq:
        self.ops.append(Raw(fn))
        return self

    def measure(self, *, kind: str | None = None, **params: object) -> Seq:
        self.ops.append(Measure(kind=kind, params=dict(params)))
        return self

    def branch(self, *, adc: int, threshold: float, then: Seq,
               else_: Seq | None = None, component: str = "I", test: str = "<") -> Seq:
        """Append a measurement-conditional branch (in-sequence feed-forward; see :class:`Branch`)."""
        self.ops.append(Branch(adc=adc, threshold=threshold, then=then, else_=else_,
                               component=component, test=test))
        return self

    def repeat(self, body: Seq, count: int) -> Seq:
        """Append a bounded hardware-loop repeat of ``body`` (see :class:`Repeat`)."""
        self.ops.append(Repeat(body=body, count=count))
        return self

    def __add__(self, other: Seq) -> Seq:
        if not isinstance(other, Seq):
            return NotImplemented
        return Seq(self.ops + other.ops)

    def __iter__(self):
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __repr__(self) -> str:
        return f"Seq({[type(op).__name__ for op in self.ops]})"


# --------------------------------------------------------------------------------------
# Optimal-control import boundary (spec §6.5, D17/D23) — the solver × QICK seam
# --------------------------------------------------------------------------------------


def from_solution(solution, *, drive_map, gain_calib, dac_rate: float | None = None) -> Seq:
    """Import a solved optimal-control pulse as a :class:`Seq` wrapping one :class:`Segment`.

    This is the typed, lossless replacement for the load-npz-and-interp1d idiom: drive→line
    mapping is explicit, physical amplitudes convert to v2 fractions through a calibrated
    reference (no int-code hop, D23), resampling is owned here, simultaneity is a
    :class:`Segment`, and provenance rides on ``Seq.meta``.

    Parameters
    ----------
    solution:
        A mapping (or object) with ``times`` — the control grid in µs — and ``controls`` — a
        mapping of drive name → real (or complex I+jQ) amplitude samples in **physical units**
        (MHz Rabi rate).
    drive_map:
        ``{drive name: (line_ref, carrier_freq_mhz)}`` — where each drive plays and at what
        carrier. ``line_ref`` is a :class:`~strumento.core.wiring.LineRef` or ``(line, role)``.
    gain_calib:
        Physical amplitude → v2 gain fraction: any callable (e.g. a cqed ``PiPulseReference``).
    dac_rate:
        Target sample rate (samples/µs). ``None`` keeps the solution grid.
    """
    times, controls = _solution_fields(solution)
    times = np.asarray(times, dtype=float)

    if dac_rate is not None:
        n = max(2, int(round((times[-1] - times[0]) * dac_rate)) + 1)
        grid = np.linspace(times[0], times[-1], n)
    else:
        grid = times

    pulses: list[Pulse] = []
    for name, samples in controls.items():
        if name not in drive_map:
            raise KeyError(f"drive {name!r} has no entry in drive_map (have {sorted(drive_map)})")
        line_ref, carrier = drive_map[name]
        if not isinstance(line_ref, LineRef):
            line_ref = LineRef(*line_ref)

        arr = np.asarray(samples)
        idata = np.interp(grid, times, np.real(arr))
        qdata = np.interp(grid, times, np.imag(arr)) if np.iscomplexobj(arr) else None

        peak = float(np.max(np.abs(arr))) or 1.0
        pulses.append(
            Pulse(
                line=line_ref,
                freq_mhz=carrier,
                gain=float(gain_calib(peak)),
                envelope=Arb(
                    idata=idata / peak,
                    qdata=(qdata / peak) if qdata is not None else None,
                ),
                start_us=0.0,
                label=f"oc_{name}",
            )
        )

    seq = Seq().play_parallel(pulses)
    seq.meta.update(
        {
            "source": "from_solution",
            "drives": sorted(controls),
            "n_samples": int(len(grid)),
            "span_us": float(grid[-1] - grid[0]),
            "dac_rate": dac_rate,
        }
    )
    return seq


def _solution_fields(solution):
    """Accept a mapping or an object with ``times``/``controls`` attributes."""
    if hasattr(solution, "times") and hasattr(solution, "controls"):
        return solution.times, solution.controls
    try:
        return solution["times"], solution["controls"]
    except (KeyError, TypeError) as err:
        raise TypeError(
            "from_solution needs `times` and `controls` (a mapping or attributes); got "
            f"{type(solution).__name__}"
        ) from err
