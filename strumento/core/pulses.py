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


Op = Pulse | Delay | Raw | Segment | Measure


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
