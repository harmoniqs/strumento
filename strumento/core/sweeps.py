"""Sweep ranges and declared loop axes (D23, D36).

Two kinds of value, one sweep model:

* A :class:`Loop` is a **declared loop axis** — ``name`` + ``points`` — declared at program
  scope (``StrumentoProgram(..., axes=[Loop("ax", 51)])``). This is the tProc-v2 model: the
  user says "these are my loop axes" by hand, and each declared axis emits exactly one
  ``add_loop``.
* A :class:`Sweep` is a **parameter range** ``start → stop`` that a pulse/delay field takes
  instead of a scalar. The **bound form** — ``Sweep(start=…, stop=…, on="ax")`` — rides the
  declared axis ``"ax"`` and takes its point count from the :class:`Loop`; several sweeps may
  bind the same axis, which is how two parameters step *together* in a single loop (a joint
  sweep). The **legacy form** — ``Sweep("amp", 0, 30000, 51)`` — auto-declares a per-name axis
  (deprecated, D36): the compiler must not invent loop structure. Constructing one emits a
  ``DeprecationWarning`` attributed to the caller's line, once; compiling it does not warn again.

**Lab-native display axis (D23).** A user sweeps in lab-native units (gain int codes, MHz, µs);
the cqed factory may :meth:`Sweep.mapped` the endpoints into hardware units (e.g. int code →
fraction) for the compiler, but the *experiment* records the lab-native values as ``xpts`` so
notebooks read familiar units. ``mapped`` therefore preserves the pre-map axis (legacy sweeps)
or the inverse map (bound sweeps).

This module is **pure**: no ``qick`` import.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Loop:
    """A declared loop axis: ``name``, stepped over ``points`` (D36).

    Declared at program scope; the compiler emits one ``add_loop`` per declared axis, never
    more. The point count lives here — a sweep *bound* to this axis carries only its range.
    """

    name: str
    points: int

    def __post_init__(self) -> None:
        if not isinstance(self.points, int) or self.points < 1:
            raise ValueError(
                f"loop axis {self.name!r} needs a positive integer point count, got {self.points!r}"
            )


@dataclass(frozen=True)
class Sweep:
    """A parameter range ``start → stop`` a pulse or delay field takes instead of a scalar.

    **Bound form (preferred, D36):** ``Sweep(start=…, stop=…, on="ax")`` — rides the declared
    axis ``"ax"``; the point count comes from its :class:`Loop`. **Legacy form (deprecated):**
    ``Sweep("amp", 0, 30000, 51)`` — auto-declares a per-name axis with ``points``, and warns
    (``DeprecationWarning``) at the line that constructs it.

    ``xpts`` is the lab-native display axis: after :meth:`mapped` it is the *pre-map* axis
    (legacy sweeps, where the point count is known), so unit conversion for the compiler does
    not change how results are plotted.
    """

    name: str | None = None
    start: float = 0.0
    stop: float = 0.0
    points: int | None = None
    on: str | None = None
    _lab_axis: tuple[float, ...] | None = None
    _lab_inverse: Callable[[float], float] | None = None

    def __post_init__(self) -> None:
        if self.on is not None and self.points is not None:
            raise ValueError(
                f"a sweep bound to declared axis {self.on!r} takes its point count from the "
                f"Loop — pass points= there, not on the Sweep"
            )
        if self.on is None and self.points is None:
            raise ValueError(
                "an unbound sweep needs a point count (legacy per-name axis, deprecated) — "
                "or bind it to a declared axis with on=…"
            )
        if self.on is None and self.name is None:
            raise ValueError("a legacy (unbound) sweep needs a name for its implicit axis")
        if self.on is None and self._lab_axis is None:
            # Warn where the user wrote the sweep: stacklevel 3 skips this __post_init__ and the
            # dataclass-generated __init__ that calls it. A sweep carrying a pinned lab axis was
            # derived by mapped() from one that already warned, so it does not warn again.
            warnings.warn(
                f"legacy per-name Sweep {self.name!r} is deprecated (D36): declare the axis on "
                f"the program (axes=[Loop({self.name!r}, {self.points})]) and bind the sweep to "
                f"it (Sweep(start, stop, on={self.name!r}))",
                DeprecationWarning,
                stacklevel=3,
            )

    @property
    def xpts(self) -> np.ndarray:
        """The lab-native display axis (D23). A bound sweep has no point count of its own —
        its axis is computed from the declared :class:`Loop` (see ``StrumentoProgram.xpts``),
        so reading ``xpts`` here raises unless :meth:`mapped` pinned a lab axis."""
        if self._lab_axis is not None:
            return np.asarray(self._lab_axis)
        if self.points is None:
            raise ValueError(
                "a bound sweep (on=…) has no point count of its own — the axis is computed "
                "from the declared Loop (StrumentoProgram.xpts)"
            )
        return np.linspace(self.start, self.stop, self.points)

    def mapped(self, fn: Callable[[float], float], inverse: Callable[[float], float] | None = None) -> Sweep:
        """Return a new Sweep with ``start``/``stop`` transformed by ``fn`` (e.g. int-code →
        fraction), keeping the lab-native display information (D23).

        Legacy sweeps pin the current linspace as the pre-map ``_lab_axis``. Bound sweeps carry
        no point count, so instead the optional ``inverse`` (hardware → lab) rides along and
        lets the program reconstruct the lab axis from the declared :class:`Loop`."""
        lab: tuple[float, ...] | None = self._lab_axis
        if lab is None and self.points is not None:
            lab = tuple(float(v) for v in np.linspace(self.start, self.stop, self.points))
        return Sweep(
            name=self.name,
            start=fn(self.start),
            stop=fn(self.stop),
            points=self.points,
            on=self.on,
            _lab_axis=lab,
            _lab_inverse=inverse,
        )

    def axis_values(self, points: int) -> np.ndarray:
        """A ``points``-step axis over ``[start, stop]`` in the compiled domain (pre-realization)."""
        return np.linspace(self.start, self.stop, points)
