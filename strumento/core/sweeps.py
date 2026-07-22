"""Sweep axes (spec §6.4, D23).

A :class:`Sweep` is a named linear axis a pulse field can take instead of a scalar (``gain``,
``freq_mhz``, or a ``Delay`` duration). The compiler lowers it to a ``QickSweep1D`` + one
``add_loop`` (Task 11).

**Lab-native display axis (D23).** A user sweeps in lab-native units (gain int codes, MHz, µs);
the cqed factory may :meth:`Sweep.mapped` the endpoints into hardware units (e.g. int code →
fraction) for the compiler, but the *experiment* records the original lab-native values as
``xpts`` so notebooks read familiar units. ``mapped`` therefore preserves the pre-map axis.

This module is **pure**: no ``qick`` import.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sweep:
    """A named linear sweep ``start → stop`` over ``points`` steps.

    ``xpts`` is the lab-native display axis: by default ``linspace(start, stop, points)``, but
    after :meth:`mapped` it is the *pre-map* axis (so unit conversion for the compiler does not
    change how results are plotted).
    """

    name: str
    start: float
    stop: float
    points: int
    _lab_axis: tuple[float, ...] | None = None

    @property
    def xpts(self) -> np.ndarray:
        if self._lab_axis is not None:
            return np.asarray(self._lab_axis)
        return np.linspace(self.start, self.stop, self.points)

    def mapped(self, fn: Callable[[float], float]) -> Sweep:
        """Return a new Sweep with ``start``/``stop`` transformed by ``fn`` (e.g. int-code →
        fraction), pinning the current ``xpts`` as the lab-native display axis (D23)."""
        return Sweep(
            name=self.name,
            start=fn(self.start),
            stop=fn(self.stop),
            points=self.points,
            _lab_axis=tuple(float(v) for v in self.xpts),
        )
