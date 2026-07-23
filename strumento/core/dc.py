"""Slow DC voltage axes — the external-voltage-source seam (spin-qubit electrostatics &c.).

QICK owns the fast (RF/baseband pulse) axes; gate-defined devices also need **slow DC** biasing
from an external voltage source (a QDAC-class instrument). This module is the modality-agnostic
seam, shaped after the pattern spinQICK proved:

* :class:`VoltageSource` — a driver **Protocol** (labs wrap their own instrument; strumento
  ships no drivers, only :class:`MockVoltageSource` for tests/offline work).
* :class:`DCAxis` — one named axis: source channel + safety clamp + optional cross-coupling.
* :class:`DCSource` — the device-aware wrapper: named-axis addressing, ``max_v`` clamping,
  cross-coupling-compensated moves, and a voltage-state snapshot (for stamping results).

Pure: no ``qick``, no instrument I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class VoltageSource(Protocol):
    """The minimal driver contract for an external DC source (implement per instrument)."""

    def set_voltage(self, channel: int, volts: float) -> None: ...

    def get_voltage(self, channel: int) -> float: ...


class MockVoltageSource:
    """An in-memory VoltageSource for tests and offline development."""

    def __init__(self) -> None:
        self.voltages: dict[int, float] = {}
        self.history: list[tuple[int, float]] = []

    def set_voltage(self, channel: int, volts: float) -> None:
        self.voltages[channel] = volts
        self.history.append((channel, volts))

    def get_voltage(self, channel: int) -> float:
        return self.voltages.get(channel, 0.0)


@dataclass(frozen=True)
class DCAxis:
    """One named slow axis: instrument channel + hard safety clamp + cross-coupling row.

    ``crosscoupling`` maps *other* axis names to the compensation coefficient applied to them
    when this axis moves (the lever-arm matrix row, as measured on the device).
    """

    channel: int
    max_v: float
    crosscoupling: dict[str, float] = field(default_factory=dict)


class DCSource:
    """Named-axis DC control over a driver, with clamping + optional compensated moves."""

    def __init__(self, source: VoltageSource, axes: dict[str, DCAxis]) -> None:
        self.source = source
        self.axes = dict(axes)

    def set(self, name: str, volts: float) -> None:
        """Set one axis (clamped to ``±max_v``); no compensation."""
        ax = self._axis(name)
        self.source.set_voltage(ax.channel, self._clamp(name, volts))

    def get(self, name: str) -> float:
        return self.source.get_voltage(self._axis(name).channel)

    def set_compensated(self, name: str, volts: float) -> None:
        """Move one axis and counter-move its cross-coupled neighbours.

        For each ``other: coeff`` in the axis's ``crosscoupling``, the other axis is shifted by
        ``-coeff * dV`` — the measured lever-arm compensation that keeps the rest of the device
        stationary in energy space while this gate moves.
        """
        ax = self._axis(name)
        dv = self._clamp(name, volts) - self.get(name)
        self.set(name, volts)
        for other, coeff in ax.crosscoupling.items():
            self.set(other, self.get(other) - coeff * dv)

    def snapshot(self) -> dict[str, float]:
        """All axis voltages by name — stamp this into ``MeasurementResult.meta``."""
        return {name: self.get(name) for name in self.axes}

    # -- internals -----------------------------------------------------------------------

    def _axis(self, name: str) -> DCAxis:
        if name not in self.axes:
            raise KeyError(f"unknown DC axis {name!r} (have {sorted(self.axes)})")
        return self.axes[name]

    def _clamp(self, name: str, volts: float) -> float:
        ax = self.axes[name]
        if abs(volts) > ax.max_v:
            raise ValueError(f"axis {name!r}: |{volts} V| exceeds max_v={ax.max_v} V")
        return volts
