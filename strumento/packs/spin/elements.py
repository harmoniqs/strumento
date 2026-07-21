"""spin (quantum-dot) element types (spec §15.5) — spinQICK-modeled (HRL).

Gate-defined quantum dots: a `Dot` qubit driven for control, with `Plunger` and `Barrier` DC gates
that set the electrostatics (charge stability, exchange). Qubit encodings are Loss–DiVincenzo
(single-spin) or exchange-only — carried on the `Dot` as ``encoding``.

Skeleton: element types + `lines()` + `ops_cls`. Pulse libraries (LD / exchange-only control) and
tune-up experiments are later work; `ops_cls` is ``None`` until they land.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from strumento.core.wiring import LineRef


@dataclass
class Dot:
    """A gate-defined quantum-dot spin qubit (LD single-spin or exchange-only)."""

    name: str
    encoding: str = "ld"  # "ld" (Loss–DiVincenzo) | "eo" (exchange-only)

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return [LineRef(self.name, "drive")]


@dataclass
class Plunger:
    """A plunger gate — a DC voltage line setting a dot's chemical potential."""

    name: str

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return [LineRef(self.name, "dc")]


@dataclass
class Barrier:
    """A barrier gate — a DC voltage line setting inter-dot tunnel coupling / exchange."""

    name: str

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return [LineRef(self.name, "dc")]
