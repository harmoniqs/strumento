"""spin (quantum-dot) element types (spec §15.5) — spinQICK-modeled.

Gate-defined quantum dots: a `Dot` qubit driven for control, with `Plunger` and `Barrier` DC gates
that set the electrostatics (charge stability, exchange). Qubit encodings are Loss–DiVincenzo
(single-spin) or exchange-only — carried on the `Dot` as ``encoding``.

Element types + `lines()` + `ops_cls` (LD drive on dots, exchange pulses on barriers); the
slow-DC electrostatics live on the external source seam (``strumento.core.dc``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from strumento.core.wiring import LineRef
from strumento.packs.spin.library.exchange import ExchangeOps
from strumento.packs.spin.library.ld import LDOps


@dataclass
class Dot:
    """A gate-defined quantum-dot spin qubit (LD single-spin or exchange-only)."""

    name: str
    encoding: str = "ld"  # "ld" (Loss–DiVincenzo) | "eo" (exchange-only)

    ops_cls: ClassVar[type | None] = LDOps

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
    """A barrier gate — a DC voltage line setting inter-dot tunnel coupling / exchange.

    The slow DC operating point lives on the external source (``core.dc``); a barrier may also
    carry a ``fast`` generator role for exchange pulses (declared in the instance wiring).
    """

    name: str

    ops_cls: ClassVar[type | None] = ExchangeOps

    def lines(self) -> list[LineRef]:
        return [LineRef(self.name, "dc")]
