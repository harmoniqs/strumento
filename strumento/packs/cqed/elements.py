"""cqed element *types* (spec §15, D21).

An element type describes a class of physical object (a transmon, a bosonic mode, a coupler, a
readout resonator). A device *instance* names concrete elements of these types in its YAML
(``qubit``, ``manipulate``, ``storage``, …); ``core.device.load`` composes the types per the
topology and binds one ops namespace per named element.

Two things every element type exposes (D21):
  * ``lines()`` — the :class:`~strumento.core.wiring.LineRef`\\ s it drives (using the *instance*
    name as the line), so ``load`` can validate the element against the wiring.
  * ``ops_cls`` — the ops class its named instances bind to (element-TYPE-keyed, never
    instance-keyed). ``None`` means "ops for this type land in a later slice".

Pure: no ``qick`` import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from strumento.core.wiring import LineRef
from strumento.packs.cqed.library.transmon import TransmonOps


@dataclass
class Transmon:
    """A transmon qubit. Drives its ``drive`` role; its sideband generator is added in Task 13."""

    name: str
    f_ge: float
    f_ef: float | None = None
    pulses: dict = field(default_factory=dict)

    ops_cls: ClassVar[type | None] = TransmonOps

    def lines(self) -> list[LineRef]:
        return [LineRef(self.name, "drive")]


@dataclass
class BosonicMode:
    """A bosonic (cavity) mode — a storage or manipulate mode. Ops land in Task 13.

    Multiple modes can share one physical line (the 7 Schuster storage modes all ride the
    single ``storage`` DAC at different frequencies), so ``line`` is separable from ``name`` and
    defaults to it.
    """

    name: str
    role: str = "storage"
    line: str | None = None

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return [LineRef(self.line or self.name, "main")]


@dataclass
class Coupler:
    """A tunable coupler / flux line, band-split into low/high generators. Ops land in Task 13."""

    name: str = "coupler"

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return [LineRef(self.name, "low"), LineRef(self.name, "high")]


@dataclass
class ReadoutResonator:
    """The dispersive readout resonator.

    The readout signal chain is modeled by :class:`~strumento.core.wiring.ReadoutWiring`, not a
    plain :class:`~strumento.core.wiring.Line`, so this element drives no ``LineRef``\\ s. Its
    readout *kind* (``DispersiveIQ``) is bound via ``core.readout`` in Task 16.
    """

    name: str = "readout"

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return []
