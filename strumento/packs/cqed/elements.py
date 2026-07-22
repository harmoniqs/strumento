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

import numpy as np

from strumento.core.wiring import LineRef
from strumento.packs.cqed.library.transmon import TransmonOps

_TWO_PI = 2 * np.pi


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

    def hamiltonian(self, ops, *, frame: str = "rotating") -> np.ndarray:
        """Duffing bare Hamiltonian (D31/§6.6): ω a†a + (α/2) a†a†aa, α = f_ef − f_ge.

        In the rotating frame the linear ω·a†a term is dropped (rotating at the qubit frequency),
        leaving the anharmonicity; ``frame="lab"`` keeps it. Angular units (2π·f).
        """
        alpha = (self.f_ef - self.f_ge) if self.f_ef is not None else 0.0
        h = (_TWO_PI * alpha / 2) * (ops.adag @ ops.adag @ ops.a @ ops.a)
        if frame == "lab":
            h = h + _TWO_PI * self.f_ge * ops.n
        return h


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
    freq_mhz: float | None = None
    kerr_mhz: float = 0.0

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return [LineRef(self.line or self.name, "main")]

    def hamiltonian(self, ops, *, frame: str = "rotating") -> np.ndarray:
        """Bare cavity Hamiltonian: ω b†b (+ optional self-Kerr K/2 b†b†bb). Angular units.

        The linear ω term is dropped in the rotating frame; the (small) self-Kerr is kept.
        """
        h = (_TWO_PI * self.kerr_mhz / 2) * (ops.adag @ ops.adag @ ops.a @ ops.a)
        if frame == "lab" and self.freq_mhz:
            h = h + _TWO_PI * self.freq_mhz * ops.n
        return h


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
