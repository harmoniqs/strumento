"""tweezer element types (spec §17) — neutral-atom AOD control (after arXiv:2506.13724).

QICK's role in such a system is the **AOD dynamic/transport controller**: each of the two
orthogonal AOD axes (X/Y) is a mux generator summing up to 32 independent DDS tones — one tone per
movable tweezer (frequency → position, amplitude → depth). The static SLM array is *not* QICK-driven.

Skeleton: element *types* + `lines()` + `ops_cls`. Pulse libraries (the `PolyChirp` move factory) and
imaging-interleaved experiments are later work; `ops_cls` is ``None`` until they land.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from strumento.core.wiring import LineRef


@dataclass
class AODAxis:
    """One AOD control axis (X or Y): a mux generator carrying up to 32 dynamic tweezer tones."""

    name: str
    axis: str = "x"

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return [LineRef(self.name, "mux")]


@dataclass
class AtomArray:
    """The trap register (SLM static array + storage/gate zones).

    Not QICK-driven — the FPGA controls the dynamic AOD tweezers, not the SLM — so this element
    drives no DAC lines. It carries the trap/zone topology the imaging readout maps ROIs onto.
    """

    name: str = "array"
    n_traps: int = 0

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return []
