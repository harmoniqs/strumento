"""spin (quantum-dot) element types (spec S15.5) — spinQICK-modeled.

Gate-defined quantum dots: a `Dot` qubit driven for control, with `Plunger` and `Barrier` DC gates
that set the electrostatics (charge stability, exchange). Qubit encodings are Loss-DiVincenzo
(single-spin), singlet-triplet (S-T0), exchange-only, or hole-spin — carried on the `Dot` as
``encoding``.

Element types + `lines()` + `ops_cls` (LD drive on dots, exchange pulses on barriers); the
slow-DC electrostatics live on the external source seam (``strumento.core.dc``).

Extended element set covers all major spin-qubit architectures:
- Dot, Plunger, Barrier — original gate-defined elements
- Reservoir — electron/hole reservoir for Elzerman readout and dot loading
- ChargeSensor — QPC or SET for charge detection
- DriveLine — dedicated MW/EDSR antenna (when separate from plunger)
- GateSensor — plunger/barrier embedded in RF resonant circuit (dispersive gate sensing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from strumento.core.wiring import LineRef
from strumento.packs.spin.library.exchange import ExchangeOps
from strumento.packs.spin.library.hole_spin import HoleSpinOps
from strumento.packs.spin.library.ld import LDOps
from strumento.packs.spin.library.singlet_triplet import SingletTripletOps


@dataclass
class Dot:
    """A gate-defined quantum-dot spin qubit (LD single-spin encoding).

    The base Dot class carries LDOps (microwave-driven EDSR pi/hpi). For other encodings,
    use the encoding-specific subclasses: STDot, EODot, HoleDot — each carries its own
    ops_cls so that ``core.device.load`` binds the correct ops per ``type(el).ops_cls``.
    """

    name: str
    encoding: str = "ld"

    ops_cls: ClassVar[type | None] = LDOps

    def lines(self) -> list[LineRef]:
        return [LineRef(self.name, "drive")]


@dataclass
class STDot(Dot):
    """A dot in the singlet-triplet (S-T0) encoding.

    The S-T0 qubit lives in a double dot; this element represents one dot of the pair.
    Control is entirely baseband: X-rotation via exchange J(epsilon) detuning pulses,
    Z-rotation via free precession under the inter-dot gradient Delta_Bz.
    No microwave drive needed.
    """

    encoding: str = "st"

    ops_cls: ClassVar[type | None] = SingletTripletOps

    def lines(self) -> list[LineRef]:
        # ST qubits use baseband detuning, not microwave drive.
        return [LineRef(self.name, "detuning")]


@dataclass
class EODot(Dot):
    """A dot in the exchange-only (triple-dot) encoding.

    The exchange-only qubit uses three dots with two exchange axes (J_12, J_23) for
    full SU(2) control — purely electrical, baseband only, no microwaves or magnetic
    gradients. The ops are exchange pulses on the two detuning axes.

    For the exchange-only encoding, the dot itself has no driven RF line — all control
    is via the adjacent barriers. The dot's ops_cls is None (the barriers carry
    ExchangeOps); the dot just holds the encoding label and participates in the topology.
    """

    encoding: str = "eo"

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        # Exchange-only: no RF drive on the dot, all control via barriers.
        return []


@dataclass
class HoleDot(Dot):
    """A Ge/SiGe hole-spin qubit (native SOC, AC gate drive).

    Hole spins are driven by AC electric fields on the plunger gate — the strong
    spin-orbit coupling converts orbital motion into spin rotation directly. No
    micromagnet or dedicated antenna needed. Gate times are typically 1-10 ns.
    """

    encoding: str = "hole"

    ops_cls: ClassVar[type | None] = HoleSpinOps

    def lines(self) -> list[LineRef]:
        # Drive goes through the plunger gate (shared with DC), plus an RF role.
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


@dataclass
class Reservoir:
    """An electron/hole reservoir for Elzerman readout and dot loading.

    The reservoir is tunnel-coupled to one or more dots. Elzerman readout works by
    pulsing the dot chemical potential above the reservoir Fermi level, allowing
    spin-selective tunneling. The reservoir itself has a DC gate controlling its
    chemical potential (back-gate or accumulation gate).
    """

    name: str

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return [LineRef(self.name, "dc")]


@dataclass
class ChargeSensor:
    """A QPC or SET for charge detection (real-time charge sensing).

    The charge sensor sits adjacent to the quantum dots and is sensitive to the
    local electrostatic environment. It provides single-shot readout via either
    DC transport (QPC conductance) or RF reflectometry (SET embedded in a tank
    circuit). The sensor has its own DC bias gate and an RF readout line.
    """

    name: str
    sensor_type: str = "qpc"  # "qpc" | "set"

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        lines = [LineRef(self.name, "dc")]
        # RF reflectometry line for fast readout
        lines.append(LineRef(self.name, "readout"))
        return lines


@dataclass
class DriveLine:
    """A dedicated MW/EDSR antenna (when separate from the plunger gate).

    Some architectures (e.g. Si/SiGe with micromagnet) use a dedicated on-chip
    stripline or antenna for EDSR, separate from the DC gates. This element
    represents that dedicated drive resource.
    """

    name: str

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return [LineRef(self.name, "drive")]


@dataclass
class GateSensor:
    """A plunger/barrier embedded in an RF resonant circuit (dispersive gate sensing).

    Dispersive gate sensing uses the quantum capacitance of a gate to detect
    charge transitions without a dedicated QPC/SET sensor. The gate is part of
    a high-Q RF resonant circuit; the resonant frequency shifts when the dot
    undergoes a charge transition.

    This is a dual-role element: DC voltage for electrostatics + RF readout for sensing.
    """

    name: str

    ops_cls: ClassVar[type | None] = None

    def lines(self) -> list[LineRef]:
        return [LineRef(self.name, "dc"), LineRef(self.name, "readout")]
