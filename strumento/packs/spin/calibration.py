"""spin calibration schema (spec §15.5) — skeleton.

Minimal typed subset: per-dot qubit frequency and (optionally) exchange coupling. Gate-voltage
operating points and full tune-up parameters land with the pulse library / experiments.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpinPulseCal(_Strict):
    """One calibrated spin-control pulse. Gain is a v2 **fraction** — spin stacks are
    fraction-native (no legacy int-code DB), so no conversion happens at the boundary."""

    gain: float
    sigma: float = 0.0        # gaussian sigma, µs (LD drive)
    length: float = 0.0       # flat length, µs (exchange)
    type: str = "gauss"


class DotCal(_Strict):
    """Per-dot calibration: qubit frequency (MHz), optional exchange, and the pulse library."""

    f_qubit_mhz: float
    exchange_j_mhz: float | None = None
    pulses: dict[str, SpinPulseCal] = {}


class GateCal(_Strict):
    """Per-gate electrostatics: the slow-DC axis (external source) + safety + lever arms.

    A barrier gate may also carry a calibrated fast ``exchange`` pulse (flat-top on the gate's
    ``fast`` generator role).
    """

    dc_channel: int
    max_v: float = 1.5
    crosscoupling: dict[str, float] = {}
    exchange: SpinPulseCal | None = None


class SpinCalibration(_Strict):
    """Quantum-dot instance calibration."""

    dots: dict[str, DotCal] = {}
    #: slow-DC gate map (plungers, barriers, sensor gates), keyed by element/axis name
    gates: dict[str, GateCal] = {}
