"""spin calibration schema (spec §15.5) — skeleton.

Minimal typed subset: per-dot qubit frequency and (optionally) exchange coupling. Gate-voltage
operating points and full tune-up parameters land with the pulse library / experiments.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DotCal(_Strict):
    """Per-dot calibration: qubit frequency (MHz) and optional exchange coupling (MHz)."""

    f_qubit_mhz: float
    exchange_j_mhz: float | None = None


class SpinCalibration(_Strict):
    """Quantum-dot instance calibration, keyed by dot name."""

    dots: dict[str, DotCal] = {}
