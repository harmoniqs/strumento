"""tweezer calibration schema (D19) — skeleton.

A typed subset of what a neutral-atom AOD instance needs. Kept minimal for the skeleton; real
move/trap parameters land with the pulse library.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    #: no calibration value is NaN or infinite (#191)
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AODAxisCal(_Strict):
    """AOD center frequency + usable span (MHz) for one axis — sets the position ↔ frequency map."""

    center_mhz: float
    span_mhz: float


class TweezerCalibration(_Strict):
    """Neutral-atom AOD instance calibration."""

    aod_x: AODAxisCal
    aod_y: AODAxisCal
    trap_depth_mk: float | None = None
