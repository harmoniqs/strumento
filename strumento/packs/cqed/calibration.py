"""cqed calibration schema + lab-native gain conversion (spec §6.1, D23).

A **typed** subset of the v1 ``hardware_config.yml → device.*`` — one field per physical
quantity, no register shadows. The schema is deliberately strict (``extra="forbid"``) so a
mistyped key in a device YAML fails loudly rather than being silently dropped; fields are added
here slice-by-slice as later ops need them (multiphoton, revival times, …).

**Gain (D23):** the calibration DB keeps lab-native **int codes** (as in v1). Conversion to the
v2 normalized **fraction** happens at the pack boundary via :func:`gain_int_to_frac` — the
compiler never sees an int code. ``maxv`` comes from the soccfg snapshot at factory time.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


def gain_int_to_frac(gain_int: float, maxv: float) -> float:
    """Convert a v1 lab-native gain **int code** to a v2 normalized **fraction** (D23).

    ``maxv`` is the generator's full-scale code from the soccfg snapshot (e.g. 32766). This is
    the single conversion that catches the v1→v2 gain-units bug: v1 code ``3289`` at
    ``maxv=32766`` is a fraction of ``≈0.1004``, not ``3289``.
    """
    return gain_int / maxv


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PulseCal(_Strict):
    """One calibrated single-qubit pulse (gain as a lab-native int code, D23)."""

    gain: float
    sigma: float
    length: float = 0.0
    type: str = "gauss"


class QubitCal(_Strict):
    """Transmon calibration: transition frequencies (MHz) + the named pulse library."""

    f_ge: float
    f_ef: float | None = None
    T1: float | None = None
    T2: float | None = None
    pulses: dict[str, PulseCal] = {}


class ReadoutCal(_Strict):
    """Dispersive-readout calibration (frequency MHz; gain as a lab-native int code)."""

    frequency: float
    gain: float
    readout_length: float
    trig_offset: float
    relax_delay: float
    phase: float | None = None
    threshold: float | None = None


class Calibration(_Strict):
    """The device instance's calibration values (calib VALUES layer, spec §5/D21)."""

    qubit: QubitCal
    readout: ReadoutCal
