"""Shared gain plumbing for cqed ops classes (D23).

Every factory converts lab-native int codes to v2 fractions at this boundary, using the
generator's ``maxv`` from the device's soccfg snapshot (resolved through the wiring so
multi-generator lines pick the right one).
"""

from __future__ import annotations

from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import gain_int_to_frac

_DEFAULT_MAXV = 2**15 - 1  # fallback when no soccfg snapshot is attached


def maxv_for(device, ref: LineRef, freq_mhz: float) -> float:
    """Full-scale gain code for the generator behind ``ref`` at ``freq_mhz``."""
    gen = device.wiring.resolve(ref, freq_mhz)
    snap = device.soccfg_snapshot
    if not snap:
        return _DEFAULT_MAXV
    gens = snap.get("gens", [])
    for g in gens:
        if g.get("tproc_ch") == gen.dac_ch:
            return g["maxv"]
    return gens[0]["maxv"] if gens else _DEFAULT_MAXV


def gain_out(calib_int_code: float, override, maxv: float):
    """The D23 three-regime rule: None -> calib int code to fraction; scalar -> raw fraction
    used as-is; Sweep -> lab-native int-code sweep mapped to a fraction sweep. The inverse map
    rides along so the display axis can come back to lab-native codes (D23/D37) — legacy sweeps
    pin ``_lab_axis``, bound sweeps (D36) reconstruct from the declared Loop via the inverse."""
    if override is None:
        return gain_int_to_frac(calib_int_code, maxv)
    if isinstance(override, Sweep):
        return override.mapped(
            lambda code: gain_int_to_frac(code, maxv),
            inverse=lambda frac: frac * maxv,
        )
    return override
