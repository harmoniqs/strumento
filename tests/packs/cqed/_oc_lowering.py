"""Shared helpers for the optimal-control import tests (#210): the π ruler, lowering, propagation.

The import path under test is the production one end to end::

    Piccolo (u_x, u_y) in GHz on a ns grid
      → piccolo_controls_to_solution      (ν = 2000 (u_x − i u_y) MHz, µs grid)
      → from_solution(gain_calib=ruler, dac_rate=fs)   (MHz → gain fraction, resampled to the DAC)
      → a Pulse a TransmonOps override can install.

Pure: numpy/scipy only, no qick.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from strumento.core.pulses import Pulse, from_solution
from strumento.interop.piccolo import piccolo_controls_to_solution
from strumento.packs.cqed.calibration import PiPulseReference

LEVELS = 3
_a = np.diag(np.sqrt(np.arange(1, LEVELS)), 1).astype(complex)
_ad = _a.conj().T
_n = _ad @ _a


def drive_map(dev, freq_field: str = "f_ge") -> dict:
    return {"qubit": (("qubit", "drive"), getattr(dev.calib.qubit, freq_field))}


def dac_rate(dev) -> float:
    fs = dev.qubit._dac_rate(dev.calib.qubit.f_ge)
    assert fs is not None, "the device needs a soccfg snapshot with the drive gen's fs"
    return fs


def pi_ruler(dev) -> PiPulseReference:
    """The import ruler anchored on the calibrated π: Ω_peak/2π = 1 / (2 ∫Î dt).

    ``∫Î dt`` is the factory π's normalized envelope area — summed over its DAC samples when it is
    an ``Arb`` (DRAG on), else the Gaussian's area over the 4σ window the compiler declares.
    """
    from strumento.core.compiler import _GAUSS_LENGTH_SIGMAS

    pi = dev.qubit.ge_pi()
    fs = dac_rate(dev)
    env = pi.envelope
    if hasattr(env, "idata"):
        area_us = float(np.sum(env.idata)) / fs
    else:
        sigma = env.sigma_us
        n = int(round(_GAUSS_LENGTH_SIGMAS * sigma * fs))
        t = np.linspace(-_GAUSS_LENGTH_SIGMAS * sigma / 2, _GAUSS_LENGTH_SIGMAS * sigma / 2, n)
        area_us = float(np.sum(np.exp(-0.5 * (t / sigma) ** 2))) / fs
    return PiPulseReference(pi_gain_frac=pi.gain, pi_rabi_mhz=1.0 / (2.0 * area_us))


def lower(dev, times_ns, u_ghz, *, ruler=None, fs="device", freq_field: str = "f_ge"):
    """Piccolo controls → the imported ``Seq`` and its one drive ``Pulse``."""
    ruler = ruler if ruler is not None else pi_ruler(dev)
    rate = dac_rate(dev) if fs == "device" else fs
    seq = from_solution(piccolo_controls_to_solution(times_ns, u_ghz, drive="qubit"),
                        drive_map=drive_map(dev, freq_field), gain_calib=ruler, dac_rate=rate)
    (pulse,) = [op for op in seq.ops[0].pulses if isinstance(op, Pulse)]
    return seq, pulse


def played_nu_mhz(pulse, ruler) -> np.ndarray:
    """What the qubit sees from a pulse: ν = Ω/2π in MHz (complex I + iQ), via the π ruler."""
    i = np.asarray(pulse.envelope.idata, dtype=float)
    q = (np.asarray(pulse.envelope.qdata, dtype=float) if pulse.envelope.qdata is not None
         else np.zeros_like(i))
    return (i + 1j * q) * pulse.gain / ruler.pi_gain_frac * ruler.pi_rabi_mhz


def unitary(nu_mhz, fs, delta_mhz) -> np.ndarray:
    """Piecewise-constant 3-level propagator, H = (δ/2) n(n−1) + ½(ν a† + ν* a), MHz and µs."""
    dt_us = 1.0 / fs
    h_anharm = (delta_mhz / 2.0) * (_n @ _n - _n)
    u = np.eye(LEVELS, dtype=complex)
    for nu in nu_mhz:
        u = expm(-2j * np.pi * (h_anharm + 0.5 * (nu * _ad + np.conj(nu) * _a)) * dt_us) @ u
    return u


#: Hand-authored Gaussian (Piccolo units): duration, knots, width.
T_NS, N_KNOTS, SIGMA_NS = 140.0, 71, 35.0
_CYCLES = {"pi": 0.5, "hpi": 0.25}


def analytic_controls(gate: str):
    """A hand-authored Gaussian π or π/2 as Piccolo ``(2, N)`` GHz controls on a ns grid.

    Its area is set from first principles (ν_peak · ∫ĝ dt = cycles), so no solver is involved.
    Returns ``(times_ns, u, nu_peak_mhz)``; ``u_y = 0`` (a real pulse).
    """
    t = np.linspace(0.0, T_NS, N_KNOTS)
    g = np.exp(-((t - T_NS / 2) ** 2) / (2 * SIGMA_NS**2))
    g -= g.min()                                     # floored ends: starts and stops at zero
    g /= g.max()
    dt_us = (t[1] - t[0]) * 1e-3
    nu_peak = _CYCLES[gate] / (float(np.sum(g)) * dt_us)   # = trapezoid: ends are zero
    from strumento.interop.piccolo import PICCOLO_GHZ_TO_RABI_MHZ

    u = np.vstack([nu_peak * g / PICCOLO_GHZ_TO_RABI_MHZ, np.zeros_like(g)])
    return t, u, nu_peak
