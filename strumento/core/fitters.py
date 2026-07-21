"""Core fit primitives (spec §9.3, D4) — ``lmfit``-backed, behind a typed result seam.

Slice 2 ships ``fit_rabi`` (amplitude Rabi = a decaying cosine of drive amplitude). Later slices
add T1 (``fit_exp``), T2 Ramsey (``fit_decaysin_detuned``), and parity fitters. Each returns a
typed :class:`FitResult` (params + stderr + derived physical quantities) — never stdout or a
figure; plotting/printing is the Display layer's job (D13).

This module is **pure**: no ``qick`` import.
"""

from __future__ import annotations

from dataclasses import dataclass

import lmfit
import numpy as np


@dataclass
class FitResult:
    """A fit's outcome: raw ``params``, their ``stderr``, and ``derived`` physical quantities."""

    params: dict[str, float]
    stderr: dict[str, float]
    derived: dict[str, float]
    success: bool

    def __getitem__(self, key: str) -> float:
        return self.params[key]


def _decaying_cos(x, offset, amp, period, phase, tau):
    return offset + amp * np.exp(-x / tau) * np.cos(2 * np.pi * x / period + phase)


def fit_rabi(xpts, y) -> FitResult:
    """Fit an amplitude-Rabi trace to a decaying cosine and derive the period + π gain.

    Model: ``offset + amp·exp(-x/tau)·cos(2π x/period + phase)``. The period is seeded from the
    dominant FFT component (robust against local minima); the π gain is half the period (the drive
    amplitude for a π rotation).
    """
    x = np.asarray(xpts, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
    span = float(x.max() - x.min()) or 1.0

    # FFT-based period seed (ignore the DC bin)
    power = np.abs(np.fft.rfft(y - y.mean()))
    freqs = np.fft.rfftfreq(x.size, d=dx)
    if power.size > 1:
        power[0] = 0.0
    f0 = freqs[int(np.argmax(power))]
    period0 = (1.0 / f0) if f0 > 0 else span

    model = lmfit.Model(_decaying_cos)
    pars = model.make_params(
        offset=float(y.mean()),
        amp=(float(y.max() - y.min()) / 2) or 1.0,
        period=period0,
        phase=0.0,
        tau=10 * span,
    )
    pars["period"].set(min=2 * dx)
    pars["tau"].set(min=1e-9)

    out = model.fit(y, pars, x=x)

    params = {k: float(v.value) for k, v in out.params.items()}
    stderr = {
        k: (float(v.stderr) if v.stderr is not None else float("nan"))
        for k, v in out.params.items()
    }
    period = abs(params["period"])
    derived = {"period": period, "pi_gain": period / 2.0}
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))
