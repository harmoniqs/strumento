"""Core fit primitives (spec §9.3, D4) — ``lmfit``-backed, behind a typed result seam.

Slice 2 ships ``fit_rabi`` (amplitude Rabi = a decaying cosine of drive amplitude). Slice 3 adds
the time-domain damped-sinusoid family — T1 (``fit_exp``), T2 Ramsey (``fit_decaysin_detuned``),
and damped Rabi (``fit_damped_rabi``) — which all share a single damped-oscillation model and
differ only in the physical quantities their ``derived`` fields expose (Dave Schuster's
"one fit, different post-processing" principle, 2026-07-21). Resonator-spectroscopy fitters
(Lorentzian / circle / Fano) live alongside in :mod:`strumento.core.resonator_fitters`.

Each returns a typed :class:`FitResult` (params + stderr + derived physical quantities) — never
stdout or a figure; plotting/printing is the Display layer's job (D13).

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


def _params_stderr(out) -> tuple[dict[str, float], dict[str, float]]:
    """Pull ``lmfit`` fit output into plain ``{name: value}`` params + stderr dicts."""
    params = {k: float(v.value) for k, v in out.params.items()}
    stderr = {
        k: (float(v.stderr) if v.stderr is not None else float("nan"))
        for k, v in out.params.items()
    }
    return params, stderr


def _seed_freq(x, y) -> float:
    """FFT frequency seed (cycles per x-unit), ignoring the DC bin — robust vs local minima."""
    dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
    power = np.abs(np.fft.rfft(y - y.mean()))
    freqs = np.fft.rfftfreq(x.size, d=dx)
    if power.size > 1:
        power[0] = 0.0
    f0 = freqs[int(np.argmax(power))]
    span = float(x.max() - x.min()) or 1.0
    return float(f0) if f0 > 0 else 1.0 / span


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


# --------------------------------------------------------------------------- #
# Oscillator / relaxation family (slice 3) — exponential + damped sinusoid.
# One fit, different post-processing (D4 / Dave Schuster 2026-07-21).
# --------------------------------------------------------------------------- #


def _exp_decay(x, y0, amp, tau):
    """y0 + amp·exp(-x/tau).  amp>0 decays down to y0; amp<0 rises up to y0."""
    return y0 + amp * np.exp(-x / tau)


def fit_exp(xpts, y) -> FitResult:
    """Fit an exponential relaxation (T1) trace to ``y0 + amp·exp(-x/tau)``.

    Handles both decaying (``amp>0``) and rising (``amp<0``) traces via a signed amplitude seed.
    ``derived["T1"]`` is the decay constant ``tau`` (same x-units as ``xpts``).
    """
    x = np.asarray(xpts, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(x.max() - x.min()) or 1.0

    model = lmfit.Model(_exp_decay)
    pars = model.make_params(
        y0=float(y[-1]),
        amp=float(y[0] - y[-1]),
        tau=span / 3.0,
    )
    pars["tau"].set(min=1e-12)

    out = model.fit(y, pars, x=x)
    params, stderr = _params_stderr(out)
    tau = abs(params["tau"])
    derived = {"T1": tau, "tau": tau}
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))


def _damped_sin(x, offset, amp, freq, phase, tau):
    """offset + amp·exp(-x/tau)·cos(2π·freq·x + phase).  ``freq`` in cycles per x-unit."""
    return offset + amp * np.exp(-x / tau) * np.cos(2 * np.pi * freq * x + phase)


def _fit_damped_sin(x, y) -> tuple[dict[str, float], dict[str, float], bool]:
    """Shared damped-cosine fit used by T2 Ramsey + damped Rabi. Returns (params, stderr, success).

    ``freq`` is fit in cycles per x-unit (e.g. MHz for x in µs). The frequency is FFT-seeded and
    the fit is retried over the phase quadrants for robustness against local minima.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(x.max() - x.min()) or 1.0
    freq0 = _seed_freq(x, y)
    amp0 = (float(y.max() - y.min()) / 2) or 1.0

    model = lmfit.Model(_damped_sin)
    best = None
    for phase0 in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        pars = model.make_params(
            offset=float(y.mean()), amp=amp0, freq=freq0, phase=phase0, tau=span
        )
        pars["freq"].set(min=0.0)
        pars["tau"].set(min=1e-12)
        try:
            out = model.fit(y, pars, x=x)
        except Exception:
            continue
        if best is None or out.chisqr < best.chisqr:
            best = out

    if best is None:  # pragma: no cover - defensive
        pars = model.make_params(offset=float(y.mean()), amp=amp0, freq=freq0, phase=0.0, tau=span)
        best = model.fit(y, pars, x=x)

    params, stderr = _params_stderr(best)
    return params, stderr, bool(best.success)


def fit_decaysin_detuned(xpts, y) -> FitResult:
    """Fit a T2 Ramsey fringe to a detuned decaying cosine and derive the detuning + T2.

    Model: ``offset + amp·exp(-x/tau)·cos(2π·freq·x + phase)``. For a Ramsey experiment the fit
    ``freq`` is the qubit–drive **detuning** and ``tau`` is **T2\\***.
    ``derived``: ``detuning``, ``freq`` (= detuning), ``T2`` (= tau).
    """
    params, stderr, success = _fit_damped_sin(xpts, y)
    freq = abs(params["freq"])
    tau = abs(params["tau"])
    derived = {"detuning": freq, "freq": freq, "T2": tau, "tau": tau}
    return FitResult(params=params, stderr=stderr, derived=derived, success=success)


def fit_damped_rabi(xpts, y) -> FitResult:
    """Fit a time-domain damped Rabi oscillation and derive the Rabi frequency + π time.

    Model: ``offset + amp·exp(-x/tau)·cos(2π·freq·x + phase)``. ``freq`` is the Rabi frequency;
    a π rotation is half a period. ``derived``: ``rabi_freq``, ``pi_time`` (= 1/(2·freq)),
    ``half_pi_time`` (= 1/(4·freq)), ``tau``.
    """
    params, stderr, success = _fit_damped_sin(xpts, y)
    freq = abs(params["freq"])
    tau = abs(params["tau"])
    period = (1.0 / freq) if freq > 0 else float("inf")
    derived = {
        "rabi_freq": freq,
        "period": period,
        "pi_time": period / 2.0,
        "half_pi_time": period / 4.0,
        "tau": tau,
    }
    return FitResult(params=params, stderr=stderr, derived=derived, success=success)
