"""Core fit primitives (D4) — ``lmfit``-backed, behind a typed result seam.

``fit_rabi`` fits amplitude Rabi (a decaying cosine of drive amplitude). Alongside it sits
the time-domain damped-sinusoid family — T1 (``fit_exp``), T2 Ramsey (``fit_decaysin_detuned``),
and damped Rabi (``fit_damped_rabi``) — which all share a single damped-oscillation model and
differ only in the physical quantities their ``derived`` fields expose ("one fit, different
post-processing"). Resonator-spectroscopy fitters
(Lorentzian / circle / Fano) live alongside in :mod:`strumento.core.resonator_fitters`.
``fit_time_of_flight`` (issue #104) adds an edge-detection fitter for a decimated raw-trace
magnitude — where the signal actually clears the noise floor, not a resonance or an oscillation.

Each returns a typed :class:`FitResult` (params + stderr + derived physical quantities) — never
stdout or a figure; plotting/printing is the Display layer's job (D13).

Every derived key comes with its 1σ error as ``derived["<name>_stderr"]`` (issue #137, the
analyzers' convention): linearly propagated from the fit covariance by :func:`_derived_stderr`,
mirrored onto aliases (``tau`` / ``T1``), and ``nan`` — present, never zero — when lmfit gives
no error. None of the fitters in this module has an exempt key.

This module is **pure**: no ``qick`` import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import lmfit
import numpy as np
from scipy.special import erf, gammaln


class _JsonValueSchema:
    """Pydantic marker: the value is any JSON value (number, string, boolean, null, list, object).

    The published ``fit`` contract (D35) is generated from :class:`FitResult`'s annotations; this
    marker makes ``derived``'s values a recursive JSON-value schema there (D38), while the type
    checker sees ``Any`` — so a numeric read like ``fit.derived["T1"] * 2`` needs no cast. The
    schema is built from ``pydantic_core`` primitives because a plain recursive alias does not
    resolve on every Pydantic 2 release (``TypeAliasType`` / ``pydantic.JsonValue`` need 2.5+).
    Pydantic is imported only when a schema is generated, so this module's import cost is unchanged.
    """

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema as cs

        ref = "JsonValue"
        item = cs.definition_reference_schema(ref)
        return cs.nullable_schema(
            cs.union_schema([
                cs.bool_schema(),
                cs.int_schema(),
                cs.float_schema(),
                cs.str_schema(),
                cs.list_schema(item),
                cs.dict_schema(cs.str_schema(), item),
            ]),
            ref=ref,
        )


#: A ``derived`` value: any JSON value (D38). Typed ``Any`` for readers; a JSON-value schema on
#: the wire — numbers stay numbers, and a consumer that needs a number checks its key.
DerivedValue = Annotated[Any, _JsonValueSchema()]


@dataclass
class FitResult:
    """A fit's outcome: raw ``params``, their ``stderr``, and ``derived`` quantities.

    ``derived`` holds the physical quantities (numbers, each with its ``<name>_stderr``) and may
    also carry JSON values an analyzer reports alongside them — a protocol name, a confusion
    matrix, the swept points, a refusal's provenance (D38).
    """

    params: dict[str, float]
    stderr: dict[str, float]
    derived: dict[str, DerivedValue]
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


def _sgn(v: float) -> float:
    """``d|v|/dv`` — the sign, with ``+1`` at zero so a single-parameter error never vanishes."""
    return -1.0 if v < 0 else 1.0


def _derived_stderr(out, grads: dict[str, dict[str, float]]) -> dict[str, float]:
    """Linearly propagate the fit covariance onto derived quantities (issue #137).

    ``grads`` maps each derived name to its gradient ``{param: ∂derived/∂param}`` evaluated at the
    fitted values; the result is ``{"<name>_stderr": sqrt(gᵀ·C·g)}`` over lmfit's covariance
    ``C`` (``out.covar``, ordered as ``out.var_names``), so a quantity built from several
    correlated parameters gets their correlation, not a naive quadrature sum. A parameter absent
    from ``var_names`` (held fixed) contributes nothing. No covariance — lmfit could not estimate
    errors — or a non-finite gradient (e.g. ``1/freq`` at ``freq = 0``) gives ``nan``: the key is
    always present, and never a fake zero.
    """
    names = list(getattr(out, "var_names", None) or [])
    covar = getattr(out, "covar", None)
    result: dict[str, float] = {}
    for name, grad in grads.items():
        sigma = float("nan")
        if covar is not None and names:
            g = np.array([float(grad.get(p, 0.0)) for p in names])
            if np.all(np.isfinite(g)):
                var = float(g @ np.asarray(covar, dtype=float) @ g)
                if np.isfinite(var):
                    sigma = float(np.sqrt(max(var, 0.0)))
        result[f"{name}_stderr"] = sigma
    return result


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
    amplitude for a π rotation). ``derived``: ``period``, ``pi_gain``, each with ``<name>_stderr``
    (``pi_gain_stderr = stderr["period"] / 2``).
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
    s = _sgn(params["period"])
    derived |= _derived_stderr(out, {"period": {"period": s}, "pi_gain": {"period": s / 2.0}})
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))


# --------------------------------------------------------------------------- #
# Oscillator / relaxation family — exponential + damped sinusoid.
# One fit, different post-processing: a shared damped-sinusoid primitive + typed wrappers.
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
    g_tau = {"tau": _sgn(params["tau"])}
    derived |= _derived_stderr(out, {"T1": g_tau, "tau": g_tau})
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))


def _damped_sin(x, offset, amp, freq, phase, tau):
    """offset + amp·exp(-x/tau)·cos(2π·freq·x + phase).  ``freq`` in cycles per x-unit."""
    return offset + amp * np.exp(-x / tau) * np.cos(2 * np.pi * freq * x + phase)


def _fit_damped_sin(x, y) -> lmfit.model.ModelResult:
    """Shared damped-cosine fit used by T2 Ramsey + damped Rabi. Returns the best lmfit result
    (the wrappers read params/stderr off it and propagate its covariance onto ``derived``).

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

    return best


def fit_decaysin_detuned(xpts, y) -> FitResult:
    """Fit a T2 Ramsey fringe to a detuned decaying cosine and derive the detuning + T2.

    Model: ``offset + amp·exp(-x/tau)·cos(2π·freq·x + phase)``. For a Ramsey experiment the fit
    ``freq`` is the qubit–drive **detuning** and ``tau`` is **T2\\***.
    ``derived``: ``detuning``, ``freq`` (= detuning), ``T2`` (= tau).
    """
    out = _fit_damped_sin(xpts, y)
    params, stderr = _params_stderr(out)
    freq = abs(params["freq"])
    tau = abs(params["tau"])
    derived = {"detuning": freq, "freq": freq, "T2": tau, "tau": tau}
    g_freq = {"freq": _sgn(params["freq"])}
    g_tau = {"tau": _sgn(params["tau"])}
    derived |= _derived_stderr(
        out, {"detuning": g_freq, "freq": g_freq, "T2": g_tau, "tau": g_tau}
    )
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))


def fit_damped_rabi(xpts, y) -> FitResult:
    """Fit a time-domain damped Rabi oscillation and derive the Rabi frequency + π time.

    Model: ``offset + amp·exp(-x/tau)·cos(2π·freq·x + phase)``. ``freq`` is the Rabi frequency;
    a π rotation is half a period. ``derived``: ``rabi_freq``, ``pi_time`` (= 1/(2·freq)),
    ``half_pi_time`` (= 1/(4·freq)), ``tau``.
    """
    out = _fit_damped_sin(xpts, y)
    params, stderr = _params_stderr(out)
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
    s_f = _sgn(params["freq"])
    # d(1/|f|)/df = -sgn(f)/f²; at f = 0 the period is inf and the gradient non-finite → nan
    d_period = -s_f / freq**2 if freq > 0 else float("inf")
    derived |= _derived_stderr(out, {
        "rabi_freq": {"freq": s_f},
        "period": {"freq": d_period},
        "pi_time": {"freq": d_period / 2.0},
        "half_pi_time": {"freq": d_period / 4.0},
        "tau": {"tau": _sgn(params["tau"])},
    })
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))


def _cosine(x, offset, amp, freq, phase):
    return offset + amp * np.cos(2 * np.pi * freq * x + phase)


def fit_parity(xpts, y) -> FitResult:
    """Fit a parity fringe (phase scan of the second half-pi, degrees) to a plain cosine.

    ``derived["contrast"]`` is the peak-to-peak fringe amplitude ``2·|amp|``;
    ``derived["phase_deg"]`` is the fringe phase offset in degrees.
    """
    x = np.asarray(xpts, dtype=float)
    y = np.asarray(y, dtype=float)

    model = lmfit.Model(_cosine)
    span = float(x.max() - x.min()) or 1.0
    pars = model.make_params(
        offset=float(np.mean(y)),
        amp=float((y.max() - y.min()) / 2.0),
        freq=1.0 / span,   # one fringe over the scanned span (360° scan -> 1/360 per degree)
        phase=0.0,
    )
    pars["freq"].set(min=0.1 / span, max=10.0 / span)

    out = model.fit(y, pars, x=x)
    params, stderr = _params_stderr(out)
    derived = {
        "contrast": 2.0 * abs(params["amp"]),
        "phase_deg": float(np.degrees(params["phase"])),
    }
    derived |= _derived_stderr(out, {
        "contrast": {"amp": 2.0 * _sgn(params["amp"])},
        "phase_deg": {"phase": 180.0 / np.pi},
    })
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))


def _gaussian_wigner(x, offset, amp, alpha_per_gain):
    """offset + amp·exp(-2·(alpha_per_gain·x)²) — displaced-vacuum parity (Wigner) vs gain."""
    return offset + amp * np.exp(-2.0 * (alpha_per_gain * x) ** 2)


def fit_gaussian_wigner(xpts, y) -> FitResult:
    """Fit displaced-vacuum parity vs displacement gain to the vacuum Wigner Gaussian.

    Model: ``offset + amp·exp(-2·(k·g)²)``. The parity of a coherent state ``D(α)|0⟩`` is
    ``⟨Π⟩ = exp(-2|α|²)`` and ``α = k·g`` is linear in the DAC gain, so a displace→parity gain
    sweep is a **Gaussian in gain, not an oscillation** — the vacuum Wigner function's closed form
    (``W_vacuum(α) = (2/π)·exp(-2|α|²)``), and the correct estimator for the
    gain→α ruler. ``derived["alpha_per_gain"]`` is ``k``; ``derived["contrast"]`` is ``2·|amp|``.
    """
    x = np.asarray(xpts, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(np.max(np.abs(x))) or 1.0

    # parity peaks at |g| = 0 and decays toward a floor at large |g|; seed a signed amplitude so
    # either readout-projection sign works, and a width that puts |α| ~ 1 around mid-sweep.
    i0 = int(np.argmin(np.abs(x)))
    far = np.abs(x) > 0.6 * span
    floor = float(np.median(y[far])) if np.any(far) else float(np.median(y))
    amp0 = (float(y[i0]) - floor) or 1.0
    k0 = 1.0 / (0.5 * span)

    model = lmfit.Model(_gaussian_wigner)
    pars = model.make_params(offset=floor, amp=amp0, alpha_per_gain=k0)
    pars["alpha_per_gain"].set(min=0.0)

    out = model.fit(y, pars, x=x)
    params, stderr = _params_stderr(out)
    k = abs(params["alpha_per_gain"])
    derived = {"alpha_per_gain": k, "contrast": 2.0 * abs(params["amp"])}
    derived |= _derived_stderr(out, {
        "alpha_per_gain": {"alpha_per_gain": _sgn(params["alpha_per_gain"])},
        "contrast": {"amp": 2.0 * _sgn(params["amp"])},
    })
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))


def _poisson_weights(nbar: float, nmax: int) -> np.ndarray:
    """Normalised Poisson weights ``P(n; nbar)`` for ``n = 0..nmax``."""
    n = np.arange(nmax + 1)
    logw = n * np.log(max(float(nbar), 1e-9)) - float(nbar) - gammaln(n + 1.0)
    w = np.exp(logw)
    s = float(w.sum())
    return w / s if s > 0 else w


def _number_comb_period_seed(f, y, f_ge, chi_lo, chi_hi, *, nmax: int = 6, ngrid: int = 200):
    """Estimate the comb's peak spacing straight from the data — a belief-free seed.

    The comb-fit objective is riddled with spacing-alias local minima, so a single ``lmfit``
    start converges only inside whichever basin its seed lands in; a stored ``chi`` far from the
    truth therefore mis-recovers (e.g. a believed 0.28 MHz against a true 1.5 MHz). This folds the
    baseline-subtracted spectrum against a Poisson-comb *template* over a fine grid of candidate
    spacings and returns the one whose template best correlates with the data. ``abs`` on the
    correlation makes it blind to a readout projection that inverts the peaks (period, not sign,
    is what the seed carries). Returns ``None`` for a flat spectrum (nothing to lock onto).
    """
    y0 = np.asarray(y, dtype=float) - float(np.median(y))
    ny = float(np.linalg.norm(y0))
    if ny == 0.0:
        return None
    span = float(f.max() - f.min()) or 1.0
    best: tuple[float, float] | None = None
    for chi in np.linspace(chi_lo, chi_hi, ngrid):
        gamma = max(chi / 3.0, span / 100.0)
        template = np.zeros_like(f, dtype=float)
        for n in range(nmax + 1):
            template = template + 1.0 / (1.0 + ((f - (f_ge - n * chi)) / gamma) ** 2)
        t0 = template - template.mean()
        nt = float(np.linalg.norm(t0))
        if nt == 0.0:
            continue
        score = abs(float(np.dot(y0, t0) / (ny * nt)))
        if best is None or score > best[0]:
            best = (score, float(chi))
    return best[1] if best is not None else None


def fit_number_splitting_comb(freqs, signal, f_ge: float, *, nmax: int = 6,
                              chi_seed: float = 1.0) -> FitResult:
    """Fit a photon-number-split qubit spectrum to a Poisson-weighted Lorentzian comb.

    A coherent state displaced into the cavity populates Fock states with Poisson weights
    ``P(n; n̄)``; each shifts the qubit by ``n·χ``, so the spectrum is a **comb** of peaks at
    ``f_ge - n·χ`` (χ > 0 for the usual below-``f_ge`` shift), spaced by χ. The dispersive shift is
    the **spacing**, which a single-Lorentzian fit cannot give — this fits the whole comb:

    ``offset + amp · Σ_n P(n; n̄) · L(f; f_ge - n·χ, γ)``

    with free ``offset, amp, chi, nbar, gamma`` at fixed ``f_ge`` (the known qubit line).
    ``derived``: ``chi`` (spacing magnitude, MHz), ``chi_estimate`` (= ``-chi``, signed for the
    below-``f_ge`` comb), ``nbar`` (fitted mean photon number), ``gamma`` (per-peak HWHM).

    **Multi-start.** The comb objective has spacing-alias local minima, so a single start
    trapped by a stored ``chi_seed`` far from the truth mis-recovers (a believed 0.28 MHz seed
    pins the fit at 0.28 when the true spacing is 1.5 MHz).
    The fit runs from a **data-driven spacing seed** (``_number_comb_period_seed``) as well as
    the caller's belief and keeps the lower-``chisqr`` result, so recovery no longer depends on
    the stored belief being close.
    """
    f = np.asarray(freqs, dtype=float)
    y = np.asarray(signal, dtype=float)
    span = float(f.max() - f.min()) or 1.0
    f_ge = float(f_ge)
    dx = span / max(len(f) - 1, 1)

    # peaks sit above a baseline; seed offset at the continuum and a positive amplitude, orienting
    # it toward whichever extremum is farther from the baseline (readout projection may invert).
    baseline = float(np.median(y))
    amp0 = float(np.max(y) - np.min(y)) or 1.0
    if abs(float(np.min(y)) - baseline) > abs(float(np.max(y)) - baseline):
        amp0 = -amp0

    # A spacing must be at least ~2 samples wide to resolve, and no wider than the span below
    # f_ge (else the n=1 tooth at f_ge - chi falls outside the scanned window). The lmfit bound
    # stays the full span; the data-driven pre-scan only searches where teeth can actually sit.
    chi_lo = max(2.0 * dx, span * 1e-3)
    chi_scan_hi = max(f_ge - float(f.min()), 3.0 * chi_lo)

    def _comb(f, offset, amp, chi, nbar, gamma):
        w = _poisson_weights(nbar, nmax)
        total = np.zeros_like(f, dtype=float)
        for n in range(nmax + 1):
            total = total + w[n] / (1.0 + ((f - (f_ge - n * chi)) / gamma) ** 2)
        return offset + amp * total

    model = lmfit.Model(_comb)

    def _fit_from(chi0: float):
        chi0 = float(min(max(abs(chi0), chi_lo), span))
        pars = model.make_params(offset=baseline, amp=amp0, chi=chi0, nbar=2.0, gamma=chi0 / 3.0)
        pars["chi"].set(min=chi_lo, max=span)
        pars["nbar"].set(min=1e-3, max=float(nmax))
        pars["gamma"].set(min=span * 1e-4, max=span)
        try:
            return model.fit(y, pars, f=f)
        except Exception:
            return None

    seeds: list[float] = []
    data_seed = _number_comb_period_seed(f, y, f_ge, chi_lo, chi_scan_hi, nmax=nmax)
    if data_seed:
        seeds.append(data_seed)
    if chi_seed:
        seeds.append(abs(float(chi_seed)))
    if not seeds:
        seeds.append(max(chi_lo, span / 10.0))

    out = None
    for seed in seeds:
        candidate = _fit_from(seed)
        if candidate is None:
            continue
        if out is None or candidate.chisqr < out.chisqr:
            out = candidate
    if out is None:  # every start raised — fall back to a single span-scaled start
        out = _fit_from(max(chi_lo, span / 10.0))

    params, stderr = _params_stderr(out)
    chi = abs(params["chi"])
    derived = {
        "chi": chi,
        "chi_estimate": -chi,
        "nbar": abs(params["nbar"]),
        "gamma": abs(params["gamma"]),
    }
    s_chi = _sgn(params["chi"])
    derived |= _derived_stderr(out, {
        "chi": {"chi": s_chi},
        "chi_estimate": {"chi": -s_chi},
        "nbar": {"nbar": _sgn(params["nbar"])},
        "gamma": {"gamma": _sgn(params["gamma"])},
    })
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))


def _double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    g1 = a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
    g2 = a2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))
    return g1 + g2


def fit_double_gaussian(xpts, y) -> FitResult:
    """Fit a bimodal histogram to two Gaussians (single-shot readout discrimination).

    ``derived``: ``threshold`` (the midpoint between the two means — the discrimination
    boundary), ``separation`` (|Δμ| / mean σ — the readout SNR figure), and the two means.
    """
    x = np.asarray(xpts, dtype=float)
    y = np.asarray(y, dtype=float)

    # seed the two peaks from the histogram halves around the global mean
    mid = float(np.average(x, weights=np.maximum(y, 1e-12)))
    lo, hi = x[x <= mid], x[x > mid]
    ylo, yhi = y[x <= mid], y[x > mid]
    mu1 = float(np.average(lo, weights=np.maximum(ylo, 1e-12))) if lo.size else float(x[0])
    mu2 = float(np.average(hi, weights=np.maximum(yhi, 1e-12))) if hi.size else float(x[-1])
    span = float(x.max() - x.min()) or 1.0

    model = lmfit.Model(_double_gaussian)
    pars = model.make_params(
        a1=float(y.max()), mu1=mu1, sigma1=span / 10.0,
        a2=float(y.max()), mu2=mu2, sigma2=span / 10.0,
    )
    for p in ("sigma1", "sigma2"):
        pars[p].set(min=span / 1e4)
    for p in ("a1", "a2"):
        pars[p].set(min=0.0)

    out = model.fit(y, pars, x=x)
    params, stderr = _params_stderr(out)
    mu_a, mu_b = sorted((params["mu1"], params["mu2"]))
    sig = (abs(params["sigma1"]) + abs(params["sigma2"])) / 2.0
    derived = {
        "threshold": (mu_a + mu_b) / 2.0,
        "separation": abs(mu_b - mu_a) / sig if sig else float("inf"),
        "mu_low": mu_a,
        "mu_high": mu_b,
    }
    # the low/high mean's parameter name (lo/hi above are the histogram halves)
    lo_key, hi_key = ("mu1", "mu2") if params["mu1"] <= params["mu2"] else ("mu2", "mu1")
    sep_grad = {"mu1": float("inf")}  # sig = 0 → separation is inf, its error nan
    if sig:
        d_sig = -abs(mu_b - mu_a) / sig**2 / 2.0  # ∂sep/∂|sigma_i| (sig is the mean of the two)
        sep_grad = {
            hi_key: 1.0 / sig,
            lo_key: -1.0 / sig,
            "sigma1": d_sig * _sgn(params["sigma1"]),
            "sigma2": d_sig * _sgn(params["sigma2"]),
        }
    derived |= _derived_stderr(out, {
        "threshold": {"mu1": 0.5, "mu2": 0.5},
        "separation": sep_grad,
        "mu_low": {lo_key: 1.0},
        "mu_high": {hi_key: 1.0},
    })
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))


# --------------------------------------------------------------------------- #
# Time-of-flight (arrival-edge step) — issue #104.
# --------------------------------------------------------------------------- #


def _tof_step(x, offset, amp, t0, sigma):
    """An erf-smoothed step: flat at ``offset`` before ``t0``, flat at ``offset+amp`` after,
    with a transition width ``sigma``. The arrival edge, not a resonance or an oscillation."""
    return offset + amp * 0.5 * (1.0 + erf((x - t0) / (np.sqrt(2.0) * sigma)))


def fit_time_of_flight(xpts, y) -> FitResult:
    """Fit a decimated raw-trace magnitude to a single arrival edge (time-of-flight bring-up).

    Model: ``offset + amp·½(1 + erf((x-t0)/(√2·sigma)))`` — flat at the noise floor before the
    reflected probe arrives, flat at the settled level once it has, with ``t0`` the point where
    the signal actually clears the noise floor (the 50% rise point) and ``sigma`` the transition
    width. Seeded from the data itself: ``offset``/the settled level from the first/last ~10% of
    samples (assumed to bracket "before" and "after"), ``t0`` from the first sample past their
    midpoint, ``sigma`` from a few sample widths — robust against where in the window the edge
    actually falls, not tied to one assumed location.

    ``derived["tof_us"]`` is the headline quantity (named for the µs convention every other
    time-domain fitter here uses; the fit itself is unit-agnostic). Never raises on a flat/noisy
    trace (e.g. an unarmed mock's all-zero decimated data) — there is simply nothing to find, and
    the caller's shape/wiring plumbing must survive that as gracefully as any other fitter here.
    """
    x = np.asarray(xpts, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(x.max() - x.min()) or 1.0
    dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0

    n_edge = max(3, x.size // 10)
    baseline = float(np.mean(y[:n_edge]))
    settled = float(np.mean(y[-n_edge:]))
    amp0 = settled - baseline

    midpoint = baseline + 0.5 * amp0
    crossings = np.where(y >= midpoint)[0] if amp0 >= 0 else np.where(y <= midpoint)[0]
    t0_seed = float(x[crossings[0]]) if crossings.size else float(x[x.size // 2])

    model = lmfit.Model(_tof_step)
    pars = model.make_params(
        offset=baseline, amp=amp0 or 1.0, t0=t0_seed, sigma=max(dx, span / 50.0)
    )
    pars["sigma"].set(min=dx / 4.0)
    pars["t0"].set(min=float(x.min()), max=float(x.max()))

    out = model.fit(y, pars, x=x)
    params, stderr = _params_stderr(out)
    t0 = params["t0"]
    derived = {
        "tof_us": t0,
        "t0": t0,
        "rise_us": abs(params["sigma"]),
        "noise_floor": params["offset"],
        "settled_level": params["offset"] + params["amp"],
    }
    derived |= _derived_stderr(out, {
        "tof_us": {"t0": 1.0},
        "t0": {"t0": 1.0},
        "rise_us": {"sigma": _sgn(params["sigma"])},
        "noise_floor": {"offset": 1.0},
        "settled_level": {"offset": 1.0, "amp": 1.0},
    })
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))
