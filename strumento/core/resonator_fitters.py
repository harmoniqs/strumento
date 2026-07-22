"""Resonator / oscillator-spectroscopy fitters (design-review requirements).

The subtle, real-world half of the fitting surface: extracting resonator quality factors from
spectroscopy data with non-idealities — environmental gain/rotation, cable delay, impedance
mismatch (asymmetry), and IQ-plane offset. Three fitters:

- :func:`fit_lorentzian`       — magnitude Lorentzian → ``f0``, ``fwhm``, ``Q_loaded`` (seed-free)
- :func:`fit_resonator_circle` — the **Probst/Khalil** circle fit of complex S21 → ``Q_loaded``,
  ``Q_internal``, ``Q_coupling``, ``f0``, with cable-delay + env-gain/rotation + asymmetry handling,
  for both **notch** (hanger / S21 transmission) and **reflection** (S11) geometries
- :func:`fit_fano`             — asymmetric Fano ("Spano") lineshape for skewed resonances

**Vendored, not depended on** (deliberately — resonator fits change rarely; a dependency is a
downstream-breakage risk for little gain). Methods are re-implemented in-repo from the published
descriptions:

- Circle fit: S. Probst *et al.*, "Efficient and robust analysis of complex scattering data under
  noise in microwave resonators", *Rev. Sci. Instrum.* **86**, 024706 (2015); with the
  diameter-correction / impedance-mismatch angle of M. S. Khalil *et al.*, *J. Appl. Phys.*
  **111**, 054510 (2012). Algebraic (Kåsa) circle fit + phase-vs-frequency tangent fit.
- Lorentzian / Fano: standard textbook lineshapes.

Each returns the typed :class:`~strumento.core.fitters.FitResult`. This module is **pure**: no
``qick`` import, no plotting, no stdout (D13).
"""

from __future__ import annotations

import lmfit
import numpy as np
import scipy.optimize

from strumento.core.fitters import FitResult, _params_stderr

__all__ = ["fit_lorentzian", "fit_fano", "fit_resonator_circle"]


# --------------------------------------------------------------------------- #
# Lorentzian magnitude fit
# --------------------------------------------------------------------------- #


def _edge_baseline(y: np.ndarray) -> float:
    """Off-resonant continuum level: median over the first + last ~10 % of the sweep."""
    k = max(1, y.size // 10)
    return float(np.median(np.concatenate([y[:k], y[-k:]])))


def _lorentzian(f, y0, amp, f0, hwhm):
    """y0 + amp / (1 + ((f-f0)/hwhm)^2).  amp>0 → peak, amp<0 → dip."""
    return y0 + amp / (1 + ((f - f0) / hwhm) ** 2)


def fit_lorentzian(freq, mag) -> FitResult:
    """Fit a magnitude spectrum to a Lorentzian and derive ``f0``, FWHM and loaded Q.

    Model: ``y0 + amp/(1 + ((f-f0)/hwhm)^2)`` — handles both a peak (``amp>0``) and a notch/dip
    (``amp<0``) via a signed-amplitude seed. ``derived``: ``f0``, ``hwhm``, ``fwhm`` (= 2·hwhm),
    ``Q_loaded`` (= f0/fwhm).
    """
    f = np.asarray(freq, dtype=float)
    y = np.asarray(mag, dtype=float)
    span = float(f.max() - f.min()) or 1.0

    baseline = _edge_baseline(y)
    dev = y - baseline
    i_ext = int(np.argmax(np.abs(dev)))
    amp0 = float(dev[i_ext])
    f0_0 = float(f[i_ext])

    model = lmfit.Model(_lorentzian)
    pars = model.make_params(y0=baseline, amp=amp0, f0=f0_0, hwhm=span / 20.0)
    pars["hwhm"].set(min=span * 1e-6)
    pars["f0"].set(min=float(f.min()), max=float(f.max()))

    out = model.fit(y, pars, f=f)
    params, stderr = _params_stderr(out)
    hwhm = abs(params["hwhm"])
    fwhm = 2 * hwhm
    f0 = params["f0"]
    Q = f0 / fwhm if fwhm else float("inf")
    derived = {"f0": f0, "hwhm": hwhm, "fwhm": fwhm, "Q_loaded": Q}
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))


# --------------------------------------------------------------------------- #
# Fano ("Spano") asymmetric lineshape
# --------------------------------------------------------------------------- #


def _fano(f, bg, amp, q, f0, gamma):
    """bg + amp·(q + eps)^2 / (1 + eps^2),  eps = (f-f0)/gamma.

    q → ±∞ recovers a Lorentzian peak; q = 0 gives a symmetric anti-resonance (dip).
    """
    eps = (f - f0) / gamma
    return bg + amp * (q + eps) ** 2 / (1 + eps**2)


def fit_fano(freq, mag) -> FitResult:
    """Fit an asymmetric Fano lineshape to a magnitude spectrum.

    Model: ``bg + amp·(q + eps)^2/(1 + eps^2)`` with ``eps = (f-f0)/gamma``; ``q`` is the Fano
    asymmetry parameter. ``derived``: ``f0``, ``gamma`` (HWHM-like), ``fwhm`` (= 2·gamma),
    ``q_fano``, ``Q_loaded`` (= f0/(2·gamma)).
    """
    f = np.asarray(freq, dtype=float)
    y = np.asarray(mag, dtype=float)
    span = float(f.max() - f.min()) or 1.0

    baseline = _edge_baseline(y)
    i_ext = int(np.argmax(np.abs(y - baseline)))
    f0_0 = float(f[i_ext])
    ptp = float(y.max() - y.min()) or 1.0

    model = lmfit.Model(_fano)
    best = None
    for q0 in (1.0, -1.0, 3.0):
        pars = model.make_params(
            bg=float(y.min()), amp=ptp, q=q0, f0=f0_0, gamma=span / 30.0
        )
        pars["gamma"].set(min=span * 1e-6)
        pars["f0"].set(min=float(f.min()), max=float(f.max()))
        pars["amp"].set(min=0.0)
        try:
            out = model.fit(y, pars, f=f)
        except Exception:
            continue
        if best is None or out.chisqr < best.chisqr:
            best = out

    params, stderr = _params_stderr(best)
    gamma = abs(params["gamma"])
    f0 = params["f0"]
    fwhm = 2 * gamma
    derived = {
        "f0": f0,
        "gamma": gamma,
        "fwhm": fwhm,
        "q_fano": params["q"],
        "Q_loaded": f0 / fwhm if fwhm else float("inf"),
    }
    return FitResult(params=params, stderr=stderr, derived=derived, success=bool(best.success))


# --------------------------------------------------------------------------- #
# Circle fit (Probst / Khalil)
# --------------------------------------------------------------------------- #


def _circle_fit_kasa(z: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kåsa) circle fit of complex points → (xc, yc, r0)."""
    x = z.real
    y = z.imag
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    xc = D / 2.0
    yc = E / 2.0
    r0 = float(np.sqrt(max(F + xc**2 + yc**2, 0.0)))
    return float(xc), float(yc), r0


def _circle_residual(tau: float, f: np.ndarray, s: np.ndarray) -> float:
    """Sum of squared distances to the best-fit circle after removing cable delay ``tau``."""
    sc = s * np.exp(2j * np.pi * f * tau)
    xc, yc, r0 = _circle_fit_kasa(sc)
    d = np.abs(sc - (xc + 1j * yc)) - r0
    return float(np.sum(d**2))


def _estimate_delay(f: np.ndarray, s: np.ndarray) -> float:
    """Estimate cable delay by finding the τ that makes the resonance trace a perfect circle.

    After removing the true delay the notch/reflection loop is an exact circle (a Möbius image of
    the real axis), so the circle-fit residual is ~0 there and strictly positive elsewhere
    (Probst). The residual has spurious local minima at large |τ| where the spiral wraps into a
    near-circle, so we **global grid-search** the residual (centred on a coarse tail-phase slope
    estimate) and then refine locally.

    The delay is searched in the **dimensionless** variable ``β = τ·span`` (≈ fringes across the
    band): the extracted Q is sensitive to the *differential* phase ramp ``2π·span·τ``, so τ must
    be pinned far tighter than a bounded minimiser's default absolute tolerance — rescaling makes
    the optimiser's ``xatol`` meaningful.
    """
    span = float(f.max() - f.min()) or 1.0

    def residual_beta(beta: float) -> float:
        return _circle_residual(beta / span, f, s)

    # coarse centre: tail-phase slope (phase ≈ -2π f τ far from resonance)
    n = f.size
    k = max(2, n // 7)
    idx = np.concatenate([np.arange(k), np.arange(n - k, n)])
    phase = np.unwrap(np.angle(s))
    tau0 = -np.polyfit(f[idx], phase[idx], 1)[0] / (2 * np.pi)
    beta0 = tau0 * span

    # global grid search over ±3 fringes, then a high-precision local refine
    grid = np.linspace(beta0 - 3.0, beta0 + 3.0, 401)
    resid = np.array([residual_beta(b) for b in grid])
    beta_g = float(grid[int(np.argmin(resid))])
    dgrid = grid[1] - grid[0]
    res = scipy.optimize.minimize_scalar(
        residual_beta,
        bounds=(beta_g - 2 * dgrid, beta_g + 2 * dgrid),
        method="bounded",
        options={"xatol": 1e-6},
    )
    beta = float(res.x) if res.success else beta_g
    return beta / span


def _phase_model(f, theta0, Ql, fr):
    """Phase-vs-frequency of the centred resonance loop: θ0 + 2·arctan(2 Ql (1 - f/fr))."""
    return theta0 + 2.0 * np.arctan(2.0 * Ql * (1.0 - f / fr))


def fit_resonator_circle(
    freq, s21, geometry: str = "notch", delay: float | None = None
) -> FitResult:
    """Probst/Khalil circle fit of complex scattering data → loaded / internal / coupling Q.

    Handles the real-world non-idealities: **cable delay** (fit from the tails + circle-residual
    refinement, or pass ``delay``), **environmental gain + IQ-plane rotation/offset** (the
    off-resonant point), and **impedance mismatch / asymmetry** (the rotation angle ``phi``).

    Pipeline (Probst *et al.* 2015): remove cable delay → algebraic circle fit → phase-vs-frequency
    tangent fit for ``f0`` + ``Q_loaded`` → off-resonant point gives env gain ``a`` / rotation
    ``alpha`` → diameter + ``phi`` give ``Q_coupling`` and ``Q_internal`` (Khalil diameter
    correction).

    Args:
        freq: frequency points (Hz).
        s21: complex scattering parameter (S21 for ``notch``, S11 for ``reflection``).
        geometry: ``"notch"`` (hanger / transmission, default) or ``"reflection"``.
        delay: cable delay (s); estimated from the data if ``None``.

    ``params``: ``f0``, ``Q_loaded``, ``Q_internal``, ``Q_coupling``, ``phi``, ``a``, ``alpha``,
    ``delay``, ``theta0``. ``derived`` exposes ``f0``, ``Q_loaded``, ``Q_internal``,
    ``Q_coupling``, ``phi``, ``fwhm``.
    """
    if geometry not in ("notch", "reflection"):
        raise ValueError(f"geometry must be 'notch' or 'reflection', got {geometry!r}")

    f = np.asarray(freq, dtype=float)
    s = np.asarray(s21, dtype=complex)

    # 1) remove cable delay
    tau = _estimate_delay(f, s) if delay is None else float(delay)
    sc = s * np.exp(2j * np.pi * f * tau)

    # 2) algebraic circle fit
    xc, yc, r0 = _circle_fit_kasa(sc)
    zc = xc + 1j * yc

    # 3) phase-vs-frequency tangent fit on the centred loop
    z = sc - zc
    phase = np.unwrap(np.angle(z))
    span = float(f.max() - f.min()) or 1.0
    fr0 = float(f[int(np.argmin(np.abs(sc)))])  # deepest excursion ≈ resonance
    Ql0 = fr0 / (span / 6.0)
    ph_model = lmfit.Model(_phase_model)
    ok = True
    best = None
    for sign in (1.0, -1.0):  # loop may be traversed either way
        pars = ph_model.make_params(theta0=float(np.mean(phase)), Ql=sign * Ql0, fr=fr0)
        pars["fr"].set(min=float(f.min()), max=float(f.max()))
        try:
            out = ph_model.fit(phase, pars, f=f)
        except Exception:
            continue
        if best is None or out.chisqr < best.chisqr:
            best = out
    if best is None:
        ok = False
        pars = ph_model.make_params(theta0=float(np.mean(phase)), Ql=Ql0, fr=fr0)
        best = ph_model.fit(phase, pars, f=f)

    theta0 = float(best.params["theta0"].value)
    Ql = abs(float(best.params["Ql"].value))
    fr = float(best.params["fr"].value)

    # 4) off-resonant point → environmental gain a + rotation alpha
    p_off = zc + r0 * np.exp(1j * (theta0 + np.pi))
    a = float(np.abs(p_off))
    alpha = float(np.angle(p_off))

    # 5) diameter + impedance-mismatch angle → coupling / internal Q (Khalil correction)
    d = 2.0 * r0 / a if a else 0.0  # normalised diameter = |Q_l / Q_c| (notch)
    phi = float(np.angle(np.exp(1j * (theta0 - alpha + np.pi))))  # wrap to (-π, π]

    if geometry == "notch":
        Qc_mag = Ql / d if d else float("inf")
    else:  # reflection carries a factor of 2 in the coupling term
        Qc_mag = 2.0 * Ql / d if d else float("inf")

    inv_Qi = 1.0 / Ql - np.cos(phi) / Qc_mag
    Qi = 1.0 / inv_Qi if inv_Qi > 0 else float("inf")

    fwhm = fr / Ql if Ql else float("inf")
    params = {
        "f0": fr,
        "Q_loaded": Ql,
        "Q_internal": Qi,
        "Q_coupling": Qc_mag,
        "phi": phi,
        "a": a,
        "alpha": alpha,
        "delay": tau,
        "theta0": theta0,
    }
    stderr = {k: float("nan") for k in params}
    if best.params["Ql"].stderr is not None:
        stderr["Q_loaded"] = float(best.params["Ql"].stderr)
    if best.params["fr"].stderr is not None:
        stderr["f0"] = float(best.params["fr"].stderr)
    derived = {
        "f0": fr,
        "Q_loaded": Ql,
        "Q_internal": Qi,
        "Q_coupling": Qc_mag,
        "phi": phi,
        "fwhm": fwhm,
    }
    success = bool(ok and best.success and np.isfinite(Qi) and Qi > 0)
    return FitResult(params=params, stderr=stderr, derived=derived, success=success)
