"""Tests for the vendored resonator / oscillator-spectroscopy fitters (design-review
2026-07-21 requirements).

Pure: fits synthetic complex S21 / magnitude data with known parameters + noise and checks
that the fitters recover them within tolerance. No qick.

Methods under test:
- ``fit_lorentzian``       — magnitude Lorentzian → f0, FWHM, Q_loaded
- ``fit_resonator_circle`` — Probst/Khalil circle fit of S21 → Ql, Qi, Qc (notch + reflection)
- ``fit_fano``             — asymmetric (Fano/"Spano") lineshape → f0, q, Q_loaded
"""

import numpy as np
import pytest

from strumento.core.fitters import FitResult
from strumento.core.resonator_fitters import (
    fit_fano,
    fit_lorentzian,
    fit_resonator_circle,
)

# --- reference models (independent of the implementation) ------------------


def _notch_s21(f, fr, Ql, Qc_mag, phi, a, alpha, tau):
    """Standard notch (hanger) resonator model, incl. cable delay + env gain/rotation."""
    env = a * np.exp(1j * alpha) * np.exp(-2j * np.pi * f * tau)
    return env * (1 - (Ql / Qc_mag) * np.exp(1j * phi) / (1 + 2j * Ql * (f / fr - 1)))


def _reflection_s11(f, fr, Ql, Qc_mag, phi, a, alpha, tau):
    """Standard reflection resonator model (factor of 2 in the coupling term)."""
    env = a * np.exp(1j * alpha) * np.exp(-2j * np.pi * f * tau)
    return env * (1 - (2 * Ql / Qc_mag) * np.exp(1j * phi) / (1 + 2j * Ql * (f / fr - 1)))


# --- Lorentzian magnitude fit ----------------------------------------------


def test_fit_lorentzian_recovers_f0_and_Q_dip():
    rng = np.random.default_rng(10)
    fr = 7.0e9
    fwhm = 2.0e6
    f = np.linspace(fr - 20e6, fr + 20e6, 601)
    # power-dip (notch magnitude): baseline minus Lorentzian
    hwhm = fwhm / 2
    mag = 1.0 - 0.8 / (1 + ((f - fr) / hwhm) ** 2) + rng.normal(0, 0.005, f.size)
    r = fit_lorentzian(f, mag)
    assert isinstance(r, FitResult)
    assert r.success
    assert r.derived["f0"] == pytest.approx(fr, rel=1e-5)
    assert r.derived["fwhm"] == pytest.approx(fwhm, rel=0.05)
    assert r.derived["Q_loaded"] == pytest.approx(fr / fwhm, rel=0.05)


def test_fit_lorentzian_recovers_peak():
    fr = 5.0e9
    fwhm = 4.0e6
    hwhm = fwhm / 2
    f = np.linspace(fr - 30e6, fr + 30e6, 401)
    mag = 0.1 + 0.9 / (1 + ((f - fr) / hwhm) ** 2)
    r = fit_lorentzian(f, mag)
    assert r.derived["f0"] == pytest.approx(fr, rel=1e-5)
    assert r.derived["Q_loaded"] == pytest.approx(fr / fwhm, rel=0.05)


# --- circle fit (notch) ----------------------------------------------------


def test_fit_resonator_circle_notch_recovers_Qs():
    rng = np.random.default_rng(11)
    fr = 6.5e9
    Ql = 5.0e4
    Qc_mag = 8.0e4
    phi = 0.0  # symmetric
    span = 8 * fr / Ql
    f = np.linspace(fr - span / 2, fr + span / 2, 801)
    s = _notch_s21(f, fr, Ql, Qc_mag, phi, a=1.0, alpha=0.0, tau=0.0)
    noise = rng.normal(0, 2e-3, f.size) + 1j * rng.normal(0, 2e-3, f.size)
    r = fit_resonator_circle(f, s + noise, geometry="notch")
    assert r.success
    assert r.derived["f0"] == pytest.approx(fr, rel=1e-5)
    assert r.derived["Q_loaded"] == pytest.approx(Ql, rel=0.05)
    assert r.derived["Q_coupling"] == pytest.approx(Qc_mag, rel=0.05)
    # 1/Ql = 1/Qi + 1/Qc  (symmetric) → Qi from the identity
    Qi_true = 1.0 / (1.0 / Ql - 1.0 / Qc_mag)
    assert r.derived["Q_internal"] == pytest.approx(Qi_true, rel=0.08)


def test_fit_resonator_circle_handles_env_and_delay_and_asymmetry():
    rng = np.random.default_rng(12)
    fr = 6.5e9
    Ql = 4.0e4
    Qc_mag = 6.0e4
    phi = 0.3  # impedance-mismatch rotation
    a, alpha, tau = 0.7, 1.1, 35e-9  # gain, rotation, cable delay
    span = 10 * fr / Ql
    f = np.linspace(fr - span / 2, fr + span / 2, 1201)
    s = _notch_s21(f, fr, Ql, Qc_mag, phi, a=a, alpha=alpha, tau=tau)
    noise = rng.normal(0, 1e-3, f.size) + 1j * rng.normal(0, 1e-3, f.size)
    r = fit_resonator_circle(f, s + noise, geometry="notch")
    assert r.success
    assert r.derived["f0"] == pytest.approx(fr, rel=1e-5)
    assert r.derived["Q_loaded"] == pytest.approx(Ql, rel=0.06)
    assert r.derived["Q_coupling"] == pytest.approx(Qc_mag, rel=0.08)
    assert r.derived["phi"] == pytest.approx(phi, abs=0.08)
    assert r.params["delay"] == pytest.approx(tau, rel=0.15)


def test_fit_resonator_circle_reflection_recovers_Qs():
    rng = np.random.default_rng(13)
    fr = 5.0e9
    Ql = 3.0e4
    Qc_mag = 6.0e4
    span = 8 * fr / Ql
    f = np.linspace(fr - span / 2, fr + span / 2, 801)
    s = _reflection_s11(f, fr, Ql, Qc_mag, phi=0.0, a=1.0, alpha=0.0, tau=0.0)
    noise = rng.normal(0, 2e-3, f.size) + 1j * rng.normal(0, 2e-3, f.size)
    r = fit_resonator_circle(f, s + noise, geometry="reflection")
    assert r.success
    assert r.derived["f0"] == pytest.approx(fr, rel=1e-5)
    assert r.derived["Q_loaded"] == pytest.approx(Ql, rel=0.06)
    assert r.derived["Q_coupling"] == pytest.approx(Qc_mag, rel=0.08)


# --- Fano (asymmetric) lineshape -------------------------------------------


def test_fit_fano_recovers_asymmetric_lineshape():
    rng = np.random.default_rng(14)
    fr = 4.5e9
    gamma = 1.5e6
    q = 2.0  # Fano asymmetry
    f = np.linspace(fr - 25e6, fr + 25e6, 601)
    eps = (f - fr) / gamma
    mag = 0.05 + 0.3 * (q + eps) ** 2 / (1 + eps**2) + rng.normal(0, 0.01, f.size)
    r = fit_fano(f, mag)
    assert isinstance(r, FitResult)
    assert r.success
    assert r.derived["f0"] == pytest.approx(fr, rel=1e-4)
    assert abs(r.derived["q_fano"]) == pytest.approx(abs(q), rel=0.15)
    assert r.derived["Q_loaded"] == pytest.approx(fr / (2 * gamma), rel=0.1)


def test_fit_fano_reduces_to_lorentzian_for_large_q():
    # large |q| → symmetric Lorentzian peak; f0 still recovered
    fr = 4.5e9
    gamma = 2.0e6
    q = 50.0
    f = np.linspace(fr - 40e6, fr + 40e6, 601)
    eps = (f - fr) / gamma
    mag = 0.0 + 1e-3 * (q + eps) ** 2 / (1 + eps**2)
    r = fit_fano(f, mag)
    assert r.derived["f0"] == pytest.approx(fr, rel=1e-4)


def test_fit_fano_successful_fit_is_unchanged_characterization():
    # Characterization (#173), captured from the pre-fix code: when a start succeeds, the
    # all-starts-failed guard must not move any fitted value, stderr or derived key.
    rng = np.random.default_rng(14)
    fr = 4.5e9
    gamma = 1.5e6
    q = 2.0
    f = np.linspace(fr - 25e6, fr + 25e6, 601)
    eps = (f - fr) / gamma
    mag = 0.05 + 0.3 * (q + eps) ** 2 / (1 + eps**2) + rng.normal(0, 0.01, f.size)
    r = fit_fano(f, mag)
    assert r.success is True
    assert r.params == pytest.approx({
        "bg": 0.05197398377188247,
        "amp": 0.2977583049101722,
        "q": 2.006835550123446,
        "f0": 4499998966.896418,
        "gamma": 1505548.9924039303,
    }, rel=1e-9)
    assert r.stderr == pytest.approx({
        "bg": 0.0011954019096637064,
        "amp": 0.001319790887447313,
        "q": 0.005122123466628871,
        "f0": 4015.047840018268,
        "gamma": 3952.907631742172,
    }, rel=1e-4)
    # The fitted values above are pinned to 1e-9 on every CI platform. The standard errors come
    # from lmfit's covariance estimate, whose last digits shift by ~2e-5 (relative) between
    # numpy/scipy releases, so they, and the derived *_stderr keys, are pinned to 1e-4.
    assert r.derived == pytest.approx({
        "f0": 4499998966.896418,
        "gamma": 1505548.9924039303,
        "fwhm": 3011097.9848078606,
        "q_fano": 2.006835550123446,
        "Q_loaded": 1494.4711163836685,
        "f0_stderr": 4015.047840018268,
        "gamma_stderr": 3952.907631742172,
        "fwhm_stderr": 7905.815263484344,
        "q_fano_stderr": 0.005122123466628871,
        "Q_loaded_stderr": 3.923749558747246,
    }, rel=1e-4)


def test_fit_fano_returns_failed_fit_when_every_start_fails():
    # #173: all-NaN data makes every multi-start lmfit fit raise; the fitter must report a
    # failed fit — the package's empty-dicts convention — instead of an AttributeError on None.
    r = fit_fano(np.linspace(5000, 5010, 101), np.full(101, np.nan))
    assert isinstance(r, FitResult)
    assert r.success is False
    assert r.params == {}
    assert r.stderr == {}
    assert r.derived == {}


def test_fit_fano_keeps_a_surviving_start_when_others_fail(monkeypatch):
    # #173: the guard only fires when *every* start fails — one surviving start is still the fit.
    import lmfit

    real_fit = lmfit.Model.fit
    calls = {"n": 0}

    def flaky_fit(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("synthetic start failure")
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(lmfit.Model, "fit", flaky_fit)
    fr = 4.5e9
    gamma = 1.5e6
    f = np.linspace(fr - 25e6, fr + 25e6, 601)
    eps = (f - fr) / gamma
    mag = 0.05 + 0.3 * (2.0 + eps) ** 2 / (1 + eps**2)
    r = fit_fano(f, mag)
    assert calls["n"] == 3
    assert r.success
    assert r.derived["f0"] == pytest.approx(fr, rel=1e-4)


# --- circle fit: characterization + degenerate data (#179) -------------------

# Captured from the pre-#179 code. Each case is a valid synthetic resonator: (model, seed, fr,
# Ql, Qc_mag, phi, a, alpha, tau, span in linewidths, points, noise sigma, geometry, delay arg).
_CIRCLE_CASES = {
    "notch_symmetric": (_notch_s21, 21, 6.5e9, 5.0e4, 8.0e4, 0.0, 1.0, 0.0, 0.0,
                        8, 801, 2e-3, "notch", None),
    "notch_env_delay_asym": (_notch_s21, 22, 7.2e9, 2.0e4, 3.5e4, -0.25, 0.6, 2.0, 42e-9,
                             10, 1201, 1e-3, "notch", None),
    "notch_given_delay": (_notch_s21, 23, 5.8e9, 8.0e4, 1.2e5, 0.15, 1.3, -0.7, 25e-9,
                          8, 601, 1.5e-3, "notch", 25e-9),
    "reflection": (_reflection_s11, 24, 5.0e9, 3.0e4, 6.0e4, 0.1, 0.9, 0.4, 10e-9,
                   8, 801, 2e-3, "reflection", None),
}

_NAN = float("nan")
_CIRCLE_PINNED = {
    "notch_symmetric": (
        {"f0": 6500000017.871214, "Q_loaded": 49913.38230886721,
         "Q_internal": 133046.21884751474, "Q_coupling": 79881.6332030854,
         "phi": -9.595634229908612e-05, "a": 1.0002192588307943,
         "alpha": -1.2190867044059295, "delay": 2.7784387023018474e-10,
         "theta0": -4.360775314338022},
        {"f0": 28.05373785231408, "Q_loaded": 19.041253374288317, "fwhm": 49.67923446192316},
        130225.59716848675,
    ),
    "notch_env_delay_asym": (
        {"f0": 7200000044.760318, "Q_loaded": 20015.93451721124,
         "Q_internal": 44849.57065560931, "Q_coupling": 35025.172884930806,
         "phi": -0.24998272106356278, "a": 0.5999498278473204,
         "alpha": 0.8511738080390405, "delay": 4.197460440515243e-08,
         "theta0": -2.5404015666143156},
        {"f0": 61.87112027367, "Q_loaded": 6.265041641498179, "fwhm": 112.59126995039702},
        359713.40926246555,
    ),
    "notch_given_delay": (
        {"f0": 5800000001.868392, "Q_loaded": 79975.74230536904,
         "Q_internal": 234458.87715919162, "Q_coupling": 120014.41701638044,
         "phi": 0.15009475559267038, "a": 1.3000692074076903,
         "alpha": -0.700063791135098, "delay": 2.5e-08,
         "theta0": -3.6915616891322207},
        {"f0": 9.87349172127324, "Q_loaded": 19.79840674175278, "fwhm": 17.953192084324506},
        72521.99022701685,
    ),
    "reflection": (
        {"f0": 5000000025.904424, "Q_loaded": 30048.606804047806,
         "Q_internal": 59776.46745361932, "Q_coupling": 60121.15912220938,
         "phi": 0.09973537602910466, "a": 0.8997939063947122,
         "alpha": -1.163777213981865, "delay": 9.750222231321044e-09,
         "theta0": -4.205634491542553},
        {"f0": 25.748696131662694, "Q_loaded": 8.284157615643325, "fwhm": 45.87432400890517},
        166397.0665432276,
    ),
}


def _circle_case_data(name):
    model, seed, fr, Ql, Qc, phi, a, alpha, tau, width, n, sigma, geometry, delay = (
        _CIRCLE_CASES[name]
    )
    rng = np.random.default_rng(seed)
    span = width * fr / Ql
    f = np.linspace(fr - span / 2, fr + span / 2, n)
    s = model(f, fr, Ql, Qc, phi, a, alpha, tau)
    s = s + rng.normal(0, sigma, f.size) + 1j * rng.normal(0, sigma, f.size)
    return f, s, geometry, delay


@pytest.mark.parametrize("name", sorted(_CIRCLE_CASES))
def test_fit_resonator_circle_valid_fit_is_unchanged_characterization(name):
    # Characterization (#179), captured from the pre-fix code: the degenerate-data guards must
    # not move any fit on a valid resonator. Unlike the Fano fit, this one runs an SVD, a delay
    # search and a least-squares circle, whose last digits differ across numpy/LAPACK builds:
    # CI measured the dimensionless values and Qs up to ~6e-7 relative apart (phi up to ~1.6e-7
    # rad absolute), while f0 and the FWHM agreed to 1e-9 relative. So those two keep 1e-9
    # relative (~6 Hz on a ~6 GHz f0) or a tenth of their own standard error, whichever is
    # larger (a relative 1e-6 would allow ~6 kHz, 100x f0's error); everything else is pinned at 1e-6 relative with a 1e-6 floor (for the near-zero
    # symmetric-notch phi). The standard errors come from the phase fit's covariance after the
    # delay search, so they inherit that search's sensitivity to the CPU: numpy's OpenBLAS picks
    # CPU-specific kernels (DYNAMIC_ARCH), and different CI runners gave f0's error as 9.84 and
    # 10.20 Hz where this machine gives 9.87 (a 3.7% spread; locally it is deterministic across
    # processes and thread counts). They are pinned at 10% so a gross change still fails; the
    # fitted values above are what this characterization protects.
    f, s, geometry, delay = _circle_case_data(name)
    r = fit_resonator_circle(f, s, geometry=geometry, delay=delay)
    params, errs, fwhm = _CIRCLE_PINNED[name]
    assert r.success is True
    assert r.params["f0"] == pytest.approx(params["f0"], rel=1e-9, abs=0.1 * errs["f0"])
    rest = {k: v for k, v in params.items() if k != "f0"}
    assert {k: r.params[k] for k in rest} == pytest.approx(rest, rel=1e-6, abs=1e-6)
    stderr_expected = {k: _NAN for k in params}
    stderr_expected["f0"] = errs["f0"]
    stderr_expected["Q_loaded"] = errs["Q_loaded"]
    assert r.stderr == pytest.approx(stderr_expected, rel=1e-1, nan_ok=True)
    values = {k: params[k] for k in ("f0", "Q_loaded", "Q_internal", "Q_coupling", "phi")}
    values["fwhm"] = fwhm
    assert r.derived["f0"] == pytest.approx(values["f0"], rel=1e-9, abs=0.1 * errs["f0"])
    assert r.derived["fwhm"] == pytest.approx(values["fwhm"], rel=1e-9, abs=0.1 * errs["fwhm"])
    dimless = {k: v for k, v in values.items() if k not in ("f0", "fwhm")}
    assert {k: r.derived[k] for k in dimless} == pytest.approx(dimless, rel=1e-6, abs=1e-6)
    assert {k: r.derived[k] for k in r.derived if k.endswith("_stderr")} == pytest.approx({
        "f0_stderr": errs["f0"],
        "Q_loaded_stderr": errs["Q_loaded"],
        "fwhm_stderr": errs["fwhm"],
        "Q_internal_stderr": _NAN,
        "Q_coupling_stderr": _NAN,
        "phi_stderr": _NAN,
    }, rel=1e-1, nan_ok=True)
    assert set(r.derived) == set(values) | {f"{k}_stderr" for k in values}


def _assert_failed_fit(r):
    assert isinstance(r, FitResult)
    assert r.success is False
    assert r.params == {}
    assert r.stderr == {}
    assert r.derived == {}


_F_DEGEN = np.linspace(5000, 5010, 101)


def test_fit_resonator_circle_all_nan_is_a_failed_fit_without_lapack_noise(capfd):
    # #179: all-NaN data used to raise LinAlgError (and print LAPACK DLASCL noise on stderr).
    r = fit_resonator_circle(_F_DEGEN, np.full(101, np.nan) + 0j)
    _assert_failed_fit(r)
    assert "DLASCL" not in capfd.readouterr().err


@pytest.mark.parametrize("bad", ["nan_point", "inf_point", "nan_freq"])
def test_fit_resonator_circle_any_non_finite_point_is_a_failed_fit(bad):
    # #179: a single non-finite sample poisons the algebraic circle fit (it raised, or for inf
    # never returned); the fit reports failure rather than guessing which points to drop.
    f, s, geometry, delay = _circle_case_data("notch_symmetric")
    f, s = f.copy(), s.copy()
    if bad == "nan_point":
        s[400] = np.nan
    elif bad == "inf_point":
        s[400] = np.inf
    else:
        f[400] = np.nan
    _assert_failed_fit(fit_resonator_circle(f, s, geometry=geometry, delay=delay))


def test_fit_resonator_circle_all_zero_is_a_failed_fit():
    # #179: all-zero data has no resonance; it used to come back success=True.
    _assert_failed_fit(fit_resonator_circle(_F_DEGEN, np.zeros(101, complex)))


@pytest.mark.parametrize("value", [1 + 1j, 1e-12 + 1e-12j, 3e6 - 2e6j])
def test_fit_resonator_circle_constant_data_is_a_failed_fit(value):
    # Constant data traces no loop at any scale (it was already success=False; now the shape too).
    _assert_failed_fit(fit_resonator_circle(_F_DEGEN, np.full(101, value, dtype=complex)))


def test_fit_resonator_circle_degeneracy_rule_is_scale_free():
    # The no-loop rule is relative to the data's own magnitude: a valid resonator recorded at a
    # tiny absolute scale (raw ADC units, attenuated chain) still fits, with the same Qs.
    f, s, geometry, delay = _circle_case_data("notch_symmetric")
    ref = fit_resonator_circle(f, s, geometry=geometry, delay=delay)
    r = fit_resonator_circle(f, 1e-9 * s, geometry=geometry, delay=delay)
    assert r.success is True
    # The fit itself is scale-invariant only to ~1e-4 (the delay search and the lmfit tolerances
    # see absolute residuals) — pre-existing behaviour, pinned loosely; the guard is the point.
    assert r.derived["Q_loaded"] == pytest.approx(ref.derived["Q_loaded"], rel=1e-3)
    assert r.derived["Q_coupling"] == pytest.approx(ref.derived["Q_coupling"], rel=1e-3)


def test_fit_resonator_circle_both_phase_starts_failing_is_a_failed_fit(monkeypatch):
    # #179: when both phase-fit starts raise, the old fallback re-fit was unguarded and raised too.
    import lmfit

    from strumento.core import resonator_fitters

    real_fit = lmfit.Model.fit

    def failing_phase_fit(self, *args, **kwargs):
        if self.func is resonator_fitters._phase_model:
            raise ValueError("synthetic phase-fit failure")
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(lmfit.Model, "fit", failing_phase_fit)
    f, s, geometry, delay = _circle_case_data("notch_symmetric")
    _assert_failed_fit(fit_resonator_circle(f, s, geometry=geometry, delay=delay))
