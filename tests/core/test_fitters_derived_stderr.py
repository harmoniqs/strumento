"""Errors on derived quantities: every core fitter reports ``derived["<name>_stderr"]`` (#137).

The convention the experiment analyzers already follow (DRAG ``alpha_opt_stderr``, error-amp
``contrast_stderr``, self-Kerr ``K_stderr_mhz``) extended to the core fitter library: each
derived key computed from fitted parameters carries a linearly propagated 1σ error, and the
error is ``nan`` — never missing, never zero — when lmfit could not estimate one.

Pure: synthetic numpy data, no qick.
"""

import math

import lmfit
import numpy as np
import pytest
from scipy.optimize import curve_fit
from scipy.special import erf

from strumento.core.fitters import (
    fit_damped_rabi,
    fit_decaysin_detuned,
    fit_double_gaussian,
    fit_exp,
    fit_gaussian_wigner,
    fit_number_splitting_comb,
    fit_parity,
    fit_rabi,
    fit_time_of_flight,
)
from strumento.core.resonator_fitters import fit_fano, fit_lorentzian, fit_resonator_circle


def _rabi_data(seed=0, n=121, noise=0.01):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 30000, n)
    y = 0.5 + 0.4 * np.cos(2 * np.pi * x / 20000.0) * np.exp(-x / 80000.0)
    return x, y + rng.normal(0, noise, x.size)


def _exp_data(seed=1, n=121, noise=0.01):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 300.0, n)
    return t, 0.1 + 0.85 * np.exp(-t / 80.0) + rng.normal(0, noise, t.size)


def _ramsey_data(seed=2):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 20.0, 401)
    y = 0.5 + 0.4 * np.exp(-t / 8.0) * np.cos(2 * np.pi * 1.5 * t + 0.3)
    return t, y + rng.normal(0, 0.008, t.size)


def _damped_rabi_data(seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2.0, 201)
    y = 0.5 - 0.45 * np.exp(-t / 3.0) * np.cos(2 * np.pi * 5.0 * t)
    return t, y + rng.normal(0, 0.01, t.size)


def _parity_data(seed=4):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 360.0, 73)
    y = 0.5 + 0.35 * np.cos(2 * np.pi * x / 360.0 + 0.7)
    return x, y + rng.normal(0, 0.01, x.size)


def _wigner_data(seed=5):
    rng = np.random.default_rng(seed)
    g = np.linspace(-1.0, 1.0, 81)
    y = 0.05 + 0.8 * np.exp(-2.0 * (2.5 * g) ** 2)
    return g, y + rng.normal(0, 0.01, g.size)


def _comb_data(seed=13, f_ge=3566.0, chi=0.9, nbar=2.0, gamma=0.12):
    rng = np.random.default_rng(seed)
    f = np.linspace(f_ge - 4.0, f_ge + 1.0, 201)
    comb = sum(
        nbar**k * math.exp(-nbar) / math.factorial(k) / (1 + ((f - (f_ge - k * chi)) / gamma) ** 2)
        for k in range(7)
    )
    return f, 0.02 + 0.9 * comb + rng.normal(0, 0.01, f.size)


def _tof_model(x, offset, amp, t0, sigma):
    return offset + amp * 0.5 * (1.0 + erf((x - t0) / (np.sqrt(2.0) * sigma)))


def _tof_data(seed=6):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 5.0, 501)
    return x, _tof_model(x, 0.02, 0.78, 0.42, 0.03) + rng.normal(0, 0.01, x.size)


def _double_gaussian_model(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2)) + a2 * np.exp(
        -((x - mu2) ** 2) / (2 * sigma2**2)
    )


def _histogram_data(seed=7):
    rng = np.random.default_rng(seed)
    x = np.linspace(-3.0, 4.0, 141)
    y = _double_gaussian_model(x, 120.0, -1.0, 0.5, 80.0, 2.0, 0.6)
    return x, y + rng.normal(0, 2.0, x.size)


def _lorentzian_model(f, y0, amp, f0, hwhm):
    return y0 + amp / (1 + ((f - f0) / hwhm) ** 2)


def _lorentzian_data(seed=10, fr=7.0e9, hwhm=1.0e6):
    rng = np.random.default_rng(seed)
    f = np.linspace(fr - 20e6, fr + 20e6, 601)
    return f, _lorentzian_model(f, 1.0, -0.8, fr, hwhm) + rng.normal(0, 0.005, f.size)


def _fano_model(f, bg, amp, q, f0, gamma):
    eps = (f - f0) / gamma
    return bg + amp * (q + eps) ** 2 / (1 + eps**2)


def _fano_data(seed=14, fr=4.5e9, gamma=1.5e6):
    rng = np.random.default_rng(seed)
    f = np.linspace(fr - 25e6, fr + 25e6, 601)
    return f, _fano_model(f, 0.05, 0.3, 2.0, fr, gamma) + rng.normal(0, 0.01, f.size)


def _notch_data(seed=12, fr=6.5e9, Ql=4.0e4, Qc=6.0e4, phi=0.3, a=0.7, alpha=1.1, tau=35e-9):
    rng = np.random.default_rng(seed)
    span = 10 * fr / Ql
    f = np.linspace(fr - span / 2, fr + span / 2, 1201)
    env = a * np.exp(1j * alpha) * np.exp(-2j * np.pi * f * tau)
    s = env * (1 - (Ql / Qc) * np.exp(1j * phi) / (1 + 2j * Ql * (f / fr - 1)))
    return f, s + rng.normal(0, 1e-3, f.size) + 1j * rng.normal(0, 1e-3, f.size)


def _independent_sigma(model, x, y, fitted: dict, names, derived_fn) -> float:
    """An independent 1σ for ``derived_fn(params)``: scipy ``curve_fit`` (its own optimizer and
    covariance, scaled by reduced χ² like lmfit) re-converged from the fitter's solution, then a
    central finite-difference gradient of the test's OWN formula — no fitter internals."""
    popt, pcov = curve_fit(model, x, y, p0=[fitted[n] for n in names])
    grad = np.zeros(len(names))
    for i in range(len(names)):
        h = 1e-6 * max(abs(popt[i]), 1e-9)
        up, dn = popt.copy(), popt.copy()
        up[i] += h
        dn[i] -= h
        grad[i] = (derived_fn(dict(zip(names, up))) - derived_fn(dict(zip(names, dn)))) / (2 * h)
    return float(np.sqrt(grad @ pcov @ grad))



# Every core fitter, on representative data (core/fitters.py + core/resonator_fitters.py).
_CASES = {
    "fit_rabi": lambda: fit_rabi(*_rabi_data()),
    "fit_exp": lambda: fit_exp(*_exp_data()),
    "fit_decaysin_detuned": lambda: fit_decaysin_detuned(*_ramsey_data()),
    "fit_damped_rabi": lambda: fit_damped_rabi(*_damped_rabi_data()),
    "fit_parity": lambda: fit_parity(*_parity_data()),
    "fit_gaussian_wigner": lambda: fit_gaussian_wigner(*_wigner_data()),
    "fit_number_splitting_comb": lambda: fit_number_splitting_comb(
        *_comb_data(), 3566.0, chi_seed=0.9
    ),
    "fit_double_gaussian": lambda: fit_double_gaussian(*_histogram_data()),
    "fit_time_of_flight": lambda: fit_time_of_flight(*_tof_data()),
    "fit_lorentzian": lambda: fit_lorentzian(*_lorentzian_data()),
    "fit_fano": lambda: fit_fano(*_fano_data()),
    "fit_resonator_circle": lambda: fit_resonator_circle(*_notch_data(), geometry="notch"),
}

# Keys documented as exempt: present, but always nan (no covariance to propagate).
_EXEMPT = {"fit_resonator_circle": {"Q_internal", "Q_coupling", "phi"}}


def _value_keys(derived):
    return {k for k in derived if not k.endswith("_stderr")}


# --- the contract: every derived key has an error, and no error means nan -------------------


@pytest.mark.parametrize("case", sorted(_CASES))
def test_every_derived_key_has_a_finite_stderr_twin(case):
    r = _CASES[case]()
    assert r.success
    keys = _value_keys(r.derived)
    assert {k for k in r.derived if k.endswith("_stderr")} == {f"{k}_stderr" for k in keys}
    for k in keys - _EXEMPT.get(case, set()):
        se = r.derived[f"{k}_stderr"]
        assert np.isfinite(se) and se > 0, f"{case}: {k}_stderr = {se}"


@pytest.fixture
def lmfit_without_errors(monkeypatch):
    """Make every lmfit fit come back as if the covariance could not be estimated — the state
    lmfit reports with ``covar = None`` and ``param.stderr = None``."""
    real_fit = lmfit.Model.fit

    def fit(self, *args, **kwargs):
        out = real_fit(self, *args, **kwargs)
        out.covar = None
        for par in out.params.values():
            par.stderr = None
        return out

    monkeypatch.setattr(lmfit.Model, "fit", fit)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_no_lmfit_error_gives_nan_derived_errors(case, lmfit_without_errors):
    r = _CASES[case]()
    stderr_keys = [k for k in r.derived if k.endswith("_stderr")]
    assert stderr_keys and len(stderr_keys) == len(_value_keys(r.derived))
    for k in stderr_keys:
        assert math.isnan(r.derived[k]), f"{case}: {k} = {r.derived[k]}"


def test_degenerate_flat_trace_gives_nan_not_zero():
    """A real degenerate fit (no signal at all): lmfit cannot estimate a covariance."""
    x = np.linspace(0, 30000, 121)
    r = fit_rabi(x, np.zeros_like(x))
    assert math.isnan(r.stderr["period"])
    assert math.isnan(r.derived["period_stderr"]) and math.isnan(r.derived["pi_gain_stderr"])


# --- characterization: no derived VALUE changed ---------------------------------------------
#
# The pre-#137 formula for every derived value, written out from the fitted params. The value
# keys (everything that is not ``*_stderr``) and their values must be exactly these.


def _old_double_gaussian(p):
    mu_a, mu_b = sorted((p["mu1"], p["mu2"]))
    sig = (abs(p["sigma1"]) + abs(p["sigma2"])) / 2.0
    return {
        "threshold": (mu_a + mu_b) / 2.0,
        "separation": abs(mu_b - mu_a) / sig if sig else float("inf"),
        "mu_low": mu_a,
        "mu_high": mu_b,
    }


_OLD_FORMULAS = {
    "fit_rabi": lambda p: {"period": abs(p["period"]), "pi_gain": abs(p["period"]) / 2.0},
    "fit_exp": lambda p: {"T1": abs(p["tau"]), "tau": abs(p["tau"])},
    "fit_decaysin_detuned": lambda p: {
        "detuning": abs(p["freq"]), "freq": abs(p["freq"]),
        "T2": abs(p["tau"]), "tau": abs(p["tau"]),
    },
    "fit_damped_rabi": lambda p: {
        "rabi_freq": abs(p["freq"]), "period": 1.0 / abs(p["freq"]),
        "pi_time": 1.0 / abs(p["freq"]) / 2.0, "half_pi_time": 1.0 / abs(p["freq"]) / 4.0,
        "tau": abs(p["tau"]),
    },
    "fit_parity": lambda p: {
        "contrast": 2.0 * abs(p["amp"]), "phase_deg": float(np.degrees(p["phase"])),
    },
    "fit_gaussian_wigner": lambda p: {
        "alpha_per_gain": abs(p["alpha_per_gain"]), "contrast": 2.0 * abs(p["amp"]),
    },
    "fit_number_splitting_comb": lambda p: {
        "chi": abs(p["chi"]), "chi_estimate": -abs(p["chi"]),
        "nbar": abs(p["nbar"]), "gamma": abs(p["gamma"]),
    },
    "fit_double_gaussian": _old_double_gaussian,
    "fit_time_of_flight": lambda p: {
        "tof_us": p["t0"], "t0": p["t0"], "rise_us": abs(p["sigma"]),
        "noise_floor": p["offset"], "settled_level": p["offset"] + p["amp"],
    },
    "fit_lorentzian": lambda p: {
        "f0": p["f0"], "hwhm": abs(p["hwhm"]), "fwhm": 2 * abs(p["hwhm"]),
        "Q_loaded": p["f0"] / (2 * abs(p["hwhm"])),
    },
    "fit_fano": lambda p: {
        "f0": p["f0"], "gamma": abs(p["gamma"]), "fwhm": 2 * abs(p["gamma"]),
        "q_fano": p["q"], "Q_loaded": p["f0"] / (2 * abs(p["gamma"])),
    },
    "fit_resonator_circle": lambda p: {
        "f0": p["f0"], "Q_loaded": p["Q_loaded"], "Q_internal": p["Q_internal"],
        "Q_coupling": p["Q_coupling"], "phi": p["phi"], "fwhm": p["f0"] / p["Q_loaded"],
    },
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_derived_values_are_unchanged_by_the_error_keys(case):
    r = _CASES[case]()
    old = _OLD_FORMULAS[case](r.params)
    assert {k: r.derived[k] for k in _value_keys(r.derived)} == old

# --- independent propagation checks ---------------------------------------------------------


def test_fit_rabi_derived_errors_follow_the_period_error():
    r = fit_rabi(*_rabi_data())
    assert np.isfinite(r.stderr["period"]) and r.stderr["period"] > 0
    assert r.derived["period_stderr"] == pytest.approx(r.stderr["period"], rel=1e-9)
    assert r.derived["pi_gain_stderr"] == pytest.approx(r.stderr["period"] / 2, rel=1e-9)


def test_fit_exp_T1_and_tau_errors_are_the_tau_error():
    r = fit_exp(*_exp_data())
    assert np.isfinite(r.stderr["tau"]) and r.stderr["tau"] > 0
    assert r.derived["T1_stderr"] == pytest.approx(r.stderr["tau"], rel=1e-9)
    assert r.derived["tau_stderr"] == r.derived["T1_stderr"]  # the alias mirrors its twin



def test_fit_decaysin_detuned_errors_follow_freq_and_tau():
    r = fit_decaysin_detuned(*_ramsey_data())
    se = r.stderr
    assert np.isfinite(se["freq"]) and np.isfinite(se["tau"])
    assert r.derived["detuning_stderr"] == pytest.approx(se["freq"], rel=1e-9)
    assert r.derived["freq_stderr"] == r.derived["detuning_stderr"]
    assert r.derived["T2_stderr"] == pytest.approx(se["tau"], rel=1e-9)
    assert r.derived["tau_stderr"] == r.derived["T2_stderr"]


def test_fit_damped_rabi_period_errors_scale_as_one_over_freq_squared():
    r = fit_damped_rabi(*_damped_rabi_data())
    f, sf = r.params["freq"], r.stderr["freq"]
    assert np.isfinite(sf) and sf > 0
    d = r.derived
    assert d["rabi_freq_stderr"] == pytest.approx(sf, rel=1e-9)
    assert d["period_stderr"] == pytest.approx(sf / f**2, rel=1e-9)
    assert d["pi_time_stderr"] == pytest.approx(sf / (2 * f**2), rel=1e-9)
    assert d["half_pi_time_stderr"] == pytest.approx(sf / (4 * f**2), rel=1e-9)
    assert d["tau_stderr"] == pytest.approx(r.stderr["tau"], rel=1e-9)


def test_fit_parity_contrast_and_phase_errors():
    r = fit_parity(*_parity_data())
    se = r.stderr
    assert np.isfinite(se["amp"]) and np.isfinite(se["phase"])
    assert r.derived["contrast_stderr"] == pytest.approx(2 * se["amp"], rel=1e-9)
    assert r.derived["phase_deg_stderr"] == pytest.approx(np.degrees(se["phase"]), rel=1e-9)


def test_fit_gaussian_wigner_ruler_and_contrast_errors():
    r = fit_gaussian_wigner(*_wigner_data())
    se = r.stderr
    assert np.isfinite(se["alpha_per_gain"]) and se["alpha_per_gain"] > 0
    assert r.derived["alpha_per_gain_stderr"] == pytest.approx(se["alpha_per_gain"], rel=1e-9)
    assert r.derived["contrast_stderr"] == pytest.approx(2 * se["amp"], rel=1e-9)


def test_fit_number_splitting_comb_errors_follow_the_comb_params():
    f, y = _comb_data()
    r = fit_number_splitting_comb(f, y, 3566.0, chi_seed=0.9)
    se, d = r.stderr, r.derived
    assert np.isfinite(se["chi"]) and se["chi"] > 0
    assert d["chi_stderr"] == pytest.approx(se["chi"], rel=1e-9)
    assert d["chi_estimate_stderr"] == d["chi_stderr"]  # the signed twin mirrors chi's error
    assert d["nbar_stderr"] == pytest.approx(se["nbar"], rel=1e-9)
    assert d["gamma_stderr"] == pytest.approx(se["gamma"], rel=1e-9)


def test_fit_time_of_flight_errors_including_the_correlated_settled_level():
    x, y = _tof_data()
    r = fit_time_of_flight(x, y)
    se, d = r.stderr, r.derived
    assert np.isfinite(se["t0"]) and se["t0"] > 0
    assert d["tof_us_stderr"] == pytest.approx(se["t0"], rel=1e-9)
    assert d["t0_stderr"] == d["tof_us_stderr"]
    assert d["rise_us_stderr"] == pytest.approx(se["sigma"], rel=1e-9)
    assert d["noise_floor_stderr"] == pytest.approx(se["offset"], rel=1e-9)
    names = ["offset", "amp", "t0", "sigma"]
    ref = _independent_sigma(
        _tof_model, x, y, r.params, names, lambda p: p["offset"] + p["amp"]
    )
    assert d["settled_level_stderr"] == pytest.approx(ref, rel=1e-3)


def test_fit_double_gaussian_errors_on_threshold_separation_and_means():
    x, y = _histogram_data()
    r = fit_double_gaussian(x, y)
    se, d = r.stderr, r.derived
    lo, hi = sorted(("mu1", "mu2"), key=lambda k: r.params[k])
    assert d["mu_low_stderr"] == pytest.approx(se[lo], rel=1e-9)
    assert d["mu_high_stderr"] == pytest.approx(se[hi], rel=1e-9)
    names = ["a1", "mu1", "sigma1", "a2", "mu2", "sigma2"]

    def sep(p):
        return abs(p["mu2"] - p["mu1"]) / ((abs(p["sigma1"]) + abs(p["sigma2"])) / 2.0)

    def ref(fn):
        return _independent_sigma(_double_gaussian_model, x, y, r.params, names, fn)

    assert d["threshold_stderr"] == pytest.approx(
        ref(lambda p: (p["mu1"] + p["mu2"]) / 2.0), rel=1e-3
    )
    assert d["separation_stderr"] == pytest.approx(ref(sep), rel=1e-3)


def test_fit_lorentzian_width_and_correlated_Q_errors():
    f, y = _lorentzian_data()
    r = fit_lorentzian(f, y)
    se, d = r.stderr, r.derived
    assert np.isfinite(se["hwhm"]) and se["hwhm"] > 0
    assert d["f0_stderr"] == pytest.approx(se["f0"], rel=1e-9)
    assert d["hwhm_stderr"] == pytest.approx(se["hwhm"], rel=1e-9)
    assert d["fwhm_stderr"] == pytest.approx(2 * se["hwhm"], rel=1e-9)
    ref = _independent_sigma(
        _lorentzian_model, f, y, r.params, ["y0", "amp", "f0", "hwhm"],
        lambda p: p["f0"] / (2 * abs(p["hwhm"])),
    )
    assert d["Q_loaded_stderr"] == pytest.approx(ref, rel=1e-3)


def test_fit_fano_width_asymmetry_and_correlated_Q_errors():
    f, y = _fano_data()
    r = fit_fano(f, y)
    se, d = r.stderr, r.derived
    assert np.isfinite(se["gamma"]) and se["gamma"] > 0
    assert d["f0_stderr"] == pytest.approx(se["f0"], rel=1e-9)
    assert d["gamma_stderr"] == pytest.approx(se["gamma"], rel=1e-9)
    assert d["fwhm_stderr"] == pytest.approx(2 * se["gamma"], rel=1e-9)
    assert d["q_fano_stderr"] == pytest.approx(se["q"], rel=1e-9)
    ref = _independent_sigma(
        _fano_model, f, y, r.params, ["bg", "amp", "q", "f0", "gamma"],
        lambda p: p["f0"] / (2 * abs(p["gamma"])),
    )
    assert d["Q_loaded_stderr"] == pytest.approx(ref, rel=1e-3)


def test_fit_resonator_circle_errors_and_documented_exemptions():
    """f0 / Q_loaded come from the lmfit phase-vs-frequency fit, so they (and fwhm = f0/Q_loaded)
    carry errors. Q_internal / Q_coupling / phi also lean on the algebraic circle fit and the
    off-resonant-point geometry, which have no covariance — they are present-with-nan (exempt,
    documented on the fitter and in docs/fitters.md), never missing."""
    r = fit_resonator_circle(*_notch_data(), geometry="notch")
    se, d = r.stderr, r.derived
    assert np.isfinite(se["Q_loaded"]) and np.isfinite(se["f0"])
    assert d["f0_stderr"] == pytest.approx(se["f0"], rel=1e-9)
    assert d["Q_loaded_stderr"] == pytest.approx(se["Q_loaded"], rel=1e-9)
    # fwhm = f0/Ql: the correlated error is bounded by the |ρ| ≤ 1 extremes of the two terms
    a = se["f0"] / d["Q_loaded"]
    b = d["f0"] * se["Q_loaded"] / d["Q_loaded"] ** 2
    assert abs(a - b) * (1 - 1e-9) <= d["fwhm_stderr"] <= (a + b) * (1 + 1e-9)
    for key in ("Q_internal", "Q_coupling", "phi"):
        assert math.isnan(d[f"{key}_stderr"])

# --- Monte Carlo coverage: the reported error is an honest 1σ ------------------------------
#
# For a fixed seed list, the fraction of fits whose |estimate − truth| ≤ reported stderr must
# sit near the Gaussian 1σ rate (68.3%). With 200 seeds the binomial spread is ±3.3%, so the
# 55–80% band is ≥3.5σ wide on either side. Observed on five disjoint 200-seed lists (bases
# 0, 1000, 20000, 31337, 77777): fit_rabi 0.645–0.74, fit_exp 0.665–0.75. The assertion is
# against the fit's own stderr, not a fixed tolerance, so it is robust to which seeds are drawn.

_MC_SEEDS = range(200)
_MC_BAND = (0.55, 0.80)


def _coverage(hits) -> float:
    return float(np.mean(list(hits)))


def test_fit_rabi_pi_gain_stderr_covers_truth_at_one_sigma():
    true_period = 20000.0

    def hit(seed):
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 30000, 61)
        y = 0.5 + 0.4 * np.cos(2 * np.pi * x / true_period) * np.exp(-x / 80000.0)
        r = fit_rabi(x, y + rng.normal(0, 0.03, x.size))
        return abs(r.derived["pi_gain"] - true_period / 2) <= r.derived["pi_gain_stderr"]

    cov = _coverage(hit(s) for s in _MC_SEEDS)
    assert _MC_BAND[0] <= cov <= _MC_BAND[1], f"pi_gain 1σ coverage {cov:.3f}"


def test_fit_exp_T1_stderr_covers_truth_at_one_sigma():
    true_T1 = 80.0

    def hit(seed):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 300.0, 61)
        y = 0.1 + 0.85 * np.exp(-t / true_T1)
        r = fit_exp(t, y + rng.normal(0, 0.03, t.size))
        return abs(r.derived["T1"] - true_T1) <= r.derived["T1_stderr"]

    cov = _coverage(hit(s) for s in _MC_SEEDS)
    assert _MC_BAND[0] <= cov <= _MC_BAND[1], f"T1 1σ coverage {cov:.3f}"
