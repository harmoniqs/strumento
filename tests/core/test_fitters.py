"""Tests for the core fitters (D4).

Pure: fits synthetic data, no qick.
"""

import math

import numpy as np
import pytest
from scipy.special import erf

from strumento.core.fitters import (
    FitResult,
    fit_damped_rabi,
    fit_decaysin_detuned,
    fit_exp,
    fit_number_splitting_comb,
    fit_rabi,
    fit_time_of_flight,
)


def test_fit_rabi_recovers_period_and_pi_gain():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 30000, 121)
    true_period = 20000.0
    y = 0.5 + 0.4 * np.cos(2 * np.pi * x / true_period) * np.exp(-x / 80000)
    y = y + rng.normal(0, 0.01, x.size)
    r = fit_rabi(x, y)
    assert isinstance(r, FitResult)
    assert r.success
    assert r.derived["period"] == pytest.approx(true_period, rel=0.05)
    assert r.derived["pi_gain"] == pytest.approx(true_period / 2, rel=0.05)


def test_fit_rabi_exposes_params_and_stderr():
    x = np.linspace(0, 30000, 121)
    y = 0.5 + 0.4 * np.cos(2 * np.pi * x / 20000)
    r = fit_rabi(x, y)
    assert {"offset", "amp", "period", "phase", "tau"} <= set(r.params)
    assert "period" in r.stderr


# --- T1 (exponential decay) ------------------------------------------------


def test_fit_exp_recovers_T1():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 300.0, 121)  # us
    true_T1 = 80.0
    y = 0.1 + 0.85 * np.exp(-t / true_T1) + rng.normal(0, 0.01, t.size)
    r = fit_exp(t, y)
    assert isinstance(r, FitResult)
    assert r.success
    assert r.derived["T1"] == pytest.approx(true_T1, rel=0.05)
    assert {"y0", "amp", "tau"} <= set(r.params)


def test_fit_exp_handles_rising_exponential():
    t = np.linspace(0, 300.0, 121)
    true_tau = 60.0
    y = 0.9 - 0.7 * np.exp(-t / true_tau)  # rises toward 0.9
    r = fit_exp(t, y)
    assert r.derived["T1"] == pytest.approx(true_tau, rel=0.05)


# --- T2 Ramsey (detuned decaying sinusoid) ---------------------------------


def test_fit_decaysin_detuned_recovers_detuning_and_T2():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 20.0, 401)  # us
    true_det = 1.5  # MHz
    true_T2 = 8.0  # us
    y = (
        0.5
        + 0.4 * np.exp(-t / true_T2) * np.cos(2 * np.pi * true_det * t + 0.3)
        + rng.normal(0, 0.008, t.size)
    )
    r = fit_decaysin_detuned(t, y)
    assert r.success
    assert r.derived["detuning"] == pytest.approx(true_det, rel=0.03)
    assert r.derived["T2"] == pytest.approx(true_T2, rel=0.1)
    assert r.derived["freq"] == pytest.approx(true_det, rel=0.03)


# --- damped Rabi (time-domain) ---------------------------------------------


def test_fit_damped_rabi_recovers_freq_and_pi_time():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 2.0, 201)  # us
    true_f = 5.0  # MHz Rabi frequency
    y = (
        0.5
        - 0.45 * np.exp(-t / 3.0) * np.cos(2 * np.pi * true_f * t)
        + rng.normal(0, 0.01, t.size)
    )
    r = fit_damped_rabi(t, y)
    assert r.success
    assert r.derived["rabi_freq"] == pytest.approx(true_f, rel=0.03)
    assert r.derived["pi_time"] == pytest.approx(1 / (2 * true_f), rel=0.03)


# --- time-of-flight (arrival-edge step) — issue #104 ------------------------


def _synthetic_tof_trace(rng, *, t0, sigma, noise_floor, settled, span_us=5.0, n=501, noise=0.01):
    """A decimated-magnitude trace: flat noise floor, then an erf-smoothed step at ``t0``."""
    x = np.linspace(0, span_us, n)
    y = noise_floor + (settled - noise_floor) * 0.5 * (1.0 + erf((x - t0) / (np.sqrt(2.0) * sigma)))
    y = y + rng.normal(0, noise, x.size)
    return x, y


def test_fit_time_of_flight_recovers_the_arrival_edge():
    rng = np.random.default_rng(4)
    true_tof = 0.42  # us
    x, y = _synthetic_tof_trace(rng, t0=true_tof, sigma=0.03, noise_floor=0.02, settled=0.8)
    r = fit_time_of_flight(x, y)
    assert isinstance(r, FitResult)
    assert r.success
    assert r.derived["tof_us"] == pytest.approx(true_tof, abs=0.02)


def test_fit_time_of_flight_exposes_params_and_stderr():
    rng = np.random.default_rng(5)
    x, y = _synthetic_tof_trace(rng, t0=0.5, sigma=0.02, noise_floor=0.0, settled=1.0)
    r = fit_time_of_flight(x, y)
    assert {"offset", "amp", "t0", "sigma"} <= set(r.params)
    assert "t0" in r.stderr


def test_fit_time_of_flight_recovers_edge_at_different_locations_and_noise_floors():
    """The fitter must find the edge wherever it is, not just at one convenient offset."""
    rng = np.random.default_rng(6)
    for true_tof, noise_floor, settled in [(0.15, 0.0, 0.5), (1.8, 0.3, 1.2), (3.2, -0.1, 0.4)]:
        x, y = _synthetic_tof_trace(
            rng, t0=true_tof, sigma=0.04, noise_floor=noise_floor, settled=settled
        )
        r = fit_time_of_flight(x, y)
        assert r.derived["tof_us"] == pytest.approx(true_tof, abs=0.03), (
            f"tof={true_tof}, noise_floor={noise_floor}, settled={settled}"
        )


def test_fit_time_of_flight_handles_a_flat_trace_without_raising():
    """Degenerate input (e.g. an unarmed mock's all-zero decimated trace) must not crash the
    fitter — the routine's shape/wiring plumbing has to survive it even with nothing to find."""
    x = np.linspace(0, 5.0, 1536)
    y = np.zeros_like(x)
    r = fit_time_of_flight(x, y)
    assert isinstance(r, FitResult)
    assert np.isfinite(r.derived["tof_us"])


# --- number-splitting comb (photon-number-resolved qubit spectrum) ----------


def _synthetic_number_comb(
    rng, *, f_ge, chi, nbar=2.0, gamma=0.12, span_below=4.0, span_above=1.0, n=201, noise=0.01
):
    """A Poisson-weighted Lorentzian comb: peaks at ``f_ge - k·chi`` weighted by ``P(k; nbar)``.

    Independent oracle for :func:`fit_number_splitting_comb` — the same physics the number-
    splitting response injects (a coherent state's photon distribution shifting the qubit by
    ``k·chi``), built here with a plain Poisson weight so the test does not lean on the fitter's
    own internals.
    """
    f = np.linspace(f_ge - span_below, f_ge + span_above, n)
    nmax = 6
    comb = np.zeros_like(f)
    for k in range(nmax + 1):
        w = nbar**k * math.exp(-nbar) / math.factorial(k)
        comb = comb + w / (1.0 + ((f - (f_ge - k * chi)) / gamma) ** 2)
    y = 0.02 + 0.9 * comb + rng.normal(0, noise, f.size)
    return f, y


def test_fit_number_splitting_comb_recovers_chi_from_a_far_seed():
    """The regression guard: the comb fit must recover the true peak SPACING even when seeded
    from a stored belief ~5x off. The comb objective has spacing-alias local minima, so a single
    belief-seeded start sticks in its own basin — the failure being guarded (believed chi
    0.28 MHz, true chi 1.5 MHz → the fit returned 0.28, a chi 5x too small, and the loose window
    gate passed anyway). Recovery requires the data-driven multi-start."""
    rng = np.random.default_rng(11)
    f_ge = 3566.0
    true_chi = 1.5
    f, y = _synthetic_number_comb(rng, f_ge=f_ge, chi=true_chi, gamma=0.2)
    r = fit_number_splitting_comb(f, y, f_ge, chi_seed=0.28)  # seed 5x too small
    assert isinstance(r, FitResult)
    assert r.success
    assert r.derived["chi"] == pytest.approx(true_chi, rel=0.1)
    assert r.derived["chi_estimate"] == pytest.approx(-true_chi, rel=0.1)


def test_fit_number_splitting_comb_recovers_small_chi_from_a_large_seed():
    """The reverse direction: a small true spacing must not be overwritten by a large seed —
    the multi-start must not bias toward wide combs. A true chi ≈ 0.28 MHz comb seeded from a
    stale large belief must still land on the narrow comb."""
    rng = np.random.default_rng(12)
    f_ge = 3566.0
    true_chi = 0.28
    f, y = _synthetic_number_comb(rng, f_ge=f_ge, chi=true_chi, gamma=0.05, n=401)
    r = fit_number_splitting_comb(f, y, f_ge, chi_seed=1.5)  # seed 5x too large
    assert r.success
    assert r.derived["chi"] == pytest.approx(true_chi, rel=0.15)


def test_fit_number_splitting_comb_exposes_derived_and_survives_flat_input():
    """The typed contract: derived carries chi/chi_estimate/nbar/gamma, and a degenerate flat
    input (mock zero-IQ shape) returns a finite typed result rather than raising."""
    rng = np.random.default_rng(13)
    f_ge = 3566.0
    f, y = _synthetic_number_comb(rng, f_ge=f_ge, chi=0.9)
    r = fit_number_splitting_comb(f, y, f_ge, chi_seed=0.9)
    assert {"chi", "chi_estimate", "nbar", "gamma"} <= set(r.derived)
    flat = fit_number_splitting_comb(f, np.zeros_like(f), f_ge, chi_seed=0.5)
    assert isinstance(flat, FitResult)
    assert np.isfinite(flat.derived["chi"])
