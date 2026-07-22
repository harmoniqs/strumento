"""Tests for the core fitters (plan Task 11 + Task 12, D4).

Pure: fits synthetic data, no qick. (Cross-check against the v1 fitting.py oracle is Task 18,
tests/oracle/test_fitters_vs_v1.py.)
"""

import numpy as np
import pytest

from strumento.core.fitters import (
    FitResult,
    fit_damped_rabi,
    fit_decaysin_detuned,
    fit_exp,
    fit_rabi,
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
