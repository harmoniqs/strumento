"""Tests for the core fitters (plan Task 11, D4).

Pure: fits synthetic data, no qick. (Cross-check against the v1 fitting.py oracle is Task 18.)
"""

import numpy as np
import pytest

from strumento.core.fitters import FitResult, fit_rabi


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
