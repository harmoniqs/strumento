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
