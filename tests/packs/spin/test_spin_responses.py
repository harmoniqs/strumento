"""Spin response model tests — fit recovery from simulated data (pure — no qick).

The closed-loop invariant: a response model generates synthetic data from a known truth;
the matching fitter recovers the truth parameters within tolerance. This is the spin
equivalent of cqed's test_responses.py.

Pure at import: exercises the response models and fitters with no simulator or board.
"""

from __future__ import annotations

import numpy as np
import pytest

# Import the responses module to trigger @responds registrations
import strumento.packs.spin.responses as spin_responses  # noqa: F401
from strumento.core.device import Device
from strumento.core.response import SimContext
from strumento.packs.spin.experiments.cpmg import CPMG
from strumento.packs.spin.experiments.exchange_oscillation import ExchangeOscillation
from strumento.packs.spin.experiments.hahn_echo import HahnEcho
from strumento.packs.spin.experiments.spin_rabi import SpinRabi
from strumento.packs.spin.experiments.spin_ramsey import SpinRamsey
from strumento.packs.spin.experiments.spin_t1 import SpinT1
from strumento.packs.spin.responses import params, truth_from_device
from tests._paths import DEVICES

DEVICES_DIR = DEVICES
SPIN_DEMO_YAML = DEVICES_DIR / "spin_demo" / "device.yaml"
HOLE_DEMO_YAML = DEVICES_DIR / "spin_hole_demo" / "device.yaml"

# Tolerance for fit recovery (relative error).
# Response models are exact closed forms of what fitters fit, so recovery should be tight.
TOLERANCE = 0.02  # 2% — tighter than cqed's 5% because no shot noise in these tests


@pytest.fixture(scope="module")
def dev():
    return Device.load(SPIN_DEMO_YAML)


@pytest.fixture(scope="module")
def hole_dev():
    return Device.load(HOLE_DEMO_YAML)


def _ctx(dev, experiment, xpts, **truth_overrides) -> SimContext:
    """Build a SimContext for testing (no soc needed — just the experiment + truth)."""
    return SimContext(
        experiment=experiment,
        device=dev,
        xpts=tuple(float(x) for x in xpts),
        truth=truth_overrides or None,
        seq=None,
    )


def _pops(response) -> np.ndarray:
    """Extract the population array from a SimResponse."""
    return np.asarray(response.populations, dtype=float)


# =============================================================================
# Truth resolution
# =============================================================================


class TestTruthResolution:
    """Truth params resolve correctly from defaults → device → armed overrides."""

    def test_defaults_present(self):
        p = truth_from_device(None)
        assert p["T1"] == 50.0
        assert p["T2_star"] == 2.0
        assert p["readout_contrast"] == 0.96

    def test_device_overrides_defaults(self, hole_dev):
        p = truth_from_device(hole_dev)
        # hole_demo has T1=50, T2*=2.5, T2_echo=15 in calibration
        assert p["T1"] == 50.0
        assert p["T2_star"] == 2.5
        assert p["T2_echo"] == 15.0

    def test_armed_truth_overrides_device(self, hole_dev):
        ctx = _ctx(hole_dev, SpinT1(hole_dev, dot_name="h1"), [0], T1=123.0)
        p = params(ctx)
        assert p["T1"] == 123.0  # armed override wins

    def test_armed_truth_overrides_default(self, dev):
        ctx = _ctx(dev, SpinT1(dev, dot_name="q0"), [0], T1=77.0)
        p = params(ctx)
        assert p["T1"] == 77.0


# =============================================================================
# Fit recovery — SpinT1
# =============================================================================


class TestSpinT1Response:
    """SpinT1 response generates an exponential; fit_exp recovers T1."""

    def test_shape_and_bounds(self, dev):
        t = np.linspace(0.0, 200.0, 101)
        resp = spin_responses.spin_t1_response(_ctx(dev, SpinT1(dev, dot_name="q0"), t, T1=68.0))
        pops = _pops(resp)
        assert pops.shape == (101,)
        assert pops[0] > pops[-1]  # decays
        assert np.all(pops >= 0) and np.all(pops <= 1)

    def test_fit_recovery(self, dev):
        from strumento.core.fitters import fit_exp

        t = np.linspace(0.0, 300.0, 101)
        pops = _pops(spin_responses.spin_t1_response(
            _ctx(dev, SpinT1(dev, dot_name="q0"), t, T1=68.0)
        ))
        fit = fit_exp(t, pops)
        assert fit.success
        assert fit.derived["tau"] == pytest.approx(68.0, rel=TOLERANCE)


# =============================================================================
# Fit recovery — SpinRabi
# =============================================================================


class TestSpinRabiResponse:
    """SpinRabi response generates a damped oscillation; fit_rabi recovers period."""

    def test_shape_and_oscillation(self, dev):
        t = np.linspace(0.0, 0.5, 201)
        resp = spin_responses.spin_rabi_response(
            _ctx(dev, SpinRabi(dev, dot_name="q0"), t, omega_rabi_mhz=20.0)
        )
        pops = _pops(resp)
        assert pops.shape == (201,)
        # Should oscillate — check it crosses the midpoint
        mid = (pops.max() + pops.min()) / 2
        crossings = np.diff(np.sign(pops - mid))
        assert np.count_nonzero(crossings) > 4

    def test_fit_recovery(self, dev):
        from strumento.core.fitters import fit_rabi

        t = np.linspace(0.0, 0.5, 201)
        omega = 20.0  # MHz — period = 0.05 µs
        pops = _pops(spin_responses.spin_rabi_response(
            _ctx(dev, SpinRabi(dev, dot_name="q0"), t, omega_rabi_mhz=omega, T_Rabi=5.0)
        ))
        fit = fit_rabi(t, pops)
        assert fit.success
        # period = 1/omega; pi_gain = period/2 = 1/(2*omega)
        expected_period = 1.0 / omega
        assert fit.derived["period"] == pytest.approx(expected_period, rel=TOLERANCE)


# =============================================================================
# Fit recovery — SpinRamsey
# =============================================================================


class TestSpinRamseyResponse:
    """SpinRamsey response generates a detuned fringe; fit recovers T2* and detuning."""

    def test_shape_and_fringe(self, dev):
        tau = np.linspace(0.0, 5.0, 201)
        resp = spin_responses.spin_ramsey_response(
            _ctx(dev, SpinRamsey(dev, dot_name="q0"), tau, T2_star=2.0, ramsey_detuning=1.5)
        )
        pops = _pops(resp)
        assert pops.shape == (201,)
        # Should oscillate at 1.5 MHz — ~7 full cycles in 5 µs
        mid = (pops.max() + pops.min()) / 2
        crossings = np.diff(np.sign(pops - mid))
        assert np.count_nonzero(crossings) > 10

    def test_fit_recovery(self, dev):
        from strumento.core.fitters import fit_decaysin_detuned

        tau = np.linspace(0.0, 8.0, 301)
        t2_star = 2.5
        detuning = 1.2  # MHz
        pops = _pops(spin_responses.spin_ramsey_response(
            _ctx(dev, SpinRamsey(dev, dot_name="q0"), tau, T2_star=t2_star, ramsey_detuning=detuning)
        ))
        fit = fit_decaysin_detuned(tau, pops)
        assert fit.success
        assert fit.derived["tau"] == pytest.approx(t2_star, rel=TOLERANCE)
        assert fit.derived["freq"] == pytest.approx(detuning, rel=TOLERANCE)


# =============================================================================
# Fit recovery — HahnEcho
# =============================================================================


class TestHahnEchoResponse:
    """HahnEcho response generates a decay vs half-evolution time; fit_exp recovers T2_echo."""

    def test_fit_recovery(self, dev):
        from strumento.core.fitters import fit_exp

        tau = np.linspace(0.0, 50.0, 101)
        t2_echo = 18.0
        pops = _pops(spin_responses.hahn_echo_response(
            _ctx(dev, HahnEcho(dev, dot_name="q0"), tau, T2_echo=t2_echo)
        ))
        fit = fit_exp(tau, pops)
        assert fit.success
        # Decay constant from fit: the model is exp(-2τ/T2e), fit sees exp(-τ/tau_fit)
        # so tau_fit = T2_echo / 2
        assert fit.derived["tau"] == pytest.approx(t2_echo / 2.0, rel=TOLERANCE)


# =============================================================================
# Fit recovery — CPMG
# =============================================================================


class TestCPMGResponse:
    """CPMG response generates a decay; fit_exp recovers tau_eff, analyzer recovers T2_DD."""

    def test_fit_recovery(self, dev):
        from strumento.core.fitters import fit_exp

        tau = np.linspace(0.0, 100.0, 101)
        t2_dd = 35.0
        n_pi = 4  # default
        pops = _pops(spin_responses.cpmg_response(
            _ctx(dev, CPMG(dev, dot_name="q0"), tau, T2_DD=t2_dd)
        ))
        fit = fit_exp(tau, pops)
        assert fit.success
        # fit_exp reports tau_eff = T2_DD / (2*N_pi) on the tau axis
        tau_eff_expected = t2_dd / (2.0 * n_pi)
        assert fit.derived["tau"] == pytest.approx(tau_eff_expected, rel=TOLERANCE)

    def test_analyzer_recovers_t2_dd(self, dev):
        """The CPMG analyzer multiplies by 2*N to recover the true T2_DD."""

        tau = np.linspace(0.0, 100.0, 101)
        t2_dd = 35.0
        n_pi = 4
        pops = _pops(spin_responses.cpmg_response(
            _ctx(dev, CPMG(dev, dot_name="q0", n_pi=n_pi), tau, T2_DD=t2_dd)
        ))
        # Simulate the analyzer path: fit on tau axis, then scale
        from strumento.core.result import MeasurementResult
        result = MeasurementResult(xpts=tau, avgi=pops.reshape(1, -1), avgq=np.zeros((1, len(tau))))
        exp = CPMG(dev, dot_name="q0", n_pi=n_pi)
        fit = exp.analyze(result)
        assert fit.success
        assert fit.derived["T2_DD"] == pytest.approx(t2_dd, rel=TOLERANCE)


# =============================================================================
# Fit recovery — ExchangeOscillation
# =============================================================================


class TestExchangeOscillationResponse:
    """Exchange oscillation generates a J-frequency oscillation; fit_rabi recovers period."""

    def test_fit_recovery(self, dev):
        from strumento.core.fitters import fit_rabi

        t = np.linspace(0.0, 0.2, 201)
        j_mhz = 50.0  # period = 0.02 µs
        pops = _pops(spin_responses.exchange_oscillation_response(
            _ctx(dev, ExchangeOscillation(dev, dot_name="q0", barrier_name="b0"), t,
                 J_mhz=j_mhz, T_J=3.0)
        ))
        fit = fit_rabi(t, pops)
        assert fit.success
        expected_period = 1.0 / j_mhz
        assert fit.derived["period"] == pytest.approx(expected_period, rel=TOLERANCE)


# =============================================================================
# Registration — all models are discoverable
# =============================================================================


class TestRegistration:
    """All spin response models are registered and resolvable."""

    @pytest.mark.parametrize("cls", [SpinT1, SpinRabi, SpinRamsey, HahnEcho, CPMG, ExchangeOscillation])
    def test_registered(self, cls):
        from strumento.core.response import _RESPONSES

        assert cls in _RESPONSES, f"{cls.__name__} not in response registry"
