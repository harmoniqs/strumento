"""Comprehensive tests for the 28 spin experiments.

Three levels of validation:
- Level 1: Construction — all 28 experiments instantiate without error.
- Level 2: Sequence validity — pulse-program experiments produce Seq with play ops.
- Level 3: Analyze — experiments using standard fitters return FitResult from synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.dc import DCAxis, DCSource, MockVoltageSource
from strumento.core.device import Device
from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult
from strumento.packs.spin.experiments.allxy import AllXY
from strumento.packs.spin.experiments.charge_noise_spectroscopy import ChargeNoiseSpectroscopy
from strumento.packs.spin.experiments.composite_pulse_calibration import CompositePulseCalibration
from strumento.packs.spin.experiments.coulomb_diamond import CoulombDiamond
from strumento.packs.spin.experiments.cpmg import CPMG
from strumento.packs.spin.experiments.dispersive_gate_calibration import DispersiveGateCalibration
from strumento.packs.spin.experiments.edsr_spectroscopy import EDSRSpectroscopy
from strumento.packs.spin.experiments.elzerman_optimize import ElzermanOptimize
from strumento.packs.spin.experiments.exchange_gate_calibration import ExchangeGateCalibration
from strumento.packs.spin.experiments.exchange_oscillation import ExchangeOscillation
from strumento.packs.spin.experiments.g_factor_map import GFactorMap
from strumento.packs.spin.experiments.gradient_calibration import GradientCalibration
from strumento.packs.spin.experiments.hahn_echo import HahnEcho
from strumento.packs.spin.experiments.interleaved_rb import InterleavedRB
from strumento.packs.spin.experiments.leakage_benchmark import LeakageBenchmark
from strumento.packs.spin.experiments.pinch_off import PinchOff
from strumento.packs.spin.experiments.psb_calibration import PSBCalibration
from strumento.packs.spin.experiments.psb_histogram import PSBHistogram
from strumento.packs.spin.experiments.randomized_benchmarking import RandomizedBenchmarking
from strumento.packs.spin.experiments.rf_reflectometry_tune import RFReflectometryTune
from strumento.packs.spin.experiments.sensor_tune import SensorTune
from strumento.packs.spin.experiments.spin_rabi import SpinRabi
from strumento.packs.spin.experiments.spin_ramsey import SpinRamsey
from strumento.packs.spin.experiments.spin_t1 import SpinT1
from strumento.packs.spin.experiments.sweet_spot_tune import SweetSpotTune
from strumento.packs.spin.experiments.tunnel_coupling_tune import TunnelCouplingTune
from strumento.packs.spin.experiments.tunnel_rate_tune import TunnelRateTune
from strumento.packs.spin.experiments.virtual_gate_calibration import VirtualGateCalibration
from tests._paths import DEVICES

DEVICES_DIR = DEVICES
SPIN_DEMO_YAML = DEVICES_DIR / "spin_demo" / "device.yaml"
SPIN_HOLE_YAML = DEVICES_DIR / "spin_hole_demo" / "device.yaml"
SPIN_ST_YAML = DEVICES_DIR / "spin_st_demo" / "device.yaml"


@pytest.fixture(scope="module")
def dev():
    """LD encoding device: dot q0, barrier b0, plunger p0."""
    return Device.load(SPIN_DEMO_YAML)


@pytest.fixture
def dc_source():
    """A mock DCSource with common spin-qubit axes for DC-sweep experiments."""
    src = MockVoltageSource()
    axes = {
        "p0": DCAxis(channel=0, max_v=2.0),
        "b0": DCAxis(channel=1, max_v=2.0),
        "v_sd": DCAxis(channel=2, max_v=2.0),
        "sensor": DCAxis(channel=3, max_v=2.0),
    }
    return DCSource(src, axes)


@pytest.fixture(scope="module")
def dev_hole():
    """Hole encoding device: dot h1, barrier b1, plunger p1."""
    return Device.load(SPIN_HOLE_YAML)


@pytest.fixture(scope="module")
def dev_st():
    """ST encoding device: dots left/right, barrier b_lr, plunger p_left."""
    return Device.load(SPIN_ST_YAML)


# ──────────────────────────────────────────────────────────────────────────────
# Level 1: Construction — all 28 experiments instantiate without error
# ──────────────────────────────────────────────────────────────────────────────

class TestExperimentConstruction:
    """Every experiment constructs without error given a valid device."""

    # -- Experiments that take a device + dot_name (LD dot "q0") --

    @pytest.mark.parametrize("cls", [
        SpinT1,
        SpinRabi,
        SpinRamsey,
        HahnEcho,
        CPMG,
        AllXY,
        EDSRSpectroscopy,
        ElzermanOptimize,
        GFactorMap,
        GradientCalibration,
        SweetSpotTune,
        ChargeNoiseSpectroscopy,
        CompositePulseCalibration,
        LeakageBenchmark,
        RandomizedBenchmarking,
        InterleavedRB,
        PSBCalibration,
        TunnelRateTune,
    ])
    def test_dot_based_experiment_constructs(self, dev, cls):
        """Experiment classes taking dot_name construct with the LD demo device."""
        exp = cls(dev, dot_name="q0")
        assert exp is not None
        assert exp.device is dev

    def test_exchange_oscillation_constructs(self, dev):
        """ExchangeOscillation needs dot_name + barrier_name."""
        exp = ExchangeOscillation(dev, dot_name="q0", barrier_name="b0")
        assert exp is not None
        assert exp.device is dev

    def test_exchange_gate_calibration_constructs(self, dev):
        """ExchangeGateCalibration needs control_dot + target_dot + barrier_name.

        Uses q0 for both since spin_demo has only one dot — construction test only.
        """
        exp = ExchangeGateCalibration(
            dev, control_dot="q0", target_dot="q0", barrier_name="b0"
        )
        assert exp is not None
        assert exp.device is dev

    # -- DC-sweep experiments (standalone classes with DCSource, no device) --

    def test_pinch_off_constructs(self, dc_source):
        """PinchOff constructs with a DCSource + probe."""
        exp = PinchOff(dc_source, gate_name="p0", probe=lambda: 0.0)
        assert exp is not None
        assert exp.dc is dc_source

    def test_coulomb_diamond_constructs(self, dc_source):
        """CoulombDiamond constructs with a DCSource + probe."""
        exp = CoulombDiamond(dc_source, plunger_axis="p0", bias_axis="v_sd", probe=lambda: 0.0)
        assert exp is not None
        assert exp.dc is dc_source

    def test_sensor_tune_constructs(self, dc_source):
        """SensorTune constructs with a DCSource + probe."""
        exp = SensorTune(dc_source, sensor_name="sensor", probe=lambda: 0.0)
        assert exp is not None
        assert exp.dc is dc_source

    def test_virtual_gate_calibration_constructs(self, dc_source):
        """VirtualGateCalibration constructs with a DCSource + probe."""
        exp = VirtualGateCalibration(dc_source, gate_a="p0", gate_b="b0", probe=lambda: 0.0)
        assert exp is not None
        assert exp.dc is dc_source

    def test_tunnel_coupling_tune_constructs(self, dc_source):
        """TunnelCouplingTune constructs with a DCSource + probe."""
        exp = TunnelCouplingTune(dc_source, barrier_name="b0", probe=lambda: 0.0)
        assert exp is not None
        assert exp.dc is dc_source

    def test_dispersive_gate_calibration_constructs(self, dc_source):
        """DispersiveGateCalibration constructs with a DCSource + probe."""
        exp = DispersiveGateCalibration(dc_source, gate_name="p0", probe=lambda: 0.0)
        assert exp is not None
        assert exp.dc is dc_source

    def test_rf_reflectometry_tune_constructs(self):
        """RFReflectometryTune constructs with probe + set_frequency (no DCSource)."""
        exp = RFReflectometryTune(probe=lambda: 0.0, set_frequency=lambda f: None)
        assert exp is not None

    def test_psb_histogram_constructs(self):
        """PSBHistogram is an analyzer-only class — no device required."""
        analyzer = PSBHistogram(bins=60)
        assert analyzer is not None
        assert analyzer.bins == 60


# ──────────────────────────────────────────────────────────────────────────────
# Level 2: Sequence validity — pulse-program experiments produce Seq with ops
# ──────────────────────────────────────────────────────────────────────────────

class TestSequenceValidity:
    """Pulse-program experiments return (Seq, ()) with play ops in the sequence."""

    @pytest.mark.parametrize("cls", [
        SpinT1,
        SpinRabi,
        SpinRamsey,
        HahnEcho,
        CPMG,
        AllXY,
        EDSRSpectroscopy,
        SweetSpotTune,
        GradientCalibration,
        ChargeNoiseSpectroscopy,
        CompositePulseCalibration,
    ])
    def test_sequence_has_play_ops(self, dev, cls):
        """Layer 2+ experiments build sequences with play ops before measure."""
        exp = cls(dev, dot_name="q0")
        seq, extra = exp.sequence()
        assert seq is not None
        assert hasattr(seq, "ops")
        # Must have at least a play + measure (>1 op)
        assert len(seq.ops) > 1, f"{cls.__name__} sequence should have play ops"
        assert extra == ()

    def test_exchange_oscillation_sequence(self, dev):
        """ExchangeOscillation builds an exchange pulse + measure."""
        exp = ExchangeOscillation(dev, dot_name="q0", barrier_name="b0")
        seq, extra = exp.sequence()
        assert seq is not None
        assert len(seq.ops) > 1
        assert extra == ()

    def test_exchange_gate_calibration_sequence(self, dev):
        """ExchangeGateCalibration builds pi(ctrl) + hpi(tgt) + exchange + hpi(tgt) + measure."""
        exp = ExchangeGateCalibration(
            dev, control_dot="q0", target_dot="q0", barrier_name="b0"
        )
        seq, extra = exp.sequence()
        assert seq is not None
        # pi + hpi + exchange + hpi + measure = 5 ops
        assert len(seq.ops) >= 5
        assert extra == ()

    def test_randomized_benchmarking_sequence(self, dev):
        """RB builds a Clifford sequence + inversion gate + measure."""
        exp = RandomizedBenchmarking(dev, dot_name="q0")
        seq, extra = exp.sequence()
        assert seq is not None
        assert len(seq.ops) > 1
        assert extra == ()

    def test_interleaved_rb_sequence(self, dev):
        """Interleaved RB builds Clifford-G-...-inversion + measure."""
        exp = InterleavedRB(dev, dot_name="q0")
        seq, extra = exp.sequence()
        assert seq is not None
        assert len(seq.ops) > 1
        assert extra == ()

    def test_leakage_benchmark_sequence(self, dev):
        """LeakageBenchmark builds a gate pulse + measure."""
        exp = LeakageBenchmark(dev, dot_name="q0")
        seq, extra = exp.sequence()
        assert seq is not None
        assert len(seq.ops) > 1
        assert extra == ()

    def test_elzerman_optimize_sequence(self, dev):
        """ElzermanOptimize builds a readout pulse + measure."""
        exp = ElzermanOptimize(dev, dot_name="q0")
        seq, extra = exp.sequence()
        assert seq is not None
        assert len(seq.ops) > 1
        assert extra == ()

    def test_psb_calibration_sequence(self, dev):
        """PSBCalibration builds a PSB pulse + measure."""
        exp = PSBCalibration(dev, dot_name="q0")
        seq, extra = exp.sequence()
        assert seq is not None
        assert len(seq.ops) > 1
        assert extra == ()

    def test_tunnel_rate_tune_sequence(self, dev):
        """TunnelRateTune builds detuning pulse + wait + measure."""
        exp = TunnelRateTune(dev, dot_name="q0")
        seq, extra = exp.sequence()
        assert seq is not None
        assert len(seq.ops) > 1
        assert extra == ()

    def test_g_factor_map_sequence(self, dev):
        """GFactorMap builds Ramsey: hpi + wait + hpi + measure."""
        exp = GFactorMap(dev, dot_name="q0")
        seq, extra = exp.sequence()
        assert seq is not None
        assert len(seq.ops) > 1
        assert extra == ()

    # -- DC-sweep experiments have run() (no sequence) --

    def test_pinch_off_run(self, dc_source):
        """PinchOff.run() sweeps the gate and returns a MeasurementResult."""
        counter = [0]
        def probe():
            counter[0] += 1
            return float(counter[0])
        exp = PinchOff(dc_source, gate_name="p0", voltages=np.linspace(0, -1.0, 5), probe=probe)
        result = exp.run()
        assert isinstance(result, MeasurementResult)
        assert result.avgi.shape == (1, 5)
        assert result.meta["axis"] == "p0"

    def test_coulomb_diamond_run(self, dc_source):
        """CoulombDiamond.run() returns a 2D grid."""
        exp = CoulombDiamond(
            dc_source, plunger_axis="p0", plunger_volts=np.linspace(-0.5, 0.5, 3),
            bias_axis="v_sd", bias_volts=np.linspace(-1, 1, 4), probe=lambda: 1.0,
        )
        result = exp.run()
        assert isinstance(result, MeasurementResult)
        assert result.avgi.shape == (4, 3)

    def test_rf_reflectometry_run(self):
        """RFReflectometryTune.run() sweeps frequency via set_frequency callable."""
        calls = []
        def set_freq(f):
            calls.append(f)
        exp = RFReflectometryTune(
            frequencies=np.linspace(300, 325, 5), probe=lambda: 1.0,
            set_frequency=set_freq,
        )
        result = exp.run()
        assert isinstance(result, MeasurementResult)
        assert result.avgi.shape == (1, 5)
        assert len(calls) == 5


# ──────────────────────────────────────────────────────────────────────────────
# Level 3: Analyze — standard fitters return FitResult from synthetic data
# ──────────────────────────────────────────────────────────────────────────────

class TestAnalyzeWithSyntheticData:
    """Experiments using standard fitters return FitResult(success=True) from clean synthetic data."""

    @staticmethod
    def _make_result(xpts: np.ndarray, signal: np.ndarray) -> MeasurementResult:
        """Build a MeasurementResult with shape (1, N) avgi."""
        return MeasurementResult(
            xpts=xpts,
            avgi=signal.reshape(1, -1),
            avgq=np.zeros((1, len(xpts))),
        )

    def test_spin_t1_analyze(self, dev):
        """SpinT1 fits exponential decay and extracts T1."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(0, 500, 101)
        # Clean exponential decay: amp=1.0, T1=100 us, y0=0.0
        signal = 1.0 * np.exp(-xpts / 100.0) + rng.normal(0, 0.005, 101)
        result = self._make_result(xpts, signal)

        exp = SpinT1(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "T1" in fit.derived or "tau" in fit.derived
        # T1 should be close to 100 us
        t1 = fit.derived.get("T1", fit.derived.get("tau"))
        assert 80.0 < t1 < 120.0, f"Expected T1 ~ 100, got {t1}"

    def test_spin_rabi_analyze(self, dev):
        """SpinRabi fits Rabi oscillation and extracts period."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(0, 0.200, 51)
        # Rabi oscillation: period = 0.050 us, decaying
        period = 0.050
        signal = (
            0.5
            + 0.5 * np.exp(-xpts / 0.5) * np.cos(2 * np.pi * xpts / period)
            + rng.normal(0, 0.01, 51)
        )
        result = self._make_result(xpts, signal)

        exp = SpinRabi(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "period" in fit.derived
        # Period should be close to 0.050 us
        assert 0.040 < fit.derived["period"] < 0.060, (
            f"Expected period ~ 0.050, got {fit.derived['period']}"
        )

    def test_spin_ramsey_analyze(self, dev):
        """SpinRamsey fits damped sinusoid and extracts T2* and detuning."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(0, 10.0, 101)
        # Detuned Ramsey: detuning = 2.0 MHz, T2* = 3.0 us
        freq = 2.0  # MHz (cycles per us)
        tau = 3.0
        signal = (
            0.5
            + 0.4 * np.exp(-xpts / tau) * np.cos(2 * np.pi * freq * xpts)
            + rng.normal(0, 0.005, 101)
        )
        result = self._make_result(xpts, signal)

        exp = SpinRamsey(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "detuning" in fit.derived or "freq" in fit.derived
        assert "T2" in fit.derived or "tau" in fit.derived
        # Detuning should be close to 2.0 MHz
        detuning = fit.derived.get("detuning", fit.derived.get("freq"))
        assert 1.5 < detuning < 2.5, f"Expected detuning ~ 2.0, got {detuning}"

    def test_hahn_echo_analyze(self, dev):
        """HahnEcho fits exponential decay and extracts T2_echo."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(0, 50.0, 101)
        # Echo decay: T2_echo = 15 us
        signal = 1.0 * np.exp(-xpts / 15.0) + rng.normal(0, 0.005, 101)
        result = self._make_result(xpts, signal)

        exp = HahnEcho(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "T1" in fit.derived or "tau" in fit.derived
        t2 = fit.derived.get("T1", fit.derived.get("tau"))
        assert 10.0 < t2 < 20.0, f"Expected T2_echo ~ 15, got {t2}"

    def test_cpmg_analyze(self, dev):
        """CPMG fits exponential decay and extracts T2_DD."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(0.001, 20.0, 101)
        # CPMG decay: T2_DD = 8 us
        signal = 1.0 * np.exp(-xpts / 8.0) + rng.normal(0, 0.005, 101)
        result = self._make_result(xpts, signal)

        exp = CPMG(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "T1" in fit.derived or "tau" in fit.derived
        t2_dd = fit.derived.get("T1", fit.derived.get("tau"))
        assert 5.0 < t2_dd < 12.0, f"Expected T2_DD ~ 8, got {t2_dd}"

    def test_exchange_oscillation_analyze(self, dev):
        """ExchangeOscillation fits Rabi-like oscillation and extracts J_MHz."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(0, 0.500, 101)
        # Exchange oscillation: J = 50 MHz -> period = 0.020 us
        period = 1.0 / 50.0  # 0.020 us
        signal = (
            0.5
            + 0.5 * np.exp(-xpts / 1.0) * np.cos(2 * np.pi * xpts / period)
            + rng.normal(0, 0.01, 101)
        )
        result = self._make_result(xpts, signal)

        exp = ExchangeOscillation(dev, dot_name="q0", barrier_name="b0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "J_MHz" in fit.derived
        assert "t_pi_exchange" in fit.derived
        # J should be close to 50 MHz
        j_mhz = fit.derived["J_MHz"]
        assert 30.0 < j_mhz < 70.0, f"Expected J ~ 50 MHz, got {j_mhz}"

    def test_randomized_benchmarking_analyze(self, dev):
        """RB fits exponential decay of survival probability."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(1, 100, 21)
        # RB decay: p = 0.99 -> tau = -1/ln(0.99) ~ 99.5
        p = 0.99
        signal = 0.5 * p**xpts + 0.5 + rng.normal(0, 0.005, 21)
        result = self._make_result(xpts, signal)

        exp = RandomizedBenchmarking(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "fidelity" in fit.derived
        assert "error_per_clifford" in fit.derived

    def test_interleaved_rb_analyze(self, dev):
        """Interleaved RB fits decay and reports gate fidelity."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(1, 100, 21)
        p = 0.98
        signal = 0.5 * p**xpts + 0.5 + rng.normal(0, 0.005, 21)
        result = self._make_result(xpts, signal)

        exp = InterleavedRB(dev, dot_name="q0", p_reference=0.995)
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "gate_fidelity" in fit.derived
        assert "p_interleaved" in fit.derived

    def test_edsr_spectroscopy_analyze(self, dev):
        """EDSRSpectroscopy finds the resonance peak."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(4995.0, 5005.0, 101)
        # Lorentzian peak at 5000 MHz
        center = 5000.0
        gamma = 0.5
        signal = (
            0.8 * gamma**2 / ((xpts - center) ** 2 + gamma**2)
            + rng.normal(0, 0.005, 101)
        )
        result = self._make_result(xpts, signal)

        exp = EDSRSpectroscopy(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "f_qubit_mhz" in fit.derived
        assert 4998.0 < fit.derived["f_qubit_mhz"] < 5002.0

    def test_allxy_analyze(self, dev):
        """AllXY compares 21-point pattern against ideal and extracts error metrics."""
        rng = np.random.default_rng(42)
        # Simulate close-to-ideal AllXY pattern with small deviations
        from strumento.packs.spin.experiments.allxy import ALLXY_PAIRS

        ideal = np.array([p[2] for p in ALLXY_PAIRS], dtype=float)
        signal = ideal + rng.normal(0, 0.02, len(ideal))
        xpts = np.arange(len(ideal), dtype=float)
        result = self._make_result(xpts, signal)

        exp = AllXY(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "amplitude_error" in fit.derived
        assert "rms_deviation" in fit.derived
        # Small errors expected from the near-ideal synthetic data
        assert abs(fit.derived["amplitude_error"]) < 0.1

    def test_sweet_spot_tune_analyze(self, dev):
        """SweetSpotTune fits Ramsey and extracts T2*."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(0, 20.0, 101)
        freq = 1.5
        tau = 6.0
        signal = (
            0.5
            + 0.4 * np.exp(-xpts / tau) * np.cos(2 * np.pi * freq * xpts)
            + rng.normal(0, 0.005, 101)
        )
        result = self._make_result(xpts, signal)

        exp = SweetSpotTune(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "t2_star_at_ss_us" in fit.derived

    def test_gradient_calibration_analyze(self, dev):
        """GradientCalibration fits S-T0 precession and extracts delta_bz_mhz."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(0, 5.0, 101)
        # S-T0 precession at f_z = 10 MHz, T2* = 2 us
        freq = 10.0
        tau = 2.0
        signal = (
            0.5
            + 0.4 * np.exp(-xpts / tau) * np.cos(2 * np.pi * freq * xpts)
            + rng.normal(0, 0.005, 101)
        )
        result = self._make_result(xpts, signal)

        exp = GradientCalibration(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "delta_bz_mhz" in fit.derived
        assert 8.0 < fit.derived["delta_bz_mhz"] < 12.0

    def test_charge_noise_spectroscopy_analyze(self, dev):
        """ChargeNoiseSpectroscopy extracts noise exponent from coherence decay."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(0.01, 50.0, 101)
        # Gaussian decay envelope (1/f noise): exp(-(tau/T2*)^2)
        t2_star = 10.0
        signal = np.exp(-(xpts / t2_star) ** 2) + rng.normal(0, 0.005, 101)
        result = self._make_result(xpts, signal)

        exp = ChargeNoiseSpectroscopy(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "noise_amplitude" in fit.derived
        assert "exponent_alpha" in fit.derived

    def test_composite_pulse_calibration_analyze(self, dev):
        """CompositePulseCalibration extracts fidelity and error suppression."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(0.8, 1.2, 51)
        # BB1 flat-top profile: high fidelity near amp_scale=1.0, quartic roll-off
        signal = 1.0 - 5.0 * (xpts - 1.0) ** 4 + rng.normal(0, 0.002, 51)
        signal = np.clip(signal, 0, 1)
        result = self._make_result(xpts, signal)

        exp = CompositePulseCalibration(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "fidelity" in fit.derived
        assert "error_suppression" in fit.derived
        assert fit.derived["fidelity"] > 0.9

    def test_leakage_benchmark_analyze(self, dev):
        """LeakageBenchmark extracts leakage rate per gate."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(1, 100, 51)
        # Exponential survival decay: leakage L = 0.005 per gate
        leakage = 0.005
        signal = np.exp(-leakage * xpts) + rng.normal(0, 0.002, 51)
        signal = np.clip(signal, 0.01, None)
        result = self._make_result(xpts, signal)

        exp = LeakageBenchmark(dev, dot_name="q0")
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "leakage_per_gate" in fit.derived
        # Should be close to 0.005
        assert 0.003 < fit.derived["leakage_per_gate"] < 0.008

    def test_psb_histogram_analyze(self):
        """PSBHistogram fits bimodal shot data to double Gaussian."""
        rng = np.random.default_rng(42)
        # Bimodal distribution: two Gaussians at -0.5 and +0.5
        shots_lo = rng.normal(-0.5, 0.15, 500)
        shots_hi = rng.normal(0.5, 0.15, 500)
        shots = np.concatenate([shots_lo, shots_hi])

        analyzer = PSBHistogram(bins=60)
        fit = analyzer.analyze(shots)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "threshold" in fit.derived
        assert "separation" in fit.derived
        # Threshold should be near 0.0 (midpoint of the two peaks)
        assert -0.2 < fit.derived["threshold"] < 0.2

    def test_exchange_gate_calibration_analyze(self, dev):
        """ExchangeGateCalibration fits oscillation and extracts t_cz."""
        rng = np.random.default_rng(42)
        xpts = np.linspace(0, 0.500, 101)
        # Exchange oscillation with J_eff = 20 MHz -> period = 0.050 us
        period = 1.0 / 20.0
        signal = (
            0.5
            + 0.5 * np.exp(-xpts / 2.0) * np.cos(2 * np.pi * xpts / period)
            + rng.normal(0, 0.01, 101)
        )
        result = self._make_result(xpts, signal)

        exp = ExchangeGateCalibration(
            dev, control_dot="q0", target_dot="q0", barrier_name="b0"
        )
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "J_eff_MHz" in fit.derived
        assert "t_cz_us" in fit.derived


# ──────────────────────────────────────────────────────────────────────────────
# Level 3 (extended): DC-sweep experiment analyzers with synthetic data
# ──────────────────────────────────────────────────────────────────────────────

class TestDCSweepAnalyzers:
    """DC-sweep experiments that use custom analysis (not standard fitters)."""

    @staticmethod
    def _make_result(xpts: np.ndarray, signal: np.ndarray) -> MeasurementResult:
        return MeasurementResult(
            xpts=xpts,
            avgi=signal.reshape(1, -1),
            avgq=np.zeros((1, len(xpts))),
        )

    def test_pinch_off_analyze(self, dc_source):
        """PinchOff extracts threshold voltage from conductance drop."""
        xpts = np.linspace(0.0, -2.0, 101)
        # Smooth step: channel pinches off around -1.0 V
        signal = 1.0 / (1.0 + np.exp(10 * (xpts + 1.0)))
        result = self._make_result(xpts, signal)

        exp = PinchOff(dc_source, gate_name="p0", voltages=xpts, probe=lambda: 0.0)
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "v_threshold" in fit.derived

    def test_coulomb_diamond_analyze(self, dc_source):
        """CoulombDiamond extracts charging energy."""
        xpts = np.linspace(-0.5, 0.5, 101)
        # Simulated blockade: suppressed current in center region
        signal = np.ones(101)
        signal[30:70] = 0.05  # blockade region
        result = self._make_result(xpts, signal)

        exp = CoulombDiamond(dc_source, plunger_axis="p0", plunger_volts=xpts,
                             bias_axis="v_sd", bias_volts=np.linspace(-1, 1, 5),
                             probe=lambda: 0.0)
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "charging_energy_mev" in fit.derived
        assert "lever_arm" in fit.derived

    def test_sensor_tune_analyze(self, dc_source):
        """SensorTune finds optimal sensor voltage (max transconductance)."""
        xpts = np.linspace(-1.0, 0.0, 101)
        # QPC-like conductance step
        signal = 1.0 / (1.0 + np.exp(-20 * (xpts + 0.5)))
        result = self._make_result(xpts, signal)

        exp = SensorTune(dc_source, sensor_name="sensor", voltages=xpts, probe=lambda: 0.0)
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "v_optimal" in fit.derived
        assert "transconductance" in fit.derived
        # Steepest point should be near -0.5 V
        assert -0.7 < fit.derived["v_optimal"] < -0.3

    def test_virtual_gate_calibration_analyze(self, dc_source):
        """VirtualGateCalibration extracts cross-capacitance coefficient."""
        xpts = np.linspace(-0.1, 0.1, 101)
        # Charge transition: sharp step at x=0
        signal = 1.0 / (1.0 + np.exp(-50 * xpts))
        result = self._make_result(xpts, signal)

        exp = VirtualGateCalibration(dc_source, gate_a="p0", gate_b="b0",
                                     voltages_a=xpts, probe=lambda: 0.0)
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert "alpha_ij" in fit.derived

    def test_dispersive_gate_calibration_analyze(self, dc_source):
        """DispersiveGateCalibration finds optimal operating point."""
        xpts = np.linspace(-0.3, 0.3, 101)
        # Dispersive signal: steep slope near epsilon=0
        signal = np.tanh(10 * xpts)
        result = self._make_result(xpts, signal)

        exp = DispersiveGateCalibration(dc_source, gate_name="p0", voltages=xpts,
                                        probe=lambda: 0.0)
        fit = exp.analyze(result)

        assert isinstance(fit, FitResult)
        assert fit.success
        assert "epsilon_optimal" in fit.derived
        assert "delta_cq_af" in fit.derived


# ──────────────────────────────────────────────────────────────────────────────
# Bonus: axes() and points
# ──────────────────────────────────────────────────────────────────────────────

class TestAxesConfiguration:
    """Experiments report their sweep axes correctly."""

    @pytest.mark.parametrize("cls,axis_name", [
        (SpinT1, "t"),
        (SpinRabi, "t"),
        (SpinRamsey, "tau"),
        (HahnEcho, "tau"),
        (CPMG, "tau"),
    ])
    def test_default_axes(self, dev, cls, axis_name):
        """Experiments declare their sweep axis from the default Sweep."""
        exp = cls(dev, dot_name="q0")
        axes = exp.axes()
        assert len(axes) == 1
        assert axes[0].name == axis_name

    def test_allxy_axes(self, dev):
        """AllXY has no hardware sweep axes (runs one pair per invocation)."""
        exp = AllXY(dev, dot_name="q0")
        axes = exp.axes()
        assert len(axes) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Single-shot acquire contract for DCSource probe()
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.qick
class TestSingleShotAcquireProbe:
    """Verify that soc.acquire(reps=1) returns a scalar-extractable result suitable as
    the ``probe()`` callable for DCSource experiments (ChargeStability, PinchOff, etc.).

    The contract: ``probe = lambda: float(np.asarray(result.avgi).ravel()[0])`` must
    yield a finite float from a single-rep acquisition of a minimal readout program.
    """

    @pytest.fixture
    def soc_and_dev(self):
        import strumento.packs.spin.responses  # noqa: F401 — triggers response model registration
        from strumento.core.simulator import SimulatorSoc
        soccfg = DEVICES / "_fixtures" / "soccfg_v2_spin.json"
        dev = Device.load(SPIN_DEMO_YAML)
        truth = {"T1": 50.0, "readout_contrast": 0.96, "p_thermal": 0.02}
        soc = SimulatorSoc.from_snapshot(soccfg, truth=truth, seed=7)
        return soc, dev

    def test_single_rep_acquire_returns_scalar_extractable_result(self, soc_and_dev):
        """A 1-rep acquisition of a minimal measure program yields a float for probe()."""
        from strumento.core.program import StrumentoProgram
        from strumento.core.pulses import Seq
        from strumento.core.response import SimContext, SimResponse

        soc, dev = soc_and_dev
        # Minimal program: one readout (the probe point)
        seq = Seq().play(dev.q0.pi()).measure()
        prog = StrumentoProgram(dev, seq=seq, sweeps=(), axes=(), reps=1, soc=soc)

        # Arm with a scalar population model (the probe returns one value)
        def probe_model(ctx):
            return SimResponse(populations=np.array(0.7), blob0=(0.0, 0.0), blob1=(1.0, 0.0), sigma=0.1)

        soc.arm(SimContext(experiment=None, device=dev, xpts=(), seq=seq, model=probe_model))
        result = prog.acquire(soc, progress=False)

        # The probe() contract: extract a single finite float
        value = float(np.asarray(result.avgi).ravel()[0])
        assert np.isfinite(value), "probe() must return a finite float from reps=1"

    def test_probe_callable_works_with_charge_stability(self, soc_and_dev):
        """End-to-end: a SimulatorSoc-backed probe() drives ChargeStability correctly."""
        from strumento.core.program import StrumentoProgram
        from strumento.core.pulses import Seq
        from strumento.core.response import SimContext, SimResponse
        from strumento.packs.spin.experiments.charge_stability import ChargeStability

        soc, dev = soc_and_dev
        dc = DCSource(MockVoltageSource(), axes={
            "p0": DCAxis(channel=0, max_v=2.0),
            "b0": DCAxis(channel=1, max_v=2.0),
        })

        # Build the probe: a single-rep acquire that extracts a scalar
        seq = Seq().play(dev.q0.pi()).measure()
        prog = StrumentoProgram(dev, seq=seq, sweeps=(), axes=(), reps=1, soc=soc)

        def probe_model(ctx):
            return SimResponse(populations=np.array(0.5), blob0=(0.0, 0.0), blob1=(1.0, 0.0), sigma=0.1)

        soc.arm(SimContext(experiment=None, device=dev, xpts=(), seq=seq, model=probe_model))

        def probe():
            result = prog.acquire(soc, progress=False)
            return float(np.asarray(result.avgi).ravel()[0])

        # Run a tiny 3x3 stability map
        cs = ChargeStability(
            dc, x_axis="p0", x_volts=np.linspace(0, 1, 3),
            y_axis="b0", y_volts=np.linspace(0, 1, 3), probe=probe,
        )
        result = cs.run()

        # Verify: 3x3 grid of finite values
        grid = np.asarray(result.avgi)
        assert grid.shape == (3, 3)
        assert np.all(np.isfinite(grid)), "all probe readings must be finite"


# ──────────────────────────────────────────────────────────────────────────────
# DC-sweep integration tests (construct → run with physics probe → analyze)
# ──────────────────────────────────────────────────────────────────────────────

class TestDCSweepIntegration:
    """End-to-end: construct a DCSource experiment with a physics-aware probe, run it,
    and verify the analyzer extracts correct derived quantities from the real sweep."""

    @pytest.fixture
    def dc(self):
        return DCSource(MockVoltageSource(), axes={
            "p0": DCAxis(channel=0, max_v=2.0),
            "b0": DCAxis(channel=1, max_v=2.0),
            "v_sd": DCAxis(channel=2, max_v=2.0),
            "sensor": DCAxis(channel=3, max_v=2.0),
        })

    def test_pinch_off_end_to_end(self, dc):
        """PinchOff: physics probe → run → analyze → correct threshold voltage."""
        from strumento.packs.spin.experiments.pinch_off import PinchOff

        voltages = np.linspace(0.0, -2.0, 101)

        # Physics probe: conductance step pinching off at -1.0 V
        # At V=0 channel is open (high signal); at V=-2 it's pinched off (low signal)
        def probe():
            v = dc.get("p0")
            return 1.0 / (1.0 + np.exp(-10.0 * (v + 1.0)))

        exp = PinchOff(dc, gate_name="p0", voltages=voltages, probe=probe)
        result = exp.run()
        fit = exp.analyze(result)

        assert fit.success
        assert "v_threshold" in fit.derived
        # Threshold should be near -1.0 V
        assert -1.3 < fit.derived["v_threshold"] < -0.7

    def test_sensor_tune_end_to_end(self, dc):
        """SensorTune: physics probe → run → analyze → correct optimal voltage."""
        from strumento.packs.spin.experiments.sensor_tune import SensorTune

        voltages = np.linspace(-1.0, 0.0, 101)

        # Physics probe: QPC conductance step with max transconductance at -0.5 V
        def probe():
            v = dc.get("sensor")
            return 1.0 / (1.0 + np.exp(-20.0 * (v + 0.5)))

        exp = SensorTune(dc, sensor_name="sensor", voltages=voltages, probe=probe)
        result = exp.run()
        fit = exp.analyze(result)

        assert fit.success
        assert "v_optimal" in fit.derived
        assert "transconductance" in fit.derived
        # Steepest point at -0.5 V
        assert -0.7 < fit.derived["v_optimal"] < -0.3

    def test_coulomb_diamond_end_to_end(self, dc):
        """CoulombDiamond: 2D physics probe → run → analyze → derived quantities."""
        from strumento.packs.spin.experiments.coulomb_diamond import CoulombDiamond

        plunger_volts = np.linspace(-0.5, 0.5, 51)
        bias_volts = np.linspace(-1.0, 1.0, 21)

        # Physics probe: Coulomb blockade diamond (suppressed current inside diamond)
        def probe():
            vp = dc.get("p0")
            vsd = dc.get("v_sd")
            # Diamond: current suppressed when |vsd| < Ec*(1 - |vp|/0.3)
            ec = 0.5  # charging energy in V_sd units
            diamond_edge = ec * max(0.0, 1.0 - abs(vp) / 0.3)
            return 1.0 if abs(vsd) > diamond_edge else 0.05

        exp = CoulombDiamond(
            dc, plunger_axis="p0", plunger_volts=plunger_volts,
            bias_axis="v_sd", bias_volts=bias_volts, probe=probe,
        )
        result = exp.run()
        fit = exp.analyze(result)

        assert fit.success
        assert "charging_energy_mev" in fit.derived
        assert "lever_arm" in fit.derived
