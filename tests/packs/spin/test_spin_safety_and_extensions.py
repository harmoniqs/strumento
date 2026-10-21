"""Tests for spin pack safety guards and extended readout/element types (pure — no qick)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from strumento.core.readout import DataSource, Discriminator, ReadoutKind
from strumento.packs.spin.calibration import GateCal, SpinCalibration, SpinPulseCal
from strumento.packs.spin.elements import (
    Barrier,
    ChargeSensor,
    Dot,
    DriveLine,
    EODot,
    GateSensor,
    HoleDot,
    Plunger,
    Reservoir,
    STDot,
)
from strumento.packs.spin.readout import (
    DispersiveGateReadout,
    ElzermanReadout,
    LatchedPSBReadout,
    PSBReadout,
)
from strumento.packs.spin.safety import (
    GainOutOfRange,
    VoltageOutOfRange,
    validate_gain,
    validate_sweep_voltage,
)

# =============================================================================
# Safety guards — validate_gain
# =============================================================================


class TestValidateGain:
    """validate_gain raises GainOutOfRange for |gain| > limit."""

    def test_valid_gain_passes(self):
        assert validate_gain(0.5, label="test") == 0.5

    def test_negative_valid_gain_passes(self):
        assert validate_gain(-0.8, label="test") == -0.8

    def test_boundary_gain_passes(self):
        assert validate_gain(1.0, label="test") == 1.0
        assert validate_gain(-1.0, label="test") == -1.0

    def test_gain_above_limit_raises(self):
        with pytest.raises(GainOutOfRange, match="exceeds safe limit"):
            validate_gain(1.01, label="q1_pi")

    def test_large_gain_raises(self):
        with pytest.raises(GainOutOfRange, match="exceeds safe limit"):
            validate_gain(5.0, label="exchange_b0")

    def test_negative_over_limit_raises(self):
        with pytest.raises(GainOutOfRange, match="exceeds safe limit"):
            validate_gain(-1.5, label="test")

    def test_custom_limit(self):
        # Tighter limit
        assert validate_gain(0.3, label="test", limit=0.5) == 0.3
        with pytest.raises(GainOutOfRange):
            validate_gain(0.6, label="test", limit=0.5)

    def test_zero_gain_passes(self):
        assert validate_gain(0.0, label="test") == 0.0


# =============================================================================
# Safety guards — validate_sweep_voltage
# =============================================================================


class TestValidateSweepVoltage:
    """validate_sweep_voltage raises VoltageOutOfRange for |endpoint| > max_v."""

    def test_valid_sweep_passes(self):
        # Should not raise
        validate_sweep_voltage(-1.0, 0.5, max_v=1.5, gate_name="P1")

    def test_start_over_limit_raises(self):
        with pytest.raises(VoltageOutOfRange, match="start=-3.0"):
            validate_sweep_voltage(-3.0, 0.5, max_v=1.5, gate_name="P1")

    def test_stop_over_limit_raises(self):
        with pytest.raises(VoltageOutOfRange, match="stop=2.0"):
            validate_sweep_voltage(-0.5, 2.0, max_v=1.5, gate_name="P1")

    def test_both_over_limit_raises_on_first(self):
        with pytest.raises(VoltageOutOfRange, match="start"):
            validate_sweep_voltage(-5.0, 5.0, max_v=1.5, gate_name="P1")

    def test_default_max_v_when_none(self):
        # DEFAULT_MAX_VOLTAGE = 2.0
        validate_sweep_voltage(-1.5, 1.5, max_v=None, gate_name="P1")
        with pytest.raises(VoltageOutOfRange):
            validate_sweep_voltage(-2.5, 0.0, max_v=None, gate_name="P1")

    def test_boundary_passes(self):
        validate_sweep_voltage(-1.5, 1.5, max_v=1.5, gate_name="P1")

    def test_error_message_includes_gate_name(self):
        with pytest.raises(VoltageOutOfRange, match="gate 'X2'"):
            validate_sweep_voltage(-3.0, 0.0, max_v=1.5, gate_name="X2")


# =============================================================================
# Safety guards — SpinPulseCal schema validation
# =============================================================================


class TestSpinPulseCalValidation:
    """SpinPulseCal rejects invalid calibration data at load time."""

    def test_valid_calibration_loads(self):
        cal = SpinPulseCal(gain=0.5, sigma=0.01, length=0.1)
        assert cal.gain == 0.5

    def test_gain_over_one_rejected(self):
        with pytest.raises(ValidationError, match="gain"):
            SpinPulseCal(gain=2.0, sigma=0.01, length=0.1)

    def test_gain_under_negative_one_rejected(self):
        with pytest.raises(ValidationError, match="gain"):
            SpinPulseCal(gain=-1.5, sigma=0.01, length=0.1)

    def test_boundary_gain_accepted(self):
        cal = SpinPulseCal(gain=1.0, sigma=0.01, length=0.1)
        assert cal.gain == 1.0
        cal = SpinPulseCal(gain=-1.0, sigma=0.01, length=0.1)
        assert cal.gain == -1.0

    def test_negative_length_rejected(self):
        with pytest.raises(ValidationError, match="length"):
            SpinPulseCal(gain=0.5, sigma=0.01, length=-0.1)

    def test_negative_sigma_rejected(self):
        with pytest.raises(ValidationError, match="sigma"):
            SpinPulseCal(gain=0.5, sigma=-0.01, length=0.1)

    def test_zero_length_and_sigma_accepted(self):
        cal = SpinPulseCal(gain=0.5, sigma=0.0, length=0.0)
        assert cal.length == 0.0 and cal.sigma == 0.0


class TestGateCalValidation:
    """GateCal rejects non-positive max_v."""

    def test_valid_max_v(self):
        cal = GateCal(dc_channel=0, max_v=1.5)
        assert cal.max_v == 1.5

    def test_zero_max_v_rejected(self):
        with pytest.raises(ValidationError, match="max_v"):
            GateCal(dc_channel=0, max_v=0.0)

    def test_negative_max_v_rejected(self):
        with pytest.raises(ValidationError, match="max_v"):
            GateCal(dc_channel=0, max_v=-1.0)


# =============================================================================
# Readout protocol conformance — new readout kinds
# =============================================================================


class TestElzermanReadout:
    """ElzermanReadout satisfies ReadoutKind/Discriminator and discriminates correctly."""

    def test_is_readout_kind(self):
        r = ElzermanReadout(threshold=0.5)
        assert isinstance(r, ReadoutKind)
        assert isinstance(r, Discriminator)
        assert r.data_source is DataSource.QICK_ADC

    def test_discriminate_blip_detection(self):
        r = ElzermanReadout(threshold=0.5)
        # Each row is a shot's time trace; max excursion per shot is the feature
        raw = np.array([
            [0.1, 0.2, 0.8, 0.3],   # max=0.8 > 0.5 → spin-up (1)
            [0.1, 0.2, 0.3, 0.1],   # max=0.3 < 0.5 → spin-down (0)
            [0.1, 0.9, 0.4, 0.2],   # max=0.9 > 0.5 → spin-up (1)
        ])
        result = r.discriminate(raw)
        assert list(result) == [1, 0, 1]

    def test_reduce_gives_spin_up_fraction(self):
        r = ElzermanReadout(threshold=0.5)
        raw = np.array([
            [0.1, 0.8],   # up
            [0.1, 0.3],   # down
            [0.1, 0.9],   # up
            [0.1, 0.2],   # down
        ])
        assert float(r.reduce(raw)) == pytest.approx(0.5)


class TestDispersiveGateReadout:
    """DispersiveGateReadout satisfies ReadoutKind/Discriminator."""

    def test_is_readout_kind(self):
        r = DispersiveGateReadout(threshold=0.5)
        assert isinstance(r, ReadoutKind)
        assert isinstance(r, Discriminator)
        assert r.data_source is DataSource.QICK_ADC

    def test_discriminate_threshold(self):
        r = DispersiveGateReadout(threshold=0.0, angle=0.0)
        raw = np.array([0.5, -0.5, 0.1, -0.1])
        result = r.discriminate(raw)
        assert list(result) == [1, 0, 1, 0]

    def test_reduce_gives_mean(self):
        r = DispersiveGateReadout(threshold=0.0)
        raw = np.array([1.0, -1.0, 1.0, 1.0])
        assert float(r.reduce(raw)) == pytest.approx(0.75)


class TestLatchedPSBReadout:
    """LatchedPSBReadout satisfies ReadoutKind/Discriminator."""

    def test_is_readout_kind(self):
        r = LatchedPSBReadout(threshold=0.5)
        assert isinstance(r, ReadoutKind)
        assert isinstance(r, Discriminator)
        assert r.data_source is DataSource.QICK_ADC

    def test_discriminate_threshold(self):
        r = LatchedPSBReadout(threshold=0.5)
        raw = np.array([1.0, 0.2, 0.8, 0.3])
        result = r.discriminate(raw)
        assert list(result) == [1, 0, 1, 0]

    def test_reduce_gives_triplet_fraction(self):
        r = LatchedPSBReadout(threshold=0.5)
        raw = np.array([1.0, 0.2, 0.8, 0.3])
        assert float(r.reduce(raw)) == pytest.approx(0.5)


# =============================================================================
# Element types — compose_elements dispatch
# =============================================================================


class TestComposeElements:
    """compose_elements dispatches on encoding and element type correctly."""

    def test_dot_ld_encoding(self):
        from strumento.packs.spin import compose_elements

        topo = [{"type": "Dot", "name": "q1", "encoding": "ld"}]
        out = compose_elements(topo, SpinCalibration())
        assert isinstance(out["q1"], Dot)
        assert out["q1"].encoding == "ld"

    def test_dot_st_encoding(self):
        from strumento.packs.spin import compose_elements

        topo = [{"type": "Dot", "name": "q1", "encoding": "st"}]
        out = compose_elements(topo, SpinCalibration())
        assert isinstance(out["q1"], STDot)
        assert out["q1"].encoding == "st"

    def test_dot_eo_encoding(self):
        from strumento.packs.spin import compose_elements

        topo = [{"type": "Dot", "name": "q1", "encoding": "eo"}]
        out = compose_elements(topo, SpinCalibration())
        assert isinstance(out["q1"], EODot)
        assert out["q1"].encoding == "eo"

    def test_dot_hole_encoding(self):
        from strumento.packs.spin import compose_elements

        topo = [{"type": "Dot", "name": "q1", "encoding": "hole"}]
        out = compose_elements(topo, SpinCalibration())
        assert isinstance(out["q1"], HoleDot)
        assert out["q1"].encoding == "hole"

    def test_dot_default_encoding_is_ld(self):
        from strumento.packs.spin import compose_elements

        topo = [{"type": "Dot", "name": "q1"}]
        out = compose_elements(topo, SpinCalibration())
        assert isinstance(out["q1"], Dot)
        assert out["q1"].encoding == "ld"

    def test_unknown_encoding_raises(self):
        from strumento.packs.spin import compose_elements

        topo = [{"type": "Dot", "name": "q1", "encoding": "nonsense"}]
        with pytest.raises(KeyError, match="nonsense"):
            compose_elements(topo, SpinCalibration())

    def test_reservoir(self):
        from strumento.packs.spin import compose_elements

        topo = [{"type": "Reservoir", "name": "res1"}]
        out = compose_elements(topo, SpinCalibration())
        assert isinstance(out["res1"], Reservoir)

    def test_charge_sensor(self):
        from strumento.packs.spin import compose_elements

        topo = [{"type": "ChargeSensor", "name": "cs1", "sensor_type": "set"}]
        out = compose_elements(topo, SpinCalibration())
        assert isinstance(out["cs1"], ChargeSensor)
        assert out["cs1"].sensor_type == "set"

    def test_drive_line(self):
        from strumento.packs.spin import compose_elements

        topo = [{"type": "DriveLine", "name": "mw1"}]
        out = compose_elements(topo, SpinCalibration())
        assert isinstance(out["mw1"], DriveLine)

    def test_gate_sensor(self):
        from strumento.packs.spin import compose_elements

        topo = [{"type": "GateSensor", "name": "gs1"}]
        out = compose_elements(topo, SpinCalibration())
        assert isinstance(out["gs1"], GateSensor)

    def test_plunger_and_barrier(self):
        from strumento.packs.spin import compose_elements

        topo = [
            {"type": "Plunger", "name": "p1"},
            {"type": "Barrier", "name": "b1"},
        ]
        out = compose_elements(topo, SpinCalibration())
        assert isinstance(out["p1"], Plunger)
        assert isinstance(out["b1"], Barrier)

    def test_unknown_type_raises(self):
        from strumento.packs.spin import compose_elements

        topo = [{"type": "Gizmo", "name": "g1"}]
        with pytest.raises(KeyError, match="Gizmo"):
            compose_elements(topo, SpinCalibration())

    def test_full_topology(self):
        """A realistic multi-element topology composes without error."""
        from strumento.packs.spin import compose_elements

        topo = [
            {"type": "Dot", "name": "q1", "encoding": "eo"},
            {"type": "Dot", "name": "q2", "encoding": "eo"},
            {"type": "Dot", "name": "q3", "encoding": "eo"},
            {"type": "Plunger", "name": "p1"},
            {"type": "Plunger", "name": "p2"},
            {"type": "Barrier", "name": "b12"},
            {"type": "Barrier", "name": "b23"},
            {"type": "Reservoir", "name": "res"},
            {"type": "ChargeSensor", "name": "sensor"},
        ]
        out = compose_elements(topo, SpinCalibration())
        assert len(out) == 9
        assert all(isinstance(out["q1"], EODot) for _ in [1])
        assert isinstance(out["sensor"], ChargeSensor)


# =============================================================================
# Calibration schema roundtrip (serialize → reload → assert equality)
# =============================================================================


class TestCalibrationRoundtrip:
    """SpinCalibration serializes to dict and reloads identically."""

    def test_full_calibration_roundtrip(self):
        """Build a full SpinCalibration, dump to dict, reload, assert equality."""
        from strumento.packs.spin.calibration import (
            DotCal,
            GateSensorCal,
            ReadoutCal,
            ReservoirCal,
            SensorCal,
        )

        cal = SpinCalibration(
            dots={
                "q1": DotCal(
                    f_qubit_mhz=5000.0,
                    exchange_j_mhz=80.0,
                    g_factor=2.0,
                    t1_us=50.0,
                    t2_star_us=2.5,
                    t2_echo_us=15.0,
                    pulses={
                        "pi": SpinPulseCal(gain=0.4, sigma=0.02, length=0.0, type="gauss"),
                        "hpi": SpinPulseCal(gain=0.2, sigma=0.02, length=0.0, type="gauss"),
                    },
                ),
            },
            gates={
                "p1": GateCal(dc_channel=0, max_v=1.5, crosscoupling={"b1": 0.12}),
                "b1": GateCal(
                    dc_channel=1,
                    max_v=1.2,
                    exchange=SpinPulseCal(gain=0.5, sigma=0.0, length=0.05, type="flat_top"),
                    lever_arm=0.08,
                    pinchoff_v=-0.9,
                ),
            },
            sensors={"cs1": SensorCal(v_operating=-0.35, g_setpoint=0.5, transconductance=0.02)},
            reservoirs={"res1": ReservoirCal(gamma_khz=50.0, t_electron_mk=30.0)},
            readout=ReadoutCal(threshold=0.5, fidelity=0.99, t_read_us=100.0, angle_rad=0.1),
            gate_sensors={"gs1": GateSensorCal(resonator_freq_mhz=312.5, q_loaded=500.0)},
            virtual_gate_matrix=[[1.0, -0.12], [-0.08, 1.0]],
        )

        # Serialize to dict (Pydantic model_dump)
        data = cal.model_dump()
        assert isinstance(data, dict)
        assert "dots" in data and "gates" in data

        # Reload from dict
        reloaded = SpinCalibration(**data)

        # Assert equality
        assert reloaded.dots["q1"].f_qubit_mhz == 5000.0
        assert reloaded.dots["q1"].pulses["pi"].gain == 0.4
        assert reloaded.dots["q1"].g_factor == 2.0
        assert reloaded.dots["q1"].t1_us == 50.0
        assert reloaded.gates["b1"].exchange.gain == 0.5
        assert reloaded.gates["b1"].lever_arm == 0.08
        assert reloaded.gates["p1"].crosscoupling == {"b1": 0.12}
        assert reloaded.sensors["cs1"].v_operating == -0.35
        assert reloaded.reservoirs["res1"].gamma_khz == 50.0
        assert reloaded.readout.fidelity == 0.99
        assert reloaded.gate_sensors["gs1"].q_loaded == 500.0
        assert reloaded.virtual_gate_matrix == [[1.0, -0.12], [-0.08, 1.0]]

    def test_empty_calibration_roundtrip(self):
        """An empty SpinCalibration roundtrips cleanly."""
        cal = SpinCalibration()
        data = cal.model_dump()
        reloaded = SpinCalibration(**data)
        assert reloaded.dots == {}
        assert reloaded.gates == {}
        assert reloaded.sensors == {}
        assert reloaded.readout is None

    def test_partial_calibration_roundtrip(self):
        """A calibration with only dots (no gates/sensors) roundtrips."""
        from strumento.packs.spin.calibration import DotCal

        cal = SpinCalibration(
            dots={"q1": DotCal(f_qubit_mhz=3200.0)},
        )
        data = cal.model_dump()
        reloaded = SpinCalibration(**data)
        assert reloaded.dots["q1"].f_qubit_mhz == 3200.0
        assert reloaded.dots["q1"].pulses == {}
        assert reloaded.gates == {}


# =============================================================================
# Readout batch shape — multi-shot / multi-read data
# =============================================================================


class TestReadoutBatchShape:
    """Readout kinds handle batch-shaped data (n_shots, n_points) correctly."""

    def test_psb_discriminate_2d(self):
        """PSBReadout.discriminate works on (n_shots, n_points) shaped data."""
        r = PSBReadout(threshold=0.5)
        # 4 shots, 3 points each — threshold applies element-wise
        raw = np.array([
            [0.8, 0.2, 0.9],
            [0.1, 0.3, 0.4],
            [0.6, 0.7, 0.2],
            [0.1, 0.9, 0.1],
        ])
        result = r.discriminate(raw)
        assert result.shape == (4, 3)
        expected = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_psb_reduce_2d(self):
        """PSBReadout.reduce averages over the last axis (per-shot mean)."""
        r = PSBReadout(threshold=0.5)
        raw = np.array([
            [0.8, 0.2, 0.9],   # 2/3 above threshold
            [0.1, 0.3, 0.4],   # 0/3 above
        ])
        result = r.reduce(raw)
        # reduce averages the discriminated array along last axis
        # discriminated: [[1,0,1],[0,0,0]] → mean along -1: [0.667, 0.0]
        assert result.shape == (2,)
        assert result[0] == pytest.approx(2.0 / 3.0)
        assert result[1] == pytest.approx(0.0)

    def test_elzerman_discriminate_2d_shots(self):
        """ElzermanReadout handles (n_shots, n_timepoints) — max per row."""
        from strumento.packs.spin.readout import ElzermanReadout

        r = ElzermanReadout(threshold=0.5)
        # 3 shots, 5 time-points each
        raw = np.array([
            [0.1, 0.2, 0.8, 0.3, 0.1],  # max=0.8 > 0.5 → 1
            [0.1, 0.2, 0.3, 0.2, 0.1],  # max=0.3 < 0.5 → 0
            [0.1, 0.9, 0.1, 0.1, 0.1],  # max=0.9 > 0.5 → 1
        ])
        result = r.discriminate(raw)
        assert list(result) == [1, 0, 1]

    def test_elzerman_reduce_gives_fraction(self):
        """ElzermanReadout.reduce returns mean spin-up fraction."""
        from strumento.packs.spin.readout import ElzermanReadout

        r = ElzermanReadout(threshold=0.5)
        raw = np.array([
            [0.1, 0.8],  # up
            [0.1, 0.3],  # down
            [0.1, 0.6],  # up
            [0.1, 0.2],  # down
            [0.1, 0.7],  # up
        ])
        result = r.reduce(raw)
        assert float(result) == pytest.approx(3.0 / 5.0)

    def test_dispersive_gate_discriminate_1d(self):
        """DispersiveGateReadout handles a flat 1D array."""
        from strumento.packs.spin.readout import DispersiveGateReadout

        r = DispersiveGateReadout(threshold=0.0)
        raw = np.array([0.5, -0.5, 0.1, -0.1, 0.3])
        result = r.discriminate(raw)
        assert list(result) == [1, 0, 1, 0, 1]

    def test_latched_psb_batch(self):
        """LatchedPSBReadout handles batch data identically to PSBReadout."""
        from strumento.packs.spin.readout import LatchedPSBReadout

        r = LatchedPSBReadout(threshold=0.5)
        raw = np.array([0.8, 0.2, 0.9, 0.1, 0.6])
        result = r.discriminate(raw)
        assert list(result) == [1, 0, 1, 0, 1]
        assert float(r.reduce(raw)) == pytest.approx(3.0 / 5.0)
