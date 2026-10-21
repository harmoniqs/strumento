"""Tests for new device fixtures and ops binding (ST, EO, Hole encodings — pure)."""

from __future__ import annotations

import pytest

from strumento.core.device import Device
from strumento.core.pulses import FlatTop, Gaussian
from strumento.core.wiring import LineRef
from strumento.packs.spin.elements import (
    Barrier,
    ChargeSensor,
    EODot,
    HoleDot,
    Plunger,
    Reservoir,
    STDot,
)
from strumento.packs.spin.library.exchange import ExchangeOps
from strumento.packs.spin.library.hole_spin import HoleSpinOps
from strumento.packs.spin.library.singlet_triplet import SingletTripletOps
from strumento.packs.spin.safety import GainOutOfRange
from tests._paths import DEVICES

ST_YAML = DEVICES / "spin_st_demo" / "device.yaml"
EO_YAML = DEVICES / "spin_eo_demo" / "device.yaml"
HOLE_YAML = DEVICES / "spin_hole_demo" / "device.yaml"


# =============================================================================
# Singlet-triplet fixture
# =============================================================================


class TestSTDevice:
    """The spin_st_demo fixture loads, composes, and binds SingletTripletOps."""

    @pytest.fixture
    def dev(self):
        return Device.load(ST_YAML)

    def test_loads_and_composes(self, dev):
        assert isinstance(dev.elements["left"], STDot)
        assert isinstance(dev.elements["right"], STDot)
        assert isinstance(dev.elements["b_lr"], Barrier)
        assert isinstance(dev.elements["p_left"], Plunger)
        assert isinstance(dev.elements["sensor"], ChargeSensor)
        assert dev.elements["left"].encoding == "st"

    def test_st_ops_bound(self, dev):
        assert isinstance(dev.left, SingletTripletOps)
        assert isinstance(dev.right, SingletTripletOps)

    def test_j_pi_from_calibration(self, dev):
        p = dev.left.j_pi()
        assert p.line == LineRef("left", "detuning")
        assert p.freq_mhz == 0.0  # baseband
        assert p.gain == pytest.approx(0.35)
        assert isinstance(p.envelope, FlatTop)
        assert p.envelope.length_us == pytest.approx(0.025)

    def test_j_hpi_half_duration(self, dev):
        p = dev.left.j_hpi()
        assert p.gain == pytest.approx(0.35)
        assert p.envelope.length_us == pytest.approx(0.0125)

    def test_singlet_prep(self, dev):
        p = dev.left.singlet_prep()
        assert p.gain == pytest.approx(0.2)
        assert p.envelope.length_us == pytest.approx(1.0)

    def test_z_idle_returns_duration(self, dev):
        dur = dev.left.z_idle(duration_ns=500.0)
        assert dur == pytest.approx(0.5)  # 500 ns = 0.5 us

    def test_exchange_on_barrier(self, dev):
        p = dev.b_lr.pulse()
        assert p.line == LineRef("b_lr", "fast")
        assert p.gain == pytest.approx(0.35)

    def test_gain_override_validated(self, dev):
        # Valid override
        p = dev.left.j_pi(gain=0.8)
        assert p.gain == pytest.approx(0.8)
        # Invalid override
        with pytest.raises(GainOutOfRange):
            dev.left.j_pi(gain=1.5)


# =============================================================================
# Exchange-only fixture
# =============================================================================


class TestEODevice:
    """The spin_eo_demo fixture loads with EODot (no dot ops) + barrier exchange."""

    @pytest.fixture
    def dev(self):
        return Device.load(EO_YAML)

    def test_loads_and_composes(self, dev):
        assert isinstance(dev.elements["q1"], EODot)
        assert isinstance(dev.elements["q2"], EODot)
        assert isinstance(dev.elements["q3"], EODot)
        assert isinstance(dev.elements["b12"], Barrier)
        assert isinstance(dev.elements["b23"], Barrier)
        assert isinstance(dev.elements["sensor"], ChargeSensor)
        assert isinstance(dev.elements["res"], Reservoir)

    def test_eo_dots_have_no_ops(self, dev):
        # EODot has ops_cls=None — no dot-level namespace
        assert not hasattr(dev, "q1")
        assert not hasattr(dev, "q2")
        assert not hasattr(dev, "q3")

    def test_barriers_have_exchange_ops(self, dev):
        assert isinstance(dev.b12, ExchangeOps)
        assert isinstance(dev.b23, ExchangeOps)

    def test_exchange_pulse_from_calibration(self, dev):
        p = dev.b12.pulse()
        assert p.line == LineRef("b12", "fast")
        assert p.freq_mhz == 0.0
        assert p.gain == pytest.approx(0.4)
        assert isinstance(p.envelope, FlatTop)
        assert p.envelope.length_us == pytest.approx(0.012)

    def test_b23_different_calibration(self, dev):
        p = dev.b23.pulse()
        assert p.gain == pytest.approx(0.45)
        assert p.envelope.length_us == pytest.approx(0.011)

    def test_exchange_gain_override_validated(self, dev):
        with pytest.raises(GainOutOfRange):
            dev.b12.pulse(gain=2.0)

    def test_eight_elements_total(self, dev):
        assert len(dev.elements) == 8


# =============================================================================
# Hole-spin fixture
# =============================================================================


class TestHoleDevice:
    """The spin_hole_demo fixture loads, composes, and binds HoleSpinOps."""

    @pytest.fixture
    def dev(self):
        return Device.load(HOLE_YAML)

    def test_loads_and_composes(self, dev):
        assert isinstance(dev.elements["h1"], HoleDot)
        assert isinstance(dev.elements["p1"], Plunger)
        assert isinstance(dev.elements["b1"], Barrier)
        assert isinstance(dev.elements["sensor"], ChargeSensor)
        assert isinstance(dev.elements["res"], Reservoir)
        assert dev.elements["h1"].encoding == "hole"

    def test_hole_ops_bound(self, dev):
        assert isinstance(dev.h1, HoleSpinOps)

    def test_pi_from_calibration(self, dev):
        p = dev.h1.pi()
        assert p.line == LineRef("h1", "drive")
        assert p.freq_mhz == pytest.approx(3200.0)
        assert p.gain == pytest.approx(0.5)
        assert isinstance(p.envelope, Gaussian)

    def test_hpi_half_gain(self, dev):
        p = dev.h1.hpi()
        assert p.gain == pytest.approx(0.25)
        assert p.freq_mhz == pytest.approx(3200.0)

    def test_pi_at_sweet_spot(self, dev):
        p = dev.h1.pi_at_sweet_spot()
        # Same as pi() — sweet-spot assumption is about the gate voltage, not the pulse
        assert p.gain == pytest.approx(0.5)
        assert p.freq_mhz == pytest.approx(3200.0)

    def test_exchange_cz_on_barrier(self, dev):
        p = dev.h1.exchange_cz(barrier="b1")
        assert p.line == LineRef("b1", "fast")
        assert p.freq_mhz == 0.0
        assert p.gain == pytest.approx(0.4)
        assert isinstance(p.envelope, FlatTop)

    def test_barrier_exchange_ops(self, dev):
        p = dev.b1.pulse()
        assert p.gain == pytest.approx(0.4)

    def test_gain_override_validated(self, dev):
        p = dev.h1.pi(gain=0.9)
        assert p.gain == pytest.approx(0.9)
        with pytest.raises(GainOutOfRange):
            dev.h1.pi(gain=1.2)

    def test_calibration_data_accessible(self, dev):
        cal = dev.calib.dots["h1"]
        assert cal.f_qubit_mhz == pytest.approx(3200.0)
        assert cal.g_factor == pytest.approx(1.8)
        assert cal.t1_us == pytest.approx(50.0)
        assert cal.t2_star_us == pytest.approx(2.5)
