"""Tests for TransmonOps pulse factories + per-instance binding (plan Task 6, D21/D23)."""

from pathlib import Path

import pytest

from strumento.core.device import Device
from strumento.core.pulses import Gaussian
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import gain_int_to_frac

YAML = Path(__file__).parents[3] / "devices" / "schuster_multimode" / "device.yaml"

MAXV = 32766  # from the committed soccfg snapshot


@pytest.fixture
def dev():
    return Device.load(YAML)


def test_ge_pi_from_calibration(dev):
    p = dev.qubit.ge_pi()
    assert p.line == LineRef("qubit", "drive")
    assert p.freq_mhz == pytest.approx(dev.calib.qubit.f_ge)
    assert p.envelope == Gaussian(sigma_us=dev.calib.qubit.pulses["pi_ge"].sigma)
    # D23: gain is a v2 FRACTION, not the int code 3289
    assert p.gain == pytest.approx(gain_int_to_frac(3289, MAXV))
    assert p.gain == pytest.approx(3289 / MAXV)
    assert p.gain != 3289


def test_ge_hpi_and_ef_pi_use_right_freq_and_gain(dev):
    hpi = dev.qubit.ge_hpi()
    assert hpi.freq_mhz == pytest.approx(dev.calib.qubit.f_ge)
    assert hpi.gain == pytest.approx(1629 / MAXV)

    ef = dev.qubit.ef_pi()
    assert ef.freq_mhz == pytest.approx(dev.calib.qubit.f_ef)  # ef transition, not ge
    assert ef.gain == pytest.approx(3038 / MAXV)


def test_overrides(dev):
    # a raw fraction passed as gain is used as-is (override convention, D23)
    p = dev.qubit.ge_pi(gain=0.5, freq=4000.0)
    assert p.gain == 0.5
    assert p.freq_mhz == 4000.0
    # phase override -> phase_deg
    assert dev.qubit.ge_pi(phase=90).phase_deg == 90


def test_binding_is_instance_scoped(dev):
    # dev.qubit ops are bound to *this* named element (D21)
    assert dev.qubit.element is dev.elements["qubit"]
    assert dev.qubit.element.name == "qubit"
