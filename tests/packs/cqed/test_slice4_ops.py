"""Slice 4 — sideband / mode / coupler ops + alias binding (pure)."""

from __future__ import annotations

from pathlib import Path

import pytest

from strumento.core.device import Device
from strumento.core.pulses import FlatTop, Gaussian
from strumento.core.wiring import LineRef

YAML = Path(__file__).parents[3] / "devices" / "multimode_demo" / "device.yaml"
MAXV = 32766


@pytest.fixture
def dev():
    return Device.load(YAML)


def test_sideband_alias_bound_from_instance_yaml(dev):
    # dev.sideband exists because the YAML declares the alias, not because the pack hardcodes it
    p = dev.sideband.f0g1()
    assert p.line == LineRef("qubit", "sideband")
    assert p.freq_mhz == pytest.approx(2100.0)
    assert p.gain == pytest.approx(12000 / MAXV)
    assert p.envelope == FlatTop(ramp_sigma_us=0.01, length_us=0.8)


def test_multiphoton_indexes_by_photon_number(dev):
    p1 = dev.sideband.multiphoton("f0-g1", "pi", photon=1)
    assert p1.freq_mhz == pytest.approx(2088.0)
    assert p1.gain == pytest.approx(12500 / MAXV)
    with pytest.raises(IndexError):
        dev.sideband.multiphoton("f0-g1", "hpi", photon=3)
    with pytest.raises(KeyError):
        dev.sideband.multiphoton("g0-e1")


def test_mode_ops_displace(dev):
    p = dev.manipulate.displace()
    assert p.line == LineRef("manipulate", "main")
    assert p.freq_mhz == pytest.approx(5200.0)
    assert p.gain == pytest.approx(6000 / MAXV)
    assert p.envelope == Gaussian(sigma_us=0.040)


def test_coupler_swap_band_gated(dev):
    p = dev.coupler.swap("manipulate-S1")
    assert p.line == LineRef("coupler", "low")        # 800 MHz < 1800 MHz band edge
    assert p.freq_mhz == pytest.approx(800.0)
    assert p.envelope == FlatTop(ramp_sigma_us=0.01, length_us=2.4)
    hpi = dev.coupler.swap("manipulate-S1", "hpi")
    assert hpi.envelope.length_us == pytest.approx(1.2)  # 50:50 beamsplitter = half length
    with pytest.raises(KeyError):
        dev.coupler.swap("manipulate-S9")
