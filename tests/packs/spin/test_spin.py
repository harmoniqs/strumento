"""Tests for the spin pack skeleton (spec §15.5)."""

from pathlib import Path

import numpy as np
import pytest

from strumento.core.device import Device
from strumento.core.readout import DataSource, Discriminator, ReadoutKind
from strumento.core.wiring import LineRef
from strumento.packs.spin.elements import Barrier, Dot, Plunger
from strumento.packs.spin.readout import PSBReadout

YAML = Path(__file__).parents[3] / "devices" / "spin_demo" / "device.yaml"


def test_instance_loads_and_composes():
    dev = Device.load(YAML)
    assert set(dev.elements) == {"q0", "p0", "b0"}
    assert isinstance(dev.elements["q0"], Dot)
    assert isinstance(dev.elements["p0"], Plunger)
    assert isinstance(dev.elements["b0"], Barrier)
    assert dev.elements["q0"].encoding == "ld"
    # the dot drives a control line; gates are DC lines
    assert dev.wiring.resolve(LineRef("q0", "drive"), freq_mhz=5000).dac_ch == 0
    assert dev.wiring.resolve(LineRef("p0", "dc"), freq_mhz=0).dac_ch == 1
    assert dev.calib.dots["q0"].f_qubit_mhz == 5000.0


def test_psb_readout_is_a_readout_kind():
    r = PSBReadout(threshold=0.0)
    assert isinstance(r, ReadoutKind)
    assert isinstance(r, Discriminator)
    assert r.data_source is DataSource.QICK_ADC
    signal = np.array([1.0, -1.0, 0.5, -0.5])  # charge-sensor signal
    assert list(r.discriminate(signal)) == [1, 0, 1, 0]
    assert float(r.reduce(signal)) == pytest.approx(0.5)


def test_no_ops_bound_yet():
    dev = Device.load(YAML)
    assert Dot.ops_cls is None
    assert not hasattr(dev, "q0")
