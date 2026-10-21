"""Tests for the spin pack skeleton."""

import numpy as np
import pytest

from strumento.core.device import Device
from strumento.core.readout import DataSource, Discriminator, ReadoutKind
from strumento.core.wiring import LineRef
from strumento.packs.spin.elements import Barrier, Dot, Plunger
from strumento.packs.spin.readout import PSBReadout
from tests._paths import DEVICES

YAML = DEVICES / "spin_demo" / "device.yaml"


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


def test_ops_now_bound():
    from strumento.packs.spin.library.ld import LDOps

    dev = Device.load(YAML)
    assert Dot.ops_cls is LDOps
    assert isinstance(dev.q0, LDOps)


def test_spin_demo_attaches_the_v2_spin_snapshot():
    # issue #132: spin_demo carries the fabricated v2 spin snapshot, so SimulatorSoc.from_device
    # can assemble against it; every wired DAC is a snapshot generator, the ADC a snapshot readout.
    dev = Device.load(YAML)
    snap = dev.soccfg_snapshot
    assert snap is not None
    assert len(snap["gens"]) == 4 and len(snap["readouts"]) == 1
    assert {0, 1, 2, 3} <= {g["tproc_ch"] for g in snap["gens"]}
    # the readout feeds the tProc's single input port
    assert snap["readouts"][0]["tproc_ch"] == 0
