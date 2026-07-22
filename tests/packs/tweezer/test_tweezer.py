"""Tests for the tweezer pack skeleton (spec §17)."""

from pathlib import Path

import numpy as np
import pytest

from strumento.core.device import Device
from strumento.core.readout import DataSource, Discriminator, ReadoutKind
from strumento.core.wiring import LineRef
from strumento.packs.tweezer.elements import AODAxis, AtomArray
from strumento.packs.tweezer.envelopes import PolyChirp
from strumento.packs.tweezer.readout import ImagingReadout

YAML = Path(__file__).parents[3] / "devices" / "tweezer_demo" / "device.yaml"


def test_instance_loads_and_composes():
    dev = Device.load(YAML)
    assert set(dev.elements) == {"aod_x", "aod_y", "array"}
    assert isinstance(dev.elements["aod_x"], AODAxis)
    assert isinstance(dev.elements["array"], AtomArray)
    # AOD axis resolves to its mux generator; the array drives no lines
    assert dev.wiring.resolve(LineRef("aod_x", "mux"), freq_mhz=100).dac_ch == 0
    assert dev.elements["array"].lines() == []
    assert dev.calib.aod_x.center_mhz == 100.0


def test_no_ops_bound_yet():
    # skeleton: element types have no pulse library, so no dev.<name> ops namespace
    dev = Device.load(YAML)
    assert AODAxis.ops_cls is None
    assert not hasattr(dev, "aod_x")


def test_imaging_readout_is_a_readout_kind():
    r = ImagingReadout(threshold=0.5)
    assert isinstance(r, ReadoutKind)
    assert isinstance(r, Discriminator)
    assert r.data_source is DataSource.EXTERNAL_DETECTOR
    brightness = np.array([0.9, 0.1, 0.8, 0.05])  # per-trap ROI brightness
    assert list(r.discriminate(brightness)) == [1, 0, 1, 0]
    assert float(r.reduce(brightness)) == pytest.approx(0.5)


def test_polychirp_envelope_is_frozen_hashable():
    m = PolyChirp(freq_coeffs=[1, 2, 3, 4, 5, 6], amp_coeffs=[1, 2, 3, 4], duration_us=10.0)
    assert hash(m) == hash(PolyChirp(freq_coeffs=[1, 2, 3, 4, 5, 6], amp_coeffs=[1, 2, 3, 4], duration_us=10.0))
    assert m.freq_coeffs == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
