"""Spin pack buildout — DC seam, LD/exchange ops, charge stability, PSB histogram (pure)."""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.dc import DCAxis, DCSource, MockVoltageSource, VoltageSource
from strumento.core.pulses import FlatTop, Gaussian
from strumento.core.wiring import LineRef
from strumento.packs.spin.experiments.charge_stability import ChargeStability
from strumento.packs.spin.experiments.psb_histogram import PSBHistogram
from tests._paths import DEVICES

YAML = DEVICES / "spin_demo" / "device.yaml"


@pytest.fixture
def dev(spin_dev):
    return spin_dev


def _dc(dev):
    axes = {
        name: DCAxis(channel=g.dc_channel, max_v=g.max_v, crosscoupling=g.crosscoupling)
        for name, g in dev.calib.gates.items()
    }
    return DCSource(MockVoltageSource(), axes)


# -- DC seam ------------------------------------------------------------------------------


def test_mock_source_satisfies_protocol():
    assert isinstance(MockVoltageSource(), VoltageSource)


def test_dc_set_get_and_clamp(dev):
    dc = _dc(dev)
    dc.set("p0", 0.8)
    assert dc.get("p0") == pytest.approx(0.8)
    with pytest.raises(ValueError, match="max_v"):
        dc.set("p0", 2.0)
    with pytest.raises(KeyError):
        dc.set("nope", 0.1)


def test_dc_compensated_move_counteracts_crosscoupling(dev):
    dc = _dc(dev)
    dc.set("b0", 0.5)
    dc.set_compensated("p0", 1.0)          # p0 moves by +1.0; b0 compensates by -0.12
    assert dc.get("p0") == pytest.approx(1.0)
    assert dc.get("b0") == pytest.approx(0.5 - 0.12 * 1.0)
    snap = dc.snapshot()
    assert set(snap) == {"p0", "b0"}


# -- ops ----------------------------------------------------------------------------------


def test_ld_ops_from_calibration(dev):
    p = dev.q0.pi()
    assert p.line == LineRef("q0", "drive")
    assert p.freq_mhz == pytest.approx(5000.0)
    assert p.gain == pytest.approx(0.4)                    # fraction-native, no conversion
    assert p.envelope == Gaussian(sigma_us=0.020)
    assert dev.q0.hpi(phase=90.0).phase_deg == pytest.approx(90.0)


def test_exchange_ops_from_calibration(dev):
    p = dev.b0.pulse()
    assert p.line == LineRef("b0", "fast")
    assert p.freq_mhz == 0.0                               # baseband
    assert p.gain == pytest.approx(0.5)
    assert p.envelope == FlatTop(ramp_sigma_us=0.006, length_us=0.05)


# -- charge stability ---------------------------------------------------------------------


def test_charge_stability_grid_and_snapshot(dev):
    dc = _dc(dev)

    def probe():
        # a fake sensor: a charge transition when p0 + 0.3*b0 crosses 0.5 V
        return 1.0 if dc.get("p0") + 0.3 * dc.get("b0") > 0.5 else 0.0

    exp = ChargeStability(
        dc, x_axis="p0", x_volts=np.linspace(0, 1, 11), y_axis="b0",
        y_volts=np.linspace(0, 1, 5), probe=probe,
    )
    res = exp.run()
    assert np.asarray(res.avgi).shape == (5, 11)
    assert res.meta["y_axis"] == "b0" and len(res.meta["ypts"]) == 5
    assert "dc_snapshot" in res.meta
    grid = np.asarray(res.avgi)
    assert grid[0, 0] == 0.0 and grid[-1, -1] == 1.0       # the transition is in the window


# -- PSB histogram ------------------------------------------------------------------------


def test_psb_histogram_threshold_recovery():
    rng = np.random.default_rng(7)
    singlets = rng.normal(-1.0, 0.25, 4000)
    triplets = rng.normal(1.0, 0.25, 2000)
    shots = np.concatenate([singlets, triplets])
    fit = PSBHistogram().analyze(shots)
    assert fit.success
    assert fit.derived["threshold"] == pytest.approx(0.0, abs=0.15)
    assert fit.derived["separation"] > 5.0
