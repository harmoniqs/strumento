"""SidebandRabi sweep_param: it must honor length sweeps, not only gain (previously sweep_param
was accepted but ignored — sequence() always swept gain). Pure: no qick needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.pulses import FlatTop, Pulse
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.experiments.sideband_rabi import SidebandRabi
from strumento.packs.cqed.readout import DispersiveIQ


def _sideband_pulse(seq) -> Pulse:
    for op in seq.ops:
        if isinstance(op, Pulse) and op.label and op.label.startswith("f0-g1"):
            return op
    raise AssertionError("no f0-g1 pulse found in the sequence")


def test_gain_sweep_is_the_default(dev):
    exp = SidebandRabi(dev, points=41)
    assert exp.sweep_param == "gain"
    assert exp.axes()[0].name == "amp"
    pulse = _sideband_pulse(exp.sequence()[0])
    assert isinstance(pulse.gain, Sweep) and pulse.gain.on == "amp"
    # amplitude is swept, the flat length stays the calibrated scalar
    assert not isinstance(pulse.envelope.length_us, Sweep)


def test_sideband_rabi_maps_residual_f_to_e_before_readout(dev):
    exp = SidebandRabi(dev, points=41)
    labels = [op.label for op in exp.sequence()[0].ops if isinstance(op, Pulse)]
    sideband_index = next(i for i, label in enumerate(labels) if label.startswith("f0-g1"))

    assert labels[:sideband_index] == ["pi_ge", "pi_ef"]
    assert labels[sideband_index + 1] == "pi_ef"
    assert labels.count("pi_ef") == 2


def test_length_sweep_sweeps_the_flat_duration(dev):
    exp = SidebandRabi(dev, sweep_param="length", lengths=Sweep(start=0.0, stop=2.5, on="t"), points=41)
    assert exp.axes()[0].name == "t"
    pulse = _sideband_pulse(exp.sequence()[0])
    assert isinstance(pulse.envelope, FlatTop)
    assert isinstance(pulse.envelope.length_us, Sweep) and pulse.envelope.length_us.on == "t"
    assert pulse.envelope.length_us.stop == pytest.approx(2.5)
    # amplitude is fixed at the calibrated value (not swept) during a length Rabi
    assert not isinstance(pulse.gain, Sweep)


def test_length_sweep_can_override_the_fixed_gain_code(dev):
    calibrated = SidebandRabi(
        dev,
        sweep_param="length",
        lengths=Sweep(start=0.02, stop=1.5, on="t"),
        points=41,
    )
    overridden = SidebandRabi(
        dev,
        sweep_param="length",
        fixed_gain_code=2700,
        lengths=Sweep(start=0.02, stop=1.5, on="t"),
        points=41,
    )

    calibrated_pulse = _sideband_pulse(calibrated.sequence()[0])
    overridden_pulse = _sideband_pulse(overridden.sequence()[0])

    assert not isinstance(overridden_pulse.gain, Sweep)
    assert float(overridden_pulse.gain) < float(calibrated_pulse.gain)


def test_length_sweep_can_override_the_f0g1_frequency(dev):
    exp = SidebandRabi(
        dev,
        sweep_param="length",
        fixed_gain_code=2700,
        fixed_frequency_mhz=2006.25,
        lengths=Sweep(start=0.02, stop=1.5, on="t"),
        points=41,
    )

    pulse = _sideband_pulse(exp.sequence()[0])

    assert float(pulse.freq_mhz) == pytest.approx(2006.25)


def test_invalid_sweep_param_raises(dev):
    with pytest.raises(ValueError, match="sweep_param"):
        SidebandRabi(dev, sweep_param="frequency")


def test_length_rabi_reports_mapped_transfer(dev):
    xpts = np.linspace(0.0, 2.0, 101)
    p_g = np.sin(np.pi * xpts / 2.0) ** 2
    p_e = 1.0 - p_g  # residual f is mapped to e by the closing ef pi
    g, e, f = np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([0.0, 2.0])
    raw = p_g[:, None] * g + p_e[:, None] * e
    result = MeasurementResult(
        xpts=xpts,
        avgi=raw[:, 0][None, :],
        avgq=raw[:, 1][None, :],
    )
    readout = DispersiveIQ(
        threshold=0.0,
        angle=np.pi / 2.0,
        g_center=tuple(g),
        e_center=tuple(e),
        f_center=tuple(f),
    )

    fit = SidebandRabi(
        dev,
        sweep_param="length",
        lengths=Sweep(start=0.0, stop=2.0, on="t"),
        points=len(xpts),
        readout=readout,
    ).analyze(result)

    assert fit.success is True
    assert fit.derived["initial_mapped_e_population"] == pytest.approx(1.0)
    assert fit.derived["max_g_population"] == pytest.approx(1.0)
    assert fit.derived["max_g_length"] == pytest.approx(1.0)
    assert fit.derived["pi_gain"] == pytest.approx(1.0)
    assert fit.derived["residual_mapped_e_at_max_g"] == pytest.approx(0.0)


def test_gain_rabi_reports_the_best_gain_not_a_length(dev):
    xpts = np.linspace(0.0, 6000.0, 101)
    p_g = np.sin(np.pi * xpts / 8000.0) ** 2
    p_e = 1.0 - p_g
    g, e, f = np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([0.0, 2.0])
    raw = p_g[:, None] * g + p_e[:, None] * e
    result = MeasurementResult(
        xpts=xpts,
        avgi=raw[:, 0][None, :],
        avgq=raw[:, 1][None, :],
    )
    readout = DispersiveIQ(
        threshold=0.0,
        angle=np.pi / 2.0,
        g_center=tuple(g),
        e_center=tuple(e),
        f_center=tuple(f),
    )

    fit = SidebandRabi(
        dev,
        sweep_param="gain",
        gains=Sweep(start=0.0, stop=6000.0, on="amp"),
        points=len(xpts),
        readout=readout,
    ).analyze(result)

    assert fit.success is True
    assert fit.derived["max_g_gain"] == pytest.approx(4020.0)
    assert fit.derived["pi_gain"] == pytest.approx(4020.0)
    assert "max_g_length" not in fit.derived
