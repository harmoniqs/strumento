"""Small checks several cqed calibration experiments share, pinned per experiment (pure).

Five experiments reset passively before every block, two build scoped ops from ``overrides=``,
three refuse a readout without g/e blob centres, two look up the qubit-drive gain limit and two
bound the DRAG ``|alpha|``. Each test below runs every experiment that carries the check, so the
returned values, the exception types and the message text stay pinned for all of them whether
the check lives in each class or in one shared helper.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from strumento.core.pulses import Arb, Delay, Measure, Pulse
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments import drag_calibration, drag_pair_calibration
from strumento.packs.cqed.experiments.drag_calibration import DRAGCalibration
from strumento.packs.cqed.experiments.drag_pair_calibration import DRAGPairCalibration
from strumento.packs.cqed.experiments.error_amplification import ErrorAmplification
from strumento.packs.cqed.experiments.ramsey_detuning_pair import RamseyDetuningPair
from strumento.packs.cqed.experiments.transmon_allxy import TransmonAllXY
from strumento.packs.cqed.experiments.transmon_rb import TransmonRB
from strumento.packs.cqed.library.transmon import TransmonOps
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_QUBIT_DRIVE_GAIN

# Small, fast builds of the five passive-reset experiments.
_RESET_BUILDS = {
    "allxy": lambda dev, **kw: TransmonAllXY(dev, **kw),
    "drag": lambda dev, **kw: DRAGCalibration(dev, points=3, **kw),
    "drag_pair": lambda dev, **kw: DRAGPairCalibration(dev, points=3, **kw),
    "ramsey_pair": lambda dev, **kw: RamseyDetuningPair(dev, stop_us=10.0, points=8,
                                                         ramsey_freq_mhz=0.05, **kw),
    "rb": lambda dev, **kw: TransmonRB(dev, depth_start=1, depth_stop=3, points=2, seed=1, **kw),
}


def _block_resets(seq):
    """The leading wait of every measured block, in order."""
    resets, first = [], None
    for op in seq.ops:
        if first is None:
            first = op
        if isinstance(op, Measure):
            assert isinstance(first, Delay), "every block must start from a passive reset"
            resets.append(float(first.duration_us))
            first = None
    return resets


# -- passive reset -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_RESET_BUILDS))
def test_reset_defaults_to_the_readout_relax_delay(dev, name):
    exp = _RESET_BUILDS[name](dev)
    assert exp.reset_delay_us is None
    relax = float(dev.calib.readout.relax_delay)
    assert exp._reset_delay() == relax
    assert isinstance(exp._reset_delay(), float)
    resets = _block_resets(exp.sequence()[0])
    assert resets and all(r == relax for r in resets)


@pytest.mark.parametrize("name", sorted(_RESET_BUILDS))
def test_reset_override_wins_and_is_a_float(dev, name):
    exp = _RESET_BUILDS[name](dev, reset_delay_us=12)
    assert exp.reset_delay_us == 12
    assert exp._reset_delay() == 12.0 and isinstance(exp._reset_delay(), float)
    assert set(_block_resets(exp.sequence()[0])) == {12.0}


@pytest.mark.parametrize("name", sorted(_RESET_BUILDS))
def test_a_zero_reset_override_is_honoured_not_defaulted(dev, name):
    exp = _RESET_BUILDS[name](dev, reset_delay_us=0.0)
    assert exp._reset_delay() == 0.0


# -- scoped overrides ops ------------------------------------------------------------------------

_OPS_BUILDS = {
    "allxy": lambda dev, **kw: TransmonAllXY(dev, **kw),
    "rb": lambda dev, **kw: TransmonRB(dev, depth_start=1, depth_stop=3, points=2, seed=1, **kw),
}


def _override(dev, gain):
    t = np.linspace(-1, 1, 896)
    return Pulse(line=LineRef("qubit", "drive"), freq_mhz=dev.calib.qubit.f_ge, gain=gain,
                 phase_deg=0.0, envelope=Arb(idata=tuple(np.exp(-4 * t**2)), qdata=tuple(0.1 * t)),
                 label="override")


@pytest.mark.parametrize("name", sorted(_OPS_BUILDS))
@pytest.mark.parametrize("overrides", [None, {}])
def test_no_overrides_plays_the_device_ops(dev, name, overrides):
    exp = _OPS_BUILDS[name](dev, overrides=overrides)
    assert exp.overrides == {}
    assert exp._ops() is dev.qubit


@pytest.mark.parametrize("name", sorted(_OPS_BUILDS))
def test_overrides_build_scoped_ops_and_leave_the_device_alone(dev, name):
    given = {"pi_ge": _override(dev, 0.2)}
    exp = _OPS_BUILDS[name](dev, overrides=given)
    assert exp.overrides == given and exp.overrides is not given  # a private copy
    ops = exp._ops()
    assert type(ops) is TransmonOps
    assert ops is not dev.qubit
    assert ops.element is dev.qubit.element
    assert ops.overrides == given
    assert ops.ge_pi().envelope == given["pi_ge"].envelope
    assert dev.qubit.overrides == {}


# -- population anchoring ------------------------------------------------------------------------


@pytest.mark.parametrize("centres", [(None, None), ((0.0, 0.0), None), (None, (1.0, 0.0))])
def test_drag_refuses_without_both_blob_centres(dev, centres):
    ro = DispersiveIQ(threshold=0.5, g_center=centres[0], e_center=centres[1])
    exp = DRAGCalibration(dev, alpha_start=-0.1, alpha_stop=0.1, points=5, readout=ro)
    y = np.linspace(0.3, 0.7, 5)
    fit = exp.analyze(MeasurementResult(xpts=np.arange(5), avgi=y, avgq=np.zeros_like(y)))
    assert fit.success is False
    assert fit.params == {} and fit.stderr == {}
    assert set(fit.derived) == {"alphas", "signal", "population_anchored"}
    assert fit.derived["population_anchored"] == 0.0


@pytest.mark.parametrize("centres", [(None, None), ((0.0, 0.0), None), (None, (1.0, 0.0))])
def test_drag_pair_refuses_without_both_blob_centres(dev, centres):
    ro = DispersiveIQ(threshold=0.5, g_center=centres[0], e_center=centres[1])
    exp = DRAGPairCalibration(dev, alpha_start=-0.1, alpha_stop=0.1, points=5, readout=ro)
    y = np.linspace(0.3, 0.7, 10)
    fit = exp.analyze(MeasurementResult(xpts=np.arange(10), avgi=y, avgq=np.zeros_like(y)))
    assert fit.success is False
    assert fit.params == {} and fit.stderr == {}
    assert set(fit.derived) == {"alphas", "signal_a", "signal_b", "difference",
                                "population_anchored"}
    assert fit.derived["population_anchored"] == 0.0


@pytest.mark.parametrize("centres", [(None, None), ((0.0, 0.0), None), (None, (1.0, 0.0))])
def test_error_amplification_fixed_period_refuses_without_both_blob_centres(dev, centres):
    ro = DispersiveIQ(threshold=0.5, g_center=centres[0], e_center=centres[1])
    exp = ErrorAmplification(dev, parameter="gain", n_pulses=7, points=5,
                             window=Sweep(start=-200.0, stop=200.0, on="ax"),
                             readout=ro, fringe_period=1000.0)
    x = np.linspace(-200.0, 200.0, 5)
    fit = exp.analyze(MeasurementResult(xpts=x, avgi=np.ones_like(x), avgq=np.zeros_like(x)))
    assert fit.success is False
    assert fit.params == {} and fit.stderr == {}
    assert math.isnan(fit.derived["gain_corrected"])
    assert fit.derived["refused"] == (
        "fixed-period fit needs a blob-anchored population (peak vs trough)"
    )
    assert fit.derived["fit_mode"] == "fixed_period"


def test_anchored_readouts_report_population_anchored(dev):
    ro = DispersiveIQ(threshold=0.5, angle=0.0, g_center=(0.0, 0.0), e_center=(1.0, 0.0))
    y = np.linspace(0.3, 0.7, 10)
    res = MeasurementResult(xpts=np.arange(10), avgi=y, avgq=np.zeros_like(y))
    for exp in (DRAGCalibration(dev, points=5, readout=ro),
                DRAGPairCalibration(dev, points=5, readout=ro)):
        assert exp.analyze(res).derived["population_anchored"] == 1.0


# -- qubit-drive gain limit ----------------------------------------------------------------------

_LIMIT_BUILDS = {
    "drag": (drag_calibration, lambda dev: DRAGCalibration(dev, points=3)),
    "drag_pair": (drag_pair_calibration, lambda dev: DRAGPairCalibration(dev, points=3)),
}


def _limits_seen(monkeypatch, module, exp):
    seen = []
    real = module.validate_gain

    def spy(gain, *, limit, **kw):
        seen.append(limit)
        return real(gain, limit=MAX_QUBIT_DRIVE_GAIN, **kw)

    monkeypatch.setattr(module, "validate_gain", spy)
    exp.sequence()
    return set(seen)


@pytest.mark.parametrize("name", sorted(_LIMIT_BUILDS))
@pytest.mark.parametrize("calibrated, expected", [
    (None, MAX_QUBIT_DRIVE_GAIN), (0.42, 0.42), (0.0, 0.0),
])
def test_drive_gain_limit_is_the_calibrated_value_else_the_pack_ceiling(
    dev, monkeypatch, name, calibrated, expected
):
    module, build = _LIMIT_BUILDS[name]
    dev.calib.qubit.max_drive_gain = calibrated
    assert _limits_seen(monkeypatch, module, build(dev)) == {expected}


@pytest.mark.parametrize("name", sorted(_LIMIT_BUILDS))
def test_a_calibrated_drive_limit_below_the_pulse_gain_refuses(dev, name):
    from strumento.packs.cqed.safety import GainOutOfRange

    _, build = _LIMIT_BUILDS[name]
    dev.calib.qubit.max_drive_gain = 1e-6
    with pytest.raises(GainOutOfRange):
        build(dev).sequence()


# -- DRAG |alpha| range --------------------------------------------------------------------------

_ALPHA_CLASSES = {"drag": DRAGCalibration, "drag_pair": DRAGPairCalibration}


@pytest.mark.parametrize("name", sorted(_ALPHA_CLASSES))
def test_alpha_limit_value(name):
    assert _ALPHA_CLASSES[name].ALPHA_LIMIT == 2.0


@pytest.mark.parametrize("name", sorted(_ALPHA_CLASSES))
@pytest.mark.parametrize("start, stop, field, value", [
    (-2.5, 0.1, "alpha_start", -2.5),
    (-0.1, 2.01, "alpha_stop", 2.01),
    (3, 4, "alpha_start", 3),          # the start is reported first
])
def test_alpha_beyond_the_limit_is_refused_with_its_message(dev, name, start, stop, field, value):
    cls = _ALPHA_CLASSES[name]
    msg = (f"{field}={value}: |alpha| > 2.0 is outside the calibrated DRAG "
           "sweep range (gain headroom path)")
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$") as info:
        cls(dev, alpha_start=start, alpha_stop=stop)
    assert type(info.value) is ValueError


@pytest.mark.parametrize("name", sorted(_ALPHA_CLASSES))
def test_alpha_at_the_limit_is_accepted(dev, name):
    exp = _ALPHA_CLASSES[name](dev, alpha_start=-2.0, alpha_stop=2.0, points=3)
    assert (exp.alpha_start, exp.alpha_stop) == (-2.0, 2.0)


@pytest.mark.parametrize("name", sorted(_ALPHA_CLASSES))
def test_alpha_limit_follows_a_subclass_override(dev, name):
    sub = type("Narrow", (_ALPHA_CLASSES[name],), {"ALPHA_LIMIT": 0.5})
    with pytest.raises(ValueError, match=re.escape("alpha_stop=0.6: |alpha| > 0.5 is outside")):
        sub(dev, alpha_start=-0.1, alpha_stop=0.6)


# -- the shared helpers, directly ----------------------------------------------------------------


def test_passive_reset_mixin_reads_the_override_else_the_relax_delay(dev):
    from strumento.packs.cqed.experiments._common import PassiveReset

    class Probe(PassiveReset):
        def __init__(self, device, reset_delay_us):
            self.device, self.reset_delay_us = device, reset_delay_us

    assert Probe(dev, None)._reset_delay() == float(dev.calib.readout.relax_delay)
    assert Probe(dev, 3)._reset_delay() == 3.0
    for name in _RESET_BUILDS:
        assert isinstance(_RESET_BUILDS[name](dev), PassiveReset)


def test_scoped_ops_helper(dev):
    from strumento.packs.cqed.experiments._common import scoped_ops

    assert scoped_ops(dev, None) is dev.qubit
    assert scoped_ops(dev, {}) is dev.qubit
    given = {"pi_ge": _override(dev, 0.2)}
    ops = scoped_ops(dev, given)
    assert type(ops) is TransmonOps and ops.overrides == given
    assert ops.element is dev.qubit.element and dev.qubit.overrides == {}


@pytest.mark.parametrize("g, e, expected", [
    (None, None, False), ((0.0, 0.0), None, False), (None, (1.0, 0.0), False),
    ((0.0, 0.0), (1.0, 0.0), True),
])
def test_is_population_anchored_helper(g, e, expected):
    from strumento.packs.cqed.experiments._common import is_population_anchored

    assert is_population_anchored(DispersiveIQ(threshold=0.5, g_center=g, e_center=e)) is expected


@pytest.mark.parametrize("calibrated, expected", [
    (None, MAX_QUBIT_DRIVE_GAIN), (0.42, 0.42), (0.0, 0.0),
])
def test_drive_gain_limit_helper(dev, calibrated, expected):
    from strumento.packs.cqed.experiments._common import drive_gain_limit

    dev.calib.qubit.max_drive_gain = calibrated
    assert drive_gain_limit(dev) == expected


def test_check_alpha_bounds_helper():
    from strumento.packs.cqed.experiments._common import check_alpha_bounds

    assert check_alpha_bounds(-2.0, 2.0, limit=2.0) is None
    msg = ("alpha_stop=2.5: |alpha| > 2.0 is outside the calibrated DRAG "
           "sweep range (gain headroom path)")
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        check_alpha_bounds(0.0, 2.5, limit=2.0)
    with pytest.raises(ValueError, match=re.escape("alpha_start=-3: |alpha| > 1 is outside")):
        check_alpha_bounds(-3, 5, limit=1)
