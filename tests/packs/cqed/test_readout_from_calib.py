"""Experiments read with the recorded readout calibration by default (#196).

When no ``readout=`` is passed, every cqed experiment builds its discriminator from the device's
``ReadoutCal`` — threshold, angle and the g/e/f blob centres — through
:meth:`DispersiveIQ.from_calib`. Before this, 46 experiments fell back to a bare
``DispersiveIQ(threshold=0.0, angle=0.0)`` and the bring-up readout dropped the centres, so an
analyzer that needs a blob-anchored population refused or failed on a device that had recorded
them, even though the simulator already placed its shots there.

The registry-driven guard below is built from ``EXPERIMENTS`` + ``inspect.signature``, so a newly
registered experiment that takes ``readout=`` is covered without editing this file.

The numbers are made up. The geometry is deliberately not the trivial one: a rotated
discrimination axis (angle near -pi/2, in radians) with the blobs separated mainly along Q.
"""

from __future__ import annotations

import inspect
import math

import pytest

from strumento.core.device import Device
from strumento.packs.cqed import EXPERIMENTS as CQED_EXPERIMENTS
from strumento.packs.cqed.calibration import ReadoutCal
from strumento.packs.cqed.readout import DispersiveIQ
from tests._paths import DEVICES, LOOPBACK_YAML, MULTIMODE_YAML

#: A made-up recorded readout: rotated axis (rad), centres separated mainly in Q.
THRESHOLD = 0.37
ANGLE = -1.52
G_CENTER = (0.12, -0.41)
E_CENTER = (0.18, 0.66)
F_CENTER = (0.93, 0.21)

#: Experiments whose default discriminator is deliberately NOT the plain calibrated one, each with
#: a written reason and the readout it must carry instead. Keep this short and justified.
DELIBERATE_DEFAULTS: dict[str, tuple[str, object]] = {
    # ActiveReset's host threshold is its own ``threshold`` argument — the same projected-units
    # value the on-board branch uses when ``branch_threshold`` is None — and its angle is tied to
    # ``readout_phase_deg`` (an --arm run rotates in the DDC and reads with angle 0). Both are
    # coupled to what the board plays, which #196 leaves out of scope; only the analysis-side
    # centres are taken from the calibration.
    "active_reset": (
        "threshold/angle are coupled to the in-program reset arguments (out of scope for #196)",
        lambda exp, ro: DispersiveIQ(
            threshold=exp.threshold, angle=0.0,
            g_center=ro.g_center, e_center=ro.e_center, f_center=ro.f_center,
        ),
    ),
}


def _recorded_device() -> Device:
    dev = Device.load(MULTIMODE_YAML)
    ro = dev.calib.readout
    ro.threshold = THRESHOLD
    ro.angle = ANGLE
    ro.g_center_iq = G_CENTER
    ro.e_center_iq = E_CENTER
    ro.f_center_iq = F_CENTER
    return dev


def _cal(**over) -> ReadoutCal:
    base = dict(frequency=7000.0, gain=1000, readout_length=2.0, trig_offset=0.5, relax_delay=100.0)
    base.update(over)
    return ReadoutCal(**base)


def _takes_readout(cls) -> bool:
    return "readout" in inspect.signature(cls.__init__).parameters


READOUT_EXPERIMENTS = sorted(name for name, cls in CQED_EXPERIMENTS.items() if _takes_readout(cls))

#: The target an experiment cannot infer on multimode_demo, which calibrates two coupler swaps
#: (``library.targets``): every other experiment constructs from the device alone.
TARGET_KWARGS = {
    "swap_rabi": {"pair": "cavity-S1"},
    "swap_spectroscopy": {"pair": "cavity-S1"},
    "storage_t1": {"mode_name": "S1"},
}


def _build(name, dev, **kwargs):
    return CQED_EXPERIMENTS[name](dev, **TARGET_KWARGS.get(name, {}), **kwargs)


# -- from_calib (pure) -------------------------------------------------------------------------


def test_from_calib_carries_threshold_angle_and_every_centre():
    ro = DispersiveIQ.from_calib(_cal(
        threshold=THRESHOLD, angle=ANGLE,
        g_center_iq=G_CENTER, e_center_iq=E_CENTER, f_center_iq=F_CENTER,
    ))
    assert ro == DispersiveIQ(
        threshold=THRESHOLD, angle=ANGLE,
        g_center=G_CENTER, e_center=E_CENTER, f_center=F_CENTER,
    )


def test_from_calib_unset_threshold_and_angle_are_zero_and_unset_centres_stay_none():
    assert DispersiveIQ.from_calib(_cal()) == DispersiveIQ(threshold=0.0, angle=0.0)


def test_from_calib_takes_each_centre_independently():
    ro = DispersiveIQ.from_calib(_cal(g_center_iq=G_CENTER, e_center_iq=E_CENTER))
    assert (ro.g_center, ro.e_center, ro.f_center) == (G_CENTER, E_CENTER, None)
    only_e = DispersiveIQ.from_calib(_cal(e_center_iq=E_CENTER))
    assert (only_e.g_center, only_e.e_center, only_e.f_center) == (None, E_CENTER, None)


def test_from_calib_centres_are_float_tuples_even_when_recorded_as_lists():
    cal = _cal()
    cal.g_center_iq = [0, -1]  # a writeback may assign a list; DispersiveIQ compares tuples
    ro = DispersiveIQ.from_calib(cal)
    assert ro.g_center == (0.0, -1.0)
    assert all(isinstance(v, float) for v in ro.g_center)


def test_from_calib_angle_stays_in_radians_and_anchors_a_q_separated_population():
    # ReadoutCal.angle is rad and DispersiveIQ.angle is rad: no conversion. With the axis at
    # ~+pi/2 (pure Q) the recorded centres map to P_e = 0 and 1 through to_population.
    ro = DispersiveIQ.from_calib(_cal(
        angle=math.pi / 2, g_center_iq=(0.05, -0.5), e_center_iq=(0.07, 0.5),
    ))
    assert ro.angle == pytest.approx(math.pi / 2)
    assert ro.to_population((0.05, -0.5)) == pytest.approx(0.0)
    assert ro.to_population((0.07, 0.5)) == pytest.approx(1.0)


# -- every experiment that takes readout= defaults to it (pure — construction only) ------------


def test_the_readout_guard_is_not_vacuous():
    # 46 bare defaults + validation_check + active_reset at the time of #196.
    assert len(READOUT_EXPERIMENTS) >= 48, READOUT_EXPERIMENTS
    stale = sorted(set(DELIBERATE_DEFAULTS) - set(READOUT_EXPERIMENTS))
    assert not stale, f"DELIBERATE_DEFAULTS names experiments without a readout= argument: {stale}"


@pytest.mark.parametrize("name", READOUT_EXPERIMENTS)
def test_every_readout_experiment_defaults_to_the_recorded_calibration(name):
    dev = _recorded_device()
    exp = _build(name, dev)
    expected = DispersiveIQ.from_calib(dev.calib.readout)
    # not tautological: from_calib on this device is the full recorded readout
    assert expected == DispersiveIQ(
        threshold=THRESHOLD, angle=ANGLE,
        g_center=G_CENTER, e_center=E_CENTER, f_center=F_CENTER,
    )
    if name in DELIBERATE_DEFAULTS:
        _reason, build = DELIBERATE_DEFAULTS[name]
        expected = build(exp, expected)
    assert exp.readout == expected


@pytest.mark.parametrize("name", READOUT_EXPERIMENTS)
def test_an_explicit_readout_still_wins(name):
    explicit = DispersiveIQ(threshold=0.5)
    exp = _build(name, _recorded_device(), readout=explicit)
    assert exp.readout is explicit


# -- the bring-up readout ----------------------------------------------------------------------


def test_bringup_readout_carries_the_recorded_centres():
    from strumento.packs.cqed import bringup

    dev = _recorded_device()
    expected = DispersiveIQ.from_calib(dev.calib.readout)
    assert bringup._dispersive_readout(dev) == expected
    # through the node factories the graph actually runs
    assert bringup._t1_factory(dev, None).readout == expected
    node = bringup._validation_factory(dev, None)
    assert node.inner.readout == expected
    assert node.inner.readout.g_center == G_CENTER
    assert node.inner.readout.e_center == E_CENTER
    assert node.inner.readout.f_center == F_CENTER


# -- the shipped demos are unchanged -----------------------------------------------------------


@pytest.mark.parametrize("path", [MULTIMODE_YAML, LOOPBACK_YAML])
def test_shipped_demos_read_exactly_the_old_default(path):
    dev = Device.load(path)
    assert DispersiveIQ.from_calib(dev.calib.readout) == DispersiveIQ(threshold=0.0, angle=0.0)


# -- AC1: the blob-anchored analyzers run on the recorded centres (qick lane) ------------------


def _blob_device() -> Device:
    dev = Device.load(MULTIMODE_YAML)
    dev.calib.readout.g_center_iq = (0.0, 0.0)
    dev.calib.readout.e_center_iq = (1.0, 0.0)
    return dev


@pytest.mark.qick
@pytest.mark.parametrize("name", ["drag_calibration", "drag_pair_calibration"])
def test_drag_fits_on_recorded_centres_without_an_explicit_readout(name):
    pytest.importorskip("qick")
    import strumento.packs.cqed.responses  # noqa: F401
    from strumento.core.simulator import SimulatorSoc
    from strumento.packs.cqed.responses import truth_from_device

    dev = _blob_device()
    soc = SimulatorSoc.from_snapshot(
        DEVICES / "_fixtures" / "soccfg_v2_cqed.json", truth=truth_from_device(dev), seed=3
    )
    out = CQED_EXPERIMENTS[name](dev).go(soc, reps=500, progress=False)
    assert out.fit.success, out.fit.derived


@pytest.mark.qick
def test_fixed_period_error_amplification_is_not_refused_without_an_explicit_readout():
    pytest.importorskip("qick")
    import strumento.packs.cqed.responses  # noqa: F401
    from strumento.core.simulator import SimulatorSoc
    from strumento.core.sweeps import Sweep
    from strumento.packs.cqed.responses import truth_from_device

    dev = _blob_device()
    g0 = float(dev.calib.qubit.pulses["pi_ge"].gain)
    n_pulses = 7
    period = 2.0 * g0 / n_pulses
    span = 0.4 * period
    exp = CQED_EXPERIMENTS["error_amplification"](
        dev, pulse=dev.qubit.ge_pi, parameter="gain", n_pulses=n_pulses,
        window=Sweep(start=g0 - span / 2, stop=g0 + span / 2, on="err_amp"),
        points=41, fringe_period=period,
    )
    soc = SimulatorSoc.from_snapshot(
        DEVICES / "_fixtures" / "soccfg_v2_cqed.json", truth=truth_from_device(dev), seed=3
    )
    out = exp.go(soc, reps=500, progress=False)
    assert "refused" not in out.fit.derived, out.fit.derived
    assert out.fit.success, out.fit.derived


def test_wigner_2d_star_defaults_to_the_recorded_calibration():
    # not a registered Experiment (a host-side driver over WignerTomography rays), so the registry
    # guard above does not reach it; its per-ray readout must still be the recorded one
    from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography2D

    dev = _recorded_device()
    star = WignerTomography2D(dev, soc=None, n_radial=3, n_angle=1)
    assert star.readout == DispersiveIQ.from_calib(dev.calib.readout)


# -- AC5: the default readout is analysis-side; it never changes what the board plays (qick) ----


def _compiled(name: str, *, record: bool, readout) -> list[dict]:
    from strumento.core.mock import MockQickSocV2
    from tests import _compiled_golden as golden
    from tests.packs import test_experiment_contract as contract

    dev, _ = contract._contract_device_and_soccfg(name)
    if record:
        ro = dev.calib.readout
        ro.threshold, ro.angle = THRESHOLD, ANGLE
        ro.g_center_iq, ro.e_center_iq, ro.f_center_iq = G_CENTER, E_CENTER, F_CENTER
    kwargs = {k: v for k, v in contract.CONTRACT_KWARGS[name].items() if k != "readout"}
    if readout is not None:
        kwargs["readout"] = readout
    exp = CQED_EXPERIMENTS[name](dev, **kwargs)
    runner = golden._CapturingRunner()
    exp.runner = runner
    exp.go(MockQickSocV2(dev.soccfg_snapshot), analyze=False, progress=False)
    assert runner.programs, f"{name}: no program was built"
    return runner.programs


@pytest.mark.qick
@pytest.mark.parametrize("name", READOUT_EXPERIMENTS)
def test_the_calibrated_default_readout_compiles_the_same_program(name):
    """The default readout built from the recorded calibration compiles byte-for-byte the program
    the old bare default did on the un-recorded device: the readout object never reaches the board."""
    pytest.importorskip("qick")
    recorded = _compiled(name, record=True, readout=None)
    bare = _compiled(name, record=False, readout=DispersiveIQ(threshold=0.0, angle=0.0))
    assert recorded == bare
