"""``SimulatorSoc`` (D32) — real v2 assembly, simulated acquire.

The whole module is marked ``qick``: the simulator is one of the qick-facing core modules (it
subclasses :class:`~strumento.core.mock.MockQickSocV2`, which *is* a ``QickConfig``), so the
purity boundary for it means **the pure lane never imports it** — the marker is the enforcement.

What is asserted here:

* the snapshot gate — a tProc-**v1** snapshot (multimode_demo with the v1 ZCU216 config attached) is a
  typed error that names ``multimode_demo`` as the cqed exemplar; the committed v2 testbench
  snapshot constructs;
* arming — an unarmed acquire is a typed :class:`ResponseError`, never zeros (that behavior stays
  in the mock, whose contract is untouched);
* the data path — an armed program's per-shot draws trace the armed response model, the averaged
  IQ is an affine image of it, and the per-shot array is what ``DispersiveIQ`` consumes;
* determinism — a pinned seed pins the data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qick")

sys.path.insert(0, str(Path(__file__).parents[1]))
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

from strumento.core.device import Device  # noqa: E402
from strumento.core.response import ResponseError, SimContext, SimResponse  # noqa: E402
from strumento.core.simulator import NonV2SnapshotError, SimulatorSoc  # noqa: E402
from strumento.core.soc import AbstractSoc, ArmableSoc, ShotSource  # noqa: E402
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi  # noqa: E402
from strumento.packs.cqed.readout import DispersiveIQ  # noqa: E402
from tests._paths import DEVICES  # noqa: E402

pytestmark = pytest.mark.qick

V1_SNAP = DEVICES / "_fixtures" / "soccfg_v1_zcu216.json"
TESTBENCH = DEVICES / "_fixtures" / "soccfg_v2_testbench.json"
LOOPBACK_YAML = DEVICES / "loopback_demo" / "device.yaml"

TRUE_PI_GAIN = 9000.0
REPS = 400


def _rabi_populations(x) -> np.ndarray:
    """The response under test: a clean Rabi arch in the lab-native gain axis."""
    g = np.asarray(x, dtype=float)
    return 0.02 + 0.96 * 0.5 * (1.0 - np.cos(np.pi * g / TRUE_PI_GAIN))


# -- the snapshot gate ---------------------------------------------------------------------


def test_v1_snapshot_is_a_typed_error_naming_the_v2_exemplar():
    with pytest.raises(NonV2SnapshotError) as err:
        SimulatorSoc.from_snapshot(V1_SNAP)
    msg = str(err.value)
    assert "tProc v2" in msg
    assert "multimode_demo" in msg  # the cqed exemplar, a device with a v2 snapshot


def test_v1_device_instance_is_rejected_through_from_device(v1_snapshot_yaml):
    dev = Device.load(v1_snapshot_yaml)
    with pytest.raises(NonV2SnapshotError, match="multimode_demo"):
        SimulatorSoc.from_device(dev)


def test_the_snapshot_errors_point_at_multimode_demo_not_loopback_demo():
    # #131: loopback_demo is the single-channel testbench, not the cqed exemplar.
    with pytest.raises(NonV2SnapshotError) as v1:
        SimulatorSoc.from_snapshot(V1_SNAP)
    with pytest.raises(NonV2SnapshotError) as none:
        SimulatorSoc.from_device(SimpleNamespace(soccfg_snapshot=None))
    for msg in (str(v1.value), str(none.value)):
        assert "multimode_demo" in msg and "loopback_demo" not in msg, msg
    assert "tProc v2" in str(v1.value)
    assert "no soccfg snapshot" in str(none.value)


def test_v2_testbench_snapshot_constructs_a_real_qickconfig():
    soc = SimulatorSoc.from_snapshot(TESTBENCH)
    assert soc["tprocs"][0]["type"] == "qick_processor"
    assert soc.freq2reg(750.0, gen_ch=0) != 0  # real unit conversions => real assembly
    assert isinstance(soc, AbstractSoc) and isinstance(soc, ArmableSoc)
    assert isinstance(soc, ShotSource)


# -- arming --------------------------------------------------------------------------------


def test_unarmed_poll_data_is_a_typed_error_not_zeros():
    soc = SimulatorSoc.from_snapshot(TESTBENCH)
    soc.start_readout(total_count=4, counter_addr=1, ch_list=[0], reads_per_shot=[1])
    with pytest.raises(ResponseError, match="not been armed"):
        soc.poll_data()


def test_arming_an_unregistered_experiment_is_a_typed_error():
    class Unmodelled:  # no registered response, no hook
        pass

    soc = SimulatorSoc.from_snapshot(TESTBENCH)
    with pytest.raises(ResponseError, match="Unmodelled"):
        soc.arm(SimContext(experiment=Unmodelled(), xpts=(0.0, 1.0)))


def test_explicit_model_is_the_raw_path_escape_hatch():
    soc = SimulatorSoc.from_snapshot(TESTBENCH, seed=3)
    soc.arm(SimContext(experiment=None, xpts=(0.0,), model=lambda ctx: np.array([1.0])))
    soc.start_readout(total_count=16, counter_addr=1, ch_list=[0], reads_per_shot=[1])
    (count, (d, _)), = soc.poll_data()
    assert count == 16
    assert d[0].shape == (16, 2) and d[0].dtype == np.int64
    # population 1.0 => every shot sits on blob1 (unit-scale default), scaled to ADC counts
    assert np.all(d[0][:, 0] > 0)


def test_truth_reaches_the_model_through_the_context():
    seen = {}

    def model(ctx):
        seen.update(ctx.truth)
        return np.array([0.5])

    soc = SimulatorSoc.from_snapshot(TESTBENCH, truth={"pi_gain": 1234.0}, model=model)
    soc.arm(SimContext(experiment=None, xpts=(0.0,)))
    soc.start_readout(total_count=8, counter_addr=1, ch_list=[0], reads_per_shot=[1])
    soc.poll_data()
    assert seen == {"pi_gain": 1234.0}


# -- the data path: a real swept program, simulated acquire --------------------------------


def _armed_rabi(seed=5, sigma=0.0):
    """Assemble the real AmplitudeRabi program against an armed simulator and acquire it."""
    from strumento.core.program import StrumentoProgram

    dev = Device.load(LOOPBACK_YAML)
    # loopback_demo has no dedicated readout line (single-generator instance) — probe on the
    # qubit line instead, same as resonator_spectroscopy's own test override (issue #102).
    exp = AmplitudeRabi(dev, line=("qubit", "drive"))
    soc = SimulatorSoc.from_snapshot(V2_SOCCFG, seed=seed)
    seq, sweeps = exp.sequence()
    prog = StrumentoProgram(dev, seq=seq, sweeps=sweeps, axes=exp.axes(), reps=REPS, soc=soc)
    soc.arm(
        SimContext(
            experiment=exp,
            device=dev,
            xpts=prog.xpts(),
            model=lambda ctx: SimResponse(
                populations=_rabi_populations(ctx.x()),
                blob0=(0.0, 0.0),
                blob1=(1.0, 0.0),
                sigma=sigma,
            ),
        )
    )
    result = prog.acquire(soc, progress=False)
    return exp, soc, result


@requires_v2_soccfg
def test_armed_program_assembles_and_returns_the_swept_axis():
    exp, soc, result = _armed_rabi()
    xpts = np.asarray(result.xpts, dtype=float)
    assert xpts.shape == (exp.points,)
    assert np.asarray(result.avgi).shape == (1, exp.points)  # (n_reads, n_expts)


@requires_v2_soccfg
def test_per_shot_draws_trace_the_response_model():
    exp, soc, result = _armed_rabi(seed=11, sigma=0.08)
    shots = soc.collect_shots()
    n = exp.points
    assert shots.shape == (n, REPS, 2)  # (sweep, shot, IQ) — the per-shot convention

    truth = _rabi_populations(np.asarray(result.xpts, dtype=float))
    measured = DispersiveIQ(threshold=0.5).reduce(shots)  # per-shot discrimination path
    tol = 5 * np.sqrt(np.clip(truth * (1 - truth), 1e-3, None) / REPS)
    assert np.all(np.abs(measured - truth) < tol)


@requires_v2_soccfg
def test_averaged_iq_is_an_affine_image_of_the_response():
    exp, soc, result = _armed_rabi(seed=13, sigma=0.05)
    signal = np.asarray(result.avgi, dtype=float).ravel()
    truth = _rabi_populations(np.asarray(result.xpts, dtype=float))
    r = float(np.corrcoef(signal, truth)[0, 1])
    # averaged acquire = the same populations, affinely scaled by the readout. The residual is
    # shot noise: sqrt(p(1-p)/reps) ~ 0.025 against a unit-range signal, so r ~ 1 - 3e-3.
    assert r > 0.99


@requires_v2_soccfg
def test_the_same_seed_gives_the_same_data():
    _, _, a = _armed_rabi(seed=99, sigma=0.2)
    _, _, b = _armed_rabi(seed=99, sigma=0.2)
    assert np.array_equal(np.asarray(a.avgi), np.asarray(b.avgi))


@requires_v2_soccfg
def test_different_seeds_give_different_noise():
    _, _, a = _armed_rabi(seed=1, sigma=0.2)
    _, _, b = _armed_rabi(seed=2, sigma=0.2)
    assert not np.array_equal(np.asarray(a.avgi), np.asarray(b.avgi))


# -- complex (transmission) responses ------------------------------------------------------


def test_complex_response_lands_in_i_and_q():
    soc = SimulatorSoc.from_snapshot(TESTBENCH, seed=2)
    soc.arm(
        SimContext(
            experiment=None,
            xpts=(0.0, 1.0),
            model=lambda ctx: np.array([1.0 + 0.0j, 0.0 + 1.0j]),
        )
    )
    soc.start_readout(total_count=2 * 32, counter_addr=1, ch_list=[0], reads_per_shot=[1])
    (_, (d, _)), = soc.poll_data()
    shots = soc.collect_shots()
    assert shots.shape == (2, 32, 2)
    assert d[0].shape == (64, 2)
    # point 0 is purely real, point 1 purely imaginary (no scatter: sigma defaults to 0)
    assert np.allclose(shots[0].mean(axis=0), (1.0, 0.0))
    assert np.allclose(shots[1].mean(axis=0), (0.0, 1.0))


def test_response_shape_mismatch_is_a_typed_error():
    soc = SimulatorSoc.from_snapshot(TESTBENCH)
    soc.arm(SimContext(experiment=None, xpts=(0.0, 1.0, 2.0), model=lambda ctx: np.zeros(7)))
    soc.start_readout(total_count=3, counter_addr=1, ch_list=[0], reads_per_shot=[1])
    with pytest.raises(ResponseError, match="shape"):
        soc.poll_data()


def test_sweep_axis_must_divide_the_acquisition_count():
    soc = SimulatorSoc.from_snapshot(TESTBENCH)
    soc.arm(SimContext(experiment=None, xpts=(0.0, 1.0, 2.0), model=lambda ctx: np.zeros(3)))
    soc.start_readout(total_count=8, counter_addr=1, ch_list=[0], reads_per_shot=[1])
    with pytest.raises(ResponseError, match="sweep"):
        soc.poll_data()


def test_multiple_reads_per_shot_broadcast_the_response():
    soc = SimulatorSoc.from_snapshot(TESTBENCH, seed=4)
    soc.arm(SimContext(experiment=None, xpts=(0.0, 1.0), model=lambda ctx: np.array([0.0, 1.0])))
    soc.start_readout(total_count=2 * 10, counter_addr=1, ch_list=[0], reads_per_shot=[2])
    (_, (d, _)), = soc.poll_data()
    assert d[0].shape == (2 * 10 * 2, 2)  # (points * shots * reads, IQ)
    assert soc.collect_shots().shape == (2, 2, 10, 2)  # (sweep, read, shot, IQ)


# -- the mock's contract is untouched ------------------------------------------------------


def test_mock_still_returns_zeros():
    from strumento.core.mock import MockQickSocV2

    mock = MockQickSocV2.from_snapshot(TESTBENCH)
    mock.start_readout(total_count=4, counter_addr=1, ch_list=[0], reads_per_shot=[1])
    (_, (d, _)), = mock.poll_data()
    assert not np.any(d[0])  # the zero-IQ contract stays exactly as it was
