"""RamseyDetuningPair — the signed qubit detuning from a ±Δa Ramsey pair.

Pure: the branch resolver (every branch, the ambiguity a single Ramsey cannot break, refusal).
qick: fit-recovery through the real path on an armed simulator, for both detuning signs and the
outer branch — the sign must come out right, not just the magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.packs.cqed.experiments.ramsey_detuning_pair import (
    RamseyDetuningPair,
    resolve_detuning,
)
from tests._paths import DEVICES

MULTIMODE_YAML = DEVICES / "multimode_demo" / "device.yaml"
CQED_SOCCFG = DEVICES / "_fixtures" / "soccfg_v2_cqed.json"


def _fringes(delta, da):
    return abs(da - delta), abs(da + delta)


@pytest.mark.parametrize("delta", [-0.02, 0.02, 0.0, -0.049, 0.031])
def test_inner_branch_recovers_the_signed_detuning(delta):
    fp, fm = _fringes(delta, 0.05)
    out = resolve_detuning(fp, fm, 0.05, tol=1e-6)
    assert out["branch"] == "inner"
    assert out["delta"] == pytest.approx(delta, abs=1e-12)


@pytest.mark.parametrize("delta,branch", [(0.12, "outer+"), (-0.12, "outer-")])
def test_outer_branches_recover_the_signed_detuning(delta, branch):
    fp, fm = _fringes(delta, 0.05)
    out = resolve_detuning(fp, fm, 0.05, tol=1e-6)
    assert out["branch"] == branch
    assert out["delta"] == pytest.approx(delta, abs=1e-12)


def test_the_single_ramsey_ambiguity_is_broken():
    # A 70 kHz fringe at Δa = +50 kHz: both δ = −20 kHz and δ = +120 kHz give f₊ = 70 kHz; the −Δa
    # fringe separates them.
    da = 0.05
    for delta in (-0.02, 0.12):
        assert _fringes(delta, da)[0] == pytest.approx(0.07)
    assert resolve_detuning(0.07, 0.03, da, tol=1e-6)["delta"] == pytest.approx(-0.02)
    assert resolve_detuning(0.07, 0.17, da, tol=1e-6)["delta"] == pytest.approx(0.12)


def test_inconsistent_fringes_are_refused_not_guessed():
    # Neither the sum nor the difference matches Δa: no branch is consistent.
    # s = 0.30, d = 0.10: residuals 0.25 / 0.05 / 0.15 against Δa = 0.05 — all far outside tol.
    out = resolve_detuning(0.20, 0.40, 0.05, tol=0.005)
    assert out["delta"] is None and out["branch"] is None


def test_constructor_refuses_an_aliased_or_zero_detuning():
    from strumento.core.device import Device

    dev = Device.load(MULTIMODE_YAML)
    with pytest.raises(ValueError):
        RamseyDetuningPair(dev, ramsey_freq_mhz=0.0)
    with pytest.raises(ValueError):  # Nyquist (76-1)/(2*300) = 0.125 MHz
        RamseyDetuningPair(dev, stop_us=300.0, points=76, ramsey_freq_mhz=0.2)


def test_sequence_resets_before_every_block_and_drives_at_plus_then_minus():
    from strumento.core.device import Device
    from strumento.core.pulses import Delay, Measure, Pulse

    dev = Device.load(MULTIMODE_YAML)
    exp = RamseyDetuningPair(dev, stop_us=40.0, points=11, ramsey_freq_mhz=0.1)
    seq, _ = exp.sequence()
    f_ge = float(dev.calib.qubit.f_ge)
    gates = [op for op in seq.ops if isinstance(op, Pulse) and op.label == "hpi_ge"]
    assert sum(isinstance(op, Measure) for op in seq.ops) == 2 * 11
    assert [g.freq_mhz for g in gates[:4]] == pytest.approx([f_ge + 0.1] * 2 + [f_ge - 0.1] * 2)
    # a reset opens the sequence and follows every read but the last: each block starts from |g>
    reset = float(dev.calib.readout.relax_delay)
    ops = list(seq.ops)
    measures = [i for i, op in enumerate(ops) if isinstance(op, Measure)]
    assert isinstance(ops[0], Delay) and ops[0].duration_us == reset
    assert all(isinstance(ops[i + 1], Delay) and ops[i + 1].duration_us == reset for i in measures[:-1])


@pytest.mark.qick
@pytest.mark.parametrize("ramsey_detuning", [0.02, -0.02, -0.09])
def test_fit_recovers_the_signed_qubit_detuning_on_the_simulator(ramsey_detuning):
    # Truth convention (shared with T2Ramsey's response): ramsey_detuning = f_ge(belief) − f_q,
    # so the analyzer's qubit_detuning_mhz (= f_q − f_ge) must come back as −ramsey_detuning.
    pytest.importorskip("qick")
    import strumento.packs.cqed.responses  # noqa: F401  (registers the response models)
    from strumento.core.device import Device
    from strumento.core.simulator import SimulatorSoc
    from strumento.packs.cqed.readout import DispersiveIQ
    from strumento.packs.cqed.responses import truth_from_device

    dev = Device.load(MULTIMODE_YAML)
    truth = truth_from_device(dev, ramsey_detuning=ramsey_detuning, T2=150.0)
    soc = SimulatorSoc.from_snapshot(CQED_SOCCFG, truth=truth, seed=7)
    exp = RamseyDetuningPair(dev, stop_us=300.0, points=151, ramsey_freq_mhz=0.05,
                             readout=DispersiveIQ(threshold=0.5), reset_delay_us=1.0)
    outcome = exp.go(soc, reps=400, progress=False)
    d = outcome.fit.derived
    assert outcome.fit.success, d.get("consistency_residual_mhz")
    expected = -ramsey_detuning
    assert d["qubit_detuning_mhz"] == pytest.approx(expected, abs=2e-3)
    assert np.sign(d["qubit_detuning_mhz"]) == np.sign(expected)
    assert d["branch"] == ("inner" if abs(expected) < 0.05 else ("outer+" if expected > 0 else "outer-"))
    assert d["f_ge_corrected"] == pytest.approx(float(dev.calib.qubit.f_ge) + expected, abs=2e-3)
    assert d["T2"] == pytest.approx(150.0, rel=0.15)
