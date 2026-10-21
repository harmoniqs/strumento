"""**Active reset on the typed ``Branch`` seam.**

Three gates, in the order they earn their keep:

(a) the composite **assembles on a real** ``AveragerProgramV2`` against the testbench soccfg,
    offline — the branch really becomes ``read_and_jump``/``label``/``jump``;
(b) the quantitative one: a simulator run with thermal excited population p₁ = 0.15 shows
    post-reset ground population **> 0.98** in the returned per-shot data, i.e. the conditional
    flow is *demonstrable*, not merely assemblable;
(c) reset-prepared shots feed the heralding / post-selection path (tests/core/test_postselect.py).

Why (b) is reachable at all: the per-shot machine (:class:`~strumento.core.response.ShotMachine`)
walks the ``Branch`` structure shot by shot, so a shot that read ``|e⟩`` really is the shot that
gets the conditional π. Independent per-read draws from a population curve could never show it —
the correlation *between* the herald read and the verification read is the whole effect.

Error budget for (b), from the pack's truth knobs (ε = ``reset_readout_error`` = 0.002,
δ = ``reset_pi_error`` = 0.01, p₁ = 0.15):

.. code::

    true excited after one round  =  p₁·[(1-ε)·δ + ε]  +  (1-p₁)·ε·(1-δ)   ≈  0.0035
    + the verification read's own assignment error ε                        ≈  0.0055
    + blob re-scatter at σ=0.15 vs threshold 0.5, i.e. Q(3.33)              ≈  0.0059
    ⇒ measured ground                                                       ≈  0.9941   (gate 0.98)

Measured: **0.99050** at the pinned seed and 2000 shots (binomial σ = 0.0022, so the gate is 4.8σ
away), 0.9905–0.9960 across seeds 1–8, and 0.99365 ± 0.00056 at 20 000 shots — i.e. the budget
above is right to the fourth decimal. Two rounds give 0.9925–0.9975. The improvement being
measured is against 0.841 with the correction disabled, which is p₁ as it should be.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from strumento.core.pulses import Branch, Delay, Measure, Pulse, Repeat, Seq
from strumento.core.response import ResponseError, SimContext
from strumento.packs.cqed.experiments.active_reset import ActiveReset
from strumento.packs.cqed.library.composite import (
    DEFAULT_FEEDBACK_SETTLE_US,
    RESET_PI_LABEL,
    active_reset,
)
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.responses import active_reset_response, truth_from_device
from tests._paths import DEVICES

sys.path.insert(0, str(Path(__file__).parents[2]))
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

LOOPBACK = DEVICES / "loopback_demo" / "device.yaml"

#: loopback_demo has a single generator (assigned to the qubit's drive role) and no dedicated
#: readout DAC line, so dev.readout.dispersive_measure()'s default ("readout", "probe") line does
#: not resolve there -- the same gap resonator_spectroscopy's CONTRACT_KWARGS entry works around.
LOOPBACK_READOUT_LINE = ("qubit", "drive")

P_THERMAL = 0.15
SHOTS = 2000
SEED = 5
GROUND_GATE = 0.98


@pytest.fixture
def dev(loopback_dev):
    return loopback_dev


# -- the composite (pure IR) ---------------------------------------------------------------


def test_one_round_is_measure_then_branch_with_a_blocking_settle(dev):
    seq = active_reset(dev, line=LOOPBACK_READOUT_LINE)
    kinds = [type(op).__name__ for op in seq.ops]
    # #100: one_round() drives the readout probe before its Measure. F1: the settle is no longer a
    # standalone Delay (a reference-time shift that never stalled the tProc, so read_and_jump read a
    # stale accumulator) -- it rides on the Branch as a processor-blocking wait_auto before the read.
    assert kinds == ["Pulse", "Measure", "Branch"]
    assert seq.ops[0].label == "ro_probe"
    branch = seq.ops[2]
    assert branch.test == ">=" and branch.else_ is None       # "if it read excited, correct it"
    assert branch.settle_us == DEFAULT_FEEDBACK_SETTLE_US      # the freshness margin (blocking wait)
    assert [p.label for p in branch.then.ops] == [RESET_PI_LABEL]


def test_branch_threshold_and_test_reach_the_branch(dev):
    # default (back-compat): the branch uses `threshold`, polarity ">=".
    branch = next(op for op in active_reset(dev, threshold=0.5, line=LOOPBACK_READOUT_LINE).ops
                  if isinstance(op, Branch))
    assert branch.threshold == 0.5 and branch.test == ">="

    # split: `branch_threshold` (ADC-accumulator units) overrides the branch and `test` flips the
    # polarity, while `threshold` (the discriminator's units) no longer reaches the branch (trap #1/#2).
    branch = next(op for op in active_reset(
        dev, threshold=0.5, branch_threshold=-33.69, test="<", line=LOOPBACK_READOUT_LINE).ops
        if isinstance(op, Branch))
    assert branch.threshold == -33.69 and branch.test == "<"


def test_the_settle_is_a_blocking_margin_on_the_branch_not_a_delay(dev):
    ops = active_reset(dev, line=LOOPBACK_READOUT_LINE).ops
    # F1: there is no free-standing Delay before the branch anymore. A Delay lowered to qick
    # `delay()` (reference-time only) never stalled the tProc, so `read_and_jump` read a stale
    # accumulator. The settle now rides on the Branch and lowers to a blocking `wait_auto(ros=True)`.
    assert not any(isinstance(op, Delay) for op in ops)
    branch = next(op for op in ops if isinstance(op, Branch))
    assert branch.settle_us == pytest.approx(DEFAULT_FEEDBACK_SETTLE_US)
    assert branch.settle_us < 1.0     # the ADC-transfer margin, not a whole readout window


def test_rounds_unroll_by_default(dev):
    seq = active_reset(dev, rounds=3, line=LOOPBACK_READOUT_LINE)
    assert [type(op).__name__ for op in seq.ops] == ["Pulse", "Measure", "Branch"] * 3


def test_hardware_loop_wraps_the_round_in_a_repeat(dev):
    seq = active_reset(dev, rounds=3, hardware_loop=True, line=LOOPBACK_READOUT_LINE)
    assert len(seq.ops) == 1
    loop = seq.ops[0]
    assert isinstance(loop, Repeat) and loop.count == 3
    assert [type(op).__name__ for op in loop.body.ops] == ["Pulse", "Measure", "Branch"]


def test_zero_rounds_is_rejected(dev):
    with pytest.raises(ValueError, match="rounds"):
        active_reset(dev, rounds=0)


def test_the_experiment_measures_last_so_the_data_read_is_the_verification(dev):
    seq, sweeps = ActiveReset(dev, line=LOOPBACK_READOUT_LINE).sequence()
    assert sweeps == ()
    assert isinstance(seq.ops[-1], Measure)
    assert seq.ops[-2].label == "ro_probe"    # #100: the verification read is driven too
    assert isinstance(seq.ops[0], Pulse) and seq.ops[0].label == "ro_probe"
    assert isinstance(seq.ops[1], Measure) and isinstance(seq.ops[2], Branch)
    assert sum(isinstance(op, Measure) for op in seq.ops) == 2   # one reset round + verification


def test_experiment_splits_branch_threshold_from_the_discriminator(dev):
    # trap #1: the on-board branch (raw accumulator units) and the host DispersiveIQ (projected
    # units) are two different quantities; the experiment must not feed one scalar to both.
    exp = ActiveReset(dev, threshold=0.5, branch_threshold=-33.69, test="<",
                      line=LOOPBACK_READOUT_LINE)
    assert exp.readout.threshold == 0.5                          # discriminator keeps its units
    branch = next(op for op in exp.sequence()[0].ops if isinstance(op, Branch))
    assert branch.threshold == -33.69 and branch.test == "<"     # branch gets accumulator units + polarity


def test_experiment_branch_threshold_defaults_to_the_single_threshold(dev):
    # back-compat: no branch_threshold ⇒ the branch uses `threshold`, polarity ">=" (today's --sim path).
    exp = ActiveReset(dev, threshold=0.5, line=LOOPBACK_READOUT_LINE)
    assert exp.readout.threshold == 0.5
    branch = next(op for op in exp.sequence()[0].ops if isinstance(op, Branch))
    assert branch.threshold == 0.5 and branch.test == ">="


def test_readout_phase_reaches_every_reset_round_binding(dev):
    # the on-board branch can't rotate (trap #1), so the discrimination angle is folded into the
    # readout DDC phase — every reset-round probe must carry it.
    seq = active_reset(dev, rounds=2, readout_phase_deg=-126.9, line=LOOPBACK_READOUT_LINE)
    probes = [op for op in seq.ops if isinstance(op, Pulse) and op.readout is not None]
    assert len(probes) == 2 and all(p.readout.phase_deg == -126.9 for p in probes)
    # default stays 0 (normal software-angle discrimination path)
    seq0 = active_reset(dev, line=LOOPBACK_READOUT_LINE)
    assert all(op.readout.phase_deg == 0.0 for op in seq0.ops
               if isinstance(op, Pulse) and op.readout is not None)


def test_experiment_threads_readout_phase_to_reset_and_verify_reads(dev):
    # reset rounds AND the verify read share the rotated frame, so the host discriminator can read
    # the aligned axis (angle=0) consistently across all reads.
    exp = ActiveReset(dev, rounds=2, readout_phase_deg=-126.9, line=LOOPBACK_READOUT_LINE)
    seq, _ = exp.sequence()
    probes = [op for op in seq.ops if isinstance(op, Pulse) and op.readout is not None]
    assert len(probes) == 3  # 2 reset rounds + the verification read
    assert all(p.readout.phase_deg == -126.9 for p in probes)


# -- the response model, purely (the physics, before any qick is involved) ------------------


def _machine_outcomes(dev, *, rounds: int = 1, truth_overrides=None, shots: int = SHOTS):
    """Run the registered active-reset response's machine over the composite, offline and pure."""
    exp = ActiveReset(dev, rounds=rounds, line=LOOPBACK_READOUT_LINE)
    seq, _ = exp.sequence()
    truth = truth_from_device(dev, p_thermal=P_THERMAL, **(truth_overrides or {}))
    ctx = SimContext(experiment=exp, device=dev, truth=truth, seq=seq)
    resp = active_reset_response(ctx)
    return resp.machine.outcomes(seq, shots, np.random.default_rng(SEED))


def test_the_response_model_alone_resets_the_thermal_population(dev):
    outcomes = _machine_outcomes(dev)
    assert outcomes.shape == (2, SHOTS)
    reset_read, verify_read = outcomes
    assert abs(reset_read.mean() - P_THERMAL) < 5 * np.sqrt(P_THERMAL * 0.85 / SHOTS)
    assert 1.0 - verify_read.mean() > GROUND_GATE


def test_a_perfect_reset_leaves_nothing_excited(dev):
    outcomes = _machine_outcomes(dev, truth_overrides={"reset_readout_error": 0.0,
                                                       "reset_pi_error": 0.0})
    assert outcomes[-1].sum() == 0


def test_more_rounds_do_not_undo_the_reset(dev):
    outcomes = _machine_outcomes(dev, rounds=2)
    assert outcomes.shape == (3, SHOTS)          # 2 reset reads + the verification read
    assert 1.0 - outcomes[-1].mean() > GROUND_GATE


def test_the_machine_needs_a_measurement_before_the_branch(dev):
    branch_first = Seq().branch(adc=0, threshold=0.5, test=">=", then=Seq())
    exp = ActiveReset(dev)
    ctx = SimContext(experiment=exp, device=dev, truth=truth_from_device(dev), seq=branch_first)
    with pytest.raises(ResponseError, match="Branch"):
        active_reset_response(ctx).machine.outcomes(branch_first, 4, np.random.default_rng(0))


# -- (a) THE ASSEMBLY GATE: real tProc-v2 ---------------------------------------------------


@pytest.mark.qick
@requires_v2_soccfg
@pytest.mark.parametrize("kw", [{}, {"rounds": 2}, {"rounds": 2, "hardware_loop": True}])
def test_the_composite_assembles_on_a_real_v2_program(dev, kw):
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram
    from strumento.core.result import MeasurementResult

    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    seq = active_reset(dev, line=LOOPBACK_READOUT_LINE, **kw).wait(0.5).measure()
    result = StrumentoProgram(dev, seq=seq, soc=soc, reps=4).acquire(soc, progress=False)
    assert isinstance(result, MeasurementResult)


@pytest.mark.qick
@requires_v2_soccfg
@pytest.mark.parametrize("phase_deg", [0.0, 40.0, -126.9])
def test_readout_ddc_phase_reaches_the_assembled_program(dev, phase_deg):
    """The inserted DDC phase survives **real v2 assembly** — verified before any board run.

    The branch reads a raw quadrature and can't rotate, so the discrimination angle is folded into
    the readout DDC phase (``ReadoutBinding.phase_deg`` -> ``add_readoutconfig(phase=...)``). This
    reads that phase back off the compiled program (register-rounded to the readout's phase word),
    confirming the assembler actually programs the demodulator with it rather than dropping it.
    """
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram

    seq = active_reset(dev, readout_phase_deg=phase_deg, line=LOOPBACK_READOUT_LINE)
    prog = StrumentoProgram(dev, seq=seq, soc=MockQickSocV2.from_snapshot(V2_SOCCFG), reps=4, trace=True)
    prog.compile()
    name = next(e["name"] for e in prog.trace()["entries"] if e["op"] == "readout_config")
    got = float(prog.get_pulse_param(name, "phase")) % 360.0
    assert got == pytest.approx(phase_deg % 360.0, abs=0.5)


@pytest.mark.qick
@requires_v2_soccfg
def test_the_experiment_runs_on_the_mock_without_shot_data(dev):
    """The mock's zero-IQ contract is untouched: no shots ⇒ an un-successful outcome that says
    why (#199) — not the bare empty fit it once was, which read like a failed reset."""
    from strumento.core.experiment import SHOTS_REFUSAL
    from strumento.core.mock import MockQickSocV2

    outcome = ActiveReset(dev, line=LOOPBACK_READOUT_LINE).go(
        MockQickSocV2.from_snapshot(V2_SOCCFG), reps=8, progress=False
    )
    assert outcome.result.shots is None
    assert outcome.fit.success is False
    assert outcome.fit.derived == {"refused": SHOTS_REFUSAL}


# -- (b) THE QUANTITATIVE GATE: reset works, through the real path --------------------------


def _run(dev, *, rounds: int = 1, reps: int = SHOTS, seed: int = SEED, truth_overrides=None):
    from strumento.core.simulator import SimulatorSoc

    truth = truth_from_device(dev, p_thermal=P_THERMAL, **(truth_overrides or {}))
    soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth=truth, seed=seed)
    return ActiveReset(dev, rounds=rounds, line=LOOPBACK_READOUT_LINE).go(soc, reps=reps, progress=False)


@pytest.mark.qick
@requires_v2_soccfg
def test_post_reset_ground_population_beats_the_gate(dev):
    outcome = _run(dev)
    ground = outcome.fit.derived["ground_population"]
    assert outcome.fit.success
    assert ground > GROUND_GATE, f"post-reset ground population {ground:.4f}"
    # ... and it is genuinely measured from the per-shot data, not from the response's parameters
    shots = outcome.result.shots
    assert shots.shape == (2, SHOTS, 2)
    from_shots = 1.0 - DispersiveIQ(threshold=0.5).discriminate(shots)[-1].mean()
    assert from_shots == pytest.approx(ground)


@pytest.mark.qick
@requires_v2_soccfg
def test_without_the_conditional_the_thermal_population_survives(dev):
    """The control: the same run with the conditional π disabled sits at p₁, not at the gate.

    ``reset_pi_error = 1`` means "the corrective pulse never lands" — the branch still fires, so
    this isolates the *effect of the correction* from everything else in the chain.
    """
    outcome = _run(dev, truth_overrides={"reset_pi_error": 1.0})
    ground = outcome.fit.derived["ground_population"]
    assert 1.0 - ground == pytest.approx(P_THERMAL, abs=0.03)
    assert ground < GROUND_GATE


@pytest.mark.qick
@requires_v2_soccfg
def test_the_reset_is_stable_across_seeds(dev):
    for seed in range(1, 6):
        ground = _run(dev, seed=seed).fit.derived["ground_population"]
        assert ground > GROUND_GATE, f"seed {seed}: {ground:.4f}"


@pytest.mark.qick
@requires_v2_soccfg
def test_a_hardware_loop_measure_is_a_typed_error_not_a_wrong_number(dev):
    """A ``Measure`` inside a ``Repeat`` reads N times but is *counted once* at assembly.

    qick counts trigger **macros**, so the acquisition declares one read per shot for a hardware
    loop while the tProc performs ``count`` of them. The simulator refuses to invent a mapping and
    says so — the offline surfacing of a real board hazard.
    """
    from strumento.core.simulator import SimulatorSoc

    truth = truth_from_device(dev, p_thermal=P_THERMAL)
    soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth=truth, seed=SEED)
    exp = ActiveReset(dev, rounds=3, hardware_loop=True, line=LOOPBACK_READOUT_LINE)
    with pytest.raises(ResponseError, match="reads per shot"):
        exp.go(soc, reps=16, progress=False)


# -- (c) heralding / post-selection ---------------------------------------------------------


@pytest.mark.qick
@requires_v2_soccfg
def test_reset_shots_feed_the_herald_and_postselect_path(dev):
    outcome = _run(dev)
    result = outcome.result

    assert result.herald_lanes == ("reset",)
    mask = result.keep_mask()
    assert mask.shape == (SHOTS,) and mask.dtype == bool
    # the lane passes the shots whose reset round read |g> — ≈1 - p₁ of them
    assert abs(mask.mean() - (1.0 - P_THERMAL)) < 0.03

    kept = result.postselect()
    assert kept.meta["postselect_survival"] == pytest.approx(float(mask.mean()))
    assert kept.heralds["reset"].size == int(mask.sum())    # lanes compacted together
    # the IQ-shaped shots lane is left intact by design (its trailing axis is (I, Q), not shots):
    # post-selection compacts the per-shot *lanes* and hands the mask to whoever reduces the IQ
    assert kept.shots.shape == result.shots.shape

    # post-selecting on the reset round tightens the state further (those shots needed no
    # correction at all), which is exactly what a herald lane is for
    verify = DispersiveIQ(threshold=0.5).discriminate(result.shots)[-1]
    assert 1.0 - verify[mask].mean() >= 1.0 - verify.mean()


@pytest.mark.qick
@requires_v2_soccfg
def test_every_round_gets_its_own_herald_lane(dev):
    result = _run(dev, rounds=3).result
    assert set(result.heralds) == {"reset_1", "reset_2", "reset"}   # earlier rounds + the last
    assert result.herald_lanes == ("reset",)                        # only the last is default
    assert all(mask.shape == (SHOTS,) for mask in result.heralds.values())


@pytest.mark.qick
@requires_v2_soccfg
def test_the_result_is_stamped_and_wire_safe(dev):
    """The D34 stamp and the D35 wire form both hold for a conditional-flow result."""
    import json

    result = _run(dev, reps=64).result
    assert result.meta["environment"]["soc"]["simulated"] is True
    json.dumps(result.to_wire())
