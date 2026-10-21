"""TransmonRB through the real compiler and the simulator (issue #211) — qick lane.

**Program memory.** The registered experiment puts every depth into one program, and qick's
tProc-v2 program memory (``tprocs[0].pmem_size`` in the soccfg snapshot) is finite. qick itself
raises only at compile time (``RuntimeError: compiled program uses N words of pmem, but the size
of that tProc memory is only M``). TransmonRB refuses before that with a pointed
:class:`~strumento.packs.cqed.experiments.transmon_rb.ProgramMemoryError`. Its estimate must
never refuse a request qick would compile, so it is a lower bound — checked here against the
real compiled size across a sweep of depths, points, seeds and envelopes — and the narrow band
it lets through is caught at compile and re-raised as the same pointed error.

**Simulated decay.** With the simulator's RB truth ``rb_decay`` = τ (survival
:math:`\\propto e^{-m/\\tau}`, so the injected depolarizing parameter is
:math:`p = e^{-1/\\tau}`), the analyzer recovers :math:`p`. The error budget follows
``tests/e2e/test_fit_recovery.py``: the RB fit is ``fit_exp`` on a decay constant, the same
estimator as the T1 gate there, so the gate is the T1 gate's 5 % — here on the error per Clifford
:math:`r = 1 - p \\approx 1/\\tau`, the quantity RB reports (a 5 % gate on :math:`p` itself would
be blind: :math:`p` is within 5 % of 1 for any usable gate). Conditioning: τ = 20 Cliffords,
depths 1…75 = 3.7 e-folds (the T1 row's coverage), 24 depths × 5000 reps. Measured over 30
simulator seeds: scatter of :math:`r` 1.0 %, max 2.4 %, mean bias −0.2 % — the 5 % gate sits 5σ
out. The 24 depths are what the 4096-word program memory of the shipped snapshot allows at that
span (≈3.4k words).
"""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

pytest.importorskip("qick")

from qick.qick_asm import QickConfig  # noqa: E402

from strumento.core.device import Device  # noqa: E402
from strumento.core.mock import MockQickSocV2  # noqa: E402
from strumento.core.program import StrumentoProgram  # noqa: E402
from strumento.core.simulator import SimulatorSoc  # noqa: E402
from strumento.packs.cqed.experiments import transmon_rb as rb  # noqa: E402
from strumento.packs.cqed.experiments.transmon_rb import (  # noqa: E402
    ProgramMemoryError,
    TransmonRB,
)
from strumento.packs.cqed.readout import DispersiveIQ  # noqa: E402
from strumento.packs.cqed.responses import truth_from_device  # noqa: E402
from tests._paths import FIXTURES, MULTIMODE_YAML  # noqa: E402

pytestmark = pytest.mark.qick

CQED_SOCCFG = FIXTURES / "soccfg_v2_cqed.json"


@pytest.fixture
def dev() -> Device:
    return Device.load(MULTIMODE_YAML)


def _compiled_words(dev, exp, pmem_size: int | None = None) -> int:
    """The real compiled pmem size of ``exp``'s program (on an enlarged pmem if asked)."""
    snapshot = copy.deepcopy(dev.soccfg_snapshot)
    if pmem_size is not None:
        for tproc in snapshot["tprocs"]:
            tproc["pmem_size"] = pmem_size
    seq, sweeps = exp._sequence_unchecked()
    prog = StrumentoProgram(dev, seq=seq, sweeps=sweeps, soc=QickConfig(snapshot), reps=100)
    return len(prog.binprog["pmem"])


# -- the estimate never refuses what would compile ---------------------------------------

SWEEP = [
    # (depth_start, depth_stop, points)
    (1, 1, 1), (1, 10, 1), (50, 50, 1), (1, 50, 6), (1, 100, 10), (1, 200, 5),
    (1, 300, 8), (1, 500, 4), (1, 100, 20), (10, 400, 12),
]


@pytest.mark.parametrize("drag_alpha", [None, -0.05], ids=["gaussian", "drag"])
def test_the_estimate_is_a_lower_bound_on_the_compiled_size(dev, drag_alpha):
    dev.calib.qubit.drag_alpha = drag_alpha
    gaps = []
    for (start, stop, points), seed in itertools.product(SWEEP, (42, 7)):
        exp = TransmonRB(dev, depth_start=start, depth_stop=stop, points=points, seed=seed)
        estimate = rb.estimate_pmem_words(exp._sequence_unchecked()[0])
        actual = _compiled_words(dev, exp, pmem_size=1 << 20)
        assert estimate <= actual, f"{(start, stop, points, seed)}: {estimate} > {actual}"
        gaps.append(actual - estimate)
    # the gap is the program's fixed prologue/epilogue — the same few words whatever the depth
    assert max(gaps) - min(gaps) <= 2 and max(gaps) <= 32, gaps


def test_the_shipped_snapshot_estimate_matches_its_own_compiled_size(dev):
    exp = TransmonRB(dev, depth_start=1, depth_stop=50, points=6, seed=42)
    estimate = rb.estimate_pmem_words(exp._sequence_unchecked()[0])
    actual = _compiled_words(dev, exp)  # the shipped 4096-word snapshot itself
    assert 0 <= actual - estimate <= 32


# -- refused before compile; the narrow band caught at compile ---------------------------


def test_an_over_size_request_is_refused_before_any_program_is_built(dev, monkeypatch):
    def _no_build(*a, **k):
        raise AssertionError("a program was built for a refused request")

    monkeypatch.setattr(StrumentoProgram, "__init__", _no_build)
    exp = TransmonRB(dev, depth_start=1, depth_stop=300, points=8, seed=42)
    with pytest.raises(ProgramMemoryError, match="4,096") as info:
        exp.go(MockQickSocV2.from_snapshot(CQED_SOCCFG), analyze=False, progress=False)
    assert not info.value.compiled


def test_a_request_the_estimate_lets_through_is_caught_at_compile(dev, monkeypatch):
    """The estimate is a lower bound, so the last few words over the limit reach qick; qick's
    overflow comes back as the same pointed error, naming the compiled size."""
    monkeypatch.setattr(rb, "estimate_pmem_words", lambda seq: 0)
    exp = TransmonRB(dev, depth_start=1, depth_stop=300, points=8, seed=42)
    with pytest.raises(ProgramMemoryError) as info:
        exp.go(MockQickSocV2.from_snapshot(CQED_SOCCFG), analyze=False, progress=False)
    err = info.value
    assert err.compiled and err.limit_words == 4096 and err.estimated_words > 4096
    assert "compiled" in str(err) and "one program per depth" in str(err)


def test_a_request_that_fits_runs_as_before(dev):
    exp = TransmonRB(dev, depth_start=1, depth_stop=50, points=6, seed=42,
                     readout=DispersiveIQ(threshold=0.5))
    outcome = exp.go(MockQickSocV2.from_snapshot(CQED_SOCCFG), analyze=False, progress=False)
    assert np.asarray(outcome.result.avgi).size == 6


# -- simulated decay: the fit recovers the injected depolarizing parameter ---------------

RB_TAU = 20.0          # Cliffords — the armed truth (the simulator default is 50)
RB_DEPTH_STOP = 75     # 3.7 e-folds
RB_POINTS = 24
RB_REPS = 5000
RB_SIM_SEED = 7
RB_GATE = 0.05         # relative, on the error per Clifford r = 1 - p (module docstring)


def test_the_rb_fit_recovers_the_injected_depolarizing_parameter(dev):
    truth = truth_from_device(dev, rb_decay=RB_TAU)
    p_true = float(np.exp(-1.0 / RB_TAU))
    soc = SimulatorSoc.from_snapshot(CQED_SOCCFG, truth=truth, seed=RB_SIM_SEED)
    exp = TransmonRB(dev, depth_start=1, depth_stop=RB_DEPTH_STOP, points=RB_POINTS, seed=42,
                     readout=DispersiveIQ(threshold=0.5))
    outcome = exp.go(soc, reps=RB_REPS, progress=False)
    fit = outcome.fit
    assert fit is not None and fit.success
    p = fit.derived["p"]
    r_true, r_fit = 1.0 - p_true, 1.0 - p
    assert abs(r_fit / r_true - 1.0) < RB_GATE, f"p {p:.5f} vs injected {p_true:.5f}"
    assert fit.derived["error_per_clifford"] == pytest.approx(r_fit / 2.0)
    assert fit.derived["fidelity"] == pytest.approx(1.0 - r_fit / 2.0)
    # not the simulator's default truth (rb_decay = 50): the fit followed the armed truth
    r_default = 1.0 - float(np.exp(-1.0 / 50.0))
    assert abs(r_fit / r_default - 1.0) > 0.5
