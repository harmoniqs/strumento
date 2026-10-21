"""``Experiment.go(record=…)`` / ``Runner.run(record=…)`` — emission counts and the gate emitter.

Two claims are under test here. The easy one is that a standard experiment emits the right record
events. The load-bearing one is that **gate verdicts are computed, not asserted**: the same fit
against the same truth produces ``pass: false`` when the tolerance is tight and ``pass: true`` when
it is not, with nothing a caller says able to change the answer.

The orchestration tests use a fake Runner (no qick, no board); the end-to-end record test drives the
real path on the simulator and is marked accordingly.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import types
from pathlib import Path

import pytest

from strumento.core.experiment import (
    DEFAULT_RECOVERY_TOLERANCE,
    Experiment,
    Runner,
)
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.record import TaskRecord, read_events
from strumento.core.result import MeasurementResult
from tests._paths import DEVICES


class _FakeRunner:
    """Mirrors the real Runner's signature — including the ``**acquire_kw`` tail."""

    def __init__(self, result):
        self._result = result
        self.seen: dict = {}

    def run(self, device, seq, sweeps, soc, *, experiment=None, reps=100, record=None, **acquire_kw):
        self.seen = {"record": record, "acquire_kw": acquire_kw}
        if record is not None:
            from strumento.core.experiment import _emit_acquired

            _emit_acquired(record, self._result, experiment=experiment, reps=reps)
        return self._result


class _T1ish(Experiment):
    """A stand-in experiment whose fit reports one derived quantity the truth also names."""

    derived = {"T1": 68.0}
    success = True

    def sequence(self):
        return (Seq(), ())

    def analyze(self, result):
        return FitResult(params={"tau": 68.0}, stderr={}, derived=dict(self.derived),
                         success=self.success)


class _StubSoc:
    """Not armable, not a shot source — just a soc that may carry a ground truth."""

    def __init__(self, truth=None):
        if truth is not None:
            self.truth = truth


def _canned(points=5):
    return MeasurementResult(xpts=list(range(points)), avgi=[0.0] * points, avgq=[0.0] * points)


def _exp(record_result=None, **kw):
    runner = _FakeRunner(record_result if record_result is not None else _canned())
    return _T1ish(device=None, runner=runner, **kw), runner


def _kinds(rec_dir):
    return [e["ev"] for e in read_events(rec_dir)]


def _gates(rec_dir):
    return {e["name"]: e for e in read_events(rec_dir) if e["ev"] == "gate"}


# -- `record` is an explicit keyword, not something a **kwargs tail swallows -------------------


@pytest.mark.parametrize("fn", [Runner.run, Experiment.go])
def test_record_is_an_explicit_keyword_only_parameter(fn):
    param = inspect.signature(fn).parameters.get("record")
    assert param is not None, f"{fn.__qualname__} must take an explicit record= keyword"
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None


def test_a_misspelled_record_keyword_raises_instead_of_being_swallowed(
    monkeypatch, tmp_path, closing
):
    """The hazard this guards: ``recrod=rec`` silently meaning "no record", so a task looks dead.

    ``record`` being explicit means a typo lands in ``**acquire_kw`` and travels on to ``acquire``,
    which rejects it — stubbed here (with a signature that rejects unknown kwargs exactly as the real
    ``AveragerProgramV2.acquire`` does) so the check runs in the pure lane.
    """
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    acquired: list[dict] = []

    class _StubProgram:
        def __init__(self, device, **kw):
            pass

        def xpts(self):
            return ()

        def acquire(self, soc, progress=True):     # the real acquire's kwargs, and only those
            acquired.append({"progress": progress})
            return _canned()

    module = types.ModuleType("strumento.core.program")
    module.StrumentoProgram = _StubProgram
    monkeypatch.setitem(sys.modules, "strumento.core.program", module)

    runner = Runner()
    # the correct spelling works and emits
    runner.run(None, Seq(), (), _StubSoc(), record=rec, progress=False)
    assert _kinds(rec.dir) == ["progress"]

    # the typo does not silently vanish
    with pytest.raises(TypeError, match="recrod"):
        runner.run(None, Seq(), (), _StubSoc(), recrod=rec, progress=False)
    assert _kinds(rec.dir) == ["progress"]         # and nothing new was recorded


# -- emission counts ---------------------------------------------------------------------------


def test_no_record_means_no_emission_and_an_unchanged_outcome(tmp_path):
    exp, _ = _exp()
    out = exp.go(soc=_StubSoc())
    assert out.fit.derived == {"T1": 68.0}
    assert list(tmp_path.iterdir()) == []          # nothing written anywhere


def test_a_run_emits_one_progress_per_stage_and_one_fit_gate(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    exp, runner = _exp(_canned(points=7))
    exp.go(soc=_StubSoc(), record=rec)

    assert runner.seen["record"] is rec            # forwarded through go -> Runner.run
    assert _kinds(rec.dir) == ["progress", "progress", "gate"]
    events = read_events(rec.dir)
    assert events[0]["metrics"] == {"points": 7, "reps": 100}
    assert "acquired 7 point(s)" in events[0]["label"]
    assert events[1]["metrics"] == {"T1": 68.0}    # the derived quantities are the metric series
    assert events[1]["step"] == 2 and events[1]["label"].endswith("fit")


def test_analysis_off_emits_the_acquisition_only(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    exp, _ = _exp()
    exp.go(soc=_StubSoc(), record=rec, analyze=False)
    assert _kinds(rec.dir) == ["progress"]         # no fit, so no fit progress and no verdict


def test_repeated_runs_accumulate_one_series(tmp_path, closing):
    """A monitor loop calling go() N times: the record holds N fit events to plot."""
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    exp, _ = _exp()
    for value in (68.0, 66.5, 70.1):
        exp.derived = {"T1": value}
        exp.go(soc=_StubSoc(), record=rec)
    series = [e["metrics"]["T1"] for e in read_events(rec.dir)
              if e["ev"] == "progress" and "T1" in e["metrics"]]
    assert series == [68.0, 66.5, 70.1]


def test_a_saved_figure_becomes_an_artifact(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))

    class _Figure:
        def savefig(self, path):
            Path(path).write_bytes(b"\x89PNG fake")

    class _Plotting(_T1ish):
        def display(self, result, fit):
            return _Figure()

    exp = _Plotting(device=None, runner=_FakeRunner(_canned()))
    exp.go(soc=_StubSoc(), record=rec, display=True)
    exp.go(soc=_StubSoc(), record=rec, display=True)

    artifacts = [e for e in read_events(rec.dir) if e["ev"] == "artifact"]
    assert [a["path"] for a in artifacts] == ["artifacts/_plotting-000.png",
                                              "artifacts/_plotting-001.png"]
    assert all(a["kind"] == "figure" for a in artifacts)
    assert (rec.dir / artifacts[0]["path"]).read_bytes().startswith(b"\x89PNG")


def test_a_figure_that_cannot_save_itself_records_no_artifact(tmp_path, closing):
    """An `artifact` path must resolve to a real file; inventing one would break the contract."""
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))

    class _Opaque(_T1ish):
        def display(self, result, fit):
            return {"figure": "a dict is not a figure"}

    _Opaque(device=None, runner=_FakeRunner(_canned())).go(
        soc=_StubSoc(), record=rec, display=True
    )
    assert [e["ev"] for e in read_events(rec.dir) if e["ev"] == "artifact"] == []


# -- the fit gate ------------------------------------------------------------------------------


def test_fit_gate_records_convergence(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    exp, _ = _exp()
    exp.go(soc=_StubSoc(), record=rec)
    gate = _gates(rec.dir)["fit"]
    assert gate["pass"] is True
    assert gate["params"]["experiment"] == "_T1ish" and gate["params"]["derived"] == ["T1"]


def test_fit_gate_records_a_failed_fit_as_a_failed_gate(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    exp, _ = _exp()
    exp.success = False
    exp.go(soc=_StubSoc(), record=rec)
    assert _gates(rec.dir)["fit"]["pass"] is False


# -- the fit-recovery gate (the vetted verdict) --------------------------------------------------


def test_fit_recovery_gate_needs_an_armed_truth(tmp_path, closing):
    """On hardware there is no truth to recover, so there is no verdict to give."""
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    exp, _ = _exp()
    exp.go(soc=_StubSoc(), record=rec)                    # a board-like soc: no truth attribute
    assert set(_gates(rec.dir)) == {"fit"}


def test_fit_recovery_passes_within_tolerance(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    exp, _ = _exp()
    exp.derived = {"T1": 69.0}
    exp.go(soc=_StubSoc(truth={"T1": 68.0}), record=rec)  # 1.5 % against a 5 % default
    gate = _gates(rec.dir)["fit-recovery"]
    assert gate["pass"] is True
    assert gate["params"]["metric"] == "T1_rel_err"
    assert gate["params"]["value"] == pytest.approx(1.0 / 68.0)
    assert gate["params"]["tolerance"] == DEFAULT_RECOVERY_TOLERANCE
    assert gate["params"]["compared"] == {"T1": pytest.approx(1.0 / 68.0)}


def test_fit_recovery_fails_outside_tolerance(tmp_path, closing):
    """The verdict is computed from the numbers — a caller cannot talk it into passing."""
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    exp, _ = _exp()
    exp.derived = {"T1": 80.0}
    exp.go(soc=_StubSoc(truth={"T1": 68.0}), record=rec)  # 17.6 %
    gate = _gates(rec.dir)["fit-recovery"]
    assert gate["pass"] is False and gate["params"]["value"] > 0.05


def test_the_same_fit_flips_the_verdict_when_the_tolerance_moves(tmp_path, closing):
    """Nothing but the numbers decides: same fit, same truth, tolerance is the only variable."""
    exp, _ = _exp()
    exp.derived = {"T1": 69.0}
    verdicts = {}
    for tol in (0.05, 1e-6):
        rec = closing(TaskRecord.create(tmp_path / f"tol-{tol}", heartbeat=False))
        exp.go(soc=_StubSoc(truth={"T1": 68.0}), record=rec, recovery_tolerance=tol)
        verdicts[tol] = _gates(rec.dir)["fit-recovery"]["pass"]
    assert verdicts == {0.05: True, 1e-6: False}


def test_the_worst_quantity_sets_the_verdict_and_all_of_them_are_recorded(tmp_path, closing):
    """One event per gate name (a reader takes the last), so the AND and the detail both ride along."""
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    exp, _ = _exp()
    exp.derived = {"T1": 68.5, "T2": 30.0, "offset": 0.5}   # T2 is 50 % off; offset has no truth
    exp.go(soc=_StubSoc(truth={"T1": 68.0, "T2": 20.0}), record=rec)
    gates = [e for e in read_events(rec.dir) if e["ev"] == "gate" and e["name"] == "fit-recovery"]
    assert len(gates) == 1
    params = gates[0]["params"]
    assert gates[0]["pass"] is False
    assert params["metric"] == "T2_rel_err" and params["value"] == pytest.approx(0.5)
    assert set(params["compared"]) == {"T1", "T2"}          # `offset` had nothing to compare against


def test_recovery_map_pairs_differently_named_quantities(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))

    class _Parityish(_T1ish):
        derived = {"phase_deg": 62.5}

        def recovery_map(self):
            return {"phase_deg": "parity_phase"}

    exp = _Parityish(device=None, runner=_FakeRunner(_canned()))
    exp.go(soc=_StubSoc(truth={"parity_phase": 62.0}), record=rec)
    gate = _gates(rec.dir)["fit-recovery"]
    assert gate["pass"] is True and set(gate["params"]["compared"]) == {"phase_deg"}


def test_a_truth_sharing_no_quantity_emits_no_recovery_verdict(tmp_path, closing):
    """Silence beats a vacuous pass: an empty comparison must not read as "recovered"."""
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    exp, _ = _exp()
    exp.go(soc=_StubSoc(truth={"something_else": 1.0}), record=rec)
    assert set(_gates(rec.dir)) == {"fit"}


def test_a_zero_truth_value_is_skipped_not_fudged(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    exp, _ = _exp()
    exp.derived = {"T1": 68.0, "detuning": 0.01}
    exp.go(soc=_StubSoc(truth={"T1": 68.0, "detuning": 0.0}), record=rec)
    assert set(_gates(rec.dir)["fit-recovery"]["params"]["compared"]) == {"T1"}


def test_an_armed_context_truth_wins_over_the_socs_default(tmp_path, closing):
    """The truth that produced the data is the armed one, not whatever the soc was constructed with."""
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))

    class _ArmedSoc(_StubSoc):
        context = types.SimpleNamespace(truth={"T1": 68.0})

    exp, _ = _exp()
    exp.derived = {"T1": 68.2}
    exp.go(soc=_ArmedSoc(truth={"T1": 40.0}), record=rec)
    gate = _gates(rec.dir)["fit-recovery"]
    assert gate["pass"] is True and gate["params"]["value"] < 0.01


# -- the whole thing, once, on the real path ------------------------------------------------------


@pytest.mark.qick
def test_a_simulated_run_produces_a_complete_record(tmp_path):
    """One real run — pulse IR → compiler → ``AveragerProgramV2`` → simulated acquire → fit → record.

    The verdicts here are the interesting part: the truth is perturbed away from what the instance
    believes, so a ``fit-recovery`` pass means the fit found the *simulator's* π gain rather than the
    device's belief — and no line of this test computes that verdict.
    """
    pytest.importorskip("qick")
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from _v2gate import V2_SOCCFG, requires_v2_soccfg

    if requires_v2_soccfg.args[0]:  # the skipif condition
        pytest.skip(requires_v2_soccfg.kwargs["reason"])

    from strumento.core.device import Device
    from strumento.core.record import derive_state
    from strumento.core.simulator import SimulatorSoc
    from strumento.core.sweeps import Sweep
    from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi
    from strumento.packs.cqed.readout import DispersiveIQ
    from strumento.packs.cqed.responses import truth_from_device

    device = Device.load(DEVICES / "loopback_demo" / "device.yaml")
    truth = truth_from_device(device, pi_gain=9350.0)      # the instance believes 8192
    soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth=truth, seed=7)

    rec = TaskRecord.create(tmp_path / "2026-07-25-rabi-ab12", title="amplitude Rabi",
                            device="loopback_demo", gates=["fit", "fit-recovery"], heartbeat=False)
    exp = AmplitudeRabi(device, gains=Sweep(start=0.0, stop=16000.0, on="amp"), points=61,
                        readout=DispersiveIQ(threshold=0.5),
                        # loopback_demo has no dedicated readout line — probe on the qubit
                        # line instead, same as resonator_spectroscopy's own test override
                        # (issue #102).
                        line=("qubit", "drive"))
    outcome = exp.go(soc, record=rec, reps=500, progress=False, recovery_tolerance=0.02)
    rec.done(summary=f"pi_gain = {outcome.fit.derived['pi_gain']:.0f}")

    assert _kinds(rec.dir) == ["progress", "progress", "gate", "gate"]
    gates = _gates(rec.dir)
    assert gates["fit"]["pass"] is True
    assert gates["fit-recovery"]["pass"] is True
    assert gates["fit-recovery"]["params"]["metric"] == "pi_gain_rel_err"
    assert gates["fit-recovery"]["params"]["value"] < 0.02

    fit_event = [e for e in read_events(rec.dir) if e.get("metrics", {}).get("pi_gain")][0]
    assert fit_event["metrics"]["pi_gain"] == pytest.approx(outcome.fit.derived["pi_gain"])

    status = derive_state(rec.dir)
    assert status.state == "done" and status.result.summary.startswith("pi_gain =")
    assert status.manifest.gates == ("fit", "fit-recovery")


# -- purity --------------------------------------------------------------------------------------


def test_module_imports_without_qick():
    code = "import sys; sys.modules['qick']=None; import strumento.core.experiment; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
