"""Tests for the Experiment/Runner/Analyzer/Display scaffold (plan Task 17, D13).

The orchestration is driven with a fake Runner, so it needs no qick/board. (The real
construct+acquire path is exercised by the v2-gated slice e2e.)
"""

from strumento.core.experiment import Experiment, ExperimentOutcome, Runner
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult


class _FakeRunner:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def run(self, device, seq, sweeps, soc, **kw):
        self.calls += 1
        return self._result


class _TrivialExp(Experiment):
    def sequence(self):
        return (Seq(), ())

    def analyze(self, result):
        return FitResult(params={}, stderr={}, derived={"answer": 42.0}, success=True)

    def display(self, result, fit):
        return {"figure": True}


def _canned():
    return MeasurementResult(xpts=None, avgi=[0.0], avgq=[0.0])


def test_go_acquires_then_analyzes():
    r = _canned()
    runner = _FakeRunner(r)
    out = _TrivialExp(device=None, runner=runner).go(soc=None)
    assert isinstance(out, ExperimentOutcome)
    assert runner.calls == 1
    assert out.result is r
    assert isinstance(out.fit, FitResult) and out.fit.derived["answer"] == 42.0
    assert out.figure is None  # display off by default


def test_go_can_skip_analysis():
    out = _TrivialExp(device=None, runner=_FakeRunner(_canned())).go(soc=None, analyze=False)
    assert out.fit is None


def test_go_display_when_requested():
    out = _TrivialExp(device=None, runner=_FakeRunner(_canned())).go(soc=None, display=True)
    assert out.figure == {"figure": True}


def test_module_imports_without_qick():
    import subprocess
    import sys

    code = "import sys; sys.modules['qick']=None; import strumento.core.experiment; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
