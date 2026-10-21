"""Tests for the legacy Pyro4 runner (#97) — the compile/acquire split for ``make_proxy()``.

Pure: ``StrumentoProgram`` is swapped out via ``sys.modules`` (the same technique
``test_experiment_record.py`` uses for ``Runner.run``), so these tests drive the *real*
``LegacyPyroRunner.run``/``.go()`` path with no qick and no Pyro4 installed — matching the
module's own constraint that it must stay in the ``pure`` CI lane.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from strumento.core.environment import ENVIRONMENT_KEY
from strumento.core.experiment import Experiment, Runner
from strumento.core.legacy_pyro_runner import LegacyPyroRunner
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult


class _RaisesIfAcquired:
    """Plays the constructor-supplied ``soccfg``: compiling against it (a subscript read, the
    way a real config is read) is fine; acquiring from it is a routing bug and raises loudly."""

    def __getitem__(self, key):
        return {}

    def acquire(self, *a, **kw):
        raise AssertionError("acquired from the compile-time soccfg — should never happen")


class _RaisesIfCompiled:
    """Plays the run-time ``soc`` (stand-in for a Pyro4 proxy): acquiring from it is the whole
    point and returns a marker; compiling against it (a subscript read) is a routing bug and
    raises loudly."""

    def __getitem__(self, key):
        raise AssertionError("compiled against the run-time soc — should never happen")

    def acquire(self, *a, **kw):
        return "ACQUIRED"


class _StubProgram:
    """Stands in for ``StrumentoProgram``: touches its constructor-supplied ``soc`` (the
    compile-time config) once, the way a real compile step reads a ``QickConfig``, and its
    acquire-time ``soc`` once, the way a real acquire pulls data from the board — enough to
    prove which object played which role, with no qick installed."""

    def __init__(self, device, seq=None, *, sweeps=(), axes=(), reps=100, final_delay=None,
                 soc=None, **kw):
        soc["probe"]  # a compile-time config read

    def xpts(self):
        return ()

    def acquire(self, soc, **kw):
        marker = soc.acquire()
        return MeasurementResult(xpts=self.xpts(), avgi=[0.0], avgq=[0.0],
                                  meta={"acquired_via": marker})


@pytest.fixture
def stub_program(monkeypatch):
    """Swap ``strumento.core.program`` for a fake module carrying ``_StubProgram`` — the same
    ``sys.modules`` trick ``test_experiment_record.py`` uses, so the local
    ``from strumento.core.program import StrumentoProgram`` inside ``run()`` resolves to the
    stub instead of needing a real qick install."""
    module = types.ModuleType("strumento.core.program")
    module.StrumentoProgram = _StubProgram
    monkeypatch.setitem(sys.modules, "strumento.core.program", module)


def test_compiles_against_the_constructor_soccfg_and_acquires_against_the_run_time_soc(stub_program):
    compile_cfg = _RaisesIfAcquired()
    runner = LegacyPyroRunner(compile_cfg)
    assert isinstance(runner, Runner)
    assert runner.soccfg is compile_cfg

    result = runner.run(None, Seq(), (), _RaisesIfCompiled())

    assert isinstance(result, MeasurementResult)
    assert result.meta["acquired_via"] == "ACQUIRED"


class _TrivialExp(Experiment):
    def sequence(self):
        return (Seq(), ())

    def analyze(self, result):
        return None


def test_go_drives_the_same_split_through_the_experiment_seam(stub_program):
    runner = LegacyPyroRunner(_RaisesIfAcquired())
    exp = _TrivialExp(device=None, runner=runner)

    outcome = exp.go(soc=_RaisesIfCompiled(), analyze=False)

    assert outcome.result.meta["acquired_via"] == "ACQUIRED"


def test_reuses_the_base_runners_arm_and_attach_shots_without_overriding_them():
    """AC1's 'reusing the base Runner's _arm/_attach_shots' — proved by identity, not behavior:

    if a future edit ever gave ``LegacyPyroRunner`` its own ``_arm``/``_attach_shots``, this
    fails immediately, before any behavioral drift could hide the duplication.
    """
    assert LegacyPyroRunner._arm is Runner._arm
    assert LegacyPyroRunner._attach_shots is Runner._attach_shots


def test_stamps_the_result_via_the_shared_stamp_environment_helper(stub_program):
    """AC1's 'reusing ... stamp_environment' — proved behaviorally: a configured environment
    source actually lands on ``meta[\"environment\"]``, exactly like the base Runner."""
    runner = LegacyPyroRunner(_RaisesIfAcquired(), environment={"fridge_mK": 11.4})

    result = runner.run(None, Seq(), (), _RaisesIfCompiled())

    assert result.meta[ENVIRONMENT_KEY]["readings"] == {"fridge_mK": 11.4}


# -- purity (Constraints & Invariants: must not import qick or Pyro4 at module load) -----------


def test_module_imports_without_qick_or_pyro4():
    import subprocess

    code = (
        "import sys; sys.modules['qick'] = None; sys.modules['Pyro4'] = None; "
        "import strumento.core.legacy_pyro_runner; print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# -- the docstring (AC2) ------------------------------------------------------------------------


def test_docstring_names_d14_and_the_recommended_alternative():
    doc = LegacyPyroRunner.__doc__
    assert "D14" in doc
    assert "TransportRunner" in doc
    assert "JobServerSoc" in doc


def test_docstring_documents_the_dead_worker_thread_risk_and_its_recovery():
    doc = LegacyPyroRunner.__doc__.lower()
    assert "worker thread" in doc
    assert "make_proxy()" in doc
    assert "fresh" in doc  # the concrete recovery step: a fresh connection...
    assert "retrying" in doc or "retry" in doc  # ...not retrying in place


# -- the doc pointer (AC4) ----------------------------------------------------------------------


def test_job_manager_doc_points_at_the_legacy_runner():
    doc = Path(__file__).parents[2].joinpath("docs", "job-manager.md").read_text()
    assert "LegacyPyroRunner" in doc
    assert "legacy_pyro_runner" in doc
    assert "TransportRunner" in doc
