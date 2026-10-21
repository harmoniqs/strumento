"""The cqed sim-smoke runs on the cqed exemplar, ``multimode_demo`` (issue #131).

``examples/sim_smoke/run_cqed.py`` is imported and its core loop run in-process: every
registered cqed experiment is constructed on ``devices/multimode_demo`` and driven through
``SimulatorSoc`` (build + compile + acquire + fit). The PASS count may not drop below the floor
measured when the script moved onto multimode_demo, no experiment may be SKIPped (an analysis
that yields no fit is not a pass), and every non-PASS carries the reason the script prints.

Marked ``qick``: ``SimulatorSoc`` is a real v2 ``QickConfig``.
"""

from __future__ import annotations

import importlib.util

import pytest

from tests._paths import MULTIMODE_YAML, ROOT

pytest.importorskip("qick")

pytestmark = pytest.mark.qick

SCRIPT = ROOT / "examples" / "sim_smoke" / "run_cqed.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("sim_smoke_run_cqed", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def smoke():
    mod = _load_script()
    return mod, mod.run_all()


def test_the_smoke_targets_the_cqed_exemplar(smoke):
    mod, _ = smoke
    assert mod.DEFAULT_DEVICE.resolve() == MULTIMODE_YAML.resolve()


def test_every_registered_cqed_experiment_is_attempted(smoke):
    from strumento.packs.cqed import EXPERIMENTS

    _, results = smoke
    assert sorted(r["experiment"] for r in results) == sorted(EXPERIMENTS)


def test_the_pass_count_holds_the_measured_floor(smoke):
    mod, results = smoke
    passed = [r["experiment"] for r in results if r["status"] == "PASS"]
    failed = {r["experiment"]: r.get("reason") for r in results if r["status"] != "PASS"}
    assert len(passed) >= mod.PASS_FLOOR, failed
    assert mod.PASS_FLOOR >= 46  # measured on multimode_demo at the move (#131)


def test_no_experiment_is_skipped(smoke):
    _, results = smoke
    skipped = [r["experiment"] for r in results if r["status"] == "SKIP" or r.get("fit") == "SKIP"]
    assert skipped == []


def test_every_failure_carries_its_reason(smoke):
    _, results = smoke
    for r in results:
        if r["status"] != "PASS":
            assert r.get("reason"), r


def test_the_exit_code_tracks_the_floor(smoke):
    mod, results = smoke
    assert mod.exit_code(results) == 0
    assert mod.exit_code(results, min_pass=len(results) + 1) == 1
