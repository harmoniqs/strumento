"""``strumento task`` + ``strumento store`` — the record format and the store, usable anywhere.

These verbs are what make the contract host-independent: a supervisor creates records with them,
CI reads state with them, and they are the fallback when a UI misbehaves. So they are tested the way
the other CLI verbs are — driving ``main(argv)`` in-process and asserting the JSON payload, plus
errors-as-data on every failure path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from strumento.cli import main
from strumento.core.config import DeviceConfig
from strumento.core.record import TaskRecord, read_manifest
from strumento.core.store import FileConfigStore, record_calibration, unvalidated


def _out(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


# -- task create ---------------------------------------------------------------------------


def test_task_create_writes_a_queued_record_with_the_rev2_fields(tmp_path, capsys):
    root = tmp_path / "experiments"
    argv = [
        "task", "create", "--json",
        "--title", "QILC CZ calibration",
        "--device", "loopback_demo",
        "--script", "scripts/qilc_cz.jl",
        "--workdir", str(tmp_path),
        "--language", "julia",
        "--task-type", "experiment",
        "--target", "sim",
        "--gates", "fit-recovery,schema-validate",
        "--authored-by", json.dumps(
            {"profile": "experimenter", "model": "prov/model-1", "variant": "v2", "session_id": "s9"}
        ),
        "--heartbeat-interval-s", "120",
        "--root", str(root),
    ]
    assert main(argv) == 0
    out = _out(capsys)
    assert out["ok"] and out["id"].endswith(tuple("0123456789abcdef"))
    task_dir = Path(out["dir"])
    assert task_dir.parent == root and task_dir.name == out["id"]

    man = read_manifest(task_dir)
    assert man.pid == 0                       # the supervisor owns pid; this record is `queued`
    assert man.title == "QILC CZ calibration" and man.device == "loopback_demo"
    assert man.language == "julia" and man.script == "scripts/qilc_cz.jl"
    assert man.task_type == "experiment" and man.target == "sim"
    assert man.gates == ("fit-recovery", "schema-validate")
    assert man.heartbeat_interval_s == 120    # per-task cadence (docs/task-record.md), via submit
    assert man.authored_by.model == "prov/model-1" and man.authored_by.session_id == "s9"
    assert (task_dir / "artifacts").is_dir()


def test_task_create_defaults_are_the_standalone_ones(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["task", "create", "--title", "T1 monitor", "--json"]) == 0
    out = _out(capsys)
    task_dir = Path(out["dir"])
    assert task_dir.parent == tmp_path / "experiments"     # <workdir>/experiments
    man = read_manifest(task_dir)
    assert (man.language, man.target, man.task_type, man.kind) == (
        "python", "sim", "experiment", "experiment"
    )
    assert man.gates == () and man.device == "" and man.heartbeat_interval_s == 15
    assert man.authored_by.to_dict() == {
        "profile": "", "model": "", "variant": "", "session_id": ""
    }                                                      # a human run: empty, never null


def test_task_create_mints_a_bringup_campaign_record(tmp_path, capsys):
    """The kind axis travels the submit chain: a supervisor mints the campaign ledger with
    `task create --kind bringup`, before any runner layer exists."""
    argv = [
        "task", "create", "--json",
        "--title", "CQED bring-up",
        "--kind", "bringup",
        "--device", "loopback_demo",
        "--root", str(tmp_path),
    ]
    assert main(argv) == 0
    out = _out(capsys)
    assert out["ok"]
    man = read_manifest(Path(out["dir"]))
    assert man.kind == "bringup" and man.device == "loopback_demo"


@pytest.mark.parametrize("bad,match", [
    (["--target", "fridge"], "target must be one of"),
    (["--language", "fortran"], "language must be one of"),
    (["--kind", "campaign"], "kind must be one of"),
    (["--authored-by", "{"], "JSON"),
    (["--authored-by", '{"agent": "x"}'], "unknown authored_by"),
    (["--authored-by", "[1]"], "JSON object"),
])
def test_task_create_errors_are_data(tmp_path, capsys, bad, match):
    argv = ["task", "create", "--title", "t", "--root", str(tmp_path / "e"), "--json", *bad]
    assert main(argv) == 1
    out = _out(capsys)
    assert out["ok"] is False and match.lower() in out["error"].lower()


# -- task status / tail / list ---------------------------------------------------------------


def test_task_status_reports_derived_state_and_staleness_inputs(tmp_path, capsys, closing):
    rec = closing(TaskRecord.create(tmp_path / "2026-07-25-t1-ab12", title="t1", heartbeat=False))
    rec.progress(0, of=2, label="iter 0", t1_us=68.2)
    assert main(["task", "status", str(rec.dir), "--json"]) == 0
    out = _out(capsys)
    assert out["ok"] and out["state"] == "running" and out["id"] == rec.dir.name
    assert out["manifest"]["pid"] == os.getpid() and out["result"] is None
    live = out["liveness"]
    assert live["pid_alive"] is True and live["n_events"] == 1
    assert live["stale_after_s"] == 45.0 and live["last_event"]["metrics"] == {"t1_us": 68.2}


def test_task_status_of_a_finished_task(tmp_path, capsys):
    rec = TaskRecord.create(tmp_path / "d", title="t", heartbeat=False)
    rec.done(summary="F = 0.9931 after 38 iters")
    assert main(["task", "status", str(rec.dir), "--json"]) == 0
    out = _out(capsys)
    assert out["state"] == "done" and out["result"]["summary"].startswith("F = 0.9931")


def test_task_status_text_mode_is_a_one_liner(tmp_path, capsys):
    rec = TaskRecord.create(tmp_path / "d", title="t", pid=0, heartbeat=False)
    assert main(["task", "status", str(rec.dir)]) == 0
    text = capsys.readouterr().out
    assert "queued" in text and rec.dir.name in text


def test_task_status_on_a_plain_directory_is_an_error_as_data(tmp_path, capsys):
    assert main(["task", "status", str(tmp_path), "--json"]) == 1
    out = _out(capsys)
    assert out["ok"] is False and "not a task directory" in out["error"]


def test_task_tail_returns_the_last_n_events(tmp_path, capsys, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    for i in range(5):
        rec.progress(i, of=5, value=float(i))
    rec.gate("fit", True, value=0.1)
    assert main(["task", "tail", str(rec.dir), "-n", "3", "--json"]) == 0
    out = _out(capsys)
    kinds = [e["ev"] for e in out["events"]]
    assert kinds == ["progress", "progress", "gate"]
    assert out["events"][-1]["pass"] is True


def test_task_tail_of_an_empty_stream(tmp_path, capsys):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    assert main(["task", "tail", str(rec.dir), "--json"]) == 0
    assert _out(capsys)["events"] == []


def test_task_list_reports_every_record_under_a_root(tmp_path, capsys):
    root = tmp_path / "experiments"
    queued = TaskRecord.create(root / "2026-07-25-queued-0001", title="waiting", pid=0,
                               device="loopback_demo", heartbeat=False)
    finished = TaskRecord.create(root / "2026-07-25-finished-0002", title="over", heartbeat=False)
    finished.done(summary="ok")
    (root / "not-a-task-dir").mkdir()

    assert main(["task", "list", str(root), "--json"]) == 0
    out = _out(capsys)
    by_id = {t["id"]: t for t in out["tasks"]}
    assert set(by_id) == {queued.dir.name, finished.dir.name}
    assert by_id[queued.dir.name]["state"] == "queued"
    assert by_id[queued.dir.name]["device"] == "loopback_demo"
    assert by_id[finished.dir.name]["state"] == "done"


def test_task_list_of_a_missing_root_is_empty_not_an_error(tmp_path, capsys):
    assert main(["task", "list", str(tmp_path / "nope"), "--json"]) == 0
    assert _out(capsys)["tasks"] == []


def test_task_without_a_subcommand_is_an_error_as_data(capsys):
    assert main(["task", "--json"]) == 1
    out = _out(capsys)
    assert out["ok"] is False and "needs a subcommand" in out["error"]
    assert "create" in out["error"] and "tail" in out["error"]


# -- the store verbs -----------------------------------------------------------------------


@pytest.fixture
def calibrated(tmp_path):
    """A ``FileConfigStore`` with two versions on ``main`` — the calibration panel's data source."""
    root = tmp_path / "store"
    store = FileConfigStore(root)
    base = DeviceConfig(
        pack="cqed",
        topology=[{"kind": "transmon", "name": "qubit"}],
        wiring={"lines": {}, "readout": {"dac_ch": 0, "adc_chs": [0]}},
        calibration={"qubit": {"f_ge": 4000.0, "pulses": {"pi_ge": {"gain": 8192}}}},
    )
    first = store.put(base)
    store.set_ref("main", first)
    second, _ = record_calibration(
        store, "main", base, {"qubit.pulses.pi_ge.gain": 9350.0},
        source="amplitude_rabi fit", timestamp="2026-07-25T12:00:00Z",
        validate=unvalidated,  # a minimal, non-composable config: the store verbs only
    )
    return root, first, second


def test_store_current_resolves_the_pointer(calibrated, capsys):
    root, _first, second = calibrated
    assert main(["store", "current", str(root), "--json"]) == 0
    out = _out(capsys)
    assert out["ok"] and out["ref"] == "main" and out["content_id"] == second
    assert out["timestamp"] == "2026-07-25T12:00:00Z"


def test_store_current_of_an_absent_store_is_empty_not_an_error(tmp_path, capsys):
    """A deviceless / never-calibrated device has no history — an empty view, not a failure."""
    absent = tmp_path / "devices" / "spin_demo" / "store"
    assert main(["store", "current", str(absent), "--json"]) == 0
    out = _out(capsys)
    assert out["ok"] and out["content_id"] == ""
    assert not absent.exists(), "a read verb must not create a store"


def test_store_current_with_an_unset_ref(calibrated, capsys):
    root, _first, _second = calibrated
    assert main(["store", "current", str(root), "--ref", "nightly", "--json"]) == 0
    assert _out(capsys)["content_id"] == ""


def test_store_log_lists_the_version_history(calibrated, capsys):
    root, first, second = calibrated
    assert main(["store", "log", str(root), "--json"]) == 0
    out = _out(capsys)
    assert [e["content_id"] for e in out["log"]] == [second]
    entry = out["log"][0]
    assert entry["source"] == "amplitude_rabi fit" and entry["parent_id"] == first
    assert entry["updates"] == {"qubit.pulses.pi_ge.gain": [8192, 9350.0]}


def test_store_log_of_an_absent_store_is_empty(tmp_path, capsys):
    assert main(["store", "log", str(tmp_path / "nope"), "--json"]) == 0
    assert _out(capsys)["log"] == []


def test_store_diff_reports_per_value_changes(calibrated, capsys):
    root, first, second = calibrated
    assert main(["store", "diff", str(root), first, second, "--json"]) == 0
    out = _out(capsys)
    assert out["changed"] == {"qubit.pulses.pi_ge.gain": [8192, 9350.0]}
    assert out["added"] == {} and out["removed"] == {}
    # the diff's paths are `apply_updates` paths — a diff can be replayed as an update
    from strumento.core.config import apply_updates
    from strumento.core.store import FileConfigStore as _Store

    base = _Store(root).get(first)
    replayed, _ = apply_updates(base, {k: new for k, (_old, new) in out["changed"].items()})
    assert replayed.content_id() == second


def test_store_diff_of_identical_versions_is_empty(calibrated, capsys):
    root, first, _second = calibrated
    assert main(["store", "diff", str(root), first, first, "--json"]) == 0
    out = _out(capsys)
    assert out["changed"] == {} and out["added"] == {} and out["removed"] == {}


def test_store_diff_of_an_unknown_version_is_an_error_as_data(calibrated, capsys):
    root, first, _second = calibrated
    assert main(["store", "diff", str(root), first, "cfg-nope", "--json"]) == 1
    out = _out(capsys)
    assert out["ok"] is False and "cfg-nope" in out["error"]


def test_store_without_a_subcommand_is_an_error_as_data(capsys):
    assert main(["store", "--json"]) == 1
    out = _out(capsys)
    assert out["ok"] is False and "current" in out["error"] and "diff" in out["error"]


# -- the CLI stays qick-free ----------------------------------------------------------------


def test_task_and_store_verbs_run_without_qick(tmp_path):
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['qick']=None\n"
        "from strumento.cli import main\n"
        f"assert main(['task','create','--title','t','--root',{str(tmp_path / 'e')!r},'--json'])==0\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
