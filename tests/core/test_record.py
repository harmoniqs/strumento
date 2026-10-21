"""The task-record contract (``strumento/core/record.py``) — the durable on-disk truth.

The tests that carry the most weight here are the *durability* ones: a record's whole reason to
exist is that it survives things (a killed process, a SIGTERM, a supervisor restart), so those paths
run in real subprocesses rather than being simulated in-process.
"""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from strumento.core.record import (
    DEFAULT_HEARTBEAT_INTERVAL_S,
    EVENT_KINDS,
    MANIFEST,
    RESULT,
    STALE_FACTOR,
    TASK_DIR_ENV,
    AuthoredBy,
    Manifest,
    RecordError,
    TaskRecord,
    artifact_path,
    derive_state,
    list_tasks,
    new_task_id,
    read_events,
    read_manifest,
    read_result,
    relative_artifact,
    slugify,
)

RECORD_PY = Path(__file__).parents[2] / "strumento" / "core" / "record.py"


def _events(rec_dir, ev=None):
    out = read_events(rec_dir)
    return [e for e in out if ev is None or e.get("ev") == ev]


# -- ids and slugs --------------------------------------------------------------------------


def test_task_id_shape():
    tid = new_task_id("QILC CZ calibration")
    day, rest = tid[:10], tid[11:]
    assert datetime.strptime(day, "%Y-%m-%d")
    assert rest.startswith("qilc-cz-calibration-")
    assert len(rest.rsplit("-", 1)[1]) == 4


def test_slugify_degrades_to_task():
    assert slugify("!!!") == "task"
    assert slugify("T1 monitor / qubit 0") == "t1-monitor-qubit-0"
    assert len(slugify("x" * 200)) <= 40


# -- manifest round-trip --------------------------------------------------------------------


def test_create_writes_the_full_rev2_manifest(tmp_path):
    rec = TaskRecord.create(
        tmp_path / "2026-07-25-t1-monitor-ab12",
        title="T1 monitor",
        device="loopback_demo",
        script="scripts/t1.py",
        workdir=str(tmp_path),
        language="python",
        task_type="experiment",
        target="sim",
        gates=["fit-recovery", "schema-validate"],
        authored_by={"profile": "experimenter", "model": "p/m-1", "variant": "v", "session_id": "s"},
        heartbeat_interval_s=30,
        heartbeat=False,
    )
    man = read_manifest(rec.dir)
    assert man.schema == 1
    assert man.id == "2026-07-25-t1-monitor-ab12" == rec.dir.name
    assert (man.title, man.kind, man.task_type, man.target) == (
        "T1 monitor", "experiment", "experiment", "sim",
    )
    assert man.gates == ("fit-recovery", "schema-validate")
    assert (man.script, man.language, man.device) == ("scripts/t1.py", "python", "loopback_demo")
    assert man.heartbeat_interval_s == 30 and man.config_content_id == ""
    assert man.authored_by == AuthoredBy("experimenter", "p/m-1", "v", "s")
    assert man.pid == os.getpid()          # standalone: stamps its own
    assert (rec.dir / "artifacts").is_dir()


def test_manifest_toml_has_no_nulls_and_is_reparseable(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", title="", heartbeat=False)
    text = (rec.dir / MANIFEST).read_text()
    assert "null" not in text and "nan" not in text.lower()
    assert 'gates = []' in text and '[authored_by]' in text
    assert 'profile = ""' in text            # absent-as-empty-string, not absent
    assert read_manifest(rec.dir) == rec.manifest


def test_authored_by_defaults_are_empty_strings_for_a_human_run(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", title="by hand", heartbeat=False)
    assert rec.manifest.authored_by.to_dict() == {
        "profile": "", "model": "", "variant": "", "session_id": ""
    }


def test_unknown_authored_by_field_is_typed_error(tmp_path):
    with pytest.raises(RecordError, match="unknown authored_by"):
        TaskRecord.create(tmp_path / "d", authored_by={"agent": "x"}, heartbeat=False)


def test_bad_target_language_and_kind_are_typed_errors(tmp_path):
    with pytest.raises(RecordError, match="target must be one of"):
        TaskRecord.create(tmp_path / "a", target="fridge", heartbeat=False)
    with pytest.raises(RecordError, match="language must be one of"):
        TaskRecord.create(tmp_path / "b", language="fortran", heartbeat=False)
    with pytest.raises(RecordError, match="kind must be one of"):
        TaskRecord.create(tmp_path / "c", kind="campaign", heartbeat=False)


def test_heartbeat_interval_is_clamped_to_at_least_one(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat_interval_s=0, heartbeat=False)
    assert rec.manifest.heartbeat_interval_s == 1  # 0 would derive `stale` instantly


def test_creating_twice_in_one_dir_is_refused(tmp_path):
    TaskRecord.create(tmp_path / "d", heartbeat=False)
    with pytest.raises(RecordError, match="already exists"):
        TaskRecord.create(tmp_path / "d", heartbeat=False)


def test_create_under_mints_the_conventional_id(tmp_path):
    rec = TaskRecord.create_under(tmp_path / "experiments", title="Sweep campaign", heartbeat=False)
    assert rec.dir.parent == tmp_path / "experiments"
    assert "sweep-campaign" in rec.dir.name and rec.manifest.id == rec.dir.name


TASK_ID_RE = r"^\d{4}-\d{2}-\d{2}-[a-z0-9-]+-[0-9a-f]{4}$"


def _script_suffixes(monkeypatch, *suffixes):
    """Make the id suffix source yield ``suffixes`` in order (the last one repeats forever)."""
    import strumento.core.record as record_mod

    queue = list(suffixes)

    def fake_uuid4():
        hex4 = queue.pop(0) if len(queue) > 1 else queue[0]
        return SimpleNamespace(hex=hex4 + "0" * 28)

    monkeypatch.setattr(record_mod.uuid, "uuid4", fake_uuid4)


def test_create_under_retries_a_colliding_suffix_into_a_fresh_directory(tmp_path, monkeypatch):
    root = tmp_path / "experiments"
    _script_suffixes(monkeypatch, "a925")
    first = TaskRecord.create_under(root, title="psb histogram", heartbeat=False)
    assert first.dir.name.endswith("-a925")

    _script_suffixes(monkeypatch, "a925", "b31c")  # the next mint repeats the existing id
    second = TaskRecord.create_under(root, title="psb histogram", heartbeat=False)
    assert second.dir != first.dir and second.dir.parent == root
    assert second.dir.name.endswith("-b31c") and second.manifest.id == second.dir.name
    assert re.match(TASK_ID_RE, second.dir.name)
    assert read_manifest(first.dir).id == first.dir.name  # the incumbent is untouched


def test_create_under_treats_a_claimed_empty_directory_as_taken(tmp_path, monkeypatch):
    """The claim is the atomic ``mkdir`` itself: a directory another writer has claimed but not
    yet given a manifest is not ours to fill, so minting moves on to a fresh suffix."""
    root = tmp_path / "experiments"
    claimed = root / new_task_id("sweep", suffix="c0de")
    claimed.mkdir(parents=True)
    _script_suffixes(monkeypatch, "c0de", "d00d")
    rec = TaskRecord.create_under(root, title="sweep", heartbeat=False)
    assert rec.dir.name.endswith("-d00d")
    assert list(claimed.iterdir()) == []  # the other writer's claim was never written into


def test_create_under_gives_up_after_bounded_collisions(tmp_path, monkeypatch):
    root = tmp_path / "experiments"
    _script_suffixes(monkeypatch, "eeee")
    TaskRecord.create_under(root, title="t1", heartbeat=False)
    with pytest.raises(RecordError, match="could not mint a fresh task id") as err:
        TaskRecord.create_under(root, title="t1", heartbeat=False)
    assert str(root) in str(err.value)
    assert [d.name for d in root.iterdir()] == [new_task_id("t1", suffix="eeee")]


def test_create_under_minted_ids_match_the_documented_format(tmp_path):
    for title in ("QILC CZ cal", "!!!", "T1 monitor / qubit 0"):
        rec = TaskRecord.create_under(tmp_path, title=title, heartbeat=False)
        assert re.match(TASK_ID_RE, rec.dir.name), rec.dir.name


def test_create_under_releases_its_claim_when_the_manifest_is_refused(tmp_path, monkeypatch):
    _script_suffixes(monkeypatch, "f00f")
    with pytest.raises(RecordError, match="target must be one of"):
        TaskRecord.create_under(tmp_path, title="bad", target="moon", heartbeat=False)
    assert list(tmp_path.iterdir()) == []  # no orphan, half-claimed task directory


def test_immutable_fields_cannot_be_updated_in_place(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", title="t", heartbeat=False)
    with pytest.raises(RecordError, match="immutable after creation"):
        rec._update_manifest(title="other")
    with pytest.raises(RecordError, match="immutable after creation"):
        rec._update_manifest(gates=["x"])
    rec._update_manifest(host="elsewhere")   # one of the three mutable fields
    assert read_manifest(rec.dir).host == "elsewhere"


def test_reader_tolerates_unknown_manifest_fields(tmp_path):
    """Writers validate, readers tolerate — a newer writer's extra fields must not break a reader."""
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    path = rec.dir / MANIFEST
    text = path.read_text().replace(
        "[authored_by]", 'future_field = "from a newer writer"\n\n[authored_by]'
    )
    path.write_text(text + 'future_agent_field = "also new"\n')
    assert read_manifest(rec.dir).id == rec.dir.name  # forward compatible, both levels


def test_manifest_kind_axis_mints_experiment_or_bringup(tmp_path):
    """``kind`` is the reserved registry axis: "experiment" (default) or "bringup" (a bring-up
    campaign's ledger — minted like any task record; the runner layer owns its semantics)."""
    rec = TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False)
    assert rec.manifest.kind == "bringup"
    assert read_manifest(rec.dir).kind == "bringup"            # the axis survives the round trip
    plain = TaskRecord.create(tmp_path / "task", heartbeat=False)
    assert plain.manifest.kind == "experiment"                 # the default is unchanged


def test_read_manifest_on_a_plain_directory_is_typed_error(tmp_path):
    with pytest.raises(RecordError, match="not a task directory"):
        read_manifest(tmp_path)


# -- the event stream -----------------------------------------------------------------------


def test_progress_events_round_trip(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    rec.progress(0, of=3, label="iter 0", infidelity=3.2e-3)
    rec.progress(1, t1_us=68.2)
    got = _events(rec.dir, "progress")
    assert [e["step"] for e in got] == [0, 1]
    assert got[0]["of"] == 3 and got[0]["label"] == "iter 0"
    assert got[0]["metrics"] == {"infidelity": 3.2e-3}
    assert "of" not in got[1] and got[1]["metrics"] == {"t1_us": 68.2}
    assert all(e["t"].endswith("Z") for e in got)


def test_progress_metrics_must_be_numbers(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    with pytest.raises(RecordError, match="must be a number"):
        rec.progress(0, note="looks good")


def test_artifact_event_records_a_relative_path(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    rec.artifact("artifacts/fit_012.json", kind="fit")
    rec.artifact(rec.artifacts / "plot.png")                    # absolute, inside the dir
    got = _events(rec.dir, "artifact")
    assert got[0] == {**got[0], "path": "artifacts/fit_012.json", "kind": "fit"}
    assert got[1]["path"] == "artifacts/plot.png" and "kind" not in got[1]


@pytest.mark.parametrize("escape", ["../evil.png", "artifacts/../../evil.png", "/etc/passwd"])
def test_artifact_escapes_are_rejected(tmp_path, escape):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    with pytest.raises(RecordError, match="escapes the task directory"):
        rec.artifact(escape)
    assert _events(rec.dir, "artifact") == []


def test_readers_reject_escapes_too(tmp_path):
    """The containment check belongs at *every* boundary — a recorded path is untrusted data."""
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    with pytest.raises(RecordError, match="escapes the task directory"):
        artifact_path(rec.dir, "../../etc/passwd")
    assert relative_artifact(rec.dir, "artifacts/ok.png") == "artifacts/ok.png"


def test_calibration_event_round_trips(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    rec.calibration("cfg-cd34ef")
    assert _events(rec.dir, "calibration")[0]["content_id"] == "cfg-cd34ef"
    with pytest.raises(RecordError, match="content id"):
        rec.calibration("  ")


def test_calibration_event_carries_extra_fields(tmp_path, closing):
    """Where the value came from rides on the event (#194); the event's own keys are reserved."""
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    rec.calibration("cfg-cd34ef", soc_kind="simulator", simulated=True, truth_hash="ab12")
    event = _events(rec.dir, "calibration")[0]
    assert event["content_id"] == "cfg-cd34ef"
    assert (event["soc_kind"], event["simulated"], event["truth_hash"]) == ("simulator", True, "ab12")
    for reserved in ("t", "ev", "content_id"):
        with pytest.raises(RecordError, match=reserved):
            rec.calibration("cfg-1", **{reserved: "x"})
    assert len(_events(rec.dir, "calibration")) == 1


def test_gate_event_round_trips(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    rec.gate("fit-recovery", True, metric="pi_gain_rel_err", value=0.004, tolerance=0.02)
    rec.gate("fit", False, reason="lmfit did not converge")
    got = _events(rec.dir, "gate")
    assert got[0]["name"] == "fit-recovery" and got[0]["pass"] is True
    assert got[0]["params"] == {"metric": "pi_gain_rel_err", "value": 0.004, "tolerance": 0.02}
    assert got[1]["pass"] is False and got[1]["params"]["reason"].startswith("lmfit")
    with pytest.raises(RecordError, match="gate name"):
        rec.gate("", True)


def test_gate_params_coerce_non_json_values(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    np = pytest.importorskip("numpy")
    rec.gate("fit", True, value=np.float64(0.5), curve=np.array([1.0, 2.0]), where=tmp_path)
    got = _events(rec.dir, "gate")[0]["params"]
    assert got["value"] == 0.5 and got["curve"] == [1.0, 2.0] and isinstance(got["where"], str)
    json.dumps(got)  # the whole point: it serializes


def test_unknown_event_kinds_survive_a_read(tmp_path, closing):
    """Forward compatibility is part of the contract: readers ignore, they do not choke."""
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    rec.heartbeat()
    with (rec.dir / "progress.jsonl").open("a") as fh:
        fh.write(json.dumps({"t": "2030-01-01T00:00:00Z", "ev": "from-the-future", "x": 1}) + "\n")
    rec.progress(1)
    kinds = [e["ev"] for e in read_events(rec.dir)]
    assert kinds == ["heartbeat", "from-the-future", "progress"]


def test_corrupt_line_is_skipped_and_the_rest_survives(tmp_path, closing):
    """kill-mid-write: the truncated line is lost, every complete line is not."""
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    rec.progress(0, value=1.0)
    rec.close()
    path = rec.dir / "progress.jsonl"
    with path.open("a") as fh:
        fh.write('{"t": "2026-07-25T00:00:00Z", "ev": "progr')   # killed here
    reopened = closing(TaskRecord.open(rec.dir))
    reopened.progress(1, value=2.0)
    got = _events(rec.dir, "progress")
    assert [e["step"] for e in got] == [0, 1]


def test_events_are_flushed_per_line_so_another_process_sees_them(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    rec.progress(0, value=1.0)
    # no close(): the file handle is still open, but the line must already be on disk
    assert '"step": 0' in (rec.dir / "progress.jsonl").read_text()


# -- heartbeats -----------------------------------------------------------------------------


def test_explicit_heartbeat_appends_an_event(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    rec.heartbeat()
    assert len(_events(rec.dir, "heartbeat")) == 1


def test_timer_thread_beats_on_its_own(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    rec.heartbeat_seconds = 0.05          # the manifest's cadence, shortened for the test
    rec.start_heartbeat()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and len(_events(rec.dir, "heartbeat")) < 2:
        time.sleep(0.05)
    rec.close()
    beats = len(_events(rec.dir, "heartbeat"))
    assert beats >= 2, f"timer thread produced {beats} heartbeats"
    time.sleep(0.2)
    assert len(_events(rec.dir, "heartbeat")) == beats   # close() stopped it


def test_default_cadence_is_the_documented_one(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    assert rec.manifest.heartbeat_interval_s == DEFAULT_HEARTBEAT_INTERVAL_S == 15
    assert rec.heartbeat_seconds == 15.0


# -- set_config -----------------------------------------------------------------------------


def test_set_config_stamps_once_when_empty(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    rec.set_config("cfg-aaa")
    assert read_manifest(rec.dir).config_content_id == "cfg-aaa"
    rec.set_config("cfg-aaa")                       # same id: a no-op, not an error
    with pytest.raises(RecordError, match="already"):
        rec.set_config("cfg-bbb")
    assert read_manifest(rec.dir).config_content_id == "cfg-aaa"
    with pytest.raises(RecordError, match="content id"):
        TaskRecord.create(tmp_path / "e", heartbeat=False).set_config("")


# -- terminal states ------------------------------------------------------------------------


def test_done_writes_result_exactly_once(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    res = rec.done(summary="F = 0.9931 after 38 iters")
    assert res.state == "done" and res.error_kind == "" and res.error == ""
    on_disk = read_result(rec.dir)
    assert on_disk == res and on_disk.summary.startswith("F = 0.9931")
    assert "null" not in (rec.dir / RESULT).read_text()
    with pytest.raises(RecordError, match="exactly once"):
        rec.done()
    with pytest.raises(RecordError, match="exactly once"):
        rec.fail("nope")


def test_fail_records_the_error_kind(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    rec.fail("job server unreachable", kind="transport-error")
    res = read_result(rec.dir)
    assert (res.state, res.error_kind) == ("failed", "transport-error")
    with pytest.raises(RecordError, match="error_kind must be one of"):
        TaskRecord.create(tmp_path / "e", heartbeat=False).fail("x", kind="cosmic-rays")


def test_result_is_written_atomically(tmp_path, monkeypatch):
    """tmp+rename: a reader never sees a partial terminal marker, so a parse error means corruption."""
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append((Path(src).name, Path(dst).name))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    rec.done()
    assert any(dst == RESULT and src != RESULT for src, dst in seen)


# -- supervised vs standalone pid ownership --------------------------------------------------


def test_standalone_from_env_creates_a_dir_and_stamps_its_own_pid(tmp_path, monkeypatch):
    monkeypatch.delenv(TASK_DIR_ENV, raising=False)
    rec = TaskRecord.from_env(root=tmp_path / "experiments", title="standalone", heartbeat=False)
    assert rec.supervised is False
    assert rec.manifest.pid == os.getpid()
    assert rec.dir.parent == tmp_path / "experiments"


def test_supervised_from_env_attaches_and_never_touches_pid(tmp_path, monkeypatch, closing):
    # the supervisor's side: create the manifest (pid = 0), then spawn with the env var set
    created = closing(
        TaskRecord.create(tmp_path / "2026-07-25-t1-ab12", title="t1", pid=0, heartbeat=False)
    )
    assert created.manifest.pid == 0                       # `queued` until the supervisor spawns
    monkeypatch.setenv(TASK_DIR_ENV, str(created.dir))

    rec = closing(TaskRecord.from_env(heartbeat=False))
    assert rec.supervised is True and rec.dir == created.dir
    rec.progress(0, x=1.0)
    assert read_manifest(rec.dir).pid == 0                 # the script never wrote pid
    with pytest.raises(RecordError, match="supervised"):
        rec.set_pid(os.getpid())


def test_standalone_record_may_set_its_own_pid(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", pid=0, heartbeat=False)
    rec.set_pid(4242)
    assert read_manifest(rec.dir).pid == 4242


def test_supervised_from_env_with_no_manifest_still_leaves_pid_alone(tmp_path, monkeypatch):
    monkeypatch.setenv(TASK_DIR_ENV, str(tmp_path / "spawned"))
    rec = TaskRecord.from_env(title="odd but recoverable", heartbeat=False)
    assert rec.supervised is True and rec.manifest.pid == 0
    assert rec.dir == tmp_path / "spawned"


# -- state derivation (the reader's total function) -------------------------------------------


def _stamp(rec, ago_s: float):
    """Age every liveness input by ``ago_s``: the manifest's mtime (the pid-write proxy) and the
    ``t`` of every recorded event (which is what liveness actually reads)."""
    when = time.time() - ago_s
    old = datetime.fromtimestamp(when, tz=timezone.utc).isoformat(timespec="milliseconds")
    old = old.replace("+00:00", "Z")
    events = rec.dir / "progress.jsonl"
    if events.exists():
        rec.close()
        lines = [json.loads(ln) for ln in events.read_text().splitlines() if ln.strip()]
        for line in lines:
            line["t"] = old
        events.write_text("".join(json.dumps(line) + "\n" for line in lines))
    os.utime(rec.dir / MANIFEST, (when, when))


def test_pid_zero_derives_queued(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", pid=0, heartbeat=False)
    assert derive_state(rec.dir).state == "queued"


def test_live_pid_with_fresh_liveness_derives_running(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)   # our own live pid
    status = derive_state(rec.dir)
    assert status.state == "running" and status.pid_alive is True


def test_dead_pid_derives_stale(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", pid=_dead_pid(), heartbeat=False)
    status = derive_state(rec.dir)
    assert status.state == "stale" and status.pid_alive is False


def test_hung_but_alive_derives_stale_before_running(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat_interval_s=1, heartbeat=False)
    rec.progress(0, x=1.0)
    _stamp(rec, ago_s=STALE_FACTOR * 1 + 2)
    status = derive_state(rec.dir)
    assert status.pid_alive is True and status.state == "stale"   # staleness precedes running
    assert status.age_s > status.manifest.stale_after_s


def test_just_spawned_with_no_events_is_running(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat_interval_s=60, heartbeat=False)
    status = derive_state(rec.dir)
    assert status.n_events == 0 and status.state == "running"


def test_streaming_progress_keeps_a_task_alive_without_heartbeats(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat_interval_s=1, heartbeat=False))
    _stamp(rec, ago_s=30)                 # manifest is ancient
    rec.progress(0, x=1.0)                # but events are fresh
    assert derive_state(rec.dir).state == "running"


def test_foreign_host_disables_the_pid_probe_but_not_staleness(tmp_path):
    rec = TaskRecord.create(
        tmp_path / "d", pid=_dead_pid(), host="some-other-host", heartbeat_interval_s=1,
        heartbeat=False,
    )
    fresh = derive_state(rec.dir)
    assert fresh.host_match is False and fresh.pid_alive is None
    assert fresh.state == "running"       # liveness is fresh: trust the heartbeats
    _stamp(rec, ago_s=STALE_FACTOR + 2)
    assert derive_state(rec.dir).state == "stale"   # and it ages into stale via rule 4


@pytest.mark.parametrize("state", ["done", "failed", "cancelled"])
def test_result_wins_over_every_liveness_input(tmp_path, state):
    rec = TaskRecord.create(tmp_path / f"d-{state}", pid=_dead_pid(), heartbeat=False)
    rec._terminal(state, summary="over")
    assert derive_state(rec.dir).state == state


def test_corrupt_result_derives_stale_with_the_reason_as_data(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    (rec.dir / RESULT).write_text("state = \n")
    status = derive_state(rec.dir)
    assert status.state == "stale" and "unreadable" in status.error


def test_out_of_enum_result_state_derives_stale(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    (rec.dir / RESULT).write_text('schema = 1\nstate = "vibing"\n')
    status = derive_state(rec.dir)
    assert status.state == "stale" and "vibing" in status.error


def test_status_payload_carries_the_staleness_inputs(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", title="t", heartbeat=False))
    rec.progress(0, x=1.0)
    payload = derive_state(rec.dir).to_dict()
    assert payload["state"] == "running" and payload["id"] == rec.dir.name
    assert payload["manifest"]["gates"] == [] and payload["result"] is None
    live = payload["liveness"]
    assert live["stale_after_s"] == STALE_FACTOR * DEFAULT_HEARTBEAT_INTERVAL_S
    assert live["n_events"] == 1 and live["last_event"]["ev"] == "progress"
    assert live["pid_alive"] is True and live["host_match"] is True
    json.dumps(payload)   # the CLI emits this verbatim


def test_list_tasks_skips_non_records_and_sorts_newest_first(tmp_path):
    root = tmp_path / "experiments"
    (root / "not-a-task").mkdir(parents=True)
    old = TaskRecord.create(root / "2026-01-01-old-0001", title="old", heartbeat=False)
    new = TaskRecord.create(root / "2026-12-31-new-0002", title="new", heartbeat=False)
    # `created` is what sorts, and both were created just now — pin it via the manifest text
    _rewrite_created(old, "2026-01-01T00:00:00.000Z")
    _rewrite_created(new, "2026-12-31T00:00:00.000Z")
    ids = [s.id for s in list_tasks(root)]
    assert ids == [new.dir.name, old.dir.name]
    assert list_tasks(tmp_path / "nope") == []


def _rewrite_created(rec, created: str) -> None:
    path = rec.dir / MANIFEST
    text = "\n".join(
        f'created = "{created}"' if line.startswith("created = ") else line
        for line in path.read_text().splitlines()
    )
    path.write_text(text + "\n")


def _dead_pid() -> int:
    """A pid that certainly is not running: spawn a trivial process and reap it."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# -- SIGTERM cancel (a real signal, a real process) -------------------------------------------


_CANCEL_SCRIPT = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {repo!r})
    from strumento.core.record import TaskRecord

    rec = TaskRecord.from_env(heartbeat=False)
    rec.on_cancel(lambda: rec.artifact(rec.artifacts / "cleanup.txt", kind="log"))
    rec.progress(0, x=1.0)
    (rec.dir / "ready").write_text("go")
    while True:
        time.sleep(0.05)
    """
)


def test_sigterm_runs_cleanup_and_writes_result_cancelled(tmp_path):
    rec = TaskRecord.create(tmp_path / "2026-07-25-cancel-me-ab12", title="cancel me", pid=0,
                            heartbeat=False)
    script = tmp_path / "task_script.py"
    script.write_text(_CANCEL_SCRIPT.format(repo=str(Path(__file__).parents[2])))
    env = {**os.environ, TASK_DIR_ENV: str(rec.dir)}
    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not (rec.dir / "ready").exists():
            time.sleep(0.05)
        assert (rec.dir / "ready").exists(), "task script never started"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 128 + int(signal.SIGTERM)
    finally:
        if proc.poll() is None:  # pragma: no cover - only on a hung test
            proc.kill()

    res = read_result(rec.dir)
    assert res.state == "cancelled" and res.error == "" and res.ended.endswith("Z")
    assert derive_state(rec.dir).state == "cancelled"
    # the cleanup hook ran, and its own event is in the stream
    assert [e["path"] for e in _events(rec.dir, "artifact")] == ["artifacts/cleanup.txt"]
    assert read_manifest(rec.dir).pid == 0     # supervised: the script never wrote pid


def test_kill_9_leaves_no_result_and_derives_stale(tmp_path):
    """An unclean stop must look unclean — cleanup did not run, so `stale` is the honest answer."""
    rec = TaskRecord.create(tmp_path / "2026-07-25-kill-me-ab12", pid=0, heartbeat_interval_s=1,
                            heartbeat=False)
    script = tmp_path / "task_script.py"
    script.write_text(_CANCEL_SCRIPT.format(repo=str(Path(__file__).parents[2])))
    env = {**os.environ, TASK_DIR_ENV: str(rec.dir)}
    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and not (rec.dir / "ready").exists():
        time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=30)

    assert not (rec.dir / RESULT).exists()
    rec._update_manifest(pid=proc.pid)          # the supervisor's field, as it would be at spawn
    assert derive_state(rec.dir).state == "stale"
    assert [e["step"] for e in _events(rec.dir, "progress")] == [0]   # the events survived


# -- purity ---------------------------------------------------------------------------------


def test_module_imports_without_qick():
    code = "import sys; sys.modules['qick']=None; import strumento.core.record; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_module_body_is_pure_stdlib():
    """The contract is public and host-independent: this one file must stand alone.

    Loaded straight off disk with every third-party dependency of the framework made unimportable —
    so a QICK user with a bare interpreter can vendor it, and a reader in CI needs nothing installed.
    """
    code = textwrap.dedent(
        f"""
        import sys, importlib.util
        for name in ("qick", "numpy", "pydantic", "yaml", "lmfit", "scipy", "strumento"):
            sys.modules[name] = None
        spec = importlib.util.spec_from_file_location("_record_standalone", {str(RECORD_PY)!r})
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        rec = mod.TaskRecord.create(sys.argv[1], title="standalone", heartbeat=False)
        rec.progress(0, of=1, value=1.0)
        rec.gate("fit", True, value=0.5)
        rec.claim("multimode_demo", "main")     # campaign emitters vendor with the file too
        rec.done(summary="ok")
        print(mod.derive_state(rec.dir).state)
        """
    )
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [sys.executable, "-c", code, str(Path(tmp) / "standalone-task")],
            capture_output=True, text=True,
        )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "done"


def test_every_import_in_the_source_is_stdlib():
    """A static guard (every import node, at any nesting), because the failure mode is a
    convenience import added later."""
    import ast

    allowed = set(sys.stdlib_module_names) | {"tomli", "tomllib"}   # tomli/tomllib = TOML parsers (tomllib is 3.11+ stdlib, tomli is backport)
    roots = set()
    for node in ast.walk(ast.parse(RECORD_PY.read_text())):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots, "no imports found — the parse went wrong"
    assert roots <= allowed, f"record.py imports non-stdlib module(s): {sorted(roots - allowed)}"


# -- the manifest is what a supervisor reads without this module ------------------------------


def test_manifest_is_plain_toml_a_python39_reader_can_parse(tmp_path):
    """No nulls, no inline tables, no dotted keys — the lowest-common-denominator subset."""
    rec = TaskRecord.create(
        tmp_path / "d", title='a "quoted" title\twith tabs', gates=["a", "b"], heartbeat=False
    )
    text = (rec.dir / MANIFEST).read_text()
    assert "\t" not in text.split("\n")[2]           # the tab was escaped, not embedded
    parsed = Manifest.from_dict(read_manifest(rec.dir).to_dict())
    assert parsed.title == 'a "quoted" title\twith tabs' and parsed.gates == ("a", "b")


def test_created_and_event_timestamps_share_one_format(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "d", heartbeat=False))
    rec.heartbeat()
    created = rec.manifest.created
    beat = _events(rec.dir, "heartbeat")[0]["t"]
    assert created.endswith("Z") and beat.endswith("Z")
    assert len(created) == len(beat)
    then = datetime.now(timezone.utc) - timedelta(seconds=5)
    assert then.isoformat(timespec="milliseconds").replace("+00:00", "Z") < created


# -- the file lifecycle: a record closes its event file (issue #123) --------------------------


def _unclosed_event_files(fn) -> list[str]:
    """Run ``fn`` (which builds and drops records), collect garbage, and return the
    ``ResourceWarning`` messages naming an unclosed ``progress.jsonl`` — the leak signature.

    Behavioural rather than white-box: a handle that is closed never warns at finalization.
    """
    import gc
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        fn()
        gc.collect()
    return [
        str(w.message) for w in caught
        if issubclass(w.category, ResourceWarning) and "progress.jsonl" in str(w.message)
    ]


def test_with_block_closes_the_event_file(tmp_path):
    def use():
        with TaskRecord.create(tmp_path / "d", heartbeat=False) as rec:
            rec.progress(0, of=1)
        assert len(_events(tmp_path / "d", "progress")) == 1

    assert _unclosed_event_files(use) == []


def test_with_block_returns_the_record_itself(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    with rec as entered:
        assert entered is rec


def test_with_block_closes_the_event_file_when_the_body_raises(tmp_path):
    def use():
        with pytest.raises(RuntimeError, match="boom"):
            with TaskRecord.create(tmp_path / "d", heartbeat=False) as rec:
                rec.progress(0, of=1)
                raise RuntimeError("boom")   # propagates: __exit__ never swallows

    assert _unclosed_event_files(use) == []
    assert len(_events(tmp_path / "d", "progress")) == 1


def test_with_block_stops_the_heartbeat_thread(tmp_path):
    with TaskRecord.create(tmp_path / "d", heartbeat=True) as rec:
        rec.heartbeat_seconds = 0.02
        rec.stop_heartbeat()
        rec.start_heartbeat()
        time.sleep(0.1)
    beats = len(_events(rec.dir, "heartbeat"))
    time.sleep(0.1)
    assert len(_events(rec.dir, "heartbeat")) == beats   # leaving the block stopped it


def test_closing_twice_is_harmless(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    rec.progress(0, of=1)
    rec.close()
    rec.close()
    with rec:
        pass
    rec.close()
    assert len(_events(rec.dir, "progress")) == 1


def test_an_event_after_close_reopens_and_appends(tmp_path):
    def use():
        rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
        rec.progress(0, of=2)
        rec.close()
        rec.progress(1, of=2)        # write-after-close reopens instead of raising
        rec.close()
        with rec:
            rec.heartbeat()          # a re-entered record appends to the same log
        steps = [e.get("step") for e in _events(rec.dir, "progress")]
        assert steps == [0, 1]
        assert len(_events(rec.dir, "heartbeat")) == 1

    assert _unclosed_event_files(use) == []


def test_a_terminal_result_closes_the_event_file(tmp_path):
    def use():
        rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
        rec.progress(0, of=1)
        rec.done(summary="ok")

    assert _unclosed_event_files(use) == []


#: the events file a fixed call sequence wrote BEFORE the lifecycle change (captured from the
#: pre-#123 code with ``now_iso`` pinned) — the on-disk format must not move by a byte.
_GOLDEN_EVENTS = (
    b'{"t": "2026-01-01T00:00:01.000Z", "ev": "claim", "device": "dev", "ref": "main"}\n'
    b'{"t": "2026-01-01T00:00:02.000Z", "ev": "step_open", "step": "a", "attempt": 1,'
    b' "child_path": "steps/x"}\n'
    b'{"t": "2026-01-01T00:00:03.000Z", "ev": "progress", "step": 0, "of": 2, "label": "first",'
    b' "metrics": {"t1_us": 68.2}}\n'
    b'{"t": "2026-01-01T00:00:04.000Z", "ev": "gate", "name": "fit", "pass": true,'
    b' "params": {"tol": 0.05}}\n'
    b'{"t": "2026-01-01T00:00:05.000Z", "ev": "heartbeat"}\n'
    b'{"t": "2026-01-01T00:00:06.000Z", "ev": "artifact", "path": "artifacts/fit.png",'
    b' "kind": "figure"}\n'
    b'{"t": "2026-01-01T00:00:07.000Z", "ev": "step_close", "step": "a", "attempt": 1,'
    b' "outcome": "pass"}\n'
    b'{"t": "2026-01-01T00:00:08.000Z", "ev": "calibration", "content_id": "abc123"}\n'
    b'{"t": "2026-01-01T00:00:09.000Z", "ev": "park", "kind": "needs_human", "reason": "why",'
    b' "step": "a"}\n'
)


def _golden_sequence(rec, *, interrupt) -> None:
    """The fixed call sequence behind ``_GOLDEN_EVENTS``; ``interrupt`` runs between calls."""
    calls = (
        lambda: rec.claim("dev", "main"),
        lambda: rec.step_open("a", 1, "steps/x"),
        lambda: rec.progress(0, of=2, label="first", t1_us=68.2),
        lambda: rec.gate("fit", True, tol=0.05),
        lambda: rec.heartbeat(),
        lambda: rec.artifact("artifacts/fit.png", kind="figure"),
        lambda: rec.step_close("a", 1, "pass"),
        lambda: rec.calibration("abc123"),
        lambda: rec.park("needs_human", "why", step="a"),
    )
    for call in calls:
        call()
        interrupt()


def _pinned_clock(monkeypatch):
    import itertools

    import strumento.core.record as record_mod

    ticks = itertools.count()
    monkeypatch.setattr(
        record_mod, "now_iso", lambda: f"2026-01-01T00:00:{next(ticks):02d}.000Z"
    )


@pytest.mark.parametrize("mode", ["held-open", "closed-between-calls", "with-block"])
def test_the_events_file_is_byte_identical_however_the_handle_is_managed(
    tmp_path, monkeypatch, mode
):
    _pinned_clock(monkeypatch)
    rec = TaskRecord.create(tmp_path / "d", title="golden", pid=0, host="h", heartbeat=False)
    if mode == "held-open":
        _golden_sequence(rec, interrupt=lambda: None)
    elif mode == "closed-between-calls":
        _golden_sequence(rec, interrupt=rec.close)
    else:
        with rec:
            _golden_sequence(rec, interrupt=lambda: None)
    rec.done(summary="ok")
    assert (rec.dir / "progress.jsonl").read_bytes() == _GOLDEN_EVENTS


def test_release_closes_the_event_file_but_keeps_the_heartbeat(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", heartbeat=False)
    rec.heartbeat_seconds = 0.02

    def use():
        rec.progress(0, of=1)
        rec.start_heartbeat()
        rec.release()                # the handle goes; the timer (someone else's) stays
        before = len(_events(rec.dir, "heartbeat"))
        time.sleep(0.15)
        assert len(_events(rec.dir, "heartbeat")) > before
        rec.close()

    assert _unclosed_event_files(use) == []


# -- the documented event vocabulary --------------------------------------------------------

DOCS_TASK_RECORD = Path(__file__).parents[2] / "docs" / "task-record.md"


def _appended_event_names() -> set[str]:
    """Every ``ev`` name a ``self._append(...)`` call site in ``record.py`` writes, read from
    the AST. A call whose event name is not a string literal fails loudly: the scan could not
    see what it writes, so the vocabulary check would be incomplete."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(RECORD_PY.read_text())):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_append"
        ):
            continue
        first = node.args[0] if node.args else None
        assert isinstance(first, ast.Constant) and isinstance(first.value, str), (
            f"record.py line {node.lineno}: _append() with a non-literal event name"
        )
        names.add(first.value)
    return names


def _exercise_every_emitter(rec: TaskRecord) -> None:
    (rec.artifacts / "fit.json").write_text("{}")
    rec.progress(1, of=2, label="iter 1", infidelity=1e-3)
    rec.artifact(rec.artifacts / "fit.json", kind="fit")
    rec.heartbeat()
    rec.calibration("cfg-1234")
    rec.gate("fit-recovery", True, tolerance=0.02)
    rec.claim("multimode_demo", "main")
    rec.step_open("resonator-find", 1, "steps/res-find-1")
    rec.step_close("resonator-find", 1, "pass")
    rec.park("propose", "outside cleared envelopes", proposal_ref="prop-1")
    rec.approve("operator", "operator-cli", "prop-1")
    rec.resume("operator", "rerun")
    rec.poison(["readout.region"], "moved outside the cleared region")


def test_every_event_the_module_writes_is_in_event_kinds(tmp_path):
    rec = TaskRecord.create(tmp_path / "d", kind="bringup", heartbeat=False)
    _exercise_every_emitter(rec)
    rec.close()
    written = {e["ev"] for e in read_events(rec.dir)}
    appended = _appended_event_names()
    # the exercise covers every call site, so a new emitter can't slip past it unexercised
    assert appended <= written, f"emitters not exercised: {sorted(appended - written)}"
    assert written | appended <= set(EVENT_KINDS), (
        f"written but undocumented: {sorted((written | appended) - set(EVENT_KINDS))}"
    )


def _documented_event_names() -> set[str]:
    """The ``ev`` names the ``progress.jsonl`` section of ``docs/task-record.md`` shows."""
    text = DOCS_TASK_RECORD.read_text()
    section = text.split("## `progress.jsonl`", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r'"ev":\s*"([a-z_]+)"', section))


def test_event_kinds_matches_the_documented_vocabulary():
    assert len(EVENT_KINDS) == len(set(EVENT_KINDS))
    assert _documented_event_names() == set(EVENT_KINDS)
