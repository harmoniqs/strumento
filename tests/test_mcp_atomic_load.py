"""Replacing an MCP session is atomic: a failed ``load_device`` leaves the previous one intact (#192).

``LabSession.load_device`` used to assign the new store and ref, seed the new store, and only then
open the soc. A soc that refused to open (a device with no soccfg snapshot, or ``sim=`` passed to
the mock) left a session half-replaced: the store and ref were the new ones, the device, soc and
task record the old ones, and the new store already held a genesis object and a ref. The next
``record_calibration`` landed in a store the agent was told had failed to open.

Each test opens session A, attempts a replace that fails at one step, and checks that every piece
of session state is the *same object* as before (identity, not equality), that the new store root
gained nothing, and that A's task record is still open and heartbeating.

Markers: the session drives a soc, so the module is dual-marked ``mcp`` + ``qick`` (the ``mcp``
lane is the one that installs both). :class:`LabSession` itself needs no SDK.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from strumento.core.record import read_events
from strumento.mcp import LabSession, SessionError
from tests._paths import DEVICES
from tests._v2gate import requires_v2_soccfg

pytestmark = [pytest.mark.mcp, pytest.mark.qick, requires_v2_soccfg]

LOOPBACK = DEVICES / "loopback_demo" / "device.yaml"
#: an instance that carries no soccfg snapshot, so no soc can open on it
SNAPSHOTLESS = DEVICES / "tweezer_demo" / "device.yaml"

#: every attribute ``load_device`` assigns; all must survive a failed replace untouched
SESSION_ATTRS = ("device", "device_path", "soc", "soc_kind", "truth", "store", "ref", "outcomes",
                 "record")

#: a short timer period so "still heartbeating" is observed in well under a second
FAST_HEARTBEAT_S = 0.05
HEARTBEAT_WAIT_S = 5.0


@pytest.fixture
def session_a(tmp_path):
    """Session A: loopback on the simulator, store ``a``/main, a task record that heartbeats fast."""
    session = LabSession()
    session.load_device(
        str(LOOPBACK), soc="simulator", store=str(tmp_path / "a"), task=str(tmp_path / "tasks"),
    )
    record = session.record
    record.stop_heartbeat()                 # restart the timer at a test-sized cadence
    record.heartbeat_seconds = FAST_HEARTBEAT_S
    record.start_heartbeat()
    session.outcomes.append({"experiment": "sentinel", "fit": None})   # non-empty session memory
    try:
        yield session
    finally:
        session.close()


def _snapshot(session: LabSession) -> dict:
    return {name: getattr(session, name) for name in SESSION_ATTRS}


def _tree(root: Path) -> list[str]:
    """Every file under ``root`` (empty directories are not entries)."""
    if not root.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def _heartbeats(task_dir: Path) -> int:
    return sum(1 for e in read_events(task_dir) if e["ev"] == "heartbeat")


def _assert_intact(session: LabSession, before: dict, store_root: Path) -> None:
    for name, value in before.items():
        assert getattr(session, name) is value, f"{name} was replaced by a failed load_device"
    assert session.store.root == store_root
    assert session.ref == "main"


def _assert_still_heartbeating(record) -> None:
    timer = record._timer
    assert timer is not None and timer.is_alive(), "the previous record's heartbeat stopped"
    seen = _heartbeats(record.dir)
    deadline = time.monotonic() + HEARTBEAT_WAIT_S
    while _heartbeats(record.dir) <= seen:
        assert time.monotonic() < deadline, "no new heartbeat from the previous record"
        time.sleep(FAST_HEARTBEAT_S)


def _assert_write_lands_in_a(session: LabSession, store_a: Path) -> None:
    before = store_a.joinpath("refs", "main").read_text()
    written = session.record_calibration({"qubit.f_ge": 4000.5})
    assert written["ref"] == "main"
    assert store_a.joinpath("refs", "main").read_text() == written["content_id"] != before
    assert session.store.root == store_a


#: each way a replace can fail: the load_device kwargs, plus an optional setup of the new store
FAILED_REPLACES = {
    "mock-with-sim": dict(path=LOOPBACK, soc="mock", ref="other", sim={"pi_gain": 9000}),
    "no-snapshot": dict(path=SNAPSHOTLESS, soc="simulator", ref="other"),
    "no-snapshot-mock": dict(path=SNAPSHOTLESS, soc="mock", ref="other"),
}


@pytest.mark.parametrize("case", list(FAILED_REPLACES.values()), ids=list(FAILED_REPLACES))
def test_a_failed_replace_leaves_the_previous_session_intact(session_a, tmp_path, case):
    store_a, store_b = tmp_path / "a", tmp_path / "b"
    before = _snapshot(session_a)
    record = session_a.record

    kwargs = dict(case)
    with pytest.raises(SessionError):
        session_a.load_device(str(kwargs.pop("path")), store=str(store_b),
                              task=str(tmp_path / "tasks-b"), **kwargs)

    _assert_intact(session_a, before, store_a)
    assert _tree(store_b) == []                        # no object, ref or log entry in b
    assert _tree(tmp_path / "tasks-b") == []           # and no task minted for the failed session
    _assert_still_heartbeating(record)
    _assert_write_lands_in_a(session_a, store_a)


def test_a_failed_task_open_leaves_the_previous_session_intact(session_a, tmp_path, monkeypatch):
    store_a, store_b = tmp_path / "a", tmp_path / "b"
    before = _snapshot(session_a)
    record = session_a.record

    def refuse(*_args, **_kwargs):
        raise OSError("task directory is not writable")

    monkeypatch.setattr(LabSession, "_open_task", staticmethod(refuse))
    with pytest.raises(OSError):
        session_a.load_device(str(LOOPBACK), soc="simulator", store=str(store_b), ref="other",
                              task=str(tmp_path / "tasks-b"))

    _assert_intact(session_a, before, store_a)
    assert _tree(store_b) == []
    _assert_still_heartbeating(record)
    _assert_write_lands_in_a(session_a, store_a)


def test_a_store_whose_ref_is_unreadable_leaves_the_previous_session_intact(session_a, tmp_path):
    store_a, store_b = tmp_path / "a", tmp_path / "b"
    (store_b / "refs").mkdir(parents=True)
    (store_b / "refs" / "other").write_text("0" * 64)    # points at an object the store lacks
    files_b = _tree(store_b)
    before = _snapshot(session_a)
    record = session_a.record

    with pytest.raises(KeyError):
        session_a.load_device(str(LOOPBACK), soc="simulator", store=str(store_b), ref="other",
                              task=str(tmp_path / "tasks-b"))

    _assert_intact(session_a, before, store_a)
    assert _tree(store_b) == files_b
    assert _tree(tmp_path / "tasks-b") == []
    _assert_still_heartbeating(record)
    _assert_write_lands_in_a(session_a, store_a)


def test_a_store_whose_ref_holds_an_invalid_config_leaves_the_previous_session_intact(
    session_a, tmp_path,
):
    """The #191 validation, on the read side: a ref whose config does not compose is refused."""
    store_a, store_b = tmp_path / "a", tmp_path / "b"
    good = json.loads(next((store_a / "objects").iterdir()).read_text())
    good["calibration"]["qubit"]["f_ge"] = "not a frequency"
    (store_b / "objects").mkdir(parents=True)
    (store_b / "refs").mkdir()
    bad_id = "f" * 64
    (store_b / "objects" / f"{bad_id}.json").write_text(json.dumps(good))
    (store_b / "refs" / "other").write_text(bad_id)
    files_b = _tree(store_b)
    before = _snapshot(session_a)
    record = session_a.record

    with pytest.raises(ValueError, match="f_ge"):     # the calibration model's ValidationError
        session_a.load_device(str(LOOPBACK), soc="simulator", store=str(store_b), ref="other",
                              task=str(tmp_path / "tasks-b"))

    _assert_intact(session_a, before, store_a)
    assert _tree(store_b) == files_b
    _assert_still_heartbeating(record)
    _assert_write_lands_in_a(session_a, store_a)


def test_a_failed_first_load_leaves_the_session_unloaded(tmp_path):
    session = LabSession()
    with pytest.raises(SessionError):
        session.load_device(str(SNAPSHOTLESS), soc="simulator", store=str(tmp_path / "b"),
                            ref="other", task=str(tmp_path / "tasks"))
    try:
        assert session.device is None and session.soc is None and session.record is None
        assert session.store is None and session.ref == "main"
        assert session.outcomes == [] and session.truth == {}
        assert session.device_path == "" and session.soc_kind == ""
        assert _tree(tmp_path / "b") == [] and _tree(tmp_path / "tasks") == []
    finally:
        session.close()


def test_a_successful_replace_still_replaces_everything_and_closes_the_old_record(
    session_a, tmp_path,
):
    old_record, old_soc = session_a.record, session_a.soc
    store_b = tmp_path / "b"

    loaded = session_a.load_device(str(LOOPBACK), soc="mock", store=str(store_b), ref="other",
                                   task=str(tmp_path / "tasks-b"))

    assert loaded["ok"] is True and loaded["store"]["seeded"] is True
    assert loaded["store"]["ref"] == "other"
    assert session_a.store.root == store_b and session_a.ref == "other"
    assert session_a.soc is not old_soc and session_a.soc_kind == "mock"
    assert session_a.outcomes == []
    assert session_a.record is not old_record and session_a.record.dir.parent == tmp_path / "tasks-b"
    assert old_record._timer is None                   # the replaced record stopped heartbeating
    assert (store_b / "refs" / "other").read_text() == loaded["store"]["content_id"]


def test_a_failed_seed_closes_the_new_record_and_leaves_the_previous_session_intact(
    session_a, tmp_path, monkeypatch,
):
    """The seed is the last step; if it raises, the record it follows is closed, not leaked."""
    from strumento.core.store import FileConfigStore

    store_a = tmp_path / "a"
    before = _snapshot(session_a)
    record = session_a.record
    opened = []
    real_open = LabSession._open_task

    def tracking_open(*args, **kwargs):
        opened.append(real_open(*args, **kwargs))
        return opened[-1]

    def refuse(self, ref, content_id):
        raise OSError("refs directory is not writable")

    monkeypatch.setattr(LabSession, "_open_task", staticmethod(tracking_open))
    monkeypatch.setattr(FileConfigStore, "set_ref", refuse)
    with pytest.raises(OSError):
        session_a.load_device(str(LOOPBACK), soc="simulator", store=str(tmp_path / "b"),
                              ref="other", task=str(tmp_path / "tasks-b"))

    assert len(opened) == 1 and opened[0]._timer is None    # the candidate record was closed
    _assert_intact(session_a, before, store_a)
    _assert_still_heartbeating(record)
    monkeypatch.undo()
    _assert_write_lands_in_a(session_a, store_a)
