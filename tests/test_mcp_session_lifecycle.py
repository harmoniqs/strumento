"""LabSession's task-record lifecycle without the MCP SDK (issue #123).

``tests/test_mcp.py`` needs the ``mcp`` extra and skips without it; :class:`LabSession` itself is
importable and drivable without the SDK (and without qick), so the one lifecycle obligation it
carries — it closes the session-scoped task record it opened — is pinned here, in the pure lane.
"""

from __future__ import annotations

import gc
import warnings

from strumento.core.record import TaskRecord, read_events
from strumento.mcp import LabSession


def _unclosed_event_files(fn) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        fn()
        gc.collect()
    return [
        str(w.message) for w in caught
        if issubclass(w.category, ResourceWarning) and "progress.jsonl" in str(w.message)
    ]


def _session_with_record(tmp_path) -> LabSession:
    session = LabSession()
    session.record = TaskRecord.create_under(tmp_path, title="session", heartbeat=False)
    session.record.progress(0, of=1)     # the handle is open, as after a run_experiment
    return session


def test_close_closes_the_session_record_and_is_idempotent(tmp_path):
    def use():
        session = _session_with_record(tmp_path)
        session.close()
        session.close()
        assert session.record is None

    assert _unclosed_event_files(use) == []


def test_a_session_is_a_context_manager(tmp_path):
    def use():
        with _session_with_record(tmp_path) as session:
            task_dir = session.record.dir
        assert session.record is None
        assert len(read_events(task_dir)) == 1   # the log stands; only the handle went

    assert _unclosed_event_files(use) == []


def test_close_without_a_record_is_harmless():
    LabSession().close()
