"""The diagnostics guard: qick/strumento WARNING+ log records and Python warnings, collected.

Shared by the experiment contract (`tests/packs/test_experiment_contract.py`, Guard 3 — read its
header note for why qick timing conflicts need a *logging* watcher, not `pytest.warns`) and any
other suite that must prove a program assembles conflict-free. One definition, so every caller
watches the same loggers with the same environment allow-list. Pure: no qick import.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager

#: Loggers whose WARNING-or-above records are treated as failures during assembly/acquire.
WATCHED_LOGGERS = ("qick", "strumento")

#: Substrings identifying records that describe the ENVIRONMENT rather than the program under test.
#: These are excluded from the fatal set — and the bar for adding one is high, because the whole
#: point of this guard is that qick reports program defects through `logging` (see the Guard 3 header
#: note in tests/packs/test_experiment_contract.py).
#: A record qualifies only if it is (a) about the toolchain/fixture rather than the sequence, and
#: (b) invariant across every test, so keeping it fatal fails all of them for one external fact and
#: tells you nothing about any individual sequence.
#:
#: `QICK library version mismatch`: `QickConfig` logs this when a soccfg snapshot's recorded
#: `sw_version` differs from the installed qick. Our committed testbench fixture was captured under
#: qick 0.2.367; CI installs qick unpinned, so it currently resolves 0.2.422 and every experiment
#: emitted this one record. It says nothing about timing. It IS a real staleness signal about the
#: fixture, so it is surfaced by `test_the_fixture_and_library_versions_are_reported` below rather
#: than dropped on the floor — the fix is to re-capture the fixture or pin qick, not to forget it.
ENVIRONMENT_NOTICES = ("QICK library version mismatch",)


@contextmanager
def _diagnostics_are_errors():
    """Collect Python warnings AND WARNING+ log records from the watched loggers.

    Yields `(records, notices, caught)`: `records` holds only what this guard treats as failures,
    and environment notices (see :data:`ENVIRONMENT_NOTICES`) are partitioned into `notices` rather
    than discarded, so a caller can assert on them.
    """

    class _Collector(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.WARNING)
            self.records: list[str] = []
            self.notices: list[str] = []

        def emit(self, record):
            line = f"{record.name}: {record.getMessage()}"
            target = self.notices if any(n in line for n in ENVIRONMENT_NOTICES) else self.records
            target.append(line)

    handler = _Collector()
    loggers = [logging.getLogger(n) for n in WATCHED_LOGGERS]
    saved = [(lg, lg.level, lg.propagate) for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield handler.records, handler.notices, caught
    finally:
        for lg in loggers:
            lg.removeHandler(handler)
        for lg, level, propagate in saved:
            lg.setLevel(level)
            lg.propagate = propagate
