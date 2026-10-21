"""Suite-wide fixtures: freshly loaded shipped devices.

Each fixture loads its device anew per test, because tests mutate calibration. ``dev`` is the
multimode demo (the device most modules exercise); a module that needs a different device, a
different scope, or a pre-mutated copy defines its own ``dev`` on the paths in
``tests/_paths.py``.

It also carries the ``--block-qick`` option (issue #124, ``tests/_block_qick.py``): the hooks are
imported here so pytest registers them with this conftest, which loads before any test module is
collected — ``pytest --block-qick -m "not qick"`` reproduces CI's pure lane on a qick machine.

And the ``--update-compiled-golden`` option (issue #151): the compiled-program snapshot gate
(``tests/packs/test_compiled_snapshot_gate.py``) then rewrites the local, gitignored reference
files that differ, and only those; without the flag it never writes. The rewritten ones are
listed at the end of the run. The helper module holding that list is looked up lazily, so this
conftest imports no pack code.
"""

from __future__ import annotations

import contextlib
import sys

import pytest

from strumento.core.device import Device
from tests import _block_qick
from tests._block_qick import pytest_configure, pytest_unconfigure  # noqa: F401
from tests._paths import FIXTURES, LOOPBACK_YAML, MULTIMODE_YAML, SPIN_YAML


def pytest_addoption(parser):
    _block_qick.pytest_addoption(parser)
    parser.addoption(
        "--update-compiled-golden",
        action="store_true",
        default=False,
        help="write the local, gitignored compiled-program references (tests/golden/compiled/) "
        "that differ — run it on main, then run the gate on your branch.",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("--update-compiled-golden"):
        return
    golden = sys.modules.get("tests._compiled_golden")
    updated = sorted(golden.UPDATED) if golden is not None else []
    terminalreporter.write_sep("-", f"compiled references rewritten: {len(updated)}")
    for entry_id in updated:
        terminalreporter.write_line(f"  {entry_id}")


@pytest.fixture
def multimode_dev() -> Device:
    return Device.load(MULTIMODE_YAML)


@pytest.fixture
def loopback_dev() -> Device:
    return Device.load(LOOPBACK_YAML)


@pytest.fixture
def spin_dev() -> Device:
    return Device.load(SPIN_YAML)


@pytest.fixture
def v1_snapshot_yaml(tmp_path):
    """A copy of multimode_demo whose snapshot is the tProc-v1 ZCU216 config: a device file that
    loads and validates, but cannot back a v2 program or the simulator (the rejection paths)."""
    text = MULTIMODE_YAML.read_text(encoding="utf-8")
    original = "soccfg_snapshot: ../_fixtures/soccfg_v2_cqed.json"
    assert original in text, "multimode_demo's snapshot line moved; update this fixture"
    v1 = (FIXTURES / "soccfg_v1_zcu216.json").as_posix()
    path = tmp_path / "multimode_demo" / "device.yaml"
    path.parent.mkdir()
    path.write_text(text.replace(original, f"soccfg_snapshot: {v1}"), encoding="utf-8")
    return path


@pytest.fixture
def dev() -> Device:
    return Device.load(MULTIMODE_YAML)


@pytest.fixture
def closing():
    """Close what the TEST itself opened, on every exit path.

    ``rec = closing(TaskRecord.create(…))`` returns its argument, entered as a context manager
    and exited at teardown. It is for objects the test owns — a record or campaign the test
    constructed. A record some package code opened is that code's to close; wrapping it here
    would hide the leak the ``ResourceWarning`` filter exists to catch (issue #123).
    """
    with contextlib.ExitStack() as stack:
        yield stack.enter_context
