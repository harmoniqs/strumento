"""The ``--block-qick`` option (issue #124): make ``import qick`` fail for the whole session.

CI's ``pure`` lane runs without qick installed. On a machine that has qick, ``pytest --block-qick
-m "not qick"`` reproduces that lane: every ``import qick`` (or ``import qick.<sub>``) during
collection or a test raises ``ImportError``, and ``pytest.importorskip("qick")`` skips.

How: at configure time — before any test module is collected, hence imported — every ``qick*``
entry already in ``sys.modules`` is evicted and ``sys.modules["qick"]`` is set to ``None``, which
the import system treats as "not importable". The evicted entries come back at unconfigure.

Limitation: an object bound to qick *before* configure (a module-level ``import qick`` in a
plugin or conftest loaded ahead of this one) keeps its reference; only imports made from
configure onwards are blocked. ``tests/conftest.py`` imports no qick, so the suite is unaffected.

Registered by ``tests/conftest.py``; a standalone session can load it with
``-p tests._block_qick``.
"""

from __future__ import annotations

import sys

import pytest

_SAVED = pytest.StashKey[dict]()


def _is_qick(name: str) -> bool:
    return name == "qick" or name.startswith("qick.")


def pytest_addoption(parser):
    parser.addoption(
        "--block-qick",
        action="store_true",
        default=False,
        help="make `import qick` raise ImportError for the session, as if qick were not "
        'installed — `pytest --block-qick -m "not qick"` reproduces the CI pure lane.',
    )


def pytest_configure(config):
    if not config.getoption("--block-qick"):
        return
    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if _is_qick(name)}
    sys.modules["qick"] = None
    config.stash[_SAVED] = saved


def pytest_unconfigure(config):
    saved = config.stash.get(_SAVED, None)
    if saved is None:
        return
    for name in [n for n in sys.modules if _is_qick(n)]:
        del sys.modules[name]
    sys.modules.update(saved)
    del config.stash[_SAVED]
