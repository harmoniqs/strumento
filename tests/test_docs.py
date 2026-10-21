"""The docs test (issue #138): every fenced Python block in the user-facing docs runs.

One test per block, id'd ``<document>:<line>`` (the line of its opening fence), plus one
``<document>:writes`` test that fails if a document's snippets changed any file in the repo.
The harness, its markers and its execution semantics are in ``tests/_snippets.py``; the
contributor-facing summary is in ``CONTRIBUTING.md`` ("Docs snippets").

The snippets run against the simulator / offline mock, so this is a qick-lane test: without qick
(the CI pure lane, or ``pytest --block-qick``) the whole module skips with that reason.

To cover another document, add its repo-relative path to ``DOCUMENTS``.
"""

from __future__ import annotations

import pytest

from tests._paths import ROOT
from tests._snippets import document_params, run_snippet

pytestmark = pytest.mark.qick

pytest.importorskip(
    "qick", reason="the docs snippets run against the qick simulator; install the [qick] extra"
)

DOCUMENTS = (
    "README.md",
    "docs/getting-started.md",
    "docs/board-bringup.md",
    "docs/write-a-runner.md",
    "docs/calibrate-a-transmon.md",
    "docs/cqed-cookbook.md",
    "docs/job-manager.md",
    "docs/interop.md",
)


@pytest.mark.parametrize("snippet", document_params(ROOT, DOCUMENTS))
def test_docs_snippet(snippet):
    run_snippet(snippet)
