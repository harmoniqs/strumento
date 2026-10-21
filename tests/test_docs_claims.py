"""Factual claims in the docs that drift from the code (issue #140).

``tests/test_docs.py`` runs the docs' Python blocks; this module checks the statements around
them that no snippet exercises:

* **Experiment counts.** Any count a doc states ("50 registered", "registers 50 experiments")
  must equal the size of the pack's ``EXPERIMENTS`` registry. The pack is the one named most
  recently before the number (``cqed``, ``packs/spin/``, …). Prefer not stating a count at all;
  ``strumento experiments`` or ``len(EXPERIMENTS)`` is always current.
* **Per-name sweeps.** No Python block in the docs uses the deprecated per-name form
  ``Sweep("amp", 0, 16000, 51)``; examples use ``Sweep(start=…, stop=…, on=…)`` and declare the
  axis.
* **Readout trigger offsets.** A ``trig_offset`` in a docs example is in µs (#98), so it is a
  small number, not a pre-#98 clock-cycle count such as 400.

The documents scanned are every Markdown file git tracks, so an untracked local folder is never
read. Pure lane: no qick needed.
"""

from __future__ import annotations

import importlib
import re
import subprocess

import pytest

from tests._paths import ROOT
from tests._snippets import parse_document

PACKS = ("cqed", "spin", "tweezer")

# The largest readout trigger offset an example may show, in µs. The demos use 0.5; a value in
# the hundreds is a pre-#98 cycle count.
MAX_EXAMPLE_TRIG_OFFSET_US = 10.0


def _tracked_markdown() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.md"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout to list the tracked docs")
    return sorted(p for p in out.split("\0") if p)


DOCS = _tracked_markdown()

_COUNT = re.compile(
    r"\b(?P<n>\d+)\s+(?:registered\b|experiments\b)|\bregisters\s+(?P<m>\d+)\s+experiments\b"
)
_PACK_MENTION = re.compile(r"(?<![\w-])(?:packs/)?(?P<pack>" + "|".join(PACKS) + r")(?![\w-])")


def _registry_size(pack: str) -> int:
    return len(importlib.import_module(f"strumento.packs.{pack}").EXPERIMENTS)


def _count_claims():
    """(doc, line, count, pack) for every experiment count stated in the tracked docs."""
    claims = []
    for doc in DOCS:
        text = (ROOT / doc).read_text(encoding="utf-8")
        for match in _COUNT.finditer(text):
            before = list(_PACK_MENTION.finditer(text, 0, match.start()))
            pack = before[-1]["pack"] if before else None
            line = text.count("\n", 0, match.start()) + 1
            claims.append((doc, line, int(match["n"] or match["m"]), pack))
    return claims


def test_the_count_scanner_finds_claims():
    # Guards the scanner itself: the docs do state counts today, so finding none means the
    # pattern broke, not that the docs are clean.
    assert _count_claims(), "no experiment counts found — has _COUNT stopped matching?"


@pytest.mark.parametrize(
    "doc, line, count, pack",
    _count_claims(),
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_stated_experiment_count_matches_the_registry(doc, line, count, pack):
    assert pack is not None, f"{doc}:{line}: states {count} experiments without naming the pack"
    actual = _registry_size(pack)
    assert count == actual, (
        f"{doc}:{line}: says {count} {pack} experiments; the registry has {actual}. "
        "Fix the number, or drop it."
    )


_PER_NAME_SWEEP = re.compile(r"\bSweep\(\s*[\"']")


def test_no_docs_snippet_uses_the_per_name_sweep():
    found = []
    for doc in DOCS:
        for snippet in parse_document(ROOT, doc):
            if snippet.kind != "code":
                continue
            for offset, text in enumerate(snippet.source.splitlines(), start=1):
                if _PER_NAME_SWEEP.search(text):
                    found.append(f"{doc}:{snippet.line + offset}: {text.strip()}")
    assert not found, "deprecated per-name Sweep in a docs snippet:\n" + "\n".join(found)


_TRIG_OFFSET = re.compile(r"[\"']?trig_offset[\"']?\s*[:=]\s*(?P<v>-?\d+(?:\.\d+)?)")


def test_example_trig_offsets_are_in_microseconds():
    found = []
    for doc in DOCS:
        for n, text in enumerate((ROOT / doc).read_text(encoding="utf-8").splitlines(), start=1):
            for match in _TRIG_OFFSET.finditer(text):
                if float(match["v"]) > MAX_EXAMPLE_TRIG_OFFSET_US:
                    found.append(f"{doc}:{n}: trig_offset {match['v']}")
    assert not found, (
        "readout trig_offset is in µs since #98 (the demos use 0.5); these look like cycle "
        "counts:\n" + "\n".join(found)
    )
