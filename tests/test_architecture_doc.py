"""``docs/architecture.md``'s layer chart and qick rule match the code (issue #206).

Two claims under "The three layers" drift silently when the tree changes, so they are checked
here, in the same way ``tests/test_readme_layout.py`` checks the README:

* **The chart's ``core`` line covers every core module.** The line is a comma-separated list of
  phrases, some grouping a few modules with ``+``. ``CORE_PHRASES`` maps each phrase to the
  modules it stands for; the phrases on the page must equal the mapping's keys, and the mapping
  must cover exactly the modules in ``strumento/core/`` (``__init__`` aside), each once.
* **The qick rule is exact.** The modules the qick sentence names equal the modules that import
  qick, found by the README test's AST scan — so this page and the README cannot disagree.

Pure lane: no qick needed.
"""

from __future__ import annotations

import ast
import re

from tests._paths import ROOT
from tests.test_readme_layout import (
    README,
    _core_modules_on_disk,
    _qick_importers_in_code,
    _qick_importers_in_readme,
)

ARCHITECTURE = ROOT / "docs" / "architecture.md"

#: Each phrase on the chart's ``core`` line (split on ``,`` and `` + ``) → the modules it names.
CORE_PHRASES: dict[str, set[str]] = {
    "wiring": {"wiring"},
    "pulse IR": {"pulses"},
    "sweeps": {"sweeps"},
    "device model": {"device"},
    "builder": {"builder"},
    "couplings": {"coupling"},
    "operator export": {"operators"},
    "compiler": {"compiler"},
    "program": {"program"},
    "mock": {"mock"},
    "result": {"result"},
    "readout/feedback protocols": {"readout"},
    "fitters (incl. resonator fitters)": {"fitters", "resonator_fitters"},
    "experiment scaffold": {"experiment"},
    "soc": {"soc"},
    "transport": {"transport"},
    "dc": {"dc"},
    "calibration store": {"store"},
    "config": {"config"},
    "task records": {"record"},
    "bring-up": {"bringup"},
    "simulator": {"simulator"},
    "response models": {"response"},
    "schemas": {"schemas"},
    "data chest": {"datachest"},
    "display": {"display"},
    "tracing": {"tracing"},
    "environment seam": {"environment"},
    "legacy Pyro runner": {"legacy_pyro_runner"},
}


def _layers_section(text: str) -> str:
    match = re.search(r"^## The three layers\n(.*?)(?=^##? |\Z)", text, re.S | re.M)
    assert match, "docs/architecture.md has no '## The three layers' section"
    return match.group(1)


def _chart(text: str) -> str:
    match = re.search(r"^```\n(.*?)^```", _layers_section(text), re.S | re.M)
    assert match, "the three-layers section has no fenced chart"
    return match.group(1)


def core_phrases(text: str) -> list[str]:
    """The phrases on the chart's ``core`` entry, continuation lines included."""
    lines = _chart(text).splitlines()
    first = next((i for i, ln in enumerate(lines) if re.match(r"^core\s", ln)), None)
    assert first is not None, "the chart has no 'core' line"
    column = re.match(r"^core\s+", lines[first]).end()  # type: ignore[union-attr]
    body = [lines[first][column:]]
    for nxt in lines[first + 1 :]:
        if len(nxt) - len(nxt.lstrip()) != column:
            break
        body.append(nxt[column:])
    joined = " ".join(" ".join(body).split())
    _, sep, items = joined.partition(":")  # drop the "modality-agnostic mechanism …:" head
    assert sep, "the core line has no ':' before its list"
    return [p.strip() for part in items.split(",") for p in part.split(" + ") if p.strip()]


def qick_importers_in_architecture(text: str) -> set[str]:
    """The ``core/<m>`` modules the section's one 'import qick' sentence names, as paths."""
    prose = " ".join(_layers_section(text).split("```")[2].split())
    sentences = [s for s in re.split(r"(?<=[.;])\s+", prose) if "import qick" in s]
    assert len(sentences) == 1, f"expected one 'import qick' sentence, found {sentences}"
    return {f"strumento/core/{m}.py" for m in re.findall(r"`core/(\w+)`", sentences[0])}


def test_chart_lines_fit_in_100_columns():
    long = [ln for ln in _chart(ARCHITECTURE.read_text(encoding="utf-8")).splitlines()
            if len(ln) > 100]
    assert not long, f"chart lines over 100 chars: {long}"


def test_the_phrase_mapping_covers_exactly_the_core_modules():
    mapped = [m for modules in CORE_PHRASES.values() for m in modules]
    assert len(mapped) == len(set(mapped)), "a module is mapped by more than one phrase"
    disk = _core_modules_on_disk()
    assert set(mapped) == disk, (
        f"CORE_PHRASES is out of date: unmapped {sorted(disk - set(mapped))}, "
        f"not in strumento/core/ {sorted(set(mapped) - disk)}"
    )


def test_chart_core_line_names_every_core_module():
    page = core_phrases(ARCHITECTURE.read_text(encoding="utf-8"))
    assert len(page) == len(set(page)), f"a phrase repeats on the core line: {page}"
    missing = sorted(set(CORE_PHRASES) - set(page))
    unknown = sorted(set(page) - set(CORE_PHRASES))
    assert not missing and not unknown, (
        f"architecture.md's core line: missing phrases {missing} "
        f"(modules {sorted(m for p in missing for m in CORE_PHRASES[p])}); "
        f"phrases with no module mapping {unknown}"
    )


def test_qick_rule_names_exactly_the_modules_that_import_qick():
    page = qick_importers_in_architecture(ARCHITECTURE.read_text(encoding="utf-8"))
    code = _qick_importers_in_code()
    assert code, "the AST scan found no qick import at all — has the scan broken?"
    assert page == code, f"architecture.md says {sorted(page)} import qick; code has {sorted(code)}"


def test_qick_rule_agrees_with_the_readme():
    page = qick_importers_in_architecture(ARCHITECTURE.read_text(encoding="utf-8"))
    readme = _qick_importers_in_readme(README.read_text(encoding="utf-8"))
    assert page == readme, f"architecture.md names {sorted(page)}; README names {sorted(readme)}"


def test_the_simulator_reaches_qick_through_the_mock():
    """The rule says the simulator needs qick only through the mock; hold it to that."""
    tree = ast.parse((ROOT / "strumento" / "core" / "simulator.py").read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "strumento.core.mock" in modules
