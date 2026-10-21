"""The README's architecture graph and repository-layout chart match the code (issue #204).

Three claims in ``README.md`` drift silently when the tree changes, so they are checked here:

* **The layout chart lists every core module.** The names on the ``core/`` line of the
  "Repository layout" chart equal the modules in ``strumento/core/`` (``__init__`` aside).
* **The layout chart lists every demo device.** Every tracked ``devices/<name>/device.yaml``
  appears in the chart, and the ``_fixtures/`` line names exactly the demos whose
  ``device.yaml`` points at a soccfg snapshot.
* **The qick sentence is exact.** The modules the "Architecture" section says import qick equal
  the modules that do (an AST scan of every tracked ``strumento/`` source file, nested imports
  included).

The architecture graph also gets a small Mermaid syntax check (no renderer is assumed): balanced
``subgraph``/``end``, every line a node or an edge of the forms the graph uses, one label per
node id, and every edge endpoint a defined node or subgraph.

Files are listed with ``git ls-files``, so an untracked local folder is never read. Pure lane:
no qick needed.
"""

from __future__ import annotations

import ast
import re
import subprocess

import pytest
import yaml

from tests._paths import ROOT, SHIPPED_DEVICE_YAMLS

README = ROOT / "README.md"


def _section(text: str, heading: str) -> str:
    """The body of the ``## heading`` section (up to the next ``## `` heading)."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match, f"README has no '## {heading}' section"
    return match.group(1)


def _fence(section: str, lang: str = "") -> str:
    match = re.search(rf"^```{lang}\n(.*?)^```", section, re.S | re.M)
    assert match, f"no ```{lang} block in the section"
    return match.group(1)


def _layout_chart(text: str) -> str:
    return _fence(_section(text, "Repository layout"))


def _architecture(text: str) -> str:
    return _section(text, "Architecture")


# --- the layout chart ------------------------------------------------------------------------


def _entry(chart: str, name: str) -> list[str]:
    """The ``·``-separated items of one chart entry (``core/``), continuation lines included.

    A continuation line carries no entry name of its own: its text starts in the same column
    as the entry's first item.
    """
    lines = chart.splitlines()
    for i, line in enumerate(lines):
        match = re.match(rf"^(\s*){re.escape(name)}\s+", line)
        if match:
            column = match.end()
            body = [line[column:]]
            for nxt in lines[i + 1 :]:
                indent = len(nxt) - len(nxt.lstrip())
                if indent != column:
                    break
                body.append(nxt[column:])
            joined = " ".join(body)
            joined = re.sub(r"\(.*?\)", "", joined)  # drop remarks like "(built)"
            return [item.strip() for item in joined.split("·") if item.strip()]
    raise AssertionError(f"layout chart has no '{name}' entry")


def _core_modules_in_chart(text: str) -> set[str]:
    return set(_entry(_layout_chart(text), "core/"))


def _core_modules_on_disk() -> set[str]:
    return {p.stem for p in (ROOT / "strumento" / "core").glob("*.py")} - {"__init__"}


def _demo_devices() -> set[str]:
    return {p.parent.name for p in SHIPPED_DEVICE_YAMLS}


def _devices_in_chart(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w/])(\w+_demo)/", _layout_chart(text)))


def _snapshot_users_in_chart(text: str) -> set[str]:
    line = next(
        (ln for ln in _layout_chart(text).splitlines() if ln.strip().startswith("_fixtures/")),
        None,
    )
    assert line, "layout chart has no '_fixtures/' entry under devices/"
    return set(re.findall(r"\b(\w+_demo)\b", line))


def _snapshot_users_on_disk() -> set[str]:
    users = set()
    for path in SHIPPED_DEVICE_YAMLS:
        if yaml.safe_load(path.read_text(encoding="utf-8")).get("soccfg_snapshot"):
            users.add(path.parent.name)
    return users


def test_layout_chart_lists_every_core_module():
    chart = _core_modules_in_chart(README.read_text(encoding="utf-8"))
    disk = _core_modules_on_disk()
    assert chart == disk, (
        f"README layout chart's core/ line is out of date: missing {sorted(disk - chart)}, "
        f"not in strumento/core/ {sorted(chart - disk)}"
    )


def test_layout_chart_lists_every_demo_device():
    chart = _devices_in_chart(README.read_text(encoding="utf-8"))
    tracked = _demo_devices()
    assert tracked, "no tracked devices/*/device.yaml found"
    assert tracked <= chart, f"README layout chart is missing devices {sorted(tracked - chart)}"
    assert chart <= tracked, f"README layout chart names untracked devices {sorted(chart - tracked)}"


def test_layout_chart_names_the_demos_that_use_a_snapshot():
    chart = _snapshot_users_in_chart(README.read_text(encoding="utf-8"))
    disk = _snapshot_users_on_disk()
    assert chart == disk, (
        f"README _fixtures/ line names {sorted(chart)}; the device.yamls with a "
        f"soccfg_snapshot are {sorted(disk)}"
    )


# --- the qick sentence -----------------------------------------------------------------------


def _tracked_sources() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", "strumento/*.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout to list the tracked sources")
    return sorted(p for p in out.split("\0") if p)


def _imports_qick(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "qick" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.split(".")[0] == "qick":
                return True
    return False


def _qick_importers_in_code() -> set[str]:
    return {
        rel
        for rel in _tracked_sources()
        if _imports_qick((ROOT / rel).read_text(encoding="utf-8"))
    }


def _qick_sentence(text: str) -> str:
    prose = " ".join(_architecture(text).split("```")[0].split())
    sentences = [s for s in re.split(r"(?<=[.;])\s+", prose) if "import qick" in s]
    assert len(sentences) == 1, f"expected one 'import qick' sentence, found {sentences}"
    return sentences[0]


def _qick_importers_in_readme(text: str) -> set[str]:
    return set(re.findall(r"`(strumento/[\w/]+\.py)`", _qick_sentence(text)))


def test_qick_sentence_names_exactly_the_modules_that_import_qick():
    readme = _qick_importers_in_readme(README.read_text(encoding="utf-8"))
    code = _qick_importers_in_code()
    assert code, "the AST scan found no qick import at all — has the scan broken?"
    assert readme == code, (
        f"README says {sorted(readme)} import qick; the code has {sorted(code)}"
    )


# --- the architecture graph's Mermaid syntax -------------------------------------------------

_ID = r"[A-Za-z][\w]*"
# A label is quoted, or plain text free of the characters Mermaid's parser treats as shape
# delimiters (the existing graph quotes "cqed (built)" for exactly this reason).
_LABEL = r'(?:\["[^"\n]*"\]|\[[^\][(){}|<>"\n]*\])'
_NODE = rf"(?P<id>{_ID})(?P<label>{_LABEL})?"
_ARROW = r"(?:-->|-\.->)(?:\|[^|\n]+\|)?"
_EDGE = re.compile(rf"^{_NODE.replace('id', 'a').replace('label', 'la')}\s*{_ARROW}\s*"
                   rf"{_NODE.replace('id', 'b').replace('label', 'lb')}$")
_LONE = re.compile(rf"^{_NODE}$")
_SUBGRAPH = re.compile(rf'^subgraph\s+(?P<id>{_ID})\["[^"\n]*"\]$')


def mermaid_problems(source: str) -> list[str]:
    """Syntax problems in a flowchart of the shape the README uses (empty list = clean)."""
    lines = [ln.strip() for ln in source.splitlines() if ln.strip()]
    if not lines or not re.match(r"^flowchart (TD|TB|LR|RL|BT)$", lines[0]):
        return ["first line must be 'flowchart <direction>'"]
    problems, depth = [], 0
    labels: dict[str, str] = {}
    subgraphs: set[str] = set()
    endpoints: list[str] = []

    def define(node_id: str, label: str | None, n: int) -> None:
        if label is None:
            return
        if node_id in labels and labels[node_id] != label:
            problems.append(f"line {n}: {node_id} relabelled {label} (was {labels[node_id]})")
        labels[node_id] = label

    for n, line in enumerate(lines[1:], start=2):
        if (m := _SUBGRAPH.match(line)) is not None:
            depth += 1
            subgraphs.add(m["id"])
        elif line == "end":
            depth -= 1
            if depth < 0:
                problems.append(f"line {n}: 'end' without a subgraph")
        elif (m := _EDGE.match(line)) is not None:
            define(m["a"], m["la"], n)
            define(m["b"], m["lb"], n)
            endpoints += [m["a"], m["b"]]
        elif (m := _LONE.match(line)) is not None and m["label"]:
            define(m["id"], m["label"], n)
        else:
            problems.append(f"line {n}: not a node, edge, subgraph or end: {line!r}")
    if depth != 0:
        problems.append(f"{depth} subgraph(s) left open")
    clash = subgraphs & set(labels)
    if clash:
        problems.append(f"ids used for both a node and a subgraph: {sorted(clash)}")
    undefined = sorted({e for e in endpoints if e not in labels and e not in subgraphs})
    if undefined:
        problems.append(f"edge endpoints never labelled: {undefined}")
    return problems


def test_architecture_graph_is_well_formed_mermaid():
    source = _fence(_architecture(README.read_text(encoding="utf-8")), "mermaid")
    assert mermaid_problems(source) == []


@pytest.mark.parametrize(
    "bad",
    [
        "flowchart TD\n    subgraph a[\"A\"]\n        X[x] --> Y[y]\n",  # unclosed subgraph
        "flowchart TD\n    X[x] --> Y\n",  # Y never labelled
        "flowchart TD\n    X[x] --> Y[y]\n    X[other] --> Y\n",  # X relabelled
        "flowchart TD\n    X[x (y)] --> Y[y]\n",  # unquoted punctuation in a label
        "flowchart TD\n    X[x] -> Y[y]\n",  # not an arrow
    ],
)
def test_the_mermaid_check_flags_a_broken_graph(bad):
    assert mermaid_problems(bad)
