"""The docs-snippet harness (issue #138): run every fenced Python block in a Markdown document.

``tests/test_docs.py`` is the consumer; ``tests/test_snippets_harness.py`` holds its meta-tests.
The conventions are documented for contributors in ``CONTRIBUTING.md`` ("Docs snippets").

Extraction. Every fenced block whose info string starts with ``python`` / ``py`` / ``python3``
is a snippet, in document order. A snippet is identified by ``<document>:<line>``, the 1-based
line of its opening fence (e.g. ``docs/getting-started.md:17``).

Markers. An HTML comment on the line immediately before a fence (blank lines allowed between):

* ``<!-- snippet: skip reason="hardware only" -->`` — not executed; reported as skipped.
* ``<!-- snippet: xfail reason="undefined name; #139" -->`` — executed as a *strict* xfail: it
  must fail, and if it starts passing the test fails (remove the marker). The reason must cite
  the issue that clears it (``#<n>``).
* ``<!-- snippet: output -->`` (optionally ``rtol=`` / ``atol=``) before a fenced block (```text)
  that directly follows a Python block — that block's printed stdout must equal it.

A marker that is malformed, unknown, has an unknown attribute, lacks its reason, or binds to no
block fails the test at its line. Nothing is skipped silently.

Execution semantics. A document's snippets run once, in order, in ONE namespace, the first time
any of its tests runs; each test then reports its own snippet's recorded outcome — so the result
does not depend on test order or ``-k`` selection. A snippet that fails leaves the namespace as
it stopped; later snippets still run, and a failure names the earlier snippets that did not
complete (a ``NameError`` downstream usually comes from one of those). Skipped snippets do not
run, so their names are undefined for the snippets after them.

Optional dependencies. A snippet that fails because ``matplotlib`` or ``mcp`` is not installed
is skipped with that reason (CI's qick lane installs neither) instead of failing.

Sandbox. Snippets run with the working directory set to a temporary mirror of the document root
(a symlink per top-level entry), so relative reads such as ``Device.load("devices/…")`` see the
repo while relative writes land in the temp dir; ``HOME``, ``TMPDIR`` and the matplotlib/XDG
config dirs point into the temp dir; network connections raise; bytecode writing is off. After
the document runs, a ``<document>:writes`` test fails if any file under the root changed. The
whole process state (cwd, environment, ``tempfile.tempdir``) is restored afterwards.

Output comparison. Both sides are compared after stripping trailing whitespace from each line
and blank lines at the start/end. Without a tolerance the comparison is exact. With ``rtol`` /
``atol``, the text between numbers must match exactly and each pair of numbers must satisfy
``math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol)``.
"""

from __future__ import annotations

import contextlib
import difflib
import io
import math
import os
import re
import socket
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pytest

PYTHON_INFO = frozenset({"python", "py", "python3"})
OPTIONAL_DEPENDENCIES = {"matplotlib": "the [plot] extra", "mcp": "the [mcp] extra"}
SNAPSHOT_SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)

_FENCE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_MARKER_START = re.compile(r"^\s*<!--\s*snippet\b")
_MARKER = re.compile(r"^\s*<!--\s*snippet:\s*(?P<kind>\S+)(?P<attrs>.*?)\s*-->\s*$")
_ATTR = re.compile(r'\s*(?P<key>\w+)=(?:"(?P<quoted>[^"]*)"|(?P<bare>[^\s"]+))')
_ISSUE = re.compile(r"#\d+")
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_ALLOWED_ATTRS = {"skip": {"reason"}, "xfail": {"reason"}, "output": {"rtol", "atol"}}


@dataclass(frozen=True)
class Marker:
    kind: str
    line: int
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Expected:
    text: str
    line: int
    rtol: float = 0.0
    atol: float = 0.0
    tolerant: bool = False


@dataclass(frozen=True)
class Snippet:
    """One test: a Python block (``kind="code"``), a marker error, or a document's writes check."""

    root: Path
    document: str
    line: int
    kind: str  # "code" | "error" | "writes"
    source: str = ""
    marker: Marker | None = None
    expected: Expected | None = None
    error: str | None = None

    @property
    def id(self) -> str:
        return f"{self.document}:{'writes' if self.kind == 'writes' else self.line}"


@dataclass
class _Block:
    line: int
    info: str
    lines: list[str]
    marker: Marker | None = None
    marker_error: str | None = None
    marker_line: int = 0
    expected: Expected | None = None
    is_output: bool = False

    @property
    def is_python(self) -> bool:
        words = self.info.split()
        return bool(words) and words[0].lower() in PYTHON_INFO


# ---------------------------------------------------------------------------- parsing


def _parse_marker(text: str, line: int) -> tuple[Marker | None, str | None]:
    m = _MARKER.match(text)
    if m is None:
        expected = "<!-- snippet: KIND ... -->"
        return None, f"malformed snippet marker {text.strip()!r}; expected {expected}"
    kind = m["kind"]
    if kind not in _ALLOWED_ATTRS:
        return None, f"unknown snippet marker {kind!r}; known: {', '.join(sorted(_ALLOWED_ATTRS))}"
    attrs: dict[str, str] = {}
    rest = m["attrs"]
    pos = 0
    for a in _ATTR.finditer(rest):
        if a.start() != pos:
            break
        attrs[a["key"]] = a["quoted"] if a["quoted"] is not None else a["bare"]
        pos = a.end()
    if rest[pos:].strip():
        return None, f"malformed attributes in snippet marker {kind!r}: {rest[pos:].strip()!r}"
    unknown = set(attrs) - _ALLOWED_ATTRS[kind]
    if unknown:
        return None, f"unknown attribute(s) {sorted(unknown)} on snippet marker {kind!r}"
    if kind in ("skip", "xfail") and not attrs.get("reason", "").strip():
        return None, f'snippet marker {kind!r} needs a non-empty reason="..."'
    if kind == "xfail" and not _ISSUE.search(attrs["reason"]):
        return None, "an xfail reason must name the issue that clears it (#<n>)"
    if kind == "output":
        for key in ("rtol", "atol"):
            if key in attrs:
                try:
                    float(attrs[key])
                except ValueError:
                    return None, f"{key}={attrs[key]!r} on snippet marker 'output' is not a number"
    return Marker(kind, line, attrs), None


def parse_document(root: Path, document: str) -> list[Snippet]:
    """The snippets of ``root/document``, in document order, plus its writes check."""
    lines = (root / document).read_text(encoding="utf-8").splitlines()
    blocks: list[_Block] = []
    errors: list[tuple[int, str]] = []
    pending: tuple[Marker | None, str | None, int] | None = None
    last_python: _Block | None = None  # the python block an output marker may attach to

    def orphan(why: str) -> None:
        nonlocal pending
        if pending is not None:
            marker, err, at = pending
            errors.append((at, err or f"snippet marker {marker.kind!r} {why}"))
            pending = None

    i = 0
    while i < len(lines):
        text = lines[i]
        lineno = i + 1
        fence = _FENCE.match(text)
        if fence and not (fence["fence"][0] == "`" and "`" in fence["info"]):
            char, length, indent = fence["fence"][0], len(fence["fence"]), len(fence["indent"])
            body: list[str] = []
            j = i + 1
            while j < len(lines):
                close = _FENCE.match(lines[j])
                if (
                    close
                    and close["fence"][0] == char
                    and len(close["fence"]) >= length
                    and not close["info"].strip()
                ):
                    break
                raw = lines[j]
                strip = len(raw) - len(raw.lstrip(" "))
                body.append(raw[min(strip, indent) :])
                j += 1
            block = _Block(line=lineno, info=fence["info"].strip(), lines=body)
            if pending is not None:
                block.marker, block.marker_error, block.marker_line = pending
                pending = None
            if block.marker is not None and block.marker.kind == "output":
                block.is_output = True
                if last_python is None:
                    errors.append(
                        (
                            block.marker.line,
                            "snippet marker 'output' must directly follow a python block",
                        )
                    )
                elif last_python.expected is not None:
                    errors.append((block.marker.line, "a python block has two output markers"))
                else:
                    rtol = float(block.marker.attrs.get("rtol", 0.0))
                    atol = float(block.marker.attrs.get("atol", 0.0))
                    tolerant = "rtol" in block.marker.attrs or "atol" in block.marker.attrs
                    last_python.expected = Expected(
                        "\n".join(body), block.marker.line, rtol, atol, tolerant
                    )
                last_python = None
            elif block.marker is not None and not block.is_python:
                kind = block.marker.kind
                errors.append((block.marker.line, f"snippet marker {kind!r} needs a python block"))
                last_python = None
            else:
                last_python = block if block.is_python else None
            blocks.append(block)
            i = j + 1
            continue
        if _MARKER_START.match(text):
            orphan("is followed by another marker, not a fenced block")
            marker, err = _parse_marker(text, lineno)
            pending = (marker, err, lineno)
        elif text.strip():
            orphan("is not immediately followed by a fenced block")
            last_python = None
        i += 1
    orphan("is not followed by a fenced block")

    snippets: list[Snippet] = []
    for block in blocks:
        if not block.is_python or block.is_output:
            continue
        error = None
        if block.marker_error is not None:
            error = f"marker at {document}:{block.marker_line}: {block.marker_error}"
        snippets.append(
            Snippet(
                root=root,
                document=document,
                line=block.line,
                kind="error" if error else "code",
                source="\n" * block.line + "\n".join(block.lines) + "\n",
                marker=block.marker,
                expected=block.expected,
                error=error,
            )
        )
    for at, err in errors:
        snippets.append(
            Snippet(root=root, document=document, line=at, kind="error", error=f"{err}")
        )
    snippets.sort(key=lambda s: s.line)
    snippets.append(Snippet(root=root, document=document, line=len(lines) + 1, kind="writes"))
    return snippets


def document_params(root: Path, documents) -> list:
    """``pytest.param`` per snippet of every document, id'd ``<document>:<line>``."""
    params = []
    for document in documents:
        for snippet in parse_document(Path(root), document):
            marks = []
            if snippet.kind == "code" and snippet.marker is not None:
                reason = snippet.marker.attrs.get("reason", "")
                if snippet.marker.kind == "skip":
                    marks.append(pytest.mark.skip(reason=f"{snippet.id}: {reason}"))
                elif snippet.marker.kind == "xfail":
                    marks.append(pytest.mark.xfail(strict=True, reason=f"{snippet.id}: {reason}"))
            params.append(pytest.param(snippet, id=snippet.id, marks=marks))
    return params


# ---------------------------------------------------------------------------- execution


@dataclass
class _Outcome:
    status: str  # "passed" | "failed" | "skipped" | "missing-dep" | "not-run"
    message: str = ""
    stdout: str = ""


@dataclass
class _DocRun:
    outcomes: dict = field(default_factory=dict)  # line -> _Outcome
    writes: list = field(default_factory=list)


_RUNS: dict[tuple[str, str], _DocRun] = {}


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    state: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SNAPSHOT_SKIP_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            state[os.path.relpath(path, root)] = (st.st_size, st.st_mtime_ns)
    return state


def _no_network(*args, **kwargs):
    raise OSError("docs snippets must not use the network (blocked by the docs-test sandbox)")


@contextlib.contextmanager
def _sandbox(root: Path):
    with tempfile.TemporaryDirectory(prefix="strumento-docs-", ignore_cleanup_errors=True) as tmp:
        base = Path(tmp)
        cwd = base / "cwd"
        cwd.mkdir()
        for entry in root.iterdir():
            if entry.name != ".git":
                (cwd / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        dirs = {name: base / name for name in ("home", "tmp", "mpl", "cache", "config")}
        for d in dirs.values():
            d.mkdir()
        saved_env = dict(os.environ)
        saved_cwd = os.getcwd()
        saved_tempdir = tempfile.tempdir
        saved_bytecode = sys.dont_write_bytecode
        saved_connect = socket.socket.connect, socket.socket.connect_ex
        saved_create = socket.create_connection
        try:
            os.environ.update(
                HOME=str(dirs["home"]),
                TMPDIR=str(dirs["tmp"]),
                TEMP=str(dirs["tmp"]),
                TMP=str(dirs["tmp"]),
                MPLBACKEND="Agg",
                MPLCONFIGDIR=str(dirs["mpl"]),
                XDG_CACHE_HOME=str(dirs["cache"]),
                XDG_CONFIG_HOME=str(dirs["config"]),
            )
            tempfile.tempdir = str(dirs["tmp"])
            sys.dont_write_bytecode = True
            socket.socket.connect = _no_network
            socket.socket.connect_ex = _no_network
            socket.create_connection = _no_network
            os.chdir(cwd)
            yield
        finally:
            os.chdir(saved_cwd)
            socket.socket.connect, socket.socket.connect_ex = saved_connect
            socket.create_connection = saved_create
            sys.dont_write_bytecode = saved_bytecode
            tempfile.tempdir = saved_tempdir
            os.environ.clear()
            os.environ.update(saved_env)
            plt = sys.modules.get("matplotlib.pyplot")
            if plt is not None:
                plt.close("all")


def _missing_optional(exc: BaseException) -> str | None:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            top = exc.name.split(".")[0]
            if top in OPTIONAL_DEPENDENCIES:
                return top
        exc = exc.__cause__ or exc.__context__
    return None


def _format(exc: BaseException) -> str:
    """``<Type>: <message>`` first (it shows in the short summary), then the traceback."""
    tb = exc.__traceback__.tb_next if exc.__traceback__ is not None else None
    summary = traceback.format_exception_only(type(exc), exc)[-1].strip()
    return summary + "\n" + "".join(traceback.format_exception(type(exc), exc, tb)).rstrip()


def _execute(snippet: Snippet, namespace: dict) -> _Outcome:
    path = snippet.root / snippet.document
    buf = io.StringIO()
    try:
        # the source is padded with newlines so traceback line numbers are the document's
        code = compile(snippet.source, str(path), "exec")
        with contextlib.redirect_stdout(buf):
            exec(code, namespace)
    except (KeyboardInterrupt, pytest.exit.Exception):
        raise
    except BaseException as exc:  # noqa: BLE001 — any failure of a snippet is its outcome
        dep = _missing_optional(exc)
        if dep is not None:
            return _Outcome("missing-dep", f"needs {OPTIONAL_DEPENDENCIES[dep]} ({dep!r})")
        return _Outcome("failed", _format(exc), buf.getvalue())
    return _Outcome("passed", stdout=buf.getvalue())


def _run_document(root: Path, document: str) -> _DocRun:
    key = (str(root), document)
    if key in _RUNS:
        return _RUNS[key]
    run = _DocRun()
    _RUNS[key] = run
    snippets = [s for s in parse_document(root, document) if s.kind in ("code", "error")]
    namespace: dict = {"__name__": "__docs__"}
    before = _snapshot(root)
    with _sandbox(root):
        for s in snippets:
            if s.kind == "error":
                run.outcomes[s.line] = _Outcome("not-run", s.error)
            elif s.marker is not None and s.marker.kind == "skip":
                run.outcomes[s.line] = _Outcome("skipped", s.marker.attrs["reason"])
            else:
                run.outcomes[s.line] = _execute(s, namespace)
    after = _snapshot(root)
    for rel in sorted(set(before) | set(after)):
        if rel not in after:
            run.writes.append(f"removed {rel}")
        elif rel not in before:
            run.writes.append(f"created {rel}")
        elif before[rel] != after[rel]:
            run.writes.append(f"modified {rel}")
    return run


def _numbers_close(actual: str, expected: Expected) -> bool:
    a_nums, e_nums = _NUMBER.findall(actual), _NUMBER.findall(expected.text)
    if _NUMBER.split(actual) != _NUMBER.split(expected.text) or len(a_nums) != len(e_nums):
        return False
    return all(
        math.isclose(float(a), float(e), rel_tol=expected.rtol, abs_tol=expected.atol)
        for a, e in zip(a_nums, e_nums, strict=True)
    )


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip("\n")


def _earlier_trouble(run: _DocRun, snippet: Snippet) -> str:
    trouble = [
        f"{snippet.document}:{line} ({o.status})"
        for line, o in sorted(run.outcomes.items())
        if line < snippet.line and o.status not in ("passed",)
    ]
    if not trouble:
        return ""
    return "\nearlier snippets in this document that did not complete: " + ", ".join(trouble)


def run_snippet(snippet: Snippet) -> None:
    """The test body: report ``snippet``'s outcome from its document's (cached) run."""
    if snippet.kind == "error":
        pytest.fail(f"{snippet.id}: {snippet.error}", pytrace=False)
    run = _run_document(snippet.root, snippet.document)
    if snippet.kind == "writes":
        if run.writes:
            pytest.fail(
                f"{snippet.id}: snippets changed files under the document root "
                f"(they must only write to a temp dir): " + ", ".join(run.writes),
                pytrace=False,
            )
        return
    outcome = run.outcomes[snippet.line]
    if outcome.status == "missing-dep":
        pytest.skip(f"{snippet.id}: {outcome.message}")
    if outcome.status == "failed":
        pytest.fail(
            f"{snippet.id}: snippet raised {outcome.message}{_earlier_trouble(run, snippet)}",
            pytrace=False,
        )
    expected = snippet.expected
    if expected is None:
        return
    actual, wanted = _normalize(outcome.stdout), _normalize(expected.text)
    if expected.tolerant:
        ok = _numbers_close(actual, Expected(wanted, 0, expected.rtol, expected.atol))
    else:
        ok = actual == wanted
    if not ok:
        diff = "\n".join(
            difflib.unified_diff(
                wanted.splitlines(), actual.splitlines(), "expected", "printed", lineterm=""
            )
        )
        pytest.fail(
            f"{snippet.id}: printed output differs from the documented output "
            f"(marker at {snippet.document}:{expected.line}"
            f"{f', rtol={expected.rtol}, atol={expected.atol}' if expected.tolerant else ''})\n"
            f"{diff}",
            pytrace=False,
        )
