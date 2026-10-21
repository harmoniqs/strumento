"""Public docs and code carry no internal references a new reader can't resolve (#141, #155).

The repo is kept public-ready, so three checks guard what a reader sees:

* **Internal references.** The public docs, READMEs, device YAMLs and examples, and every tracked
  text file under the package (``strumento/``) and the tests (``tests/``) -- comments and
  docstrings surface in ``help()`` and generated API docs -- never cite things that only exist
  outside the repo: plan steps ("Task 3", "Slice 6", "Phase 2", "plan D1"), sections of an
  unpublished spec ("spec §6.5", "§10.3", "§Approval"), bare plan labels ("the B3 seam",
  "the 7a shots slot", "the run path (S5)"), internal planning-document filenames,
  internal campaign names, lab-local directories (an untracked ``Experiments/`` or ``Bugs/``),
  machine-specific environment setup, or audit notes (dated or not). Say it in plain language,
  or link to something in the repo. A justified exception goes in ``ALLOWED`` with its reason.
  This file is the one exclusion by design: it defines the rules and spells examples of them.
* **Decision ids resolve.** Every ``D<n>`` in those files is a row of the design-decision index
  at the end of ``docs/architecture.md``.
* **No external lab, institution or collaborator named in any tracked file.** Methods are
  described generically, never by whose lab uses them or who suggested them, and the partner lab
  is written ``####``. The names themselves are never
  written into the repo: each lowercase alphanumeric word token of every tracked text file (and
  of every tracked path) is hashed and compared to the digests in ``LAB_NAME_SHA256S``.

Files are listed with ``git ls-files``, so an untracked local folder is never read. Pure lane:
no qick needed.
"""

from __future__ import annotations

import hashlib
import re
import subprocess

import pytest

from tests._paths import DEVICES, ROOT

# SHA-256 (hex) of each external lab's, institution's or collaborator's name in lowercase, compared
# against word tokens; the names are deliberately never spelled in the repo. Add a digest here to
# keep another name out.
LAB_NAME_SHA256S = (
    "0ede38339e19d0865df41223e1a23c179226574aaa51f0ba81a01dc1aaf3dc84",  # the partner lab
    "2547234fa27544224aef8495747a5d06ec75a98897759bd022f248ab0504b071",  # a cited multimode lab
    "de19c97d557a0e8a8cb2eb074915a22154e5f7606bd248aaf4ae6d24782f1409",  # an institution
    "1f3aec7c35c1441c1fd8eb6ad8fd31d2ecd23ecd05588145a09728a051b4d762",  # a collaborator
    "742287bd18fe61104648fa4215e2f5dd2c0f08ef9502bd967a2a6ca9f3c9483f",  # a collaborator
    "265ce76730503a448331353d6562c3d2092636a9da448c246d6366abd19bddb7",  # a cited tweezer lab
)

# (pattern name, regex). Case-insensitive unless the pattern sets its own flags.
FORBIDDEN = (
    ("plan step 'Task <n>'", r"\btask\s+\d+\b"),
    ("plan step 'Slice <n>'", r"\bslices?[\s–-]*\d+[a-z]?\b"),
    ("plan step 'Phase <n>'", r"\bphase\s+\d+\b"),
    ("plan item 'plan D<n>' / 'DRAG plan'", r"\bplan\s+D\d+\b|\bDRAG\s+plan\b"),
    ("'follows the plan'", r"\bfollow(?:s|ed|ing)?\s+the\s+plan\b"),
    ("unpublished spec section", r"\bspec\s*(?:§|R\d|D\d)"),
    ("internal planning-document filename", r"\b(?:plan|spec|method|ledger|session)-20\d{6}"),
    ("internal campaign name", r"\bcat[- ]state\s+campaign\b"),
    ("lab-local directory", r"(?-i:(?<![\w./-])(?:Experiments|Bugs)/)"),
    ("lab-local path", r"\blab-local\b|/home/|~/harmoniqs\b"),
    ("machine-specific environment", r"\bconda\s+activate\b|\bon this machine\b"),
    ("internal audit note", r"\baudit\s+note\b"),
    # Another lab's codebase, cited by its own identifiers ("MM_helper.do_thing") or as an
    # unnamed reference implementation: describe the behaviour instead.
    ("external codebase identifier 'MM_<name>'", r"(?-i:\bMM_\w+)"),
    ("unnamed external reference codebase", r"\btproc-v1\s+reference\b|\.refs-v1\b"),
    # Provenance from a predecessor codebase ("the v1 idiom", "v1's method", "ported from"): say what
    # the code does and cite the published method instead.
    (
        "predecessor-codebase provenance",
        r"(?<![\w-])v1\s+(?:[a-z]+\s+){0,2}(?:idiom|convention|reference|kernel|method|code)\b"
        r"|(?-i:(?<![\w-])v1's\s+(?!clears|only|initial))"
        r"|\bported\s+from\b|(?<!\bn)_expts\b|\bwait_after_readout\b",
    ),
    # Bring-up gate labels ("the S1 gate", "D14/S5"): name the gate instead.
    ("plan gate label 'S<n> gate'", r"(?-i:\bS\d\s+gate\b|\bD\d+/S\d\b)"),
    # Bare labels of plan items ("the B3 seam", "B3/B4's surface", "the 7a shots slot", "the run
    # path (S5)"): name the feature instead. Case-sensitive, and "B1 field" / "B0" (magnetic
    # fields), "S21", "S-T0" and T1/T2 stay legitimate physics.
    ("bring-up plan label 'B<n>'", r"(?-i:\bB[1-9]\b(?![\s-]*field))"),
    ("plan sub-step label '<n>a'", r"(?-i:(?<![\w.-])\d{1,2}[ab](?![\w-]))"),
    ("plan label '(S<n>)'", r"(?-i:\(S\d\))"),
    # "Audit fix (2026-09-13)", "audit finding, 2026-…", "audit 2026-10-07": a dated internal
    # review. Say what was wrong and what the code does now; the date means nothing to a reader.
    ("dated audit note", r"\baudit\w*\W{1,4}(?:\w+\W{1,4})?20\d\d-\d\d-\d\d"),
)

# A '§<n>' or '§Name' section reference is fine when it points at a doc in the repo (the same
# line names a '.md' file, e.g. "[getting-started §7](getting-started.md#…)"); otherwise it is a
# section of a document the reader doesn't have.
_SECTION = re.compile(r"§\s*[\dA-Za-z]")

# Justified exceptions: (repo path, exact text on the line). Keep this list tiny, and give each
# entry its reason.
ALLOWED: tuple[tuple[str, str], ...] = ()

_DECISION = re.compile(r"(?<![\w])D(\d+)(?![\w])")
_INDEX_ROW = re.compile(r"^\|\s*D(\d+)(?:\s*[–-]\s*D(\d+))?\s*\|", re.MULTILINE)


def _git_ls_files(*pathspec: str) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", *pathspec],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout to list the tracked files")
    return sorted(p for p in out.split("\0") if p and not (ROOT / p).is_symlink())


# The shipped-devices folder as a repo-relative git pathspec, from the shared path constants.
_DEVICES_REL = DEVICES.relative_to(ROOT).as_posix()


def _public_docs() -> list[str]:
    """The tracked files a reader of the public repo sees as documentation."""
    markdown = [
        p for p in _git_ls_files("*.md") if not p.startswith(("strumento/", "tests/"))
    ]
    return sorted(set(markdown + _git_ls_files(f"{_DEVICES_REL}/*.yaml", "examples/*.py")))


PUBLIC_DOCS = _public_docs()

# Text files under the package and the tests that the checks read; data files (e.g. ``.json``
# snapshots) are left out.
_TEXT_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".toml", ".txt")

# Excluded by design, not through ``ALLOWED``: this file defines the forbidden patterns, and its
# docstring and bite tests spell examples of them, so it would always flag itself.
_THIS_FILE = "tests/test_docs_hygiene.py"


def _code_files(top: str) -> list[str]:
    """The tracked text files under ``top/``, minus this file."""
    return [
        p for p in _git_ls_files(f"{top}/") if p.endswith(_TEXT_SUFFIXES) and p != _THIS_FILE
    ]


# Every scanned file, by scope: the public docs, the package, and the tests.
SCOPES = {
    "docs": PUBLIC_DOCS,
    "package": _code_files("strumento"),
    "tests": _code_files("tests"),
}


def _lines(path: str):
    text = (ROOT / path).read_text(encoding="utf-8")
    yield from enumerate(text.splitlines(), start=1)


def _allowed(path: str, line: str) -> bool:
    return any(path == p and snippet in line for p, snippet in ALLOWED)


def line_references(line: str) -> list[str]:
    """The names of the rules ``line`` breaks (empty when it is clean)."""
    names = [name for name, pattern in FORBIDDEN if re.search(pattern, line, re.IGNORECASE)]
    if _SECTION.search(line) and ".md" not in line:
        names.append("section of a document not in the repo")
    return names


def internal_references(path: str) -> list[str]:
    return [
        f"{path}:{n}: {name}: {line.strip()}"
        for n, line in _lines(path)
        if not _allowed(path, line)
        for name in line_references(line)
    ]


def decision_index() -> set[int]:
    text = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    ids: set[int] = set()
    for row in _INDEX_ROW.finditer(text):
        first = int(row[1])
        ids.update(range(first, int(row[2] or first) + 1))
    return ids


def unresolved_decisions(path: str, index: set[int]) -> list[str]:
    return [
        f"{path}:{n}: D{m[1]} is not in docs/architecture.md's design-decision index"
        for n, line in _lines(path)
        for m in _DECISION.finditer(line)
        if int(m[1]) not in index
    ]


def lab_name_hits(path: str, text: str, digests: str | tuple[str, ...]) -> list[str]:
    """Locations (``path:line``) of word tokens in ``text`` whose SHA-256 is one of ``digests``."""
    wanted = {digests} if isinstance(digests, str) else set(digests)
    hits = []
    for n, line in enumerate(text.lower().splitlines(), start=1):
        for token in sorted(set(re.findall(r"[a-z0-9]+", line))):
            if hashlib.sha256(token.encode()).hexdigest() in wanted:
                hits.append(f"{path}:{n}")
    return hits


def _tracked_text() -> list[tuple[str, str]]:
    files = []
    for path in _git_ls_files():
        data = (ROOT / path).read_bytes()
        if b"\0" in data[:8192]:
            continue  # binary
        try:
            files.append((path, data.decode("utf-8")))
        except UnicodeDecodeError:
            continue
    return files


def test_the_scanners_have_files_to_read():
    # A broken file discovery would make every check below pass vacuously.
    assert "README.md" in PUBLIC_DOCS and "docs/architecture.md" in PUBLIC_DOCS
    assert any(p.startswith(f"{_DEVICES_REL}/") for p in PUBLIC_DOCS)
    assert any(p.startswith("examples/") for p in PUBLIC_DOCS)
    assert len(_tracked_text()) > 100
    assert "strumento/__init__.py" in SCOPES["package"]
    assert "tests/conftest.py" in SCOPES["tests"] and _THIS_FILE not in SCOPES["tests"]
    assert {1, 17, 37} <= decision_index()


@pytest.mark.parametrize("scope", sorted(SCOPES))
def test_no_internal_references(scope):
    found = [hit for path in SCOPES[scope] for hit in internal_references(path)]
    assert not found, (
        "internal references a public reader can't resolve — say it in plain language or link to "
        "something in the repo:\n" + "\n".join(found)
    )


@pytest.mark.parametrize(
    "line",
    [
        "# see Task 3",
        "the slice-1 program, slices 4–6 and slice 7b",
        "# Audit fix (2026-09-13): the endpoints are validated now",
        "audit finding, 2026-09-15: the mode name was hardcoded",
        "the analyzer now refuses it (audit 2026-10-07)",
        "Under test (spec §Autonomy):",
        "per the bring-up plan's §Pack libraries",
        "see §6.6",
        "ge only in Phase 1",
        "the shared-builder invariant (DRAG plan D2)",
        "herald post-selection (spec R1(ii))",
        "the table in spec-20260823-070524-strumento-bringup-plan.md",
        "copied from Experiments/run_42",
        "a lab-local path",
        "the engine wholesale (the B3 seam)",
        "keeps the B3/B4 direct-run surface",
        "the 7a ``shots`` slot",
        "**The run path (S5).**",
        "matching the v1 reference (MM_helper.do_thing)",
        "the v2 port of the tProc-v1 reference's wait",
        "**The S1 gate**: offline fit-recovery",
        "**The off-board bridge (D14/S5).**",
        "the v1 EF readout idiom",
        "(v1's method: a weak tone)",
        "Ported from the v1 reference",
        "a wait_all(wait_after_readout) before every read",
        "from the lab_expts fitting module",
    ],
)
def test_the_rules_flag_a_synthetic_reference(line):
    # Bite test: each rule must flag a line written the way the reference appears in practice.
    assert line_references(line), line


@pytest.mark.parametrize(
    "line",
    [
        "the phase-180 pulse",
        "slice the trace into 4 blocks",
        "a sliced array: data[slice(0, 3)]",
        "timestamp='2026-07-23T00:00:00Z'",
        "the audit trail records every write",
        "[getting-started §Install](getting-started.md#install)",
        "the B-field sweep and the B0 axis",
        "the B1 field of the EDSR drive",
        "an S-T0 qubit's S21 transmission, T1 and T2",
        "a 2D map, 1e-3 tolerance, version 1.2a and hex 0x0a",
        "the storage modes S1 and S2, the cavity-S1 swap",
        "the tProc-v1 idiom of a wait_all before every read",
        "the lab-session gate and the fit-recovery gate",
        "qick's raw shape (n_ch, n_reads, n_expts, 2)",
        "v1's clears nothing; the v1 cleared region",
    ],
)
def test_the_rules_leave_ordinary_text_alone(line):
    assert line_references(line) == [], line


@pytest.mark.parametrize("scope", sorted(SCOPES))
def test_every_decision_id_resolves_to_the_architecture_index(scope):
    index = decision_index()
    found = [hit for path in SCOPES[scope] for hit in unresolved_decisions(path, index)]
    assert not found, "\n".join(found)


def test_the_lab_name_check_detects_a_matching_token():
    # Non-vacuity proof for the hash scan, using a harmless word.
    digest = hashlib.sha256(b"zanzibar").hexdigest()
    assert lab_name_hits("x.md", "line one\nthe Zanzibar_multimode device\n", digest) == ["x.md:2"]
    assert lab_name_hits("x.md", "zanzibarish", digest) == []


def test_the_lab_name_check_detects_every_digest():
    # Non-vacuity proof for a multi-digest scan: each listed name is caught on its own line, and a
    # word matching none of them is not.
    digests = (hashlib.sha256(b"zanzibar").hexdigest(), hashlib.sha256(b"timbuktu").hexdigest())
    text = "Zanzibar config\nan ordinary line\nthe Timbuktu method\nzanzibar and timbuktu\n"
    assert lab_name_hits("x.md", text, digests) == ["x.md:1", "x.md:3", "x.md:4", "x.md:4"]
    assert lab_name_hits("x.md", "samarkand", digests) == []


def test_no_tracked_file_names_an_external_lab():
    assert len(LAB_NAME_SHA256S) >= 2 and len(set(LAB_NAME_SHA256S)) == len(LAB_NAME_SHA256S)
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in LAB_NAME_SHA256S)
    found = [f"{p} (its path)" for p in _git_ls_files() if lab_name_hits(p, p, LAB_NAME_SHA256S)]
    for path, text in _tracked_text():
        found += lab_name_hits(path, text, LAB_NAME_SHA256S)
    assert not found, (
        "an external lab is named here; describe the method generically (the partner lab is "
        "written ####):\n" + "\n".join(found)
    )
