"""The compiled-program snapshot gate (issue #151): what a board would play, pinned to files.

Every registered experiment is compiled, with its experiment-contract arguments, on the shipped
demo device its contract run uses, and the compiled program (qick's own ``dump_prog()`` payload
via ``compiled_payload``) is compared with a reference under ``tests/golden/compiled/``.
A refactor near compilation is behaviour-neutral on the board exactly when this gate stays green.

References are **local and gitignored**, never committed. Generate them on ``main``:
``pytest tests/packs/test_compiled_snapshot_gate.py --update-compiled-golden``, then run this file
without the flag on your branch; any diff is a change to what is played. With no references on
disk the comparison tests skip, saying how to generate them; a partial set fails (CONTRIBUTING.md,
"Compiled-program snapshot gate").

The machinery (matrix, normalization, diff, update) lives in ``tests/_compiled_golden.py``. The
meta-tests below pin that machinery itself — the diff names what changed, the update flag writes
only changed files, nothing is written without it, no references means skip — and run always, so
a green gate cannot be a blind one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from strumento.core.device import Device
from tests import _compiled_golden as golden
from tests._paths import DEVICES, ROOT, SHIPPED_DEVICE_YAMLS
from tests.packs.test_experiment_contract import (
    ALL_REGISTERED,
    REGISTRIES,
    _contract_device_and_soccfg,
)

_UPDATE = "--update-compiled-golden"

# -- meta: the diff names what changed -----------------------------------------------------------


def _record(**program) -> dict:
    return {"experiment": "t1", "programs": [{"program": program, "acquire": {"reps": 100}}]}


def test_the_diff_names_pack_device_experiment_and_the_differing_paths():
    entry = golden.Entry("cqed", "multimode_demo", "t1")
    old = _record(gen_chs={"0": {"nqz": 1, "mixer_freq": 0.0}}, loop_dims=[100, 41])
    new = _record(gen_chs={"0": {"nqz": 2, "mixer_freq": 0.0}}, loop_dims=[100, 41])

    report = golden.diff_report(entry, old, new)

    assert "cqed/multimode_demo/t1" in report
    assert "$.programs[0].program.gen_chs.0.nqz: 1 → 2" in report
    assert "mixer_freq" not in report and "loop_dims" not in report  # only what differs


def test_the_diff_reports_added_removed_and_resized_fields():
    old = {"a": 1, "b": [1, 2, 3]}
    new = {"c": 1, "b": [1, 2, 3, 4]}

    lines = golden.json_diff(old, new)

    assert "$.a: removed (was 1)" in lines
    assert "$.c: added (now 1)" in lines
    assert "$.b: length 3 → 4" in lines
    assert "$.b[3]: added (now 4)" in lines


def test_the_diff_truncates_long_arrays_and_long_values():
    old = {"w": list(range(500)), "s": "x" * 1000}
    new = {"w": [v + 1 for v in range(500)], "s": "y" * 1000}

    lines = golden.json_diff(old, new)

    assert len(lines) <= golden.MAX_DIFF_LINES + 1
    assert lines[-1].startswith("… and ") and "more difference" in lines[-1]
    assert all(len(line) <= 3 * golden.MAX_VALUE_CHARS for line in lines)
    assert any(line.startswith("$.s: ") for line in lines)


def test_identical_records_have_no_diff():
    rec = _record(prog_list=[{"CMD": "NOP"}])
    assert golden.json_diff(rec, json.loads(json.dumps(rec))) == []


# -- meta: the update flag writes only changed files, and nothing is written without it ----------


def test_without_the_update_flag_nothing_is_ever_written(tmp_path):
    same, changed, missing = tmp_path / "same.json", tmp_path / "changed.json", tmp_path / "no.json"
    same.write_text("A\n")
    changed.write_text("old\n")
    before = {p: p.stat().st_mtime_ns for p in (same, changed)}

    assert golden.sync(same, "A\n", update=False) == "same"
    assert golden.sync(changed, "new\n", update=False) == "differs"
    assert golden.sync(missing, "new\n", update=False) == "missing"

    assert changed.read_text() == "old\n" and not missing.exists()
    assert {p: p.stat().st_mtime_ns for p in (same, changed)} == before


def test_with_the_update_flag_only_changed_files_are_rewritten(tmp_path):
    same, changed = tmp_path / "same.json", tmp_path / "changed.json"
    missing = tmp_path / "pack" / "dev" / "new.json"
    same.write_text("A\n")
    changed.write_text("old\n")
    same_mtime = same.stat().st_mtime_ns

    assert golden.sync(same, "A\n", update=True) == "same"
    assert golden.sync(changed, "new\n", update=True) == "written"
    assert golden.sync(missing, "fresh\n", update=True) == "written"

    assert same.stat().st_mtime_ns == same_mtime
    assert changed.read_text() == "new\n" and missing.read_text() == "fresh\n"


def test_the_update_flag_is_registered_and_off_by_default():
    from tests import conftest

    class _Parser:
        def __init__(self):
            self.options: dict[str, dict] = {}

        def addoption(self, name, **kw):
            self.options[name] = kw

    parser = _Parser()
    conftest.pytest_addoption(parser)

    assert parser.options[_UPDATE]["action"] == "store_true"
    assert parser.options[_UPDATE]["default"] is False
    assert "--block-qick" in parser.options  # registering ours must not displace it


# -- meta: references are local; without any, the comparison skips; a partial set fails ---------


def test_with_no_local_references_the_comparison_skips_and_says_how_to_generate_them(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(golden, "GOLDEN_DIR", tmp_path / "compiled")
    monkeypatch.setattr(golden, "UPDATED", [])  # keep the run's rewritten-list summary honest
    entry = golden.matrix()[0]

    with pytest.raises(pytest.skip.Exception) as skipped:
        _skip_without_references(update=False)
    with pytest.raises(pytest.skip.Exception):
        test_the_reference_files_are_exactly_the_matrix()

    assert not golden.references_exist()
    assert "on main" in str(skipped.value) and _UPDATE in str(skipped.value)
    _skip_without_references(update=True)  # the update run must not skip: it writes them
    assert _compare(entry, "{}\n", update=True) == "written" and golden.references_exist()


def test_a_partial_reference_set_fails_rather_than_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(golden, "GOLDEN_DIR", tmp_path / "compiled")
    present, absent = golden.matrix()[:2]
    present.path.parent.mkdir(parents=True)
    present.path.write_text("{}\n")

    _skip_without_references(update=False)  # some exist: no skip
    assert _compare(present, "{}\n", update=False) == "same"
    with pytest.raises(pytest.fail.Exception, match=absent.id):
        _compare(absent, "{}\n", update=False)
    with pytest.raises(AssertionError, match=_UPDATE):
        test_the_reference_files_are_exactly_the_matrix()


# -- meta: normalization is listed in one place, each with a reason ------------------------------


def test_every_normalization_carries_a_reason():
    for path, reason in golden.NORMALIZATIONS.items():
        assert path.startswith("$.") and reason.strip(), (path, reason)


def test_normalize_touches_only_the_listed_fields():
    rec = _record(prog_list=[{"CMD": "NOP"}], envelopes=[{"data": "AAAA"}])
    original = json.loads(json.dumps(rec))

    out = golden.normalize(rec)

    assert rec == original  # never mutates its input
    listed = set(golden.NORMALIZATIONS)
    changed = {p.split(":")[0] for p in golden.json_diff(original, out)}
    assert changed <= listed


def test_render_is_canonical_sorted_and_newline_terminated():
    text = golden.render({"b": 1, "a": {"d": 2, "c": [1.5, "x"]}})
    assert text.endswith("}\n") and text.index('"a"') < text.index('"b"')
    assert json.loads(text) == {"a": {"c": [1.5, "x"], "d": 2}, "b": 1}
    assert golden.render(json.loads(text)) == text


# -- guard: the matrix is explicit and complete ---------------------------------------------------


def test_every_registered_experiment_is_gated_or_listed_as_not_compilable():
    gated = [e.experiment for e in golden.matrix()]
    listed = list(golden.CANNOT_COMPILE)

    assert sorted(gated + listed) == ALL_REGISTERED, "each experiment exactly once"
    assert len(set(gated)) == len(gated)
    for name, reason in golden.CANNOT_COMPILE.items():
        assert reason.strip(), f"{name}: CANNOT_COMPILE entry has no reason"


def test_each_entry_is_labelled_with_its_pack_and_the_device_the_contract_builds():
    for entry in golden.matrix():
        assert entry.experiment in REGISTRIES[entry.pack]
        dev, _ = _contract_device_and_soccfg(entry.experiment)
        labelled = Device.load(DEVICES / entry.device / "device.yaml")
        assert dev.soccfg_snapshot == labelled.soccfg_snapshot, entry.id
        assert type(dev.calib) is type(labelled.calib), entry.id


def _snapshot_demos() -> set[str]:
    return {y.parent.name for y in SHIPPED_DEVICE_YAMLS if Device.load(y).soccfg_snapshot}


def test_every_shipped_demo_with_a_snapshot_is_gated_or_excused():
    used = {e.device for e in golden.matrix()}
    shipped = {y.parent.name for y in SHIPPED_DEVICE_YAMLS}

    assert used <= _snapshot_demos(), "the gate only compiles on tracked demos with a snapshot"
    assert _snapshot_demos() - used == set(golden.UNGATED_SNAPSHOT_DEVICES)
    assert set(golden.UNGATED_SNAPSHOT_DEVICES) <= shipped
    assert all(r.strip() for r in golden.UNGATED_SNAPSHOT_DEVICES.values())


@pytest.mark.qick
def test_the_not_compilable_list_is_still_true():
    """Each listed experiment must still fail to compile; once it compiles, take it off the list
    (and generate its reference) — the list can't silently outlive the failure."""
    pytest.importorskip("qick")
    stale = []
    for name in golden.CANNOT_COMPILE:
        try:
            golden.capture(golden.entry_for(name))
        except Exception:
            continue
        stale.append(name)
    assert not stale, f"these compile now; remove them from CANNOT_COMPILE: {stale}"


# -- determinism: the payload is a function of the code and the snapshot, nothing else -----------


_SUBPROCESS = """
import json, sys
sys.path.insert(0, sys.argv[1])
from tests import _compiled_golden as golden
print(json.dumps(golden.digests(golden.matrix())))
"""


@pytest.mark.qick
def test_compiling_is_byte_identical_in_process_in_a_fresh_process_and_from_another_cwd(tmp_path):
    pytest.importorskip("qick")
    entries = golden.matrix()
    first, second = golden.digests(entries), golden.digests(entries)
    assert first == second, "two in-process compiles differ"

    env = {**os.environ, "PYTHONHASHSEED": "4321", "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS, str(ROOT)],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    ).stdout
    fresh = json.loads(out.strip().splitlines()[-1])
    assert fresh == first, sorted(k for k in first if fresh.get(k) != first[k])


@pytest.mark.qick
def test_the_payload_carries_nothing_host_specific():
    """Why NORMALIZATIONS can stay this short: no path, user or library version is in it."""
    qick = pytest.importorskip("qick")
    text = "".join(golden.render(golden.capture(e)) for e in golden.matrix())
    for needle in {str(ROOT), os.path.expanduser("~"), os.getcwd(), qick.__version__}:
        assert needle not in text, needle


# -- the gate -------------------------------------------------------------------------------------


def _skip_without_references(*, update: bool) -> None:
    """No local references at all (and not generating them) → skip, saying how to make them."""
    if not update and not golden.references_exist():
        pytest.skip(golden.NO_REFERENCES)


def _compare(entry: golden.Entry, text: str, *, update: bool) -> str:
    """Sync ``entry``'s reference with ``text``; fail on a missing or differing one."""
    status = golden.sync(entry.path, text, update=update)
    if status == "written":
        golden.UPDATED.append(entry.id)
    elif status == "missing":
        pytest.fail(
            f"{entry.id}: no reference at {entry.rel} although other references exist; "
            f"regenerate the full set on main with pytest {_UPDATE}"
        )
    elif status == "differs":
        old = json.loads(entry.path.read_text(encoding="utf-8"))
        pytest.fail(golden.diff_report(entry, old, json.loads(text)), pytrace=False)
    return status


@pytest.mark.qick
@pytest.mark.parametrize("entry", golden.matrix(), ids=lambda e: e.id)
def test_compiled_program_matches_its_reference(entry, request):
    pytest.importorskip("qick")
    update = request.config.getoption(_UPDATE)
    _skip_without_references(update=update)
    _compare(entry, golden.render(golden.capture(entry)), update=update)


# Last, so an --update-compiled-golden run has written the files before this looks.
def test_the_reference_files_are_exactly_the_matrix():
    _skip_without_references(update=False)
    on_disk = {
        p.relative_to(golden.GOLDEN_DIR).as_posix() for p in golden.GOLDEN_DIR.rglob("*.json")
    }
    expected = {e.path.relative_to(golden.GOLDEN_DIR).as_posix() for e in golden.matrix()}

    assert expected - on_disk == set(), f"run pytest {_UPDATE} on main to create them"
    assert on_disk - expected == set(), "orphan reference files: delete them"
