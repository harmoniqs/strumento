"""An MCP run keeps its raw data: one DataChest entry per ``run_experiment`` (#200, option B).

What a session now does with a run's measured data:

* **where it goes** — ``load_device(..., chest=<path>)`` names a
  :class:`~strumento.core.datachest.FileDataChest`; without ``chest=`` but with ``store=<root>`` the
  chest is the sibling directory ``<root>-data`` (beside the store, never inside it, so the store's
  fresh/existing rule and its object layout are untouched); with neither, nothing is saved and every
  reply says so. The directory is created on the first save, never by ``load_device``.
* **what is saved** — the run's ``MeasurementResult`` (``xpts``, ``avgi``, ``avgq``, ``shots``,
  heralds, ``meta``), the sequence that was played and the fit summary, through the chest's own
  ``save``. The entry's ``meta`` carries the #194 origin fields (``soc_kind``, ``simulated``,
  ``truth_hash``) so saved data is never mistaken for a board's.
* **who cites it** — the reply (``datachest: {saved, chest, entry}``), the task record (an
  ``artifact`` event, kind ``datachest``, for a pointer file under ``artifacts/``), and a
  calibration write labelled with that run (``extra["data"]`` on its provenance line).
* **a save that fails** costs the citation, never the run: the reply keeps its fit and says
  ``saved: false`` with the reason.

Markers: the sessions drive a soc, so ``mcp`` + ``qick``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402

from strumento.core.datachest import FileDataChest  # noqa: E402
from strumento.core.experiment import Experiment  # noqa: E402
from strumento.core.record import read_events  # noqa: E402
from strumento.core.store import Provenance  # noqa: E402
from strumento.mcp import (  # noqa: E402
    SIM_STORE_MARKER,
    TOOL_NAMES,
    LabSession,
    SessionError,
    build_server,
    coerce_params,
    experiments_for,
    store_has_content,
    truth_hash,
)
from tests._paths import DEVICES  # noqa: E402
from tests._v2gate import requires_v2_soccfg  # noqa: E402

pytestmark = [pytest.mark.mcp, pytest.mark.qick, requires_v2_soccfg]

LOOPBACK = DEVICES / "loopback_demo" / "device.yaml"
TRUTH = {"pi_gain": 9350.0}
PI_GAIN_PATH = "qubit.pulses.pi_ge.gain"
RABI = {"gains": {"start": 0.0, "stop": 16380.0, "on": "amp"}, "points": 31,
        "readout": {"threshold": 0.5}, "reps": 200, "line": ["qubit", "drive"]}


@pytest.fixture
def lab():
    session = LabSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def captured(monkeypatch):
    """Every outcome ``Experiment.go`` returns, so a test can compare the saved data to the run's."""
    outcomes: list = []
    real_go = Experiment.go

    def spy(self, *args, **kwargs):
        outcome = real_go(self, *args, **kwargs)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(Experiment, "go", spy)
    return outcomes


def _tree(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _log_lines(root: Path, ref: str = "main") -> list[dict]:
    return [json.loads(line) for line in (root / "log" / f"{ref}.jsonl").read_text().splitlines()
            if line.strip()]


def _assert_same_arrays(saved, run) -> None:
    assert np.array_equal(np.asarray(saved.xpts), np.asarray(run.xpts))
    assert np.array_equal(np.asarray(saved.avgi), np.asarray(run.avgi))
    assert np.array_equal(np.asarray(saved.avgq), np.asarray(run.avgq))
    if run.shots is None:
        assert saved.shots is None
    else:
        assert saved.shots is not None
        assert np.array_equal(np.asarray(saved.shots), np.asarray(run.shots))


# -- AC1: the entry reloads to the run's arrays, and the reply names it ------------------------


def test_a_run_with_a_chest_reloads_to_the_runs_arrays_and_the_reply_names_the_entry(
    tmp_path, lab, captured,
):
    chest = tmp_path / "chest"
    loaded = lab.load_device(str(LOOPBACK), soc="simulator", sim=TRUTH, chest=str(chest))
    assert loaded["chest"]["opened"] is True and loaded["chest"]["source"] == "argument"
    assert not chest.exists(), "load_device must not create the chest; the first save does"

    run = lab.run_experiment("amplitude_rabi", RABI)

    saved = run["datachest"]
    assert saved["saved"] is True
    assert Path(saved["chest"]) == chest.resolve()
    entry = FileDataChest(chest).load(saved["entry"])
    original = captured[-1].result
    assert original.shots is not None, "the simulator run should be shot-resolved"
    _assert_same_arrays(entry["result"], original)
    assert entry["sequence"] is not None and entry["sequence"]["ops"], "the played sequence"
    listed = FileDataChest(chest).list_entries()
    assert listed[0]["id"] == saved["entry"] and listed[0]["experiment"] == "amplitude_rabi"


def test_a_mock_run_without_shots_reloads_with_shots_none(tmp_path, lab, captured):
    chest = tmp_path / "chest"
    lab.load_device(str(LOOPBACK), soc="mock", chest=str(chest))
    run = lab.run_experiment("amplitude_rabi", RABI)
    assert run["datachest"]["saved"] is True
    entry = FileDataChest(chest).load(run["datachest"]["entry"])
    _assert_same_arrays(entry["result"], captured[-1].result)


def test_two_runs_with_identical_data_each_cite_their_own_entry(tmp_path, lab):
    # The mock returns zero IQ, so both runs' arrays are byte-identical. A content address over
    # the arrays alone would give them one entry carrying the first run's metadata, and the
    # second run's citation would resolve to the first run.
    chest = tmp_path / "chest"
    lab.load_device(str(LOOPBACK), soc="mock", chest=str(chest))
    first = lab.run_experiment("amplitude_rabi", RABI)
    second = lab.run_experiment("amplitude_rabi", {**RABI, "reps": RABI["reps"] + 100})
    a, b = first["datachest"]["entry"], second["datachest"]["entry"]
    assert a != b, "identical data from two runs must not share an entry"
    reps = {a: RABI["reps"], b: RABI["reps"] + 100}
    for run, entry in ((first, a), (second, b)):
        meta = FileDataChest(chest).load(entry)["meta"]["mcp_run"]
        assert meta["index"] == run["index"]
        assert meta["run"] == {"reps": reps[entry]}


def test_the_entry_metadata_says_where_the_data_came_from(tmp_path, lab):
    chest = tmp_path / "chest"
    lab.load_device(str(LOOPBACK), soc="simulator", sim=TRUTH, chest=str(chest))
    run = lab.run_experiment("amplitude_rabi", RABI)
    meta = FileDataChest(chest).load(run["datachest"]["entry"])["meta"]
    assert meta["soc_kind"] == "simulator"
    assert meta["simulated"] is True
    assert meta["truth_hash"] == truth_hash(TRUTH)
    assert meta["mcp_run"]["experiment"] == "amplitude_rabi"
    assert meta["mcp_run"]["index"] == run["index"]
    assert meta["mcp_run"]["run"] == {"reps": RABI["reps"]}
    assert meta["environment"] == run["environment"]


def test_a_mock_entry_has_no_truth_hash(tmp_path, lab):
    chest = tmp_path / "chest"
    lab.load_device(str(LOOPBACK), soc="mock", chest=str(chest))
    run = lab.run_experiment("amplitude_rabi", RABI)
    meta = FileDataChest(chest).load(run["datachest"]["entry"])["meta"]
    assert meta["soc_kind"] == "mock" and meta["simulated"] is True
    assert "truth_hash" not in meta


def test_the_reply_names_the_entry_over_mcp_and_the_tool_list_is_unchanged(tmp_path):
    chest = tmp_path / "chest"

    async def go():
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(build_server()) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            out = []
            for name, args in [
                ("load_device", {"path": str(LOOPBACK), "sim": TRUTH, "chest": str(chest)}),
                ("run_experiment", {"name": "amplitude_rabi", "params": RABI}),
            ]:
                result = await client.call_tool(name, args)
                assert not result.isError, result
                out.append(json.loads(result.content[0].text))
            return names, out

    names, (loaded, run) = anyio.run(go)
    assert sorted(names) == sorted(TOOL_NAMES)
    assert loaded["ok"] and loaded["chest"]["opened"] is True
    assert run["ok"] and run["datachest"]["saved"] is True
    assert (chest / f"{run['datachest']['entry']}.npz").is_file()


# -- AC2: refitting the reloaded result reproduces the reply's fit -----------------------------


def test_refitting_the_reloaded_result_reproduces_the_replys_fit(tmp_path, lab):
    chest = tmp_path / "chest"
    lab.load_device(str(LOOPBACK), soc="simulator", sim=TRUTH, chest=str(chest))
    run = lab.run_experiment("amplitude_rabi", RABI)
    assert run["fit"]["success"]

    reloaded = FileDataChest(chest).load(run["datachest"]["entry"])["result"]
    cls = experiments_for(lab.device)["amplitude_rabi"]
    given = {k: v for k, v in RABI.items() if k != "reps"}
    refit = cls(lab.device, **coerce_params(cls, given)).analyze(reloaded)

    assert set(refit.params) == set(run["fit"]["params"])
    for key, value in run["fit"]["params"].items():
        assert refit.params[key] == pytest.approx(value, rel=1e-12, abs=1e-12), key
    assert refit.derived["pi_gain"] == pytest.approx(run["fit"]["derived"]["pi_gain"], rel=1e-12)


# -- the default location: beside the store -----------------------------------------------------


def test_without_chest_the_chest_lives_beside_the_store_and_leaves_the_store_alone(tmp_path, lab):
    store = tmp_path / "store"
    loaded = lab.load_device(str(LOOPBACK), soc="simulator", store=str(store), sim=TRUTH)
    beside = tmp_path / "store-data"
    assert loaded["chest"] == {"opened": True, "path": str(beside.resolve()),
                               "source": "beside-store"}
    before = _tree(store)

    run = lab.run_experiment("amplitude_rabi", RABI)

    assert run["datachest"]["saved"] is True
    assert Path(run["datachest"]["chest"]) == beside.resolve()
    assert (beside / f"{run['datachest']['entry']}.npz").is_file()
    assert _tree(store) == before, "saving a run wrote into the store"


def test_a_chest_beside_a_store_does_not_make_the_store_existing(tmp_path, lab):
    """#194's rule — fresh = no file under objects/, refs/ or log/ — sees no chest data."""
    store = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="simulator", chest=str(tmp_path / "store-data"))
    lab.run_experiment("amplitude_rabi", RABI)                   # data beside a store-to-be
    assert not store_has_content(store)

    loaded = lab.load_device(str(LOOPBACK), soc="simulator", store=str(store))
    assert loaded["store"]["seeded"] is True and loaded["store"]["simulation_mark"] is True
    assert (store / SIM_STORE_MARKER).is_file()
    lab.run_experiment("amplitude_rabi", RABI)
    assert lab.record_calibration({"readout.relax_delay": 300.0})["ok"]


def test_an_explicit_chest_wins_over_the_store_default(tmp_path, lab):
    chest = tmp_path / "elsewhere"
    loaded = lab.load_device(str(LOOPBACK), soc="simulator", store=str(tmp_path / "store"),
                             chest=str(chest))
    assert loaded["chest"]["path"] == str(chest.resolve())
    run = lab.run_experiment("amplitude_rabi", RABI)
    assert Path(run["datachest"]["chest"]) == chest.resolve()
    assert not (tmp_path / "store-data").exists()


@pytest.mark.parametrize("sub", ["objects", "refs", "log"])
def test_a_chest_inside_the_stores_own_directories_is_refused(tmp_path, lab, sub):
    store = tmp_path / "store"
    with pytest.raises(SessionError, match="chest"):
        lab.load_device(str(LOOPBACK), soc="simulator", store=str(store),
                        chest=str(store / sub / "data"))
    assert lab.device is None


def test_a_chest_path_that_is_a_file_is_refused(tmp_path, lab):
    not_a_dir = tmp_path / "chest"
    not_a_dir.write_text("not a directory\n")
    with pytest.raises(SessionError, match="chest"):
        lab.load_device(str(LOOPBACK), soc="simulator", chest=str(not_a_dir))


# -- AC3: nowhere to save ------------------------------------------------------------------------


def test_a_session_with_nowhere_to_save_keeps_working_and_says_nothing_was_saved(tmp_path, lab):
    loaded = lab.load_device(str(LOOPBACK), soc="simulator", sim=TRUTH)
    assert loaded["chest"]["opened"] is False
    assert "not saved" in loaded["chest"]["reason"]

    run = lab.run_experiment("amplitude_rabi", RABI)

    assert run["ok"] is True and run["fit"]["success"] is True
    assert run["datachest"]["saved"] is False
    assert "chest=" in run["datachest"]["reason"] and "not saved" in run["datachest"]["reason"]
    assert "entry" not in run["datachest"]
    assert lab.device_info()["chest"]["opened"] is False
    assert list(tmp_path.iterdir()) == []


def test_a_failed_save_keeps_the_fit_and_says_why(tmp_path, lab, monkeypatch):
    lab.load_device(str(LOOPBACK), soc="simulator", sim=TRUTH, chest=str(tmp_path / "chest"))

    def refuse(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(FileDataChest, "save", refuse)
    run = lab.run_experiment("amplitude_rabi", RABI)

    assert run["ok"] is True and run["fit"]["success"] is True
    assert run["datachest"]["saved"] is False
    assert "OSError" in run["datachest"]["reason"] and "disk full" in run["datachest"]["reason"]
    assert lab.outcomes[-1]["datachest"]["saved"] is False


# -- AC4: a calibration write cites the run's entry ---------------------------------------------


def test_a_write_labelled_with_an_outcome_cites_its_entry_in_provenance(tmp_path, lab):
    store = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="simulator", store=str(store), sim=TRUTH)
    run = lab.run_experiment("amplitude_rabi", RABI)
    lab.record_calibration({PI_GAIN_PATH: run["fit"]["derived"]["pi_gain"]}, outcome=run["index"])

    line = _log_lines(store)[-1]
    assert line["source"] == "amplitude_rabi fit"
    assert line["extra"]["data"] == {"saved": True, "chest": run["datachest"]["chest"],
                                     "entry": run["datachest"]["entry"]}
    assert line["extra"]["fit"] == run["fit"]["derived"]               # #194's label still there
    assert Provenance(**line).extra["data"]["entry"] == run["datachest"]["entry"]   # old readers
    history = lab.calibration_history()["history"]
    assert history[-1]["extra"]["data"]["entry"] == run["datachest"]["entry"]
    entry = FileDataChest(Path(line["extra"]["data"]["chest"])).load(line["extra"]["data"]["entry"])
    assert entry["result"].avgi is not None


def test_a_manual_write_cites_no_data_and_older_lines_still_read(tmp_path, lab):
    store = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="simulator", store=str(store))
    lab.run_experiment("amplitude_rabi", RABI)
    lab.record_calibration({"readout.relax_delay": 300.0})
    genesis, manual = _log_lines(store)
    assert "data" not in manual["extra"] and "data" not in genesis["extra"]


def test_a_write_labelled_with_an_unsaved_run_says_so(tmp_path, lab, monkeypatch):
    store = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="simulator", store=str(store), sim=TRUTH)
    monkeypatch.setattr(FileDataChest, "save", lambda self, *a, **k: (_ for _ in ()).throw(
        OSError("read-only file system")))
    run = lab.run_experiment("amplitude_rabi", RABI)
    lab.record_calibration({PI_GAIN_PATH: run["fit"]["derived"]["pi_gain"]}, outcome=run["index"])
    data = _log_lines(store)[-1]["extra"]["data"]
    assert data["saved"] is False and "read-only" in data["reason"]


# -- the task record cites the entry --------------------------------------------------------------


def test_the_task_record_cites_each_runs_entry(tmp_path, lab):
    store = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="simulator", store=str(store), sim=TRUTH,
                    task=str(tmp_path / "tasks"))
    first = lab.run_experiment("amplitude_rabi", RABI)
    second = lab.run_experiment("amplitude_rabi", RABI)
    written = lab.record_calibration({PI_GAIN_PATH: first["fit"]["derived"]["pi_gain"]},
                                     outcome=first["index"])
    task_dir = lab.record.dir

    events = read_events(task_dir)
    artifacts = [e for e in events if e["ev"] == "artifact" and e.get("kind") == "datachest"]
    assert len(artifacts) == 2
    pointers = [json.loads((task_dir / e["path"]).read_text()) for e in artifacts]
    assert [p["entry"] for p in pointers] == [first["datachest"]["entry"],
                                              second["datachest"]["entry"]]
    assert all(p["chest"] == first["datachest"]["chest"] for p in pointers)
    assert [p["index"] for p in pointers] == [first["index"], second["index"]]
    assert first["datachest"]["task_artifact"] == artifacts[0]["path"]

    manifest = FileDataChest(Path(first["datachest"]["chest"])).list_entries()
    assert {m["task_id"] for m in manifest} == {lab.record.id}

    calibration = next(e for e in events if e["ev"] == "calibration")
    assert calibration["content_id"] == written["content_id"]
    assert calibration["data"]["entry"] == first["datachest"]["entry"]


# -- #192: a failed load creates no chest and keeps the previous one -----------------------------


def test_a_failed_load_with_chest_creates_nothing_and_keeps_the_previous_chest(tmp_path, lab):
    chest_a, chest_b = tmp_path / "chest-a", tmp_path / "chest-b"
    lab.load_device(str(LOOPBACK), soc="simulator", chest=str(chest_a))
    before = (lab.device, lab.soc, lab.chest)

    with pytest.raises(SessionError):
        lab.load_device(str(LOOPBACK), soc="mock", sim=TRUTH, chest=str(chest_b),
                        store=str(tmp_path / "b"))        # mock + truth: refused at the soc

    assert (lab.device, lab.soc, lab.chest) == before
    assert lab.chest is before[2]
    assert not chest_b.exists(), "a refused load_device created its chest"
    assert not (tmp_path / "b-data").exists()
    run = lab.run_experiment("amplitude_rabi", RABI)
    assert Path(run["datachest"]["chest"]) == chest_a.resolve()
    assert (chest_a / f"{run['datachest']['entry']}.npz").is_file()


@pytest.mark.parametrize("step", ["task", "seed"])
def test_a_load_failing_after_the_soc_opens_creates_no_chest(tmp_path, lab, monkeypatch, step):
    """The later #192 failure points: the task record cannot open, or the store seed fails."""
    from strumento.core.store import FileConfigStore

    chest_a = tmp_path / "chest-a"
    lab.load_device(str(LOOPBACK), soc="simulator", chest=str(chest_a))
    before = (lab.device, lab.soc, lab.chest, lab.chest_source)

    def refuse(*_args, **_kwargs):
        raise OSError("not writable")

    if step == "task":
        monkeypatch.setattr(LabSession, "_open_task", staticmethod(refuse))
    else:
        monkeypatch.setattr(FileConfigStore, "set_ref", refuse)
    with pytest.raises(OSError):
        lab.load_device(str(LOOPBACK), soc="simulator", store=str(tmp_path / "b"),
                        task=str(tmp_path / "tasks-b"))
    monkeypatch.undo()

    assert (lab.device, lab.soc, lab.chest, lab.chest_source) == before
    assert not (tmp_path / "b-data").exists(), "a failed load_device created its chest"
    assert lab.run_experiment("amplitude_rabi", RABI)["datachest"]["chest"] == str(chest_a.resolve())
