"""Simulator and hardware writes can be told apart (#194).

Three things a store write from an MCP session now says honestly:

* **where it came from** — every provenance line the session writes carries, in ``extra``,
  ``soc_kind``, ``simulated`` and (for the simulator) ``truth_hash``, the SHA-256 of the canonical
  JSON of the session's ground truth. ``extra`` because older readers build ``Provenance(**line)``.
* **whether the store is a simulation store** — a simulator or mock session marks a *fresh* store
  it creates, and refuses to write to an *existing* store without the mark (both
  ``record_calibration`` and seeding a new ref) unless it was opened with ``allow_sim_writes=True``.
* **what it is** — ``record_calibration(updates)`` is ``manual``; only
  ``record_calibration(updates, outcome=<index>)`` is ``<experiment> fit`` with that fit attached.

Markers: the sessions drive a soc, so ``mcp`` + ``qick``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402

from strumento.core.device import Device  # noqa: E402
from strumento.core.record import read_events, read_manifest  # noqa: E402
from strumento.core.store import FileConfigStore, Provenance  # noqa: E402
from strumento.core.store import record_calibration as library_record_calibration  # noqa: E402
from strumento.mcp import (  # noqa: E402
    SIM_STORE_MARKER,
    LabSession,
    SessionError,
    build_server,
    truth_hash,
)
from tests._paths import DEVICES  # noqa: E402
from tests._v2gate import requires_v2_soccfg  # noqa: E402

pytestmark = [pytest.mark.mcp, pytest.mark.qick, requires_v2_soccfg]

LOOPBACK = DEVICES / "loopback_demo" / "device.yaml"
TRUTH = {"pi_gain": 9350.0}
PI_GAIN_PATH = "qubit.pulses.pi_ge.gain"
RELAX_PATH = "readout.relax_delay"
RABI = {"gains": {"start": 0.0, "stop": 16380.0, "on": "amp"}, "points": 61,
        "readout": {"threshold": 0.5}, "reps": 500, "line": ["qubit", "drive"]}


def _canonical_hash(truth: dict) -> str:
    """The definition, restated independently of the implementation."""
    text = json.dumps(truth, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` and its bytes — what "the store is unchanged" compares."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _log_lines(root: Path, ref: str = "main") -> list[dict]:
    return [json.loads(line) for line in (root / "log" / f"{ref}.jsonl").read_text().splitlines()
            if line.strip()]


def _existing_unmarked_store(root: Path) -> str:
    """A store a human made through the library path — a lab's store, never marked."""
    store = FileConfigStore(root)
    config = Device.load(LOOPBACK).to_config()
    cid = store.put(config)
    store.append_provenance("main", Provenance(content_id=cid, source="loaded device.yaml"))
    store.set_ref("main", cid)
    library_record_calibration(store, "main", config, {RELAX_PATH: 250.0},
                               source="manual", timestamp="2026-10-01T00:00:00Z")
    return store.resolve_ref("main") or ""


@pytest.fixture
def lab():
    session = LabSession()
    try:
        yield session
    finally:
        session.close()


# -- AC1: provenance records the soc kind -----------------------------------------------------


def test_truth_hash_is_sha256_of_canonical_json():
    assert truth_hash(TRUTH) == _canonical_hash(TRUTH)
    assert truth_hash({"b": 1, "a": 2.5}) == truth_hash({"a": 2.5, "b": 1})   # key order is free
    assert truth_hash({}) == hashlib.sha256(b"{}").hexdigest()                # simulator, no truth
    assert truth_hash({"pi_gain": 9350}) != truth_hash({"pi_gain": 9351})


def test_a_simulator_write_carries_soc_kind_simulated_and_truth_hash(tmp_path, lab):
    root = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="simulator", store=str(root), sim=TRUTH)
    lab.record_calibration({RELAX_PATH: 300.0})
    genesis, write = _log_lines(root)
    for line in (genesis, write):
        assert line["extra"]["soc_kind"] == "simulator"
        assert line["extra"]["simulated"] is True
        assert line["extra"]["truth_hash"] == _canonical_hash(TRUTH)


def test_a_simulator_without_truth_still_hashes_its_empty_truth(tmp_path, lab):
    root = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="simulator", store=str(root))
    lab.record_calibration({RELAX_PATH: 300.0})
    assert _log_lines(root)[-1]["extra"]["truth_hash"] == _canonical_hash({})


def test_a_mock_write_carries_soc_kind_mock(tmp_path, lab):
    root = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="mock", store=str(root))
    lab.record_calibration({RELAX_PATH: 300.0})
    for line in _log_lines(root):
        assert line["extra"]["soc_kind"] == "mock"
        assert line["extra"]["simulated"] is True
        assert "truth_hash" not in line["extra"]     # the mock has no ground truth to hash


# -- AC2: an existing unmarked store is refused -----------------------------------------------


@pytest.mark.parametrize("soc", ["simulator", "mock"])
def test_a_simulated_session_refuses_record_calibration_on_an_unmarked_store(tmp_path, lab, soc):
    root = tmp_path / "lab-store"
    current = _existing_unmarked_store(root)
    loaded = lab.load_device(str(LOOPBACK), soc=soc, store=str(root))   # reading is fine
    assert loaded["store"]["content_id"] == current and loaded["store"]["seeded"] is False
    before = _tree(root)
    with pytest.raises(SessionError) as refused:
        lab.record_calibration({RELAX_PATH: 300.0})
    message = str(refused.value)
    assert str(root) in message
    assert "allow_sim_writes=True" in message and "fresh store" in message
    assert _tree(root) == before, "a refused write left a trace in the store"
    assert lab.device.calib.readout.relax_delay == 250.0   # the session belief did not move


def test_a_simulated_session_refuses_to_seed_a_new_ref_in_an_unmarked_store(tmp_path, lab):
    root = tmp_path / "lab-store"
    _existing_unmarked_store(root)
    lab.load_device(str(LOOPBACK), soc="simulator", store=str(tmp_path / "a"))
    previous = (lab.device, lab.soc, lab.store, lab.ref)
    before = _tree(root)
    with pytest.raises(SessionError) as refused:
        lab.load_device(str(LOOPBACK), soc="simulator", store=str(root), ref="nightly")
    assert str(root) in str(refused.value) and "allow_sim_writes=True" in str(refused.value)
    assert _tree(root) == before, "a refused seed wrote to the store"
    assert (lab.device, lab.soc, lab.store, lab.ref) == previous   # the old session stands


def test_allow_sim_writes_lifts_the_refusal_and_still_says_simulated(tmp_path, lab):
    root = tmp_path / "lab-store"
    _existing_unmarked_store(root)
    loaded = lab.load_device(str(LOOPBACK), soc="simulator", store=str(root), ref="nightly",
                             sim=TRUTH, allow_sim_writes=True)
    assert loaded["ok"] and loaded["store"]["seeded"] is True
    lab.record_calibration({RELAX_PATH: 300.0})
    for line in _log_lines(root, "nightly"):
        assert line["extra"]["simulated"] is True and line["extra"]["soc_kind"] == "simulator"
    assert not (root / SIM_STORE_MARKER).exists(), "an allowed write must not re-mark a lab store"


def test_the_refusal_comes_back_as_data_over_mcp(tmp_path):
    root = tmp_path / "lab-store"
    _existing_unmarked_store(root)
    before = _tree(root)

    async def go():
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(build_server()) as client:
            calls = [
                ("load_device", {"path": str(LOOPBACK), "store": str(root)}),
                ("record_calibration", {"updates": {RELAX_PATH: 300.0}}),
                ("load_device", {"path": str(LOOPBACK), "store": str(root), "ref": "nightly"}),
                ("load_device", {"path": str(LOOPBACK), "store": str(root),
                                 "allow_sim_writes": True}),
                ("record_calibration", {"updates": {RELAX_PATH: 300.0}}),
            ]
            out = []
            for name, args in calls:
                result = await client.call_tool(name, args)
                assert not result.isError, result
                out.append(json.loads(result.content[0].text))
            return out

    opened, refused_write, refused_seed, allowed, written = anyio.run(go)
    assert opened["ok"]
    assert refused_write["ok"] is False and str(root) in refused_write["error"]
    assert refused_seed["ok"] is False and str(root) in refused_seed["error"]
    assert allowed["ok"] and written["ok"]
    assert len(_log_lines(root)) == 3        # the two lab lines + the one allowed write
    assert set(before) <= set(_tree(root))


# -- AC3: a fresh store is marked ---------------------------------------------------------------


@pytest.mark.parametrize("precreate", [False, True], ids=["absent", "empty-dirs"])
def test_a_fresh_store_is_marked_and_its_writes_succeed(tmp_path, lab, precreate):
    root = tmp_path / "store"
    if precreate:
        FileConfigStore(root)          # empty objects/ refs/ log/ — still a fresh store
    loaded = lab.load_device(str(LOOPBACK), soc="simulator", store=str(root))
    assert loaded["store"]["seeded"] is True
    marker = json.loads((root / SIM_STORE_MARKER).read_text())
    assert marker["simulated"] is True and marker["soc_kind"] == "simulator"
    assert lab.record_calibration({RELAX_PATH: 300.0})["ok"]
    # a later session (mock or simulator) keeps writing to the store it marked
    lab.load_device(str(LOOPBACK), soc="mock", store=str(root))
    assert lab.record_calibration({RELAX_PATH: 310.0})["ok"]


def test_a_refused_load_does_not_mark_a_fresh_store(tmp_path, lab):
    root = tmp_path / "store"
    with pytest.raises(SessionError):
        lab.load_device(str(LOOPBACK), soc="mock", store=str(root), sim=TRUTH)   # mock + truth
    assert not (root / SIM_STORE_MARKER).exists()


# -- AC4: writes are labelled honestly ----------------------------------------------------------


def test_a_write_without_outcome_is_manual_and_one_with_outcome_carries_that_fit(tmp_path, lab):
    root = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="simulator", store=str(root), sim=TRUTH)
    run = lab.run_experiment("amplitude_rabi", RABI)
    assert run["fit"]["success"]

    lab.record_calibration({RELAX_PATH: 300.0})
    manual = _log_lines(root)[-1]
    assert manual["source"] == "manual"
    assert "fit" not in manual["extra"]

    lab.record_calibration({PI_GAIN_PATH: run["fit"]["derived"]["pi_gain"]}, outcome=run["index"])
    fitted = _log_lines(root)[-1]
    assert fitted["source"] == "amplitude_rabi fit"
    assert fitted["extra"]["fit"] == run["fit"]["derived"]
    assert fitted["extra"]["simulated"] is True


@pytest.mark.parametrize("outcome", [1, -1, True, "0"], ids=["past-end", "negative", "bool", "str"])
def test_an_unknown_outcome_is_refused_before_the_store(tmp_path, lab, outcome):
    root = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="simulator", store=str(root), sim=TRUTH)
    lab.run_experiment("amplitude_rabi", RABI)            # index 0 is the only outcome
    before = _tree(root)
    with pytest.raises(SessionError, match="outcome"):
        lab.record_calibration({RELAX_PATH: 300.0}, outcome=outcome)
    assert _tree(root) == before


def test_an_outcome_without_a_fit_cannot_label_a_write(tmp_path, lab):
    root = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="simulator", store=str(root))
    lab.outcomes.append({"index": 0, "experiment": "unfitted", "fit": None})
    before = _tree(root)
    with pytest.raises(SessionError, match="no fit"):
        lab.record_calibration({RELAX_PATH: 300.0}, outcome=0)
    assert _tree(root) == before


def test_outcome_indices_stay_unique_past_the_memory_cap(tmp_path, lab):
    """``index`` is what ``outcome=`` names, so it must never repeat inside one session."""
    from strumento.mcp import MAX_OUTCOMES

    lab.load_device(str(LOOPBACK), soc="simulator")
    for i in range(MAX_OUTCOMES + 2):
        lab._remember("fake", {}, {}, _FakeOutcome())
    indices = [o["index"] for o in lab.outcomes]
    assert indices == list(range(2, MAX_OUTCOMES + 2))


class _FakeResult:
    xpts = None
    avgi = avgq = shots = None
    heralds: list = []
    meta: dict = {}


class _FakeOutcome:
    result = _FakeResult()
    fit = None


# -- AC5: history shows the fields; old lines still read ---------------------------------------


def test_calibration_history_shows_the_fields_and_reads_old_lines(tmp_path, lab):
    root = tmp_path / "store"
    lab.load_device(str(LOOPBACK), soc="simulator", store=str(root), sim=TRUTH)
    log = root / "log" / "main.jsonl"
    genesis = _log_lines(root)[0]["content_id"]
    # two lines as an older strumento wrote them: one with empty extra, one with no extra key
    with log.open("a") as fh:
        fh.write(json.dumps({"content_id": genesis, "parent_id": genesis, "source": "manual",
                             "updates": {}, "timestamp": "2026-09-01T00:00:00Z", "extra": {}}) + "\n")
        fh.write(json.dumps({"content_id": genesis, "parent_id": genesis, "source": "old",
                             "updates": {}, "timestamp": None}) + "\n")
    lab.record_calibration({RELAX_PATH: 300.0})
    history = lab.calibration_history()["history"]
    assert [h["source"] for h in history][1:3] == ["manual", "old"]
    assert history[1]["extra"] == {} and history[2]["extra"] == {}
    assert history[-1]["extra"]["soc_kind"] == "simulator"
    assert history[-1]["extra"]["truth_hash"] == _canonical_hash(TRUTH)


# -- AC6: content ids are unchanged --------------------------------------------------------------


def test_a_session_write_hashes_exactly_as_the_library_write(tmp_path, lab):
    root = tmp_path / "store"
    seeded = lab.load_device(str(LOOPBACK), soc="simulator", store=str(root), sim=TRUTH)
    session_cid = lab.record_calibration({RELAX_PATH: 300.0})["content_id"]

    plain = FileConfigStore(tmp_path / "plain")
    config = Device.load(LOOPBACK).to_config()
    assert plain.put(config) == seeded["store"]["content_id"]
    plain.set_ref("main", config.content_id())
    library_cid, _ = library_record_calibration(plain, "main", config, {RELAX_PATH: 300.0})
    assert session_cid == library_cid


# -- the task record ------------------------------------------------------------------------------


@pytest.mark.parametrize("soc", ["simulator", "mock"])
def test_the_task_record_says_sim_and_its_calibration_event_carries_the_fields(tmp_path, lab, soc):
    lab.load_device(str(LOOPBACK), soc=soc, store=str(tmp_path / "store"),
                    sim=TRUTH if soc == "simulator" else None, task=str(tmp_path / "tasks"))
    written = lab.record_calibration({RELAX_PATH: 300.0})
    task_dir = lab.record.dir
    assert read_manifest(task_dir).target == "sim"
    event = next(e for e in read_events(task_dir) if e["ev"] == "calibration")
    assert event["content_id"] == written["content_id"]
    assert event["soc_kind"] == soc and event["simulated"] is True
    if soc == "simulator":
        assert event["truth_hash"] == _canonical_hash(TRUTH)
    else:
        assert "truth_hash" not in event
