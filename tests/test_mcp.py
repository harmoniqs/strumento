"""**The lab-session gate**: the stateful lab session, driven by a real in-process MCP client (D33).

Every test here talks to :mod:`strumento.mcp` through the MCP SDK's **memory transport** — a real
client session, real tool dispatch, real JSON payloads — so what is exercised is the protocol
surface an external agent gets, not the Python object behind it.

The capstone (:func:`test_the_capstone_agent_calibrates_a_simulated_device`) is the load-bearing
one: *belief → measure → write back → the belief was wrong and now it isn't.* It runs the whole
chain (pulse IR → compiler → real ``AveragerProgramV2`` → simulated acquire → readout projection →
fitter → content-addressed store → device reload) through nothing but tool calls.

Markers: the module needs the ``mcp`` extra (``importorskip``, the repo idiom — so the plain
``qick`` CI job simply skips the file); tests that open a session against the simulator or the mock
additionally carry ``qick``.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402  (an mcp SDK dependency — the sync test bodies' event-loop runner)

sys.path.insert(0, str(Path(__file__).parent))
from _v2gate import requires_v2_soccfg  # noqa: E402

from strumento.mcp import TOOL_NAMES, LabSession, build_server  # noqa: E402
from tests._paths import DEVICES  # noqa: E402

pytestmark = pytest.mark.mcp

LOOPBACK = DEVICES / "loopback_demo" / "device.yaml"
#: the v2-assembly multimode instance — a cqed device with dedicated readout wiring
MULTIMODE_V2 = DEVICES / "multimode_demo" / "device.yaml"

#: the belief in ``devices/loopback_demo`` is 8192; the simulator is told the π gain is really this.
TRUE_PI_GAIN = 9350.0

#: the dotted calibration path the loop writes back (the same grammar ``strumento store diff`` emits)
PI_GAIN_PATH = "qubit.pulses.pi_ge.gain"

#: the fit-recovery gate's Rabi scan (61 points over 30 000 int codes = 1.6 cosine periods) at its reps.
#: `line`: loopback_demo has no dedicated readout wiring line (single-generator instance) --
#: AmplitudeRabi's calibrated default (`("readout","probe")`, issue #102) has nothing to resolve
#: to there, so every call site needs the same override the other 9 pre-existing test files
#: already carry (`("qubit","drive")`, the shared generator loopback_demo actually has).
RABI = {"gains": {"start": 0.0, "stop": 16380.0, "on": "amp"}, "points": 61,
        "readout": {"threshold": 0.5}, "reps": 500, "line": ["qubit", "drive"]}


# -- the in-process client -------------------------------------------------------------------


@asynccontextmanager
async def _client():
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(build_server()) as client:
        yield client


async def _call(client, name: str, args: dict | None = None) -> dict:
    """One tool call, decoded. A tool that raises would come back ``isError`` — errors are data."""
    result = await client.call_tool(name, args or {})
    assert not result.isError, f"{name} leaked an exception instead of returning data: {result}"
    assert result.content and result.content[0].type == "text"
    return json.loads(result.content[0].text)


def _rel(got: float, want: float) -> float:
    return abs(float(got) / float(want) - 1.0)


# -- session-logic only (no board, no soc) ---------------------------------------------------


def test_the_tool_list_is_the_d33_contract_and_is_strategy_free():
    async def go():
        async with _client() as client:
            return [tool.name for tool in (await client.list_tools()).tools]

    names = anyio.run(go)
    assert sorted(names) == sorted(TOOL_NAMES)
    # PERMANENT non-goal (D33): what to measure next is the calibration-strategy layer's job,
    # upstream of this server. A tool here that answered it would make strumento a strategy.
    forbidden = ("recommend", "suggest", "next", "plan", "decide", "advise", "strategy")
    assert not [n for n in names if any(word in n.lower() for word in forbidden)]


def test_schema_mirrors_the_stateless_verb_and_unknown_names_are_data():
    async def go():
        async with _client() as client:
            return (
                await _call(client, "schema", {"what": "result"}),
                await _call(client, "schema", {"what": "nope"}),
            )

    good, bad = anyio.run(go)
    assert good["ok"] and good["contract"] == "result"
    assert "properties" in good["schema"] and "avgi" in good["schema"]["properties"]
    assert bad["ok"] is False
    assert "operator-model" in bad["error"]  # lists the valid names


def test_schema_publishes_a_packs_calibration_contract_and_missing_packs_are_data():
    async def go():
        async with _client() as client:
            return (
                await _call(client, "schema", {"what": "calibration", "pack": "cqed"}),
                await _call(client, "schema", {"what": "calibration"}),  # no device loaded
                await _call(client, "schema", {"what": "calibration", "pack": "nosuch"}),
            )

    good, no_pack, unknown = anyio.run(go)
    assert good["ok"] and good["contract"] == "calibration" and good["pack"] == "cqed"
    defs = good["schema"]["$defs"]
    assert "MHz" in defs["QubitCal"]["properties"]["f_ge"]["description"]
    assert "µs" in defs["ReadoutCal"]["properties"]["relax_delay"]["description"]
    for bad in (no_pack, unknown):
        assert bad["ok"] is False
        assert all(name in bad["error"] for name in ("cqed", "spin", "tweezer"))


def test_validate_takes_a_path_or_inline_yaml_and_reports_failures_as_data():
    async def go():
        async with _client() as client:
            return (
                await _call(client, "validate", {"yaml": str(LOOPBACK)}),
                await _call(client, "validate", {"yaml": "pack: cqed\ntopology: []\n"}),
            )

    from_path, inline = anyio.run(go)
    assert from_path["ok"] and from_path["pack"] == "cqed"
    assert "qubit" in from_path["elements"]
    assert inline["ok"] is False and "wiring" in inline["error"]


def test_every_stateful_tool_on_a_fresh_session_says_no_device_is_loaded():
    async def go():
        async with _client() as client:
            return {
                name: await _call(client, name, args)
                for name, args in (
                    ("device_info", None),
                    ("list_experiments", None),
                    ("run_experiment", {"name": "amplitude_rabi"}),
                    ("calibration_history", None),
                    ("record_calibration", {"updates": {PI_GAIN_PATH: 1.0}}),
                )
            }

    for name, payload in anyio.run(go).items():
        assert payload["ok"] is False, name
        assert "no device loaded" in payload["error"], name


def test_the_session_object_is_importable_and_starts_empty():
    """The tools are thin over :class:`LabSession`; a host without MCP can drive it directly."""
    session = LabSession()
    assert session.device is None and session.soc is None and session.store is None
    assert session.ref == "main" and session.outcomes == []


def test_the_module_imports_without_qick_and_without_the_mcp_sdk():
    """Hard rule 1 for a module that *opens* soc sessions: the soc imports are lazy, both ways.

    ``strumento.mcp`` sits above the purity boundary — it reaches the qick-facing layer only inside
    ``load_device`` — and it must also import without the optional SDK, so :class:`LabSession` is
    usable from plain Python. Both are checked in a subprocess with the module made unimportable.
    (This lives in the SDK-gated file, so it runs in the ``mcp`` CI job — which is precisely the
    lane where qick *is* installed and a stray top-level import would otherwise go unnoticed.)
    """
    import subprocess

    for blocked in ("qick", "mcp"):
        code = (
            f"import sys; sys.modules[{blocked!r}] = None; "
            "import strumento.mcp as m; s = m.LabSession(); "
            "assert s.device is None and m.TOOL_NAMES; print('ok')"
        )
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert run.returncode == 0 and "ok" in run.stdout, f"blocking {blocked}: {run.stderr}"


# -- sessions that drive a soc ----------------------------------------------------------------


@pytest.mark.qick
@requires_v2_soccfg
def test_the_capstone_agent_calibrates_a_simulated_device(tmp_path, capsys):
    """**The lab-session gate.** An agent measures a device whose belief is wrong, and fixes it.

    Every step is a tool call: open a session on the v2-assembly instance against a simulator
    holding a *perturbed* ground truth → run a Rabi → the fit recovers the truth, which is not what
    the device believes → write the fit back → the session device reloads from the new store
    version → run again → the belief-vs-truth error has collapsed, and the history shows the chain.
    """

    async def go():
        async with _client() as client:
            loaded = await _call(client, "load_device", {
                "path": str(LOOPBACK),
                "soc": "simulator",
                "store": str(tmp_path / "store"),
                "sim": {"pi_gain": TRUE_PI_GAIN},
            })
            before = await _call(client, "device_info")
            first = await _call(client, "run_experiment", {"name": "amplitude_rabi", "params": RABI})
            fitted = first["fit"]["derived"]["pi_gain"]
            written = await _call(client, "record_calibration",
                                  {"updates": {PI_GAIN_PATH: fitted}, "outcome": first["index"]})
            after = await _call(client, "device_info")
            second = await _call(client, "run_experiment", {"name": "amplitude_rabi", "params": RABI})
            history = await _call(client, "calibration_history")
            return loaded, before, first, written, after, second, history

    loaded, before, first, written, after, second, history = anyio.run(go)

    # 1. the session opened, seeded the store from the instance, and tracks the ref
    assert loaded["ok"] and loaded["soc"]["kind"] == "simulator"
    assert loaded["store"]["ref"] == "main" and loaded["store"]["seeded"] is True
    genesis = loaded["store"]["content_id"]

    # 2. the fit recovers the *truth*, which is not the device's belief
    belief_before = before["calibration_paths"][PI_GAIN_PATH]
    fitted = first["fit"]["derived"]["pi_gain"]
    assert first["ok"] and first["fit"]["success"]
    assert _rel(fitted, TRUE_PI_GAIN) < 0.02, f"fit {fitted} vs truth {TRUE_PI_GAIN}"
    assert _rel(fitted, belief_before) > 0.05, "the fit came back at the belief — truth ignored"
    assert first["environment"]["soc"]["simulated"] is True   # D34: simulated data says so

    # 3. the write-back moved the store pointer and *reloaded the session device*
    assert written["ok"] and written["content_id"] != genesis
    assert written["updates"][PI_GAIN_PATH] == [belief_before, fitted]
    belief_after = after["calibration_paths"][PI_GAIN_PATH]
    assert belief_after == fitted, "the session device did not reload from the new config"

    # 4. THE CLAIM: the error against the truth shrank
    err_before, err_after = _rel(belief_before, TRUE_PI_GAIN), _rel(belief_after, TRUE_PI_GAIN)
    with capsys.disabled():
        print(f"\n[capstone] pi-gain error vs truth: before {err_before:.4%} "
              f"-> after {err_after:.4%}  (belief {belief_before} -> {belief_after}, "
              f"truth {TRUE_PI_GAIN})")
    assert err_after < err_before / 5.0, f"error {err_before:.4%} -> {err_after:.4%}"

    # 5. and the now-calibrated belief still measures the same truth (it was not a lucky draw)
    assert _rel(second["fit"]["derived"]["pi_gain"], TRUE_PI_GAIN) < 0.02

    # 6. the provenance chain is readable through the session
    assert history["ok"] and len(history["history"]) == 2
    first_entry, second_entry = history["history"]
    assert first_entry["content_id"] == genesis
    assert second_entry["parent_id"] == genesis
    assert second_entry["content_id"] == history["current"] == written["content_id"]
    assert "amplitude_rabi" in second_entry["source"]
    assert second_entry["updates"][PI_GAIN_PATH] == [belief_before, fitted]


@pytest.mark.qick
@requires_v2_soccfg
def test_a_loaded_device_supplies_the_calibration_schemas_pack():
    async def go():
        async with _client() as client:
            await _call(client, "load_device", {"path": str(LOOPBACK), "soc": "simulator"})
            return (
                await _call(client, "schema", {"what": "calibration"}),
                await _call(client, "schema", {"what": "calibration", "pack": "spin"}),
            )

    defaulted, explicit = anyio.run(go)
    assert defaulted["ok"] and defaulted["pack"] == "cqed"  # loopback_demo is a cqed device
    assert "QubitCal" in defaulted["schema"]["$defs"]
    assert explicit["ok"] and explicit["pack"] == "spin"     # an explicit pack still wins


@pytest.mark.qick
@requires_v2_soccfg
def test_a_store_less_session_says_so_when_asked_for_memory(tmp_path):
    async def go():
        async with _client() as client:
            await _call(client, "load_device", {"path": str(LOOPBACK), "soc": "simulator"})
            return (
                await _call(client, "calibration_history"),
                await _call(client, "record_calibration", {"updates": {PI_GAIN_PATH: 9000.0}}),
            )

    for payload in anyio.run(go):
        assert payload["ok"] is False
        assert "no store opened for this session" in payload["error"]


@pytest.mark.qick
@requires_v2_soccfg
def test_list_experiments_comes_from_the_packs_registry_and_unknown_names_list_it():
    async def go():
        async with _client() as client:
            await _call(client, "load_device", {"path": str(LOOPBACK), "soc": "simulator"})
            return (
                await _call(client, "list_experiments"),
                await _call(client, "run_experiment", {"name": "rabbi"}),
            )

    listed, unknown = anyio.run(go)
    from strumento.packs import cqed

    assert listed["ok"] and listed["pack"] == "cqed"
    assert sorted(e["name"] for e in listed["experiments"]) == sorted(cqed.EXPERIMENTS)
    rabi = next(e for e in listed["experiments"] if e["name"] == "amplitude_rabi")
    assert rabi["class"] == "AmplitudeRabi" and "gains" in rabi["params"]

    assert unknown["ok"] is False
    assert "amplitude_rabi" in unknown["error"] and "rabbi" in unknown["error"]


@pytest.mark.qick
@pytest.mark.parametrize("device", [MULTIMODE_V2, DEVICES / "spin_demo" / "device.yaml"])
def test_list_experiments_says_which_experiments_need_per_shot_data(device):
    """#199: every listed experiment carries ``needs_shots``, read from the class declaration —
    so an agent knows before running which experiments an averages-only soc cannot serve."""

    async def go():
        async with _client() as client:
            await _call(client, "load_device", {"path": str(device), "soc": "mock"})
            return await _call(client, "list_experiments")

    listed = anyio.run(go)
    from strumento.packs import cqed, spin

    registry = {"cqed": cqed.EXPERIMENTS, "spin": spin.EXPERIMENTS}[listed["pack"]]
    flags = {e["name"]: e["needs_shots"] for e in listed["experiments"]}
    assert flags == {name: cls.needs_shots for name, cls in registry.items()}
    assert all(isinstance(v, bool) for v in flags.values())
    assert any(flags.values()), "no experiment flagged: the declaration never reached the listing"


@pytest.mark.qick
@requires_v2_soccfg
def test_a_shot_experiment_on_the_mock_reports_why_it_refused():
    """#199 through the protocol: the refusal's reason arrives as data, not as a bare failure."""
    from strumento.core.experiment import SHOTS_REFUSAL

    async def go():
        async with _client() as client:
            await _call(client, "load_device", {"path": str(MULTIMODE_V2), "soc": "mock"})
            return await _call(client, "run_experiment",
                               {"name": "readout_state", "params": {"reps": 8}})

    run = anyio.run(go)
    assert run["ok"] and run["fit"] is not None
    assert run["fit"]["success"] is False
    assert run["fit"]["derived"]["refused"] == SHOTS_REFUSAL
    assert "assignment_fidelity" not in run["fit"]["derived"]


@pytest.mark.qick
@requires_v2_soccfg
def test_an_experiment_with_no_fitter_returns_a_null_fit_and_a_shape_summary(monkeypatch):
    """``fit: null`` is a first-class outcome — an acquisition without an analyzer still reports."""
    from strumento.core.experiment import Experiment
    from strumento.core.pulses import Seq
    from strumento.packs import cqed

    class Unfitted(Experiment):
        """A bare pi-pulse acquisition with no analyzer."""

        def sequence(self):
            return Seq().play(self.device.qubit.ge_pi()).measure(), ()

        def analyze(self, result):
            return None

    monkeypatch.setitem(cqed.EXPERIMENTS, "unfitted", Unfitted)

    async def go():
        async with _client() as client:
            # the mock soc: no response model needed, and its zero-IQ contract is untouched
            await _call(client, "load_device", {"path": str(LOOPBACK), "soc": "mock"})
            return await _call(client, "run_experiment", {"name": "unfitted", "params": {"reps": 4}})

    payload = anyio.run(go)
    assert payload["ok"] and payload["fit"] is None
    assert payload["data"]["points"] == 1
    assert payload["data"]["avgi_shape"] == [1]
    assert payload["environment"]["source"] == "NullEnvironment"


@pytest.mark.qick
@requires_v2_soccfg
def test_a_fits_non_numeric_derived_values_arrive_intact_over_the_session():
    """D38 through the protocol (#178): a list-valued ``derived`` entry is not flattened to null.

    ``readout_state`` on the simulator has per-shot clouds, so its fit reports a
    ``confusion_matrix`` — a list — beside its numbers. The tool result must carry it as a list.
    """

    async def go():
        async with _client() as client:
            await _call(client, "load_device", {"path": str(MULTIMODE_V2), "soc": "simulator"})
            run = await _call(client, "run_experiment",
                              {"name": "readout_state", "params": {"readout": {"threshold": 0.5}}})
            info = await _call(client, "device_info")
            return run, info

    run, info = anyio.run(go)
    assert run["ok"] and run["fit"] is not None
    matrix = run["fit"]["derived"]["confusion_matrix"]
    assert isinstance(matrix, list) and len(matrix) == 4, matrix
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in matrix), matrix
    assert abs(matrix[0] + matrix[1] - 1.0) < 1e-9 and abs(matrix[2] + matrix[3] - 1.0) < 1e-9
    assert isinstance(run["fit"]["derived"]["threshold"], float)        # numbers stay numbers
    # the session's memory of the run carries it too
    assert info["recent_outcomes"][-1]["derived"]["confusion_matrix"] == matrix


#: the T1 scan's delay sweep as JSON, both shapes over the same 0 → 250 µs, 201 points: the bound
#: form (D36, a declared axis ``"t"`` + the experiment's ``points=``) and the legacy per-name dict,
#: which still coerces but constructs a deprecated Sweep — and so must warn
T1_DELAYS = {
    "bound": {"delays": {"start": 0.0, "stop": 250.0, "on": "t"}, "points": 201},
    "legacy": {"delays": {"name": "t", "start": 0.0, "stop": 250.0, "points": 201}},
}


@pytest.mark.qick
@requires_v2_soccfg
@pytest.mark.parametrize("shape", ["bound", "legacy"])
def test_run_experiment_coerces_json_params_into_the_typed_constructor_arguments(shape, caplog):
    """A JSON host has no ``Sweep`` object — the session builds one from the annotation."""

    async def go():
        async with _client() as client:
            await _call(client, "load_device", {"path": str(LOOPBACK), "soc": "simulator"})
            # the fit-recovery gate's T1 conditioning (250 µs = 3.7 e-folds, 201 points, 1000 reps) — an
            # under-conditioned scan would fail the recovery assertion for reasons that have
            # nothing to do with the coercion this test is about
            return await _call(client, "run_experiment", {
                "name": "t1",
                "params": {**T1_DELAYS[shape],
                           "readout": {"threshold": 0.5}, "reps": 1000,
                           "line": ["qubit", "drive"]},
            })

    if shape == "legacy":
        # The legacy dict still coerces (and still runs) — it just carries the D36 deprecation.
        # pytest.warns cannot see it: the MCP server handles every request inside its own
        # ``warnings.catch_warnings(record=True)`` and reports what it caught as an INFO log line
        # (``Warning: DeprecationWarning: …``). The suite's ``error:legacy per-name Sweep`` filter
        # would turn the warning into a SessionError inside that handler, so this one case relaxes
        # it locally and asserts the warning on the server's record instead.
        caplog.set_level(logging.INFO, logger="mcp.server.lowlevel.server")
        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            payload = anyio.run(go)
        assert "DeprecationWarning: legacy per-name Sweep 't'" in caplog.text
    else:
        payload = anyio.run(go)
    assert payload["ok"] and payload["fit"]["success"]
    assert payload["data"]["points"] == 201
    # the axis is AS-PLAYED (D37): 1.25 µs steps are 537.6 cycles at the testbench's 430.08 MHz
    # tProc clock; the compiled register steps top at 249.721 µs — the requested 250.0 was
    # never playable. Numeric golden against the committed snapshot.
    assert payload["data"]["xpts_last"] == pytest.approx(249.72098214285714)
    # the instance believes T1 = 100 µs and the simulator was given no perturbation, so the fit
    # recovers the belief — the "simulator tells the truth" default of `sim=None`
    assert _rel(payload["fit"]["derived"]["T1"], 100.0) < 0.05


@pytest.mark.qick
def test_a_v1_instance_cannot_back_a_simulator_session(v1_snapshot_yaml):
    async def go():
        async with _client() as client:
            return await _call(client, "load_device", {"path": str(v1_snapshot_yaml), "soc": "simulator"})

    payload = anyio.run(go)
    assert payload["ok"] is False
    assert "multimode_demo" in payload["error"]  # names the cqed exemplar


# -- the durable session: task records (load_device(task=…)) ---------------------------------


@pytest.mark.qick
@requires_v2_soccfg
def test_a_task_recorded_session_writes_the_full_loop_to_disk(tmp_path):
    """The autoresearch shape: with ``task=`` the session's runs, gates, and calibration writes
    land in a task directory — the durable record a supervisor (Amicode) reads, per the task
    substrate contract. The tool list itself does not grow: the record rides on the session.
    """
    from strumento.core.record import read_events

    async def go():
        async with _client() as client:
            loaded = await _call(client, "load_device", {
                "path": str(LOOPBACK),
                "soc": "simulator",
                "store": str(tmp_path / "store"),
                "sim": {"pi_gain": TRUE_PI_GAIN},
                "task": str(tmp_path / "tasks"),
            })
            first = await _call(client, "run_experiment", {"name": "amplitude_rabi", "params": RABI})
            written = await _call(client, "record_calibration",
                                  {"updates": {PI_GAIN_PATH: first["fit"]["derived"]["pi_gain"]}})
            return loaded, first, written

    loaded, first, written = anyio.run(go)

    # a fresh task was minted under the root, and the session says so
    assert loaded["task"]["opened"] is True
    task_dir = Path(loaded["task"]["dir"])
    assert task_dir.is_dir() and (task_dir / "task.toml").exists()
    assert loaded["task"]["id"] == task_dir.name
    assert len(TOOL_NAMES) == 8          # the record added a session option, not a ninth tool

    # the run emitted its events through the vetted emitter: acquisition + fit progress, the
    # `fit` gate, and — truth armed — the `fit-recovery` gate with the comparison attached
    events = read_events(task_dir)
    kinds = [(e["ev"], e.get("name") or e.get("label") or "") for e in events]
    assert any(t == "progress" and "acquired" in lbl for t, lbl in kinds)
    assert any(t == "progress" and "fit" in lbl for t, lbl in kinds)
    gates = {e["name"]: e for e in events if e["ev"] == "gate"}
    assert gates["fit"]["pass"] is True
    assert gates["fit-recovery"]["pass"] is True
    assert gates["fit-recovery"]["params"]["experiment"] == "AmplitudeRabi"

    # and the store write is cross-linked in the same durable log
    calibrations = [e for e in events if e["ev"] == "calibration"]
    assert [e["content_id"] for e in calibrations] == [written["content_id"]]


@pytest.mark.qick
@requires_v2_soccfg
def test_a_session_resumes_an_existing_task_dir(tmp_path):
    """``task=<dir holding a task.toml>`` resumes: events append to the same log, and the manifest
    is not rewritten (the single-writer rule) — a restarted loop keeps one history.
    """
    from strumento.core.record import TaskRecord, read_events

    prior = TaskRecord.create_under(tmp_path / "tasks", title="prior loop", device="device.yaml")
    prior_dir, manifest_before = prior.dir, (prior.dir / "task.toml").read_text()
    prior.progress(0, label="loop starting")
    n_prior = len(read_events(prior_dir))
    prior.close()

    session = LabSession()
    loaded = session.load_device(
        str(LOOPBACK), soc="simulator", task=str(prior_dir),
    )
    assert loaded["task"] == {"opened": True, "id": prior.id, "dir": str(prior_dir)}
    assert session.run_experiment("amplitude_rabi", {
        "gains": {"start": 0.0, "stop": 16000.0, "on": "amp"}, "points": 11,
        "readout": {"threshold": 0.5}, "reps": 100, "line": ["qubit", "drive"],
    })
    session.record.close()

    events = read_events(prior_dir)
    assert len(events) > n_prior                      # the same log grew — one history
    assert any(e["ev"] == "gate" and e.get("name") == "fit" for e in events)
    assert (prior_dir / "task.toml").read_text() == manifest_before   # the manifest stands


# -- a write-back is validated before the store ref moves (#191) -----------------------------

#: one update per way a write-back used to land in the store unchecked: a typo'd key, a value of
#: the wrong type, a NaN, a negative time and a negative gain code
INVALID_WRITEBACKS = {
    "typo-key": {"qubit.pulses.pi_ge.gian": 9000},
    "wrong-type": {"qubit.f_ge": "abc"},
    "nan-time": {"readout.relax_delay": float("nan")},
    "negative-time": {"readout.trig_offset": -5},
    "negative-gain": {"readout.gain": -100},
}


def _store_state(root: Path) -> tuple:
    """Everything a write leaves behind: the ref's content id, the object set and the log bytes."""
    log = root / "log" / "main.jsonl"
    return (
        (root / "refs" / "main").read_text(),
        sorted(p.name for p in (root / "objects").iterdir()),
        log.read_bytes() if log.exists() else b"",
    )


@pytest.mark.qick
@requires_v2_soccfg
@pytest.mark.parametrize("updates", list(INVALID_WRITEBACKS.values()), ids=list(INVALID_WRITEBACKS))
def test_an_invalid_write_back_is_refused_and_leaves_the_store_untouched(tmp_path, updates):
    root = tmp_path / "store"

    async def write():
        async with _client() as client:
            loaded = await _call(client, "load_device", {
                "path": str(MULTIMODE_V2), "soc": "simulator", "store": str(root),
            })
            before = await _call(client, "device_info")
            state = _store_state(root)
            written = await _call(client, "record_calibration", {"updates": updates})
            return loaded, before, state, written

    async def reopen():
        async with _client() as client:
            reloaded = await _call(client, "load_device", {
                "path": str(MULTIMODE_V2), "soc": "simulator", "store": str(root),
            })
            return reloaded, await _call(client, "device_info")

    loaded, before, state, written = anyio.run(write)
    assert loaded["ok"]
    assert written["ok"] is False, f"{updates} was accepted: {written}"
    assert next(iter(updates)) in written["error"], "the refusal does not name the field"
    assert _store_state(root) == state, "a refused write-back left a trace in the store"

    reloaded, after = anyio.run(reopen)
    assert reloaded["ok"], f"the store no longer opens after a refused write: {reloaded}"
    assert reloaded["store"]["content_id"] == loaded["store"]["content_id"]
    assert after["calibration_paths"] == before["calibration_paths"]


@pytest.mark.qick
@requires_v2_soccfg
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=repr)
def test_a_non_finite_write_back_from_python_is_refused_before_the_store(tmp_path, value):
    """JSON carries no NaN, so a host driving :class:`LabSession` directly is the path that can."""
    root = tmp_path / "store"
    lab = LabSession()
    lab.load_device(str(MULTIMODE_V2), soc="simulator", store=str(root))
    belief = lab.device.calib.readout.relax_delay
    state = _store_state(root)
    with pytest.raises(ValueError, match="relax_delay"):
        lab.record_calibration({"readout.relax_delay": value})
    assert _store_state(root) == state
    assert lab.device.calib.readout.relax_delay == belief     # the session belief is untouched


# -- an agent cannot raise its own gain ceiling (#190) ---------------------------------------

#: each per-element gain limit, set to a value the schema itself accepts (above the pack default,
#: under the hard cap) — so the refusal can only come from the session, never from the validator
CEILING_WRITES = {
    "qubit": {"qubit.max_drive_gain": 0.6},
    "readout": {"readout.max_readout_gain": 0.7},
    "mode": {"modes.cavity.max_displace_gain": 0.5},
    "mixed": {"qubit.max_drive_gain": 0.6, PI_GAIN_PATH: 30000},
    "pi-first": {PI_GAIN_PATH: 9000, "readout.max_readout_gain": 0.7},
}


def _refused_ceiling(updates: dict) -> str:
    return next(p for p in updates if p != PI_GAIN_PATH)


@pytest.mark.qick
@requires_v2_soccfg
@pytest.mark.parametrize("updates", list(CEILING_WRITES.values()), ids=list(CEILING_WRITES))
def test_a_ceiling_write_over_mcp_is_refused_whole_and_leaves_everything_untouched(
    tmp_path, updates
):
    root = tmp_path / "store"

    async def write():
        async with _client() as client:
            loaded = await _call(client, "load_device", {
                "path": str(MULTIMODE_V2), "soc": "simulator", "store": str(root),
            })
            before = await _call(client, "device_info")
            history = await _call(client, "calibration_history")
            state = _store_state(root)
            written = await _call(client, "record_calibration", {"updates": updates})
            after = await _call(client, "device_info")
            history_after = await _call(client, "calibration_history")
            return loaded, before, history, state, written, after, history_after

    loaded, before, history, state, written, after, history_after = anyio.run(write)
    assert loaded["ok"]
    assert written["ok"] is False, f"{updates} was accepted: {written}"
    path = _refused_ceiling(updates)
    assert path in written["error"], f"the refusal does not name {path}: {written['error']}"
    assert "ceiling" in written["error"].lower()
    assert _store_state(root) == state, "a refused ceiling write left a trace in the store"
    assert history_after["current"] == history["current"] == loaded["store"]["content_id"]
    assert after["calibration_paths"] == before["calibration_paths"], "the session device moved"


@pytest.mark.qick
@requires_v2_soccfg
def test_a_subtree_write_that_carries_a_ceiling_is_refused(tmp_path):
    """Replacing a whole section is a ceiling write too, when the section's ceiling changes."""
    root = tmp_path / "store"
    lab = LabSession()
    lab.load_device(str(MULTIMODE_V2), soc="simulator", store=str(root))
    section = lab.device.to_config().calibration["modes"]["cavity"]
    state = _store_state(root)
    device = lab.device
    with pytest.raises(Exception, match="modes.cavity.max_displace_gain"):
        lab.record_calibration({"modes.cavity": {**section, "max_displace_gain": 0.5}})
    assert _store_state(root) == state
    assert lab.device is device


@pytest.mark.qick
@requires_v2_soccfg
def test_an_ordinary_write_back_still_lands_next_to_the_ceiling_guard(tmp_path):
    root = tmp_path / "store"
    lab = LabSession()
    lab.load_device(str(MULTIMODE_V2), soc="simulator", store=str(root))
    section = lab.device.to_config().calibration["modes"]["cavity"]
    out = lab.record_calibration({
        PI_GAIN_PATH: 9000, "modes.cavity": {**section, "T1": 900.0},
    })
    assert out["ok"] and lab.device.calib.qubit.pulses["pi_ge"].gain == 9000
    assert lab.device.calib.modes["cavity"].T1 == 900.0
