"""The campaign events — the emitters a bring-up campaign's ledger rides on (#51, #54).

They are **generic substrate mechanism**, not bring-up semantics: each emitter carries an event
name plus a payload, validates structurally at append (a malformed append is rejected before
anything reaches ``progress.jsonl`` — never discovered at read time), and the events *mean*
something only to the orchestration layer that reads them (the runner slice; this slice does
not interpret). These tests pin that mechanism: round-trips, append-time rejection of
malformed payloads, crash-safe re-derivation by a fresh process, and the old-reader opacity
rule — unknown ``ev`` values and unknown manifest ``kind`` values are skipped, never fatal.
The freshness engine (#54) adds the seventh emitter (``poison`` — excursion poisoning's
durable home); a reader that has never heard of it skips it, exactly as the
forward-compatibility axis promises.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from strumento.core.record import (
    MANIFEST,
    RecordError,
    TaskRecord,
    derive_state,
    list_tasks,
    read_events,
    read_manifest,
)

MCP_PY = Path(__file__).parents[2] / "strumento" / "mcp.py"


def _events(rec_dir, ev=None):
    out = read_events(rec_dir)
    return [e for e in out if ev is None or e.get("ev") == ev]


# -- mint + resume (the campaign ledger is a task record) --------------------------------------


def test_bringup_record_mints_and_resumes(tmp_path, closing):
    """A task record minted with kind="bringup" resumes correctly: the campaign events append
    to the same event stream, and the manifest — written once at creation — is never
    rewritten by a resume or by any campaign append."""
    rec = closing(TaskRecord.create(
        tmp_path / "2026-08-29-cqed-bringup-a1b2", title="CQED bring-up",
        kind="bringup", pid=0, heartbeat=False,
    ))
    manifest_before = (rec.dir / MANIFEST).read_text()
    rec.close()

    resumed = closing(TaskRecord.open(rec.dir))      # a fresh writer: a restarted runner
    resumed.claim("multimode_demo", "main")

    got = _events(rec.dir, "claim")
    assert len(got) == 1
    assert got[0]["ev"] == "claim" and got[0]["t"].endswith("Z")
    assert got[0]["device"] == "multimode_demo" and got[0]["ref"] == "main"
    assert (rec.dir / MANIFEST).read_text() == manifest_before   # the manifest never moved
    assert read_manifest(rec.dir).kind == "bringup"


# -- malformed appends are refused at the emitter, never discovered at read --------------------


@pytest.mark.parametrize("bad", [
    {"device": "", "ref": "main"},
    {"device": "  ", "ref": "main"},
    {"device": "multimode_demo", "ref": ""},
    {"device": "multimode_demo", "ref": "  "},
])
def test_malformed_claim_is_rejected_at_append(tmp_path, bad):
    """The same validation rigor as the existing kinds: an empty payload field is a typed
    error raised by the emitter, and the event stream is untouched by the refusal."""
    rec = TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False)
    with pytest.raises(RecordError, match="claim"):
        rec.claim(**bad)
    assert read_events(rec.dir) == []              # nothing landed — rejection leaves no trace


# -- step attempts ------------------------------------------------------------------------------


def test_step_open_and_step_close_round_trip(tmp_path, closing):
    """A step attempt brackets a child task record: step_open before the attempt runs (the
    authoritative child mapping), step_close after it, with the outcome it produced."""
    rec = closing(TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False))
    rec.step_open("resonator-find", 1, "steps/2026-08-29-res-find-1-c1d2")
    rec.step_close("resonator-find", 1, "pass")

    opened, closed = _events(rec.dir, "step_open"), _events(rec.dir, "step_close")
    assert (opened[0]["step"], opened[0]["attempt"]) == ("resonator-find", 1)
    assert opened[0]["child_path"] == "steps/2026-08-29-res-find-1-c1d2"
    assert (closed[0]["step"], closed[0]["attempt"], closed[0]["outcome"]) == (
        "resonator-find", 1, "pass"
    )
    assert all(e["t"].endswith("Z") for e in opened + closed)


def test_step_open_accepts_a_path_object_for_the_child(tmp_path, closing):
    rec = closing(TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False))
    child = Path("steps/2026-08-29-punch-2-e3f4")   # recorded as given — the runner owns layout
    rec.step_open("punchout", 2, child)
    assert _events(rec.dir, "step_open")[0]["child_path"] == "steps/2026-08-29-punch-2-e3f4"


@pytest.mark.parametrize("bad", [
    ("", 1, "steps/child"),                       # empty step
    ("  ", 1, "steps/child"),
    ("resonator-find", 0, "steps/child"),          # attempts are 1-based
    ("resonator-find", -1, "steps/child"),
    ("resonator-find", "one", "steps/child"),      # attempt must be an integer
    ("resonator-find", 1, ""),                     # empty child mapping
    ("resonator-find", 1, "  "),
])
def test_malformed_step_open_is_rejected_at_append(tmp_path, bad):
    rec = TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False)
    with pytest.raises(RecordError, match="step_open"):
        rec.step_open(*bad)
    assert read_events(rec.dir) == []


@pytest.mark.parametrize("bad", [
    ("", 1, "pass"),
    ("resonator-find", 0, "pass"),
    ("resonator-find", 1, ""),
    ("resonator-find", 1, "  "),
])
def test_malformed_step_close_is_rejected_at_append(tmp_path, bad):
    rec = TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False)
    with pytest.raises(RecordError, match="step_close"):
        rec.step_close(*bad)
    assert read_events(rec.dir) == []


# -- parks, approvals, resumes -------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["propose", "needs_human", "sensor-drift-hold"])
def test_park_round_trip_and_its_kind_is_an_open_string(tmp_path, kind, closing):
    """park.kind is a NON-EMPTY OPEN string. Core may interpret `propose` and `needs_human`,
    but an unknown park kind is recorded verbatim — on derivation it HOLDS the campaign as
    parked-unresolved (the runner's rule); the substrate's job is only to keep it intact."""
    rec = closing(TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False))
    rec.park(kind, "no cleared envelope for readout.max_power")
    got = _events(rec.dir, "park")
    assert got[0]["kind"] == kind
    assert got[0]["reason"] == "no cleared envelope for readout.max_power"
    assert got[0]["t"].endswith("Z")


@pytest.mark.parametrize("bad_kind,bad_reason", [("", "why"), ("  ", "why"), ("propose", "")])
def test_malformed_park_is_rejected_at_append(tmp_path, bad_kind, bad_reason):
    rec = TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False)
    with pytest.raises(RecordError, match="park"):
        rec.park(bad_kind, bad_reason)
    assert read_events(rec.dir) == []


def test_propose_park_round_trips_its_staged_proposal_payload(tmp_path, closing):
    """Autonomy layer: a propose park carries its staged proposal's linkage — the step, the
    operator-addressable proposal_ref, the per-channel envelope it would clear once
    approved, the declared region, the bounded writeback paths, the child record the
    proposal is banked in, and the config stamp it was compiled against. All of it
    round-trips verbatim; the optional fields of a plain park stay absent."""
    rec = closing(TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False))
    rec.park(
        "propose",
        "channel 'rf' applies 0.05 with no cleared envelope",
        step="punchout",
        proposal_ref="punchout:1@abcd1234ef56",
        envelope={"rf": 0.05, "lo": 0.0},
        region="stability.map",
        bounds=["readout.max_power"],
        child_path="steps/2026-08-30-punchout-1-a1b2",
        config_content_id="c0ffee1234",
    )
    got = _events(rec.dir, "park")[0]
    assert got["kind"] == "propose"
    assert got["step"] == "punchout"
    assert got["proposal_ref"] == "punchout:1@abcd1234ef56"
    assert got["envelope"] == {"rf": 0.05, "lo": 0.0}
    assert got["region"] == "stability.map"
    assert got["bounds"] == ["readout.max_power"]
    assert got["child_path"] == "steps/2026-08-30-punchout-1-a1b2"
    assert got["config_content_id"] == "c0ffee1234"

    rec.park("needs_human", "waiting on the operator")  # a plain park stays plain
    plain = _events(rec.dir, "park")[1]
    assert "proposal_ref" not in plain and "envelope" not in plain


@pytest.mark.parametrize(
    "payload",
    [
        {"proposal_ref": "  "},
        {"step": ""},
        {"envelope": {"rf": "lots"}},
        {"envelope": {"rf": True}},
        {"envelope": "rf"},
        {"bounds": "readout.max_power"},  # a bare string is not a path list
        {"bounds": ["ok", ""]},
        {"region": "  "},
        {"child_path": ""},
        {"config_content_id": "  "},
    ],
)
def test_malformed_propose_park_payload_is_rejected_at_append(tmp_path, payload):
    """Autonomy layer: the staged-proposal payload validates structurally at append, like every
    campaign event — an empty ref, a non-numeric envelope maximum, a non-list bounds,
    or an empty child mapping never reaches the stream."""
    rec = TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False)
    with pytest.raises(RecordError, match="park"):
        rec.park("propose", "why", **payload)
    assert read_events(rec.dir) == []


def test_approve_round_trip(tmp_path, closing):
    """An approval is a privileged event: actor + channel (the authentication channel, e.g.
    `operator-cli`) + the proposal it clears, linked by proposal_ref. Only the operator CLI /
    supervisor appends these — the agent door has no path to them (asserted separately)."""
    rec = closing(TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False))
    rec.approve("operator", "operator-cli", "prop-3f2c1d")
    got = _events(rec.dir, "approve")
    assert (got[0]["actor"], got[0]["channel"], got[0]["proposal_ref"]) == (
        "operator", "operator-cli", "prop-3f2c1d"
    )
    assert got[0]["t"].endswith("Z")


@pytest.mark.parametrize("bad", [
    ("", "operator-cli", "prop-1"),
    ("operator", "", "prop-1"),
    ("operator", "operator-cli", ""),
    ("  ", "operator-cli", "prop-1"),
])
def test_malformed_approve_is_rejected_at_append(tmp_path, bad):
    rec = TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False)
    with pytest.raises(RecordError, match="approve"):
        rec.approve(*bad)
    assert read_events(rec.dir) == []


def test_resume_round_trip(tmp_path, closing):
    """A resume resolves a parked campaign: the actor plus the directive that picks the
    edge and states whether retries reset (the directive's meaning is the runner's)."""
    rec = closing(TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False))
    rec.park("needs_human", "readout saturated at every power")
    rec.resume("operator", "retry punchout with narrower span")
    got = _events(rec.dir, "resume")
    assert (got[0]["actor"], got[0]["directive"]) == ("operator", "retry punchout with narrower span")


@pytest.mark.parametrize("bad", [("", "go"), ("operator", ""), ("  ", "go"), ("operator", "  ")])
def test_malformed_resume_is_rejected_at_append(tmp_path, bad):
    rec = TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False)
    with pytest.raises(RecordError, match="resume"):
        rec.resume(*bad)
    assert read_events(rec.dir) == []


# -- crash safety: a fresh process re-derives campaign state -------------------------------------


_DERIVE_SCRIPT = textwrap.dedent(
    """
    import sys, json
    sys.path.insert(0, {repo!r})
    from strumento.core.record import derive_state, read_events

    status = derive_state(sys.argv[1])
    events = read_events(sys.argv[1])
    print(json.dumps({{
        "state": status.state,
        "n_events": status.n_events,
        "last_ev": events[-1]["ev"],
        "kinds": [e["ev"] for e in events],
    }}))
    """
)


def test_a_fresh_process_rederives_campaign_state_from_disk(tmp_path):
    """Crash-safe by construction: the campaign state — every event, in order — lives on
    disk alone, so a fresh process (sharing nothing with the writer, which is gone) derives
    it. The six campaign events all survive, all count, and the last one is what the
    derivation sees last."""
    rec = TaskRecord.create(tmp_path / "campaign", kind="bringup", pid=0, heartbeat=False)
    rec.claim("multimode_demo", "main")
    rec.step_open("resonator-find", 1, "steps/2026-08-29-res-find-1-c1d2")
    rec.step_close("resonator-find", 1, "pass")
    rec.park("propose", "qubit spec is outside cleared envelopes")
    rec.approve("operator", "operator-cli", "prop-3f2c1d")
    rec.resume("operator", "rerun with the envelope cleared")
    rec.close()                     # the writer is gone; nothing survives but the directory

    script = _DERIVE_SCRIPT.format(repo=str(Path(__file__).parents[2]))
    r = subprocess.run([sys.executable, "-c", script, str(rec.dir)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got["state"] == "queued"  # pid 0, no result.toml — derived from the record alone
    assert got["kinds"] == ["claim", "step_open", "step_close", "park", "approve", "resume"]
    assert got["n_events"] == 6 and got["last_ev"] == "resume"


def test_campaign_events_are_liveness(tmp_path, closing):
    """A campaign ledger has no `progress` loop — its events are the campaign events, and
    they keep the record alive exactly like any other event kind (liveness is the newest
    event of ANY kind; the reader knows no per-kind semantics)."""
    # our live pid
    rec = closing(TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False))
    rec.park("needs_human", "waiting on the operator")
    status = derive_state(rec.dir)
    assert status.state == "running" and status.n_events == 1
    assert status.last_event["ev"] == "park"


# -- old-reader opacity: unknown values skip, never fail ----------------------------------------


def test_readers_skip_unknown_manifest_kinds(tmp_path, closing):
    """A reader that predates the campaign events ignores unknown `kind` values: a future
    writer's manifest (a kind this version has never heard of) reads, derives, and lists
    without failing — and a bringup record is a first-class task in listings, never filtered
    by kind (the old reader knows no kinds; only the minter validates)."""
    root = tmp_path / "experiments"
    bringup = closing(TaskRecord.create(
        root / "2026-08-29-cqed-bringup-a1b2", title="bring-up", kind="bringup",
        pid=0, heartbeat=False,
    ))
    experiment = closing(TaskRecord.create(
        root / "2026-08-29-t1-monitor-b3c4", title="t1", pid=0, heartbeat=False,
    ))
    path = bringup.dir / MANIFEST                        # a future writer's axis value, patched in
    path.write_text(path.read_text().replace('kind = "bringup"', 'kind = "from-2088"'))

    man = read_manifest(bringup.dir)                     # no failure; the value survives opaquely
    assert man.kind == "from-2088"
    assert derive_state(bringup.dir).state == "queued"
    assert {s.id for s in list_tasks(root)} == {bringup.dir.name, experiment.dir.name}
    resumed = closing(TaskRecord.open(bringup.dir))  # attach-as-reader is kind-agnostic too
    resumed.heartbeat()
    assert len(_events(bringup.dir, "heartbeat")) == 1


def test_readers_skip_unknown_event_values(tmp_path, closing):
    """The reader rule: unknown `ev` values are skipped, never fatal. Campaign events in the
    stream are just events to a reader with no campaign knowledge, and a hand-appended future
    event rides alongside them without breaking the read or the derivation."""
    rec = closing(TaskRecord.create(tmp_path / "campaign", kind="bringup", pid=0, heartbeat=False))
    rec.claim("multimode_demo", "main")
    with (rec.dir / "progress.jsonl").open("a") as fh:
        fh.write(json.dumps({"t": "2030-01-01T00:00:00Z", "ev": "from-the-future", "x": 1}) + "\n")
    rec.park("propose", "staged")

    kinds = [e["ev"] for e in read_events(rec.dir)]
    assert kinds == ["claim", "from-the-future", "park"]   # returned, not dropped — consumers skip
    status = derive_state(rec.dir)
    assert status.state == "queued" and status.n_events == 3
    assert status.last_event["ev"] == "park"


# -- the agent door: no write path to campaign records -------------------------------------------


def test_the_agent_door_gains_no_campaign_write_path():
    """AC #51/5, the negative scope: the MCP surface gains NO write path to campaign records.
    The session record path may log *experiments* (progress · artifact · gate · calibration) —
    it must carry no campaign emitter call site, no campaign tool, and never mint a bringup
    record. Source-level guard, pure lane; the live 8-tool list is asserted by the
    mcp-marked suite (unchanged by this slice)."""
    import strumento.mcp as mcp_mod

    source = MCP_PY.read_text()
    for verb in ("claim", "step_open", "step_close", "park", "approve", "resume"):
        assert f".{verb}(" not in source, (
            f"strumento/mcp.py must not call the {verb}() emitter: "
            f"the agent door has no write path to campaign records"
        )
    assert "bringup" not in source          # the session record path stays kind="experiment"
    assert len(mcp_mod.TOOL_NAMES) == 8
    assert not [
        t for t in mcp_mod.TOOL_NAMES
        if any(v in t.lower() for v in ("campaign", "bringup", "claim", "park", "approve", "resume"))
    ]


# -- the poison event (excursion poisoning's durable home) ----------------------------------------


def test_poison_round_trip(tmp_path, closing):
    """``poison(paths, reason)`` records which calibration paths an excursion forced
    stale and why — the durable, crash-safe home of spatial-excursion poisoning (a fresh
    process re-derives it; the runner's derivation collects it)."""
    rec = closing(TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False))
    rec.poison(["stability.map"], "dot-formation moved outside the cleared region")
    got = _events(rec.dir, "poison")
    assert len(got) == 1
    assert got[0]["paths"] == ["stability.map"]
    assert got[0]["reason"] == "dot-formation moved outside the cleared region"

    rec.poison(["stability.map", "dot.compensation"], "a second excursion, two paths")
    got = _events(rec.dir, "poison")
    assert len(got) == 2  # append-only: both excursions are history


@pytest.mark.parametrize(
    "bad",
    [
        ([], "why"),
        ([""], "why"),
        (["ok", "  "], "why"),
        ("stability.map", "why"),  # a bare string is not a path list
        (["ok"], ""),
        (["ok"], "   "),
    ],
)
def test_malformed_poison_is_rejected_at_append(tmp_path, bad):
    """Structural validation at the emitter, like every campaign event: an empty path
    list, an empty path, a non-list, or an empty reason never reaches the stream."""
    rec = TaskRecord.create(tmp_path / "campaign", kind="bringup", heartbeat=False)
    with pytest.raises(RecordError, match="poison"):
        rec.poison(*bad)
    assert read_events(rec.dir) == []


def test_poison_events_surface_in_campaign_derivation(tmp_path, closing):
    """The runner's derivation collects poison events into ``CampaignState.poisoned`` —
    path -> the LATEST poison's timestamp (a later poison of the same path wins; a resume
    clears no poison — data validity is not a human's word)."""
    from strumento.core.bringup import BringupCampaign, BringupPlan, BringupStep

    def _mk(name):
        return BringupStep(
            name=name, experiment=lambda d, c: None, requires=(), provides=(),
            gates=("fit",), on_pass="calibrated", on_fail={1: "park"}, max_retries=1,
            envelope={"rf": 0.05},
        )

    plan = BringupPlan(steps=(_mk("a"),), entry="a")
    campaign = closing(BringupCampaign.create(tmp_path, plan, device="d", ref="main"))
    assert campaign.derive_state().poisoned == {}

    campaign.record.poison(["stability.map"], "excursion one")
    campaign.record.poison(["stability.map"], "excursion two — the later stamp wins")
    campaign.record.poison(["dot.compensation"], "excursion three")

    state = campaign.derive_state()
    assert sorted(state.poisoned) == ["dot.compensation", "stability.map"]
    from strumento.core.record import read_events

    stamps = [e["t"] for e in read_events(campaign.record.dir) if e.get("ev") == "poison"]
    assert state.poisoned["stability.map"] == stamps[1]  # the LATEST poison of the path

    campaign.record.resume("operator", "keep going, the data is fine")  # a human's word
    assert campaign.derive_state().poisoned["stability.map"] == stamps[1]  # still poisoned
