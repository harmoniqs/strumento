"""The BringupRunner — one-step advance, verdict routing, derived state (issue #53).

The campaign events record and the plan vocabulary describes; neither *executes* a graph.
These tests pin the runner: ``advance()`` executes at most one step per call under an ephemeral
breakable lock, invokes only sanctioned runnables and routes advancement from the emitter's
**returned** outcome (never from a child ledger), and the whole campaign state — position,
attempt counts, park state, cleared envelopes, claims — re-derives from the event stream alone
so a fresh process resumes after any crash. Directory names are caches; events are authority.

The sanctioned-runnable fixture is a real ``Experiment`` subclass with a scripted fake Runner
(the injectable-Runner seam the scaffold ships for exactly this), so the runner's verdict path
exercises ``go(soc, record=child)`` — the record-arming path — end to end.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from strumento.core.bringup import (
    BringupCampaign,
    BringupError,
    BringupPlan,
    BringupRunner,
    BringupStep,
)
from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.record import now_iso
from strumento.core.result import MeasurementResult

# -- fixtures: a sanctioned experiment whose fit verdict is scripted ---------------------------


class _ScriptedRunner:
    """A fake Runner (the injectable seam): one fixed acquisition, no board, no qick."""

    def __init__(self, result: MeasurementResult) -> None:
        self.result = result

    def run(self, device, seq, sweeps, soc, **run_kw) -> MeasurementResult:
        return self.result


class ScriptedExperiment(Experiment):
    """A sanctioned runnable whose fit outcome is scripted per construction.

    The runner invokes it through the record-arming path (``go(soc, record=child)``), so the
    child ledger receives the fit progress + ``fit`` gate events the real emitter writes — and
    the verdict the graph routes on is the one ``go`` *returns*.
    """

    def __init__(self, device, *, success: bool, derived: dict | None = None) -> None:
        super().__init__(
            device,
            runner=_ScriptedRunner(
                MeasurementResult(xpts=[0.0, 1.0], avgi=[0.0, 0.0], avgq=[0.0, 0.0])
            ),
        )
        self.success = bool(success)
        self.derived = dict(derived or {})

    def sequence(self):
        return None, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return FitResult(params={}, stderr={}, derived=self.derived, success=self.success)


def _factory(success: bool):
    """A step experiment factory returning a sanctioned runnable with a scripted verdict."""

    def make(device, ctx):
        return ScriptedExperiment(device, success=success)

    return make


def _step(name: str, **overrides) -> BringupStep:
    kwargs: dict = dict(
        name=name,
        experiment=_factory(True),
        requires=(),
        provides=(),
        gates=("fit",),
        on_pass="calibrated",
        on_fail={1: "park"},
        max_retries=1,
        envelope={"rf": 0.05},
    )
    kwargs.update(overrides)
    return BringupStep(**kwargs)


def _chain_plan() -> BringupPlan:
    """a -> b -> calibrated: the minimal multi-step graph for one-step-per-call semantics."""
    return BringupPlan(
        steps=(
            _step("a", on_pass="b"),
            _step("b", on_pass="calibrated"),
        ),
        entry="a",
    )


def _campaign(tmp_path, plan, *, device="demo-device", ref="main"):
    return BringupCampaign.create(tmp_path, plan, device=device, ref=ref)


def _runner(campaign, *, device=None, **kw):
    return BringupRunner(campaign, device=device if device is not None else object(),
                         soc=SimpleNamespace(), **kw)


# -- one step per call --------------------------------------------------------------------------


def test_advance_executes_at_most_one_step_per_call(tmp_path):
    """The core contract: a single advance() runs exactly one step attempt, routes, and stops —
    the next step waits for the next call. The campaign claims its device + ref at construction
    and the derived position/attempt counts come from the events the advance wrote."""
    plan = _chain_plan()
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)

    out = runner.advance()

    assert out.step == "a" and out.attempt == 1 and out.passed is True
    assert out.detour is False and out.held == ""
    state = campaign.derive_state()
    assert state.position == "b"  # routed by the pass — not executed
    assert state.attempts["a"] == 0  # no failures since the pass
    assert state.attempts["b"] == 0  # b never ran — the map is total over the plan
    assert state.claim is not None
    assert (state.claim.device, state.claim.ref, state.claim_live) == (
        "demo-device", "main", True,
    )
    steps_dir = campaign.record.dir / "steps"
    children = [d for d in steps_dir.iterdir() if d.is_dir()]
    assert len(children) == 1  # one child record — step "a" attempt 1, nothing else ran


# -- verdict routing: the returned outcome is the only control channel -------------------------


class _Verdicts:
    """A scripted verdict sequence: each factory call consumes the next success flag."""

    def __init__(self, *sequence: bool) -> None:
        self.sequence = list(sequence)
        self.i = 0

    def next(self) -> bool:
        value = self.sequence[min(self.i, len(self.sequence) - 1)]
        self.i += 1
        return value


def test_forged_gate_event_does_not_advance(tmp_path):
    """AC: a hand-appended gate event in a child ledger does not advance the graph; only
    the emitter's returned outcome does. The child record is a *record* — the runner
    routes on what ``go`` returned, so forging a pass verdict into the ledger changes
    nothing, while the honest failing return routes the ladder exactly as it should."""

    verdicts = _Verdicts(False, True)
    plan = BringupPlan(
        steps=(
            _step("a", on_pass="b", experiment=lambda device, ctx: ScriptedExperiment(
                device, success=verdicts.next()
            )),
            _step("b", on_pass="calibrated"),
        ),
        entry="a",
    )
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)

    out = runner.advance()  # the returned outcome: fit success=False
    assert out.passed is False
    state = campaign.derive_state()
    assert state.attempts["a"] == 1
    assert state.parked is not None and state.parked.kind == "needs_human"  # rung 1 parks

    child = next((campaign.record.dir / "steps").iterdir())
    ledger = child / "progress.jsonl"
    # the emitter's honest verdict is in the child ledger…
    lines = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    assert any(e["ev"] == "gate" and e["name"] == "fit" and e["pass"] is False for e in lines)
    # …and now the forgery: a hand-appended pass verdict, same shape a real one has
    with ledger.open("a") as fh:
        fh.write(json.dumps({
            "t": "2026-08-29T00:00:00.000Z", "ev": "gate", "name": "fit", "pass": True,
            "params": {"forged": True},
        }) + "\n")

    state = campaign.derive_state()  # re-derive from disk, forgery included
    assert state.attempts["a"] == 1          # still one failed attempt
    assert state.parked is not None and state.parked.kind == "needs_human"
    assert state.position == "a"             # the graph did NOT advance

    campaign.record.resume("operator", "retry after the fixture check")
    out2 = runner.advance()  # the returned outcome NOW says pass — and only now
    assert out2.passed is True and out2.attempt == 2
    assert campaign.derive_state().position == "b"
    # the forged event is still sitting in the ledger, ignored for all time
    lines = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    assert sum(1 for e in lines if e["ev"] == "gate" and e["pass"] is True) == 1


# -- sanctioned runnables ----------------------------------------------------------------------


def test_runner_refuses_unsanctioned_runnable(tmp_path):
    """AC: a callable outside the sanctioned factories is refused with an actionable
    error — the runner names the two legal paths, and the refusal burns no attempt."""
    def rogue(device, ctx):
        return lambda soc, record=None: "verdict"  # arbitrary callable — never sanctioned

    plan = BringupPlan(
        steps=(_step("a", experiment=rogue),), entry="a"
    )
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)

    with pytest.raises(BringupError) as excinfo:
        runner.advance()
    message = str(excinfo.value)
    assert "sanctioned" in message and "Experiment" in message
    assert "register_host_bridge" in message  # actionable: the two legal paths are named
    assert "'a'" in message                   # the offending step is named

    state = campaign.derive_state()
    assert state.position == "a" and state.attempts["a"] == 0  # nothing ran, nothing burned
    steps_dir = campaign.record.dir / "steps"
    assert not steps_dir.exists() or not any(steps_dir.iterdir())


class _MockHostBridge:
    """A registered host-bridge adapter: DC/host experiments ride this seam — no ``go()``,
    verdicts returned from ``run`` itself."""

    def __init__(self, verdicts):
        self.verdicts = dict(verdicts)
        self.received_record = None

    def run(self, *, record=None, **kw):
        self.received_record = record
        return self.verdicts


def test_registered_host_bridge_adapter_is_sanctioned(tmp_path):
    """AC (sanctioned factories): a host-bridge adapter registered with core is a legal
    runnable — invoked as ``run(record=child)``, its returned verdict map routes the
    graph exactly like an Experiment's returned outcome."""
    from strumento.core import bringup

    bridge = _MockHostBridge({"demod-balance": True, "noise-floor": True})
    bringup.register_host_bridge(_MockHostBridge)
    try:
        plan = BringupPlan(
            steps=(
                _step(
                    "sensor-turn-on",
                    gates=("demod-balance", "noise-floor"),
                    experiment=lambda device, ctx: bridge,
                    on_pass="calibrated",
                ),
            ),
            entry="sensor-turn-on",
        )
        campaign = _campaign(tmp_path, plan)
        runner = _runner(campaign)

        out = runner.advance()

        assert out.step == "sensor-turn-on" and out.passed is True
        assert out.verdicts == {"demod-balance": True, "noise-floor": True}
        assert bridge.received_record is not None  # the child record was handed over
        assert bridge.received_record.manifest.kind == "experiment"
        assert campaign.derive_state().position == "sensor-turn-on"  # on_pass → calibrated
    finally:
        bringup._HOST_BRIDGES.remove(_MockHostBridge)  # the registry is global; no leaks


def test_host_bridge_verdict_map_gates_the_step(tmp_path):
    """A registered adapter whose returned verdict map misses (or falsifies) a declared
    gate fails the step — absence is a verdict, and the ladder routes it."""
    from strumento.core import bringup

    bridge = _MockHostBridge({"demod-balance": False})
    bringup.register_host_bridge(_MockHostBridge)
    try:
        plan = BringupPlan(
            steps=(
                _step(
                    "sensor-turn-on",
                    gates=("demod-balance",),
                    experiment=lambda device, ctx: bridge,
                    on_fail={1: "park"},
                ),
            ),
            entry="sensor-turn-on",
        )
        campaign = _campaign(tmp_path, plan)
        runner = _runner(campaign)

        out = runner.advance()
        assert out.passed is False
        state = campaign.derive_state()
        assert state.attempts["sensor-turn-on"] == 1
        assert state.parked is not None and state.parked.kind == "needs_human"
    finally:
        bringup._HOST_BRIDGES.remove(_MockHostBridge)


# -- crashed attempts derive as failures ---------------------------------------------------------


class _CrashingExperiment(ScriptedExperiment):
    """A sanctioned runnable that dies mid-acquisition — the killed-runner fixture. The
    crash flag is a shared box (the factory mints a fresh experiment per attempt)."""

    def __init__(self, device, *, crash: dict) -> None:
        super().__init__(device, success=True)
        self._crash = crash

    def go(self, soc, **kw):
        if self._crash.get("pending"):
            self._crash["pending"] = False
            raise RuntimeError("board transport died mid-acquisition")
        return super().go(soc, **kw)


def test_crashed_attempt_derives_as_a_failed_attempt(tmp_path):
    """AC: a crashed attempt — ``step_open`` with no ``step_close`` — derives as a failed
    attempt, routes its ladder rung, and the next advance runs the NEXT attempt number.
    Crash-safety by construction: nothing but the open attempt survives the crash."""
    crash = {"pending": True}
    plan = BringupPlan(
        steps=(
            _step(
                "a",
                on_pass="calibrated",
                on_fail={1: "a", 2: "park"},
                max_retries=2,
                experiment=lambda device, ctx: _CrashingExperiment(device, crash=crash),
            ),
        ),
        entry="a",
    )
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)

    with pytest.raises(RuntimeError, match="board transport died"):
        runner.advance()  # the attempt crashes; advance propagates

    state = campaign.derive_state()  # a fresh derivation sees the open attempt
    assert state.attempts["a"] == 1  # …and counts it as a failed attempt
    assert state.position == "a"     # rung 1 routed the retry (on_fail[1] == "a")
    assert state.parked is None

    out = runner.advance()  # the next advance resumes: attempt 2, which passes
    assert out.step == "a" and out.attempt == 2 and out.passed is True
    final = campaign.derive_state()
    assert final.attempts["a"] == 0  # the pass reset the ladder
    assert final.terminal == "calibrated"


# -- crash safety: a fresh process re-derives and resumes ----------------------------------------


def test_state_rederives_in_a_fresh_process_and_ignores_directory_names(tmp_path):
    """AC: position, attempt counts, park state, cleared envelopes, and claims all
    re-derive from the event stream in a fresh process — the writer is gone, nothing is
    remembered, and directory names (the child-dir cache) are ignored: renaming every
    child directory on disk changes nothing."""
    crash = {"pending": True}
    # step "a" crashes once, then passes; step "b" fails (parks needs_human on rung 1)
    plan = BringupPlan(
        steps=(
            _step(
                "a",
                on_pass="b",
                on_fail={1: "a", 2: "park"},
                max_retries=2,
                experiment=lambda device, ctx: _CrashingExperiment(device, crash=crash),
            ),
            _step("b", on_pass="calibrated", on_fail={1: "park"}, max_retries=1,
                  experiment=lambda device, ctx: ScriptedExperiment(device, success=False)),
        ),
        entry="a",
    )
    campaign = _campaign(tmp_path, plan, device="demo-device", ref="main")
    runner = _runner(campaign)

    with pytest.raises(RuntimeError):
        runner.advance()      # a attempt 1: crashes (open, unclosed)
    runner.advance()          # a attempt 2: passes -> position b
    runner.advance()          # b attempt 1: fails -> parks needs_human
    campaign.record.approve("operator", "operator-cli", "prop-9f2c")  # clears an envelope
    campaign.record.close()   # the writer is gone; nothing survives but the directory

    # directory names are caches, never authority — scramble them
    steps_dir = campaign.record.dir / "steps"
    for i, child in enumerate(sorted(steps_dir.iterdir())):
        child.rename(steps_dir / f"misleading-name-{i}")

    expected = campaign.derive_state()
    assert expected.position == "b"
    assert expected.attempts["a"] == 0 and expected.attempts["b"] == 1
    assert expected.parked is not None and expected.parked.kind == "needs_human"
    assert expected.cleared_envelopes == frozenset({"prop-9f2c"})
    assert expected.claim is not None and expected.claim_live

    script = textwrap.dedent(
        """
        import json, sys
        sys.path.insert(0, {repo!r})
        from strumento.core.bringup import BringupCampaign, BringupPlan, BringupStep

        def step(name, **kw):
            return BringupStep(
                name=name, experiment=lambda d, c: None, requires=(), provides=(),
                gates=("fit",), envelope={{}}, **kw
            )

        plan = BringupPlan(steps=(
            step("a", on_pass="b", on_fail={{1: "a", 2: "park"}}, max_retries=2),
            step("b", on_pass="calibrated", on_fail={{1: "park"}}, max_retries=1),
        ), entry="a")
        campaign = BringupCampaign.open(sys.argv[1], plan)
        s = campaign.derive_state()
        print(json.dumps({{
            "position": s.position,
            "attempts": dict(s.attempts),
            "parked_kind": s.parked.kind if s.parked else None,
            "cleared": sorted(s.cleared_envelopes),
            "claim": [s.claim.device, s.claim.ref, s.claim_live] if s.claim else None,
        }}))
        """
    ).format(repo=str(Path(__file__).parents[2]))
    r = subprocess.run([sys.executable, "-c", script, str(campaign.record.dir)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got == {
        "position": "b",
        "attempts": {"a": 0, "b": 1},
        "parked_kind": "needs_human",
        "cleared": ["prop-9f2c"],
        "claim": ["demo-device", "main", True],
    }


# -- claim exclusivity --------------------------------------------------------------------------


def test_second_live_campaign_on_the_same_device_or_ref_is_rejected(tmp_path):
    """AC: a campaign claims BOTH a device and a calibration ref at construction, and a
    second live campaign colliding on either is rejected loudly — the error names the
    resource and the campaign that already holds it. Non-bringup and unknown-kind task
    dirs are opaque to the scan (an experiment record claims nothing)."""
    plan = _chain_plan()
    first = BringupCampaign.create(tmp_path, plan, device="demo-device", ref="main")

    with pytest.raises(BringupError, match="demo-device") as ei:
        BringupCampaign.create(tmp_path, plan, device="demo-device", ref="backup")
    assert str(first.record.dir) in str(ei.value)  # the incumbent is named, actionably

    with pytest.raises(BringupError, match=r"ref.*main|main.*ref"):
        BringupCampaign.create(tmp_path, plan, device="spare-device", ref="main")

    # an experiment record under the same root claims nothing — no collision
    from strumento.core.record import TaskRecord

    TaskRecord.create(tmp_path / "2026-08-29-t1-monitor-0001", kind="experiment",
                      pid=0, heartbeat=False)
    # a manifest kind this scan has never heard of is opaque — skipped, never fatal
    unknown = TaskRecord.create(tmp_path / "2026-08-29-from-2088-0002", pid=0,
                                heartbeat=False)
    manifest = unknown.dir / "task.toml"
    manifest.write_text(manifest.read_text().replace('kind = "experiment"', 'kind = "from-2088"'))

    third = BringupCampaign.create(tmp_path, plan, device="spare-device", ref="backup")
    assert third.derive_state().claim_live  # both resources free -> constructs and claims


def _script_suffixes(monkeypatch, *suffixes):
    """Make the task-id suffix source yield ``suffixes`` in order (the last one repeats)."""
    import strumento.core.record as record_mod

    queue = list(suffixes)

    def fake_uuid4():
        hex4 = queue.pop(0) if len(queue) > 1 else queue[0]
        return SimpleNamespace(hex=hex4 + "0" * 28)

    monkeypatch.setattr(record_mod.uuid, "uuid4", fake_uuid4)


def test_a_colliding_campaign_id_mints_a_fresh_directory(tmp_path, monkeypatch):
    """A campaign whose minted id is already taken (another record, same day, same suffix)
    retries into a fresh directory rather than aborting on "a task record already exists"."""
    from strumento.core.record import TaskRecord, new_task_id

    incumbent = TaskRecord.create(tmp_path / new_task_id("bringup", suffix="a925"),
                                  kind="experiment", pid=0, heartbeat=False)
    _script_suffixes(monkeypatch, "a925", "b31c")
    campaign = _campaign(tmp_path, _chain_plan())
    assert campaign.record.dir.parent == tmp_path
    assert campaign.record.dir != incumbent.dir
    assert campaign.record.dir.name.endswith("-b31c")
    assert campaign.derive_state().claim_live


def test_a_colliding_step_child_id_mints_a_fresh_directory(tmp_path, monkeypatch):
    """Re-running a step on the same day mints sibling ``steps/`` dirs that differ only in
    their suffix; a repeated suffix must not abort the advance."""
    from strumento.core.record import TaskRecord, new_task_id

    campaign = _campaign(tmp_path, _chain_plan())
    taken = campaign.record.dir / "steps" / new_task_id("a", suffix="a925")
    TaskRecord.create(taken, kind="experiment", pid=0, heartbeat=False)  # a sibling attempt
    _script_suffixes(monkeypatch, "a925", "b31c")
    out = _runner(campaign).advance()
    assert out.passed is True
    children = sorted(d.name for d in (campaign.record.dir / "steps").iterdir())
    assert children == sorted([taken.name, new_task_id("a", suffix="b31c")])


def test_a_terminal_campaign_state_releases_the_claim(tmp_path):
    """AC: a terminal campaign state releases the claim — once the incumbent's record is
    terminal, the same device + ref constructs again."""
    plan = _chain_plan()
    first = BringupCampaign.create(tmp_path, plan, device="demo-device", ref="main")
    first.record.done(summary="calibrated")
    assert first.derive_state().claim_live is False

    second = BringupCampaign.create(tmp_path, plan, device="demo-device", ref="main")
    assert second.derive_state().claim_live is True


def test_open_reattaches_without_reclaiming(tmp_path):
    """Re-attaching an existing campaign (the fresh-process resume path) emits no second
    claim — the stream carries exactly one, from construction."""
    plan = _chain_plan()
    first = BringupCampaign.create(tmp_path, plan, device="demo-device", ref="main")
    first.record.close()

    resumed = BringupCampaign.open(first.record.dir, plan)
    resumed.record.close()

    from strumento.core.record import read_events

    claims = [e for e in read_events(first.record.dir) if e.get("ev") == "claim"]
    assert len(claims) == 1


# -- approvals: valid ones clear, forged ones are ignored ----------------------------------------


def test_an_approve_missing_actor_or_channel_is_ignored_at_derivation(tmp_path):
    """AC: an approve event missing actor or channel is ignored at derivation — the
    emitters refuse such appends, so these are hand-forged lines in the stream. An
    ignored approve clears no envelope and no propose park; a valid one clears both.
    And the one execution path: an approve only appends an event — nothing executes
    until the next advance()."""
    plan = _chain_plan()
    campaign = _campaign(tmp_path, plan)
    campaign.record.park("propose", "readout.max_power is not cleared")

    ledger = campaign.record.dir / "progress.jsonl"
    with ledger.open("a") as fh:  # forgeries the emitter would have refused
        fh.write(json.dumps({
            "t": "2026-08-29T00:00:00.000Z", "ev": "approve",
            "channel": "operator-cli", "proposal_ref": "prop-forged-1",
        }) + "\n")
        fh.write(json.dumps({
            "t": "2026-08-29T00:00:00.001Z", "ev": "approve",
            "actor": "operator", "proposal_ref": "prop-forged-2",
        }) + "\n")

    state = campaign.derive_state()
    assert state.cleared_envelopes == frozenset()          # nothing was cleared
    assert state.parked is not None and state.parked.kind == "propose"  # the park held
    assert len(state.pending_approvals) == 1               # and is still pending

    runner = _runner(campaign)
    out = runner.advance()  # a forged approval appended nothing — still held
    assert out.held == "parked:propose" and out.step is None
    assert campaign.derive_state().position == "a"

    campaign.record.approve("operator", "operator-cli", "prop-3f2c")  # the valid one
    state = campaign.derive_state()
    assert state.cleared_envelopes == frozenset({"prop-3f2c"})
    assert state.parked is None and state.pending_approvals == ()

    # one execution path: the approve appended an event and executed nothing…
    from strumento.core.record import read_events

    opens = [e for e in read_events(campaign.record.dir) if e.get("ev") == "step_open"]
    assert opens == []
    out = runner.advance()  # …the next advance executes
    assert out.step == "a" and out.passed is True


# -- unknown park kinds hold (parked-unresolved) --------------------------------------------------


def test_unknown_park_kind_holds_the_campaign_until_resume(tmp_path):
    """AC: an unknown park kind HOLDS the campaign — derived as parked-unresolved, an
    opaque kind is never released, and only a ``resume`` clears it (a valid approve
    clears propose parks, not unknown ones)."""
    plan = _chain_plan()
    campaign = _campaign(tmp_path, plan)
    campaign.record.park("vendor-hold", "the vendor's interlock wants a firmware word")

    state = campaign.derive_state()
    assert state.parked is not None
    assert state.parked.kind == "vendor-hold" and state.parked.unresolved is True

    runner = _runner(campaign)
    out = runner.advance()
    assert out.held == "parked-unresolved:vendor-hold"
    assert out.step is None and out.passed is None

    campaign.record.approve("operator", "operator-cli", "prop-irrelevant")
    assert campaign.derive_state().parked is not None  # an approve is not a resume

    out = runner.advance()
    assert out.held == "parked-unresolved:vendor-hold"  # still holding
    assert campaign.derive_state().position == "a"      # the graph never moved

    campaign.record.resume("operator", "firmware word accepted, continue")
    assert campaign.derive_state().parked is None

    out = runner.advance()  # only now does the campaign move
    assert out.step == "a" and out.passed is True


# -- the ephemeral advance() mutex ----------------------------------------------------------------


class _ReentrantExperiment(ScriptedExperiment):
    """A sanctioned runnable that tries a SECOND advance() while the first holds the
    lock — the concurrent-supervisor fixture."""

    def __init__(self, device, *, runner_ref) -> None:
        super().__init__(device, success=True)
        self.runner_ref = runner_ref
        self.inner_error: BringupError | None = None

    def go(self, soc, **kw):
        try:
            self.runner_ref.advance()  # concurrent advance — must fail loudly
        except BringupError as err:
            self.inner_error = err
        return super().go(soc, **kw)


def test_a_concurrent_advance_fails_loudly(tmp_path):
    """AC: two concurrent advance calls — the second fails loudly. The lock is an
    ephemeral mutex under the campaign dir: while the holder is fresh and alive it holds;
    the first advance completes normally and releases it."""
    def factory(device, ctx):  # late-binds `runner` at call time
        return _ReentrantExperiment(device, runner_ref=runner)

    plan = BringupPlan(steps=(_step("a", on_pass="calibrated", experiment=factory),), entry="a")
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)

    out = runner.advance()  # the inner, concurrent advance failed; this one completed

    assert out.step == "a" and out.passed is True
    assert campaign.derive_state().terminal == "calibrated"
    assert not (campaign.record.dir / "bringup.lock").exists()  # released on completion


def test_a_concurrent_advance_fails_loudly_inner_error_named(tmp_path):
    """The loud failure itself: an advance attempted while the lock is freshly held
    raises BringupError naming the lock and its holder — same process, same pid, the
    mutex still refuses (freshness, not pid equality, is what breaks a lock)."""
    captured = {}

    def factory(device, ctx):  # late-binds `runner` at call time
        exp = _ReentrantExperiment(device, runner_ref=runner)
        captured["exp"] = exp
        return exp

    plan = BringupPlan(steps=(_step("a", on_pass="calibrated", experiment=factory),), entry="a")
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)

    runner.advance()
    assert captured["exp"].inner_error is not None
    message = str(captured["exp"].inner_error)
    assert "advance()" in message and "bringup.lock" in message
    assert str(os.getpid()) in message


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_a_stale_lock_is_breakable(tmp_path):
    """AC: a stale lock is breakable — it is a mutex, never authority. Heartbeat-aged
    (even with a live pid), dead-pid-fresh, and corrupt lock files all give way; the
    advance proceeds and cleans up after itself."""
    plan = _chain_plan()
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)
    lock = campaign.record.dir / "bringup.lock"

    from datetime import datetime, timedelta, timezone

    aged = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    lock.write_text(json.dumps({
        "pid": os.getpid(), "host": "not-this-host", "t": aged, "heartbeat_interval_s": 15,
    }))
    out = runner.advance()  # foreign host, but hours stale: broken
    assert out.step == "a" and out.passed is True
    assert not lock.exists()

    campaign2 = _campaign(tmp_path, plan, device="second-device", ref="second")
    runner2 = _runner(campaign2)
    lock2 = campaign2.record.dir / "bringup.lock"
    lock2.write_text(json.dumps({  # fresh, but the holder is dead
        "pid": _dead_pid(), "host": socket.gethostname(), "t": now_iso(),
        "heartbeat_interval_s": 15,
    }))
    out = runner2.advance()
    assert out.step == "a" and out.passed is True
    assert not lock2.exists()

    campaign3 = _campaign(tmp_path, plan, device="third-device", ref="third")
    runner3 = _runner(campaign3)
    lock3 = campaign3.record.dir / "bringup.lock"
    lock3.write_text("{not json at all")
    out = runner3.advance()  # corrupt: unreadable locks are stale locks
    assert out.step == "a" and out.passed is True
    assert not lock3.exists()


def test_a_fresh_foreign_lock_holds(tmp_path):
    """The complement that makes the lock honest: a fresh lock on a live foreign host
    holds (its pid is unknowable here; freshness decides), and the advance fails loudly."""
    plan = _chain_plan()
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)
    lock = campaign.record.dir / "bringup.lock"
    lock.write_text(json.dumps({
        "pid": 12345, "host": "some-other-host", "t": now_iso(), "heartbeat_interval_s": 15,
    }))

    with pytest.raises(BringupError, match="another advance"):
        runner.advance()
    # the holder's lock is left exactly as it was — breaking is for stale locks only
    assert json.loads(lock.read_text())["host"] == "some-other-host"


# -- children stay out of top-level listings ------------------------------------------------------


def test_children_under_the_campaign_dir_are_excluded_from_top_level_listings(tmp_path):
    """AC: children under the campaign dir are excluded from top-level task listings —
    the depth convention (root task dirs only). The parent's ``step_open`` events carry
    the authoritative child mapping; the listing shows the campaign itself, and the
    children remain addressable as task dirs (just not listed at the top level)."""
    from strumento.core.record import derive_state, list_tasks, read_events

    plan = _chain_plan()
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)
    runner.advance()  # mints steps/<child> for step "a"

    steps_dir = campaign.record.dir / "steps"
    children = [d for d in steps_dir.iterdir() if d.is_dir()]
    assert len(children) == 1
    assert (children[0] / "task.toml").exists()  # a real, addressable task dir

    listed = {s.id for s in list_tasks(tmp_path)}
    assert listed == {campaign.record.dir.name}  # the campaign lists; the child does not

    # the authoritative mapping is the event, not any directory name
    opens = [e for e in read_events(campaign.record.dir) if e.get("ev") == "step_open"]
    assert opens[0]["child_path"].startswith("steps/")
    assert derive_state(children[0]).manifest.kind == "experiment"  # addressable directly


# -- the success terminal ------------------------------------------------------------------------


def test_calibrated_finalizes_the_record_and_releases_the_claim(tmp_path):
    """Reaching the success terminal finalizes the campaign: the runner writes the
    record's terminal state, the claim derives released, further advances hold, and the
    freed device + ref construct again."""
    plan = BringupPlan(steps=(_step("a", on_pass="calibrated"),), entry="a")
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)

    out = runner.advance()
    assert out.passed is True and out.routed_to == "calibrated"

    state = campaign.derive_state()
    assert state.terminal == "calibrated"
    assert state.record_state == "done"   # the runner finalized the campaign record
    assert state.claim_live is False      # …which releases the claim

    out2 = runner.advance()  # idempotent: a terminal campaign never re-runs
    assert out2.held == "terminal:calibrated" and out2.step is None

    second = BringupCampaign.create(tmp_path, plan, device="demo-device", ref="main")
    assert second.derive_state().claim_live is True  # the resources are free again


def test_a_crash_between_the_pass_and_the_finalize_recovers(tmp_path):
    """Crash-safety of the finalize itself: a stream whose last pass routed to
    ``calibrated`` but whose record was never closed (the runner died in between) still
    derives terminal with a live claim — and the NEXT advance finalizes it."""
    plan = BringupPlan(steps=(_step("a", on_pass="calibrated"),), entry="a")
    campaign = _campaign(tmp_path, plan)
    campaign.record.step_open("a", 1, "steps/hand-made")
    campaign.record.step_close("a", 1, "pass")  # the runner died right here

    state = campaign.derive_state()
    assert state.terminal == "calibrated"
    assert state.claim_live is True and state.record_state == "queued"

    runner = _runner(campaign)
    out = runner.advance()  # a fresh process finalizes what the crashed one left open
    assert out.held == "terminal:calibrated"
    final = campaign.derive_state()
    assert final.record_state == "done" and final.claim_live is False


# -- the freshness seam: a staleness detour consumes the call -------------------------------------


def test_a_staleness_detour_consumes_the_call(tmp_path):
    """The freshness seam (its logic is the next slice's; the hook is this one's): when
    the injected staleness hook names a detour step, THAT step executes — under its own
    ladder — and the main step runs on the next advance."""
    detour_once = {"active": True}
    seen_positions: list[str] = []

    def staleness(step, state):
        seen_positions.append(step.name)
        if detour_once["active"] and step.name == "a":
            detour_once["active"] = False
            return "b"
        return None

    plan = BringupPlan(
        steps=(
            _step("a", on_pass="calibrated"),
            _step("b", on_pass="a"),  # a refresh detour that hands control back
        ),
        entry="a",
    )
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign, staleness=staleness)

    out = runner.advance()  # the detour consumed the call
    assert out.step == "b" and out.detour is True and out.passed is True
    assert out.attempt == 1  # b's own ladder — its first attempt
    state = campaign.derive_state()
    assert state.position == "a"  # b passed control back; a never ran yet
    assert state.attempts["b"] == 0 and state.attempts["a"] == 0

    out2 = runner.advance()  # the main step, on the next call
    assert out2.step == "a" and out2.detour is False and out2.passed is True
    assert seen_positions == ["a", "a"]  # the hook always sees the position step


def test_a_detour_naming_an_unknown_step_is_refused(tmp_path):
    """A hook naming a nonexistent detour is a loud error, not a silent no-op — the next
    slice's freshness logic rides this seam and must fail visibly when it mis-names."""
    plan = _chain_plan()
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign, staleness=lambda step, state: "ghost-step")

    with pytest.raises(BringupError, match="ghost-step"):
        runner.advance()
    assert campaign.derive_state().attempts["a"] == 0  # nothing burned


# -- purity: the runner's whole path, not just its import ------------------------------------------


def test_the_runtime_path_imports_no_qick_and_no_packs(tmp_path):
    """AC: the core bring-up module imports no qick and no pack modules — the plan-vocabulary
    tests pin the import; this one drives the RUNTIME path (create a campaign, advance it
    through a sanctioned experiment, derive state) with qick blocked and the pack namespace stubbed,
    so a lazy import anywhere in the execution path fails the test."""
    import textwrap

    script = textwrap.dedent(
        """
        import sys, types
        sys.modules["qick"] = None
        sys.modules["strumento.packs"] = types.ModuleType("strumento.packs")
        sys.path.insert(0, {repo!r})

        from types import SimpleNamespace

        from strumento.core.bringup import BringupCampaign, BringupPlan, BringupStep
        from strumento.core.experiment import Experiment
        from strumento.core.fitters import FitResult
        from strumento.core.result import MeasurementResult

        class PureRunner:
            def run(self, device, seq, sweeps, soc, **kw):
                return MeasurementResult(xpts=[0.0], avgi=[0.0], avgq=[0.0])

        class PureExperiment(Experiment):
            def sequence(self):
                return None, ()
            def analyze(self, result):
                return FitResult(params={{}}, stderr={{}}, derived={{}}, success=True)

        plan = BringupPlan(steps=(
            BringupStep(
                name="a", experiment=lambda d, c: PureExperiment(d, runner=PureRunner()),
                requires=(), provides=(), gates=("fit",), on_pass="calibrated",
                on_fail={{1: "park"}}, max_retries=1, envelope={{"rf": 0.05}},
            ),
        ), entry="a")
        campaign = BringupCampaign.create(sys.argv[1], plan, device="pure-device", ref="main")
        runner = __import__(
            "strumento.core.bringup", fromlist=["BringupRunner"]
        ).BringupRunner(campaign, device=object(), soc=SimpleNamespace())
        out = runner.advance()
        state = campaign.derive_state()
        assert out.passed is True and state.terminal == "calibrated"
        print("ok")
        """
    ).format(repo=str(Path(__file__).parents[2]))
    r = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# -- derivation-side opacity + plan/stream honesty -------------------------------------------------


def test_derivation_skips_unknown_event_values(tmp_path):
    """Forward compatibility at derivation: an unknown ``ev`` value hand-appended between
    campaign events is skipped — the derived state is exactly what it would be without
    the future writer's line (the campaign-event tests pin the substrate reader; this
    pins the walker)."""
    plan = _chain_plan()
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)
    runner.advance()  # a passes -> position b

    ledger = campaign.record.dir / "progress.jsonl"
    with ledger.open("a") as fh:
        fh.write(json.dumps({
            "t": "2030-01-01T00:00:00.000Z", "ev": "from-the-future", "payload": {"x": 1},
        }) + "\n")

    state = campaign.derive_state()
    assert state.position == "b"
    assert state.attempts["a"] == 0 and state.parked is None


def test_derivation_refuses_a_stream_naming_a_foreign_step(tmp_path, closing):
    """A stream naming a step the attached plan does not carry is a plan/stream mismatch —
    refused loudly (the campaign was re-attached to the wrong plan), never silently
    mis-routed."""
    plan = _chain_plan()
    campaign = closing(_campaign(tmp_path, plan))
    campaign.record.step_open("not-in-this-plan", 1, "steps/ghost")

    with pytest.raises(BringupError, match="not-in-this-plan"):
        campaign.derive_state()


def test_a_resumed_campaign_reruns_under_the_clamped_ladder(tmp_path):
    """After a ladder-exhausting park, a resume clears the hold and the step re-runs —
    its next failure re-parks at the terminal rung (the ladder never escalates past
    ``max_retries``; the directive's retry-reset semantics is the autonomy slice's, and
    this slice ships the conservative clamp)."""
    plan = BringupPlan(
        steps=(
            _step("a", on_pass="calibrated", on_fail={1: "park"}, max_retries=1,
                  experiment=lambda device, ctx: ScriptedExperiment(device, success=False)),
        ),
        entry="a",
    )
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)

    out = runner.advance()
    assert out.passed is False and out.attempt == 1
    assert campaign.derive_state().parked is not None

    campaign.record.resume("operator", "one more try, then we park")
    out2 = runner.advance()  # the resume cleared the hold: the step re-runs
    assert out2.step == "a" and out2.attempt == 2 and out2.passed is False
    state = campaign.derive_state()
    assert state.parked is not None and state.parked.kind == "needs_human"
    assert state.attempts["a"] == 2  # both failures counted since the last pass


# -- the runner lets go of every event file it opens (issue #123) ------------------------------


def _unclosed_event_files(fn) -> list[str]:
    """Run ``fn``, collect garbage, return ResourceWarnings naming an unclosed ``progress.jsonl``.

    ``fn`` owns nothing past its return: every record it built is dropped, so a handle still
    open at finalization is a handle somebody forgot to close."""
    import gc
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        fn()
        gc.collect()
    return [
        str(w.message) for w in caught
        if issubclass(w.category, ResourceWarning) and "progress.jsonl" in str(w.message)
    ]


def test_a_created_campaign_holds_no_open_event_file(tmp_path):
    """The claim is written at creation; creating a campaign does not leave its handle open."""
    assert _unclosed_event_files(lambda: _campaign(tmp_path, _chain_plan())) == []


def test_the_runner_closes_every_record_on_the_done_path(tmp_path):
    def run():
        runner = _runner(_campaign(tmp_path, _chain_plan()))
        runner.advance()                          # a passes -> b (child closed, open-ended)
        runner.advance()                          # b passes -> calibrated, record finalized
        assert runner.advance().held == "terminal:calibrated"

    assert _unclosed_event_files(run) == []


def test_the_runner_closes_every_record_on_the_parked_path(tmp_path):
    def run():
        plan = BringupPlan(steps=(_step("a", experiment=_factory(False)),), entry="a")
        runner = _runner(_campaign(tmp_path, plan))
        out = runner.advance()                    # a fails -> rung 1 parks
        assert out.passed is False and out.routed_to == "park"
        assert runner.advance().held.startswith("parked")   # the hold path writes nothing

    assert _unclosed_event_files(run) == []


def test_the_runner_closes_every_record_on_the_propose_park_path(tmp_path):
    from strumento.core.bringup import Clearances

    def run():
        runner = _runner(_campaign(tmp_path, _chain_plan()), safety=Clearances())
        out = runner.advance()                    # rf not cleared -> stage a proposal, park
        assert out.proposed is True

    assert _unclosed_event_files(run) == []


def test_the_runner_closes_every_record_on_the_error_path(tmp_path):
    def run():
        crash = {"pending": True}
        plan = BringupPlan(
            steps=(
                _step(
                    "a",
                    experiment=lambda device, ctx: _CrashingExperiment(device, crash=crash),
                ),
            ),
            entry="a",
        )
        runner = _runner(_campaign(tmp_path, plan))
        with pytest.raises(RuntimeError, match="board transport died"):
            runner.advance()

    assert _unclosed_event_files(run) == []


def test_the_campaign_record_still_appends_after_an_advance_lets_go(tmp_path):
    """Letting go of the handle is not finality: the operator's next write reopens and appends."""
    plan = BringupPlan(steps=(_step("a", experiment=_factory(False)),), entry="a")
    campaign = _campaign(tmp_path, plan)
    _runner(campaign).advance()                   # parks needs_human
    campaign.record.resume("operator", "go on")
    with campaign:                                 # a campaign is a context manager too
        assert campaign.derive_state().parked is None
