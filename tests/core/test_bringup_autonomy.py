"""The bring-up autonomy layer — auto/propose decision, staged proposals, approvals (issue #55).

New surface per the issue's Testing Decisions: the runner fixtures are REUSED from
``tests.core.test_bringup_runner`` (the sanctioned-experiment fixtures) and the
store/config helpers from ``tests.core.test_bringup_writeback`` — the flagged
cross-slice edges. The autonomy DECISION is new unit surface (envelope + region +
bound + force_propose math); the propose/approve/resume cycle runs end-to-end through
``advance()`` with a mock soc.

Under test (the autonomy and approval rules):

- a step runs ``auto`` iff every channel's applied excitation is within a cleared
  envelope AND every DC move is inside a cleared region AND it writes no uncleared
  bound AND ``force_propose`` is False — otherwise ``propose``;
- propose steps are compile-and-stage only: zero hardware side effects, the child
  record banks the proposal (params, program, gates, config stamp) and carries no
  acquisition event until approval;
- ``approve(actor, channel, proposal_ref)`` clears a whole escalation ladder once at
  its max rung (one approve linked by proposal_ref); approvals persist for the
  campaign's life; a forged approve (missing actor or channel) is ignored;
- the first ``advance()`` after an approve re-validates the proposal against the
  CURRENT ref — a config content-id mismatch re-proposes, never executes; execution
  lands in a fresh child record stamped with the config actually used;
- ``needs_human`` parks resolve via ``resume(actor, directive)`` — the directive
  picks the edge and states whether retries reset.

The autonomy gate arms when the runner is constructed with the **device safety
config** (``safety=``) — v1's clears no RF excitation and no DC region, so entry
steps propose by the rule itself. A runner constructed without one keeps the
direct-run surface (the pre-autonomy supervisor seam, exactly as ``store=None`` keeps
the storeless runner).
"""

from __future__ import annotations

import json

from strumento.core.bringup import BringupPlan, Clearances
from strumento.core.record import read_events
from strumento.core.store import record_calibration, unvalidated
from tests.core.test_bringup_runner import (
    ScriptedExperiment,
    _campaign,
    _step,
)
from tests.core.test_bringup_writeback import _store_with_ref

#: the v1 initial device safety config: clears no RF excitation and no DC region,
#: so entry and region-establishing steps propose by the rule itself.
V1_SAFETY = Clearances()


def _autonomy_plan(*, extra=None):
    """rabi(entry, envelope {rf: 0.05}) -> calibrated — the minimal gated step."""
    steps = [_step("rabi", on_pass="calibrated", envelope={"rf": 0.05})]
    if extra:
        steps.extend(extra)
    return BringupPlan(steps=tuple(steps), entry="rabi")


def _gate(campaign, **kw):
    """A runner with the device safety config wired — the autonomy gate armed."""
    from tests.core.test_bringup_runner import _runner

    return _runner(campaign, safety=V1_SAFETY, **kw)


# -- the auto/propose decision (envelope + region + bounds + force_propose) ------------------------


def test_decides_auto_the_envelope_region_and_bound_math():
    """AC (unit surface): the decision is one pure function — every channel within a
    cleared envelope, every DC region cleared, no uncleared bound written, and not
    force_propose. Each clause fires alone; within/covered/inside all run auto."""
    from strumento.core.bringup import Provides, decides_auto

    envelope_step = _step("s", envelope={"rf": 0.05})
    # envelope: no clearance -> propose; smaller-or-equal clearance -> auto; larger -> auto
    assert decides_auto(envelope_step, Clearances()) is False
    assert decides_auto(envelope_step, Clearances(envelopes={"rf": 0.05})) is True
    assert decides_auto(envelope_step, Clearances(envelopes={"rf": 9.9})) is True
    assert decides_auto(envelope_step, Clearances(envelopes={"rf": 0.01})) is False
    assert decides_auto(envelope_step, Clearances(envelopes={"other": 9.9})) is False
    # a step declaring NO channel is within every envelope (nothing applied)
    assert decides_auto(_step("s", envelope={}), Clearances()) is True

    region_step = _step("s", envelope={}, region="stability.map")
    assert decides_auto(region_step, Clearances()) is False  # region not cleared
    assert decides_auto(region_step, Clearances(regions={"stability.map"})) is True
    assert decides_auto(_step("s", envelope={}), Clearances()) is True  # no region declared

    bound_step = _step(
        "s",
        envelope={},
        provides=(Provides(path="readout", mapping={"p": "max_power"}, max_abs=-10.0),),
    )
    assert decides_auto(bound_step, Clearances()) is False  # writes an uncleared bound
    assert decides_auto(bound_step, Clearances(bounds={"readout.max_power"})) is True
    unbounded_step = _step(
        "s",
        envelope={},
        provides=(Provides(path="readout", mapping={"p": "max_power"}),),
    )
    assert decides_auto(unbounded_step, Clearances()) is True  # no bound declared

    gated = _step("s", envelope={}, force_propose=True)
    assert decides_auto(gated, Clearances(envelopes={"rf": 9.9})) is False  # always gated


def test_initial_config_clears_no_envelope_so_entry_steps_propose(tmp_path):
    """AC: v1's initial device safety config clears no RF excitation and no DC region,
    so an entry step PROPOSES by the rule itself — nothing ran, the campaign parked
    with kind `propose`, and the campaign ledger carries no step attempt at all."""
    campaign = _campaign(tmp_path, _autonomy_plan())
    runner = _gate(campaign)

    out = runner.advance()

    assert out.step == "rabi" and out.passed is None and out.proposed is True
    assert out.held == "parked:propose" and out.routed_to == "park:propose"
    state = campaign.derive_state()
    assert state.parked is not None and state.parked.kind == "propose"
    assert state.position == "rabi" and state.attempts["rabi"] == 0
    steps_dir = campaign.record.dir / "steps"
    assert steps_dir.exists() and any(steps_dir.iterdir())  # the staged child exists
    events = read_events(campaign.record.dir)
    assert not any(e["ev"] == "step_open" for e in events)  # nothing executed
    assert not any(e["ev"] == "step_close" for e in events)


def test_step_within_cleared_envelope_runs_auto(tmp_path):
    """AC: a step whose every channel sits within a cleared envelope runs ``auto`` — the
    propose park the previous advance raised is cleared by the operator's approve
    (linked by proposal_ref), and the step executes (step_open + step_close)."""
    campaign = _campaign(tmp_path, _autonomy_plan())
    runner = _gate(campaign)
    runner.advance()  # stages the proposal -> parks propose
    campaign.record.approve("operator", "operator-cli", _staged_ref(campaign))

    out = runner.advance()  # the auto path — envelope cleared, step runs

    assert out.step == "rabi" and out.passed is True and out.proposed is False
    assert campaign.derive_state().terminal == "calibrated"


def test_force_propose_true_forces_propose_even_when_cleared(tmp_path):
    """AC: ``force_propose=True`` forces propose regardless of cleared state — the
    pack's override for steps that must always be human-gated (cold-start entry steps
    with no priors). It proposes even with the envelope cleared by the safety
    config itself."""
    plan = BringupPlan(
        steps=(
            _step(
                "rabi",
                on_pass="calibrated",
                envelope={"rf": 0.05},
                force_propose=True,
            ),
        ),
        entry="rabi",
    )
    # the safety config pre-clears the channel — force_propose proposes ANYWAY
    from tests.core.test_bringup_runner import _runner

    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign, safety=Clearances(envelopes={"rf": 1.0}))

    out = runner.advance()

    assert out.proposed is True and out.passed is None
    assert campaign.derive_state().parked.kind == "propose"


# -- compile-and-stage: zero hardware side effects --------------------------------------------------


def test_propose_steps_bank_a_proposal_and_never_touch_hardware(tmp_path):
    """AC: a propose step is compile-and-stage only — the child record banks the
    proposal (params, program, gates to check, config stamp) and its ledger contains
    NO acquisition event (no gate verdicts, no fit progress); the campaign ledger
    links the proposal to its child by proposal_ref + child_path."""
    campaign = _campaign(tmp_path, _autonomy_plan())
    runner = _gate(campaign)

    out = runner.advance()  # stages the proposal

    assert out.proposed is True
    steps_dir = campaign.record.dir / "steps"
    children = sorted(steps_dir.iterdir())
    assert len(children) == 1  # the proposal's child record — compute-and-record
    ledger = read_events(children[0])
    kinds = {e["ev"] for e in ledger}
    assert kinds == {"progress", "artifact"}  # the staging marker + the banked proposal
    assert not any(e["ev"] == "gate" for e in ledger)  # no verdicts — nothing ran
    park = _staged_park(campaign)
    assert park["kind"] == "propose"
    assert park["proposal_ref"] == "rabi:1"  # stable, operator-addressable ref
    assert park["step"] == "rabi"
    assert park["child_path"].startswith("steps/")
    assert park["envelope"] == {"rf": 0.05}


def test_a_propose_step_child_record_has_no_acquisition_event_until_approval(tmp_path):
    """AC (the mandatory one): the child record of a propose step contains NO
    acquisition event until approval — zero hardware side effects, compute-and-record
    only. The acquisition lands only after the approve clears the envelope AND the
    runner re-validates and EXECUTES in a fresh child record."""
    store = _store_with_ref(tmp_path)
    campaign = _campaign(tmp_path, _autonomy_plan(), ref="main")
    runner = _gate(campaign, store=store)
    runner.advance()  # stage
    staged = next((campaign.record.dir / "steps").iterdir())
    assert not any(e["ev"] == "gate" for e in read_events(staged))  # no verdicts — no run

    campaign.record.approve("operator", "operator-cli", _staged_ref(campaign))
    out = runner.advance()  # re-validated and executed

    assert out.passed is True and out.proposed is False
    executed_children = [d for d in (campaign.record.dir / "steps").iterdir() if d != staged]
    assert len(executed_children) == 1  # execution landed in a FRESH child record
    assert any(e["ev"] == "gate" for e in read_events(executed_children[0]))  # the run


# -- approval: ladder, persistence, forged events --------------------------------------------------


def test_one_approve_clears_a_whole_escalation_ladder_at_its_max_rung(tmp_path):
    """AC: a whole escalation ladder is approved once with its max rung — one approve
    linked by proposal_ref clears the full envelope set (the max-rung envelope), so
    the rung-holding step runs auto AND a lower-rung step within the same cleared
    envelope runs auto with no proposal of its own; approvals persist for the
    campaign's life."""
    plan = _autonomy_plan(
        extra=[_step("rabi-low", on_pass="calibrated", envelope={"rf": 0.02})],
    )
    plan = BringupPlan(
        steps=(
            _step("qubit-spec", on_pass="rabi-low", envelope={"rf": 0.20}),
            _step("rabi-low", on_pass="calibrated", envelope={"rf": 0.02}),
        ),
        entry="qubit-spec",
    )
    campaign = _campaign(tmp_path, plan)
    runner = _gate(campaign)
    out1 = runner.advance()
    assert out1.proposed is True  # no cleared envelope

    campaign.record.approve("operator", "operator-cli", _staged_ref(campaign))

    out2 = runner.advance()  # the max rung runs auto under the one approval
    assert out2.proposed is False and out2.passed is True and out2.step == "qubit-spec"
    out3 = runner.advance()  # the lower rung is WITHIN the cleared 0.20 — auto too
    assert out3.proposed is False and out3.passed is True and out3.step == "rabi-low"
    assert campaign.derive_state().terminal == "calibrated"
    # the approval persists: the cleared ref stays derived for the campaign's life
    assert _staged_ref(campaign) in campaign.derive_state().cleared_envelopes


def test_an_approve_missing_actor_or_channel_is_ignored_at_derivation(tmp_path, closing):
    """AC: a forged approve (missing actor or channel — hand-appended, since the
    emitters refuse such appends) is ignored at derivation: it clears no envelope and
    no propose park. The valid one clears both."""
    campaign = closing(_campaign(tmp_path, _autonomy_plan()))
    campaign.record.park("propose", "no cleared envelope")

    ledger = campaign.record.dir / "progress.jsonl"
    with ledger.open("a") as fh:  # the forgeries
        fh.write(json.dumps({
            "t": "2026-08-30T00:00:00.000Z", "ev": "approve",
            "channel": "operator-cli", "proposal_ref": "prop-forged-1",
        }) + "\n")
        fh.write(json.dumps({
            "t": "2026-08-30T00:00:00.001Z", "ev": "approve",
            "actor": "operator", "proposal_ref": "prop-forged-2",
        }) + "\n")

    state = campaign.derive_state()
    assert state.cleared_envelopes == frozenset()
    assert state.parked is not None and state.parked.kind == "propose"
    assert len(state.pending_approvals) == 1


def test_an_approve_for_a_different_proposal_does_not_clear_the_outstanding_park(tmp_path):
    """The ref-linking handoff gap (the runner once cleared THE outstanding propose park on any
    valid approve): with proposal_refs on parks, an approve naming a DIFFERENT
    proposal leaves the outstanding park held — outstanding proposals are
    disambiguated by ref."""
    campaign = _campaign(tmp_path, _autonomy_plan())
    runner = _gate(campaign)
    runner.advance()  # stages proposal "rabi:1" -> parks propose

    campaign.record.approve("operator", "operator-cli", "rabi:999")  # a different ref

    state = campaign.derive_state()
    assert state.parked is not None and state.parked.kind == "propose"
    assert state.cleared_envelopes == frozenset({"rabi:999"})  # banked, but inert
    out = runner.advance()  # still held — the outstanding proposal was not approved
    assert out.held == "parked:propose" and out.step is None


# -- re-validation: the config actually used --------------------------------------------------------


def test_a_moved_config_ref_re_proposes_never_executes(tmp_path):
    """AC: at the first advance() after an approve, the runner re-validates the
    proposal against the CURRENT ref (config content-id match); a moved config
    (content-id mismatch) RE-PROPOSES, never executes — and the re-proposal parks
    again with a fresh proposal_ref for the operator to approve."""
    store = _store_with_ref(tmp_path)
    campaign = _campaign(tmp_path, _autonomy_plan(), ref="main")
    runner = _gate(campaign, store=store)
    runner.advance()  # stages the proposal against the CURRENT ref
    campaign.record.approve("operator", "operator-cli", _staged_ref(campaign))

    # the config moves under the ref — the staged proposal is against a stale world
    base = store.get(store.resolve_ref("main"))
    record_calibration(store, "main", base, {"sensor.bias": 0.123}, source="hand-adjust",
                       validate=unvalidated)  # a synthetic, non-composable config

    out = runner.advance()  # re-validation FAILS -> re-propose, never execute

    assert out.proposed is True and out.passed is None
    state = campaign.derive_state()
    assert state.parked is not None and state.parked.kind == "propose"
    assert state.attempts["rabi"] == 0  # nothing burned — a mismatch is not a failure
    events = read_events(campaign.record.dir)
    assert not any(e["ev"] == "step_close" for e in events)  # never executed
    # the re-proposal carries a FRESH ref (the config stamp moved)…
    assert _staged_ref(campaign) != "rabi:1"
    # …so the old approval cannot wave it through
    out2 = runner.advance()
    assert out2.held == "parked:propose" and out2.step is None


def test_execution_lands_in_a_fresh_child_record_stamped_with_the_config_used(tmp_path):
    """AC: execution lands in a fresh child record stamped with the config actually
    used — the executed child's manifest carries the config_content_id the runner
    validated against (re-validation passed)."""
    store = _store_with_ref(tmp_path)
    campaign = _campaign(tmp_path, _autonomy_plan(), ref="main")
    runner = _gate(campaign, store=store)
    runner.advance()  # stage
    campaign.record.approve("operator", "operator-cli", _staged_ref(campaign))

    out = runner.advance()  # execute

    assert out.passed is True
    executed = _executed_child(campaign)
    assert executed.manifest.config_content_id  # stamped with the config used
    assert executed.manifest.config_content_id == store.resolve_ref("main")


# -- needs_human parks resolve via resume(actor, directive) -------------------------------------------


def _failing_plan():
    return BringupPlan(
        steps=(
            _step(
                "ramsey",
                on_pass="calibrated",
                on_fail={1: "ramsey", 2: "park:needs_human"},
                max_retries=2,
                envelope={"rf": 0.05},
                experiment=lambda d, c: ScriptedExperiment(d, success=False),
            ),
        ),
        entry="ramsey",
    )


def test_needs_human_park_resolves_via_resume_directive_picking_the_edge(tmp_path):
    """AC: ``needs_human`` parks resolve via ``resume(actor, directive)`` — the
    directive picks the edge: a ``retry`` directive routes the SAME step (retries
    keep counting — the conservative default, so a plain-English directive behaves
    exactly like the direct-run surface's clamped re-run). The autonomy gate is not armed here: the
    resume semantics under test are derivation-level."""
    from tests.core.test_bringup_runner import _runner

    campaign = _campaign(tmp_path, _failing_plan())
    runner = _runner(campaign)

    out1 = runner.advance()  # fails (rung 1)
    assert out1.passed is False and out1.attempt == 1
    out2 = runner.advance()  # fails again (rung 2) -> parks needs_human
    assert out2.passed is False and out2.attempt == 2
    assert campaign.derive_state().parked.kind == "needs_human"

    campaign.record.resume("operator", "retry: the setup was warmed up now")
    out3 = runner.advance()  # the directive picked the retry edge: same step
    assert out3.step == "ramsey" and out3.attempt == 3
    assert out3.passed is False
    assert campaign.derive_state().attempts["ramsey"] == 3  # retries did NOT reset


def test_resume_directive_advance_follows_the_on_pass_edge(tmp_path):
    """AC: the directive picks the edge — an ``advance`` directive resolves the
    needs_human park by following the parked step's ``on_pass`` edge WITHOUT executing
    it (the operator accepts the current state and moves on); a step whose on_pass is
    the success terminal derives calibrated."""
    from tests.core.test_bringup_runner import _runner

    plan = BringupPlan(
        steps=(
            _step(
                "ramsey",
                on_pass="calibrated",
                on_fail={1: "park:needs_human"},
                max_retries=1,
                envelope={"rf": 0.05},
                experiment=lambda d, c: ScriptedExperiment(d, success=False),
            ),
        ),
        entry="ramsey",
    )
    campaign = _campaign(tmp_path, plan)
    runner = _runner(campaign)
    out1 = runner.advance()  # fails -> parks needs_human at the terminal rung
    assert out1.passed is False and campaign.derive_state().parked.kind == "needs_human"

    campaign.record.resume("operator", "advance: the T2* is good enough as measured")
    state = campaign.derive_state()
    assert state.parked is None and state.terminal == "calibrated"

    out2 = runner.advance()  # nothing more to run — the operator's edge was the terminal
    assert out2.held == "terminal:calibrated" and out2.step is None


def test_resume_directive_reset_resets_retries(tmp_path):
    """AC: the directive states whether retries reset — a ``reset`` directive zeroes
    the step's failure count before the retry runs (the ladder restarts from rung 1)."""
    from tests.core.test_bringup_runner import _runner

    campaign = _campaign(tmp_path, _failing_plan())
    runner = _runner(campaign)
    runner.advance()  # fail 1
    runner.advance()  # fail 2 -> park needs_human
    assert campaign.derive_state().parked.kind == "needs_human"

    campaign.record.resume("operator", "reset: the device was re-thermally-cycled")
    out3 = runner.advance()
    assert out3.step == "ramsey" and out3.attempt == 1  # the ladder restarted
    assert campaign.derive_state().attempts["ramsey"] == 1


# -- helpers -----------------------------------------------------------------------------------------


def _staged_park(campaign) -> dict:
    """The most recent propose park event in the campaign stream."""
    parks = [
        e for e in read_events(campaign.record.dir)
        if e.get("ev") == "park" and e.get("kind") == "propose"
    ]
    assert parks, "no staged proposal in the stream"
    return parks[-1]


def _staged_ref(campaign) -> str:
    """The proposal_ref of the most recent staged proposal park in the stream."""
    ref = _staged_park(campaign).get("proposal_ref")
    assert ref, "staged proposal carries no proposal_ref"
    return ref


def _executed_child(campaign):
    """The child record of the step that EXECUTED (the child carrying gate events) —
    as distinct from the staged proposal's child."""
    from strumento.core.record import TaskRecord

    steps_dir = campaign.record.dir / "steps"
    for child in sorted(steps_dir.iterdir(), reverse=True):
        if any(e.get("ev") == "gate" for e in read_events(child)):
            return TaskRecord.open(child)
    raise AssertionError("no executed child record found")
