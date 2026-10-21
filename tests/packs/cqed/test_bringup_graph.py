"""The cqed bring-up graph as pack data (issue #56).

The bring-up graph is **pack data** the core never names: the cqed pack authors it against the
core plan vocabulary (``BringupPlan`` / ``BringupStep`` / ``Provides`` / ``Requirement``), and
construction-time validation refuses a malformed graph before anything runs. These tests pin
the graph's shape as authored data:

- the full chain loads and passes every construction check (AC 1);
- the entry step forces propose (the cold-start gate), end to end through the autonomy gate;
- every escalation ladder terminates in a needs_human park; the TLS scan and the spurs check
  are escalation-only — no ``on_pass`` edge ever routes to them;
- the punchout banks the RF envelope path and the operating point, and the kappa estimate is
  gate-only (in no provides);
- Ramsey's ladder routes the TLS scan on rung 1 and parks needs_human on rung 2;
- the validation terminal requires the recorded T1/T2 paths and gates the detuning window and
  the readout fidelity against the SAME pack-prior objects the readout-state node consumes;
- the no-contrast ladder is the serialized readout-chain revisit through existing nodes, with
  the joint 2D revisit a named waiver no v1 edge routes to;
- the hardware-mode physical priors are pack data (the windows the spec enumerates).

Pure: construction-time validation only — no execution, no store, no board (the autonomy
test loads a device instance, which stays pure).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from strumento.core.bringup import PARK_NEEDS_HUMAN, _park_kind
from strumento.packs.cqed.bringup import (
    ANHARMONICITY_MHZ,
    ASSIGNMENT_FIDELITY_MIN,
    KAPPA_MHZ,
    PHYSICAL_PRIORS,
    RABI_CONTRAST_MIN,
    RABI_RATE_MHZ,
    RAMSEY_DETUNING_MAX_KHZ,
    cqed_bringup_plan,
)
from tests._paths import MULTIMODE_YAML


@pytest.fixture(scope="module")
def plan():
    """The full cqed bring-up graph, exactly as the pack ships it."""
    return cqed_bringup_plan()


def _step(plan, name):
    return {s.name: s for s in plan.steps}[name]


# -- AC 1: the full cqed graph loads as pack data through the plan schema ---------------------


def test_the_full_cqed_graph_loads_as_pack_data_and_validates(plan):
    """AC: the full cqed graph loads as pack data through the plan schema and passes
    construction validation — every write-exclusivity, refine-ordering, edge, ladder and
    declaration check the ``BringupPlan`` constructor runs, on the pack's own graph."""
    names = {s.name for s in plan.steps}
    assert plan.entry == "resonator-find-coarse"
    for name in (
        "resonator-find-coarse",
        "resonator-find",
        "punchout",
        "resonator-refind",
        "qubit-spec",
        "amplitude-rabi",
        "readout-length",
        "readout-state",
        "t1",
        "ramsey",
        "tls-scan",
        "spurs-check",
        "validation",
    ):
        assert name in names, f"missing graph node {name!r}"
    # the chain the spec's graph names, edge for edge
    chain = [
        "resonator-find-coarse",
        "resonator-find",
        "punchout",
        "resonator-refind",
        "qubit-spec",
        "amplitude-rabi",
        "readout-length",
        "readout-state",
        "t1",
        "ramsey",
        "validation",
    ]
    for a, b in zip(chain, chain[1:]):
        assert _step(plan, a).on_pass == b, f"the on_pass chain breaks at {a!r}"
    assert _step(plan, "validation").on_pass == "calibrated"
    # construction itself is the plan validation: no BringupPlanError above


def test_entry_step_force_proposes(plan):
    """The entry step (resonator find-coarse) is force_propose — a cold-start wide sweep at
    minimum power proposes even when the operator's safety config pre-clears its envelope."""
    entry = _step(plan, plan.entry)
    assert entry.force_propose is True


def test_entry_step_proposes_under_a_fully_cleared_safety_config(tmp_path):
    """Entry autonomy, end to end: the entry step proposes even against a safety config that
    pre-clears its whole envelope — force_propose overrides the cleared state (the pack-level
    override for cold-start entry steps with no priors). Zero hardware side effects."""
    from strumento.core.bringup import BringupCampaign, BringupRunner, Clearances
    from strumento.core.device import Device
    from strumento.core.record import read_events
    from strumento.core.store import FileConfigStore

    device = Device.load(MULTIMODE_YAML)
    plan = cqed_bringup_plan()
    entry = _step(plan, "resonator-find-coarse")
    store = FileConfigStore(tmp_path / "cfgstore")
    cfg = device.to_config()
    store.put(cfg)
    store.set_ref("main", cfg.content_id())
    campaign = BringupCampaign.create(tmp_path, plan, device="multimode-demo", ref="main")
    # a safety config that clears every channel the entry declares — it proposes anyway
    runner = BringupRunner(
        campaign, device=device, soc=object(), store=store,
        safety=Clearances(envelopes=dict(entry.envelope)),
    )

    out = runner.advance()

    assert out.proposed is True and out.passed is None
    assert out.held == "parked:propose"
    state = campaign.derive_state()
    assert state.parked is not None and state.parked.kind == "propose"
    assert state.position == "resonator-find-coarse"  # staged, never executed
    assert not any(e["ev"] == "step_open" for e in read_events(campaign.record.dir))


def test_pack_priors_are_the_spec_windows_as_data():
    """The hardware-mode physical priors are pack data with the spec's windows: κ ∈ [0.1, 10]
    MHz, |α| ∈ [100, 400] MHz, Rabi rate ∈ [0.1, 50] MHz, rabi_contrast > 0.7, assignment
    fidelity ≥ the pack threshold, and the 50 kHz Ramsey-detuning terminal window."""
    assert KAPPA_MHZ == (0.1, 10.0)
    assert ANHARMONICITY_MHZ == (100.0, 400.0)
    assert RABI_RATE_MHZ == (0.1, 50.0)
    assert RABI_CONTRAST_MIN == 0.7
    # the readout-length window prior consumes the kappa window: [1/κ] in µs
    lo, hi = 1.0 / KAPPA_MHZ[1], 1.0 / KAPPA_MHZ[0]
    assert (lo, hi) == pytest.approx((0.1, 10.0))
    assert 0.0 < ASSIGNMENT_FIDELITY_MIN < 1.0
    assert RAMSEY_DETUNING_MAX_KHZ == 50.0
    # one prior object, several consumers (the spec's words): the same threshold object gates
    # the readout-state step's fidelity gate and the terminal validation's
    assert PHYSICAL_PRIORS == {
        "kappa_mhz": KAPPA_MHZ,
        "anharmonicity_mhz": ANHARMONICITY_MHZ,
        "rabi_rate_mhz": RABI_RATE_MHZ,
        "rabi_contrast_min": RABI_CONTRAST_MIN,
        "assignment_fidelity_min": ASSIGNMENT_FIDELITY_MIN,
        "ramsey_detuning_max_khz": RAMSEY_DETUNING_MAX_KHZ,
    }


# -- AC 7: every ladder terminates in a park; TLS scan is escalation-only ----------------------


def test_every_ladder_terminates_in_a_needs_human_park_after_max_retries(plan):
    """AC: every ladder routes every rung 1..max_retries and its terminal rung parks
    needs_human — walked explicitly over the plan (construction already enforces it; this
    pins the pack's authored ladders against silent re-shaping)."""
    for step in plan.steps:
        rungs = sorted(step.on_fail)
        assert rungs == list(range(1, step.max_retries + 1)), (
            f"step {step.name!r}: the on_fail ladder must route every rung 1..max_retries"
        )
        assert _park_kind(step.on_fail[step.max_retries]) == PARK_NEEDS_HUMAN, (
            f"step {step.name!r}: the terminal rung must park needs_human"
        )


def test_tls_scan_and_spurs_check_are_escalation_only(plan):
    """AC: the TLS scan (and the spurs check) are escalation-only — no step's ``on_pass``
    edge routes to them; Ramsey's ladder is what reaches the TLS scan (on_fail[1])."""
    for step in plan.steps:
        assert step.on_pass not in ("tls-scan", "spurs-check"), (
            f"step {step.name!r}: on_pass routes to an escalation-only node — the TLS scan"
            " (and the spurs check) are never on the on_pass chain"
        )


def test_ramsey_ladder_routes_tls_then_parks_and_tls_banks_nothing(plan):
    """Ramsey dead: on_fail[1] → the TLS scan node (escalation-only, banks nothing —
    gate-only evidence for the park decision); on_fail[2] → needs_human park."""
    ramsey = _step(plan, "ramsey")
    assert ramsey.on_fail == {1: "tls-scan", 2: "park:needs_human"}
    assert ramsey.max_retries == 2
    tls = _step(plan, "tls-scan")
    assert tls.provides == ()  # banks nothing — gate-only evidence for the park decision
    assert tls.on_pass == "ramsey"  # the evidence gathered, Ramsey retries
    assert tls.on_fail == {1: "park:needs_human"}  # a TLS failure parks too


def test_the_joint_2d_revisit_is_a_named_waiver_never_an_edge():
    """The joint 2D readout revisit is the named post-waiver follow-up: it is authoring data
    in the plan's ``waived`` set and NO v1 edge routes to it (construction would refuse)."""
    plan = cqed_bringup_plan()
    assert "joint-2d-revisit" in plan.waived
    for step in plan.steps:
        reached = [step.on_pass, *step.on_fail.values()]
        assert "joint-2d-revisit" not in reached, (
            f"step {step.name!r} routes to the waived joint 2D revisit — no v1 edge may"
        )


# -- the escalation ladders the spec enumerates ------------------------------------------------


def test_resonator_miss_ladder_is_three_rungs_terminating_in_a_park():
    """Resonator miss (widen ×4 → drop ~20 dB → spurs check; 3 rungs): the find-coarse
    ladder covers three rungs — widen (self-retry, the factory widens ×4), the spurs-check
    node, and the terminal needs_human park (the core's law: the max_retries rung parks).
    The ~20 dB drop is the third attempt's re-parameterization, after the spurs detour
    returns control to the coarse find."""
    plan = cqed_bringup_plan()
    coarse = _step(plan, "resonator-find-coarse")
    assert coarse.max_retries == 3
    assert coarse.on_fail == {
        1: "resonator-find-coarse",
        2: "spurs-check",
        3: "park:needs_human",
    }
    spurs = _step(plan, "spurs-check")
    assert spurs.on_pass == "resonator-find-coarse"  # the drop-power retry follows the check
    assert spurs.on_fail == {1: "park:needs_human"}  # and the spurs check itself parks on fail


def test_punchout_no_show_ladder_is_two_rungs():
    """Punchout no-show (verify coupling → re-identify dip; 2 rungs): rung 1 re-enters the
    targeted find (re-identify the dip), and the terminal rung parks needs_human."""
    plan = cqed_bringup_plan()
    punchout = _step(plan, "punchout")
    assert punchout.max_retries == 2
    assert punchout.on_fail == {1: "resonator-find", 2: "park:needs_human"}


def test_no_contrast_ladder_is_the_serialized_readout_chain_revisit():
    """No contrast: the SERIALIZED readout-chain revisit — frequency, then power, then
    threshold, each an EXISTING node — terminating in a needs_human park; the joint 2D
    revisit is the named waiver and no edge routes to it."""
    plan = cqed_bringup_plan()
    rabi = _step(plan, "amplitude-rabi")
    assert rabi.on_fail == {
        1: "resonator-refind",  # frequency
        2: "punchout",          # power
        3: "readout-state",     # threshold
        4: "park:needs_human",  # the terminal park
    }
    assert rabi.max_retries == 4
    # the revisit targets are the graph's own existing nodes, in chain order
    names = {s.name for s in plan.steps}
    for rung_target in ("resonator-refind", "punchout", "readout-state"):
        assert rung_target in names, "each revisit target is an existing node"


# -- the provides/requires contract per node ---------------------------------------------------


def test_punchout_banks_envelope_and_operating_point_and_kappa_writes_nothing(plan):
    """AC: the punchout banks the RF envelope path (readout.max_power, with the declared
    bound) AND the operating point (readout.power) — and the kappa estimate is gate-only:
    no provides path carries it, so a passing punchout can never write it."""
    punch = _step(plan, "punchout")
    paths = {
        f"{provides.path}.{subfield}": provides
        for provides in punch.provides
        for subfield in provides.mapping.values()
    }
    assert set(paths) == {"readout.max_power", "readout.power"}
    # the RF envelope path is the bound-declared one (bounds refused, never clamped)
    bounded = {path for path, provides in paths.items() if provides.max_abs is not None}
    assert bounded == {"readout.max_power"}
    # ...and the kappa estimate stays gate-only: no provides mapping names it
    assert not any("kappa" in key for provides in punch.provides for key in provides.mapping)
    # ...and the kappa window is still a live gate on the step
    assert "kappa-window" in punch.gates


def test_write_exclusivity_shape_of_the_full_graph(plan):
    """The graph's writer discipline, asserted on the pack data: the coarse find is the
    set-writer of the resonator frequency (refind refines it), the punchout is the set-writer
    of the operating point, and Ramsey refines qubit.f_ge (set by the qubit-spec step)."""
    assert _step(plan, "resonator-find-coarse").provides[0].kind == "set"
    assert _step(plan, "resonator-refind").provides[0].kind == "refine"
    ramsey = _step(plan, "ramsey")
    kinds = {provides.kind for provides in ramsey.provides}
    assert kinds == {"set", "refine"}
    # the Ramsey compound: detuning correction (set), q_freq (refine), T2 (set)
    qubit_paths = {
        f"{provides.path}.{subfield}": provides.kind
        for provides in ramsey.provides
        for subfield in provides.mapping.values()
    }
    assert set(qubit_paths) == {
        "qubit.freq_correction",  # the measured detuning (set)
        "qubit.f_ge",             # f_spec + Δ_Ramsey at the final Rabi gain
        "qubit.T2",               # T2* from the Ramsey envelope
    }
    assert qubit_paths["qubit.freq_correction"] == "set"
    assert qubit_paths["qubit.f_ge"] == "refine"
    assert qubit_paths["qubit.T2"] == "set"


def test_validation_terminal_requires_the_recorded_quantities(plan):
    """AC: the validation terminal requires T1 AND T2 recorded (the freshness engine refuses
    to run it until the store holds both) plus the measured detuning/fidelity gates, and its
    pass edge is the success terminal."""
    validation = _step(plan, "validation")
    required = {req.path for req in validation.requires}
    assert {"qubit.T1", "qubit.T2"} <= required
    assert validation.on_pass == "calibrated"


def test_the_readout_state_step_gates_the_shared_pack_threshold(plan):
    """The readout-state node writes the readout threshold/angle its fidelity gate consumed,
    and the terminal validation requires that threshold: a readout-state failure
    failure-poisons it, and the freshness engine detours validation to re-check — the same
    pack-prior threshold object gates both consumers."""
    readout_state = _step(plan, "readout-state")
    validation = _step(plan, "validation")
    written = {
        f"{p.path}.{sf}" for p in readout_state.provides for sf in p.mapping.values()
    }
    assert written == {"readout.threshold", "readout.angle"}
    required = {req.path for req in validation.requires}
    assert "readout.threshold" in required


# -- purity ------------------------------------------------------------------------------------


def test_pack_bringup_module_imports_without_qick():
    """The graph is pack data in the pure lane: importing the pack's bring-up module — and
    constructing the plan from it — requires no qick (mirrors the core purity pins)."""
    code = (
        "import sys; sys.modules['qick']=None;"
        "import strumento.packs.cqed.bringup as b;"
        "plan = b.cqed_bringup_plan();"
        "print(len(plan.steps))"
    )
    repo = Path(__file__).parents[3]
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=repo)
    assert r.returncode == 0 and "13" in r.stdout, r.stderr
