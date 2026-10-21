"""Construction-time validation of typed bring-up plans (issue #52).

The bring-up layer needs its typed graph vocabulary — steps, multi-path provides with
set/refine discipline, retry-indexed escalation ladders, declared envelopes and regions —
so that a malformed graph is refused at *authoring* time, not mid-campaign on a device.
These tests pin the pure construction-time contract: no execution, no store access, no
task-dir scanning. The claim/exclusivity scan and the runner are tested in their own files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import strumento.core.bringup as bringup
from strumento.core.bringup import BringupPlan, BringupStep, Provides, Requirement

BRINGUP_PY = Path(bringup.__file__)


def _step(name: str, **overrides) -> BringupStep:
    """A valid step with spec-shaped defaults; tests override only what they are pinning."""
    kwargs: dict = dict(
        name=name,
        experiment=lambda device, ctx: None,  # factory — never invoked at construction
        requires=(),
        provides=(),
        gates=("fit",),
        frame_check=None,
        on_pass="calibrated",
        on_fail={1: "park"},
        max_retries=1,
        envelope={"rf": 0.05},
        on_stale="refresh",
        poisons=(),
        region=None,
        force_propose=False,
    )
    kwargs.update(overrides)
    return BringupStep(**kwargs)


def test_valid_plan_constructs():
    """A spec-shaped graph — set-writer, downstream refine, escalation back-edge, park
    terminal — builds cleanly: the vocabulary must accept everything the packs will author."""
    coarse = _step(
        "resonator-find-coarse",
        provides=(Provides(path="readout.frequency", mapping={"f_res": "frequency"}),),
        on_pass="punchout",
    )
    punch = _step(
        "punchout",
        provides=(
            Provides(path="readout.max_power", mapping={"p_max": "power"}),
            Provides(path="readout.power", mapping={"p_op": "power"}),
        ),
        requires=(Requirement(path="readout.frequency", max_age_s=None),),
        on_pass="resonator-refind",
        on_fail={1: "resonator-find-coarse", 2: "park"},
        max_retries=2,
    )
    refind = _step(
        "resonator-refind",
        provides=(
            Provides(path="readout.frequency", kind="refine", mapping={"f_res": "frequency"}),
        ),
        requires=(Requirement(path="readout.power", max_age_s=3600.0, drift_sensitive=True),),
        on_pass="calibrated",
    )
    plan = BringupPlan(steps=(coarse, punch, refind), entry="resonator-find-coarse")

    assert plan.entry == "resonator-find-coarse"
    assert [s.name for s in plan.steps] == [
        "resonator-find-coarse",
        "punchout",
        "resonator-refind",
    ]


# -- declared envelopes and regions (AC: envelope numeric per channel; region a named
# -- artifact or absent; on_stale a policy word) --------------------------------------------


def test_envelope_values_must_be_numeric_maxima():
    """A per-channel envelope declares numeric applied-excitation maxima — a string, a
    bool, NaN, an inf, or a negative 'maximum' is refused, naming the step and channel.
    Ints are as numeric as floats."""
    for bad in ("high", True, float("nan"), float("inf"), -0.05):
        with pytest.raises(bringup.BringupPlanError, match=r"rabi.*rf"):
            BringupPlan(steps=(_step("rabi", envelope={"rf": bad}),), entry="rabi")
    BringupPlan(steps=(_step("rabi", envelope={"rf": 1}),), entry="rabi")  # int is numeric


def test_envelope_must_map_channels():
    with pytest.raises(bringup.BringupPlanError, match=r"rabi.*envelope"):
        BringupPlan(steps=(_step("rabi", envelope=[("rf", 0.05)]),), entry="rabi")


def test_envelope_channels_must_be_named():
    for bad in ("", "   "):
        with pytest.raises(bringup.BringupPlanError, match=r"rabi.*envelope.*channel"):
            BringupPlan(steps=(_step("rabi", envelope={bad: 0.05}),), entry="rabi")


def test_region_is_a_named_artifact_or_absent():
    """A declared DC region names a banked artifact — an empty or blank string is
    neither a name nor absent."""
    for bad in ("", "   "):
        with pytest.raises(bringup.BringupPlanError, match=r"dot-formation.*region"):
            BringupPlan(steps=(_step("dot-formation", region=bad),), entry="dot-formation")
    BringupPlan(steps=(_step("dot-formation", region="stability.map"),), entry="dot-formation")


def test_on_stale_is_a_policy_word():
    """on_stale routes staleness — refresh | park | proceed — it is not a node edge;
    a typo would silently mis-route every stale prerequisite this step has."""
    with pytest.raises(bringup.BringupPlanError, match=r"rabi.*on_stale.*refesh"):
        BringupPlan(steps=(_step("rabi", on_stale="refesh"),), entry="rabi")


# -- validator robustness: refuse, never crash -----------------------------------------------


def test_on_fail_must_be_a_mapping():
    """The validator refuses malformed graphs; it never crashes on them."""
    with pytest.raises(bringup.BringupPlanError, match=r"rabi.*on_fail"):
        BringupPlan(steps=(_step("rabi", on_fail=[(1, "park")]),), entry="rabi")


# -- escalation ladders (AC: every ladder terminates in a park, never a silent stop) --------


def test_ladder_terminal_rung_must_park():
    """A ladder whose max_retries rung routes to another step — not a park — is
    refused: the campaign must park needs_human after max_retries, never escalate on."""
    with pytest.raises(bringup.BringupPlanError, match=r"ramsey.*rung 2.*park"):
        BringupPlan(
            steps=(
                _step("ramsey", on_fail={1: "tls-scan", 2: "tls-scan"}, max_retries=2),
                _step("tls-scan", on_pass="calibrated"),
            ),
            entry="ramsey",
        )


def test_ladder_silent_stop_rejected():
    """A retry budget with unrouteable failures is refused — a failure with no rung is
    the silent stop the spec forbids. Covers a missing first rung and an empty ladder."""
    with pytest.raises(bringup.BringupPlanError, match=r"ramsey.*rung 1"):
        BringupPlan(
            steps=(_step("ramsey", on_fail={2: "park"}, max_retries=2),),
            entry="ramsey",
        )
    with pytest.raises(bringup.BringupPlanError, match=r"ramsey.*on_fail"):
        BringupPlan(steps=(_step("ramsey", on_fail={}, max_retries=2),), entry="ramsey")


def test_ladder_rung_beyond_max_retries_rejected():
    """Rungs past max_retries are unreachable — the ladder covers exactly
    rungs 1..max_retries and its terminal parks."""
    with pytest.raises(bringup.BringupPlanError, match=r"ramsey.*rung 3"):
        BringupPlan(
            steps=(_step("ramsey", on_fail={1: "park", 2: "park", 3: "park"}, max_retries=2),),
            entry="ramsey",
        )


def test_ladder_terminal_park_must_be_needs_human():
    """The terminal park resolves needs_human ('terminates in a needs_human park after
    max_retries'); a propose park as the terminal rung is refused. Bare 'park' and
    'park:needs_human' are the same park and accepted."""
    with pytest.raises(bringup.BringupPlanError, match=r"ramsey.*propose"):
        BringupPlan(
            steps=(_step("ramsey", on_fail={1: "park:propose"}, max_retries=1),),
            entry="ramsey",
        )
    BringupPlan(steps=(_step("ramsey", on_fail={1: "park"}, max_retries=1),), entry="ramsey")
    BringupPlan(
        steps=(_step("ramsey", on_fail={1: "park:needs_human"}, max_retries=1),),
        entry="ramsey",
    )


def test_ladder_rung_keys_must_be_retry_ordinals():
    with pytest.raises(bringup.BringupPlanError, match=r"ramsey.*2\.5"):
        BringupPlan(
            steps=(_step("ramsey", on_fail={1: "park", 2.5: "park"}, max_retries=2),),
            entry="ramsey",
        )


def test_max_retries_must_be_a_positive_retry_budget():
    with pytest.raises(bringup.BringupPlanError, match=r"ramsey.*max_retries"):
        BringupPlan(steps=(_step("ramsey", max_retries=0),), entry="ramsey")


# -- edge existence (AC: on_pass/on_fail edges to nonexistent or waived nodes rejected) ---


def test_on_pass_to_nonexistent_node_rejected():
    with pytest.raises(bringup.BringupPlanError, match=r"sensor-turn-on.*on_pass.*ghost-node"):
        BringupPlan(steps=(_step("sensor-turn-on", on_pass="ghost-node"),), entry="sensor-turn-on")


def test_on_fail_to_nonexistent_node_rejected():
    with pytest.raises(bringup.BringupPlanError, match=r"ramsey.*on_fail.*tls-ghost"):
        BringupPlan(
            steps=(_step("ramsey", on_fail={1: "tls-ghost", 2: "park"}, max_retries=2),),
            entry="ramsey",
        )


def test_edge_to_waived_node_rejected():
    """A target naming a waived node is rejected *as waived* — the spec's waiver list
    (e.g. the joint 2D readout revisit, C8) is authoring data, and no v1 edge may
    route to it."""
    with pytest.raises(bringup.BringupPlanError, match=r"readout-state.*joint-2d-revisit.*waived"):
        BringupPlan(
            steps=(_step("readout-state", on_pass="joint-2d-revisit"),),
            entry="readout-state",
            waived=frozenset({"joint-2d-revisit"}),
        )


def test_entry_must_name_a_step():
    with pytest.raises(bringup.BringupPlanError, match=r"entry.*ghost-entry"):
        BringupPlan(steps=(_step("sensor-turn-on"),), entry="ghost-entry")


def test_park_targets_are_legal_edges():
    """Park targets — bare and kinded — and the success terminal are the reserved
    non-step edge vocabulary; a plan using them constructs."""
    step = _step("ramsey", on_fail={1: "park:propose", 2: "park:needs_human"}, max_retries=2)
    plan = BringupPlan(steps=(step,), entry="ramsey")
    assert plan.entry == "ramsey"


def test_reserved_step_names_rejected():
    """A step may not take the name of reserved edge vocabulary (a park target or the
    success terminal) — the grammar would be ambiguous."""
    with pytest.raises(bringup.BringupPlanError, match=r"park:custom.*reserved"):
        BringupPlan(steps=(_step("park:custom"),), entry="park:custom")
    with pytest.raises(bringup.BringupPlanError, match=r"calibrated.*reserved"):
        BringupPlan(steps=(_step("calibrated"),), entry="calibrated")


def test_duplicate_step_names_rejected():
    """Two steps sharing a name make every edge to it ambiguous — refused."""
    with pytest.raises(bringup.BringupPlanError, match=r"duplicate step name.*sensor-turn-on"):
        BringupPlan(
            steps=(_step("sensor-turn-on"), _step("sensor-turn-on")),
            entry="sensor-turn-on",
        )


# -- set -> refine ordering (AC: refine-before-set rejected) --------------------------------


def test_refine_without_set_writer_rejected():
    """A refine-writer with no set-writer on its path is refused — there is nothing to
    refine; the first writer of a path must declare kind='set'."""
    with pytest.raises(bringup.BringupPlanError, match=r"resonator-refind.*readout\.frequency"):
        BringupPlan(
            steps=(
                _step(
                    "resonator-refind",
                    provides=(
                        Provides(path="readout.frequency", kind="refine", mapping={"f": "f"}),
                    ),
                ),
            ),
            entry="resonator-refind",
        )


def test_refine_before_set_writer_rejected():
    """A refine-writer upstream of its set-writer is refused: set→refine ordering is
    enforced on the *graph* — the set-writer must reach the refine-writer, else the
    refine can run against a path never set. (A beside-the-set refine — reachable
    from entry via an on_fail detour the set-writer never passes through — is the
    same failure.)"""
    set_late = _step(
        "resonator-find",
        provides=(Provides(path="readout.frequency", mapping={"f": "f"}),),
        on_pass="calibrated",
    )
    refine_first = _step(
        "resonator-refind",
        provides=(Provides(path="readout.frequency", kind="refine", mapping={"f": "f"}),),
        on_pass="resonator-find",
    )
    with pytest.raises(bringup.BringupPlanError, match=r"resonator-refind.*readout\.frequency") as ei:
        BringupPlan(steps=(refine_first, set_late), entry="resonator-refind")
    assert "resonator-find" in str(ei.value)  # the set-writer it outruns is named

    # beside-the-set: entry passes to the set-writer, but a failure detour reaches the
    # refine-writer without ever passing through it
    entry = _step("sensor-turn-on", on_pass="resonator-find", on_fail={1: "resonator-refind", 2: "park"}, max_retries=2)
    with pytest.raises(bringup.BringupPlanError, match="resonator-refind"):
        BringupPlan(
            steps=(
                entry,
                _step("resonator-find", provides=(Provides(path="readout.frequency", mapping={"f": "f"}),), on_pass="calibrated"),
                _step("resonator-refind", provides=(Provides(path="readout.frequency", kind="refine", mapping={"f": "f"}),), on_pass="calibrated"),
            ),
            entry="sensor-turn-on",
        )


def test_provides_kind_vocabulary():
    """A provides kind outside {set, refine} is refused — a typo'd kind is invisible to
    both write-exclusivity checks and would silently write an undisciplined path."""
    with pytest.raises(bringup.BringupPlanError, match=r"punchout.*readout\.power.*sets"):
        BringupPlan(
            steps=(_step("punchout", provides=(Provides(path="readout.power", kind="sets", mapping={"p": "power"}),)),),
            entry="punchout",
        )


# -- write exclusivity: one set-writer per path (AC: duplicate set-writers rejected) --------


def test_duplicate_set_writers_rejected():
    """Two steps both declaring kind="set" on the same calibration path are refused at
    construction — the path would have two first-writers, and "last wins" is a refine
    discipline, not a set one. The error names both steps and the path."""
    a = _step(
        "resonator-find",
        provides=(Provides(path="readout.frequency", mapping={"f_res": "frequency"}),),
        on_pass="qubit-spec",
    )
    mid = _step("qubit-spec", on_pass="resonator-refind")
    b = _step(
        "resonator-refind",
        provides=(Provides(path="readout.frequency", mapping={"f_res": "frequency"}),),
        on_pass="calibrated",
    )

    with pytest.raises(bringup.BringupPlanError, match=r"readout\.frequency") as ei:
        BringupPlan(steps=(a, mid, b), entry="resonator-find")
    assert "resonator-find" in str(ei.value) and "resonator-refind" in str(ei.value)


def test_intra_step_duplicate_provides_path_rejected():
    """One step writing the same path twice in its own provides tuple is refused — a
    single writeback snapshot cannot carry two writes of one path. The error names the
    offending step and the path."""
    with pytest.raises(bringup.BringupPlanError, match=r"punchout.*readout\.power"):
        BringupPlan(
            steps=(
                _step(
                    "punchout",
                    provides=(
                        Provides(path="readout.power", mapping={"p_op": "power"}),
                        Provides(path="readout.power", mapping={"p_op2": "power"}),
                    ),
                ),
            ),
            entry="punchout",
        )


def test_write_exclusivity_is_per_full_calibration_path():
    """Exclusivity and the set→refine ordering are per FULL calibration path — the one the
    writeback lands (``path.subfield``), the same path the snapshot and the freshness
    requirements name. Two steps may each be the first writer of DIFFERENT leaves under one
    calibration parent (a compound writeback's set + refine + set, e.g. Ramsey's detuning
    correction, refined qubit frequency and T2 under one parent), while a second writer of
    the SAME leaf must declare kind='refine'."""
    setter = _step(
        "qubit-spec",
        provides=(Provides(path="qubit", mapping={"f_spec": "f_ge"}),),
        on_pass="ramsey",
    )
    compound = _step(
        "ramsey",
        provides=(
            Provides(path="qubit", mapping={"detuning": "freq_correction"}),
            Provides(path="qubit", mapping={"freq_refine": "f_ge"}, kind="refine"),
            Provides(path="qubit", mapping={"T2": "T2"}),
        ),
        on_pass="calibrated",
    )
    plan = BringupPlan(steps=(setter, compound), entry="qubit-spec")
    assert [s.name for s in plan.steps] == ["qubit-spec", "ramsey"]

    # the same-leaf discipline still holds: a second SET writer of one leaf is refused
    second = _step(
        "t2-again",
        provides=(Provides(path="qubit", mapping={"tau": "T2"}),),
    )
    with pytest.raises(bringup.BringupPlanError, match=r"qubit\.T2"):
        BringupPlan(steps=(setter, compound, second), entry="qubit-spec")


# -- purity (AC invariant: core enforces generically — no qick, no pack symbols, ever) --------


def test_module_imports_without_qick():
    code = "import sys; sys.modules['qick']=None; import strumento.core.bringup; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_module_imports_no_pack_modules():
    """The pack namespace is blocked outright: any ``strumento.packs.*`` import — even
    transitively — fails the module's import."""
    code = (
        "import sys, types\n"
        "sys.modules['qick'] = None\n"
        "sys.modules['strumento.packs'] = types.ModuleType('strumento.packs')\n"
        "import strumento.core.bringup\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_module_names_no_pack_symbol():
    """Core enforces generically and branches on no pack symbol (D18): the module's
    own text may not so much as name a pack — not an import, not a string constant,
    not a branch. Naming one in prose is exactly the smell this test exists for."""
    import re

    source = BRINGUP_PY.read_text()
    assert "strumento.packs" not in source
    for word in ("cqed", "tweezer", "spin"):
        assert not re.search(rf"\b{word}\b", source), f"pack symbol named: {word!r}"
