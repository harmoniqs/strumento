"""The bring-up writeback policy — one snapshot per pass, bounds refused never clamped (issue #54).

New surface per the issue's Testing Decisions; the runner fixtures are REUSED from
``tests.core.test_bringup_runner`` (the flagged cross-slice edge: the runner tests own
``ScriptedExperiment`` and the campaign/runner helpers — these tests import them rather than
redefining, so a runner fixture rename breaks here loudly, by import).

The writeback contract under test: a passing step's multi-path ``provides`` lands in ONE
``record_calibration`` snapshot stamped ``source=f"{campaign}:{step}"`` with the refinement
level in provenance; undeclared paths raise; ``max_abs``/``max_slew`` violations are refused
(never clamped); host-bridge sweep extents beyond a declared envelope bound raise before
anything runs; artifact-backed paths round-trip through a content reference vetted readers
resolve.
"""

from __future__ import annotations

import pytest

from strumento.core.bringup import (
    BringupError,
    BringupPlan,
    Provides,
)
from strumento.core.config import DeviceConfig
from strumento.core.store import FileConfigStore
from tests.core.test_bringup_runner import (
    ScriptedExperiment,
    _campaign,
    _runner,
    _step,
)


def _base_config() -> DeviceConfig:
    """A minimal spin-shaped calibration tree (core tests name no pack module — the tree is
    plain data with the parent dicts the dotted writeback paths descend through)."""
    return DeviceConfig(
        pack="spin",
        topology=[],
        wiring={},
        calibration={
            "sensor": {"carrier_f": 5.0e9, "demod_phase": 0.0, "bias": 0.0},
            "readout": {"max_power": -30.0},
            "stability": {},
            "dot": {},
            "psb": {},
        },
    )


def _store_with_ref(tmp_path, ref: str = "main") -> FileConfigStore:
    store = FileConfigStore(tmp_path / "cfgstore")
    cfg = _base_config()
    store.put(cfg)
    store.set_ref(ref, cfg.content_id())
    return store


def _derived_factory(derived: dict):
    """A step factory whose sanctioned experiment carries the given derived quantities —
    the writeback's value source."""

    def make(device, ctx):
        return ScriptedExperiment(device, success=True, derived=derived)

    return make


SENSOR_TURNOVER_PROVIDES = (
    Provides(path="sensor", mapping={"f": "carrier_f", "ph": "demod_phase"}),
    Provides(path="readout", mapping={"p": "max_power"}),
)


# -- one snapshot per pass -------------------------------------------------------------------------


def test_multi_path_writeback_lands_in_one_snapshot_with_the_campaign_step_stamp(tmp_path):
    """AC: a passing step's multi-path writeback lands in ONE calibration snapshot — every
    provides path in a single ``record_calibration`` call — stamped
    ``source=f"{campaign}:{step}"``, with the refinement level recorded in provenance, the
    ref advanced, and the campaign ledger cross-linked to the new content id."""
    store = _store_with_ref(tmp_path)
    plan = BringupPlan(
        steps=(
            _step(
                "sensor-turn-on",
                provides=SENSOR_TURNOVER_PROVIDES,
                experiment=_derived_factory({"f": 5.001e9, "ph": 12.5, "p": -27.0}),
            ),
        ),
        entry="sensor-turn-on",
    )
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out = runner.advance()

    assert out.passed is True
    log = store.log("main")
    assert len(log) == 1  # ONE write-back call — one snapshot, not one per path
    entry = log[0]
    assert sorted(entry.updates) == [
        "readout.max_power",
        "sensor.carrier_f",
        "sensor.demod_phase",
    ]
    assert entry.updates["sensor.carrier_f"] == [5.0e9, 5.001e9]  # the value diff, old -> new
    assert entry.source == f"{campaign.record.dir.name}:sensor-turn-on"
    assert entry.timestamp is not None  # the freshness engine ages from this stamp
    assert entry.extra["refinement"] == {
        "sensor.carrier_f": "set",
        "sensor.demod_phase": "set",
        "readout.max_power": "set",
    }
    assert store.resolve_ref("main") == entry.content_id  # the ref advanced to the snapshot
    written = store.get(entry.content_id)
    assert written.calibration["sensor"]["carrier_f"] == 5.001e9
    assert written.calibration["readout"]["max_power"] == -27.0

    # the campaign ledger cross-links the calibration move (the substrate's calibration event)
    from strumento.core.record import read_events

    cal_events = [
        e for e in read_events(campaign.record.dir) if e.get("ev") == "calibration"
    ]
    assert len(cal_events) == 1 and cal_events[0]["content_id"] == entry.content_id


def test_a_refine_writeback_records_the_refinement_level(tmp_path):
    """AC (provenance records the refinement level): a ``kind="refine"`` provides lands in
    provenance marked refine — set→refine is visible in the log, not just the graph."""
    store = _store_with_ref(tmp_path)
    plan = BringupPlan(
        steps=(
            _step(
                "sensor-turn-on",
                provides=(Provides(path="sensor", mapping={"f": "carrier_f"}),),
                experiment=_derived_factory({"f": 5.001e9}),
                on_pass="sensor-refine",
            ),
            _step(
                "sensor-refine",
                provides=(Provides(path="sensor", mapping={"f": "carrier_f"}, kind="refine"),),
                experiment=_derived_factory({"f": 5.0005e9}),
            ),
        ),
        entry="sensor-turn-on",
    )
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    runner.advance()
    runner.advance()

    log = store.log("main")
    assert [e.extra["refinement"]["sensor.carrier_f"] for e in log] == ["set", "refine"]
    assert store.get(store.resolve_ref("main")).calibration["sensor"]["carrier_f"] == 5.0005e9


def test_a_providing_step_without_a_store_is_refused_loudly(tmp_path):
    """A step that declares provides cannot pass silently un-recorded: a runner with no
    store wired refuses before any attempt is opened — a misconfiguration, not a data point."""
    plan = BringupPlan(
        steps=(
            _step(
                "sensor-turn-on",
                provides=(Provides(path="sensor", mapping={"f": "carrier_f"}),),
                experiment=_derived_factory({"f": 5.001e9}),
            ),
        ),
        entry="sensor-turn-on",
    )
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign)  # no store: the writeback has nowhere to land

    with pytest.raises(BringupError, match="store"):
        runner.advance()
    state = campaign.derive_state()
    assert state.position == "sensor-turn-on" and state.attempts["sensor-turn-on"] == 0
    steps_dir = campaign.record.dir / "steps"
    assert not steps_dir.exists() or not any(steps_dir.iterdir())  # nothing ran


# -- bounds are refused, never clamped -------------------------------------------------------------


def test_writeback_endpoint_beyond_max_abs_is_refused_never_clamped(tmp_path):
    """AC: a writeback beyond a declared ``max_abs`` endpoint is refused — never clamped.
    Nothing lands in the store, the config value stays as it was, and the refused attempt
    derives as a failed attempt (a pass the store never recorded is never routed)."""
    store = _store_with_ref(tmp_path)
    plan = BringupPlan(
        steps=(
            _step(
                "bias-set",
                provides=(Provides(path="sensor", mapping={"b": "bias"}, max_abs=1.0),),
                experiment=_derived_factory({"b": 2.5}),
            ),
        ),
        entry="bias-set",
    )
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    with pytest.raises(BringupError, match="max_abs"):
        runner.advance()

    assert store.log("main") == []  # nothing was written
    assert store.get(store.resolve_ref("main")).calibration["sensor"]["bias"] == 0.0
    state = campaign.derive_state()
    assert state.attempts["bias-set"] == 1  # the refusal derives as a failed attempt
    assert state.parked is not None and state.parked.kind == "needs_human"  # rung 1


def test_writeback_within_max_abs_lands_exactly_never_clamped(tmp_path):
    """The complement: a value inside the bound lands EXACTLY as measured — the runner
    never reshapes a value to fit a bound."""
    store = _store_with_ref(tmp_path)
    plan = BringupPlan(
        steps=(
            _step(
                "bias-set",
                provides=(Provides(path="sensor", mapping={"b": "bias"}, max_abs=1.0),),
                experiment=_derived_factory({"b": 0.937}),
            ),
        ),
        entry="bias-set",
    )
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out = runner.advance()

    assert out.passed is True
    cfg = store.get(store.resolve_ref("main"))
    assert cfg.calibration["sensor"]["bias"] == 0.937  # exact — no clamping, no rounding


def test_writeback_slew_beyond_max_slew_is_refused(tmp_path):
    """AC: a writeback whose move exceeds the declared ``max_slew`` (against the current
    config value at that path) is refused — never clamped, nothing written."""
    store = _store_with_ref(tmp_path)
    from strumento.core.store import record_calibration, unvalidated

    # the base already holds bias = 0.5 (an earlier, honest write; a synthetic config)
    base = store.get(store.resolve_ref("main"))
    record_calibration(store, "main", base, {"sensor.bias": 0.5}, source="prior fit",
                       validate=unvalidated)

    plan = BringupPlan(
        steps=(
            _step(
                "bias-refine",
                provides=(
                    Provides(path="sensor", mapping={"b": "bias"}, max_slew=0.2),
                ),
                experiment=_derived_factory({"b": 0.9}),  # slew 0.4 > 0.2
            ),
        ),
        entry="bias-refine",
    )
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    with pytest.raises(BringupError, match="max_slew"):
        runner.advance()

    log = store.log("main")
    assert len(log) == 1  # only the prior write — the refused slew added nothing
    assert store.get(store.resolve_ref("main")).calibration["sensor"]["bias"] == 0.5


def test_writeback_slew_within_bound_lands(tmp_path):
    """The complement: a move inside the declared slew bound lands."""
    store = _store_with_ref(tmp_path)
    from strumento.core.store import record_calibration, unvalidated

    base = store.get(store.resolve_ref("main"))
    record_calibration(store, "main", base, {"sensor.bias": 0.5}, source="prior fit",
                       validate=unvalidated)  # a synthetic, non-composable config

    plan = BringupPlan(
        steps=(
            _step(
                "bias-refine",
                provides=(
                    Provides(path="sensor", mapping={"b": "bias"}, max_slew=0.2),
                ),
                experiment=_derived_factory({"b": 0.68}),  # slew 0.18 <= 0.2
            ),
        ),
        entry="bias-refine",
    )
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out = runner.advance()

    assert out.passed is True
    assert store.get(store.resolve_ref("main")).calibration["sensor"]["bias"] == 0.68


# -- host-bridge sweep extents ----------------------------------------------------------------------


class _SweepBridge:
    """A registered host-bridge adapter that declares its planned sweep extents — the
    pre-run surface the runner checks against the step's declared envelope."""

    def __init__(self, extents, *, verdicts=None):
        self.sweep_extents = dict(extents)
        self.verdicts = dict(verdicts or {"demod-balance": True})
        self.ran = False

    def run(self, *, record=None, **kw):
        self.ran = True
        return self.verdicts


def _bridged_plan(bridge, **step_kw):
    return BringupPlan(
        steps=(_step("dc-nudge", gates=("demod-balance",), experiment=lambda d, c: bridge,
                     **step_kw),),
        entry="dc-nudge",
    )


def test_sweep_extent_beyond_declared_bound_raises_before_running(tmp_path):
    """AC: a host-bridge sweep extent beyond its declared envelope bound raises — BEFORE
    the sweep runs (the adapter never executed, no attempt opened, nothing burned)."""
    from strumento.core import bringup

    bridge = _SweepBridge({"gate_v": (-3.0, 3.0)})
    bringup.register_host_bridge(_SweepBridge)
    try:
        plan = _bridged_plan(bridge, envelope={"gate_v": 1.0})
        campaign = _campaign(tmp_path, plan)
        runner = _runner(campaign)

        with pytest.raises(BringupError, match="gate_v.*beyond.*1"):
            runner.advance()

        assert bridge.ran is False  # refused before the sweep touched anything
        state = campaign.derive_state()
        assert state.attempts["dc-nudge"] == 0  # no attempt burned
        steps_dir = campaign.record.dir / "steps"
        assert not steps_dir.exists() or not any(steps_dir.iterdir())
    finally:
        bringup._HOST_BRIDGES.remove(_SweepBridge)


def test_sweep_extent_on_an_undeclared_channel_is_refused(tmp_path):
    """An extent on a channel the envelope never declared is refused too — an undeclared
    sweep channel is unbounded excitation, never run."""
    from strumento.core import bringup

    bridge = _SweepBridge({"gate_v": (-0.5, 0.5), "stub_v": (-0.1, 0.1)})
    bringup.register_host_bridge(_SweepBridge)
    try:
        plan = _bridged_plan(bridge, envelope={"gate_v": 1.0})
        campaign = _campaign(tmp_path, plan)
        runner = _runner(campaign)

        with pytest.raises(BringupError, match="stub_v.*does not declare"):
            runner.advance()
        assert bridge.ran is False
    finally:
        bringup._HOST_BRIDGES.remove(_SweepBridge)


def test_sweep_extent_within_the_envelope_runs(tmp_path):
    """The complement: extents inside the declared envelope run normally and route."""
    from strumento.core import bringup

    bridge = _SweepBridge({"gate_v": (-0.8, 0.9)})
    bringup.register_host_bridge(_SweepBridge)
    try:
        plan = _bridged_plan(bridge, envelope={"gate_v": 1.0})
        campaign = _campaign(tmp_path, plan)
        runner = _runner(campaign)

        out = runner.advance()

        assert out.passed is True and bridge.ran is True
        assert campaign.derive_state().terminal == "calibrated"
    finally:
        bringup._HOST_BRIDGES.remove(_SweepBridge)



def test_an_unresolvable_ref_aborts_the_writeback(tmp_path):
    """The writeback reads its base from the claimed ref; a ref that resolves to nothing is
    refused loudly (nothing partial written), and the open attempt derives as failed."""
    store = FileConfigStore(tmp_path / "cfgstore")  # a store with NO ref set
    plan = BringupPlan(
        steps=(
            _step(
                "sensor-turn-on",
                provides=(Provides(path="sensor", mapping={"f": "carrier_f"}),),
                experiment=_derived_factory({"f": 5.001e9}),
            ),
        ),
        entry="sensor-turn-on",
    )
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    with pytest.raises(BringupError, match="main"):
        runner.advance()
    state = campaign.derive_state()
    assert state.attempts["sensor-turn-on"] == 1  # crash semantics: refused writeback = failed attempt
    assert store.log("main") == []  # nothing was written


# -- artifact-backed paths round-trip --------------------------------------------------------------


def test_artifact_backed_paths_round_trip_through_the_content_reference(tmp_path):
    """AC: an artifact-backed provides (stability map, dot compensation — 2D data) banks its
    content in the child record and the calibration path holds a CONTENT REFERENCE that
    vetted readers resolve back to the banked bytes; the child ledger records the artifact."""
    from strumento.core.bringup import resolve_calibration_artifact

    stability_map = {
        "extent": {"bounds": [[0.0, 1.0], [0.0, 1.0]]},  # the generic spatial predicate data
        "grid": [[0.1, 0.4, 0.9], [0.2, 0.5, 0.8]],
    }
    compensation = {"offsets": [[12.5, -3.25], [0.0, 44.0]]}
    store = _store_with_ref(tmp_path)
    plan = BringupPlan(
        steps=(
            _step(
                "stability-map",
                provides=(
                    Provides(path="stability", mapping={"map": "map"}, artifact=True),
                    Provides(path="dot", mapping={"comp": "compensation"}, artifact=True),
                ),
                experiment=_derived_factory({"map": stability_map, "comp": compensation}),
            ),
        ),
        entry="stability-map",
    )
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out = runner.advance()

    assert out.passed is True
    cfg = store.get(store.resolve_ref("main"))
    map_ref = cfg.calibration["stability"]["map"]
    comp_ref = cfg.calibration["dot"]["compensation"]
    assert isinstance(map_ref, str) and map_ref.startswith("artifact:")
    assert isinstance(comp_ref, str) and comp_ref.startswith("artifact:")

    # a vetted reader resolves the reference back to the banked content, byte-checked
    assert resolve_calibration_artifact(tmp_path, cfg, "stability.map") == stability_map
    assert resolve_calibration_artifact(tmp_path, cfg, "dot.compensation") == compensation

    # the 2D bytes live in the child record, recorded as artifacts in its ledger
    from strumento.core.record import read_events

    children = list((campaign.record.dir / "steps").iterdir())
    assert len(children) == 1
    ledger = read_events(children[0])
    artifact_events = [e for e in ledger if e.get("ev") == "artifact"]
    assert len(artifact_events) == 2
    assert any(e.get("kind") == "stability.map" for e in artifact_events)
    assert any(e.get("kind") == "dot.compensation" for e in artifact_events)


def test_a_tampered_artifact_is_refused_at_resolution(tmp_path):
    """The content reference carries the banked bytes' digest: a file edited after banking
    does not resolve — the reader refuses bytes that no longer match what was banked."""
    from strumento.core.bringup import resolve_calibration_artifact

    store = _store_with_ref(tmp_path)
    plan = BringupPlan(
        steps=(
            _step(
                "stability-map",
                provides=(Provides(path="stability", mapping={"map": "map"}, artifact=True),),
                experiment=_derived_factory({"map": {"extent": {"bounds": [[0, 1], [0, 1]]}}}),
            ),
        ),
        entry="stability-map",
    )
    campaign = _campaign(tmp_path, plan, ref="main")
    _runner(campaign, store=store).advance()

    cfg = store.get(store.resolve_ref("main"))
    ref = cfg.calibration["stability"]["map"]
    child = next((campaign.record.dir / "steps").iterdir())
    rel = ref.split("@", 1)[0][len("artifact:"):].split("/", 3)[3]
    banked = child / rel
    assert banked.exists()
    banked.write_text('{"tampered": true}')  # edited after banking

    with pytest.raises(BringupError, match="no longer match"):
        resolve_calibration_artifact(tmp_path, cfg, "stability.map")


# -- an undeclared path in a writeback raises ------------------------------------------------------


class _CalibratingBridge:
    """A registered host-bridge adapter returning verdicts PLUS an explicit calibration
    payload — the emitter-authored writeback surface."""

    def __init__(self, calibration, *, verdicts=None):
        self.calibration = dict(calibration)
        self.verdicts = dict(verdicts or {"demod-balance": True})

    def run(self, *, record=None, **kw):
        return {**self.verdicts, "calibration": self.calibration}


def test_an_undeclared_path_in_a_writeback_raises(tmp_path):
    """AC: a writeback carrying a path the step's provides do not declare is refused —
    loudly, by path, with nothing written. The bridge's explicit payload carries one
    declared path and one rogue one; the rogue one refuses the whole writeback."""
    from strumento.core import bringup

    bridge = _CalibratingBridge({
        "sensor.carrier_f": 5.001e9,   # declared below — fine
        "sensor.sneaky_gain": 8192.0,  # declared NOWHERE — the rogue write
    })
    bringup.register_host_bridge(_CalibratingBridge)
    try:
        store = _store_with_ref(tmp_path)
        plan = BringupPlan(
            steps=(
                _step(
                    "sensor-turn-on",
                    gates=("demod-balance",),
                    provides=(Provides(path="sensor", mapping={"f": "carrier_f"}),),
                    experiment=lambda d, c: bridge,
                ),
            ),
            entry="sensor-turn-on",
        )
        campaign = _campaign(tmp_path, plan, ref="main")
        runner = _runner(campaign, store=store)

        with pytest.raises(BringupError, match="sensor.sneaky_gain"):
            runner.advance()

        assert store.log("main") == []  # nothing was written — not even the declared path
        state = campaign.derive_state()
        assert state.attempts["sensor-turn-on"] == 1  # the refusal derives as a failed attempt
    finally:
        bringup._HOST_BRIDGES.remove(_CalibratingBridge)


def test_a_declared_explicit_payload_lands_in_the_snapshot(tmp_path):
    """The complement: an explicit payload whose paths are all declared lands — the
    emitter's precise values beat the derived projection."""
    from strumento.core import bringup

    bridge = _CalibratingBridge({"sensor.carrier_f": 5.002e9})
    bringup.register_host_bridge(_CalibratingBridge)
    try:
        store = _store_with_ref(tmp_path)
        plan = BringupPlan(
            steps=(
                _step(
                    "sensor-turn-on",
                    gates=("demod-balance",),
                    provides=(Provides(path="sensor", mapping={"f": "carrier_f"}),),
                    experiment=lambda d, c: bridge,
                ),
            ),
            entry="sensor-turn-on",
        )
        campaign = _campaign(tmp_path, plan, ref="main")
        runner = _runner(campaign, store=store)

        out = runner.advance()

        assert out.passed is True
        cfg = store.get(store.resolve_ref("main"))
        assert cfg.calibration["sensor"]["carrier_f"] == 5.002e9
    finally:
        bringup._HOST_BRIDGES.remove(_CalibratingBridge)
