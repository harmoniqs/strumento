"""The bring-up freshness engine — three poisoning triggers, age as the backstop (issue #54).

New surface per the issue's Testing Decisions; fixtures are reused across the freshness seam:
``tests.core.test_bringup_runner`` owns the sanctioned-experiment fixtures, and the
writeback module (``tests.core.test_bringup_writeback``) owns the store/config helpers —
the flagged cross-slice edges. A store wired into the runner is what turns the engine on
(the storeless runner is never stale, by construction).

Freshness under test: a prerequisite is stale when (1) older than its ``max_age_s``;
(2) value-poisoned — a writeback changed a path this step transitively requires, regardless
of age; (3) failure-poisoned — a failed step's ``poisons`` are forced stale before its
retry; or (4) excursion-poisoned — a DC step's moves left the cleared region in which a
drift-sensitive artifact was acquired (in-region moves do NOT poison). Stale handling
routes per ``on_stale``: a refresh detour consumes the advance and inherits the providing
step's ladder; ``park`` holds needs_human; ``proceed`` is refused for drift-sensitive
paths.
"""

from __future__ import annotations

import pytest

from strumento.core.bringup import BringupError, BringupPlan, Provides, Requirement
from strumento.core.store import record_calibration, unvalidated
from tests.core.test_bringup_runner import _campaign, _runner, _step
from tests.core.test_bringup_writeback import _derived_factory, _store_with_ref

#: a timestamp far enough in the past that any declared max_age_s has lapsed
ANCIENT = "2020-01-01T00:00:00Z"


def _two_step_plan(*, on_stale="refresh", max_age_s=3600.0, drift_sensitive=False):
    """coulomb(entry) requires sensor.carrier_f from sensor-turn-on, which hands control
    back on pass — the minimal refresh-detour graph."""
    return BringupPlan(
        steps=(
            _step(
                "coulomb-peak",
                requires=(
                    Requirement(
                        path="sensor.carrier_f",
                        max_age_s=max_age_s,
                        drift_sensitive=drift_sensitive,
                    ),
                ),
                on_stale=on_stale,
                on_pass="calibrated",
                on_fail={1: "park"},
                max_retries=1,
            ),
            _step(
                "sensor-turn-on",
                provides=(Provides(path="sensor", mapping={"f": "carrier_f"}),),
                experiment=_derived_factory({"f": 5.001e9}),
                on_pass="coulomb-peak",
                on_fail={1: "park"},
                max_retries=1,
            ),
        ),
        entry="coulomb-peak",
    )


def _old_write(store, tmp_path, *, path="sensor.carrier_f", value=5.001e9,
               step="sensor-turn-on", timestamp=ANCIENT):
    """A write the store's public API made long ago (the engine reads provenance — it never
    cares WHO wrote, only what and when)."""
    base = store.get(store.resolve_ref("main"))
    record_calibration(
        store, "main", base, {path: value},
        source=f"{tmp_path.name}:{step}", timestamp=timestamp,
        validate=unvalidated,  # a synthetic, non-composable config: the store mechanics only
    )


def test_age_stale_requirement_detours_to_its_provider(tmp_path):
    """AC: a prerequisite older than its ``max_age_s`` is stale (the backstop trigger), and
    a stale requirement routes ``on_stale`` — default refresh — as a DETOUR to the step
    that provides it."""
    store = _store_with_ref(tmp_path)
    _old_write(store, tmp_path)  # sensor.carrier_f written ANCIENT — max_age long lapsed
    plan = _two_step_plan()
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out = runner.advance()

    assert out.step == "sensor-turn-on"  # the detour, not the position step
    assert out.detour is True and out.passed is True
    assert out.attempt == 1  # under the PROVIDER's own ladder
    state = campaign.derive_state()
    assert state.position == "coulomb-peak"  # the provider passed control back
    assert state.attempts["coulomb-peak"] == 0  # the main step never burned an attempt

    # the detour re-provided fresh (a recent timestamp in provenance)
    last = store.log("main")[-1]
    assert last.updates["sensor.carrier_f"][1] == 5.001e9
    assert last.timestamp != ANCIENT

    out2 = runner.advance()  # fresh now — the main step runs
    assert out2.step == "coulomb-peak" and out2.detour is False and out2.passed is True
    assert campaign.derive_state().terminal == "calibrated"


def test_a_refresh_detour_consumes_the_call_and_the_main_step_runs_next(tmp_path):
    """AC: a refresh detour consumes the advance call — the main step does NOT run in the
    same call (no attempt opened for it), and runs on the next advance."""
    from strumento.core.record import read_events

    store = _store_with_ref(tmp_path)
    _old_write(store, tmp_path)
    plan = _two_step_plan()
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out = runner.advance()  # the detour consumed this call

    assert out.detour is True
    opens = [
        e for e in read_events(campaign.record.dir)
        if e.get("ev") == "step_open" and e.get("step") == "coulomb-peak"
    ]
    assert opens == []  # the main step did not run — not even opened
    assert campaign.derive_state().position == "coulomb-peak"

    out2 = runner.advance()  # the main step, on the next call
    assert out2.step == "coulomb-peak" and out2.detour is False and out2.attempt == 1


def test_a_fresh_requirement_does_not_detour(tmp_path):
    """The complement: a recent write inside ``max_age_s`` runs the main step directly."""
    store = _store_with_ref(tmp_path)
    from strumento.core.record import now_iso

    _old_write(store, tmp_path, timestamp=now_iso())  # fresh, by the public API
    plan = _two_step_plan()
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out = runner.advance()

    assert out.step == "coulomb-peak" and out.detour is False and out.passed is True


# -- value poisoning: a writeback changed a transitively-required path ------------------------------


def _chain_plan():
    """turn (provides sensor.carrier_f) -> map (provides stability.map, requires
    carrier_f) -> dot (entry-side consumer requiring stability.map, NO age bound) —
    the transitive chain: poisoning turn's output re-stales map's, which re-stales dot."""
    return BringupPlan(
        steps=(
            _step(
                "dot-formation",
                requires=(Requirement(path="stability.map", max_age_s=None),),
                on_pass="calibrated",
                on_fail={1: "park"},
                max_retries=1,
            ),
            _step(
                "stability-map",
                requires=(Requirement(path="sensor.carrier_f", max_age_s=None),),
                provides=(Provides(path="stability", mapping={"m": "map"}),),
                experiment=_derived_factory({"m": 0.42}),
                on_pass="dot-formation",
                on_fail={1: "park"},
                max_retries=1,
            ),
            _step(
                "sensor-turn-on",
                provides=(Provides(path="sensor", mapping={"f": "carrier_f"}),),
                experiment=_derived_factory({"f": 5.001e9}),
                on_pass="stability-map",
                on_fail={1: "park"},
                max_retries=1,
            ),
        ),
        entry="dot-formation",
    )


def test_value_poisoned_requirement_detours_regardless_of_age(tmp_path):
    """AC: a writeback that changed a path this step TRANSITIVELY requires re-stales the
    chain — dot requires stability.map (no age bound, minutes old), but turn's drift
    re-measure changed sensor.carrier_f AFTER stability.map was written, so stability.map
    is value-poisoned and dot detours to the map step."""
    store = _store_with_ref(tmp_path)
    _old_write(store, tmp_path, path="sensor.carrier_f", step="sensor-turn-on",
               timestamp="2026-08-28T00:00:00Z")  # t1: turn wrote carrier_f
    _old_write(store, tmp_path, path="stability.map", value=0.41,
               step="stability-map",
               timestamp="2026-08-28T01:00:00Z")  # t2: map wrote stability.map FROM it
    _old_write(store, tmp_path, path="sensor.carrier_f", step="sensor-turn-on",
               timestamp="2026-08-28T02:00:00Z")  # t3: turn RE-measured — the poison

    plan = _chain_plan()
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out = runner.advance()

    # stability.map is value-poisoned (its input changed after it), age boundlessly fresh
    assert out.step == "stability-map" and out.detour is True
    state = campaign.derive_state()
    assert state.position == "dot-formation"  # the map handed control back
    assert state.attempts["dot-formation"] == 0

    out2 = runner.advance()  # the chain re-ran root-first; dot is fresh now
    assert out2.step == "dot-formation" and out2.detour is False and out2.passed is True


def test_an_undisturbed_chain_does_not_detour(tmp_path):
    """The control: no upstream write after the chain's outputs — nothing is stale, the
    consumer runs directly even with the same (unbounded) age contract."""
    store = _store_with_ref(tmp_path)
    _old_write(store, tmp_path, path="sensor.carrier_f", step="sensor-turn-on",
               timestamp="2026-08-28T00:00:00Z")
    _old_write(store, tmp_path, path="stability.map", value=0.41,
               step="stability-map", timestamp="2026-08-28T01:00:00Z")

    plan = _chain_plan()
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out = runner.advance()

    assert out.step == "dot-formation" and out.detour is False and out.passed is True


# -- failure poisoning: a failed step's poisons are forced stale ------------------------------------


class _Sequence:
    """A scripted per-factory-call verdict sequence (one flip per attempt)."""

    def __init__(self, *flips: bool) -> None:
        self.flips = list(flips)
        self.i = 0

    def next(self) -> bool:
        value = self.flips[min(self.i, len(self.flips) - 1)]
        self.i += 1
        return value


def _sequence_factory(sequence: _Sequence, derived: dict | None = None):
    """A sanctioned experiment whose verdict flips per attempt and whose derived values
    (the writeback's source) are fixed."""

    def make(device, ctx):
        from tests.core.test_bringup_runner import ScriptedExperiment

        return ScriptedExperiment(device, success=sequence.next(), derived=derived or {})

    return make


def test_failure_poisoned_paths_are_forced_stale(tmp_path):
    """AC: a failed step's declared ``poisons`` are forced stale — regardless of age (the
    path is minutes old with NO age bound) and regardless of writes. PSB passed, its
    consumer failed, PSB re-ran and FAILED: its poisons (psb.window) are stale for the
    consumer, which detours to PSB before anything else runs."""
    psb = _Sequence(True, False, True)   # pass, then fail on the re-run, then pass
    edsr = _Sequence(False, True)        # fail once, then pass
    plan = BringupPlan(
        steps=(
            _step(
                "psb-histogram",
                provides=(Provides(path="psb", mapping={"w": "window"}),),
                poisons=("psb.window",),  # its own output is suspect when it fails
                experiment=_sequence_factory(psb, derived={"w": 0.7}),
                on_pass="edsr-scan",
                on_fail={1: "edsr-scan", 2: "park"},
                max_retries=2,
            ),
            _step(
                "edsr-scan",
                requires=(Requirement(path="psb.window", max_age_s=None),),
                experiment=_sequence_factory(edsr),
                on_pass="calibrated",
                on_fail={1: "psb-histogram", 2: "park"},
                max_retries=2,
            ),
        ),
        entry="psb-histogram",
    )
    store = _store_with_ref(tmp_path)
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out1 = runner.advance()  # psb attempt 1: PASSES — psb.window written, minutes-fresh
    assert out1.step == "psb-histogram" and out1.passed is True
    out2 = runner.advance()  # edsr attempt 1: fresh (no trigger yet) — runs and FAILS
    assert out2.step == "edsr-scan" and out2.passed is False
    out3 = runner.advance()  # psb attempt 2 (edssr's rung routed back): FAILS — poisons fire
    assert out3.step == "psb-histogram" and out3.passed is False
    assert campaign.derive_state().attempts["psb-histogram"] == 1

    out4 = runner.advance()  # position edsr: psb.window is FORCED stale by the failure
    assert out4.step == "psb-histogram" and out4.detour is True
    assert out4.attempt == 2  # the refresh runs under PSB's own ladder (post-pass reset)
    state = campaign.derive_state()
    assert state.position == "edsr-scan"  # psb passed control back
    assert state.attempts["psb-histogram"] == 0  # the passing detour reset its ladder

    out5 = runner.advance()  # psb.window freshly re-provided — edsr runs and passes
    assert out5.step == "edsr-scan" and out5.detour is False and out5.passed is True
    assert campaign.derive_state().terminal == "calibrated"


# -- spatial-excursion poisoning: the predicate is spatial, not envelope class -----------------------


class _MovesBridge:
    """A registered host-bridge adapter that reports its DC moves — the surface the
    runner's spatial predicate checks against the cleared region's extent."""

    def __init__(self, moves, *, verdicts=None):
        self.moves = [list(p) for p in moves]
        self.verdicts = dict(verdicts or {"demod-balance": True})

    def run(self, *, record=None, **kw):
        return {**self.verdicts, "moves": self.moves}


REGION_MAP = {
    "extent": {"bounds": [[0.0, 1.0], [0.0, 1.0]]},  # the cleared region's measured extent
    "grid": [[0.1, 0.5, 0.9], [0.2, 0.6, 0.8]],
}


def _widening_map_factory(box):
    """A stability-map experiment whose measured extent WIDENS on re-measurement — the
    first map is the narrow cleared region; a refresh re-measures (covering where the
    excursion went), so the second run is in-region."""

    def make(device, ctx):
        content = {
            "extent": {"bounds": box["bounds"]},
            "grid": [[0.1, 0.5, 0.9], [0.2, 0.6, 0.8]],
        }
        return __import__(
            "tests.core.test_bringup_runner", fromlist=["ScriptedExperiment"]
        ).ScriptedExperiment(device, success=True, derived={"map": content})

    return make


def _region_plan(bridge, *, drift_sensitive=True, map_factory=None):
    """stability-map banks the region artifact; dot-formation moves within it (or not);
    psb requires the drift-sensitive artifact."""
    return BringupPlan(
        steps=(
            _step(
                "stability-map",
                provides=(Provides(path="stability", mapping={"map": "map"}, artifact=True),),
                experiment=map_factory or _derived_factory({"map": REGION_MAP}),
                on_pass="dot-formation",
                on_fail={1: "park"},
                max_retries=1,
            ),
            _step(
                "dot-formation",
                gates=("demod-balance",),
                region="stability.map",  # the declared DC region: a banked artifact
                experiment=lambda d, c: bridge,
                on_pass="psb-histogram",
                on_fail={1: "park"},
                max_retries=1,
            ),
            _step(
                "psb-histogram",
                requires=(
                    Requirement(path="stability.map", max_age_s=None,
                                drift_sensitive=drift_sensitive),
                ),
                on_pass="calibrated",
                on_fail={1: "park"},
                max_retries=1,
            ),
        ),
        entry="stability-map",
    )


def test_an_excursion_poisons_the_drift_sensitive_region_artifact(tmp_path):
    """AC: a DC step whose moves LEAVE the cleared region poisons the drift-sensitive
    artifact acquired in it — the consumer requiring stability.map detours to re-measure
    the map, even though the write is minutes old with no age bound."""
    from strumento.core import bringup
    from strumento.core.record import read_events

    bridge = _MovesBridge([(0.2, 0.8), (5.0, 5.0)])  # the second move leaves the region
    box = {"bounds": [[0.0, 1.0], [0.0, 1.0]]}  # first measurement: the narrow region
    bringup.register_host_bridge(_MovesBridge)
    try:
        store = _store_with_ref(tmp_path)
        plan = _region_plan(bridge, map_factory=_widening_map_factory(box))
        campaign = _campaign(tmp_path, plan, ref="main")
        runner = _runner(campaign, store=store)

        out1 = runner.advance()  # banks the region artifact (the cleared extent)
        assert out1.step == "stability-map" and out1.passed is True
        out2 = runner.advance()  # the DC step runs — its second move EXCURSES
        assert out2.step == "dot-formation" and out2.passed is True

        poisons = [e for e in read_events(campaign.record.dir) if e.get("ev") == "poison"]
        assert len(poisons) == 1 and poisons[0]["paths"] == ["stability.map"]

        box["bounds"] = [[0.0, 10.0], [0.0, 10.0]]  # the re-measure will cover the excursion
        out3 = runner.advance()  # psb: stability.map is excursion-poisoned -> refresh
        assert out3.step == "stability-map" and out3.detour is True

        out4 = runner.advance()  # the DC step re-runs — in-region under the new extent
        assert out4.step == "dot-formation" and out4.passed is True
        out5 = runner.advance()  # the re-measured map cleared the poison: psb runs
        assert out5.step == "psb-histogram" and out5.detour is False and out5.passed is True
        assert campaign.derive_state().terminal == "calibrated"
    finally:
        bringup._HOST_BRIDGES.remove(_MovesBridge)


def test_in_region_moves_do_not_poison(tmp_path):
    """AC: moves INSIDE the cleared region do not poison — the artifact's validity region
    contains them (the property that makes in-region auto moves safe)."""
    from strumento.core import bringup
    from strumento.core.record import read_events

    bridge = _MovesBridge([(0.1, 0.9), (0.5, 0.5), (1.0, 1.0)])  # boundary-inclusive
    bringup.register_host_bridge(_MovesBridge)
    try:
        store = _store_with_ref(tmp_path)
        plan = _region_plan(bridge)
        campaign = _campaign(tmp_path, plan, ref="main")
        runner = _runner(campaign, store=store)

        runner.advance()  # bank the region
        runner.advance()  # the DC step moves strictly inside
        assert [
            e for e in read_events(campaign.record.dir) if e.get("ev") == "poison"
        ] == []

        out3 = runner.advance()  # psb runs directly — nothing was poisoned
        assert out3.step == "psb-histogram" and out3.detour is False and out3.passed is True
    finally:
        bringup._HOST_BRIDGES.remove(_MovesBridge)


def test_an_excursion_against_a_non_drift_sensitive_region_does_not_poison(tmp_path):
    """The poison keys to the drift-sensitive flag: an artifact no requirement flags
    drift-sensitive is robust to the excursion — no poison event, no detour."""
    from strumento.core import bringup
    from strumento.core.record import read_events

    bridge = _MovesBridge([(9.0, 9.0)])  # a wild excursion
    bringup.register_host_bridge(_MovesBridge)
    try:
        store = _store_with_ref(tmp_path)
        plan = _region_plan(bridge, drift_sensitive=False)
        campaign = _campaign(tmp_path, plan, ref="main")
        runner = _runner(campaign, store=store)

        runner.advance()
        runner.advance()
        assert [e for e in read_events(campaign.record.dir) if e.get("ev") == "poison"] == []

        out3 = runner.advance()  # not flagged drift-sensitive: nothing was poisoned
        assert out3.step == "psb-histogram" and out3.detour is False
    finally:
        bringup._HOST_BRIDGES.remove(_MovesBridge)


def test_a_region_naming_a_non_artifact_path_is_refused(tmp_path):
    """A declared region names a banked artifact; a plain calibration value at that path
    is a loud authoring error, never a silent no-region run."""
    from strumento.core import bringup

    class _BiasBridge(_MovesBridge):
        pass

    bringup.register_host_bridge(_BiasBridge)
    try:
        store = _store_with_ref(tmp_path)
        plan = BringupPlan(
            steps=(
                _step(
                    "dot-formation",
                    gates=("demod-balance",),
                    region="sensor.bias",  # a plain number, not a banked artifact
                    experiment=lambda d, c: _BiasBridge([(0.1, 0.1)]),
                    on_pass="calibrated",
                    on_fail={1: "park"},
                    max_retries=1,
                ),
            ),
            entry="dot-formation",
        )
        campaign = _campaign(tmp_path, plan, ref="main")
        runner = _runner(campaign, store=store)

        with pytest.raises(BringupError, match="content reference"):
            runner.advance()
    finally:
        bringup._HOST_BRIDGES.remove(_BiasBridge)


def test_a_region_declared_step_that_reports_no_moves_is_refused(tmp_path):
    """A region-declared DC step must report its moves — unreported moves cannot be
    checked, and an unchecked region is a silent safety hole, not a conservative pass."""
    from strumento.core import bringup

    class _SilentBridge:
        def run(self, *, record=None, **kw):
            return {"demod-balance": True}  # verdicts, no moves

    bringup.register_host_bridge(_SilentBridge)
    try:
        store = _store_with_ref(tmp_path)
        plan = BringupPlan(
            steps=(
                _step(
                    "stability-map",
                    provides=(Provides(path="stability", mapping={"map": "map"}, artifact=True),),
                    experiment=_derived_factory({"map": REGION_MAP}),
                    on_pass="silent-step",
                    on_fail={1: "park"},
                    max_retries=1,
                ),
                _step(
                    "silent-step",
                    gates=("demod-balance",),
                    region="stability.map",
                    experiment=lambda d, c: _SilentBridge(),
                    on_pass="calibrated",
                    on_fail={1: "park"},
                    max_retries=1,
                ),
            ),
            entry="stability-map",
        )
        campaign = _campaign(tmp_path, plan, ref="main")
        runner = _runner(campaign, store=store)

        runner.advance()  # bank the region the silent step declares
        with pytest.raises(BringupError, match="reported no moves"):
            runner.advance()
    finally:
        bringup._HOST_BRIDGES.remove(_SilentBridge)


# -- on_stale routing: proceed is refused for drift-sensitive paths; park holds -----------------------


def test_on_stale_proceed_is_refused_for_an_artifact_backed_drift_sensitive_path(tmp_path):
    """AC: ``on_stale: proceed`` is refused for artifact-backed drift-sensitive paths — the
    flag is the pack's declared contract; waving a stale region artifact through is a loud
    error, and nothing runs."""
    store = _store_with_ref(tmp_path)
    _old_write(store, tmp_path, path="stability.map", value=0.41,
               step="stability-map")  # stale by age
    plan = BringupPlan(
        steps=(
            _step(
                "psb-histogram",
                requires=(
                    Requirement(path="stability.map", max_age_s=3600.0, drift_sensitive=True),
                ),
                on_stale="proceed",
                on_pass="calibrated",
                on_fail={1: "park"},
                max_retries=1,
            ),
            _step(
                "stability-map",
                provides=(Provides(path="stability", mapping={"map": "map"}, artifact=True),),
                experiment=_derived_factory({"map": REGION_MAP}),
                on_pass="psb-histogram",
                on_fail={1: "park"},
                max_retries=1,
            ),
        ),
        entry="psb-histogram",
    )
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    with pytest.raises(BringupError, match="drift-sensitive"):
        runner.advance()

    state = campaign.derive_state()
    assert state.position == "psb-histogram"  # nothing ran, nothing burned
    assert state.attempts["psb-histogram"] == 0
    steps_dir = campaign.record.dir / "steps"
    assert not steps_dir.exists() or not any(steps_dir.iterdir())


def test_on_stale_proceed_is_refused_for_any_drift_sensitive_requirement(tmp_path):
    """The refusal keys on the declared flag, not the path's backing: a drift-sensitive
    scalar requirement is waved through by no ``proceed`` either."""
    store = _store_with_ref(tmp_path)
    _old_write(store, tmp_path)  # sensor.carrier_f, stale by age
    plan = _two_step_plan(on_stale="proceed", drift_sensitive=True)
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    with pytest.raises(BringupError, match="proceed"):
        runner.advance()


def test_on_stale_proceed_runs_a_non_drift_sensitive_step(tmp_path):
    """The complement: ``proceed`` is legal for paths not flagged drift-sensitive — the
    stale value is consumed as-is and the step runs."""
    store = _store_with_ref(tmp_path)
    _old_write(store, tmp_path)
    plan = _two_step_plan(on_stale="proceed", drift_sensitive=False)
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out = runner.advance()

    assert out.step == "coulomb-peak" and out.detour is False and out.passed is True
    assert campaign.derive_state().terminal == "calibrated"


def test_on_stale_park_holds_needs_human(tmp_path):
    """AC: ``on_stale: park`` emits a ``needs_human`` park naming the stale prerequisite —
    the campaign holds (nothing executes) until a resume."""
    store = _store_with_ref(tmp_path)
    _old_write(store, tmp_path)
    plan = _two_step_plan(on_stale="park")
    campaign = _campaign(tmp_path, plan, ref="main")
    runner = _runner(campaign, store=store)

    out = runner.advance()

    assert out.step is None and out.passed is None
    assert out.held == "parked:needs_human"
    state = campaign.derive_state()
    assert state.parked is not None and state.parked.kind == "needs_human"
    assert "sensor.carrier_f" in state.parked.reason
    assert state.attempts["coulomb-peak"] == 0  # the hold burned nothing

    out2 = runner.advance()  # the park holds the campaign — advancing does nothing
    assert out2.held == "parked:needs_human" and out2.step is None

    campaign.record.resume("operator", "I re-measured by hand; the value is fresh")
    out3 = runner.advance()
    # a resume is a word about routing, never about data: the value is still stale, so
    # the park policy re-parks — honestly, until the data itself is fresh
    assert out3.held == "parked:needs_human" and out3.step is None

    from strumento.core.record import now_iso

    _old_write(store, tmp_path, timestamp=now_iso())  # the operator re-measured by hand
    campaign.record.resume("operator", "the hand re-measure is in; continue")
    out4 = runner.advance()
    assert out4.step == "coulomb-peak" and out4.passed is True  # fresh data released it
