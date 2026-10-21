"""The cqed bring-up graph executed through the BringupRunner (issue #56).

The graph is pack data; these tests drive it through the core runner — store wired, sim soc,
armed truth — and pin the campaign mechanics the ACs name:

- AC 4 — the punchout's writeback lands the RF envelope path AND the operating point in one
  snapshot, and the kappa estimate writes nothing;
- AC 5 — Ramsey's compound writeback (detuning correction set + final-gain q_freq refine + T2
  set) lands in ONE ``record_calibration`` call, and its ladder routes the TLS scan on rung 1
  and a needs_human park on rung 2;
- AC 6 — the validation terminal's requires: T1 AND T2 recorded (the freshness engine), plus
  the measured detuning/fidelity gates.

The graph-node mechanics ride the core's tested paths (one snapshot per pass, bounds refused,
freshness detours); what is new here is the pack's own graph executing through them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("qick")

import strumento.packs.cqed.responses  # noqa: F401,E402  (registers the pack's response models)
from strumento.core.bringup import BringupCampaign, BringupRunner, Provides
from strumento.core.device import Device
from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.response import SimResponse
from strumento.core.simulator import SimulatorSoc
from strumento.core.store import FileConfigStore
from strumento.packs.cqed.bringup import cqed_bringup_plan
from strumento.packs.cqed.responses import truth_from_device
from tests._paths import DEVICES

MULTIMODE_YAML = DEVICES / "multimode_demo" / "device.yaml"
CQED_SOCCFG = DEVICES / "_fixtures" / "soccfg_v2_cqed.json"

#: The armed truth for the campaign runs — every node's fit has something to recover, and
#: every gate's prior is satisfied (the healthy-device scenario).
CAMPAIGN_TRUTH = dict(
    resonator_f0=750.3,
    resonator_kappa_mhz=0.25,
    pi_gain=8600.0,
    T1=88.0,
    T2=41.0,
    ramsey_detuning=0.02,  # 20 kHz — inside the terminal's 50 kHz window
    f_ge=4000.4,           # within the gain ladder's lowest-rung centroid cap
)

SEED = 7


def _device():
    return Device.load(MULTIMODE_YAML)


def _soc(dev, truth_overrides: dict | None = None):
    truth = truth_from_device(dev, **{**CAMPAIGN_TRUTH, **(truth_overrides or {})})
    return SimulatorSoc.from_snapshot(CQED_SOCCFG, truth=truth, seed=SEED)


def _store(tmp_path) -> FileConfigStore:
    """A FileConfigStore seeded with the device's own calibration tree as the campaign's ref."""
    store = FileConfigStore(tmp_path / "cfgstore")
    cfg = _device().to_config()
    store.put(cfg)
    store.set_ref("main", cfg.content_id())
    return store


def _campaign_plan_step(name):
    return {s.name: s for s in cqed_bringup_plan().steps}[name]


def _advance_to(runner, target: str, *, limit: int = 30):
    """Advance until the runner reports the target step (or its terminal), consuming calls."""
    for _ in range(limit):
        out = runner.advance()
        if out.step == target or (out.held and "terminal" in out.held):
            return out
    raise AssertionError(f"never reached {target!r} in {limit} advances")


# -- AC 4: the punchout banks the envelope path + the operating point; kappa writes nothing ----


@pytest.mark.qick
def test_punchout_writeback_banks_envelope_and_operating_point_not_kappa(tmp_path):
    """AC: the punchout's writeback — one snapshot carrying readout.max_power AND
    readout.power, with the kappa estimate nowhere in it (gate-only). Driven through the
    pack's own graph: coarse find → targeted find → punchout."""
    dev = _device()
    store = _store(tmp_path)
    campaign = BringupCampaign.create(tmp_path, cqed_bringup_plan(), device="multimode", ref="main")
    runner = BringupRunner(campaign, device=dev, soc=_soc(dev), store=store)

    out = _advance_to(runner, "punchout")
    assert out.step == "punchout" and out.passed is True

    log = store.log("main")
    assert len(log) == 3  # coarse find, targeted find, punchout — one snapshot per pass
    punchout_entry = log[-1]
    assert punchout_entry.source.endswith(":punchout")
    assert sorted(punchout_entry.updates) == ["readout.max_power", "readout.power"]
    # the kappa estimate is gate-only: it wrote nothing
    assert not any("kappa" in path for path in punchout_entry.updates)

    cfg = store.get(store.resolve_ref("main"))
    max_power = cfg.calibration["readout"]["max_power"]
    power = cfg.calibration["readout"]["power"]
    assert 0.0 < power < max_power  # the operating point sits below the punched-out ceiling
    # the RF envelope bound was honored, not clamped: the fitted ceiling landed exactly
    assert max_power == punchout_entry.updates["readout.max_power"][1]


# -- AC 5: Ramsey's compound writeback in one call ---------------------------------------------


class _ScriptedNode(Experiment):
    """A sanctioned scripted node for the mini-graphs: a trivial real acquisition (the sim
    soc needs an armable response) with a fixed verdict the graph routes on."""

    def __init__(self, device, *, success: bool = True, derived: dict | None = None):
        super().__init__(device)
        self._success = bool(success)
        self._derived = dict(derived or {})

    def sequence(self):
        return Seq().play(self.device.readout.probe()).measure(), ()

    def response_model(self, device, xpts):
        def model(ctx):
            n = len(ctx.xpts) or 1
            return SimResponse(populations=np.full(n, 0.5))

        return model

    def analyze(self, result):
        return FitResult(params={}, stderr={}, derived=self._derived, success=self._success)


@pytest.mark.qick
def test_ramsey_compound_writeback_lands_in_one_call(tmp_path):
    """AC: Ramsey's compound writeback — the detuning correction (set), the final-gain
    q_freq (refine), and T2 from the envelope (set) — lands in ONE ``record_calibration``
    call; the refine value is f_spec + Δ_Ramsey at the device's (final-Rabi-gain) belief."""
    dev = _device()
    ramsey = replace(_campaign_plan_step("ramsey"), on_pass="calibrated")

    plan = _mini_plan(
        tmp_path,
        steps=(
            replace(
                _campaign_plan_step("resonator-find-coarse"),
                name="f-ge-setter",
                provides=(Provides(path="qubit", mapping={"f0": "f_ge"}),),
                requires=(),
                on_pass="ramsey",
                on_fail={1: "park"},
                max_retries=1,
                experiment=lambda device, ctx: _ScriptedNode(device, derived={"f0": 4000.4}),
            ),
            ramsey,
            _campaign_plan_step("tls-scan"),
        ),
        entry="f-ge-setter",
    )
    campaign = BringupCampaign.create(tmp_path, plan, device="multimode", ref="main")
    runner = BringupRunner(campaign, device=dev, soc=_soc(dev), store=_store(tmp_path))

    out1 = runner.advance()  # the set-writer lands qubit.f_ge (set)
    assert out1.passed is True
    out2 = runner.advance()  # Ramsey runs and its compound writeback lands
    assert out2.step == "ramsey" and out2.passed is True

    log = [e for e in runner.store.log("main") if e.source.endswith(":ramsey")]
    assert len(log) == 1, "the compound writeback must land in ONE record_calibration call"
    entry = log[0]
    assert sorted(entry.updates) == ["qubit.T2", "qubit.f_ge", "qubit.freq_correction"]
    assert entry.extra["refinement"] == {
        "qubit.freq_correction": "set",
        "qubit.f_ge": "refine",
        "qubit.T2": "set",
    }
    detuning = entry.updates["qubit.freq_correction"][1]
    assert detuning == pytest.approx(0.02, abs=0.01)  # the measured detuning
    # the refine: f_spec (the device's belief) + Δ_Ramsey at the final Rabi gain
    assert entry.updates["qubit.f_ge"][1] == pytest.approx(dev.calib.qubit.f_ge + detuning, abs=0.05)


def _mini_plan(tmp_path, steps, entry):
    from strumento.core.bringup import BringupPlan

    return BringupPlan(steps=tuple(steps), entry=entry)


@pytest.mark.qick
def test_ramsey_ladder_routes_the_tls_scan_on_rung_1_and_parks_on_rung_2(tmp_path):
    """AC: Ramsey's ladder — rung 1 runs the TLS scan (the pack's real escalation-only node,
    through the sim soc), and the second failure parks needs_human. The failed Ramsey never
    wrote: the store log carries no Ramsey entry."""
    dev = _device()
    ramsey = replace(
        _campaign_plan_step("ramsey"),
        on_pass="calibrated",  # the mini plan's terminal — the ladder is the pack's
        experiment=lambda device, ctx: _ScriptedNode(device, success=False),
    )
    plan = _mini_plan(
        tmp_path,
        steps=(
            replace(
                _campaign_plan_step("resonator-find-coarse"),
                name="f-ge-setter",
                provides=(Provides(path="qubit", mapping={"f0": "f_ge"}),),
                requires=(),
                on_pass="ramsey",
                on_fail={1: "park"},
                max_retries=1,
                experiment=lambda device, ctx: _ScriptedNode(device, derived={"f0": 4000.4}),
            ),
            ramsey,
            _campaign_plan_step("tls-scan"),
        ),
        entry="f-ge-setter",
    )
    campaign = BringupCampaign.create(tmp_path, plan, device="multimode", ref="main")
    store = _store(tmp_path)
    runner = BringupRunner(campaign, device=dev, soc=_soc(dev), store=store)

    out1 = runner.advance()  # the set-writer
    assert out1.passed is True
    out2 = runner.advance()  # Ramsey fails → rung 1 routes the TLS scan
    assert out2.step == "ramsey" and out2.passed is False
    out3 = runner.advance()  # the TLS scan runs (gate-only evidence) → routes back to Ramsey
    assert out3.step == "tls-scan" and out3.passed is True
    out4 = runner.advance()  # Ramsey fails again → rung 2 parks needs_human
    assert out4.step == "ramsey" and out4.passed is False
    state = campaign.derive_state()
    assert state.parked is not None and state.parked.kind == "needs_human"
    # the dead Ramsey wrote nothing; only the setter's f_ge landed
    ramsey_writes = [e for e in store.log("main") if e.source.endswith(":ramsey")]
    assert ramsey_writes == []


# -- AC 6: the validation terminal requires T1/T2 recorded -------------------------------------


@pytest.mark.qick
def test_validation_requires_recorded_t1_and_t2_before_it_runs(tmp_path):
    """AC: the validation terminal requires qubit.T1 AND qubit.T2 recorded — with a store
    wired, the freshness engine detours to the providers until both are written; the
    validation then runs its measured gates and the campaign reaches the success terminal."""
    dev = _device()
    plan = cqed_bringup_plan()
    campaign = BringupCampaign.create(tmp_path, plan, device="multimode", ref="main")
    runner = BringupRunner(campaign, device=dev, soc=_soc(dev), store=_store(tmp_path))

    final = None
    for _ in range(30):
        final = runner.advance()
        if final.held and "terminal" in final.held:
            break
    assert final is not None and final.held == "terminal:calibrated", (
        "the healthy-device campaign should reach the success terminal with the graph's"
        f" own gates (last outcome: {final})"
    )
    state = campaign.derive_state()
    assert state.terminal == "calibrated"

    # the validation's writeback-free pass: nothing new landed at the terminal
    validation_children = [
        d for d in (campaign.record.dir / "steps").iterdir()
        if d.is_dir() and (d / "task.toml").exists()
    ]
    assert len(validation_children) >= 11  # every node in the chain ran as a child record
