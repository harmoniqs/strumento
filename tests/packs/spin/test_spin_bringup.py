"""The spin bring-up library (issue #57) — the six-step graph as pack data, the
host-bridge protocol, and the DC-utility promotion.

The graph is six steps, cold start to calibrated: sensor turn-on → Coulomb peak → stability
map → dot formation → PSB histogram → EDSR scan. Each step's literal provides / requires /
poisons / frame check (the step table, authored in ``strumento/packs/spin/bringup.py``) is
the design of record, and this suite pins that the graph-as-data carries them through
construction-time plan validation.

What each test covers, by the issue's Acceptance Criteria:

- AC1 — the six-step graph loads as pack data with the table's literals;
- AC2 — the frame checks per the table (barrier-nudge monotonicity, transition
  coherence, single-vs-double-dot geometry, PSB visibility + dark-space/valley);
- AC3 — the DC utilities are registered as contract claims (not shells) and drive the
  graph's steps through the host bridge;
- AC4 — host-bridge adapters stream per-point progress, honor cancellation between
  points, and declare settling times (sweep extents are the pre-run opt-in contract);
- AC5 — a stability-map refresh value-poisons dot formation, transitively re-staling
  PSB and EDSR;
- AC6 — in-region DC moves run auto; excursions propose and poison drift-sensitive
  paths;
- AC7 — EDSR's on_fail rungs: 1 → PSB refresh detour, 2 → power/span ladder, 3+ → park.

Two spec-table literals conflicted with the landed core and are resolved here (the
issue asks for exactly this accounting in the return):

- **Full-path write exclusivity.** The landed core keys set-writers by the FULL
  calibration path (``path + "." + subfield``), so two steps writing different
  leaves under the same container are each independent set-writers of their own
  leaf. Coulomb peak's ``sensor.operating_point`` and EDSR's ``dot.f_resonance``
  declare ``kind="set"`` because no prior step writes those specific leaves (the
  earlier steps write other leaves under the same parent container).
- **One Provides per container per step.** The landed core refuses two ``Provides``
  entries with the same container path in one step, and the artifact flag is
  per-Provides — so dot formation's mixed row (``dot.operating_point`` / ``dot.lever_arms``
  plain, ``dot.compensation`` artifact) is unexpressible as literally flagged. The
  whole dot container banks artifact-backed (the compensation flag — the 2D-data
  invariant the parent spec cares about — is honored; operating_point and lever_arms
  ride the same digest-checked content-reference round-trip).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strumento.core.bringup import (
    CALIBRATED,
    BringupCampaign,
    BringupError,
    BringupPlan,
    BringupRunner,
)
from strumento.core.dc import DCAxis, DCSource, MockVoltageSource
from strumento.core.store import FileConfigStore, record_calibration, unvalidated
from strumento.packs.spin.bringup import PSB_VISIBILITY_MIN

# -- the graph-as-data surface under test -----------------------------------------------


# ======================================================================================
# A physics-ish mock charge sensor — the host bridge's hardware seam
# ======================================================================================


class MockSpinDevice:
    """A mock charge-sensing spin device over the core ``MockVoltageSource``.

    One probe callable serves every step honestly: it returns a single-shot sensor
    read whose *mean* carries the electrostatics (an RF-reflectometry dip on the
    demodulated quadrature, a Coulomb peak on the sensor gate that the nearby barrier
    shifts — the frame-truth knob — and the dot's charging-transition line in the
    (plunger, barrier) plane) and whose *shot noise* carries the spin (a bright-mode
    probability the EDSR drive flips on resonance).

    The knobs the pathology tests flip: ``coupling_barrier`` (a barrier-blind peak is
    the RF-pickup impostor), ``interdot_family`` (a second transition family is the
    double-dot masquerade), ``p_dark``/``spin_delta`` (a smeared PSB histogram).
    """

    def __init__(self, *, seed: int = 11) -> None:
        self.source = MockVoltageSource()
        self.dc = DCSource(
            self.source,
            axes={
                "sensor": DCAxis(channel=0, max_v=1.5),
                "p0": DCAxis(channel=1, max_v=1.5, crosscoupling={"b0": 0.12}),
                "b0": DCAxis(channel=2, max_v=1.5),
            },
        )
        self.rf = {
            "carrier_mhz": None, "demod_phase": 0.0, "power_dbm": -30.0,
            "drive_mhz": None,
        }
        self.rng = np.random.default_rng(seed)
        self.probe_calls = 0
        # reflectometry: a Lorentzian dip on one demod quadrature
        self.f_dip_mhz = 312.5
        self.dip_depth = 0.30
        self.dip_gamma_mhz = 1.5
        self.phi_dip = 0.7
        # the charge sensor: a Coulomb peak the nearby gates shift
        self.v_peak0 = -0.50
        self.peak_amp = 0.40
        self.peak_width = 0.06
        self.coupling_barrier = 0.5  # -> the peak shifts 0.05 V under a 0.1 V nudge
        self.coupling_plunger = 0.05
        # the dot: ONE charging-transition line, p0 + slope*b0 = 0 (single dot)
        self.map_slope = 0.5
        self.line_offset = 0.0
        self.interdot_family = False  # -> the double-dot masquerade
        # the spin: bright-mode shots the EDSR drive flips on resonance
        self.p_dark = 0.70
        self.spin_delta = 0.25
        self.shot_sigma = 0.05
        self.f_res_mhz = 8200.0
        self.flip_width_mhz = 40.0
        self.flip_gain = 0.5

    # the host RF callables the adapters drive
    def set_carrier(self, f: float) -> None:
        self.rf["carrier_mhz"] = float(f)

    def set_drive(self, f: float) -> None:
        self.rf["drive_mhz"] = float(f)

    def set_demod(self, phi: float) -> None:
        self.rf["demod_phase"] = float(phi)

    def probe(self) -> float:
        """One single-shot sensor read at the current DC + RF state."""
        self.probe_calls += 1
        v = 1.0  # the flat band
        f_carrier = self.rf["carrier_mhz"]
        if f_carrier is not None:
            lorentz = 1.0 / (
                1.0 + ((f_carrier - self.f_dip_mhz) / self.dip_gamma_mhz) ** 2
            )
            v -= (
                self.dip_depth
                * lorentz
                * math.cos(self.rf["demod_phase"] - self.phi_dip)
            )
        v_sensor = self.dc.get("sensor")
        p0 = self.dc.get("p0")
        b0 = self.dc.get("b0")
        v_peak_eff = (
            self.v_peak0
            + self.coupling_barrier * b0
            + self.coupling_plunger * p0
        )
        v += self.peak_amp * math.exp(
            -((v_sensor - v_peak_eff) ** 2) / (2.0 * self.peak_width**2)
        )
        # the charging-transition line (and, when masquerading, the interdot family)
        v += 0.5 * self._ramp(p0 + self.map_slope * b0 - self.line_offset, 0.30)
        if self.interdot_family:
            # a second, gentler transition family (the interdot anti-crossing): too
            # weak to hijack the stability map's dominant-transition fits, steep
            # enough to be a second family in the dot-formation geometry cuts
            v += 0.15 * self._ramp(p0 - self.map_slope * b0 - 0.04, 0.12)
        # the spin bit: EDSR flips the bright-mode probability on resonance
        p_bright = 1.0 - self.p_dark
        f_drive = self.rf["drive_mhz"]
        if f_drive is not None:
            flip = 1.0 / (
                1.0 + ((f_drive - self.f_res_mhz) / self.flip_width_mhz) ** 2
            )
            p_bright = min(1.0, p_bright + self.flip_gain * flip)
        if self.rng.random() < p_bright:
            v += self.spin_delta
        return float(v + self.rng.normal(0.0, self.shot_sigma))

    @staticmethod
    def _ramp(x: float, period: float) -> float:
        """A charging ramp: piecewise-linear in the cell, kinked at each addition line."""
        return (x / period - math.floor(x / period)) if period > 0 else 0.0


@pytest.fixture()
def mock_spin() -> MockSpinDevice:
    return MockSpinDevice()


def _child_record(tmp_path):
    """A fresh child task record the bridge streams into (the runner's child shape)."""
    from strumento.core.record import TaskRecord

    return TaskRecord.create(
        tmp_path / "tasks" / "child-0001", title="coulomb-peak", kind="experiment",
        device="spin-demo", pid=0, heartbeat=False,
    )


def _point_events(record):
    """The per-point progress events a bridge streamed into ``record``."""
    from strumento.core.record import read_events

    return [
        e for e in read_events(record.dir)
        if e.get("ev") == "progress" and str(e.get("label", "")).startswith("dc point")
    ]


# -- AC2: the frame checks per the step table --------------------------------------------


def test_coulomb_peak_frame_check_requires_monotonic_shift_under_a_barrier_nudge(
    tmp_path, mock_spin, closing
):
    """AC2 (frame): the Coulomb peak must shift monotonically under a nearby barrier
    nudge — the frame-truth rule. A barrier-blind peak (the RF-pickup impostor: a
    signal the sensor's own frame cannot distinguish from a Coulomb peak) fails its
    own frame check, never a confidently wrong operating point."""
    from strumento.packs.spin.bringup import CoulombPeakBridge

    honest = CoulombPeakBridge(
        mock_spin.dc, mock_spin.probe, sensor_axis="sensor", barrier_axis="b0",
    )
    out = honest.run(record=closing(_child_record(tmp_path)))
    assert out["barrier_nudge_monotonic"] is True  # the peak follows the barrier
    assert out["fit"] is True

    impostor = MockSpinDevice(seed=11)
    impostor.coupling_barrier = 0.0  # the peak no longer responds to the barrier
    rogue = CoulombPeakBridge(
        impostor.dc, impostor.probe, sensor_axis="sensor", barrier_axis="b0",
    )
    out = rogue.run(record=closing(_child_record(tmp_path / "rogue")))
    assert out["fit"] is True  # the fit alone is happy — the peak looks like a peak
    assert out["barrier_nudge_monotonic"] is False  # the frame check catches it


def test_stability_map_frame_check_requires_transitions_coherent_with_both_axes(
    tmp_path, mock_spin, closing
):
    """AC2 (frame): the stability map's transitions must move coherently with BOTH
    axes — transition lines have slope in the (plunger, barrier) plane. A map whose
    transitions ignore one axis is not a stability diagram the dot can be formed
    from."""
    from strumento.packs.spin.bringup import ChargeStabilityBridge

    bridge = ChargeStabilityBridge(
        mock_spin.dc, mock_spin.probe, x_axis="p0", y_axis="b0",
    )
    out = bridge.run(record=closing(_child_record(tmp_path)))
    assert out["fit"] is True
    assert out["transition_coherence"] is True
    # the banked map artifact carries the generic extent the region predicate reads
    content = out["derived"]["map"]
    assert content["extent"]["bounds"] == [
        [float(bridge.x_volts.min()), float(bridge.x_volts.max())],
        [float(bridge.y_volts.min()), float(bridge.y_volts.max())],
    ]

    degenerate = MockSpinDevice(seed=11)
    degenerate.map_slope = 0.0  # transition lines ignore the barrier axis
    flat = ChargeStabilityBridge(
        degenerate.dc, degenerate.probe, x_axis="p0", y_axis="b0",
    )
    out = flat.run(record=closing(_child_record(tmp_path / "flat")))
    assert out["transition_coherence"] is False


# -- the campaign harness: the six-step graph driven by the runner over the mock ---------


def _spin_store(tmp_path) -> FileConfigStore:
    """A claimed-ref calibration store with the graph's parent dicts seeded (the
    writeback's dotted paths descend through existing parents)."""
    from strumento.core.config import DeviceConfig

    store = FileConfigStore(tmp_path / "cfgstore")
    base = DeviceConfig(
        pack="spin", topology=[], wiring={},
        calibration={"sensor": {}, "stability": {}, "dot": {}, "psb": {}},
    )
    store.put(base)
    store.set_ref("main", base.content_id())
    return store


def _spin_campaign(tmp_path, mock_spin, **plan_kw):
    """A cold-start spin bring-up campaign over the mock device: the plan's factories
    wired to the mock's DC + probe + RF seams, a claimed (device, ref), and a runner
    with the store attached (writeback + freshness armed)."""
    from strumento.packs.spin.bringup import spin_bringup_plan

    runner_kw = plan_kw.pop("runner_kw", {})
    store = _spin_store(tmp_path)
    plan = spin_bringup_plan(
        root=tmp_path, store=store, ref="main",
        dc=mock_spin.dc, probe=mock_spin.probe,
        set_carrier_frequency=mock_spin.set_carrier,
        set_drive_frequency=mock_spin.set_drive,
        set_demod_phase=mock_spin.set_demod,
        **plan_kw,
    )
    campaign = BringupCampaign.create(
        tmp_path, plan, device="spin-demo", ref="main"
    )
    runner = BringupRunner(
        campaign, device=object(), soc=object(), store=store, **runner_kw
    )
    return campaign, runner, store


def test_the_six_step_campaign_runs_cold_start_to_calibrated_on_the_mock(tmp_path, mock_spin):
    """The integration tracer bullet: the whole graph, cold start to calibrated, one
    step per advance — every step's gates computed from real acquisitions through the
    registered bridges, and every provides path from the step table landing
    in the claimed store."""
    campaign, runner, store = _spin_campaign(tmp_path, mock_spin)

    order = []
    while campaign.derive_state().terminal is None and len(order) < 12:
        out = runner.advance()
        assert out.held == "", f"unexpected hold: {out.held}"
        assert out.step is not None
        order.append((out.step, out.passed, out.detour))
        assert out.detour is False, f"unexpected detour: {out}"
        assert out.passed is True, f"{out.step} did not pass: verdicts={out.verdicts}"

    assert [step for step, _, _ in order] == [
        "sensor-turn-on", "coulomb-peak", "stability-map",
        "dot-formation", "psb-histogram", "edsr-scan",
    ]
    assert campaign.derive_state().terminal == "calibrated"

    # every literal path from the step table landed in the claimed ref
    config = store.get(store.resolve_ref("main"))
    calib = config.calibration
    assert calib["sensor"]["carrier_f"] == pytest.approx(
        mock_spin.f_dip_mhz, abs=2.0
    )
    assert calib["sensor"]["demod_phase"] == pytest.approx(mock_spin.phi_dip, abs=0.4)
    assert "rf_power" in calib["sensor"] and "bias" in calib["sensor"]
    assert "operating_point" in calib["sensor"]
    assert calib["stability"]["map"].startswith("artifact:")
    for leaf in ("operating_point", "lever_arms", "compensation"):
        assert calib["dot"][leaf].startswith("artifact:")
    assert calib["psb"]["visibility"] >= PSB_VISIBILITY_MIN
    window, threshold = calib["psb"]["window"], calib["psb"]["threshold"]
    assert window[0] < threshold < window[1]  # the readout window straddles its threshold
    assert window[1] - window[0] >= mock_spin.spin_delta  # and spans the two modes
    assert calib["dot"]["f_resonance"] == pytest.approx(
        mock_spin.f_res_mhz, abs=100.0
    )


# -- AC4: the host-bridge protocol — streaming, cancellation, settling, extents ----------


def test_host_bridge_adapters_stream_per_point_progress_events(tmp_path, mock_spin, closing):
    """AC4: an adapter's sweep streams one per-point progress event into the child
    record per point acquired — the live series a supervisor plots — each carrying
    its index, the total, and the measured signal."""
    from strumento.packs.spin.bringup import CoulombPeakBridge

    bridge = CoulombPeakBridge(
        mock_spin.dc, mock_spin.probe, sensor_axis="sensor", barrier_axis="b0",
    )
    # the Coulomb-peak acquisition is the base sweep plus the two barrier-nudged frame
    # sweeps — every acquired point streams, whichever pass it belongs to
    n_points = 3 * len(bridge.sweep_points())
    child = closing(_child_record(tmp_path))

    out = bridge.run(record=child)

    events = _point_events(child)
    assert len(events) == n_points  # exactly one per acquired point
    assert [e["step"] for e in events] == list(range(n_points))
    assert all(e.get("of") == n_points for e in events)
    assert all("sensor_signal" in e.get("metrics", {}) for e in events)
    assert out["fit"] is True  # the promoted utility's analysis produced the verdict


def test_host_bridge_honors_cancellation_between_points(tmp_path, mock_spin, closing):
    """AC4: cancellation is honored BETWEEN points — a cancelled bridge stops before
    the next DC move, streams no further progress, and surfaces the typed
    cancellation so the attempt derives as stopped, never as a silent partial pass."""
    from strumento.packs.spin.bringup import (
        CoulombPeakBridge,
        HostBridgeCancelled,
    )

    bridge = CoulombPeakBridge(
        mock_spin.dc, mock_spin.probe, sensor_axis="sensor", barrier_axis="b0",
        reps_per_point=1,  # one shot per point: the count IS the point count
        # the operator cancels once two points are on disk — the next point boundary
        # must honor it before any further DC move
        cancel_check=lambda: mock_spin.probe_calls >= 2,
    )
    child = closing(_child_record(tmp_path))

    with pytest.raises(HostBridgeCancelled):
        bridge.run(record=child)

    events = _point_events(child)
    assert len(events) == 2  # points 0 and 1 streamed; nothing after
    # no DC move past the cancellation point: the probe was never called again
    assert mock_spin.probe_calls == 2


def test_host_bridge_adapters_declare_settling_times(tmp_path, mock_spin, closing):
    """AC4: every adapter DECLARES its per-point settling time and waits it between
    the DC move and the probe — the host bridge's honest timing contract."""
    from strumento.packs.spin.bringup import CoulombPeakBridge

    sleeps: list[float] = []
    bridge = CoulombPeakBridge(
        mock_spin.dc, mock_spin.probe, sensor_axis="sensor", barrier_axis="b0",
        settling_time_s=0.05, sleep=sleeps.append,
    )
    assert bridge.settling_time_s == 0.05  # declared, up front
    child = closing(_child_record(tmp_path))

    bridge.run(record=child)

    assert sleeps == [0.05] * 3 * len(bridge.sweep_points())  # once per point


def test_host_bridge_sweep_extents_are_the_pre_run_opt_in_contract(mock_spin):
    """AC4: the adapters expose ``sweep_extents`` — channel -> (lo, hi) applied range —
    the pre-run contract the runner refuses beyond declared bounds."""
    from strumento.packs.spin.bringup import CoulombPeakBridge

    bridge = CoulombPeakBridge(
        mock_spin.dc, mock_spin.probe, sensor_axis="sensor", barrier_axis="b0",
    )
    extents = bridge.sweep_extents()
    assert set(extents) == {"sensor", "b0"}  # every channel the step sweeps
    lo_sensor, hi_sensor = extents["sensor"]
    assert lo_sensor == pytest.approx(float(bridge.sensor_volts.min()))
    assert hi_sensor == pytest.approx(float(bridge.sensor_volts.max()))
    lo_b, hi_b = extents["b0"]  # the barrier-nudge range the frame check applies
    assert lo_b == pytest.approx(-bridge.nudge_v)
    assert hi_b == pytest.approx(+bridge.nudge_v)



# -- AC2: dot-formation geometry and the PSB pass, per the table -------------------------


def test_dot_formation_frame_check_rejects_the_double_dot_masquerade(tmp_path, mock_spin):
    """AC2 (frame): dot formation passes single-dot vs double-dot geometry — ONE
    transition family per cut is a single dot; a second family is the interdot
    anti-crossing masquerading, and the frame check refuses to bank a formed dot
    from it. The map banked coherent and single-family; the anti-crossing shows
    when the fine geometry cuts resolve it — exactly the drift/retune masquerade."""
    campaign, runner, _ = _spin_campaign(tmp_path, mock_spin)
    while campaign.derive_state().position != "dot-formation":
        out = runner.advance()
        assert out.passed is True, f"{out.step} did not pass: {out.verdicts}"
    out = runner.advance()  # dot-formation, honest single dot
    assert out.step == "dot-formation" and out.passed is True
    assert out.verdicts["single_dot_geometry"] is True

    masquerade = MockSpinDevice(seed=11)
    campaign2, runner2, _ = _spin_campaign(tmp_path / "double", masquerade)
    for _ in range(3):
        out = runner2.advance()  # the map banks while the device is still single-dot
        assert out.passed is True, f"{out.step} did not pass: {out.verdicts}"
    masquerade.interdot_family = True  # the interdot anti-crossing masquerade
    out = runner2.advance()  # dot-formation: TWO families cross the cuts
    assert out.step == "dot-formation"
    assert out.verdicts["single_dot_geometry"] is False
    assert out.passed is False  # the frame check refuses to bank the masquerade


def test_psb_pass_is_visibility_at_the_pack_threshold_and_the_valley_check(
    tmp_path, mock_spin
):
    """AC2 (PSB): the pass is the table's literal — visibility ≥ the pack threshold
    AND the dark-space/valley check. A smeared histogram fails the visibility half;
    a bright-dominant histogram (no populated dark space) fails the valley check
    while its visibility is fine — the two gates are independent verdicts."""
    from strumento.packs.spin.bringup import PSBHistogramBridge

    campaign, runner, _ = _spin_campaign(tmp_path, mock_spin)
    while campaign.derive_state().position != "psb-histogram":
        out = runner.advance()
        assert out.passed is True, f"{out.step} did not pass: {out.verdicts}"
    out = runner.advance()
    assert out.step == "psb-histogram" and out.passed is True
    assert out.verdicts["psb_visibility"] is True
    assert out.verdicts["psb_valley"] is True
    # the bridge's pass is literally "visibility ≥ pack threshold AND the valley check"
    assert PSBHistogramBridge is not None  # the registered analyzer adapter exists

    smeared = MockSpinDevice(seed=11)
    smeared.spin_delta = 0.08  # the two modes nearly merge: no discrimination
    campaign2, runner2, _ = _spin_campaign(tmp_path / "smeared", smeared)
    while campaign2.derive_state().position != "psb-histogram":
        runner2.advance()
    out = runner2.advance()
    assert out.step == "psb-histogram"
    assert out.verdicts["psb_visibility"] is False  # separation < the pack threshold
    assert out.passed is False

    bright = MockSpinDevice(seed=11)
    bright.p_dark = 0.10  # no populated dark space: the histogram is one bright mode
    campaign3, runner3, _ = _spin_campaign(tmp_path / "bright", bright)
    while campaign3.derive_state().position != "psb-histogram":
        runner3.advance()
    out = runner3.advance()
    assert out.step == "psb-histogram"
    assert out.verdicts["psb_visibility"] is True  # the modes ARE separated...
    assert out.verdicts["psb_valley"] is False  # ...but the dark space is empty
    assert out.passed is False  # the pass needs BOTH halves of the literal


# -- AC3: the DC utilities are registered contract claims ---------------------------------


def test_the_seven_dc_utilities_are_registered_contract_claims():
    """AC3: the pack's DC-sweep utilities are promoted into REGISTERED host-bridge
    adapters — the registration is the contract claim (the sanctioned-runnable
    registry core checks), not shells. Every one of the seven unregistered
    utilities rides a registered adapter; the adapters genuinely drive their
    utility's run()/analyze() through the host-bridge protocol (pinned by the
    frame-check and tracer suites for the graph-driving five, and by the
    not-a-shell drive test below for the remaining four)."""
    from strumento.core import bringup as core_bringup
    from strumento.packs.spin import host_bridge as hb

    registered = set(core_bringup._HOST_BRIDGES)
    promoted = {
        hb.SensorTurnOnBridge,   # wraps RFReflectometryTune
        hb.CoulombPeakBridge,    # SensorTune
        hb.ChargeStabilityBridge,
        hb.DotFormationBridge,   # VirtualGateCalibration
        hb.PSBHistogramBridge,   # the PSBHistogram analyzer
        hb.EDSRScanBridge,
        hb.PinchOffBridge,               # the four remaining DC utilities of the seven
        hb.CoulombDiamondBridge,         #   the issue promotes into contract claims
        hb.TunnelCouplingTuneBridge,
        hb.DispersiveGateCalibrationBridge,
    }
    assert promoted <= registered  # every promoted adapter is a registered contract claim


def test_promoted_dc_utility_adapters_drive_their_utilities_not_shells(tmp_path, mock_spin, closing):
    """AC3 (not shells): the four non-graph DC utilities' adapters drive their
    utility's real run()+analyze() through the host-bridge protocol — the derived
    leaves are the UTILITY's own analysis keys, the sweep streams per point, and
    the extents declare the swept channels."""
    from strumento.packs.spin import host_bridge as hb

    cases = [
        (hb.PinchOffBridge, {"gate_axis": "p0"}, {"v_threshold"}, "pinch-off"),
        (
            hb.CoulombDiamondBridge,
            dict(plunger_axis="p0", bias_axis="b0"),
            {"charging_energy_mev", "lever_arm"},
            "coulomb-diamond",
        ),
        (hb.TunnelCouplingTuneBridge, {"barrier_axis": "b0"}, {"tunnel_coupling_ghz"},
         "tunnel-coupling"),
        (
            hb.DispersiveGateCalibrationBridge,
            dict(gate_axis="p0"),
            {"epsilon_optimal", "delta_cq_af"},
            "dispersive-gate",
        ),
    ]
    for cls, kw, derived_keys, name in cases:
        sleeps: list[float] = []
        bridge = cls(
            mock_spin.dc, mock_spin.probe, reps_per_point=1,
            sleep=sleeps.append, **kw,
        )
        child = closing(_child_record(tmp_path / name))
        out = bridge.run(record=child)

        # the verdict came from the utility's own analyze() — its derived keys surface
        assert derived_keys <= set(out["derived"]), (
            f"{name}: the adapter must surface the wrapped utility's derived keys"
        )
        # ...and through the protocol: streaming, settling, extents
        events = _point_events(child)
        assert len(events) == len(bridge.sweep_points())  # one per acquired point
        assert bridge.settling_time_s > 0 and len(sleeps) == len(bridge.sweep_points())
        extents = bridge.sweep_extents()
        assert extents, f"{name}: the extents contract is not opt-out for a sweep"


# -- AC4: the runner refuses a sweep extent beyond its declared envelope -------------------


def test_the_runner_refuses_a_sweep_extent_beyond_the_declared_envelope(
    tmp_path, mock_spin
):
    """AC4: sweep extents are the pre-run opt-in contract — a bridge whose planned
    extent exceeds its step's declared envelope bound is REFUSED by the runner
    before anything runs: no attempt opens, no probe call, no child record."""
    import dataclasses

    from strumento.packs.spin.bringup import spin_bringup_plan
    from strumento.packs.spin.host_bridge import CoulombPeakBridge

    store = _spin_store(tmp_path)
    plan = spin_bringup_plan(
        root=tmp_path, store=store, ref="main",
        dc=mock_spin.dc, probe=mock_spin.probe,
        set_carrier_frequency=mock_spin.set_carrier,
        set_drive_frequency=mock_spin.set_drive,
        set_demod_phase=mock_spin.set_demod,
    )
    by = {s.name: s for s in plan.steps}
    wide = CoulombPeakBridge(
        mock_spin.dc, mock_spin.probe, sensor_axis="sensor", barrier_axis="b0",
        sensor_volts=np.linspace(-3.0, -0.1, 41),  # beyond the 1.5 V envelope
    )
    plan = BringupPlan(
        steps=(
            by["sensor-turn-on"],
            dataclasses.replace(by["coulomb-peak"], experiment=lambda device, ctx: wide),
            *(
                s for s in plan.steps if s.name not in ("sensor-turn-on", "coulomb-peak")
            ),
        ),
        entry="sensor-turn-on",
    )
    campaign = BringupCampaign.create(tmp_path, plan, device="spin-demo", ref="main")
    runner = BringupRunner(campaign, device=object(), soc=object(), store=store)

    out = runner.advance()  # sensor-turn-on passes — nothing wide yet
    assert out.passed is True
    probe_calls_before = mock_spin.probe_calls

    with pytest.raises(BringupError, match="beyond the declared envelope"):
        runner.advance()  # the wide extent is refused BEFORE anything runs

    assert mock_spin.probe_calls == probe_calls_before  # nothing ran
    state = campaign.derive_state()
    assert state.position == "coulomb-peak"  # no attempt opened — no step_open
    assert state.attempts["coulomb-peak"] == 0
    assert state.parked is None and state.terminal is None


# -- AC5: the value-poison chain — a map refresh re-stales the whole downstream ------------


def test_a_stability_map_refresh_value_poisons_dot_formation_transitively(
    tmp_path, mock_spin
):
    """AC5: a stability-map refresh value-poisons dot formation, transitively
    re-staling PSB and EDSR — the full downstream chain re-runs. Seeded: the
    campaign drove to EDSR (every upstream write landed). A map refresh re-writes
    ``stability.map`` AFTER the dot/psb writes; EDSR's advance must then detour to
    the ROOT stale cause (the map — reached THROUGH psb.window's transitive chain,
    proving the transitive rule) and the chain re-runs map → dot → PSB → EDSR."""

    campaign, runner, store = _spin_campaign(tmp_path, mock_spin)
    for _ in range(5):  # sensor-turn-on .. psb-histogram all pass; position EDSR
        out = runner.advance()
        assert out.passed is True, f"{out.step} did not pass: {out.verdicts}"
    assert campaign.derive_state().position == "edsr-scan"

    # the drift event: the map is re-measured (a fresh stability.map write lands
    # AFTER every downstream write — the refresh that re-stales the whole chain)
    base = store.get(store.resolve_ref("main"))
    record_calibration(
        store, "main", base, {"stability.map": 0.41},
        source=f"{tmp_path.name}:stability-map",
        validate=unvalidated,  # the campaign's own writes bank paths the schema lacks
    )

    out = runner.advance()  # EDSR's requirements are transitively value-poisoned
    assert out.step == "stability-map" and out.detour is True
    # the detour named the ROOT: from EDSR's position the chain walked
    # psb.window -> dot.compensation -> stability.map (transitive value-poison)
    assert campaign.derive_state().attempts["edsr-scan"] == 0

    refreshed = []
    while campaign.derive_state().terminal is None and len(refreshed) < 4:
        out = runner.advance()
        refreshed.append((out.step, out.detour))
        assert out.passed is True, f"{out.step} did not pass: {out.verdicts}"
    # the refresh re-ran the whole downstream chain, in order, root-first
    assert [step for step, _ in refreshed] == [
        "dot-formation", "psb-histogram", "edsr-scan",
    ]
    assert campaign.derive_state().terminal == "calibrated"

    # the age backstop cannot explain it: all writes are seconds old (max_age 600s)
    # and nothing failed — the trigger was the value-poison of the re-written map
    config = store.get(store.resolve_ref("main"))
    assert config.calibration["stability"]["map"].startswith("artifact:")


# -- AC6: region semantics — the cleared region IS the map artifact's extent ----------------


def _spin_clearances():
    """The device safety config that clears the honest chain's envelopes and the
    stability-map region (the operator-approved state after a bring-up pass)."""
    from strumento.core.bringup import Clearances

    return Clearances(
        envelopes={
            "sensor": 1.5, "b0": 1.5, "p0": 1.5,
            "carrier_mhz": 400.0, "drive_mhz": 20000.0,
        },
        regions={"stability.map"},
    )


def test_in_region_dc_moves_run_auto_under_a_cleared_region(tmp_path, mock_spin):
    """AC6: in-region DC moves run auto — with the stability.map region cleared (its
    artifact extent IS the v1 cleared region) and the envelopes cleared, the whole
    six-step chain runs auto: no proposal parks, no excursion poisons, calibrated."""
    from strumento.core.bringup import Clearances
    from strumento.core.record import read_events

    campaign, runner, _ = _spin_campaign(
        tmp_path, mock_spin,
        runner_kw=dict(
            safety=Clearances(
                envelopes={
                    "sensor": 1.5, "b0": 1.5, "p0": 1.5,
                    "carrier_mhz": 400.0, "drive_mhz": 20000.0,
                },
                regions={"stability.map"},
            )
        ),
    )
    while campaign.derive_state().terminal is None:
        out = runner.advance()
        assert out.held == "", f"unexpected hold: {out.held}"
        assert out.proposed is False, f"{out.step} proposed: {out.held}"
        assert out.passed is True, f"{out.step} did not pass: {out.verdicts}"
    poisons = [e for e in read_events(campaign.record.dir) if e.get("ev") == "poison"]
    assert poisons == []  # in-region moves poison nothing
    assert campaign.derive_state().poisoned == {}


def test_region_steps_propose_until_the_region_is_cleared(tmp_path, mock_spin):
    """AC6 (propose half): a step whose declared DC region is not cleared PROPOSES —
    compile-and-stage, zero hardware — even with every envelope cleared; the
    approval clears the region for the whole downstream chain."""
    from strumento.core.bringup import Clearances

    envelopes = {
        "sensor": 1.5, "b0": 1.5, "p0": 1.5,
        "carrier_mhz": 400.0, "drive_mhz": 20000.0,
    }
    campaign, runner, _store = _spin_campaign(
        tmp_path, mock_spin, runner_kw=dict(safety=Clearances(envelopes=envelopes))
    )
    for _ in range(3):
        out = runner.advance()  # no region declared: auto within the cleared envelopes
        assert out.proposed is False and out.passed is True, f"{out}"
    out = runner.advance()  # dot-formation: region stability.map NOT cleared
    assert out.proposed is True
    assert out.step == "dot-formation" and out.passed is None
    assert campaign.derive_state().parked.kind == "propose"
    assert "stability.map" in campaign.derive_state().parked.reason

    # the operator approves the staged proposal: the region clears and the step runs
    ref = campaign.derive_state().pending_approvals[-1].proposal_ref
    campaign.record.approve("operator", "operator-cli", ref)
    out = runner.advance()  # executed under the approval
    assert out.step == "dot-formation" and out.proposed is False and out.passed is True
    while campaign.derive_state().terminal is None:
        out = runner.advance()  # the approval persists: the rest runs auto
        assert out.proposed is False, f"{out.step} proposed: {out}"
    assert campaign.derive_state().terminal == "calibrated"


def test_an_excursion_poisons_the_drift_sensitive_region_and_the_chain_refreshes(
    tmp_path, mock_spin
):
    """AC6 (excursion): a move outside the cleared region's extent — the banked
    stability-map artifact's measured extent — poisons the drift-sensitive region
    path (a durable poison event); the consumer requiring the map detours to
    re-measure it, the re-provided map heals the poison, and the chain re-runs."""
    import dataclasses

    from strumento.core import bringup as core_bringup
    from strumento.core.bringup import BringupPlan
    from strumento.core.record import read_events
    from strumento.packs.spin.bringup import spin_bringup_plan
    from strumento.packs.spin.host_bridge import SpinHostBridge

    class ExcursingBridge(SpinHostBridge):
        """The dot-formation step's reported moves leave the map's extent ONCE — the
        next run of the same factory is in-region (the freshness one-shot pattern: the
        re-measured map covers the excursion)."""

        def __init__(self, *args, seen=None, **kw):
            super().__init__(**kw)
            self._seen = seen

        def run(self, *, record=None, point_emitter=None):
            self._bind(record, point_emitter)
            self._points_streamed += 1
            self._seen["calls"] += 1
            if self._seen["calls"] == 1:
                moves = [(0.0, 0.0), (9.0, 9.0)]  # the excursion
            else:
                moves = [(0.0, 0.0)]
            return {
                "fit": True,
                "single_dot_geometry": True,
                "derived": {
                    "operating_point": {
                        "axes": ["p0", "b0"], "point": [0.0, 0.0],
                        "cell_offset_v": 0.05,
                    },
                    "lever_arms": {"axes": ["p0", "b0"], "p0": 1.0, "b0": 0.5},
                    "compensation": {
                        "axes": ["p0", "b0"], "matrix": [[1.0, 0.0], [0.0, 1.0]],
                    },
                },
                "moves": moves,
            }

    core_bringup.register_host_bridge(ExcursingBridge)
    try:
        plan = spin_bringup_plan(
            root=tmp_path, store=_spin_store(tmp_path), ref="main",
            dc=mock_spin.dc, probe=mock_spin.probe,
            set_carrier_frequency=mock_spin.set_carrier,
            set_drive_frequency=mock_spin.set_drive,
            set_demod_phase=mock_spin.set_demod,
        )
        by = {s.name: s for s in plan.steps}

        seen = {"calls": 0}  # shared across the factory's invocations: ONE excursion

        def excursing_factory(device, ctx):
            return ExcursingBridge(seen=seen)

        plan = BringupPlan(
            steps=(
                by["sensor-turn-on"],
                by["coulomb-peak"],
                by["stability-map"],
                dataclasses.replace(
                    by["dot-formation"], experiment=excursing_factory
                ),
                by["psb-histogram"],
                by["edsr-scan"],
            ),
            entry="sensor-turn-on",
        )
        store = _spin_store(tmp_path)
        campaign = BringupCampaign.create(
            tmp_path, plan, device="spin-demo", ref="main"
        )
        runner = BringupRunner(campaign, device=object(), soc=object(), store=store)

        for _ in range(3):  # sensor .. map pass; the map banks the region artifact
            out = runner.advance()
            assert out.passed is True, f"{out.step} did not pass: {out.verdicts}"
        out = runner.advance()  # dot-formation EXCURSES (its second reported move)
        assert out.step == "dot-formation" and out.passed is True  # routing on its own verdicts
        poisons = [e for e in read_events(campaign.record.dir) if e.get("ev") == "poison"]
        assert len(poisons) == 1 and poisons[0]["paths"] == ["stability.map"]
        assert campaign.derive_state().poisoned == {"stability.map": poisons[0]["t"]}

        out = runner.advance()  # psb requires stability.map: excursion-poisoned -> refresh
        assert out.step == "stability-map" and out.detour is True
        # the re-measured map passed AFTER the poison: the poison is healed in the
        # freshness view (the chain re-runs and re-calibrates below)
        out = runner.advance()  # dot-formation re-runs — the retry moves are in-region
        assert out.step == "dot-formation" and out.passed is True
        out = runner.advance()  # psb: the re-provided map healed the poison
        assert out.step == "psb-histogram" and out.detour is False and out.passed is True
        while campaign.derive_state().terminal is None:
            out = runner.advance()
        assert campaign.derive_state().terminal == "calibrated"
    finally:
        core_bringup._HOST_BRIDGES.remove(ExcursingBridge)


# -- AC7: EDSR's on_fail rungs — PSB refresh detour, power/span ladder, park ----------------


def test_edsr_on_fail_rungs_route_the_ladder(tmp_path, mock_spin):
    """AC7: EDSR's on_fail rungs — rung 1 routes the PSB refresh detour (a fresh
    readout window, then the retry); rung 2 re-enters the step itself as the
    power/span ladder (the factory widens the drive span with the failure count);
    rung 3 parks needs_human. An impostor drive window (a fitted resonance outside
    the physical prior) makes every attempt fail the resonance_prior gate."""
    from strumento.packs.spin.bringup import spin_bringup_plan

    drives: list[float] = []
    store = _spin_store(tmp_path)
    plan = spin_bringup_plan(
        root=tmp_path, store=store, ref="main",
        dc=mock_spin.dc, probe=mock_spin.probe,
        set_carrier_frequency=mock_spin.set_carrier,
        set_drive_frequency=drives.append,  # the history the span ladder shows
        set_demod_phase=mock_spin.set_demod,
        drive_lo_mhz=100.0, drive_hi_mhz=200.0,  # an impostor window: no real
        # resonance in it, so every fitted f0 fails the (5-20 GHz) prior
    )
    campaign = BringupCampaign.create(tmp_path, plan, device="spin-demo", ref="main")
    runner = BringupRunner(campaign, device=object(), soc=object(), store=store)

    for _ in range(5):
        out = runner.advance()
        assert out.passed is True, f"{out.step} did not pass: {out.verdicts}"
    out = runner.advance()  # edsr attempt 1: the prior gate fails it
    assert out.step == "edsr-scan" and out.passed is False
    assert out.routed_to == "psb-histogram"  # rung 1 -> the PSB refresh detour
    out = runner.advance()  # the PSB refresh detour ran
    assert out.step == "psb-histogram" and out.passed is True
    out = runner.advance()  # edsr attempt 2 (span_scale 2^0 — the same window)
    assert out.step == "edsr-scan" and out.passed is False
    assert out.routed_to == "edsr-scan"  # rung 2 -> the power/span ladder re-entry
    out = runner.advance()  # edsr attempt 3: the factory WIDENED the drive span
    assert out.step == "edsr-scan" and out.passed is False
    assert min(drives) == pytest.approx(50.0)  # the span doubled: 150 +/- 100 MHz
    assert out.routed_to == "park"
    out = runner.advance()  # rung 3: the campaign holds for a human
    assert out.held == "parked:needs_human"
    assert campaign.derive_state().attempts["edsr-scan"] == 3


def test_a_second_live_spin_campaign_on_the_same_device_or_ref_is_rejected(tmp_path):
    """v1 scope (pinned): single-dot, one claimed (device, ref) — a second live spin
    campaign on the same device OR the same ref is rejected loudly at campaign
    construction; core derives the exclusivity, the pack pins it through its plan."""
    from strumento.packs.spin.bringup import spin_bringup_plan

    plan = spin_bringup_plan()
    BringupCampaign.create(tmp_path, plan, device="spin-demo", ref="main")
    with pytest.raises(BringupError, match="already claimed"):
        BringupCampaign.create(tmp_path, plan, device="spin-demo", ref="main")


def _paths(step):
    """Every calibration leaf a step's provides write on pass: ``path.subfield``."""
    return {
        f"{provides.path}.{subfield}"
        for provides in step.provides
        for subfield in provides.mapping.values()
    }


def _requirements(step):
    return {requirement.path: requirement for requirement in step.requires}


def test_the_six_step_graph_loads_as_pack_data_with_the_tables_literals():
    """AC1: the full spin plan constructs and passes plan validation carrying the
    step table's literal provides/requires/poisons/frame-checks — six steps,
    one entry, every path exactly where the table puts it."""
    from strumento.packs.spin.bringup import spin_bringup_plan

    plan = spin_bringup_plan()
    assert isinstance(plan, BringupPlan)  # construction validation ran and passed

    by = {step.name: step for step in plan.steps}
    assert set(by) == {
        "sensor-turn-on", "coulomb-peak", "stability-map",
        "dot-formation", "psb-histogram", "edsr-scan",
    }
    assert plan.entry == "sensor-turn-on"

    # sensor turn-on — provides the four sensor leaves, all set; no requirements
    turn_on = by["sensor-turn-on"]
    assert _paths(turn_on) == {
        "sensor.carrier_f", "sensor.demod_phase", "sensor.rf_power", "sensor.bias",
    }
    assert turn_on.requires == ()
    assert turn_on.poisons == ()
    assert turn_on.gates == ("demod_balance", "noise_floor")
    assert turn_on.frame_check is None

    # Coulomb peak — provides sensor.operating_point; requires the four sensor
    # leaves, all LONG; frame: peak shifts monotonically under a nearby barrier nudge
    coulomb = by["coulomb-peak"]
    assert _paths(coulomb) == {"sensor.operating_point"}
    assert set(_requirements(coulomb)) == {
        "sensor.carrier_f", "sensor.demod_phase", "sensor.rf_power", "sensor.bias",
    }
    assert coulomb.poisons == ()
    assert coulomb.frame_check == "barrier_nudge_monotonic"

    # stability map — provides stability.map as an ARTIFACT; requires the sensor
    # operating point (long); poisons itself on fail; drift-sensitive by frame
    stable = by["stability-map"]
    assert _paths(stable) == {"stability.map"}
    assert set(_requirements(stable)) == {"sensor.operating_point"}
    assert stable.poisons == ("stability.map",)
    assert stable.frame_check == "transition_coherence"
    provides_map = next(p for p in stable.provides if p.path == "stability")
    assert provides_map.artifact is True

    # dot formation — provides the dot geometry; requires stability.map (short);
    # poisons dot.* on fail; frame: single-dot vs double-dot geometry
    dot = by["dot-formation"]
    assert _paths(dot) == {
        "dot.operating_point", "dot.lever_arms", "dot.compensation",
    }
    assert set(_requirements(dot)) == {"stability.map"}
    assert set(dot.poisons) == {
        "dot.operating_point", "dot.lever_arms", "dot.compensation",
    }
    assert dot.frame_check == "single_dot_geometry"
    assert dot.region == "stability.map"  # the v1 cleared region IS the map's extent

    # PSB histogram — provides the three psb leaves; requires dot.compensation and
    # stability.map (short); poisons psb.* on fail; pass := visibility AND valley
    psb = by["psb-histogram"]
    assert _paths(psb) == {"psb.window", "psb.threshold", "psb.visibility"}
    assert set(_requirements(psb)) == {"dot.compensation", "stability.map"}
    assert set(psb.poisons) == {"psb.window", "psb.threshold", "psb.visibility"}
    assert psb.gates == ("psb_visibility", "psb_valley")
    assert psb.frame_check is None
    assert psb.region == "stability.map"

    # EDSR scan — provides dot.f_resonance; requires psb.window, dot.operating_point,
    # dot.compensation (all short); no poisons; resonance fit + prior
    edsr = by["edsr-scan"]
    assert _paths(edsr) == {"dot.f_resonance"}
    assert set(_requirements(edsr)) == {
        "psb.window", "dot.operating_point", "dot.compensation",
    }
    assert edsr.poisons == ()
    assert edsr.gates == ("fit", "resonance_prior")
    assert edsr.region == "stability.map"

    # the short requirements are the drift-sensitive chain (artifact-backed, flagged)
    for name in ("dot-formation", "psb-histogram", "edsr-scan"):
        for requirement in by[name].requires:
            assert requirement.drift_sensitive, (
                f"{name}: {requirement.path} is a short/drift-sensitive requirement"
            )

    # every on_fail ladder routes every rung and terminates in a needs_human park
    for step in plan.steps:
        assert set(step.on_fail) == set(range(1, step.max_retries + 1))
        assert step.on_fail[step.max_retries] in ("park", "park:needs_human")

    # EDSR's rungs, literally: 1 → PSB refresh detour, 2 → power/span ladder, 3+ → park
    assert edsr.max_retries == 3
    assert edsr.on_fail == {1: "psb-histogram", 2: "edsr-scan", 3: "park"}

    # the on_pass chain threads all six steps to the success terminal
    assert by["sensor-turn-on"].on_pass == "coulomb-peak"
    assert by["coulomb-peak"].on_pass == "stability-map"
    assert by["stability-map"].on_pass == "dot-formation"
    assert by["dot-formation"].on_pass == "psb-histogram"
    assert by["psb-histogram"].on_pass == "edsr-scan"
    assert by["edsr-scan"].on_pass == CALIBRATED
