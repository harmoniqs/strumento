"""The spin bring-up library — the six-step graph as pack data (issue #57).

Single dot + shared charge sensor, cold start to calibrated: sensor turn-on → Coulomb
peak → stability map → dot formation → PSB histogram → EDSR scan. The graph is pack
data authored in core's vocabulary (:class:`~strumento.core.bringup.BringupStep` /
:class:`~strumento.core.bringup.Provides` / :class:`~strumento.core.bringup.Requirement`);
core owns the mechanism (D18) and this module owns the graph, its escalation ladders,
its physical priors, and its frame-check definitions — the step table and the literals
below are the design of record.

**Region provenance (v1, stated):** the cleared region for dot-formation / PSB / EDSR
moves IS the ``stability.map`` artifact's measured extent — no separate region artifact
is banked. In-region moves run auto; an excursion outside the banked extent poisons the
drift-sensitive artifact's path and the chain refreshes.

**Host bridge (this module's second half):** the DC/host steps run through registered
host-bridge adapters (:class:`SpinHostBridge`) — the sanctioned-runnable contract for
experiments that have no ``go()``. Adapters stream per-point progress events into the
child record, honor cancellation between points, and declare settling times; their
sweep extents are the pre-run opt-in contract the runner refuses beyond declared
bounds. The seven DC-sweep utilities (plus ``ChargeStability`` and the ``PSBHistogram``
analyzer) are promoted into registered adapters — registration is the contract claim;
the utility's own ``analyze()`` is the verdict source, never a shell.

**Two spec-table literals vs the landed core** (resolved, and pinned in
``tests/packs/spin/test_spin_bringup.py``):

- the core keys write exclusivity by the FULL calibration path
  (``path + "." + subfield``), so two steps writing different leaves under
  the same container parent are each independent set-writers of their own
  leaf: sensor-turn-on sets ``sensor.carrier_f`` etc., Coulomb peak sets
  ``sensor.operating_point`` — both ``kind="set"`` because no prior step
  writes that specific leaf. Same for dot-formation (sets ``dot.operating_point``
  etc.) and EDSR (sets ``dot.f_resonance``);
- one ``Provides`` per container per step, with the artifact flag on the whole entry —
  so dot formation's mixed row banks the entire dot container artifact-backed (the
  ``compensation`` flag honored; ``operating_point`` and ``lever_arms`` ride the same
  digest-checked content-reference round-trip).

**v1 scope, stated:** single-dot; the sensor calibration ref is device-global (the one
claimed ``(device, ref)``); a second live spin campaign on the same device or the same
ref is rejected at campaign construction (core derives claim exclusivity — pinned).

The host-bridge protocol (:mod:`strumento.packs.spin.host_bridge`), refresh scope and region
provenance are described above.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from strumento.core.bringup import BringupStep, Provides, Requirement

# The host-bridge protocol and the registered adapters live beside the graph; importing
# them here registers every promoted DC utility with core's sanctioned-runnable
# registry (the contract claim) and re-exports the adapter surface.
from strumento.packs.spin.host_bridge import (  # noqa: F401
    BARRIER_NUDGE_V,
    DEFAULT_SETTLING_TIME_S,
    DEMOD_BALANCE_RATIO_MIN,
    EDSR_FREQ_PRIOR_MHZ,
    NOISE_FLOOR_SNR_MIN,
    PSB_DARK_FRACTION_MIN,
    PSB_VALLEY_MAX,
    PSB_VISIBILITY_MIN,
    ChargeStabilityBridge,
    CoulombDiamondBridge,
    CoulombPeakBridge,
    DispersiveGateCalibrationBridge,
    DotFormationBridge,
    EDSRScanBridge,
    HostBridgeCancelled,
    PinchOffBridge,
    PSBHistogramBridge,
    RecordPointEmitter,
    SensorTurnOnBridge,
    SpinHostBridge,
    TunnelCouplingTuneBridge,
)

if TYPE_CHECKING:  # the runtime import is lazy, inside the builder
    from strumento.core.bringup import BringupPlan

# -- freshness constants (pack data; the table's "long"/"short") ------------------------

#: A LONG requirement is the stable electrostatic chain — re-measured per cooldown,
#: not per drift event (the sensor's RF setup and bias).
LONG_MAX_AGE_S: float = 86400.0

#: A SHORT requirement is the drift-sensitive artifact chain — a stability map, a dot
#: compensation, a PSB window all go stale on the minutes scale; the value-poison
#: triggers fire long before the backstop does.
SHORT_MAX_AGE_S: float = 600.0

def _sensor_requirements() -> tuple[Requirement, ...]:
    """The four sensor leaves Coulomb peak consumes — all LONG (the stable chain)."""
    return tuple(
        Requirement(path=f"sensor.{leaf}", max_age_s=LONG_MAX_AGE_S)
        for leaf in ("carrier_f", "demod_phase", "rf_power", "bias")
    )


def _short(path: str) -> Requirement:
    """A drift-sensitive, short-max-age requirement on the artifact chain."""
    return Requirement(path=path, max_age_s=SHORT_MAX_AGE_S, drift_sensitive=True)


def spin_bringup_plan(
    *,
    root=None,
    store=None,
    ref: str = "main",
    dc=None,
    probe=None,
    set_carrier_frequency=None,
    set_drive_frequency=None,
    set_demod_phase=None,
    sensor_axis: str = "sensor",
    plunger_axis: str = "p0",
    barrier_axis: str = "b0",
    max_v: float = 1.5,
    **adapter_kw,
) -> BringupPlan:
    """Build the six-step spin bring-up graph (pack data) per the step table.

    The graph's literals are fixed; the builder parameterizes only the device shape —
    axis names, the DC envelope maxima (volts, lab-native), and the hardware seams the
    step factories close over (``dc``, ``probe``, the RF host callables, and the
    ``store``/``root`` pair the bridge adapters read calibration artifacts through).
    All parameters are optional: the plan validates as pure data with none of them,
    and a factory invoked without its seam refuses loudly rather than running nothing.

    ``root`` is the task-registry root the campaign will live under (artifact content
    references are registry-root-relative); ``store`` is the claimed calibration store's
    :class:`~strumento.core.store.ConfigStore`.
    """
    from strumento.core.bringup import BringupError, BringupPlan

    carrier_max_mhz = adapter_kw.pop("carrier_max_mhz", 400.0)
    drive_max_mhz = adapter_kw.pop("drive_max_mhz", float(EDSR_FREQ_PRIOR_MHZ[1]))

    # -- per-step adapter kwargs: the pack's step factories accept a SHARED adapter_kw
    # -- mapping, but each bridge constructor only takes its own subset. A key no
    # -- bridge accepts is an authoring error (refused loudly); a key one bridge
    # -- accepts flows only to the factories whose bridge consumes it.

    import inspect

    _ADAPTER_PARAMS_BY_CLASS = {
        cls: frozenset(inspect.signature(cls.__init__).parameters) - {"self"}
        for cls in (
            SensorTurnOnBridge, CoulombPeakBridge, ChargeStabilityBridge,
            DotFormationBridge, PSBHistogramBridge, EDSRScanBridge,
        )
    }
    _ADAPTER_PARAMS = frozenset().union(*_ADAPTER_PARAMS_BY_CLASS.values())
    _plan_params = {"carrier_max_mhz", "drive_max_mhz", "drive_lo_mhz", "drive_hi_mhz"}
    unknown = set(adapter_kw) - _plan_params - _ADAPTER_PARAMS
    if unknown:
        raise BringupError(
            f"spin_bringup_plan(): unknown adapter parameter(s) {sorted(unknown)} —"
            " the graph's builder refuses kwargs no adapter consumes (they would be"
            " silently dropped otherwise)"
        )

    def kw_for(cls):
        params = _ADAPTER_PARAMS_BY_CLASS.get(cls, frozenset())
        return {k: v for k, v in adapter_kw.items() if k in params}

    # -- the step factories: each constructs its registered host-bridge adapter from
    # -- the builder's hardware seams; a missing seam refuses loudly, never runs nothing.

    def _need(*, what: str, **seams) -> None:
        missing = sorted(name for name, value in seams.items() if value is None)
        if missing:
            raise BringupError(
                f"step {what!r}: its host bridge has no hardware seam wired —"
                f" pass {', '.join(missing)} to spin_bringup_plan() (the graph's"
                " factories construct their adapters from the seams; construction"
                " never runs, but an advance without them has nothing to drive)"
            )

    def make_sensor_turn_on(device, ctx):
        _need(what="sensor-turn-on", dc=dc, probe=probe,
              set_carrier_frequency=set_carrier_frequency)
        return SensorTurnOnBridge(
            dc, probe, set_carrier_frequency,
            set_demod_phase=set_demod_phase, sensor_axis=sensor_axis,
            **kw_for(SensorTurnOnBridge),
        )

    def make_coulomb_peak(device, ctx):
        _need(what="coulomb-peak", dc=dc, probe=probe)
        return CoulombPeakBridge(
            dc, probe, sensor_axis=sensor_axis, barrier_axis=barrier_axis,
            **kw_for(CoulombPeakBridge),
        )

    def make_stability_map(device, ctx):
        _need(what="stability-map", dc=dc, probe=probe)
        return ChargeStabilityBridge(
            dc, probe, x_axis=plunger_axis, y_axis=barrier_axis,
            **kw_for(ChargeStabilityBridge),
        )

    def make_dot_formation(device, ctx):
        _need(what="dot-formation", dc=dc, probe=probe, store=store, root=root)
        return DotFormationBridge(
            dc, probe, root=root, store=store, ref=ref,
            x_axis=plunger_axis, y_axis=barrier_axis, **kw_for(DotFormationBridge),
        )

    def make_psb_histogram(device, ctx):
        _need(what="psb-histogram", dc=dc, probe=probe, store=store, root=root)
        return PSBHistogramBridge(
            dc, probe, root=root, store=store, ref=ref,
            x_axis=plunger_axis, y_axis=barrier_axis, **kw_for(PSBHistogramBridge),
        )

    def make_edsr_scan(device, ctx):
        _need(what="edsr-scan", dc=dc, probe=probe,
              set_drive_frequency=set_drive_frequency, store=store, root=root)
        # rung 2 of the on_fail ladder is the power/span escalation: the factory
        # widens the drive span with the step's failure count (the ladder's rungs
        # re-run THIS step with more reach)
        attempts = ctx.state.attempts.get("edsr-scan", 0)
        span_scale = 2.0 ** max(0, attempts - 1)
        center = 0.5 * (
            float(adapter_kw.get("drive_lo_mhz", 8000.0))
            + float(adapter_kw.get("drive_hi_mhz", 8400.0))
        )
        half = 0.5 * span_scale * (
            float(adapter_kw.get("drive_hi_mhz", 8400.0))
            - float(adapter_kw.get("drive_lo_mhz", 8000.0))
        )
        return EDSRScanBridge(
            dc, probe, set_drive_frequency,
            root=root, store=store, ref=ref,
            x_axis=plunger_axis, y_axis=barrier_axis,
            drive_mhz=np.linspace(center - half, center + half, 81),
            **kw_for(EDSRScanBridge),
        )

    steps = (
        BringupStep(
            name="sensor-turn-on",
            experiment=make_sensor_turn_on,
            requires=(),
            provides=(
                Provides(
                    path="sensor",
                    mapping={
                        "carrier_f": "carrier_f",
                        "demod_phase": "demod_phase",
                        "rf_power": "rf_power",
                        "bias": "bias",
                    },
                ),
            ),
            gates=("demod_balance", "noise_floor"),
            on_pass="coulomb-peak",
            on_fail={1: "sensor-turn-on", 2: "park"},
            max_retries=2,
            envelope={"sensor": max_v, "carrier_mhz": carrier_max_mhz},
        ),
        BringupStep(
            name="coulomb-peak",
            experiment=make_coulomb_peak,
            requires=_sensor_requirements(),
            provides=(
                Provides(path="sensor", mapping={"operating_point": "operating_point"}),
            ),
            gates=("fit",),
            frame_check="barrier_nudge_monotonic",
            on_pass="stability-map",
            on_fail={1: "coulomb-peak", 2: "park"},
            max_retries=2,
            envelope={"sensor": max_v, barrier_axis: max_v},
        ),
        BringupStep(
            name="stability-map",
            experiment=make_stability_map,
            requires=(
                Requirement(path="sensor.operating_point", max_age_s=LONG_MAX_AGE_S),
            ),
            provides=(
                Provides(path="stability", mapping={"map": "map"}, artifact=True),
            ),
            gates=("fit",),
            frame_check="transition_coherence",
            on_pass="dot-formation",
            on_fail={1: "stability-map", 2: "park"},
            max_retries=2,
            envelope={plunger_axis: max_v, barrier_axis: max_v},
            poisons=("stability.map",),
        ),
        BringupStep(
            name="dot-formation",
            experiment=make_dot_formation,
            requires=(_short("stability.map"),),
            provides=(
                Provides(
                    path="dot",
                    mapping={
                        "operating_point": "operating_point",
                        "lever_arms": "lever_arms",
                        "compensation": "compensation",
                    },
                    artifact=True,
                ),
            ),
            gates=("fit",),
            frame_check="single_dot_geometry",
            on_pass="psb-histogram",
            on_fail={1: "dot-formation", 2: "park"},
            max_retries=2,
            envelope={plunger_axis: max_v, barrier_axis: max_v},
            poisons=(
                "dot.operating_point",
                "dot.lever_arms",
                "dot.compensation",
            ),
            region="stability.map",
        ),
        BringupStep(
            name="psb-histogram",
            experiment=make_psb_histogram,
            requires=(_short("dot.compensation"), _short("stability.map")),
            provides=(
                Provides(
                    path="psb",
                    mapping={
                        "window": "window",
                        "threshold": "threshold",
                        "visibility": "visibility",
                    },
                ),
            ),
            gates=("psb_visibility", "psb_valley"),
            on_pass="edsr-scan",
            on_fail={1: "psb-histogram", 2: "park"},
            max_retries=2,
            envelope={plunger_axis: max_v, barrier_axis: max_v},
            poisons=("psb.window", "psb.threshold", "psb.visibility"),
            region="stability.map",
        ),
        BringupStep(
            name="edsr-scan",
            experiment=make_edsr_scan,
            requires=(
                _short("psb.window"),
                _short("dot.operating_point"),
                _short("dot.compensation"),
            ),
            provides=(
                Provides(path="dot", mapping={"f_resonance": "f_resonance"}),
            ),
            gates=("fit", "resonance_prior"),
            on_pass="calibrated",
            on_fail={1: "psb-histogram", 2: "edsr-scan", 3: "park"},
            max_retries=3,
            envelope={plunger_axis: max_v, barrier_axis: max_v,
                      "drive_mhz": drive_max_mhz},
            region="stability.map",
        ),
    )
    return BringupPlan(steps=steps, entry="sensor-turn-on")
