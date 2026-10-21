"""The spin host bridge — the protocol DC/host experiments run through (issue #57).

Core's runner invokes two kinds of sanctioned runnable: an ``Experiment`` through
``go(soc, record=child)``, and a **host-bridge adapter** registered with
:func:`~strumento.core.bringup.register_host_bridge`, invoked as ``run(record=child)``.
The spin pack's electrostatic steps have no ``go()`` — they are host-driven sweeps over
an external :class:`~strumento.core.dc.DCSource` (and host-RF setters) with a probe
callable — so this module is where they become runnable inside ``advance()``.

**The protocol, stated once (the adapters all inherit it):**

- adapters run via ``run(record, point_emitter)`` — the child task record is the
  destination and a point emitter is the streaming seam; when the runner invokes
  ``run(record=child)`` the bridge binds a :class:`RecordPointEmitter` to that record;
- **per-point progress events stream into the child record** — one ``progress`` event
  per acquired point, carrying its index, the total, and the measured signal, so a
  supervisor plots the sweep live from the ledger;
- **cancellation is honored between points** — the declared ``cancel_check`` fires at
  every point boundary *before the move*, so a cancelled sweep stops with no further
  hardware motion, and the typed :class:`HostBridgeCancelled` propagates (the attempt
  derives as stopped, never as a silent partial pass);
- **settling times are declared** — every bridge carries ``settling_time_s`` and waits
  it between the move and the acquisition (the wait is injectable for tests);
- **sweep extents are the pre-run opt-in contract** — every bridge exposes
  ``sweep_extents`` mapping each swept channel to its applied ``(lo, hi)`` range, which
  the runner refuses beyond the step's declared envelope bound *before anything runs*.

**DC-utility promotion:** each adapter wraps one of the pack's DC-sweep utilities —
the utility's own ``run()`` drives the sweep through this module's wrapped seams and
the utility's own ``analyze()`` is the verdict source. Registration with the core
registry (the *contract claim*) happens at module import; an adapter is never a shell.

Purity: no ``qick`` — the probe callable owns any fast-hardware access.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np

from strumento.core.bringup import register_host_bridge
from strumento.core.dc import DCSource

#: the declared per-point settling time (s) — the move settles before the acquisition.
DEFAULT_SETTLING_TIME_S: float = 0.001

# -- physical priors (pack data — the frame checks' "fit + priors") ---------------------

#: The PSB readout's minimum fitted visibility (|Δμ| / mean σ from the double-Gaussian
#: fit) — "visibility ≥ pack threshold" is the first half of the PSB pass.
PSB_VISIBILITY_MIN: float = 3.0

#: The maximum fraction of shots allowed in the valley BETWEEN the two histogram modes
#: — a good PSB histogram is bimodal with an empty valley (the dark-space check).
PSB_VALLEY_MAX: float = 0.15

#: The minimum dark (singlet) fraction at the PSB readout point — the histogram must
#: actually be two modes with a populated dark space, not one smeared peak.
PSB_DARK_FRACTION_MIN: float = 0.2

#: The demod quadrature balance: the dip's amplitude on its optimal phase must exceed
#: the orthogonal quadrature's by this factor — a real reflectometry dip is
#: phase-localized; RF pickup is not (the frame-truth rule).
DEMOD_BALANCE_RATIO_MIN: float = 3.0

#: The noise-floor sanity: the fitted dip depth over the sweep's residual scatter — a
#: dip indistinguishable from the noise floor is not a sensor.
NOISE_FLOOR_SNR_MIN: float = 5.0

#: The EDSR resonance prior window (MHz): a fitted resonance outside the physical
#: prior is an impostor, not a resonance.
EDSR_FREQ_PRIOR_MHZ: tuple[float, float] = (5000.0, 20000.0)

#: The barrier-nudge amplitude (V) for the Coulomb-peak frame check — "a nearby barrier
#: nudge"; small enough to stay in the same charge cell, large enough to shift a peak.
BARRIER_NUDGE_V: float = 0.1

#: the minimum SNR a gradient spike must clear over its cut's robust noise floor to
#: count as a transition — a flat or noisy cut has no dominant transition (``None``).
TRANSITION_SNR_MIN: float = 5.0


class HostBridgeCancelled(RuntimeError):
    """A bridge stopped at a point boundary because its cancellation check fired.

    The sweep made no move past the boundary and streamed no further points; the
    campaign's step attempt stays open and derives as stopped — cancellation is a fact
    about the operator's intent, never a verdict about the data.
    """


class RecordPointEmitter:
    """The default point emitter: streams one progress event per acquired point into
    the child record — a pure streamer. Cancellation is the bridge's job, not the
    emitter's: the wrapped DC/RF seams check at every point boundary, BEFORE the move,
    so a cancelled sweep stops with no further hardware motion (see
    :class:`SpinHostBridge`).

    A caller may pass its own ``point_emitter`` to :meth:`SpinHostBridge.run` — the
    protocol is the call shape (``point(index, of=..., **metrics)``), not this class.
    """

    def __init__(self, record: Any = None) -> None:
        self.record = record

    def point(self, index: int, *, of: int, **metrics: Any) -> None:
        """One acquired point, streamed into the record."""
        if self.record is not None:
            self.record.progress(index, of=of, label="dc point", **metrics)


class _CancellableDC:
    """A :class:`DCSource` view whose every move honors the cancellation check first —
    the point boundary a cancelled sweep stops at, before the hardware moves."""

    def __init__(self, dc: DCSource, bridge: SpinHostBridge) -> None:
        self._dc = dc
        self._bridge = bridge

    def set(self, name: str, volts: float) -> None:
        self._bridge._check_cancelled()
        self._dc.set(name, volts)

    def set_compensated(self, name: str, volts: float) -> None:
        self._bridge._check_cancelled()
        self._dc.set_compensated(name, volts)

    def get(self, name: str) -> float:
        return self._dc.get(name)

    def snapshot(self) -> dict[str, float]:
        return self._dc.snapshot()


class _CancellableSetter:
    """A host-RF setter (``set_frequency``-style) wrapped with the same point-boundary
    cancellation check — an RF sweep honors cancellation exactly like a DC one."""

    def __init__(self, setter: Callable[[float], None], bridge: SpinHostBridge) -> None:
        self._setter = setter
        self._bridge = bridge

    def __call__(self, value: float) -> None:
        self._bridge._check_cancelled()
        self._setter(value)


class SpinHostBridge:
    """The spin pack's host-bridge adapter base — the sanctioned-runnable contract.

    Subclasses wrap one DC-sweep utility and implement :meth:`sweep_extents` plus their
    acquisition; the protocol (cancellation between points, declared settling, per-point
    streaming) lives here, once. ``run(*, record=None, point_emitter=None)`` is the
    runner's entry: it binds a point emitter to the child record, drives the wrapped
    utility, and returns the verdict mapping (gate name -> bool, the reserved payload
    keys ``moves`` / ``calibration`` / ``derived`` riding beside them).
    """

    def __init__(
        self,
        *,
        settling_time_s: float | None = None,
        sleep: Callable[[float], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        #: the declared per-point settling time (s) — waited between the move and the
        #: acquisition, once per point.
        self.settling_time_s: float = (
            DEFAULT_SETTLING_TIME_S if settling_time_s is None else float(settling_time_s)
        )
        self._sleep: Callable[[float], None] = (
            sleep if sleep is not None else time.sleep
        )
        self.cancel_check = cancel_check
        self._emitter: Any = None
        self._points_streamed = 0

    # -- the protocol pieces subclasses compose ---------------------------------------

    def _check_cancelled(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise HostBridgeCancelled(
                "cancellation requested — the bridge stops between points, making no"
                " further DC moves"
            )

    def _bind(self, record: Any, point_emitter: Any) -> None:
        """Bind this run's streaming seam: the caller's emitter, or one on the record."""
        self._emitter = (
            point_emitter if point_emitter is not None else RecordPointEmitter(record)
        )
        self._points_streamed = 0

    def _wrap_dc(self, dc: DCSource) -> _CancellableDC:
        return _CancellableDC(dc, self)

    def _wrap_setter(self, setter: Callable[[float], None]) -> _CancellableSetter:
        return _CancellableSetter(setter, self)

    def _wrap_probe(
        self,
        probe: Callable[[], float],
        *,
        of: int,
        reps: int = 1,
        signal_key: str = "sensor_signal",
    ) -> Callable[[], float]:
        """The acquisition seam: settle (the declared time), average ``reps`` single
        shots, stream the point, return the value."""

        def wrapped() -> float:
            if self.settling_time_s > 0:
                self._sleep(self.settling_time_s)
            if reps <= 1:
                value = float(probe())
            else:
                value = float(np.mean([probe() for _ in range(reps)]))
            index = self._points_streamed
            self._points_streamed += 1
            if self._emitter is not None:
                self._emitter.point(index, of=of, **{signal_key: value})
            return value

        return wrapped

    # -- the contract surface ----------------------------------------------------------

    def sweep_extents(self) -> Mapping[str, tuple[float, float]]:
        """The pre-run opt-in contract: every swept channel -> its applied (lo, hi)
        range. The runner refuses an extent beyond the step's declared envelope bound
        BEFORE anything runs; an empty mapping is the opt-out."""
        return {}

    def run(self, *, record: Any = None, point_emitter: Any = None) -> Mapping[str, Any]:
        """Drive the acquisition; return the verdict mapping the runner routes on."""
        raise NotImplementedError  # pragma: no cover - the subclass owns the physics


def _peak_position(x: np.ndarray, y: np.ndarray) -> float:
    """The peak-top position as a soft centroid of the curve's upper region — a smooth
    tracker (robust to flank flips and single-point noise) whose shift follows the
    peak monotonically, which is exactly what the frame checks need to measure."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    lo, hi = float(y.min()), float(y.max())
    weights = np.clip(y - (hi - 0.3 * (hi - lo)), 0.0, None)
    if weights.sum() <= 0:
        return float(x[int(np.argmax(y))])
    return float(np.average(x, weights=weights))


class CoulombPeakBridge(SpinHostBridge):
    """Coulomb peak — the promoted :class:`SensorTune` as a graph-step runnable.

    Sweeps the sensor gate through its Coulomb peak and extracts the operating point
    (maximum transconductance) via the utility's own analysis. The barrier-nudge frame
    check rides the same seam: it re-sweeps under a nearby barrier nudge either side of
    the current setting and requires the peak position to shift monotonically with it —
    a peak that does not respond to the barrier is an impostor (RF pickup) the sensor's
    own frame cannot tell apart.
    """

    def __init__(
        self,
        dc: DCSource,
        probe: Callable[[], float],
        *,
        sensor_axis: str = "sensor",
        barrier_axis: str = "b0",
        sensor_volts=np.linspace(-0.9, -0.1, 41),
        nudge_v: float = 0.1,
        reps_per_point: int = 32,
        compensated: bool = False,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        self.dc = dc
        self.probe = probe
        self.sensor_axis = sensor_axis
        self.barrier_axis = barrier_axis
        self.sensor_volts = np.asarray(sensor_volts, dtype=float)
        #: the barrier-nudge amplitude (V) the frame check applies either side of the
        #: current barrier setting.
        self.nudge_v = float(nudge_v)
        self.reps_per_point = int(reps_per_point)
        self.compensated = compensated

    def sweep_points(self) -> list[tuple[float, ...]]:
        """The DC points this bridge visits per sweep, in the sensor-axis basis."""
        return [(float(v),) for v in self.sensor_volts]

    def sweep_extents(self):
        base = float(self.dc.get(self.barrier_axis))
        return {
            self.sensor_axis: (
                float(self.sensor_volts.min()),
                float(self.sensor_volts.max()),
            ),
            self.barrier_axis: (base - self.nudge_v, base + self.nudge_v),
        }

    def run(self, *, record=None, point_emitter=None):
        from strumento.packs.spin.experiments.sensor_tune import SensorTune

        self._bind(record, point_emitter)
        dc = self._wrap_dc(self.dc)
        of = 3 * len(self.sensor_volts)  # base sweep + the two nudged frame sweeps
        probe = self._wrap_probe(self.probe, of=of, reps=self.reps_per_point)

        base_barrier = self.dc.get(self.barrier_axis)
        positions: dict[str, float] = {}
        fit = None
        for label, barrier_v in (
            ("base", base_barrier),
            ("nudge_minus", base_barrier - self.nudge_v),
            ("nudge_plus", base_barrier + self.nudge_v),
        ):
            dc.set(self.barrier_axis, barrier_v)
            utility = SensorTune(
                dc,
                sensor_name=self.sensor_axis,
                voltages=self.sensor_volts,
                probe=probe,
                compensated=self.compensated,
            )
            result = utility.run()
            analysis = utility.analyze(result)
            positions[label] = _peak_position(result.xpts, result.avgi)
            if label == "base":
                fit = analysis

        ordered = [positions["nudge_minus"], positions["base"], positions["nudge_plus"]]
        grid = (
            float(abs(self.sensor_volts[1] - self.sensor_volts[0]))
            if len(self.sensor_volts) > 1
            else 0.0
        )
        resolved = max(grid / 2.0, 1e-9)
        steps = [b - a for a, b in zip(ordered, ordered[1:])]
        monotonic = fit is not None and fit.success and (
            all(s > resolved for s in steps) or all(s < -resolved for s in steps)
        )
        return {
            "fit": bool(fit is not None and fit.success),
            "barrier_nudge_monotonic": bool(monotonic),
            "derived": {
                "operating_point": float(
                    fit.derived.get("v_optimal", float("nan")) if fit else float("nan")
                ),
            },
            "moves": self.sweep_points(),
        }


def _smoothed(values: np.ndarray, *, size: int = 5) -> np.ndarray:
    """A centered moving average with honest edges (nearest-value, never zero-padded —
    zero padding fakes huge gradients at the window boundary and swamps real kinks)."""
    from scipy.ndimage import uniform_filter1d

    return uniform_filter1d(
        np.asarray(values, dtype=float), size=max(1, int(size)), mode="nearest"
    )


def _dominant_transition(x: np.ndarray, y: np.ndarray) -> float | None:
    """The dominant transition position along one cut: the centroid of the largest
    peak in |d smoothed / dx|, or ``None`` when no peak clears the noise floor — a
    real transition is a kink in the mean signal whose gradient spike stands well
    above the cut's own baseline slope and shot noise."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) < 5:
        return None
    gradient = np.abs(np.gradient(_smoothed(y), x))
    baseline = float(np.median(gradient))
    noise = _robust_noise(gradient - baseline)
    top = float(gradient.max())
    if top <= baseline or top - baseline < max(TRANSITION_SNR_MIN * noise, 1e-12):
        return None
    weights = np.clip(gradient - 0.5 * (top + baseline), 0.0, None)
    if weights.sum() <= 0:
        return None
    return float(np.average(x, weights=weights))


def _robust_noise(deviation: np.ndarray) -> float:
    """The robust σ of a baseline-subtracted gradient — the MAD estimator over the
    whole cut. On a cut fine enough to resolve its families the transition samples
    are a small minority and cannot inflate the floor."""
    return 1.4826 * float(np.median(np.abs(np.asarray(deviation, dtype=float))))


def _count_transitions(
    x: np.ndarray, y: np.ndarray, *, min_separation: float, smooth: int = 5
) -> int:
    """How many distinct transition families cross one cut — baseline-subtracted
    gradient peaks that clear the cut's robust noise floor (the charging ramp's own
    slope IS the baseline, not a transition), merged when closer than
    ``min_separation``. One family is a single charging line; two or more is an
    anti-crossing region."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) < 5:
        return 0
    gradient = np.abs(np.gradient(_smoothed(y, size=smooth), x))
    baseline = float(np.median(gradient))
    deviation = gradient - baseline
    floor = max(TRANSITION_SNR_MIN * _robust_noise(deviation), 1e-12)
    positions: list[float] = []
    for i in range(1, len(x) - 1):
        if (
            deviation[i] >= floor
            and deviation[i] >= deviation[i - 1]
            and deviation[i] >= deviation[i + 1]
        ):
            if not positions or abs(x[i] - positions[-1]) >= min_separation:
                positions.append(float(x[i]))
    return len(positions)


class ChargeStabilityBridge(SpinHostBridge):
    """Stability map — the promoted :class:`ChargeStability` as a graph-step runnable.

    Sweeps plunger × barrier and banks the 2D map as the artifact whose measured
    extent IS the v1 cleared region (the calibration path ``stability.map`` holds the
    content reference; ``resolve_calibration_artifact`` is the reader's side). The
    frame check requires the transitions to move coherently with BOTH axes —
    transition lines have slope in the (plunger, barrier) plane; a map whose
    transitions ignore an axis is not a stability diagram.
    """

    def __init__(
        self,
        dc: DCSource,
        probe: Callable[[], float],
        *,
        x_axis: str = "p0",
        y_axis: str = "b0",
        x_volts=np.linspace(-0.15, 0.15, 31),
        y_volts=np.linspace(-0.24, 0.24, 25),
        reps_per_point: int = 64,
        compensated: bool = False,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        self.dc = dc
        self.probe = probe
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.x_volts = np.asarray(x_volts, dtype=float)
        self.y_volts = np.asarray(y_volts, dtype=float)
        self.reps_per_point = int(reps_per_point)
        self.compensated = compensated

    def sweep_points(self) -> list[tuple[float, ...]]:
        """The 2D grid in the region basis (x, y) — the order the utility sweeps it."""
        return [
            (float(vx), float(vy))
            for vy in self.y_volts
            for vx in self.x_volts
        ]

    def sweep_extents(self):
        return {
            self.x_axis: (float(self.x_volts.min()), float(self.x_volts.max())),
            self.y_axis: (float(self.y_volts.min()), float(self.y_volts.max())),
        }

    def run(self, *, record=None, point_emitter=None):
        from strumento.packs.spin.experiments.charge_stability import ChargeStability

        self._bind(record, point_emitter)
        dc = self._wrap_dc(self.dc)
        probe = self._wrap_probe(
            self.probe,
            of=len(self.x_volts) * len(self.y_volts),
            reps=self.reps_per_point,
        )
        utility = ChargeStability(
            dc,
            x_axis=self.x_axis,
            x_volts=self.x_volts,
            y_axis=self.y_axis,
            y_volts=self.y_volts,
            probe=probe,
            compensated=self.compensated,
        )
        result = utility.run()
        analysis = utility.analyze(result)
        grid = np.asarray(result.avgi, dtype=float)
        coherent = self._transitions_coherent(grid)

        content = {
            "extent": {
                "bounds": [
                    [float(self.x_volts.min()), float(self.x_volts.max())],
                    [float(self.y_volts.min()), float(self.y_volts.max())],
                ]
            },
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "x_pts": [float(v) for v in self.x_volts],
            "y_pts": [float(v) for v in self.y_volts],
            "grid": [[float(v) for v in row] for row in grid],
        }
        found = bool(
            analysis.get("n_transitions_x", 0) >= 1
            and analysis.get("n_transitions_y", 0) >= 1
        )
        return {
            "fit": found,
            "transition_coherence": bool(coherent),
            "derived": {"map": content},
            "moves": self.sweep_points(),
        }

    def _transitions_coherent(self, grid: np.ndarray) -> bool:
        """The frame check: the dominant x-transition must MOVE with the y axis and the
        dominant y-transition with the x axis — each by more than a grid step."""
        n_y, n_x = grid.shape
        if n_y < 5 or n_x < 5:
            return False
        x_step = float(abs(self.x_volts[1] - self.x_volts[0]))
        y_step = float(abs(self.y_volts[1] - self.y_volts[0]))
        rows = sorted({1, n_y // 2, n_y - 2})  # interior cuts, spread across the window
        x_positions = [
            _dominant_transition(self.x_volts, grid[j, :]) for j in rows
        ]
        if any(p is None for p in x_positions):
            return False
        xs = cast("list[float]", x_positions)  # the None check above: every entry is a float
        if max(xs) - min(xs) <= 3.0 * x_step:
            return False  # the x-transition does not move with the y axis
        # quarter-interior columns: cuts whose y-transition can sit inside the window
        # (edge columns' lines may exit the mapped extent — a map, not a coherence fact)
        cols = sorted({n_x // 4, n_x // 2, 3 * n_x // 4})
        y_positions = [
            _dominant_transition(self.y_volts, grid[:, i]) for i in cols
        ]
        if any(p is None for p in y_positions):
            return False
        ys = cast("list[float]", y_positions)  # the None check above: every entry is a float
        return (max(ys) - min(ys)) > 3.0 * y_step


class PinchOffBridge(SpinHostBridge):
    """Pinch-off — the promoted :class:`PinchOff` as a sanctioned host-bridge runnable.

    Sweeps a gate from accumulation toward depletion; the utility's own ``analyze()``
    extracts the pinch-off threshold (the first voltage where the channel conductance
    drops below the cutoff fraction of its maximum). Registered as a contract claim
    (issue #57's DC-utility promotion) — any graph step or escalation ladder may
    drive it through the sanctioned path, and the protocol (streaming, cancellation,
    settling, extents) is inherited, not re-authored.
    """

    def __init__(
        self,
        dc: DCSource,
        probe: Callable[[], float],
        *,
        gate_axis: str = "p0",
        voltages=np.linspace(0.0, -1.2, 41),
        cutoff_fraction: float = 0.1,
        reps_per_point: int = 16,
        compensated: bool = False,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        self.dc = dc
        self.probe = probe
        self.gate_axis = gate_axis
        self.voltages = np.asarray(voltages, dtype=float)
        self.cutoff_fraction = float(cutoff_fraction)
        self.reps_per_point = int(reps_per_point)
        self.compensated = compensated

    def sweep_points(self) -> list[tuple[float, ...]]:
        return [(float(v),) for v in self.voltages]

    def sweep_extents(self):
        return {self.gate_axis: (float(self.voltages.min()), float(self.voltages.max()))}

    def run(self, *, record=None, point_emitter=None):
        from strumento.packs.spin.experiments.pinch_off import PinchOff

        self._bind(record, point_emitter)
        dc = self._wrap_dc(self.dc)
        probe = self._wrap_probe(
            self.probe, of=len(self.voltages), reps=self.reps_per_point
        )
        utility = PinchOff(
            dc, gate_name=self.gate_axis, voltages=self.voltages, probe=probe
        )
        result = utility.run()
        fit = utility.analyze(result)
        return {
            "fit": bool(fit.success),
            "derived": dict(fit.derived),
            "moves": self.sweep_points(),
        }


class CoulombDiamondBridge(SpinHostBridge):
    """Coulomb diamond — the promoted :class:`CoulombDiamond` as a graph-step runnable.

    2D sweep of plunger × bias; the utility's own analysis extracts the charging
    energy and lever arm from the diamond geometry. Registered as a contract claim —
    a sanctioned host-bridge runnable any graph step may drive.
    """

    def __init__(
        self,
        dc: DCSource,
        probe: Callable[[], float],
        *,
        plunger_axis: str = "p0",
        bias_axis: str = "b0",
        plunger_volts=np.linspace(-0.5, 0.5, 31),
        bias_volts=np.linspace(-0.8, 0.8, 17),
        reps_per_point: int = 8,
        compensated: bool = False,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        self.dc = dc
        self.probe = probe
        self.plunger_axis = plunger_axis
        self.bias_axis = bias_axis
        self.plunger_volts = np.asarray(plunger_volts, dtype=float)
        self.bias_volts = np.asarray(bias_volts, dtype=float)
        self.reps_per_point = int(reps_per_point)

    def sweep_points(self) -> list[tuple[float, ...]]:
        return [
            (float(v_plunger), float(v_bias))
            for v_bias in self.bias_volts
            for v_plunger in self.plunger_volts
        ]

    def sweep_extents(self):
        return {
            self.plunger_axis: (
                float(self.plunger_volts.min()), float(self.plunger_volts.max())
            ),
            self.bias_axis: (float(self.bias_volts.min()), float(self.bias_volts.max())),
        }

    def run(self, *, record=None, point_emitter=None):
        from strumento.packs.spin.experiments.coulomb_diamond import CoulombDiamond

        self._bind(record, point_emitter)
        dc = self._wrap_dc(self.dc)
        probe = self._wrap_probe(
            self.probe,
            of=len(self.bias_volts) * len(self.plunger_volts),
            reps=self.reps_per_point,
        )
        utility = CoulombDiamond(
            dc,
            plunger_axis=self.plunger_axis,
            plunger_volts=self.plunger_volts,
            bias_axis=self.bias_axis,
            bias_volts=self.bias_volts,
            probe=probe,
        )
        result = utility.run()
        fit = utility.analyze(result)
        return {
            "fit": bool(fit.success),
            "derived": dict(fit.derived),
            "moves": self.sweep_points(),
        }


class TunnelCouplingTuneBridge(SpinHostBridge):
    """Tunnel-coupling tune — the promoted :class:`TunnelCouplingTune` as a
    graph-step runnable: a detuning sweep across the interdot transition at fixed
    barrier; the utility's own Lorentzian fit extracts the tunnel coupling."""

    def __init__(
        self,
        dc: DCSource,
        probe: Callable[[], float],
        *,
        barrier_axis: str = "b0",
        voltages=np.linspace(-0.3, 0.3, 41),
        reps_per_point: int = 64,
        compensated: bool = False,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        self.dc = dc
        self.probe = probe
        self.barrier_axis = barrier_axis
        self.voltages = np.asarray(voltages, dtype=float)
        self.reps_per_point = int(reps_per_point)
        self.compensated = compensated

    def sweep_points(self) -> list[tuple[float, ...]]:
        return [(float(v),) for v in self.voltages]

    def sweep_extents(self):
        return {self.barrier_axis: (float(self.voltages.min()), float(self.voltages.max()))}

    def run(self, *, record=None, point_emitter=None):
        from strumento.packs.spin.experiments.tunnel_coupling_tune import (
            TunnelCouplingTune,
        )

        self._bind(record, point_emitter)
        dc = self._wrap_dc(self.dc)
        probe = self._wrap_probe(
            self.probe, of=len(self.voltages), reps=self.reps_per_point
        )
        utility = TunnelCouplingTune(
            dc, barrier_name=self.barrier_axis, voltages=self.voltages, probe=probe
        )
        result = utility.run()
        fit = utility.analyze(result)
        return {
            "fit": bool(fit.success),
            "derived": dict(fit.derived),
            "moves": [(float(v),) for v in self.voltages],
        }


class DispersiveGateCalibrationBridge(SpinHostBridge):
    """Dispersive gate calibration — the promoted utility as a graph-step runnable.

    Sweeps the gate detuning across the charge transition and locates the maximum
    dispersive shift through the utility's own analysis (the steepest-slope point).
    """

    def __init__(
        self,
        dc: DCSource,
        probe: Callable[[], float],
        *,
        gate_axis: str = "p0",
        voltages=np.linspace(-0.3, 0.3, 51),
        reps_per_point: int = 64,
        compensated: bool = False,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        self.dc = dc
        self.probe = probe
        self.gate_axis = gate_axis
        self.voltages = np.asarray(voltages, dtype=float)
        self.reps_per_point = int(reps_per_point)
        self.compensated = compensated

    def sweep_points(self) -> list[tuple[float, ...]]:
        return [(float(v),) for v in self.voltages]

    def sweep_extents(self):
        return {self.gate_axis: (float(self.voltages.min()), float(self.voltages.max()))}

    def run(self, *, record=None, point_emitter=None):
        from strumento.packs.spin.experiments.dispersive_gate_calibration import (
            DispersiveGateCalibration,
        )

        self._bind(record, point_emitter)
        dc = self._wrap_dc(self.dc)
        probe = self._wrap_probe(
            self.probe, of=len(self.voltages), reps=self.reps_per_point
        )
        utility = DispersiveGateCalibration(
            dc, gate_name=self.gate_axis, voltages=self.voltages, probe=probe
        )
        result = utility.run()
        fit = utility.analyze(result)
        return {
            "fit": bool(fit.success),
            "derived": dict(fit.derived),
            "moves": self.sweep_points(),
        }


class SensorTurnOnBridge(SpinHostBridge):
    """Sensor turn-on — the promoted :class:`RFReflectometryTune` as a graph-step
    runnable.

    Sweeps the reflectometry carrier through its resonance dip (the utility's own
    Lorentzian fit), then scans the demodulation phase at the found carrier to locate
    the dip's quadrature: a real dip is phase-localized (the ``demod_balance`` gate),
    and a dip distinguishable from its own noise floor is the ``noise_floor`` gate —
    "fit + priors", the table's sensor-turn-on frame. Banks the operating carrier
    frequency, demod phase, carrier power, and the sensor bias it applied.
    """

    def __init__(
        self,
        dc: DCSource,
        probe: Callable[[], float],
        set_carrier_frequency: Callable[[float], None],
        *,
        set_demod_phase: Callable[[float], None] | None = None,
        sensor_axis: str = "sensor",
        bias_v: float = -0.5,
        power_dbm: float = -30.0,
        carrier_mhz=np.linspace(300.0, 325.0, 51),
        phase_grid=np.linspace(0.0, math.pi, 9),
        reps_per_point: int = 64,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        self.dc = dc
        self.probe = probe
        self.set_carrier_frequency = set_carrier_frequency
        self.set_demod_phase = (
            set_demod_phase if set_demod_phase is not None else (lambda phi: None)
        )
        self.sensor_axis = sensor_axis
        self.bias_v = float(bias_v)
        self.power_dbm = float(power_dbm)
        self.carrier_mhz = np.asarray(carrier_mhz, dtype=float)
        self.phase_grid = np.asarray(phase_grid, dtype=float)
        if len(self.phase_grid) < 3:  # pragma: no cover - an authoring-time guard
            raise ValueError("the demod phase scan needs at least three phase settings")
        self.reps_per_point = int(reps_per_point)

    def sweep_extents(self):
        return {
            "carrier_mhz": (
                float(self.carrier_mhz.min()),
                float(self.carrier_mhz.max()),
            )
        }

    def run(self, *, record=None, point_emitter=None):
        from strumento.packs.spin.experiments.rf_reflectometry_tune import (
            RFReflectometryTune,
            _lorentzian_dip,
        )

        self._bind(record, point_emitter)
        dc = self._wrap_dc(self.dc)
        set_carrier = self._wrap_setter(self.set_carrier_frequency)
        set_demod = self._wrap_setter(self.set_demod_phase)
        of = len(self.carrier_mhz) + len(self.phase_grid)
        probe = self._wrap_probe(self.probe, of=of, reps=self.reps_per_point)

        dc.set(self.sensor_axis, self.bias_v)  # the starting sensor bias, applied

        utility = RFReflectometryTune(
            frequencies=self.carrier_mhz,
            probe=probe,
            set_frequency=set_carrier,
        )
        result = utility.run()
        fit = utility.analyze(result)

        gates: dict[str, bool] = {"fit": bool(fit.success)}
        derived: dict[str, Any] = {}
        balance_ratio = 0.0
        noise_snr = 0.0
        phase_opt = 0.0
        if fit.success:
            params = {k: float(v) for k, v in fit.params.items()}
            model = _lorentzian_dip(
                np.asarray(result.xpts, dtype=float).ravel(),
                params["f0"], params["q"], params["depth"], params["offset"],
            )
            residual = (
                np.asarray(result.avgi, dtype=float).ravel() - model
            )
            scatter = float(np.std(residual)) or 1e-12
            noise_snr = params["depth"] / scatter

            # the demod quadrature: at the fitted carrier, the dip must live on ONE
            # phase — an impostor (pickup) responds equally on every quadrature
            set_carrier(params["f0"])
            samples = []
            for phi in self.phase_grid:
                set_demod(float(phi))
                samples.append(probe())
            flat = float(np.mean(samples))
            # the dip's quadrature: the phase where the dip is DEEPEST (the reflected
            # dip subtracts on its own quadrature) — the anti-quadrature carries the
            # inverted dip and the mean sits between them, so neither |flat - s| nor a
            # magnitude metric identifies the quadrature; the deepest point does.
            phase_opt = float(self.phase_grid[int(np.argmin(samples))])
            amplitudes = np.abs(flat - np.asarray(samples))
            balance_ratio = float(amplitudes.max()) / (
                float(amplitudes.min()) + 1e-12
            )
            set_demod(phase_opt)  # park the demod on the operating quadrature

            gates["demod_balance"] = bool(balance_ratio >= DEMOD_BALANCE_RATIO_MIN)
            gates["noise_floor"] = bool(noise_snr >= NOISE_FLOOR_SNR_MIN)
            derived = {
                "carrier_f": params["f0"],
                "demod_phase": phase_opt,
                "rf_power": self.power_dbm,
                "bias": self.bias_v,
            }
        else:
            gates["demod_balance"] = False
            gates["noise_floor"] = False

        return {
            **gates,
            "derived": derived,
            "moves": [(self.bias_v,)],
        }


class EDSRScanBridge(SpinHostBridge):
    """EDSR scan — the host-driven drive-frequency sweep for the spin resonance.

    Biases the dot at its formed operating point (from the ``dot.operating_point``
    artifact) and sweeps the EDSR drive frequency, reading the spin signal per point:
    on resonance the bright-mode probability rises. A Lorentzian peak fit extracts the
    resonance; the ``resonance_prior`` gate checks the fitted frequency against the
    pack's physical prior window — an out-of-window "resonance" is an impostor.
    """

    def __init__(
        self,
        dc: DCSource,
        probe: Callable[[], float],
        set_drive_frequency: Callable[[float], None],
        *,
        root=None,
        store=None,
        ref: str = "main",
        x_axis: str = "p0",
        y_axis: str = "b0",
        drive_mhz=np.linspace(8000.0, 8400.0, 81),
        reps_per_point: int = 64,
        compensated: bool = True,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        self.dc = dc
        self.probe = probe
        self.set_drive_frequency = set_drive_frequency
        self.root = root
        self.store = store
        self.ref = ref
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.drive_mhz = np.asarray(drive_mhz, dtype=float)
        self.reps_per_point = int(reps_per_point)
        self.compensated = compensated

    def sweep_extents(self):
        return {
            "drive_mhz": (float(self.drive_mhz.min()), float(self.drive_mhz.max()))
        }

    def _lorentzian_peak(self, f, f0, gamma, amp, offset):
        return offset + amp / (1.0 + ((f - f0) / gamma) ** 2)

    def run(self, *, record=None, point_emitter=None):
        import lmfit

        self._bind(record, point_emitter)
        dc = self._wrap_dc(self.dc)
        set_drive = self._wrap_setter(self.set_drive_frequency)
        probe = self._wrap_probe(
            self.probe, of=len(self.drive_mhz), reps=self.reps_per_point
        )

        op = self._resolve_operating_point()
        move = dc.set_compensated if self.compensated else dc.set
        move(self.x_axis, op[0])
        move(self.y_axis, op[1])

        signal = []
        for f in self.drive_mhz:
            set_drive(float(f))
            signal.append(probe())
        f_grid = np.asarray(self.drive_mhz, dtype=float)
        y = np.asarray(signal, dtype=float)

        span = float(f_grid.max() - f_grid.min()) or 1.0
        model = lmfit.Model(self._lorentzian_peak)
        pars = model.make_params(
            f0=float(f_grid[int(np.argmax(y))]),
            gamma=span / 20.0,
            amp=float(y.max() - y.min()),
            offset=float(y.min()),
        )
        pars["f0"].set(min=float(f_grid.min()), max=float(f_grid.max()))
        pars["gamma"].set(min=span / 1e4)
        pars["amp"].set(min=0.0)
        out = model.fit(y, pars, f=f_grid)
        fitted = {k: float(v.value) for k, v in out.params.items()}

        f_res = fitted.get("f0", float("nan"))
        return {
            "fit": bool(out.success and fitted.get("amp", 0.0) > 0.0),
            "resonance_prior": bool(
                EDSR_FREQ_PRIOR_MHZ[0] <= f_res <= EDSR_FREQ_PRIOR_MHZ[1]
            ),
            "derived": {"f_resonance": fitted.get("f0", float("nan"))},
            "moves": [(float(op[0]), float(op[1]))],
        }

    def _resolve_operating_point(self) -> tuple[float, float]:
        from strumento.core.bringup import BringupError, resolve_calibration_artifact

        if self.store is None or self.root is None:
            raise BringupError(
                "the EDSR bridge reads dot.operating_point — wire store= and root= to"
                " the plan (the calibration artifact the step consumes)"
            )
        config = self.store.get(self.store.resolve_ref(self.ref))
        content = resolve_calibration_artifact(self.root, config, "dot.operating_point")
        point = content.get("point") if isinstance(content, Mapping) else None
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"dot.operating_point carries {point!r} — expected a 2-axis point"
            )
        return float(point[0]), float(point[1])


class DotFormationBridge(SpinHostBridge):
    """Dot formation — the promoted :class:`VirtualGateCalibration` as a graph-step
    runnable.

    Consumes the stability-map artifact (its transition-line fit picks the operating
    point one cell inside the charging line), scans a small window around that point
    for the single-dot vs double-dot geometry (ONE transition family per cut is a
    single dot; a second family is the interdot anti-crossing masquerading), and
    measures the cross-capacitance compensation through the promoted utility's own
    analysis. Banks the operating point, the lever arms (from the fitted line slope),
    and the compensation matrix — the whole dot container artifact-backed.
    """

    def __init__(
        self,
        dc: DCSource,
        probe: Callable[[], float],
        *,
        root=None,
        store=None,
        ref: str = "main",
        x_axis: str = "p0",
        y_axis: str = "b0",
        cell_offset_v: float = 0.05,
        geometry_halfwidth_v: float = 0.12,
        cut_offsets=(-0.08, -0.04, 0.0, 0.04, 0.08),
        vgc_halfwidth_v: float = 0.08,
        reps_per_point: int = 128,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        self.dc = dc
        self.probe = probe
        self.root = root
        self.store = store
        self.ref = ref
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.cell_offset_v = float(cell_offset_v)
        self.geometry_halfwidth_v = float(geometry_halfwidth_v)
        self.cut_offsets = tuple(float(o) for o in cut_offsets)
        self.vgc_halfwidth_v = float(vgc_halfwidth_v)
        self.reps_per_point = int(reps_per_point)

    def _resolve_map(self) -> dict:
        from strumento.core.bringup import BringupError, resolve_calibration_artifact

        if self.store is None or self.root is None:
            raise BringupError(
                "the dot-formation bridge reads stability.map — wire store= and"
                " root= to the plan (the calibration artifact the step consumes)"
            )
        config = self.store.get(self.store.resolve_ref(self.ref))
        return resolve_calibration_artifact(self.root, config, "stability.map")

    def _map_bounds(self) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        """The map's measured extent (the v1 cleared region) and its axis steps —
        every dot-formation move is clamped inside it by construction."""
        content = self._resolve_map()
        x_pts = np.asarray(content["x_pts"], dtype=float)
        y_pts = np.asarray(content["y_pts"], dtype=float)
        (x_lo, x_hi), (y_lo, y_hi) = content["extent"]["bounds"]
        x_step = float(abs(x_pts[1] - x_pts[0]))
        y_step = float(abs(y_pts[1] - y_pts[0]))
        return (x_lo, x_hi), (y_lo, y_hi), x_step, y_step

    def sweep_extents(self):
        if self.store is None or self.root is None:
            return {self.x_axis: (0.0, 0.0), self.y_axis: (0.0, 0.0)}
        (x_lo, x_hi), (y_lo, y_hi), x_step, y_step = self._map_bounds()
        # every move this bridge makes is clamped inside the map's measured extent —
        # the extents declare exactly that applied range
        return {
            self.x_axis: (x_lo + x_step, x_hi - x_step),
            self.y_axis: (y_lo + y_step, y_hi - y_step),
        }

    def run(self, *, record=None, point_emitter=None):
        from strumento.packs.spin.experiments.virtual_gate_calibration import (
            VirtualGateCalibration,
        )

        self._bind(record, point_emitter)
        dc = self._wrap_dc(self.dc)
        content = self._resolve_map()
        x_pts = np.asarray(content["x_pts"], dtype=float)
        y_pts = np.asarray(content["y_pts"], dtype=float)
        grid = np.asarray(content["grid"], dtype=float)
        (x_lo, x_hi), (y_lo, y_hi), x_step, y_step = self._map_bounds()

        # the charging line's fit from the map's per-row dominant transitions
        ys, xs = [], []
        for j in range(1, len(y_pts) - 1):
            position = _dominant_transition(x_pts, grid[j, :])
            if position is not None:
                ys.append(float(y_pts[j]))
                xs.append(position)
        line_ok = len(xs) >= 3
        slope, intercept = (
            np.polyfit(ys, xs, 1)
            if line_ok
            else (0.0, float(np.mean(x_pts) if xs else 0.0))
        )

        y_mid = 0.5 * (y_lo + y_hi)
        x_line = float(slope * y_mid + intercept)
        op_x = float(np.clip(x_line - self.cell_offset_v, x_lo, x_hi))
        op_y = float(np.clip(y_mid, y_lo, y_hi))

        # the geometry cuts: windows across the line at several offsets, each clamped
        # inside the map's measured extent — in-region moves only
        cuts: list[tuple[float, np.ndarray]] = []
        for offset in self.cut_offsets:
            cut_y = float(np.clip(op_y + offset, y_lo, y_hi))
            cut_x_line = float(slope * cut_y + intercept)
            lo = max(cut_x_line - self.geometry_halfwidth_v, x_lo + x_step)
            hi = min(cut_x_line + self.geometry_halfwidth_v, x_hi - x_step)
            if hi > lo:
                cuts.append((cut_y, np.linspace(lo, hi, 61)))

        vgc_lo_x = max(op_x - self.vgc_halfwidth_v, x_lo + x_step)
        vgc_hi_x = min(op_x + self.vgc_halfwidth_v, x_hi - x_step)
        vgc_lo_y = max(op_y - self.vgc_halfwidth_v, y_lo + y_step)
        vgc_hi_y = min(op_y + self.vgc_halfwidth_v, y_hi - y_step)
        vgc_x = np.linspace(vgc_lo_x, vgc_hi_x, 21)
        vgc_y = np.linspace(vgc_lo_y, vgc_hi_y, 21)
        of = sum(len(cut_x) for _, cut_x in cuts) + len(vgc_x) + len(vgc_y) + 1
        probe = self._wrap_probe(self.probe, of=of, reps=self.reps_per_point)

        moves: list[tuple[float, ...]] = []
        cut_counts = []
        for cut_y, cut_x in cuts:
            dc.set(self.y_axis, cut_y)
            signal = []
            for vx in cut_x:
                dc.set(self.x_axis, float(vx))
                signal.append(probe())
            signal = np.asarray(signal, dtype=float)
            moves.extend((float(vx), cut_y) for vx in cut_x)
            cut_counts.append(
                _count_transitions(
                    cut_x, signal, min_separation=3.0 * x_step, smooth=3
                )
            )
        single_dot = line_ok and bool(cut_counts) and all(c == 1 for c in cut_counts)

        # the compensation, through the promoted VirtualGateCalibration at the point
        alphas: dict[str, float] = {}
        vgc_ok = True
        for gate_a, gate_b, voltages, base in (
            (self.x_axis, self.y_axis, vgc_x, (op_x, op_y)),
            (self.y_axis, self.x_axis, vgc_y, (op_x, op_y)),
        ):
            dc.set(self.x_axis, op_x)
            dc.set(self.y_axis, op_y)
            utility = VirtualGateCalibration(
                dc, gate_a=gate_a, gate_b=gate_b, voltages_a=voltages, probe=probe
            )
            result = utility.run()
            analysis = utility.analyze(result)
            vgc_ok = vgc_ok and bool(analysis.success)
            alphas[f"{gate_a}->{gate_b}"] = float(
                analysis.derived.get("alpha_ij", 0.0) or 0.0
            )
            for v in voltages:
                moves.append(
                    (float(v), base[1])
                    if gate_a == self.x_axis
                    else (base[0], float(v))
                )

        dc.set(self.x_axis, op_x)  # form the dot at the operating point
        dc.set(self.y_axis, op_y)
        moves.append((op_x, op_y))

        alpha_xy = alphas.get(f"{self.x_axis}->{self.y_axis}", 0.0)
        alpha_yx = alphas.get(f"{self.y_axis}->{self.x_axis}", 0.0)
        return {
            "fit": bool(line_ok and vgc_ok),
            "single_dot_geometry": bool(single_dot),
            "derived": {
                "operating_point": {
                    "axes": [self.x_axis, self.y_axis],
                    "point": [op_x, op_y],
                    "cell_offset_v": self.cell_offset_v,
                },
                "lever_arms": {
                    "axes": [self.x_axis, self.y_axis],
                    self.x_axis: 1.0,
                    self.y_axis: float(abs(slope)),
                },
                "compensation": {
                    "axes": [self.x_axis, self.y_axis],
                    "matrix": [[1.0, -alpha_xy], [-alpha_yx, 1.0]],
                },
            },
            "moves": moves,
        }


class PSBHistogramBridge(SpinHostBridge):
    """PSB histogram — the promoted :class:`PSBHistogram` analyzer as a graph-step
    runnable.

    Moves (compensated) to the formed dot's operating point, detunes to the PSB
    readout point, and acquires single-shot PSB data; the analyzer's double-Gaussian
    fit yields the discrimination threshold and visibility. The pass is the table's
    literal: visibility ≥ the pack threshold AND the dark-space/valley check (the
    valley between the modes stays empty and the dark space populated — the
    singlet-fraction-at-detuning sanity).
    """

    def __init__(
        self,
        dc: DCSource,
        probe: Callable[[], float],
        *,
        root=None,
        store=None,
        ref: str = "main",
        x_axis: str = "p0",
        y_axis: str = "b0",
        detuning_v: float = 0.03,
        shots: int = 400,
        bins: int = 60,
        compensated: bool = True,
        **base_kw,
    ) -> None:
        super().__init__(**base_kw)
        self.dc = dc
        self.probe = probe
        self.root = root
        self.store = store
        self.ref = ref
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.detuning_v = float(detuning_v)
        self.shots = int(shots)
        self.bins = int(bins)
        self.compensated = compensated

    def _resolve_artifact(self, path: str) -> Any:
        from strumento.core.bringup import BringupError, resolve_calibration_artifact

        if self.store is None or self.root is None:
            raise BringupError(
                f"the PSB bridge reads {path} — wire store= and root= to the plan"
                " (the calibration artifacts the step consumes)"
            )
        config = self.store.get(self.store.resolve_ref(self.ref))
        return resolve_calibration_artifact(self.root, config, path)

    def sweep_extents(self):
        return {}  # a point acquisition, not a sweep — the moves carry the region check

    def run(self, *, record=None, point_emitter=None):
        from strumento.packs.spin.experiments.psb_histogram import PSBHistogram

        self._bind(record, point_emitter)
        dc = self._wrap_dc(self.dc)

        op_content = self._resolve_artifact("dot.operating_point")
        point = op_content.get("point")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"dot.operating_point carries {point!r} — expected a 2-axis point"
            )
        self._resolve_artifact("dot.compensation")  # consumed: the round-trip is real

        op = (float(point[0]), float(point[1]))
        psb = (op[0] + self.detuning_v, op[1])
        move = dc.set_compensated if self.compensated else dc.set
        move(self.x_axis, op[0])
        move(self.y_axis, op[1])
        move(self.x_axis, psb[0])
        move(self.y_axis, psb[1])

        probe = self._wrap_probe(self.probe, of=self.shots, reps=1)
        shot_values = np.asarray([probe() for _ in range(self.shots)], dtype=float)

        fit = PSBHistogram(bins=self.bins).analyze(shot_values)

        params = {k: float(v) for k, v in fit.params.items()}
        sigma_bar = 0.5 * (abs(params.get("sigma1", 1.0)) + abs(params.get("sigma2", 1.0)))
        mu_low = float(fit.derived["mu_low"])
        mu_high = float(fit.derived["mu_high"])
        threshold = float(fit.derived["threshold"])
        separation = float(fit.derived["separation"])

        valley_lo, valley_hi = mu_low + 2.0 * sigma_bar, mu_high - 2.0 * sigma_bar
        in_valley = (
            (shot_values > valley_lo) & (shot_values < valley_hi)
            if valley_hi > valley_lo
            else np.zeros(len(shot_values), dtype=bool)
        )
        valley_fraction = float(np.count_nonzero(in_valley)) / len(shot_values)
        dark_fraction = float(np.count_nonzero(shot_values < threshold)) / len(shot_values)

        return {
            "psb_visibility": bool(fit.success and separation >= PSB_VISIBILITY_MIN),
            "psb_valley": bool(
                valley_fraction <= PSB_VALLEY_MAX and dark_fraction >= PSB_DARK_FRACTION_MIN
            ),
            "derived": {
                "window": [mu_low - 3.0 * sigma_bar, mu_high + 3.0 * sigma_bar],
                "threshold": threshold,
                "visibility": separation,
            },
            "moves": [op, psb],
        }


register_host_bridge(CoulombPeakBridge)
register_host_bridge(ChargeStabilityBridge)
register_host_bridge(SensorTurnOnBridge)
register_host_bridge(DotFormationBridge)
register_host_bridge(PSBHistogramBridge)
register_host_bridge(EDSRScanBridge)
register_host_bridge(PinchOffBridge)
register_host_bridge(CoulombDiamondBridge)
register_host_bridge(TunnelCouplingTuneBridge)
register_host_bridge(DispersiveGateCalibrationBridge)
