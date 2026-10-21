"""Backend-agnostic data import — bring external measurement data into Strumento.

Any control backend (QUA/OPX, Qblox, Zurich Instruments, custom scripts, CSV files)
produces arrays of I/Q data. This module converts those arrays into a
:class:`~strumento.core.result.MeasurementResult` so the entire Strumento analysis
pipeline — fitters, display, datachest, bringup verdicts — works regardless of where
the data came from.

This is the **temporary multi-backend bridge**: it decouples analysis from acquisition.
A full backend wrapper (compiling ``Seq`` to QUA/Qblox programs) is future work; this
module makes the analysis side useful *today* for anyone with raw data.

Pure: no ``qick``, no ``qm``, no ``qblox`` — only numpy (``xarray`` is optional and
lazily imported only when :func:`from_xarray` is called).

Usage::

    from strumento.interop.backends import from_iq, from_complex, from_xarray

    # From any Python script that produces numpy arrays:
    result = from_iq(freqs, i_data, q_data, backend="opx")

    # Complex S21 from a VNA or network analyzer:
    result = from_complex(freqs, s21_complex)

    # From a Quantify / Qblox xarray Dataset:
    result = from_xarray(dataset, x="x0", i="y0", q="y1", backend="qblox")

    # Then use Strumento's full analysis pipeline:
    from strumento.core.resonator_fitters import fit_lorentzian
    fit = fit_lorentzian(result.xpts, np.hypot(result.avgi, result.avgq))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from strumento.core.environment import stamp_environment
from strumento.core.experiment import (
    DEFAULT_RECOVERY_TOLERANCE,
    Experiment,
    ExperimentOutcome,
    _emit_figure,
    _emit_fit,
    _emit_gates,
)
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult


def from_iq(
    xpts,
    i: Any,
    q: Any,
    *,
    shots: Any | None = None,
    backend: str = "external",
    meta: dict | None = None,
) -> MeasurementResult:
    """Build a :class:`MeasurementResult` from separate I and Q arrays.

    This is the universal entry point — any backend that produces real-valued I and Q
    arrays (one per sweep point) can use this.

    Args:
        xpts: the swept independent axis (frequencies, delays, gains, …). ``None`` for
            a single-point measurement.
        i: in-phase signal, shape ``(n_points,)`` or scalar.
        q: quadrature signal, same shape as ``i``.
        shots: optional per-shot data, shape ``(n_points, n_shots, 2)`` or similar.
        backend: a tag for the source backend (``"opx"``, ``"qblox"``, ``"zurich"``,
            ``"custom"``, …). Stamped on ``meta["backend"]``.
        meta: additional metadata to attach (merged with the backend tag).

    Returns:
        A :class:`MeasurementResult` ready for any Strumento fitter or display function.
    """
    m = {"backend": backend, "source": "strumento.interop.backends.from_iq"}
    if meta:
        m.update(meta)
    return MeasurementResult(
        xpts=np.asarray(xpts) if xpts is not None else None,
        avgi=np.asarray(i, dtype=float),
        avgq=np.asarray(q, dtype=float),
        shots=np.asarray(shots) if shots is not None else None,
        meta=m,
    )


def from_complex(
    xpts,
    signal: Any,
    *,
    shots: Any | None = None,
    backend: str = "external",
    meta: dict | None = None,
) -> MeasurementResult:
    """Build a :class:`MeasurementResult` from a complex-valued signal array.

    Many backends and instruments (VNAs, network analyzers, QUA ``demod.full``) return
    complex S-parameters or demodulated IQ as a single complex array. This splits it
    into I (real) and Q (imaginary) for Strumento.

    Args:
        xpts: the swept independent axis, or ``None``.
        signal: complex-valued array, shape ``(n_points,)``. Each element is ``I + jQ``.
        shots: optional per-shot complex data.
        backend: source backend tag.
        meta: additional metadata.
    """
    s = np.asarray(signal, dtype=complex)
    shot_arr = None
    if shots is not None:
        shot_c = np.asarray(shots, dtype=complex)
        shot_arr = np.stack([shot_c.real, shot_c.imag], axis=-1)
    return from_iq(xpts, s.real, s.imag, shots=shot_arr, backend=backend, meta=meta)


def from_counts(
    xpts,
    excited_fraction: Any,
    *,
    total_shots: int | None = None,
    backend: str = "external",
    meta: dict | None = None,
) -> MeasurementResult:
    """Build a :class:`MeasurementResult` from population / state-assignment data.

    Some backends return discriminated results (excited-state probability per point)
    rather than raw IQ. This maps the population to the I channel (Q = 0) so fitters
    that operate on a scalar signal (Rabi, T1 decay) work directly.

    Args:
        xpts: the swept independent axis, or ``None``.
        excited_fraction: probability of the excited state at each point, shape
            ``(n_points,)``, values in [0, 1].
        total_shots: if provided, stamped on meta for provenance.
        backend: source backend tag.
        meta: additional metadata.
    """
    p = np.asarray(excited_fraction, dtype=float)
    m: dict[str, Any] = {"backend": backend, "source": "strumento.interop.backends.from_counts",
                         "signal_kind": "excited_fraction"}
    if total_shots is not None:
        m["total_shots"] = int(total_shots)
    if meta:
        m.update(meta)
    return MeasurementResult(
        xpts=np.asarray(xpts) if xpts is not None else None,
        avgi=p,
        avgq=np.zeros_like(p),
        meta=m,
    )


def from_multiread(
    xpts,
    reads: list[tuple[Any, Any]],
    *,
    backend: str = "external",
    meta: dict | None = None,
) -> MeasurementResult:
    """Build a :class:`MeasurementResult` from multiple readouts per point.

    Some protocols (active reset, QND checks, validation) acquire multiple reads per
    sweep point. Each read is an ``(I, Q)`` pair of arrays with shape ``(n_points,)``.
    The result stacks them on axis 0: ``avgi`` and ``avgq`` have shape
    ``(n_reads, n_points)``.

    Args:
        xpts: the swept independent axis, or ``None``.
        reads: list of ``(i_array, q_array)`` tuples, one per read.
        backend: source backend tag.
        meta: additional metadata.
    """
    i_stack = np.stack([np.asarray(i, dtype=float) for i, _ in reads])
    q_stack = np.stack([np.asarray(q, dtype=float) for _, q in reads])
    m = {"backend": backend, "source": "strumento.interop.backends.from_multiread",
         "n_reads": len(reads)}
    if meta:
        m.update(meta)
    return MeasurementResult(
        xpts=np.asarray(xpts) if xpts is not None else None,
        avgi=i_stack,
        avgq=q_stack,
        meta=m,
    )


def from_xarray(
    dataset,
    *,
    x: str = "x0",
    i: str = "y0",
    q: str = "y1",
    x_scale: float = 1.0,
    backend: str = "external",
    meta: dict | None = None,
) -> MeasurementResult:
    """Build a :class:`MeasurementResult` from an ``xarray.Dataset``.

    Quantify-core's ``MeasurementControl`` and Qblox's ``InstrumentCoordinator``
    return results as ``xarray.Dataset`` objects. This extracts the coordinate and
    data variables by name and delegates to :func:`from_iq`.

    ``xarray`` is imported **lazily** — this function is the only place it is needed,
    and it raises a clear ``ImportError`` if the package is not installed. The rest of
    the module stays pure-numpy.

    Args:
        dataset: an ``xarray.Dataset`` (or any object whose ``[key]`` returns objects
            with a ``.values`` attribute — so a plain dict of array-likes works too).
        x: name of the coordinate variable for the swept axis (default ``"x0"``).
        i: name of the data variable for the I channel (default ``"y0"``).
        q: name of the data variable for the Q channel (default ``"y1"``).
        x_scale: multiplicative scale applied to the x-axis values (e.g. ``1e-6`` to
            convert Hz → MHz). Default ``1.0`` (no conversion).
        backend: source backend tag.
        meta: additional metadata.

    Raises:
        KeyError: if ``x``, ``i``, or ``q`` are not found in the dataset. The error
            message lists the available keys so the caller can fix the names.

    Usage::

        # Quantify MeasurementControl returns an xarray Dataset
        dataset = mc.run()
        result = from_xarray(dataset, x="x0", i="y0", q="y1",
                             x_scale=1e-6, backend="qblox")

        # Also works with a Zurich Instruments result dataset
        result = from_xarray(zi_dataset, x="frequency", i="real", q="imag",
                             backend="zurich")
    """
    _available = _xarray_keys(dataset)

    for name, key in [("x (coordinate)", x), ("i (I channel)", i), ("q (Q channel)", q)]:
        if key not in _available:
            raise KeyError(
                f"from_xarray: {name} key {key!r} not found in dataset "
                f"(available: {sorted(_available)})"
            )

    xpts = np.asarray(_xarray_values(dataset, x)) * x_scale
    i_data = np.asarray(_xarray_values(dataset, i))
    q_data = np.asarray(_xarray_values(dataset, q))

    ds_meta: dict[str, Any] = {"xarray_keys": {"x": x, "i": i, "q": q}}
    if x_scale != 1.0:
        ds_meta["x_scale"] = x_scale
    if meta:
        ds_meta.update(meta)

    return from_iq(xpts, i_data, q_data, backend=backend, meta=ds_meta)


def from_xarray_complex(
    dataset,
    *,
    x: str = "x0",
    signal: str = "y0",
    x_scale: float = 1.0,
    backend: str = "external",
    meta: dict | None = None,
) -> MeasurementResult:
    """Build a :class:`MeasurementResult` from an ``xarray.Dataset`` with a complex variable.

    Some instruments and frameworks store the result as a single complex-valued data
    variable (``I + jQ``). This extracts it and delegates to :func:`from_complex`.

    Args:
        dataset: an ``xarray.Dataset`` or dict-like.
        x: name of the coordinate variable for the swept axis.
        signal: name of the complex-valued data variable.
        x_scale: multiplicative scale for the x-axis.
        backend: source backend tag.
        meta: additional metadata.
    """
    _available = _xarray_keys(dataset)
    for name, key in [("x (coordinate)", x), ("signal (complex)", signal)]:
        if key not in _available:
            raise KeyError(
                f"from_xarray_complex: {name} key {key!r} not found in dataset "
                f"(available: {sorted(_available)})"
            )

    xpts = np.asarray(_xarray_values(dataset, x)) * x_scale
    s = np.asarray(_xarray_values(dataset, signal))

    ds_meta: dict[str, Any] = {"xarray_keys": {"x": x, "signal": signal}}
    if x_scale != 1.0:
        ds_meta["x_scale"] = x_scale
    if meta:
        ds_meta.update(meta)

    return from_complex(xpts, s, backend=backend, meta=ds_meta)


# ---------------------------------------------------------------------------
# Quick-analyze helpers (import + fit in one call)
# ---------------------------------------------------------------------------


def analyze_resonator(
    xpts,
    i,
    q,
    *,
    method: str = "lorentzian",
    backend: str = "external",
    meta: dict | None = None,
) -> FitResult:
    """Import frequency-sweep data and fit the resonator response in one call.

    Args:
        xpts: frequency axis (MHz).
        i, q: in-phase and quadrature signal arrays.
        method: ``"lorentzian"`` (magnitude dip) or ``"circle"`` (full complex S21).
        backend: source backend tag.
        meta: additional metadata.

    Returns:
        A :class:`~strumento.core.fitters.FitResult` with derived quantities:
        ``f0``, ``hwhm``, ``fwhm``, ``Q_loaded`` (lorentzian); plus ``Q_internal``,
        ``Q_coupling``, ``phi`` (circle).
    """
    from strumento.core.resonator_fitters import fit_lorentzian, fit_resonator_circle

    result = from_iq(xpts, i, q, backend=backend, meta=meta)
    if method == "circle":
        return fit_resonator_circle(result.xpts, result.avgi + 1j * result.avgq)
    return fit_lorentzian(result.xpts, np.hypot(result.avgi, result.avgq))


def analyze_rabi(
    xpts,
    i,
    q,
    *,
    backend: str = "external",
    meta: dict | None = None,
) -> FitResult:
    """Import amplitude-sweep data and fit the Rabi oscillation in one call.

    Fits the I channel (``avgi``). Returns ``period`` and ``pi_gain``.
    """
    from strumento.core.fitters import fit_rabi

    result = from_iq(xpts, i, q, backend=backend, meta=meta)
    return fit_rabi(result.xpts, result.avgi)


def analyze_t1(
    xpts,
    i,
    q,
    *,
    backend: str = "external",
    meta: dict | None = None,
) -> FitResult:
    """Import delay-sweep data and fit the T1 exponential decay in one call.

    Returns ``T1`` and ``tau`` (same value, both for compatibility).
    """
    from strumento.core.fitters import fit_exp

    result = from_iq(xpts, i, q, backend=backend, meta=meta)
    return fit_exp(result.xpts, result.avgi)


def analyze_t2_ramsey(
    xpts,
    i,
    q,
    *,
    backend: str = "external",
    meta: dict | None = None,
) -> FitResult:
    """Import Ramsey delay-sweep data and fit the damped fringes in one call.

    Returns ``T2``, ``tau``, ``detuning``, and ``freq``.
    """
    from strumento.core.fitters import fit_decaysin_detuned

    result = from_iq(xpts, i, q, backend=backend, meta=meta)
    return fit_decaysin_detuned(result.xpts, result.avgi)


def analyze_spectroscopy(
    xpts,
    i,
    q,
    *,
    backend: str = "external",
    meta: dict | None = None,
) -> FitResult:
    """Import two-tone spectroscopy data and fit the Lorentzian dip in one call.

    Returns ``f0``, ``hwhm``, ``fwhm``, ``Q_loaded``.
    """
    from strumento.core.resonator_fitters import fit_lorentzian

    result = from_iq(xpts, i, q, backend=backend, meta=meta)
    return fit_lorentzian(result.xpts, np.hypot(result.avgi, result.avgq))


# ---------------------------------------------------------------------------
# ExperimentOutcome builder (external data → verdict-ready object)
# ---------------------------------------------------------------------------


def external_outcome(
    xpts,
    i,
    q,
    *,
    fit_fn,
    signal_fn=None,
    shots=None,
    backend: str = "external",
    meta: dict | None = None,
) -> ExperimentOutcome:
    """Build an :class:`~strumento.core.experiment.ExperimentOutcome` from external data.

    Produces the same object that ``Experiment.go()`` returns — a bundle of
    ``MeasurementResult`` + ``FitResult`` — so external data can participate in
    the verdict system (bringup gates, datachest, record emission).

    Args:
        xpts: swept axis, or ``None``.
        i, q: in-phase and quadrature signal arrays.
        fit_fn: a fitter function ``(xpts, signal) -> FitResult`` — e.g.
            ``fit_rabi``, ``fit_exp``, ``fit_lorentzian``.
        signal_fn: optional projection ``(MeasurementResult) -> array`` to extract
            the scalar signal from IQ before fitting. Default: uses ``avgi`` directly.
            Example: ``lambda r: np.hypot(r.avgi, r.avgq)`` for magnitude.
        shots: optional per-shot data.
        backend: source backend tag.
        meta: additional metadata.

    Returns:
        An :class:`~strumento.core.experiment.ExperimentOutcome` with ``.result``
        and ``.fit`` populated. ``.figure`` is always ``None`` (no display context).

    Usage::

        from strumento.interop.backends import external_outcome
        from strumento.core.fitters import fit_rabi

        outcome = external_outcome(gains, I, Q, fit_fn=fit_rabi, backend="opx")
        print(outcome.fit.derived["pi_gain"])
        print(outcome.fit.success)  # the verdict
    """
    result = from_iq(xpts, i, q, shots=shots, backend=backend, meta=meta)
    signal = signal_fn(result) if signal_fn is not None else result.avgi
    try:
        fit = fit_fn(result.xpts, signal)
    except Exception:
        fit = None
    return ExperimentOutcome(result=result, fit=fit)


# ---------------------------------------------------------------------------
# External experiment adapter (slot existing scripts into bringup campaigns)
# ---------------------------------------------------------------------------


class ExternalExperiment(Experiment):
    """Wraps an external acquisition function as a Strumento :class:`Experiment`.

    This adapter lets an existing script (QUA, Qblox, custom Python) participate
    in the full experiment lifecycle — ``go()`` returns an ``ExperimentOutcome``
    with gate verdicts, records, and artifacts, exactly as a native Strumento
    experiment would. It can be used as a bringup step, stored in DataChest, and
    its verdicts feed the bringup runner's routing logic.

    The adapter overrides ``go()`` to call the user's ``acquire_fn`` instead of
    the ``Runner`` → compile → acquire path. The ``sequence()`` and ``analyze()``
    abstract methods are implemented but only used internally.

    Args:
        device: the Strumento Device instance (for calibration context).
        acquire_fn: a callable ``(device) -> (xpts, i, q)`` or
            ``(device) -> MeasurementResult`` that runs the external experiment
            and returns the data. The simplest form returns three arrays; if it
            returns a ``MeasurementResult`` directly, that is used as-is.
        fit_fn: a fitter function ``(xpts, signal) -> FitResult``.
        signal_fn: optional IQ projection ``(MeasurementResult) -> array``.
            Default: uses ``avgi``.
        backend: source backend tag.
        meta: additional metadata stamped on every result.

    Usage::

        from strumento.interop.backends import ExternalExperiment
        from strumento.core.fitters import fit_rabi

        def my_opx_rabi(device):
            # ... your QUA program, execute, fetch ...
            return gains, I, Q

        exp = ExternalExperiment(dev, acquire_fn=my_opx_rabi, fit_fn=fit_rabi,
                                  backend="opx")

        # Works everywhere a Strumento Experiment works:
        outcome = exp.go(soc=None)
        outcome = exp.go(soc=None, record=child_record)  # emits gate verdicts
    """

    def __init__(
        self,
        device,
        *,
        acquire_fn,
        fit_fn,
        signal_fn=None,
        backend: str = "external",
        meta: dict | None = None,
        runner=None,
    ):
        super().__init__(device, runner=runner)
        self._acquire_fn = acquire_fn
        self._fit_fn = fit_fn
        self._signal_fn = signal_fn
        self._backend = backend
        self._meta = meta or {}

    def sequence(self) -> tuple[Seq, tuple]:
        """Not used for external experiments — go() bypasses the Runner."""
        return Seq(), ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        """Fit the externally acquired result."""
        signal = self._signal_fn(result) if self._signal_fn is not None else result.avgi
        return self._fit_fn(result.xpts, signal)

    def go(
        self,
        soc=None,
        *,
        analyze: bool = True,
        display: bool = False,
        record=None,
        recovery_tolerance: float = DEFAULT_RECOVERY_TOLERANCE,
        **run_kw,
    ) -> ExperimentOutcome:
        """Run the external acquisition, fit, and emit verdicts.

        The ``soc`` parameter is accepted for API compatibility with
        ``Experiment.go()`` but is NOT passed to the acquire function —
        the external script manages its own hardware connection.
        """
        # Acquire via the external function
        raw = self._acquire_fn(self.device)

        # Accept either (xpts, i, q) tuple or a MeasurementResult directly
        if isinstance(raw, MeasurementResult):
            result = raw
            if "backend" not in result.meta:
                result.meta["backend"] = self._backend
        else:
            xpts, i, q = raw
            result = from_iq(xpts, i, q, backend=self._backend, meta=dict(self._meta))

        # Environment stamp — same invariant as Runner.run(): every result exiting
        # the experiment lifecycle has a stamp, even a null one. The external script
        # has no soc, so the stamp carries the device's signal-chain metadata and a
        # NullEnvironment reading. An existing stamp (e.g. from a MeasurementResult
        # the acquire_fn built itself) is not overwritten.
        result = stamp_environment(result, device=self.device, soc=soc)

        # Fit
        fit = self.analyze(result) if analyze else None
        figure = self.display(result, fit) if display else None

        # Emit record events — same contract as Experiment.go()
        if record is not None:
            if fit is not None:
                _emit_fit(record, self, fit)
                _emit_gates(record, self, soc, fit, tolerance=recovery_tolerance)
            if figure is not None:
                _emit_figure(record, self, figure)

        return ExperimentOutcome(result=result, fit=fit, figure=figure)


# ---------------------------------------------------------------------------
# xarray helpers (lazy import, duck-typed for plain dicts too)
# ---------------------------------------------------------------------------


def _xarray_keys(dataset) -> set[str]:
    """Return all available keys in a dataset (coords + data vars, or dict keys)."""
    keys: set[str] = set()
    # xarray.Dataset: coords + data_vars
    if hasattr(dataset, "coords") and hasattr(dataset, "data_vars"):
        keys.update(dataset.coords)
        keys.update(dataset.data_vars)
    # dict-like fallback
    elif hasattr(dataset, "keys"):
        keys.update(dataset.keys())
    else:
        raise TypeError(
            f"from_xarray expects an xarray.Dataset or a dict-like, "
            f"got {type(dataset).__name__}"
        )
    return keys


def _xarray_values(dataset, key: str):
    """Extract values from a dataset key, handling both xarray and plain dicts."""
    item = dataset[key]
    # xarray DataArray → .values
    if hasattr(item, "values"):
        return item.values
    return item


# ---------------------------------------------------------------------------
# Optimal-control pulse export (Piccolo → any backend)
# ---------------------------------------------------------------------------


@dataclass
class WaveformExport:
    """A single OC-optimized drive waveform, ready for injection into any backend.

    The waveform is **normalized** (``idata`` and ``qdata`` peak at 1.0) with
    the calibrated gain in ``gain``. To get absolute-amplitude samples for upload:
    ``samples = wf.idata * wf.gain``.
    """

    idata: np.ndarray
    """In-phase samples, normalized to [-1, 1]."""

    qdata: np.ndarray | None
    """Quadrature samples, normalized to [-1, 1], or ``None`` for real-valued drives."""

    gain: float
    """Calibrated gain fraction (0 to 1) — the peak physical amplitude converted
    through the ``gain_calib`` reference."""

    freq_mhz: float
    """Carrier frequency in MHz — the frequency the backend's NCO / mixer targets."""

    duration_us: float
    """Total pulse duration in microseconds."""

    n_samples: int
    """Number of waveform samples."""

    dac_rate: float | None
    """Sample rate in samples/us, or ``None`` if the solution's native grid was kept."""

    line: tuple[str, str]
    """The hardware line as ``(line_name, role)`` — maps to the backend's port/element."""

    drive_name: str
    """The Piccolo drive label (e.g. ``"qubit:x"``) — round-trips with ``to_piccolo_system``."""

    def absolute_samples(self) -> np.ndarray:
        """I-channel samples at absolute amplitude: ``idata * gain``."""
        return self.idata * self.gain

    def absolute_samples_iq(self) -> tuple[np.ndarray, np.ndarray | None]:
        """``(I, Q)`` samples at absolute amplitude, or ``(I, None)`` for real drives."""
        q = self.qdata * self.gain if self.qdata is not None else None
        return self.idata * self.gain, q


def extract_oc_pulses(
    solution,
    *,
    drive_map: dict,
    gain_calib,
    dac_rate: float | None = None,
) -> dict[str, WaveformExport]:
    """Extract Piccolo-optimized waveforms as plain arrays for any backend.

    Same inputs as :func:`~strumento.core.pulses.from_solution` (solution, drive_map,
    gain_calib, dac_rate) but returns a dict of :class:`WaveformExport` objects instead
    of a Strumento ``Seq``. Each export carries the processed waveform arrays (resampled,
    normalized, gain-calibrated) ready for injection into QUA configs, Quantify schedules,
    Zurich HDAWG uploads, or any other backend's pulse definition format.

    Args:
        solution: a mapping or object with ``times`` (control grid in us) and
            ``controls`` (mapping of drive name → amplitude samples in physical
            units, MHz Rabi rate). This is the Piccolo solve output.
        drive_map: ``{drive_name: (line_ref, carrier_freq_mhz)}`` — where each
            drive plays and at what carrier. ``line_ref`` is a
            :class:`~strumento.core.wiring.LineRef` or a ``(line, role)`` tuple.
        gain_calib: physical amplitude → gain fraction: any callable (e.g.
            ``PiPulseReference`` or ``SpinGainReference``).
        dac_rate: target sample rate (samples/us). ``None`` keeps the solution grid.

    Returns:
        ``{drive_name: WaveformExport}`` — one entry per drive in ``controls``.

    Usage::

        from strumento.interop.backends import extract_oc_pulses

        pulses = extract_oc_pulses(solution, drive_map=..., gain_calib=gain_ref)

        # QUA: inject into config
        wf = pulses["qubit:x"]
        config["waveforms"]["oc_I"] = {
            "type": "arbitrary", "samples": wf.absolute_samples().tolist()
        }

        # Qblox: inject into schedule
        sched.add(NumericalPulse(samples=wf.absolute_samples(), ...))

        # Zurich: upload to AWG
        awg.set_wave("oc_pulse", wf.absolute_samples())
    """
    from strumento.core.pulses import _solution_fields
    from strumento.core.wiring import LineRef

    times, controls = _solution_fields(solution)
    times = np.asarray(times, dtype=float)

    if dac_rate is not None:
        n = max(2, int(round((times[-1] - times[0]) * dac_rate)) + 1)
        grid = np.linspace(times[0], times[-1], n)
    else:
        grid = times

    duration_us = float(grid[-1] - grid[0])
    exports: dict[str, WaveformExport] = {}

    for name, samples in controls.items():
        if name not in drive_map:
            raise KeyError(
                f"drive {name!r} has no entry in drive_map "
                f"(have {sorted(drive_map)})"
            )
        line_ref, carrier = drive_map[name]
        if isinstance(line_ref, LineRef):
            line_tuple = (line_ref.line, line_ref.role)
        else:
            line_tuple = (str(line_ref[0]), str(line_ref[1]))

        arr = np.asarray(samples)
        idata = np.interp(grid, times, np.real(arr))
        qdata = np.interp(grid, times, np.imag(arr)) if np.iscomplexobj(arr) else None

        peak = float(np.max(np.abs(arr))) or 1.0

        exports[name] = WaveformExport(
            idata=idata / peak,
            qdata=(qdata / peak) if qdata is not None else None,
            gain=float(gain_calib(peak)),
            freq_mhz=float(carrier),
            duration_us=duration_us,
            n_samples=len(grid),
            dac_rate=dac_rate,
            line=line_tuple,
            drive_name=name,
        )

    return exports
