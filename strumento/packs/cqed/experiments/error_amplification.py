"""Error-amplification experiment (issue #82).

Repeats a calibrated pulse ``n_pulses`` times, sweeping a target parameter (frequency or gain)
across a narrow window, and fits the amplified deviation to the zero-error parameter value — the
mechanism behind sub-linewidth pulse fine-tuning: a small per-pulse error accumulates N-fold.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from typing import cast

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_parity
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._common import is_population_anchored
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ

#: Default window half-span around the pulse's calibrated value (only meaningful for the
#: DEFAULT pulse, ``dev.qubit.ge_pi`` -- a caller who overrides ``pulse=`` to a different
#: transition/ops class should also supply ``window=``, exactly as a caller of AmplitudeRabi
#: or QubitSpectroscopy supplies their own sweep once they are off the qubit-ge default).
_DEFAULT_FREQ_SPAN_MHZ = 0.4
_DEFAULT_GAIN_SPAN = 400.0


class ErrorAmplification(Experiment):
    """Amplify a coherent frequency/gain error by repeating a calibrated pulse ``n_pulses`` times.

    A single application of the pulse under test, off its ideal frequency/gain by a small
    amount, does not perfectly complete its intended rotation. After ``n_pulses`` repeats that
    residual compounds to roughly ``n_pulses`` times the per-pulse error, so the excited-state
    population *as a function of the swept parameter* traces an oscillation whose frequency (in
    the swept-parameter axis) scales with ``n_pulses`` — tighter fringes for more repeats, which
    is the mechanism behind sub-linewidth pulse fine-tuning. That is exactly ``fit_parity``'s
    existing cosine model (``offset + amp*cos(2*pi*freq*x + phase)``), fit against the swept
    parameter instead of a phase; the corrected parameter value is the argument's zero-crossing,
    ``x0 = -phase / (2*pi*freq)`` (mod the fringe period, folded to the crossing nearest the
    window's center).

    Sequence: ``n_pulses`` copies of the named pulse (each carrying the SAME swept
    ``freq=``/``gain=`` window), then ONE measure — unrolled by construction (the same choice
    ``active_reset`` made for its own multi-round composite), which trivially keeps
    ``reads_per_shot`` correct regardless of ``n_pulses`` and sidesteps an open question about a
    ``Sweep``-carrying pulse inside a hardware ``Repeat`` body.

    Parameters
    ----------
    pulse:
        The pulse factory **method, already bound to a device's ops object** — e.g.
        ``dev.qubit.ge_pi``, ``dev.qubit.ef_pi``, ``dev.sideband.f0g1``. Every factory in this
        pack already accepts ``freq=``/``gain=`` overrides, so this works for any of them with
        no name->factory resolution machinery. Defaults to ``device.qubit.ge_pi``.
    parameter:
        ``"frequency"`` (sweeps the factory's ``freq=`` kwarg, in MHz) or ``"gain"`` (sweeps
        ``gain=``, in lab-native int codes — the factory's own safety validation
        (``validate_gain_sweep``) fires exactly as it would for any other gain-swept pulse).
    n_pulses:
        The amplification factor ``N`` (caller-supplied; defaults to 7, a common convention
        for error-amplification sequences). The experiment does not choose it.
    window:
        The swept :class:`~strumento.core.sweeps.Sweep`, in the SAME absolute units the
        factory's ``freq=``/``gain=`` kwarg expects (MHz for frequency; lab-native int codes for
        gain — matching :class:`~strumento.packs.cqed.experiments.amplitude_rabi.AmplitudeRabi`'s
        convention). Defaults to a narrow span centred on the calibrated ``ge_pi`` value; a
        caller who overrides ``pulse=`` should also override ``window=``.
    points:
        Sweep point count for the window's declared loop axis.

    Derived quantities (``FitResult.derived``):
        ``<parameter>_corrected`` : the zero-error frequency/gain (absolute, same units as
            ``window``) — e.g. ``derived["frequency_corrected"]``.
        ``n_pulses``              : the amplification factor ``N`` actually used.
        (plus ``fit_parity``'s own ``contrast``/``phase_deg``.)

    Usage::

        exp = ErrorAmplification(dev, n_pulses=7, window=Sweep(start=3999.5, stop=4000.5, on="detune"))
        outcome = exp.go(soc, reps=1000)
        outcome.fit.derived["frequency_corrected"]
    """

    #: ``parameter`` -> the factory kwarg it sweeps.
    _PARAM_KWARG = {"frequency": "freq", "gain": "gain"}

    #: Fixed-period fit: the peak must stand above the noise by this many standard errors.
    MIN_FRINGE_T = 3.0

    def __init__(self, device, *, pulse=None, parameter: str = "frequency",
                 n_pulses: int = 7, window: Sweep | None = None, points: int = 41,
                 readout: DispersiveIQ | None = None, line=("readout", "probe"),
                 fringe_period: float | None = None, runner=None):
        super().__init__(device, runner=runner)
        if parameter not in self._PARAM_KWARG:
            raise ValueError(
                f"parameter must be one of {sorted(self._PARAM_KWARG)}; got {parameter!r}"
            )
        if int(n_pulses) < 1:
            raise ValueError(f"n_pulses must be >= 1; got {n_pulses}")
        if fringe_period is not None and not float(fringe_period) > 0.0:
            raise ValueError(f"fringe_period must be > 0 (swept-parameter units); got {fringe_period}")
        #: The amplified fringe's period in the swept parameter, when physics fixes it (e.g. a gain
        #: sweep of N pulses each rotating θ₀ at gain g₀: period = 2π·g₀/(N·θ₀)). ``None`` keeps the
        #: free-frequency ``fit_parity`` path. Set it whenever the window spans less than ~1 fringe:
        #: there the free fit is degenerate (it can lock onto a period ~10× wrong and still pass).
        self.fringe_period = None if fringe_period is None else float(fringe_period)
        self.pulse = pulse if pulse is not None else device.qubit.ge_pi
        self.parameter = parameter
        self.n_pulses = int(n_pulses)
        self.points = points
        self.window = window if window is not None else self._default_window(device, parameter)
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)
        self.line = LineRef(*line) if not isinstance(line, LineRef) else line

    @staticmethod
    def _default_window(device, parameter: str) -> Sweep:
        """A window centred on the calibrated ``ge_pi`` value — only correct for that default
        pulse; a caller who passes a non-default ``pulse=`` should also pass ``window=``."""
        if parameter == "gain":
            center = float(device.calib.qubit.pulses["pi_ge"].gain)
            half = _DEFAULT_GAIN_SPAN / 2.0
        else:
            center = float(device.calib.qubit.f_ge)
            half = _DEFAULT_FREQ_SPAN_MHZ / 2.0
        return Sweep(start=center - half, stop=center + half, on="err_amp")

    def axes(self) -> tuple:
        if isinstance(self.window, Sweep) and self.window.on is not None:
            return (Loop(self.window.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        kwarg = self._PARAM_KWARG[self.parameter]
        seq = Seq()
        for _ in range(self.n_pulses):
            seq.play(self.pulse(**{kwarg: self.window}))
        seq.play(self.device.readout.dispersive_measure(line=self.line))
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        if self.fringe_period is not None:
            return self._analyze_fixed_period(result)
        fit = fit_parity(result.xpts, readout_signal(result, self.readout))
        fit.derived[f"{self.parameter}_corrected"] = self._zero_crossing(fit)
        fit.derived["n_pulses"] = float(self.n_pulses)
        return fit

    def _analyze_fixed_period(self, result: MeasurementResult) -> FitResult:
        """Known-period fringe: linear least squares for ``y = a + b·cos(kx) + c·sin(kx)``.

        Requires a POPULATION (g/e blob centres on the readout): the zero-error point is the
        fringe's maximum in P_e, and a projected signal's sign is arbitrary, so the peak/trough
        choice would be a coin toss. The peak ``x₀ = atan2(c, b)/k`` is folded to the representative
        nearest the window centre; its stderr propagates the (b, c) covariance. Success requires a
        peak inside the window that stands ``MIN_FRINGE_T`` standard errors above the noise.
        """
        from strumento.packs.cqed.experiments._signal import data_read

        assert self.fringe_period is not None  # analyze() routes here only when it is set
        key = f"{self.parameter}_corrected"
        x = np.asarray(result.xpts, dtype=float).ravel()
        derived: dict = {"n_pulses": float(self.n_pulses), "fit_mode": "fixed_period",
                         "fringe_period": self.fringe_period}
        anchored = is_population_anchored(self.readout)
        if not anchored:
            derived[key] = float("nan")
            derived["refused"] = "fixed-period fit needs a blob-anchored population (peak vs trough)"
            return FitResult(params={}, stderr={}, derived=derived, success=False)
        iq = np.stack([np.asarray(result.avgi, dtype=float), np.asarray(result.avgq, dtype=float)], axis=-1)
        y = np.asarray(data_read(self.readout.to_population(iq)), dtype=float).ravel()

        k = 2.0 * np.pi / self.fringe_period
        design = np.column_stack([np.ones_like(x), np.cos(k * x), np.sin(k * x)])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        a, b, c = (float(v) for v in coef)
        dof = max(x.size - 3, 1)
        resid = y - design @ coef
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.pinv(design.T @ design)
        amp = float(np.hypot(b, c))
        # amp stderr (gradient of hypot) and peak-phase stderr (gradient of atan2)
        g_amp = np.array([0.0, b / amp, c / amp]) if amp > 0 else np.zeros(3)
        g_phi = np.array([0.0, -c / amp**2, b / amp**2]) if amp > 0 else np.zeros(3)
        # g @ cov @ g of a 1-d g is a scalar (numpy's stubs type it as an array)
        amp_se = float(np.sqrt(max(cast(float, g_amp @ cov @ g_amp), 0.0)))
        x0_se = float(np.sqrt(max(cast(float, g_phi @ cov @ g_phi), 0.0))) / k

        x0 = float(np.arctan2(c, b)) / k
        center = 0.5 * (float(self.window.start) + float(self.window.stop))
        period = self.fringe_period
        x0 = center + ((x0 - center + period / 2.0) % period) - period / 2.0
        lo, hi = sorted((float(self.window.start), float(self.window.stop)))

        derived.update({
            key: x0,
            f"{key}_stderr": x0_se,
            "contrast": 2.0 * amp,
            "contrast_stderr": 2.0 * amp_se,
            "offset": a,
            # P_e at the fitted peak — ≈ 1 (minus readout infidelity) at the zero-error point. A
            # value far from 1 means the blob anchors are stale for this session (re-anchor them, e.g.
            # on same-session AllXY poles).
            "peak_population": a + amp,
            "residual_rms": float(np.sqrt(s2)),
        })
        significant = amp_se > 0 and amp / amp_se >= self.MIN_FRINGE_T
        ok = bool(significant and lo <= x0 <= hi and np.isfinite(x0_se))
        params = {"offset": a, "b": b, "c": c}
        stderr = {"offset": float(np.sqrt(cov[0, 0])), "b": float(np.sqrt(cov[1, 1])),
                  "c": float(np.sqrt(cov[2, 2]))}
        return FitResult(params=params, stderr=stderr, derived=derived, success=ok)

    def _zero_crossing(self, fit: FitResult) -> float:
        """``x0 = -phase / (2*pi*freq)``, folded to the crossing nearest the window's center.

        Two corrections on top of the raw formula, both because ``fit_parity``'s cosine model
        under-determines its own representation:

        1. **Amplitude-sign canonicalization.** ``offset + amp*cos(arg)`` with ``(amp, phase)``
           is the IDENTICAL curve to ``(-amp, phase + pi)`` — a legal lmfit degeneracy, not a
           fitter bug, so the optimizer is free to converge to either. This model's own
           convention puts the zero-error point at the fringe's PEAK (``argument = 0`` when
           ``x = x_ideal``, by construction in ``error_amplification_response``), so a fit that
           landed on the negative-amplitude branch is canonicalized back to that convention
           before reading ``phase`` off it — otherwise the reported correction would silently
           land on the fringe's ANTI-node (half a period off) purely from optimizer happenstance.
        2. **Period fold.** The raw formula is exact only up to an integer number of fringe
           periods (a fringe completing more than one cycle across the window is degenerate
           under that shift); the window's own center is the practical disambiguator — the
           swept range is a caller's best-current-belief neighbourhood, so the crossing *inside*
           (or nearest) it is the physically meaningful one.
        """
        freq = float(fit.params.get("freq", 0.0))
        if not freq:
            return float("nan")
        phase = float(fit.params["phase"])
        if float(fit.params.get("amp", 0.0)) < 0.0:
            phase += np.pi
        period = 1.0 / freq
        x0 = -phase / (2.0 * np.pi * freq)
        center = 0.5 * (float(self.window.start) + float(self.window.stop))
        return center + ((x0 - center + period / 2.0) % period) - period / 2.0
