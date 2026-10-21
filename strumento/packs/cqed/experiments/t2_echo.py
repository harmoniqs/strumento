"""T2 Echo (Hahn echo) — measure T2 without low-frequency noise (cqed expansion).

Half-pi, wait half-delay tau, pi refocusing pulse, wait tau, half-pi, measure. The refocusing
pulse cancels quasi-static dephasing, isolating the intrinsic T2 from slow frequency
fluctuations. The decay is fit to a simple exponential over total evolution time 2*tau.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_exp
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class T2Echo(Experiment):
    """Measure the qubit coherence time T2 using a Hahn spin-echo sequence.

    Sequence: ``ge_hpi`` -- wait(t) -- ``ge_pi`` (refocusing) -- wait(t) --
    ``ge_hpi`` -- measure, with the total echo delay ``2t`` swept.  The pi
    refocusing pulse cancels quasi-static dephasing, isolating the intrinsic
    T2 from slow frequency fluctuations.
    Fitter: ``fit_exp`` (exponential decay), with ``T2_echo`` derived as twice the fitted
    half-delay decay constant.

    Derived quantities:
        T2_echo : echo coherence time (us), the headline quantity
        tau     : raw half-delay decay constant

    Parameters:
        delays  : nonnegative half-delay sweep; total evolution is twice this axis
        points  : number of sweep points (default 101)
        readout : ``DispersiveIQ`` readout config; defaults to the recorded calibration
                    (``DispersiveIQ.from_calib``)

    Usage::

        exp = T2Echo(dev)
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["T2_echo"]
    """

    def __init__(self, device, *, delays: Sweep | None = None, points: int = 101,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        t1 = getattr(device.calib.qubit, "t1", 100.0)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=5.0 * t1, on="t")
        validate_sweep_range(self.delays, min_val=MIN_TIME_SWEEP,
                             max_val=MAX_TIME_SWEEP, label="delay")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        seq = (
            Seq()
            .play(q.ge_hpi())
            .wait(self.delays)
            .play(q.ge_pi())
            .wait(self.delays)
            .play(q.ge_hpi())
            .play(self.device.readout.dispersive_measure())
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        fit = fit_exp(result.xpts, readout_signal(result, self.readout))
        # ``fit_exp`` labels every exponential constant T1; echo's raw axis is a half-delay.
        fit.derived.pop("T1", None)
        fit.derived["T2_echo"] = 2.0 * fit.derived["tau"]
        return fit
