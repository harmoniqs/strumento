"""T1 relaxation experiment.

Excite with a ge pi pulse, idle for a swept delay, measure. The population relaxes as
``exp(-t/T1)``; the analyzer projects the averaged IQ onto the discrimination axis and fits
:func:`~strumento.core.fitters.fit_exp`.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_exp
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class T1(Experiment):
    """Measure the qubit energy-relaxation time T1 via excited-state decay.

    Sequence: ``ge_pi`` -- wait(t) -- measure, with the idle delay ``t`` swept.
    The excited-state population decays as exp(-t/T1); the analyzer projects averaged
    IQ data onto the discrimination axis and fits the decay.
    Fitter: ``fit_exp`` (exponential decay ``y0 + amp * exp(-x/tau)``).

    Derived quantities:
        T1  : energy-relaxation time in the same units as the delay sweep (typically us)
        tau : decay constant (same value as T1)

    Parameters:
        delays  : delay sweep; defaults to 0-500 us on axis ``"t"``
        points  : number of sweep points (default 101)
        readout : ``DispersiveIQ`` readout config; defaults to the recorded calibration
                    (``DispersiveIQ.from_calib``)
        line    : wiring line reference for the readout probe tone (default
                  ``("readout", "probe")``) — override on a single-generator instance that has
                  no dedicated readout line (e.g. ``("qubit", "drive")``, issue #102).

    Usage::

        exp = T1(dev)
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["T1"]
    """

    def __init__(self, device, *, delays: Sweep | None = None, points: int = 101,
                 readout: DispersiveIQ | None = None, line=("readout", "probe"), runner=None):
        super().__init__(device, runner=runner)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=500.0, on="t")
        validate_sweep_range(self.delays, min_val=MIN_TIME_SWEEP,
                             max_val=MAX_TIME_SWEEP, label="delay")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)
        self.line = LineRef(*line) if not isinstance(line, LineRef) else line

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        seq = (
            Seq()
            .play(self.device.qubit.ge_pi())
            .wait(self.delays)
            .play(self.device.readout.dispersive_measure(line=self.line))
        )
        return seq, ()  # the sweep travels in the Delay; the compiler discovers it

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_exp(result.xpts, readout_signal(result, self.readout))
