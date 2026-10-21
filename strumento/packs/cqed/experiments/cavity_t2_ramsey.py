"""Cavity T2 Ramsey experiment.

Measure cavity dephasing time by preparing a superposition of vacuum and coherent state,
waiting a swept delay, and observing the decay of the parity fringe contrast.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_decaysin_detuned
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.library.composite import parity
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class CavityT2Ramsey(Experiment):
    """Measure the cavity dephasing time T2 via a Ramsey-style parity contrast decay.

    Sequence: ``displace(alpha)`` -- wait(t) -- ``displace(alpha, phase=180)``
    (the second, interfering arm) -- parity mapping -- measure, with the delay ``t`` swept.
    This is a cavity **Ramsey** interferometer, not a refocusing (Hahn) echo: the phase-180
    displacement is the return arm of the interferometer, with no mid-wait pi to cancel low-
    frequency noise, so it measures T2* (not T2-echo).  The second displacement interferes with
    the first; dephasing causes the parity fringe contrast to decay as a damped sinusoid.
    Fitter: ``fit_decaysin_detuned`` (damped cosine
    ``offset + amp * exp(-x/tau) * cos(2*pi*freq*x + phase)``).

    Derived quantities:
        T2       : cavity dephasing time (us)
        tau      : decay constant (same value as T2)
        detuning : oscillation frequency from the fit (cycles per x-unit)
        freq     : same as detuning

    Parameters:
        delays    : delay sweep; defaults to 0-5000 us on axis ``"t"``
        points    : number of sweep points (default 101)
        mode_name : device mode to displace and read parity on (default: the device's cavity)
        readout   : ``DispersiveIQ`` readout config; defaults to the recorded calibration
                    (``DispersiveIQ.from_calib``)

    Usage::

        exp = CavityT2Ramsey(dev, mode_name="cavity")
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["T2"]
    """

    def __init__(self, device, *, delays: Sweep | None = None, points: int = 101,
                 mode_name: str | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        mode_name = resolve_mode(device, mode_name, purpose=type(self).__name__)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=5000.0, on="t")
        validate_sweep_range(self.delays, min_val=MIN_TIME_SWEEP,
                             max_val=MAX_TIME_SWEEP, label="delay")
        self.points = points
        self.mode_name = mode_name
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        mode = getattr(dev, self.mode_name)
        # #100: parity() drives the readout and measures internally by default now -- chaining
        # our own .measure() here would double-measure.
        seq = (
            Seq()
            .play(mode.displace())
            .wait(self.delays)
            .play(mode.displace(phase=180.0))
            + parity(dev, self.mode_name)
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_decaysin_detuned(result.xpts, readout_signal(result, self.readout))
