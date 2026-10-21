"""Swap Rabi experiment.

Calibrate swap duration for pi (full swap) and pi/2 by sweeping the coupler pulse
duration at the swap resonance frequency.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_rabi
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.library.composite import Fock, parity, prepare
from strumento.packs.cqed.library.targets import resolve_swap_pair
from strumento.packs.cqed.readout import DispersiveIQ


class SwapRabi(Experiment):
    """Sweep coupler pulse duration to calibrate pi and pi/2 swap times.

    Sequence: prepare Fock |1> in the source mode, apply a coupler swap pulse at
    the known swap frequency with swept duration, parity-map the source mode,
    readout. The parity oscillation's period gives the full-swap (pi) time.
    Fitter: ``fit_rabi`` -- decaying-cosine fit of the parity oscillation.

    Derived quantities:
        period  : oscillation period in sweep units (us).
        pi_gain : half-period value -- the pi swap duration.

    Parameters:
        durations : duration sweep (``Sweep``); defaults to 0.01--5.0 us on axis ``"t"``.
        points    : number of sweep points.
        pair      : coupler pair label (e.g. ``"cavity-S1"``); defaults to the only calibrated swap.

    Usage::

        exp = SwapRabi(dev, pair="cavity-S1", durations=Sweep(0.01, 3.0, on="t"))
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["pi_gain"]
    """

    def __init__(self, device, *, durations: Sweep | None = None, points: int = 41,
                 pair: str | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        pair = resolve_swap_pair(device, pair, purpose=type(self).__name__)
        self.durations = durations if durations is not None else Sweep(start=0.01, stop=5.0, on="t")
        self.points = points
        self.pair = pair
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.durations, Sweep) and self.durations.on is not None:
            return (Loop(self.durations.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        mode = self.pair.split("-")[0]
        swap_pulse = dev.coupler.swap(self.pair, "pi", length=self.durations)
        seq = prepare(dev, Fock(1))
        seq += Seq().play(swap_pulse)
        # #100: parity() drives the readout and measures internally by default now -- chaining
        # our own .measure() here would double-measure.
        seq += parity(dev, mode)
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_rabi(result.xpts, readout_signal(result, self.readout))
