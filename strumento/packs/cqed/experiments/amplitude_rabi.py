"""Amplitude-Rabi experiment.

Sweeps the ge π-pulse drive amplitude and fits the resulting oscillation to find the π gain. It
subclasses the :class:`~strumento.core.experiment.Experiment` scaffold (D13): :meth:`sequence`
builds the swept program, :meth:`analyze` reduces the readout signal and fits it — no plotting.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_rabi
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ


class AmplitudeRabi(Experiment):
    """Drive-amplitude Rabi on the ge transition.

    ``gains`` is a lab-native (int-code) :class:`Sweep`; the cqed factory maps it to a fraction
    sweep for the compiler and keeps the int-code axis as ``xpts`` (D23). The default rides the
    declared loop axis ``"amp"`` (D36): ``axes()`` emits ``Loop("amp", points)`` — a caller-supplied
    *bound* sweep (``on=…``) renames the axis; a legacy per-name sweep takes the deprecated
    auto-declared path. ``readout`` projects the averaged IQ onto the discrimination axis before
    fitting.
    """

    def __init__(self, device, *, gains: Sweep | None = None, points: int = 51,
                 readout: DispersiveIQ | None = None, line=("readout", "probe"), runner=None):
        super().__init__(device, runner=runner)
        # Default kept within MAX_QUBIT_DRIVE_GAIN (0.5 frac, ~16383 int code on a 32766-maxv
        # generator): 16000 -> ~0.488, safely under. Was 0->30000 (~0.916 frac) while the
        # sweep-endpoint safety check wasn't wired in at all — now that it is, a caller who
        # deliberately needs a wider sweep passes gains= explicitly and takes responsibility
        # for it (validated the same way regardless).
        self.gains = gains if gains is not None else Sweep(start=0, stop=16000, on="amp")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)
        self.line = LineRef(*line) if not isinstance(line, LineRef) else line

    def axes(self) -> tuple:
        if isinstance(self.gains, Sweep) and self.gains.on is not None:
            return (Loop(self.gains.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        seq = (
            Seq()
            .play(self.device.qubit.ge_pi(gain=self.gains))
            .play(self.device.readout.dispersive_measure(line=self.line))
        )
        return seq, ()  # the sweep travels in the pulse; the compiler discovers it

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_rabi(result.xpts, readout_signal(result, self.readout))
