"""Amplitude-Rabi experiment (spec §9, plan Task 11, slice 2).

Sweeps the ge π-pulse drive amplitude and fits the resulting oscillation to find the π gain. It
subclasses the :class:`~strumento.core.experiment.Experiment` scaffold (D13): :meth:`sequence`
builds the swept program, :meth:`analyze` reduces the readout signal and fits it — no plotting.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_rabi
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.readout import DispersiveIQ


class AmplitudeRabi(Experiment):
    """Drive-amplitude Rabi on the ge transition.

    ``gains`` is a lab-native (int-code) :class:`Sweep`; the cqed factory maps it to a fraction
    sweep for the compiler and keeps the int-code axis as ``xpts`` (D23). ``readout`` projects the
    averaged IQ onto the discrimination axis before fitting.
    """

    def __init__(self, device, *, gains: Sweep | None = None, readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.gains = gains if gains is not None else Sweep("amp", 0, 30000, 51)
        self.readout = readout if readout is not None else DispersiveIQ(threshold=0.0, angle=0.0)

    def sequence(self) -> tuple[Seq, tuple]:
        seq = Seq().play(self.device.qubit.ge_pi(gain=self.gains)).measure()
        return seq, ()  # the sweep travels in the pulse; the compiler discovers it

    def analyze(self, result: MeasurementResult) -> FitResult:
        iq = np.stack([np.asarray(result.avgi), np.asarray(result.avgq)], axis=-1)
        signal = self.readout.project(iq)
        return fit_rabi(result.xpts, signal)
