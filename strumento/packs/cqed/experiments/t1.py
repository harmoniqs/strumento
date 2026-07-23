"""T1 relaxation experiment (slice 3).

Excite with a ge pi pulse, idle for a swept delay, measure. The population relaxes as
``exp(-t/T1)``; the analyzer projects the averaged IQ onto the discrimination axis and fits
:func:`~strumento.core.fitters.fit_exp`.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_exp
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.readout import DispersiveIQ


class T1(Experiment):
    """Relaxation-time measurement: ``ge_pi · wait(t) · measure`` over a swept idle ``t`` (µs)."""

    def __init__(self, device, *, delays: Sweep | None = None, readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.delays = delays if delays is not None else Sweep("t", 0.0, 500.0, 101)
        self.readout = readout if readout is not None else DispersiveIQ(threshold=0.0, angle=0.0)

    def sequence(self) -> tuple[Seq, tuple]:
        seq = Seq().play(self.device.qubit.ge_pi()).wait(self.delays).measure()
        return seq, ()  # the sweep travels in the Delay; the compiler discovers it

    def analyze(self, result: MeasurementResult) -> FitResult:
        iq = np.stack([np.asarray(result.avgi), np.asarray(result.avgq)], axis=-1)
        return fit_exp(result.xpts, self.readout.project(iq))
