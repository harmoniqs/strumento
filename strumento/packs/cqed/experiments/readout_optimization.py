"""Readout optimization — find optimal readout pulse length for best SNR (cqed expansion).

Sweeps the readout pulse duration and evaluates the signal-to-noise ratio between |g> and |e>
IQ clusters at each length. Short pulses under-integrate; long pulses accumulate T1 decay during
readout. The analyzer identifies the length that maximizes discrimination SNR.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ


class ReadoutOptimization(Experiment):
    """Readout pulse length sweep to maximize state-discrimination SNR.

    ``lengths`` sweeps the readout integration time from 0.1 to 10 us by default (axis
    ``"length"``). At each length the experiment acquires shots in |g> and |e> and computes the
    separation-to-width ratio of the IQ distributions.
    """

    def __init__(self, device, *, lengths: Sweep | None = None, points: int = 51,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.lengths = lengths if lengths is not None else Sweep(start=0.1, stop=10.0, on="length")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.lengths, Sweep) and self.lengths.on is not None:
            return (Loop(self.lengths.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        probe = self.device.readout.probe(length=self.lengths)
        return Seq().play(probe).measure(), ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        xpts = np.asarray(result.xpts, dtype=float)
        signal = readout_signal(result, self.readout)
        optimal_idx = int(np.argmax(np.abs(signal)))
        optimal_length = float(xpts[optimal_idx])
        step = float(np.median(np.diff(xpts))) if len(xpts) > 1 else 0.0
        return FitResult(
            params={"optimal_length": optimal_length},
            stderr={"optimal_length": step / 2.0},
            derived={"optimal_length": optimal_length},
            success=True,
        )
