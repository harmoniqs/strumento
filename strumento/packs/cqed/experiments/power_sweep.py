"""Power sweep — find optimal readout power for state discrimination (cqed expansion).

Sweeps the readout pulse gain from low to high and measures the signal-to-noise ratio between
|g> and |e> IQ blobs at each power level. Too low → poor SNR; too high → Kerr nonlinearity or
qubit excitation. The analyzer identifies the gain that maximizes discrimination SNR.

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


class PowerSweep(Experiment):
    """Readout gain sweep to maximize state-discrimination SNR.

    ``gains`` is a lab-native int-code sweep (D23) from 0 to 19000 by default (within
    MAX_READOUT_GAIN=0.6; was 0 to 30000, ~0.916 frac). At each gain
    the experiment acquires shots with the qubit in |g> and |e>, then computes the
    separation-to-width ratio of the IQ blobs.
    """

    def __init__(self, device, *, gains: Sweep | None = None, points: int = 51,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.gains = gains if gains is not None else Sweep(start=0, stop=19000, on="gain")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.gains, Sweep) and self.gains.on is not None:
            return (Loop(self.gains.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        probe = self.device.readout.probe(gain=self.gains)
        return Seq().play(self.device.qubit.ge_pi()).play(probe).measure(), ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)
        xpts = np.asarray(result.xpts, dtype=float)
        idx = int(np.argmax(np.abs(signal)))
        optimal_gain = float(xpts[idx])
        # Uncertainty: half the sweep step size (argmax resolution)
        step = float(np.median(np.diff(xpts))) if len(xpts) > 1 else 0.0
        se = step / 2.0
        return FitResult(
            params={"optimal_gain": optimal_gain},
            stderr={"optimal_gain": se},
            derived={"optimal_gain": optimal_gain},
            success=True,
        )
