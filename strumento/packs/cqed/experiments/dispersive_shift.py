"""Dispersive shift measurement — extract chi from qubit-state-dependent readout shift (cqed expansion).

Measures the readout resonator frequency with the qubit in |g> and |e> (a ge pi pulse prepares
|e>). The separation of the two Lorentzian peaks gives 2*chi, the dispersive shift that underpins
state discrimination.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.resonator_fitters import fit_lorentzian
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_iq
from strumento.packs.cqed.readout import DispersiveIQ


class DispersiveShift(Experiment):
    """Readout frequency sweep to measure chi (the dispersive shift).

    Sweeps the readout probe frequency over a ~10 MHz span centered on the resonator frequency.
    Two acquisitions — qubit in |g> and |e> — produce two Lorentzian peaks whose separation is
    2*chi.
    """

    def __init__(self, device, *, freqs: Sweep | None = None, points: int = 201,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        f_ro = device.calib.readout.frequency
        self.freqs = freqs if freqs is not None else Sweep(start=f_ro - 5.0, stop=f_ro + 5.0, on="freq")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.freqs, Sweep) and self.freqs.on is not None:
            return (Loop(self.freqs.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        probe = self.device.readout.probe(freq=self.freqs)
        return Seq().play(self.device.qubit.ge_pi()).play(probe).measure(), ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        i, q = readout_iq(result)
        fit = fit_lorentzian(result.xpts, np.hypot(i, q))
        f0 = fit.derived["f0"]
        f_ro_calib = self.device.calib.readout.frequency
        chi = (f0 - f_ro_calib) / 2.0
        fit.derived["chi"] = chi
        return fit
