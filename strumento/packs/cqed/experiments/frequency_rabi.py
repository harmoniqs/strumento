"""Frequency Rabi (chevron) — Rabi oscillation vs drive detuning (cqed expansion).

Sweeps the qubit drive frequency at fixed amplitude and duration to produce a chevron pattern.
On resonance the Rabi frequency is minimal (set by the drive amplitude); off resonance the
effective Rabi frequency increases as sqrt(Omega^2 + delta^2). A Lorentzian fit of the signal
at the target duration extracts the resonance center.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.resonator_fitters import fit_lorentzian
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ


class FrequencyRabi(Experiment):
    """Drive-frequency sweep (chevron slice) at fixed amplitude and duration.

    Sequence: ge pi pulse at swept frequency, readout. At the resonance
    frequency the Rabi signal peaks (or dips, depending on the pulse area);
    a Lorentzian fit identifies the center.
    Fitter: ``fit_lorentzian`` -- Lorentzian peak/dip fit of the dispersive signal.

    Derived quantities:
        f0       : resonance center frequency (MHz).
        hwhm     : half-width at half-maximum (MHz).
        fwhm     : full-width at half-maximum (MHz).
        Q_loaded : loaded quality factor f0 / fwhm.

    Parameters:
        freqs  : frequency sweep (``Sweep``); defaults to f_ge +/- 20 MHz on axis ``"freq"``.
        points : number of sweep points.

    Usage::

        exp = FrequencyRabi(dev, freqs=Sweep(4980, 5020, on="freq"))
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["f0"]
    """

    def __init__(self, device, *, freqs: Sweep | None = None, points: int = 101,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        f_ge = device.calib.qubit.f_ge
        self.freqs = freqs if freqs is not None else Sweep(start=f_ge - 20.0, stop=f_ge + 20.0, on="freq")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.freqs, Sweep) and self.freqs.on is not None:
            return (Loop(self.freqs.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        seq = (
            Seq()
            .play(q.ge_pi(freq=self.freqs))
            .play(self.device.readout.dispersive_measure())
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)
        return fit_lorentzian(result.xpts, signal)
