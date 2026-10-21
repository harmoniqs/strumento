"""Swap spectroscopy experiment.

Find the coupler pump frequency for a beamsplitter swap interaction between two
cavity modes by sweeping the coupler drive frequency.

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
from strumento.packs.cqed.library.composite import Fock, parity, prepare
from strumento.packs.cqed.library.targets import resolve_swap_pair
from strumento.packs.cqed.readout import DispersiveIQ


class SwapSpectroscopy(Experiment):
    """Sweep coupler pump frequency to locate the beamsplitter swap resonance.

    Sequence: prepare Fock |1> in the source mode, apply a coupler swap pulse at
    swept frequency, parity-map the source mode onto the qubit, readout. The
    parity signal dips at the frequency where the photon swaps out.
    Fitter: ``fit_lorentzian`` -- Lorentzian fit of the parity signal vs frequency.

    Derived quantities:
        f0       : swap resonance frequency (MHz).
        hwhm     : half-width at half-maximum (MHz).
        fwhm     : full-width at half-maximum (MHz).
        Q_loaded : loaded quality factor f0 / fwhm.

    Parameters:
        freqs  : frequency sweep (``Sweep``); defaults to 0--1 on axis ``"freq"``.
        points : number of sweep points.
        pair   : coupler pair label (e.g. ``"cavity-S1"``); defaults to the only calibrated swap.

    Usage::

        exp = SwapSpectroscopy(dev, pair="cavity-S1")
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["f0"]
    """

    def __init__(self, device, *, freqs: Sweep | None = None, points: int = 201,
                 pair: str | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        pair = resolve_swap_pair(device, pair, purpose=type(self).__name__)
        self.freqs = freqs if freqs is not None else Sweep(start=0.0, stop=1.0, on="freq")
        self.points = points
        self.pair = pair
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.freqs, Sweep) and self.freqs.on is not None:
            return (Loop(self.freqs.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        mode = self.pair.split("-")[0]
        swap_pulse = dev.coupler.swap(self.pair, "pi", freq=self.freqs)
        seq = prepare(dev, Fock(1))
        seq += Seq().play(swap_pulse)
        # #100: parity() drives the readout and measures internally by default now -- chaining
        # our own .measure() here would double-measure.
        seq += parity(dev, mode)
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)
        return fit_lorentzian(result.xpts, signal)
