"""Readout-resonator spectroscopy.

Sweeps a constant probe tone across the readout resonator and fits the response — Lorentzian
magnitude by default, or the full circle fit (cable delay + IQ rotation/offset + Fano asymmetry,
loaded/internal/coupling Q) with ``method="circle"``.

The probe plays on a wiring line (default ``("readout", "probe")``) — the instance YAML declares
which generator drives the readout resonator. The analyzer is where the headline resonator
fitters enter the experiment lifecycle.

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.resonator_fitters import fit_lorentzian, fit_resonator_circle
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._signal import readout_iq


class ResonatorSpectroscopy(Experiment):
    """Sweep a probe tone across the readout resonator and extract the resonance frequency and quality factors.

    Sequence: a constant-amplitude probe pulse plays on the readout line while its
    frequency is swept across the resonator.  The magnitude (or complex S21) response
    is recorded at each frequency point.
    Fitter: ``fit_lorentzian`` (default) for a magnitude Lorentzian, or
    ``fit_resonator_circle`` (Probst/Khalil circle fit of complex S21) when
    ``method="circle"``.

    Derived quantities (method="lorentzian"):
        f0        : resonance frequency (same units as the sweep)
        hwhm      : half-width at half-maximum
        fwhm      : full-width at half-maximum (2 * hwhm)
        Q_loaded  : loaded quality factor (f0 / fwhm)

    Derived quantities (method="circle"):
        f0        : resonance frequency
        Q_loaded  : loaded quality factor
        Q_internal: internal quality factor (loss)
        Q_coupling: coupling quality factor
        phi       : impedance-mismatch angle (rad)
        fwhm      : full-width at half-maximum

    Parameters:
        freqs     : frequency sweep; defaults to +/-5 units around the calibrated readout frequency
        points    : number of sweep points (default 201)
        line      : wiring line reference for the probe tone (default ``("readout", "probe")``)
        gain      : probe tone gain; ``None`` uses the device default
        length_us : probe pulse length in us; defaults to the calibrated readout length
        method    : ``"lorentzian"`` (default) or ``"circle"`` for the full circle fit

    Usage::

        exp = ResonatorSpectroscopy(dev, method="lorentzian")
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["f0"]
    """

    def __init__(self, device, *, freqs: Sweep | None = None, points: int = 201,
                 line=("readout", "probe"),
                 gain=None, length_us: float | None = None, method: str = "lorentzian", runner=None):
        super().__init__(device, runner=runner)
        ro = device.calib.readout
        span = 5.0
        self.freqs = freqs if freqs is not None else Sweep(start=ro.frequency - span, stop=ro.frequency + span, on="f")
        self.points = points
        self.line = LineRef(*line) if not isinstance(line, LineRef) else line
        self.gain = gain
        self.length_us = length_us if length_us is not None else ro.readout_length
        if method not in ("lorentzian", "circle"):
            raise ValueError("method must be 'lorentzian' or 'circle'")
        self.method = method

    def axes(self) -> tuple:
        if isinstance(self.freqs, Sweep) and self.freqs.on is not None:
            return (Loop(self.freqs.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        return self.device.readout.dispersive_measure(
            freq=self.freqs, gain=self.gain, length=self.length_us, line=self.line,
        ), ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        i, q = readout_iq(result)
        if self.method == "circle":
            return fit_resonator_circle(result.xpts, i + 1j * q)
        return fit_lorentzian(result.xpts, np.hypot(i, q))
