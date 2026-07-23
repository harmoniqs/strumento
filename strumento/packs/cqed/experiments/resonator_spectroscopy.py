"""Readout-resonator spectroscopy (slice 4 fitter wiring).

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
from strumento.core.pulses import Const, Pulse, Seq
from strumento.core.resonator_fitters import fit_lorentzian, fit_resonator_circle
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.library._gain import gain_out, maxv_for


class ResonatorSpectroscopy(Experiment):
    """Probe-tone frequency scan across the readout resonator."""

    def __init__(self, device, *, freqs: Sweep | None = None, line=("readout", "probe"),
                 gain=None, length_us: float | None = None, method: str = "lorentzian", runner=None):
        super().__init__(device, runner=runner)
        ro = device.calib.readout
        span = 5.0
        self.freqs = freqs if freqs is not None else Sweep("f", ro.frequency - span, ro.frequency + span, 201)
        self.line = LineRef(*line) if not isinstance(line, LineRef) else line
        self.gain = gain
        self.length_us = length_us if length_us is not None else ro.readout_length
        if method not in ("lorentzian", "circle"):
            raise ValueError("method must be 'lorentzian' or 'circle'")
        self.method = method

    def sequence(self) -> tuple[Seq, tuple]:
        ro = self.device.calib.readout
        maxv = maxv_for(self.device, self.line, self.freqs.start)
        probe = Pulse(
            line=self.line,
            freq_mhz=self.freqs,
            gain=gain_out(ro.gain, self.gain, maxv),
            envelope=Const(length_us=self.length_us),
            label="ro_probe",
        )
        return Seq().play(probe).measure(), ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        i, q = np.asarray(result.avgi, dtype=float), np.asarray(result.avgq, dtype=float)
        if self.method == "circle":
            return fit_resonator_circle(result.xpts, i + 1j * q)
        return fit_lorentzian(result.xpts, np.hypot(i, q))
