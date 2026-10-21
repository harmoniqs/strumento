"""Qubit spectroscopy — two-tone measurement of the transmon g-e frequency (cqed expansion).

Drives the readout resonator at its known frequency while sweeping a second tone across the
expected qubit g-e transition. When the probe hits f_ge the dispersive shift changes the readout
response, producing a Lorentzian dip/peak in the projected signal.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Const, Pulse, Seq
from strumento.core.resonator_fitters import fit_lorentzian
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._common import drive_gain_limit
from strumento.packs.cqed.experiments._signal import readout_iq
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import validate_gain


class QubitSpectroscopy(Experiment):
    """Two-tone spectroscopy to locate the transmon g-e transition frequency.

    A probe tone sweeps ``freqs`` (centered on the expected f_ge, span 50 MHz by default) while
    the readout resonator is monitored at its dressed frequency. The dispersive response is fit
    to a Lorentzian to extract f_ge.
    """

    def __init__(self, device, *, freqs: Sweep | None = None, points: int = 201,
                 probe_gain: float = 0.03,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        f_ge = device.calib.qubit.f_ge
        self.freqs = freqs if freqs is not None else Sweep(start=f_ge - 25.0, stop=f_ge + 25.0, on="freq")
        self.points = points
        self.probe_gain = probe_gain
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.freqs, Sweep) and self.freqs.on is not None:
            return (Loop(self.freqs.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        qubit_name = self.device.qubit.element.name
        line = LineRef(qubit_name, "drive")
        validate_gain(self.probe_gain, limit=drive_gain_limit(self.device), role="qubit_drive", label="spec_probe")
        probe = Pulse(
            line=line,
            freq_mhz=self.freqs,
            gain=self.probe_gain,
            envelope=Const(length_us=10.0),
            label="spec_probe",
        )
        # The spec tone above drives the QUBIT line, never the resonator (#103) -- the
        # dispersive readout that actually reports the qubit's state still needs its own
        # probe on the readout line, alongside (not instead of) the spec tone.
        seq = Seq().play(probe) + self.device.readout.dispersive_measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        i, q = readout_iq(result)
        return fit_lorentzian(result.xpts, np.hypot(i, q))
