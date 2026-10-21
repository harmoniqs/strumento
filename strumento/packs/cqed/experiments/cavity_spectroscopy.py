"""Cavity spectroscopy — find cavity/storage cavity mode frequencies (cqed expansion).

Sweeps a probe tone across a bosonic mode (the cavity or a storage mode, selected by ``mode_name``) and fits
the transmitted/reflected response to a Lorentzian to extract the bare or dressed cavity
frequency. The span and center default to the mode's calibrated frequency.

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
from strumento.packs.cqed.experiments._common import displace_gain_limit
from strumento.packs.cqed.experiments._signal import readout_iq
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import validate_gain


class CavitySpectroscopy(Experiment):
    """Probe-tone frequency scan across a cavity mode (cavity or storage).

    ``mode_name`` selects which cavity mode to probe (default: the device's cavity). The frequency
    sweep defaults to the mode's calibrated frequency +/- a span suited for locating the mode.
    A Lorentzian fit extracts the resonance frequency.
    """

    def __init__(self, device, *, freqs: Sweep | None = None, points: int = 401,
                 mode_name: str | None = None, probe_gain: float = 0.03,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        mode_name = resolve_mode(device, mode_name, purpose=type(self).__name__)
        self.mode_name = mode_name
        mode_ops = getattr(device, mode_name)
        f_cav = getattr(mode_ops.element, 'freq_mhz', None) or 5200.0
        self.freqs = freqs if freqs is not None else Sweep(start=f_cav - 5.0, stop=f_cav + 5.0, on="freq")
        self.points = points
        self.probe_gain = probe_gain
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.freqs, Sweep) and self.freqs.on is not None:
            return (Loop(self.freqs.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        mode_ops = getattr(self.device, self.mode_name)
        el = mode_ops.element
        line = LineRef(el.line or el.name, "main")
        validate_gain(self.probe_gain, limit=displace_gain_limit(self.device, el.name),
                      role="cavity_probe", label="cav_probe", element=el.name)
        probe = Pulse(
            line=line,
            freq_mhz=self.freqs,
            gain=self.probe_gain,
            envelope=Const(length_us=10.0),
            label="cav_probe",
        )
        return Seq().play(probe).measure(), ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        i, q = readout_iq(result)
        return fit_lorentzian(result.xpts, np.hypot(i, q))
