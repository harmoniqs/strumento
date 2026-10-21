"""EDSR spectroscopy — Larmor frequency finding (Layer 2).

Sweeps the microwave drive frequency around the expected Larmor frequency f_L = g·µ_B·B/h
and measures the spin-flip probability after a long burst. The resonance appears as a peak
(or dip, depending on readout contrast) in P_up(f).

Applicable to: LD (EDSR via micromagnet or SOC), Ge hole (AC gate drive).

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Const, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.spin.experiments._signal import readout_signal
from strumento.packs.spin.readout import PSBReadout
from strumento.packs.spin.safety import validate_gain


class EDSRSpectroscopy(Experiment):
    """Frequency sweep to find the spin resonance (EDSR or ESR).

    Swept: microwave drive frequency f_MW (MHz) around expected f_L.
    Measured: spin-flip probability P_up(f) after a long drive burst at each frequency.
    Extracts: f_qubit (resonance peak position), linewidth.

    The burst duration ``burst_duration_us`` is fixed (long enough for saturation).
    The frequency sweep rides the declared loop axis ``"f"``.
    ``dot_name`` selects the dot whose drive line is used.
    """

    def __init__(self, device, *, dot_name: str = "q1",
                 freqs: Sweep | None = None, points: int = 101,
                 burst_duration_us: float = 0.500, drive_gain: float | None = None,
                 readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self._dot_name = dot_name
        self.burst_duration_us = burst_duration_us
        self._drive_gain = drive_gain
        # Default sweep: ±5 MHz around the calibrated qubit frequency
        if freqs is not None:
            self.freqs = freqs
        else:
            ref = self._dot.pi()
            center = ref.freq_mhz
            self.freqs = Sweep(start=center - 5.0, stop=center + 5.0, on="f")
        self.points = points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.freqs, Sweep) and self.freqs.on is not None:
            return (Loop(self.freqs.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # A fixed-duration drive burst with swept frequency
        gain = self._drive_gain
        if gain is None:
            gain = self._dot.pi().gain
        validate_gain(gain, label=f"{self._dot_name}_edsr_burst")

        burst = Pulse(
            line=LineRef(self._dot_name, "drive"),
            freq_mhz=self.freqs,  # swept frequency
            gain=gain,
            phase_deg=0.0,
            envelope=Const(length_us=self.burst_duration_us),
        )
        seq = Seq().play(burst).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)
        xpts = np.asarray(result.xpts, dtype=float)
        y = np.asarray(signal, dtype=float)

        # Find the resonance peak (maximum spin-flip probability)
        peak_idx = int(np.argmax(y))
        f_qubit = float(xpts[peak_idx])

        # Estimate linewidth from FWHM
        half_max = (y.max() + y.min()) / 2.0
        above = xpts[y >= half_max]
        linewidth = float(above[-1] - above[0]) if len(above) > 1 else 0.0

        derived = {
            "f_qubit_mhz": f_qubit,
            "linewidth_mhz": linewidth,
        }
        return FitResult(
            params={"f_peak": f_qubit, "linewidth": linewidth},
            stderr={"f_peak": 0.0, "linewidth": 0.0},
            derived=derived,
            success=True,
        )
