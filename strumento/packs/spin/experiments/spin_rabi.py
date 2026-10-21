"""Spin Rabi oscillation — pi-pulse calibration (Layer 2).

Sweeps the microwave burst duration at the resonance frequency and measures the spin-flip
probability. The resulting oscillation gives the Rabi frequency and calibrates the pi and
pi/2 pulse durations.

Applicable to: LD (EDSR), Ge hole (AC gate). Routine recalibration (multiple times/day).

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_rabi
from strumento.core.pulses import Const, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.spin.experiments._signal import readout_signal
from strumento.packs.spin.readout import PSBReadout
from strumento.packs.spin.safety import validate_gain


class SpinRabi(Experiment):
    """Drive-duration Rabi oscillation for pi-pulse calibration.

    Swept: microwave burst duration t_burst (µs) at fixed frequency and power.
    Measured: P_up(t_burst) — oscillates as cos(2π·Ω_R·t).
    Extracts: Rabi period, pi_time (= period/2), half_pi_time (= period/4).

    The burst duration is swept via the declared loop axis ``"t"`` — the sweep travels
    inside the pulse envelope (a ``Const`` whose ``length_us`` is the sweep). The dot's
    calibrated frequency and gain are used for the drive.

    ``dot_name`` selects which dot on the device provides the drive parameters.
    """

    def __init__(self, device, *, dot_name: str = "q1", durations: Sweep | None = None,
                 points: int = 51, readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self._dot_name = dot_name
        self.durations = durations if durations is not None else Sweep(start=0.006, stop=0.200, on="t")
        self.points = points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.durations, Sweep) and self.durations.on is not None:
            return (Loop(self.durations.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # Build a drive burst with swept duration using the dot's calibrated frequency/gain.
        # The pi pulse's parameters give us the correct frequency and gain; we replace the
        # envelope with a Const of swept length.
        ref_pulse = self._dot.pi()
        validate_gain(ref_pulse.gain, label=f"{self._dot_name}_rabi_burst")
        burst = Pulse(
            line=LineRef(self._dot_name, "drive"),
            freq_mhz=ref_pulse.freq_mhz,
            gain=ref_pulse.gain,
            phase_deg=0.0,
            envelope=Const(length_us=self.durations),
        )
        seq = Seq().play(burst).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_rabi(result.xpts, readout_signal(result, self.readout))
