"""Length Rabi — sweep pulse duration at fixed gain to find pi time (cqed expansion).

Drives the qubit at a fixed amplitude while sweeping the pulse length. The resulting Rabi
oscillation in the time domain lets you extract the pi-pulse duration independently of
amplitude-Rabi calibration — useful as a cross-check or when the amplitude DAC is near
saturation.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_damped_rabi
from strumento.core.pulses import Const, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import gain_int_to_frac
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.library._gain import maxv_for
from strumento.packs.cqed.readout import DispersiveIQ


class LengthRabi(Experiment):
    """Drive-duration Rabi oscillation on the ge transition at fixed amplitude.

    Sequence: constant-envelope pulse on the qubit drive at the calibrated ge pi
    gain, with swept pulse length, then readout. The time-domain Rabi oscillation
    gives the pi time independently of amplitude calibration.
    Fitter: ``fit_damped_rabi`` -- damped sinusoid fit in the time domain.

    Derived quantities:
        rabi_freq    : Rabi frequency (cycles per us).
        period       : oscillation period (us).
        pi_time      : half-period -- the pi-pulse duration (us).
        half_pi_time : quarter-period -- the pi/2-pulse duration (us).
        tau          : exponential decay time of the envelope (us).

    Parameters:
        lengths : duration sweep (``Sweep``); defaults to 0--0.5 us on axis ``"t"``.
        points  : number of sweep points.

    Usage::

        exp = LengthRabi(dev, lengths=Sweep(0.0, 1.0, on="t"))
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["pi_time"]
    """

    def __init__(self, device, *, lengths: Sweep | None = None, points: int = 101,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.lengths = lengths if lengths is not None else Sweep(start=0.0, stop=0.5, on="t")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.lengths, Sweep) and self.lengths.on is not None:
            return (Loop(self.lengths.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        qc = self.device.calib.qubit
        pc = qc.pulses["pi_ge"]
        element_name = self.device.qubit.element.name
        freq_mhz = qc.f_ge
        maxv = maxv_for(self.device, LineRef(element_name, "drive"), freq_mhz)
        gain = gain_int_to_frac(pc.gain, maxv)

        pulse = Pulse(
            line=LineRef(element_name, "drive"),
            freq_mhz=freq_mhz,
            gain=gain,
            envelope=Const(length_us=self.lengths),
            label="length_rabi",
        )
        seq = Seq().play(pulse).play(self.device.readout.dispersive_measure())
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_damped_rabi(result.xpts, readout_signal(result, self.readout))
