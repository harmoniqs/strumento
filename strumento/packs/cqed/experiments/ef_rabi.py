"""E-F Rabi — calibrate the e-f pi pulse amplitude (cqed expansion).

After exciting the qubit to |e> with a calibrated ge pi pulse, sweeps the ef drive amplitude to
observe Rabi oscillations on the e-f transition. A sinusoidal fit extracts the ef pi-pulse gain,
needed for leakage studies and higher-level state preparation.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_rabi
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ


class EFRabi(Experiment):
    """Drive-amplitude Rabi on the e-f transition after ge pi excitation.

    Sequence: ge pi (prepare |e>), ef pi pulse at swept gain, a second ge pi (map |e> -> |g> so the
    readout distinguishes |e> from |f>), readout. The oscillation between |e> and
    |f> yields the ef pi-pulse gain needed for leakage studies and higher-level state preparation.
    Fitter: ``fit_rabi`` -- decaying-cosine fit of the Rabi oscillation.

    Derived quantities:
        period  : oscillation period in gain units (DAC code).
        pi_gain : half-period value -- the ef pi-pulse amplitude.

    Parameters:
        gains  : amplitude sweep (``Sweep``); defaults to 0-16000 on axis ``"amp"`` (within
                 MAX_QUBIT_DRIVE_GAIN=0.5; was 0-30000, ~0.916 frac).
        points : number of sweep points.

    Usage::

        exp = EFRabi(dev, gains=Sweep(0, 20000, on="amp"))
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["pi_gain"]
    """

    def __init__(self, device, *, gains: Sweep | None = None, points: int = 51,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        # See amplitude_rabi.py's identical note: default lowered to stay within
        # MAX_QUBIT_DRIVE_GAIN now that sweep endpoints are actually validated.
        self.gains = gains if gains is not None else Sweep(start=0, stop=16000, on="amp")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.gains, Sweep) and self.gains.on is not None:
            return (Loop(self.gains.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        # A second ge pi maps |e> -> |g> before readout so the ge discriminator distinguishes |e>
        # from |f>. Without it the |f> population reads out on a muddied ge axis.
        seq = (
            Seq()
            .play(q.ge_pi())
            .play(q.ef_pi(gain=self.gains))
            .play(q.ge_pi())
            .play(self.device.readout.dispersive_measure())
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_rabi(result.xpts, readout_signal(result, self.readout))
