"""Displacement calibration experiment.

Map displacement pulse gain to coherent state amplitude alpha by sweeping the
displacement gain and fitting parity oscillations.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_gaussian_wigner
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.library.composite import parity
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ


class DisplacementCalibration(Experiment):
    """Calibrate the displacement pulse by mapping DAC gain to coherent-state amplitude alpha.

    Sequence: ``displace(gain)`` -- parity mapping -- measure, with the
    displacement gain swept.  The parity of the displaced *vacuum* is the vacuum
    Wigner function ``⟨Π⟩ = exp(-2|alpha|^2)`` with ``alpha = alpha_per_gain * gain``
    linear in the DAC gain, so the measured parity is a **Gaussian decay in gain**
    (NOT an oscillation); its width gives the gain-to-alpha conversion.
    Fitter: ``fit_gaussian_wigner`` (``offset + amp * exp(-2*(k*gain)^2)``), reporting
    ``alpha_per_gain`` = ``k`` directly.

    Derived quantities:
        alpha_per_gain : coherent-state amplitude per unit of DAC gain (the calibration factor)
        contrast       : peak-to-peak parity fringe amplitude
        phase_deg      : fringe phase offset in degrees

    Parameters:
        gains     : gain sweep; defaults to 0-13000 on axis ``"gain"`` (within
                    MAX_DISPLACEMENT_GAIN=0.4; was 0-30000, ~0.916 frac, which
                    mode.py's sweep-endpoint validation now rejects)
        points    : number of sweep points (default 101)
        mode_name : device mode to displace (default: the device's cavity)
        readout   : ``DispersiveIQ`` readout config; defaults to the recorded calibration
                    (``DispersiveIQ.from_calib``)

    Usage::

        exp = DisplacementCalibration(dev, mode_name="cavity")
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["alpha_per_gain"]
    """

    def __init__(self, device, *, gains: Sweep | None = None, points: int = 101,
                 mode_name: str | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        mode_name = resolve_mode(device, mode_name, purpose=type(self).__name__)
        self.gains = gains if gains is not None else Sweep(start=0, stop=13000, on="gain")
        self.points = points
        self.mode_name = mode_name
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.gains, Sweep) and self.gains.on is not None:
            return (Loop(self.gains.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        mode = getattr(dev, self.mode_name)
        displace_pulse = mode.displace(gain=self.gains)
        # #100: parity() drives the readout and measures internally by default now -- chaining
        # our own .measure() here would double-measure.
        seq = Seq().play(displace_pulse) + parity(dev, self.mode_name)
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)
        # Parity of a displaced vacuum is the vacuum Wigner Gaussian exp(-2|alpha|^2); with
        # alpha = alpha_per_gain * gain, that is a Gaussian in gain whose width IS the ruler.
        return fit_gaussian_wigner(result.xpts, signal)
