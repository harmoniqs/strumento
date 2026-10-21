"""E-F spectroscopy — locate the transmon e-f transition frequency (cqed expansion).

First excites the qubit to |e> with a calibrated ge pi pulse, then sweeps a second tone to find
the e-f transition. The analyzer fits a Lorentzian to extract f_ef, which is needed for leakage
characterization and ef-pulse calibration.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Const, Pulse, Seq
from strumento.core.resonator_fitters import fit_lorentzian
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._common import drive_gain_limit
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import validate_gain


class EFSpectroscopy(Experiment):
    """Two-tone spectroscopy of the e-f transition after a ge pi excitation.

    Sequence: ge pi (prepare |e>), long constant probe tone at swept frequency, a second ge pi
    (map |e> -> |g> so the readout distinguishes |e> from |f>), readout. The probe
    drives the |e> to |f> transition; a Lorentzian fit locates f_ef precisely.
    Fitter: ``fit_lorentzian`` -- Lorentzian peak/dip fit of the dispersive signal.

    Derived quantities:
        f0       : e-f transition frequency (MHz).
        hwhm     : half-width at half-maximum (MHz).
        fwhm     : full-width at half-maximum (MHz).
        Q_loaded : loaded quality factor f0 / fwhm.

    Parameters:
        freqs      : frequency sweep (``Sweep``); defaults to f_ef_est +/- 25 MHz on axis ``"freq"``.
        points     : number of sweep points.
        probe_gain : probe tone amplitude (fraction of full scale, default 0.03).

    Usage::

        exp = EFSpectroscopy(dev, probe_gain=0.05)
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["f0"]
    """

    def __init__(self, device, *, freqs: Sweep | None = None, points: int = 201,
                 probe_gain: float = 0.03,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        f_ge = device.calib.qubit.f_ge
        f_ef_est = getattr(device.calib.qubit, "f_ef", f_ge - 200.0)
        self.freqs = freqs if freqs is not None else Sweep(start=f_ef_est - 25.0, stop=f_ef_est + 25.0, on="freq")
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
        validate_gain(self.probe_gain, limit=drive_gain_limit(self.device), role="qubit_drive", label="ef_probe")
        probe = Pulse(
            line=line,
            freq_mhz=self.freqs,
            gain=self.probe_gain,
            envelope=Const(length_us=10.0),
            label="ef_probe",
        )
        # The probe above drives the QUBIT line (the e-f transition), never the resonator
        # (#103) -- add the real readout probe alongside it rather than leaving .measure() bare.
        # A second ge pi maps |e> -> |g> before readout so the ge discriminator distinguishes |e>
        # (off-resonance probe, no e-f drive) from |f> (on-resonance).
        seq = (
            Seq().play(self.device.qubit.ge_pi()).play(probe).play(self.device.qubit.ge_pi())
            + self.device.readout.dispersive_measure()
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)
        return fit_lorentzian(result.xpts, signal)

    def recovery_map(self) -> dict[str, str]:
        """Pair the Lorentzian center with the simulator's E-F transition truth."""
        return {"f0": "f_ef"}
