"""Number splitting spectroscopy — resolve photon-number-dependent qubit frequency (cqed expansion).

Displaces the cavity to populate multiple Fock states, then sweeps a qubit spectroscopy tone.
Each photon number n shifts the qubit frequency by n*chi, producing a comb of Lorentzian peaks
separated by chi. This directly measures the dispersive interaction strength and verifies the
cavity photon distribution.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_number_splitting_comb
from strumento.core.pulses import Const, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._common import drive_gain_limit
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import validate_gain


class NumberSplitting(Experiment):
    """Photon-number-resolved qubit spectroscopy after a coherent cavity displacement.

    ``displacement_alpha`` sets the coherent state amplitude (default 2.0, populating ~4 Fock
    states). The frequency sweep spans ~10*chi around f_ge to resolve individual number peaks.
    A Poisson-weighted Lorentzian **comb** fit (``fit_number_splitting_comb``) extracts chi from
    the peak **spacing** (not one peak's offset) plus the mean photon number. ``mode_name`` selects
    which cavity mode to displace; by default the device's cavity, the one ``BosonicMode`` with
    ``role: cavity`` (see :mod:`strumento.packs.cqed.library.targets`).
    """

    def __init__(self, device, *, freqs: Sweep | None = None, points: int = 401,
                 displacement_alpha: float = 2.0, probe_gain: float = 0.03,
                 mode_name: str | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        mode_name = resolve_mode(device, mode_name, purpose=type(self).__name__)
        f_ge = device.calib.qubit.f_ge
        chi = getattr(device.calib.qubit, "chi", 1.0)
        span = 10.0 * abs(chi)
        self.freqs = freqs if freqs is not None else Sweep(start=f_ge - span / 2, stop=f_ge + span / 2, on="freq")
        self.points = points
        self.displacement_alpha = displacement_alpha
        self.probe_gain = probe_gain
        self.mode_name = mode_name
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.freqs, Sweep) and self.freqs.on is not None:
            return (Loop(self.freqs.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        mode_ops = getattr(self.device, self.mode_name)
        disp = mode_ops.displace()

        qubit_name = self.device.qubit.element.name
        line = LineRef(qubit_name, "drive")
        validate_gain(self.probe_gain, limit=drive_gain_limit(self.device), role="qubit_drive", label="ns_probe")
        probe = Pulse(
            line=line,
            freq_mhz=self.freqs,
            gain=self.probe_gain,
            envelope=Const(length_us=10.0),
            label="ns_probe",
        )
        # `probe` above drives the QUBIT line, never the resonator (#103) -- add the real
        # readout probe alongside it rather than leaving .measure() bare.
        seq = Seq().play(disp).play(probe) + self.device.readout.dispersive_measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        # The spectrum is a Poisson-weighted comb of peaks at f_ge - n*chi; chi is the peak
        # SPACING, which a single Lorentzian cannot give. Fit the whole comb, seeding chi from the
        # mode's believed dispersive shift (falls back to the qubit belief, then a span-scaled seed).
        signal = readout_signal(result, self.readout)
        f_ge = float(self.device.calib.qubit.f_ge)
        mode_cal = self.device.calib.modes.get(self.mode_name)
        chi_seed = abs(float(mode_cal.chi)) if (mode_cal is not None and getattr(mode_cal, "chi", None)) else 0.0
        if not chi_seed:
            qchi = getattr(self.device.calib.qubit, "chi", None)
            chi_seed = abs(float(qchi)) if qchi else 1.0
        return fit_number_splitting_comb(result.xpts, signal, f_ge, chi_seed=chi_seed)
