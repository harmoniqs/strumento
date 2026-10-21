"""Sideband spectroscopy experiment.

Find the f0-g1 sideband transition frequency by sweeping the sideband drive near
f_cavity - f_ef. The analyzer fits a Lorentzian to locate the peak.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.resonator_fitters import fit_lorentzian
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ


class SidebandSpectroscopy(Experiment):
    """Sweep sideband drive frequency near f_cavity - f_ef to locate the f0-g1 transition.

    Sequence: ge pi, ef pi, sideband probe at swept frequency, readout. The
    qubit is shelved in |f> so the sideband drive addresses the |f,0> to |g,1>
    transition directly.
    Fitter: ``fit_lorentzian`` -- Lorentzian peak/dip fit of the dispersive signal.

    **Probe strength.** By default the probe reuses the calibrated f0-g1 ``pi`` pulse (its full
    gain and length) -- fine for locating the line, but a strong drive both power-broadens the
    resonance and AC-Stark-shifts it, so the fitted ``f0`` sits slightly off the low-power value.
    For a clean, transform-limited line set a weak ``probe_gain`` and a long ``probe_length_us``
    (a weak, ~1-2 us tone). These route through ``SidebandOps.f0g1`` so the
    ``MAX_SIDEBAND_GAIN`` safety and the calibrated ramp still apply; a low-gain long flat-top is a
    ``const`` probe in all but the 5 ns edges. Both ``None`` (the default) reproduces the calibrated
    pi exactly, so the demo-fixture contract path is unchanged.

    Derived quantities:
        f0       : sideband transition frequency (MHz).
        hwhm     : half-width at half-maximum (MHz).
        fwhm     : full-width at half-maximum (MHz).
        Q_loaded : loaded quality factor f0 / fwhm.

    Parameters:
        freqs           : frequency sweep (``Sweep``); defaults to 0--1 on axis ``"freq"``.
        points          : number of sweep points.
        probe_gain      : probe amplitude as a v2 gain **fraction** (D23); ``None`` uses the
                          calibrated f0-g1 pi gain. Set weak (e.g. 0.03) for spectroscopy.
        probe_length_us : flat probe length (us); ``None`` uses the calibrated pi length. Set long
                          (e.g. 2.0) to narrow the transform-limited linewidth. Only valid on a
                          flat-top-calibrated ladder (the f0-g1 pulses here are flat_top).

    Usage::

        exp = SidebandSpectroscopy(dev, freqs=Sweep(1994, 2004, on="freq"),
                                   probe_gain=0.03, probe_length_us=2.0)
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["f0"]
    """

    def __init__(self, device, *, freqs: Sweep | None = None, points: int = 201,
                 probe_gain: float | None = None, probe_length_us: float | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.freqs = freqs if freqs is not None else Sweep(start=0.0, stop=1.0, on="freq")
        self.points = points
        self.probe_gain = probe_gain
        self.probe_length_us = probe_length_us
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.freqs, Sweep) and self.freqs.on is not None:
            return (Loop(self.freqs.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        # Weak, long probe for spectroscopy: overriding gain (a v2 fraction, D23) and the flat
        # length routes through the SidebandOps.f0g1 factory, so the MAX_SIDEBAND_GAIN safety and
        # the calibrated ramp still apply -- a low-gain long flat-top is a `const` probe in all
        # but the 5 ns edges. Both None -> the calibrated pi shape (default), so the demo-fixture
        # contract path is byte-identical to before this knob existed.
        probe = dev.sideband.multiphoton(
            "f0-g1", "pi", photon=0, freq=self.freqs,
            gain=self.probe_gain, length=self.probe_length_us,
        )
        # `probe` above drives the sideband generator on the QUBIT line, never the resonator
        # (#103) -- add the real readout probe alongside it rather than leaving .measure() bare.
        seq = (
            Seq().play(dev.qubit.ge_pi()).play(dev.qubit.ef_pi()).play(probe)
            + dev.readout.dispersive_measure()
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)
        return fit_lorentzian(result.xpts, signal)
