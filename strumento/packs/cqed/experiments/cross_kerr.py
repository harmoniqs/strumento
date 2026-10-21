"""Cross-Kerr measurement experiment.

Measure the cross-Kerr interaction (chi) between the qubit and a cavity mode by
performing a Ramsey experiment on the qubit with the cavity displaced into a
coherent state. The frequency shift yields chi.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_decaysin_detuned
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class CrossKerrMeasurement(Experiment):
    """Measure the cross-Kerr interaction chi between the qubit and a cavity mode.

    Sequence: ``displace(alpha)`` on the cavity -- ``ge_hpi`` -- wait(t) --
    ``ge_hpi`` -- measure, with the Ramsey delay ``t`` swept.  The cavity
    photon population shifts the qubit frequency by chi per photon; the
    Ramsey fringe frequency with the cavity displaced gives chi directly.
    Fitter: ``fit_decaysin_detuned`` (damped cosine), with ``chi`` aliased
    from the fitted oscillation frequency.

    Derived quantities:
        chi      : cross-Kerr shift (cycles per x-unit, i.e. MHz for delay in us)
        detuning : oscillation frequency from the damped-cosine fit (same as chi)
        freq     : same as detuning
        T2       : Ramsey decay time during the measurement (us)
        tau      : same as T2

    Parameters:
        delays    : Ramsey delay sweep; defaults to 0-2000 us on axis ``"t"``
        points    : number of sweep points (default 101)
        mode_name : cavity mode to displace (default: the device's cavity)
        readout   : ``DispersiveIQ`` readout config; defaults to the recorded calibration
                    (``DispersiveIQ.from_calib``)

    Usage::

        exp = CrossKerrMeasurement(dev, mode_name="cavity")
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["chi"]
    """

    def __init__(self, device, *, delays: Sweep | None = None, points: int = 101,
                 mode_name: str | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        mode_name = resolve_mode(device, mode_name, purpose=type(self).__name__)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=2000.0, on="t")
        validate_sweep_range(self.delays, min_val=MIN_TIME_SWEEP,
                             max_val=MAX_TIME_SWEEP, label="delay")
        self.points = points
        self.mode_name = mode_name
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        mode = getattr(dev, self.mode_name)
        q = dev.qubit
        seq = (
            Seq()
            .play(mode.displace())
            .play(q.ge_hpi())
            .wait(self.delays)
            .play(q.ge_hpi())
            .play(dev.readout.dispersive_measure())
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        fit = fit_decaysin_detuned(result.xpts, readout_signal(result, self.readout))
        fit.derived["chi"] = fit.derived.get("freq", 0.0)
        return fit
