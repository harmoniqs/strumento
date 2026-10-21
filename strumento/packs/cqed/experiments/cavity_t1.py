"""Cavity T1 experiment.

Measure the photon lifetime of a cavity mode by displacing into a coherent state,
waiting a swept delay, and measuring the remaining photon population via parity.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_exp
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.library.composite import parity
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class CavityT1(Experiment):
    """Measure the photon lifetime of a cavity mode via displacement and parity readout.

    Sequence: ``displace(alpha)`` -- wait(t) -- parity mapping -- measure, with
    the idle delay ``t`` swept.  The cavity is displaced into a coherent state;
    photon loss during the wait reduces the mean photon number as
    ``n̄(t) = n̄₀·exp(-t/T1_cavity)``.
    Fitter: ``fit_exp`` (exponential decay ``y0 + amp * exp(-x/tau)``).

    Bias caveat: the *parity* of a decaying coherent state is ``⟨Π(t)⟩ = exp(-2·n̄(t))`` — a
    NESTED exponential, not a single one — so a plain ``fit_exp`` recovers T1 without bias only in
    the small-displacement limit ``n̄₀ ≪ 1`` (where ``⟨Π⟩ ≈ 1 - 2·n̄₀·exp(-t/T1)`` is a clean
    exponential). At the calibrated ``|alpha| ≈ 1`` the single-exponential T1 is biased; keep the
    displacement small, or measure buffer T1 the unbiased way — prepare
    Fock ``|1⟩`` via the ``|f0⟩↔|g1⟩`` sideband and watch ``P(1)`` decay (see StorageT1 for the
    sideband-mapped variant).

    Derived quantities:
        T1  : cavity photon lifetime in the same units as the delay sweep (typically us)
        tau : decay constant (same value as T1)

    Parameters:
        delays    : delay sweep; defaults to 0-5000 us on axis ``"t"``
        points    : number of sweep points (default 101)
        mode_name : device mode to displace and read parity on (default: the device's cavity)
        readout   : ``DispersiveIQ`` readout config; defaults to the recorded calibration
                    (``DispersiveIQ.from_calib``)

    Usage::

        exp = CavityT1(dev, mode_name="cavity")
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["T1"]
    """

    def __init__(self, device, *, delays: Sweep | None = None, points: int = 101,
                 mode_name: str | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        mode_name = resolve_mode(device, mode_name, purpose=type(self).__name__)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=5000.0, on="t")
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
        # #100: parity() drives the readout and measures internally by default now -- chaining
        # our own .measure() here would double-measure.
        seq = (
            Seq()
            .play(mode.displace())
            .wait(self.delays)
            + parity(dev, self.mode_name)
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_exp(result.xpts, readout_signal(result, self.readout))
