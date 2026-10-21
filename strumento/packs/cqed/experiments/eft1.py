"""E-F T1 — energy-relaxation time on the e-f transition, after a ge-pi shelve (cqed expansion).

Shelves the qubit in |e> with a ge pi pulse (the pack's ge-shelving idiom -- see ``EFRabi``/
``EFSpectroscopy``), then excites to |f> with an ef pi pulse and idles for a swept delay before
readout. The |f> population relaxes as ``exp(-t/T1_ef)``; the fit model is identical to the ge
``T1`` experiment's -- only the state preparation differs, so its fitter is reused verbatim.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_exp
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class EFT1(Experiment):
    """Measure the e-f transition's energy-relaxation time via excited-state decay.

    Sequence: ``ge_pi`` (shelve into |e>) -- ``ef_pi`` (excite to |f>) -- wait(t) -- ``ge_pi``
    (map |e> -> |g> for readout discrimination) -- measure, with the idle delay ``t`` swept.
    Matches the pack's ge-shelving idiom: a ge pi pulse first, then act on the ef transition, and
    a closing ge pi so the ge discriminator distinguishes |e> from |f>.
    Fitter: ``fit_exp`` (exponential decay ``y0 + amp * exp(-x/tau)``), reused verbatim from the
    ge ``T1`` experiment -- the fit model is identical, only the state preparation differs.

    Derived quantities:
        T1  : e-f relaxation time in the same units as the delay sweep (typically us)
        tau : decay constant (same value as T1)

    Parameters:
        delays  : delay sweep; defaults to 0-500 us on axis ``"t"``
        points  : number of sweep points (default 101)
        readout : ``DispersiveIQ`` readout config; defaults to the recorded calibration
                    (``DispersiveIQ.from_calib``)

    Usage::

        exp = EFT1(dev)
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["T1"]
    """

    def __init__(self, device, *, delays: Sweep | None = None, points: int = 101,
                 readout: DispersiveIQ | None = None, line=("readout", "probe"), runner=None):
        super().__init__(device, runner=runner)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=500.0, on="t")
        validate_sweep_range(self.delays, min_val=MIN_TIME_SWEEP,
                             max_val=MAX_TIME_SWEEP, label="delay")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)
        self.line = LineRef(*line) if not isinstance(line, LineRef) else line

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        # A closing ge pi maps the surviving |e> population back to |g> so the ge discriminator
        # reads the |f> population as the bright state. The fitted
        # decay constant is unchanged; the mapping only fixes which state reads out bright.
        seq = (
            Seq()
            .play(q.ge_pi())
            .play(q.ef_pi())
            .wait(self.delays)
            .play(q.ge_pi())
            .play(self.device.readout.dispersive_measure(line=self.line))
        )
        return seq, ()  # the sweep travels in the Delay; the compiler discovers it

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_exp(result.xpts, readout_signal(result, self.readout))
