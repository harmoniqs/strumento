"""E-F T2 echo — Hahn echo on the e-f transition, after a ge-pi shelve (cqed expansion).

Shelves the qubit in |e> with a ge pi pulse (the pack's ge-shelving idiom -- see ``EFRabi``/
``EFSpectroscopy``/``EFT1``/``EFT2Ramsey``), then runs the same half-pi -- wait -- pi -- wait --
half-pi Hahn echo the ge ``T2Echo`` experiment uses, but on the e-f transition, with a closing
ge pi for readout discrimination: ge_pi, ef_hpi, wait(t), ef_pi (refocus), wait(t), ef_hpi,
ge_pi, measure. The refocusing ef pi cancels quasi-static dephasing, isolating the intrinsic ef
T2 from slow frequency fluctuations. The decay is fit to a simple exponential over total
evolution time 2*tau, exactly as the ge echo -- only the state preparation and readout mapping
differ.

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


class EFT2Echo(Experiment):
    """Measure the e-f transition's coherence time T2 using a Hahn spin-echo sequence.

    Sequence: ``ge_pi`` (shelve into |e>) -- ``ef_hpi`` -- wait(t) -- ``ef_pi`` (refocusing) --
    wait(t) -- ``ef_hpi`` -- ``ge_pi`` (map |e> -> |g> for readout discrimination) -- measure,
    with the total echo delay ``2t`` swept. Matches the pack's ge-shelving idiom and its EF
    readout mapping (the closing ge pi distinguishes |e> from |f>); the refocusing
    ef pi cancels quasi-static dephasing, isolating the intrinsic ef T2 from slow frequency
    fluctuations. Fitter: ``fit_exp`` (exponential decay), reused verbatim from the ge ``T2Echo``
    experiment, with ``T2_echo`` derived as twice the fitted half-delay decay constant -- only the
    state preparation and readout mapping differ.

    Derived quantities:
        T2_echo : e-f echo coherence time (us), the headline quantity
        tau     : raw half-delay decay constant

    Parameters:
        delays  : nonnegative half-delay sweep; total evolution is twice this axis
        points  : number of sweep points (default 101)
        readout : ``DispersiveIQ`` readout config; defaults to the recorded calibration
                    (``DispersiveIQ.from_calib``)
        line    : wiring line reference for the readout probe tone (default ``("readout", "probe")``)

    Usage::

        exp = EFT2Echo(dev)
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["T2_echo"]
    """

    def __init__(self, device, *, delays: Sweep | None = None, points: int = 101,
                 readout: DispersiveIQ | None = None, line=("readout", "probe"), runner=None):
        super().__init__(device, runner=runner)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=100.0, on="t")
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
        seq = (
            Seq()
            .play(q.ge_pi())
            .play(q.ef_hpi())
            .wait(self.delays)
            .play(q.ef_pi())
            .wait(self.delays)
            .play(q.ef_hpi())
            .play(q.ge_pi())
            .play(self.device.readout.dispersive_measure(line=self.line))
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        fit = fit_exp(result.xpts, readout_signal(result, self.readout))
        # ``fit_exp`` labels every exponential constant T1; echo's raw axis is a half-delay.
        fit.derived.pop("T1", None)
        fit.derived["T2_echo"] = 2.0 * fit.derived["tau"]
        return fit
