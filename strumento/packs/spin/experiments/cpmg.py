"""CPMG / dynamical decoupling — extended T2 and noise spectroscopy (Layer 2).

Multiple refocusing pi pulses (N_pi) extend coherence beyond the Hahn echo limit.
The filter function of CPMG peaks at f = N/(2T_total), so varying N maps the noise
spectral density S(f).

Sequence: hpi → [wait(tau) → pi → wait(tau)] × N_pi → hpi → measure.
The total evolution time is 2·N_pi·tau; at each sweep point tau changes, keeping N_pi fixed.

Applicable to: all encodings. Characterization experiment.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_exp
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.spin.experiments._signal import readout_signal
from strumento.packs.spin.readout import PSBReadout


class CPMG(Experiment):
    """CPMG dynamical decoupling: hpi → [tau → pi → tau]^N → hpi → measure.

    Swept: inter-pulse delay tau (µs) at fixed number of pi pulses N_pi.
        Total evolution time = 2·N_pi·tau.
    Measured: P_up decay envelope vs total evolution time.
    Extracts: T2_DD (dynamically-decoupled coherence time).

    ``n_pi`` sets the number of refocusing pulses (default 4). The sweep is over tau,
    riding the declared loop axis ``"tau"``. The pi pulse train uses a hardware repeat
    when ``n_pi > 1``.
    """

    def __init__(self, device, *, dot_name: str = "q1", n_pi: int = 4,
                 delays: Sweep | None = None, points: int = 101,
                 readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self.n_pi = n_pi
        self.delays = delays if delays is not None else Sweep(start=0.001, stop=20.0, on="tau")
        self.points = points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # Build the refocusing unit: wait(tau) → pi → wait(tau)
        refocus_unit = Seq().wait(self.delays).play(self._dot.pi()).wait(self.delays)

        seq = Seq().play(self._dot.hpi())
        # Use Repeat for the N_pi refocusing blocks (hardware loop)
        seq.repeat(refocus_unit, self.n_pi)
        seq.play(self._dot.hpi()).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        fit = fit_exp(result.xpts, readout_signal(result, self.readout))
        # fit_exp reports the decay constant on the tau axis: tau_eff = T2_DD / (2*N_pi).
        # Recover T2_DD by scaling by the total-time factor.
        tau_eff = fit.derived.get("T1", 0.0)
        t2_dd = tau_eff * 2.0 * self.n_pi
        fit.derived["T2_DD"] = t2_dd
        fit.derived["n_pi"] = self.n_pi
        return fit
