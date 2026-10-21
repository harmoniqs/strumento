"""Hahn echo — T2 coherence time measurement (Layer 2).

Refocusing sequence: hpi → wait(tau) → pi → wait(tau) → hpi → measure.
The pi pulse refocuses quasistatic noise, extending coherence beyond T2*.
The decay of the echo signal gives T2_echo.

Applicable to: LD, Ge hole, S-T (using exchange pi pulse). Characterization experiment.

**Encoding-dependent semantics:**

- **LD / Ge hole**: the wait periods are free Larmor precession; the π pulse refocuses
  quasistatic Zeeman noise (B-field fluctuations). T2_echo reflects the high-frequency
  noise the echo cannot refocus.

- **S-T0**: the wait periods are free precession under ΔB_z (the inter-dot gradient);
  the π pulse (an exchange pulse) refocuses quasistatic charge noise that shifts J.
  T2_echo here measures resilience to charge noise, not magnetic noise — a physically
  different quantity from LD despite the identical sequence structure.

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


class HahnEcho(Experiment):
    """Hahn echo: hpi → tau → pi → tau → hpi → measure.

    Swept: half-evolution time tau (µs); total evolution is 2·tau.
    Measured: P_up(2·tau) — decays without oscillation (if on resonance).
    Extracts: T2_echo from exponential decay fit.

    The sweep is over tau (half the total evolution time), riding the declared
    loop axis ``"tau"``. ``dot_name`` selects the qubit dot.
    """

    def __init__(self, device, *, dot_name: str = "q1", delays: Sweep | None = None,
                 points: int = 101, readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=50.0, on="tau")
        self.points = points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        seq = (
            Seq()
            .play(self._dot.hpi())
            .wait(self.delays)
            .play(self._dot.pi())
            .wait(self.delays)
            .play(self._dot.hpi())
            .measure()
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_exp(result.xpts, readout_signal(result, self.readout))
