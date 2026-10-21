"""Spin T1 — relaxation time measurement (Layer 2: qubit characterization).

Initializes the spin in the excited state with a pi pulse, waits a variable delay, then
reads out. The excited-state population decays exponentially with time constant T1
(spin-lattice relaxation via phonon emission or spin-orbit mediated processes).

Applicable to: LD, Ge hole spin. Typical T1 ~ 1-100 ms in Si at 1 T.

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


class SpinT1(Experiment):
    """Spin relaxation time T1: pi → wait(t) → measure.

    Swept: wait time t between excitation and readout (µs).
    Measured: spin-up probability P_up(t) — decays as exp(-t/T1).
    Extracts: T1 from exponential fit.

    The default delay sweep rides the declared loop axis ``"t"`` (D36).
    ``dot_name`` selects which dot on the device provides the pi pulse (default ``"q1"``).
    """

    def __init__(self, device, *, dot_name: str = "q1", delays: Sweep | None = None,
                 points: int = 101, readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=500.0, on="t")
        self.points = points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        seq = Seq().play(self._dot.pi()).wait(self.delays).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_exp(result.xpts, readout_signal(result, self.readout))
