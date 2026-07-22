"""T2 Ramsey experiment (slice 3).

Half-pi, swept idle, second half-pi, measure. Run slightly detuned (set the drive off-resonance
or advance the second half-pi's phase) so the decay envelope carries visible fringes; the fit's
oscillation frequency IS the qubit-drive detuning, and ``tau`` is T2*.

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_decaysin_detuned
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.readout import DispersiveIQ


class T2Ramsey(Experiment):
    """Ramsey fringe measurement: ``ge_hpi · wait(t) · ge_hpi(phase) · measure``.

    ``ramsey_freq_mhz`` records the *intended* software detuning for analysis bookkeeping (the
    conventional lab practice is a real drive detuning; a per-point software phase advance needs
    a swept-phase pulse, which lands with the swept-phase compiler support). The second half-pi
    carries a static ``advance_phase_deg`` (useful for echo-style phase flips).
    """

    def __init__(
        self,
        device,
        *,
        delays: Sweep | None = None,
        advance_phase_deg: float = 0.0,
        ramsey_freq_mhz: float = 0.0,
        readout: DispersiveIQ | None = None,
        runner=None,
    ):
        super().__init__(device, runner=runner)
        self.delays = delays if delays is not None else Sweep("t", 0.0, 100.0, 101)
        self.advance_phase_deg = advance_phase_deg
        self.ramsey_freq_mhz = ramsey_freq_mhz
        self.readout = readout if readout is not None else DispersiveIQ(threshold=0.0, angle=0.0)

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        seq = (
            Seq()
            .play(q.ge_hpi())
            .wait(self.delays)
            .play(q.ge_hpi(phase=self.advance_phase_deg))
            .measure()
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        iq = np.stack([np.asarray(result.avgi), np.asarray(result.avgq)], axis=-1)
        return fit_decaysin_detuned(result.xpts, self.readout.project(iq))
