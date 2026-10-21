"""Leakage benchmark — leakage to non-computational states (Layer 3).

Quantifies the leakage rate from the qubit subspace (computational states) to
non-computational states (e.g. S_z = 3/2 quadruplet in exchange-only, or higher
orbital states). Leakage occurs when pulses are too fast (non-adiabatic) relative
to the energy gap to the leaked state.

The implementation builds one program with ``points`` gate-repetition blocks (one per
gate count N), each followed by a readout (multi-read, no loop axis). Each block applies
N identity-equivalent gate sequences (X·X = I) and measures the survival probability in
the computational subspace. Population loss beyond depolarization indicates leakage.

Applicable to: exchange-only (triple dot), LD (orbital leakage), Ge (heavy/light hole).

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_exp
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.spin.readout import PSBReadout


class LeakageBenchmark(Experiment):
    """Measure leakage rate per gate to non-computational states.

    Builds ``points`` blocks of increasing gate count in a single program (multi-read,
    no loop axis). Each block applies N identity-equivalent sequences (pairs of π pulses:
    X·X = I) to stress the qubit without intended rotation — leakage appears as
    non-recoverable population loss exceeding the depolarization baseline.

    The analyzer fits the survival probability decay and extracts the leakage rate L
    per gate from the exponential envelope.

    Prerequisites: basic gates working (Rabi calibrated).
    """

    def __init__(self, device, *, dot_name: str = "q1",
                 gate_count_start: int = 2, gate_count_stop: int = 100,
                 points: int = 10, readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self._dot_name = dot_name
        self.gate_count_start = gate_count_start
        self.gate_count_stop = gate_count_stop
        self.points = points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)
        # Pre-compute the gate-count schedule (even numbers for X·X pairs)
        self._gate_counts = np.linspace(
            gate_count_start, gate_count_stop, points, dtype=int
        )
        # Ensure all counts are even (for complete X·X identity pairs)
        self._gate_counts = (self._gate_counts // 2) * 2
        self._gate_counts[self._gate_counts < 2] = 2

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        seq = Seq()
        for n_gates in self._gate_counts:
            # Apply n_gates/2 identity-equivalent pairs (X·X = I)
            n_pairs = int(n_gates) // 2
            for _ in range(n_pairs):
                seq.play(self._dot.pi())
                seq.play(self._dot.pi())
            seq.measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        measured = np.asarray(result.avgi, dtype=float).ravel()

        # Use result.xpts when provided (synthetic/unit-test path), else internal schedule
        if result.xpts is not None and len(np.asarray(result.xpts).ravel()) > 0:
            xpts = np.asarray(result.xpts, dtype=float).ravel()
        else:
            xpts = self._gate_counts.astype(float)

        # Truncate if shapes don't match (defensive)
        n = min(len(measured), len(xpts))

        # Fit survival decay: P_survive(N) ~ exp(-L*N)
        fit = fit_exp(xpts[:n], measured[:n])

        tau = fit.derived.get("T1", 1.0)
        # Leakage per gate = 1/tau (the decay constant in gates)
        leakage_per_gate = 1.0 / tau if tau > 0 else 0.0
        leakage_per_gate = max(0.0, leakage_per_gate)

        fit.derived["leakage_per_gate"] = leakage_per_gate
        return fit
