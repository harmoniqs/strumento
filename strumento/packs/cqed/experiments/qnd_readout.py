"""QND readout verification experiment (cqed expansion).

Verify the quantum non-demolition property of dispersive readout by performing
repeated measurements on the same shot. A perfect QND readout gives the same result
on every repetition; measurement-induced transitions (T1 during readout, photon-number
jumps) cause the outcome to flip between reads.

The implementation uses the multi-read pattern: prepare |e⟩ once, then measure
``n_repeats`` times in sequence (no loop axis). The analyzer computes the QND fidelity
from the consistency of the repeated readout signals: how many consecutive reads agree.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.cqed.readout import DispersiveIQ


class QNDReadout(Experiment):
    """Repeated measurement to verify QND property of dispersive readout.

    Builds a multi-read program: one ge_pi preparation followed by ``n_repeats`` consecutive
    Measure ops. If the readout is QND, all reads return the same result (|e⟩). The analyzer
    reports the fraction of reads that agree with the first read (QND fidelity) and the
    average transition probability between consecutive reads.
    """

    def __init__(self, device, *, n_repeats: int = 5,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.n_repeats = n_repeats
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        seq = Seq().play(self.device.qubit.ge_pi())
        for _ in range(self.n_repeats):
            seq.play(self.device.readout.dispersive_measure())
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        # Each of the n_repeats reads gives one averaged signal
        avgi = np.asarray(result.avgi, dtype=float).ravel()
        avgq = np.asarray(result.avgq, dtype=float).ravel()
        iq = np.stack([avgi, avgq], axis=-1)
        signal = self.readout.project(iq).ravel()

        n = len(signal)
        if n < 2:
            return FitResult(
                params={}, stderr={},
                derived={"qnd_fidelity": float(signal[0]) if n == 1 else 0.0},
                success=True,
            )

        # QND fidelity: fraction of reads that agree with the first read
        # (in the projected basis, excited state is signal > 0.5)
        first_outcome = signal[0] > 0.5
        agreements = np.sum((signal > 0.5) == first_outcome)
        qnd_fidelity = float(agreements) / n

        # Transition probability: fraction of consecutive read pairs that disagree
        outcomes = signal > 0.5
        transitions = np.sum(outcomes[1:] != outcomes[:-1])
        transition_prob = float(transitions) / (n - 1)

        return FitResult(
            params={},
            stderr={},
            derived={
                "qnd_fidelity": qnd_fidelity,
                "transition_prob": transition_prob,
                "n_repeats": n,
            },
            success=True,
        )
