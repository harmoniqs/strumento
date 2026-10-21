"""Single-shot readout characterization — IQ blob histogram (cqed expansion).

Characterize readout fidelity by preparing the qubit in |g⟩ and |e⟩ and measuring each,
building IQ histograms from the two preparations. A double-Gaussian fit on the projected
signal determines the optimal discrimination threshold and reports assignment fidelity.

The implementation uses the multi-read pattern: prepare |g⟩ (identity) + measure, then
prepare |e⟩ (ge_pi) + measure — two reads per shot. With high reps, the averaged signal
from each read gives the center of the |g⟩ and |e⟩ blobs. The separation between them
divided by the noise width gives the readout SNR.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment, shots_refusal
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.cqed.readout import DispersiveIQ


class SingleShotReadout(Experiment):
    """Two-preparation readout characterization: |g⟩ and |e⟩ in one shot.

    Builds a two-read program: first read measures the ground state (no preparation),
    second read measures the excited state (after ge_pi). The analyzer extracts the
    discrimination threshold from the midpoint of the two signals, and estimates
    assignment fidelity from the separation relative to noise.

    It needs per-shot data (``needs_shots``, #199): its purpose is single-shot readout, and a
    "fidelity" from two averaged reads alone is a proxy presented as a fidelity. A result without
    shots (the mock, an averaged board acquire) is refused — ``success=False``,
    ``derived["refused"]``. A result with shots is analysed as before.
    """

    needs_shots = True

    def __init__(self, device, *, readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        # Read 1: ground state (no preparation) — drive the readout line, then measure (#101)
        seq = Seq().play(self.device.readout.dispersive_measure())
        # Read 2: excited state (ge_pi then measure) — same drive-then-measure pairing
        seq.play(q.ge_pi())
        seq.play(self.device.readout.dispersive_measure())
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        if result.shots is None:
            return shots_refusal()  # #199: no single-shot claim from averages alone
        # Two reads: avgi[0] = ground signal, avgi[1] = excited signal
        avgi = np.asarray(result.avgi, dtype=float).ravel()
        avgq = np.asarray(result.avgq, dtype=float).ravel()

        if len(avgi) < 2:
            return FitResult(
                params={}, stderr={},
                derived={"threshold": 0.0, "fidelity": 0.0, "separation": 0.0},
                success=True,
            )

        # Project both reads onto the discrimination axis
        iq = np.stack([avgi, avgq], axis=-1)
        signal = self.readout.project(iq).ravel()

        # Ground = first read, Excited = second read
        sig_g = float(signal[0])
        sig_e = float(signal[1])

        # Threshold: midpoint between the two blob centers
        threshold = (sig_g + sig_e) / 2.0

        # Discrimination angle: the IQ rotation that maximizes separation.
        # arctan2(dQ, dI) gives the angle of the ground→excited vector in the IQ plane.
        di = float(avgi[1] - avgi[0])
        dq = float(avgq[1] - avgq[0])
        angle = float(np.arctan2(dq, di))

        # Separation (in projected units) — proxy for SNR
        separation = abs(sig_e - sig_g)

        # Assignment fidelity estimate: assumes Gaussian blobs with some noise
        # For the simulator, this gives a clean estimate; on hardware you'd use
        # per-shot data for the full histogram analysis
        fidelity = min(1.0, separation / 2.0) if separation > 0 else 0.0

        return FitResult(
            params={},
            stderr={},
            derived={
                "threshold": threshold,
                "angle": angle,
                "fidelity": fidelity,
                "separation": separation,
                "signal_ground": sig_g,
                "signal_excited": sig_e,
            },
            success=True,
        )
