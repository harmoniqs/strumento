"""Photon-parity fringe experiment (slice 5).

Scans the second half-pi's phase through a full turn after the parity revival wait; the fringe
contrast is the parity signal. Analyzer fits a plain cosine (``fit_parity``).

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_parity
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.library.composite import parity
from strumento.packs.cqed.readout import DispersiveIQ


class ParityFringe(Experiment):
    """Parity fringe: ``hpi · wait(revival) · hpi(phase)`` with the phase swept 0–360°."""

    def __init__(self, device, *, mode: str = "manipulate", phases: Sweep | None = None,
                 fast: bool = False, readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.mode = mode
        self.fast = fast
        self.phases = phases if phases is not None else Sweep("phase", 0.0, 360.0, 61)
        self.readout = readout if readout is not None else DispersiveIQ(threshold=0.0, angle=0.0)

    def sequence(self) -> tuple[Seq, tuple]:
        seq = parity(self.device, self.mode, second_phase=self.phases, fast=self.fast).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        iq = np.stack([np.asarray(result.avgi), np.asarray(result.avgq)], axis=-1)
        return fit_parity(result.xpts, self.readout.project(iq))
