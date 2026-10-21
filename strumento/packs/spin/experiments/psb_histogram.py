"""PSB readout tune-up: single-shot histogram → double-Gaussian threshold.

Repeated singlet/triplet SPAM shots produce a bimodal sensor histogram; fitting a
double Gaussian yields the discrimination threshold and readout visibility. The analyzer is the
useful offline piece (feed it shots from any acquisition); the sequence side firms up with the
SPAM step machinery.

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.fitters import FitResult, fit_double_gaussian


class PSBHistogram:
    """Analyzer for single-shot PSB data: histogram + double-Gaussian threshold fit."""

    def __init__(self, *, bins: int = 60) -> None:
        self.bins = bins

    def analyze(self, shots) -> FitResult:
        shots = np.asarray(shots, dtype=float).ravel()
        counts, edges = np.histogram(shots, bins=self.bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return fit_double_gaussian(centers, counts)
