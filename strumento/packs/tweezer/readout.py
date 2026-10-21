"""tweezer imaging readout (D19) — the external-detector acid test of the readout seam.

Detection is a **camera + PC frame grabber**, not a QICK ADC (arXiv 2506.13724): fluorescence images
→ per-trap ROI brightness → occupancy. Implementing this against the same `core/readout.py` protocol
as `DispersiveIQ` is what proves the readout seam is genuinely IQ-agnostic (D15) — the reason
`DataSource.EXTERNAL_DETECTOR` exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strumento.core.readout import DataSource


@dataclass
class ImagingReadout:
    """Threshold per-trap ROI brightness → occupancy (0/1); reduce → per-trap loading fraction."""

    threshold: float
    data_source: DataSource = DataSource.EXTERNAL_DETECTOR

    def discriminate(self, raw):
        """Per-ROI occupancy: 1 if a trap's brightness exceeds the threshold, else 0."""
        return (np.asarray(raw, dtype=float) > self.threshold).astype(int)

    def reduce(self, raw):
        """Loading fraction, averaged over the trailing shot axis."""
        return np.mean(self.discriminate(raw), axis=-1)
