"""spin PSB / charge-sensing readout (spec §15.5) — the third ReadoutKind.

Pauli spin blockade read out via a charge sensor (RF reflectometry off the ADC): the sensor signal
is thresholded to a singlet/triplet (blocked/unblocked) outcome. Implementing this against the same
`core/readout.py` protocol as `DispersiveIQ` (cqed) and `ImagingReadout` (tweezer) validates the
readout seam a third, independent way — spinQICK already fits T1/T2 through this lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strumento.core.readout import DataSource


@dataclass
class PSBReadout:
    """Threshold a charge-sensor signal → singlet/triplet (0/1); reduce → triplet fraction."""

    threshold: float
    data_source: DataSource = DataSource.QICK_ADC  # RF reflectometry off the ADC

    def discriminate(self, raw):
        return (np.asarray(raw, dtype=float) > self.threshold).astype(int)

    def reduce(self, raw):
        return np.mean(self.discriminate(raw), axis=-1)
