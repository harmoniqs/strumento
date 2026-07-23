"""cqed dispersive-IQ readout kind (spec §16.2).

``DispersiveIQ`` rotates the IQ cloud so the g/e separation lies along I, thresholds each shot to
a discrete state, and reduces to the excited-state population. It is the ``ReadoutKind`` the
slice-2+ fitters consume so they fit *populations* (matching the QILC measurement-vector
convention), not raw IQ.

This module is **pure**: no ``qick`` import.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strumento.core.readout import DataSource


@dataclass
class DispersiveIQ:
    """Threshold-and-angle dispersive discriminator.

    ``angle`` (radians) rotates ``(I, Q)`` so the state separation is along the rotated I axis;
    a shot is ``|e⟩`` (1) when rotated-I exceeds ``threshold``, else ``|g⟩`` (0). Raw arrays are
    ``(..., 2)`` with the last axis ``(I, Q)``; ``reduce`` averages over the trailing shot axis.
    """

    threshold: float
    angle: float = 0.0
    data_source: DataSource = DataSource.QICK_ADC

    def project(self, raw):
        """Rotate ``(I, Q)`` onto the discrimination axis and return the signal (rotated I).

        Works on *averaged* IQ (no thresholding), so fitters can fit the readout signal directly
        when per-shot data isn't kept — the common case for a swept averaging program.
        """
        raw = np.asarray(raw, dtype=float)
        i, q = raw[..., 0], raw[..., 1]
        return i * np.cos(self.angle) + q * np.sin(self.angle)

    def _rotated_i(self, raw):
        return self.project(raw)

    def discriminate(self, raw):
        """Per-shot state labels (0 = |g⟩, 1 = |e⟩), shape = raw without the trailing IQ axis."""
        return (self._rotated_i(raw) > self.threshold).astype(int)

    def reduce(self, raw):
        """Excited-state population, averaged over the trailing shot axis."""
        return np.mean(self.discriminate(raw), axis=-1)

    def herald(self, raw, *, keep_state: int = 0):
        """Per-shot **pass mask** for a herald readout event (spec R1(ii)).

        Discriminates each shot and returns ``True`` where the outcome equals ``keep_state`` —
        the shots to keep in post-selection (default: keep ``|g⟩``, i.e. a bare-ground herald).
        Feed the result into ``MeasurementResult.heralds[<lane>]``; ``postselect`` ANDs the lanes.
        """
        return self.discriminate(raw) == keep_state
