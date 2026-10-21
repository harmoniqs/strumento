"""Synthetic g/e readout geometry and populations-to-IQ builders, shared by blob-anchored tests.

Hardware-shaped analyzer tests feed averaged IQ anchored to g/e blob centres (raw IQ, neither
near 0.5) — the shape production emits — rather than a hand-built ``[0, 1]`` curve. This module is
the one definition of that synthetic geometry. Import as ``from tests._synthetic_readout import ...``.

The angle and threshold are DERIVED from the centres, in ``DispersiveIQ``'s convention: rotated-I
is ``I·cos θ + Q·sin θ``, so ``θ = atan2(ΔQ, ΔI)`` of ``E − G`` puts the g→e separation on the
rotated-I axis, and the threshold is the projected midpoint of the two centres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from strumento.core.result import MeasurementResult
from strumento.packs.cqed.readout import DispersiveIQ


@dataclass(frozen=True)
class BlobGeometry:
    """A g/e blob pair in raw IQ, with the discrimination axis and threshold derived from it."""

    g_center: tuple[float, float]
    e_center: tuple[float, float]

    @property
    def angle(self) -> float:
        """The g→e direction (radians): rotating by it puts the separation on rotated-I."""
        return math.atan2(self.e_center[1] - self.g_center[1], self.e_center[0] - self.g_center[0])

    def _project(self, c: tuple[float, float]) -> float:
        return c[0] * math.cos(self.angle) + c[1] * math.sin(self.angle)

    @property
    def threshold(self) -> float:
        """The projected midpoint of the g and e centres along :attr:`angle`."""
        return 0.5 * (self._project(self.g_center) + self._project(self.e_center))

    def readout(self, *, with_centres: bool = True) -> DispersiveIQ:
        """A calibrated ``DispersiveIQ`` on this axis — with the blob centres (what the Runner
        threads in) or, ``with_centres=False``, the bare calibrated axis with no centres."""
        if not with_centres:
            return DispersiveIQ(threshold=self.threshold, angle=self.angle)
        return DispersiveIQ(
            threshold=self.threshold,
            angle=self.angle,
            g_center=self.g_center,
            e_center=self.e_center,
        )

    def result_from_populations(self, pops, *, xpts=None) -> MeasurementResult:
        """Averaged IQ for a GIVEN excited-population array, linearly placed on the g→e segment."""
        p_e = np.asarray(pops, dtype=float)
        g = np.asarray(self.g_center, dtype=float)
        e = np.asarray(self.e_center, dtype=float)
        iq = g[None, :] * (1.0 - p_e[:, None]) + e[None, :] * p_e[:, None]
        xs = None if xpts is None else np.asarray(xpts)
        return MeasurementResult(xpts=xs, avgi=iq[:, 0], avgq=iq[:, 1], shots=None)


# Synthetic ge blob centres (raw IQ): g and e separate on a rotated axis, neither near 0.5 — the
# hardware-like shape a [0, 1]-assuming analyzer mishandles.
GE_BLOBS = BlobGeometry(g_center=(-4.0, -6.0), e_center=(2.5, -4.5))
G_CENTER = GE_BLOBS.g_center
E_CENTER = GE_BLOBS.e_center
ANGLE = GE_BLOBS.angle
THRESHOLD = GE_BLOBS.threshold


def dispersive_iq(geometry: BlobGeometry = GE_BLOBS, *, with_centres: bool = True) -> DispersiveIQ:
    """``geometry.readout(...)`` — the calibrated readout, defaulting to :data:`GE_BLOBS`."""
    return geometry.readout(with_centres=with_centres)


def result_from_populations(
    pops, *, xpts=None, geometry: BlobGeometry = GE_BLOBS
) -> MeasurementResult:
    """``geometry.result_from_populations(...)``, defaulting to :data:`GE_BLOBS`."""
    return geometry.result_from_populations(pops, xpts=xpts)
