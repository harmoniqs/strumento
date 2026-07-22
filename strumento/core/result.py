"""The measurement-result seam (spec §9, §18/D24).

:class:`MeasurementResult` is the typed object every ``acquire`` returns — the boundary between
the qick-facing program layer and the pure analysis/fitting layer. It carries the swept axis
(``xpts``), the averaged I/Q, optional per-shot ``raw`` data, the herald lanes used for
post-selection, and a free-form ``meta`` dict.

``meta["environment"]`` is a **reserved** slot for station telemetry (fridge temperature, LO
readings, …). Environment stamping is the one *retroactively-impossible* primitive — data taken
without a stamp can never be re-stamped — so the key is reserved here from day one even though
the Runner only starts writing it in Task 19.

This module is **pure**: it never imports ``qick``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class MeasurementResult:
    """A single acquisition's data + provenance.

    Attributes:
        xpts: the swept independent axis (lab-native units), or ``None`` for a single point.
        avgi, avgq: averaged in-phase / quadrature readout signal.
        raw: optional per-shot data (populated when shots are kept, e.g. for post-selection).
        herald_lanes: names of the herald measurements enabled for post-selection.
        meta: free-form provenance; ``meta["environment"]`` is the reserved telemetry slot.
    """

    xpts: Any
    avgi: Any
    avgq: Any
    raw: Any | None = None
    herald_lanes: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[Any, Any, Any]:
        """Return ``(xpts, avgi, avgq)`` — the shape legacy analysis code expects."""
        return (self.xpts, self.avgi, self.avgq)

    def postselect(self, *, lanes: tuple[str, ...] | None = None) -> MeasurementResult:
        """Post-select on the enabled herald lanes (logical AND across lanes).

        Slice-1 stub: with no ``herald_lanes`` this is the identity. The full per-shot masking
        logic lands with the active-reset slice (slice 7); until then a result with heralds
        declared is returned unchanged so callers can already thread the API.
        """
        active = self.herald_lanes if lanes is None else lanes
        if not active:
            return replace(self)
        # Deferred to slice 7 (spec R1); identity for now so the seam is stable.
        return replace(self)
