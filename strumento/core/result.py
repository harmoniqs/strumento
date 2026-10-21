"""The measurement-result seam (D13, D24).

:class:`MeasurementResult` is the typed object every ``acquire`` returns — the boundary between
the qick-facing program layer and the pure analysis/fitting layer. It carries the swept axis
(``xpts``), the averaged I/Q, optional per-shot ``raw`` data, the herald lanes used for
post-selection, and a free-form ``meta`` dict.

``meta["environment"]`` is a **reserved** slot for station telemetry (fridge temperature, LO
readings, …). Environment stamping is the one *retroactively-impossible* primitive — data taken
without a stamp can never be re-stamped — so the key is reserved here from day one even though
the Runner writes it (D34).

:meth:`MeasurementResult.to_wire` / :meth:`MeasurementResult.from_wire` are the JSON form (D35):
numpy arrays become lists, herald lanes become **bool** lists, and ``meta`` is coerced JSON-safe
so a result can cross a process boundary (agent, store, job server) without a custom encoder.
``strumento schema result`` publishes that shape; see :mod:`strumento.core.schemas`.

This module is **pure**: it never imports ``qick``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


def _listify(value: Any) -> Any:
    """numpy → nested lists; scalars and ``None`` pass through unchanged."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_listify(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    """Coerce arbitrary provenance into JSON primitives (last resort: ``str``).

    ``meta`` is free-form by design, so serialization must never fail on it: arrays and numpy
    scalars convert, tuples become lists, dicts recurse, and anything else (a ``Path``, a live
    object) degrades to its string form rather than raising at the wire boundary.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class MeasurementResult:
    """A single acquisition's data + provenance.

    Attributes:
        xpts: the swept independent axis (lab-native units), or ``None`` for a single point.
        avgi, avgq: averaged in-phase / quadrature readout signal.
        raw: optional raw acquisition payload (the native soc return), for reprocessing.
        shots: optional **per-shot** data for the data readout — leading axes index sweep
            points, the *last* axis is the shot axis. Kept when a program acquires shot-resolved
            (heralding, post-selection, single-shot readout). QEC analysis is shot-resolved by
            definition, and this is the one primitive that is **retroactively impossible** to add
            to already-taken data — so it is first-class here from day one.
        heralds: per-shot **pass masks** by lane name — ``heralds["reset"]`` is a boolean array
            over the shot axis, ``True`` where that herald round passed. Produced by a readout
            kind's herald helper (e.g. :meth:`DispersiveIQ.herald`).
        herald_lanes: the lanes enabled for :meth:`postselect` by default (a subset of
            ``heralds``; empty ⇒ all present heralds).
        meta: free-form provenance; ``meta["environment"]`` is the reserved telemetry slot,
            ``meta["postselect_survival"]`` is written by :meth:`postselect`.
    """

    xpts: Any
    avgi: Any
    avgq: Any
    raw: Any | None = None
    shots: Any | None = None
    heralds: dict[str, Any] = field(default_factory=dict)
    herald_lanes: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[Any, Any, Any]:
        """Return ``(xpts, avgi, avgq)`` — the shape legacy analysis code expects."""
        return (self.xpts, self.avgi, self.avgq)

    # -- wire form (D35) -----------------------------------------------------------------

    def to_wire(self) -> dict:
        """The JSON-safe dict form — ``json.dumps`` works on it with no encoder hook.

        Arrays (``xpts``/``avgi``/``avgq``/``shots``/``raw``) become nested lists, herald lanes
        become **bool** lists (a herald is a pass mask, not a count), ``herald_lanes`` a list of
        names, and ``meta`` is coerced leaf-by-leaf (see :func:`_jsonable`). Schema:
        ``strumento schema result``.
        """
        return {
            "xpts": _listify(self.xpts),
            "avgi": _listify(self.avgi),
            "avgq": _listify(self.avgq),
            "raw": _jsonable(self.raw),
            "shots": _listify(self.shots),
            "heralds": {
                str(name): np.asarray(mask, dtype=bool).tolist()
                for name, mask in self.heralds.items()
            },
            "herald_lanes": [str(name) for name in self.herald_lanes],
            "meta": _jsonable(self.meta),
        }

    @classmethod
    def from_wire(cls, wire: dict) -> MeasurementResult:
        """Rebuild from :meth:`to_wire`, restoring arrays (heralds as ``bool`` arrays).

        Round-trip stable: ``from_wire(w).to_wire() == w``. Scalars stay scalars, so a
        single-point result survives unchanged.
        """

        def _arr(value):
            return None if value is None else (
                np.asarray(value) if isinstance(value, list) else value
            )

        return cls(
            xpts=_arr(wire.get("xpts")),
            avgi=_arr(wire["avgi"]),
            avgq=_arr(wire["avgq"]),
            raw=wire.get("raw"),
            shots=_arr(wire.get("shots")),
            heralds={
                str(name): np.asarray(mask, dtype=bool)
                for name, mask in (wire.get("heralds") or {}).items()
            },
            herald_lanes=tuple(wire.get("herald_lanes") or ()),
            meta=dict(wire.get("meta") or {}),
        )

    def keep_mask(self, lanes: tuple[str, ...] | None = None):
        """The per-shot survival mask: logical AND of the active herald lanes' pass masks.

        ``lanes=None`` uses ``herald_lanes`` if set, else every lane in ``heralds``. Returns
        ``None`` when no herald lanes apply (nothing to post-select on ⇒ keep everything).
        """
        active = tuple(lanes) if lanes is not None else (self.herald_lanes or tuple(self.heralds))
        if not active:
            return None
        masks = [np.asarray(self.heralds[name], dtype=bool) for name in active]
        combined = masks[0].copy()
        for extra in masks[1:]:
            combined = combined & extra
        return combined

    def postselect(self, *, lanes: tuple[str, ...] | None = None) -> MeasurementResult:
        """Keep only shots that pass the AND of the enabled herald lanes.

        With no herald lanes this is the identity (records survival 1.0). With a **flat shot
        axis** (the common heralded case — active reset, state-prep verification) the surviving
        shots are selected out of ``shots`` and every herald lane; the survival fraction lands in
        ``meta["postselect_survival"]``. For a multi-dimensional shot axis (per-point masks over a
        sweep, which would be ragged if compacted) the data is left intact and the mask is exposed
        via :meth:`keep_mask` for a reducer to weight by — filtering is not forced into a ragged
        shape.
        """
        mask = self.keep_mask(lanes)
        if mask is None:
            return replace(self, meta={**self.meta, "postselect_survival": 1.0})

        active = tuple(lanes) if lanes is not None else (self.herald_lanes or tuple(self.heralds))
        survival = float(np.mean(mask)) if mask.size else 0.0
        meta = {**self.meta, "postselect_survival": survival, "postselect_lanes": active}

        if mask.ndim != 1:
            # ragged if compacted per-point; expose the mask instead of forcing a shape
            return replace(self, meta={**meta, "keep_mask": mask})

        def _filter(arr):
            a = np.asarray(arr)
            return a[..., mask] if a.ndim and a.shape[-1] == mask.shape[0] else arr

        new_shots = _filter(self.shots) if self.shots is not None else None
        new_heralds = {name: _filter(v) for name, v in self.heralds.items()}
        return replace(self, shots=new_shots, heralds=new_heralds, meta=meta)
