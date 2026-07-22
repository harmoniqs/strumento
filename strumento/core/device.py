"""Device instance loading + topology-derived namespaces (spec §5, D21).

A :class:`Device` is one physical fridge: a :class:`~strumento.core.wiring.Wiring` map, a parsed
calibration object, a soccfg snapshot (plain dict), and the composed element instances. It is
built by :meth:`Device.load` from a single instance YAML (topology + wiring + calib values).

**Physics-free core (D21/§15).** ``core`` never imports a pack by name. ``load`` resolves the
pack named in the YAML through the ``strumento.packs`` entry-point group, then delegates
calibration parsing and element composition to that pack. So the same ``load`` serves cqed,
tweezer, spin, … packs without change.

**Topology-derived namespaces (D21).** ``load`` walks the *named* element instances and binds
``dev.<name>`` to an instance of that element type's ``ops_cls``. ``dev.qubit`` exists because
the Schuster YAML names its transmon ``qubit`` — not because the pack hardcodes it. Element
types whose ops land in a later slice have ``ops_cls is None`` and are simply not bound yet.

This module is **pure**: it never imports ``qick``.
"""

from __future__ import annotations

import json
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import yaml

from strumento.core.wiring import LineRef, Wiring

#: attribute names on Device that an element instance name must not shadow.
_RESERVED = frozenset({"wiring", "calib", "soccfg_snapshot", "elements", "pack", "load"})


def _load_pack(name: str) -> Any:
    """Resolve a modality pack by name via the ``strumento.packs`` entry-point group."""
    for ep in entry_points(group="strumento.packs"):
        if ep.name == name:
            return ep.load()
    available = sorted(ep.name for ep in entry_points(group="strumento.packs"))
    raise KeyError(f"no pack {name!r} under 'strumento.packs' entry points (have {available})")


def _parse_couplings(raw: list[dict]) -> list:
    """Build typed coupling edges from a YAML ``couplings:`` list (form is a registered name)."""
    if not raw:
        return []
    from strumento.core.coupling import Coupling

    return [
        Coupling(a=c["a"], b=c["b"], form=c["form"], mediated_by=c.get("mediated_by"))
        for c in raw
    ]


class Device:
    """A loaded device instance: wiring + calibration + soccfg + composed elements/namespaces."""

    def __init__(
        self,
        *,
        wiring: Wiring,
        calib: Any,
        soccfg_snapshot: dict | None,
        elements: dict[str, Any],
        pack: Any,
        couplings: list | tuple = (),
    ) -> None:
        self.wiring = wiring
        self.calib = calib
        self.soccfg_snapshot = soccfg_snapshot
        self.elements = elements
        self.pack = pack
        self.couplings = list(couplings)  # typed interaction edges (D31)
        self._validate_wiring()
        self._bind_namespaces()

    @classmethod
    def load(cls, path: str | Path) -> "Device":
        """Parse an instance YAML → validate → compose elements → bind wiring & namespaces."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text())

        pack = _load_pack(raw["pack"])
        wiring = Wiring(**raw["wiring"])
        calib = pack.CALIBRATION(**raw["calibration"])
        elements = pack.compose_elements(raw["topology"], calib)

        soccfg_snapshot: dict | None = None
        snap_name = raw.get("soccfg_snapshot")
        if snap_name:
            soccfg_snapshot = json.loads((path.parent / snap_name).read_text())

        couplings = _parse_couplings(raw.get("couplings", []))

        return cls(
            wiring=wiring,
            calib=calib,
            soccfg_snapshot=soccfg_snapshot,
            elements=elements,
            pack=pack,
            couplings=couplings,
        )

    # -- internals ---------------------------------------------------------------------

    def _validate_wiring(self) -> None:
        """Every element's declared LineRefs must exist (line + role) in the wiring."""
        for name, el in self.elements.items():
            for ref in el.lines():
                line = self.wiring.lines.get(ref.line)
                if line is None:
                    raise ValueError(
                        f"element {name!r} drives line {ref.line!r}, absent from wiring"
                    )
                if not any(g.role == ref.role for g in line.generators):
                    raise ValueError(
                        f"element {name!r} needs role {ref.role!r} on line {ref.line!r}, "
                        f"which has none"
                    )

    def _bind_namespaces(self) -> None:
        """Bind ``dev.<name>`` to ``ops_cls(dev, element)`` for each element that has one (D21)."""
        for name, el in self.elements.items():
            ops_cls = getattr(type(el), "ops_cls", None)
            if ops_cls is None:
                continue
            if name in _RESERVED or hasattr(self, name):
                raise ValueError(f"element name {name!r} collides with a reserved Device attribute")
            setattr(self, name, ops_cls(self, el))

    def to_operator_model(self, *, frame: str = "rotating", cutoffs: dict[str, int], include=None):
        """Export topology + couplings + calibration as a framework-neutral Hamiltonian spec (D31).

        Mirror of the ``from_solution`` import seam — see
        :func:`strumento.core.operators.to_operator_model`. Core stays Piccolo-free; the upstream
        ``to_piccolo_system`` (or QuTiP) bridge consumes the returned :class:`OperatorModel`.
        """
        from strumento.core.operators import to_operator_model

        return to_operator_model(self, frame=frame, cutoffs=cutoffs, include=include)

    def __repr__(self) -> str:
        return f"Device(pack={self.pack.__name__!r}, elements={sorted(self.elements)})"
