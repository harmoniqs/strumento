"""Device instance loading + topology-derived namespaces (D21).

A :class:`Device` is one physical fridge: a :class:`~strumento.core.wiring.Wiring` map, a parsed
calibration object, a soccfg snapshot (plain dict), and the composed element instances. It is
built by :meth:`Device.load` from a single instance YAML (topology + wiring + calib values).

**Physics-free core (D15, D21).** ``core`` never imports a pack by name. ``load`` resolves the
pack named in the YAML through the ``strumento.packs`` entry-point group, then delegates
calibration parsing and element composition to that pack. So the same ``load`` serves cqed,
tweezer, spin, … packs without change.

**Topology-derived namespaces (D21).** ``load`` walks the *named* element instances and binds
``dev.<name>`` to an instance of that element type's ``ops_cls``. ``dev.qubit`` exists because
the instance YAML names its transmon ``qubit`` — not because the pack hardcodes it. Element
types that ship no ops yet have ``ops_cls is None`` and are simply not bound.

This module is **pure**: it never imports ``qick``.
"""

from __future__ import annotations

import json
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import yaml

from strumento.core.wiring import Wiring

#: attribute names on Device that an element instance name must not shadow.
_RESERVED = frozenset({"wiring", "calib", "soccfg_snapshot", "elements", "pack", "load"})


def installed_packs() -> list[str]:
    """The names of the installed modality packs (the ``strumento.packs`` entry points), sorted."""
    return sorted(ep.name for ep in entry_points(group="strumento.packs"))


def _load_pack(name: str) -> Any:
    """Resolve a modality pack by name via the ``strumento.packs`` entry-point group."""
    for ep in entry_points(group="strumento.packs"):
        if ep.name == name:
            return ep.load()
    available = installed_packs()
    raise KeyError(f"no pack {name!r} under 'strumento.packs' entry points (have {available})")


def _form_name(form: Any) -> str:
    """A coupling form's registered/class name (for serializing a live edge back to config)."""
    return form if isinstance(form, str) else getattr(form, "__name__", str(form))


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
        aliases: dict | None = None,
        config: Any | None = None,
    ) -> None:
        self.wiring = wiring
        self.calib = calib
        self.soccfg_snapshot = soccfg_snapshot
        self.elements = elements
        self.pack = pack
        self.couplings = list(couplings)  # typed interaction edges (D31)
        self._config = config  # the canonical source form (DeviceConfig), retained for round-trip
        self._validate_wiring()
        self._bind_namespaces()
        self._bind_aliases(aliases or {})

    @classmethod
    def from_config(cls, config: Any) -> Device:
        """Compose a live ``Device`` from a canonical :class:`~strumento.core.config.DeviceConfig`.

        This is the one composition path; :meth:`load` and ``DeviceBuilder.build`` both funnel
        through it, so ``Device ⇄ DeviceConfig`` is a faithful round-trip (D30). The config is
        retained on the device (``to_config``) so a live device can be re-serialized, hashed,
        and versioned in the config store (D9).
        """
        pack = _load_pack(config.pack)
        wiring = Wiring(**config.wiring)
        calib = pack.CALIBRATION(**config.calibration)
        elements = pack.compose_elements(config.topology, calib)
        couplings = _parse_couplings(config.couplings)
        return cls(
            wiring=wiring,
            calib=calib,
            soccfg_snapshot=config.soccfg_snapshot,
            elements=elements,
            pack=pack,
            couplings=couplings,
            aliases=config.aliases,
            config=config,
        )

    def to_config(self) -> Any:
        """The canonical :class:`~strumento.core.config.DeviceConfig` this device composed from.

        Reflects the *current* calibration: after a write-back (``config.apply_updates``) the new
        config is what you snapshot into the store.
        """
        from strumento.core.config import DeviceConfig

        if self._config is not None:
            return self._config
        # A device constructed directly (not via from_config/load) reconstructs its config from
        # live state — calibration and wiring re-serialize via Pydantic; topology from elements.
        return DeviceConfig(
            pack=self.pack.__name__.rsplit(".", 1)[-1],
            topology=[{"type": type(el).__name__, "name": name} for name, el in self.elements.items()],
            wiring=self.wiring.model_dump(),
            calibration=self.calib.model_dump(),
            couplings=[{"a": c.a, "b": c.b, "form": _form_name(c.form), "mediated_by": c.mediated_by}
                       for c in self.couplings],
            soccfg_snapshot=self.soccfg_snapshot,
        )

    @classmethod
    def load(cls, path: str | Path) -> Device:
        """Parse an instance YAML → :class:`DeviceConfig` → compose (via :meth:`from_config`)."""
        from strumento.core.config import DeviceConfig

        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))  # device files are UTF-8 by authorship (µ, —); never locale-dependent

        snap_name = raw.get("soccfg_snapshot")
        soccfg_snapshot: dict | None = None
        if snap_name:
            soccfg_snapshot = json.loads((path.parent / snap_name).read_text(encoding="utf-8"))

        config = DeviceConfig(
            pack=raw["pack"],
            topology=raw["topology"],
            wiring=raw["wiring"],
            calibration=raw["calibration"],
            couplings=raw.get("couplings", []),
            aliases=raw.get("aliases", {}),
            soccfg_snapshot=soccfg_snapshot,
        )
        return cls.from_config(config)

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

    def _bind_aliases(self, aliases: dict) -> None:
        """Bind instance-declared alias namespaces (D21): ``dev.<alias>`` -> a pack alias-ops
        class over a named element (e.g. ``sideband`` -> the transmon's sideband generator).

        The pack opts in by exposing ``ALIAS_OPS: dict[str, type]``; the instance YAML says
        ``aliases: {<alias>: {element: <name>, ops: <key>}}``.
        """
        if not aliases:
            return
        alias_ops = getattr(self.pack, "ALIAS_OPS", {})
        for alias, spec in aliases.items():
            ops_key, el_name = spec["ops"], spec["element"]
            if ops_key not in alias_ops:
                raise KeyError(
                    f"pack {self.pack.__name__!r} has no alias ops {ops_key!r} "
                    f"(have {sorted(alias_ops)})"
                )
            if el_name not in self.elements:
                raise KeyError(f"alias {alias!r} points at unknown element {el_name!r}")
            if alias in _RESERVED or hasattr(self, alias):
                raise ValueError(f"alias name {alias!r} collides with a Device attribute")
            setattr(self, alias, alias_ops[ops_key](self, self.elements[el_name]))

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
