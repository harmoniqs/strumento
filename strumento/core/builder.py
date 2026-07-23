"""``DeviceBuilder`` — config-as-code that *produces* the canonical device model (spec §5.3, D30).

A device instance is canonically **data** (the serialized Pydantic ``Device`` / ``Wiring``). Hand-
writing YAML is repetitive and forecloses programmatic generation, so this adds a **pure,
pack-agnostic** builder that produces the *same* ``Device`` — never a parallel state store. Three
authoring surfaces, one source of truth::

    Device (Pydantic)   ⇄   YAML/JSON   ⇄   DeviceBuilder

``DeviceBuilder(...).build(calib=...)`` and ``Device.load(equivalent.yaml)`` yield the identical
model. The builder is the same "producer of canonical data" shape the design uses elsewhere
(``from_board_description``, ``from_solution``), so single-source-of-truth (D10) is preserved.

**Pack-agnostic by construction.** The builder never names ``Transmon`` or any modality type; it
composes over the *loaded pack's* registered element types (resolved through the ``strumento.packs``
entry-point group). So a new pack gets device authoring for free, with no ``core`` change.

**Carve-out (D30).** Calibration *values* are never authored in code — they change every run (the
calibration loop / a device-model report writes them) and stay data, passed to ``build(calib=...)``
as a dict or a YAML/JSON file. The builder composes **topology + wiring (+ couplings)** only.

This module is **pure**: no ``qick`` import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from strumento.core.device import Device, _form_name, _load_pack
from strumento.core.wiring import Generator, Line, ReadoutWiring


class DeviceBuilder:
    """Compose a device's topology + wiring in code; ``build(calib=...)`` emits a canonical ``Device``."""

    def __init__(self, pack: str) -> None:
        self.pack_name = pack
        self._topology: list[dict] = []
        self._lines: dict[str, list[Generator]] = {}
        self._readout: dict | None = None
        self._couplings: list = []
        self._soccfg: dict | None = None

    # -- authoring -------------------------------------------------------------------

    def add(self, type_name: str, name: str, **fields) -> DeviceBuilder:
        """Add a named element of a pack-registered type (validated against the pack at build)."""
        self._topology.append({"type": type_name, "name": name, **fields})
        return self

    def wire(self, line_name: str, **generators) -> DeviceBuilder:
        """Wire a line's generators; each keyword is the generator's **role**.

        Values are ``Generator`` instances (role filled from the key) or plain field dicts.
        """
        gens: list[Generator] = []
        for role, gen in generators.items():
            if isinstance(gen, Generator):
                gens.append(gen.model_copy(update={"role": role}))
            elif isinstance(gen, dict):
                gens.append(Generator(role=role, **gen))
            else:
                raise TypeError(
                    f"wire({line_name!r}): role {role!r} must map to a Generator or a dict, "
                    f"got {type(gen).__name__}"
                )
        self._lines[line_name] = gens
        return self

    def readout(self, **fields) -> DeviceBuilder:
        """Set the readout wiring (``dac_ch``, ``adc_chs``, …)."""
        self._readout = fields
        return self

    def coupling(self, a: str, b: str, *, form, mediated_by: str | None = None) -> DeviceBuilder:
        """Declare a typed interaction edge between two elements (D31).

        ``form`` is a pack-registered coupling form (e.g. ``BeamSplitter``); ``mediated_by`` ties
        the edge to a driven element (a coupler). Strengths come from calibration, not here.
        """
        self._couplings.append({"a": a, "b": b, "form": form, "mediated_by": mediated_by})
        return self

    def soccfg(self, snapshot: dict) -> DeviceBuilder:
        """Attach a soccfg snapshot (as ``Device.load`` would from the instance file)."""
        self._soccfg = snapshot
        return self

    # -- build -----------------------------------------------------------------------

    def build(self, *, calib: Any) -> Device:
        """Validate, then emit a canonical :class:`DeviceConfig` and compose a ``Device`` from it.

        Routing through :meth:`Device.from_config` (rather than constructing the ``Device``
        directly) means a builder-authored device round-trips through the config store identically
        to a YAML-loaded one — the D30 "three authoring surfaces, one canonical model" guarantee.
        """
        from strumento.core.config import DeviceConfig

        _load_pack(self.pack_name)  # fail fast on an unknown pack
        self._validate()

        calib_data = calib
        if isinstance(calib, (str, Path)):
            calib_data = yaml.safe_load(Path(calib).read_text())

        wiring = {
            "lines": {
                name: Line(name=name, generators=gens).model_dump()
                for name, gens in self._lines.items()
            },
            "readout": ReadoutWiring(**self._readout).model_dump(),
        }
        config = DeviceConfig(
            pack=self.pack_name,
            topology=list(self._topology),
            wiring=wiring,
            calibration=dict(calib_data),
            couplings=[
                {"a": c["a"], "b": c["b"], "form": _form_name(c["form"]), "mediated_by": c["mediated_by"]}
                for c in self._couplings
            ],
            soccfg_snapshot=self._soccfg,
        )
        return Device.from_config(config)

    # -- construction-time validation (early, specific diagnostics) -------------------

    def _validate(self) -> None:
        if self._readout is None or not self._readout.get("adc_chs"):
            raise ValueError("readout() must be called with at least one ADC target (adc_chs) before build()")

        names = {entry["name"] for entry in self._topology}
        for c in self._couplings:
            for endpoint in (c["a"], c["b"]):
                if endpoint not in names:
                    raise ValueError(f"coupling references unknown element {endpoint!r} (add() it first)")
            if c["mediated_by"] is not None and c["mediated_by"] not in names:
                raise ValueError(f"coupling mediated_by unknown element {c['mediated_by']!r}")
