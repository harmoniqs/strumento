"""The canonical device configuration — the serializable source form + content addressing (D9, D30).

A device instance is **data** (D10: single source of truth). :class:`DeviceConfig` is that
data in one typed object: the pack name, the topology, the wiring, the calibration values, the
typed couplings, alias namespaces, and the soccfg snapshot. A live :class:`~strumento.core.device.Device`
composes *from* a config and can emit its config back — so ``Device ⇄ DeviceConfig ⇄ YAML`` all
yield the identical model (the D30 round-trip, now real rather than asserted).

**Content addressing (D9).** ``config.content_id()`` is the SHA-256 of the config's canonical JSON
(sorted keys, no whitespace). Immutability is the load-bearing property: a content id resolves
forever to a byte-identical config, so a stored calibration version is permanently reproducible.
Provenance (timestamps, fit summaries) is *never* part of the hashed config — it lives beside the
object in the store's log, so identical calibration always hashes identically.

**Write-back (the seam that makes strumento operate devices, not just run experiments).**
:func:`apply_updates` produces a *new* config with calibration values changed by dotted path
(``"qubit.pulses.pi_ge.gain"``) — never mutating in place — and returns the value diff so the
change is auditable.

This module is **pure**: no ``qick`` import.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class DeviceConfig:
    """The canonical, serializable source form of a device instance.

    Every field is plain JSON-able data (dicts/lists/scalars) so the config serializes
    deterministically and hashes reproducibly. ``elements`` are *derived* (composed by the pack)
    and deliberately absent — the config is the source, not the composition.
    """

    pack: str
    topology: list[dict]
    wiring: dict
    calibration: dict
    couplings: list[dict] = field(default_factory=list)
    aliases: dict = field(default_factory=dict)
    soccfg_snapshot: dict | None = None

    def canonical_json(self) -> str:
        """Deterministic JSON (sorted keys, tight separators) — the byte string that is hashed."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def content_id(self) -> str:
        """A stable content id: ``cfg-<sha256>`` over :meth:`canonical_json`."""
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"cfg-{digest}"

    def short_id(self) -> str:
        """The content id abbreviated for display (``cfg-<12 hex>``)."""
        return self.content_id()[: len("cfg-") + 12]

    def to_dict(self) -> dict:
        """A plain dict (e.g. for YAML/JSON emission)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DeviceConfig:
        """Build from a plain dict (a parsed YAML/JSON instance or a stored object)."""
        return cls(
            pack=data["pack"],
            topology=list(data.get("topology", [])),
            wiring=dict(data["wiring"]),
            calibration=dict(data["calibration"]),
            couplings=list(data.get("couplings", [])),
            aliases=dict(data.get("aliases", {})),
            soccfg_snapshot=data.get("soccfg_snapshot"),
        )


def apply_updates(config: DeviceConfig, updates: dict[str, Any]) -> tuple[DeviceConfig, dict[str, tuple]]:
    """Return a new config with calibration values changed by dotted path, plus a value diff.

    ``updates`` maps a dotted path *into the calibration tree* to its new value, e.g.
    ``{"qubit.f_ge": 4001.2, "qubit.pulses.pi_ge.gain": 8300}``. The update is explicit by design:
    strumento never guesses which calibration field a fit result belongs to — the caller (an
    experiment, a calibration loop) states the mapping, so the write-back is auditable and
    unit-safe (D23: a gain field takes whatever unit that field stores).

    Immutable: ``config`` is untouched; a deep-copied calibration carries the change. The returned
    diff maps each path to ``(old_value, new_value)``.
    """
    new_calib = copy.deepcopy(config.calibration)
    diff: dict[str, tuple] = {}
    for path, new_value in updates.items():
        keys = path.split(".")
        node: Any = new_calib
        for key in keys[:-1]:
            node = _descend(node, key, path)
        leaf = keys[-1]
        old_value = _descend(node, leaf, path, missing_ok=True)
        _assign(node, leaf, new_value, path)
        diff[path] = (old_value, new_value)
    return replace(config, calibration=new_calib), diff


def _descend(node: Any, key: str, path: str, *, missing_ok: bool = False) -> Any:
    """Index into a dict by key or a list by integer index; raise a clear error on a bad path."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        if missing_ok:
            return None
        raise KeyError(f"calibration path {path!r}: no key {key!r} in {sorted(node)}")
    if isinstance(node, list):
        try:
            return node[int(key)]
        except (ValueError, IndexError) as err:
            raise KeyError(f"calibration path {path!r}: bad list index {key!r}") from err
    if missing_ok:
        return None
    raise KeyError(f"calibration path {path!r}: cannot descend into {type(node).__name__} at {key!r}")


def _assign(node: Any, key: str, value: Any, path: str) -> None:
    if isinstance(node, dict):
        node[key] = value
    elif isinstance(node, list):
        node[int(key)] = value
    else:  # pragma: no cover - guarded by _descend on the way down
        raise KeyError(f"calibration path {path!r}: cannot assign into {type(node).__name__}")
