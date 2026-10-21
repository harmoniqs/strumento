"""The versioned configuration store (D9) — immutable snapshots + a movable pointer.

This is the seam that turns strumento from "runs experiments" into "operates a device over time":
every calibration is a content-addressed, write-once snapshot, and a named ref (``main``) moves
across them as the device drifts and re-calibrates. Because the object id is the hash of the
config's bytes, a version is **permanently reproducible** — an id resolves forever to a
byte-identical device.

Design (D9): **immutability is the load-bearing property**, not any particular
backend. :class:`ConfigStore` is the interface; :class:`FileConfigStore` is a dependency-free
default (a directory of write-once JSON objects + ref files + an append-only log) with no queue
and no database — a lab or a fleet can supply another backend behind the same protocol.

Provenance (who/when/why, the fit summary, the value diff) is recorded in the append-only log,
**never inside the hashed object**, so identical calibration always hashes identically.

This module is **pure**: no ``qick`` import.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from strumento.core.config import DeviceConfig, apply_updates


@dataclass
class Provenance:
    """Why a config version exists — recorded beside the object, not hashed into it."""

    content_id: str
    parent_id: str | None = None
    source: str = ""                       # e.g. "amplitude_rabi fit", "manual", "device-model report"
    updates: dict[str, Any] = field(default_factory=dict)   # {path: [old, new]}
    timestamp: str | None = None           # ISO-8601, caller-supplied (kept out of the hashed config)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content_id": self.content_id,
            "parent_id": self.parent_id,
            "source": self.source,
            "updates": self.updates,
            "timestamp": self.timestamp,
            "extra": self.extra,
        }


@runtime_checkable
class ConfigStore(Protocol):
    """A content-addressed, immutable store of :class:`DeviceConfig` with a movable ref."""

    def put(self, config: DeviceConfig) -> str: ...
    def get(self, content_id: str) -> DeviceConfig: ...
    def set_ref(self, ref: str, content_id: str) -> None: ...
    def resolve_ref(self, ref: str) -> str | None: ...
    def append_provenance(self, ref: str, prov: Provenance) -> None: ...
    def log(self, ref: str) -> list[Provenance]: ...


class FileConfigStore:
    """A dependency-free :class:`ConfigStore`: write-once objects + ref files + an append-only log.

    Layout under ``root``::

        objects/<content-id>.json     write-once canonical config snapshots
        refs/<ref>                     a single line: the content id the ref points at
        log/<ref>.jsonl                append-only provenance, one JSON record per version
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        (self.root / "refs").mkdir(parents=True, exist_ok=True)
        (self.root / "log").mkdir(parents=True, exist_ok=True)

    def put(self, config: DeviceConfig) -> str:
        """Store a config (write-once); return its content id. Re-putting identical bytes is a no-op."""
        cid = config.content_id()
        obj = self.root / "objects" / f"{cid}.json"
        if not obj.exists():                       # write-once: never rewrite an existing object
            obj.write_text(config.canonical_json())
        return cid

    def get(self, content_id: str) -> DeviceConfig:
        obj = self.root / "objects" / f"{content_id}.json"
        if not obj.exists():
            raise KeyError(f"no config object {content_id!r} in store")
        return DeviceConfig.from_dict(json.loads(obj.read_text()))

    def set_ref(self, ref: str, content_id: str) -> None:
        if not (self.root / "objects" / f"{content_id}.json").exists():
            raise KeyError(f"cannot point {ref!r} at unknown object {content_id!r}")
        (self.root / "refs" / ref).write_text(content_id)

    def resolve_ref(self, ref: str) -> str | None:
        f = self.root / "refs" / ref
        return f.read_text().strip() if f.exists() else None

    def append_provenance(self, ref: str, prov: Provenance) -> None:
        with (self.root / "log" / f"{ref}.jsonl").open("a") as fh:
            fh.write(json.dumps(prov.to_dict()) + "\n")

    def log(self, ref: str) -> list[Provenance]:
        f = self.root / "log" / f"{ref}.jsonl"
        if not f.exists():
            return []
        out: list[Provenance] = []
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out.append(Provenance(**d))
        return out


def compose_device(config: DeviceConfig) -> Any:
    """The default write-back validator: compose a device from ``config`` (raises if it is invalid).

    ``Device.from_config`` runs the pack's calibration model (``extra="forbid"``, typed and
    bounded fields) and the wiring checks. It is imported here, at call time, so this module stays
    free of any pack import: the pack is resolved by entry point inside ``Device``.
    """
    from strumento.core.device import Device

    return Device.from_config(config)


def unvalidated(config: DeviceConfig) -> None:
    """The explicit bypass: accept any candidate config without composing it.

    For a writer whose calibration tree is deliberately wider than a pack's schema — the bring-up
    engine banks values the pack model does not declare — or whose config is not a composable
    device at all. Passing it is a visible, greppable statement that the write is unchecked.
    """
    return None


def record_calibration(
    store: ConfigStore,
    ref: str,
    base: DeviceConfig,
    updates: dict[str, Any],
    *,
    source: str = "",
    timestamp: str | None = None,
    extra: dict | None = None,
    validate: Callable[[DeviceConfig], Any] = compose_device,
) -> tuple[str, DeviceConfig]:
    """Apply a calibration write-back, validate it, snapshot it immutably, and advance ``ref`` to it.

    The one call a calibration loop makes each time it re-measures a device: it produces a *new*
    version from ``base`` with ``updates`` applied (dotted calibration paths → new values),
    validates it, stores it, records provenance (the value diff, the parent version, a
    caller-supplied source label and timestamp), and moves ``ref``. Returns
    ``(content_id, new_config)``.

    **Validation comes first (#191).** ``validate`` is called on the candidate config before
    anything is written; whatever it raises propagates, and the store is left exactly as it was —
    no object, no provenance line, no ref move. The default composes a device from the candidate
    (:func:`compose_device`), so a typo'd key, a wrong type, a NaN or a negative time is refused
    by the pack's calibration schema. A caller can pass its own callable, or :func:`unvalidated`
    to write without the check.

    Reproducibility: pass the same ``base`` and ``updates`` and you get the same ``content_id`` —
    provenance (timestamp/source) does not perturb the hash, and validation never touches the
    bytes that are hashed.
    """
    new_config, diff = apply_updates(base, updates)
    validate(new_config)
    parent = store.resolve_ref(ref)
    cid = store.put(new_config)
    prov = Provenance(
        content_id=cid,
        parent_id=parent,
        source=source,
        updates={path: list(pair) for path, pair in diff.items()},
        timestamp=timestamp,
        extra=extra or {},
    )
    store.append_provenance(ref, prov)
    store.set_ref(ref, cid)
    return cid, new_config
