"""Persistent measurement-data store — write-once results alongside the device (D9 pattern).

The datachest is the measurement counterpart of the :class:`~strumento.core.store.ConfigStore`:
the store holds immutable calibration snapshots (what the device *believes*); the datachest holds
immutable measurement results (what was *measured*). Both live at the device instance directory;
both are write-once, content-addressed, and permanently reproducible.

:class:`DataChest` is the protocol; :class:`FileDataChest` is the dependency-free default
(a directory of ``.npz`` result files + a ``.jsonl`` manifest). A lab or fleet can supply another
backend behind the same protocol.

Each entry stores:
- **result arrays** — ``xpts``, ``avgi``, ``avgq``, ``shots``, ``heralds`` (as ``.npz``)
- **metadata** — environment stamp, dc_snapshot, soc info (as JSON inside the ``.npz``)
- **sequence** — the ``Seq`` that was played, serialized (every ``Pulse`` with its parameters,
  including ``Arb`` waveform samples for OC-imported pulses) — the waveform provenance
- **manifest entry** — experiment name, timestamp, task linkage, derived quantities

The task record's ``artifact`` events link to datachest entry IDs, so a campaign ledger says
"step 3 produced result ``abc123``" and the datachest at the device path holds the arrays.

This module is **pure**: no ``qick`` import, no plotting dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from strumento.core.result import MeasurementResult, _jsonable

# =============================================================================
# Seq serialization — waveform provenance
# =============================================================================


def seq_to_wire(seq) -> dict:
    """Serialize a ``Seq`` to a JSON-safe dict — every Pulse with its parameters.

    ``Arb`` envelope samples are included inline as lists (they may be large; the caller
    can extract them to separate ``.npz`` files for storage efficiency). Sweep-valued fields
    are serialized as their ``{start, stop, on}`` representation.
    """
    from strumento.core.pulses import (
        Arb,
        Const,
        Delay,
        FlatTop,
        Gaussian,
        Measure,
        Pulse,
        Raw,
        Segment,
    )

    def _envelope(env) -> dict:
        if isinstance(env, Gaussian):
            return {"type": "Gaussian", "sigma_us": env.sigma_us}
        if isinstance(env, FlatTop):
            return {"type": "FlatTop", "ramp_sigma_us": _field(env.ramp_sigma_us), "length_us": _field(env.length_us)}
        if isinstance(env, Const):
            return {"type": "Const", "length_us": _field(env.length_us)}
        if isinstance(env, Arb):
            d: dict[str, Any] = {"type": "Arb", "idata": list(env.idata)}
            if env.qdata is not None:
                d["qdata"] = list(env.qdata)
            return d
        return {"type": type(env).__name__}

    def _field(value) -> Any:
        """Serialize a possibly-swept field."""
        from strumento.core.sweeps import Sweep
        if isinstance(value, Sweep):
            return {"__sweep__": True, "start": value.start, "stop": value.stop, "on": value.on, "name": value.name}
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        return float(value)

    def _pulse(p: Pulse) -> dict:
        wire = {
            "type": "Pulse",
            "line": [p.line.line, p.line.role],
            "freq_mhz": _field(p.freq_mhz),
            "gain": _field(p.gain),
            "phase_deg": _field(p.phase_deg),
            "start_us": _field(p.start_us),
            "envelope": _envelope(p.envelope),
            "label": p.label,
        }
        if p.readout is not None:
            wire["readout"] = {
                "adc_ch": p.readout.adc_ch,
                "phase_deg": _field(p.readout.phase_deg),
                "window_us": _field(p.readout.window_us),
            }
        return wire

    def _op(op) -> dict:
        if isinstance(op, Pulse):
            return _pulse(op)
        if isinstance(op, Delay):
            return {"type": "Delay", "duration_us": _field(op.duration_us)}
        if isinstance(op, Measure):
            return {"type": "Measure"}
        if isinstance(op, Segment):
            return {"type": "Segment", "pulses": [_pulse(p) for p in op.pulses]}
        if isinstance(op, Raw):
            return {"type": "Raw"}
        # Branch, Repeat — serialize structurally
        return {"type": type(op).__name__}

    return {
        "ops": [_op(op) for op in seq.ops],
        "meta": _jsonable(getattr(seq, "meta", {})),
    }


# =============================================================================
# DataChest protocol
# =============================================================================


@runtime_checkable
class DataChest(Protocol):
    """The measurement-data storage protocol — content-addressed, write-once."""

    def save(
        self,
        result: MeasurementResult,
        *,
        seq=None,
        experiment: str = "",
        fit: dict | None = None,
        task_id: str = "",
        salt: str | None = None,
    ) -> str:
        """Write a result + sequence. Returns a content ID (see :meth:`FileDataChest.save`)."""
        ...

    def load(self, entry_id: str) -> dict:
        """Load an entry by ID. Returns ``{result, sequence, meta, fit}``."""
        ...

    def list_entries(self, *, experiment: str | None = None, limit: int = 100) -> list[dict]:
        """List manifest entries, newest first."""
        ...


# =============================================================================
# FileDataChest — dependency-free default
# =============================================================================


@dataclass
class FileDataChest:
    """A directory of write-once ``.npz`` result files + a ``.jsonl`` manifest.

    Default storage layout under a device instance::

        devices/multimode_demo/
            data/
                manifest.jsonl
                abc123.npz
                def456.npz

    Each ``.npz`` contains the result arrays, serialized sequence, and metadata.
    The manifest is an append-only index for fast queries.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

    # -- save ------------------------------------------------------------------

    def save(
        self,
        result: MeasurementResult,
        *,
        seq=None,
        experiment: str = "",
        fit: dict | None = None,
        task_id: str = "",
        salt: str | None = None,
    ) -> str:
        """Write result arrays + sequence + metadata. Returns a content-addressed ID.

        The ID is a hash of the result arrays, so identical data shares one write-once entry
        (and the first save's metadata, sequence and fit). ``salt`` gives a save its own entry
        for identical data: it is folded into the hash only when given, so an unsalted ID is
        unchanged. Pass a per-run identity when each save must cite its own run, as an MCP
        session does — otherwise two runs returning identical data (e.g. the mock's zero IQ)
        would resolve to the first run's record.
        """
        # Build arrays dict for npz
        arrays: dict[str, Any] = {}
        if result.xpts is not None:
            arrays["xpts"] = np.asarray(result.xpts)
        arrays["avgi"] = np.asarray(result.avgi)
        arrays["avgq"] = np.asarray(result.avgq)
        if result.shots is not None:
            arrays["shots"] = np.asarray(result.shots)
        for name, mask in result.heralds.items():
            arrays[f"herald_{name}"] = np.asarray(mask, dtype=bool)

        # Metadata (JSON-safe)
        meta = _jsonable(result.meta)

        # Sequence provenance
        seq_wire = seq_to_wire(seq) if seq is not None else None

        # Fit (for convenience — can be re-derived)
        fit_wire = _jsonable(fit) if fit is not None else None

        # Content ID from a hash of the result arrays
        hasher = hashlib.sha256()
        for key in sorted(arrays):
            hasher.update(arrays[key].tobytes())
        if salt is not None:
            hasher.update(b"\x00salt\x00" + salt.encode())
        entry_id = hasher.hexdigest()[:12]

        # Write the npz
        entry_path = self.path / f"{entry_id}.npz"
        if not entry_path.exists():
            # Pack metadata and sequence as JSON bytes inside the npz
            meta_bytes = json.dumps(meta, indent=2).encode()
            arrays["__meta__"] = np.frombuffer(meta_bytes, dtype=np.uint8)
            if seq_wire is not None:
                seq_bytes = json.dumps(seq_wire).encode()
                arrays["__sequence__"] = np.frombuffer(seq_bytes, dtype=np.uint8)
            if fit_wire is not None:
                fit_bytes = json.dumps(fit_wire, indent=2).encode()
                arrays["__fit__"] = np.frombuffer(fit_bytes, dtype=np.uint8)
            np.savez_compressed(entry_path, **arrays)

        # Append to manifest
        timestamp = datetime.now(timezone.utc).isoformat()
        manifest_entry = {
            "id": entry_id,
            "experiment": experiment,
            "timestamp": timestamp,
            "task_id": task_id,
            "has_shots": result.shots is not None,
            "has_sequence": seq is not None,
        }
        if fit_wire and isinstance(fit_wire, dict):
            derived = fit_wire.get("derived")
            if derived:
                manifest_entry["derived"] = derived
        manifest_path = self.path / "manifest.jsonl"
        with open(manifest_path, "a") as f:
            f.write(json.dumps(manifest_entry) + "\n")

        return entry_id

    # -- load ------------------------------------------------------------------

    def load(self, entry_id: str) -> dict:
        """Load an entry by ID. Returns ``{result, sequence, meta, fit}``."""
        entry_path = self.path / f"{entry_id}.npz"
        if not entry_path.exists():
            raise KeyError(f"datachest entry {entry_id!r} not found at {entry_path}")

        data = np.load(entry_path, allow_pickle=False)

        # Extract metadata
        meta = {}
        if "__meta__" in data:
            meta = json.loads(bytes(data["__meta__"]))

        # Extract sequence
        sequence = None
        if "__sequence__" in data:
            sequence = json.loads(bytes(data["__sequence__"]))

        # Extract fit
        fit = None
        if "__fit__" in data:
            fit = json.loads(bytes(data["__fit__"]))

        # Rebuild result
        heralds = {}
        herald_lanes = []
        for key in data.files:
            if key.startswith("herald_"):
                lane = key[len("herald_"):]
                heralds[lane] = data[key]
                herald_lanes.append(lane)

        result = MeasurementResult(
            xpts=data["xpts"] if "xpts" in data else None,
            avgi=data["avgi"],
            avgq=data["avgq"],
            shots=data["shots"] if "shots" in data else None,
            heralds=heralds,
            herald_lanes=tuple(herald_lanes),
            meta=meta,
        )

        return {
            "result": result,
            "sequence": sequence,
            "meta": meta,
            "fit": fit,
        }

    # -- list ------------------------------------------------------------------

    def list_entries(self, *, experiment: str | None = None, limit: int = 100) -> list[dict]:
        """List manifest entries, newest first."""
        manifest_path = self.path / "manifest.jsonl"
        if not manifest_path.exists():
            return []
        entries = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if experiment is not None and entry.get("experiment") != experiment:
                    continue
                entries.append(entry)
        # Newest first
        entries.reverse()
        return entries[:limit]
