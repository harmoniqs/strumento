# The calibration store — operating a device over time

A `Device` is a snapshot: the calibration values true of one fridge *right now*. Real operation
is a *timeline* — the device drifts, you re-measure, values change — and a QEC or fleet workload
needs that timeline to be versioned, reproducible, and auditable. That's what `core/store.py`
and `core/config.py` add: immutable, content-addressed calibration versions with a movable
pointer.

## The canonical config

Every device round-trips through a `DeviceConfig` — the serializable source form (pack,
topology, wiring, calibration, couplings, aliases, soccfg). It is what the builder emits, what a
YAML instance parses to, and what a live device re-serializes to:

```python
from strumento import Device
dev = Device.load("devices/multimode_demo/device.yaml")

cfg = dev.to_config()          # the canonical DeviceConfig
cfg.content_id()               # 'cfg-<sha256>' — a stable, calibration-sensitive id
Device.from_config(cfg)        # composes an identical device back
```

The id is the SHA-256 of the config's canonical JSON: an id resolves **forever** to a
byte-identical device, so any historical calibration is permanently reproducible. Provenance
(timestamps, fit labels) is *never* hashed into the config, so identical calibration always
hashes identically.

## Write-back — the seam that closes the loop

A calibration run measures a value; write-back records it as a *new* version (never a mutation):

```python
from strumento import apply_updates
new_cfg, diff = apply_updates(cfg, {"qubit.f_ge": 4001.2,
                                    "qubit.pulses.pi_ge.gain": 8300})
diff["qubit.f_ge"]             # (4000.0, 4001.2)  — old -> new, auditable
```

Updates are **explicit** dotted paths into the calibration tree — strumento never guesses which
field a fit result belongs to, so the write-back is unit-safe (a gain field takes whatever unit
it stores, D23) and auditable.

## The store

```python
from strumento import FileConfigStore, record_calibration

store = FileConfigStore("~/.strumento/multimode")
store.put(cfg); store.set_ref("main", cfg.content_id())

# each re-calibration advances `main` and logs provenance
cid, cfg2 = record_calibration(
    store, "main", cfg,
    {"qubit.f_ge": 4001.2},
    source="amplitude_rabi fit", timestamp="2026-07-23T00:00:00Z",
)

store.resolve_ref("main")      # -> cid (the current version)
store.log("main")              # -> [Provenance, ...] the full drift history, parent-linked
store.get(cfg.content_id())    # any past version, byte-identical
```

`FileConfigStore` is dependency-free — write-once JSON objects, ref files, an append-only log —
and it's one implementation of the `ConfigStore` protocol; a lab or a fleet can supply another
backend (a database, a shared object store) behind the same interface. **Immutability is the
load-bearing property**, not the backend (spec D9).

## Where this is going

This is the accumulator the codesign loop (D26) and closed-loop calibration build on: every
characterization run updates the measured device model *for free*, and the drift history is what
fleet-level modeling and reproducible QEC operation consume. A measured-vs-design diff report
(`DeviceModelReport`) is the natural next layer on top of the store.
