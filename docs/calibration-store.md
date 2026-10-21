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

`record_calibration` **validates before it writes**. The candidate config is composed into a
device first (the `validate=` argument; by default `Device.from_config`), so a typo'd key, a
value of the wrong type, a NaN or infinity, a negative time, or a `readout.trig_offset` above
10 µs raises a `ValidationError` naming
the field — and the store is left exactly as it was: no new object, no provenance line, no ref
move. Validation never changes the bytes that are hashed, so a valid write-back's content id is
the same as before. A writer that deliberately stores values outside a pack's schema passes
`validate=unvalidated` (from `strumento.core.store`) — the bring-up engine does, since its
campaigns bank provides paths the pack schema need not declare.

`FileConfigStore` is dependency-free — write-once JSON objects, ref files, an append-only log —
and it's one implementation of the `ConfigStore` protocol; a lab or a fleet can supply another
backend (a database, a shared object store) behind the same interface. **Immutability is the
load-bearing property**, not the backend (D9).

## Simulated writes

A lab session (`strumento.mcp.LabSession`, the MCP server's session) can run on a simulator or
the mock, and a simulated fit must never pass for a measured one. Three rules keep them apart.

**Every write says where it came from.** Each provenance line the session writes — the genesis
seed of a ref and every `record_calibration` — carries these fields in its `extra`:

| Field | Value |
|---|---|
| `soc_kind` | the soc the session opened: `"simulator"` or `"mock"` |
| `simulated` | `true` for both kinds (neither measures a board) |
| `truth_hash` | simulator only: the SHA-256 hex digest of the canonical JSON of the session's `sim=` ground truth |
| `fit` | only for a write labelled with a run (below): that run's `derived` values |
| `data` | only for a write labelled with a run: where that run's data was saved, `{"saved": true, "chest": "<dir>", "entry": "<id>"}`, or `{"saved": false, "reason": "…"}` (see [The run's data](#the-runs-data)) |

The canonical JSON is `json.dumps(truth, sort_keys=True, separators=(",", ":"),
ensure_ascii=True)` — keys sorted at every depth, no whitespace, numbers as Python's `json`
writes them, so `9350` and `9350.0` are different truths. A simulator with no truth hashes `{}`.
`strumento.mcp.truth_hash` computes it. The fields live in `extra`, not beside `source`, because
readers build `Provenance(**line)`: a log line written before these fields existed reads exactly
as before (its `extra` is whatever it was, `{}` when absent). Provenance is never hashed, so a
content id is the same whoever wrote it. `calibration_history` and `strumento store log --json`
show `extra` on every entry. With a task record open, the record's `calibration` event carries
`soc_kind`, `simulated` and `truth_hash` too, and the record's `target` is `"sim"`.

**A simulated session writes only to a simulation store.** A store is *fresh* when it holds no
file under `objects/`, `refs/` or `log/` — an absent root, or the empty directories a
`FileConfigStore(root)` creates. A simulator or mock session that opens a fresh store marks it,
by writing `simulation-store.json` at the store root, before it seeds the ref. Any other store is
*existing*, and a simulated session refuses to write into an existing store that has no mark:

- `load_device` is refused when it would seed a ref that does not yet resolve there, and
- `record_calibration` is refused.

The refusal is `ok: false` (a `SessionError` from Python), names the store, and gives the two ways
out: open a fresh store, or re-open this one with `load_device(..., allow_sim_writes=True)`.
Nothing is written — no mark, no object, no line, no ref move — and a refused `load_device`
leaves the previous session as it was. Reading is never refused: a simulated session may open an
existing store whose ref resolves, and run against that belief. With `allow_sim_writes=True` the
writes proceed, still stamped `simulated: true`, and the store is not marked.

The refusal is the session's policy, not the store's. `strumento.core.store.record_calibration`
— the library call a human or the bring-up engine makes — writes into any store.

**A write is labelled with what it is.** `record_calibration(updates)` is recorded with
`source: "manual"` and no fit attached. `record_calibration(updates, outcome=<index>)`, where
`<index>` is the `index` a `run_experiment` returned in the same session, is recorded with
`source: "<experiment> fit"` and that run's derived values in `extra.fit`. An index the session
does not remember (never returned, older than the last 32 runs, or not an integer) or a run with
no fit is refused, with nothing written.

## The run's data

Each `run_experiment` in a lab session saves the run as one entry of a
`strumento.core.datachest.FileDataChest`: the `MeasurementResult` arrays (`xpts`, `avgi`,
`avgq`, `shots` when the run is shot-resolved, heralds), its `meta` plus `soc_kind`, `simulated`,
`truth_hash` (simulator) and `mcp_run` (experiment, index, params, run keys, device, ref and the
config content id the run used), the sequence that was played, and the fit summary. The reply
names the entry:

```jsonc
"datachest": {"saved": true, "chest": "/lab/store-data", "entry": "3f9c0a1b2d4e"}
```

The chest is `load_device(..., chest=<dir>)`. Without `chest=`, a session opened with
`store=<root>` keeps its data in the sibling directory `<root>-data` — beside the store, never
inside it, so a chest never makes a fresh store look existing and never sits among its objects
(a `chest=` inside the store's `objects/`, `refs/` or `log/` is refused). With neither, nothing
is saved, and `load_device`, `device_info` and every run say so. `load_device` only resolves the
path; the first save creates the directory, so a refused `load_device` leaves no chest behind.
A save that fails (an unwritable directory, a full disk) does not cost the run: the reply keeps
its fit and says `{"saved": false, "reason": "…"}`.

A write labelled with `outcome=<index>` cites that run's entry in its provenance's `extra.data`,
so the measurement behind a calibration value can be reloaded and refitted:

```python
from strumento.core.datachest import FileDataChest

cited = lab.calibration_history()["history"][-1]["extra"]["data"]
result = FileDataChest(cited["chest"]).load(cited["entry"])["result"]   # a MeasurementResult
```

With a task record open, each run also writes a small pointer file under the record's
`artifacts/` (`{chest, entry, experiment, index}`) and records it as an `artifact` event of
kind `datachest`; the record's `calibration` event carries the same `data` as the provenance
line. Each run's entry is its own: the session salts the entry id with a per-run identity, so
two runs whose arrays are byte-identical (the mock's zeros, for example) still get separate
entries, each carrying its own run's metadata, sequence and fit. Outside a session,
`FileDataChest.save` addresses an entry by its arrays alone unless it is given a `salt`.

## Where this is going

This is the accumulator the codesign loop (D26) and closed-loop calibration build on: every
characterization run updates the measured device model *for free*, and the drift history is what
fleet-level modeling and reproducible QEC operation consume. A measured-vs-design diff report
(`DeviceModelReport`) is the natural next layer on top of the store.
