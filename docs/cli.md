# The `strumento` CLI

A small, agent-forward command-line surface over the pure layers. Every command prints
human-readable text by default and machine-readable JSON with `--json`; errors come back as
`{"ok": false, "error": ...}` with a nonzero exit code, never a traceback.

```
strumento packs                          # installed modality packs (entry points)
strumento validate <device.yaml>         # load + validate an instance
strumento info <device.yaml>             # topology, namespaces, wiring table
strumento soccfg <dump_cfg.json>         # board, tProc v1-vs-v2 verdict, generators
strumento fit <kind> <data.csv>          # rabi | t1 | t2 | parity | lorentzian | circle
strumento compile <device.yaml>          # lower a canned Rabi; list the program calls
strumento check <device.yaml>            # validate wiring against the device's soccfg snapshot
strumento schema <contract>              # device | fit | result | operator-model (JSON Schema)
strumento schema calibration --pack <p>  # a pack's calibration schema, units in descriptions
strumento task create|status|tail|list   # long-running experiment records
strumento store current|log|diff         # the calibration store, read-only
```

Examples:

```console
$ strumento soccfg devices/_fixtures/soccfg_v2_cqed.json
ZCU216 · qick 0.2.367 · tproc qick_processor · 6 gen(s), 1 readout(s)
tProc v2 — real AveragerProgramV2 assembly works offline

$ strumento fit t1 t1_trace.csv --json
{ "ok": true, "kind": "t1", "success": true, "derived": { "T1": 41.98, ... } }
```

`fit` reads a headered CSV: two columns (`x,y`) for the scalar fits, three (`x,i,q`) for the
resonator `circle` fit. `compile` needs no `qick` for the lowering listing; with `qick`
installed it also reports the DAC-grid realized values of the calibrated π pulse.

## `check` — wiring against the soccfg snapshot, as an exit code

```console
$ strumento check devices/multimode_demo/device.yaml; echo $?
PASS: devices/multimode_demo/device.yaml validates against its soccfg
  6 gen(s), 1 readout(s)
0
```

Warnings are reported but do not fail the check. The single-channel testbench instance passes
with one:

```console
$ strumento check devices/loopback_demo/device.yaml; echo $?
PASS: devices/loopback_demo/device.yaml validates against its soccfg
  WARN: tproc f_time=430.08 differs from gen f_fabric={599.04} — may cause timing-conflict warnings on back-to-back pulses
  1 gen(s), 1 readout(s)
0
```

`check` cross-references the device YAML against the `dump_cfg()` snapshot it attaches
(`soccfg_snapshot:`): every wired `dac_ch` must be a generator, every `adc_ch` a readout, and the
tProc must be v2. Its verdict is the exit code, so scripts and CI can gate on it:

| result | `ok` (`--json`) | exit |
|---|---|---|
| PASS, no warnings | `true` | 0 |
| PASS with warnings only (e.g. tProc `f_time` ≠ gen `f_fabric`, small `pmem`) | `true` | 0 |
| FAIL — any `ERROR` (`issues` non-empty) | `false` | 1 |
| FAIL — no soccfg snapshot attached | `false` | 1 |

`--json` returns `{ok, issues, warnings, n_gens, n_readouts}`; the no-snapshot case returns
`{ok: false, error, issues: [error], warnings: []}`. `ok` always agrees with the exit code.

## `schema` — the typed contracts, machine-readable (D35)

```console
$ strumento schema device > device.schema.json   # text mode prints the bare schema document
$ strumento schema result --json                 # wrapped: {"ok": true, "contract": …, "schema": …}
$ strumento schema calibration --pack cqed --json # wrapped, plus "pack": "cqed"
```

| contract | describes |
|---|---|
| `device` | an instance's source form — the YAML on disk *and* `DeviceConfig.to_dict()` (`soccfg_snapshot` may be the relative path or the resolved snapshot) |
| `fit` | `FitResult` — every fitter's and analyzer's return (`params` / `stderr` / `derived` / `success`); `derived` values are any JSON value (D38) |
| `result` | `MeasurementResult.to_wire()` — arrays as lists, herald lanes as bool lists, JSON-safe `meta` |
| `operator-model` | `OperatorModel.to_dict()` — the framework-neutral Hamiltonian export |
| `calibration` (needs `--pack <name>`) | the pack's `CALIBRATION` model (`cqed` `Calibration`, `spin` `SpinCalibration`, `tweezer` `TweezerCalibration`); each field's description states its unit |

Each schema is emitted from the same Pydantic definition that *validates* the payload
(`strumento.core.schemas.validate`), so the published shape cannot drift from the objects it
describes; every shipped `devices/*/device.yaml` is asserted against the `device` schema in CI, and
every registered `cqed` and `spin` experiment's simulated fit against the `fit` schema.
An unknown contract name comes back as data, listing the valid ones, with exit code 1.

`calibration` is **pack-scoped**: the `device` contract types `calibration` as an open mapping,
because its content belongs to the pack, so the pack's own shape is published separately. Core
resolves the pack by name through the `strumento.packs` entry points and emits its `CALIBRATION`
model. `calibration` without `--pack`, or with a pack that is not installed, comes back as data
listing the installed packs (exit code 1); `--pack` on any other contract is refused the same way.

## `task` and `store` — long-running experiments and their calibration provenance

```console
$ strumento task create --title "T1 monitor" --device multimode_demo --script scripts/t1.py \
      --language python --target sim --gates fit-recovery --root ./experiments --json
{ "ok": true, "id": "2026-07-25-t1-monitor-4f1c", "dir": "./experiments/2026-07-25-t1-monitor-4f1c" }

$ strumento task status ./experiments/2026-07-25-t1-monitor-4f1c
2026-07-25-t1-monitor-4f1c: running · pid 81234 (alive=True) · 37 event(s) · 1.2s since liveness (stale after 45s)
```

| verb | returns |
|---|---|
| `task create` | a fresh task directory + manifest (`pid = 0`, i.e. `queued`) → `{id, dir}` |
| `task status <dir>` | derived state **plus every staleness input behind it** (pid liveness, host match, age, the manifest's own interval) |
| `task tail <dir> -n 50` | the last N `progress.jsonl` events, corrupt lines skipped |
| `task list <root>` | every record under a root, newest first (a missing root is an empty list) |
| `store current <root> --ref main` | the resolved pointer: ref, content id, timestamp |
| `store log <root> --ref main` | version history — content id, timestamp, `source`, value updates, `extra` |
| `store diff <root> <id-a> <id-b>` | per-value calibration diff, keyed by `apply_updates` dotted paths |

The record format itself — the directory layout, the manifest fields, the event families, the state
derivation, and the gate trust boundary — is documented in
[docs/task-record.md](task-record.md). It is a public contract implemented in one pure-stdlib file,
so these verbs are equally usable from CI, from a bare interpreter, or from a supervisor written in
another language. An **absent** calibration store is an empty view rather than an error (a deviceless
or never-calibrated device legitimately has no history), and the store verbs never create one.

## Why a CLI *and* an MCP server (D33)

Strumento has **two agent doors**, and the split between them is not CLI vs MCP — it is
**stateless vs stateful**.

| | `strumento` (CLI) | `strumento-mcp` (`strumento/mcp.py`) |
|---|---|---|
| shape | one-shot verbs: load, answer, exit | a long-lived **session** |
| holds | nothing between calls | `{device, soc, store, ref, recent outcomes}` |
| answers | "what is this instance / this contract / this fit?" | "measure this device, then measure it again knowing what you learned" |
| reached by | `exec` — native everywhere, including headless CI | the MCP protocol over stdio (`[mcp]` extra) |
| autonomy | n/a | full on simulator/mock; a hardware-backed session inherits the propose-vs-execute gate |

A calibration loop is *measure → believe → measure again*, and the "believe" step has to survive
between calls. A process-per-verb CLI is structurally incapable of that: it would have to re-load the
device, re-open the store and re-arm the soc on every step, and there would be nowhere for "the fit I
just took" to live. That is the whole reason the second door exists — not a nicer syntax.

The reverse also holds: nothing about the session is a *better* CLI. For a one-shot question a
subprocess is simpler, cheaper, and already installed.

Both doors are thin over the **same importable functions** — `strumento.cli.device_summary`,
`flatten_calibration`, `provenance_entries`, `strumento.core.schemas.schema_for` / `validate`,
`strumento.core.store.record_calibration` — so `strumento info` and the session's `device_info` can
never drift into two different answers. `schema` and `validate` exist on *both* doors deliberately:
a shell-less host still needs the contracts.

### The session's eight tools

```
load_device(path, soc="simulator"|"mock", store=None, ref="main", sim=None, task=None,
            allow_sim_writes=False, chest=None)
device_info()                      # topology, wiring, namespaces + the live calibration
list_experiments()                 # from the loaded pack's EXPERIMENTS registry; each entry
                                   # says whether it needs per-shot data (needs_shots)
run_experiment(name, params=None)  # params = constructor kwargs (+ reps); typed fit back,
                                   # the run saved to a data chest entry
calibration_history()              # the tracked ref's provenance chain
record_calibration(updates, outcome=None)  # write back, advance the ref, RELOAD the device
schema(what, pack=None)            # mirror of `strumento schema`; pack defaults to the device's
validate(yaml)                     # mirror of `strumento validate`
```

Eight details carry weight:

- **`record_calibration` reloads the device.** It snapshots a new immutable version, advances the
  ref, *and recomposes the session's device from that config* — so the next `run_experiment` plays
  pulses built from the new belief. Without the reload a session would keep measuring with the value
  it just replaced and no loop would ever close.
- **An agent cannot raise its own gain ceiling.** An update that touches a path the pack declares
  in `CALIBRATION_CEILINGS` (for cqed: `qubit.max_drive_gain`, `readout.max_readout_gain`,
  `modes.<name>.max_displace_gain`) is refused whole, before anything is written, with
  `ok: false` naming the path. Ceilings are set by a human, in the device YAML or through
  `strumento.core.store.record_calibration`.
- **Simulated writes say so, and stay out of real stores.** Every provenance line the session
  writes records `soc_kind`, `simulated` and, for the simulator, `truth_hash` in its `extra`.
  A simulator or mock session marks a fresh store it creates and refuses to write into an
  existing store without the mark unless opened with `allow_sim_writes=True`. A write is labelled
  `manual` unless `outcome=<index>` names the run whose fit it records. See
  [calibration-store.md](calibration-store.md#simulated-writes).
- **A run keeps its data.** Each `run_experiment` saves the run (arrays including `shots`,
  metadata with `soc_kind` / `simulated` / `truth_hash`, the played sequence, the fit) as one
  `FileDataChest` entry and returns `datachest: {saved, chest, entry}`. The chest is
  `load_device(..., chest=<dir>)`, or by default `<store>-data` beside the store; with neither,
  or if the save fails, the run still returns its fit and `datachest` says `saved: false` and why.
  A write labelled with `outcome=<index>` cites the entry in its provenance (`extra.data`). See
  [calibration-store.md](calibration-store.md#the-runs-data).
- **`sim=` is the simulator's ground truth** (a partial calibration dict, or a path to one), held
  separately from the device's belief. Omitted, the simulator tells the truth and every fit recovers
  what the device already believes — which is the boring case. Perturb a key and the session becomes
  a real closed-loop demo.
- **The fit arrives as JSON, values intact.** `run_experiment` returns
  `fit: {success, params, stderr, derived}`. `params` and `stderr` are numbers; `derived` also
  carries the strings, booleans, lists and objects an analyzer reports (a protocol name, a
  confusion matrix, a refusal's reason), and they arrive as given. A NaN or infinite number, at
  any depth, arrives as `null` — NaN is not JSON, and an unestimated stderr is the common case.
- **Some experiments need per-shot data.** Each `list_experiments` entry carries `needs_shots`.
  On a soc that returns averages only (`soc="mock"`, or a board until it has a shot path) such an
  experiment refuses rather than guessing: `fit.success` is `false` and `fit.derived.refused`
  says it needs per-shot data. The simulator provides shots, so they run normally there. See
  [cqed-cookbook.md](cqed-cookbook.md#experiments-that-need-per-shot-data).
- **Errors are data.** Every tool returns `{"ok": true, …}` or `{"ok": false, "error": "Type: msg"}`.
  A traceback never crosses the protocol; a store-less session asked for history says
  *"no store opened for this session"* and tells you how to open one.

### The permanent non-goal: no "what next?" tool

There is deliberately **no** next-experiment, recommendation, or planning tool, and one must never be
added. Choosing what to measure next is the **calibration-strategy layer's** job, above this one;
strumento is the strategy-agnostic substrate underneath it (D18). A tool here that answered "what
next" would quietly make strumento a strategy and every consumer would inherit it. The tool list is
mechanism only.

A related boundary: an automated calibration harness should drive strumento through the Python API
and this CLI **directly**. An LLM looping over `run_experiment` would reintroduce LLM-as-state-machine — the
MCP session is the *interactive / external* agent surface, not the inner loop of an automated one.

### Running it

```console
$ pip install -e ".[mcp]"     # the SDK is optional; `import strumento.mcp` works without it
$ strumento-mcp               # serves one session over stdio
```

The session object is importable too — `LabSession().load_device(...)` — so a host that already
speaks Python needs no protocol at all. The capstone test (`tests/test_mcp.py`) drives the real
protocol over the SDK's in-memory transport: open a session on `devices/loopback_demo` against a
simulator holding a perturbed truth, run a Rabi, write the fit back, run it again, and watch the
belief-vs-truth error collapse.
