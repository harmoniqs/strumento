# The task record — a durable contract for long-running experiments

An experiment that takes hours is not a function call. A T1 monitor, an overnight sweep campaign, a
50-iteration calibration loop: each one outlives the terminal that launched it, the editor session
that watched it, and any supervisor process that spawned it. So its truth cannot live in memory. It
lives in a **task directory**, and this document is that directory's contract.

`strumento/core/record.py` is the reference implementation — **one file, pure standard library**, no
`qick`, no `numpy`, nothing from the rest of strumento. That is deliberate: the format is public, so
a plain script from any QICK user should be able to emit it, and a reader in any language should be
able to consume it. Copy the file into a bare interpreter and it works (there is a test that proves
exactly this).

```
experiments/2026-07-23-qilc-cz-cal-a3f2/
  task.toml          manifest — written once; pid / host / config_content_id update in place
  progress.jsonl     append-only event stream, one JSON object per line, flushed per line
  artifacts/         whatever the script saves (plots, fit CSVs, pulses)
  result.toml        terminal state — written exactly once, atomically (tmp+rename)
```

`task-id` is `YYYY-MM-DD-<slug>-<4 hex>`, and **the id is always the directory's basename** — one
identity instead of two that can disagree. Default root: `<workspace>/experiments/`. Minting an id
claims its directory with an atomic `mkdir`, so two writers can never own one id; if the directory
already exists, a fresh suffix is drawn and the claim retried (a bounded number of times).

## Design rules (the reasons, not just the shapes)

**The directory is the truth; everything else is a view.** Derived state (`queued`, `running`,
`stale`) is recomputed at read time from the record plus process liveness, and stored nowhere.
`result.toml`'s *existence* is the only durable terminal marker there is.

**No TOML nulls, anywhere.** Absent strings are `""`, absent integers `0`. Readers include a
python3.9 + `tomli` parser, where a null is a parse error; keeping the convention project-wide makes
every reader trivially compatible. The manifest is also written in the lowest-common-denominator TOML
subset — no inline tables, no dotted keys, no multiline strings.

**Crash safety by construction.** `progress.jsonl` is opened append-only and flushed after every
line, so `kill -9` loses at most the line in flight; a reader skips an incomplete line and keeps the
rest (JSONL is per-line recoverable). `task.toml` and `result.toml` are written tmp+fsync+rename, so
a partial document is not a state a reader can observe — which is why an unparseable `result.toml`
is treated as *corruption*, not as a race.

**The event file is closed by whoever opened the record.** It opens lazily on the first event and
stays open between events; a terminal marker closes it, and so do `close()` and leaving a `with`
block. A record that never reaches a terminal state — a parked campaign, a bring-up step's child —
is closed by its owner (the bring-up runner closes every child it mints and lets go of the campaign
file at the end of every advance). Closing changes nothing on disk.

**One writer per field.** `pid` is the field two processes might both want. A **supervised** run (the
supervisor set `STRUMENTO_TASK_DIR`) never touches it: the supervisor knows the pid before the script
boots and writes it synchronously at spawn. A **standalone** run (`TaskRecord.create`) stamps its
own. No path has two writers.

**Forward compatibility is part of the contract.** Readers must ignore `ev` values they do not
know and manifest fields they do not know — **including unknown `kind` values** (a future axis
value reads, derives, and lists without failing). Writers validate their own input strictly;
readers tolerate.

## `task.toml`

```toml
schema = 1
id = "2026-07-23-qilc-cz-cal-a3f2"
title = "QILC CZ calibration"
kind = "experiment"            # reserved registry axis: "experiment" | "bringup"
                              # ("bringup" = a bring-up campaign's ledger; the runner layer
                              #  reads its events — the record itself is the same shape)
task_type = "experiment"       # telemetry key; a supervisor suffixes -sim/-hw from `target`
target = "sim"                 # "sim" | "hw"
gates = []                     # gate names this task claims; one telemetry row per name
script = "scripts/qilc_cz.jl"  # relative to workdir
workdir = "/abs/path"
language = "julia"             # "python" | "julia"
device = "multimode_demo"      # device id = devices/<id>/ dir name; "" if deviceless
created = "2026-07-23T03:31:00.000Z"
pid = 0                        # 0 = not started (⇒ derives `queued`)
host = "some-workstation"
heartbeat_interval_s = 15      # this task's honest liveness cadence
config_content_id = ""         # the DeviceConfig version the task ran against

[authored_by]                  # agent provenance; empty strings for a human/standalone run
profile = ""
model = ""                     # provider/model-id of the authoring session
variant = ""
session_id = ""
```

| field | mutability | why it is here |
|---|---|---|
| `pid`, `host`, `config_content_id` | **updatable in place** | the spawner writes the first two; a script stamps the third once, when it learns which calibration it ran against |
| everything else | immutable after creation | a supervisor queues, lists, and routes on these *before* any script runs, so they have to travel the whole submit chain (tool input → submit payload → `task create` flags) rather than being discovered later |

`heartbeat_interval_s` is clamped to ≥ 1 (0 would derive `stale` immediately). Staleness is always
judged against **this task's** interval, never a global constant — see the GIL note below for why
that matters.

`authored_by` being empty is load-bearing downstream, not just tidy: a supervisor forwards telemetry
rows only when `model` is non-empty, so human outcomes never become routing signal.

## `progress.jsonl`

One JSON object per line; `t` is UTC ISO-8601 with milliseconds and a `Z` suffix.

```jsonc
{"t": "…", "ev": "progress", "step": 12, "of": 50, "label": "iter 12",
 "metrics": {"infidelity": 3.2e-3}}
{"t": "…", "ev": "artifact", "path": "artifacts/fit_012.json", "kind": "fit"}
{"t": "…", "ev": "heartbeat"}
{"t": "…", "ev": "calibration", "content_id": "cfg-cd34…",   // the store pointer moved
 "soc_kind": "simulator", "simulated": true, "truth_hash": "9c1f…"}   // optional: its origin
{"t": "…", "ev": "gate", "name": "fit-recovery", "pass": true,
 "params": {"metric": "pi_gain_rel_err", "value": 0.004, "tolerance": 0.02}}
// campaign events — a kind="bringup" record's ledger (generic mechanism; see below)
{"t": "…", "ev": "claim", "device": "multimode_demo", "ref": "main"}
{"t": "…", "ev": "step_open", "step": "resonator-find", "attempt": 1,
 "child_path": "steps/2026-08-29-res-find-1-c1d2"}
{"t": "…", "ev": "step_close", "step": "resonator-find", "attempt": 1, "outcome": "pass"}
{"t": "…", "ev": "park", "kind": "propose", "reason": "outside cleared envelopes"}
{"t": "…", "ev": "approve", "actor": "operator", "channel": "operator-cli",
 "proposal_ref": "prop-3f2c…"}
{"t": "…", "ev": "resume", "actor": "operator", "directive": "rerun with envelope cleared"}
{"t": "…", "ev": "poison", "paths": ["readout.region"],
 "reason": "flux bias moved outside the cleared region"}
```

- **The names above are exported as `record.EVENT_KINDS`**, for a reader that wants the documented
  vocabulary as data. It is a list of what is documented, not a check: readers still ignore any `ev`
  they do not know.
- **`metrics` is an open string → number map.** A T1 monitor reports `t1_us`, a calibration loop
  `infidelity`, and a reader plots whatever series it finds — no schema change per experiment type.
  Non-numeric metrics are refused at the boundary, so a reader can always plot the map.
- **`of` may be absent** for an open-ended monitor; a UI then shows a counter, not a bar.
- **`artifact.path` is task-dir-relative and must resolve inside the task dir.** Both the writer and
  the readers reject `..` escapes — a recorded path is data in a file anyone could have written, so
  the containment check belongs at every boundary that turns it into a filesystem path.
- **`gate` carries a verdict; it does not compute one.** See the trust boundary below.

### Campaign events (`kind="bringup"` records)

Seven emitters — `claim`, `step_open`, `step_close`, `park`, `approve`, `resume`, `poison` — are
**generic substrate mechanism**: each carries an event name plus a payload of non-empty strings
(attempt numbers are 1-based integers; `poison`'s `paths` is a non-empty list of them), validates structurally at append exactly like the other kinds, and
means something only to the orchestration layer that *reads* the stream. This substrate never
interprets them: what a `pass` outcome is, which park kinds core knows (`propose`, `needs_human`),
what an approve clears — those are the runner layer's, and an unknown `park` kind is recorded
verbatim (on derivation it holds a campaign as parked-unresolved; opacity there means *parked*,
never released).

`poison` marks calibration paths as forced stale (excursion poisoning). Its payload is `paths`, a
non-empty list of non-empty calibration paths, and `reason`, a non-empty string saying what moved
where. The runner appends one when a region-declared step's moves leave the cleared region in which
a drift-sensitive artifact was acquired; each named path then derives as stale until a newer write
of it lands, and the event's `t` is the poison's clock. A `resume` clears no poison.

Three rules fix the mechanism's boundaries:

- **The child ledger is a record, never a control channel.** Appending an event — any event — never
  triggers execution. `step_open` is the authoritative child mapping (the child directory's naming
  is a cache); a crashed attempt is exactly a `step_open` with no matching `step_close`.
- **Appends validate at the emitter, not at read.** A malformed payload (empty device, attempt 0,
  an approve missing its channel) is a typed error raised before anything reaches
  `progress.jsonl` — rejection leaves no trace in the stream.
- **No agent-reachable surface writes campaign records.** The approvals/parks/resumes ride
  the operator CLI / supervisor; the MCP door has no campaign tool and no file-write path to campaign
  directories — asserted by a source-level guard plus the unchanged 8-tool list.

**The runner's derivation rules** (`strumento.core.bringup`, which reads these events): campaign
position, per-step attempt counts, park state, pending approvals, cleared envelopes, and claims all
derive from the event stream alone — never from directory names — so a fresh process re-attaches
and resumes after any crash. `step_open` is the authority on where the campaign is; a `step_close`
outcome of `pass` follows `on_pass` and resets that step's failure count, anything else counts a
failure and follows the `on_fail` rung matching the count; an open attempt superseded by a later
`step_open` counts as a failure where it crashed, and one left open at end-of-stream counts *and*
routes (the killed-runner resume). A `resume` clears any park; a valid `approve` (one carrying both
actor and channel — a line missing either is ignored at derivation, clearing nothing) clears a
`propose` park and its proposal's envelope. Verdicts route from the sanctioned emitter's *returned*
outcome, never from the child ledger; the campaign's `advance()` takes an ephemeral breakable lock
(`bringup.lock`) that is a mutex, never authority — exclusivity derives from live claims, and a
terminal campaign state (the record's `result.toml`, which the runner writes on reaching
`calibrated`) releases the claim.

## `result.toml`

```toml
schema = 1
state = "done"                # "done" | "failed" | "cancelled"
ended = "2026-07-23T05:10:00.000Z"
summary = "F = 0.9931 after 38 iters"
error_kind = ""               # "" | "script-error" | "transport-error"
error = ""                    # short message; the full traceback belongs in artifacts/stderr.log
```

`error_kind` separates failures by **who fixes them**: `script-error` is scientific (the experiment
is wrong), `transport-error` is operational (the job server is unreachable). That distinction is the
only reason the field exists.

## Derived state

A pure, **total** function of (record, pid-liveness), evaluated in this order — first match wins:

1. `result.toml` exists ⇒ its `state`. Unparseable, or a state outside the enum ⇒ `stale`, with the
   reason surfaced as data.
2. `pid == 0` ⇒ `queued` (never started: a queue is holding it, or it awaits re-enqueue).
3. pid dead ⇒ `stale`. **Skipped when the manifest's `host` is not this host** — a pid number from
   another machine says nothing here, and probing it could hit an unrelated process. A foreign record
   ages into `stale` through rule 4 instead.
4. liveness older than `3 × heartbeat_interval_s` ⇒ `stale`, where liveness :=
   max(`task.toml` mtime, newest event of **any** kind). So a just-spawned task with no events yet is
   `running` until its interval elapses, and a task streaming `progress` whose heartbeat thread died
   never goes stale. Hung-but-alive is exactly what this rule is for, which is why it **precedes**
   `running`.
5. otherwise ⇒ `running`.

Note the guard's honest coverage: rule 3's host check protects against probing *cross-host* records,
not same-host pid recycling after a reboot. A canceller should do a cheap identity check (process
start time ≤ pid-write time) before signalling.

## Python API

```python
from strumento.core.record import TaskRecord

rec = TaskRecord.from_env()                    # supervised dir, or a fresh one standalone
rec.on_cancel(soc.close)                       # SIGTERM → cleanup → result.toml(cancelled)
rec.set_config(store.resolve_ref("main"))      # which calibration this run saw (once)

for i, delay in enumerate(delays):
    fit = T1(dev, delays=delay).go(soc).fit
    rec.progress(i, of=len(delays), t1_us=fit.derived["T1"])

rec.artifact("artifacts/t1.png", kind="figure")
rec.calibration(new_content_id)                # the store pointer moved
rec.done(summary=f"T1 = {fit.derived['T1']:.1f} us")
```

| call | does |
|---|---|
| `TaskRecord.create(dir, title=…, …)` | create a task dir with that basename as its id |
| `TaskRecord.create_under(root, title=…)` | mint the conventional id and create `<root>/<id>/` |
| `TaskRecord.from_env(**create_kw)` | attach to `$STRUMENTO_TASK_DIR`, else create standalone |
| `TaskRecord.open(dir)` | attach as a reader (no heartbeat thread) |
| `rec.progress(step, of=None, label=None, **metrics)` | one loop step |
| `rec.artifact(path, kind=None)` | something saved (escape-checked) |
| `rec.heartbeat()` | explicit liveness |
| `rec.set_config(content_id)` | stamp `config_content_id` once, while empty |
| `rec.calibration(content_id, **fields)` | the store pointer moved; `fields` (JSON values) say where the write came from |
| `rec.gate(name, passed, **params)` | route a **vetted** verdict into the record |
| `rec.claim(device, ref)` | campaign ledger: the persistent claim on (device, ref) |
| `rec.step_open(step, attempt, child_path)` | attempt starting; the authoritative child mapping |
| `rec.step_close(step, attempt, outcome)` | attempt finished, with its open-string outcome |
| `rec.park(kind, reason)` | holding; `kind` is a non-empty open string |
| `rec.approve(actor, channel, proposal_ref)` | privileged approval — operator surfaces only |
| `rec.resume(actor, directive)` | a parked campaign resumes |
| `rec.done(summary)` / `rec.fail(error, kind)` | the terminal marker, written once |
| `rec.on_cancel(cleanup)` | install the SIGTERM → cleanup → `cancelled` path |
| `rec.close()` / `with rec:` | stop the heartbeat and close the event file — idempotent, and not final: a later event reopens the file and appends |
| `rec.release()` | close the event file only, leaving the heartbeat running |

Readers: `read_manifest`, `read_result`, `read_events`, `derive_state`, `list_tasks`,
`artifact_path` (escape-checked resolution).

## Experiments emit their own records

`Experiment.go(record=…)` (and `Runner.run(record=…)`, for the escape-hatch path that drives a Runner
directly) wires the whole thing up, so a standard experiment needs no plumbing of its own:

| event | when |
|---|---|
| `progress` step 1 | the acquisition completed — `points`, `reps`, and the experiment's name. The acquire is one call into the board, so that is the only honest granularity at this layer; a per-point series would be invented |
| `progress` step 2 | the fit landed, carrying **every derived quantity as a metric** — the series a monitor wants plotted |
| `artifact` | a figure that was actually written into the record's `artifacts/` (duck-typed on `savefig`; a figure that cannot save itself yields no event, because an `artifact` path must resolve to a real file) |
| `gate` `fit` | always, when analysis ran: did the fitter converge? |
| `gate` `fit-recovery` | additionally, when the soc carries an armed ground truth (a simulator) |

`record` is an **explicit keyword parameter** on both methods — a misspelled `recrod=` must not
silently mean "no record", which would make a healthy task look dead. It therefore lands in
`**acquire_kw` and raises at `acquire`, instead of vanishing.

A lab session (`strumento.mcp.LabSession`) opened with `task=` adds one event per run on top of
these: the run's data is saved to a data chest outside the task directory, so the session writes a
pointer file `artifacts/datachest-<entry>-<n>.json` (`{chest, entry, experiment, index}`) and
records it as an `artifact` event with `kind: "datachest"`. A `calibration` event for a write
labelled with that run carries the same citation as `data` (see
[calibration-store.md](calibration-store.md#the-runs-data)).

`fit-recovery` pairs a fit's `derived` quantities with the truth **by name** (`derived["T1"]` against
`truth["T1"]`) — a naming convention, not physics, so `core` applies it without interpreting a key;
an experiment's `recovery_map()` adds pairs a name match cannot see. Exactly **one** event is emitted
per run, `pass` = every compared quantity inside `recovery_tolerance`, with the worst offender in
`params` and the full per-quantity map alongside: a reader takes the *last* gate event per name, so
several events under one name would silently discard all but one verdict. A truth that shares no
quantity with the fit emits nothing at all — silence beats a vacuous pass. On hardware there is no
truth to recover, so the gate simply does not appear; *physics agreement* is a different gate with a
different meaning.

## The gate trust boundary

A `gate` event is a **verdict computed by vetted code**. The reason is not ceremony: gate outcomes
feed a learned routing table that decides which model gets which work, and a table fed self-graded
homework is worse than no table at all. So `Experiment.go` computes the verdicts and a task script's
job is to *route* them, not to author them.

Be precise about how strong that is:

- **What holds structurally — for a *shipped* experiment.** The comparison is between a pack fitter's
  output and the armed truth. A script that runs `AmplitudeRabi(...).go(soc, record=rec)` cannot reach
  into it: it chooses whether the comparison happens, not what it says.
- **What does not.** Two ways out, both real. `rec.gate` is a public method, so a script can write
  `rec.gate("fit-recovery", True)` with no fit behind it. And `analyze()` belongs to the subclass, so
  a script that defines its **own** `Experiment` can hand the emitter a fabricated `FitResult` and get
  a genuine-looking pass. "Vetted code computes the verdict" is true of the shipped experiments, not
  of arbitrary agent-authored subclasses. This is a **contract**, not a sandbox.
- **Where the teeth are.** Supervisor-side: verdicts are re-derived from the record before being
  forwarded, only gates the manifest declared (`gates`) are forwarded at all, and forwarding requires
  a non-empty `authored_by.model`. Enforcement lives where the incentive to cheat is, which is the
  side that consumes the number — not the side that emits it.

## Liveness across a language boundary

A Julia task writes its record **through this Python writer** (over PythonCall): one implementation,
zero format drift. That creates one hazard worth pinning down.

Julia holds the interpreter lock while executing Julia code, so the Python heartbeat thread
**cannot fire** during long pure-Julia stretches — exactly the re-solve and fit steps of a
calibration loop. Therefore:

1. The Julia side **owns its liveness**: call `rec.heartbeat()` explicitly at iteration boundaries.
2. It declares an **honest** `heartbeat_interval_s` (e.g. 120 for a loop whose solve step takes a
   minute or two), not the 15 s default. The field is immutable after creation, so a supervised task
   passes it through the submit chain.

The timer thread is documented as *best-effort supplementary* coverage: it fires during acquire waits
and whenever the other language releases the lock into Python. Without rule 1, every healthy Julia
task would falsely derive `stale` — which is why staleness is judged against the manifest's interval
rather than a global constant.

## CLI

```console
$ strumento task create --title "T1 monitor" --device loopback_demo \
      --script scripts/t1.py --workdir . --language python \
      --task-type experiment --target sim --gates fit-recovery \
      --heartbeat-interval-s 15 --root ./experiments --json
{ "ok": true, "id": "2026-07-25-t1-monitor-4f1c", "dir": "…/experiments/2026-07-25-t1-monitor-4f1c" }

$ strumento task status ./experiments/2026-07-25-t1-monitor-4f1c
2026-07-25-t1-monitor-4f1c: running · pid 81234 (alive=True) · 37 event(s) · 1.2s since liveness (stale after 45s)

$ strumento task tail  <dir> -n 50 --json      # the last N events
$ strumento task list  <root>      --json      # every record under a root, newest first
```

`task create` writes `pid = 0` — a fresh manifest is `queued` until whoever spawns the process
writes the pid. A supervisor always passes `--root` explicitly, resolved by the same rule as its
rescan roots, so every supervised record is rediscoverable by a restarted supervisor. The
`<workdir>/experiments/` default exists for standalone runs.

The companion `strumento store current|log|diff` verbs read the calibration store
(`docs/calibration-store.md`) — the other half of a calibration task's provenance. An absent store is
an empty view, not an error: a deviceless or never-calibrated device legitimately has no history, and
these verbs never create one.

```console
$ strumento store current devices/loopback_demo/store --ref main --json
$ strumento store log     devices/loopback_demo/store --ref main --json
$ strumento store diff    devices/loopback_demo/store cfg-aa… cfg-bb… --json
```

`store diff`'s paths are `apply_updates` paths (`qubit.pulses.pi_ge.gain`), so a diff can be replayed
as an update.
