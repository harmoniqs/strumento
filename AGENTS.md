# AGENTS.md — working on strumento

Orientation for AI agents (and fast-moving humans). Everything here is verified against the
tree it ships with; when in doubt, the tests are the ground truth.

## What this repo is

A first-principles **QICK tProc-v2 experiment framework**: typed device model → pulse IR →
compiler → `AveragerProgramV2` → acquire → typed fitting. Read
**[docs/architecture.md](docs/architecture.md)** first — it defines the three layers
(core / packs / devices), the tenets, and the design-decision ids (D-numbers) that code
docstrings cite.

## Environment

- **Python 3.10–3.12** (the CI matrix). Work in a virtual environment on one of those versions;
  a system `python3` may be older or newer than the supported range.
- `pip install -e ".[test]"` — pure layers. `pip install -e ".[test,qick]"` adds the
  compiler/program/mock/simulator layers (the extra pins `qick>=0.2.367,<0.3`); qick installs fine
  on x86 (no pynq off-board).

## Tests — run them like CI does

```bash
pytest -m "not qick" -q      # pure suite — must pass with NO qick installed
pytest -q                    # full suite — real tProc-v2 assembly runs offline
pytest -q -m mcp             # the agent-surface suite (needs the [mcp] extra + qick)
ruff check .                 # lint (config in pyproject)
```

- Marker rule: any test that imports `strumento.core.{mock,program,simulator,compiler-with-qick}`
  or `qick` itself is `@pytest.mark.qick`. Everything else must run qick-free — CI's `pure` job
  enforces this by running without qick installed. Tests needing the MCP SDK are `mcp`-marked and
  live behind `pytest.importorskip("mcp")`; a session that drives a soc is **dual-marked**
  (`mcp` + `qick`), and the `mcp` CI job is the only lane that installs both.
- **The v2 fixture gate** (`tests/_v2gate.py`): v2-assembly tests use
  `devices/_fixtures/soccfg_v2.json` (your board's `dump_cfg()`, gitignored) when present, else
  the committed official testbench config (`soccfg_v2_testbench.json`, single gen + single
  readout). Every snapshot-carrying demo assembles for real offline: use `devices/multimode_demo`
  (the cqed exemplar, on `soccfg_v2_cqed.json`) for anything multi-channel, and
  `devices/loopback_demo` only for single-channel v2 assembly against the testbench.
- TDD is the working style here: failing test → minimal code → green → commit.

### The experiment contract — and why it exists

`tests/packs/test_experiment_contract.py` holds five **structural** guards, parametrized over the
pack `EXPERIMENTS` registries so coverage grows automatically. Read its docstring before adding an
experiment; the short version of the post-mortem it encodes:

> Four of five cqed analyzers were once broken on realistically-shaped data **while the suite was
> green**. An averaged single-read acquisition arrives shaped `(n_reads, n_points)` = `(1, N)`, but
> the analyzer unit tests hand-built **1-D** arrays and called `analyze()` directly — so they were
> verified against a shape production never emits. The mock's zero-IQ made e2e blind too, and
> nothing was parametrized over the registries, so four breakages were four silences.

Rules that follow, and are enforced:

1. **Analyzer tests consume Runner-shaped data.** A bare 1-D array is a deliberate *degenerate
   case*, never the only coverage. The real-path guard covers the production shape for every
   registered experiment.
2. **Registering an experiment without covering it fails the suite** (`CONTRACT_KWARGS`). An
   `Experiment` subclass that is deliberately unregistered needs a written reason in
   `NOT_REGISTERED` — the inverse gap is closed too.
3. **qick reports timing conflicts through `logging`, not `warnings`** (`asm_v2.py:1016`
   `logger.warning("Readout time … appears to conflict …")`). `pytest.warns`, `filterwarnings` and
   `-W error` are all **blind** to them. The diagnostics guard watches the `qick` and `strumento`
   loggers at WARNING *and* Python warnings; a conflict is a correctness bug qick declines to
   raise, so without it the bug lands in a passing run and only appears on a board.
4. **`reads_per_shot` must equal the reads the sequence performs.** qick increments `ro['trigs']`
   once per trigger *macro at assembly*, so a `Measure` inside a `Repeat` is counted once however
   many times the tProc runs it — under-declared reads are unretrievable. Unroll, or reject loudly.
5. **Never make a guard pass by weakening it.** Enlarge a scan, fix the code, or leave a
   `strict=True` xfail with a precise reason. Two of the guards carry explicit non-vacuity proofs
   (a known-bad back-to-back-`Measure` sequence; a `Repeat`-containing-`Measure`) so they cannot
   silently go blind — if those proofs ever start failing, the guard broke, not the fixture.

## Hard rules

1. **Purity boundary**: only `core/compiler.py`, `core/program.py`, `core/mock.py`,
   `core/simulator.py` may import `qick`. `import strumento` must never pull qick in. There are
   tests for this. The simulator's *pure* half (response registry + mechanism) is
   `core/response.py`, and pack response models (`packs/cqed/responses.py`) stay pure too — packs
   are imported by `Device.load`, which runs in the pure lane. `mcp.py` sits *above* the boundary:
   it reaches a soc class only inside `load_device` (lazy import), so `import strumento.mcp` is
   qick-free — and SDK-free, so `LabSession` works without the `[mcp]` extra.
2. **`core` is physics-free**: element types, pulse libraries, readout kinds, calibration
   schemas live in `packs/`. Core never imports a pack by name (entry-point discovery only).
3. **Gain units (D23)**: `Pulse.gain` is always the v2 normalized fraction. Int-code ↔ fraction
   conversion happens only at the cqed pack boundary; physical-unit conversion only in
   `from_solution` via a calibrated reference.
4. **Config is data**: calibration values live in instance YAMLs, never in code. The
   `DeviceBuilder` produces the canonical model; it never becomes a second store.
5. **Escape hatch**: never wall off raw qick. Any new abstraction needs a bypass.
6. **No collaborator names** in code, docs, comments, or commit messages — this tree is kept
   public-ready. Published-paper citations (arXiv ids) are fine. `tests/test_docs_hygiene.py`
   checks the public docs for internal references and every tracked file for the partner-lab
   name; run it before large doc changes.
7. **Typed results**: fitters return `FitResult` (params/stderr/derived), experiments return
   `ExperimentOutcome`. No printing from analyzers; no derived quantities to stdout.

## Repo map

```
strumento/core/        wiring · pulses (IR + from_solution) · sweeps · device (+ builder,
                       coupling, operators) · compiler (envelope registry) · program (+ CompiledJob
                       extraction) · mock · simulator (SimulatorSoc) · result (+ wire form) ·
                       readout · fitters · resonator_fitters · response (response registry + sim
                       mechanism) · schemas (JSON Schema contracts) · experiment · soc
                       (+ ArmableSoc, ShotSource) · transport (wire contract + JobServerSoc +
                       transport-backed run) · environment (D34 stamp) ·
                       record (task-record contract — pure stdlib, zero deps) · dc · cli ·
                       datachest · display · tracing · bringup · backend · environment · schemas
strumento/mcp.py       the stateful lab session (D33): LabSession + 8 MCP tools + strumento-mcp
strumento/packs/cqed/  elements · calibration · library/ (transmon, sideband, mode, coupler,
                       readout, composite) · readout (DispersiveIQ) · responses · safety ·
                       experiments/ (spectroscopy, rabi, t1/t2, cavity,
                       sideband, swap, RB, DRAG, readout opt, active reset, bringup, …) ·
                       EXPERIMENTS registry
strumento/packs/spin/  elements (Dot/ST/EO/Hole, Plunger, Barrier, Sensor, Reservoir,
                       DriveLine, GateSensor) · calibration · library/ (ld, exchange,
                       hole_spin, singlet_triplet, composite, eo_clifford) ·
                       readout (PSB, Elzerman, DispersiveGate, LatchedPSB) · responses ·
                       safety · experiments/ (21 registered + 8 DCSource-based)
strumento/packs/tweezer/  skeleton (elements, PolyChirp envelope, imaging readout)
devices/               multimode_demo (docs/tests anchor) · loopback_demo (single-channel testbench) ·
                       tweezer_demo · spin_demo · _fixtures/ (soccfg snapshots)
examples/jobserver/    reference execute-agent (the D14 server side — deployment, not core)
docs/                  architecture · getting-started · cqed-cookbook · packs-and-devices ·
                       fitters · spin-pack · job-manager · qick-v2-api-notes · hardware-ci · cli ·
                       calibration-store · task-record · timeline
```

## Extension recipes

- **New pulse factory**: one function on the right ops class reading calibration; add a schema
  field if needed. Test: factory → expected `Pulse` fields.
- **New envelope**: frozen dataclass + `register_envelope(Type, declare_fn,
  pulse_kwargs=style_fn)` — packs do this at import (see tweezer's `PolyChirp` slot).
- **New experiment**: subclass `Experiment`; implement `sequence()` + `analyze()`; wire a
  fitter; structural e2e on the mock + fit-recovery on synthetic data. Then **two registrations**:
  add it to the pack's `EXPERIMENTS` dict (name → class, so a host can run it by name — D33; the
  constructor takes the device positionally and everything else keyword-only), and register a
  **response model** in the pack's `responses.py` (`@responds(YourExperiment)`) to make it runnable
  offline. Add its fit-recovery case to `tests/e2e/test_fit_recovery.py`.
- **New response model** (what makes an experiment simulatable, D32): a function
  `(SimContext) -> SimResponse` in the pack's `responses.py` under `@responds(TheExperiment)`.
  Read parameters through the pack's `params(ctx)` layering — **simulator defaults → the device's
  belief → the armed ground truth** — never a hardcoded constant, and never from `core`
  (physics stays in packs). Return `populations=` for a closed form over the swept axis, `iq=` for a
  complex response, or `machine=` for a per-shot walk of `Branch`/`Repeat` (active reset). Size the
  scan from the error budget in `tests/e2e/test_fit_recovery.py`'s docstring, not from round numbers.
- **New fitter**: lmfit model → typed `FitResult` in `core/fitters.py`; synthetic-data
  recovery test with tolerance; add to the CLI `fit` map.
- **New pack**: directory under `packs/` + `CALIBRATION` + `compose_elements` + `EXPERIMENTS`
  (`{}` is fine for a skeleton) + entry point in pyproject. See `docs/packs-and-devices.md`.
- **New device instance**: a YAML under `devices/` (or out-of-tree — instances resolve packs
  via entry points; instances can live out-of-tree).

## Two agent doors — stateless verbs and a stateful session (D33)

```
strumento packs | validate <yaml> | info <yaml> | soccfg <json> | fit <kind> <csv> | compile <yaml>
          | check <yaml>                      # wiring vs soccfg snapshot; exit 1 on any error
          | schema <device|fit|result|operator-model>
          | task create|status|tail|list      # long-running experiment records
          | store current|log|diff            # the calibration store, read-only

strumento-mcp    # the stateful lab session, over stdio ([mcp] extra; strumento/mcp.py)
                 # load_device · device_info · list_experiments · run_experiment
                 # calibration_history · record_calibration · schema · validate
```

All CLI commands take `--json`; errors are `{"ok": false, "error": ...}` + exit 1 (the MCP tools
return the same shape as data). A negative verdict is a failure too: `check` exits 1 whenever its
`ok` is false (warnings alone exit 0). The split is **stateless vs stateful**, not CLI vs MCP: the session
holds `{device, soc, store, ref, recent outcomes}` because a calibration loop is *measure → believe →
measure again* and the "believe" step must survive between calls. Both doors are thin over the same
importable functions (`cli.device_summary` / `flatten_calibration` / `provenance_entries`,
`core.schemas`, `core.store.record_calibration`), so they cannot drift. See
[docs/cli.md](docs/cli.md).

Two rules that are not negotiable:

- **The MCP tool list is strategy-free, permanently.** No next-experiment / recommendation /
  planning tool, ever. What to measure next belongs to the calibration-strategy layer *above*
  strumento (D18); a tool here that answered it would make strumento a strategy.
- **`record_calibration` reloads the session device** from the new config after advancing the ref.
  That reload is what closes a loop — without it a session keeps measuring with the belief it just
  replaced.

The task-record format itself (a public, pure-stdlib contract: `core/record.py` +
`docs/task-record.md`) is what a long experiment writes to disk so it can outlive the process that
started it — including the **gate trust boundary**: gate verdicts are computed by vetted code
(`Experiment.go(record=…)`), and scripts route them, they never author them.

## What needs a board (and what doesn't)

| Capability | Board needed? |
|---|---|
| Pulse IR, device model, builder, couplings, operator export, fitters, CLI | no — no qick at all |
| Real v2 program assembly + mock acquire | no — committed testbench soccfg |
| Simulated acquire + offline fit-recovery (`SimulatorSoc`) | no — committed testbench soccfg |
| A whole closed calibration loop through the MCP session (the capstone) | no — committed testbench soccfg |
| Transport-backed run against the reference job server | no — in-process or subprocess |
| Numeric-golden gate (realized values) | no |
| Multi-channel v2 assembly beyond 1 gen/1 readout | your board's `dump_cfg()` JSON |
| Physics-agreement gate (real T1/T2/Rabi) | yes — self-hosted runner (docs/hardware-ci.md) |

## CI

`.github/workflows/ci.yml`: `lint` (ruff) + `pure` (3.10–3.12, no qick) + `qick` (3.10/3.11, full
suite, offline v2 assembly, `-m "not mcp"`) + `mcp` (3.10/3.11, `.[test,qick,mcp]`, `-m mcp` — the
agent surface and the capstone). `hardware.yml` is the board-in-the-loop job — inert until a
self-hosted runner is registered (`HAS_QICK_RUNNER` repo variable); label a PR `run-hardware` to
request a board run. PRs must be green on all four hosted jobs.
