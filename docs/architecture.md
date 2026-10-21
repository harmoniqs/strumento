# strumento architecture

The design of the package in one document: the tenets, the three layers, the seams, and an
index of the design decisions (D-ids) that code docstrings cite. Where a docstring cites a
decision such as D17, the [design-decision index](#design-decision-index) at the end of this
document is where it resolves.

## What strumento is

A first-principles **QICK tProc-v2 experiment framework**: a typed device model, a pulse IR, a
compiler, an experiment/fitting lifecycle, and a raw-qick escape hatch at every level. You
describe operations on *named elements* (`dev.qubit.ge_pi()`); strumento resolves wiring,
compiles to v2, acquires, and reduces to typed results.

It replaces the failure modes of register-level tProc-v1 lab stacks — god-mixin program bases,
positional pulse descriptors, string-`eval` dispatch, contradictory channel maps, multi-sourced
configs — with bounded, independently-testable modules and a single source of truth for every
physical quantity.

## Tenet 0 — every abstraction has an escape hatch

`StrumentoProgram` **is** an `AveragerProgramV2`. Raw qick is reachable three ways, always:

1. `.raw(lambda p: ...)` splices any qick call mid-`Seq`;
2. `self.<any AveragerProgramV2 method>` inside a hand-written `_body`;
3. bypass the builder entirely — write `_body` yourself, keep using the pulse factories.

No abstraction ships without one of these. You can always hack the bottom layer while keeping
the rest.

## Tenet 1 — model the device, not the FPGA

Two layers between physics and DACs:

- **Device language** (what you write): `dev.qubit.ge_pi()`, `dev.coupler.swap("A-B")` — named
  elements, physical units, zero channel numbers.
- **Wiring** (data): each *line* owns one or more *generators* (role → DAC channel, Nyquist
  zone, optional frequency band gate). `Wiring.resolve(LineRef, freq)` is the **only** channel
  map. Re-patching a cable is a one-line config edit; every experiment is untouched.

A `Pulse` targets a `(line, role)` pair — never a bare integer. Band-gated lines (e.g. a
coupler's low/high split) resolve by frequency; the "magic threshold" lives in wiring *data*.

## The three layers

```
core      modality-agnostic mechanism (no physics): wiring, pulse IR, sweeps, device
          model + builder + couplings + operator export, compiler, program, mock, result,
          readout/feedback protocols, fitters (incl. resonator fitters), experiment scaffold,
          soc + transport, dc, calibration store + config, task records, bring-up, simulator +
          response models, schemas, data chest, display, tracing, environment seam, legacy
          Pyro runner
packs     one kit per physics class: element TYPES + pulse libraries + readout kind +
          calibration SCHEMA (cqed built; spin built; tweezer skeleton)
devices   one instance per physical fridge: topology + wiring + calibration VALUES (config)
```

- A pack is a *kit of parts*; a device instance is *one assembly* of those parts. "Multimode"
  is a cqed **instance**, not a pack (D20).
- Packs are discovered via the `strumento.packs` **entry-point group** — third-party packs
  install rather than fork; `core` never imports a pack by name (D15).
- Only `core/program` and `core/mock` import qick. The compiler and the rest of `core` stay
  qick-free; the simulator needs qick only through the mock, which it builds on. Everything
  else — the bulk of the logic — imports and tests with no qick installed.

### Namespaces are topology-derived (D21)

The pack maps element *types* → ops classes (`Transmon → TransmonOps`); `Device.load` binds one
namespace per **named element instance**. `dev.qubit` exists because the YAML names its
transmon `qubit`; a multi-qubit chip gets `dev.q0`/`dev.q1` for free. Instance-declared
**aliases** bind extra namespaces (e.g. `sideband` → the transmon's sideband generator ops). No
instance name appears in pack code.

## The pulse IR (D22, D23)

Typed, physical-unit, sweep-aware — plain dataclasses on the hot path:

- **Envelopes are a typed union**: `Gaussian` / `FlatTop` / `Const` / `Arb` frozen dataclasses,
  each carrying its own parameters. The compiler lowers via a **type-keyed registry** (declare
  hook + pulse-style hook), so a pack registers new envelopes without touching core. Frozen ⇒
  hashable ⇒ envelope dedup per `(generator, envelope)` by construction.
- **`Pulse.gain` is the v2-native normalized fraction** (D23). The cqed factories convert
  legacy int codes at the pack boundary (`gain_int / maxv`, `maxv` from the soccfg snapshot);
  the OC import converts physical amplitudes via a calibrated reference; fraction-native packs
  (spin, tweezer) pass through. One unit inside the IR; conversions live at boundaries.
- **Sweep structure is declared, not inferred** (D36): a `Loop` is a declared loop axis (name +
  point count) declared at program scope; a `Sweep` is a range bound to an axis with `on=`. The
  compiler emits one `add_loop` per *declared axis* — never one per swept parameter — so several
  parameters can step *together* on one axis (a joint sweep), and a sweep bound to an undeclared
  axis is a typed error. A legacy per-name `Sweep("amp", …, 51)` still auto-declares its axis
  (deprecated: constructing one warns at the caller's line). Band-boundary-crossing frequency
  sweeps are rejected.
- **Result axes are as-compiled values** (D37): the swept axis is read back from the compiled
  program — `get_pulse_param` for a swept pulse field, `get_time_param` for a swept delay — so
  `xpts` is the per-step axis the tProc *actually steps* (register-rounded, the loop engine's
  quantized step sizes included), mapped to lab-native units through the factory's inverse
  where one exists — never the requested values, and never a second client-side conversion.
  The requested axis is preserved at `meta["requested_xpts"]`. This is what makes a fitted
  π-gain the *played* π-gain, and it is the same discipline as `realized()`: one source of
  truth, the compiled program.
- **Simultaneity is first-class**: a `Segment` groups pulses sharing a start;
  `.play_parallel(...)` plays multi-line; every OC import is a `Seq` wrapping one `Segment`.
- **Three authoring styles**: chaining for one-liners, `s += op` accumulation (diff-friendly),
  `with Seq() as s:` blocks. `a + b` concatenates immutably.

`StrumentoProgram.realized(pulse)` reports the as-compiled values (frequency, gain fraction,
length) read back from the compiled program via qick's `get_pulse_param` — the queryable middle
between "requested" and "played", and the only honest answer when a field is swept (the loop
engine's register-quantized step sizes included).

## The device model (D30, D31)

Config-as-data is canonical: the instance YAML (topology + wiring + calibration values) is the
single source of truth, serialized and diffable. Around it:

- **`DeviceBuilder`** (D30) — config-as-code that *produces* the canonical model (never a
  parallel store): `add(type, name)` / `wire` / `coupling` / `readout` / `build(calib=…)`.
  Pack-agnostic by construction; packs ship builder sugar as plain functions. Calibration
  *values* are never authored in code — they change every run and stay data.
- **Typed coupling edges** (D31) — an interaction is `coupling(a, b, form, mediated_by=…)`
  where the *form* (Dispersive / ZZ / Exchange / BeamSplitter / JaynesCummings, …) is a
  pack-registered union carrying symmetry + an operator template. The driven coupler *element*
  is separate from the *interaction*. Strengths are calibration data.
- **Hamiltonian export** (D31) — `dev.to_operator_model(frame, cutoffs, include=…)` lowers
  topology + couplings + calibration into a framework-neutral operator spec (drift + drive
  operators + basis + drive map + provenance); `include=` extracts a subgraph for tractability.
  A thin upstream bridge builds a solver-native system from it; core stays solver-free.

## The solver seam — both directions

- **Model out**: `to_operator_model` (above) — what a pulse-optimization stack consumes.
- **Pulse in** (D17): `from_solution(solution, drive_map, gain_calib, dac_rate)` imports
  a solved waveform as normalized `Arb` envelopes in a `Segment` — explicit drive→line mapping,
  physical→fraction gain via a calibrated π reference, owned resampling, provenance on
  `Seq.meta`. `prepare(target)` / `gate(op)` address operations by *what they do*; the
  realization (analytic ladder or OC waveform) is the device's choice.

The two directions close a loop: export the model → solve upstream → import the pulse → measure
→ refine the model. strumento owns measurement, board-aware compilation, and structured export;
design and optimization stay upstream.

## Readout, experiments, fitting (D13, D4)

- **Readout kinds are protocols** (`core/readout.py`): `ReadoutKind` + `Discriminator` + a data
  source (QICK ADC vs external detector). Implementations live in packs: `DispersiveIQ` (cqed),
  `PSBReadout` (spin), `ImagingReadout` (tweezer skeleton) — three structurally different
  readouts through one seam.
- **The experiment lifecycle splits into collaborators**: Runner (build + acquire →
  `MeasurementResult`), Analyzer (fitters → typed `FitResult`, no plotting), Display (optional
  figure). Derived quantities are fields, never stdout.
- **Fitters are vendored, lmfit-based, typed** (`core/fitters.py`, `core/resonator_fitters.py`):
  one damped-sinusoid primitive with typed wrappers (Rabi / T1 / T2-Ramsey / parity /
  double-Gaussian), plus the headline resonator set — Lorentzian, Fano, and the full circle fit
  (cable-delay removal, IQ rotation/offset, loaded/internal/coupling Q; Probst RSI 2015 /
  Khalil JAP 2012). Vendored deliberately: resonator fits change rarely and an external
  dependency is a liability.
- **Three experiment archetypes** exist across the packs:
  1. **Runner-based** (the standard `Experiment` subclass): `sequence()` + `analyze()`, compiled
     through `StrumentoProgram`, acquired via a soc. Registered in the pack's `EXPERIMENTS` dict
     and covered by the experiment contract. Most cqed and spin experiments are this shape.
  2. **Host-driven, standalone** (`ChargeStability`, `PinchOff`, `CoulombDiamond`, etc.; `cqed`'s
     `TimeOfFlight`, #104): their own `run()` + `analyze()`, never registered in `EXPERIMENTS`,
     never an `Experiment` subclass. Two shapes share the archetype: the DC-sweep classes above
     have no `StrumentoProgram` at all — the slow axes live on an external `DCSource`, and the
     host steps voltages and calls a `probe()` at each point, owning its own fitting (direct
     `lmfit` where core's vendored fitters don't cover the domain model); `TimeOfFlight` instead
     plays one compiled `StrumentoProgram` directly and reads it back via `acquire_decimated`
     (there is no DC axis to sweep — it plays a single long probe, once), and its fitting is a
     vendored `core/fitters.py` entry (`fit_time_of_flight`) like a Runner-based experiment's.
  3. **Composite-only** (`active_reset`, `prepare`): sequence-construction functions in pack
     libraries that return a `Seq` composed into other experiments (not standalone).

## The board seam (D14, D25)

- **A soccfg snapshot is the offline board.** `QickSoc.dump_cfg()` JSON → `QickConfig(json)`
  rebuilds everything program construction needs — build, compile, and v2 assembly run with
  **no hardware**. The repo commits a v1 board snapshot (unit math) and the official qick
  testbench **v2** config (real `AveragerProgramV2` assembly offline and in CI); drop your own
  board's dump at `devices/_fixtures/soccfg_v2.json` and the v2 suite runs against your
  firmware.
- **`MockQickSocV2`** is a real `QickConfig` with only the leaf hardware calls stubbed — v2
  validators fire on malformed programs with no FPGA.
- **`SimulatorSoc`** (D32, realizing D29) is the same thing with `poll_data` replaced by a
  pack-owned **response model** rather than zeros: real IR → real compiler → real
  `AveragerProgramV2` → simulated per-shot IQ → real readout projection → real fitter. Its ground
  truth is an ordinary calibration dict held *apart* from the device's belief, which is what makes an
  offline closed-loop demo non-trivial. The mock's zero-IQ contract is untouched — zeros cannot
  exercise a fit path, which is exactly why the simulator exists.
- **Transport is job-server-native, no object proxies** (D14): compile off-board on a fast
  machine; the board is a dumb executor (load overlay → load program → run → return buffers).
  `JobServerSoc` submits a `CompiledJob` and polls; a firmware overlay corresponds 1:1 to a
  soccfg snapshot. A raw board-exec verb is retained (Tenet 0). See `docs/job-manager.md`.
- **Environment seam** (D24, shape in D34): `EnvironmentSource` protocol (read-only station
  telemetry), result stamping at acquire time — and **client-side at transport receipt**, since the
  telemetry is lab-side and the board stays dumb. An existing stamp is never overwritten: whoever
  received the data first was closest to the station. Optional per-line signal-chain metadata rides
  the existing wiring types. Instrument *control* is permanently out of scope.
- **Slow-DC axes** (`core/dc.py`): a `VoltageSource` driver protocol + `DCSource` wrapper
  (named axes, safety clamps, lever-arm-compensated moves, voltage snapshots) for gate-defined
  devices — QICK owns the fast axes, the external source owns the slow ones.

## The agent surface (D33, D35)

Strumento has **two agent doors**, split on *stateless vs stateful*, not on CLI vs MCP:

- `strumento` (`strumento/cli.py`) is one-shot verbs — load, answer, exit. `exec`-able everywhere,
  including headless CI, and structurally incapable of holding a device open across measurements.
- `strumento-mcp` (`strumento/mcp.py`, optional `[mcp]` extra) is a long-lived **session** holding
  `{device, soc, store, ref, recent outcomes}`. That state is the point: a calibration loop is
  *measure → believe → measure again*, and the "believe" step has to survive between calls.
  `record_calibration` snapshots an immutable version, advances the ref **and reloads the session's
  device from it** — the step without which no loop closes. With `load_device(task=…)` the session
  also holds a **task record**: a root mints a fresh task, an existing task dir is resumed, and
  every run's progress and gate verdicts plus each calibration write land in the task's append-only
  log — the durable record a supervisor reads (the task directory is the truth; the in-memory
  outcome list stays a recent-labels cache).

Both doors are thin over the same importable functions, so they cannot answer differently. Packs
publish an `EXPERIMENTS` registry (name → `Experiment` subclass) so a host names an experiment
instead of importing it, and `strumento schema <contract>` (D35) publishes the wire shapes — emitted
from the same definitions that *validate* payloads, so the schema cannot drift from the objects.
A pack's calibration shape, units included, is the pack-scoped `calibration` contract
(`strumento schema calibration --pack <name>`): core resolves the pack by name and asks for its
`CALIBRATION` model, so core stays pack-agnostic.

**The tool list is strategy-free, permanently.** No next-experiment or recommendation tool, ever:
choosing what to measure next belongs to the calibration-strategy layer *above* strumento (D18).
Relatedly, a deterministic harness should drive strumento through the Python API or the CLI directly
— an LLM looping over `run_experiment` would reintroduce LLM-as-state-machine. The MCP session is the
*interactive / external* agent surface. See `docs/cli.md`.

## Testing — the gates

The three original gates test *layers*:

1. **Structural** (mock): programs build + compile; readout shapes match; v2 validators fire.
2. **Numeric golden** (offline): realized values read back from real v2 assembly
   (`StrumentoProgram.realized`) match hand-computed goldens — the gate that catches
   unit-conversion bugs (int-code vs fraction gain, grid snapping). Runs in CI against the
   committed v2 testbench config.
3. **Physics agreement** (board): T1 / T2 / Rabi on real hardware, fit sensibly, agree with a
   trusted reference stack. This gate needs a board wired to a cooled device.

Two more are still board-free but test *lifecycles*, end to end:

- **Offline fit-recovery** (`SimulatorSoc`, D32): every shipped experiment's fitter recovers the
  simulator's perturbed ground truth through the real path, to a tolerance derived from the shot
  noise and the scan's conditioning rather than picked round.
- **The capstone** (the MCP session, D33): an in-process agent client opens a session on
  `devices/loopback_demo`, measures a device whose belief is wrong, writes the fit back, re-measures,
  and the belief-vs-truth error collapses — with the provenance chain readable from the store.

Markers: `qick` (needs the qick package — still no board) and `mcp` (needs the `[mcp]` extra;
sessions that drive a soc are dual-marked). An unmarked test is pure — no qick, no board. CI runs all of them on hosted runners;
a board-attached self-hosted runner picks up `-m qick` for gate 3 (see `docs/hardware-ci.md`).

## Design-decision index

Docstrings cite decisions by id. One-line summaries (details throughout this doc):

| id | decision |
|---|---|
| D1–D3 | first-principles greenfield; broad multimode core surface; standalone package |
| D4 | new lmfit fitters behind a typed seam, validated against a legacy oracle |
| D5 | pulse-IR + compiler architecture (thin program base, bounded modules) |
| D6 | every abstraction escapable (Tenet 0) |
| D7 | model the device, not the FPGA (Tenet 1) |
| D8 | fluent `Seq` builder as the primary sequence surface |
| D9 | config versioning mechanism — open; store interface planned, immutability load-bearing |
| D10 | single source of truth; nothing derived is stored |
| D11 | typed factory functions; no string keys, no eval |
| D12 | one loaded `Device` object; soccfg snapshot kept as plain data |
| D13 | Experiment contract kept; Runner/Analyzer/Display split inside |
| D14 | job-server-native board transport; no object-proxy layer; compile off-board |
| D15 | design-for-extension: core + packs + instances; entry-point discovery |
| D16 | pydantic for config layers; plain dataclasses on the hot path |
| D17 | OC import = typed `from_solution` boundary; operations addressed by intent |
| D18 | strumento is the strategy-agnostic substrate under closed-loop calibration stacks |
| D19 | tweezer pack grounded in published AOD-control architecture (arXiv:2506.13724) |
| D20 | pack = kit of element types; device = per-fridge instance of a pack |
| D21 | topology-derived namespaces; instance-declared aliases |
| D22 | typed envelope union + type-keyed lowering registry |
| D23 | `Pulse.gain` is the v2 fraction; conversions live at pack boundaries |
| D24 | environment/station seam: read-only telemetry + result stamping, never control |
| D25 | v2 API ground truth: subclass `AveragerProgramV2`; `dump_cfg()` → offline assembly |
| D26 | `DeviceModel` as a measured artifact; characterization path = calibration path |
| D27 | board-description schema → `Wiring` (single source of truth down to copper) |
| D28 | transfer-function-aware compilation: per-line H(ω) + pre-distortion at import |
| D29 | `SimulatorSoc`: same acquire protocol against a physics simulation — **realized by D32** |
| D30 | `DeviceBuilder`: config-as-code that produces canonical config-as-data |
| D31 | typed coupling edges + framework-neutral Hamiltonian export |
| D32 | the simulator's mechanism, split on the qick boundary: pure response registry + per-shot machinery (`core/response.py`) + the soc (`core/simulator.py`) + pack-owned response models; **ground truth is data**, held apart from the device's belief |
| D33 | **two agent doors**: CLI = stateless verbs, `strumento-mcp` = a stateful lab session holding `{device, soc, store, ref, outcomes}`; packs gain an `EXPERIMENTS` registry; the tool list is strategy-free, permanently |
| D34 | the D24 environment seam, minimal shape: `EnvironmentSource` + `Runner` stamping + client-side stamping at transport receipt (never overwritten) |
| D35 | `strumento schema <contract>`: the typed contracts as JSON Schema, emitted from the same definitions that validate — plus `MeasurementResult.to_wire()/from_wire()`; a pack's calibration model is the pack-scoped `calibration` contract (`--pack <name>`) |
| D36 | sweep **loop structure is declared, not inferred**: `Loop` axes at program scope + `Sweep(on=…)` binding; one `add_loop` per axis; joint sweeps in one loop; legacy per-name loops deprecated |
| D37 | **result axes are as-compiled values**: `xpts` is read back from the compiled program (`get_pulse_param` / `get_time_param`), never re-derived client-side; requested axis kept at `meta["requested_xpts"]` |
| D38 | **`FitResult.derived` carries JSON values**: numbers, strings, booleans, null, lists and objects, and the published `fit` schema says so (a protocol name, a confusion matrix, a refusal's provenance live beside the numbers); numbers stay numbers, and a consumer that needs one checks its key. Rejected: keeping `derived` numeric and moving the rest to a new field, which would move keys readers already use. A simulator cross-check validates every registered experiment's fit against the schema |

D26–D29 are the **codesign enablement** arc (device-model measurement + board-aware compile +
structured export, with design/optimization held upstream); the model/report pieces are designed
seams, built as they're scheduled. **D29 is no longer one of them** — D32 built it, and the
offline-simulation lane (fit-recovery against a perturbed ground truth, active reset on the `Branch`
seam, the whole capstone calibration loop) runs on the committed testbench soccfg with no board.

D32–D35 are the **agentic-substrate** arc: something to measure (D32), a way to say what was
measured (D34) in a shape another program can read (D35), and two doors an agent can drive it
through (D33).
