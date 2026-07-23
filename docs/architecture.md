# strumento architecture

The design of the package in one document: the tenets, the three layers, the seams, and an
index of the design decisions (D-ids) that code docstrings cite. This is the public
distillation of the internal design-of-record; where a docstring says "spec §6.5 / D17", this
document is where that resolves.

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
          readout/feedback protocols, fitters, experiment scaffold, soc + transport, dc
packs     one kit per physics class: element TYPES + pulse libraries + readout kind +
          calibration SCHEMA (cqed built; tweezer, spin growing)
devices   one instance per physical fridge: topology + wiring + calibration VALUES (config)
```

- A pack is a *kit of parts*; a device instance is *one assembly* of those parts. "Multimode"
  is a cqed **instance**, not a pack (D20).
- Packs are discovered via the `strumento.packs` **entry-point group** — third-party packs
  install rather than fork; `core` never imports a pack by name (D15).
- Only `core/compiler`, `core/program`, and `core/mock` import qick. Everything else — the
  bulk of the logic — imports and tests with no qick installed.

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
- **Sweeps are values**: any pulse/delay field takes a scalar or a `Sweep`; the compiler lowers
  to `QickSweep1D` + `add_loop` and rejects sweeps that cross a generator band boundary. Sweep
  axes stay lab-native for plotting (`.xpts` survives unit mapping).
- **Simultaneity is first-class**: a `Segment` groups pulses sharing a start;
  `.play_parallel(...)` plays multi-line; every OC import is a `Seq` wrapping one `Segment`.
- **Three authoring styles**: chaining for one-liners, `s += op` accumulation (diff-friendly),
  `with Seq() as s:` blocks. `a + b` concatenates immutably.

`quantize(pulse, soccfg, wiring)` reports the DAC-grid realized values (frequency, gain
fraction, cycles) without touching the IR — the queryable middle between "requested" and
"played".

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
- **Pulse in** (§6.5, D17): `from_solution(solution, drive_map, gain_calib, dac_rate)` imports
  a solved waveform as normalized `Arb` envelopes in a `Segment` — explicit drive→line mapping,
  physical→fraction gain via a calibrated π reference, owned resampling, provenance on
  `Seq.meta`. `prepare(target)` / `gate(op)` address operations by *what they do*; the
  realization (analytic ladder or OC waveform) is the device's choice.

The two directions close a loop: export the model → solve upstream → import the pulse → measure
→ refine the model. strumento owns measurement, board-aware compilation, and structured export;
design and optimization stay upstream (§19 framing).

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

## The board seam (D14, D25)

- **A soccfg snapshot is the offline board.** `QickSoc.dump_cfg()` JSON → `QickConfig(json)`
  rebuilds everything program construction needs — build, compile, and v2 assembly run with
  **no hardware**. The repo commits a v1 board snapshot (unit math) and the official qick
  testbench **v2** config (real `AveragerProgramV2` assembly offline and in CI); drop your own
  board's dump at `devices/_fixtures/soccfg_v2.json` and the v2 suite runs against your
  firmware.
- **`MockQickSocV2`** is a real `QickConfig` with only the leaf hardware calls stubbed — v2
  validators fire on malformed programs with no FPGA.
- **Transport is job-server-native, no object proxies** (D14): compile off-board on a fast
  machine; the board is a dumb executor (load overlay → load program → run → return buffers).
  `JobServerSoc` submits a `CompiledJob` and polls; a firmware overlay corresponds 1:1 to a
  soccfg snapshot. A raw board-exec verb is retained (Tenet 0). See `docs/job-manager.md`.
- **Environment seam** (D24): `EnvironmentSource` protocol (read-only station telemetry), result
  stamping at acquire time, optional per-line signal-chain metadata. Instrument *control* is
  permanently out of scope.
- **Slow-DC axes** (`core/dc.py`): a `VoltageSource` driver protocol + `DCSource` wrapper
  (named axes, safety clamps, lever-arm-compensated moves, voltage snapshots) for gate-defined
  devices — QICK owns the fast axes, the external source owns the slow ones.

## Testing — three gates (§10)

1. **Structural** (mock): programs build + compile; readout shapes match; v2 validators fire.
2. **Numeric golden** (offline): realized values from `quantize()` / real v2 assembly match
   hand-computed goldens — the gate that catches unit-conversion bugs (int-code vs fraction
   gain, grid snapping). Runs in CI against the committed v2 testbench config.
3. **Physics agreement** (board): T1 / T2 / Rabi on real hardware, fit sensibly, agree with a
   trusted reference stack. This gate needs a board and runs with a lab collaboration.

Markers: `pure` (default; no qick), `qick` (needs the qick package — still no board). CI runs
both on hosted runners; a board-attached self-hosted runner picks up `-m qick` for gate 3
(see `docs/hardware-ci.md`).

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
| D29 | `SimulatorSoc`: same acquire protocol against a physics simulation (design-time) |
| D30 | `DeviceBuilder`: config-as-code that produces canonical config-as-data |
| D31 | typed coupling edges + framework-neutral Hamiltonian export |

D26–D29 are the **codesign enablement** arc (device-model measurement + board-aware compile +
structured export, with design/optimization held upstream); the model/report/simulator pieces
are designed seams, built as they're scheduled.
