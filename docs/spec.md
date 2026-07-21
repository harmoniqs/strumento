---
type: spec
date: 2026-07-16
session_id: "e3dfcc03-f686-47a0-b129-29407ed941d0"
status: draft
priority: P2
platform: bosonic
tags: [spec, strumento, bosonic, multimode, cqed, atoms, tweezer, neutral-atom, thompson-group, aod, imaging-readout, spin, quantum-dot, spinqick, psb-readout, qick, tproc-v2, schuster-lab, multimode_expts, pulse-ir, device-model, modality-pack, device-instance, fluent-builder, config-versioning, single-source-of-truth, active-reset, host-in-loop-feedback, mock-testing, fitting, lmfit, job-server, data-lifecycle, envelope-union, topology-namespaces, station-seam, environment, fridge, maybell, codesign, device-model, system-identification, transfer-function, pre-distortion, board-description, crosstalk, simulator-backend]
linked_plan: "./plan.md"
---

# `strumento` — a first-principles QICK tProc-v2 experiment framework (first target: the Schuster multimode device)

> **Renamed 2026-07-17: `strumento` → `strumento`.** The package is renamed to avoid claiming QICK's
> namespace (it reads as quasi-official and near-collides with KIT's `qiclib`), to be OSS-safe (the
> `strumento` PyPI name is free; `strumento`'s vibe wasn't the issue, the brand was), and to join the
> Harmoniqs Italian-musical family — *strumento* = "instrument" (QICK is the **Instrument**ation Control
> Kit; strumento is the instrument on top). The vault filename keeps `strumento` as a stable historical id;
> the GitHub repo and Python package are `strumento`.

> **What this is.** A design-of-record for a **fresh, standalone Python package**, `strumento` — a
> general **QICK tProc-v2 experiment framework** (device model + pulse IR + compiler + experiment/fitting
> lifecycle + a raw-qick escape hatch). Its **first target** is the *core functionality* of Stanford's
> [`SchusterLab/multimode_expts`](https://github.com/SchusterLab/multimode_expts)
> on **QICK tProc v2** — from first principles, **not** as a port-in-place. It keeps
> the physics (qubit + manipulate cavity + storage modes + coupler, with f0g1,
> beamsplitter swaps, parity, active reset) and throws out the architecture: the
> 2745-line `MM_base` god-mixin, the positional 7-row pulse descriptor, the
> `eval`-dispatched string pulse-creator, the contradictory magic-int channel maps,
> and the multi-sourced config.
>
> **Status.** Draft for review. The design was worked out interactively with Aaron;
> every major fork below records that decision. One genuine feasibility risk
> (measurement-conditional feedback on tProc v2 — "active reset") is flagged for the
> QICK collaboration (Sho Uemura).
>
> **Relationship to existing work.** Sibling to `expt_service` (the coarse board-side
> 3-verb service for the closed-loop optimizer) and `IntonatoQICK.jl`. This package is
> a general QICK experiment framework whose **first target is the native Stanford multimode stack** on
> v2 — a much richer surface than the 3-verb contract. It does not replace them; if a closed-loop path is later wanted, one adapter
> maps GHz-Rabi → `Pulse.gain`.

## 1. Context & motivation

`multimode_expts` is a tProc **v1** codebase (confirmed: zero `asm_v2` usage anywhere).
Its own TODO reads *"Migrate to tProcv2 at some point."* The stack today is four layers:

1. **`Experiment` classes** (**~73** ending in `Experiment`, of ~83 named `*Experiment*`, in
   `single_qubit/` alone; **140** across `experiments/`) — expand cfg, instantiate a Program,
   `prog.acquire(soc)`, unpack
   `(xpts, avgi, avgq)`, `analyze()`/`display()`.
2. **Three program bases** — `MMAveragerProgram` / `MMRAveragerProgram` /
   `MMNDAveragerProgram` = v1 `AveragerProgram`/`RAveragerProgram`/`NDAveragerProgram`
   **+ `MM_base` mixin**, each with hand-written `initialize()`/`body()`.
3. **`MM_base`** (2745 lines) — every pulse primitive (`custom_pulse`,
   `register_long_pulse`, `active_reset`, `measure_wrapper`, `reset_and_sync`, parity,
   dual-rail) written against v1's **register-level** API (`setup_and_pulse`,
   `freq2reg`/`deg2reg`/`us2cycles`, `sync_all`, `add_gauss`).
4. **`prepulse_creator2`** — builds the pulse descriptor `[[freq],[gain],[len],[phase],[ch],[shape],[sigma]]`, dispatched via literal `eval(f"creator.{channel_name}(...)")`.

### 1.1 The messes a redesign fixes (all found in-tree)

- **God-mixin + pre-converted registers.** `parse_config` dumps ~50 attributes onto
  `self` and pre-converts physical values to register units at parse time — a pure
  v1-ism, since v2 `add_pulse` takes physical units (MHz/µs/deg) directly.
- **Contradictory magic-int channel model — *three* disagreeing legends.** `channel_table` says
  `4→man, 6→storage`; `custom_pulse`'s docstring says `4→storage, 6→manipulate`; and
  `prepulse_creator2.__init__` says `3→manipulate, 4→storage` — and none matches the actual config DACs.
- **Positional untyped descriptor.** 7 parallel object-dtype lists indexed by physical row;
  string-typed shapes (`"gaussian"`/`"gauss"`/`"g"`/`"flat_top"`/`"f"`/…).
- **Multi-sourced config (no single source of truth).** See §7.1 — f0g1 defined in two files,
  `man()` duplicates `multiphoton('fn-gn+1')`, qubit ge freq in 3+ places, channel map both in
  config and hardcoded in code.
- **High-friction extension.** Adding one pulse type touches a config location (one of *5*
  schemas), a `prepulse_creator2` method, maybe a `custom_pulse` branch, and
  `channel_assign`/`channel_table` — four edits, four desync chances.

### 1.2 Assets to reuse (conceptually — reimplemented in the new package)

- **`MockQickSoc` strategy** — the real qick library runs through build→compile→acquire; only
  the leaf `soc.*` calls are stubbed. This validates programs (v2 param validators fire) with no
  FPGA. Directly reusable for v2 (same three-phase split).
- **`soccfg_snapshot.json`** — committed `QickConfig` snapshot for off-board work.
- **`BranchManager`** — a git-reflog-style config-versioning idea (good model; the *versioning mechanism* it implies is **reopened** — see §7.3 / §14.1, D9).

## 2. Locked decisions (from the design session)

*IDs reflect chronological order; D13/D14 are listed next to their thematically-adjacent decisions
(D4, D9), so the table is grouped rather than strictly numeric. All of D1–D31 are present. D9 is
**open** (§14.1); D14 is **resolved** (§14.2, landed 2026-07-21); D18's expt_service **execution** circulates (§16); the rest are locked. D21–D24 were
added in the naming/extensibility/elegance/station review round (2026-07-17); D25 folds in the qick-v2
API review; D26–D29 add the codesign-enablement dimension (§19); D30 adds the `DeviceBuilder` authoring seam (§5.3, landed 2026-07-21); D31 adds typed-edge couplings + the device-model → Hamiltonian export seam (§5.3 / §6.6, landed 2026-07-21).*

| # | Question | Decision |
|---|---|---|
| D1 | Migration goal | **First-principles greenfield**, not port-in-place. Keep core functionality, free to redesign architecture. |
| D2 | Functional surface of first slice | **Broad multimode core**: single-qubit (ge/ef) + f0g1 + storage swaps + parity + active_reset. |
| D3 | Where it lives | **Fresh standalone package** (`strumento`), outside `multimode_expts`. |
| D4 | Fitting/data-processing | **New `lmfit`-based core fitters** (Rabi/T1/T2/parity) behind a typed result seam, **validated against the current `fitting.py` as a physics oracle** (§9). Full fit-layer refactor is a separate follow-on. |
| D13 | Data lifecycle | **Keep the high-level `Experiment` contract** (`go`/`acquire`/`analyze`/`display`/`save`) but **separate the internals** into distinct collaborators — Runner (acquire) / Analyzer (lmfit fitters) / Display (plot). The v1 conflation lives in the *implementation*, not the interface (§9). |
| D5 | Architecture | **Approach A — Pulse-IR + compiler**, thin base class; logic in bounded, independently-testable modules. |
| D6 | Escape hatch | **Every abstraction is escapable** (Tenet 0). `StrumentoProgram` *is* an `AveragerProgramV2`; raw qick reachable at three levels. |
| D7 | Channel abstraction | **Model the device, not the FPGA** (Tenet 1). Two layers: device language on top, swappable DAC wiring underneath. |
| D8 | Sequence surface | **Fluent builder** `Seq().play(...).wait(...).raw(...).measure()` as primary. |
| D9 | Config versioning | **OPEN — to circulate (§14.1).** Tension between append-only immutable snapshots (conflict-free by construction; the *current* model) and git-native (simpler, but race-prone under the shared worker + concurrent users). The earlier lean toward git-native is reopened given the operational reality. |
| D10 | Config integrity | **Single source of truth** for every physical quantity; nothing derived is stored; low-friction extension. |
| D14 | Job-server / multi-user | **RESOLVED 2026-07-21 — job-server-native, no-Pyro, compile-off-board + thin execute-agent (§14.2.1, topology (ii)).** Own hardware removes the blocking constraint: we run on a **separate, dedicated FPGA QICK board in parallel on the fridge** (alongside the Schuster multimode board), so there is no shared worker/queue to stay compatible with — we build the clean end-state directly and do **not** stage through Option-A wire-compatibility. Board = dumb executor (load overlay → load compiled program → run → return ADC buffers); off-board owns build/compile/analyze/fit; `JobServerSoc` transport; firmware overlays = per-snapshot `dump_cfg()` JSON; raw board-exec/debug verb retained (Tenet 0). Option A/B remain the description of the compatibility path for anyone sharing the Schuster board; we do not take it. |
| D11 | Pulse API | **Typed factory functions**, no string keys, no `eval`; pulses are *computed from calibration*, not a stored bank keyed by strings. |
| D12 | `dev` | The one loaded `Device` object (wiring + calibrations + soccfg **snapshot** + datasets); factory namespaces hang off it. The compiler builds the live `QickConfig` from the snapshot, keeping `device.py` qick-free. |
| D15 | Generality / horizontal scope | **Design-for-extension** (§15): modality-agnostic `core` + per-modality **modality packs** + per-fridge **device instances** (§5.1). **Three packs: `cqed` (build, Stanford) + `tweezer` (build, Thompson — §17) + `spin` (designed, spinQICK-modeled, build-gated on a spin user — §15.5).** `cqed`+`tweezer` are two real, very different modalities in hand; `spin` is the third data point (shipped external framework) that keeps `core` honest. |
| D16 | Config schema | **Pydantic** for the config layer (wiring, device, pack calibration) — validation + JSON round-trip for the §14.2 job blob, matching spinQICK. Hot-path IR (`Pulse`/`Delay`/`Seq`) stays plain dataclasses. |
| D17 | Optimal-control / Piccolo seam | **Drop `dev.optimal_control`** → address by operation (`dev.prepare(target)` / `dev.gate(op)`; realization analytic *or* OC). First-class **`from_solution(...)` import boundary** (artifact → `Seq` wrapping a normalized-ARB `Segment`, drive→line map, physical→DAC gain calib, resample, provenance) + **`Segment`/simultaneity** in the IR — **in the minimal core** (§6.5). **Catalog auto-retrieval deferred.** |
| D18 | Calibration-stack relationship | **strumento is the open substrate at the bottom of the QILC stack** (Intonatissimo→Intonato→IntonatoQICK→strumento→board); it stays **strategy-agnostic** (QILC method stays in Intonatissimo). **`expt_service` unifies as a thin adapter over strumento** (direction locked; execution to circulate with Sho — §16). `from_solution` (§6.5) is the QILC-iterate-on-hardware seam; readout reducers emit Intonato's measurement vectors (§16). |
| D20 | Pack/instance model + `spin` pack | **Two-layer device model (§5.1):** a **modality pack** = element *types* + libraries + readout kind + calib *schema* (code, reusable); a **device instance** = topology + wiring + calibration *values* (config, per fridge, §14.1). "Multimode" is a **`cqed` instance**, not a pack; transmon / multi-transmon / transmon+bosonic are free topologies. Add a **`spin`** (quantum-dot) modality pack (spinQICK-modeled — Dot/Plunger/Barrier, LD + exchange-only, PSB readout), **designed now, build-gated on a spin user** (§15.5). |
| D19 | Tweezer pack + core sharpenings | Build the **`tweezer` modality pack** (Thompson instance, **grounded in arXiv 2506.13724**): X/Y AOD axes = **32-tone mux** generators, moves = a new **`PolyChirp`** envelope (5th-order freq + 3rd-order amp; pack-registered, §6.1/D22), **`ImagingReadout`** (camera + frame grabber) + occupancy discriminator, **host-in-loop rearrangement** (PC→tProc, ~25 ms); transport via Piccolissimo `TweezerTransportProblem` → `from_solution`. **Three `core` additions confirmed** (§17): (a) external-detector readout, (b) persistent-gen + multi-detection lifecycle, (c) host-in-the-loop feedback. Targets a **Thompson custom QICK firmware variant** (polynomial-chirp DDS); remaining details need a Thompson requirements pass. |
| D21 | Namespace binding | **Topology-derived namespaces (§5.2).** The pack maps element *types* → ops classes (`Transmon→TransmonOps`, `BosonicMode→ModeOps`, `Coupler→CouplerOps`); `Device.load` binds one namespace per **named element instance** — `dev.qubit`/`dev.manipulate` exist because the *instance* names them. Multi-qubit (`dev.q0`/`dev.q1`) and non-Schuster naming fall out free; `dev.sideband` is an instance-declared alias; no instance name appears in pack code ("manipulate" was Schuster jargon leaking into the pack). |
| D22 | Envelope IR | **Typed envelope union (§6.1)** replaces the flat `Shape` enum + shape-specific `Pulse` fields: `Gaussian`/`FlatTop`/`Const`/`Arb` frozen dataclasses; the compiler lowers via a **type-keyed registry**, so packs register new envelopes + lowerings (`PolyChirp` lives in the `tweezer` pack — core stays stock-v2). Fixes a real gap (PolyChirp had no coefficient fields in the IR) and dedup keys on the frozen envelope (R6). |
| D23 | Gain units in the IR | **`Pulse.gain` carries the v2-native normalized fraction.** The **cqed factories** convert legacy int codes (`gain_frac = gain_int/maxv`, maxv from the soccfg snapshot) at the pack boundary; the calib DB keeps storing ints. Core and the raw-qick escape hatch speak one unit; physical→fraction stays confined to the §6.5 import boundary; sweep axes stay lab-native at the surface (§6.1). *(Supersedes the earlier int-codes-in-the-IR lean.)* |
| D24 | Environment / station seam | **Model the station's read-only seam, never fridge control (§18):** `core/environment.py` `EnvironmentSource` protocol; the Runner stamps `MeasurementResult.meta["environment"]` at acquire boundaries; optional per-line **signal-chain metadata** in the wiring. Fridge-vendor integration (Maybell-class) = a config profile + a thin adapter, not a fork. Gating, drivers, and any control-plane deferred. |
| D25 | qick-v2 API confirmed (Sho/Fermilab) | **R2 resolved by Sho Uemura's IntonatoQICK PR #5 review (2026-07-13).** Five facts fold in (all *confirm* strumento's existing design; ref: his [`qick_demos_sho` notebook](https://github.com/meeg/qick_demos_sho/blob/main/tprocv2/00_intro.ipynb)): (1) **subclass `AveragerProgramV2`, override `_initialize()`/`_body()`** (not append-to-object) — strumento §8 already does this, and doing it in Python sidesteps his "doable in Julia?" doubt; (2) **`dump_cfg()` → `QickConfig(json)` builds a program with NO board** — the soccfg snapshot (§3) is obtained this way, so the numeric-golden gate (§10.2) runs offline/CI for real, not just shape-mocked; (3) **multiple pulses per generator + const/flat-top envelopes** (arb-only wastes memory) — this is D22's envelope union exactly; (4) **sweeps need `add_loop`; `reps=1` is single-shot** — strumento §8 already; (5) **sample rates come from the config, not hand-passed** (`dac_rate`/`adc_rate` args were a confusion) — strumento reads them from the soccfg snapshot. strumento is also the clean answer to his PR-#5 program-authoring question (his option 3/4 = a Python-side `AveragerProgramV2` subclass = `StrumentoProgram`); IntonatoQICK's Julia backend delegates to strumento rather than assembling inline (§16.4). |
| D26 | Device model as a measured artifact (codesign loop-closer, §19) | strumento is a **device-model measurement instrument**, not only a runtime. Define a structured, serializable **`DeviceModel`** (element params — freqs/anharmonicities/couplings/T1/T2 — plus per-line transfer functions and a crosstalk matrix). A `Device` declares `source: design \| measured \| hybrid`; the Analyzer emits a **`DeviceModelReport`** = the measured `DeviceModel` + `diff(design, measured)` + provenance/timestamp/environment stamp (ties D24). Every characterization run updates the measured model — so the calibration path *is* the device-characterization path, and the exported model is the artifact a design → fab → characterize → redesign loop consumes. |
| D27 | Board-description → `Wiring` (Layer 3, §19) | A neutral, typed **board-description schema** (Pydantic: channel roles + generator→DAC/ADC bindings + RF-periphery annotations feeding the §18 `SignalChain`) and a `Wiring.from_board_description(desc)` builder. Extends single-source-of-truth (§5, §7) *down to the physical board*: the board description is the wiring config's upstream source; a vendor netlist export is **one adapter/producer**, never a coupling. |
| D28 | Transfer-function-aware compilation (§19) | The board/line is a **linear filter** between DAC and qubit; in-band linear distortion is **pre-compensable in software**. The `SignalChain` (§18) carries a per-line transfer function **H(ω)** (measured via characterization, or modeled from the board description) + inter-line **crosstalk**; the `from_solution` import boundary (§6.5) gains an optional **pre-distortion stage** (regularized deconvolution with out-of-band / DAC-clip guards) so H·ũ ≈ u at the device. Sits alongside the resample + physical→fraction gain-calib the boundary already owns. |
| D29 | Simulator soc backend (§19) | A **`SimulatorSoc`** implementing the same soc protocol as `MockQickSocV2` / `PyQickSoc`, running a `Device` + `Seq` against a physics simulation, so a **design-time `Device`** (`source: design`) is exercisable *before it is fabricated*. The two ends of the codesign loop — simulation (design-time) and board (as-built) — share one `acquire` interface; the same experiment/fitters run against both. Design-for-extension seam (deferred build). |
| D30 | Device authoring — builder vs config (§5.3) | **Config-as-data stays canonical; add a `DeviceBuilder` (config-as-code) that PRODUCES it, never replaces it (landed 2026-07-21).** A pure, **pack-agnostic** builder in `core` composes a device from the *loaded pack's registered element types* — `add(type, name)` / `coupling` / `wire` / `readout` — with construction-time topology validation; `build()` emits the canonical Pydantic `Device`/`Wiring`, and `Device ⇄ YAML/JSON ⇄ builder` all yield the identical model (three equal authoring surfaces, one source of truth). Motivated by DRY (the 7 storage modes = a loop), programmatic / design-time generation (D29 `source: design`), and early validation — without forking single-source-of-truth (D10) or the pack/instance boundary (D20). Same *"producer of canonical data"* shape as `from_board_description` (D27) / `from_solution` (§6.5); packs may ship domain builder sugar (parametric sub-assemblies + connectivity patterns) as plain functions over the builder, so a new modality gets authoring for free. **Carve-out: calibration *values* are never authored in code** — they change every run (written by the calibration loop / `DeviceModelReport`, D26) and stay data, passed to `build(calib=…)`; the builder composes topology + wiring only. |
| D31 | Typed-edge couplings + Hamiltonian export (§5.3, §6.6) | **Couplings are first-class typed edges, and the device model exports a Hamiltonian for Piccolo (landed 2026-07-21).** (a) A coupling = `coupling(a, b, form, mediated_by=…)`: an operator **form** (pack-registered union à la D22 — `Dispersive` / `ZZ` / `Exchange` / `BeamSplitter` / `JaynesCummings`; `spin` → `HeisenbergExchange`), endpoints, an optional mediating driven element, strength from calib (D30 carve-out). The form carries symmetry + an **operator template**; the physical *coupler element* (`add("Coupler", …)`, has a DAC) is separate from the *interaction*; patterns (`ring_couple`, `grid_nn`) are functions emitting typed edges. (b) Because forms carry operator templates and element types carry bare-Hamiltonian templates, `dev.to_operator_model(frame, cutoffs, include=…)` lowers topology + couplings + calib → a **framework-neutral** operator spec (drift $H_0$ + drive ops $\{H_k\}$ + basis + `drive_map` + provenance); `include=` extracts a **subgraph** for tractability. A thin **upstream** `to_piccolo_system(model)` bridge builds the Piccolo `QuantumSystem` (strumento `core` stays Piccolo-free; the neutral middle also serves QuTiP). This is the **mirror of `from_solution` (§6.5)** — model-out ↔ pulse-in — closing the codesign loop (D26 `DeviceModelReport` refines the model → next solve). §19.4 boundary preserved: strumento owns the *export*, the *solve* stays upstream. Required inputs (not derivable): frame, Fock cutoffs, computational-subspace / target. |

## 3. Package architecture (Approach A)

Dependency arrows point **down**; only `core/compiler`/`core/program`/`core/mock` import qick.

**Three layers (§5.1, §15):** modality-agnostic **`core`** → reusable **modality `packs`** (physics
kits: element *types* + libraries + readout kind + calibration *schema*) → **device *instances*** (a
topology of elements + wiring + calibration *values*, config, one per physical fridge, versioned
§14.1). A pack is a *kit of parts* for a physics class; a device is *one assembly* of those parts.
Three packs in the map: **`cqed`** (superconducting transmon+bosonic — Stanford, §1), **`tweezer`**
(neutral-atom AOD — Thompson, §17), **`spin`** (quantum-dot — spinQICK-modeled, §15.5).

```
strumento/
  core/                       ─────────────── modality-AGNOSTIC (no physics knowledge) ───────────────
    wiring.py       Wiring: line/generator → DAC binding (Pydantic; mux/mux4_v2 tone combs;
                      optional per-line signal-chain metadata §18)                                 (pure)
    pulses.py       Envelope union (Gaussian/FlatTop/Const/Arb — pack-extensible, D22) +
                      Pulse / Delay / Raw / Segment / Measure / Seq builder + from_solution        (pure)
    sweeps.py       Sweep axis → QickSweep1D lowering (incl. tone chirps)                         (pure)
    readout.py      ReadoutKind + Discriminator PROTOCOLS + data-source abstraction:              (pure)
                      QICK-ADC IQ  OR  external trigger-synced detector (camera) — §17 finding (a)
    program.py      StrumentoProgram(AveragerProgramV2): _initialize/_body, play/play_seq, acquire; (qick)
                      persistent/background generation + multiple detection events — §17 finding (b)
    feedback.py     feedback protocol: in-sequence branch (active_reset) OR host-in-the-loop      (pure)
    environment.py  EnvironmentSource protocol — read-only station telemetry (fridge etc., §18)   (pure)
    compiler.py     lower a Seq/Pulse onto a live AveragerProgramV2; quantize()                   (qick)
    result.py       MeasurementResult typed seam                                                  (pure)
    fitters.py      generic lmfit fit primitives (decaying sinusoid, exponential, …)              (pure)
    device.py       Device (topology of elements) + load(): compose pack elements + wiring + calib (pure*)
    builder.py      DeviceBuilder: config-as-code that PRODUCES the canonical Device/Wiring (§5.3, D30)  (pure)
    mock.py         MockQickSocV2 no-FPGA harness (+ mock external detector)                       (qick)
  packs/                      ── modality kits: element TYPES + libraries + readout kind + calib SCHEMA ──
    cqed/                     ─── superconducting (transmon + bosonic) — build, Stanford ───
      elements.py   element TYPES: Transmon / BosonicMode / Coupler / ReadoutResonator → lines    (pure)
      calibration.py  Pydantic calib SCHEMA + swap-dataset accessors (get_pi/poly models)          (pure)
      library/      transmon(ge/ef) · sideband(f0g1) · mode(displace) · coupler(swap) · composite —
                      element-TYPE keyed; namespaces bound per named instance element (§5.2, D21)   (pure)
      readout.py    DispersiveIQ readout kind + discriminator                                      (pure)
      experiments/  Rabi / T1 / T2 / parity / active_reset                                         (pure)
    tweezer/                  ─── neutral-atom AOD — build, Thompson (§17) ───
      elements.py   element TYPES: AtomArray / AOD-axis (mux, ≤32 tones); site↔tone               (pure)
      library/      set_array / move (PolyChirp: pack-registered envelope + lowering, D22) /
                      rearrange · readout.py ImagingReadout                                         (pure)
    spin/                     ─── quantum-dot — spinQICK-modeled, build-gated on a user (§15.5) ───
      elements.py   element TYPES: Dot / Plunger / Barrier                                         (pure)
      library/      LD + exchange-only pulses · readout.py PSB / charge-sensing                    (pure)
  devices/                    ── device INSTANCES: topology + wiring + calibration VALUES (config, per fridge, §14.1) ──
    schuster_multimode/  1 Transmon + 1 Manipulate + 7 Storage + Coupler + Readout   (instance of cqed)
    thompson_tweezer/    X/Y AOD (32 tones/axis) + SLM static + camera               (instance of tweezer)
    …                    transmon_chip / transmon+multi-bosonic / N-dot spin → free topologies of a pack
  tests/
```

`*pure` = importable and unit-testable with **no qick, no board** — the bulk of the logic. qick is
confined to `core/compiler`, `core/program`, `core/mock`. **Config-layer types (`wiring`,
`device`, pack `calibration`) are Pydantic models** — typed validation *and* clean JSON round-trip,
which the job-server config blob (§14.2) needs. The hot-path IR (`Pulse`/`Delay`/`Seq`) stays plain
dataclasses (runtime objects, not serialized config).

> **`soccfg` and purity.** `Device` holds the **JSON soccfg snapshot** (a plain dict — pure, no qick
> import), *not* a live `qick.QickConfig`. The snapshot is produced on a board **once** via
> `QickSoc.dump_cfg()` and committed; the **compiler/program** reconstructs a `QickConfig` from it with
> `QickConfig(json)` — **no board required** (confirmed by Sho, D25) — and `quantize()` (§6.1) takes the
> built `QickConfig`. So `device.py` stays qick-free; only the qick-facing modules touch `QickConfig`,
> and the *entire* program build/compile path is exercisable offline against the snapshot.

Each unit answers "what does it do / how do I use it / what does it depend on" in isolation. The
Pulse IR is the spine: experiments never touch qick; only the compiler does.

## 4. Tenet 0 — every abstraction has an escape hatch

`StrumentoProgram` **subclasses** `AveragerProgramV2`; the device layer only *adds* helpers, never walls
qick off. Three escape levels, always available:

1. **Raw op mid-sequence** — `.raw(lambda p: p.delay_auto(0.01))` splices any qick call into a `Seq`
   (invoked with the live program `p` at that point).
2. **Raw method on the program** — inside a hand-written `_body`, `self.<any AveragerProgramV2 method>`.
3. **Bypass the builder entirely** — hand-write `_body` with raw qick while still calling library
   factories for the pulses you don't want to hand-roll.

> Design rule: no abstraction ships without one of these. "You must be able to hack one thing at the
> very bottom and revert to raw qick at any moment while still using the abstractions."

## 5. Tenet 1 — the device model (model the device, not the FPGA)

### 5.1 Pack vs. device — modality *kit* vs physical *instance*

"Device" is **two things**, and separating them is what makes the map generalize:

- A **modality pack** (`packs/<modality>/`, code, reusable) supplies **element *types*** (what building
  blocks exist — `Transmon`, `BosonicMode`, `Dot`, `AODaxis`, …), their **pulse libraries**, the
  **readout kind**, **experiment types**, and the calibration **schema**. One pack per physics class.
- A **device instance** (`devices/<name>/`, config, per fridge, versioned §14.1) is a **topology** of
  element *instances* (how many of each, connectivity) + **wiring** (each element's drive → DAC) +
  calibration **values**. **Generated individually for each actual device.**

So *transmon*, *transmon+bosonic*, *multi-transmon*, and the Schuster *"multimode"* system are **not
separate packs** — they are different **topologies (device instances) of the one `cqed` pack**, which
provides the `Transmon`/`BosonicMode`/`Coupler`/`ReadoutResonator` element types. Schuster =
`1×Transmon + 1×BosonicMode(manipulate) + 7×BosonicMode(storage) + Coupler + Readout`. A pure transmon
chip = `N×Transmon + Couplers + Readouts`, no bosonic. Both load the same `cqed` pack; only the
instance config differs.

`Device.load(instance_config)` composes the pack's element types per the instance's topology, binds the
wiring, and attaches calibration values. `dev` is that assembled object; the rest of §5 is how one
element resolves to hardware.

### 5.2 The device language and wiring (per resolved element)

**Layer 1 — what you write (device language, zero DACs):**

```python
dev.qubit.ge_pi(gain=amp)      # π on qubit g↔e
dev.qubit.ef_pi()              # π on e↔f
dev.sideband.f0g1(photon=0)    # |f,0⟩↔|g,1⟩ exchange (qubit ↔ manipulate cavity)
dev.coupler.swap("M1-S3")      # beamsplitter swap manipulate ↔ storage 3
dev.manipulate.displace(...)   # direct cavity drive
dev.readout.measure()
dev.parity(mode="manipulate")  # composite → a sub-sequence
dev.prepare(Fock(1))           # state prep → analytic OR optimal-control waveform (§6.5)
```

**Namespaces are derived from the instance topology, not hardcoded in the pack (D21).** The pack maps
element *types* → ops classes — `Transmon → TransmonOps` (ge/ef rotations), `BosonicMode → ModeOps`
(displace, …), `Coupler → CouplerOps` (swaps) — and `Device.load` binds one namespace per **named
element instance**. `dev.qubit` / `dev.manipulate` exist above because the *Schuster instance* names
its elements `qubit` / `manipulate`; a lab whose yaml says `cavity` gets `dev.cavity`, and a
multi-qubit chip gets `dev.q0` / `dev.q1`. No instance name appears in pack code — "manipulate" is
Schuster jargon, and the earlier five-fixed-namespace design leaked it into the pack. `dev.sideband`
(f0g1 / multiphoton ladder) is an **instance-declared alias namespace** — a generator-role grouping
bound to the transmon's sideband ops; it is distinct from `dev.qubit` because it is a distinct
generator, though it enters the same port. *(This also resolves the earlier `qubit.f0g1()` vs
`sideband.f0g1()` open point: the instance declares the alias.)* Device-level composites
(`dev.parity` / `dev.prepare` / `dev.gate`) stay on `dev` and take element names as arguments
(`mode="manipulate"`).

**Layer 2 — the wiring (plain config data, `wiring.py`, the ONLY channel map).**

Built from the v1 `hardware_config.yml → hw.soc.dacs/adcs` (this is the **v1 snapshot**; the target
v2 board's map is regenerated from its own config at implementation time — R2). A **line** owns one or
more **generators** (role → DAC); one line can have several DACs:

```
LINE        GENERATOR(role → DAC)                       notes
qubit       drive    → DAC 2 (nyq 1, full)              ge/ef rotations
            sideband → DAC 0 (nyq 1, full)              f0g1 / multiphoton ladder — a DIFFERENT DAC,
                                                          combined into the qubit port (same LINE)
coupler     low/high → DAC 1 (nyq 1, full)              beamsplitter swaps; band-gated in general,
                                                          but flux_low.ch == flux_high.ch == 1 HERE,
                                                          so both bands collapse to one generator
manipulate  main     → DAC 3 (nyq 2, full)              direct cavity drive
storage     main     → DAC 6 (nyq 2)                    direct storage drive (rarely used) — see gen_type note
readout     dac      → DAC 5 (full) + ADC 0             (a second ADC `cavity_out` = ch 1 exists)
```

- **There is no separate "sideband" *line*.** `sideband` is a **generator role under the qubit line**
  (DAC 0), distinct from the qubit `drive` generator (DAC 2) but feeding the same physical port. The
  Layer-1 `dev.sideband.*` namespace (§5, above) is an *API grouping*; its factories emit
  `LineRef("qubit", "sideband")`. "Different DAC" and "same line/port" are both true and consistent.
- **Coupler band-gate:** the v1 code picks flux_low vs flux_high by `freq < 1800`, but in this snapshot
  both resolve to DAC 1, so the choice is a no-op *here*. The schema still models the low/high split as
  two band-gated generators (a different board may separate them); when they collapse to one DAC, the
  compiler declares it once and both bands route to it.
- **Storage `gen_type` — a YAML-vs-firmware discrepancy to resolve.** `hardware_config.yml` records
  `storage_in: type: full, nyquist: 2`, but the v1 `MM_base` comment ([:275-279]) reports the actual
  firmware block is `axis_sg_mux4_v2` (which `declare_gen` rejects without `mixer_freq`/`mux_freqs`),
  and v1 therefore *excludes* storage from auto-declare. The schema supports `gen_type="mux4_v2"` as a
  capability; the correct value for the target v2 board must be read from its config (R4), not assumed.

**Wiring schema** (`wiring.py` — the module implementing "model the device, *not the FPGA channels*"
shouldn't itself be called `channels`) — this is the seam an implementer copies, so it is typed, not prose:

```python
@dataclass(frozen=True)
class Generator:
    role: str                       # "drive" | "sideband" | "low" | "high" | "main" | ...
    dac_ch: int
    nyquist: int = 1
    gen_type: str = "full"          # "full" | "int4" | "mux4" | "mux4_v2"
    band_hz: tuple[float,float] | None = None   # freq gate; None = always eligible
    mixer_freq: float | None = None             # required for int4 / mux*
    mux_freqs: list[float] | None = None        # required for mux4 / mux4_v2 (e.g. storage)

@dataclass(frozen=True)
class Line:
    name: str                       # instance-chosen names: "qubit" | "coupler" | "manipulate" | "storage"  (NOT "sideband")
    generators: list[Generator]     # >1 when band-gated (coupler low/high) or multi-source
                                    #   (qubit line owns drive@DAC2 + sideband@DAC0)
    chain: SignalChain | None = None  # OPTIONAL per-stage attenuation/amp chain metadata (§18)

@dataclass(frozen=True)
class ReadoutWiring:                 # named to avoid colliding with core/readout.py (ReadoutKind protocols)
    dac_ch: int; adc_chs: list[int]
    mux_freqs: list[float] | None = None        # freq→lane map for multiplexed readout
    lengths_us: list[float] = ...; thresholds: list[float] = ...; trig_offset: int = ...

class LineRef(NamedTuple):           # what a Pulse carries
    line: str; role: str            # e.g. ("qubit","drive"), ("qubit","sideband"), ("coupler","low")

class Wiring:
    lines: dict[str, Line]; readout: ReadoutWiring
    def resolve(self, ref: LineRef, freq_mhz: float) -> Generator:
        """Pick the generator for (line, role, freq): role match + band_hz gate.
        Raises if no eligible generator (e.g. a swept freq leaves the band — see §6.4)."""
```

The compiler's declare step reads `Generator` to emit `declare_gen(ch, nqz, mixer_freq=…,
mux_freqs=…)` — including the `mux4_v2` storage args, so the R4 quirk is handled by data, not a comment.

> **Config types are Pydantic.** `Generator`/`Line`/`ReadoutWiring`/`Wiring` are shown as dataclasses for
> brevity but are implemented as Pydantic `BaseModel`s — typed validation **and** clean JSON round-trip
> for the §14.2 job-server config blob (matching spinQICK's Pydantic config, §15). `LineRef` stays a
> lightweight `NamedTuple`: it rides on the runtime `Pulse`, not the serialized config.

Consequences:
- **f0g1 = a second generator on the qubit line.** `QubitLine` has `{drive, sideband}`; `ge_pi` uses
  `drive`, `f0g1` uses `sideband` — same line, different generator.
- **The `freq<1800`/`freq<1000` magic thresholds die** — they become the coupler's
  generator-selection rule in the wiring, where "which DAC realizes this frequency" belongs.
- **`channel_table` and `channel_assign` are deleted**; all code reads the wiring.
- **Re-patch a cable / move a drive to a different DAC = edit one line in Layer 2.** Every experiment
  and pulse factory is untouched — they only speak Layer-1 device language.

A `Pulse` targets a `(line, generator-role)` — resolved to a DAC by the wiring at compile — never a
bare integer.

### 5.3 Authoring a device instance — the `DeviceBuilder` (config-as-code, D30)

A device instance (§5.1) is **canonically data** — the serialized Pydantic `Device`/`Wiring` is what the
job server ships (§14.2), what versioning tracks (§14.1), and what the codesign loop diffs (§19). But
hand-writing YAML is repetitive (the Schuster instance alone is 7 storage modes) and forecloses
programmatic and design-time generation (`SimulatorSoc`, D29). So `core/builder.py` adds a **pure,
pack-agnostic `DeviceBuilder` that *produces* the canonical model — never a parallel state store.** Three
authoring surfaces, one source of truth, freely interconvertible:

```
Device (Pydantic)   ⇄   YAML/JSON   ⇄   DeviceBuilder
```

`Device.from_yaml(...)`, a hand-built Pydantic `Device`, and `DeviceBuilder(...).build()` all yield the
**identical** `Device`; `dev.to_yaml()` writes it back. The builder is the same *"producer of canonical
data"* shape the design already uses for `Wiring.from_board_description` (D27) and `from_solution` (§6.5)
— code that lowers into the data model, so **single-source-of-truth (D10) is preserved**: `build()`'s
output (committed, or deterministically re-emitted from the same inputs) is the one authority.

**Pack-agnostic by construction — this is the extensibility.** The builder never names `Transmon` or any
modality type; it composes over the **loaded pack's registered element types** (§5.1; entry-point
discovery, §15.3). So `spin` / `tweezer` — and any third-party pack — get device authoring **for free**,
with no `core` change:

```python
b = DeviceBuilder(pack="cqed")             # element types resolved from the pack registry
b.add("Transmon", "qubit")
b.add("BosonicMode", "manipulate")
for i in range(1, 8):
    b.add("BosonicMode", f"storage{i}")    # the loop YAML cannot express
b.add("Coupler", "coupler")                            # the physical, driven coupler element (has a DAC)
b.coupling("manipulate", "storage3", form=BeamSplitter, mediated_by="coupler")   # a TYPED edge (below)
b.wire("qubit", drive=Generator(dac_ch=2), sideband=Generator(dac_ch=0))
b.readout(dac_ch=5, adc_chs=[0], mux_freqs=[...])
dev = b.build(calib="devices/schuster_multimode/calib.yml")   # topology+wiring from code; calib from DATA
dev.to_yaml("devices/schuster_multimode/")                    # data remains canonical
```

**Couplings are typed edges — not just "there's a coupler here" (D31).** A physical coupler with its own
drive is an *element* you `add` (`dev.coupler.swap()` still works); an *interaction* is a separate,
first-class **typed edge** — `coupling(a, b, form=…, mediated_by=…)` — carrying an operator **form**, its
endpoints, an optional mediating element, and (from calib) a strength. **Coupling forms are a
pack-registered union**, exactly the D22 envelope pattern applied to interactions: `cqed` registers
`Dispersive` / `ZZ` / `Exchange` / `BeamSplitter` / `JaynesCummings`, `spin` registers `HeisenbergExchange`,
a new pack adds its own with **no `core` change**. The form carries its **symmetry** (`BeamSplitter` / `ZZ`
symmetric; `CrossResonance` directional, control→target) and — crucially — its **operator template**, which
is what makes the model Hamiltonian-extractable (§6.6). Strengths ($g$, $\chi$, $J$) are calibration
*values* → data (the D30 carve-out), never in the edge declaration. `mediated_by` ties the edge to a driven
element (same physics as `dev.<coupler>.swap()`); omit it for an always-on direct coupling.

**Composition algebra — elegance via plain Python.** Parametric sub-assemblies and connectivity patterns
are just functions over the builder, so complex topologies compose from small reusable pieces instead of a
monolithic schema:

```python
def transmon_unit(b, name, *, drive_dac, ro_lane):   # a reusable unit
    b.add("Transmon", name)
    b.wire(name, drive=Generator(dac_ch=drive_dac))
    b.readout_lane(name, lane=ro_lane)

b = DeviceBuilder(pack="cqed")
for i in range(4):
    transmon_unit(b, f"q{i}", drive_dac=i, ro_lane=i)
ring_couple(b, [f"q{i}" for i in range(4)], form=ZZ)   # a pattern → emits typed edges
dev = b.build(calib="devices/ring4/calib.yml")
```

A pack **may ship domain builder sugar** (`transmon_unit`, `aod_axis`, …) as ordinary functions over the
builder — discovered like the rest of the pack — so patterns live in the pack / userland, not `core`.

**Validation layers, and the naming → namespace tie.** Pydantic validates the *data* (field types,
required fields); the builder adds *construction-time topology* checks with early, specific diagnostics
(`coupling` rejects an unknown endpoint or missing mediator; `readout` requires ≥1 target; a `wire` role must exist on that
element). The `name` you `add` is exactly what binds the `dev.<name>` namespace (D21), so the builder
literally authors the topology D21 consumes.

**Carve-out — calibration values are never authored in code.** The builder composes **topology + wiring
only**; calibration *values* change every run (written by the calibration loop / `DeviceModelReport`, D26)
and stay **data**, passed to `build(calib=…)` as a file. This keeps the one place a device mutates most in
the diffable, versioned form and prevents a split-brain source of truth.

**Escape hatch (Tenet 0).** The builder is optional at every level: hand-write the YAML, or construct the
Pydantic `Device` directly, or use the builder and then hand-edit the emitted data. No path is privileged
over the canonical model.

## 6. The pulse layer

### 6.1 Pulse IR (`pulses.py`) — typed, physical-unit, sweep-aware

```python
# Typed ENVELOPE union (D22) — each shape is a small frozen dataclass carrying its OWN
# parameters; the compiler lowers via a TYPE-KEYED REGISTRY, so a pack can add an
# envelope + its lowering without touching core. PolyChirp (freq-poly deg 5 + amp-poly
# deg 3, §17.2) is registered by the TWEEZER pack with its custom-firmware lowering —
# core never names a shape stock v2 cannot play.
@dataclass(frozen=True)
class Gaussian: sigma_us: float
@dataclass(frozen=True)
class FlatTop:  ramp_sigma_us: float; length_us: float | Sweep    # flat portion (sweepable)
@dataclass(frozen=True)
class Const:    length_us: float | Sweep
@dataclass(frozen=True)
class Arb:      idata: np.ndarray; qdata: np.ndarray | None = None  # NORMALIZED [-1,1] (× gain at
                                                                    # lowering) — computed by a
                                                                    # factory OR imported (§6.5)
Envelope = Gaussian | FlatTop | Const | Arb    # open union — packs register more (tweezer: PolyChirp)

@dataclass
class Pulse:
    line: LineRef                       # (namespace, generator-role) — resolved to DAC at compile
    freq_mhz:  float | Sweep
    gain:      float | Sweep            # v2-NATIVE normalized fraction (D23) — see gain-units note
    phase_deg: float | Sweep = 0.0
    envelope:  Envelope = ...           # shape + its own typed params — no None-field conventions
    start_us:  float | None = None      # top level: absolute start (None ⇒ sequential/auto);
                                        #   inside a Segment: OFFSET from segment start (default 0)
    label: str | None = None

@dataclass
class Delay:   duration_us: float | Sweep
@dataclass
class Raw:     fn: Callable[["StrumentoProgram"], None]   # escape hatch element
@dataclass
class Segment: pulses: list[Pulse]      # pulses sharing a start (multi-line, same t)
@dataclass
class Measure: ...                      # readout op; carries a ReadoutKind (default: the device's) —
                                        #   what the builder's .measure() emits

Seq = fluent builder over [Pulse | Delay | Raw | Segment | Measure]
```

- **Simultaneity is first-class.** Real experiments (and every OC pulse, §6.5) drive **multiple lines
  at once**. A `Pulse.start_us` (absolute) or a `Segment` (pulses sharing a start) expresses this; the
  builder exposes `.play_parallel(x: list[Pulse] | Segment | Seq)` (a `Seq` is flattened — accepting the
  single-`Segment` `Seq` that `from_solution`/`prepare` return). The compiler lowers to v2's
  absolute/relative `t` on `pulse()`. Sequential `.play()` (the common case) leaves `start_us=None` and
  the compiler chains. Within a `Segment`, `start_us` is an **offset relative to the segment start**
  (default 0) — staggered overlaps are expressible; at top level it is absolute. **Type contract:** `from_solution`/`prepare`/`gate` all return a `Seq` (an OC
  import is a `Seq` wrapping one `Segment`); `.play()` takes a `Pulse` or `Seq`; `.play_parallel()` the
  union above.

- **Physical units throughout.** All `freq2reg`/`us2cycles`/`deg2reg` conversion moves *out* of
  config-parsing and *into* the compiler — the single biggest v1 simplification.
- **Gain units — a real v1→v2 semantics change (do not skip).** v1 `set_pulse_registers`/`add_pulse`
  take an **integer DAC gain code** (0..32766) and the existing calibration DB stores these ints
  (e.g. `pi_ge.gain = 3289`); QICK **v2** `add_pulse` takes a **normalized float** fraction of full
  scale (≈ −1..1). Decision (**D23**): **`Pulse.gain` carries the v2-native fraction** — the core IR
  and the raw-qick escape hatch speak *one* unit (a `.raw()` user sees the same numbers the IR
  carries), and no lab's legacy unit is baked into the modality-agnostic core (spin/tweezer users and
  stock v2 think in fractions). The **cqed factories convert at the pack boundary**: the calib DB
  keeps storing what the lab measures (**int codes**, DB continuity preserved) and the factory
  computes `gain_frac = gain_int / maxv` (`maxv` from the soccfg snapshot for the resolved generator —
  verified against the target qick's `maxv` convention; distinct from the signed-int16 register bound
  ±32767 used in `expt_service/gain.py` `MAX_GAIN`). A pack unit test asserts a known int code maps to
  the intended fraction. *(Supersedes the earlier int-codes-in-the-IR lean, which created a two-unit
  world inside strumento itself. Verify the exact v2 convention against the qick version Sho targets.)*
  - **Three input regimes, one IR unit.** The **cqed factories** take int codes (lab-native,
    converted at the boundary); the **OC / Piccolo import path (§6.5) takes physical units**
    (GHz-Rabi / rad·s⁻¹, Piccolo's native output) and calibrates **directly to fraction** via the
    π-pulse reference (no int-code hop — same physical→gain adapter as the `expt_service` contract);
    **tweezer / spinQICK-style packs are fraction-native already**. All produce the same
    `Pulse.gain` fraction; unit adaptation lives at the boundaries, never smeared across the library.
  - **Sweep axes stay lab-native at the surface.** A gain sweep entered in int codes
    (`ge_pi(gain=Sweep("amp", 0, 30000, 51))`) is converted to a fraction sweep by the factory; the
    experiment records the **user's original axis** for `xpts`/plots, so existing notebooks keep
    reading familiar units.
- **Quantization is queryable, IR stays pure.** `quantize(pulse, qickcfg) → Realized(freq_mhz_actual,
  length_us_actual, gain_frac, cycles, …)` gives exactly what the DAC will play (for phase/evolution-time
  math) without polluting the requested-value IR. It is a side-effect-free query that *takes* a built
  `QickConfig` (from the compiler layer, per the purity note in §3) — the IR itself never holds one.

### 6.2 Pulses are computed factory functions, not a data bank (`library/`)

Each pulse type is a typed function that *computes* its parameters from calibration — no string keys,
no `eval`. Factory namespaces are bound to `dev`.

**Uniform override surface (required for sweeps).** Every factory accepts keyword overrides for
*every* physical field — `freq`, `gain`, `phase`, `length` — where `None` ⇒ take the calibrated value
and a scalar-or-`Sweep` ⇒ override. This is what lets `dev.qubit.ge_pi(freq=f, gain=a)` (a chevron)
type-check: any field a factory produces can be swept.

```python
Num = float | Sweep | None      # None ⇒ from calibration; scalar/Sweep ⇒ override

# library/transmon.py → TransmonOps, bound per named element (§5.2/D21; Schuster instance: dev.qubit.*)
def ge_pi(dev, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse: ...  # Gaussian, calib.qubit.pi_ge
                                                                               # (gain= accepts int codes; factory converts to fraction, D23)
def ge_hpi(dev, *, freq=None, gain=None, phase=0.0) -> Pulse: ...
def ef_pi(dev, *, freq=None, gain=None, phase=0.0) -> Pulse: ...

# library/sideband.py → sideband ops, bound via the instance alias (§5.2/D21; Schuster: dev.sideband.*)
#                        (identity args positional; physical overrides keyword-only)
def f0g1(dev, kind="pi", photon=0, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse: ...  # DEFAULT f0g1
def multiphoton(dev, transition, kind="pi", photon=0, *, freq=None, gain=None, phase=0.0) -> Pulse:  # ladder

# library/coupler.py  → CouplerOps (Schuster: dev.coupler.*)
def swap(dev, mode="M1-S1", kind="pi", *, freq=None, gain=None, phase=0.0, length=None) -> Pulse: ...  # ds_storage
def floquet_swap(dev, mode, pi_frac=1.0, *, freq=None, gain=None, phase=0.0) -> Pulse: ...             # ds_floquet

# library/composite.py → dev.parity(...), dev.prepare(...), dev.gate(...), dev.storage_write(...)
def parity(dev, mode="manipulate", *, second_phase=180, fast=False) -> Seq: ...  # hpi · wait(revival) · hpi
def prepare(dev, target, *, via=None) -> Seq: ...   # state prep; target=Fock(1)/GKP("Z")/…
def gate(dev, op, *, via=None) -> Seq: ...          # gate; op="CZ" / Unitary(U)
```

- **Address OC pulses by operation, not method.** `dev.prepare(Fock(1))` / `dev.gate("CZ")` replace v1's
  `optimal_control("fock","1")`. The realization — an **analytic sequence** (multiphoton ladder) *or* an
  **optimal-control waveform** (§6.5) — is the device's choice; `via="oc"`/`via="analytic"` forces it.
  Addressing by *what it does* is what makes analytic and synthesized pulses interchangeable.
- **"default f0g1 vs `f0g1_multiphoton` are different functions"** → `f0g1()` is the common case,
  `multiphoton(transition=…)` the general one; each with its own explicit signature, no overloaded string.
- **Adding to the library = writing one function.** No registration ceremony to *call* one (plain
  typed functions, IDE-discoverable). Optional `@pulse_factory` decorator only feeds a catalogue for
  `list_pulses()` introspection.
- **Arb factories may compute samples** (from a synthesis routine) or **import a solved pulse from the
  Harmoniqs stack** (§6.5) rather than load a bare `.npy` — the IR just holds normalized `idata/qdata`.

### 6.3 Composition — the fluent builder (`Seq`)

```python
seq = (Seq()
   .play(dev.qubit.ge_pi(gain=amp))
   .wait(0.5)
   .play(dev.sideband.f0g1())
   .play(dev.coupler.swap("M1-S3", phase=180))
   .play(dev.parity(mode="manipulate"))  # .play accepts a Pulse OR a sub-sequence
   .raw(lambda p: p.delay_auto(0.01))    # escape hatch
   .measure())
```

`Seq()` is a plain container (no `dev` — pulses are already device-resolved). Composing existing
library pulses into larger programs and adding pulses to the library are the two first-class daily
workflows. **Simultaneous multi-line play** uses `.play_parallel([...])` (a `Segment`) or explicit
`start_us` — e.g. `seq.play_parallel(dev.prepare(Fock(1)))` drives qubit + manipulate together (§6.5).

### 6.4 Sweeps and generator selection (edge cases the compiler must handle)

- **A frequency `Sweep` that straddles a generator band boundary is rejected.** A v2 generator is bound
  to one DAC for the whole program, so a coupler-swap frequency `Sweep` whose `[start, stop]` spans the
  low/high gate cannot switch DACs mid-loop. `Wiring.resolve` is evaluated over the sweep's full range;
  if `[start, stop]` maps to more than one generator, the compiler raises a clear diagnostic ("sweep
  spans coupler low/high boundary at f=…; split into per-band programs"). (In the v1 snapshot both bands
  are DAC 1, so this never triggers there — but a separated board would.)
- **Field-width / quantization limits are caught, not silently wrapped.** A swept `length` (or const
  length) exceeding v2's timing field width, or a step below DDS/cycle quantization, surfaces as a v2
  validator error via the mock (§10) — the same line a real board would fail on. The compiler does not
  clamp or wrap silently. (v1's long-pulse chunking — `register_long_pulse`/`play_long_pulse*` — is a
  known pattern to port here if long constant drives are needed; out of the minimal core.)

### 6.5 Optimal-control / Piccolo import boundary (D17) — the Piccolo × QICK seam

Optimal-control pulses come from **your stack** (Piccolo/Piccolissimo → `extract_pulse`; the
armonissima pulse catalog). This is a headline seam, not a corner case — so the import is a **typed,
lossless conversion**, replacing v1's `np.load(.npz)` + hand-`interp1d` + `maxv`-scale in `custom_pulse`:

```python
# core/pulses.py — modality-agnostic (a spin/atom pack imports its own solutions the same way)
def from_solution(
    solution,              # Harmoniqs pulse artifact: extract_pulse output / NamedTrajectory / catalog entry
    *, drive_map,          # named solution drive → strumento line ('qubit'→dev.qubit, 'cavity'→dev.manipulate)
    gain_calib,            # physical amplitude (GHz-Rabi / rad·s⁻¹) → v2 gain fraction, via π-pulse reference (D23)
    dac_rate,              # resample the control grid → DAC sample grid
) -> Seq: ...              # → Seq wrapping a Segment of normalized-ARB Pulses (one per line, same start) + provenance
```

What this nails that the `.npz` loader fudged:

1. **Drive→line mapping is explicit**, resolved through the wiring (§5) — not the hardcoded
   `qubit_ch`/`man_ch` the v1 opt_cont path assumed.
2. **Physical units → gain fraction via calibration** (§6.1/D23) — Piccolo controls are physical; this
   is *the* place physical-unit input belongs (straight to fraction, no int-code hop).
3. **Resampling** onto the DAC grid is owned here (typed `dac_rate`), not a buried `interp1d`.
4. **Multi-line simultaneity** → a `Segment` (§6.1), played with `.play_parallel(...)`.
5. **Provenance travels with the pulse** — source solve, target, fidelity, sample rate, units — feeding
   config versioning (§14.1) and reproducibility.

`dev.prepare(target)` / `dev.gate(op)` (§6.2) are the operation-addressed front doors; when they resolve
to an OC realization they call `from_solution(...)` under the hood. **The same path is the closed-loop
calibration hardware seam** (§16.1): an Intonato QILC iterate → PythonCall → `from_solution → acquire →
reduce → Vector{Measurement}`.

**In the minimal core (D17):** `from_solution(...)` importing an artifact you hand it, + `prepare`/`gate`
naming, + `Segment`/simultaneity. **Deferred:** **catalog auto-retrieval** — `dev.prepare(Fock(1))`
looking the solved pulse up in the armonissima catalog by *(device system, target)* and importing it
automatically. That closes the full Piccolo×QICK loop but belongs partly in the catalog layer; land it
once the import boundary is proven.

### 6.6 Device-model → Hamiltonian export (D31) — the Piccolo *input* seam (mirror of §6.5)

§6.5 imports a solved pulse *onto* the device; this is the **other direction** — the device model *exports
the Hamiltonian* a solver optimizes against. Once couplings carry operator forms (§5.3) and element types
carry bare-Hamiltonian templates, the topology graph + calibrated values **are** a Hamiltonian spec:

$$H(t) \;=\; \sum_i H_i^{\text{elem}} \;+\; \sum_{\langle i,j\rangle} H_{ij}^{\text{coupling}} \;+\; \sum_k H_k^{\text{drive}}\,u_k(t)$$

- $H_i^{\text{elem}}$ — the element *type*'s operator template, filled by calibrated params (`Transmon` →
  Duffing $\omega\,a^\dagger a + \tfrac{\alpha}{2}\,a^\dagger a^\dagger a a$; `BosonicMode` → $\omega\,b^\dagger b$,
  + optional self-Kerr).
- $H_{ij}^{\text{coupling}}$ — each typed edge's **form → operator**, scaled by its calibrated strength.
- $H_k^{\text{drive}}$ — from the **wiring**: each line/generator → the element + operator it drives (qubit
  drive → $a + a^\dagger$; the `sideband` generator → the $f_0g_1$ coupling operator). The $u_k(t)$ are
  exactly the controls Piccolo optimizes.

```python
# core — pure, framework-NEUTRAL, no qick, no Piccolo import
def to_operator_model(dev, *, frame="rotating", cutoffs, include=None) -> OperatorModel: ...
#   OperatorModel = drift H0 (sparse) + drives {line → H_k} + basis labels + drive_map + provenance
```

- **Subgraph extraction (`include=`) is the point.** The topology is a graph, so
  `include=["qubit","manipulate"]` pulls just the $f_0g_1$ subsystem instead of the full 9-mode Hilbert
  space — the line between a tractable and an impossible Piccolo solve. The Fock truncation also *defines*
  the computational subspace of the gate target (ties `EmbeddedOperator` / Pedersen fidelity).
- **Framework-neutral middle, upstream bridge.** `to_operator_model` emits a neutral spec (sparse operators
  + labels); a thin **`to_piccolo_system(model)`** adapter — living **upstream**, not in strumento `core` —
  builds the Piccolo `QuantumSystem`. This keeps `core` Piccolo-free and lets QuTiP (or any other consumer)
  use the same spec. Per §19.4, strumento owns the *export*; the *solve* stays upstream.
- **The loop closes coherently.** The same `drive_map` the model emits feeds `from_solution` (§6.5) on the
  way back, so labels round-trip:

  $$\texttt{to\_operator\_model} \to \text{Piccolo solve} \to \texttt{extract\_pulse}
  \xrightarrow{\text{drive\_map}} \texttt{from\_solution} \to \text{device} \to \text{measure}
  \to \texttt{DeviceModelReport (D26)} \to \text{refine model}$$

  so the *measured* device continuously sharpens the very Hamiltonian the next solve uses.
- **Required inputs (not derivable, hence arguments):** the **frame** per element (the model stores
  lab-frame freqs), **Fock cutoffs** per bosonic mode, and the **computational-subspace / target** definition.

**Sequencing.** The operator-template contract on element types + coupling forms is cheap and lands with the
pack; `to_operator_model` + the Piccolo bridge are a natural follow-on once the `cqed` templates exist
(candidate alongside the §19 codesign slice — not on the 1–6 critical path).

## 7. Config: single source of truth (versioning mechanism is §14.1)

### 7.1 Single-source-of-truth violations today (concrete)

| Same fact, multiple places | Evidence |
|---|---|
| f0g1 / multiphoton ladder defined twice | `hardware_config.yml → device.multiphoton` **and** `configs/multiphoton_config.yml` (loaded but its reader is commented out — dead duplicate) |
| `man()` ≡ `multiphoton('fn-gn+1')` | both read `device.multiphoton.pi['fn-gn+1']`; comments admit it |
| Channel map in code *and* config | `hw.soc.dacs.*` shadowed by `channel_table` + `channel_assign`, which contradict |
| Qubit ge freq in 3+ places | `device.qubit.f_ge`, `device.multiphoton.pi['gn-en'].frequency[0]`, `self.f_ge`/`self.f_ge_reg` |
| `flux_low.ch == flux_high.ch == 1` | low/high split is calibration-only; wiring collapses them — proves the DAC map must be *data* |

### 7.2 The single-source design (`device.py` + `calibration.py` + `wiring.py`)

One typed **`Device`**, loaded once, with:
- **`wiring` (Layer 2)** — the only channel map (§5).
- **`calib` (Layer 1)** — one *uniform* schema for "a calibrated operation," organized by family
  (qubit rotations, sideband ladder, storage swaps, manipulate, optimal-control, readout,
  active-reset). Each physical quantity appears **once**.
- **Rule: nothing derived is stored.** Register values computed at compile from soccfg; shared
  physical values (ge freq) *referenced*, never copied.

Result — **adding functionality is one edit each**: new pulse type → one calibration row (uniform
schema) + one factory; no `channel_table` edit (wiring is data), no `custom_pulse` branch (compiler
lowers shapes generically), no second config file. `multiphoton_config.yml` **deleted**; `man()`
**deleted**; the 169 KB `experiment_config.yml` keyed-by-classname blob becomes per-experiment typed
defaults co-located with each experiment.

### 7.3 Versioning — OPEN design decision (see §14.1)

Single-source-of-truth (§7.2) is settled; the *versioning mechanism* is **not**. The earlier lean
toward git-native reopens once the operational reality is admitted (single shared checkout, one
long-lived worker reading configs off disk, concurrent submitters — §14.2). `Device.load(...)` accepts
a version handle regardless of which mechanism wins; §14.1 lays out the tension for circulation.

## 8. `StrumentoProgram` interface

```python
class StrumentoProgram(AveragerProgramV2):          # IS a qick v2 program
    def __init__(self, device, seq=None, sweeps=(), reps=..., rounds=..., final_delay=...): ...

    def _initialize(self, cfg):
        self._auto_declare()                   # declare gens/readout + add envelopes referenced by
        for s in self.sweeps:                  #   seq/sweeps from wiring; manual declare still allowed
            self.add_loop(s.name, s.count)

    def _body(self, cfg):
        self.play_seq(self.seq)                # DEFAULT: run the declarative Seq

    # device-aware helpers
    def play(self, pulse_or_seq): ...          # lower onto self (add_pulse + pulse / delay / raw)
    def play_seq(self, seq): ...
    def measure(self, ...): ...
    def active_reset(self, ...): ...           # ⚠ wraps v2 feedback — see §11 risk
    def reset_and_sync(self): ...
    # + every AveragerProgramV2 method (escape hatch)
```

**Two `_body` modes:** declarative (hand it a `Seq`, never write `_body`) or imperative (subclass,
write `_body`, mix device helpers + raw qick).

**Sweeps as first-class values.** A `Sweep` is a named axis; any pulse/delay parameter may be a scalar
*or* a `Sweep`. The compiler lowers a `Sweep` to v2's `QickSweep1D` and auto-registers `add_loop`. The
entire v1 `RAverager`/`NDAverager` register-update machinery disappears.

```python
# Amplitude Rabi (1D over gain)
amp = Sweep("amp", 0, 30000, 51)
rabi = StrumentoProgram(dev, reps=1000, sweeps=[amp],
    seq=Seq().play(dev.qubit.ge_pi(gain=amp)).measure())
res = rabi.acquire(soc)                           # -> MeasurementResult (see below)
res.avgi.shape                                    # (n_ro, 51); res.xpts == amp axis

# T1 (1D over an idle delay)
t = Sweep("t", 0, 500, 101)                      # µs
t1 = StrumentoProgram(dev, reps=1000, sweeps=[t],
    seq=Seq().play(dev.qubit.ge_pi()).wait(t).measure())

# Chevron (2D nested; list order = loop nesting)
f = Sweep("freq", 3550, 3575, 51); a = Sweep("amp", 0, 30000, 51)
res = StrumentoProgram(dev, sweeps=[f, a],
    seq=Seq().play(dev.qubit.ge_pi(freq=f, gain=a)).measure()).acquire(soc)
res.avgi.shape                                    # (n_ro, 51, 51)
```

**`StrumentoProgram.acquire(soc, ...)` returns a `MeasurementResult`** (§9), *not* v1's `(xpts, avgi, avgq)`
tuple and not v2's raw native structure. It wraps `AveragerProgramV2.acquire`'s native return (which
differs from v1) into the typed seam, attaching the sweep axes as `xpts`. A `.as_tuple()` convenience
returns `(xpts, avgi, avgq)` for anyone porting v1 analysis code incrementally.

**Nothing v2 is hidden** — multiplexed/decimated/DDR4 readout (`declare_readout`/`trigger(ros=…)`),
`delay`/`delay_auto`, per-pulse `phase`/`phrst`, `QickParam` arithmetic, feedback control-flow: all on
`self`, with device helpers only for the common path.

## 9. Data lifecycle — run / analyze / fit / display (D13, D4)

### 9.1 Keep the high-level Experiment contract, split the internals

The slab `Experiment` base already exposes the right seams — `go(save, analyze, display)` →
`acquire()` / `analyze()` / `display()` / `save_data()`. That contract is **respected and kept**. The
v1 problem is in the *implementation*: each concrete `analyze()` **inlines** `fitter.fitdecaysin(...)`
*and prints* derived quantities, and `display()` **inlines** matplotlib — run, fit, analyze, and plot
are conflated on one class. `strumento` keeps the Experiment as the user-facing orchestrator but
delegates to distinct, independently-testable collaborators:

```
Experiment (thin orchestrator; keeps go/acquire/analyze/display/save)
  ├─ Runner    — builds StrumentoProgram from device+params, acquires → MeasurementResult   (no fitting, no plotting)
  ├─ Analyzer  — MeasurementResult → typed FitResult via lmfit fitters                 (no plotting, no I/O)
  └─ Display   — FitResult + MeasurementResult → figure                                (no fitting, no I/O)
```

Each collaborator answers "what does it do / how used / what depends on" alone: the Runner never
fits, the Analyzer never plots, the Display never re-fits. Derived-quantity printing (v1's inline
`print(f'Pi gain …')`) becomes typed fields on `FitResult`, not stdout side-effects.

### 9.2 `result.py` — `MeasurementResult` (the seam D4 protects)

Typed carrier of `xpts` (sweep axes), per-readout `avgi/avgq` (loop/channel axes), **raw per-shot data
+ herald-lane indices** (for post-selection), and metadata (device ref, sweep axes, timestamp). Exposes
`.as_tuple()` (v1-compat `(xpts, avgi, avgq)`) and `.postselect(...)` (ANDs enabled heralds per R1(ii)).
It does **not** inherit the 8.3 kloc fitting mess.

### 9.3 `fitters.py` — clean `lmfit` fitters + the current code as a physics oracle (D4)

- **Built on `lmfit`** (already a dependency — `slab/dsfit.py`, `experiments/qsim/utils.py`). Each core
  fit is an `lmfit.Model` with named `Parameters`, bounds, and returned uncertainties — replacing the
  v1 `def xfunc(x,*p)` + `def fitx(...)` scipy pattern and the derived-quantity printing.
- **Core set only:** **Rabi** (decaying sinusoid → π/π-2 gain), **T1** (exponential), **T2 Ramsey**
  (decaying sinusoid ± detuning), **parity** (contrast/phase). Take a `MeasurementResult`, return a
  typed `FitResult` (params + stderr + derived quantities).
- **Oracle validation.** The new fitters are cross-checked against the **current `fitting.py`**
  (`fitdecaysin`, `fitexp`, …) on the *same* data: a test asserts the new lmfit params agree with the
  legacy fit within tolerance. The trusted old code confirms the new code before the old code is retired.
- The full `fit_display*` / `fit_display_classes` / `wigner` / `noise_psd` refactor (≈8.3 kloc) is an
  explicit **follow-on spec**, not this one.

## 10. Testing & acceptance

- **`mock.py` — `MockQickSocV2`.** Reimplement the proven strategy: real qick runs build→compile→acquire;
  only leaf `soc.*` calls are stubbed (correctly-shaped zeros, right dtypes). v2's param validators fire
  on a buggy program with no FPGA. (v2's acquire path calls a slightly different soc surface than v1 —
  the mock covers v2's.)
- **Committed soccfg snapshot** for off-board runs — captured once on the target board via
  `QickSoc.dump_cfg()` and rebuilt with `QickConfig(json)` (no board thereafter — confirmed by Sho, D25).
  This makes gate 2 below a **real offline construction** of the concrete `AveragerProgramV2`, not merely
  a shape mock.
- **Three acceptance gates.** Structural alone is too weak for a reimplementation — a shapes-only mock
  would pass the int-vs-float gain bug (§6.1) unnoticed:
  1. **Structural gate (mock).** Program builds + compiles on `MockQickSocV2`; readout shapes match the
     declared sweep/readout axes; v2 validators fire on malformed programs. (Per `expt_service` §5.4.)
  2. **Numeric golden-equivalence gate (offline correctness).** For each core operation, compile the
     v1 program and the `strumento` program and assert the **realized physical values match**: register
     frequencies, **DAC gain codes / v2 gain fractions**, cycle counts, and envelope samples — via
     `quantize()` / the emitted ASM. The cheap middle gate between "it builds" and "it works on the
     board," and exactly what catches the gain-units / unit-conversion class of bugs.
  3. **Physics-agreement gate (on-board, the real acceptance).** On the actual board (with the
     collaboration): run **T1 / T2-Ramsey / amplitude-Rabi**, fit with the new `lmfit` fitters, and
     require (i) **agreement with physics** (sensible τ₁, τ₂, π-gain; Ramsey detuning matches the set
     detuning) and (ii) **agreement with the current stack** — the same raw data fit by the legacy
     `fitting.py` oracle (§9.3) yields matching parameters. This is the criterion that says the migration
     actually works, not just that it compiles.
  Gates 1–2 are mock/offline (no FPGA); gate 3 needs the board and is run with the QICK collaboration.
- **Unit tests** hit the pure layers directly (no qick): channel/wiring resolution (incl. band-gate
  rejection §6.4), factory→Pulse correctness against calibration, Seq composition, sweep→loop lowering
  (assert shapes), quantize() round-trips, the cqed factory's int-code→fraction gain conversion (D23),
  fitters against synthetic data.

## 11. Minimal core deliverable & sequencing (tracer bullets)

Each slice is end-to-end (device → factory → Seq → compile → mock-acquire → result → fit) and
independently demoable.

1. **Skeleton + wiring + one pulse, one readout.** Establishes the `core` / `packs/cqed` / `devices`
   split (§3, §5.1): `Device.load`, `core.wiring`, a `Pulse`, compiler lowers `ge_pi` + `measure` onto
   `StrumentoProgram`, runs on `MockQickSocV2`, returns a `MeasurementResult`. *Proves the whole vertical.*
2. **Amplitude Rabi** — `Sweep`-as-value + `add_loop` + Rabi fitter. *Proves sweeps + fitting.*
3. **T1 + T2 Ramsey** — swept delay, phase, `wait`. *Proves timing/phase.*
4. **f0g1 + storage swap** — `sideband`/`coupler` namespaces, freq-gated wiring, `ds_storage`. *Proves
   multi-line device model.*
5. **Parity** — composite factory (`dev.parity`), revival-time calibration.
6. **Optimal-control import (the Piccolo × QICK seam, §6.5).** `from_solution(...)` on a Piccolo /
   `extract_pulse` artifact → a `Seq` wrapping a `Segment` of normalized-ARB pulses across qubit +
   manipulate, `.play_parallel`, physical→DAC gain calibration, provenance; `dev.prepare(Fock(1))` front door.
   *Proves simultaneity + the physical-unit import path.* Validated by numeric-golden vs a known solution
   (§10 gate 2).
7. **Active reset** — split per R1: **(7a)** herald / post-selection data path in the result layer
   (no v2 risk) + host-side post-selection fallback; **(7b)** in-sequence v2 feedback (⚠ gated on Sho).
   `active_reset` is in the D2 core *surface*; sequencing it last is a schedule choice, not a scope cut —
   7a lands regardless, 7b lands when the v2 feedback API is confirmed.

Slices 1–6 have no v2-feasibility risk (pulses, delays, sweeps, readout, arb import are well-defined v2 API).
Only slice **7b** carries real uncertainty; 7a and everything before it do not.

> **Out of the minimal core (deferred, but modeled):** **catalog auto-retrieval** for `dev.prepare/gate`
> (§6.5), `dev.coupler.floquet_swap`, and `register_long_pulse`-style long constant drives appear in the
> design so it accommodates them, but are **not** in the tracer bullets above — they follow once the core
> is green. *(The OC import boundary itself IS in-core, slice 6; only the catalog auto-lookup is deferred.)*

## 12. Risks & open questions

- **⚠ R1 — active reset on tProc v2 (the one real risk).** This has **two separable parts**; only the
  first carries feasibility risk:
  - **(i) In-sequence measurement-conditional feedback.** v1 uses low-level ASM: `safe_regwi` /
    `read(0,0,"lower",reg)` / `condj(page,reg,"<",thresh,label)` / `label(...)` with hand-assigned
    register pages. tProc v2 expresses this through a **different control-flow model**; how much the
    target `qick.asm_v2` exposes (and its exact API) must be confirmed with **Sho Uemura**. Quarantined
    to one `active_reset` helper + `compiler.py`.
  - **(ii) Herald / post-selection data path.** v1 `active_reset` also emits up to **two independent
    post-selection herald rounds** (bare-|g⟩ and parity-mapped), each its own readout lane, and does
    **per-shot filtering** into `avgi/avgq` gated on `_pre_selection_filtering`. This is *not* a feedback
    problem — it is a **result-layer** concern: `MeasurementResult` must carry raw per-shot data + the
    herald lane indices, and expose a `postselect(...)` that ANDs the enabled heralds. This part has no
    v2 risk and can land independent of (i).
  - **Fallback if v2 feedback is unavailable/insufficient:** **host-side post-selection** — run without
    in-sequence reset, herald in software from raw shots — so slice 7 degrades gracefully instead of
    being all-or-nothing. (Reset *quality* differs, but the experiment surface stays usable.)
  Slices 1–5 do not depend on any of this.
- **R2 — exact v2 API names — LARGELY RESOLVED (Sho Uemura / Fermilab, IntonatoQICK PR #5 review,
  2026-07-13; see D25).** `AveragerProgramV2`, `_initialize`/`_body`, `declare_gen`, `add_gauss`,
  `add_pulse`, `add_loop`, `QickSweep1D`, `pulse`, `delay`/`delay_auto`, `trigger`/`readout` were used
  from recollection; Sho's review **confirms** the shape and the reference is his notebook
  [`meeg/qick_demos_sho` `tprocv2/00_intro.ipynb`](https://github.com/meeg/qick_demos_sho/blob/main/tprocv2/00_intro.ipynb).
  Key correction: you **subclass `AveragerProgramV2` and override `_initialize()`/`_body()`** — you do
  *not* append declarations to a bare program object (that idiom is `QickProgramV2` only). strumento's §8
  `StrumentoProgram(AveragerProgramV2)` already picked this pattern; doing it **in Python** sidesteps the
  "is this doable in Julia?" problem that made the IntonatoQICK inline-assembly approach awkward (§16.4).
  Remaining concrete artifact: the **`dump_cfg()` JSON** from the target board (ask Sho) — see D25 / §10.
  The `storage_in` = `mux4_v2` generator quirk (rejects `declare_gen` without mixer/mux) must be handled
  explicitly in the wiring/compiler.
- **R3 — multi-qubit generality.** The core is single-qubit-centric (`qTest`). The device model
  (`Qubit` element, per-line wiring) generalizes, but the first slices assume one qubit. Multi-qubit is
  out of scope for the minimal core; the model should not preclude it.
- **R4 — storage line vs coupler-addressed storage.** Storage *swaps* go through the coupler flux line;
  a separate `storage_in` DAC (mux4_v2) exists for direct storage drive (rarely used). Model both; the
  core exercises the coupler path.
- **R5 — optimal-control provenance.** `.npy`/`.npz` waveforms currently reference Windows share paths
  (`H:\Shared drives\...` in `device.optimal_control[...]['filename']`). The arb-factory should accept a
  configurable waveform source (or compute), not hardcode paths.
- **R6 — envelope naming / dedup (carry the v1 lesson).** v1 `add_gauss` is **not idempotent** — it
  re-allocates an envelope on every call, so per-gate naming blows up the waveform buffer with sequence
  length (v1 added an opt-in `dedupe_waveforms` flag as a patch). The v2 compiler's `_auto_declare` must
  add each distinct envelope (keyed by `(generator, envelope)` — the frozen envelope dataclass is
  hashable by construction, D22) **once** and reference it by name from multiple pulses — dedup by
  construction, not as an afterthought.

## 13. Appendix — pulse-type catalogue (what backs each, from `prepulse_creator2` + configs)

| v1 verb | Physical op | Data source | Shape | Line (Layer 1) | new home |
|---|---|---|---|---|---|
| `qubit('ge'/'ef','pi'/'hpi')` | qubit rotation | `calib.qubit.{pi,hpi}_{ge,ef}` | gaussian | qubit | `dev.qubit.*` |
| `qubit('ge_broadband')` | broadband ge | own row | gaussian | qubit | `dev.qubit.ge_broadband` |
| `qubit('ge','parity_Mx')` | parity wait | `manipulate.revival_time[x]` | delay | qubit | folded into `dev.parity` |
| `multiphoton(gn-en/en-fn/fn-gn+1,kind,n)` | multiphoton ladder | `calib.multiphoton[kind][trans]` by photon n | gauss/flat_top | qubit or sideband | `dev.sideband.multiphoton` |
| `man('M1','pi'/'hpi')` | f0g1 (legacy, M1) | `multiphoton.pi['fn-gn+1']` | flat_top | sideband | **deleted** → `dev.sideband.f0g1` |
| `storage('M1-Sx','pi'/'hpi')` | beamsplitter swap | `ds_storage` (+ poly freq↔gain) | flat_top | coupler (freq-gated) | `dev.coupler.swap` |
| `floquet('M1-Sx',pi_frac)` | Floquet swap | `ds_floquet` (`get_pulse_envelope`) | flat_top/gauss | coupler | `dev.coupler.floquet_swap` |
| `optimal_control(enc,state)` | arb IQ (2-ch) | `.npz` → **`from_solution` import** (Piccolo/catalog, §6.5) | arb `Segment` | qubit + manipulate | `dev.prepare(...)` / `dev.gate(...)` |
| `buffer(t)` / `wait(t)` | idle gap | — | const/delay | — | `Seq.wait` / `Delay` |

## 14. Open design decisions (to circulate before locking)

These two are deliberately **not decided** in this spec. Aaron to circulate with the team; the answers
shape the plan but not the core module design (§3–§9), which stands either way.

### 14.1 Config versioning mechanism (D9)

Single-source-of-truth (§7.2) is settled. The *versioning mechanism* is the open question, because the
operational reality (§14.2) — one shared checkout, a long-lived worker reading configs off disk,
concurrent submitters — makes the naive choice race-prone.

| Option | How | Pros | Cons |
|---|---|---|---|
| **A. Append-only immutable + pointer** (the *current* model) | Content-addressed snapshots (`CFG-…` ids, SHA256 dedup, write-once files) + one movable `main` pointer; writes serialized (SQLite WAL). | **Conflict-free by construction**; a new version can never disturb an in-progress read; reproducibility is permanent (an id resolves forever to byte-identical files); no textual merge of numeric calibration data. | Bespoke store (SQLite + `configs/versions/`); "not just git"; currently gitignored/per-machine. |
| **B. Git-native** | Config dir *is* a git repo; branches/tags/merges are git's. | Zero bespoke code; familiar; portable history. | **Reintroduces the races** the current model avoids: line-merge of YAML/CSV can silently corrupt structure; `checkout`/`merge`/`reset` rewrite files **in place, non-atomically, under the live worker's open reads**; one shared working tree defeats per-user branches; rebase/gc/force-push make old runs unreproducible. |
| **C. Hybrid (immutable-in-git)** | Append-only immutable snapshots, but **stored/synced via git that only ever ADDS files** — never in-place edits, never content merges; the pointer is a single serialized writer. | Keeps "configs handled by git" (backup, sync, portability) **while preserving conflict-freedom** (immutability, not merge semantics, is what makes it safe). | Needs discipline/tooling to guarantee git stays additive; two systems (git + pointer). |

**Recommendation:** A or C — **immutability is the load-bearing property**; git, if used, must be
additive-only. Pure B is not safe under the current shared-worker/multi-user topology. *(The verified
race analysis is in the session's job-server exploration; happy to attach it to the circulation.)*

**Bearing of later decisions (added after §15 / D16):**
- **This is now a framework-wide decision, not a multimode one.** The versioning mechanism lives in
  `core` and versions whatever config a *pack* defines (§15); a spin/atom lab using strumento wants the
  same mechanism. Choose it for the framework, not just for Stanford.
- **Pydantic (D16) strengthens A and C.** Canonical, typed serialization yields deterministic content
  hashes, so content-addressed dedup and diffs are robust across machines.
- **New tension the generality goal surfaces.** Option A *as literally reusing the existing store*
  hard-couples config versioning to Stanford's **`job_server` SQLite + `ConfigVersionManager`** —
  lab-specific infra a general framework shouldn't require. So even if immutability + pointer wins,
  `core` should expose versioning behind a small **store interface** (default: a standalone
  file-snapshot + pointer implementation with **no queue/DB dependency**; Stanford's
  `ConfigVersionManager` becomes *one* backend). This keeps A/C viable without importing the job server.

### 14.2 Job-server / multi-user integration (D14)

**Current model (verified):** FastAPI HTTP server + SQLite queue + a **single polling worker that is
the sole hardware owner** (PID-locked) + Pyro4 to the RFSoC. A user process **serializes the full
station config to JSON** and submits `experiment_class` / `experiment_module` **strings** (+ optional
`program_class`); the worker **dynamically imports** the class, applies the shipped config blob (rebuilding
`ds_storage`/`ds_floquet` from CSV records), and runs `expt.go()` → `acquire(soc)` where `soc` is a
**Pyro4 proxy**. Result = HDF5 + a pickled expt object; the user postprocesses in their own process.
Exclusivity is structural (one worker, serial FIFO+priority queue, SQLite WAL) — there is no per-user
device lock.

**Hard constraints this places on `strumento` regardless of the decision** (bake into slice 1 so the
compatible-with path stays open):
- Experiments must be **worker-importable by `module:class` path**.
- Experiments must be **constructable from the serialized config blob** — no reliance on live objects
  the worker doesn't have (the worker rebuilds datasets from CSV records).
- `StrumentoProgram` must run under a **Pyro4 `soc` proxy** in the worker (not just an in-process soc/mock).
  *(Being reconsidered — the collaboration proposes dropping Pyro; see §14.2.1. The general form is "runs
  under the job server's `soc` transport"; Pyro is one implementation.)*
- Results must be **serializable** (HDF5 + picklable `MeasurementResult`).

| Option | How | Trade-off |
|---|---|---|
| **A. Be compatible, don't reimplement** | strumento experiments satisfy the contract above; the existing queue/worker is untouched. | Smallest; keeps multi-user working immediately; no queue rewrite. Inherits the current runner/serialization ergonomics. |
| **B. Clean re-take on the runner** | strumento ships a tidy submit/runner layer (re-doing `CharacterizationRunner`/`SweepRunner`) that talks to the *existing* queue. | Better ergonomics/serialization; larger surface; must stay wire-compatible with the worker. |
| **C. Out of scope for now** | Direct in-process `acquire` + mock only; queue integration later. | Fastest to first light; but defers the multi-user reality the lab runs on daily. |

**Recommendation (LANDED 2026-07-21 — see §14.2.1, D14):** go **job-server-native, no-Pyro,
compile-off-board + thin execute-agent** (topology (ii)) **directly** — we do **not** stage through
Option-A wire-compatibility. The Option-A "keep the shared multi-user workflow working" rationale is moot
for us: we run on a **separate, dedicated FPGA QICK board in parallel on the fridge**, so there is no
shared worker/queue to stay compatible with. Options A/B above stay on record as the *compatibility path*
for anyone sharing the Schuster board; we take neither.

#### 14.2.1 Emerging direction (2026-07-19): job-server-native execution, drop Pyro

The QICK collaboration (Connie & Guan with Dave, cc Sho) is rethinking the instrument-manager layer, and
the direction reshapes this decision. Today the board runs a **QICK instrument server that selects a
firmware overlay and hosts a Pyro4 proxy** the measurement PC talks to. The proposal: **since a job
server exists anyway, eliminate the Pyro layer entirely**, and additionally allow the **firmware to be
reprogrammed per job/user**.

- **The enabler is D25's `dump_cfg()`.** Pyro exists to proxy a *live `QickSoc`* for fine-grained remote
  method calls. But once `dump_cfg()` → `QickConfig(json)` lets the whole **build + compile** of the
  `AveragerProgramV2` happen off-board, the only board-side operation left is **coarse**: *load compiled
  program → run → return ADC buffers*. That is a job submission, not an object proxy — Pyro is the wrong
  tool for it (and Pyro4 is EOL and version-couples the board's and PC's qick installs).
- **Two clean topologies, both Pyro-free.** (i) **Worker-on-board** — run the job-server worker on the
  board's ARM against a *local* `QickSoc` (no proxy, no nameserver); the PC only hits the job API. (ii)
  **Compile-off-board + thin execute-agent** — compile/analyze/fit on a fast machine (D25 makes this
  possible), ship compiled artifacts to a minimal board agent that loads+runs+reads. (ii) is preferable
  where the weak ARM should do as little as possible.
- **Firmware-on-the-fly = select among pre-built overlays, not synthesize gateware.** FPGA synthesis is
  hours-long/offline, so the achievable form is a **library of pre-synthesized overlay bitstreams** on the
  board, with "download overlay X" as a serialized **job step** (PYNQ `Overlay.download()`, ~seconds,
  resets the tProc; needs exclusive access — which the single-worker queue already serializes). **Each
  overlay has its own `dump_cfg()` JSON**, so a firmware personality is exactly one **soccfg snapshot**
  (§3, D25) — a clean 1:1 with strumento's per-instance snapshot model.
- **Keep an interactive escape hatch (Tenet 0).** The one thing Pyro gives that a coarse contract does
  not is live exploratory poking of the board from the PC; the job contract should carry a raw
  board-side-exec / debug verb (or those users keep an on-board notebook), mirroring §4.

**Bearing on the decision.** The hard constraint above ("`StrumentoProgram` must run under a **Pyro4**
`soc` proxy") **softens to "under whatever transport the job server uses."** Because strumento already abstracts
the board behind a small soc protocol (`MockQickSocV2` / `PyQickSoc` / `SimulatorSoc`, §19.3), **dropping
Pyro is swapping one transport implementation — nothing above it changes.** This is a shared cleanup: it
also retires the Pyro4 leg in `expt_service` (§16.1). **Decision (LANDED 2026-07-21):** go
**job-server-native, no-Pyro, compile-off-board + thin execute-agent** (topology (ii)) **directly** — add a
`JobServerSoc` transport and do **not** stage through Option-A wire-compatibility. Rationale: we run on a
**separate, dedicated FPGA QICK board in parallel on the fridge** (alongside the Schuster multimode board),
so the shared-worker blocking concern that motivated staging through Option-A does not apply to us. The
clean split: board = **dumb executor** (load overlay → load compiled program → run → return ADC buffers);
the fast machine owns **build + compile + analyze + fit**; each firmware overlay is exactly one
`dump_cfg()` soccfg snapshot (§3, D25); a raw board-exec/debug verb is retained (Tenet 0). *(Still worth
flagging to the QICK collaboration — the no-Pyro cleanup also benefits the shared board and `expt_service`
— but it no longer gates our work.)*

## 15. Generality / horizontal scope (D15, D16, D20)

**Ambition:** `strumento` as a general QICK tProc-v2 framework for experimental labs — usable by
practitioners across modalities (superconducting, spin qubits, neutral-atom, sensing), not just the
Schuster multimode device. This section records how far that reaches and the disciplined path to it.

### 15.1 Evidence the architecture already generalizes

**spinQICK** (HRL Laboratories, shipped 2025-07) is an open QICK framework for *electrostatic spin
qubits* that **independently converged on the same layering** `strumento` designs — and it targets tProc
**V2** too. Module-for-module:

| spinQICK (spins) | strumento (`cqed` example) | strumento layer |
|---|---|---|
| `qick_code_v2/` (tProc-V2 API + asm) | `core/compiler`, `core/program` | **core** (agnostic) |
| `Core/dot_experiment` | `Experiment` orchestrator (§9.1) | **core** |
| `helper_function/analysis`, `plot_tools` | `core/fitters` + Display (§9) | **core** |
| `Models/` (**Pydantic** config) | `core/wiring`, `core/device` + pack calib | **core schema + pack contents** |
| `Core/awg_pulses, eo_pulses, ld_pulses` | pack `library/` | **modality pack** |
| `Core/readout_v2, psb_setup` (PSB) | pack `readout` (DispersiveIQ) | **pack (readout kind differs)** |
| `Experiments/` (charge-stability, exchange) | pack `experiments/` (parity, dual-rail) | **modality pack** |

In our terms, **spinQICK is, in effect, a shipped `spin` pack + `core`** — which is exactly why strumento
models its `spin` pack directly on it (§15.5), and why the pack/`core` seam is not speculative.

The ecosystem already builds per-modality frameworks on the shared QICK substrate: **spinQICK** (spins),
**QICK-DAWG** (NV/defect sensing), **Qibosoq** (circuits). That is both the market signal and the
competitive reality.

### 15.2 What generalizes for free vs. what a new modality costs

- **Generic by construction (`core`, ~70–80%):** Pulse IR, compiler, wiring, sweeps, fluent builder,
  program, `MeasurementResult`, lmfit fit primitives, mock, escape hatch. A spin or atom experiment
  compiles pulses onto DACs, sweeps parameters, acquires, and fits T1/T2/Ramsey the same way.
- **Per-pack (modality-specific):** device model, pulse library, calibration schema, experiments, and
  — the one genuinely new engineering surface — the **readout kind**. Dispersive IQ (superconducting)
  vs PSB/charge-sensing (spin) vs fluorescence/photon counting (atoms/sensing) are structurally
  different. `core/readout.py` therefore defines `ReadoutKind` + `Discriminator` **protocols**; each
  pack implements them (`cqed` = `DispersiveIQ`). This is where generality costs real design work;
  everything else is packaging. The reducers must also emit the measurement vectors the QILC loop
  consumes (populations / wigner / tomography, keyed by knot index — §16.2).

### 15.3 Stance — design-for-extension; three modality packs in the map (D15, D20)

- **Factor the `core` / `packs` / `devices` seam now** (§3, §5.1). Good design regardless — the same
  seam that makes the Schuster instance clean.
- **Third-party packs are installable, not forkable:** pack discovery via Python **entry points**
  (`strumento.packs` group), so an external group — or spinQICK itself (§15.5) — can ship a pack as a
  separate package that `Device.load` resolves by name.
- **Three modality packs, spanning genuinely different physics:**
  - **`cqed`** (superconducting transmon + bosonic) — **build**, Stanford (§1, §11). The Schuster
    "multimode" system is its first *device instance*; pure-transmon / multi-transmon / transmon+bosonic
    are other topologies of the *same* pack (§5.1).
  - **`tweezer`** (neutral-atom AOD) — **build**, Thompson (§17). A real second user with genuinely
    different acquisition (camera imaging) and feedback (rearrangement) — which flushed out three `core`
    assumptions to fix (external-detector readout, persistent-field lifecycle, host-in-loop feedback).
  - **`spin`** (quantum-dot) — **designed** (spinQICK-modeled, §15.5); *build gated on a spin
    user*. A third, independently-shipped modality that validates the seam without a build commitment.
- **Rule of three, satisfied honestly:** `cqed` + `tweezer` are two real, very different modalities in
  hand; `spin` is the third data point (from a shipped external framework, spinQICK) that keeps `core`
  general without speculation.
- **Sequencing:** `cqed` leads (Stanford deliverable, §11); the three `core` sharpenings (§17) land so
  `core` is honestly general; `tweezer` builds with a Thompson requirements pass; `spin` is designed
  now, built when a spin user appears.

### 15.4 Caveats

- **Incumbents + ownership.** spinQICK/QICK-DAWG/Qibosoq have users; "better core, they'll switch" is
  not a plan. A general framework is a *product* (docs, support, governance) — a Harmoniqs
  horizontal-bet decision, not a code decision.
- **Name comity.** `strumento` reads as quasi-official QICK; confirm the name with the QICK
  collaboration (the Sho conversation is the natural channel). Note also KIT's `qiclib` (their
  QiController client library) — visually close, unrelated stack.
- **Tweezers are the *hardest* pack — which is why they're valuable to build.** Optical-tweezer
  control is a *different paradigm* — continuous multi-tone AOD RF + real-time rearrangement/feedback,
  not "compile a pulse program → acquire → fit." QICK's multi-tone mux generators suit AOD combs
  cleanly (§17), but the acquisition/feedback lifecycle stresses `core` in three real places. Rather
  than a reason to defer, that stress is exactly what makes the tweezer pass sharpen `core`'s
  abstractions honestly (§17 findings a–c, all now confirmed). §17 is **grounded in Thompson's actual
  paper** (arXiv 2506.13724); only firmware/protocol details (tProc version, command-streaming, the
  custom polynomial-chirp DDS) need a **Thompson requirements pass** (§17.5).

### 15.5 The `spin` (quantum-dot) pack — spinQICK-modeled (D20)

A third modality pack, modeled directly on **spinQICK** (HRL; §15.1). **Designed** now (it sharpens the
pack/`core` seam with a shipped-framework reference); **built when a spin user appears** (per D15).

- **Element types:** `Dot`, `Plunger`, `Barrier` (gate-defined quantum dots). Qubit encodings:
  **Loss-DiVincenzo** (single-spin, `ld_pulses`) and **exchange-only** (`eo_pulses`).
- **Readout:** **PSB (Pauli spin blockade) / charge-sensing** — a third `ReadoutKind` implementing the
  `core/readout.py` protocol (after `DispersiveIQ` and `ImagingReadout`), validating the readout seam a
  third, independent way.
- **Experiments:** charge-stability / electrostatic tune-up, LD single- and two-qubit, exchange-only
  control + exchange calibration, T1 / T2* / T2-echo / Ramsey / All-XY — all via the same
  Runner/Analyzer/Display lifecycle (§9); spinQICK already fits T1/T2 this way.
- **What it adds to `core` (little):** charge-stability is a **2D DC gate-voltage sweep** (plunger vs
  barrier) — slow ramps within the existing Sweep model; PSB is a new readout *kind* but no new `core`
  mechanism beyond the three §17 sharpenings. So `spin` mostly **exercises** the seam rather than
  extending it — which is exactly what makes it a cheap honesty check that `core` isn't SC/atom-shaped.
- **spinQICK relationship (open):** whether strumento's `spin` interoperates with / could re-home
  spinQICK, or stands parallel, is a partnership question (like §14.2 / §16) — not decided here.
  spinQICK is the design oracle either way.

## 16. Relationship to the Intonato/Intonatissimo calibration stack (D18)

strumento is the **open Python measurement/execution substrate at the bottom of the QILC calibration
stack.** It is *driven by* the calibration layers; it does not contain the calibration method.

```
Intonatissimo   PRIVATE — QILC method / ILCStrategy
  Intonato       PUBLIC — strategy-generic QILC chassis; solve!(PulseTuningProblem):
                          extract_pulse → run_experiment(::AbstractExperiment,::AbstractPulse)→Vector{Measurement}
    IntonatoQICK.jl  PUBLIC — AbstractHardwareBackend: upload_pulse! / trigger! / readout (PythonCall)
       strumento   OPEN — this spec: from_solution → compile → acquire → reduce
          board
```

### 16.1 Three alignments (the design already fits)

1. **`expt_service` ⊂ strumento — unify (D18).** expt_service's 3-verb contract — `upload_pulse`
   (GHz-Rabi IQ → gain via `PiPulseReference`) / `trigger` (assemble `AveragerProgramV2` + acquire) /
   `readout` (reduce to a measurement vector; kinds `iq`/`wigner`/`tomography_1q`) — is a strict subset
   of strumento (`from_solution` §6.5 → compiler → `acquire` → readout-reduce). `gain.py`'s
   `PiPulseReference` **is** strumento's `gain_calib`. **Direction locked: the 3-verb contract becomes a
   thin adapter/server over strumento** (one substrate). *Execution to circulate with Sho* — it touches
   the live IntonatoQICK collaboration.
2. **`from_solution` (§6.5) = the QILC-iterate-on-hardware seam.** Each `solve!` iteration extracts an
   `AbstractPulse`/`NamedTrajectory` and calls `run_experiment(pulse)`. On hardware that is
   Intonato's `HardwareExperiment(pulse -> run)` → PythonCall → strumento
   `from_solution → compile → acquire → reduce → Vector{Measurement}`. §6.5 is not just a Piccolo
   convenience — it is the closed-loop calibration hardware path.
3. **π-reference closure.** strumento's *experimental* calibration (amplitude Rabi → π-gain, slice 2)
   **produces** the `pi_gain` of the `PiPulseReference` directly (and `pi_rabi_ghz` from the known
   π-pulse's area/duration) that the physical→DAC gain conversion (§6.1) depends on. The framework
   calibrates the very reference its optimal-control import needs.

### 16.2 What the QILC loop requires of strumento

Intonato's `MeasurementModel` consumes measurement vectors keyed by knot index (`Measurement.data`,
`.index`). Its `measurement_functions/` are **`populations`, `wigner`, `displaced_parity`,
`density_matrix_measurement` / `rho_measurement_functions`** (ρ reconstruction — the closest analogue
of "tomography"). These **overlap** expt_service's readout kinds (`iq` / `wigner` / `tomography_1q`) but
are not a one-for-one identity: `wigner` matches directly; `tomography_1q` maps to Intonato's
ρ-reconstruction; `populations` / `displaced_parity` are additional Intonato functions; `iq` is the raw
form. **Design target (not identity):** the readout layer must be able to emit these reduced vectors
from discriminated IQ, keyed by knot. Placement per §15.2 — `core/readout.py` defines the
`ReadoutKind`/`Discriminator` **protocols**; the **`cqed` pack implements the specific reducers**
(`DispersiveIQ`, `populations`, `wigner`, `displaced_parity`, ρ-reconstruction), since `wigner` /
`displaced_parity` are bosonic-pack concepts, not core.

### 16.3 The boundary to hold (public/private)

strumento stays **strategy-agnostic**: it exposes *import-pulse / acquire / discriminate / reduce*; the
**QILC method stays in Intonatissimo**. Same open-core split as IntonatoQICK (public interface) vs
ILCStrategy (private method) — strumento is the open layer the private strategy ultimately runs on. No
ILC/QILC logic ever lands in strumento. (This also keeps strumento honestly general for non-Harmoniqs
practitioners, §15 — a spinQICK-style user gets the substrate without the calibration method.)

### 16.4 strumento resolves IntonatoQICK's v2 program-authoring question (Sho, PR #5)

Sho's IntonatoQICK PR #5 review (D25) surfaced the central seam: an `AveragerProgramV2` must be
**subclassed** (override `_initialize()`/`_body()`), so assembling one *inline from Julia* (append
declarations to a program object) is the wrong idiom. His options were (1) reimplement the averager loop
in Julia on `QickProgramV2` — he dislikes (duplicates qick); (2) subclass `AveragerProgramV2` in Julia —
he doubts it's doable; (3) a **Python-side program class**, parametrized by a config dict — his
preference; (4) a new QICK-side class taking two `AsmV2` objects.

**strumento is option 3/4 done right.** `StrumentoProgram(AveragerProgramV2)` (§8) *is* the Python-side
subclass, but instead of losing the ability to express the sequence (his worry about option 3), the
sequence is authored in the **`Seq` IR** (Julia/Intonato hands over a pulse/solution via `from_solution`,
§6.5) and lowered by the compiler. So the division of labour becomes:

```
Intonato (Julia)  → AbstractPulse / solution
  IntonatoQICK.jl → thin backend: PythonCall → strumento, hand over the pulse, get Measurements back
    strumento (Py)  → StrumentoProgram(AveragerProgramV2): _initialize/_body, compile Seq, acquire, reduce
```

IntonatoQICK's `PyQickSoc.play_program!` stops assembling `AveragerProgramV2` inline and instead
**delegates to strumento** — which removes exactly the awkwardness Sho flagged. This is the concrete
execution of D18 ("expt_service/IntonatoQICK unify over strumento") and what to socialize with Sho.

## 17. The `tweezer` modality pack — Thompson neutral atoms (D19)

> **Pack vs instance (§5.1):** `tweezer` is a *modality pack* (element types + libraries + imaging
> readout); the Thompson system (`devices/thompson_tweezer`) is its first *device instance*.

> **Grounding (verified).** From the Thompson group's *"Logical qubits with erasure conversion using
> metastable neutral atoms"* (Zhang, Liu, …, Thompson; **arXiv 2506.13724**), Methods §"Real-time AOD
> waveform generation" (+ "Experimental implementation"). The QICK-tweezer control is real and concrete;
> the facts below are quoted/paraphrased from that paper. Residual unknowns → §17.5.

### 17.1 The physical model (from 2506.13724 Methods)

- **Two tweezer paths.** A **static array from an SLM** (Hamamatsu; **88 traps** = 3×26 storage/loading
  + 10 gate zone) and a **dynamic array from a pair of orthogonal AODs (X/Y)**. Metastable ³P₀ ¹⁷¹Yb,
  **zone-based architecture** (atoms transported between storage and gate zones). QICK's role is the
  **AOD dynamic/transport controller** — *not* the SLM static array.
- **ZCU216 RFSoC running a *variant of the QICK firmware*.** Each of the X/Y channel outputs is the
  **sum of 32 independent DDS generators** — a multi-tone mux, up to **32 tones (movable tweezers) per
  axis**. One DDS tone = one dynamic tweezer (freq → position, amplitude → depth).
- **Move primitive: a polynomial chirp.** Each DDS tone performs a **frequency chirp = piecewise
  fifth-order polynomial** (min-jerk / zero-jerk trajectories) **as a single segment**, plus a
  **third-order polynomial for the amplitude** — transmitted as *coefficients*, "much more efficiently
  than a raw waveform."
- **Detection = camera + PC frame grabber.** Fluorescence imaging; a **PC frame grabber processes the
  camera images and streams commands to the FPGA real-time processor (tProc)**; **~25 ms** latency from
  image exposure to the next waveform step (they note future work: process images *on* the FPGA).
- **Transport is heating-limited** (acoustic lensing — the AOD acts as a cylindrical lens as its drive
  frequency sweeps); the trajectory (zero-jerk polynomial) is chosen to minimize it (§17.4).

### 17.2 What maps cleanly onto the core/pack seam (validates the extension points)

- **An AOD axis is a mux generator (32 tones).** `Line{generators:[Generator(gen_type="mux…",
  mux_freqs=<≤32 tones>)]}` — the wiring schema (§5) already carries `mux_freqs`; X and Y are two mux
  lines. Each tone = a movable tweezer.
- **A move is a `PolyChirp` — a new, compact envelope type.** Their move is a *piecewise 5th-order
  frequency polynomial + 3rd-order amplitude polynomial, single segment*. This adds **one envelope**:
  `PolyChirp(freq_poly deg 5, amp_poly deg 3)` — **registered by the tweezer pack** together with its
  lowering via the §6.1 envelope registry (D22), so core stays stock-v2 while the pack targets the
  custom chirp DDS. It is a *compact parameterized* pulse — exactly strumento's philosophy (like
  `Gaussian`: coefficients, not samples), and a better fit than a generic sampled `Sweep`. A
  simultaneous multi-tone move = a `Segment` of `PolyChirp`s (§6.1).
- **Imaging is a `ReadoutKind`.** The pack supplies `ImagingReadout` (camera + frame grabber → ROI
  occupancy) + a discriminator, implementing the `core/readout.py` external-detector protocol — *instead
  of* `DispersiveIQ`. Acid test of §15.2 — passes: the readout seam is genuinely IQ-agnostic.

### 17.3 The three `core` sharpenings — all CONFIRMED by 2506.13724

1. **(a) Readout data-source abstraction — confirmed.** Detection is a **camera + PC frame grabber**,
   not a QICK-ADC IQ buffer. `core/readout.py` must let a `ReadoutKind` declare an **external,
   trigger/stream-synchronized detector** as its data source. (`core/mock.py` gains a mock external
   detector.)
2. **(b) Persistent generation + multiple detection events — confirmed.** Persistent **SLM static
   traps** + **AOD dynamic** movers, with imaging interleaved across the transport phases (ramp-up
   handoff → move → ramp-down). Not `Seq → acquire → done`. `core/program.py` needs **persistent/
   background generation** and **multiple detection events** per sequence.
3. **(c) Host-in-the-loop feedback — confirmed.** **PC frame grabber → streaming commands → tProc**,
   ~25 ms latency; not active_reset's `if IQ<thresh` branch. `core/feedback.py` defines a feedback
   protocol with two modes: **in-sequence branch** (active_reset) and **host-in-the-loop** (image →
   host solve → next control). The paper's stated future ("process images on the FPGA") means the
   protocol should allow an FPGA-side path too.

Designing (a)–(c) into `core` from the start is what makes `core` honestly modality-general (D15) — the
`cqed` pack simply doesn't exercise them.

**Plus one firmware note.** The polynomial-chirp DDS (`PolyChirp`, §17.2) is a **Thompson custom QICK
firmware variant**, not stock QICK. So "built with Thompson" includes targeting that firmware: the
tweezer pack's compiler path must emit the polynomial-segment DDS commands their variant exposes (vs
strumento's stock-v2 assumption elsewhere). The envelope registry (D22) **confines this boundary to the
pack**: `PolyChirp` + its lowering live in `packs/tweezer`, and core never names a shape stock v2
cannot play. This is a real integration boundary, flagged in §17.5.

### 17.4 The Piccolissimo transport connection — stronger than expected

The paper makes the wedge obvious: Thompson **already** parameterizes transport as *optimized polynomial
trajectories* (zero-jerk beats min-jerk once acoustic lensing is modeled) to minimize heating, and
their **CZ gate is GRAPE optimal control** (Ref 37, Jandura–Thompson–Pupillo). That is squarely the
Harmoniqs optimal-control wheelhouse:

- **Piccolissimo's `TweezerTransportProblem` is a direct upgrade to their hand-derived zero-jerk
  polynomial** — solving the transport waveform against the *actual* heating objective (acoustic lensing
  + scattering) rather than a symmetry-constrained 5th-order form.
- `from_solution` (§6.5) imports the solved chirp onto the AOD (as a `PolyChirp` or sampled `ARB`),
  closing the loop: **Piccolissimo solves → strumento plays on the AOD.**

Concrete Harmoniqs × Thompson wedge: **replace the zero-jerk polynomial with an OC-optimized transport
waveform**, measured against their own heating/scattering budget.

### 17.5 Sequencing (tweezer tracer bullets) + open items

Depends on the three `core` sharpenings landing first; then, per pack:

1. **Static array** — AOD mux line + N-tone `Segment` → `set_array(positions)`; mock external detector.
2. **Imaging readout** — `ImagingReadout` (camera trigger + ROI photon-count) + occupancy discriminator
   (histogram/MLE) → `MeasurementResult` of site occupancies.
3. **Move** — a single-tweezer chirp (`Sweep` on a tone) → `move(site→site)`.
4. **Rearrange (host-in-loop)** — image → host assignment solve → multi-tone chirps → re-image.
5. **Transport import** — Piccolissimo `TweezerTransportProblem` → `from_solution` → played move (§17.4).

**Verified from the paper:** 32 DDS tones/axis, X/Y crossed AODs, the `PolyChirp` move primitive
(5th-order freq + 3rd-order amp), camera + PC frame-grabber detection, host-in-loop feedback at ~25 ms,
SLM static (88 traps) + AOD dynamic, transport heating-limited by acoustic lensing.

**Residual unknowns (smaller now) — a Thompson requirements pass:** the explicit **tProc version**
(v1/v2) of their QICK variant; the **PC↔tProc command-streaming protocol**; the **custom
polynomial-chirp DDS firmware** interface (and whether strumento targets stock v2 + a chirp block, or
their firmware variant — §17.3); the rearrangement/assignment algorithm; and whether intensity
servoing is in scope. Like the `cqed` pack (Stanford/Sho) and expt_service (§16), the tweezer pack is
**designed here, built with the collaboration** (Thompson).

## 18. Environment / station seam — fridge & instrument integration (D24)

**Frame.** Experiments happen in a *station*: the SoC plus a dilution refrigerator, signal chains,
amplifiers, and auxiliary instruments. Tenet 1 extends one rung — *model the station's read-only
seam* — while **fridge/instrument control stays out of scope permanently** (fridge automation is the
vendor's product; strumento never owns heaters, valves, or magnets). The concrete driver is a
Maybell-class fridge-vendor integration (compact dilution refrigerators + standardized high-density
wiring), but every primitive below pays off with zero partnerships.

### 18.1 Three primitives (cheap now, impossible to retrofit)

1. **`EnvironmentSource` protocol** (`core/environment.py`, pure, ~20 lines):
   `snapshot() → dict[str, float | str]` of typed telemetry (stage temperatures, pressures,
   cooldown id). Implementations are adapters — a Maybell client, a Bluefors client, or a
   **QCoDeS-instrument wrapper** (one adapter buys the whole QCoDeS driver ecosystem). strumento ships
   the protocol + a mock; vendors/labs ship adapters.
2. **Result stamping.** The Runner snapshots all configured sources at acquire start/end into
   `MeasurementResult.meta["environment"]` (reserved key, schema-light). This is the
   *retroactively-impossible* one — data taken without stamps can never be re-stamped — and it is
   directly load-bearing beyond partnerships: calibration-drift-vs-temperature correlation is exactly
   what closed-loop calibration trend analysis and fleet-level device modeling consume.
3. **Signal-chain metadata on the wiring** (optional `chain:` block per `Line`, §5.2): per-stage
   attenuation and the amplifier chain (HEMT/TWPA) between DAC and device. The missing rung between
   DAC units and power-at-device — it strengthens the §6.1/§6.5 physical↔gain story and makes fridge
   wiring a *described* artifact instead of tribal knowledge.

### 18.2 The partnership shape (Maybell-class)

A fridge-vendor integration is then **a config profile + a thin adapter, not a fork**: the vendor's
standardized wiring looms ship as machine-readable **fridge profiles** (a `chain:` fragment merged
into the device-instance yaml) and their telemetry API implements `EnvironmentSource`. The joint
story — fridge + RFSoC + strumento as a turnkey "cold + control + experiments" reference stack —
targets exactly §15's audience (new QICK labs standing up their first system). *Recorded as a design
consideration, not a committed partnership.*

### 18.3 Deferred explicitly

Condition gating ("don't run if $T_\text{MXC} > 30$ mK" — belongs in the §14.2 job-server layer as a
predicate over snapshots), instrument *control* of any kind, vendor drivers themselves, and a full
`Station` abstraction (the named growth path if the seam ever needs to own instruments; QCoDeS's
`Station` is the prior art).

## 19. Codesign enablement — strumento as a device-model measurement instrument (D26–D29)

**Frame.** A control framework usually flows one way: config → pulses → board. strumento is designed so
information also flows **back** — from measurements to a structured device model that can inform the
*next* design of the hardware it drives. In other words, strumento is not only a *runtime*; it is a
**device-model measurement instrument** that closes a loop:

```
design-time device model  ──►  predict / exercise (simulator, §19.3)
        ▲                              │
        │                              ▼
   (re)design                     fabricate / wire up
        ▲                              │
        │                              ▼
   DeviceModelReport  ◄──  measure + characterize (fitters, §9)  ◄──  board-aware compile (§19.2)
   (measured model + diff, §19.1)
```

strumento owns three arcs of this loop — **board-aware compilation** (§19.2), **measurement +
characterization** (§9), and **structured export of the measured model** (§19.1) — plus the ability to
**exercise a design-time model before it exists** (§19.3). It deliberately does **not** own the
design/optimization arc (re-specifying the hardware, or optimizing pulses through it): that lives
*upstream* in the trajectory-optimization stack (Piccolo/Piccolissimo, §6.5) and consumes strumento's
exported model through a clean, serialized boundary. strumento's job is to make the loop's
**measurement and board-aware-play ends first-class and honest**, so whatever sits upstream has a
faithful, machine-readable picture of the as-built device.

Everything here is a generally-useful capability for any lab that iterates on its own hardware; it needs
no special partner or program to be worth building.

### 19.1 The `DeviceModel`, provenance, and the design-vs-measured diff (D26)

Today a `Device` (§5) bundles wiring + calibration values for *one* fridge, transcribed from what the lab
has measured. Codesign needs that same structure to be **authored from a prediction** and **compared to
reality**, so we factor out a **`DeviceModel`** — the structured physical description, separate from the
runtime `Device`:

```python
# core/device_model.py  (pure; Pydantic — serializable, diffable)
class DeviceModel(BaseModel):
    elements:  dict[str, ElementParams]   # per named element: f_ge, anharmonicity, couplings, T1, T2, …
    transfer:  dict[str, TransferFn]      # per line: H(ω) (see §19.2)
    crosstalk: CrosstalkMatrix            # inter-line coupling
    source:    Literal["design","measured","hybrid"]
    provenance: dict                      # timestamp, environment stamp (D24), fit uncertainties, …
```

- A `Device` gains `dev.model: DeviceModel` and declares its `source`. A **design-time** device
  (`source="design"`) is authored from predicted/proposed parameters — it can be loaded and exercised
  (§19.3) before any hardware exists. A **measured** model is produced by characterization.
- The Analyzer (§9.1) gains a terminal artifact, the **`DeviceModelReport`**: the measured `DeviceModel`
  + `diff(design, measured)` (per-parameter deltas with uncertainties) + provenance. This is the loop's
  feedback artifact — a faithful, serialized picture of the as-built device that an upstream (re)design
  process consumes. It is a natural superset of per-experiment `FitResult`s: the fitters already
  recover $f_{ge}$, $T_1$, $T_2$, revival times, etc. (§9.3); D26 is the discipline of **collecting them
  into one structured model, keyed to the device, versioned, and exportable** — rather than leaving them
  as scattered per-run fit outputs.
- **"The calibration path is the characterization path."** Because closed-loop calibration (§16) already
  measures device parameters every run, the measured `DeviceModel` is updated *for free* by ordinary use;
  strumento is the accumulator. No separate characterization campaign is required to keep the feedback
  artifact current.

### 19.2 Board-description → `Wiring`, and transfer-function-aware compilation (D27, D28)

**Board description (D27).** The wiring model (§5) already separates device language (Layer 1) from the
DAC map (Layer 2). Codesign adds a **Layer 3 upstream**: a neutral, typed **board-description schema**
(channel roles, generator→DAC/ADC bindings, and RF-periphery annotations — filters, PA/attenuation
stages, that populate the §18 `SignalChain`). `Wiring.from_board_description(desc)` generates the Layer-2
`Wiring` from it, extending single-source-of-truth (§7) **down to the physical board**: the board
*is* the wiring config's upstream source, and re-binding a trace is a one-line board-description edit
that leaves every pulse factory untouched (the §5 Layer-1/Layer-2 property, now proven from copper). The
schema is vendor-neutral; any netlist/CAD export is **one producer/adapter**, not a coupling.

**Transfer functions and pre-distortion (D28).** The path from a DAC to the qubit is a **linear filter**
(traces, filters, line parasitics, an AOD's transfer function). Its in-band distortion is a known,
measurable transfer function **H(ω)** — and in-band linear distortion is **pre-compensable in software**:

- The `SignalChain` (§18) carries a per-line **`TransferFn`** (H(ω)) — obtained by **measurement**
  (VNA / round-trip characterization → stored on the measured `DeviceModel`, §19.1) or **modeled** from
  the board description (above) — and inter-line **crosstalk** coefficients.
- The `from_solution(...)` import boundary (§6.5) gains an optional **pre-distortion stage**: given a
  solved pulse $u(t)$ and H(ω), emit the pre-distorted waveform $\tilde{u}$ with $H * \tilde{u} \approx
  u$ at the device (regularized deconvolution; explicit out-of-band and DAC-clip guards). It composes
  with the resample and physical→fraction gain-calib the boundary already owns (D23).
- **Scope discipline:** pre-distortion covers the *linear, in-band, known-H* regime — the common case.
  It cannot fix passband notches, insufficient bandwidth, dynamic range, or crosstalk; strumento's job is
  to **measure and expose** those (via the `DeviceModel`), making the boundary between "fixable in
  software" and "must change the hardware" explicit and quantitative. Optimizing the pulse *through* the
  filter (as opposed to pre-correcting a fixed pulse) belongs upstream in the optimizer, not here.

### 19.3 Simulator soc backend (D29)

strumento's soc surface is a small protocol (build → acquire → typed data), implemented three ways:
`MockQickSocV2` (shapes/validators, no physics), `PyQickSoc` (the real board), and — added here — a
**`SimulatorSoc`** that runs a `Device` + `Seq` against a **physics simulation** (the natural backend is
the Harmoniqs simulation stack; a `DeviceModel` → Hamiltonian map). Because it satisfies the same
`acquire` protocol, **the same experiment modules and fitters run against it unchanged** — so a
**design-time `Device`** (`source="design"`, §19.1) can be exercised, characterized, and its
`DeviceModelReport` produced *before the hardware is fabricated*. Simulation (design-time) and board
(as-built) become the two ends of one interface; the diff between their `DeviceModel`s is the whole point
of the loop. This is a design-for-extension seam — the protocol is defined now; the physics backend is a
**deferred build** (spec §11 slice list; it is not on the core critical path).

### 19.4 The boundary strumento holds

strumento provides the **measurement, board-aware compilation, and structured export** ends of the
codesign loop, behind clean serialized interfaces (`DeviceModel`, `DeviceModelReport`, the
board-description schema). It does **not** contain hardware-design logic, hardware qualification policy,
or any optimization over hardware parameters — those consume strumento's artifacts from *upstream* and
stay out of the open substrate, exactly as the QILC *method* stays out of strumento (§16.3). This keeps
strumento honestly general (any lab iterating on its own hardware benefits) while leaving the
design-side machinery to the layers above it.

---

*End of spec. §14 items are open pending circulation; §16's expt_service **execution** circulates with
Sho (direction locked); §17's tweezer pack (D19) needs a Thompson requirements pass. §15 records the
generality stance (two real packs); §18 records the environment/station seam (D24); §19 records the
codesign-enablement dimension (D26–D29) — measurement + board-aware compilation + structured
device-model export, with the design/optimization side held upstream. Next steps per the brainstorming
workflow: spec review → user review → `writing-plans`.*
