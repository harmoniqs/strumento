# Changelog

All notable changes to Strumento are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-21

The first public release.

Known limitations of this release are listed in the README's
[Status → Known limitations](README.md#known-limitations) section.

### Added

- **Core (modality-agnostic, no physics).** A device model with a wiring layer that maps named
  elements to generators and readouts; a typed pulse IR (`Gaussian`, `FlatTop`, `Const`, `Arb`)
  and a `Seq` sequence builder with a raw-qick escape hatch mid-sequence; `Sweep` / `Loop` axes;
  and a compiler that lowers a sequence to a QICK tProc-v2 `AveragerProgramV2`
  (`StrumentoProgram`). The pure layers import and test with no `qick` installed. A pulse's
  `freq_mhz`, `gain` and `phase_deg`, a `FlatTop` or `Const` envelope's `length_us`, and
  `Seq.wait` accept a `Sweep` as well as a number, and the type annotations say so; `mypy`
  checks them in CI.
- **Declared sweep axes** (D36). Declare a loop with `axes=[Loop(name, n)]` and bind a field to
  it with `Sweep(start=…, stop=…, on=name)`. The legacy per-name `Sweep("name", start, stop, n)`
  form still works, but it is deprecated: it emits a `DeprecationWarning` once, at the line that
  wrote it.
- **Device authoring.** Devices load from YAML or are built in code with `DeviceBuilder`, and
  round-trip through a canonical, content-addressed `DeviceConfig`. Typed couplings and
  Hamiltonian export (`dev.to_operator_model`) feed an optimal-control solver; pulses come back
  in through `from_solution`.
- **Packs**, discovered through the `strumento.packs` entry-point group:
  - `cqed` (built) — transmons, bosonic modes, couplers and readout resonators, dispersive IQ
    readout, and 50 registered experiments across spectroscopy, calibration, benchmarking, cavity
    characterization and bring-up. Every calibration field documents its unit in the schema
    (its field description), with a units table in the packs-and-devices page. The ops
    factories' `gain=` has three documented meanings (`None` is the calibrated DAC integer
    code, a number is a fraction of full scale used as is, a `Sweep` is in DAC integer codes),
    stated once in `strumento.packs.cqed.library` and in a "Gain units" section of the guide
    and the cookbook.
  - `spin` (built) — quantum-dot workflows: loss-DiVincenzo, singlet-triplet, exchange-only and
    hole-spin elements, PSB / Elzerman / dispersive-gate readouts, a slow-DC seam, and 21
    registered experiments.
  - `tweezer` (skeleton) — element types, calibration schema and an imaging readout; no
    experiments yet.
- **Demo devices.** `multimode_demo` and `loopback_demo` (cqed); `spin_demo`, `spin_st_demo`,
  `spin_eo_demo` and `spin_hole_demo` (spin); and `tweezer_demo`. `multimode_demo`,
  `loopback_demo` and `spin_demo` carry soccfg snapshots and pass `strumento check`.
- **Offline simulator.** `SimulatorSoc` returns synthetic data from a ground truth you choose, so
  a whole calibration runs and fits with no board.
- **CLI.** `strumento packs | validate | info | soccfg | fit | compile | check | schema | task |
  store`, with `--json` output and errors as data. `strumento schema calibration --pack <name>`
  publishes a pack's calibration schema, with each field's unit in its description.
- **MCP session** (`[mcp]` extra). `strumento-mcp` serves a stateful lab session over stdio:
  load a device, run experiments, write calibrations back, read their history.
- **Calibration store.** Immutable, content-addressed calibration versions with a movable ref,
  explicit dotted-path write-back (`apply_updates`, `record_calibration`) and provenance.
- **Task records.** A durable, pure-standard-library directory contract for long-running
  experiments (manifest, append-only progress stream, artifacts, atomic result), with task ids
  minted by an atomic directory claim so two writers can never share one. `record.EVENT_KINDS`
  lists the documented event names, `poison` included; a test checks that every event the record
  module writes is in it and that it matches [docs/task-record.md](docs/task-record.md).
- **Bring-up graphs.** A typed vocabulary for bring-up campaigns, construction-time validation,
  and a resumable runner that executes one step per call from the campaign's event stream; the
  `cqed` and `spin` packs ship plans.
- **Fitters.** Resonator circle fit, Fano, the damped-sinusoid family and double-Gaussian, with
  standard errors propagated to derived quantities (`derived["<name>_stderr"]`). The two
  multi-start resonator fitters report a failed fit, `success=False` with empty `params`,
  `stderr` and `derived`, rather than raising: `fit_fano` when every starting point fails, and
  `fit_resonator_circle` on non-finite samples, data with no loop (all-zero or constant), or
  when both phase-fit starts fail. Every experiment's fit holds plain Python values, so it
  serialises with the standard `json.dumps`.
- **The `fit` contract** (D38). `FitResult.derived` holds JSON values: numbers with their
  `_stderr`, plus the strings, booleans, null, lists and objects an analyzer reports (a protocol
  name, a confusion matrix, a refusal's provenance). The published `strumento schema fit` says
  so, and a test validates the simulated fit of every registered `cqed` and `spin` experiment
  against it.
- **Transport.** The `CompiledJob` / `RawAcquisition` wire contract, `JobServerSoc` and a
  transport-backed run path for compiling off-board and running on a remote executor, with a
  reference execute-agent in `examples/jobserver/`.
- **Plots** (`[plot]` extra). `go(..., display=True)` draws figures with matplotlib; without it
  every experiment still runs. Every registered `cqed` experiment has a display; the displays
  register from one table, and a test fails if a newly registered experiment has none.
- **Documentation.** A getting-started guide (including a before-every-session checklist for
  hardware and a troubleshooting section organised by error message and by silent symptom), a board bring-up guide
  ([docs/board-bringup.md](docs/board-bringup.md): from a board's recorded description to a
  validated device rehearsed on the simulator, including Nyquist zones, first calibration values,
  and comparing the live board with the stored description), a calibration walkthrough
  ([docs/calibrate-a-transmon.md](docs/calibrate-a-transmon.md): resonator to randomized
  benchmarking on the simulator, with how to judge each step and the common traps),
  a guide to writing your own run
  script ([docs/write-a-runner.md](docs/write-a-runner.md)) with an example to copy
  (`examples/runner/run_amplitude_rabi.py`: live-configuration check, a crash-safe board lock, a
  preflight on the compiled program, a task record and data chest, quality checks on the fit, and
  a proposed — never applied — calibration edit), the `cqed` cookbook (including the host-batched
  diagnostics `ladder_population`, `self_kerr_ramsey` and `kerr_cat_witness`), and reference
  pages for the architecture and decision index, packs and devices, the timeline, the calibration
  store, task records, fitters, the job manager, the CLI, the `spin` pack and hardware CI.
  [docs/interop.md](docs/interop.md) covers importing data from other control stacks (QUA/OPX,
  Qblox, Zurich, a VNA) and exporting optimal-control waveforms to them. The docs' Python blocks
  run as tests, and further tests keep their factual claims in step with the code.
- **CI.** Lint; a `typecheck` lane running `mypy` on Python 3.11; a qick-free pure lane on
  Python 3.10–3.12; qick and mcp lanes on 3.10 and 3.11, with the qick lane also installing the
  `plot` extra so the display tests run; and a board-in-the-loop workflow for a self-hosted
  runner.
- `DispersiveIQ.from_calib(readout_cal)` builds the discriminator a device has recorded: the
  `ReadoutCal` threshold (ADC units) and angle (rad), `0.0` when unset, and each of the
  `g_center_iq` / `e_center_iq` / `f_center_iq` blob centres that is set (#196).
- MCP session writes say where they came from (#194). Every provenance line a `LabSession`
  writes (the genesis seed and each `record_calibration`) carries `soc_kind`, `simulated` and,
  for the simulator, `truth_hash` (SHA-256 of the canonical JSON of the `sim=` truth;
  `strumento.mcp.truth_hash`) in its `extra`, and the task record's `calibration` event carries
  the same fields (`TaskRecord.calibration(content_id, **fields)`). `calibration_history` and
  `strumento store log --json` now show each entry's `extra`. Older log lines read unchanged and
  content ids are unchanged: provenance is never hashed.
- MCP runs keep their data (#200). Each `run_experiment` saves the run to a `FileDataChest`
  entry (the `MeasurementResult` arrays including `shots`, its `meta` plus `soc_kind`,
  `simulated`, `truth_hash` and an `mcp_run` block, the played sequence and the fit summary) and
  returns `datachest: {saved, chest, entry}`; `FileDataChest(chest).load(entry)["result"]` reloads
  it for refitting. The chest is the new `load_device(..., chest=<dir>)`, or by default the
  sibling `<store>-data` directory beside the store; with neither, or when the save fails, the
  run still returns its fit and `datachest` is `{saved: false, reason}`. `load_device` and
  `device_info` report the chest. A task record cites each entry with an `artifact` event of kind
  `datachest`, and `record_calibration(updates, outcome=<index>)` cites that run's entry in its
  provenance (`extra.data`) and in the task record's `calibration` event. The tool list is
  unchanged.
- `strumento.interop.piccolo.piccolo_controls_to_solution(times_ns, controls_ghz, *, drive)`
  converts Piccolo's two-quadrature transmon controls, a `(2, N)` array of `u_x`, `u_y` in GHz on
  an ns grid, to the `{times, controls}` input of `from_solution`: ν = 2000 (u_x − i u_y) MHz on a
  µs grid (`PICCOLO_GHZ_TO_RABI_MHZ = 2000.0`). A naive `GHz × 1000` import played the pulse at
  half amplitude with Q sign-flipped. Malformed controls raise `ValueError` (#210).

### Changed

- **Bosonic experiments resolve their target from the device.** No experiment or factory defaults
  to an element name. An unset `mode_name=` (or `mode=`) is the `BosonicMode` declared with
  `role: cavity`, or the device's only mode; an unset `pair=` is the only calibrated coupler swap.
  With several candidates and none singled out, the experiment refuses and names them.
- **`multimode_demo` has its own channel layout and a generic cavity name.** The cavity mode is
  `cavity` (swaps `cavity-S1`, `cavity-S2`), and the lines are rewired: readout DAC0, qubit drive
  DAC1, cavity DAC2, storage DAC3, coupler DAC4, sideband DAC5.
- **Wigner tomography is reimplemented from the published methods.** Least squares in a Hermitian
  basis with the trace fixed exactly, the Smolin-Gambetta-Smith projection onto physical states,
  and an A-optimal displacement design with an analytic gradient; the module cites each source.
- **A gain refusal says how to raise the limit.** `GainOutOfRange` now names the calibration
  field that raises the role's limit for that element (`qubit.max_drive_gain`,
  `readout.max_readout_gain`, `modes.<name>.max_displace_gain`) and its hard cap, or says the role
  has no per-element override yet (the coupler pump and the sideband drive). The getting-started
  guide has a new "Raising a limit" section.

- **Behaviour change: cqed experiments read with the recorded readout calibration by default**
  (#196). When no `readout=` is passed, every cqed experiment that takes one now uses
  `DispersiveIQ.from_calib(device.calib.readout)` instead of a bare
  `DispersiveIQ(threshold=0.0, angle=0.0)`, and the bring-up graph's readout and
  `validation_check` now carry the recorded centres as well as the threshold and angle. On a
  device that records a non-zero `readout.threshold` or `readout.angle`, or any blob centre, the
  default analysis changes: the averaged signal is projected on the recorded angle, and analyzers
  that need a blob-anchored population (`drag_calibration`, `drag_pair_calibration`,
  fixed-period `error_amplification`, and the population / level-population conversions) now
  use the recorded centres instead of failing or refusing. `active_reset` keeps its `threshold`
  argument and angle `0.0` (they are tied to what the board plays) and takes only the recorded
  centres. Pass `readout=` explicitly to keep the old analysis. Compiled programs are unchanged;
  the shipped demos record zero threshold and angle and no centres, so their results are
  unchanged.
- **Behaviour change: a simulated MCP session refuses an unmarked store** (#194). A simulator or
  mock session marks a fresh store it creates (`simulation-store.json` at the root) and refuses,
  with `ok: false` and nothing written, to write into an existing store without that mark: both
  `record_calibration` and seeding a new ref in `load_device`. `load_device(...,
  allow_sim_writes=True)` lifts the refusal; the writes still record `simulated: true`. The
  library `strumento.core.store.record_calibration` is unaffected.
- **Behaviour change: `record_calibration` writes are `manual` unless they name a run** (#194).
  The session no longer labels every write with the last fit. `record_calibration(updates)` is
  recorded as `source: "manual"` with no fit attached; `record_calibration(updates,
  outcome=<index>)` (the `index` a `run_experiment` returned) is recorded as
  `"<experiment> fit"` with that run's derived values. Closed-loop clients that want the fit
  label must now pass `outcome`. Outcome indices no longer repeat once a session passes 32 runs.
- **Behaviour change: experiments that need per-shot data refuse explicitly without it** (#199).
  `Experiment.needs_shots` (a class attribute, `False` by default) is `True` for cqed
  `active_reset`, `readout_state`, `single_shot_readout` and `validation_check`, and spin
  `heralded_init`; `list_experiments` reports it per experiment as `needs_shots`. On a result
  without `shots` (the mock, or a board soc, which has no shot path yet) these return
  `success=False` with `derived["refused"]` = `strumento.core.experiment.SHOTS_REFUSAL`
  ("needs per-shot data; …") instead of a bare failure or a shot quantity computed from
  averages. Previously `active_reset` returned an empty fit, `readout_state` fit a double
  Gaussian to the two averaged reads and reported an assignment fidelity from it,
  `single_shot_readout` reported an averaged fidelity proxy with `success=True`, `heralded_init`
  reported the averaged signal as a population with `success=True`, and `validation_check`
  raised on the mock. `readout_state` keeps `angle` and `validation_check` keeps its averaged
  Ramsey quantities; both list the withheld quantities in `derived["refused_parts"]`. A result
  that carries `shots` (the simulator, or a caller that attaches them) is analysed as before.

### Fixed

- **The Wigner operator was the transpose of the true one.** It computed `W(α*)`, so a state with a
  complex amplitude reconstructed mirrored across the real axis (states on the real axis, such as
  real cats and Fock states, were unaffected). It is now `(2/π) D(α) Π D(α)†`, checked against an
  independent matrix-exponential construction.
- **`StorageT1` uses `mode_name`.** It was accepted and ignored; it now selects the swap and is the
  storage mode measured.
- **A raised qubit-drive or displacement limit reaches every experiment.** `QubitSpectroscopy`,
  `EFSpectroscopy`, `NumberSplitting`, `TlsScan` and `GainLadderSpectroscopy` checked their probe
  against the pack default of 0.5 even when the qubit's `max_drive_gain` was set, and
  `CavitySpectroscopy` checked against 0.4 regardless of the mode's `max_displace_gain`. They now
  use the element's calibrated limit, like the pulse factories.

- A gate override plays at the cursor (#210). `from_solution` gives its pulses `start_us=0.0`,
  an offset inside their parallel segment; installed as a `TransmonOps` override (`pi_ge`,
  `hpi_ge`, `pi_ef`, `hpi_ef`, including the `overrides=` of `TransmonAllXY` and `TransmonRB`)
  that offset used to survive, so the compiler placed the gate absolutely: both gates of an
  AllXY pair and the readout stacked at the pair's start, and qick only logged "pulse time 0.0
  appears to conflict". The override now drops its start offset. `from_solution` itself, and
  every calibrated gate, are unchanged.
- `record_calibration` validates a write-back before it writes: an invalid update (a typo'd
  key, a wrong type, NaN or infinity, a negative time or DAC gain code) raises and leaves the
  store, its ref and its log unchanged, instead of moving the ref to a config that no longer
  loads. The calibration schemas now refuse non-finite values in every pack, negative µs times
  in cqed and spin, and negative DAC gain codes in cqed — in a device YAML as well as a
  write-back.
- An agent can no longer raise its own gain ceiling (#190). The MCP session's
  `record_calibration` refuses, as a whole and before anything is written, any update that
  touches a path the pack declares in the new `CALIBRATION_CEILINGS` (for cqed:
  `qubit.max_drive_gain`, `readout.max_readout_gain`, `modes.<name>.max_displace_gain`). Those
  per-element limits now have hard caps in `packs/cqed/safety.py` (`QUBIT_DRIVE_GAIN_CAP = 0.8`,
  `READOUT_GAIN_CAP = 0.8`, `DISPLACEMENT_GAIN_CAP = 0.6`, fractions of full scale) and the
  schema bounds each to `0 < limit <= cap`, so a device with a limit above its cap does not
  load. The default limits are unchanged, and no shipped device sets a ceiling.
- `readout.trig_offset` is bounded to `0 <= trig_offset <= 10` µs (#195), in the cqed
  calibration and in the `wiring.readout.trig_offset` fallback, which had no bound at all. A
  device carried over from before #98 with a clock-cycle count (e.g. `400`, now 400 µs) no
  longer loads; the error names the unit and the likely cause. A time-of-flight or MCP
  write-back outside the range is refused before anything is stored. The schemas publish the
  bound.
- Replacing an MCP session is atomic (#192). A `load_device` that fails (a device with no soccfg
  snapshot, `sim=` with the mock soc, an unreadable store ref, a task record that cannot open)
  now leaves the previous session's device, soc, store, ref, task record and outcomes as they
  were, and writes nothing to the new store. Before, the session kept the old device and soc
  but switched to the new store and ref, which had already been seeded, so the next
  `record_calibration` landed there.
- An over-size transmon RB request is refused before it compiles (#211). `TransmonRB` puts every
  depth into one program, and deep RB used to overflow tProc program memory with only qick's
  compile-time `RuntimeError`. `sequence()` now raises `ProgramMemoryError` (a `ValueError`)
  when a lower-bound estimate of the program's size exceeds the board's program memory, read from
  `tprocs[0].pmem_size` in the device's soccfg snapshot. The error names the estimated size, the
  limit and the remedy: fewer depths per program, or one program per depth on the host. The few
  words the estimate does not count are caught at compile and raised as the same error. The
  ceiling is stated in the `TransmonRB` docstring. Requests that fit play and compile as before.

[Unreleased]: https://github.com/harmoniqs/strumento/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/harmoniqs/strumento/releases/tag/v0.1.0
