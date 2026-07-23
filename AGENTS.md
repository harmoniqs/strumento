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

- **Python ≥ 3.11.** On this machine: `source $(conda info --base)/etc/profile.d/conda.sh &&
  conda activate strumento` (the system `python3` may be older — never use it).
- `pip install -e ".[test]"` — pure layers. Add `.[qick]` (or `pip install qick`) for the
  compiler/program/mock layers; qick installs fine on x86 (no pynq off-board).

## Tests — run them like CI does

```bash
pytest -m "not qick" -q      # pure suite — must pass with NO qick installed
pytest -q                    # full suite — real tProc-v2 assembly runs offline
ruff check .                 # lint (config in pyproject)
```

- Marker rule: any test that imports `strumento.core.{mock,program,compiler-with-qick}` or
  `qick` itself is `@pytest.mark.qick`. Everything else must run qick-free — CI's `pure` job
  enforces this by running without qick installed.
- **The v2 fixture gate** (`tests/_v2gate.py`): v2-assembly tests use
  `devices/_fixtures/soccfg_v2.json` (your board's `dump_cfg()`, gitignored) when present, else
  the committed official testbench config (`soccfg_v2_testbench.json`, single gen + single
  readout — use `devices/loopback_demo` for anything that must really assemble).
- TDD is the working style here: failing test → minimal code → green → commit.

## Hard rules

1. **Purity boundary**: only `core/compiler.py`, `core/program.py`, `core/mock.py` may import
   `qick`. `import strumento` must never pull qick in. There are tests for this.
2. **`core` is physics-free**: element types, pulse libraries, readout kinds, calibration
   schemas live in `packs/`. Core never imports a pack by name (entry-point discovery only).
3. **Gain units (D23)**: `Pulse.gain` is always the v2 normalized fraction. Int-code ↔ fraction
   conversion happens only at the cqed pack boundary; physical-unit conversion only in
   `from_solution` via a calibrated reference.
4. **Config is data**: calibration values live in instance YAMLs, never in code. The
   `DeviceBuilder` produces the canonical model; it never becomes a second store.
5. **Escape hatch**: never wall off raw qick. Any new abstraction needs a bypass.
6. **No collaborator names** in code, docs, comments, or commit messages — this tree is kept
   public-ready. Published-paper citations (arXiv ids) are fine. Run the release-audit grep
   before large doc changes if unsure (the maintainers keep the list out-of-tree).
7. **Typed results**: fitters return `FitResult` (params/stderr/derived), experiments return
   `ExperimentOutcome`. No printing from analyzers; no derived quantities to stdout.

## Repo map

```
strumento/core/        wiring · pulses (IR + from_solution) · sweeps · device (+ builder,
                       coupling, operators) · compiler (envelope registry) · program · mock ·
                       result · readout · fitters · resonator_fitters · experiment · soc ·
                       transport · environment? · dc · cli
strumento/packs/cqed/  elements · calibration · library/ (transmon, sideband, mode, coupler,
                       composite) · readout (DispersiveIQ) · experiments/ (rabi, t1, t2,
                       parity, resonator_spectroscopy)
strumento/packs/spin/  elements · calibration · library/ (ld, exchange) · readout (PSB) ·
                       experiments/ (charge_stability, psb_histogram)
strumento/packs/tweezer/  skeleton (elements, PolyChirp envelope, imaging readout)
devices/               multimode_demo (docs/tests anchor) · loopback_demo (v2 assembly) ·
                       tweezer_demo · spin_demo · _fixtures/ (soccfg snapshots)
docs/                  architecture · getting-started · cqed-cookbook · packs-and-devices ·
                       fitters · spin-pack · job-manager · qick-v2-api-notes · hardware-ci · cli
```

## Extension recipes

- **New pulse factory**: one function on the right ops class reading calibration; add a schema
  field if needed. Test: factory → expected `Pulse` fields.
- **New envelope**: frozen dataclass + `register_envelope(Type, declare_fn,
  pulse_kwargs=style_fn)` — packs do this at import (see tweezer's `PolyChirp` slot).
- **New experiment**: subclass `Experiment`; implement `sequence()` + `analyze()`; wire a
  fitter; structural e2e on the mock + fit-recovery on synthetic data.
- **New fitter**: lmfit model → typed `FitResult` in `core/fitters.py`; synthetic-data
  recovery test with tolerance; add to the CLI `fit` map.
- **New pack**: directory under `packs/` + `CALIBRATION` + `compose_elements` + entry point in
  pyproject. See `docs/packs-and-devices.md`.
- **New device instance**: a YAML under `devices/` (or out-of-tree — instances resolve packs
  via entry points; see the private companion repo pattern).

## CLI (agents: prefer `--json`)

```
strumento packs | validate <yaml> | info <yaml> | soccfg <json> | fit <kind> <csv> | compile <yaml>
```

All commands take `--json`; errors are `{"ok": false, "error": ...}` + exit 1. See
[docs/cli.md](docs/cli.md).

## What needs a board (and what doesn't)

| Capability | Board needed? |
|---|---|
| Pulse IR, device model, builder, couplings, operator export, fitters, CLI | no — no qick at all |
| Real v2 program assembly + mock acquire | no — committed testbench soccfg |
| Numeric-golden gate (realized values) | no |
| Multi-channel v2 assembly beyond 1 gen/1 readout | your board's `dump_cfg()` JSON |
| Physics-agreement gate (real T1/T2/Rabi) | yes — self-hosted runner (docs/hardware-ci.md) |

## CI

`.github/workflows/ci.yml`: `lint` (ruff) + `pure` (3.11/3.12, no qick) + `qick` (full suite,
offline v2 assembly). `hardware.yml` is the board-in-the-loop job — inert until a self-hosted
runner is registered (`HAS_QICK_RUNNER` repo variable); label a PR `run-hardware` to request a
board run. PRs must be green on all three hosted jobs.
