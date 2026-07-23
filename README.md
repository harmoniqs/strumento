# Strumento

[![CI](https://github.com/harmoniqs/strumento/actions/workflows/ci.yml/badge.svg)](https://github.com/harmoniqs/strumento/actions/workflows/ci.yml)

**Write quantum experiments in the language of your device — run them on QICK tProc-v2.**

Strumento is a first-principles [QICK](https://github.com/openquantumhardware/qick) tProc-v2
experiment framework: a device model, a typed pulse IR, a compiler, and an experiment/fitting
lifecycle — with a raw-qick escape hatch at every level. You describe operations on *named elements*
(`dev.qubit.ge_pi()`, a gain sweep, a readout); Strumento resolves the wiring, compiles to v2, and
gives you back typed results. Nothing is walled off — when you need raw qick, it's one call away.

It's designed from first principles against the tProc-v2 API — informed by the architecture
lessons of mature tProc-v1 lab stacks, keeping the physics and discarding the legacy — and it's
built to generalize: one modality-agnostic `core`, swappable **packs** for superconducting /
neutral-atom / spin qubits.

> 🔒 Private while the core is proven; open-source is the intent.

## A quick taste

```python
from strumento import Device, Seq, Sweep

dev = Device.load("devices/multimode_demo/device.yaml")

# Amplitude Rabi — in device language, no register-level assembly.
# The gain sweep is just a value in the pulse; the compiler turns it into a loop.
rabi = Seq().play(dev.qubit.ge_pi(gain=Sweep("amp", 0, 30000, 51))).measure()
```

```python
# Run it as an experiment (acquire + fit), on hardware or the offline mock:
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi

outcome = AmplitudeRabi(dev).go(soc)
outcome.fit.derived["pi_gain"]        # the drive amplitude for a π rotation
```

Full walkthrough → **[docs/getting-started.md](docs/getting-started.md)**.

## Why Strumento

- **Model the device, not the FPGA.** Physical ops on named elements; a wiring layer maps them to
  DACs. Re-patching a cable is a one-line config edit — every experiment is untouched.
- **Typed pulses, computed from calibration.** No string keys, no `eval`. A pulse is a typed value
  from a factory that reads calibration.
- **Escape hatch everywhere.** `StrumentoProgram` *is* an `AveragerProgramV2` — splice a raw call
  mid-sequence, or hand-write a body, while still using the abstractions for everything else.
- **The pure layers need no board.** The IR, device model, builder, compiler, fitters, and job
  transport all import and unit-test with no `qick` installed.
- **A measurement *instrument*, not just a runtime.** Author a device in code or YAML, measure the
  as-built device, and export its Hamiltonian for optimal control — closing a hardware↔pulse loop.

## Status

The modality-agnostic `core` and the `cqed` pack are built and green end-to-end through the
first six tracer-bullet slices: pulses, sweeps, amplitude Rabi, T1/T2 Ramsey, f0g1 + storage
swaps, parity, resonator spectroscopy, and the optimal-control import boundary
(`from_solution`). The device model — `DeviceBuilder`, typed couplings, Hamiltonian export —
is built; the `spin` pack covers quantum-dot workflows (slow-DC seam, charge stability, PSB
readout tune-up); `tweezer` ships as a skeleton. A vendored fitter library (resonator circle
fit, Fano, the damped-sinusoid family, double-Gaussian) and a `strumento` CLI round it out.
The suite runs offline; most of the logic needs no `qick` at all.

> **The v2 story:** building a *real* `AveragerProgramV2` needs a **tProc-v2** (`qick_processor`)
> soccfg. The repo commits the official qick **testbench** v2 config (single-channel), so real v2
> assembly runs offline and in CI; drop your own board's `dump_cfg()` at
> `devices/_fixtures/soccfg_v2.json` and the v2 suite runs against *your* firmware instead.

Roadmap: active reset (herald/post-selection data path, then in-sequence v2 feedback), the
tweezer pack ops, spinQICK-parity long tail (see [docs/spin-pack.md](docs/spin-pack.md)), and
the codesign arc (device-model reports, board descriptions, simulator soc).

## Documentation

| Doc | What's in it |
|---|---|
| **[architecture.md](docs/architecture.md)** | The design: tenets, three layers, every seam, and the decision index (D1–D31) code docstrings cite. |
| **[getting-started.md](docs/getting-started.md)** | Hands-on walkthrough with runnable examples — device, pulses, sequences, sweeps, fitting, experiments, the escape hatch, transport. |
| **[cqed-cookbook.md](docs/cqed-cookbook.md)** | A tour of the `cqed` pack by example. Every snippet verified against the live API. |
| **[packs-and-devices.md](docs/packs-and-devices.md)** | How elements compose into a device, the `DeviceBuilder`, typed couplings + Hamiltonian export, and **how to add a modality pack**. |
| **[timeline.md](docs/timeline.md)** | The cursor model — exactly when each op plays — plus the QICK-v2 redundancy audit. |
| **[calibration-store.md](docs/calibration-store.md)** | Operating a device over time — immutable, content-addressed calibration versions + write-back. |
| **[fitters.md](docs/fitters.md)** | The vendored fitter library — the resonator circle fit and friends. |
| **[spin-pack.md](docs/spin-pack.md)** | Quantum-dot workflows: the slow-DC seam, charge stability, PSB — and the coverage roadmap. |
| **[job-manager.md](docs/job-manager.md)** | How Strumento reaches a remote/shared board — the job server and the compile-off-board wire contract. |
| **[hardware-ci.md](docs/hardware-ci.md)** | Board-in-the-loop CI: the runner-on-the-board pattern and how to stand one up. |
| **[cli.md](docs/cli.md)** | The `strumento` CLI (agent-forward; `--json` everywhere). |
| **[qick-v2-api-notes.md](docs/qick-v2-api-notes.md)** | The exact offline qick tProc-v2 API Strumento builds against. |
| **[AGENTS.md](AGENTS.md)** | Orientation for AI agents: rules, recipes, what needs a board. |

## Architecture

Three layers; dependency arrows point down. Only the compiler, program, and mock touch qick.

```mermaid
flowchart TD
    subgraph core["core — modality-agnostic (no physics)"]
        W[wiring] --> P[pulse IR + Seq builder]
        P --> C[compiler] --> PR[program]
        SW[sweeps] --> C
        DEV[device + builder] --> P
        CPL[couplings + operators] --> DEV
        EX[experiment · fitters · result] --> PR
        RK[readout protocols] -.-> PR
        SOC[soc] --> PR
        TR[transport: JobServerSoc] -.-> SOC
    end
    subgraph packs["packs — modality kits"]
        CQED["cqed (built)"]
        TWZ["tweezer (skeleton)"]
        SPIN["spin (skeleton)"]
    end
    subgraph devices["devices — per-fridge instances"]
        SM[multimode_demo]
    end
    packs --> core
    devices --> packs
```

- **`core`** — modality-agnostic mechanism: wiring, the typed pulse IR, sweeps, the device model +
  `DeviceBuilder`, typed couplings + Hamiltonian export, the compiler + program, fitters, the
  experiment scaffold, the no-FPGA mock, and the soc + transport seams.
- **`packs`** — one reusable kit per physics class (element types + libraries + readout kind + calib
  schema), discovered via the `strumento.packs` entry-point group. A new pack is a new directory,
  not a core change.
- **`devices`** — one instance per physical fridge (topology + wiring + calibration values).

## Repository layout

```
strumento/
  strumento/
    core/         wiring · pulses · sweeps · device · builder · coupling · compiler ·
                  program · mock · result · readout · fitters · experiment · operators · soc · transport
    packs/
      cqed/       elements · calibration · couplings · library/ · readout · experiments/   (built)
      tweezer/    elements · calibration · envelopes · readout                             (skeleton)
      spin/       elements · calibration · readout                                         (skeleton)
    interop/      piccolo    — upstream bridges; core stays framework-neutral
  devices/
    multimode_demo/   device.yaml + soccfg snapshot   (cqed instance)
    tweezer_demo/     device.yaml                     (tweezer instance)
    spin_demo/            device.yaml                     (spin instance)
  docs/           spec · plan · getting-started · packs-and-devices · qick-v2-api-notes
  tests/          core/ · packs/ · interop/ · e2e/
```

## Where it fits in the Harmoniqs stack

Strumento is the open measurement/execution substrate at the bottom of the closed-loop calibration
stack. It knows nothing about calibration *strategy* — it imports a pulse, compiles, acquires, and
reduces to a measurement vector.

```
Intonatissimo   →  calibration method (private)
  Intonato       →  strategy-generic calibration chassis
    IntonatoQICK →  hardware backend
      strumento    →  from_solution → compile → acquire → reduce   ← this repo
        board
```

Optimal-control pulses enter through `from_solution` (Slice 6); the reverse seam,
`dev.to_operator_model(...)`, exports the device Hamiltonian for a solver — closing the codesign
loop.

## Install & develop

Strumento targets **Python 3.11+**. The pure layers need only `pydantic` / `numpy` / `scipy` /
`lmfit` / `pyyaml`; `qick` (with `qick.asm_v2`) is an optional extra for the compiler / program / mock.

```bash
pip install -e ".[test]"          # pure layers + pytest
pip install -e ".[test]" qick     # + qick-facing layers (installs off-board on x86 — no pynq)

pytest -m "not qick"              # the pure layers — no board
pytest                            # everything (v2-only tests skip without a tProc-v2 soccfg)
```

Development follows the plan task-by-task with a `pytest` TDD cycle.

## Naming

*Strumento* is Italian for **instrument** — QICK is the *Instrumentation* Control Kit, and Strumento
is the instrument you play it with. It joins the Harmoniqs family of Italian-musical names; we
avoided a `*QICK` name so it reads as *our layer on QICK*, not a fork.

## License

MIT — see [LICENSE](LICENSE).
