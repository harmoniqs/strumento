# The `spin` pack — quantum-dot workflows on the strumento core

Gate-defined quantum dots put two demands on a control framework that superconducting stacks
don't: **slow DC electrostatics** (an external voltage source biases plungers/barriers alongside
QICK's fast axes) and **charge-sensed readout** (PSB spin-to-charge, not dispersive IQ). The
`spin` pack covers both on the unchanged modality-agnostic core — the same device model, pulse
IR, compiler, and experiment lifecycle as the `cqed` pack.

The pack's shape follows the workflows proven by
[spinQICK](https://github.com/HRL-Laboratories/spinqick), the open quantum-dot package on QICK
tProc-v2, so practitioners coming from it will recognize the moving parts. Prior art is cited
where a design was adopted directly.

## What's here

| Piece | Where | What it does |
|---|---|---|
| Element types | `packs/spin/elements.py` | `Dot` (LD or exchange-only encoding), `Plunger`, `Barrier` — topology nodes with typed lines |
| LD drive ops | `packs/spin/library/ld.py` | `dev.q0.pi()/hpi(phase=…)` EDSR rotations from per-dot calibration (fraction-native gains) |
| Exchange ops | `packs/spin/library/exchange.py` | `dev.b0.pulse()` — fast baseband flat-top on the barrier's `fast` role |
| Slow-DC seam | `core/dc.py` | `VoltageSource` driver **Protocol** + `DCAxis`/`DCSource`: named-axis addressing, `max_v` clamps, lever-arm-compensated moves, voltage snapshots (spinQICK's proven pattern, made modality-agnostic) |
| Charge stability | `packs/spin/experiments/charge_stability.py` | host-driven 2D DC sweep with a fast sensor probe at each point |
| PSB readout | `packs/spin/readout.py` + `experiments/psb_histogram.py` | threshold discriminator + double-Gaussian histogram fit (`fit_double_gaussian`) → threshold + separation |
| Demo instance | `devices/spin_demo/` | a double-dot with gate map, cross-coupling, and calibrated pulses |

The DC seam is deliberately in **core** (not the pack): any modality with slow external axes —
flux biasing, laser piezo offsets — reuses `DCSource` unchanged.

## Design notes

- **No drivers ship here.** `VoltageSource` is a Protocol; wrap your instrument (QDAC, Basel
  LNHR, …) in ~10 lines. `MockVoltageSource` covers tests and offline work.
- **Gains are fractions.** Spin stacks are v2-native — there is no legacy int-code database, so
  calibration stores fractions and no boundary conversion happens (unlike `cqed`, D23).
- **Voltage state is data.** Every `ChargeStability` result carries a `dc_snapshot` in its
  metadata — the full gate-voltage state the map was taken at.

## Coverage roadmap (vs. the spinQICK experiment surface)

| Workflow | Status here |
|---|---|
| Gate turn-on / gate action / 1D sweeps | 🟡 compose from `DCSource` + a probe (no packaged experiment yet) |
| GvG charge stability (DC-swept) | ✅ `ChargeStability` |
| GvG baseband-fast (QICK gens DC-coupled) | ⏳ roadmap |
| Cross-capacitance extraction | 🟡 lever-arm matrix consumed (`crosscoupling`); auto-extraction roadmap |
| Sensor auto-retune | ⏳ roadmap |
| PSB histogram / threshold | ✅ `PSBHistogram` + `fit_double_gaussian` |
| PSB window scans (flush / idle / measure) | ⏳ roadmap (needs the SPAM step machinery) |
| LD spectroscopy / Rabi / Ramsey / echo / All-XY | 🟡 core experiments (Rabi/T1/T2) generalize; LD-specific wrappers roadmap |
| Exchange fingerprint / calibration | ⏳ roadmap |
| Noise / DSO capture | ⏳ roadmap |

The roadmap items are pack-local work — none of them needs a core change, which is the point of
the pack seam.
