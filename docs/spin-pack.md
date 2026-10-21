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
| Element types | `packs/spin/elements.py` | `Dot` (LD), `STDot` (singlet-triplet), `EODot` (exchange-only), `HoleDot` (Ge hole), `Plunger`, `Barrier`, `Reservoir`, `ChargeSensor`, `DriveLine`, `GateSensor` — 10 topology node types |
| LD drive ops | `packs/spin/library/ld.py` | `dev.q0.pi()/hpi(phase=…)` EDSR rotations from per-dot calibration (fraction-native gains) |
| Exchange ops | `packs/spin/library/exchange.py` | `dev.b0.pulse()` — fast baseband flat-top on the barrier's `fast` role |
| Hole-spin ops | `packs/spin/library/hole_spin.py` | AC gate drive through plunger (SOC-mediated EDSR), `exchange_cz` for dispersive CZ |
| S-T ops | `packs/spin/library/singlet_triplet.py` | `j_pi()/j_hpi()` exchange rotations, `z_idle()`, `singlet_prep()` |
| Composite library | `packs/spin/library/composite.py` | BB1, CORPSE, Knill composites; active reset (Branch-based); `load_oc_pulse` for OC import |
| EO Clifford | `packs/spin/library/eo_clifford.py` | 24 Cliffords decomposed into J₁₂/J₂₃ exchange pulses for exchange-only RB |
| Readout kinds | `packs/spin/readout.py` | `PSBReadout`, `ElzermanReadout`, `DispersiveGateReadout`, `LatchedPSBReadout` |
| Slow-DC seam | `core/dc.py` | `VoltageSource` driver **Protocol** + `DCAxis`/`DCSource` (modality-agnostic) |
| Experiments | `packs/spin/experiments/` | 21 registered + 8 DCSource-based + 2 standalone |
| Demo instances | `devices/spin_demo/`, `spin_st_demo/`, `spin_eo_demo/`, `spin_hole_demo/` | LD, S-T, EO, and hole-spin device fixtures |

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
| Gate turn-on / gate action / 1D sweeps | ✅ `PinchOff`, `SensorTune`, `TunnelCouplingTune`, `DispersiveGateCalibration` (DCSource-based) |
| GvG charge stability (DC-swept) | ✅ `ChargeStability` (+ `analyze()` with gradient edge detection) |
| GvG baseband-fast (QICK gens DC-coupled) | ⏳ roadmap |
| Cross-capacitance extraction | ✅ `VirtualGateCalibration` (DCSource); lever-arm matrix in `crosscoupling` |
| Sensor auto-retune | ✅ `SensorTune` (DCSource) |
| PSB histogram / threshold | ✅ `PSBHistogram` + `fit_double_gaussian` |
| PSB window scans (flush / idle / measure) | ✅ `PSBCalibration` (registered) |
| LD spectroscopy / Rabi / Ramsey / echo / All-XY | ✅ `EDSRSpectroscopy`, `SpinRabi`, `SpinRamsey`, `HahnEcho`, `CPMG`, `AllXY` (all registered) |
| Exchange fingerprint / calibration | ✅ `ExchangeOscillation`, `ExchangeGateCalibration` (registered) |
| Benchmarking | ✅ `RandomizedBenchmarking`, `InterleavedRB`, `LeakageBenchmark`, `CompositePulseCalibration` |
| Active reset | ✅ `HeraldedInit` (Branch-based, ShotMachine response; needs per-shot data — on a result without shots it refuses, `derived["refused"]`) |
| Noise / DSO capture | ⏳ roadmap |

The roadmap items are pack-local work — none of them needs a core change, which is the point of
the pack seam.

## Writing a device YAML

A spin device YAML declares `pack: spin`, then three sections: `wiring`, `topology`, and
`calibration`. The structure varies by encoding. Shipped examples: `devices/spin_demo/` (LD),
`devices/spin_eo_demo/` (exchange-only), `devices/spin_st_demo/` (singlet-triplet),
`devices/spin_hole_demo/` (Ge hole).

### Topology — element types and encoding

```yaml
topology:
  - {type: Dot, name: q0}                      # LD (default encoding)
  - {type: Dot, name: q0, encoding: hole}       # Ge hole-spin
  - {type: Dot, name: q1, encoding: st}          # singlet-triplet
  - {type: Dot, name: q1, encoding: eo}          # exchange-only
  - {type: Plunger, name: p0}
  - {type: Barrier, name: b0}
  - {type: ChargeSensor, name: sensor, sensor_type: qpc}
  - {type: Reservoir, name: res}
  - {type: DriveLine, name: antenna}             # dedicated MW antenna (optional)
  - {type: GateSensor, name: gs0}                # dispersive gate sensor (optional)
```

The encoding determines which ops class binds to the dot namespace:
- `ld` (default) → `LDOps` (`dot.pi()`, `dot.hpi()` — EDSR)
- `hole` → `HoleSpinOps` (AC gate drive through plunger, `drive` role)
- `st` → `SingletTripletOps` (exchange + detuning, `detuning` role)
- `eo` → no dot-level ops (all control via barrier `ExchangeOps`)

### Wiring — lines, generators, readout

Each element with a driven line needs a wiring entry mapping its roles to DAC channels:

```yaml
wiring:
  lines:
    q0:                                          # LD dot: one drive generator
      name: q0
      generators:
        - {role: drive, dac_ch: 0, nyquist: 1}
    b0:                                          # barrier: DC + fast exchange
      name: b0
      generators:
        - {role: dc, dac_ch: 2, nyquist: 1}
        - {role: fast, dac_ch: 2, nyquist: 1}
  readout:
    dac_ch: 3
    adc_chs: [0]
```

Line roles by element type:
- `Dot` (LD/hole): `drive` — the EDSR or AC gate drive
- `Dot` (st): `detuning` — baseband detuning axis
- `Dot` (eo): no lines (control via barriers)
- `Barrier`: `dc` (slow bias) + `fast` (exchange pulse)
- `Plunger`: `dc`
- `ChargeSensor`: `dc` + `readout`
- `GateSensor`: `dc` + `readout`

### Calibration — per-element values

```yaml
calibration:
  dots:
    q0:
      f_qubit_mhz: 5000.0                       # Larmor / resonance frequency
      pulses:
        pi:  {gain: 0.4, sigma: 0.02}           # Gaussian pi (LD/hole)
        hpi: {gain: 0.2, sigma: 0.02}           # Gaussian pi/2
      t1_us: 50.0                                # measured T1 (optional)
      t2_star_us: 2.5                            # measured T2* (optional)
      g_factor: 2.0                              # effective g (optional, hole-spin)
  gates:
    b0:
      dc_channel: 1
      max_v: 1.5
      exchange: {gain: 0.4, length: 0.012, type: flat_top}   # calibrated exchange pi
      crosscoupling: {p0: 0.08}                  # lever-arm compensation
    p0:
      dc_channel: 0
      max_v: 2.0
  sensors:
    sensor: {v_operating: -0.35, transconductance: 0.02}
  readout:
    threshold: 0.5
    fidelity: 0.95
```

Gains are always v2 fractions (D23). Pulse `sigma` is in microseconds; `length` is flat-top
duration in microseconds. The calibration schema (`packs/spin/calibration.py`) validates all
fields strictly — a mistyped key in the YAML fails loudly at load time.
