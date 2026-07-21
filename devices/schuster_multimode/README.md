# `schuster_multimode` — a device instance

A **device instance** of the `cqed` pack: the Stanford Schuster-lab "multimode" system —
`1×Transmon + 1×BosonicMode(manipulate) + 7×BosonicMode(storage) + Coupler + Readout`. This is a
worked example of the config-as-data model: everything about *this fridge* lives here as data;
the pack (element types, pulse libraries, readout kind) is generic.

```python
from strumento import Device
dev = Device.load("devices/schuster_multimode/device.yaml")
```

## Files

- **`device.yaml`** — the instance: `pack`, a `topology` of named elements, the `wiring`
  (line/generator → DAC), and `calibration` values. Numbers are transcribed from the v1
  `hardware_config.yml` (`device.*`), flattening its one-element-list idiom to scalars.
- **`soccfg_snapshot.json`** — a committed `QickConfig.dump_cfg()` snapshot, so off-board work
  (unit conversions, `maxv` for gain, and — with a v2 snapshot — full program assembly) needs no
  board. *This committed snapshot is tProc-**v1*** (the multimode board); a tProc-v2 snapshot for
  real `AveragerProgramV2` assembly goes at `../_fixtures/soccfg_v2.json` (see that README).

## Topology → namespaces

Each named element binds a `dev.<name>` namespace (only element types with ops bound so far are
callable; the rest are topology nodes today):

| Element | Type | `dev.<name>` | Status |
|---|---|---|---|
| `qubit` | Transmon | `dev.qubit.ge_pi()` / `ge_hpi()` / `ef_pi()` | ✅ |
| `manipulate` | BosonicMode | `dev.manipulate` | ⏳ ops in slice 4 |
| `S1`…`S7` | BosonicMode (storage) | — | ⏳ slice 4 |
| `coupler` | Coupler | `dev.coupler.swap(...)` | ⏳ slice 4 |
| `readout` | ReadoutResonator | readout kind (`DispersiveIQ`) | ✅ |

The 7 storage modes share one physical line (DAC 6) at different frequencies — modeled as separate
elements with a common `line: storage`.

## Wiring (from v1 `hw.soc.dacs`)

| Line | Role | DAC | Nyquist |
|---|---|---|---|
| `qubit` | drive | 2 | 1 |
| `qubit` | sideband | 0 | 1 |
| `manipulate` | main | 3 | 2 |
| `storage` | main | 6 | 2 |
| `coupler` | low / high (band-gated) | 1 | 1 |
| readout | probe DAC 5 + ADC 0 | 5 | — |

`resolve` picks the generator by line + role (and frequency, for band-gated lines). Re-patching a
cable is a one-line edit to this table — every experiment and pulse factory is untouched, because
they speak only device language.

## Calibration (excerpt)

`qubit.f_ge = 3562.6105 MHz`, `f_ef = 3419.0360 MHz`; `pi_ge = {gain: 3289, sigma: 0.035 µs}`
(gain is a lab-native int code — the factory converts it to a v2 fraction, `3289/32766 ≈ 0.1004`);
`readout = {frequency: 749.4329 MHz, gain: 800, readout_length: 1.5 µs, relax_delay: 2500 µs}`.

Calibration *values* are the one thing that changes every run, so they stay data (never authored in
code): a calibration loop or a device-model report rewrites them, and versioning tracks them.
