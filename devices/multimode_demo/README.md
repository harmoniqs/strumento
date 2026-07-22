# `multimode_demo` — a generic multimode device instance

A **device instance** of the `cqed` pack: a generic "multimode" system —
`1×Transmon + 1×BosonicMode(manipulate) + 2×BosonicMode(storage) + Coupler + Readout`. This is the
worked example used throughout the docs and tests, and a template for describing your own device:
everything about *one physical fridge* lives here as data; the pack (element types, pulse
libraries, readout kind) is generic. The calibration values are synthetic but plausible.

```python
from strumento import Device
dev = Device.load("devices/multimode_demo/device.yaml")
```

## Files

- **`device.yaml`** — the instance: `pack`, a `topology` of named elements, the `wiring`
  (line/generator → DAC), and `calibration` values.
- The soccfg snapshot is shared from `../_fixtures/` — a committed `QickConfig.dump_cfg()`
  snapshot, so off-board work (unit conversions, `maxv` for gain) needs no board. See
  `devices/_fixtures/README.md` for the v1/v2 snapshot story.

## Topology → namespaces

Each named element binds a `dev.<name>` namespace (element types with ops classes are callable;
the rest are topology nodes until their ops land):

| Element | Type | `dev.<name>` |
|---|---|---|
| `qubit` | Transmon | `dev.qubit.ge_pi()` / `ge_hpi()` / `ef_pi()` |
| `manipulate` | BosonicMode | `dev.manipulate` |
| `S1`, `S2` | BosonicMode (storage) | `dev.S1`, `dev.S2` |
| `coupler` | Coupler | `dev.coupler` |
| `readout` | ReadoutResonator | readout kind (`DispersiveIQ`) |

The storage modes share one physical line (DAC 6) at different frequencies — modeled as separate
elements with a common `line: storage`.

## Wiring

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

`qubit.f_ge = 4000.0 MHz`, `f_ef = 3820.0 MHz`; `pi_ge = {gain: 8192, sigma: 0.030 µs}`
(gain is a lab-native int code — the factory converts it to a v2 fraction, `8192/32766 ≈ 0.2500`);
`readout = {frequency: 750.0 MHz, gain: 1000, readout_length: 2.0 µs}`.

Calibration *values* are the one thing that changes every run, so they stay data (never authored
in code): a calibration loop or a device-model report rewrites them, and versioning tracks them.
