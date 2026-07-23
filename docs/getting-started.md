# Getting started with Strumento

A hands-on walkthrough. Every example runs against the committed `multimode_demo` instance; the only step
that needs real hardware (or a tProc-v2 soccfg) is the final acquire, which is flagged where it
appears. If you're coming from a register-level tProc-v1 stack or raw QICK, this shows how the same experiments
look in device language.

## 1. Install

```bash
pip install -e ".[test]"          # pure layers + pytest
pip install -e ".[test]" qick     # add the compiler / program / mock (qick installs off-board on x86)
```

Quick check — the pure surface imports with no board and no `qick`:

```python
from strumento import Device, Seq, Sweep, Gaussian
```

## 2. Load a device

A **device instance** is a single YAML file (topology + wiring + calibration values). Loading it
gives you a `Device` whose elements are addressable by name:

```python
from strumento import Device

dev = Device.load("devices/multimode_demo/device.yaml")
print(dev)
# Device(pack='strumento.packs.cqed', elements=['S1', ..., 'coupler', 'manipulate', 'qubit', 'readout'])

dev.calib.qubit.f_ge          # 4000.0  (MHz)
dev.wiring.lines.keys()       # qubit, manipulate, storage, coupler
```

`dev.qubit` exists because the YAML names a transmon `qubit`. The pack never hardcodes that name —
a different instance yields `dev.q0` / `dev.cavity`. This is what "model the device, not the FPGA"
buys you: the wiring maps `qubit → drive → DAC 2`, and if you re-patch a cable you edit the YAML,
not a single experiment.

```python
from strumento import LineRef
dev.wiring.resolve(LineRef("qubit", "drive"), freq_mhz=4000).dac_ch    # -> 2
```

## 3. Build a pulse

Pulses are **typed values computed from calibration** — factory methods on an element's ops, not
lookups from a string bank:

```python
pi = dev.qubit.ge_pi()
pi.freq_mhz      # 4000.0        — the g–e frequency from calib
pi.gain          # 0.25001526…   — a v2 FRACTION
pi.envelope      # Gaussian(sigma_us=0.030)
```

That gain is worth a look: the calibration stores the lab-native int code `8192`, and the cqed
factory converts it to a fraction (`8192 / 32766`) at the pack boundary — so the compiler only ever
sees v2 fractions, and the v1→v2 gain-units bug can't happen. Override any field:

```python
dev.qubit.ge_pi(phase=90)        # rotate the axis
dev.qubit.ge_pi(gain=0.2)        # an explicit gain is a raw v2 fraction, used as-is
dev.qubit.ge_hpi()               # a π/2
dev.qubit.ef_pi()                # the e–f transition (uses f_ef)
```

## 4. Compose a sequence

`Seq` is a fluent builder over an ordered op list:

```python
from strumento import Seq

seq = (Seq()
       .play(dev.qubit.ge_pi())
       .wait(0.5)               # µs
       .measure())

[type(op).__name__ for op in seq.ops]     # ['Pulse', 'Delay', 'Measure']
```

`.play` also flattens a nested `Seq`, `.play_parallel([...])` plays pulses at the same time, and
`.raw(fn)` drops to raw qick (see §9).

## 5. Sweep a parameter

A `Sweep` stands in for any pulse field. Sweep in the units your lab thinks in — the factory maps
them to hardware units, but keeps your axis for plotting:

```python
from strumento import Sweep

amp = Sweep("amp", 0, 30000, 51)          # lab-native gain int codes
p = dev.qubit.ge_pi(gain=amp)

p.gain.start, p.gain.stop                 # 0.0, 0.915…   — the fraction sweep the hardware runs
p.gain.xpts[0], p.gain.xpts[-1]           # 0, 30000      — the axis your result reports

rabi = Seq().play(p).measure()
```

The compiler turns this into one `add_loop` over 51 points plus a `QickSweep1D` on the gain — you
never touch a loop register. (A frequency sweep that would cross a generator's band boundary is
rejected with a clear error.)

## 6. Run it

Running goes through `StrumentoProgram`, which *is* an `AveragerProgramV2`:

```python
from strumento.core.program import StrumentoProgram
from strumento.core.mock import MockQickSocV2

soc = MockQickSocV2.from_snapshot("devices/_fixtures/soccfg_v2.json")
prog = StrumentoProgram(dev, seq=rabi, reps=1000, soc=soc)
result = prog.acquire(soc)                  # -> MeasurementResult
xpts, avgi, avgq = result.as_tuple()        # xpts == the 0..30000 int-code axis
```

> **This step needs a tProc-v2 soccfg.** `StrumentoProgram` assembles a real `AveragerProgramV2`,
> which requires a `qick_processor` config. The committed board snapshot is tProc-**v1**; drop a
> v2 `dump_cfg()` at `devices/_fixtures/soccfg_v2.json` to enable it (see
> `devices/_fixtures/README.md`). Until then, everything up to and including *compilation* works —
> the compiler is verified against a recording program — you just can't acquire offline yet.

## 7. Fit the result

Fitters are `lmfit`-backed and return a typed `FitResult` — no printing, no plots:

```python
from strumento.core.fitters import fit_rabi

fit = fit_rabi(xpts, signal)         # signal = readout-projected IQ vs the gain axis
fit.derived["pi_gain"]               # the drive amplitude for a π rotation
fit.derived["period"]
fit.params["amp"], fit.stderr["amp"]
```

To turn raw IQ into that `signal`, use a **readout kind**. The cqed `DispersiveIQ` rotates the IQ
cloud onto the discrimination axis:

```python
from strumento.packs.cqed.readout import DispersiveIQ
disc = DispersiveIQ(threshold=0.0, angle=0.3)
disc.reduce(iq)          # per-point excited-state population
disc.discriminate(iq)    # per-shot 0/1
```

## 8. Wrap it as an experiment

The `Experiment` scaffold pairs a sequence with a fit and separates acquire / analyze / display:

```python
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi

exp = AmplitudeRabi(dev, gains=Sweep("amp", 0, 30000, 51))

seq, sweeps = exp.sequence()             # inspect what it will run (no board needed)
outcome = exp.go(soc)                    # acquire + analyze  (needs the v2 soc from §6)
outcome.fit.derived["pi_gain"]
```

`analyze` works on any `MeasurementResult`, so you can develop and test the analysis with synthetic
data before you ever touch hardware.

## 9. The escape hatch (Tenet 0)

Nothing is walled off. Three ways to reach raw qick, from most to least surgical:

```python
# a) splice one raw call mid-sequence
Seq().play(dev.qubit.ge_pi()).raw(lambda p: p.delay_auto(0.01)).measure()

# b) StrumentoProgram IS an AveragerProgramV2 — call any qick method on it
# c) hand-write _body(self, cfg) for a fully bespoke program, still using dev/Seq elsewhere
```

## 10. The board seam — soc + transport

Strumento talks to the board through a small **soc protocol** (`AbstractSoc`) — the mock, a real
`QickSoc`, and a simulator all satisfy it. To reach a *remote* board, a **transport** ships a job:

```python
from strumento.core.transport import CompiledJob, JobServerSoc

job = CompiledJob(overlay_id="cqed-multimode-v2", program=compiled, acquire=acq_cfg)
raw = JobServerSoc(client).run(job)        # submit + poll-to-done + deserialize
result = raw.to_measurement_result(xpts)
```

`JobServerSoc` is deliberately coarse — one job in, IQ out — not a per-call proxy. Swapping the
board, or the transport to it, changes nothing above this seam.

## Where to next

- [architecture.md](architecture.md) — the design: tenets, layers, seams, and the decision index (D1–D31).
- `docs/qick-v2-api-notes.md` — the exact offline qick tProc-v2 API strumento builds against.
- `devices/multimode_demo/README.md` — anatomy of a device instance.
