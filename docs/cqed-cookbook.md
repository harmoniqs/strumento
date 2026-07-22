# cqed cookbook

A tour of the `cqed` pack by example — superconducting transmon + bosonic. Every snippet here was
run against the live API; the printed values are real. Copy-paste from the top down, or jump to a
section.

Unless noted, examples are **pure** — no board, no `qick` needed. The one exception is *acquiring*
on the mock (§6), which needs a tProc-v2 soccfg; it's flagged where it appears.

**Contents**
1. [Load & inspect a device](#1-load--inspect-a-device)
2. [Single-qubit pulses](#2-single-qubit-pulses)
3. [Sequences](#3-sequences)
4. [Sweeps](#4-sweeps)
5. [What will the hardware play? — `quantize`](#5-what-will-the-hardware-play--quantize)
6. [Running on the mock](#6-running-on-the-mock)
7. [Readout — `DispersiveIQ`](#7-readout--dispersiveiq)
8. [Fitting — `fit_rabi`](#8-fitting--fit_rabi)
9. [The `AmplitudeRabi` experiment](#9-the-amplituderabi-experiment)
10. [Authoring a device in code — `DeviceBuilder`](#10-authoring-a-device-in-code--devicebuilder)
11. [Couplings, forms & patterns](#11-couplings-forms--patterns)
12. [Hamiltonian export — `to_operator_model`](#12-hamiltonian-export--to_operator_model)
13. [Calibration & gain units](#13-calibration--gain-units)
14. [Roadmap — not yet built](#14-roadmap--not-yet-built)

---

## 1. Load & inspect a device

```python
from strumento import Device, LineRef

dev = Device.load("devices/schuster_multimode/device.yaml")

dev                                   # Device(pack='strumento.packs.cqed', elements=['S1', ... 'qubit', 'readout'])
dev.calib.qubit.f_ge                  # 3562.6105332943075   (MHz)
list(dev.wiring.lines)                # ['qubit', 'manipulate', 'storage', 'coupler']
dev.wiring.resolve(LineRef("qubit", "drive"), freq_mhz=3562).dac_ch   # 2
```

`dev.qubit` is bound because the instance YAML names a transmon `qubit` (a different YAML yields
`dev.q0`/`dev.cavity`). The 7 storage modes (`S1…S7`) share one physical line.

## 2. Single-qubit pulses

Pulses are typed values computed from calibration — factory methods on the transmon's ops:

```python
p = dev.qubit.ge_pi()
p.freq_mhz        # 3562.6105332943075   — the g–e frequency, from calib
p.gain            # 0.10037844…          — a v2 FRACTION (calib int code 3289 / maxv 32766)
p.envelope        # Gaussian(sigma_us=0.035)

dev.qubit.ge_pi(phase=90).phase_deg     # 90     — override the phase (degrees)
dev.qubit.ge_pi(gain=0.2).gain          # 0.2    — an explicit gain is a raw v2 fraction, used as-is
dev.qubit.ge_hpi().gain                 # 0.049716…   — a π/2 (calib code 1629)
dev.qubit.ef_pi().freq_mhz              # 3419.036012060942   — the e–f transition (uses f_ef)
```

## 3. Sequences

`Seq` is a fluent builder over an ordered op list:

```python
from strumento import Seq

seq = Seq().play(dev.qubit.ge_pi()).wait(0.5).measure()
[type(op).__name__ for op in seq.ops]        # ['Pulse', 'Delay', 'Measure']   (wait is in µs)

# play two pulses simultaneously
seg = Seq().play_parallel([dev.qubit.ge_pi(), dev.qubit.ge_hpi()])
len(seg.ops[0].pulses)                        # 2

# drop to raw qick mid-sequence, then keep going (Tenet 0)
Seq().play(dev.qubit.ge_pi()).raw(lambda prog: prog.delay_auto(0.01)).measure()

# .play also flattens a nested Seq; Seq + Seq concatenates without mutating either
```

## 4. Sweeps

A `Sweep` stands in for any pulse field. Sweep in lab-native units; the factory maps them to
hardware units but keeps your axis for plotting (`.xpts`):

```python
from strumento import Sweep

amp = Sweep("amp", 0, 30000, 51)          # lab-native gain int codes
p = dev.qubit.ge_pi(gain=amp)

p.gain.start, p.gain.stop                 # 0.0, 0.9156…   — the fraction sweep the hardware runs
p.gain.xpts[-1]                           # 30000.0        — the axis your result reports

rabi = Seq().play(p).measure()            # the compiler turns the sweep into one add_loop
```

A frequency sweep that would cross a generator's band boundary is rejected with a clear error.

## 5. What will the hardware play? — `quantize`

`quantize` snaps a pulse to the generator's DAC resolution — inspect it with no board (the unit math
only needs the soccfg snapshot):

```python
from qick import QickConfig
from strumento.core.compiler import quantize

soccfg = QickConfig(dev.soccfg_snapshot)
r = quantize(dev.qubit.ge_pi(), soccfg, dev.wiring)
r.freq_mhz_actual   # 3562.6105   — grid-snapped
r.gain_frac         # 0.100378    — realized to amplitude resolution
r.cycles            # 56          — a 4σ (0.14 µs) Gaussian window
```

## 6. Running on the mock

Running goes through `StrumentoProgram`, which *is* an `AveragerProgramV2`:

```python
from strumento.core.program import StrumentoProgram
from strumento.core.mock import MockQickSocV2

soc = MockQickSocV2.from_snapshot("devices/_fixtures/soccfg_v2.json")
prog = StrumentoProgram(dev, seq=rabi, reps=1000, soc=soc)
result = prog.acquire(soc)                # -> MeasurementResult
xpts, avgi, avgq = result.as_tuple()      # xpts is the 0..30000 int-code axis
```

> ⚠️ **Needs a tProc-v2 soccfg.** `StrumentoProgram` assembles a real `AveragerProgramV2`
> (`qick_processor`). The committed Schuster snapshot is tProc-**v1**, so drop a v2 `dump_cfg()` at
> `devices/_fixtures/soccfg_v2.json` to run this (see `devices/_fixtures/README.md`). Everything
> above (build, sweep, quantize) and the analysis below need no board.

## 7. Readout — `DispersiveIQ`

Turn raw IQ into physics: rotate the cloud onto the discrimination axis, threshold per shot, reduce
to a population.

```python
import numpy as np
from strumento.packs.cqed.readout import DispersiveIQ

disc = DispersiveIQ(threshold=0.0, angle=0.0)      # angle in radians
iq = np.array([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1]])   # per-shot (I, Q)
disc.discriminate(iq)     # array([1, 0, 1])   — per-shot |g>=0 / |e>=1
disc.reduce(iq)           # 0.667              — excited-state population
disc.project(iq)          # rotated-I signal (for fitting averaged IQ directly)
```

## 8. Fitting — `fit_rabi`

`lmfit`-backed, returns a typed `FitResult` (no printing, no plots):

```python
import numpy as np
from strumento.core.fitters import fit_rabi

x = np.linspace(0, 30000, 121)
y = 0.5 + 0.4 * np.cos(2 * np.pi * x / 20000)     # a synthetic Rabi trace, period 20000
fit = fit_rabi(x, y)
fit.derived["period"]     # 20000.0
fit.derived["pi_gain"]    # 10000.0   — half the period, the π drive amplitude
fit.params["amp"], fit.stderr["amp"]
```

## 9. The `AmplitudeRabi` experiment

Pairs the sequence with the fit and separates acquire / analyze / display. `analyze` works on any
`MeasurementResult`, so you can develop the analysis with synthetic data before touching hardware:

```python
import numpy as np
from strumento import Sweep
from strumento.core.result import MeasurementResult
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi

exp = AmplitudeRabi(dev, gains=Sweep("amp", 0, 30000, 51))
seq, sweeps = exp.sequence()          # inspect what it will run — no board needed
[type(o).__name__ for o in seq.ops]   # ['Pulse', 'Measure']

# analyze a (here synthetic) result:
x = np.linspace(0, 30000, 51)
res = MeasurementResult(xpts=x, avgi=0.5 + 0.4 * np.cos(2 * np.pi * x / 20000), avgq=np.zeros_like(x))
exp.analyze(res).derived["pi_gain"]   # 10000.0

# on hardware / the v2 mock, the whole thing is one call:
# outcome = exp.go(soc); outcome.fit.derived["pi_gain"]
```

## 10. Authoring a device in code — `DeviceBuilder`

`DeviceBuilder` produces the *same* canonical `Device` as loading YAML — useful for programmatic or
parametric topologies. Calibration values stay **data** (passed to `build(calib=…)`):

```python
from strumento.core.builder import DeviceBuilder
from strumento.core.wiring import Generator

calib = {
    "qubit": {"f_ge": 4200.0, "f_ef": 4050.0, "pulses": {"pi_ge": {"gain": 9000, "sigma": 0.03}}},
    "readout": {"frequency": 700.0, "gain": 500, "readout_length": 2.0, "trig_offset": 400, "relax_delay": 2000},
}
dev2 = (DeviceBuilder(pack="cqed")
        .add("Transmon", "q")
        .add("ReadoutResonator", "ro")
        .wire("q", drive=Generator(dac_ch=0))       # the keyword ('drive') is the role
        .readout(dac_ch=5, adc_chs=[0])
        .build(calib=calib))

sorted(dev2.elements)     # ['q', 'ro']
dev2.q.ge_pi().freq_mhz   # 4200.0
```

## 11. Couplings, forms & patterns

A physical coupler (with a DAC) is an *element* you `add`; an *interaction* is a first-class typed
**edge**. Build a 4-transmon ring with one line of code:

```python
from strumento.core.builder import DeviceBuilder
from strumento.core.wiring import Generator
from strumento.core.coupling import ring_couple
from strumento.packs.cqed.couplings import ZZ

calib = {
    "qubit": {"f_ge": 5000.0, "f_ef": 4800.0, "pulses": {}},
    "readout": {"frequency": 700.0, "gain": 500, "readout_length": 2.0, "trig_offset": 400, "relax_delay": 2000},
    "couplings": {"q0-q1": 0.3, "q1-q2": 0.3, "q2-q3": 0.3, "q3-q0": 0.3},   # strengths (MHz) = DATA
}
rb = DeviceBuilder(pack="cqed")
for i in range(4):
    rb.add("Transmon", f"q{i}").wire(f"q{i}", drive=Generator(dac_ch=i))
rb.add("ReadoutResonator", "ro").readout(dac_ch=5, adc_chs=[0])
ring_couple(rb, [f"q{i}" for i in range(4)], form=ZZ)     # a pattern emits typed edges
ring = rb.build(calib=calib)

[(c.a, c.b, c.form.__name__, c.symmetry) for c in ring.couplings]
# [('q0','q1','ZZ','symmetric'), ('q1','q2','ZZ','symmetric'),
#  ('q2','q3','ZZ','symmetric'), ('q3','q0','ZZ','symmetric')]
```

cqed registers five forms — `Dispersive`, `ZZ`, `Exchange`, `BeamSplitter`, `JaynesCummings` — each
carrying a symmetry tag and an operator template. A single mediated edge looks like:

```python
from strumento.packs.cqed.couplings import BeamSplitter
rb.coupling("manipulate", "S3", form=BeamSplitter, mediated_by="coupler")
```

## 12. Hamiltonian export — `to_operator_model`

The topology + couplings + calibration **are** a Hamiltonian spec. Export it for a solver — the
mirror of `from_solution`:

```python
import numpy as np

model = ring.to_operator_model(frame="rotating", cutoffs={f"q{i}": 2 for i in range(4)})
model.H0.shape                              # (16, 16)   — 4 qubits × 2 levels
np.allclose(model.H0, model.H0.conj().T)    # True       — hermitian
sorted(model.drives)                        # ['q0', 'q1', 'q2', 'q3']   — one drive op per driven line
model.drive_map["q0"]                       # 'q0:x'

# include= extracts a subgraph, so you build a tractable Hilbert space instead of the whole device:
sub = ring.to_operator_model(frame="rotating", cutoffs={"q0": 3, "q1": 3}, include=["q0", "q1"])
sub.H0.shape                                # (9, 9)
sub.to_dict()                               # JSON-serializable wire format for an upstream consumer

# shape it for Piccolo (upstream bridge; core stays Piccolo-free):
from strumento.interop.piccolo import to_piccolo_system
spec = to_piccolo_system(sub)
spec.drive_labels                           # ['q0', 'q1']
spec.H_drift.shape                          # (9, 9)
```

`frame="lab"` keeps each element's bare ω·n; `"rotating"` drops it, leaving anharmonicity + couplings.

## 13. Calibration & gain units

Calibration is a typed Pydantic schema; the lab-native gain **int codes** convert to v2 **fractions**
at the pack boundary (this is the one place the v1→v2 gain-units bug can't sneak in):

```python
from strumento.packs.cqed.calibration import gain_int_to_frac

gain_int_to_frac(3289, 32766)     # 0.100378…   — calib code 3289 at maxv 32766
dev.calib.qubit.pulses["pi_ge"]   # PulseCal(gain=3289, sigma=0.035, length=0.0, type='gauss')
dev.calib.readout.frequency       # 749.4328673317561
```

## 14. Roadmap — not yet built

These are designed (see `docs/spec.md`) but not yet implemented — they'll appear as the slices land,
following the same patterns above:

- **Slice 3 — T1 / T2 Ramsey:** swept-delay experiments + `fit_exp` / `fit_decaysin_detuned`.
- **Slice 4 — f0g1 + storage swap:** `dev.sideband.f0g1()`, `dev.coupler.swap("M1-S3")`, the
  `FlatTop` envelope, and the multi-line device model.
- **Slice 5 — parity:** `dev.parity(mode="manipulate")` composite factories.
- **Slice 6 — optimal-control import:** `from_solution(...)`, the Piccolo × QICK seam (the return
  leg of §12's export).

See **[getting-started.md](getting-started.md)** for the narrative walkthrough and
**[packs-and-devices.md](packs-and-devices.md)** for how the pieces compose and how to add a pack.
