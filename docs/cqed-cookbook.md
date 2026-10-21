# cqed cookbook

A reference, by example, for the `cqed` pack: superconducting transmons, readout resonators and
bosonic modes. Each section is self-contained enough to look something up; read
[getting-started.md](getting-started.md) first for the walkthrough (install, the mock and the
simulator, the calibration loop, going to hardware).

Every Python block here runs in the docs test (`tests/test_docs.py`), top to bottom in one
session, from the repository root, and the output shown under a block is checked. Comments of
the form `# 4000.0` are illustrations and are not checked. The examples need the `qick` extra
only where they build or run a program (sections 5, 6, 9 and 15, and the host-batched
diagnostics at the end of section 14); everything else is pure.

**Contents**
1. [Load and inspect a device](#1-load-and-inspect-a-device)
2. [Single-qubit pulses](#2-single-qubit-pulses)
3. [Sequences](#3-sequences)
4. [Sweeps and axes](#4-sweeps-and-axes)
5. [What will the hardware play? `realized`](#5-what-will-the-hardware-play-realized)
6. [Building and acquiring a program yourself](#6-building-and-acquiring-a-program-yourself)
7. [Readout: `DispersiveIQ`](#7-readout-dispersiveiq)
8. [Fitting: `fit_rabi`](#8-fitting-fit_rabi)
9. [The `AmplitudeRabi` experiment](#9-the-amplituderabi-experiment)
10. [Authoring a device in code: `DeviceBuilder`](#10-authoring-a-device-in-code-devicebuilder)
11. [Couplings, forms and patterns](#11-couplings-forms-and-patterns)
12. [Hamiltonian export: `to_operator_model`](#12-hamiltonian-export-to_operator_model)
13. [Calibration and gain units](#13-calibration-and-gain-units)
14. [Experiments by category](#14-experiments-by-category)
15. [Calibration write-back](#15-calibration-write-back)
16. [See also](#16-see-also)

---

## 1. Load and inspect a device

`devices/multimode_demo` is the cqed exemplar: a transmon, a readout resonator, three bosonic
modes and a coupler, wired to a six-generator tProc-v2 board.

```python
from strumento import Device, LineRef

dev = Device.load("devices/multimode_demo/device.yaml")

print(dev)
print(dev.calib.qubit.f_ge)                                            # MHz
print(list(dev.wiring.lines))
print(dev.wiring.resolve(LineRef("qubit", "drive"), freq_mhz=4000).dac_ch)
```

<!-- snippet: output -->
```text
Device(pack='strumento.packs.cqed', elements=['S1', 'S2', 'cavity', 'coupler', 'qubit', 'readout'])
4000.0
['qubit', 'cavity', 'storage', 'coupler', 'readout']
1
```

Element namespaces come from the instance: `dev.qubit` exists because this YAML names a transmon
`qubit` (a different YAML yields `dev.q0`, `dev.c1`). The storage modes `S1` and `S2` share
one physical line, `storage`. `wiring.resolve` maps a line and role to the generator that plays
it at a given frequency.

## 2. Single-qubit pulses

Pulses are typed values computed from calibration, by factory methods on the element's ops:

```python
p = dev.qubit.ge_pi()
print(p.freq_mhz, f"{p.gain:.8f}", p.envelope)

print(dev.qubit.ge_pi(phase=90).phase_deg)    # override the phase, in degrees
print(dev.qubit.ge_pi(gain=0.2).gain)         # an explicit gain is a v2 fraction, used as is
print(f"{dev.qubit.ge_hpi().gain:.8f}")       # π/2: calibration code 4096
print(dev.qubit.ef_pi().freq_mhz)             # the e–f transition, from f_ef
```

<!-- snippet: output -->
```text
4000.0 0.25001526 Gaussian(sigma_us=0.03)
90
0.2
0.12500763
3820.0
```

The calibration stores the π gain as the lab's integer code, 8192; the pulse carries it as the
tProc-v2 fraction 8192 / 32766 (section 13).

## 3. Sequences

`Seq` is a builder over an ordered list of ops. The op types (`Delay`, `Measure`, …) live in
`strumento.core.pulses`:

```python
from strumento import Seq
from strumento.core.pulses import Delay, Measure

seq = Seq().play(dev.qubit.ge_pi()).wait(0.5).measure()       # wait is in µs
print([type(op).__name__ for op in seq.ops])

# play two pulses simultaneously
seg = Seq().play_parallel([dev.qubit.ge_pi(), dev.qubit.ge_hpi()])
print(type(seg.ops[0]).__name__, len(seg.ops[0].pulses))

# drop to raw qick mid-sequence, then keep going (Tenet 0: every abstraction has a bypass)
raw = Seq().play(dev.qubit.ge_pi()).raw(lambda prog: prog.delay_auto(0.01)).measure()
print([type(op).__name__ for op in raw.ops])
```

<!-- snippet: output -->
```text
['Pulse', 'Delay', 'Measure']
Segment 2
['Pulse', 'Raw', 'Measure']
```

`.play` also flattens a nested `Seq`, and `a + b` concatenates two sequences without mutating
either. There is deliberately no `seq + pulse` form.

### Three ways to write a sequence

Chaining suits one-liners, but it diffs badly: change one pulse in a long chain and the whole
expression is one changed line. For a sequence that lives in a repository, build it line by line
with the `+=` accumulator, so each op is its own line, optionally scoped in a `with` block:

```python
# 1. chain: one-liners and quick notebook cells
chained = Seq().play(dev.qubit.ge_pi()).wait(0.5).measure()

# 2. accumulator: diff-friendly, and ops can be added across notebook cells
accumulated = Seq()
accumulated += dev.qubit.ge_pi()
accumulated += Delay(0.5)
accumulated += Measure()

# 3. with-block: the same accumulator, visually scoped
with Seq() as ramsey:
    ramsey += dev.qubit.ge_hpi()
    ramsey.wait(0.2)                    # the chaining methods work on the same object
    ramsey += dev.qubit.ge_hpi(phase=90)
    ramsey.measure()

print(chained.ops == accumulated.ops)
print([type(op).__name__ for op in ramsey.ops])
```

<!-- snippet: output -->
```text
True
['Pulse', 'Delay', 'Pulse', 'Measure']
```

The three forms build the same `Seq`; mix them freely.

### Where the config-to-parameter boilerplate went

In register-level tProc-v1 code, most of a program is indexing a configuration dict into flat
pulse parameters, roughly ten lines per pulse of `cfg['device']['qubit']['pulses']['pi_ge'][...]`
plus unit conversions. In Strumento that mapping is the factory: the calibration schema is typed,
the factory reads it, and units convert once, at the pack boundary.

| tProc-v1 idiom (per pulse) | Strumento |
|---|---|
| gather frequency, gain and sigma from nested config keys | `dev.qubit.ge_pi()` |
| `freq2reg` / `us2cycles` / `deg2reg` by hand | the compiler owns every unit conversion |
| an integer gain code pasted into registers | the factory converts the code to a v2 fraction (D23) |
| override by mutating the config dict | keyword overrides: `dev.qubit.ge_pi(gain=…, freq=…)` |

## 4. Sweeps and axes

A `Sweep` stands in for a pulse field. It is a range, `start` to `stop`, in the field's
lab-native units, bound to a named loop axis with `on=`. The axis and its number of points are a
`Loop`, declared on the program (or by the experiment). The factory maps the range to hardware
units:

```python
from strumento import Loop, Sweep

amp = Sweep(start=0, stop=16000, on="amp")      # lab-native integer gain codes
p = dev.qubit.ge_pi(gain=amp)
print(p.gain.start, f"{p.gain.stop:.5f}")       # the fraction range the hardware runs

rabi = Seq().play(p).measure()
axes = [Loop("amp", 51)]                        # one add_loop: 51 points on the "amp" axis
```

<!-- snippet: output -->
```text
0.0 0.48831
```

- **Units.** A scalar `gain=` is a v2 fraction; a gain `Sweep` is in the lab's integer codes.
- **Joint sweeps.** Several sweeps bound to the same axis step together, in one loop.
- **Declared, not inferred** (D36). A sweep bound to an axis the program does not declare is a
  `ValueError`; the program never guesses a point count.
- **Safety limits.** A gain sweep whose `stop` exceeds its role's limit raises `GainOutOfRange`
  when the pulse is built: 0.5 of full scale for the qubit drive, 0.6 for the readout probe, 0.4
  for a cavity displacement (the full table is in [getting-started §7](getting-started.md#7-sweeps-and-safety-limits)).
  The qubit-drive, readout and displacement limits can be raised per element, up to a hard cap,
  through `qubit.max_drive_gain`, `readout.max_readout_gain` and `modes.<name>.max_displace_gain`
  ([Raising a limit](getting-started.md#raising-a-limit)).
- **Bands.** A frequency sweep whose endpoints resolve to different generators is rejected when
  the program compiles.

The result's `xpts` come back in your lab units, as the hardware plays them: read back from the
compiled program, not re-derived from your request (D37).

## 5. What will the hardware play? `realized`

`StrumentoProgram.realized(pulse)` reads a pulse's values back from the *compiled* program,
through qick's own `get_pulse_param`, so the numbers are the register-rounded values the board
will play. Assembly runs offline against the device's own soccfg snapshot, so the mock is enough:

```python
from strumento.core.mock import MockQickSocV2
from strumento.core.program import StrumentoProgram

mock = MockQickSocV2(dev.soccfg_snapshot)
pi = dev.qubit.ge_pi()
played = StrumentoProgram(dev, seq=Seq().play(pi).measure(), soc=mock).realized(pi)
print(f"{played.freq_mhz_actual:.8f}")    # on the DDS grid, not exactly the requested 4000
print(f"{played.gain_frac:.8f}")          # rounded to the amplitude resolution
print(f"{played.length_us_actual:.5f}")   # the 4σ window (0.12 µs) on the fabric clock grid
```

<!-- snippet: output -->
```text
4000.00000036
0.25001526
0.12019
```

For a swept field, `realized` returns the per-step array the loop engine writes.

## 6. Building and acquiring a program yourself

`StrumentoProgram` *is* an `AveragerProgramV2`. Build one from a sequence and its declared axes,
then acquire on any soc:

```python
prog = StrumentoProgram(dev, seq=rabi, axes=axes, reps=1000, soc=mock)
result = prog.acquire(mock, progress=False)
xpts, avgi, avgq = result.as_tuple()
print(xpts[:3], xpts[-1])       # the swept gain, in integer codes
print(avgi.shape)               # (reads per shot, sweep points)
```

<!-- snippet: output -->
```text
[  0. 320. 640.] 16000.0
(1, 51)
```

The mock's data is all zeros; it answers "does this assemble and run?". For data worth fitting,
run an experiment on the `SimulatorSoc` (section 9, and [getting-started §4](getting-started.md#4-hello-world)): an experiment arms
the simulator with its response model automatically. A raw program on the simulator must be
armed by hand (`soc.arm(SimContext(...))`), and the error says so if you forget.

## 7. Readout: `DispersiveIQ`

Turn raw IQ into a population: rotate the cloud onto the discrimination axis, threshold each
shot, average.

```python
import numpy as np
from strumento.packs.cqed.readout import DispersiveIQ

disc = DispersiveIQ(threshold=0.0, angle=0.0)          # angle in radians
iq = np.array([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1]])   # per-shot (I, Q)
print(disc.discriminate(iq))     # per shot: |g> = 0, |e> = 1
print(f"{disc.reduce(iq):.3f}")  # excited-state population
print(disc.project(iq))          # the rotated I signal, for fitting averaged IQ directly
```

<!-- snippet: output -->
```text
[1 0 1]
0.667
[ 1.  -1.   0.9]
```

When you pass no `readout=`, a cqed experiment reads with `DispersiveIQ.from_calib(dev.calib.readout)`:
the recorded `readout.threshold` and `readout.angle` (`0.0` when unset) and whichever of the
`g_center_iq` / `e_center_iq` / `f_center_iq` blob centres are recorded. Pass `readout=` to
override it.

## 8. Fitting: `fit_rabi`

The fitters are `lmfit`-backed plain functions that return a typed `FitResult`, with no printing
and no plotting:

```python
from strumento.core.fitters import fit_rabi

x = np.linspace(0, 30000, 121)
rng = np.random.default_rng(0)
y = 0.5 + 0.4 * np.cos(2 * np.pi * x / 20000) + rng.normal(0, 0.01, x.size)  # period 20000
fit = fit_rabi(x, y)
print(fit.success, f'{fit.derived["period"]:.0f}')
print(f'{fit.derived["pi_gain"]:.0f}')       # half the period: the π drive amplitude
print(sorted(fit.params) == sorted(fit.stderr))
```

<!-- snippet: output rtol=0.005 -->
```text
True 20000
10000
True
```

`fit.params` and `fit.stderr` hold the fitted model parameters and their standard errors;
`fit.derived` holds the physical quantities, each with its own `<name>_stderr`.
[fitters.md](fitters.md) lists every fitter.

## 9. The `AmplitudeRabi` experiment

An experiment pairs a sequence with a fit and keeps acquire, analyze and display separate.
`analyze` works on any `MeasurementResult`, so you can develop analysis on synthetic data:

```python
from strumento.core.result import MeasurementResult
from strumento.core.simulator import SimulatorSoc
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi

exp = AmplitudeRabi(dev, gains=Sweep(start=0, stop=16000, on="amp"), points=51)
seq, _ = exp.sequence()                   # inspect what it will run; no board needed
print([type(o).__name__ for o in seq.ops])
print(exp.axes())

# analyze a synthetic result
x = np.linspace(0, 16000, 51)
fake = MeasurementResult(xpts=x, avgi=0.5 - 0.5 * np.cos(np.pi * x / 9000), avgq=np.zeros_like(x))
print(f'{exp.analyze(fake).derived["pi_gain"]:.0f}')

# or the whole loop in one call, on the simulator, the mock or a board
outcome = exp.go(SimulatorSoc.from_device(dev, seed=1), reps=2000, progress=False)
print(outcome.fit.success)
```

<!-- snippet: output -->
```text
['Pulse', 'Pulse', 'Measure']
(Loop(name='amp', points=51),)
9000
True
```

The sequence is the qubit pulse, the readout probe and the measurement. `go` returns an
`ExperimentOutcome` with `.result`, `.fit` and, with `display=True`, `.figure`.

## 10. Authoring a device in code: `DeviceBuilder`

`DeviceBuilder` produces the same canonical `Device` as loading YAML, which suits programmatic or
parametric topologies. Calibration values stay data, passed to `build(calib=…)`:

```python
from strumento.core.builder import DeviceBuilder
from strumento.core.wiring import Generator

calib = {
    "qubit": {"f_ge": 4200.0, "f_ef": 4050.0, "pulses": {"pi_ge": {"gain": 9000, "sigma": 0.03}}},
    "readout": {"frequency": 700.0, "gain": 500, "readout_length": 2.0, "trig_offset": 0.5,
                "relax_delay": 2000},
}
dev2 = (DeviceBuilder(pack="cqed")
        .add("Transmon", "q")
        .add("ReadoutResonator", "ro")
        .wire("q", drive=Generator(dac_ch=0))       # the keyword ('drive') is the role
        .readout(dac_ch=5, adc_chs=[0])
        .build(calib=calib))

print(sorted(dev2.elements), dev2.q.ge_pi().freq_mhz)
```

<!-- snippet: output -->
```text
['q', 'ro'] 4200.0
```

Units match the YAML: frequencies in MHz, times (including the readout `trig_offset`) in µs,
gains in the lab's integer codes.

## 11. Couplings, forms and patterns

A physical coupler with its own DAC is an *element* you `add`. An *interaction* is a typed
**edge**, a `Coupling` with fields `a`, `b`, `form` and `mediated_by`. Patterns are plain
functions that emit edges onto a builder; here, a four-transmon ring:

```python
from strumento.core.coupling import ring_couple
from strumento.packs.cqed.couplings import ZZ

calib = {
    "qubit": {"f_ge": 5000.0, "f_ef": 4800.0, "pulses": {}},
    "readout": {"frequency": 700.0, "gain": 500, "readout_length": 2.0, "trig_offset": 0.5,
                "relax_delay": 2000},
    "couplings": {"q0-q1": 0.3, "q1-q2": 0.3, "q2-q3": 0.3, "q3-q0": 0.3},  # strengths, MHz
}
rb = DeviceBuilder(pack="cqed")
for i in range(4):
    rb.add("Transmon", f"q{i}").wire(f"q{i}", drive=Generator(dac_ch=i))
rb.add("ReadoutResonator", "ro").readout(dac_ch=5, adc_chs=[0])
ring_couple(rb, [f"q{i}" for i in range(4)], form=ZZ)
ring = rb.build(calib=calib)

for c in ring.couplings:
    print(c.a, c.b, c.form, c.symmetry, c.mediated_by)
```

<!-- snippet: output -->
```text
q0 q1 ZZ symmetric None
q1 q2 ZZ symmetric None
q2 q3 ZZ symmetric None
q3 q0 ZZ symmetric None
```

The built device stores each form by its registered name (`c.form` is the string `'ZZ'`);
`c.symmetry` and `c.operator(...)` resolve it to the form class. The strengths are calibration
data, keyed `"a-b"`. cqed registers five forms, each with a symmetry tag and an operator template:
`Dispersive`, `ZZ`, `Exchange`, `BeamSplitter` and `JaynesCummings`. A single edge, optionally
mediated by a coupler element, is one call on the builder:

```python
from strumento.packs.cqed.couplings import BeamSplitter

rb.coupling("q0", "q2", form=BeamSplitter, mediated_by="ro")   # mediated_by names the mediating element
```

## 12. Hamiltonian export: `to_operator_model`

The topology, couplings and calibration together are a Hamiltonian specification. Export it for a
solver (the mirror of `Pulse.from_solution`, which imports one):

```python
model = ring.to_operator_model(frame="rotating", cutoffs={f"q{i}": 2 for i in range(4)})
print(model.H0.shape, np.allclose(model.H0, model.H0.conj().T))   # 4 qubits × 2 levels, Hermitian
print(sorted(model.drives), model.drive_map["q0"])               # one drive operator per driven line

# include= extracts a subgraph, so the Hilbert space stays tractable
sub = ring.to_operator_model(frame="rotating", cutoffs={"q0": 3, "q1": 3}, include=["q0", "q1"])
print(sub.H0.shape, type(sub.to_dict()).__name__)                # to_dict(): the JSON wire form

# shaped for Piccolo (an upstream bridge; core stays Piccolo-free)
from strumento.interop.piccolo import to_piccolo_system

spec = to_piccolo_system(sub)
print(spec.drive_labels, spec.H_drift.shape)
```

<!-- snippet: output -->
```text
(16, 16) True
['q0', 'q1', 'q2', 'q3'] q0:x
(9, 9) dict
['q0', 'q1'] (9, 9)
```

`frame="lab"` keeps each element's bare $\omega \hat n$ term; `frame="rotating"` drops it, leaving
the anharmonicity and the couplings.

## 13. Calibration and gain units

Calibration is a typed Pydantic schema. The lab-native integer gain codes convert to v2
fractions at the pack boundary, the one place a v1-to-v2 gain-units mistake could happen:

```python
from strumento.packs.cqed.calibration import gain_int_to_frac

print(f"{gain_int_to_frac(8192, 32766):.8f}")   # code 8192 at maxv 32766
print(dev.calib.qubit.pulses["pi_ge"])
print(dev.calib.readout.frequency, dev.calib.readout.trig_offset)   # MHz, µs
```

<!-- snippet: output -->
```text
0.25001526
gain=8192.0 sigma=0.03 length=0.0 type='gauss'
7000.0 0.5
```

### Gain units

A factory's `gain=` means one of three things, by type: `None` takes the calibrated DAC integer
code and divides it by the generator's full-scale `maxv`; a number is a fraction of full scale,
used as is; a `Sweep` is in DAC integer codes, divided by `maxv` at each endpoint. The pulse
always carries the fraction. The rule is the same for every cqed ops factory:

```python
from functools import partial

from strumento import Sweep

codes = Sweep(start=0, stop=10000, on="amp")
for name, make, code in [
    ("ge_pi", dev.qubit.ge_pi, dev.calib.qubit.pulses["pi_ge"].gain),
    ("probe", dev.readout.probe, dev.calib.readout.gain),
    ("displace", dev.cavity.displace, dev.calib.modes["cavity"].displace.gain),
    ("swap", partial(dev.coupler.swap, "cavity-S1"), dev.calib.swaps["cavity-S1"].gain),
    ("f0g1", dev.sideband.f0g1, dev.calib.multiphoton["f0-g1"].pi[0].gain),
]:
    swept = make(gain=codes).gain
    print(f"{name:<8} None: {code:.0f} -> {make().gain:.6f} | 0.1 -> {make(gain=0.1).gain}"
          f" | Sweep 0..10000 -> {swept.start:.6f}..{swept.stop:.6f}")
```

<!-- snippet: output -->
```text
ge_pi    None: 8192 -> 0.250015 | 0.1 -> 0.1 | Sweep 0..10000 -> 0.000000..0.305194
probe    None: 1000 -> 0.030519 | 0.1 -> 0.1 | Sweep 0..10000 -> 0.000000..0.305194
displace None: 6000 -> 0.183117 | 0.1 -> 0.1 | Sweep 0..10000 -> 0.000000..0.305194
swap     None: 9000 -> 0.274675 | 0.1 -> 0.1 | Sweep 0..10000 -> 0.000000..0.305194
f0g1     None: 12000 -> 0.366233 | 0.1 -> 0.1 | Sweep 0..10000 -> 0.000000..0.305194
```

Every result then faces its role's gain limit, and so does a pulse override: a `TransmonOps`
built with pulse `overrides=` plays the override's own gain (a fraction) for `gain=None`, and an
override above the drive limit raises `GainOutOfRange` like any other pulse. The full rule,
including its one variation (`displace_alpha` takes no `gain=`), is in the docstring of
`strumento.packs.cqed.library`; the calibration's units are listed in
[packs-and-devices.md](packs-and-devices.md).

## 14. Experiments by category

Every cqed experiment follows one pattern: construct it with a device and optional overrides,
then call `.go(soc)` for the full loop, or `.sequence()` and `.analyze()` separately. The pack's
`EXPERIMENTS` registry maps a stable name to each class; the MCP session's `run_experiment`
resolves names through it, and `len(EXPERIMENTS)` is the current count.

```python
from strumento.packs.cqed import EXPERIMENTS

print(EXPERIMENTS["amplitude_rabi"].__name__, EXPERIMENTS["single_shot_readout"].__name__)
```

<!-- snippet: output -->
```text
AmplitudeRabi SingleShotReadout
```

The blocks below construct experiments on the demo device; each comment names what the fit
reports in `outcome.fit.derived`. Sweep ranges respect the safety limits from section 4.

**Which mode or swap a bosonic experiment acts on.** No experiment hardcodes an element name.
Leave `mode_name=` (or `mode=` for `ParityFringe` and `parity`) unset and the experiment uses the
device's cavity: the one `BosonicMode` declared with `role: cavity`, or the device's only
bosonic mode. Leave `pair=` unset and a swap experiment uses the only calibrated coupler swap.
When the device has several candidates and none is singled out, the experiment refuses and names
them, so you choose:

```python
from strumento.packs.cqed.experiments.cavity_t1 import CavityT1
from strumento.packs.cqed.experiments.storage_t1 import StorageT1
from strumento.packs.cqed.experiments.swap_rabi import SwapRabi

print(CavityT1(dev).mode_name)                    # the mode declared with role: cavity
print(StorageT1(dev, mode_name="S2").pair)        # the one swap that reaches S2
try:
    SwapRabi(dev)                                  # two swaps are calibrated on this device
except ValueError as err:
    print(err)
```

<!-- snippet: output -->
```text
cavity
cavity-S2
SwapRabi: cannot choose a coupler swap — calibrated: ['cavity-S1', 'cavity-S2']; pass pair= to choose one
```

The f0–g1 experiments (`SidebandRabi`, `SidebandSpectroscopy`, `MultiphotonLadder` and Fock
preparation) read `calib.multiphoton["f0-g1"]` by that name: they prepare |f,0⟩ and drive the
|f,0⟩↔|g,1⟩ transition, so the key names the physics, not a device's choice.

### Experiments that need per-shot data

A few experiments are built on shot statistics, not on averages. Each declares
`needs_shots = True`, and `list_experiments` reports the flag for each experiment:

```python
from strumento.packs.cqed import EXPERIMENTS

print(sorted(name for name, cls in EXPERIMENTS.items() if cls.needs_shots))
```

<!-- snippet: output -->
```text
['active_reset', 'readout_state', 'single_shot_readout', 'validation_check']
```

Per-shot data reaches a result only when the soc provides it. `SimulatorSoc` does. The mock does
not, and a board soc does not yet either, because the job-server wire form carries averaged IQ only.
On a result without `shots`, these experiments refuse: `fit.success` is `False` and
`fit.derived["refused"]` gives the reason ("needs per-shot data; …"). They never report a shot
quantity computed from averages. Two of them keep the values that averages do support, and list
everything they withheld in `fit.derived["refused_parts"]`:

| Experiment | Kept without shots | Withheld |
|---|---|---|
| `active_reset` | nothing | the populations and the per-round corrections |
| `single_shot_readout` | nothing | everything (its fidelity would be an averaged proxy) |
| `readout_state` | `angle` (from the averaged blob centres) | `threshold`, `separation`, `assignment_fidelity`, `readout_contrast`, `confusion_matrix`, `Ig`, `Ie` |
| `validation_check` | the averaged Ramsey (`detuning`, `T2`, …) | `assignment_fidelity`, `readout_separation`, `readout_threshold` |

The refusal depends only on the result. If your own runner reads the per-shot IQ from the board
and sets `result.shots`, `exp.analyze(result)` analyses it as usual. On the bring-up graph, a
refused `readout_state` or `validation_check` fails its `fit` gate, so a run on a soc without
per-shot data cannot bank a threshold or pass validation.

### Spectroscopy

```python
from strumento.packs.cqed.experiments.cavity_spectroscopy import CavitySpectroscopy
from strumento.packs.cqed.experiments.ef_spectroscopy import EFSpectroscopy
from strumento.packs.cqed.experiments.qubit_spectroscopy import QubitSpectroscopy

# qubit g–e: two-tone sweep around f_ge, Lorentzian fit -> "f0"
exp = QubitSpectroscopy(dev, points=201)

# e–f: a ge π pulse first, then a sweep around f_ef -> "f0"
exp = EFSpectroscopy(dev, points=201)

# a cavity mode: sweep around the mode's calibrated frequency -> "f0"
exp = CavitySpectroscopy(dev, points=401)             # the cavity; mode_name="S1" for a storage mode

# all three accept probe_gain= (a fraction) to override the default 0.03
exp = QubitSpectroscopy(dev, probe_gain=0.05)
```

### Cavity characterization

```python
from strumento.packs.cqed.experiments.cavity_t1 import CavityT1
from strumento.packs.cqed.experiments.displacement_calibration import DisplacementCalibration
from strumento.packs.cqed.experiments.revival_time_calibration import RevivalTimeCalibration

# photon lifetime: displace, wait t, parity readout, exponential fit -> "T1"
exp = CavityT1(dev, delays=Sweep(start=0, stop=5000, on="t"), mode_name="cavity")

# parity fringe against displacement gain -> "alpha_per_gain"
# (the displacement limit is 0.4 of full scale, about 13106 in codes)
exp = DisplacementCalibration(dev, gains=Sweep(start=0, stop=12000, on="gain"))

# damped χ oscillation -> "revival_time"
exp = RevivalTimeCalibration(dev, delays=Sweep(start=0.5, stop=5.0, on="t"))
```

### Sideband and swap operations

```python
from strumento.packs.cqed.experiments.sideband_rabi import SidebandRabi
from strumento.packs.cqed.experiments.sideband_spectroscopy import SidebandSpectroscopy
from strumento.packs.cqed.experiments.swap_rabi import SwapRabi
from strumento.packs.cqed.experiments.swap_spectroscopy import SwapSpectroscopy

# the f0–g1 transition: frequency sweep, Lorentzian fit -> "f0"
exp = SidebandSpectroscopy(dev, freqs=Sweep(start=490, stop=510, on="freq"))

# sideband Rabi: gain sweep, fit_rabi -> "pi_gain" (the half-period, in gain codes)
exp = SidebandRabi(dev, gains=Sweep(start=0, stop=15000, on="amp"))

# the beamsplitter swap frequency for a mode pair -> "f0"
exp = SwapSpectroscopy(dev, pair="cavity-S1")

# swap Rabi: pulse duration sweep, in µs -> "pi_gain" (here the π swap duration, in µs)
exp = SwapRabi(dev, pair="cavity-S1", durations=Sweep(start=0, stop=2, on="t"))
```

### Benchmarking and DRAG

```python
from strumento.packs.cqed import (
    DRAGCalibration,
    DRAGPairCalibration,
    TransmonAllXY,
    TransmonRB,
)

# Clifford RB: one read per depth, exact inversion through the Cayley table
# -> "fidelity", "error_per_clifford"
exp = TransmonRB(dev, depth_start=1, depth_stop=100, points=10, seed=42)

# AllXY: 21 gate pairs in one program (21 reads, no loop)
exp = TransmonAllXY(dev)

# DRAG, preferred: the offset-cancelling pair (see below) -> "alpha_opt" (alias "drag_alpha")
exp = DRAGPairCalibration(dev, alpha_start=-0.25, alpha_stop=0.25, points=31)

# DRAG, simpler: a single X/2·Y/2 equator crossing -> "alpha_opt"
exp = DRAGCalibration(dev, alpha_start=-0.25, alpha_stop=0.25, points=41)
```

The DRAG experiments in more detail:

- **What they play.** Each block plays exactly the factory's `ge_hpi` / `ge_pi` at
  `drag_alpha = alpha` (one shared builder, a 4σ window at the drive DAC rate), and every block
  starts from a passive reset (the readout's `relax_delay`). A measurement is not a reset: without
  the delay the blocks chain and lock at $P_e = 0.5$. `alpha` is the Q/I peak ratio, with
  $|\alpha| \le 2$; $|\alpha| > 1$ is played through gain headroom (the envelope scaled by
  $1/|\alpha|$, the gain by $|\alpha|$, validated against the drive limit).
- **The pair.** Per `alpha`, `DRAGPairCalibration` plays X(π/2)·Y(π) and then Y(π/2)·X(π). The
  DRAG error tilts them in opposite directions, so $P_A - P_B$ crosses zero at the optimum, and
  readout offset and scale, thermal population, $T_1$ and π/2-amplitude error cancel. Detuning is
  only reduced (about sixfold), so tune `f_ge` first. Its fit also reports `equator_offset` (the
  common-mode $P_e$ offset that cancels), the slope and the fit window, each with a stderr. It
  performs `2 * points` reads: keep `2 * points * reps` under the ADC averaging buffer (8192 on a
  ZCU216).
- **The single crossing.** `DRAGCalibration` is simpler, but any common $P_e$ offset at the
  equator moves its crossing (an offset of +0.03 shifts it by about −0.044 for σ = 35 ns).
- **Refusals.** Both analyzers fit a line in a window around the crossing nearest `alpha = 0`
  (iterated), never "the first sign change", and refuse (`success=False`, no `alpha_opt`) when the
  window holds fewer than 5 points, the slope's $t = |m| / \sigma_m$ is below 3, the crossing lies
  outside the fitted points, or the readout has no calibrated g/e blob centres
  (`g_center_iq` / `e_center_iq`, needed to convert to population).
- **Using the result.** Once `calib.qubit.drag_alpha` is set, `dev.qubit.ge_pi()` and `ge_hpi()`
  return that DRAG pulse (unset, the plain Gaussian). The factory refuses $|\alpha| > 1$ in the
  calibration, since it would ship a Q-clipped pulse that differs from the one measured. DRAG is
  g–e only for now.
- **Verifying it.** Run AllXY with DRAG on and off: the pairs (x,Y)−(y,X) and (X,y)−(Y,x) are the
  DRAG pair at one `alpha`, so with DRAG off they read about `slope * (0 - alpha_opt)`, and about 0
  with it on.

### Readout optimization

```python
from strumento.packs.cqed.experiments.power_sweep import PowerSweep
from strumento.packs.cqed.experiments.readout_optimization import ReadoutOptimization
from strumento.packs.cqed.experiments.single_shot_readout import SingleShotReadout

# single shot: ground and excited reads -> "threshold", "angle", "fidelity"
exp = SingleShotReadout(dev)

# readout gain sweep -> "optimal_gain", the power with the best SNR
# (the readout limit is 0.6 of full scale, about 19660 in codes)
exp = PowerSweep(dev, gains=Sweep(start=0, stop=19000, on="gain"))

# integration time sweep, in µs -> "optimal_length"
exp = ReadoutOptimization(dev, lengths=Sweep(start=0.1, stop=10, on="length"))
```

### The readout probe factory

Readout-probing experiments build their probe with `dev.readout.probe()` rather than a raw
`Pulse(...)`. The factory resolves the line, converts the gain (D23) and validates it against
`MAX_READOUT_GAIN`, or against `calib.readout.max_readout_gain` when that is set (that per-element
limit is itself capped at `READOUT_GAIN_CAP`):

```python
probe = dev.readout.probe(freq=Sweep(start=6995, stop=7005, on="f"), gain=0.05, length=2.0)
print(probe.line, probe.gain, probe.envelope)
```

<!-- snippet: output -->
```text
LineRef(line='readout', role='probe') 0.05 Const(length_us=2.0)
```

### Host-batched diagnostics

Three modules in `strumento/packs/cqed/experiments/` are **not** in `EXPERIMENTS`, on purpose.
Each diagnostic needs several *independent* programs (one per ladder block, Wigner ring point
or coherent amplitude), with the device relaxed before each one. A registered experiment is one
program, and stacking the pieces into one program does not work: qick's relax fires once per
shot, after the whole body, so each piece would start from the previous piece's end state. The
modules therefore hold only the pure halves, the per-program sequence builders and the
analyzers; your runner builds one program per piece, acquires each, and feeds the results to
the analyzer.

| Module | What it measures | Sequence builders | Analyzer |
|---|---|---|---|
| `ladder_population` | where the Fock-state prep ladder (`ge_pi · ef_pi · f0g1`, repeated) strands population, stage by stage | `iter_blocks`, `block_sequence`, `round_trip_sequence` | `analyze_ladder_populations` |
| `self_kerr_ramsey` | a cavity mode's signed self-Kerr `K` (and drive detuning), shared across amplitude rows | `row_sequence` | `analyze_self_kerr_ramsey` (`expected_parity` is its model) |
| `kerr_cat_witness` | when free Kerr evolution of a coherent state gives the most phase-space structure, from a ring of Wigner samples | `ring_points`, `witness_sequence` | `analyze_witness`, `analyze_interleaved_witness` |

The ladder diagnostic, run block by block on the mock. Each `(stage, variant)` block is its own
program with its own relax (`final_delay`), long only once a photon is in the cavity:

```python
from strumento.packs.cqed.experiments import ladder_population as ladder

reads = {}
for stage, variant in ladder.iter_blocks(fock_n=1):
    seq, axes = ladder.block_sequence(dev, stage, variant, fock_n=1)
    relax_us = 10_000.0 if ladder.photons_after(stage) else dev.calib.readout.relax_delay
    prog = StrumentoProgram(dev, seq=seq, axes=axes, reps=100, final_delay=relax_us, soc=mock)
    reads[stage, variant] = prog.acquire(mock, progress=False)
print(len(reads), ladder.n_stages(1), ladder.LADDER_VARIANTS)
```

<!-- snippet: output -->
```text
12 4 ('none', 'ge', 'ef')
```

The analyzer takes direct three-level populations `(P_g, P_e, P_f)` per block, in the same
order. It refuses anything else: a two-level g/e discriminator has no honest `P_f`, so convert
each block's IQ with a g/e/f blob calibration first. Here, made-up populations whose last stage
leaves 26% in `f`:

```python
populations = {0: (0.97, 0.02, 0.01), 1: (0.02, 0.96, 0.02),
               2: (0.02, 0.04, 0.94), 3: (0.70, 0.04, 0.26)}
blocks = [populations[stage] for stage, _ in ladder.iter_blocks(fock_n=1)]
derived = ladder.analyze_ladder_populations(blocks, fock_n=1)
print(int(derived["leak_stage"]), round(derived["final_f"], 2))
```

<!-- snippet: output -->
```text
3 0.26
```

`leak_stage` is the first stage whose expected level holds less than `1 - leak_threshold`
(default 0.10); `-1` means the ladder is clean.

The self-Kerr Ramsey diagnostic is one program per amplitude row. Each row displaces, waits on a
declared delay axis, displaces back with a virtual phase advancing at `reference_detuning_mhz`,
then maps parity. Acquire every amplitude at two return-phase offsets (0° and 90°): with one
offset, `+K` and `-K` fit equally well, and the analyzer flags it as `K_alias_ambiguous`.

```python
from strumento.packs.cqed.experiments import self_kerr_ramsey as skr

amplitudes = np.array([0.8, 1.4, 1.4, 2.0])
offsets = np.array([0.0, 0.0, 90.0, 45.0])
delays = np.linspace(0.0, 60.0, 25)
for alpha, offset in zip(amplitudes, offsets):
    seq, _ = skr.row_sequence(dev, alpha=alpha, delays=Sweep(start=0.0, stop=60.0, on="t"),
                              mode_name="cavity", reference_detuning_mhz=0.025,
                              return_phase_offset_deg=offset)
    row = StrumentoProgram(dev, seq=seq, axes=[Loop("t", 25)], reps=100, soc=mock)
    row.acquire(mock, progress=False)  # a real runner keeps each row's parity

# fit synthetic parity rows built from the analyzer's own model, K = 4 kHz
rows = np.stack([
    skr.expected_parity(alpha, delays, k_mhz=0.004, detuning_mhz=-0.008,
                        reference_detuning_mhz=0.025, return_phase_offset_deg=offset, cutoff=16)
    for alpha, offset in zip(amplitudes, offsets)
])
fit = skr.analyze_self_kerr_ramsey(amplitudes, delays, 0.9 * rows,
                                   reference_detuning_mhz=0.025,
                                   return_phase_offsets_deg=offsets, cutoff=16)
print(fit["success"], f"{fit['K_mhz']:.4f}", fit["K_alias_ambiguous"])
```

<!-- snippet: output -->
```text
True 0.0040 False
```

The Kerr-cat witness follows the same shape: `ring_points(radius=…, count=…)` gives the Wigner
coordinates, and `witness_sequence(dev, sample_alpha=…, kerr_wait=…, init_alpha=…,
mode_name=…)` builds one program per coordinate (prepare a coherent state, free-evolve, sample
displaced parity). `analyze_witness` takes one control row per coordinate plus a sweep over waits;
`analyze_interleaved_witness` takes a matched zero-wait control acquired beside each wait, and
subtracts it coordinate by coordinate to remove the static spatial baseline. Both need the readout (`dev.readout`) and the mode's calibration (for
`alpha_scale`, `parity_sign` and `pe_midpoint`), and report `best_wait_us`. Their contrast is a
witness, not a fidelity: a null result is still a result, and the runner should keep it.

## 15. Calibration write-back

An experiment never changes the device. To close the loop, write a fit's derived values back
with `record_calibration`: it snapshots a new immutable calibration version, advances the store's
ref and reloads the session's device, so the next experiment builds pulses from the new belief.
You name each field explicitly; Strumento never guesses which calibration value a fit result
belongs to, and `outcome=` names the run whose fit the version is labelled with (omit it and the
version is recorded as `manual`). With `LabSession` (on the simulator here):

```python
from strumento.mcp import LabSession

with LabSession() as lab:
    lab.load_device("devices/multimode_demo/device.yaml", store="calibration-store")
    run = lab.run_experiment("single_shot_readout", {"reps": 1000})
    derived = run["fit"]["derived"]
    lab.record_calibration({
        "readout.threshold": derived["threshold"],
        "readout.angle": derived["angle"],
    }, outcome=run["index"])
    print(lab.device.calib.readout.threshold == derived["threshold"])
    print("readout.threshold" in lab.device_info()["calibration_paths"])
```

<!-- snippet: output -->
```text
True
True
```

`device_info()["calibration_paths"]` lists every writable dotted path, the same grammar
`record_calibration` accepts. Typical write-backs:

| Experiment | Derived value | Calibration path |
|---|---|---|
| `amplitude_rabi` | `pi_gain` | `qubit.pulses.pi_ge.gain` |
| `single_shot_readout` | `threshold`, `angle` | `readout.threshold`, `readout.angle` |
| `drag_pair_calibration` | `drag_alpha` (only when `fit.success`; verify with AllXY first) | `qubit.drag_alpha` |
| `cross_kerr` (`mode_name="cavity"`) | `chi` | `modes.cavity.chi` |

Without the reload a session would keep measuring with the value it had just replaced. Outside a
session, `strumento.record_calibration` writes to a calibration store directly; see
[calibration-store.md](calibration-store.md).

## 16. See also

- [getting-started.md](getting-started.md): the walkthrough, from install to hardware.
- [packs-and-devices.md](packs-and-devices.md): how the pieces compose, and how to add a pack.
- [timeline.md](timeline.md): exactly when each op plays.
- [fitters.md](fitters.md): every fitter and what it reports.
