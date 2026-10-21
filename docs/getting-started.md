# Getting started with Strumento

This guide takes you from a fresh clone to a fitted calibration value, then shows you how to
describe your own device and take it to hardware. Read it top to bottom: each section builds on
the one before. Every Python block runs as written from the repository root, and the docs test
(`tests/test_docs.py`) checks the printed output shown under it. The only blocks that are not run
are the ones that need a real board; they are marked as such.

## 1. What you need

- **Python 3.10–3.12**, ideally in a fresh virtual environment.
- **A clone of this repository.** Strumento is installed from source for now, and the examples
  load the demo devices under `devices/`.
- **No board.** Everything up to [section 12](#12-going-to-hardware) runs on a laptop.

Strumento's dependencies are split into extras, so you only install what you use:

| Extra | Install | What it unlocks |
|---|---|---|
| *(none)* | `pip install -e .` | The device model, pulses, sweeps, fitters, the calibration store and the `strumento` CLI. No `qick`. |
| `qick` | `pip install -e ".[qick]"` | Everything that builds a real tProc-v2 program: the compiler, the mock board and the simulator. **This guide needs it.** Pins `qick>=0.2.367,<0.3`. |
| `mcp` | `pip install -e ".[qick,mcp]"` | The `strumento-mcp` server, which lets an AI agent drive a lab session. |
| `plot` | `pip install -e ".[qick,plot]"` | `matplotlib`, for the figures `go(..., display=True)` returns (see [section 10](#10-plots)). |
| `test` | `pip install -e ".[test,qick]"` | `pytest`, to run the test suite. |
| `dev` | `pip install -e ".[dev]"` | `pytest`, `ruff` and `mypy`, for contributors. |

Plotting is optional: `matplotlib` comes with the `plot` extra. Without it, everything works
except plotting (see [section 10](#10-plots)).

## 2. Install

```bash
git clone https://github.com/harmoniqs/strumento.git
cd strumento
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[qick]"
```

`qick` installs on an ordinary x86 machine; the board-only parts it depends on are skipped off
the board. Check that the install worked:

```bash
strumento packs
```

```text
installed packs: cqed, spin, tweezer
```

A **pack** is the physics kit for one kind of hardware. `cqed` covers superconducting circuits,
`spin` covers quantum dots, and `tweezer` is a skeleton for neutral atoms. This guide uses `cqed`.

The first time a program is built, you will see this notice on stderr:

```text
QICK library version mismatch: 0.2.367 remote (the board), 0.2.431 local (the PC)
                        This may cause errors, usually KeyError in QickConfig initialization.
                        If this happens, you must bring your versions in sync.
```

It is harmless here. The demo devices carry a board description recorded with qick 0.2.367, and
qick compares that with the version you installed (the second number is yours). On a real board,
keep the two in sync.

## 3. Three ways to run

Strumento builds the same real tProc-v2 program whichever way you run it. What changes is where
the data comes from:

| | Mock | Simulator | Hardware |
|---|---|---|---|
| Class | `MockQickSocV2` | `SimulatorSoc` | `QickSoc` on the board, or a remote board (section 12) |
| Program assembly | real | real | real |
| Data returned | all zeros | synthetic, from a ground truth you choose | measured |
| Needs a board | no | no | yes |
| Use it to | check that a program builds | develop and test analysis, rehearse a calibration | measure |

The **mock** answers "does this program assemble?". The **simulator** goes one step further: you
give it a *ground truth* (for example, the true π gain), and it returns per-shot IQ data that a
qubit with that truth would produce. That is the only offline path that gives you data worth
fitting, so it is what the rest of this guide uses.

The simulator's physics models are simple closed forms, one per experiment, not a Hamiltonian
simulation. They are there to exercise everything between your experiment and your fit, not to
predict a real device.

## 4. Hello world

Run this from the repository root:

```python
from strumento import Device
from strumento.core.simulator import SimulatorSoc
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi

dev = Device.load("devices/multimode_demo/device.yaml")
print(dev.calib.qubit.pulses["pi_ge"].gain)   # what the device believes the π gain is

# A simulated board whose true π gain (9350) disagrees with that belief
soc = SimulatorSoc.from_device(dev, truth={"pi_gain": 9350}, seed=3)

outcome = AmplitudeRabi(dev).go(soc, reps=2000, progress=False)
fit = outcome.fit
print(fit.success, f'{fit.derived["pi_gain"]:.0f} ± {fit.derived["pi_gain_stderr"]:.0f}')
```

<!-- snippet: output -->
```text
8192.0
True 9333 ± 71
```

The device believed its π gain was 8192. The amplitude-Rabi experiment measured 9333 ± 71,
within one standard error of the true 9350. That is a calibration: the experiment found the
value the device should have had.

## 5. What just happened

**The device.** `devices/multimode_demo/device.yaml` describes one (fictional) fridge: a
transmon, a readout resonator, three bosonic modes and a coupler, how each is wired to the
board's DACs and ADCs, and their calibration values. Loading it gives you a `Device` whose parts
are named the way the YAML names them:

```python
print(dev)
print(dev.calib.qubit.f_ge)   # the qubit's g–e frequency, in MHz
```

<!-- snippet: output -->
```text
Device(pack='strumento.packs.cqed', elements=['S1', 'S2', 'cavity', 'coupler', 'qubit', 'readout'])
4000.0
```

`dev.qubit` exists because the YAML has an element called `qubit`; a device with an element
called `q0` would give you `dev.q0`. If you move a cable, you edit the wiring in the YAML, not
your experiments.

**Calibration is belief.** The calibration values are what the lab currently *believes*. Pulses
are computed from them:

```python
pi = dev.qubit.ge_pi()
print(pi.freq_mhz, f"{pi.gain:.5f}", pi.envelope)
```

<!-- snippet: output -->
```text
4000.0 0.25002 Gaussian(sigma_us=0.03)
```

The calibration stores the gain as the lab's integer code, 8192. The pulse carries it as a
fraction of full scale, 8192 / 32766, because that is what tProc-v2 uses. You keep thinking in
your lab's units; the conversion happens once, inside the pack.

**The experiment.** `AmplitudeRabi(dev)` sweeps the π pulse's gain from 0 to 16000 in 51 steps,
measures after each, and fits a decaying cosine. `go(soc)` runs it on whatever board you pass
and returns an `ExperimentOutcome`:

```python
print(type(outcome).__name__)
print(outcome.result.xpts[:4])     # the swept gain, in the lab's integer codes
print(outcome.result.avgi.shape)   # (reads per shot, sweep points)
print(sorted(fit.derived))
```

<!-- snippet: output -->
```text
ExperimentOutcome
[  0. 320. 640. 960.]
(1, 51)
['period', 'period_stderr', 'pi_gain', 'pi_gain_stderr']
```

- `outcome.result` is a `MeasurementResult`: the sweep axis (`xpts`) and the averaged I and Q
  (`avgi`, `avgq`), plus the per-shot data when the board provides it.
- `outcome.fit` is a `FitResult`: `params` and `stderr` are the fitted model parameters and their
  standard errors, and `derived` holds the physical quantities computed from them, each with its
  own `<name>_stderr`. `fit.success` says whether the fit converged.
- `outcome.figure` is a plot, if you asked for one (section 10).

`reps=2000` sets the number of repetitions averaged at each sweep point. `progress=False` turns
off qick's progress bar.

**Nothing was written back.** Running an experiment never changes the device. `dev` still
believes 8192; section 9 shows how to record the new value.

## 6. Inspect before you run

On a real board you want to know what a program will do before you play it. Start with the CLI:

```bash
strumento info devices/multimode_demo/device.yaml
```

```text
pack: strumento.packs.cqed
elements: S1:BosonicMode, S2:BosonicMode, cavity:BosonicMode, coupler:Coupler, qubit:Transmon, readout:ReadoutResonator
namespaces: S1, S2, cavity, coupler, qubit, sideband
  qubit.drive     -> DAC 1 (nqz 1)
  qubit.sideband  -> DAC 5 (nqz 1)
  cavity.main      -> DAC 2 (nqz 2)
  storage.main      -> DAC 3 (nqz 2)
  coupler.low       -> DAC 4 (nqz 1)
  coupler.high      -> DAC 4 (nqz 1)
  readout.probe     -> DAC 0 (nqz 2)
  readout       -> DAC 0 + ADC [0]
```

```bash
strumento check devices/multimode_demo/device.yaml
```

```text
PASS: devices/multimode_demo/device.yaml validates against its soccfg
  6 gen(s), 1 readout(s)
```

`check` compares the device's wiring with the board description it carries (the *soccfg
snapshot*, section 11) and exits with status 1 if anything does not match. `strumento compile
<device.yaml>` shows the program calls for a canned Rabi. Every command takes `--json`; see
[cli.md](cli.md).

In Python, an experiment shows you its sequence and its sweep axes without running anything, and
a program built on the mock tells you exactly what the hardware will play:

```python
from strumento.core.mock import MockQickSocV2
from strumento.core.program import StrumentoProgram

exp = AmplitudeRabi(dev)
seq, _ = exp.sequence()
print([type(op).__name__ for op in seq.ops])
print(exp.axes())

mock = MockQickSocV2(dev.soccfg_snapshot)
prog = StrumentoProgram(dev, seq=seq, axes=exp.axes(), soc=mock)
played = prog.realized(seq.ops[0])   # the qubit pulse, read back from the compiled program
print(f"{played.freq_mhz_actual:.6f} MHz, {played.length_us_actual:.4f} µs")
print(played.gain_frac[[0, 1, -1]].round(5))
```

<!-- snippet: output -->
```text
['Pulse', 'Pulse', 'Measure']
(Loop(name='amp', points=51),)
4000.000000 MHz, 0.1202 µs
[0.      0.00977 0.48831]
```

The sequence is the qubit pulse, the readout probe pulse and the measurement. `realized` reports
the values after the hardware's rounding, so the frequency, the pulse length and every step of
the gain sweep are what the board will actually produce. `prog.print_trace()` prints the full
compiled program.

The mock runs the whole experiment too, but its data is all zeros, so the fit means nothing:

```python
zeros = exp.go(mock, progress=False)
print(zeros.result.avgi.max())
```

<!-- snippet: output -->
```text
0.0
```

## 7. Sweeps and safety limits

**Sweeps are declared.** A `Sweep` is a range, `start` to `stop`, that stands in for a pulse
field. It is bound to a named loop axis with `on=`, and the axis, with its number of points, is a
`Loop`. Experiments declare their own axes, so for them you only choose the range and the number
of points:

```python
from strumento import Sweep

finer = AmplitudeRabi(dev, gains=Sweep(start=0, stop=12000, on="amp"), points=41)
print(finer.axes())
```

<!-- snippet: output -->
```text
(Loop(name='amp', points=41),)
```

When you build a program yourself, you declare the axis. Several sweeps bound to the same axis
step together, in one loop. Here the gain and the frequency move in lockstep over 7 points:

```python
from strumento import Loop, Seq

g = Sweep(start=0, stop=16000, on="ax")
f = Sweep(start=3980, stop=4020, on="ax")
joint = Seq().play(dev.qubit.ge_pi(gain=g, freq=f)).measure()
prog = StrumentoProgram(dev, joint, axes=[Loop("ax", 7)], soc=mock)
print(prog.xpts())
```

<!-- snippet: output -->
```text
[    0.  2666.  5332.  7998. 10664. 13330. 15996.]
```

The axis comes back in your lab's units and as the hardware plays it: the steps are 2666 rather
than 2666.67, because the gain register is an integer. Leave out `axes=` and the program refuses
to guess:

```python
try:
    StrumentoProgram(dev, joint, soc=mock)
except ValueError as err:
    print(err)
```

<!-- snippet: output -->
```text
sweep [0.0, 0.4883110541414881] is bound to undeclared axis 'ax': declare it on the program (axes=[Loop('ax', n)]) — loop structure is explicit (D36)
```

**Gain limits.** The `cqed` pack refuses to build a pulse, or a gain sweep, whose amplitude
could heat or damage the device. Each role has a ceiling, as a fraction of the DAC's full scale:

| Role | Limit | On the demo's DACs (full scale 32766) |
|---|---|---|
| Qubit drive (g–e and e–f) | 0.5 | ≈ 16383 |
| Readout probe | 0.6 | ≈ 19660 |
| Cavity displacement | 0.4 | ≈ 13106 |
| Coupler pump | 0.5 | ≈ 16383 |
| Sideband drive | 0.5 | ≈ 16383 |

That is why `AmplitudeRabi`'s default sweep stops at 16000. Go past a limit and you get a
`GainOutOfRange` error before anything is compiled:

```python
from strumento.packs.cqed.safety import GainOutOfRange

for gain in (0.6, Sweep(start=0, stop=20000, on="amp")):
    try:
        dev.qubit.ge_pi(gain=gain)
    except GainOutOfRange as err:
        print(err)
```

<!-- snippet: output -->
```text
|gain|=0.6000 (role=qubit_drive, pulse=pi_ge) exceeds safe limit 0.5 for cqed; refusing to build pulse that could damage the device. If this element needs more, raise qubit.max_drive_gain in the device's calibration (up to 0.8; see docs/getting-started.md, section 7). Past the cap, construct the Pulse directly (Tenet 0).
gain sweep stop=20000 (fraction=0.6104) on qubit_drive exceeds safe limit 0.5; refusing to build sweep. If this element needs more, raise qubit.max_drive_gain in the device's calibration (up to 0.8; see docs/getting-started.md, section 7). Past the cap, construct the Pulse directly (Tenet 0).
```

### Raising a limit

The limits above are **defaults**, set for a typical wiring. An element behind more attenuation or
a weaker amplifier needs a larger DAC fraction for the same drive at the device, so three roles
let you set the limit per element, in its calibration. Each field is bounded by a hard cap that
stays below full scale:

| Role | Calibration field | Default | Hard cap |
|---|---|---|---|
| Qubit drive (every qubit-drive pulse and probe) | `qubit.max_drive_gain` | 0.5 | 0.8 |
| Readout probe | `readout.max_readout_gain` | 0.6 | 0.8 |
| Cavity displacement and cavity-spectroscopy probe | `modes.<name>.max_displace_gain` | 0.4 | 0.6 |
| Coupler pump | none yet | 0.5 | — |
| Sideband drive | none yet | 0.5 | — |

To let this device's qubit drive reach 0.6 of full scale instead of 0.5, set the field in the
device file:

```yaml
calibration:
  qubit:
    max_drive_gain: 0.6     # fraction of full scale; replaces the 0.5 default, at most 0.8
```

Or apply it in Python, to try the value before you commit it to the file. `apply_updates` takes
the same dotted paths the calibration store uses:

```python
from strumento import apply_updates

config, _ = apply_updates(dev.to_config(), {"qubit.max_drive_gain": 0.6})
strong = Device.from_config(config)
print(strong.qubit.ge_pi(gain=0.6).gain)

try:
    dev.qubit.ge_pi(gain=0.6)          # the original device keeps the 0.5 default
except GainOutOfRange:
    print("still refused on dev")
```

<!-- snippet: output -->
```text
0.6
still refused on dev
```

Everything built from `strong` faces the new limit: the pulse factories, gain sweeps, and the
experiments that play the qubit drive (`QubitSpectroscopy`, `EFSpectroscopy`, `NumberSplitting`,
`TlsScan`, `GainLadderSpectroscopy`, the DRAG calibrations and the rest). A raised limit does not
change any experiment's default sweep. `AmplitudeRabi` still stops at 16000 unless you pass a
wider `gains=` range.

A value above the cap does not load. The device's calibration is refused, naming the field:

```python
from pydantic import ValidationError

try:
    Device.from_config(apply_updates(dev.to_config(), {"qubit.max_drive_gain": 0.9})[0])
except ValidationError as err:
    print("\n".join(str(err).splitlines()[1:3]))
```

<!-- snippet: output -->
```text
qubit.max_drive_gain
  Input should be less than or equal to 0.8 [type=less_than_equal, input_value=0.9, input_type=float]
```

To keep the new limit as a version of the device's calibration, write it with the library's
`record_calibration` (section 9 and [calibration-store.md](calibration-store.md)), with a
`source=` saying who raised it and why. These fields are **set by a person**: the MCP session's
`record_calibration` refuses any write that touches one, so an agent running the loop cannot raise
its own limit.

Past a cap, or past the coupler or sideband default, the only way is to build the `Pulse`
yourself, outside the factories (Tenet 0 in [architecture.md](architecture.md)). Nothing then
checks its amplitude, and you take responsibility for it.

### Gain units

Every cqed pulse factory that takes `gain=` (`dev.qubit.ge_pi`, `dev.readout.probe`,
`dev.cavity.displace`, `dev.coupler.swap`, `dev.sideband.f0g1` and the rest) reads it one of
three ways, and the pulse it returns always carries a fraction of the DAC's full scale:

| You pass | It means | The pulse's gain |
|---|---|---|
| nothing (`gain=None`) | the calibrated gain, a DAC integer code | code / maxv |
| a number, `gain=0.1` | a fraction of full scale | the number, as is |
| a `Sweep` | DAC integer codes | each endpoint / maxv |

`maxv` is the generator's full-scale code, 32766 on the demo devices. So the calibrated π pulse,
an explicit fraction and a sweep in codes come out as:

```python
pi_code = dev.calib.qubit.pulses["pi_ge"].gain                 # a DAC integer code
print(pi_code, f"{dev.qubit.ge_pi().gain:.6f}")                 # code / maxv
print(dev.qubit.ge_pi(gain=0.1).gain)                           # used as is
swept = dev.qubit.ge_pi(gain=Sweep(start=0, stop=16000, on="amp")).gain
print(swept.start, f"{swept.stop:.6f}")                         # 0 / 32766, 16000 / 32766
```

<!-- snippet: output -->
```text
8192.0 0.250015
0.1
0.0 0.488311
```

Passing the wrong form is caught where it can be: a code given as a number (`gain=8192`) is over
every limit, and a `Sweep` endpoint between 0 and 1 looks like a fraction and is refused. The
calibration's own units are listed in [packs-and-devices.md](packs-and-devices.md), and the full
rule, including pulse overrides (checked against the limit like every other pulse) and its one
variation, is in the docstring of `strumento.packs.cqed.library`.

## 8. Your own analysis

`analyze` works on any `MeasurementResult`, wherever it came from. That means you can develop and
test analysis code on synthetic data before you have a board:

```python
import numpy as np
from strumento.core.result import MeasurementResult

x = np.linspace(0, 16000, 51)
rng = np.random.default_rng(2)
signal = 0.5 - 0.5 * np.cos(np.pi * x / 9350) + rng.normal(0, 0.02, x.size)
fake = MeasurementResult(xpts=x, avgi=signal, avgq=np.zeros_like(x))

fit = AmplitudeRabi(dev).analyze(fake)
print(fit.success, f'{fit.derived["pi_gain"]:.0f} ± {fit.derived["pi_gain_stderr"]:.0f}')
```

<!-- snippet: output -->
```text
True 9367 ± 159
```

The fitters are plain functions, so you can also call them directly
(`from strumento.core.fitters import fit_rabi`; `fit_rabi(x, signal)`), or re-analyze a result you
saved earlier. [fitters.md](fitters.md) lists them.

## 9. Closing the loop with the calibration store

The hello world measured a better π gain, but nothing recorded it. The **calibration store** does
that: every calibration you record becomes a new, immutable version, with a log of what changed
and why. `LabSession` puts a device, a board and a store together, and gives you the full loop:
measure, record, and carry on with the new belief.

```python
from strumento.mcp import LabSession

with LabSession() as lab:
    lab.load_device("devices/multimode_demo/device.yaml",
                    store="calibration-store", sim={"pi_gain": 9350})
    run = lab.run_experiment("amplitude_rabi", {"reps": 2000})
    pi, err = run["fit"]["derived"]["pi_gain"], run["fit"]["derived"]["pi_gain_stderr"]
    print(f"measured {pi:.0f} ± {err:.0f}")

    lab.record_calibration({"qubit.pulses.pi_ge.gain": round(pi)}, outcome=run["index"])
    print("belief now", lab.device.calib.qubit.pulses["pi_ge"].gain)
    print("versions", len(lab.calibration_history()["history"]))
```

<!-- snippet: output rtol=0.01 -->
```text
measured 9405 ± 93
belief now 9405.0
versions 2
```

- `load_device(..., store=...)` opens the store (creating it if needed). `sim=` is the
  simulator's ground truth; the session uses the simulator by default, or the mock with
  `soc="mock"`.
- `run_experiment` takes an experiment's name and its parameters, and returns a plain dict.
- `record_calibration` writes the new value as a new version and reloads the session's device,
  so the next experiment plays pulses built from it. You name the field explicitly: Strumento
  never guesses which calibration value a fit result belongs to. `outcome=run["index"]` labels
  the version with the run it came from (`amplitude_rabi fit`, with the fit attached); without it
  the version is recorded as `manual`.
- A simulator or mock session marks a store it creates as a simulation store, and will not write
  into an existing store without that mark (see
  [calibration-store.md](calibration-store.md#simulated-writes)).

Each run's data is kept, too. The session saves every run to a data chest beside the store (here
`calibration-store-data`; pass `load_device(..., chest=<dir>)` to put it elsewhere), and the
reply's `datachest` names the entry, so you can reload the measurement and refit it later:

```python
from strumento.core.datachest import FileDataChest

saved = run["datachest"]
result = FileDataChest(saved["chest"]).load(saved["entry"])["result"]
print(saved["saved"], list(result.avgi.shape) == run["data"]["avgi_shape"])
```

<!-- snippet: output -->
```text
True True
```

The store, not the YAML, is now the device's belief. Open a new session on the same store and it
picks up where you left off:

```python
with LabSession() as lab:
    info = lab.load_device("devices/multimode_demo/device.yaml", store="calibration-store")
    print(info["store"]["seeded"], lab.device.calib.qubit.pulses["pi_ge"].gain)
```

<!-- snippet: output -->
```text
False 9405.0
```

`LabSession` is an ordinary Python class and needs no extra; the `mcp` extra only adds the server
that exposes the same session to an AI agent. To use the store directly, see
[calibration-store.md](calibration-store.md). `strumento store` reads a store from the command
line.

## 10. Plots

Ask `go` for a figure with `display=True`. With `matplotlib` installed you get a matplotlib
figure; without it, `outcome.figure` is `None` and everything else works the same:

```python
outcome = AmplitudeRabi(dev).go(soc, reps=2000, progress=False, display=True)
if outcome.figure is not None:
    outcome.figure.savefig("rabi.png")
```

Install the `plot` extra (`pip install -e ".[qick,plot]"`) if you want plots. The fitters never
plot, and plotting never changes a fit.

## 11. Describe your own device

This section is the short version. For a new board, [board-bringup.md](board-bringup.md) walks the
whole path in order: what the board must run, recording its description, wiring and Nyquist
zones, first calibration values and where they come from, comparing the live board with the
stored description, the day-one signal-chain checks, and a simulator rehearsal.

A device is one YAML file plus a board description. The quickest start is a copy of the
exemplar:

```bash
mkdir devices/my_fridge
cp devices/multimode_demo/device.yaml devices/my_fridge/device.yaml
```

Then edit `devices/my_fridge/device.yaml`:

- **`topology`**: the elements you have, by type and name. Delete what you don't have.
- **`wiring`**: which DAC drives each line, its Nyquist zone, and which ADCs read out.
- **`calibration`**: your current values (frequencies in MHz, times in µs, gains in your lab's
  integer codes).
- **`soccfg_snapshot`**: a path, relative to the YAML, to your board's description.

The board description is the JSON that qick's `dump_cfg()` produces. Record it once, on the
board:

<!-- snippet: skip reason="needs a board: QickSoc only runs on the RFSoC" -->
```python
from pathlib import Path
from qick import QickSoc

soc = QickSoc()
Path("my_board_soccfg.json").write_text(soc.dump_cfg())
```

Copy the file next to your YAML and set `soccfg_snapshot: my_board_soccfg.json`. Strumento needs
a **tProc-v2** board (`qick_processor` firmware). Until you have your own snapshot, the copied
`../_fixtures/soccfg_v2_cqed.json` keeps working.

Now validate the YAML, then check it against the board:

```bash
strumento validate devices/my_fridge/device.yaml
strumento check devices/my_fridge/device.yaml
```

`validate` checks that the file is a well-formed device of its pack. `check` catches wiring the
board cannot satisfy. For example, a qubit drive wired to a DAC the board doesn't have:

```text
FAIL: 1 issue(s) found
  ERROR: line 'qubit' role 'drive': dac_ch=9 not found in soccfg gens (have [0, 1, 2, 3, 4, 5])
  6 gen(s), 1 readout(s)
```

Finally, simulate it. With no `truth=`, the simulator agrees with your calibration, so the fit
should recover your own values. That confirms the device loads, assembles and fits end to end:

```python
path = "devices/multimode_demo/device.yaml"   # use devices/my_fridge/device.yaml
mine = Device.load(path)
sim = SimulatorSoc.from_device(mine, seed=1)
fit = AmplitudeRabi(mine).go(sim, reps=2000, progress=False).fit
print(fit.success, f'{fit.derived["pi_gain"]:.0f} ± {fit.derived["pi_gain_stderr"]:.0f}')
```

<!-- snippet: output -->
```text
True 8223 ± 45
```

The demo believes 8192, and the fit agrees within one standard error. [packs-and-devices.md](packs-and-devices.md)
explains the device model in depth, including building a device in Python with `DeviceBuilder`,
and `devices/multimode_demo/README.md` walks through the exemplar file.

## 12. Going to hardware

None of the blocks in this section run in the docs test: they need a board. If this is a new
board, work through [board-bringup.md](board-bringup.md) first. The calls below run one
experiment once; [write-a-runner.md](write-a-runner.md) shows the script around them that a run
on hardware needs (comparing the live board with its description, a lock, a preflight, a run
record, and quality checks on the fit).

**Check the signal chain first.** `examples/fallback/` holds plain qick programs that bypass
Strumento entirely: an ADC trace, a DAC-to-ADC loopback, resonator spectroscopy, Rabi, T1 and
Ramsey. Run them in that order on a new board before you trust anything built on top. Their
[README](../examples/fallback/README.md) explains each one.

**On the board.** When your Python session runs on the RFSoC itself, pass a `QickSoc` where you
passed the simulator. Nothing else changes:

<!-- snippet: skip reason="needs a board: QickSoc only runs on the RFSoC" -->
```python
from qick import QickSoc

soc = QickSoc()
dev = Device.load("devices/my_fridge/device.yaml")
outcome = AmplitudeRabi(dev).go(soc, reps=2000)
print(outcome.fit.success, outcome.fit.derived["pi_gain"])
```

**From your PC, over Pyro.** If your lab already serves the board with qick's Pyro4 server, use
`LegacyPyroRunner`. It compiles on your PC against the board's configuration and only acquires
through the proxy:

<!-- snippet: skip reason="needs a board: connects to a running qick Pyro4 server" -->
```python
from qick.pyro import make_proxy
from strumento.core.legacy_pyro_runner import LegacyPyroRunner

soc, soccfg = make_proxy(ns_host="192.168.1.10")   # your nameserver's address
exp = AmplitudeRabi(dev, runner=LegacyPyroRunner(soccfg))
outcome = exp.go(soc, reps=2000)
```

Pyro4 is a fragile transport for this: if its worker thread dies, every later call in the session
fails until you reconnect with a fresh `make_proxy()`. [job-manager.md](job-manager.md) explains
why, and describes the job server, which is the long-term way to reach a remote or shared board.

**Safety still applies.** The gain limits in section 7 hold on hardware too. On a new device, run
`strumento check` and rehearse the experiment on the simulator first. If a line's attenuation
means a calibrated pulse needs more than the default limit, raise that element's limit in the
device's calibration ([section 7](#raising-a-limit)), rather than building pulses by hand.

**Before every session on a board:**

1. **Same qick on both sides.** Programs are compiled against the board's configuration on the
   machine that builds them. A `QICK library version mismatch` notice against a live board means
   the PC and the board run different qick versions: install the board's.
2. **The board is the board you described.** Compare the board's live description with the
   device's stored snapshot (the example run script in [write-a-runner.md](write-a-runner.md)
   does this on every run), then run `strumento check`. On any difference, re-record the snapshot
   and review the calibration that depends on it.
3. **One script at a time.** Two scripts acquiring on one board interleave their programs. Use one
   lock file per board, shared by every script that talks to it.
4. **Re-anchor the readout.** The IQ frame can rotate between sessions. Re-measure the readout
   blob centres first, then check the qubit frequency and the π gain (a Ramsey detuning pair and
   a π error amplification) before any precision step. Calibration drifts between sessions;
   [calibrate-a-transmon.md](calibrate-a-transmon.md) gives the order.
5. **Stay with the board.** An armed run has no automatic timeout or abort yet (#197): watch it,
   and stop it by hand if it stalls.
6. **Keep the record.** Run with a task record and save each run's data, so a surprising result
   can be re-analysed rather than re-measured.

[hardware-ci.md](hardware-ci.md) describes running the test suite with a board in the loop.

## 13. Troubleshooting

**`QICK library version mismatch: 0.2.367 remote (the board), … local (the PC)`.** On stderr,
the first time a program is built. The demo devices' board descriptions were recorded with qick
0.2.367; the notice compares that with your installed qick. It is harmless offline. On a real
board, install the qick version the board runs.

**`GainOutOfRange: … exceeds safe limit 0.5 …`.** A pulse or sweep asked for more amplitude than
its role allows (section 7). Lower the gain or the sweep's `stop`, or, if the element really
needs more (a heavily attenuated line), raise its limit in the device's calibration up to the
role's cap, as the message says ([Raising a limit](#raising-a-limit)). Remember that a scalar `gain=`
is a fraction (0.25) while a gain `Sweep` is in integer codes (8192). A sweep endpoint between 0
and 1 is rejected for that reason, with a message saying it looks like a fraction.

**`KeyError: "no line 'readout' in wiring (have ['qubit'])"`.** The experiment uses a line your
device doesn't wire. You will see this if you run a `cqed` experiment on `loopback_demo`: that
device mirrors qick's official testbench, which has one generator and one readout, so it wires a
single line. Use `multimode_demo`, or add the missing line to your device's `wiring`.

**`ValueError: sweep […] is bound to undeclared axis 'amp': declare it on the program (axes=[Loop('amp', n)]) …`.**
You built a `StrumentoProgram` from a sequence with a bound sweep but did not pass `axes=`. Pass
`axes=[Loop("amp", 51)]`, or `axes=exp.axes()` for an experiment's own sequence (section 7).

**`NonV2SnapshotError: device snapshot is not tProc v2 (tprocs[0].type='axis_tproc64x32_x8', need 'qick_processor') …`.**
The device's board description is from tProc-v1 firmware, which cannot run tProc-v2 programs. Use
a snapshot from tProc-v2 firmware. A device with no snapshot at all gives
`NonV2SnapshotError: this device instance carries no soccfg snapshot …` instead; add
`soccfg_snapshot:` to its YAML (section 11).

**`outcome.figure` is `None`.** `matplotlib` is not installed. From your clone, run
`pip install -e ".[qick,plot]"` to get plots; nothing else depends on it.

**`ImportError: strumento-mcp needs the MCP SDK: pip install 'strumento[mcp]'`.** You started
the agent server without the `mcp` extra. From your clone, run `pip install -e ".[qick,mcp]"`.
`LabSession` itself does not need it.

**`strumento check` prints `FAIL` and exits with status 1.** It lists each mismatch between the
wiring and the board description, by line and role. Fix the wiring, or the snapshot if it is not
your board's.

**`CompilationError: … compilation failed (build). unsupported operand type(s) for *: 'NoneType' and 'float'`.**
The device's readout calibration has no `phase`. The file loads and checks without it, but the
compiler needs it (#227). Add `phase: 0.0` to the readout calibration.

**`ProgramMemoryError: … needs an estimated … words … of tProc program memory, but this board's program memory holds …`.**
The program does not fit the board's tProc program memory: randomized benchmarking puts every
depth in one program. Use fewer or shorter depths, or run one program per depth in a loop
([calibrate-a-transmon.md](calibrate-a-transmon.md#11-randomized-benchmarking)).

**A Pyro call fails with a `Pyro4.errors` exception (for example `ConnectionClosedError`) after
earlier calls worked.** The proxy's worker thread has died, and every later call in the session
fails the same way. Reconnect with a fresh `make_proxy()`; a run script should open a fresh
connection for every run and release it when the run ends.

**The board is locked.** A run that crashed hard (a killed process, a segfault) leaves its lock
file behind. The example run script reclaims a lock whose owning process is no longer running on
the same host; otherwise check that no script is still using the board, then delete the lock
file by hand.

**The live board differs from the stored description.** The firmware, clocking or channel layout
changed since you recorded the snapshot. Re-record it (`dump_cfg()`), run `strumento check`, and
review every calibration that depends on the changed fields. Do not run until the two agree.

**The Ramsey detuning pair refuses (`success` is false; its two fringes disagree).** The qubit is
further from `f_ge` than the pair's delay sampling can resolve, so one fringe has aliased. Run a
wider pair (shorter delays, a larger artificial detuning) to find the offset, apply it, then
repeat the default pair ([calibrate-a-transmon.md](calibrate-a-transmon.md#5-ramsey-detuning-pair)).

**An error-amplification fringe peaks far from an excited population of 1, or populations fall
outside 0 to 1.** The readout blob centres are stale: the IQ frame has rotated since they were
measured. Re-measure them before trusting any population.

Some problems give no error at all:

- **The fit "succeeded" but there was no signal.** A fitter can report success on a trace with no
  signal, leaving its parameters at their starting guesses (#230). Check the trace for a signal
  well above its noise before trusting the fit ([write-a-runner.md](write-a-runner.md#7-judge-the-fit)).
- **The readout is weak, or gains behave differently than calibrated.** Check each generator's
  Nyquist zone against its frequencies. A tone declared in the other zone still plays, with less
  output power (#226; [board-bringup.md](board-bringup.md#wiring)).
- **The resonator frequency is slightly off.** The default magnitude fit is biased on an
  asymmetric line; use `method="circle"` ([board-bringup.md](board-bringup.md#7-rehearse-on-the-simulator)).
- **An error that cannot happen.** A standard-library object missing an attribute, a segfault in
  pure-Python code, or a failure that passes when re-run points at the environment, not the
  experiment: the Python build, a mix of incompatible native packages, or an unstable host. Re-run
  once; if it recurs, reproduce it in a second, independently built environment, and confirm any
  result that matters by a second method.

## 14. Where next

- [cqed-cookbook.md](cqed-cookbook.md): the `cqed` pack's experiments and pulse library, by
  example.
- [packs-and-devices.md](packs-and-devices.md): the device model, `DeviceBuilder`, couplings and
  Hamiltonian export, and how to write a new pack.
- [spin-pack.md](spin-pack.md): the quantum-dot pack; `devices/spin_demo` runs on the simulator
  the same way `multimode_demo` does.
- [calibration-store.md](calibration-store.md) and [task-record.md](task-record.md): running a
  device over time.
- [cli.md](cli.md): every `strumento` command.
- [timeline.md](timeline.md): exactly when each operation plays.
- [architecture.md](architecture.md): the design, the layers, and the decision index (the "D"
  numbers you see in error messages and docstrings).
- [qick-v2-api-notes.md](qick-v2-api-notes.md): the qick tProc-v2 API Strumento builds on.
