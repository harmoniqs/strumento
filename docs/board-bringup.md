# Bring up a new board

This guide takes you from a board on the network to a device that strumento validates against
your hardware and rehearses on the simulator. It walks one path, in order. Each step either runs
as written (the docs test executes every Python block here and checks the printed output) or is
marked as needing a board.

The running example is a **single transmon with a readout resonator**, the smallest useful cQED
device. `devices/multimode_demo/` shows a richer one (cavities, storage modes, a coupler);
[packs-and-devices.md](packs-and-devices.md) explains how elements compose.

When you finish, continue with [getting-started.md §12](getting-started.md#12-going-to-hardware)
to run on the board itself.

## 1. What the board must run

- **tProc-v2 firmware** (`qick_processor`). Strumento compiles to tProc-v2 programs only; a
  tProc-v1 board can load a device but cannot assemble a program.
- **The same qick version on your PC as on the board.** Programs are compiled on whichever
  machine builds them, against the board's configuration. When the two qick versions differ,
  qick prints `QICK library version mismatch: <x> remote (the board), <y> local (the PC)`. With
  the shipped fixtures that notice is harmless; against a real board, install the qick version the
  board runs before you trust a result.
- **A way to reach it**: either Python on the RFSoC itself (`QickSoc()`), or qick's Pyro4 server
  with a name server your PC can reach (`make_proxy(ns_host=…)`).

## 2. Record the board description

Strumento never guesses the hardware. It reads the board's own description, the JSON that qick's
`dump_cfg()` produces: every generator (DAC), its sample rate and full-scale code, every readout
(ADC), and the tProc version. Record it on the board:

<!-- snippet: skip reason="needs a board: QickSoc only runs on the RFSoC" -->
```python
from pathlib import Path
from qick import QickSoc

soc = QickSoc()
Path("my_board_soccfg.json").write_text(soc.dump_cfg())
```

Over Pyro, the configuration object the proxy returns has the same method:

<!-- snippet: skip reason="needs a board: connects to a running qick Pyro4 server" -->
```python
import json
from pathlib import Path
from qick.pyro import make_proxy

soc, soccfg = make_proxy(ns_host="192.168.1.10")    # your name server's address
Path("my_board_soccfg.json").write_text(json.dumps(soccfg.get_cfg()))
```

Make a folder for the device and put the description in it. This guide uses the shipped fixture
in place of your board's description, so every step below runs as written:

```python
import shutil
from pathlib import Path

folder = Path("my_fridge")
folder.mkdir(exist_ok=True)
shutil.copy("devices/_fixtures/soccfg_v2_cqed.json", folder / "my_board_soccfg.json")
print(sorted(p.name for p in folder.iterdir()))
```

<!-- snippet: output -->
```text
['my_board_soccfg.json']
```

`strumento soccfg` summarizes a description, including the firmware type and the qick version:

```bash
strumento soccfg my_fridge/my_board_soccfg.json
```

```text
ZCU216 · qick 0.2.367 · tproc qick_processor · 6 gen(s), 1 readout(s)
tProc v2 — real AveragerProgramV2 assembly works offline
```

`qick 0.2.367` is the qick version that recorded the file, and `qick_processor` means tProc-v2.
To see each generator's sample rate and full-scale code, which the wiring and gain steps below need:

```python
import json

cfg = json.loads((folder / "my_board_soccfg.json").read_text())
for ch, gen in enumerate(cfg["gens"]):
    print(f"DAC {ch}: fs = {gen['fs']:.2f} MHz, full scale = {gen['maxv']}")
```

<!-- snippet: output -->
```text
DAC 0: fs = 9584.64 MHz, full scale = 32766
DAC 1: fs = 9584.64 MHz, full scale = 32766
DAC 2: fs = 9584.64 MHz, full scale = 32766
DAC 3: fs = 9584.64 MHz, full scale = 32766
DAC 4: fs = 9584.64 MHz, full scale = 32766
DAC 5: fs = 9584.64 MHz, full scale = 32766
```

## 3. Describe the device

A device file has four parts: the board description, the elements (**topology**), which
generator plays each line (**wiring**), and your current values (**calibration**). Here is the
whole file for one transmon and its readout resonator:

```python
device_yaml = """\
pack: cqed
soccfg_snapshot: my_board_soccfg.json

topology:
  - {type: Transmon, name: qubit}
  - {type: ReadoutResonator, name: readout}

wiring:
  lines:
    qubit:
      name: qubit
      generators:
        - {role: drive, dac_ch: 1, nyquist: 1}
    readout:
      name: readout
      generators:
        - {role: probe, dac_ch: 0, nyquist: 2}
  readout:
    dac_ch: 0
    adc_chs: [0]

calibration:
  qubit:
    f_ge: 4000.0
    pulses:
      pi_ge:  {gain: 6000, sigma: 0.030, type: gauss}
      hpi_ge: {gain: 3000, sigma: 0.030, type: gauss}
  readout:
    frequency: 7000.0
    gain: 1000
    readout_length: 2.0
    trig_offset: 0.5
    relax_delay: 500
    phase: 0.0
"""
(folder / "device.yaml").write_text(device_yaml)
```

`soccfg_snapshot` is a path relative to the device file. Building the same device in Python is an
alternative to YAML; see `DeviceBuilder` in [packs-and-devices.md](packs-and-devices.md).

### Wiring

Each line names the generators that play it. For each generator:

- **`dac_ch`** is the generator's index in the board description (`DAC 0`…`DAC 5` above).
  Follow your cabling: the DAC whose output reaches the qubit's drive line is the qubit's `drive`
  generator.
- **`nyquist`** is the Nyquist zone the tone lives in. A DAC sampling at $f_s$ covers zone $n$
  from $(n-1)\,f_s/2$ to $n\,f_s/2$. Here $f_s = 9584.64$ MHz, so zone 1 is 0 to 4792 MHz and
  zone 2 is 4792 to 9585 MHz: the 4000 MHz qubit drive is zone 1 and the 7000 MHz readout probe
  is zone 2. The setting does not move the tone: a DAC always produces the requested frequency
  together with its images. It selects the DAC's output mode, which puts more power at
  frequencies in zone 1 or in zones 2 and 3.

**Get the zone right by hand.** Nothing checks it against your frequencies yet: a readout at
7000 MHz declared in zone 1 passes `strumento check` and runs on the simulator without complaint.
On a real board the tone still plays, but with less output power than the matching mode would
give: a weaker readout, and gain codes that do not mean what they would in the right mode. A
tone near a zone boundary loses power in either mode, so there the choice is a judgement; what
matters is that the gains are calibrated with the setting you keep, and recalibrated if you
change it. Compute the zone for every calibrated frequency:

```python
import math

fs = cfg["gens"][5]["fs"]
for name, f in (("qubit drive", 4000.0), ("readout probe", 7000.0)):
    print(f"{name}: {f:.0f} MHz is in Nyquist zone {math.floor(f / (fs / 2)) + 1}")
```

<!-- snippet: output -->
```text
qubit drive: 4000 MHz is in Nyquist zone 1
readout probe: 7000 MHz is in Nyquist zone 2
```

`wiring.readout` names the probe DAC and the ADC (`adc_chs`, the index in the board's
`readouts` list) that digitizes the readout signal.

### First calibration values

Every calibration field has a unit, listed in
[packs-and-devices.md](packs-and-devices.md#cqed-calibration-fields-and-their-units). For a new
device you will not know most values yet. Start from estimates; the rehearsal below and the
calibration chain replace them.

| Field | Unit | Where a first estimate comes from |
|---|---|---|
| `readout.frequency` | MHz | the resonator's design value, or a wide transmission scan on the board |
| `qubit.f_ge` | MHz | the design value, or a wide qubit spectroscopy |
| `pulses.pi_ge.gain` | DAC integer code | a guess well under the drive ceiling (below); amplitude Rabi measures it |
| `pulses.hpi_ge.gain` | DAC integer code | half of `pi_ge` until it is measured separately |
| `pulses.*.sigma` | µs | your gate-length choice; 0.030 µs gives a 0.12 µs Gaussian (4σ) |
| `readout.gain` | DAC integer code | low, well under the readout ceiling, then raised while watching the resonator |
| `readout.readout_length` | µs | a few times the resonator's ring-down time; 1–3 µs is typical |
| `readout.trig_offset` | µs | the time of flight from probe to ADC; must be below `readout_length` |
| `readout.relax_delay` | µs | at least 5 × T1, so the qubit relaxes between shots |
| `readout.phase` | deg | 0.0 |

**Gains are integer codes, limited by role.** A calibrated gain is a DAC code from 0 to the
generator's full scale (32766 here). Strumento converts it to a fraction of full scale when it
builds the pulse and refuses anything above the role's ceiling: 0.5 of full scale for a qubit
drive, 0.6 for the readout probe ([getting-started.md §7](getting-started.md#7-sweeps-and-safety-limits)).
On this board that is codes 16383 and 19659. The ceiling protects the hardware from mistakes. It
is not a suggested operating point: start low. If your line's attenuation means a calibrated pulse
genuinely needs more, raise that element's limit in the calibration (for example
`qubit.max_drive_gain: 0.6`, up to a hard cap); see
[getting-started.md, Raising a limit](getting-started.md#raising-a-limit).

**Set `readout.phase` explicitly.** The device loads without it, but every experiment then fails
to compile with an error from inside qick (`unsupported operand type(s) for *: 'NoneType' and
'float'`). `phase: 0.0` is the right starting value.

## 4. Validate and check

Two commands, both offline:

```bash
strumento validate my_fridge/device.yaml
strumento check my_fridge/device.yaml
```

```text
OK — Device(pack='strumento.packs.cqed', elements=['qubit', 'readout'])
PASS: my_fridge/device.yaml validates against its soccfg
  6 gen(s), 1 readout(s)
```

`validate` checks that the file is a well-formed device of its pack: every required field present
and of the right type. `check` cross-references it against the board description: every
wired `dac_ch` must be a generator, every `adc_ch` a readout, the tProc must be v2. Both return a
non-zero exit code on failure, so scripts can gate on them ([cli.md](cli.md)). The same device
loads in Python:

```python
from strumento.core.device import Device

mine = Device.load(folder / "device.yaml")
print(sorted(mine.elements))
```

<!-- snippet: output -->
```text
['qubit', 'readout']
```

## 5. Compare the live board with the stored description

`strumento check` validates the device against the **stored** description, the file you saved
when you recorded the board. Every gain, frequency and timing strumento computes comes from that
file too. If the board's firmware or clocking changed since you recorded it, nothing notices:
programs are compiled against the old description and can play at the wrong scale or rate.

So compare the live board with the stored file at the start of every session. A small helper
lists every field that differs:

```python
def config_differences(live, stored, path=""):
    """Every field path where two board descriptions disagree."""
    if isinstance(live, dict) and isinstance(stored, dict):
        out = []
        for key in sorted(set(live) | set(stored)):
            out += config_differences(live.get(key), stored.get(key), f"{path}.{key}" if path else key)
        return out
    if isinstance(live, list) and isinstance(stored, list) and len(live) == len(stored):
        out = []
        for i, (a, b) in enumerate(zip(live, stored)):
            out += config_differences(a, b, f"{path}[{i}]")
        return out
    return [] if live == stored else [path]

stored = json.loads((folder / "my_board_soccfg.json").read_text())
print(config_differences(stored, stored))
```

<!-- snippet: output -->
```text
[]
```

An empty list means the board matches. Here is what a firmware change looks like. Suppose a
board update changed one generator's sample rate:

```python
import copy

changed = copy.deepcopy(stored)
changed["gens"][2]["fs"] = 6881.28
print(config_differences(changed, stored))
```

<!-- snippet: output -->
```text
['gens[2].fs']
```

On the board, compare the live description with the stored one. The round trip through JSON
makes the live description comparable with the file (qick returns some fields as tuples, which
the file stores as lists):

<!-- snippet: skip reason="needs a board: connects to a running qick Pyro4 server" -->
```python
soc, soccfg = make_proxy(ns_host="192.168.1.10")
live = json.loads(json.dumps(soccfg.get_cfg()))
print(config_differences(live, stored))
```

**On a mismatch, stop.** Re-record the board description, run `strumento check` again, and
review any calibration that depends on the changed fields before you trust a result. Put this
comparison at the start of every run script, so it happens on every run rather than when you
remember it: [write-a-runner.md](write-a-runner.md) builds it into an example run script.

## 6. Check the signal chain

Before you trust anything strumento builds, confirm the hardware path with plain qick programs.
`examples/fallback/` holds six of them, which bypass strumento entirely. Run them in order on a
new board:

| Script | What it answers |
|---|---|
| `adc_trace.py` | Is the ADC alive? An open input still shows a noise floor; all zeros means the acquisition path is broken |
| `dac_loopback.py` | Does a tone leave a DAC and arrive at an ADC (with a loopback cable or board loopback) at the programmed frequency? |
| `resonator_spec.py` | Is there a resonator dip where you expect it? |
| `rabi.py` | Does the qubit respond to its drive? |
| `t1.py`, `t2_ramsey.py` | Are the coherence times plausible? |

Each takes the shared flags `--soccfg`, `--gen-ch`, `--ro-ch`, `--reps` and `--plot`, plus its own
(`--help` lists them). Offline, against a description, they assemble and run on a mock board that
returns zeros, which proves only that the program builds:

```bash
cd examples/fallback
python resonator_spec.py --soccfg ../../my_fridge/my_board_soccfg.json --gen-ch 0 --center-freq 7000 --nqz 2
```

On the board, run them in a Python session on the RFSoC and pass the board in: each script's
`main(soc=…)` takes a live `QickSoc`. [examples/fallback/README.md](../examples/fallback/README.md)
explains each one.

**These scripts have no gain guards.** Their `--gain` is a fraction of full scale and nothing
caps it. Keep it low on a new board.

## 7. Rehearse on the simulator

The simulator plays your device's programs against a physical response model: real tProc-v2
assembly, simulated acquisition. Give it **ground truth** that differs from your first
estimates, and the first experiments of the calibration chain should measure the truth, not your
guess. That proves the device, wiring and experiments work end to end before the board is
involved.

The truth keys name fitted quantities: `resonator_f0` (MHz), `f_ge` (MHz) and `pi_gain` (DAC
integer code). Here the truth is 0.6 MHz, 3 MHz and 550 codes away from the guesses in the
device file:

```python
from strumento.core.simulator import SimulatorSoc
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi
from strumento.packs.cqed.experiments.qubit_spectroscopy import QubitSpectroscopy
from strumento.packs.cqed.experiments.resonator_spectroscopy import ResonatorSpectroscopy

sim = SimulatorSoc.from_device(mine, seed=1, truth={"resonator_f0": 7000.6, "f_ge": 4003.0, "pi_gain": 6550})

res = ResonatorSpectroscopy(mine, freqs=Sweep(start=6995.0, stop=7005.0, on="f"), method="circle")
print(f"resonator: {res.go(sim, reps=200, progress=False).fit.derived['f0']:.2f} MHz")

qub = QubitSpectroscopy(mine, freqs=Sweep(start=3990.0, stop=4010.0, on="freq"))
print(f"qubit:     {qub.go(sim, reps=200, progress=False).fit.derived['f0']:.2f} MHz")

rabi = AmplitudeRabi(mine, gains=Sweep(start=0, stop=16000, on="amp"), points=81)
fit = rabi.go(sim, reps=2000, progress=False).fit
print(f"pi gain:   {fit.derived['pi_gain']:.0f} ± {fit.derived['pi_gain_stderr']:.0f}")
```

<!-- snippet: output -->
```text
resonator: 7000.60 MHz
qubit:     4003.00 MHz
pi gain:   6558 ± 14
```

All three recover the truth: the resonator and qubit frequencies exactly, the π gain within one
standard error. Three things this rehearsal teaches:

- **Each scan must contain its answer.** A sweep is centred on your current belief unless you
  give it `freqs=` or `gains=`; centre a first scan wide enough to include the real value. A
  frequency sweep binds to the experiment's axis (`on="f"` for the resonator, `on="freq"` for
  the qubit, `on="amp"` for the Rabi gain).
- **Use the circle fit for the readout resonator.** A real resonator line is asymmetric
  (impedance mismatch). The default magnitude-Lorentzian fit reads this asymmetric line
  0.14 MHz high, more than 20 times its own error bar; `method="circle"` fits the complex response
  and removes the bias.
- **Truth keys are flat.** `truth={"f_ge": 4003.0}` works; a nested dict shaped like the
  calibration (`{"qubit": {"f_ge": …}}`) is not a truth key, and the simulator silently tells the
  truth of your own device file instead.

Write each measured value back into the device file as you go, by hand or through the
calibration store ([calibration-store.md](calibration-store.md)), and re-run §4.

## 8. Next

- [getting-started.md §12](getting-started.md#12-going-to-hardware): run the same experiments on
  the board, on the RFSoC or over Pyro.
- [calibrate-a-transmon.md](calibrate-a-transmon.md): the calibration chain from here to randomized
  benchmarking.
- [cqed-cookbook.md](cqed-cookbook.md): every cQED experiment by example.
- [calibration-store.md](calibration-store.md): keep a versioned history of the device's
  calibration.
