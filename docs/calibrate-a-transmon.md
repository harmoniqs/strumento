# Calibrate a transmon

This guide calibrates a single transmon from first estimates to randomized benchmarking. Each step
measures one quantity, judges the result, and writes one calibration edit, which the next step
then plays with. The order matters: every step relies on the ones before it.

| # | Step | Measures | Updates |
|---|---|---|---|
| 1 | Resonator spectroscopy | readout resonator frequency | `readout.frequency` |
| 2 | Qubit spectroscopy | qubit frequency, coarsely | `qubit.f_ge` |
| 3 | Amplitude Rabi | π gain, coarsely | `pulses.pi_ge.gain`, `pulses.hpi_ge.gain` |
| 4 | Readout blobs | where \|g⟩ and \|e⟩ land in the IQ plane | `readout.g_center_iq`, `e_center_iq`, `angle`, `threshold` |
| 5 | Ramsey detuning pair | the remaining qubit detuning, with its sign | `qubit.f_ge` |
| 6 | π error amplification | π gain, finely | `pulses.pi_ge.gain` |
| 7 | π/2 error amplification | π/2 gain, finely and separately | `pulses.hpi_ge.gain` |
| 8 | DRAG pair | the DRAG coefficient | `qubit.drag_alpha` |
| 9 | AllXY | a check of the whole gate set | nothing |
| 10 | T1, T2 Ramsey, T2 echo | coherence | `qubit.T1`, `T2`, `T2_echo` |
| 11 | Randomized benchmarking | error per Clifford | nothing |

Everything below runs on the simulator, and the docs test executes every block. On a board, run
the same experiments through a run script ([write-a-runner.md](write-a-runner.md)) and apply each
edit after reviewing the run. The device is the single transmon from
[board-bringup.md](board-bringup.md).

## The simulated board

The simulator models each experiment separately: a qubit spectroscopy reads the true qubit
frequency, a Ramsey reads a true *detuning*, an error amplification reads a true *gain error*. To
make them behave like one physical qubit, a small helper derives every experiment's truth from one
description of the real device and the calibration you currently believe. The real device differs
from the first estimates in every quantity the chain measures:

```python
import math
import shutil
from pathlib import Path

import numpy as np

from strumento.core.config import apply_updates
from strumento.core.device import Device
from strumento.core.simulator import SimulatorSoc
from strumento.core.sweeps import Sweep

# The real device: what the chain should find.
REAL = {
    "resonator_mhz": 7000.6, "qubit_mhz": 4003.16, "pi_gain": 6620, "hpi_gain": 3340,
    "drag_alpha": -0.05, "T1": 90.0, "T2": 60.0, "T2_echo": 110.0, "epc": 2.5e-3,
}

def board(device, pulse="pi_ge", seed=1):
    """A simulated board that behaves like REAL, as seen from `device`'s current calibration."""
    q = device.calib.qubit
    true_gain = REAL["pi_gain"] if pulse == "pi_ge" else REAL["hpi_gain"]
    truth = {
        "resonator_f0": REAL["resonator_mhz"], "f_ge": REAL["qubit_mhz"], "pi_gain": REAL["pi_gain"],
        "ramsey_detuning": float(q.f_ge) - REAL["qubit_mhz"],
        "err_amp_gain_delta": float(q.pulses[pulse].gain) - true_gain,
        "drag_alpha_opt": REAL["drag_alpha"],
        "T1": REAL["T1"], "T2": REAL["T2"], "T2_echo": REAL["T2_echo"],
        "rb_decay": -1.0 / math.log(1.0 - 2.0 * REAL["epc"]),
    }
    return SimulatorSoc.from_device(device, seed=seed, truth=truth)

def update(device, edits):
    """Apply calibration edits and print what changed: old -> new."""
    config, diff = apply_updates(device.to_config(), edits)
    for path, (old, new) in diff.items():
        print(f"  {path}: {old} -> {new}")
    return Device.from_config(config)
```

The starting device file, with first estimates (see the board bring-up guide for where they come
from):

```python
folder = Path("my_fridge")
folder.mkdir(exist_ok=True)
shutil.copy("devices/_fixtures/soccfg_v2_cqed.json", folder / "my_board_soccfg.json")
(folder / "device.yaml").write_text("""\
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
""")
dev = Device.load(folder / "device.yaml")
```

**What the simulator cannot show.** It is a set of phenomenological models, not a qubit. It does
not model the IQ frame of the readout rotating, it places shots on whatever blob centres you
recorded (so it never disagrees with your readout calibration), its AllXY is a near-ideal
staircase whatever the gates, and its RB decay is a parameter rather than a consequence of the
gates. The steps below say where that matters and what to watch for on a board.

## 1. Resonator spectroscopy

Find the readout resonator. Scan wide enough to contain it, and use the circle fit: a real
resonator line is asymmetric, and the default magnitude fit reads it biased (see the bring-up
guide).

```python
from strumento.packs.cqed.experiments.resonator_spectroscopy import ResonatorSpectroscopy

res = ResonatorSpectroscopy(dev, freqs=Sweep(start=6995.0, stop=7005.0, on="f"), method="circle")
fit = res.go(board(dev), reps=200, progress=False).fit
print(f"resonator: {fit.derived['f0']:.2f} MHz")
dev = update(dev, {"readout.frequency": round(fit.derived["f0"], 2)})
```

<!-- snippet: output -->
```text
resonator: 7000.60 MHz
  readout.frequency: 7000.0 -> 7000.6
```

**Judge it:** the fitted frequency is inside the scan and not at an edge, and the dip is clearly
above the noise. **Then:** everything in the readout's IQ plane depends on this frequency. If you
change it later, re-measure the readout blobs (step 4).

## 2. Qubit spectroscopy

Find the qubit. A spectroscopy line is broad (here about 4 MHz wide), so its centre is a coarse
estimate: good to tens of kilohertz at best.

```python
from strumento.packs.cqed.experiments.qubit_spectroscopy import QubitSpectroscopy

spec = QubitSpectroscopy(dev, freqs=Sweep(start=3985.0, stop=4015.0, on="freq"), points=61)
fit = spec.go(board(dev), reps=200, progress=False).fit
print(f"qubit: {fit.derived['f0']:.3f} ± {fit.derived['f0_stderr']:.3f} MHz")
dev = update(dev, {"qubit.f_ge": round(fit.derived["f0"], 3)})
```

<!-- snippet: output -->
```text
qubit: 4003.130 ± 0.027 MHz
  qubit.f_ge: 4000.0 -> 4003.13
```

The real qubit is at 4003.16 MHz: the spectroscopy is 30 kHz low, about one standard error. Step 5
removes the rest.

## 3. Amplitude Rabi

Find the gain that rotates the qubit by π. Until step 7 measures it, take the π/2 gain as half.

```python
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi

rabi = AmplitudeRabi(dev, gains=Sweep(start=0, stop=16000, on="amp"), points=81)
fit = rabi.go(board(dev), reps=2000, progress=False).fit
pi = round(fit.derived["pi_gain"])
print(f"pi gain: {fit.derived['pi_gain']:.0f} ± {fit.derived['pi_gain_stderr']:.0f}")
dev = update(dev, {"qubit.pulses.pi_ge.gain": pi, "qubit.pulses.hpi_ge.gain": round(pi / 2)})
```

<!-- snippet: output -->
```text
pi gain: 6618 ± 16
  qubit.pulses.pi_ge.gain: 6000 -> 6618
  qubit.pulses.hpi_ge.gain: 3000 -> 3309
```

**Judge it:** the scan shows a clear oscillation (not just a fit that converged: see
[write-a-runner.md](write-a-runner.md)), and the π gain sits well inside it. Treat Rabi as the
coarse step it is: on hardware, amplitude Rabi and error amplification can disagree by more than
Rabi's error bar (distortion of the drive, a slightly detuned drive), and error amplification is
the one to trust.

## 4. Readout blobs

Precision steps convert the measured signal into an excited-state population, anchored on where
the |g⟩ and |e⟩ readouts land in the IQ plane. `SingleShotReadout` measures |g⟩, then |e⟩ (after
a π pulse), in one program; the two averaged reads are the two blob centres. The discrimination
angle points from |g⟩ to |e⟩, and the threshold is the midpoint along it:

```python
from strumento.packs.cqed.experiments.single_shot_readout import SingleShotReadout

out = SingleShotReadout(dev).go(board(dev), reps=2000, progress=False)
i, q = np.ravel(out.result.avgi), np.ravel(out.result.avgq)
g, e = (i[0], q[0]), (i[1], q[1])
angle = math.atan2(e[1] - g[1], e[0] - g[0])
along = lambda p: p[0] * math.cos(angle) + p[1] * math.sin(angle)
print(f"g = ({g[0]:.3f}, {g[1]:.3f}), e = ({e[0]:.3f}, {e[1]:.3f})")
dev = update(dev, {
    "readout.g_center_iq": [round(float(g[0]), 4), round(float(g[1]), 4)],
    "readout.e_center_iq": [round(float(e[0]), 4), round(float(e[1]), 4)],
    "readout.angle": round(angle, 5),
    "readout.threshold": round(float(along(g) + along(e)) / 2, 4),
})
```

<!-- snippet: output -->
```text
g = (0.475, -0.028), e = (15.585, -0.050)
  readout.g_center_iq: None -> [0.475, -0.0278]
  readout.e_center_iq: None -> [15.5847, -0.0501]
  readout.angle: None -> -0.00148
  readout.threshold: None -> 8.0299
```

**Judge it:** the two centres are far apart compared with the blobs' width. Single-shot
fidelity needs per-shot data, which the averaged hardware path does not return; the centres do
not.

**Re-measure the blobs whenever the readout changes**: its frequency, its gain, or at the start
of a new session. The IQ plane's orientation is set by phases and delays along the whole readout
chain, and it can rotate substantially between sessions. With stale centres every population is
wrong, and the steps below fit the wrong numbers without complaint. The simulator cannot show
this, because it places its shots on whatever centres you recorded. Step 6's peak-population check
is the guard: a full excitation that reads far from 1 means the anchors are stale.

## 5. Ramsey detuning pair

Spectroscopy left the qubit tens of kilohertz from where you believe it is. A Ramsey fringe
measures that detuning far more precisely, but a single Ramsey cannot tell its sign. The **pair**
runs two Ramseys, driven at $f_{ge} + \Delta_a$ and $f_{ge} - \Delta_a$ with an artificial
detuning $\Delta_a$ (50 kHz by default); the two fringes, $|\Delta_a - \delta|$ and
$|\Delta_a + \delta|$, fix the qubit detuning $\delta = f_q - f_{ge}$ with its sign:

```python
from strumento.packs.cqed.experiments.ramsey_detuning_pair import RamseyDetuningPair

fit = RamseyDetuningPair(dev).go(board(dev), reps=500, progress=False).fit
d = fit.derived
print(f"f_q - f_ge = {1000 * d['qubit_detuning_mhz']:+.1f} ± {1000 * d['qubit_detuning_stderr']:.1f} kHz, T2* = {d['T2']:.0f} us")
dev = update(dev, {"qubit.f_ge": round(d["f_ge_corrected"], 5)})
```

<!-- snippet: output rtol=0.01 -->
```text
f_q - f_ge = +30.1 ± 0.1 kHz, T2* = 62 us
  qubit.f_ge: 4003.13 -> 4003.16005
```

**The trap: a detuning too large for the pair.** The default pair samples the fringes every
4 µs, so it can resolve fringes up to 125 kHz. If the qubit is further from your belief than the
pair can see, one fringe aliases, the two no longer agree, and the pair **refuses** rather than
reporting a wrong frequency. Here is a belief 160 kHz low, well within what a broad
spectroscopy line can leave behind:

```python
stale = update(dev, {"qubit.f_ge": REAL["qubit_mhz"] - 0.16})
fit = RamseyDetuningPair(stale).go(board(stale), reps=500, progress=False).fit
d = fit.derived
print(f"success: {fit.success}; fringes {1000 * d['fringe_plus_mhz']:.1f} and {1000 * d['fringe_minus_mhz']:.1f} kHz, "
      f"disagreeing by {1000 * d['consistency_residual_mhz']:.1f} kHz")
```

<!-- snippet: output rtol=0.01 -->
```text
  qubit.f_ge: 4003.16005 -> 4003.0
success: False; fringes 109.9 and 40.0 kHz, disagreeing by 15.1 kHz
```

**The recovery:** a *wide* pair, with shorter delays and a larger artificial detuning, sees much
larger offsets (here up to about 0.9 MHz). It finds the offset; the default pair then confirms
it:

```python
wide = RamseyDetuningPair(stale, stop_us=40.0, ramsey_freq_mhz=0.4)
fit = wide.go(board(stale), reps=500, progress=False).fit
print(f"wide pair: f_q - f_ge = {1000 * fit.derived['qubit_detuning_mhz']:+.1f} kHz")
recovered = update(stale, {"qubit.f_ge": round(fit.derived["f_ge_corrected"], 5)})
fit = RamseyDetuningPair(recovered).go(board(recovered), reps=500, progress=False).fit
print(f"default pair: f_q - f_ge = {1000 * fit.derived['qubit_detuning_mhz']:+.1f} kHz")
```

<!-- snippet: output rtol=0.02 atol=0.2 -->
```text
wide pair: f_q - f_ge = +160.0 kHz
  qubit.f_ge: 4003.0 -> 4003.16
default pair: f_q - f_ge = +0.1 kHz
```

A refusal followed by a wide pair and a passing default pair is the expected path from a stale
qubit frequency. A Ramsey chevron (a 2D map of Ramsey fringes against drive detuning) is another
way to find a large offset.

## 6. π error amplification

Amplitude Rabi finds the π gain to about a percent. Error amplification repeats the pulse: after
$N$ π pulses a small over-rotation $\epsilon$ has grown to $N\epsilon$, so the excited population,
swept across gains near the belief, oscillates with a fringe whose peak sits at the true π gain.
With $N$ odd ($N = 7$), the peak is a full excitation. The fringe period is fixed by the physics,
$2 g_\pi / N$, so the fit is told it:

```python
from strumento.packs.cqed.experiments.error_amplification import ErrorAmplification

def amplify(device, pulse, n):
    """Error-amplify one pulse's gain in a ±400-code window around its current belief."""
    g0 = float(device.calib.qubit.pulses[pulse].gain)
    angle_per_pulse = math.pi if pulse == "pi_ge" else math.pi / 2
    exp = ErrorAmplification(
        device, pulse=device.qubit.ge_pi if pulse == "pi_ge" else device.qubit.ge_hpi,
        parameter="gain", n_pulses=n, window=Sweep(start=g0 - 400, stop=g0 + 400, on="g"),
        fringe_period=2 * math.pi * g0 / (n * angle_per_pulse),
    )
    return exp.go(board(device, pulse), reps=500, progress=False).fit

fit = amplify(dev, "pi_ge", 7)
d = fit.derived
print(f"pi gain: {d['gain_corrected']:.1f} ± {d['gain_corrected_stderr']:.1f}, peak P_e = {d['peak_population']:.2f}")
dev = update(dev, {"qubit.pulses.pi_ge.gain": round(d["gain_corrected"])})
```

<!-- snippet: output rtol=0.01 -->
```text
pi gain: 6615.7 ± 2.5, peak P_e = 0.98
  qubit.pulses.pi_ge.gain: 6618 -> 6616
```

**Judge it:** the peak population is close to 1. This check catches stale readout centres (step
4): a peak far from 1 means the populations are being computed with the wrong anchors, so
re-measure the blobs before trusting the gain. Repeat the step if the correction is a large
fraction of the window: the fringe period is computed from the belief, so a belief far off
distorts it. A second run that lands within its error bars of the first confirms the value.

## 7. π/2 error amplification

Measure the π/2 gain on its own. Half the π gain is not the right π/2 gain: amplitude-to-angle is
not perfectly linear on real hardware, and the pulses' own shapes differ slightly. With $N = 10$
π/2 pulses ($N \equiv 2 \bmod 4$, a total of 5π), the peak is again a full excitation:

```python
fit = amplify(dev, "hpi_ge", 10)
d = fit.derived
print(f"pi/2 gain: {d['gain_corrected']:.1f} ± {d['gain_corrected_stderr']:.1f}, peak P_e = {d['peak_population']:.2f}")
dev = update(dev, {"qubit.pulses.hpi_ge.gain": round(d["gain_corrected"])})
print(f"pi / (pi/2) = {dev.calib.qubit.pulses['pi_ge'].gain / dev.calib.qubit.pulses['hpi_ge'].gain:.3f}")
```

<!-- snippet: output rtol=0.01 -->
```text
pi/2 gain: 3341.7 ± 1.4, peak P_e = 0.98
  qubit.pulses.hpi_ge.gain: 3309 -> 3342
pi / (pi/2) = 1.980
```

The ratio is 1.98, not 2: half of π would have left the π/2 gain 1 % low. Measure the π/2 gain
every time you re-calibrate π; the ratio is not a constant of the device.

## 8. DRAG pair

A transmon is not a two-level system: its next level (|f⟩) is only the anharmonicity away, and a
short pulse leaks towards it and picks up a phase error. DRAG adds a derivative component on the
other quadrature, scaled by the coefficient α. The **pair** plays X(π/2)·Y(π) and Y(π/2)·X(π)
across α; the difference of the two populations crosses zero at the right α, and taking the
difference cancels readout offsets and amplitude errors:

```python
from strumento.packs.cqed.experiments.drag_pair_calibration import DRAGPairCalibration

fit = DRAGPairCalibration(dev, points=31).go(board(dev), reps=500, progress=False).fit
d = fit.derived
print(f"alpha = {d['alpha_opt']:.3f} ± {d['alpha_opt_stderr']:.3f}, equator offset {d['equator_offset']:+.3f}")
dev = update(dev, {"qubit.drag_alpha": round(d["alpha_opt"], 4)})
```

<!-- snippet: output rtol=0.02 atol=0.002 -->
```text
alpha = -0.054 ± 0.003, equator offset +0.003
  qubit.drag_alpha: None -> -0.054
```

**Then:** DRAG changes the pulses every later step plays. If α moved by more than its error bar,
repeat steps 6 and 7 with it.

## 9. AllXY

AllXY plays 21 pairs of gates that should leave the qubit in |g⟩, on the equator, or in |e⟩: a
staircase of 0, ½ and 1. Its deviations diagnose the gate set as a whole:

```python
from strumento.packs.cqed.experiments.transmon_allxy import TransmonAllXY

fit = TransmonAllXY(dev).go(board(dev), reps=500, progress=False).fit
pops = np.asarray(fit.derived["populations"], dtype=float)
drag_pairs = ((pops[9] - pops[10]) + (pops[11] - pops[12])) / 2   # (xY - yX) and (Xy - Yx)
print(f"rms deviation {fit.derived['rms_deviation']:.3f}; DRAG-sensitive pair difference {drag_pairs:+.3f}")
```

<!-- snippet: output atol=0.005 -->
```text
rms deviation 0.017; DRAG-sensitive pair difference +0.000
```

**Judge it with the right quantity.** The overall rms is a weak check: it averages 21 points,
and a wrong DRAG coefficient moves only a few of them, so it can leave the rms small. Check the
pair difference that is sensitive to DRAG, computed above: the mean of $(xY - yX)$ and
$(Xy - Yx)$, pairs 9–10 and 11–12 of the staircase. Ideal gates give zero; a DRAG error pushes the
two terms the same way. A difference of several times its shot noise (about $1/\sqrt{\text{reps}}$)
means the DRAG coefficient is wrong. The simulator's AllXY is a near-ideal staircase whatever the
gates, so here it reads zero.

## 10. Coherence

T1 (energy relaxation), T2* (Ramsey dephasing) and T2 echo (dephasing with a refocusing π pulse)
set the floor for the gates' error. Scan each delay out to several times the expected value:

```python
from strumento.packs.cqed.experiments.t1 import T1
from strumento.packs.cqed.experiments.t2_echo import T2Echo
from strumento.packs.cqed.experiments.t2_ramsey import T2Ramsey

t1 = T1(dev, delays=Sweep(start=0.0, stop=400.0, on="t"), points=61).go(board(dev), reps=500, progress=False).fit
t2 = T2Ramsey(dev, delays=Sweep(start=0.0, stop=200.0, on="t"), ramsey_freq_mhz=0.05).go(board(dev), reps=500, progress=False).fit
echo = T2Echo(dev, delays=Sweep(start=0.0, stop=300.0, on="t"), points=61).go(board(dev), reps=500, progress=False).fit
print(f"T1 = {t1.derived['T1']:.0f} ± {t1.derived['T1_stderr']:.0f} us, "
      f"T2* = {t2.derived['T2']:.0f} ± {t2.derived['T2_stderr']:.0f} us, T2 echo = {echo.derived['T2_echo']:.0f} us")
dev = update(dev, {"qubit.T1": round(t1.derived["T1"], 1), "qubit.T2": round(t2.derived["T2"], 1),
                   "qubit.T2_echo": round(echo.derived["T2_echo"], 1)})
```

<!-- snippet: output rtol=0.02 atol=0.5 -->
```text
T1 = 89 ± 2 us, T2* = 60 ± 2 us, T2 echo = 108 us
  qubit.T1: None -> 88.7
  qubit.T2: None -> 60.1
  qubit.T2_echo: None -> 107.9
```

**Judge it:** the decay is well inside the scan, and a single exponential describes it. The
scans here reach 4.5 T1, 3.3 T2* and 2.8 echo times. A scan that stops much short of three
decay times leaves the fit's floor unconstrained and the time biased; extend it and re-measure. A
T2* far below its echo (here 60 against 108 µs) means slow noise that the echo refocuses: the
Ramsey-based steps (5) see the shorter one.

## 11. Randomized benchmarking

RB plays random sequences of Clifford gates of increasing length $m$, each followed by the
Clifford that undoes them, and measures how often the qubit returns to |g⟩. The survival decays as
$A p^m + B$; the error per Clifford is $(1 - p)/2$. It measures the average error of the gate set
as it is actually used.

**Program memory limits the depths.** `TransmonRB` puts every depth in one program, and on this
board one program holds 4096 words, so only a handful of depths up to a few hundred Cliffords:

```python
from strumento.packs.cqed.experiments.transmon_rb import ProgramMemoryError, TransmonRB

try:
    TransmonRB(dev, depth_start=1, depth_stop=600, points=4).sequence()
except ProgramMemoryError as error:
    print(str(error).split(". Remedy")[0])
```

<!-- snippet: output -->
```text
TransmonRB: 4 depths (1…600, 1,201 Cliffords in total) in one program needs an estimated 4,507 words (a lower bound) of tProc program memory, but this board's program memory holds 4,096 words (tprocs[0].pmem_size in the soccfg snapshot)
```

For long sequences, run one program per depth and seed in a loop, as the run-script guide's
"one program per run" limit describes. Depths up to 400 fit:

```python
rb = TransmonRB(dev, depth_start=1, depth_stop=400, points=5)
out = rb.go(board(dev), reps=500, progress=False)
print(out.fit.success, [int(m) for m in rb._depths])
```

<!-- snippet: output -->
```text
True [1, 100, 200, 300, 400]
```

The simulator's RB response is a placeholder, a plain exponential towards zero, not RB physics,
so its number is not a measurement. The analysis lesson below uses data shaped like real RB.

**Pin the asymptote, and reach far enough.** For a qubit, random Cliffords leave a fully mixed
state, so the survival decays towards $B = 1/2$. A fit that leaves $B$ free has three parameters,
and when the depths reach less than a couple of decay constants, $B$ and the decay trade off:
the error per Clifford it reports then depends on which depths you chose. Here is the effect on
survival data with a true error per Clifford of $2.5\times10^{-3}$:

```python
from scipy.optimize import curve_fit

def rb_fits(depths, survival):
    """Error per Clifford from a free-asymptote fit and from one pinned at B = 1/2."""
    (_, p_free, b), _ = curve_fit(lambda m, a, p, b: a * p**m + b, depths, survival,
                                  p0=[0.5, 0.99, 0.5], bounds=([0, 0.9, 0], [1, 1, 1]))
    (_, p_pinned), _ = curve_fit(lambda m, a, p: a * p**m + 0.5, depths, survival,
                                 p0=[0.5, 0.99], bounds=([0, 0.9], [1, 1]))
    return (1 - p_free) / 2, b, (1 - p_pinned) / 2

rng = np.random.default_rng(7)
true_p = 1 - 2 * 2.5e-3
for stop in (150, 600):
    depths = np.linspace(1, stop, 8).round()
    survival = 0.48 * true_p**depths + 0.5 + rng.normal(0, 0.006, depths.size)
    free, b, pinned = rb_fits(depths, survival)
    print(f"depths to {stop}: free {free:.2e} (B = {b:.2f}), pinned {pinned:.2e}")
```

<!-- snippet: output rtol=0.01 atol=0.005 -->
```text
depths to 150: free 3.39e-03 (B = 0.59), pinned 2.49e-03
depths to 600: free 2.48e-03 (B = 0.50), pinned 2.48e-03
```

With depths reaching only 0.8 decay constants, the free fit over-estimates the error by 36 %; the
pinned fit is right, and so is the free fit once the depths reach three decay constants. On a
board, convert the readout to population with the blob centres from step 4, pin $B = 1/2$, and
choose depths that reach at least twice the decay constant.

**Judge it against coherence.** Decoherence alone sets a floor on the error per Clifford:
$t_C/(6T_1) + t_C/(3T_2)$, where $t_C$ is the average Clifford's duration. The shipped Clifford
decomposition averages 1.875 pulses per Clifford, each 4σ long:

```python
t_clifford = 1.875 * 4 * dev.calib.qubit.pulses["pi_ge"].sigma
limit = t_clifford / (6 * dev.calib.qubit.T1) + t_clifford / (3 * dev.calib.qubit.T2)
print(f"coherence limit: {limit:.2e} per Clifford")
```

<!-- snippet: output rtol=0.01 -->
```text
coherence limit: 1.67e-03 per Clifford
```

A measured error close to the limit means the gates are coherence-limited; well above it means
calibration error remains, so revisit steps 5–8. **Below** the limit is a warning sign, not a
success: something in the measurement is optimistic, such as identity Cliffords that do not idle
for a gate's length, stale blob centres, or too little depth. Using T2* here gives the
conservative floor; with T2 echo it is lower, because random Clifford sequences refocus some slow
noise, and a measured error between the two is plausible.

## 12. Keep the result

Write the calibrated device back to a file, with the board description as a relative path, and
confirm it loads and checks:

```python
import yaml

saved = dev.to_config().to_dict()
saved["soccfg_snapshot"] = "my_board_soccfg.json"
(folder / "calibrated.yaml").write_text(yaml.safe_dump(saved, sort_keys=False))
again = Device.load(folder / "calibrated.yaml")
q = again.calib.qubit
print(f"f_ge {q.f_ge} MHz, pi {q.pulses['pi_ge'].gain:.0f}, pi/2 {q.pulses['hpi_ge'].gain:.0f}, drag {q.drag_alpha}")
```

<!-- snippet: output -->
```text
f_ge 4003.16005 MHz, pi 6616, pi/2 3342, drag -0.054
```

Against the real device (4003.16 MHz, 6620, 3340, −0.05) every value is within two standard
errors.
Run `strumento check my_fridge/calibrated.yaml` before using it. To keep a versioned history of
every edit instead of overwriting files, use the calibration store
([calibration-store.md](calibration-store.md)).
