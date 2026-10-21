# Importing data from other control backends

`strumento.interop.backends` brings data you acquired with another control stack (a QUA program
on an OPX, a Qblox or Quantify run, a Zurich Instruments sweep, a VNA, a CSV file) into a
`MeasurementResult`, the object Strumento's fitters, plots, verdicts and records consume. It
converts arrays; it does not drive the other stack. Your script still runs the instrument and
fetches the data, and Strumento does the analysis.

The module is pure: it needs only `numpy`, never `qick`, `qm` or `qblox`. `xarray` is not
needed either, since the `from_xarray` functions also accept a plain dict of arrays.

Every Python block here runs in the docs test (`tests/test_docs.py`), top to bottom in one
session, from the repository root, on made-up arrays, and the output shown under a block is
checked.

| Name | What it does |
|---|---|
| `from_iq(xpts, i, q, *, shots=None, backend=, meta=)` | separate I and Q arrays, one value per sweep point |
| `from_complex(xpts, signal, ...)` | one complex array (`I + jQ`), e.g. a VNA's S21 or a demodulated trace |
| `from_counts(xpts, excited_fraction, *, total_shots=None, ...)` | discriminated data: the excited fraction goes on `avgi`, and `avgq` is zero |
| `from_multiread(xpts, reads, ...)` | several reads per point, a list of `(i, q)` pairs; `avgi` / `avgq` have shape `(n_reads, n_points)` |
| `from_xarray(dataset, *, x=, i=, q=, x_scale=1.0, ...)` | an `xarray.Dataset` (Quantify, Qblox) or a dict, by variable name |
| `from_xarray_complex(dataset, *, x=, signal=, ...)` | the same, with one complex data variable |
| `analyze_resonator`, `analyze_spectroscopy`, `analyze_rabi`, `analyze_t1`, `analyze_t2_ramsey` | import and fit in one call; each returns a `FitResult` |
| `external_outcome(xpts, i, q, *, fit_fn, signal_fn=None, ...)` | import and fit into an `ExperimentOutcome`, the object `Experiment.go()` returns |
| `ExternalExperiment(device, *, acquire_fn, fit_fn, ...)` | wrap your acquisition function as an `Experiment`, so it can be a bring-up step |
| `extract_oc_pulses(solution, *, drive_map, gain_calib, dac_rate=None)` | the other direction: optimal-control waveforms as plain arrays (`WaveformExport`) for another stack's pulse format |
| `piccolo_controls_to_solution(times_ns, controls_ghz, *, drive)` | Piccolo's `(u_x, u_y)` controls (GHz, ns) as the `{times, controls}` input of `from_solution` (MHz, µs); see [Piccolo convention](#piccolo-convention) |

`backend=` is a free-form tag (`"opx"`, `"qblox"`, `"zurich"`, …) stamped on
`result.meta["backend"]`; anything in `meta=` is merged into the same dict.

## Arrays in, a `MeasurementResult` out

A Rabi gain sweep, as an OPX script might hand it back:

```python
import numpy as np

from strumento.core.fitters import fit_rabi
from strumento.interop.backends import from_iq

gains = np.linspace(0.0, 1.0, 41)
i = 0.5 - 0.5 * np.cos(2 * np.pi * gains / 0.8)    # a Rabi oscillation with period 0.8
q = 0.1 * i

result = from_iq(gains, i, q, backend="opx")
fit = fit_rabi(result.xpts, result.avgi)
print(result.avgi.shape, result.meta["backend"])
print(fit.success, f"{fit.derived['pi_gain']:.3f}")    # half the period
```

<!-- snippet: output -->
```text
(41,) opx
True 0.400
```

A complex S21 trace goes through `from_complex`, which splits it into I and Q. The `analyze_*`
helpers import and fit in one step; `analyze_resonator` fits a Lorentzian to the magnitude
(`method="circle"` fits the full complex response instead):

```python
from strumento.interop.backends import analyze_resonator, from_complex

freqs = np.linspace(7000.0, 7010.0, 201)                    # MHz
s21 = 1 - 0.8 / (1 + 2j * (freqs - 7005.0) / 0.5)
res = from_complex(freqs, s21, backend="vna")
print(np.allclose(res.avgi + 1j * res.avgq, s21))

fit = analyze_resonator(res.xpts, res.avgi, res.avgq)
print(fit.success, f"{fit.derived['f0']:.2f}")
```

<!-- snippet: output -->
```text
True
True 7005.00
```

The other helpers fit the I channel (`avgi`): `analyze_rabi` with `fit_rabi`, `analyze_t1` with
`fit_exp`, `analyze_t2_ramsey` with `fit_decaysin_detuned`, and `analyze_spectroscopy` fits a
Lorentzian to the magnitude, like `analyze_resonator`. Rotate or project the data first if the
signal is not on I.

## Datasets by variable name

`from_xarray` reads the swept coordinate and the two data variables by name. A Quantify dataset
uses `x0`, `y0`, `y1`, which are the defaults; `x_scale` converts units on the way in. A dict of
arrays works the same way, which is what this example uses:

```python
from strumento.interop.backends import from_xarray

dataset = {"x0": freqs * 1e6, "y0": s21.real, "y1": s21.imag}      # Hz, as Quantify stores it
res = from_xarray(dataset, x_scale=1e-6, backend="qblox")           # to MHz
print(f"{res.xpts[0]:.1f}", res.meta["backend"])

try:
    from_xarray(dataset, x="frequency")
except KeyError as err:
    print(err)
```

<!-- snippet: output -->
```text
7000.0 qblox
"from_xarray: x (coordinate) key 'frequency' not found in dataset (available: ['x0', 'y0', 'y1'])"
```

A missing name raises a `KeyError` that lists the names the dataset does have.

## External data in the experiment lifecycle

`external_outcome` returns an `ExperimentOutcome` (`.result` and `.fit`), the object
`Experiment.go()` returns. If the fitter raises, `.fit` is `None` rather than an exception.
`ExternalExperiment` goes one step further: it wraps your acquisition function, a callable taking
the device and returning `(xpts, i, q)` or a `MeasurementResult`, as an `Experiment`. Its `go()`
calls that function instead of compiling and acquiring a program, stamps the environment like
any other run, fits, and, given `record=`, emits the same fit and gate events, so the step can
sit in a bring-up plan. `soc` is accepted for compatibility and never passed to your function:
your script owns its own connection.

```python
from strumento import Device
from strumento.interop.backends import ExternalExperiment

dev = Device.load("devices/multimode_demo/device.yaml")


def my_rabi(device):
    # run the program on the other stack, fetch, and return the arrays
    return gains, i, q


exp = ExternalExperiment(dev, acquire_fn=my_rabi, fit_fn=fit_rabi, backend="opx")
outcome = exp.go()
print(outcome.fit.success, f"{outcome.fit.derived['pi_gain']:.3f}")
print(sorted(outcome.result.meta))
```

<!-- snippet: output -->
```text
True 0.400
['backend', 'environment', 'source']
```

## Optimal-control pulses out

`extract_oc_pulses` takes the same inputs as `strumento.core.pulses.from_solution` (a solution
with `times` in µs and `controls` in MHz Rabi rate, a `drive_map` of
`{drive: (line, carrier_mhz)}`, and a `gain_calib` callable from physical amplitude to gain
fraction) and returns one `WaveformExport` per drive: normalized `idata` / `qdata`, the `gain`,
the carrier, the duration and the line. `absolute_samples()` gives the samples to upload.

```python
from strumento.interop.backends import extract_oc_pulses

t = np.linspace(0.0, 0.1, 11)                                       # µs
solution = {"times": t, "controls": {"qubit:x": 20.0 * np.sin(np.pi * t / 0.1)}}
pulses = extract_oc_pulses(
    solution,
    drive_map={"qubit:x": (("qubit", "drive"), 4000.0)},
    gain_calib=lambda rabi_mhz: rabi_mhz / 80.0,                    # an illustrative calibration
    dac_rate=1000.0,                                                # samples per µs
)
wf = pulses["qubit:x"]
print(wf.n_samples, wf.gain, wf.line, f"{wf.absolute_samples().max():.3f}")
```

<!-- snippet: output -->
```text
101 0.25 ('qubit', 'drive') 0.250
```

## Piccolo convention

A Piccolo transmon solve returns two controls, `u_x` and `u_y`, in GHz on an ns grid, for the
drive

$$H_\text{Piccolo} = 2\pi\,\big[u_x\,(a + a^\dagger) + u_y\, i\,(a - a^\dagger)\big].$$

`from_solution` consumes the hardware convention, $H = \tfrac12(\Omega a^\dagger + \Omega^* a)$
with $\Omega \propto I + iQ$, as $\nu = \Omega/2\pi$ in MHz on a µs grid. Matching the two terms
gives

$$\nu = 2000\,(u_x - i\,u_y)\ \text{MHz}.$$

`strumento.interop.piccolo.piccolo_controls_to_solution(times_ns, controls_ghz, drive=...)` does
this conversion (pure numpy). `controls_ghz` must be `(2, N)` with rows `u_x` and `u_y`, and `N`
must equal the length of `times_ns`. Anything else raises `ValueError`. The constant is
`PICCOLO_GHZ_TO_RABI_MHZ = 2000.0`. There are two traps:

- **The factor of 2.** It comes from the ½ in the hardware convention. A naive `GHz × 1000`
  import plays the pulse at half amplitude.
- **The sign of Q.** Piccolo's $+u_y\, i(a - a^\dagger)$ is $Q = -u_y$ in hardware terms. A naive
  import mirrors a DRAG-like correction, so it increases leakage instead of suppressing it.

The derivation is in the function's docstring. The interop tests check both traps against the two
Hamiltonians, including the time evolution of a pulse with both quadratures non-zero.

```python
from strumento.core.device import Device
from strumento.core.pulses import from_solution
from strumento.interop.piccolo import piccolo_controls_to_solution
from strumento.packs.cqed.calibration import PiPulseReference
from strumento.packs.cqed.library.transmon import TransmonOps

t_ns = np.array([0.0, 50.0, 100.0])                             # Piccolo's grid, ns
u = np.array([[0.003, 0.003, 0.003],                            # u_x, GHz
              [0.001, 0.001, 0.001]])                           # u_y, GHz
sol = piccolo_controls_to_solution(t_ns, u, drive="qubit")
print(sol["times"], sol["controls"]["qubit"][0])

dev = Device.load("devices/multimode_demo/device.yaml")
seq = from_solution(
    sol,
    drive_map={"qubit": (("qubit", "drive"), dev.calib.qubit.f_ge)},
    gain_calib=PiPulseReference(pi_gain_frac=0.25, pi_rabi_mhz=10.0),   # illustrative numbers
    dac_rate=9584.64,                                   # the drive generator's fs, samples per µs
)
(oc,) = seq.ops[0].pulses
ops = TransmonOps(dev, dev.qubit.element, overrides={"pi_ge": oc})
print(oc.start_us, ops.ge_pi().start_us, f"{ops.ge_pi().gain:.3f}")
```

<!-- snippet: output -->
```text
[0.   0.05 0.1 ] (6-2j)
0.0 None 0.158
```

Pass the drive generator's own sample rate as `dac_rate`. The Arb plays its samples at the DAC
rate, so a coarse rate, or `None` with a solver grid that is not the DAC grid, shortens the played
pulse. The import's `meta` (`n_samples`, `span_us`, `dac_rate`) records what was played.

**Placement.** `from_solution` gives each pulse `start_us=0.0`. That is an offset inside its
parallel segment, and inside a segment it is unchanged. When the pulse is installed as a gate
override (`TransmonOps(overrides={"pi_ge": ...})`, or the `overrides=` of `TransmonAllXY` and
`TransmonRB`), the ops drop that offset (`start_us=None`). The override then plays at the cursor
like the calibrated gate it replaces. If the offset were kept, the compiler would place the gate
absolutely and would not advance the cursor. Both gates of an AllXY pair and the readout would
then stack at the pair's start, and the only symptom would be qick's logged "pulse time 0.0
appears to conflict" (see [timeline.md](timeline.md)).

## See also

- [cqed-cookbook.md](cqed-cookbook.md): the fitters and experiments these results feed.
- [fitters.md](fitters.md): every fitter and what it reports.
- [task-record.md](task-record.md): the record `ExternalExperiment.go(record=...)` writes to.
