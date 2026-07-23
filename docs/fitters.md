# The fitter library

Typed, vendored, lmfit-based. Every fitter takes `(xpts, y)` arrays and returns a
`FitResult{params, stderr, derived, success}` — no plotting, no printing; derived physical
quantities are fields (D4/D13). The CLI exposes all of them: `strumento fit <kind> <csv>`.

## The headline set — resonators (`core/resonator_fitters.py`)

Resonator/oscillator spectroscopy is where fitting earns its keep, because real traces carry
non-idealities: cable delay winding the IQ plane, Fano interference from parasitic paths,
rotation/offset from the measurement chain. The library treats those as first-class, and the
methods are the published ones:

| fitter | model | derived |
|---|---|---|
| `fit_lorentzian(freq, mag)` | Lorentzian dip/peak in magnitude | `f0`, `fwhm`, `Q` |
| `fit_fano(freq, mag)` | Fano lineshape (asymmetry q) | `f0`, `q`, width |
| `fit_resonator_circle(freq, s21)` | full complex circle fit — cable-delay removal, IQ rotation/offset, Fano asymmetry; notch or reflection geometry | `f0`, `Q_loaded`, `Q_internal`, `Q_coupling`, delay |

The circle fit follows Probst *et al.*, Rev. Sci. Instrum. **86**, 024706 (2015) and Khalil
*et al.*, J. Appl. Phys. **111**, 054510 (2012). Implementation note: the cable-delay search is
performed in the dimensionless β = τ·span variable — searching τ directly with a generic
minimizer's default tolerance biases Q by several percent on clean data.

**Vendored deliberately.** Resonator fits change rarely; carrying them in-tree costs little and
removes a whole class of downstream breakage from an external fitting dependency.

## The damped-sinusoid family (`core/fitters.py`)

One general primitive, many typed wrappers — the "boring" family is a single fit with different
parameter meanings and preconditioning:

| fitter | for | derived |
|---|---|---|
| `fit_rabi` | amplitude/time Rabi | `period`, `pi_gain`, `hpi_gain` |
| `fit_damped_rabi` | Rabi with visible decay | + decay `tau` |
| `fit_exp` | T1 relaxation | `T1` |
| `fit_decaysin_detuned` | T2 Ramsey | `T2`, `detuning` (the fit frequency IS the drive detuning) |
| `fit_parity` | parity fringe (phase scan) | `contrast`, `phase_deg` |
| `fit_double_gaussian` | single-shot readout histograms | `threshold`, `separation`, `mu_low/high` |

## Where fitters meet experiments

Analyzers wire them into the lifecycle: `AmplitudeRabi→fit_rabi`, `T1→fit_exp`,
`T2Ramsey→fit_decaysin_detuned`, `ParityFringe→fit_parity`,
`ResonatorSpectroscopy→fit_lorentzian|fit_resonator_circle`, `PSBHistogram→fit_double_gaussian`.

Adding one: lmfit model → typed `FitResult` → synthetic-data recovery test (5–10% tolerance is
typical) → wire into an Analyzer → add to the CLI `fit` map.
