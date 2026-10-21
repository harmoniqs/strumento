# The fitter library

Typed, vendored, lmfit-based. Every fitter takes `(xpts, y)` arrays and returns a
`FitResult{params, stderr, derived, success}` — no plotting, no printing; derived physical
quantities are fields (D4/D13). The CLI exposes all of them: `strumento fit <kind> <csv>`.

`derived` maps names to JSON values (D38). A fitter's physical quantities are numbers; an
experiment's analyzer may add other JSON values beside them — `CompositePulseCalibration`'s
`protocol` string, `ReadoutState`'s `confusion_matrix` list, a refusal's provenance. Code that needs
a number reads a key it knows is numeric, and `strumento schema fit` publishes exactly this shape.

## Errors on derived quantities: `derived["<name>_stderr"]`

Every derived key a core fitter computes from fitted parameters comes with its 1σ error under
`derived["<name>_stderr"]` — the same convention the experiment analyzers use (DRAG's
`alpha_opt_stderr`, error amplification's `contrast_stderr`). So `fit_rabi` reports
`pi_gain` **and** `pi_gain_stderr` (= `stderr["period"] / 2`); you never propagate by hand.

- **Linear propagation over the fit covariance.** The error is `sqrt(gᵀ·C·g)`, with `g` the
  gradient of the derived formula at the fitted values and `C` lmfit's parameter covariance,
  so a quantity built from several correlated parameters (a Lorentzian `Q_loaded = f0/fwhm`,
  a histogram `threshold`, a time-of-flight `settled_level = offset + amp`) gets the
  correlation, not a naive quadrature sum. `|x|` propagates with slope `sign(x)`.
- **Aliases mirror their twin.** `tau` next to `T1`, `freq` next to `detuning`, `t0` next to
  `tof_us`, and the signed `chi_estimate` next to `chi` carry the same error.
- **No error is `nan`, never missing and never zero.** When lmfit cannot estimate a
  covariance (a flat or degenerate trace), every `*_stderr` key is still present, set to `nan`.
- **Exempt keys are present-with-`nan`.** One fitter has derived values with no covariance to
  propagate: `fit_resonator_circle`'s `Q_internal`, `Q_coupling` and `phi` also depend on the
  algebraic (Kåsa) circle fit and the off-resonant-point geometry, which are not lmfit fits. Their
  `*_stderr` keys exist and are always `nan`; its `f0`, `Q_loaded` and `fwhm` (from the lmfit
  phase fit) carry real errors.
- **The errors are honest 1σ.** A Monte Carlo test over 200 seeds checks that
  `|estimate − truth| ≤ stderr` holds at the Gaussian rate (~68%) for `fit_rabi`'s `pi_gain`
  and `fit_exp`'s `T1` — the band 55–80%, observed about 65–75% across five disjoint seed lists. Like any
  linearized error, it understates the spread for a fit that is barely constrained (few
  points per period, a decay much longer or shorter than the window).

> **The `xpts` you fit against is the *realized* axis (D37).** A result's `xpts` is read back
> from the compiled program — the register-rounded per-step values the tProc *actually plays*
> (quantized loop step sizes included) — so a fitted π-gain, T₁, or detuning is the value at
> the played points, not the requested ones. The requested axis is preserved at
> `meta["requested_xpts"]` if you need it for comparison.

## The headline set — resonators (`core/resonator_fitters.py`)

Resonator/oscillator spectroscopy is where fitting earns its keep, because real traces carry
non-idealities: cable delay winding the IQ plane, Fano interference from parasitic paths,
rotation/offset from the measurement chain. The library treats those as first-class, and the
methods are the published ones:

Every derived key below also has its `<name>_stderr` (see above); † marks the exempt keys,
whose error is always `nan`.

| fitter | model | derived |
|---|---|---|
| `fit_lorentzian(freq, mag)` | Lorentzian dip/peak in magnitude | `f0`, `hwhm`, `fwhm`, `Q_loaded` |
| `fit_fano(freq, mag)` | Fano lineshape (asymmetry q) | `f0`, `gamma`, `fwhm`, `q_fano`, `Q_loaded` |
| `fit_resonator_circle(freq, s21)` | full complex circle fit — cable-delay removal, IQ rotation/offset, Fano asymmetry; notch or reflection geometry | `f0`, `Q_loaded`, `fwhm`, `Q_internal`†, `Q_coupling`†, `phi`† (cable delay is `params["delay"]`) |

`fit_fano` tries three starting values of `q` and keeps the best fit. If all three fail, it
returns `success=False` with empty `params`, `stderr` and `derived` instead of raising.
`fit_resonator_circle` returns the same failed fit when the data cannot support a circle: any
NaN or infinite sample in the frequencies or the S21 trace, a trace with no loop (all-zero or
constant data, i.e. a spread about its mean within √ε ≈ 1.5e-8 of its own magnitude, the
resolution floor of the algebraic circle fit), a zero-radius circle, or both phase-fit starts
failing.

The circle fit follows Probst *et al.*, Rev. Sci. Instrum. **86**, 024706 (2015) and Khalil
*et al.*, J. Appl. Phys. **111**, 054510 (2012). Implementation note: the cable-delay search is
performed in the dimensionless β = τ·span variable — searching τ directly with a generic
minimizer's default tolerance biases Q by several percent on clean data.

**Vendored deliberately.** Resonator fits change rarely; carrying them in-tree costs little and
removes a whole class of downstream breakage from an external fitting dependency.

## The damped-sinusoid family (`core/fitters.py`)

One general primitive, many typed wrappers — the "boring" family is a single fit with different
parameter meanings and preconditioning. Every derived key below also has its
`<name>_stderr`; none of these fitters has an exempt key.

| fitter | for | derived |
|---|---|---|
| `fit_rabi` | amplitude/time Rabi | `period`, `pi_gain` |
| `fit_damped_rabi` | Rabi with visible decay | `rabi_freq`, `period`, `pi_time`, `half_pi_time`, decay `tau` |
| `fit_exp` | T1 relaxation | `T1` (alias `tau`) |
| `fit_decaysin_detuned` | T2 Ramsey | `T2` (alias `tau`), `detuning` (alias `freq` — the fit frequency IS the drive detuning) |
| `fit_parity` | parity fringe (phase scan) | `contrast`, `phase_deg` |
| `fit_gaussian_wigner` | displaced-vacuum parity vs gain | `alpha_per_gain`, `contrast` |
| `fit_number_splitting_comb` | photon-number-split qubit spectrum | `chi` (signed alias `chi_estimate`), `nbar`, `gamma` |
| `fit_double_gaussian` | single-shot readout histograms | `threshold`, `separation`, `mu_low/high` |
| `fit_time_of_flight` | raw-trace arrival edge | `tof_us` (alias `t0`), `rise_us`, `noise_floor`, `settled_level` |

## Where fitters meet experiments

The mapping from analyzer to fitter, across both packs:

| Fitter | cqed experiments | spin experiments |
|---|---|---|
| `fit_rabi` | AmplitudeRabi, EFRabi, SidebandRabi, SwapRabi | SpinRabi, ExchangeOscillation, ExchangeGateCalibration |
| `fit_damped_rabi` | LengthRabi | — |
| `fit_exp` | T1, T2Echo, CavityT1, StorageT1, TransmonRB | SpinT1, HahnEcho, LeakageBenchmark |
| `fit_decaysin_detuned` | T2Ramsey, CavityT2Ramsey, RevivalTimeCalibration, CrossKerrMeasurement | SpinRamsey, CPMG, GradientCalibration, GFactorMap, SweetSpotTune, ChargeNoiseSpectroscopy |
| `fit_parity` | ParityFringe, DisplacementCalibration | — |
| `fit_lorentzian` | ResonatorSpectroscopy, QubitSpectroscopy, EFSpectroscopy, CavitySpectroscopy, DispersiveShift, NumberSplitting, FrequencyRabi, SidebandSpectroscopy, SwapSpectroscopy | EDSRSpectroscopy |
| `fit_resonator_circle` | ResonatorSpectroscopy (method="circle") | — |
| `fit_double_gaussian` | — | PSBHistogram |
| manual `FitResult` | ActiveReset, TransmonAllXY, SingleShotReadout, PowerSweep, ReadoutOptimization, StatePrepFidelity, WignerTomography, DRAGCalibration, MultiphotonLadder, PhotonNumberDistribution, QNDReadout | AllXY, CompositePulseCalibration, HeraldedInit |

Adding one: lmfit model → typed `FitResult` → a `<name>_stderr` for each derived key (the
`_derived_stderr` helper takes the gradients) → synthetic-data recovery test (5–10% tolerance
is typical) plus an independent check of the errors → wire into an Analyzer → add to the CLI
`fit` map.
