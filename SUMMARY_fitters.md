# Fitter work — resonator/oscillator fitters (Dave Schuster, 2026-07-21 meeting)

Branch: `wip/fitters` (off clean `main`). **Not pushed, not merged, no PR** — per instructions.

Implements Dave's fitter requirements: a general lmfit primitive + **typed wrappers** returning the
existing `FitResult` seam (`params` / `stderr` / `derived` / `success`), matching
`core/fitters.py::fit_rabi` exactly. The headline is the resonator/oscillator fits with real-world
non-idealities; the "boring" T1/T2/Rabi family is one damped-oscillation model with different
post-processing.

## What was built

### Oscillator family — `strumento/core/fitters.py` (extended)
One shared damped-cosine model (`_damped_sin`, phase-quadrant multi-start) + one shared exponential
model; the wrappers differ only in the physics their `derived` fields expose (Dave's "one fit,
different post-processing" principle). Shared helpers `_params_stderr`, `_seed_freq` added.

| Fitter | Model | `derived` fields |
|---|---|---|
| `fit_rabi` (pre-existing, untouched) | decaying cosine of drive amplitude | `period`, `pi_gain` |
| `fit_exp` (T1) | `y0 + amp*exp(-x/tau)` (handles decay **and** rising) | `T1`, `tau` |
| `fit_decaysin_detuned` (T2 Ramsey) | `offset + amp*exp(-x/tau)*cos(2*pi*f*x + phi)` | `detuning`, `freq`, `T2`, `tau` |
| `fit_damped_rabi` (time-domain Rabi) | same damped cosine | `rabi_freq`, `period`, `pi_time`, `half_pi_time`, `tau` |

`fit_exp` and `fit_decaysin_detuned` are exactly the slice-3 fitters the plan (Task 12) earmarked;
`fit_damped_rabi` rounds out the damped-sinusoid family.

### Resonator family — `strumento/core/resonator_fitters.py` (new module)
Pure, no qick, no plotting/stdout (D13). Imports `FitResult` from `core/fitters.py`.

| Fitter | Method | `derived` fields |
|---|---|---|
| `fit_lorentzian` | magnitude Lorentzian, signed-amp seed (peak **or** dip) | `f0`, `hwhm`, `fwhm`, `Q_loaded` |
| `fit_resonator_circle` | **Probst/Khalil circle fit** of complex S21/S11; `geometry="notch"` (hanger/transmission) or `"reflection"` | `f0`, `Q_loaded`, `Q_internal`, `Q_coupling`, `phi`, `fwhm` (plus `params`: `a`, `alpha`, `delay`, `theta0`) |
| `fit_fano` | asymmetric Fano ("Spano") lineshape, q multi-start | `f0`, `gamma`, `fwhm`, `q_fano`, `Q_loaded` |

`fit_resonator_circle` handles the non-idealities Dave called out: **cable delay** (auto-estimated),
**environmental gain + IQ-plane rotation/offset** (the off-resonant point -> `a`, `alpha`), and
**impedance mismatch / asymmetry** (`phi`, Khalil diameter correction -> `Q_internal` vs `Q_coupling`).

Circle-fit pipeline (Probst 2015): remove cable delay -> algebraic (Kasa) circle fit -> phase-vs-
frequency tangent fit for `f0` + `Q_loaded` -> off-resonant point gives env gain/rotation ->
normalised diameter + mismatch angle give `Q_coupling` and `Q_internal` (`1/Q_l = 1/Q_i + Re(1/Q_c)`).

## Methods vendored (no new dependency — Dave's guidance)

Re-implemented in-repo from published descriptions; only numpy/scipy/lmfit (already deps) used:
- **Circle fit**: S. Probst et al., Rev. Sci. Instrum. 86, 024706 (2015); diameter/impedance-
  mismatch correction from M. S. Khalil et al., J. Appl. Phys. 111, 054510 (2012).
- **Lorentzian / Fano**: standard textbook lineshapes (Lorentzian also mirrors v1 `lorfunc`).

No external resonator-fit package (e.g. `resonator_tools`, `scresonators`) was added.

## Tests (TDD — synthetic-recovery tests written first)

- `tests/core/test_fitters.py` — extended with T1 (decay + rising), T2 Ramsey (recovers detuning +
  T2), damped-Rabi (freq + pi-time) synthetic-recovery tests.
- `tests/core/test_resonator_fitters.py` (new) — synthetic S21/S11 with known params + noise:
  Lorentzian dip/peak; circle fit notch (Q_l/Q_i/Q_c), circle fit with env gain + rotation + cable
  delay + asymmetry, circle fit reflection; Fano asymmetric + large-q limit.
- `tests/oracle/test_fitters_vs_v1.py` (new, `oracle` marker) — cross-checks new fitters vs the
  vendored v1 `multimode_expts/fitting/fitting.py` (`fitexp`, `fitdecaysin`, `fitlor`) on identical
  data. `.refs-v1/` **is present**, so the oracle tests run (loaded via `.refs-v1/multimode_expts`,
  overridable with `$STRUMENTO_V1_FITTING`); they skip gracefully if it is ever absent.

### Passing pytest summary

```
$ python -m pytest tests/core/test_fitters.py tests/core/test_resonator_fitters.py tests/oracle/ -v
tests/core/test_fitters.py::test_fit_rabi_recovers_period_and_pi_gain PASSED
tests/core/test_fitters.py::test_fit_rabi_exposes_params_and_stderr PASSED
tests/core/test_fitters.py::test_fit_exp_recovers_T1 PASSED
tests/core/test_fitters.py::test_fit_exp_handles_rising_exponential PASSED
tests/core/test_fitters.py::test_fit_decaysin_detuned_recovers_detuning_and_T2 PASSED
tests/core/test_fitters.py::test_fit_damped_rabi_recovers_freq_and_pi_time PASSED
tests/core/test_resonator_fitters.py::test_fit_lorentzian_recovers_f0_and_Q_dip PASSED
tests/core/test_resonator_fitters.py::test_fit_lorentzian_recovers_peak PASSED
tests/core/test_resonator_fitters.py::test_fit_resonator_circle_notch_recovers_Qs PASSED
tests/core/test_resonator_fitters.py::test_fit_resonator_circle_handles_env_and_delay_and_asymmetry PASSED
tests/core/test_resonator_fitters.py::test_fit_resonator_circle_reflection_recovers_Qs PASSED
tests/core/test_resonator_fitters.py::test_fit_fano_recovers_asymmetric_lineshape PASSED
tests/core/test_resonator_fitters.py::test_fit_fano_reduces_to_lorentzian_for_large_q PASSED
tests/oracle/test_fitters_vs_v1.py::test_exp_agrees_with_v1_fitexp PASSED
tests/oracle/test_fitters_vs_v1.py::test_decaysin_agrees_with_v1_fitdecaysin PASSED
tests/oracle/test_fitters_vs_v1.py::test_lorentzian_agrees_with_v1_fitlor PASSED
16 passed in 1.00s

$ python -m pytest -q       # whole suite
125 passed, 3 skipped in 2.04s   # 3 skips are the tProc-v2/qick-gated tests (unchanged)
```

## Key decisions

- **New module `core/resonator_fitters.py`** rather than piling into `fitters.py`: the resonator fits
  are a distinct, heavier family (circle geometry, complex data) — cleaner module boundary. It reuses
  `FitResult` and `_params_stderr` from `fitters.py`, so the typed seam is identical.
- **Circle-fit delay estimation was the one real trap.** The extracted Q is sensitive to the
  *differential* phase ramp `2*pi*span*tau`, so tau must be pinned to ~`1e-8` s; `scipy.minimize_scalar`'s
  default `xatol=1e-5` is enormous at that scale and silently never refined (caused a systematic ~6 %
  Q bias even on noiseless data). Fixed by searching in the **dimensionless** variable `beta = tau*span`
  (~ fringes across the band) with a global grid search + high-precision local refine of the circle
  residual (which is exactly zero at the true delay). Q now recovers to <1 % on clean data.
- **Q-factor conventions**: `Q_coupling` is the magnitude `|Q_c|`; notch uses `|Q_l/Q_c| = diameter`,
  reflection carries the factor of 2 (`|2*Q_l/Q_c| = diameter`); `Q_internal` from
  `1/Q_l = 1/Q_i + cos(phi)/|Q_c|` (Khalil). Self-consistent with `1/Q_l = 1/Q_i + 1/Q_c` at `phi=0`.

## Blockers / notes

- **No blockers.** `.refs-v1/` oracle is present; oracle tests run and pass. `ruff`/`mypy` are not
  installed in the `strumento` conda env (dev extra), so lint was checked manually (all new lines
  <=100 chars per `[tool.ruff] line-length`); worth a `pip install -e ".[dev]"` + `ruff` pass in CI.
- Wiring these into `packs/cqed/experiments/` (t1, t2_ramsey, resonator_spec Analyzers, Task 12/17)
  is out of scope here — this delivers the pure fitter primitives + tests only.
