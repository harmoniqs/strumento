# Fallback diagnostic scripts

Bare `AveragerProgramV2` programs that bypass strumento entirely. Use them to
verify the signal chain (DAC output, ADC capture, frequency math, timing) before
trusting any higher-level abstraction on top.

## Purpose

These scripts are the **day-one hardware bring-up toolkit**. When the board
powers on for the first time, you run them in order to answer:

1. **ADC alive?** — `adc_trace.py` arms the readout and captures IQ. On a real
   board, even an open input shows a noise floor; zeros mean the acquisition
   pipeline is broken.
2. **DAC → ADC path?** — `dac_loopback.py` plays a constant tone on a generator
   and reads it back on the ADC (if the board supports loopback or with a cable).
   The carrier frequency and amplitude should match what you programmed.
3. **Resonator?** — `resonator_spec.py` sweeps a probe tone across a readout
   resonator. A clear dip means the resonator is coupled and the signal chain is
   calibrated enough to proceed.
4. **Qubit response?** — `rabi.py` sweeps the drive amplitude and looks for
   oscillation. If you see a Rabi oscillation, the qubit is alive.
5. **Coherence?** — `t1.py` and `t2_ramsey.py` measure energy relaxation and
   dephasing. If the fits converge, you have a working qubit.

## Usage

Each script is self-contained. It needs `qick` installed but does **not** import
`strumento`. It loads a soccfg from a JSON snapshot for offline assembly testing,
or uses a live board when available.

```bash
# Offline: verify the program assembles (uses MockQickSocV2 if strumento is installed)
python adc_trace.py --soccfg ../../devices/_fixtures/soccfg_v2_testbench.json

# Live board: pass a soc object in an interactive session, or omit --soccfg to
# use the board's own config (requires the board to be connected)
```

### Common CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--soccfg PATH` | testbench fixture | Path to a `dump_cfg()` JSON snapshot |
| `--gen-ch N` | 0 | Generator (DAC) channel index |
| `--ro-ch N` | 0 | Readout (ADC) channel index |
| `--reps N` | 100 | Hardware averages per point |
| `--plot` | off | Show a matplotlib figure after acquire |

Each script adds experiment-specific flags (frequencies, gains, sweep ranges).
Run `python <script>.py --help` for the full list.

## Offline testing

With `strumento` installed, the scripts use `MockQickSocV2` to run the full
acquire path (program builds, compiles, acquires with zero-valued IQ). This
verifies:

- The program assembles without errors (declare_gen, add_pulse, add_gauss)
- The sweep loop structure is correct (add_loop, QickSweep1D)
- The acquisition dimensions are right (reps × sweep points × readout channels)

Without `strumento`, the scripts fall back to `qick.QickConfig` for
build-and-compile-only testing (no acquire).

## What these scripts are NOT

- Not a replacement for strumento's experiment framework. Once the signal chain
  is verified, switch to strumento for proper calibration workflows.
- Not fitted. These scripts do not fit the data — they produce raw I/Q traces
  for eyeball verification. Fitting is strumento's job.
- Not safe for unattended operation. No power guards, no frequency sanity checks.
  These are lab-bench scripts for a human at the controls.
