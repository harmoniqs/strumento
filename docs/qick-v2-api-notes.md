# qick tProc-v2 API notes (D25)

Last verified against **qick 0.2.418**. Strumento accepts any `qick>=0.2.367,<0.3` (the `[qick]`
extra), so yours may differ; check it with `qick.__version__`. Cross-checked against the upstream
[qick tProc-v2 demo notebooks](https://github.com/openquantumhardware/qick). Everything below runs
**offline, no board**, from the committed `devices/_fixtures/soccfg_v1_zcu216.json`.

> **Version skew is real and expected.** These notes were first written against 0.2.291; the
> feed-forward section at the bottom was re-verified against 0.2.418. The committed soccfg
> fixtures were captured under qick 0.2.367, so with any other installed version `QickConfig`
> logs a one-line `QICK library version mismatch` notice — the experiment-contract's diagnostics
> guard partitions that off as an environment notice (not a program defect) and surfaces it via
> `test_the_fixture_and_library_versions_are_reported`. qick's control-flow API is not frozen:
> re-check signatures here against `qick.__version__` before trusting them.

## Offline config

```python
from qick import QickConfig
soccfg = QickConfig(soccfg_dict)   # a DICT (a str arg is treated as a file path!)
soccfg['gens'], soccfg['readouts'], soccfg['tprocs']   # __getitem__ -> raw cfg
soccfg.freq2reg(f_mhz, gen_ch, ro_ch)  # unit helpers
soccfg.us2cycles(us, gen_ch, ro_ch);  soccfg.cycles2us(cyc, gen_ch, ro_ch)
soccfg.deg2reg(deg, gen_ch, ro_ch)
```

Gen record fields we use: `tproc_ch`, `dac`, `fs`, `maxv` (=32766 on this board),
`samps_per_clk`, `has_mixer`, `f_dds`, `type` (`axis_signal_gen_v6` full gens; the readout mux
DAC is `axis_sg_mux4_v2`).

## Program base — `qick.asm_v2.AveragerProgramV2`

Subclass it; override `_initialize(self, cfg)` and `_body(self, cfg)`. **Do not** append to a
bare program object (that idiom is `QickProgramV2`).

```
AveragerProgramV2(soccfg, reps, final_delay, final_wait=0, initial_delay=1.0,
                  reps_innermost=False, before_reps=None, after_reps=None, cfg=None)
```

Build methods (all confirmed present):
- `declare_gen(ch, nqz=1, mixer_freq=None, mux_freqs=None, mux_gains=None, mux_phases=None, ro_ch=None)`
- `declare_readout(ch, length, freq=None, phase=0, sel='product', gen_ch=None)`
- `add_gauss(ch, name, sigma, length, maxv=None, even_length=False)`
- `add_pulse(ch, name, **kwargs)`  (style/freq/phase/gain/envelope, per mode)
- `add_loop(name, count, ...)` + `QickSweep1D(name, start, stop)`  (sweeps)
- `pulse(ch, name, t=0)`, `delay(t)`, `delay_auto(t=0, gens=True, ros=True)`
- `trigger(ros=None, pins=None, t=0, width=None)`
- feed-forward + loops (`read_and_jump`, `read_input`, `cond_jump`, `label`/`jump`,
  `open_loop`/`close_loop`) and the blocking `wait`/`wait_auto` — see the dedicated section at the
  bottom (the `delay`-vs-`wait` distinction is load-bearing for active reset).

## acquire() contract — what the mock soc must satisfy

`acquire(soc, soft_avgs=1, load_pulses=True, start_src='internal', threshold=None, angle=None, ...)`

Returns averaged IQ, shape `(n_ch, n_reads, 2)` (or `(n_ch, n_reads, n_expts, 2)` with sweeps).
Internally builds `d_buf[ii]` of shape `(*loop_dims, nreads, 2)` int64 and fills it from
`poll_data()`.

**Full `soc.*` surface the offline path calls** (from `config_all` / `load_pulses` /
`config_gens` / `config_readouts` / `config_bufs` / the acquire loop):

```
load_bin_program, stop_tproc, start_src, load_pulse_data,
set_nyquist, set_mixer_freq, config_mux_gen,
configure_readout, config_mux_readout,
config_avg, config_buf,
reload_mem, start_readout, poll_data
```

All are no-ops for a mock **except**:
- `start_readout(total_count, counter_addr, ch_list, reads_per_shot)` — capture `total_count`
  and `reads_per_shot`.
- `poll_data()` -> `[(new_points, (d, stats))]` where `d[ii]` is int `(new_points*nreads, 2)`.
  The mock returns everything in one chunk: `[(total_count, ([zeros((total_count*nr,2), int) for
  nr in reads_per_shot], None))]`.

So `MockQickSocV2(QickConfig)` = a real config (declare/add_pulse/compile validate for real) +
these stubbed leaf methods. This makes the numeric-golden gate validate **actual v2 assembly**,
not just shapes.
```
```

## Feed-forward and control flow — the `Branch` / `Repeat` seam (verified 0.2.418)

Active reset and every mid-circuit feed-forward / QEC sequence lower through these. They are **not**
on the demo-notebook happy path, and the `delay`-vs-`wait` distinction below is a genuine trap — it
is what made an early active-reset port read a *stale* accumulator.

**Control flow** — untimed: the tProc executes these at its *execution* time, as fast as it can, not
on the timed pulse queue.
- `label(name)` / `jump(name)` — an unconditional jump target and jump.
- `cond_jump(label, arg1, test, op=None, arg2=None)` — test a register, jump if it passes.
  `test` ∈ `1/0` (always/never), `Z/NZ` (==0/!=0), `S/NS` (<0/>=0), `F/NF` (external flag).
- `read_input(ro_ch)` — copy an accumulated I/Q from a tProc input into `s_port_l`/`s_port_h`.
  **"The readout must have already pushed the value into the input, otherwise you will get a stale
  value."** (verbatim from the docstring — this is the whole hazard.)
- `read_and_jump(ro_ch, component, threshold, test, label)` — `read_input` + `cond_jump` in one.
  `component` ∈ `{"I","Q"}` (→ `s_port_l`/`s_port_h`); `test` ∈ `{">=","<"}` (→ `NS`/`S`);
  `threshold` is a **24-bit signed int** accumulator value (or a register name). It **cannot
  rotate** — it compares one raw quadrature — so the discrimination angle must be folded into the
  readout DDC phase upstream. strumento lowers `Branch` to this (and realizes a float threshold
  half-away-from-zero — `core/compiler.py:_realize_threshold`).
- `open_loop(n, name=None)` / `close_loop()` — a bounded hardware loop; strumento lowers `Repeat`
  to this. Under `AveragerProgramV2`, use `add_loop` for *swept* (measured) axes instead.

**Timeline — `delay*` moves the reference time; `wait*` blocks the processor. They are different:**

| call | effect | blocks the tProc? |
| --- | --- | --- |
| `delay(t)` / `delay_auto(t=0, gens=True, ros=True)` | increment the **reference time** (pushes later *timed* ops — pulses/triggers — further out) | **no** |
| `wait(t)` / `wait_auto(t=0, gens=False, ros=True)` | **pause tProc execution** until the time is reached (rel. to reference / to the last pulse/readout end) | **yes** |
| `resync(t=0.05)` | bump reference to exceed execution time by ≥ `t` (recover slack after an unknown-length wait) | no |

**The trap.** `read_input`/`read_and_jump` run at execution time, *off* the timed queue. A `delay`
before them does nothing to guarantee the readout has completed — it only shifts where later *timed*
ops land, and qick's `Wait` macro is the only thing that stalls the processor (its `Delay` macro
does not touch the timestamps). So without a blocking wait the read is **stale**. You must emit
`wait_auto(t, ros=True)` (or `wait`) after the readout trigger and before the read — the v2
equivalent of the tProc-v1 idiom of a `wait_all` before every `read`/`condj`.
strumento does this in the `Branch` lowering (`compiler.py:_play_branch`), keyed to the branch's
`settle_us`; a `wait_auto(ros=True)` with no readout in scope compiles to nothing, so it is safe to
emit unconditionally.
