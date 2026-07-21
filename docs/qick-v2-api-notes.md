# qick tProc-v2 API notes (Task 0, spec R2/D25)

Verified against **qick 0.2.291** (installed to match the board's soccfg `sw_version`, so
`QickConfig` builds with no version-mismatch warning). Confirms Sho Uemura's PR #5 review and
his `meeg/qick_demos_sho/tprocv2/00_intro.ipynb`. Everything below runs **offline, no board**,
from the committed `devices/schuster_multimode/soccfg_snapshot.json`.

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
