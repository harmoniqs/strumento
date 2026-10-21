# Test fixtures — soccfg snapshots

A **soccfg snapshot** is the JSON produced by `QickSoc.dump_cfg()` on a board; rebuilt offline via
`QickConfig(json)`, it drives unit conversions, `maxv` gain math, and — for tProc-v2 configs —
full `AveragerProgramV2` assembly with **no board attached**.

## `soccfg_v1_zcu216.json` (committed)

A tProc-**v1** (`axis_tproc64x32_x8`) ZCU216 config. Used by the pure unit math (`quantize`,
`maxv` gain conversion) and the v1-era mock tests. It **cannot** drive v2 assembly.
The `v1_snapshot_yaml` test fixture (`tests/conftest.py`) attaches it to a temporary copy of
`multimode_demo`, so tests can exercise the v1-snapshot rejection paths (the simulator and the
MCP session refuse it with a typed error).

## `soccfg_v2_testbench.json` (committed)

The official qick **testbench** firmware config (from
[`openquantumhardware/qick`](https://github.com/openquantumhardware/qick)
`firmware/testbench/qick_testbench/soccfg.json`, MIT license): ZCU216, `qick_processor`
(tProc **v2**), **one** full-speed generator + **one** dynamic readout. It is a simulation
firmware's config, not a lab board's — single-channel only — but it is a *real* v2 config, so
real v2 program assembly runs offline and in CI against it (see `devices/loopback_demo`).

## `soccfg_v2_cqed.json` (committed, fabricated)

A **fabricated** tProc-v2 config for `devices/multimode_demo`: six full-speed generators (0–5)
and **one** dynamic readout, laid out on the testbench pattern. Like the testbench, its tProc has
a single input port (`in_port_qty: 1`), so the readout's `tproc_ch` (the tProc input it feeds) is
`0`; its `tproc_ctrl` (the output that configures it) sits after the generators, at `6`. A
program addresses the readout by its index in `readouts` (`ro_ch` 0, the device's `adc_chs: [0]`).
`strumento check devices/multimode_demo/device.yaml` passes against it.

## `soccfg_v2_spin.json` (committed, fabricated)

A **fabricated** tProc-v2 config for `devices/spin_demo`: four full-speed generators (0–3) and
**one** dynamic readout, on the same testbench pattern. Its tProc likewise has a single input port
(`in_port_qty: 1`), so the readout's `tproc_ch` is `0` (it was fabricated as `4`, an input the
tProc does not have — the same defect fixed in the cqed fixture); its `tproc_ctrl` sits after the
generators, at `4`. The device addresses the readout as `ro_ch` 0 (`adc_chs: [0]`).
`strumento check devices/spin_demo/device.yaml` passes against it. The other spin demos
(`spin_eo_demo`, `spin_hole_demo`, `spin_st_demo`) do not attach a snapshot yet.

## `soccfg_v2.json` (not committed — drop yours here)

The v2 gate (`tests/_v2gate.py`) prefers a local `soccfg_v2.json` over the testbench snapshot.
Drop a `dump_cfg()` JSON from **your** tProc-v2 board here and the v2-gated tests (real assembly +
numeric goldens) run against your firmware:

- `tests/core/test_program.py::test_construct_and_acquire_returns_measurement_result`
- `tests/e2e/test_loopback_vertical.py::test_loopback_full_vertical_on_mock`

Everything else (pure logic + `quantize` unit math) runs against the committed snapshots.
