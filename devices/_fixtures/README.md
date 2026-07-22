# Test fixtures — soccfg snapshots

A **soccfg snapshot** is the JSON produced by `QickSoc.dump_cfg()` on a board; rebuilt offline via
`QickConfig(json)`, it drives unit conversions, `maxv` gain math, and — for tProc-v2 configs —
full `AveragerProgramV2` assembly with **no board attached**.

## `soccfg_v1_zcu216.json` (committed)

A tProc-**v1** (`axis_tproc64x32_x8`) ZCU216 config. Used by the pure unit math (`quantize`,
`maxv` gain conversion) and the v1-era mock tests. It **cannot** drive v2 assembly.

## `soccfg_v2_testbench.json` (committed)

The official qick **testbench** firmware config (from
[`openquantumhardware/qick`](https://github.com/openquantumhardware/qick)
`firmware/testbench/qick_testbench/soccfg.json`, MIT license): ZCU216, `qick_processor`
(tProc **v2**), **one** full-speed generator + **one** dynamic readout. It is a simulation
firmware's config, not a lab board's — single-channel only — but it is a *real* v2 config, so
real v2 program assembly runs offline and in CI against it (see `devices/loopback_demo`).

## `soccfg_v2.json` (not committed — drop yours here)

The v2 gate (`tests/_v2gate.py`) prefers a local `soccfg_v2.json` over the testbench snapshot.
Drop a `dump_cfg()` JSON from **your** tProc-v2 board here and the v2-gated tests (real assembly +
numeric goldens) run against your firmware:

- `tests/core/test_program.py::test_construct_and_acquire_returns_measurement_result`
- `tests/e2e/test_slice1_skeleton.py::test_slice1_full_vertical_on_mock`

Everything else (pure logic + `quantize` unit math) runs against the committed snapshots.
