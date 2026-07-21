# Test fixtures

## `soccfg_v2.json` (not committed — drop it here to enable v2 tests)

Several tests build a real `AveragerProgramV2`, which requires a **tProc-v2** (`qick_processor`)
soccfg. The committed `devices/schuster_multimode/soccfg_snapshot.json` is a tProc-**v1**
(`axis_tproc64x32_x8`) config from the Schuster multimode board, so it cannot drive v2 assembly.

Those tests are skip-gated on the presence of a v2 config at:

```
devices/_fixtures/soccfg_v2.json
```

To enable them, drop a `QickConfig.dump_cfg()` JSON from any tProc-v2 board here (e.g. Sho
Uemura's board — the `dump_cfg()` the Slack request asks for). The gate lives in
`tests/_v2gate.py`; the tests it guards:

- `tests/core/test_program.py::test_construct_and_acquire_returns_measurement_result`
- `tests/e2e/test_slice1_skeleton.py::test_slice1_full_vertical_on_mock`

Everything else (pure logic + `quantize` unit math) runs against the v1 snapshot.
