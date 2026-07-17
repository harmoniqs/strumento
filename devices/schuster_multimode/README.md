# `schuster_multimode` — device instance (placeholder)

A **device instance** of the `cqed` pack: the Stanford Schuster-lab "multimode"
system — `1×Transmon + 1×BosonicMode(manipulate) + 7×BosonicMode(storage) +
Coupler + Readout`.

This directory will hold the instance config (not yet authored — see plan Task 5):

- `device.yaml` — topology + wiring (line/generator → DAC) + calibration values,
  transcribed from the v1 `hardware_config.yml`.
- `storage_swaps.csv` — the beamsplitter swap dataset (plan Task 13).
- `soccfg_snapshot.json` — committed QickConfig snapshot for off-board work.

Wiring is *data*, not code: re-patching a cable is a one-line edit here, and every
experiment and pulse factory is untouched (they speak only device language).
