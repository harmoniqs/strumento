# `loopback_demo` — the single-channel v2 testbench instance

A minimal `cqed` instance shaped to the official qick **testbench** tProc-v2 firmware config
(`devices/_fixtures/soccfg_v2_testbench.json`): one full-speed generator (DAC ch 0) and one
dynamic readout (ADC 0). The qubit drive and the readout share that one generator. It exists so
that a **real `AveragerProgramV2` builds offline** — in tests and CI, with no board — against a
genuine `qick_processor` (v2) config.

**What it supports:** real v2 program assembly, on a single channel. It passes
`strumento check`, and single-line programs build, compile and run through `SimulatorSoc`.

**What it doesn't:** the multi-line cqed experiments. They need separate drive and probe lines
(and, for the cavity experiments, cavity, sideband and coupler lines), which the testbench
firmware does not have, so the registered cqed experiments do not run on it with their defaults.
Its wiring deliberately stays single-channel: inventing a second line would misdescribe the
testbench firmware.

The cqed exemplar is `multimode_demo` — use it for the device-model walkthrough and for running
the cqed experiments (`examples/sim_smoke/run_cqed.py`). Use this instance when you need real v2
assembly against the official testbench config (see `tests/_v2gate.py` for the config
resolution order).
