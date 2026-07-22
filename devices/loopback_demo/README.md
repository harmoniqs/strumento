# `loopback_demo` — the offline v2-assembly instance

A minimal `cqed` instance shaped to the official qick **testbench** tProc-v2 firmware config
(`devices/_fixtures/soccfg_v2_testbench.json`): one full-speed generator (DAC ch 0) and one
dynamic readout (ADC 0). It exists so that a **real `AveragerProgramV2` builds offline** — in
tests and CI, with no board — against a genuine `qick_processor` (v2) config.

Use `multimode_demo` for the full device-model walkthrough; use this instance when you need real
v2 assembly (see `tests/_v2gate.py` for the config resolution order).
