# The `strumento` CLI

A small, agent-forward command-line surface over the pure layers. Every command prints
human-readable text by default and machine-readable JSON with `--json`; errors come back as
`{"ok": false, "error": ...}` with a nonzero exit code, never a traceback.

```
strumento packs                          # installed modality packs (entry points)
strumento validate <device.yaml>         # load + validate an instance
strumento info <device.yaml>             # topology, namespaces, wiring table
strumento soccfg <dump_cfg.json>         # board, tProc v1-vs-v2 verdict, generators
strumento fit <kind> <data.csv>          # rabi | t1 | t2 | parity | lorentzian | circle
strumento compile <device.yaml>          # lower a canned Rabi; list the program calls
```

Examples:

```console
$ strumento soccfg devices/_fixtures/soccfg_v2_testbench.json
ZCU216 · qick 0.2.367 · tproc qick_processor · 1 gen(s), 1 readout(s)
tProc v2 — real AveragerProgramV2 assembly works offline

$ strumento fit t1 t1_trace.csv --json
{ "ok": true, "kind": "t1", "success": true, "derived": { "T1": 41.98, ... } }
```

`fit` reads a headered CSV: two columns (`x,y`) for the scalar fits, three (`x,i,q`) for the
resonator `circle` fit. `compile` needs no `qick` for the lowering listing; with `qick`
installed it also reports the DAC-grid realized values of the calibrated π pulse.

## Why a CLI (and not an MCP server)?

Agents (Claude Code, Amicode, CI jobs) execute CLIs natively in every environment, including
headless runners — no server lifecycle, no transport. The commands are thin wrappers over
importable functions (`strumento.cli`), so an MCP server later is a ~50-line shim over the same
surface if a hosted-agent use case ever wants it.
