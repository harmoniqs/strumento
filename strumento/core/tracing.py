"""Opt-in compilation and execution tracing for strumento + qick debugging.

Two layers, both opt-in:

**Compiler trace** — records every decision the :class:`~strumento.core.compiler.Compiler` makes:
which generator resolved for each ``LineRef``, what frequency snapped to on the DDS grid, how
``delay_auto`` advanced the cursor, which envelopes were deduped, which pulse names were assigned.
Enabled via ``StrumentoProgram(dev, seq, trace=True)``.

**Program dump** — after compilation, extracts the full compiled state from both strumento (the
compiler's name maps, sweep bindings, declared channels) and qick (the ASM instruction list,
waveform memory, register values, loop structure). Uses stock qick APIs (``dump_prog()``,
``get_pulse_param``, ``pulses``, ``gen_chs``, ``ro_chs``) — no modified qick needed.

This module is **qick-facing** (it reads compiled program attributes), so it sits alongside
``core/program.py`` in the import hierarchy — inside the qick boundary.

Usage::

    from strumento.core.program import StrumentoProgram

    prog = StrumentoProgram(dev, seq=seq, reps=100, soc=soc, trace=True)
    # The compiler trace is recorded during __init__ (which compiles)

    # Print the full trace (strumento compiler + qick compiled state)
    prog.print_trace()

    # Or get the structured data
    trace = prog.trace()     # compiler decisions
    dump = prog.dump()       # qick compiled state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEntry:
    """One compiler decision or qick call."""

    phase: str          # "declare" or "play"
    op: str             # "gen_resolve", "pulse", "delay_auto", "trigger", "envelope_dedup", ...
    detail: dict        # op-specific fields


@dataclass
class CompilerTrace:
    """Accumulates compiler decisions when tracing is enabled."""

    entries: list[TraceEntry] = field(default_factory=list)

    def emit(self, phase: str, op: str, **detail: Any) -> None:
        self.entries.append(TraceEntry(phase=phase, op=op, detail=detail))

    def print_report(self) -> None:
        """Human-readable trace report to stdout."""
        for e in self.entries:
            detail_str = ", ".join(f"{k}={v}" for k, v in e.detail.items())
            print(f"  [{e.phase}] {e.op}: {detail_str}")


def dump_program(prog) -> dict:
    """Extract the full compiled state from a ``StrumentoProgram``.

    Combines strumento's compiler metadata with qick's ``dump_prog()`` output.
    Works with stock qick — no modifications needed.

    Returns a dict with two sections:
    - ``strumento``: compiler name maps, sweep bindings, declared channels
    - ``qick``: ASM instructions, waveform memory, gen/ro configs, loop structure
    """
    result: dict[str, Any] = {}

    # -- Strumento layer: compiler decisions --
    compiler = getattr(prog, "_compiler", None)
    if compiler is not None:
        result["strumento"] = {
            "declared_gens": dict(compiler._declared_gens),
            "declared_readouts": sorted(compiler._declared_ros),
            "envelope_names": {
                f"ch{k[0]}_{k[1].__class__.__name__}": v
                for k, v in compiler._env_names.items()
            },
            "pulse_names": {
                f"ch{k[0]}": v
                for k, v in compiler._pulse_names.items()
            },
            "declared_loops": sorted(compiler._declared_loops),
            # Delay sites are keyed internally by tuples; provenance dumps must stay JSON-safe.
            "delay_tags": {str(key): value for key, value in compiler._delay_tags.items()},
        }

    # -- Qick layer: compiled program state --
    qick_state: dict[str, Any] = {}

    # ASM instructions
    prog_list = getattr(prog, "prog_list", None)
    if prog_list is not None:
        qick_state["asm_instructions"] = len(prog_list)
        qick_state["asm"] = [dict(inst) if isinstance(inst, dict) else str(inst) for inst in prog_list]

    # Generator channels
    gen_chs = getattr(prog, "gen_chs", None)
    if gen_chs:
        qick_state["gen_chs"] = {
            str(ch): dict(cfg) if hasattr(cfg, "items") else str(cfg)
            for ch, cfg in gen_chs.items()
        }

    # Readout channels
    ro_chs = getattr(prog, "ro_chs", None)
    if ro_chs:
        qick_state["ro_chs"] = {
            str(ch): dict(cfg) if hasattr(cfg, "items") else str(cfg)
            for ch, cfg in ro_chs.items()
        }

    # Pulses (named pulse objects with raw register values)
    pulses = getattr(prog, "pulses", None)
    if pulses:
        pulse_info = {}
        for name, pulse_obj in pulses.items():
            pulse_info[str(name)] = str(pulse_obj)
        qick_state["pulses"] = pulse_info

    # Loop structure
    loop_dims = getattr(prog, "loop_dims", None)
    if loop_dims is not None:
        qick_state["loop_dims"] = list(loop_dims) if hasattr(loop_dims, "__iter__") else loop_dims
    avg_level = getattr(prog, "avg_level", None)
    if avg_level is not None:
        qick_state["avg_level"] = avg_level

    # Labels (loop variable names -> register addresses)
    labels = getattr(prog, "labels", None)
    if labels:
        qick_state["labels"] = dict(labels)

    # Waveforms
    waves = getattr(prog, "waves", None)
    if waves:
        qick_state["waveforms"] = [
            {"name": getattr(w, "name", None), "length": getattr(w, "length", None)}
            for w in waves
        ]

    # Full dump_prog (the transport payload)
    try:
        dump = prog.dump_prog()
        qick_state["dump_prog_keys"] = sorted(dump.keys())
    except Exception:
        pass

    result["qick"] = qick_state
    return result


def print_program_dump(prog, *, max_asm: int = 50) -> None:
    """Print a human-readable dump of the compiled program.

    Shows both the strumento compiler's decisions and qick's compiled output.
    """
    dump = dump_program(prog)

    print("=" * 70)
    print("STRUMENTO PROGRAM DUMP")
    print("=" * 70)

    # Strumento layer
    stru = dump.get("strumento", {})
    if stru:
        print("\n--- Strumento Compiler ---")
        print(f"  Declared generators: {stru.get('declared_gens', {})}")
        print(f"  Declared readouts: {stru.get('declared_readouts', [])}")
        if stru.get("envelope_names"):
            print("  Envelopes:")
            for k, v in stru["envelope_names"].items():
                print(f"    {k} -> {v}")
        if stru.get("pulse_names"):
            print("  Pulse names:")
            for k, v in stru["pulse_names"].items():
                print(f"    {k} -> {v}")
        if stru.get("declared_loops"):
            print(f"  Loops: {stru['declared_loops']}")

    # Qick layer
    qick = dump.get("qick", {})
    if qick:
        print("\n--- Qick Compiled State ---")
        if "gen_chs" in qick:
            print("  Generator channels:")
            for ch, cfg in qick["gen_chs"].items():
                print(f"    ch {ch}: {cfg}")
        if "ro_chs" in qick:
            print("  Readout channels:")
            for ch, cfg in qick["ro_chs"].items():
                print(f"    ch {ch}: {cfg}")
        if "pulses" in qick:
            print("  Pulses:")
            for name, info in qick["pulses"].items():
                print(f"    {name}: {info}")
        if "loop_dims" in qick:
            print(f"  Loop dims: {qick['loop_dims']}")
        if "labels" in qick:
            print(f"  Labels: {qick['labels']}")
        if "waveforms" in qick:
            print("  Waveforms:")
            for w in qick["waveforms"]:
                print(f"    {w['name']}: {w['length']} samples")
        if "asm" in qick:
            n = qick["asm_instructions"]
            print(f"\n  ASM ({n} instructions):")
            for inst in qick["asm"][:max_asm]:
                print(f"    {inst}")
            if n > max_asm:
                print(f"    ... ({n - max_asm} more)")

    # Compiler trace (if tracing was enabled)
    trace = getattr(prog, "_trace", None)
    if trace and trace.entries:
        print(f"\n--- Compiler Trace ({len(trace.entries)} entries) ---")
        trace.print_report()

    print("\n" + "=" * 70)
