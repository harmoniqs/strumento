"""Tracing tests — opt-in compiler trace and program dump.

Marked qick: uses StrumentoProgram (which imports qick).
"""

from __future__ import annotations

import pytest

pytest.importorskip("qick")

from strumento.core.device import Device
from strumento.core.mock import MockQickSocV2
from strumento.core.program import StrumentoProgram
from strumento.core.pulses import Seq
from tests._paths import DEVICES, LOOPBACK_YAML

SOCCFG = DEVICES / "_fixtures" / "soccfg_v2_testbench.json"


@pytest.fixture(scope="module")
def dev():
    return Device.load(LOOPBACK_YAML)


@pytest.fixture(scope="module")
def soc():
    return MockQickSocV2.from_snapshot(SOCCFG)


@pytest.mark.qick
class TestCompilerTrace:
    """The compiler trace records decisions when trace=True."""

    def test_trace_off_by_default(self, dev, soc):
        seq = Seq().play(dev.qubit.ge_pi()).measure()
        prog = StrumentoProgram(dev, seq=seq, reps=1, soc=soc)
        assert prog._trace is None
        assert prog.trace() is None

    def test_trace_records_pulse_and_measure(self, dev, soc):
        seq = Seq().play(dev.qubit.ge_pi()).wait(0.5).measure()
        prog = StrumentoProgram(dev, seq=seq, reps=1, soc=soc, trace=True)
        trace = prog.trace()

        assert trace is not None
        entries = trace["entries"]
        assert len(entries) > 0

        # Should have a pulse entry and a measure entry
        ops = [e["op"] for e in entries]
        assert "pulse" in ops
        assert "measure" in ops
        assert "delay" in ops

    def test_trace_pulse_has_details(self, dev, soc):
        seq = Seq().play(dev.qubit.ge_pi()).measure()
        prog = StrumentoProgram(dev, seq=seq, reps=1, soc=soc, trace=True)
        trace = prog.trace()

        pulse_entries = [e for e in trace["entries"] if e["op"] == "pulse"]
        assert len(pulse_entries) >= 1
        p = pulse_entries[0]
        assert "ch" in p
        assert "name" in p
        assert "freq" in p
        assert "gain" in p
        assert "label" in p
        assert "envelope" in p

    def test_trace_gen_declaration(self, dev, soc):
        seq = Seq().play(dev.qubit.ge_pi()).measure()
        prog = StrumentoProgram(dev, seq=seq, reps=1, soc=soc, trace=True)
        trace = prog.trace()

        gen_entries = [e for e in trace["entries"] if e["op"] == "gen"]
        assert len(gen_entries) >= 1
        assert "ch" in gen_entries[0]
        assert "nqz" in gen_entries[0]
        assert "line" in gen_entries[0]


@pytest.mark.qick
class TestProgramDump:
    """prog.dump() returns the full compiled state from strumento + qick."""

    def test_dump_has_both_layers(self, dev, soc):
        seq = Seq().play(dev.qubit.ge_pi()).measure()
        prog = StrumentoProgram(dev, seq=seq, reps=1, soc=soc)
        dump = prog.dump()

        assert "strumento" in dump
        assert "qick" in dump

    def test_strumento_layer(self, dev, soc):
        seq = Seq().play(dev.qubit.ge_pi()).measure()
        prog = StrumentoProgram(dev, seq=seq, reps=1, soc=soc)
        stru = prog.dump()["strumento"]

        assert "declared_gens" in stru
        assert "declared_readouts" in stru
        assert "envelope_names" in stru
        assert "pulse_names" in stru

    def test_qick_layer(self, dev, soc):
        seq = Seq().play(dev.qubit.ge_pi()).measure()
        prog = StrumentoProgram(dev, seq=seq, reps=1, soc=soc)
        qick = prog.dump()["qick"]

        assert "asm_instructions" in qick
        assert qick["asm_instructions"] > 0
        assert "asm" in qick
        assert "gen_chs" in qick
        assert "ro_chs" in qick
        assert "pulses" in qick

    def test_print_trace_does_not_crash(self, dev, soc, capsys):
        """print_trace() produces output without errors."""
        seq = Seq().play(dev.qubit.ge_pi()).wait(0.5).measure()
        prog = StrumentoProgram(dev, seq=seq, reps=1, soc=soc, trace=True)
        prog.print_trace()

        captured = capsys.readouterr()
        assert "STRUMENTO PROGRAM DUMP" in captured.out
        assert "Compiler Trace" in captured.out
        assert "ASM" in captured.out
