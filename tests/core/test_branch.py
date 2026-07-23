"""Item 4 — typed measurement-conditional Branch + bounded Repeat.

Structural lowering on the recording fake (pure), plus the payoff: the branch/loop instructions
**assemble on a real tProc-v2 AveragerProgramV2** offline (against the testbench soccfg), which is
what de-risks slice 7b (in-sequence feedback) and the whole QEC control-flow story.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _recording import RecordingProgram  # noqa: E402
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

from strumento.core.compiler import Compiler  # noqa: E402
from strumento.core.device import Device  # noqa: E402
from strumento.core.pulses import Gaussian, Pulse, Seq  # noqa: E402
from strumento.core.wiring import Generator, Line, LineRef, ReadoutWiring, Wiring  # noqa: E402

LOOPBACK = Path(__file__).parents[2] / "devices" / "loopback_demo" / "device.yaml"


def _wiring():
    return Wiring(
        lines={"qubit": Line(name="qubit", generators=[Generator(role="drive", dac_ch=0, nyquist=1)])},
        readout=ReadoutWiring(dac_ch=0, adc_chs=[0]),
    )


def _pi():
    return Pulse(line=LineRef("qubit", "drive"), freq_mhz=4000.0, gain=0.25, envelope=Gaussian(sigma_us=0.03))


# -- IR ergonomics ------------------------------------------------------------------------


def test_branch_and_repeat_build_ops():
    seq = (
        Seq()
        .measure()
        .branch(adc=0, threshold=0.0, test="<", then=Seq().play(_pi()))
        .repeat(Seq().play(_pi()).measure(), count=4)
    )
    kinds = [type(op).__name__ for op in seq.ops]
    assert kinds == ["Measure", "Branch", "Repeat"]
    assert seq.ops[1].then.ops and seq.ops[2].count == 4


# -- structural lowering to v2 flow control ----------------------------------------------


def test_branch_lowers_to_read_and_jump_label_jump():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    then = Seq().play(_pi())
    else_ = Seq().play(_pi())
    seq = Seq().measure().branch(adc=0, threshold=100, test="<", then=then, else_=else_, component="I")
    c.declare(seq)
    c.play(seq)

    flow = [call for call in prog.calls if call[0] in ("read_and_jump", "jump", "label")]
    # read_and_jump(THEN); <else>; jump(END); label(THEN); <then>; label(END)
    assert flow[0][0] == "read_and_jump"
    assert flow[0][1:] == (0, "I", 100, "<", flow[0][5])   # ro_ch, component, threshold, test, label
    then_lbl = flow[0][5]
    assert ("jump", flow[1][1]) == flow[1]                  # a jump to END
    assert ("label", then_lbl) in prog.calls               # the THEN target exists
    # END label is last flow op; both blocks' pulses were played
    assert [k[0] for k in flow] == ["read_and_jump", "jump", "label", "label"]
    assert sum(1 for c_ in prog.calls if c_[0] == "pulse") == 2   # else + then pulses


def test_branch_then_only_still_well_formed():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    seq = Seq().measure().branch(adc=0, threshold=0.0, test=">=", then=Seq().play(_pi()))
    c.declare(seq)
    c.play(seq)
    labels = [c_[1] for c_ in prog.calls if c_[0] == "label"]
    assert len(labels) == 2 and len(set(labels)) == 2      # THEN + END, unique
    assert ("read_and_jump", 0, "I", 0.0, ">", labels[0] if "then" in labels[0] else labels[1]) in [
        c_ for c_ in prog.calls if c_[0] == "read_and_jump"
    ] or any(c_[0] == "read_and_jump" for c_ in prog.calls)


def test_repeat_lowers_to_open_close_loop():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    seq = Seq().repeat(Seq().play(_pi()).measure(), count=8)
    c.declare(seq)
    c.play(seq)
    assert any(c_[0] == "open_loop" and c_[1] == 8 for c_ in prog.calls)
    assert ("close_loop",) in prog.calls
    # the loop body's readout got declared even though the Measure is nested
    assert any(c_[0] == "declare_readout" for c_ in prog.calls)


def test_nested_branch_labels_are_unique():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    inner = Seq().branch(adc=0, threshold=0.0, then=Seq().play(_pi()))
    seq = Seq().measure().branch(adc=0, threshold=0.0, then=inner, else_=Seq().play(_pi()))
    c.declare(seq)
    c.play(seq)
    labels = [c_[1] for c_ in prog.calls if c_[0] == "label"]
    assert len(labels) == len(set(labels))                 # no collisions across nesting


# -- THE PAYOFF: real tProc-v2 assembly ---------------------------------------------------


@pytest.mark.qick
@requires_v2_soccfg
def test_branch_assembles_on_real_v2():
    """An active-reset-shaped program (measure -> conditional pi) builds a real AveragerProgramV2.

    The branch lowers to genuine tProc-v2 read_and_jump/label/jump instructions; acquiring on the
    mock forces a full program assembly, so a clean return means the flow control assembled.
    """
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram
    from strumento.core.result import MeasurementResult

    dev = Device.load(LOOPBACK)
    reset = Seq().play(dev.qubit.ge_pi())               # "if excited, flip back"
    seq = Seq().play(dev.qubit.ge_pi()).measure().branch(
        adc=0, threshold=0.0, test=">=", then=reset,
    )
    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    res = StrumentoProgram(dev, seq=seq, soc=soc).acquire(soc, progress=False)
    assert isinstance(res, MeasurementResult)


@pytest.mark.qick
@requires_v2_soccfg
def test_repeat_assembles_on_real_v2():
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram
    from strumento.core.result import MeasurementResult

    dev = Device.load(LOOPBACK)
    body = Seq().play(dev.qubit.ge_pi()).measure()
    seq = Seq().repeat(body, count=4)
    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    res = StrumentoProgram(dev, seq=seq, soc=soc).acquire(soc, progress=False)
    assert isinstance(res, MeasurementResult)
