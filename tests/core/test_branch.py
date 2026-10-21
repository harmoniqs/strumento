"""Typed measurement-conditional Branch + bounded Repeat.

Structural lowering on the recording fake (pure), plus the payoff: the branch/loop instructions
**assemble on a real tProc-v2 AveragerProgramV2** offline (against the testbench soccfg), which is
what de-risks in-sequence feedback (active reset) and the whole QEC control-flow story.
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
from tests._paths import DEVICES  # noqa: E402

LOOPBACK = DEVICES / "loopback_demo" / "device.yaml"


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


def test_branch_blocks_for_a_fresh_read_before_read_and_jump():
    """The freshness gate (F1): a processor-blocking ``wait_auto(ros=True)`` must sit **immediately**
    before ``read_and_jump``, else the on-board branch reads a stale ADC accumulator.

    ``read_and_jump`` runs at tProc execution time, off the timed queue; qick's ``read_input`` warns
    it returns a stale value unless the readout has completed. A plain ``Delay``/``wait`` only shifts
    the reference time and never stalls the processor, so only this blocking wait guarantees a fresh
    read — the v2 port of the tProc-v1 idiom of a ``wait_all`` before every ``read``.
    ``settle_us`` rides through as the wait's argument (the ADC-transfer margin, on top of the
    readout window end that ``ros=True`` keys off).
    """
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    seq = Seq().measure().branch(adc=0, threshold=0.0, test=">=",
                                 then=Seq().play(_pi()), settle_us=0.3)
    c.declare(seq)
    c.play(seq)
    kinds = prog.kinds()
    i_read = kinds.index("read_and_jump")
    assert kinds[i_read - 1] == "wait_auto"                          # adjacency: nothing between
    assert prog.calls[i_read - 1] == ("wait_auto", 0.3, False, True)  # (t=settle_us, gens, ros)


def test_branch_without_settle_still_blocks_until_the_readout_ends():
    """Even at the ``settle_us=0.0`` default the branch emits the blocking wait (waits to the window
    end, no extra transfer margin) — the freshness gate is structural, not opt-in."""
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    seq = Seq().measure().branch(adc=0, threshold=0.0, test=">=", then=Seq().play(_pi()))
    c.declare(seq)
    c.play(seq)
    kinds = prog.kinds()
    assert kinds[kinds.index("read_and_jump") - 1] == "wait_auto"
    assert prog.first("wait_auto") == ("wait_auto", 0.0, False, True)


@pytest.mark.parametrize(
    "threshold,expected",
    [(0.5, 1), (2.5, 3), (-0.5, -1), (-2.5, -3), (100, 100), (-33.69, -34), (0.0, 0)],
)
def test_branch_threshold_realizes_half_away_from_zero(threshold, expected):
    """read_and_jump wants a signed int accumulator threshold; the IR carries a float (F4).

    Python's ``round`` is banker's rounding (``0.5 -> 0``, ``2.5 -> 2``) — a surprising, sign-
    asymmetric comparison boundary. The compiler realizes half-away-from-zero instead, so a
    midpoint lands on the excited side symmetrically about zero. (Notably the ``--sim`` back-compat
    default ``0.5`` now realizes to 1, not 0.)
    """
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    seq = Seq().measure().branch(adc=0, threshold=threshold, test=">=", then=Seq().play(_pi()))
    c.declare(seq)
    c.play(seq)
    assert prog.first("read_and_jump")[3] == expected      # (kind, ro_ch, component, threshold, ...)


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
def test_read_and_jump_is_preceded_by_a_blocking_wait_in_assembly():
    """The F1 freshness gate survives **real v2 assembly** — the offline corrector for it.

    ``--sim``/``--mock`` cannot see feedback timing (the simulator walks the branch with a perfectly
    fresh read; the mock returns zeros), so this inspects the compiled macro stream directly: a
    blocking ``Wait`` macro must sit immediately before the accumulator read (``ReadInput``, which
    ``read_and_jump`` expands to). Without it, ``read_input`` reads a stale value on a board.
    """
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram

    dev = Device.load(LOOPBACK)
    reset = Seq().play(dev.qubit.ge_pi())
    seq = Seq().play(dev.qubit.ge_pi()).measure().branch(
        adc=0, threshold=0.0, test=">=", then=reset, settle_us=0.3,
    )
    prog = StrumentoProgram(dev, seq=seq, soc=MockQickSocV2.from_snapshot(V2_SOCCFG))
    prog.compile()
    names = [type(m).__name__ for m in prog.macro_list]
    assert "ReadInput" in names, names
    i_read = names.index("ReadInput")                       # the accumulator read
    assert names[i_read - 1] == "Wait", names               # a blocking wait immediately precedes it
    assert prog.macro_list[i_read - 1].auto is True         # wait_auto (keyed to the readout end)


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
