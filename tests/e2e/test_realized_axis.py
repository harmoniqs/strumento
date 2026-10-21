"""Realized (as-compiled) result axes, end to end (D37).

The result's ``xpts`` is the axis the hardware *actually steps*, read back from the compiled
program — the register-rounded per-step values with the loop engine's quantized step sizes —
and the requested axis is preserved at ``meta["requested_xpts"]``. The rule: never emit the
original lab-native values; emit the lab-native values that correspond to the as-compiled
values.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

pytest.importorskip("qick")

from strumento.core.pulses import Seq  # noqa: E402
from strumento.core.sweeps import Loop, Sweep  # noqa: E402
from tests._paths import DEVICES  # noqa: E402

YAML = DEVICES / "loopback_demo" / "device.yaml"


@pytest.fixture
def dev(loopback_dev):
    return loopback_dev


@pytest.mark.qick
@requires_v2_soccfg
def test_freq_sweep_xpts_is_the_compiled_axis(dev):
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram

    f = Sweep(start=3980.0, stop=4020.0, on="ax")
    p = dev.qubit.ge_pi(freq=f)
    seq = Seq().play(p).measure()
    prog = StrumentoProgram(dev, seq, axes=[Loop("ax", 7)], reps=4)
    res = prog.acquire(MockQickSocV2.from_snapshot(V2_SOCCFG))

    requested = np.linspace(3980.0, 4020.0, 7)
    # the requested axis is preserved verbatim in meta
    assert np.asarray(res.meta["requested_xpts"]) == pytest.approx(requested)
    # xpts IS the compiled axis: the same per-step array realized() reads back (independently
    # pinned by test_program.py) — not a second derivation of the requested points
    realized = np.asarray(res.xpts, dtype=float)
    assert realized.shape == requested.shape
    assert np.all(np.diff(realized) > 0)  # monotonic: a faithful stepped axis
    assert realized[0] == pytest.approx(requested[0], abs=1e-3)
    assert realized[-1] == pytest.approx(requested[-1], abs=1e-3)


@pytest.mark.qick
@requires_v2_soccfg
def test_gain_sweep_xpts_reports_the_compiled_register_steps(dev):
    """The display axis carries the loop engine's truncated register steps.

    qick quantizes the sweep span and step in register space: 11 steps over 0..8191 codes
    compile to a step of 819, so the compiled axis stops at 8190 — one register step below the
    requested 8191. A per-point round-trip of the requested values reports 8191; the readback
    reports the truth. This is the discrepancy class flagged in review (PR #25).
    """
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram

    g = Sweep(start=0, stop=8191, on="ax")
    p = dev.qubit.ge_pi(gain=g)
    seq = Seq().play(p).measure()
    prog = StrumentoProgram(dev, seq, axes=[Loop("ax", 11)], reps=4)
    res = prog.acquire(MockQickSocV2.from_snapshot(V2_SOCCFG))

    xpts = np.asarray(res.xpts, dtype=float)
    assert xpts.shape == (11,)
    assert xpts[-1] == pytest.approx(8190.0, abs=1.5)  # the compiled top, not the requested 8191
    # and the axis is the compiled one throughout: uniform register steps of 819 codes
    steps = np.diff(xpts)
    assert np.allclose(steps, 819.0, atol=1.0)


@pytest.mark.qick
@requires_v2_soccfg
def test_int_code_gain_sweep_steps_evenly_to_identity(dev):
    """A range that divides evenly on the register grid realizes to itself: the honest display
    is the codes the tProc steps, which here coincide with the requested linspace."""
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram

    g = Sweep(start=0, stop=16380, on="ax")
    seq = Seq().play(dev.qubit.ge_pi(gain=g)).measure()
    prog = StrumentoProgram(dev, seq, axes=[Loop("ax", 7)], reps=4)
    res = prog.acquire(MockQickSocV2.from_snapshot(V2_SOCCFG))

    assert np.asarray(res.xpts, dtype=float) == pytest.approx(np.linspace(0, 16380, 7), rel=1e-6)


@pytest.mark.qick
@requires_v2_soccfg
def test_swept_delay_axis_is_read_back_from_the_compiled_program(dev):
    """A swept Delay's axis comes from qick's time-param readback — the tProc-cycle-rounded
    durations, not the requested linspace (T1/Ramsey ride exactly this path)."""
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram

    t = Sweep(start=0.0, stop=1.0, on="t")
    seq = Seq().play(dev.qubit.ge_pi()).wait(t).measure()
    prog = StrumentoProgram(dev, seq, axes=[Loop("t", 5)], reps=4)
    res = prog.acquire(MockQickSocV2.from_snapshot(V2_SOCCFG))

    xpts = np.asarray(res.xpts, dtype=float)
    assert xpts.shape == (5,)
    # every played delay is an integer number of tProc cycles (the register grid) — via qick's
    # own conversion, no second belief about the clock; 0.25 µs steps are 107.52 cycles at the
    # testbench's 430.08 MHz timing clock, so truncation makes this bite (not a vacuous check)
    cycles = np.array([prog.soccfg.us2cycles(float(x)) for x in xpts])
    assert np.allclose(cycles, np.round(cycles), atol=1e-6)
    assert np.all(np.diff(xpts) > 0)
    # the requested axis is preserved verbatim
    assert np.asarray(res.meta["requested_xpts"]) == pytest.approx(np.linspace(0.0, 1.0, 5))
