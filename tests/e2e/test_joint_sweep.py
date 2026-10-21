"""Joint sweep e2e (D36) — two parameters bound to ONE declared axis step together in a
single loop.

The old structure (one compiler-invented loop per swept-parameter name) made this
inexpressible: two named sweeps always produced a 2D outer product. With declared axes the
user says "these two ranges ride the same loop" — a 1D joint scan, which is exactly the
tProc-v2 model (declare the loop axes by hand; bind parameter ranges to them).
"""

import sys
from pathlib import Path

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
def test_joint_sweep_acquires_a_1d_scan(dev):
    """gain AND freq swept on one declared axis → ONE add_loop → a 1D result, not 7×7."""
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram

    g = Sweep(start=0, stop=16380, on="ax")        # lab-native int codes → fraction at the factory
    f = Sweep(start=3980.0, stop=4020.0, on="ax")  # MHz around f_ge = 4000
    seq = Seq().play(dev.qubit.ge_pi(gain=g, freq=f)).measure()

    prog = StrumentoProgram(dev, seq, axes=[Loop("ax", 7)], reps=4)
    res = prog.acquire(MockQickSocV2.from_snapshot(V2_SOCCFG))

    assert res.avgi.size == 7          # one loop of 7 — not a 49-point outer product
    assert res.xpts is not None and len(res.xpts) == 7
    # the display axis is lab-native (int codes), reconstructed via the factory's inverse map
    assert res.xpts[0] == pytest.approx(0) and res.xpts[-1] == pytest.approx(16380)


@pytest.mark.qick
@requires_v2_soccfg
def test_joint_sweep_undeclared_axis_is_a_typed_error(dev):
    from strumento.core.mock import MockQickSocV2  # noqa: F401
    from strumento.core.program import StrumentoProgram

    seq = Seq().play(dev.qubit.ge_pi(gain=Sweep(start=0, stop=16380, on="nope"))).measure()
    with pytest.raises(ValueError, match="undeclared axis"):
        StrumentoProgram(dev, seq, reps=1)
