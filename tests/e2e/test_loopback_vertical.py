"""End-to-end vertical on the loopback demo device.

Proves the whole pure vertical — instance YAML -> device -> ops -> Seq -> compiler — produces the
expected pi-pulse-and-measure program (structural gate), and that ``StrumentoProgram.realized``
reads the calibrated ge_pi back from the compiled program (numeric-golden gate). The
structural gate is pure (RecordingProgram); the golden gate assembles a real v2 program on the
loopback demo's committed testbench snapshot. The full construct+acquire+MeasurementResult
vertical needs a real tProc-v2 soccfg and is gated (see tests/_v2gate.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _recording import RecordingProgram  # noqa: E402
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

from strumento.core.compiler import Compiler  # noqa: E402
from strumento.core.device import Device  # noqa: E402
from strumento.core.pulses import Seq  # noqa: E402
from tests._paths import DEVICES  # noqa: E402

LOOPBACK_YAML = DEVICES / "loopback_demo" / "device.yaml"
MAXV = 32766


def test_loopback_vertical_structural(dev):
    """dev.qubit.ge_pi() + measure compiles into the expected build calls."""
    prog = RecordingProgram()
    seq = Seq().play(dev.qubit.ge_pi()).measure()
    c = Compiler(prog, dev.wiring)
    c.declare(seq)
    c.play(seq)

    # generator declared on the qubit drive DAC (ch 1), Nyquist zone 1
    dg = prog.first("declare_gen")
    assert dg[1] == 1 and dg[2]["nqz"] == 1

    # the Gaussian envelope was lowered with the calibrated sigma
    ag = prog.first("add_gauss")
    assert ag[3]["sigma"] == pytest.approx(0.030)

    # the pulse carries the real calibrated frequency + fraction gain (D23)
    ap = prog.first("add_pulse")
    assert ap[3]["freq"] == pytest.approx(4000.0)
    assert ap[3]["gain"] == pytest.approx(8192 / MAXV)

    # timeline order: play the pulse, then trigger the readout
    kinds = prog.kinds()
    assert kinds.index("pulse") < kinds.index("trigger")
    assert prog.first("trigger")[1] == (0,)  # readout on ADC 0


@pytest.mark.qick
def test_loopback_numeric_golden_realized():
    """realized() reads the calibrated ge_pi back from the compiled v2 program."""
    from strumento.core.program import StrumentoProgram  # noqa: E402

    dev = Device.load(LOOPBACK_YAML)
    p = dev.qubit.ge_pi()
    r = StrumentoProgram(dev, seq=Seq().play(p).measure()).realized(p)

    # frequency readback is on the DDS grid within a few Hz of the requested 4000.0 MHz
    assert r.freq_mhz_actual == pytest.approx(4000.0, abs=1e-3)
    # gain realized to the amplitude grid == the calibrated code 8192 / maxv
    assert r.gain_frac == pytest.approx(8192 / MAXV)
    # a 4-sigma (0.12 us) Gaussian window realized on the fabric grid is never shorter
    assert r.length_us_actual >= 0.030 * 4


@pytest.mark.qick
@requires_v2_soccfg
def test_loopback_full_vertical_on_mock():
    dev = Device.load(LOOPBACK_YAML)
    """Full vertical: build a real AveragerProgramV2 and acquire on the mock (needs v2 soccfg)."""
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram
    from strumento.core.result import MeasurementResult

    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    prog = StrumentoProgram(dev, seq=Seq().play(dev.qubit.ge_pi()).measure(), soc=soc)
    res = prog.acquire(soc, progress=False)
    assert isinstance(res, MeasurementResult)
    xpts, avgi, avgq = res.as_tuple()
    assert np.asarray(avgi).size >= 1  # single-readout shape
