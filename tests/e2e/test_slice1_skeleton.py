"""Slice 1 end-to-end vertical (plan Task 10).

Proves the whole pure vertical — instance YAML -> device -> ops -> Seq -> compiler — produces the
expected slice-1 program (structural gate, spec §10.1), and that ``quantize`` realizes the real
calibrated ge_pi to the DAC grid (numeric-golden gate, spec §10.2). The structural + quantize
gates run now against the committed (v1) soccfg. The full construct+acquire+MeasurementResult
vertical needs a real tProc-v2 soccfg and is gated (see tests/_v2gate.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _recording import RecordingProgram  # noqa: E402
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

from strumento.core.compiler import Compiler, quantize  # noqa: E402
from strumento.core.device import Device  # noqa: E402
from strumento.core.pulses import Gaussian, Seq  # noqa: E402

YAML = Path(__file__).parents[2] / "devices" / "multimode_demo" / "device.yaml"
LOOPBACK_YAML = Path(__file__).parents[2] / "devices" / "loopback_demo" / "device.yaml"
MAXV = 32766


@pytest.fixture
def dev():
    return Device.load(YAML)


def test_slice1_vertical_structural(dev):
    """dev.qubit.ge_pi() + measure compiles into the expected build calls."""
    prog = RecordingProgram()
    seq = Seq().play(dev.qubit.ge_pi()).measure()
    c = Compiler(prog, dev.wiring)
    c.declare(seq)
    c.play(seq)

    # generator declared on the qubit drive DAC (ch 2), Nyquist zone 1
    dg = prog.first("declare_gen")
    assert dg[1] == 2 and dg[2]["nqz"] == 1

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


def test_slice1_numeric_golden_quantize(dev):
    """quantize() realizes the calibrated ge_pi to hand-computed DAC-grid values (v1 soccfg)."""
    pytest.importorskip("qick")
    from qick import QickConfig

    soccfg = QickConfig(dev.soccfg_snapshot)
    p = dev.qubit.ge_pi()
    r = quantize(p, soccfg, dev.wiring)

    # frequency snaps to the DDS grid within a few Hz of the requested 4000.0 MHz
    assert r.freq_mhz_actual == pytest.approx(4000.0, abs=1e-3)
    # gain realized to amplitude resolution == the calibrated code 3289 / maxv
    assert r.gain_frac == pytest.approx(8192 / MAXV)
    # a 4-sigma (0.14 us) Gaussian window is a positive integer number of cycles
    assert r.cycles == soccfg.us2cycles(0.030 * 4, gen_ch=2)
    assert r.cycles > 0


@requires_v2_soccfg
def test_slice1_full_vertical_on_mock():
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
