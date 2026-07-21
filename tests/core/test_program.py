"""Tests for StrumentoProgram + acquire -> MeasurementResult (plan Task 9).

The acquire->result wrapping is pure-tested with a synthetic raw array. The full
construct+acquire path needs a real ``AveragerProgramV2`` (tProc-v2 firmware), so it is gated on
a v2 soccfg fixture (see tests/_v2gate.py) and skips until one is present.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

pytest.importorskip("qick")

from strumento.core.program import _to_measurement_result  # noqa: E402
from strumento.core.result import MeasurementResult  # noqa: E402


def test_to_measurement_result_single_readout():
    # AveragerProgramV2.acquire returns (n_ch, n_reads, 2); build a synthetic one.
    raw = np.array([[[0.3, -0.7]]])  # 1 ch, 1 read, (I, Q)
    res = _to_measurement_result(raw, xpts=None)
    assert isinstance(res, MeasurementResult)
    assert res.avgi.reshape(-1)[0] == pytest.approx(0.3)
    assert res.avgq.reshape(-1)[0] == pytest.approx(-0.7)
    assert res.xpts is None
    x, i, q = res.as_tuple()
    assert i is res.avgi


def test_to_measurement_result_attaches_sweep_axis():
    raw = np.zeros((1, 1, 5, 2))  # 1 ch, 1 read, 5 expts, (I, Q)
    xpts = np.linspace(0, 30000, 5)
    res = _to_measurement_result(raw, xpts=xpts)
    assert res.avgi.shape[-1] == 5
    assert res.xpts is xpts


@requires_v2_soccfg
def test_construct_and_acquire_returns_measurement_result():
    # Only runs when a real tProc-v2 soccfg is present (Sho's dump_cfg).
    from strumento.core.device import Device
    from strumento.core.mock import MockQickSocV2
    from strumento.core.pulses import Seq

    dev = Device.load(Path(__file__).parents[2] / "devices" / "schuster_multimode" / "device.yaml")
    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    from strumento.core.program import StrumentoProgram

    prog = StrumentoProgram(dev, seq=Seq().play(dev.qubit.ge_pi()).measure(), soc=soc)
    res = prog.acquire(soc, progress=False)
    assert isinstance(res, MeasurementResult)
    assert res.as_tuple()[1] is not None
