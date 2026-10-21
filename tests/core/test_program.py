"""Tests for StrumentoProgram + acquire -> MeasurementResult.

The acquire->result wrapping is pure-tested with a synthetic raw array. The full
construct+acquire path needs a real ``AveragerProgramV2`` (tProc-v2 firmware), so it is gated on
a v2 soccfg fixture (see tests/_v2gate.py) and skips until one is present.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

pytest.importorskip("qick")

from strumento.core.program import _resolve_trig_offset, _to_measurement_result  # noqa: E402
from strumento.core.result import MeasurementResult  # noqa: E402
from tests._paths import DEVICES  # noqa: E402


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


# --- trig_offset resolution (#98): calibration-first, wiring-fallback, 0 if neither ---
#
# _resolve_trig_offset is the pure half of the fix (mirrors final_delay's own
# getattr(device.calib, "readout", None) pattern), so — exactly like _to_measurement_result
# above — it is unit-testable with lightweight duck-typed devices, no board and no full
# pydantic-validated Device needed.


def test_resolve_trig_offset_prefers_calibration_over_wiring():
    device = SimpleNamespace(
        calib=SimpleNamespace(readout=SimpleNamespace(trig_offset=0.45)),
        wiring=SimpleNamespace(readout=SimpleNamespace(trig_offset=0.10)),
    )
    assert _resolve_trig_offset(device) == pytest.approx(0.45)


def test_resolve_trig_offset_falls_back_to_wiring_when_calibration_has_none():
    # a pack whose calibration schema has no concept of trig_offset at all (core never assumes
    # cqed's shape) -> getattr(ro, "trig_offset", None) misses, so the core-level wiring field
    # (populated by any pack) is the fallback.
    device = SimpleNamespace(
        calib=SimpleNamespace(readout=SimpleNamespace()),
        wiring=SimpleNamespace(readout=SimpleNamespace(trig_offset=0.10)),
    )
    assert _resolve_trig_offset(device) == pytest.approx(0.10)


def test_resolve_trig_offset_is_zero_when_neither_calibration_nor_wiring_set_it():
    device = SimpleNamespace(
        calib=SimpleNamespace(readout=SimpleNamespace()),
        wiring=SimpleNamespace(readout=SimpleNamespace()),
    )
    assert _resolve_trig_offset(device) == 0.0


@pytest.mark.qick
@requires_v2_soccfg
def test_construct_and_acquire_returns_measurement_result():
    # Only runs when a tProc-v2 soccfg fixture is present (see tests/_v2gate.py).
    from strumento.core.device import Device
    from strumento.core.mock import MockQickSocV2
    from strumento.core.pulses import Seq

    dev = Device.load(DEVICES / "loopback_demo" / "device.yaml")
    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    from strumento.core.program import StrumentoProgram

    prog = StrumentoProgram(dev, seq=Seq().play(dev.qubit.ge_pi()).measure(), soc=soc)
    res = prog.acquire(soc, progress=False)
    assert isinstance(res, MeasurementResult)
    assert res.as_tuple()[1] is not None


# --- realized(): values read back from the compiled program, never re-converted client-side ---
#
# These tests build on loopback_demo, which pins its own (testbench v2) snapshot, so they are
# independent of the V2_SOCCFG resolution order and run anywhere qick installs.
_LOOPBACK = DEVICES / "loopback_demo" / "device.yaml"
_MAXV = 32766  # the testbench generator's full-scale gain in int codes


@pytest.mark.qick
def test_measure_triggers_at_the_devices_calibrated_trig_offset():
    """AC1, end-to-end (#98): loopback_demo's calib.readout.trig_offset (already set, nonzero
    in the committed fixture) reaches a real compiled program's Measure trigger via the
    Compiler it builds — not a hardcoded 0. The compiler-level mechanism itself is covered by
    tests/core/test_compiler.py; this proves StrumentoProgram.__init__/_initialize actually wire
    the resolved value through end to end."""
    from strumento.core.device import Device
    from strumento.core.program import StrumentoProgram
    from strumento.core.pulses import Seq

    dev = Device.load(_LOOPBACK)
    assert dev.calib.readout.trig_offset != 0  # a meaningful, nonzero fixture value
    prog = StrumentoProgram(dev, seq=Seq().play(dev.qubit.ge_pi()).measure())
    assert prog._compiler.trig_offset == pytest.approx(dev.calib.readout.trig_offset)


#: the cqed demos' readout trigger offset in µs (#133) — a realistic time-of-flight, well inside
#: each demo's 2.0 µs readout window (the 400 these YAMLs used to carry predates #98's µs units).
_DEMO_TRIG_OFFSET_US = 0.5


@pytest.mark.qick
@pytest.mark.parametrize("demo", ["loopback_demo", "multimode_demo"])
def test_demo_readout_trigger_compiles_at_its_offset_in_us(demo):
    """#133: the *compiled* program for a readout on each cqed demo fires the ADC trigger at the
    demo's µs offset — read back from the assembled Trigger macro's tProc-cycle register, not the
    compiler argument — and that offset sits inside the declared readout window."""
    from strumento.core.device import Device
    from strumento.core.program import StrumentoProgram
    from strumento.core.pulses import Seq

    dev = Device.load(DEVICES / demo / "device.yaml")
    prog = StrumentoProgram(dev, seq=Seq().measure())
    triggers = [m for m in prog.macro_list if type(m).__name__ == "Trigger"]
    assert len(triggers) == 1
    t_us = prog.soccfg.cycles2us(triggers[0].t_regs["t"])
    assert t_us == pytest.approx(_DEMO_TRIG_OFFSET_US, abs=prog.soccfg.cycles2us(1))
    assert t_us < dev.calib.readout.readout_length


@pytest.mark.qick
def test_realized_static_pulse_is_grid_snapped():
    from strumento.core.device import Device
    from strumento.core.program import StrumentoProgram
    from strumento.core.pulses import Seq

    dev = Device.load(_LOOPBACK)
    p = dev.qubit.ge_pi()
    r = StrumentoProgram(dev, seq=Seq().play(p).measure()).realized(p)
    # the readback is the compiled value: on the DDS grid, not the requested number
    assert r.freq_mhz_actual == pytest.approx(4000.0, abs=1e-3)
    assert r.freq_mhz_actual != 4000.0
    # gain comes back as the fraction of the calibrated int code (8192), amplitude-grid truncated
    assert r.gain_frac == pytest.approx(8192 / _MAXV)
    # requested 4-sigma window is 0.12 us; the realized length is never shorter
    assert r.length_us_actual >= 0.12


@pytest.mark.qick
def test_realized_swept_freq_is_the_compiled_axis():
    from strumento.core.device import Device
    from strumento.core.program import StrumentoProgram
    from strumento.core.pulses import Seq
    from strumento.core.sweeps import Loop, Sweep

    dev = Device.load(_LOOPBACK)
    p = dev.qubit.ge_pi(freq=Sweep(start=3900.0, stop=4100.0, on="f"))
    r = StrumentoProgram(dev, seq=Seq().play(p).measure(), axes=[Loop("f", 11)]).realized(p)
    assert isinstance(r.freq_mhz_actual, np.ndarray)
    assert r.freq_mhz_actual.shape == (11,)
    assert r.freq_mhz_actual[0] == pytest.approx(3900.0, abs=1e-3)
    assert np.all(np.diff(r.freq_mhz_actual) > 0)


@pytest.mark.qick
def test_realized_swept_gain_reflects_register_step_quantization():
    """The compiled axis is not the per-point round-trip of the requested axis.

    qick quantizes the sweep *span* and *step* in register space (truncation): 11 steps over
    0..8191 codes compile to a step of 819, so the top of the realized axis is 8190 codes —
    below the requested 8191. Snapping each requested point to the grid independently (the old
    quantize() discipline) would report the top as 8191. This is the discrepancy class flagged
    in review (PR #25).
    """
    from strumento.core.device import Device
    from strumento.core.program import StrumentoProgram
    from strumento.core.pulses import Seq
    from strumento.core.sweeps import Loop, Sweep

    dev = Device.load(_LOOPBACK)
    p = dev.qubit.ge_pi(gain=Sweep(start=0, stop=8191, on="amp"))
    r = StrumentoProgram(dev, seq=Seq().play(p).measure(), axes=[Loop("amp", 11)]).realized(p)
    assert isinstance(r.gain_frac, np.ndarray)
    assert r.gain_frac.shape == (11,)
    assert r.gain_frac[0] == pytest.approx(0.0, abs=1 / _MAXV)
    assert r.gain_frac[-1] < 8191 / _MAXV  # the compiled axis stops one register step short
    assert r.gain_frac[-1] == pytest.approx(8190 / _MAXV, abs=1.5 / _MAXV)


@pytest.mark.qick
def test_realized_rejects_a_pulse_that_is_not_in_the_program():
    from strumento.core.device import Device
    from strumento.core.program import StrumentoProgram
    from strumento.core.pulses import Seq

    dev = Device.load(_LOOPBACK)
    prog = StrumentoProgram(dev, seq=Seq().play(dev.qubit.ge_pi()).measure())
    with pytest.raises(ValueError, match="not declared"):
        prog.realized(dev.qubit.ge_hpi())
