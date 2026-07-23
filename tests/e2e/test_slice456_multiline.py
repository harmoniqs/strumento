"""Slices 4–6 e2e — multi-line compile (structural) + OC arb import on real v2 assembly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _recording import RecordingProgram  # noqa: E402
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

from strumento.core.compiler import Compiler  # noqa: E402
from strumento.core.device import Device  # noqa: E402
from strumento.core.pulses import Seq, from_solution  # noqa: E402
from strumento.core.sweeps import Sweep  # noqa: E402
from strumento.core.wiring import LineRef  # noqa: E402
from strumento.packs.cqed.experiments.resonator_spectroscopy import (
    ResonatorSpectroscopy,  # noqa: E402
)

YAML = Path(__file__).parents[2] / "devices" / "multimode_demo" / "device.yaml"
LOOPBACK_YAML = Path(__file__).parents[2] / "devices" / "loopback_demo" / "device.yaml"


@pytest.fixture
def dev():
    return Device.load(YAML)


def test_slice4_multiline_ladder_compiles(dev):
    """ge_pi · ef_pi · f0g1 · swap — four generators, three envelope styles, one program."""
    seq = (
        Seq()
        .play(dev.qubit.ge_pi())
        .play(dev.qubit.ef_pi())
        .play(dev.sideband.f0g1())
        .play(dev.coupler.swap("manipulate-S1"))
        .measure()
    )
    prog = RecordingProgram()
    c = Compiler(prog, dev.wiring)
    c.declare(seq)
    c.play(seq)

    declared = sorted(call[1] for call in prog.calls if call[0] == "declare_gen")
    assert declared == [0, 1, 2]                     # drive DAC2, sideband DAC0, coupler DAC1
    styles = [call[3]["style"] for call in prog.calls if call[0] == "add_pulse"]
    assert styles == ["arb", "arb", "flat_top", "flat_top"]
    flat = [call for call in prog.calls if call[0] == "add_pulse" and call[3]["style"] == "flat_top"]
    assert flat[0][3]["length"] == pytest.approx(0.8)     # f0g1 flat length
    assert flat[1][3]["length"] == pytest.approx(2.4)     # swap flat length
    kinds = prog.kinds()
    assert kinds.index("pulse") < kinds.index("trigger")


def test_resonator_spectroscopy_const_probe_compiles(dev):
    exp = ResonatorSpectroscopy(dev, freqs=Sweep("f", 745.0, 755.0, 101))
    seq, _ = exp.sequence()
    prog = RecordingProgram()
    c = Compiler(prog, dev.wiring)
    c.declare(seq)
    c.play(seq)

    ap = prog.first("add_pulse")
    assert ap[3]["style"] == "const" and "envelope" not in ap[3]
    assert ap[3]["length"] == pytest.approx(2.0)          # calibrated readout_length
    assert prog.first("add_loop")[1] == "f"               # the frequency sweep became a loop
    assert prog.first("declare_gen")[1] == 5              # the probe generator


def test_resonator_spectroscopy_analyzers(dev):
    from strumento.core.result import MeasurementResult

    f = np.linspace(745.0, 755.0, 201)
    f0, kappa = 750.0, 0.8
    mag = 1.0 - 0.9 / (1.0 + ((f - f0) / (kappa / 2)) ** 2)
    fit = ResonatorSpectroscopy(dev).analyze(MeasurementResult(xpts=f, avgi=mag, avgq=np.zeros_like(f)))
    assert fit.success
    assert fit.derived["f0"] == pytest.approx(f0, abs=0.05)


@pytest.mark.qick
@requires_v2_soccfg
def test_oc_import_assembles_on_real_v2():
    """from_solution -> Arb envelope -> real AveragerProgramV2 build + acquire (loopback)."""
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram
    from strumento.core.result import MeasurementResult

    dev = Device.load(LOOPBACK_YAML)
    t = np.linspace(0.0, 0.5, 201)
    solution = {"times": t, "controls": {"qubit": 5.0 * np.sin(np.pi * t / t[-1]) ** 2}}
    seq = from_solution(
        solution,
        drive_map={"qubit": (LineRef("qubit", "drive"), 4000.0)},
        gain_calib=lambda amp_mhz: amp_mhz / 20.0,
        dac_rate=1000.0,
    ).measure()

    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    res = StrumentoProgram(dev, seq=seq, soc=soc).acquire(soc, progress=False)
    assert isinstance(res, MeasurementResult)
