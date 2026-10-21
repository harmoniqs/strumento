"""End to end: multi-line compile (structural) + OC arb import on real v2 assembly."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _recording import RecordingProgram  # noqa: E402
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

from strumento.core.compiler import Compiler  # noqa: E402
from strumento.core.config import apply_updates  # noqa: E402
from strumento.core.device import Device  # noqa: E402
from strumento.core.pulses import Seq, from_solution  # noqa: E402
from strumento.core.sweeps import Sweep  # noqa: E402
from strumento.core.wiring import LineRef  # noqa: E402
from strumento.packs.cqed.experiments.resonator_spectroscopy import (
    ResonatorSpectroscopy,  # noqa: E402
)
from tests._paths import DEVICES

LOOPBACK_YAML = DEVICES / "loopback_demo" / "device.yaml"


def test_multiline_ladder_compiles(dev):
    """ge_pi · ef_pi · f0g1 · swap — four generators, three envelope styles, one program."""
    seq = (
        Seq()
        .play(dev.qubit.ge_pi())
        .play(dev.qubit.ef_pi())
        .play(dev.sideband.f0g1())
        .play(dev.coupler.swap("cavity-S1"))
        .measure()
    )
    prog = RecordingProgram()
    c = Compiler(prog, dev.wiring)
    c.declare(seq)
    c.play(seq)

    declared = sorted(call[1] for call in prog.calls if call[0] == "declare_gen")
    assert declared == [1, 4, 5]                     # drive DAC1, coupler DAC4, sideband DAC5
    styles = [call[3]["style"] for call in prog.calls if call[0] == "add_pulse"]
    assert styles == ["arb", "arb", "flat_top", "flat_top"]
    flat = [call for call in prog.calls if call[0] == "add_pulse" and call[3]["style"] == "flat_top"]
    assert flat[0][3]["length"] == pytest.approx(0.8)     # f0g1 flat length
    assert flat[1][3]["length"] == pytest.approx(2.4)     # swap flat length
    kinds = prog.kinds()
    assert kinds.index("pulse") < kinds.index("trigger")


def test_resonator_spectroscopy_const_probe_compiles(dev):
    exp = ResonatorSpectroscopy(dev, freqs=Sweep(start=745.0, stop=755.0, on="f"), points=101)
    seq, _ = exp.sequence()
    prog = RecordingProgram()
    c = Compiler(prog, dev.wiring, axes=exp.axes())
    c.declare(seq)
    c.play(seq)

    ap = prog.first("add_pulse")
    assert ap[3]["style"] == "const" and "envelope" not in ap[3]
    assert ap[3]["length"] == pytest.approx(2.0)          # calibrated readout_length
    assert prog.first("add_loop")[1] == "f"               # the frequency sweep became a loop
    assert prog.first("declare_gen")[1] == 0              # the probe generator


def test_resonator_spectroscopy_binds_dynamic_adc_to_the_probe(dev):
    """The normal experiment path must emit Script 14's configure -> pulse -> trigger shape."""
    exp = ResonatorSpectroscopy(dev, points=11)
    seq, _ = exp.sequence()
    prog = RecordingProgram()
    compiler = Compiler(prog, dev.wiring, axes=exp.axes(), trig_offset=0.0)
    compiler.declare(seq)
    compiler.play(seq)

    adc = dev.wiring.readout.adc_chs[0]
    pulse = prog.first("add_pulse")
    config = prog.first("add_readoutconfig")
    declared_readout = prog.first("declare_readout")

    assert pulse[3]["ro_ch"] == adc
    assert config[1] == adc
    assert config[3]["gen_ch"] == pulse[1]
    assert declared_readout[1] == adc
    assert declared_readout[2] == pytest.approx(exp.length_us)

    runtime = [call[0] for call in prog.calls if call[0] in {
        "send_readoutconfig", "pulse", "trigger", "delay_auto",
    }]
    assert runtime[:4] == ["send_readoutconfig", "pulse", "trigger", "delay_auto"]


@pytest.mark.qick
def test_resonator_spectroscopy_exposes_compiled_program_artifacts(dev):
    from strumento.core.program import StrumentoProgram

    exp = ResonatorSpectroscopy(dev, points=11)
    seq, _ = exp.sequence()
    program = StrumentoProgram(dev, seq=seq, axes=exp.axes(), reps=1, trace=True)

    trace = program.trace()
    dump = program.dump()
    generator = dev.wiring.resolve(LineRef("readout", "probe"), dev.calib.readout.frequency)
    adc = dev.wiring.readout.adc_chs[0]

    assert trace is not None
    assert any(entry["op"] == "pulse" and entry["ch"] == generator.dac_ch for entry in trace["entries"])
    assert any(entry["op"] == "measure" and entry["adc"] == adc for entry in trace["entries"])
    assert dump["strumento"]["declared_gens"] == {generator.dac_ch: generator.nyquist}
    assert dump["strumento"]["declared_readouts"] == [adc]
    assert dump["strumento"]["declared_loops"] == ["f"]
    assert dump["qick"]["loop_dims"] == [1, 11]

    config = next(entry for entry in trace["entries"] if entry["op"] == "readout_config")
    probe = next(entry for entry in trace["entries"] if entry["op"] == "pulse")
    assert np.allclose(
        program.get_pulse_param(probe["name"], "freq", as_array=True),
        program.get_pulse_param(config["name"], "freq", as_array=True),
    )
    adc_sample_us = 1.0 / dev.soccfg_snapshot["readouts"][adc]["f_output"]
    assert dump["qick"]["ro_chs"][str(adc)]["length_us"] == pytest.approx(
        exp.length_us,
        abs=adc_sample_us,
    )

    wports = [entry["DST"] for entry in dump["qick"]["asm"] if entry["CMD"] == "WPORT_WR"]
    assert wports[:2] == [
        str(dev.soccfg_snapshot["readouts"][adc]["tproc_ctrl"]),
        str(dev.soccfg_snapshot["gens"][generator.dac_ch]["tproc_ch"]),
    ]
    runtime_ops = [entry["op"] for entry in trace["entries"] if entry["op"] in {
        "readout_config", "pulse", "measure",
    }]
    assert runtime_ops == ["readout_config", "pulse", "measure"]


@pytest.mark.qick
def test_raw_dynamic_readout_config_matches_the_probe_frequency(dev):
    from strumento.core.program import StrumentoProgram

    config, _ = apply_updates(dev.to_config(), {"readout.trig_offset": 0.0})
    probe_device = Device.from_config(config)
    frequency = probe_device.calib.readout.frequency
    probe = dataclasses.replace(probe_device.readout.probe(length=3.0), start_us=0.0)

    def configure_adc(program):
        program.add_readoutconfig(
            ch=0,
            name="matched_adc",
            freq=frequency,
            phase=0.0,
            gen_ch=0,
        )
        program.send_readoutconfig(ch=0, name="matched_adc", t=0.0)

    program = StrumentoProgram(
        probe_device,
        seq=Seq().play(probe).raw(configure_adc).measure(),
        reps=1,
        trace=True,
    )

    assert program.get_pulse_param("matched_adc", "freq") == pytest.approx(
        program.get_pulse_param("pulse0_0", "freq"),
        abs=1e-5,
    )
    assert any(entry["op"] == "raw" for entry in program.trace()["entries"])


@pytest.mark.qick
def test_raw_only_lambda_can_define_the_complete_probe_and_adc_path(dev):
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram

    frequency = dev.calib.readout.frequency
    gain = dev.calib.readout.gain / 32766

    soc = MockQickSocV2(dev.soccfg_snapshot)
    program = StrumentoProgram(
        dev,
        seq=Seq().raw(
            lambda raw: (
                raw.declare_gen(ch=0, nqz=2),
                raw.declare_readout(ch=0, length=3.0),
                raw.add_pulse(
                    ch=0,
                    name="raw_probe",
                    style="const",
                    freq=frequency,
                    phase=0.0,
                    gain=gain,
                    length=3.0,
                ),
                raw.add_readoutconfig(
                    ch=0,
                    name="raw_adc",
                    freq=frequency,
                    phase=0.0,
                    gen_ch=0,
                ),
                raw.send_readoutconfig(ch=0, name="raw_adc", t=0.0),
                raw.pulse(ch=0, name="raw_probe", t=0.0),
                raw.trigger(ros=[0], t=0.0),
                raw.delay_auto(0.0, gens=False, ros=True),
            )
        ),
        reps=1,
        soc=soc,
        trace=True,
    )
    raw_iq = program.acquire_decimated(soc, rounds=1, progress=False)

    assert program.get_pulse_param("raw_adc", "freq") == pytest.approx(
        program.get_pulse_param("raw_probe", "freq"),
        abs=1e-5,
    )
    assert np.asarray(raw_iq[0]).shape[1] == 2


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
