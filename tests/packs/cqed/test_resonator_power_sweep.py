"""Framework-native resonator power-map behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qick")

from strumento.core.datachest import FileDataChest
from strumento.core.device import Device
from strumento.core.program import StrumentoProgram
from strumento.core.pulses import Pulse
from strumento.core.record import TaskRecord, read_events
from strumento.core.simulator import SimulatorSoc
from strumento.packs.cqed import EXPERIMENTS
from strumento.packs.cqed.experiments.resonator_power_sweep import (
    DEFAULT_FINAL_DELAY_US,
    DEFAULT_REPS,
    ResonatorPowerSweep,
)
from strumento.packs.cqed.responses import truth_from_device

ROOT = Path(__file__).parents[3]
DEVICE_YAML = ROOT / "devices" / "multimode_demo" / "device.yaml"
SOCCFG = ROOT / "devices" / "_fixtures" / "soccfg_v2_cqed.json"


def test_power_map_is_registered_without_redefining_punchout():
    """Hosts can name the two-dimensional validation map separately from Punchout."""
    assert EXPERIMENTS["resonator_power_sweep"] is ResonatorPowerSweep
    assert EXPERIMENTS["punchout"] is not ResonatorPowerSweep


@pytest.mark.qick
def test_power_map_defaults_match_the_reference_grid_and_dynamic_readout_path():
    """The framework row carries the same validation grid through matched DAC/ADC lowering."""
    device = Device.load(DEVICE_YAML)
    exp = ResonatorPowerSweep(device)
    seq, _ = exp.sequence()
    probe = next(op for op in seq.ops if isinstance(op, Pulse))
    program = StrumentoProgram(
        device,
        seq=seq,
        axes=exp.axes(),
        reps=50,
        final_delay=5.0,
        soc=SimulatorSoc.from_snapshot(SOCCFG, truth=truth_from_device(device), seed=7),
        trace=True,
    )

    assert exp.gain_codes == tuple(range(0, 1201, 100))
    assert exp.points == 61
    assert exp.freqs.start == pytest.approx(device.calib.readout.frequency - 5.0)
    assert exp.freqs.stop == pytest.approx(device.calib.readout.frequency + 5.0)
    assert probe.gain == pytest.approx(device.calib.readout.gain / 32766.0)

    realized = program.realized(probe)
    assert int(round(float(realized.gain_frac) * 32766.0)) == device.calib.readout.gain
    assert np.asarray(realized.freq_mhz_actual).shape == (61,)

    trace = program.trace()
    assert trace is not None
    config = next(entry for entry in trace["entries"] if entry["op"] == "readout_config")
    pulse = next(entry for entry in trace["entries"] if entry["op"] == "pulse")
    assert np.allclose(
        program.get_pulse_param(config["name"], "freq", as_array=True),
        program.get_pulse_param(pulse["name"], "freq", as_array=True),
        atol=1e-9,
    )


@pytest.mark.qick
def test_power_map_realizes_every_requested_gain_code_exactly():
    """The validation map does not relabel a rounded or fraction-domain gain grid as codes."""
    device = Device.load(DEVICE_YAML)
    exp = ResonatorPowerSweep(device)
    soc = SimulatorSoc.from_snapshot(SOCCFG, truth=truth_from_device(device), seed=7)

    realized_codes = []
    for gain_code in exp.gain_codes:
        sequence, _ = exp._row_sequence(gain_code)
        probe = next(op for op in sequence.ops if isinstance(op, Pulse))
        program = StrumentoProgram(
            device,
            seq=sequence,
            axes=exp.axes(),
            reps=DEFAULT_REPS,
            final_delay=DEFAULT_FINAL_DELAY_US,
            soc=soc,
        )
        realized_codes.append(int(round(float(program.realized(probe).gain_frac) * 32766.0)))

    assert realized_codes == list(exp.gain_codes)


@pytest.mark.qick
def test_power_map_rejects_a_row_that_exceeds_adc_accumulator_capacity():
    """Host batching never turns an unsafe row into a partial map."""
    device = Device.load(DEVICE_YAML)
    soc = SimulatorSoc.from_snapshot(SOCCFG, truth=truth_from_device(device), seed=7)

    with pytest.raises(ValueError, match=r"one gain row needs 8235 accumulated points"):
        ResonatorPowerSweep(device).go(soc, reps=135, progress=False)


@pytest.mark.qick
def test_power_map_records_one_aggregate_acquisition(tmp_path, closing):
    """Task progress reflects the complete map rather than a misleading single row."""
    device = Device.load(DEVICE_YAML)
    soc = SimulatorSoc.from_snapshot(SOCCFG, truth=truth_from_device(device), seed=7)
    record = closing(TaskRecord.create(
        str(tmp_path / "resonator-power-map"),
        title="resonator power map",
        kind="experiment",
        pid=0,
        heartbeat=False,
    ))

    ResonatorPowerSweep(device, gain_codes=(0, 1000), points=11).go(
        soc,
        reps=10,
        record=record,
        progress=False,
    )

    acquired = [
        event
        for event in read_events(record.dir)
        if event.get("ev") == "progress" and event.get("step") == 1
    ]
    assert len(acquired) == 1
    assert acquired[0]["metrics"]["points"] == 22


@pytest.mark.qick
def test_power_map_display_is_a_frequency_by_gain_magnitude_map():
    """The normal display makes the validation surface inspectable without a scratch script."""
    pytest.importorskip("matplotlib")  # display layer is optional; no figure without it
    device = Device.load(DEVICE_YAML)
    soc = SimulatorSoc.from_snapshot(SOCCFG, truth=truth_from_device(device), seed=7)

    outcome = ResonatorPowerSweep(device, gain_codes=(0, 1000), points=11).go(
        soc,
        reps=10,
        progress=False,
        display=True,
    )

    assert outcome.figure is not None
    axis = outcome.figure.axes[0]
    assert axis.get_xlabel() == "Frequency (MHz)"
    assert axis.get_ylabel() == "Probe gain (QICK code)"
    assert axis.collections


@pytest.mark.qick
def test_power_map_runs_through_go_and_preserves_gain_rows():
    """A caller receives one ordered I/Q map, not thirteen unrelated row outcomes."""
    device = Device.load(DEVICE_YAML)
    soc = SimulatorSoc.from_snapshot(
        SOCCFG,
        truth=truth_from_device(device, resonator_f0=7000.08),
        seed=7,
    )

    outcome = ResonatorPowerSweep(device, gain_codes=(0, 1000), points=11).go(
        soc,
        reps=10,
        progress=False,
    )

    assert outcome.result.avgi.shape == (2, 11)
    assert outcome.result.avgq.shape == (2, 11)
    assert outcome.result.meta["gain_codes"] == [0, 1000]
    assert np.asarray(outcome.result.xpts).shape == (11,)
    assert outcome.fit is not None and outcome.fit.success


@pytest.mark.qick
def test_default_power_map_reproduces_the_full_direct_reference_grid():
    """The public default is the 13-by-61, 50-repetition validation measurement."""
    device = Device.load(DEVICE_YAML)
    soc = SimulatorSoc.from_snapshot(
        SOCCFG,
        truth=truth_from_device(device, resonator_f0=7000.08),
        seed=7,
    )

    outcome = ResonatorPowerSweep(device).go(soc, progress=False)
    magnitude = np.hypot(outcome.result.avgi, outcome.result.avgq)
    reference_row = outcome.result.meta["gain_codes"].index(device.calib.readout.gain)

    assert outcome.result.avgi.shape == (13, 61)
    assert outcome.result.meta["gain_codes"] == list(range(0, 1201, 100))
    assert outcome.result.meta["acquisition_points"] == 793
    assert outcome.result.meta["reps_per_row"] == 50
    assert outcome.result.meta["final_delay_us"] == pytest.approx(5.0)
    assert magnitude[0].mean() < magnitude[reference_row].mean()
    assert outcome.fit is not None and outcome.fit.success


@pytest.mark.qick
def test_power_map_round_trips_through_the_datachest(tmp_path):
    """The merged map is directly comparable with the self-contained direct-QICK artifact."""
    device = Device.load(DEVICE_YAML)
    soc = SimulatorSoc.from_snapshot(SOCCFG, truth=truth_from_device(device), seed=7)
    result = ResonatorPowerSweep(device, gain_codes=(0, 1000), points=11).go(
        soc,
        reps=10,
        progress=False,
    ).result

    chest = FileDataChest(tmp_path / "power-map")
    entry_id = chest.save(result, experiment="resonator_power_sweep")
    restored = chest.load(entry_id)["result"]

    assert restored.avgi.shape == (2, 11)
    assert restored.avgq.shape == (2, 11)
    assert restored.meta["gain_codes"] == [0, 1000]
