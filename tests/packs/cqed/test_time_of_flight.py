"""Time-of-flight bring-up routine (issue #104) — host-driven, not an ``Experiment``.

Covered here, per acceptance criterion:

- AC 1 — ``run()`` plays the probe and acquires decimated over a configurable window.
- AC 2 — ``analyze()`` returns a typed ``FitResult`` with ``derived["tof_us"]`` (the fitter
  itself is covered in ``tests/core/test_fitters.py``; this file covers the integration).
- AC 3 — ``TimeOfFlight`` is not an ``Experiment`` subclass and is not registered in any pack's
  ``EXPERIMENTS``.
- AC 4 — ``write_back()`` stores the measured value into ``calib.readout.trig_offset`` via the
  existing calibration-store record pattern (reload-after-write).

Structural/purity checks are plain (no qick); anything that builds a program or acquires needs
qick and is marked accordingly.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from scipy.special import erf

from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult
from strumento.packs.cqed.experiments.time_of_flight import TimeOfFlight
from tests._paths import DEVICES

CQED_SOCCFG = DEVICES / "_fixtures" / "soccfg_v2_cqed.json"


# ──────────────────────────────────────────────────────────────────────────────
# Structure (AC 3) — pure, no qick
# ──────────────────────────────────────────────────────────────────────────────


def test_time_of_flight_is_not_an_experiment_subclass():
    from strumento.core.experiment import Experiment

    assert not issubclass(TimeOfFlight, Experiment)


def test_time_of_flight_is_not_registered_in_any_pack_experiments_dict():
    from strumento.packs.cqed import EXPERIMENTS as CQED_EXPERIMENTS
    from strumento.packs.spin import EXPERIMENTS as SPIN_EXPERIMENTS
    from strumento.packs.tweezer import EXPERIMENTS as TWEEZER_EXPERIMENTS

    assert TimeOfFlight not in CQED_EXPERIMENTS.values()
    assert TimeOfFlight not in SPIN_EXPERIMENTS.values()
    assert TimeOfFlight not in TWEEZER_EXPERIMENTS.values()


def test_module_imports_without_qick():
    """StrumentoProgram (qick-facing) is a local import inside run() — exactly like
    core/experiment.py's Runner (hard rule 1) — so importing this module never needs qick."""
    code = (
        "import sys; sys.modules['qick']=None; "
        "import strumento.packs.cqed.experiments.time_of_flight; print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_constructs_with_defaults(dev):
    tof = TimeOfFlight(dev)
    assert tof.device is dev
    assert tof.window_us > 0
    assert tof.pre_trigger_us == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# run(): plays the probe, acquires decimated over a configurable window (AC 1)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.qick
def test_run_returns_a_decimated_window_spanning_the_configured_length(dev):
    from strumento.core.mock import MockQickSocV2

    soc = MockQickSocV2.from_snapshot(CQED_SOCCFG)
    tof = TimeOfFlight(dev, window_us=4.0, reps=1, rounds=1)
    result = tof.run(soc, progress=False)

    assert isinstance(result, MeasurementResult)
    assert result.xpts[0] == pytest.approx(0.0, abs=1e-6)
    # register-quantized (D37): the realized window may fall a bit short of the requested one,
    # but must span the large majority of it, not some unrelated (e.g. default) length.
    assert 4.0 * 0.9 <= result.xpts[-1] <= 4.0 + 1e-6
    assert len(result.xpts) == len(result.avgi) == len(result.avgq)


@pytest.mark.qick
def test_run_window_is_configurable(dev):
    from strumento.core.mock import MockQickSocV2

    soc = MockQickSocV2.from_snapshot(CQED_SOCCFG)
    short = TimeOfFlight(dev, window_us=1.0, reps=1, rounds=1).run(soc, progress=False)
    long = TimeOfFlight(dev, window_us=6.0, reps=1, rounds=1).run(soc, progress=False)

    assert len(long.xpts) > len(short.xpts) * 3  # ~6x the window -> ~6x the samples


@pytest.mark.qick
def test_run_accepts_a_separate_compile_soccfg(dev):
    """A remote runtime soc may be a Pyro proxy, while v2 assembly needs a local config."""
    from strumento.core.mock import MockQickSocV2

    compile_soccfg = MockQickSocV2.from_snapshot(CQED_SOCCFG)
    runtime_soc = MockQickSocV2.from_snapshot(CQED_SOCCFG)

    result = TimeOfFlight(dev, window_us=3.0, reps=1, rounds=1).run(
        runtime_soc,
        compile_soccfg=compile_soccfg,
        progress=False,
    )

    assert isinstance(result, MeasurementResult)
    assert len(result.xpts) == len(result.avgi) == len(result.avgq)


@pytest.mark.qick
def test_run_does_not_mutate_the_caller_s_device(dev):
    """The trig_offset=0 / widened-window overrides live on an internal clone — the caller's
    device (and its calibration belief) must come back untouched."""
    from strumento.core.mock import MockQickSocV2

    original_trig_offset = dev.calib.readout.trig_offset
    original_lengths_us = dev.wiring.readout.lengths_us
    soc = MockQickSocV2.from_snapshot(CQED_SOCCFG)

    TimeOfFlight(dev, window_us=2.0, reps=1, rounds=1).run(soc, progress=False)

    assert dev.calib.readout.trig_offset == original_trig_offset
    assert dev.wiring.readout.lengths_us == original_lengths_us


@pytest.mark.qick
def test_run_shape_agrees_between_mock_and_simulator(dev):
    """Guard-5-style (tests/packs/test_experiment_contract.py): the mock's zero-IQ contract is
    deliberate, but the two socs must agree on shape."""
    from strumento.core.mock import MockQickSocV2
    from strumento.core.simulator import SimulatorSoc

    mock_result = TimeOfFlight(dev, window_us=3.0, reps=1, rounds=1).run(
        MockQickSocV2.from_snapshot(CQED_SOCCFG), progress=False
    )
    sim_result = TimeOfFlight(dev, window_us=3.0, reps=1, rounds=1).run(
        SimulatorSoc.from_snapshot(CQED_SOCCFG), progress=False
    )

    assert mock_result.avgi.shape == sim_result.avgi.shape
    assert mock_result.avgq.shape == sim_result.avgq.shape
    assert len(mock_result.xpts) == len(sim_result.xpts)


@pytest.mark.qick
def test_run_triggers_the_adc_with_the_probe_not_after_it(dev):
    """The whole point of a ToF bring-up: the ADC must start listening when the probe STARTS
    driving, not after it finishes. A sequential ``.play(probe).measure()`` would advance the
    cursor past the whole probe first (docs/timeline.md) and fire the trigger only afterwards —
    which is exactly the bug this routine's ``start_us=0.0`` placement avoids (see the module
    docstring). Pinned directly on the compiled trace so a regression here fails loudly instead
    of just producing data nobody would notice is wrong offline (the mock returns zeros either
    way)."""
    from strumento.core.mock import MockQickSocV2

    soc = MockQickSocV2.from_snapshot(CQED_SOCCFG)
    tof = TimeOfFlight(dev, window_us=2.0, pre_trigger_us=0.0, reps=1, rounds=1)
    result = tof.run(soc, progress=False, trace=True)

    entries = result.meta["trace"]["entries"]
    play_entries = [e for e in entries if e["phase"] == "play"]
    pulse_entry = next(e for e in play_entries if e["op"] == "pulse")
    measure_entry = next(e for e in play_entries if e["op"] == "measure")

    assert pulse_entry["t"] == 0.0
    assert measure_entry["t"] == 0.0
    # No delay_auto between them: the cursor never advanced past the probe before the trigger
    # fired — that absence is what makes the two simultaneous rather than sequential.
    assert [e["op"] for e in play_entries] == ["readout_config", "pulse", "measure"]


@pytest.mark.qick
def test_run_honors_a_nonzero_pre_trigger_us(dev):
    from strumento.core.mock import MockQickSocV2

    soc = MockQickSocV2.from_snapshot(CQED_SOCCFG)
    tof = TimeOfFlight(dev, window_us=2.0, pre_trigger_us=0.05, reps=1, rounds=1)
    result = tof.run(soc, progress=False, trace=True)

    measure_entry = next(e for e in result.meta["trace"]["entries"] if e["op"] == "measure")
    assert measure_entry["t"] == pytest.approx(0.05)


# ──────────────────────────────────────────────────────────────────────────────
# analyze(): a typed FitResult with derived["tof_us"] (AC 2 integration)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.qick
def test_analyze_returns_a_typed_fit_result_with_tof_us(dev):
    from strumento.core.mock import MockQickSocV2

    soc = MockQickSocV2.from_snapshot(CQED_SOCCFG)
    tof = TimeOfFlight(dev, window_us=3.0, reps=1, rounds=1)
    result = tof.run(soc, progress=False)
    fit = tof.analyze(result)

    assert isinstance(fit, FitResult)
    assert "tof_us" in fit.derived
    assert np.isfinite(fit.derived["tof_us"])


@pytest.mark.qick
def test_run_then_analyze_recovers_a_known_arrival_edge(dev):
    """End to end on run()'s own realized time axis: a hand-built trace with a KNOWN edge must
    be recovered by analyze(), not just produce a well-shaped but meaningless result."""
    from strumento.core.mock import MockQickSocV2

    soc = MockQickSocV2.from_snapshot(CQED_SOCCFG)
    tof = TimeOfFlight(dev, window_us=3.0, reps=1, rounds=1)
    shape_probe = tof.run(soc, progress=False)  # the real, register-realized time axis

    true_tof_us = 0.4
    x = np.asarray(shape_probe.xpts, dtype=float)
    magnitude = 50.0 + 900.0 * 0.5 * (1.0 + erf((x - true_tof_us) / (np.sqrt(2.0) * 0.03)))
    rng = np.random.default_rng(0)
    synthetic = MeasurementResult(
        xpts=x,
        avgi=magnitude + rng.normal(0, 10.0, x.size),
        avgq=rng.normal(0, 10.0, x.size),
    )

    fit = tof.analyze(synthetic)

    assert fit.success
    assert fit.derived["tof_us"] == pytest.approx(true_tof_us, abs=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# write_back(): calib.readout.trig_offset, reload-after-write (AC 4) — pure, no qick
# ──────────────────────────────────────────────────────────────────────────────


def test_write_back_updates_trig_offset_and_reloads_the_device(tmp_path, dev):
    from strumento.core.store import FileConfigStore

    store = FileConfigStore(tmp_path / "store")
    tof = TimeOfFlight(dev, window_us=3.0)
    fit = FitResult(params={}, stderr={}, derived={"tof_us": 0.417}, success=True)

    new_device = tof.write_back(fit, store, "main")

    assert new_device.calib.readout.trig_offset == pytest.approx(0.417)
    # the reload-after-write contract: self.device now IS the new belief.
    assert tof.device is new_device
    assert tof.device.calib.readout.trig_offset == pytest.approx(0.417)


def test_write_back_does_not_mutate_the_original_device_object(tmp_path, dev):
    """record_calibration never mutates in place — the OLD device object (if a caller kept a
    reference to it) must still read the pre-write-back value."""
    from strumento.core.store import FileConfigStore

    store = FileConfigStore(tmp_path / "store")
    original_trig_offset = dev.calib.readout.trig_offset
    tof = TimeOfFlight(dev, window_us=3.0)
    fit = FitResult(params={}, stderr={}, derived={"tof_us": 0.417}, success=True)

    tof.write_back(fit, store, "main")

    assert dev.calib.readout.trig_offset == original_trig_offset
    assert tof.device is not dev


def test_write_back_advances_the_store_ref_and_logs_provenance(tmp_path, dev):
    from strumento.core.store import FileConfigStore

    store = FileConfigStore(tmp_path / "store")
    tof = TimeOfFlight(dev, window_us=3.0)
    fit = FitResult(params={}, stderr={}, derived={"tof_us": 0.417}, success=True)

    tof.write_back(fit, store, "main", source="time_of_flight bring-up")

    log = store.log("main")
    assert len(log) == 1
    assert log[0].source == "time_of_flight bring-up"
    assert log[0].updates["readout.trig_offset"] == [dev.calib.readout.trig_offset, 0.417]
    assert store.resolve_ref("main") is not None


def test_write_back_defaults_a_source_label_when_none_given(tmp_path, dev):
    from strumento.core.store import FileConfigStore

    store = FileConfigStore(tmp_path / "store")
    tof = TimeOfFlight(dev, window_us=3.0)
    fit = FitResult(params={}, stderr={}, derived={"tof_us": 0.5}, success=True)

    tof.write_back(fit, store, "main")

    assert store.log("main")[0].source  # non-empty, without the caller having to supply one


# ──────────────────────────────────────────────────────────────────────────────
# The full pipeline, chained (measure -> fit -> believe)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.qick
def test_full_pipeline_run_analyze_write_back(tmp_path, dev):
    from strumento.core.mock import MockQickSocV2
    from strumento.core.store import FileConfigStore

    soc = MockQickSocV2.from_snapshot(CQED_SOCCFG)
    store = FileConfigStore(tmp_path / "store")
    tof = TimeOfFlight(dev, window_us=3.0, reps=1, rounds=1)

    result = tof.run(soc, progress=False)
    fit = tof.analyze(result)
    new_device = tof.write_back(fit, store, "main")

    assert new_device.calib.readout.trig_offset == pytest.approx(fit.derived["tof_us"])
    assert tof.device is new_device
