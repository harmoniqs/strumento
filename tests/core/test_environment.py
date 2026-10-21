"""The environment seam (D34, realizing D24) — every result carries a telemetry stamp.

Two halves, both tested here:

* the **pure** half (this file's module-level tests): the ``EnvironmentSource`` protocol, the
  three shipped sources, the coercion helper, and :func:`stamp_environment`'s invariants
  (JSON-safe, never overwrites, never raises).
* the **invariant** (``@pytest.mark.qick`` below): a ``MeasurementResult`` coming back through
  the experiment path carries ``meta["environment"]`` on **both** the mock and the simulator
  soc — the one property that is retroactively impossible to add to data already taken.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from strumento.core.environment import (
    NO_ENVIRONMENT,
    CallableEnvironment,
    EnvironmentSource,
    NullEnvironment,
    StaticEnvironment,
    environment_source,
    stamp_environment,
)
from strumento.core.result import MeasurementResult
from tests._paths import DEVICES

sys.path.insert(0, str(Path(__file__).parents[1]))
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

LOOPBACK = DEVICES / "loopback_demo" / "device.yaml"
# loopback_demo has no dedicated readout line (single-generator instance) — every AmplitudeRabi
# below probes on the qubit line instead, same as resonator_spectroscopy's own test override
# (issue #102).


def _result() -> MeasurementResult:
    return MeasurementResult(xpts=None, avgi=0.0, avgq=0.0)


# -- the sources ---------------------------------------------------------------------------


def test_static_environment_snapshots_its_readings():
    src = StaticEnvironment({"mxc_temperature_mK": 21.4, "still_K": 0.9})
    assert src.snapshot() == {"mxc_temperature_mK": 21.4, "still_K": 0.9}
    assert isinstance(src, EnvironmentSource)  # structural protocol


def test_static_environment_snapshot_is_a_copy():
    src = StaticEnvironment({"a": 1})
    src.snapshot()["a"] = 999
    assert src.snapshot() == {"a": 1}


def test_callable_environment_reads_at_snapshot_time():
    readings = [{"t": 1.0}, {"t": 2.0}]
    src = CallableEnvironment(lambda: readings.pop(0))
    assert src.snapshot() == {"t": 1.0}
    assert src.snapshot() == {"t": 2.0}  # read *at acquire time*, not at construction


def test_null_environment_is_an_empty_dict():
    assert NullEnvironment().snapshot() == {}
    assert NO_ENVIRONMENT.snapshot() == {}


def test_environment_source_coerces_the_shorthands():
    assert environment_source(None) is NO_ENVIRONMENT
    assert isinstance(environment_source({"a": 1}), StaticEnvironment)
    assert isinstance(environment_source(lambda: {"a": 1}), CallableEnvironment)
    src = StaticEnvironment({"a": 1})
    assert environment_source(src) is src  # already a source: passed through


def test_environment_source_rejects_a_non_source():
    with pytest.raises(TypeError, match="environment source"):
        environment_source(42)


# -- stamping ------------------------------------------------------------------------------


def test_stamp_writes_a_json_safe_dict():
    r = stamp_environment(_result(), {"mxc_temperature_mK": np.float64(21.4)})
    env = r.meta["environment"]
    assert isinstance(env, dict)
    assert json.loads(json.dumps(env))["readings"]["mxc_temperature_mK"] == pytest.approx(21.4)
    assert env["source"] == "StaticEnvironment"
    assert env["stamped_at"].endswith("+00:00")  # UTC, unambiguous


def test_stamp_with_no_source_still_stamps():
    """The invariant is the *key*, not the readings — an unstamped result must not exist."""
    env = stamp_environment(_result()).meta["environment"]
    assert env["readings"] == {} and env["source"] == "NullEnvironment"


def test_stamp_does_not_overwrite_an_existing_stamp():
    r = _result()
    r.meta["environment"] = {"readings": {"from": "the transport receipt"}}
    stamp_environment(r, {"mxc_temperature_mK": 21.4})
    assert r.meta["environment"] == {"readings": {"from": "the transport receipt"}}


def test_stamp_overwrite_is_explicit():
    r = _result()
    r.meta["environment"] = {"readings": {"stale": True}}
    stamp_environment(r, {"fresh": True}, overwrite=True)
    assert r.meta["environment"]["readings"] == {"fresh": True}


def test_a_failing_telemetry_read_never_loses_the_data():
    def boom():
        raise TimeoutError("fridge logger did not answer")

    env = stamp_environment(_result(), boom).meta["environment"]
    assert env["readings"] == {}
    assert "fridge logger" in env["error"]  # errors-as-data: the acquisition survives


def test_a_non_dict_reading_is_reported_not_raised():
    env = stamp_environment(_result(), lambda: 21.4).meta["environment"]
    assert env["readings"] == {} and "dict" in env["error"]


def test_stamp_carries_the_per_line_signal_chain_when_the_instance_has_one():
    from strumento.core.wiring import Generator, Line, ReadoutWiring, SignalChain, Wiring

    wiring = Wiring(
        lines={
            "qubit": Line(
                name="qubit",
                generators=[Generator(role="drive", dac_ch=0)],
                chain=SignalChain(attenuation_db=20.0, amplifier="hemt"),
            ),
            "flux": Line(name="flux", generators=[Generator(role="bias", dac_ch=1)]),
        },
        readout=ReadoutWiring(dac_ch=0, adc_chs=[0]),
    )
    device = type("Dev", (), {"wiring": wiring})()
    env = stamp_environment(_result(), device=device).meta["environment"]
    assert env["signal_chain"]["qubit"]["attenuation_db"] == 20.0
    assert "flux" not in env["signal_chain"]  # only annotated lines appear


def test_stamp_omits_signal_chain_when_no_line_is_annotated():
    from strumento.core.wiring import Generator, Line, ReadoutWiring, Wiring

    wiring = Wiring(
        lines={"qubit": Line(name="qubit", generators=[Generator(role="drive", dac_ch=0)])},
        readout=ReadoutWiring(dac_ch=0, adc_chs=[0]),
    )
    device = type("Dev", (), {"wiring": wiring})()
    assert "signal_chain" not in stamp_environment(_result(), device=device).meta["environment"]


def test_stamp_includes_a_reporting_socs_contribution():
    class Reporter:
        def environment(self):
            return {"kind": "simulator", "seed": 7}

    env = stamp_environment(_result(), soc=Reporter()).meta["environment"]
    assert env["soc"] == {"kind": "simulator", "seed": 7}


def test_a_non_reporting_soc_contributes_nothing():
    assert "soc" not in stamp_environment(_result(), soc=object()).meta["environment"]


def test_the_stamp_survives_the_wire_form():
    r = stamp_environment(_result(), {"mxc_temperature_mK": 21.4})
    wire = r.meta["environment"]
    assert MeasurementResult.from_wire(r.to_wire()).meta["environment"] == wire


# -- the invariant: every result from the mock AND the simulator path is stamped ------------


@pytest.mark.qick
@requires_v2_soccfg
def test_the_mock_path_stamps_every_result():
    from strumento.core.device import Device
    from strumento.core.mock import MockQickSocV2
    from strumento.core.sweeps import Sweep
    from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi

    dev = Device.load(LOOPBACK)
    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    outcome = AmplitudeRabi(dev, gains=Sweep(start=0.0, stop=16000.0, on="amp"), points=5,
                            line=("qubit", "drive")).go(
        soc, reps=4, progress=False, environment={"mxc_temperature_mK": 21.4}
    )
    env = outcome.result.meta["environment"]
    assert env["readings"]["mxc_temperature_mK"] == 21.4
    assert "soc" not in env  # the mock reports no environment; the stamp still exists
    json.dumps(env)


@pytest.mark.qick
@requires_v2_soccfg
def test_the_simulator_path_stamps_a_synthetic_environment():
    from strumento.core.device import Device
    from strumento.core.simulator import SimulatorSoc
    from strumento.core.sweeps import Sweep
    from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi
    from strumento.packs.cqed.responses import truth_from_device

    dev = Device.load(LOOPBACK)
    truth = truth_from_device(dev, pi_gain=9350.0)
    soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth=truth, seed=11)
    outcome = AmplitudeRabi(dev, gains=Sweep(start=0.0, stop=16000.0, on="amp"), points=5,
                            line=("qubit", "drive")).go(
        soc, reps=8, progress=False
    )
    env = outcome.result.meta["environment"]
    assert env["soc"]["simulated"] is True and env["soc"]["seed"] == 11
    # provenance, not an oracle: the truth's KEYS are recorded, never its values (a closed-loop
    # agent must measure the truth, not read it off the result's provenance)
    assert "pi_gain" in env["soc"]["truth_keys"]
    assert "9350" not in json.dumps(env)
    json.dumps(env)


@pytest.mark.qick
@requires_v2_soccfg
def test_a_default_runner_stamps_even_with_no_telemetry_configured():
    from strumento.core.device import Device
    from strumento.core.mock import MockQickSocV2
    from strumento.core.sweeps import Sweep
    from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi

    dev = Device.load(LOOPBACK)
    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    outcome = AmplitudeRabi(dev, gains=Sweep(start=0.0, stop=16000.0, on="amp"), points=5,
                            line=("qubit", "drive")).go(
        soc, reps=4, progress=False
    )
    assert isinstance(outcome.result.meta["environment"], dict)


@pytest.mark.qick
@requires_v2_soccfg
def test_a_runner_level_source_is_used_for_every_run():
    from strumento.core.device import Device
    from strumento.core.experiment import Runner
    from strumento.core.mock import MockQickSocV2
    from strumento.core.sweeps import Sweep
    from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi

    dev = Device.load(LOOPBACK)
    runner = Runner(environment=StaticEnvironment({"station": "bench-1"}))
    exp = AmplitudeRabi(dev, gains=Sweep(start=0.0, stop=16000.0, on="amp"), points=5,
                        line=("qubit", "drive"), runner=runner)
    outcome = exp.go(MockQickSocV2.from_snapshot(V2_SOCCFG), reps=4, progress=False)
    assert outcome.result.meta["environment"]["readings"]["station"] == "bench-1"


# -- purity ---------------------------------------------------------------------------------


def test_module_imports_without_qick():
    import subprocess

    code = "import sys; sys.modules['qick']=None; import strumento.core.environment; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
