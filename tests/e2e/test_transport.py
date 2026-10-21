"""**The transport gate**: the transport-backed run path, end to end against a reference execute-agent.

``tests/core/test_transport.py`` covers the wire *contract* against a hand-written fake. This file
covers the *path*: a real program, really compiled off-board, extracted into a
:class:`~strumento.core.transport.CompiledJob`, shipped through
:class:`~strumento.core.transport.JobServerSoc` to the reference agent in ``examples/jobserver/``,
reconstructed there with qick's own ``load_prog``, acquired, and returned as a stamped
:class:`~strumento.core.result.MeasurementResult`.

Three claims are load-bearing:

1. **submit → poll → stamped result.** The transport path returns the same object the in-process path
   returns, carrying ``meta["environment"]`` written *at receipt* (D34) — the board never stamps.
2. **The payload survives the wire byte-exactly.** ``json.dumps`` → ``json.loads`` →
   ``from_wire`` → ``to_wire`` → ``json.dumps`` is byte-identical, including the envelope arrays. If
   that were not true the board would be loading a different program than the one compiled.
3. **The agent really runs what it was sent.** With a simulator behind the wire, a Rabi shipped as a
   job comes back as data whose fit recovers the simulator's perturbed ground truth. Zeros could not
   do that; neither could a payload that lost its waveforms.

Everything here is board-free: the committed testbench soccfg backs the single-channel case, and the
agent runs both in-process and as a real subprocess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qick")

_TESTS = Path(__file__).parents[1]
_REPO = _TESTS.parent
sys.path.insert(0, str(_TESTS))
sys.path.insert(0, str(_REPO))          # examples/ is deployment, not an installed package
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

from examples.jobserver.server import (  # noqa: E402
    InProcessClient,
    JobServer,
    SubprocessClient,
    build_soc,
)
from strumento.core.environment import stamp_environment  # noqa: E402
from strumento.core.experiment import Runner  # noqa: E402
from strumento.core.fitters import fit_rabi  # noqa: E402
from strumento.core.mock import MockQickSocV2  # noqa: E402
from strumento.core.pulses import Seq  # noqa: E402
from strumento.core.response import SimContext  # noqa: E402
from strumento.core.sweeps import Loop, Sweep  # noqa: E402
from strumento.core.transport import (  # noqa: E402
    CompiledJob,
    JobError,
    JobServerSoc,
    RawAcquisition,
    TransportRunner,
    run_transport,
)
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi  # noqa: E402
from strumento.packs.cqed.experiments.qubit_spectroscopy import QubitSpectroscopy  # noqa: E402
from strumento.packs.cqed.experiments.t1 import T1  # noqa: E402
from strumento.packs.cqed.readout import DispersiveIQ  # noqa: E402
from strumento.packs.cqed.responses import truth_from_device  # noqa: E402

pytestmark = [pytest.mark.qick, requires_v2_soccfg]

LOOPBACK = _REPO / "devices" / "loopback_demo" / "device.yaml"

#: the same perturbed truth the fit-recovery gate uses (belief 8192), so a recovery here means the same thing.
TRUE_PI_GAIN = 9350.0
AMP = Loop("amp", 61)  # the declared axis the Rabi gain sweep rides (D36)
GAINS = Sweep(start=0.0, stop=16380.0, on="amp")  # divides evenly by 60 (points-1); just under
                                                  # the MAX_QUBIT_DRIVE_GAIN=0.5 ceiling (16383.0)
GAINS_XPTS = GAINS.axis_values(AMP.points)  # the lab-native axis the sweep requests
REPS = 500


@pytest.fixture
def dev(loopback_dev):
    return loopback_dev


def _rabi_seq(dev, gains=GAINS):
    return Seq().play(dev.qubit.ge_pi(gain=gains)).measure()


def _mock_server():
    return JobServer(build_soc(V2_SOCCFG, "mock"))


def _simulator_server(dev, gains=GAINS):
    """A board-side simulator, armed by whoever owns it (the escape hatch — see build_soc's docs)."""
    from strumento.core.simulator import SimulatorSoc

    soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth=truth_from_device(dev, pi_gain=TRUE_PI_GAIN))
    soc.arm(SimContext(experiment=AmplitudeRabi(dev, gains=gains, points=AMP.points), device=dev,
                       xpts=gains.axis_values(AMP.points)))
    return JobServer(soc)


def _rel(got: float, want: float) -> float:
    return abs(float(got) / float(want) - 1.0)


# -- claim 1: submit -> poll -> a stamped MeasurementResult ---------------------------------


def test_the_transport_path_returns_a_stamped_measurement_result(dev):
    soc = JobServerSoc(InProcessClient(_mock_server()))
    result = run_transport(
        dev, _rabi_seq(dev), transport=soc, axes=[AMP], reps=8,
        environment={"fridge_mK": 11.4, "bench": "offline"},
    )

    assert np.shape(result.avgi) == (1, 61)      # (reads per shot, sweep points)
    assert np.allclose(result.xpts, GAINS_XPTS)  # the axis the sweep declared, not a re-derivation

    stamp = result.meta["environment"]           # D34: stamped client-side, at receipt
    assert stamp["readings"] == {"fridge_mK": 11.4, "bench": "offline"}
    assert stamp["stamped_at"] and stamp["source"] == "StaticEnvironment"
    assert "soc" not in stamp                    # the board is remote: it has no reachable reporter

    job = result.meta["transport"]               # provenance: what was actually shipped
    assert job["kind"] == "JobServerSoc" and job["reps"] == 8
    assert job["expts"] == 61 and job["reads_per_shot"] == [1]


def test_the_receipt_stamp_is_never_overwritten_by_a_later_layer(dev):
    """Whoever received the data first was closest to the station (D34's never-overwrite rule)."""
    soc = JobServerSoc(InProcessClient(_mock_server()))
    result = run_transport(dev, _rabi_seq(dev), transport=soc, axes=[AMP], reps=4,
                           environment={"fridge_mK": 11.4})
    first = result.meta["environment"]

    stamp_environment(result, {"fridge_mK": 999.0}, device=dev)
    assert result.meta["environment"] is first
    assert result.meta["environment"]["readings"] == {"fridge_mK": 11.4}


def test_an_unstamped_source_still_produces_a_stamp(dev):
    """No configured telemetry is still a stamp — an unstamped result cannot come out of this path."""
    result = run_transport(dev, _rabi_seq(dev), axes=[AMP],
                           transport=JobServerSoc(InProcessClient(_mock_server())), reps=4)
    assert result.meta["environment"]["source"] == "NullEnvironment"
    assert result.meta["environment"]["readings"] == {}


# -- claim 2: the payload survives the wire byte-exactly ------------------------------------


def test_the_wire_payloads_survive_serialize_deserialize_byte_exactly(dev):
    from strumento.core.program import StrumentoProgram

    prog = StrumentoProgram(dev, seq=_rabi_seq(dev), axes=[AMP], reps=13,
                            soc=build_soc(V2_SOCCFG, "mock"))
    job = prog.to_compiled_job(overlay_id="cqed-loopback-v2", soft_avgs=3)

    sent = json.dumps(job.to_wire(), sort_keys=True)
    received = CompiledJob.from_wire(json.loads(sent))
    assert json.dumps(received.to_wire(), sort_keys=True) == sent

    # the envelope arrays are the part a naive encoder would lose (base64 + shape + dtype)
    envelopes = job.program["envelopes"]
    assert any(env["envs"] for env in envelopes), "no envelope survived the extraction"
    assert received.acquire == {"reps": 13, "soft_avgs": 3, "ro_chs": [0],
                                "reads_per_shot": [1], "expts": 61}

    raw = RawAcquisition(iq=[np.zeros((1, 61, 2), dtype=np.int64)])
    raw_sent = json.dumps(raw.to_wire(), sort_keys=True)
    assert json.dumps(RawAcquisition.from_wire(json.loads(raw_sent)).to_wire(),
                      sort_keys=True) == raw_sent


def test_the_extracted_job_carries_qicks_own_program_form(dev):
    """The payload is ``dump_prog()``: the board agent's executor is ``load_prog`` and nothing else."""
    from strumento.core.program import StrumentoProgram

    prog = StrumentoProgram(dev, seq=_rabi_seq(dev), axes=[AMP], reps=5,
                            soc=build_soc(V2_SOCCFG, "mock"))
    payload = prog.to_compiled_job().program
    assert set(payload) == {"envelopes", "ro_chs", "gen_chs", "counter_addr", "loop_dims",
                            "avg_level", "waves", "prog_list", "labels"}
    assert payload["loop_dims"] == [5, 61] and payload["avg_level"] == 0
    assert payload["prog_list"], "an empty ASM listing means nothing would run"


# -- claim 3: the agent really runs the program it was sent ---------------------------------


def test_a_rabi_shipped_over_the_wire_recovers_the_simulators_truth(dev):
    """The gate's sharpest end: the fit through the transport is the fit through a local soc."""
    soc = JobServerSoc(InProcessClient(_simulator_server(dev)))
    result = run_transport(dev, _rabi_seq(dev), transport=soc, axes=[AMP], reps=REPS)

    signal = DispersiveIQ(threshold=0.5).project(np.stack([result.avgi, result.avgq], axis=-1))
    fit = fit_rabi(result.xpts, signal[-1])      # the data read is the last read
    assert fit.success
    assert _rel(fit.derived["pi_gain"], TRUE_PI_GAIN) < 0.02
    assert _rel(fit.derived["pi_gain"], dev.calib.qubit.pulses["pi_ge"].gain) > 0.05


def test_an_experiment_runs_unchanged_over_a_transport_runner(dev):
    """Every shipped experiment reaches a remote board through the Runner seam, with no local soc."""
    runner = TransportRunner(JobServerSoc(InProcessClient(_simulator_server(dev))))
    # loopback_demo has no dedicated readout line (single-generator instance) — probe on the
    # qubit line instead, same as resonator_spectroscopy's own test override (issue #102).
    outcome = AmplitudeRabi(dev, gains=GAINS, points=AMP.points,
                            readout=DispersiveIQ(threshold=0.5),
                            line=("qubit", "drive"),
                            runner=runner).go(None, reps=REPS)
    assert outcome.fit.success
    assert _rel(outcome.fit.derived["pi_gain"], TRUE_PI_GAIN) < 0.02
    assert outcome.result.meta["environment"]["source"] == "NullEnvironment"
    assert outcome.result.shots is None          # per-shot data is a soc capability; there is no soc


# -- declared loop axes ride the transport (issue #148) -------------------------------------

#: shipped experiments whose DEFAULT sweep is bound to a declared loop axis (D36): a gain, a delay
#: and a frequency sweep, all run with no arguments on the multimode demo.
BOUND_SWEEP_EXPERIMENTS = {
    "amplitude_rabi": AmplitudeRabi,
    "t1": T1,
    "qubit_spectroscopy": QubitSpectroscopy,
}


class _RecordingClient:
    """An :class:`InProcessClient` that keeps every job wire the job server receives."""

    def __init__(self, server: JobServer) -> None:
        self.inner = InProcessClient(server)
        self.received: list[dict] = []

    def submit(self, job_wire: dict) -> str:
        self.received.append(json.loads(json.dumps(job_wire)))
        return self.inner.submit(job_wire)

    def poll(self, job_id: str) -> dict:
        return self.inner.poll(job_id)


class _CompilingRunner(Runner):
    """The local ``Runner``, keeping each built program's canonical compiled payload."""

    def __init__(self) -> None:
        super().__init__()
        self.programs: list[dict] = []

    def _arm(self, soc, prog, device, experiment, seq) -> None:
        from strumento.core.program import acquire_config, compiled_payload

        if prog.binprog is None:
            prog.compile()
        self.programs.append({"program": compiled_payload(prog), "acquire": acquire_config(prog)})
        super()._arm(soc, prog, device, experiment, seq)


def _local_and_transport(dev, cls):
    """Run ``cls(dev)`` (defaults) on the local Runner and through JobServerSoc, on the mock."""
    runner = _CompilingRunner()
    local = cls(dev, runner=runner).go(MockQickSocV2(dev.soccfg_snapshot), analyze=False,
                                       progress=False)
    client = _RecordingClient(JobServer(MockQickSocV2(dev.soccfg_snapshot)))
    remote = cls(dev, runner=TransportRunner(JobServerSoc(client))).go(None, analyze=False,
                                                                       progress=False)
    return runner.programs, local.result, client.received, remote.result


@pytest.mark.parametrize("name", sorted(BOUND_SWEEP_EXPERIMENTS))
def test_a_default_bound_sweep_experiment_runs_through_the_job_server(multimode_dev, name):
    cls = BOUND_SWEEP_EXPERIMENTS[name]
    assert cls(multimode_dev).axes(), "the default sweep must be bound to a declared axis"
    _, _, received, result = _local_and_transport(multimode_dev, cls)
    assert len(received) == 1
    assert result.meta["transport"]["kind"] == "JobServerSoc"


@pytest.mark.parametrize("name", sorted(BOUND_SWEEP_EXPERIMENTS))
def test_the_program_sent_over_the_transport_is_the_one_the_local_runner_compiles(
    multimode_dev, name,
):
    local_programs, _, received, _ = _local_and_transport(multimode_dev,
                                                          BOUND_SWEEP_EXPERIMENTS[name])
    assert len(local_programs) == len(received) == 1
    sent = CompiledJob.from_wire(received[0])
    assert sent.program == local_programs[0]["program"]
    assert json.dumps(sent.program, sort_keys=True) == json.dumps(local_programs[0]["program"],
                                                                  sort_keys=True)
    assert sent.acquire["expts"] == local_programs[0]["acquire"]["expts"]
    assert sent.acquire["reads_per_shot"] == local_programs[0]["acquire"]["reads_per_shot"]


@pytest.mark.parametrize("name", sorted(BOUND_SWEEP_EXPERIMENTS))
def test_the_transport_result_has_the_local_results_shape_and_axis(multimode_dev, name):
    _, local, _, remote = _local_and_transport(multimode_dev, BOUND_SWEEP_EXPERIMENTS[name])
    assert np.shape(remote.avgi) == np.shape(local.avgi)
    assert np.shape(remote.avgq) == np.shape(local.avgq)
    assert np.shape(remote.xpts) == np.shape(local.xpts)
    assert np.array_equal(np.asarray(remote.xpts), np.asarray(local.xpts))


# -- the agent is picky where a dumb executor should be ------------------------------------


def test_a_job_whose_declared_shape_contradicts_its_program_is_refused(dev):
    from strumento.core.program import StrumentoProgram

    prog = StrumentoProgram(dev, seq=_rabi_seq(dev), axes=[AMP], reps=8,
                            soc=build_soc(V2_SOCCFG, "mock"))
    job = prog.to_compiled_job()
    job.acquire["reads_per_shot"] = [4]          # a lie the board is the only side able to catch

    with pytest.raises(JobError, match="reads_per_shot"):
        JobServerSoc(InProcessClient(_mock_server())).run(job)


def test_a_broken_payload_fails_the_job_and_not_the_server(dev):
    server = _mock_server()
    soc = JobServerSoc(InProcessClient(server))
    with pytest.raises(JobError):
        soc.run(CompiledJob(overlay_id="", program={"nonsense": True}, acquire={"reps": 1}))

    # the board keeps serving: the next submitter is unaffected
    from strumento.core.program import StrumentoProgram

    good = StrumentoProgram(dev, seq=_rabi_seq(dev), axes=[AMP], reps=4,
                            soc=build_soc(V2_SOCCFG, "mock")).to_compiled_job()
    assert np.shape(soc.run(good).iq[0]) == (1, 61, 2)


def test_an_unavailable_overlay_is_a_failed_job_not_a_guess(dev):
    from strumento.core.program import StrumentoProgram

    server = JobServer(build_soc(V2_SOCCFG, "mock"), overlays=("cqed-loopback-v2",))
    prog = StrumentoProgram(dev, seq=_rabi_seq(dev), axes=[AMP], reps=4,
                            soc=build_soc(V2_SOCCFG, "mock"))
    with pytest.raises(JobError, match="not available on this board"):
        JobServerSoc(InProcessClient(server)).run(prog.to_compiled_job(overlay_id="some-other"))


# -- and it runs as a real process ---------------------------------------------------------


def test_the_reference_agent_runs_as_a_subprocess_over_a_real_process_boundary(dev):
    """The same job, the same result — through a pipe, with a genuine serialization round-trip."""
    with SubprocessClient(V2_SOCCFG, soc="mock") as client:
        result = run_transport(dev, _rabi_seq(dev), transport=JobServerSoc(client), axes=[AMP],
                               reps=6,
                               environment={"bench": "subprocess"})
    assert np.shape(result.avgi) == (1, 61)
    assert result.meta["environment"]["readings"] == {"bench": "subprocess"}
    assert result.meta["transport"]["reps"] == 6
