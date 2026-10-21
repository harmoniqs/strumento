"""Tests for the job-server transport (D14).

Pure: exercises the wire contract (CompiledJob / RawAcquisition) and the JobServerSoc client
against an in-process FakeJobServer — no qick, no board, no HTTP. This is the strumento side of
the no-Pyro / compile-off-board / thin-execute-agent architecture: the fast machine ships a
CompiledJob, the board returns raw IQ.
"""

import numpy as np
import pytest

from strumento.core.result import MeasurementResult
from strumento.core.transport import (
    CompiledJob,
    JobError,
    JobServerClient,
    JobServerSoc,
    RawAcquisition,
    Transport,
)


def _job(expts=None, n_ro=1, reads=None):
    reads = reads or [1] * n_ro
    return CompiledJob(
        overlay_id="cqed-multimode-v2",
        program={"blob": "opaque-compiled-program-bytes"},  # opaque to the transport
        acquire={
            "reps": 100, "soft_avgs": 1,
            "ro_chs": list(range(n_ro)), "reads_per_shot": reads, "expts": expts,
        },
    )


class FakeJobServer:
    """In-process JobServerClient that 'runs' a job into zero IQ of the right shape."""

    def __init__(self, pending_polls=0):
        self._jobs, self._pending, self._pending_polls, self._n = {}, {}, pending_polls, 0

    def submit(self, job_wire):
        jid = str(self._n)
        self._n += 1
        acq = job_wire["acquire"]
        expts = acq.get("expts")
        iq = []
        for nr in acq["reads_per_shot"]:
            shape = (nr, 2) if expts is None else (nr, expts, 2)
            iq.append(np.zeros(shape).tolist())
        self._jobs[jid] = {"status": "done", "acquisition": {"iq": iq}}
        self._pending[jid] = self._pending_polls
        return jid

    def poll(self, jid):
        if self._pending[jid] > 0:
            self._pending[jid] -= 1
            return {"status": "pending"}
        return self._jobs[jid]


def test_compiled_job_wire_roundtrip():
    j = _job(expts=51)
    assert CompiledJob.from_wire(j.to_wire()) == j


def test_raw_acquisition_wire_roundtrip_and_to_result():
    ra = RawAcquisition(iq=[np.array([[0.3, -0.7]])])  # 1 ro ch, 1 read, (I, Q)
    ra2 = RawAcquisition.from_wire(ra.to_wire())
    assert np.allclose(ra2.iq[0], ra.iq[0])
    res = ra.to_measurement_result()
    assert isinstance(res, MeasurementResult)
    assert res.avgi.reshape(-1)[0] == pytest.approx(0.3)
    assert res.avgq.reshape(-1)[0] == pytest.approx(-0.7)


def test_job_server_soc_round_trips_to_raw_acquisition():
    soc = JobServerSoc(FakeJobServer())
    ra = soc.run(_job(expts=51))
    assert isinstance(ra, RawAcquisition)
    assert np.asarray(ra.iq[0]).shape == (1, 51, 2)
    res = ra.to_measurement_result(xpts=np.linspace(0, 30000, 51))
    assert res.avgi.shape[-1] == 51
    assert res.xpts is not None


def test_job_server_soc_polls_until_done():
    soc = JobServerSoc(FakeJobServer(pending_polls=3))
    assert isinstance(soc.run(_job()), RawAcquisition)


def test_job_server_soc_raises_on_error_status():
    class ErrServer:
        def submit(self, job_wire): return "x"
        def poll(self, jid): return {"status": "error", "error": "board offline"}

    with pytest.raises(JobError, match="board offline"):
        JobServerSoc(ErrServer()).run(_job())


def test_protocol_conformance():
    assert isinstance(JobServerSoc(FakeJobServer()), Transport)
    assert isinstance(FakeJobServer(), JobServerClient)
