"""Job-server transport (spec §14.2, resolved D14) — the strumento side of no-Pyro execution.

The resolved architecture: the fast machine builds + compiles a program off-board and ships a
**coarse** job to the board (*load overlay → load compiled program → run → return ADC buffers*);
the board is a dumb executor. This module is that seam on the client side — the **wire contract**
plus the :class:`JobServerSoc` transport. It is deliberately *not* a per-method soc proxy: making
each ``acquire`` call a remote round-trip is exactly the Pyro pattern D14 drops. One job in, raw
IQ out.

Layering (vs :mod:`strumento.core.soc`): an :class:`~strumento.core.soc.AbstractSoc` is the
board-*object* surface (used in-process by the mock/simulator, or by the thin on-board agent
against its local ``QickSoc``). A :class:`Transport` is the off-board→board *channel*. Swapping
Pyro for the job server is swapping the transport; nothing above it changes.

What crosses the wire is intentionally opaque to the transport: :class:`CompiledJob.program` is
the compiled payload (tproc binary + envelopes) produced off-board and loaded verbatim by the
board agent. Producing that payload from a built ``StrumentoProgram`` is the one v2-gated bridge
(it needs a real compiled ``AveragerProgramV2``); this module locks the *contract* the board agent
builds against, independent of that bridge.

This module is **pure** — dataclasses + client logic + numpy; no ``qick`` import.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from strumento.core.result import MeasurementResult


class JobError(RuntimeError):
    """Raised when the job server reports a failed job."""


@dataclass
class CompiledJob:
    """A board-ready job: which firmware overlay, the compiled program, and how to acquire.

    ``overlay_id`` selects a pre-synthesized firmware personality (one overlay == one
    ``dump_cfg()`` soccfg snapshot, D25). ``program`` is the compiled payload, opaque here.
    ``acquire`` carries the run parameters (``reps``, ``soft_avgs``, ``ro_chs``,
    ``reads_per_shot``, ``expts``).
    """

    overlay_id: str
    program: dict[str, Any]
    acquire: dict[str, Any]

    def to_wire(self) -> dict:
        return {"overlay_id": self.overlay_id, "program": self.program, "acquire": self.acquire}

    @classmethod
    def from_wire(cls, d: dict) -> "CompiledJob":
        return cls(overlay_id=d["overlay_id"], program=d["program"], acquire=d["acquire"])


@dataclass
class RawAcquisition:
    """What the board returns: per-readout-channel IQ arrays, shaped ``(nreads, [expts,] 2)``.

    Averaging happens on the board's local ``acquire`` (topology (ii)); off-board owns the
    analysis/fitting that turns this into physics.
    """

    iq: list

    def to_wire(self) -> dict:
        return {"iq": [np.asarray(a).tolist() for a in self.iq]}

    @classmethod
    def from_wire(cls, d: dict) -> "RawAcquisition":
        return cls(iq=[np.asarray(a) for a in d["iq"]])

    def to_measurement_result(self, xpts=None) -> MeasurementResult:
        arr = np.asarray(self.iq[0])  # first readout channel
        return MeasurementResult(xpts=xpts, avgi=arr[..., 0], avgq=arr[..., 1], raw=self.iq)


@runtime_checkable
class Transport(Protocol):
    """A channel that runs a :class:`CompiledJob` on a board and returns its raw IQ."""

    def run(self, job: CompiledJob) -> RawAcquisition: ...


@runtime_checkable
class JobServerClient(Protocol):
    """The minimal job-server API the transport needs (HTTP details live in the deployment).

    ``submit`` enqueues a serialized job and returns a job id; ``poll`` returns a status dict
    (``{"status": "pending" | "done" | "error", ...}``; a done job carries ``"acquisition"``).
    """

    def submit(self, job_wire: dict) -> str: ...

    def poll(self, job_id: str) -> dict: ...


class JobServerSoc:
    """A :class:`Transport` that submits a compiled job to a job server and polls for its result.

    Deliberately coarse — one submit + poll-to-completion + one deserialize — not a per-method
    proxy (D14 drops Pyro precisely to avoid fine-grained remote calls). The HTTP wiring lives in
    the injected ``client``; an in-process client makes this fully testable without a board.
    """

    def __init__(self, client: JobServerClient, *, poll_interval_s: float = 0.0, max_polls: int = 100_000):
        self.client = client
        self.poll_interval_s = poll_interval_s
        self.max_polls = max_polls

    def run(self, job: CompiledJob) -> RawAcquisition:
        job_id = self.client.submit(job.to_wire())
        for _ in range(self.max_polls):
            status = self.client.poll(job_id)
            state = status.get("status")
            if state == "done":
                return RawAcquisition.from_wire(status["acquisition"])
            if state == "error":
                raise JobError(status.get("error", f"job {job_id} failed"))
            if self.poll_interval_s > 0:
                time.sleep(self.poll_interval_s)
        raise TimeoutError(f"job {job_id} did not complete within {self.max_polls} polls")
