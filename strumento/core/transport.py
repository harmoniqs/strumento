"""Job-server transport (D14) — the strumento side of no-Pyro execution.

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
(it needs a real compiled ``AveragerProgramV2``) and therefore lives in
:meth:`strumento.core.program.StrumentoProgram.to_compiled_job`; this module locks the *contract*
the board agent builds against and stays pure.

**The run path.** :func:`run_transport` is the whole seam in one call: build the program against
the device's own soccfg snapshot → extract the job → ship it → receive
:class:`RawAcquisition` → **stamp the environment client-side at receipt** → return a
:class:`~strumento.core.result.MeasurementResult` with ``xpts`` from the sweep. Stamping happens
*here*, not on the board: telemetry is lab-side by design (D24/D34) and the board stays dumb, so the
moment the data arrives is the earliest true reading — which is also why
:func:`~strumento.core.environment.stamp_environment` never overwrites an existing stamp.
:class:`TransportRunner` plugs the same path under any :class:`~strumento.core.experiment.Experiment`.

This module is **pure** — dataclasses + client logic + numpy; no ``qick`` import (the program build
inside :func:`run_transport` is a local import, the same idiom the experiment ``Runner`` uses).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from strumento.core.environment import stamp_environment
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
    def from_wire(cls, d: dict) -> CompiledJob:
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
    def from_wire(cls, d: dict) -> RawAcquisition:
        return cls(iq=[np.asarray(a) for a in d["iq"]])

    def to_measurement_result(self, xpts=None, *, environment=None, device=None,
                              meta: dict | None = None) -> MeasurementResult:
        """Wrap the received IQ as a :class:`MeasurementResult`, **stamped at receipt** (D34).

        The same shape the in-process path returns, so nothing downstream can tell where the data
        came from — except the stamp, which says so.

        ``environment`` is the **lab-side** telemetry source (a dict of readings, a zero-argument
        callable, or anything with ``snapshot()``). There is deliberately no ``soc=`` here: the board
        is remote, so its own environment reporter is not reachable from this process, and the
        station reading that matters is the one the client can take. ``device`` contributes the
        instance's per-line signal-chain annotations.
        """
        arr = np.asarray(self.iq[0])  # first readout channel
        result = MeasurementResult(
            xpts=xpts, avgi=arr[..., 0], avgq=arr[..., 1], raw=self.iq, meta=dict(meta or {}),
        )
        return stamp_environment(result, environment, device=device)


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


# --------------------------------------------------------------------------------------
# the transport-backed run path
# --------------------------------------------------------------------------------------


def run_transport(
    device,
    seq,
    sweeps=(),
    *,
    transport: Transport,
    axes=(),
    overlay_id: str = "",
    reps: int = 100,
    final_delay=None,
    soft_avgs: int = 1,
    environment=None,
    soccfg=None,
) -> MeasurementResult:
    """Build off-board, ship, receive, stamp — one acquisition over a :class:`Transport`.

    The mirror image of :meth:`strumento.core.experiment.Runner.run`: same inputs, same
    :class:`MeasurementResult` out, but the acquire happens on the far side of a wire instead of on a
    local soc object.

    The program is built against **the device's own soccfg snapshot** (``soccfg=`` overrides it),
    which is what makes off-board compilation legitimate: an overlay corresponds 1:1 to a snapshot
    (D25), so a program compiled against the instance's snapshot is a program the selected overlay
    can run. ``xpts`` comes from the built program, so the axis on the result and the axis the sweep
    declared are the same object, never two derivations.

    Args:
        device: the loaded instance (supplies wiring, calibration, and the soccfg snapshot).
        seq: the :class:`~strumento.core.pulses.Seq` to run.
        sweeps: explicit sweeps; a swept pulse field carries its own and is discovered.
        transport: anything satisfying :class:`Transport` — :class:`JobServerSoc` in practice.
        axes: the declared loop axes (D36) a bound sweep rides, forwarded to the program exactly
            as :meth:`strumento.core.experiment.Runner.run` forwards them.
        overlay_id: the firmware personality to select; ``""`` = whatever is loaded.
        reps: hardware repetitions (averaged on the board).
        final_delay: inter-shot relax delay (µs); ``None`` reads the device's calibration.
        soft_avgs: rounds the board averages in software.
        environment: the lab-side telemetry source, read at receipt (D34).
        soccfg: build against this ``QickConfig`` instead of the device's snapshot.
    """
    # local import: qick lives behind this call, not behind importing this module
    from strumento.core.program import StrumentoProgram

    prog = StrumentoProgram(
        device, seq=seq, sweeps=sweeps, axes=axes, reps=reps, final_delay=final_delay,
        soc=soccfg,
    )
    job = prog.to_compiled_job(overlay_id=overlay_id, soft_avgs=soft_avgs)
    raw = transport.run(job)
    return raw.to_measurement_result(
        xpts=prog.xpts(),
        environment=environment,
        device=device,
        meta={"transport": {"kind": type(transport).__name__, "overlay_id": job.overlay_id,
                            **job.acquire}},
    )


class TransportRunner:
    """A drop-in experiment ``Runner`` that acquires over a :class:`Transport` instead of a soc.

    Because every shipped experiment goes through the Runner seam, this is all it takes to run one
    against a remote board::

        exp = AmplitudeRabi(dev, gains=Sweep(start=0, stop=16000, on="amp"), points=61,
                            runner=TransportRunner(JobServerSoc(client)))
        outcome = exp.go(None)      # no local soc: the board is on the far side of the wire

    The experiment's declared loop axes (D36) are forwarded to the program built here, as
    ``Runner`` and ``LegacyPyroRunner`` forward them, so a bound sweep (every shipped experiment's
    default) compiles to the same program it compiles to locally.

    ``soc`` is accepted and ignored (the signature is the Runner's), which is why ``go(None)`` reads
    honestly: in a job-server deployment there *is* no client-side board object. Arming (D32) and
    per-shot collection are soc capabilities and therefore do not apply here — a simulator behind the
    wire is armed on the far side, by whoever owns it.
    """

    def __init__(self, transport: Transport, *, overlay_id: str = "", soft_avgs: int = 1,
                 environment=None) -> None:
        self.transport = transport
        self.overlay_id = overlay_id
        self.soft_avgs = soft_avgs
        self.environment = environment

    def run(self, device, seq, sweeps, soc=None, *, axes=(), experiment=None, reps: int = 100,
            final_delay=None, environment=None, record=None, **acquire_kw) -> MeasurementResult:
        result = run_transport(
            device, seq, sweeps,
            transport=self.transport,
            axes=axes,
            overlay_id=self.overlay_id,
            reps=reps,
            final_delay=final_delay,
            soft_avgs=self.soft_avgs,
            environment=environment if environment is not None else self.environment,
        )
        if record is not None:
            from strumento.core.experiment import _emit_acquired

            _emit_acquired(record, result, experiment=experiment, reps=reps)
        return result
