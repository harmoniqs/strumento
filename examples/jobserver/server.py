"""A reference **execute-agent**: the board side of the D14 wire contract.

This is *deployment*, not core (see ``docs/job-manager.md``): the client-side contract lives in
``strumento/core/transport.py`` and ships with the package, while the process that sits next to a
board and drains a queue is a deployment concern a lab owns. This file is the smallest honest
implementation of that process, so the contract has a second implementer and the whole path is
testable with no board and no HTTP.

**What the agent does — and pointedly does not do.** The board is a dumb executor: *select overlay →
load the compiled program → run → hand back ADC buffers.* It does not know what a Rabi is, does not
fit, does not hold calibration, and never decides anything. Every expensive step already happened
off-board; what arrives is a payload plus an acquire block.

The executor is three lines of qick, because the payload *is* qick's own dump form:

.. code::

    prog = LoadedProgram(soccfg, job.program)     # load_prog: ASM + waveforms + envelopes + dims
    raw = prog.acquire(soc, rounds=soft_avgs)     # the board's local acquire loop
    return RawAcquisition(iq=raw)

**Two ways to run it.**

* *in-process* — :class:`JobServer` + :class:`InProcessClient`, which is how the tests drive it
  against ``MockQickSocV2`` and ``SimulatorSoc``;
* *as a subprocess* — ``python examples/jobserver/server.py --soccfg <dump_cfg.json>`` speaks
  newline-delimited JSON on stdin/stdout, and :class:`SubprocessClient` is a
  :class:`~strumento.core.transport.JobServerClient` over it. That is a real process boundary with a
  real serialization round-trip, which is the point of having it: a deployment swaps this loop for
  HTTP and nothing above the client changes.

**Integrity check on the wire.** Before running anything the agent compares the *declared* acquire
block against the program it actually reconstructed (readout channels, reads per shot, experiment
count). A mismatch is a failed job with a message, not a silently mis-shaped array — the one place a
dumb executor should be picky, because it is the only side that can see both.

Single-threaded on purpose: hardware exclusivity is structural in this design (one worker owns the
board and drains a FIFO), so a reference implementation with one queue and no locks is the honest
shape. A real deployment gives the worker its own thread or process; here
:meth:`JobServer.work` is called from :meth:`JobServer.poll`, so a poll is when the worker gets its
turn.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from strumento.core.transport import CompiledJob, RawAcquisition


class LoadedProgram:
    """A compiled payload, reconstituted into a runnable ``AveragerProgramV2``.

    ``AveragerProgramV2`` normally builds itself from ``_initialize``/``_body`` at construction; here
    there is nothing to build — the low-level program crossed the wire already — so ``compile`` is
    replaced by ``load_prog``. That is qick's documented inverse of ``dump_prog`` (it restores the
    ASM, the waveforms, the envelope arrays and the acquisition dims, then makes the binary), which is
    why the executor needs no knowledge of what it is running.

    Implemented as a factory rather than a module-level subclass so importing this file never imports
    ``qick`` — the same lazy-import discipline the strumento core keeps.
    """

    def __new__(cls, soccfg, program: dict, *, reps: int = 1, final_delay=None):
        from qick.asm_v2 import AveragerProgramV2

        class _Loaded(AveragerProgramV2):
            def __init__(self, soccfg, program, *, reps, final_delay):
                self._payload = program
                super().__init__(soccfg, reps=reps, final_delay=final_delay)

            def compile(self):
                if self.binprog is None:
                    # deepcopy: load_prog decodes the envelope arrays in place, and the payload must
                    # stay byte-identical to what arrived (a caller may still be holding the wire form)
                    self.load_prog(copy.deepcopy(self._payload))

            def _initialize(self, cfg):
                pass        # nothing to initialize: the program is already compiled

            def _body(self, cfg):
                pass        # nothing to play: ditto

        return _Loaded(soccfg, program, reps=reps, final_delay=final_delay)


def execute(job: CompiledJob, soc, soccfg=None) -> RawAcquisition:
    """Run one job on a soc and return its raw IQ. The entire board-side responsibility.

    ``soc`` is the board object (a real ``QickSoc``, or the mock/simulator offline); ``soccfg``
    defaults to it, since a ``QickSoc`` *is* a ``QickConfig``.
    """
    acquire = dict(job.acquire)
    prog = LoadedProgram(soccfg if soccfg is not None else soc, job.program,
                         reps=int(acquire.get("reps", 1)))
    _check_declared_shape(prog, acquire)
    raw = prog.acquire(soc, rounds=int(acquire.get("soft_avgs", 1)), progress=False)
    return RawAcquisition(iq=list(raw))


def _check_declared_shape(prog, acquire: dict) -> None:
    """The declared acquire block must match the reconstructed program (see the module docstring)."""
    from strumento.core.program import acquire_config

    actual = acquire_config(prog, soft_avgs=int(acquire.get("soft_avgs", 1)))
    for key in ("ro_chs", "reads_per_shot", "expts"):
        if key in acquire and acquire[key] != actual[key]:
            raise ValueError(
                f"job declares {key}={acquire[key]!r} but the compiled program it shipped has "
                f"{key}={actual[key]!r}; refusing to run a job whose declared shape does not match "
                f"its program (the returned buffers would be silently mis-shaped)"
            )


class JobServer:
    """A single-worker queue over one board: ``submit`` enqueues, ``work`` runs, ``poll`` reports.

    The status dicts are exactly what :class:`~strumento.core.transport.JobServerClient` promises:
    ``{"status": "pending"}`` while queued, ``{"status": "done", "acquisition": {...}}`` on success,
    ``{"status": "error", "error": "..."}`` on failure. A failed job never takes the server down —
    the board keeps serving the next submitter.
    """

    def __init__(self, soc, soccfg=None, *, overlays: tuple[str, ...] = ()) -> None:
        self.soc = soc
        self.soccfg = soccfg
        self.overlays = tuple(overlays)      # empty ⇒ single-personality board: accept any id
        self.loaded_overlay = ""
        self.queue: list[str] = []
        self.jobs: dict[str, dict] = {}
        self._n = 0

    # -- the JobServerClient-facing surface -------------------------------------------

    def submit(self, job_wire: dict) -> str:
        job_id = str(self._n)
        self._n += 1
        self.jobs[job_id] = {"status": "pending", "job": job_wire}
        self.queue.append(job_id)
        return job_id

    def poll(self, job_id: str) -> dict:
        if job_id not in self.jobs:
            return {"status": "error", "error": f"unknown job {job_id!r}"}
        if self.jobs[job_id]["status"] == "pending":
            self.work()          # single-threaded reference worker: a poll is its turn
        return {k: v for k, v in self.jobs[job_id].items() if k != "job"}

    # -- the worker --------------------------------------------------------------------

    def work(self) -> str | None:
        """Run the job at the head of the queue (FIFO). Returns its id, or ``None`` when idle."""
        if not self.queue:
            return None
        job_id = self.queue.pop(0)
        record = self.jobs[job_id]
        try:
            job = CompiledJob.from_wire(record["job"])
            self._select_overlay(job.overlay_id)
            record.update(status="done", acquisition=execute(job, self.soc, self.soccfg).to_wire())
        except Exception as err:
            record.update(status="error", error=f"{type(err).__name__}: {err}")
        return job_id

    def _select_overlay(self, overlay_id: str) -> None:
        """Select a firmware personality (D25: one overlay ⇔ one soccfg snapshot).

        ``""`` means "whatever is loaded" — the single-personality case. A named overlay this board
        does not have is a failed job, not a guess: running a program compiled against a different
        soccfg is exactly the class of error the 1:1 overlay↔snapshot rule exists to prevent. Loading
        real gateware is a deployment's job; this reference agent only records the selection.
        """
        if not overlay_id or overlay_id == self.loaded_overlay:
            return
        if self.overlays and overlay_id not in self.overlays:
            raise ValueError(
                f"overlay {overlay_id!r} is not available on this board (have: "
                f"{', '.join(self.overlays)})"
            )
        self.loaded_overlay = overlay_id


class InProcessClient:
    """A :class:`~strumento.core.transport.JobServerClient` straight onto a :class:`JobServer`.

    No transport at all — the point is that ``JobServerSoc`` is testable end to end without a server
    process, and that the client contract is small enough to satisfy in four lines.
    """

    def __init__(self, server: JobServer) -> None:
        self.server = server

    def submit(self, job_wire: dict) -> str:
        return self.server.submit(job_wire)

    def poll(self, job_id: str) -> dict:
        return self.server.poll(job_id)


# --------------------------------------------------------------------------------------
# subprocess mode: newline-delimited JSON on stdin/stdout
# --------------------------------------------------------------------------------------
#
# One request per line, one response per line. Deliberately the dumbest wire that is still a real
# process boundary: it exercises the same serialize/deserialize the HTTP deployment will, with no
# dependency and nothing to configure.
#
#   -> {"op": "submit", "job": {...}}          <- {"ok": true, "job_id": "0"}
#   -> {"op": "poll", "job_id": "0"}           <- {"ok": true, "status": {...}}
#   -> {"op": "shutdown"}                       <- {"ok": true}


def handle(server: JobServer, request: dict) -> dict:
    """One request → one response. Errors come back as data, exactly like the CLI's convention."""
    op = request.get("op")
    try:
        if op == "submit":
            return {"ok": True, "job_id": server.submit(request["job"])}
        if op == "poll":
            return {"ok": True, "status": server.poll(request["job_id"])}
        if op == "shutdown":
            return {"ok": True, "shutdown": True}
        return {"ok": False, "error": f"unknown op {op!r} (submit | poll | shutdown)"}
    except Exception as err:
        return {"ok": False, "error": f"{type(err).__name__}: {err}"}


def serve_stdio(server: JobServer, stdin=None, stdout=None) -> None:
    """Read requests from ``stdin``, write responses to ``stdout``, until EOF or ``shutdown``."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    for line in stdin:
        if not line.strip():
            continue
        response = handle(server, json.loads(line))
        stdout.write(json.dumps(response) + "\n")
        stdout.flush()
        if response.get("shutdown"):
            return


class SubprocessClient:
    """A :class:`~strumento.core.transport.JobServerClient` over a spawned :func:`serve_stdio` agent.

    Use as a context manager (or call :meth:`close`). This is the shape a real client takes — submit
    and poll are the only two calls that cross the boundary — with a pipe standing in for HTTP.
    """

    def __init__(self, soccfg_path: str | Path, *, soc: str = "mock", python: str | None = None) -> None:
        import subprocess

        self.proc = subprocess.Popen(
            [python or sys.executable, str(Path(__file__).resolve()),
             "--soccfg", str(soccfg_path), "--soc", soc],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )

    def _rpc(self, request: dict) -> dict:
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("the job-server subprocess closed its stdout without responding")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"job server: {reply.get('error')}")
        return reply

    def submit(self, job_wire: dict) -> str:
        return self._rpc({"op": "submit", "job": job_wire})["job_id"]

    def poll(self, job_id: str) -> dict:
        return self._rpc({"op": "poll", "job_id": job_id})["status"]

    def close(self) -> None:
        try:
            self._rpc({"op": "shutdown"})
        except Exception:  # pragma: no cover - the process may already be gone
            pass
        self.proc.stdin.close()
        self.proc.wait(timeout=10)
        self.proc.stdout.close()  # both pipe ends are ours; an unclosed one leaks its fd

    def __enter__(self) -> SubprocessClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def build_soc(soccfg_path: str | Path, kind: str = "mock"):
    """The board object this agent drives. Offline: the mock (zeros) or the simulator.

    On a real board this is ``QickSoc(bitfile)`` and the ``--soccfg`` argument is unnecessary — the
    board reports its own config. Offline it is reconstructed from a committed ``dump_cfg()``
    snapshot, which is what makes the whole path exercisable in CI.

    A ``SimulatorSoc`` here must be armed by whoever owns it (a board-side simulator has no
    experiment to infer one from) — pass ``SimulatorSoc(..., model=fn)`` or arm it directly. That is
    the documented raw-program escape hatch, not a special case.
    """
    snapshot = json.loads(Path(soccfg_path).read_text())
    if kind == "simulator":
        from strumento.core.simulator import SimulatorSoc

        return SimulatorSoc(snapshot)
    if kind == "mock":
        from strumento.core.mock import MockQickSocV2

        return MockQickSocV2(snapshot)
    raise ValueError(f"unknown soc kind {kind!r} (mock | simulator)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--soccfg", required=True, type=Path,
                        help="a dump_cfg() JSON snapshot for the board this agent serves")
    parser.add_argument("--soc", default="mock", choices=["mock", "simulator"],
                        help="offline board object to drive (default: mock)")
    args = parser.parse_args(argv)
    soc = build_soc(args.soccfg, args.soc)
    serve_stdio(JobServer(soc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
