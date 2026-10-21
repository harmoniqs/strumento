"""A legacy, Pyro4-compatible ``Runner`` — the compile/acquire split for ``qick.pyro.make_proxy()``.

Ships **alongside**, never replacing, :class:`~strumento.core.transport.TransportRunner` (the
resolved D14 path — job-server-native, no object proxies; see :mod:`strumento.core.transport`
and ``docs/job-manager.md``). D14 dropped Pyro4 precisely because its fine-grained, multi-call,
background-worker-thread acquisition pattern is fragile over RPC. Some labs' real infrastructure
still runs on ``qick.pyro.make_proxy()`` today, though, and migrating to a job-server deployment
is a separate, larger effort — this module is the honest escape hatch for that infrastructure,
not a recommendation.

Like the base :class:`~strumento.core.experiment.Runner`, this module never imports ``qick`` or
``Pyro4``: both the compile-time config and the acquire-time proxy are accepted as duck-typed
objects, so it costs no new dependency and stays testable in the ``pure`` CI lane.
"""

from __future__ import annotations

from strumento.core.environment import stamp_environment
from strumento.core.experiment import Runner
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult


class LegacyPyroRunner(Runner):
    """Compiles against a constructor-supplied ``soccfg``; acquires against ``qick.pyro``'s proxy.

    **Why this exists (D14).** The resolved architecture —
    :class:`~strumento.core.transport.TransportRunner` /
    :class:`~strumento.core.transport.JobServerSoc`, see :mod:`strumento.core.transport` and
    ``docs/job-manager.md`` — drops Pyro4 entirely: compile off-board, ship one coarse job, run
    on a dumb executor. **Prefer ``TransportRunner``/``JobServerSoc`` for anything new.** This
    class is for the lab that is on ``qick.pyro.make_proxy()`` *today*, where migrating to a
    job-server deployment is out of scope for now — a supported, labeled legacy path.

    **The problem it works around.** ``qick.pyro.make_proxy()`` returns a Pyro4 proxy that is
    fine for a coarse acquire call but a poor stand-in for the ``QickConfig`` a
    ``StrumentoProgram`` needs at *compile* time — v2 assembly makes many fine-grained
    attribute/subscript reads while building a program, and routing every one of those over
    Pyro4's RPC boundary is exactly the fragile pattern D14 exists to avoid. The fix is the
    split this class encodes: compile against a real ``QickConfig`` snapshot supplied at
    construction (``soccfg``, e.g. from the board's own ``dump_cfg()``), and acquire against
    whatever live proxy is handed to ``.run()``/``.go()``.

    **Known risk: a dead worker thread breaks the session silently.** Confirmed directly during
    a real hardware bring-up session: Pyro4 runs each proxy's calls through a background worker
    thread, and a thread killed by *any* earlier exception silently breaks every subsequent
    accumulate-mode call in the same kernel session — there is no way to check that thread's
    liveness through Pyro4's exposed-method boundary, so the failure looks like a hang or a
    generic RPC error far from its real cause. **Recovery:** open a fresh
    ``qick.pyro.make_proxy()`` connection and re-arm from there — retrying the same call on the
    same proxy will not revive a dead worker thread. This class does not add a pre-flight
    liveness canary before acquire; that is real hardening, deferred to a later pass.

    Reuses the base :class:`~strumento.core.experiment.Runner`'s ``_arm``, ``_attach_shots``,
    and :func:`~strumento.core.environment.stamp_environment` unchanged — only *which object
    plays the compile-time config* differs from the base Runner.
    """

    def __init__(self, soccfg, *, environment=None) -> None:
        super().__init__(environment=environment)
        self.soccfg = soccfg

    def run(self, device, seq: Seq, sweeps, soc, *, axes=(), experiment=None, reps: int = 100,
            final_delay=None, environment=None, record=None, **acquire_kw) -> MeasurementResult:
        """Compile against ``self.soccfg``; acquire against ``soc`` — the rest is the base Runner's.

        ``soc`` is whatever ``qick.pyro.make_proxy()`` (or a fake of it) returns; it is used
        **only** for ``.acquire()``, never for compilation. ``self.soccfg`` — supplied at
        construction — is used **only** to compile, never acquired from.
        """
        from strumento.core.program import StrumentoProgram

        prog = StrumentoProgram(device, seq=seq, sweeps=sweeps, axes=axes, reps=reps,
                                 final_delay=final_delay, soc=self.soccfg)
        self._arm(soc, prog, device, experiment, seq)
        result = prog.acquire(soc, **acquire_kw)
        result = self._attach_shots(soc, result)
        source = environment if environment is not None else self.environment
        result = stamp_environment(result, source, device=device, soc=soc)
        if record is not None:
            from strumento.core.experiment import _emit_acquired

            _emit_acquired(record, result, experiment=experiment, reps=reps)
        return result
