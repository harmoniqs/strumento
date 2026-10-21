"""The board-object seam — real board, mock and simulator behind one protocol (D25, D32).

``AbstractSoc`` is the runtime-checkable protocol that ``AveragerProgramV2.acquire`` drives. The
concrete implementations are:

* :class:`~strumento.core.mock.MockQickSocV2` — offline, no FPGA (build/compile real, acquire faked).
* ``PyQickSoc`` — a real board's ``QickSoc`` (or the thin on-board execute-agent's local soc).
* :class:`~strumento.core.simulator.SimulatorSoc` — real v2 assembly, data from a response model.
  D29 named this class and left its mechanism open; **D32 realizes it** (response registry in
  :mod:`strumento.core.response`, the soc itself in :mod:`strumento.core.simulator`).

The protocol lists the acquire-loop-critical data-path methods; the full soc surface the acquire
path calls is enumerated in ``docs/qick-v2-api-notes.md``. Formalizing the seam is what makes the
board — and, above it, the *transport* to the board (see :mod:`strumento.core.transport`) — a swap
rather than a rewrite. That is the property the resolved D14 (no-Pyro, job-server-native) rests on:
dropping Pyro swaps one transport for another; nothing above this protocol changes.

Two **optional** capabilities extend that seam without any layer above it importing a simulator:
:class:`ArmableSoc` (a soc that must be told what is being run before acquire — the D32 arming
seam) and :class:`ShotSource` (a soc that can hand back the per-shot data behind its last
acquisition). Both are structural, so the mock and hardware socs simply do not satisfy them and
the experiment ``Runner`` leaves those paths untouched.

This module is **pure** — structural Protocols, no ``qick`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only (keeps this module import-light)
    from strumento.core.response import SimContext


@runtime_checkable
class AbstractSoc(Protocol):
    """The minimal board surface a program's ``acquire`` loop needs.

    ``isinstance(x, AbstractSoc)`` is a structural check (method presence). It intentionally lists
    the *data-path* methods that distinguish a runnable soc — ``start_readout`` (declares the
    acquisition dims) and ``poll_data`` (returns the IQ chunks), plus ``reload_mem`` / ``start_src``
    — rather than the full config surface, which is qick-internal and documented separately.
    """

    def start_readout(self, total_count, counter_addr=None, ch_list=None, reads_per_shot=None): ...

    def poll_data(self): ...

    def reload_mem(self, *args, **kwargs): ...

    def start_src(self, *args, **kwargs): ...


@runtime_checkable
class ArmableSoc(Protocol):
    """A soc that needs the run *context* before acquire (the D32 arming seam).

    A soc only ever sees ``start_readout``/``poll_data``, so a soc that generates its own data
    must be told what is being run by the component that knows — the experiment ``Runner``, which
    builds a :class:`~strumento.core.response.SimContext` (experiment, device, lab-native swept
    axis) and calls :meth:`arm` before delegating to ``acquire``.

    ``isinstance(soc, ArmableSoc)`` is a structural check for ``arm``, so the mock and a real
    board are *not* armable and their run paths are unchanged.
    """

    def arm(self, ctx: SimContext) -> None: ...


@runtime_checkable
class ShotSource(Protocol):
    """A soc that can return the **per-shot** data behind its last acquisition.

    The averaged-acquire return loses the shot axis; a soc that generated (or buffered) the shots
    exposes them here so the ``Runner`` can pin them on ``MeasurementResult.shots`` and the
    post-selection path (``heralds``/``keep_mask``, a readout kind's ``discriminate``) runs on the
    same shapes it runs on for board data. Returns ``None`` when no shots are available.
    """

    def collect_shots(self, ch: int = 0): ...
