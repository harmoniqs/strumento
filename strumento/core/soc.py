"""The board-object seam (spec §19.3).

``AbstractSoc`` is the runtime-checkable protocol that ``AveragerProgramV2.acquire`` drives. The
concrete implementations are:

* :class:`~strumento.core.mock.MockQickSocV2` — offline, no FPGA (build/compile real, acquire faked).
* ``PyQickSoc`` — a real board's ``QickSoc`` (or the thin on-board execute-agent's local soc).
* ``SimulatorSoc`` — a physics simulator behind the same surface (spec §19, deferred).

The protocol lists the acquire-loop-critical data-path methods; the full soc surface the acquire
path calls is enumerated in ``docs/qick-v2-api-notes.md``. Formalizing the seam is what makes the
board — and, above it, the *transport* to the board (see :mod:`strumento.core.transport`) — a swap
rather than a rewrite. That is the property the resolved D14 (no-Pyro, job-server-native) rests on:
dropping Pyro swaps one transport for another; nothing above this protocol changes.

This module is **pure** — a structural Protocol, no ``qick`` import.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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
