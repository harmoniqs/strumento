"""The environment seam (D34, realizing D24) — read-only station telemetry, stamped on every result.

Every :class:`~strumento.core.result.MeasurementResult` should be able to answer "what was the
station doing when this was taken?" — fridge temperatures, LO readings, attenuator settings, which
soc produced it. ``MeasurementResult.meta["environment"]`` has been the reserved slot since the
result type was written; this module is what finally fills it.

**Why it lands early.** Stamping is the one primitive that is *retroactively impossible*: data
taken before the seam exists is unstamped forever, and no later code can reconstruct the reading.
So the seam is deliberately minimal and lands ahead of the surfaces that will use it.

Three pieces:

* :class:`EnvironmentSource` — the protocol, one method: ``snapshot() -> dict``. Structural, so a
  lab's own logger class satisfies it without importing anything from here.
* the shipped sources — :class:`StaticEnvironment` (fixed readings: a bench label, a nominal
  temperature), :class:`CallableEnvironment` (a zero-argument reader called **at acquire time**,
  which is how a live logger plugs in), and :class:`NullEnvironment` (the default: no readings,
  but the stamp still exists).
* :func:`stamp_environment` — what the experiment ``Runner`` calls once per acquisition.

**Read-only, always.** This seam *observes*. It never sets a temperature, never opens a valve,
never touches an instrument's state — fridge/instrument *control* is out of scope by design, and a
vendor integration is a config profile plus a thin adapter that still only reads. Two consequences
are enforced here: a telemetry read that raises is recorded as an error and the acquisition
survives (losing data because a logger timed out would be absurd), and an existing stamp is never
overwritten (the transport path stamps client-side at receipt; that stamp is the true one).

Optional per-line signal-chain metadata rides the **existing** wiring types
(:class:`~strumento.core.wiring.SignalChain` on a :class:`~strumento.core.wiring.Line`) rather
than a parallel schema — the annotation an instance carries today is copied into the stamp.

This module is **pure**: it never imports ``qick``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from strumento.core.result import MeasurementResult, _jsonable

#: the reserved ``meta`` key (documented on :class:`~strumento.core.result.MeasurementResult`).
ENVIRONMENT_KEY = "environment"


@runtime_checkable
class EnvironmentSource(Protocol):
    """Something that can report the station's environment as a plain dict.

    One method by design. Keys and units are the lab's business — core neither validates nor
    interprets them (the same config-is-data rule calibration follows); it only guarantees the
    reading reaches the result JSON-safely.
    """

    def snapshot(self) -> dict: ...


@runtime_checkable
class EnvironmentReporter(Protocol):
    """A soc that can describe *its own* environment (the simulator's synthetic stamp).

    Structural, like :class:`~strumento.core.soc.ArmableSoc`: the mock and a real board simply do
    not implement ``environment()`` and contribute nothing, so their paths are unchanged.
    """

    def environment(self) -> dict: ...


@dataclass(frozen=True)
class StaticEnvironment:
    """Fixed readings — a bench label, a nominal fridge temperature, a cable-run note."""

    readings: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return dict(self.readings)


@dataclass(frozen=True)
class CallableEnvironment:
    """Wraps a zero-argument reader, called **at acquire time** (that is the whole point)."""

    reader: Callable[[], Any]

    def snapshot(self) -> dict:
        value = self.reader()
        if not isinstance(value, dict):
            raise TypeError(
                f"an environment reader must return a dict of readings, got "
                f"{type(value).__name__}"
            )
        return dict(value)


class NullEnvironment:
    """No telemetry configured — the stamp still exists, with empty readings."""

    def snapshot(self) -> dict:
        return {}


#: the default source: every result is stamped even when a station reports nothing.
NO_ENVIRONMENT = NullEnvironment()


def environment_source(spec: Any = None) -> EnvironmentSource:
    """Coerce the shorthands into an :class:`EnvironmentSource`.

    ``None`` → :data:`NO_ENVIRONMENT`; a dict → :class:`StaticEnvironment`; a callable →
    :class:`CallableEnvironment`; anything with ``snapshot()`` passes through unchanged. Anything
    else is a ``TypeError`` at configuration time rather than a surprise at acquire time.
    """
    if spec is None:
        return NO_ENVIRONMENT
    if isinstance(spec, EnvironmentSource):
        return spec
    if isinstance(spec, dict):
        return StaticEnvironment(spec)
    if callable(spec):
        return CallableEnvironment(spec)
    raise TypeError(
        f"not an environment source: {type(spec).__name__}. Pass a dict of readings, a "
        f"zero-argument callable returning one, or an object with snapshot() -> dict"
    )


def snapshot_of(spec: Any = None) -> tuple[dict, str | None]:
    """Read a source, returning ``(readings, error)`` — a failed read is data, not an exception."""
    source = environment_source(spec)
    try:
        readings = source.snapshot()
    except Exception as exc:  # read-only telemetry must never cost an acquisition
        return {}, f"{type(exc).__name__}: {exc}"
    return {str(k): _jsonable(v) for k, v in readings.items()}, None


def signal_chains(device: Any) -> dict:
    """Per-line signal-chain annotations from the device's wiring (empty when none are set)."""
    lines = getattr(getattr(device, "wiring", None), "lines", None) or {}
    out: dict[str, Any] = {}
    for name, line in lines.items():
        chain = getattr(line, "chain", None)
        if chain is None:
            continue
        raw = chain.model_dump() if hasattr(chain, "model_dump") else chain
        out[str(name)] = _jsonable(raw)
    return out


def environment_stamp(spec: Any = None, *, device: Any = None, soc: Any = None) -> dict:
    """Build the stamp: when, from what source, the readings, the signal chain, the soc's own note.

    Shape (JSON-safe throughout, so ``MeasurementResult.to_wire()`` needs no special case)::

        {"stamped_at": <UTC ISO-8601>, "source": <source class name>, "readings": {...},
         "signal_chain": {<line>: {...}},   # only when an instance annotates a line
         "soc": {...}}                      # only when the soc is an EnvironmentReporter
    """
    readings, error = snapshot_of(spec)
    stamp: dict[str, Any] = {
        "stamped_at": datetime.now(timezone.utc).isoformat(),
        "source": type(environment_source(spec)).__name__,
        "readings": readings,
    }
    if error is not None:
        stamp["error"] = error
    chains = signal_chains(device)
    if chains:
        stamp["signal_chain"] = chains
    if isinstance(soc, EnvironmentReporter):
        try:
            stamp["soc"] = _jsonable(soc.environment())
        except Exception as exc:  # same rule as a telemetry read: report, never raise
            stamp["soc_error"] = f"{type(exc).__name__}: {exc}"
    return stamp


def stamp_environment(
    result: MeasurementResult,
    spec: Any = None,
    *,
    device: Any = None,
    soc: Any = None,
    overwrite: bool = False,
) -> MeasurementResult:
    """Write ``meta["environment"]`` on ``result`` (in place, returning it) — the Runner's call.

    An **existing** stamp wins unless ``overwrite=True``: whoever received the data first was
    closer to the station (the transport path stamps client-side at receipt), and re-stamping in a
    later layer would silently record the wrong moment.
    """
    if result.meta.get(ENVIRONMENT_KEY) and not overwrite:
        return result
    result.meta[ENVIRONMENT_KEY] = environment_stamp(spec, device=device, soc=soc)
    return result
