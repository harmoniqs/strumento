"""Typed bring-up graphs — the vocabulary and the runner that executes them (issues #52–#54).

A bring-up campaign chains characterization experiments from a cold-started device to a
calibrated one: coarse-to-fine writers, retry-indexed escalation ladders, declared
excitation envelopes, and DC regions an operator has cleared. That graph is *pack data*
(the modality packs author it); this module is the pack-agnostic vocabulary it is
written in, the construction-time validation that refuses a malformed graph at authoring
time, and — since #53 — the runner that executes it one step per call.

**The runner (all state derived, nothing remembered):** :meth:`BringupRunner.advance` executes
AT MOST ONE step attempt per call, under an ephemeral lock file in the campaign directory
(a mutex, never authority — a heartbeat-aged or dead-pid lock is breakable). It invokes only
**sanctioned runnables** — an :class:`~strumento.core.experiment.Experiment` through its
record-arming path ``go(soc, record=child)``, or a host-bridge adapter registered with
:meth:`register_host_bridge` — and refuses arbitrary callables with an actionable error.
Advancement routes from the runnable's **returned** outcome, never from the child ledger: a
hand-appended ``gate`` event in a child record is a record entry, not a control input. Every
piece of campaign state — position, per-step attempt counts, park state, pending approvals,
cleared envelopes, excursion poisons, claims — is **derived from the campaign's event
stream** on each call, so a fresh process re-attaches and resumes after any crash (a
``step_open`` with no ``step_close`` derives as a failed attempt). Directory names are
caches; events are authority.

**Freshness (since #54, on when a store is wired):** a prerequisite is stale when
value-poisoned (a writeback changed a path this step transitively requires — the
provenance log's append order is the ordering), failure-poisoned (a step that failed since
its last pass forces its declared ``poisons`` stale, regardless of age), or
excursion-poisoned (a region-declared DC step's moves left the cleared region's extent —
the *spatial predicate*, never the envelope class; active until a providing step passes
after the poison event in the stream) — with **age as the backstop only** (a short
``max_age_s`` catches what poison cannot see). Stale handling routes per the step's
``on_stale``: ``refresh`` detours to the ROOT stale cause's providing step (consuming the
advance, under the provider's own ladder), ``park`` emits ``needs_human``, and ``proceed``
is refused for drift-sensitive requirements. The injected ``staleness`` callable REPLACES
the engine wholesale (an override hook, preserved for tests and supervisors that know better).

**Writeback (since #54):** a passing step's whole ``provides`` lands in ONE
:func:`~strumento.core.store.record_calibration` call — multi-path is the norm — stamped
``source=f"{campaign}:{step}"`` with the refinement level in provenance; an undeclared
path in an explicit payload is refused; ``max_abs``/``max_slew`` violations are refused,
never clamped, before any store mutation; 2D data banks as an artifact in the child record
with the calibration path holding a digest-checked content reference
(:func:`resolve_calibration_artifact` is the vetted reader's side); host-bridge sweep
extents beyond a declared envelope bound raise before anything runs. A refused writeback
leaves the attempt open — it derives as a failed attempt, never a phantom pass.

**Verdict vocabulary (v1):** the gate verdicts this runner derives from a returned
:class:`~strumento.core.experiment.ExperimentOutcome` are ``{"fit": fit.success}`` — a step
declaring any other gate does not pass it until the outcome grows its verdict map. Host-bridge
adapters return their verdict map directly (a mapping, or an object with a ``gates`` mapping;
the reserved payload keys ``moves``/``calibration``/``derived`` are not gates).

**What construction validates (all pure, at ``BringupPlan`` construction):**

- write exclusivity per calibration path — one set-writer; later writers must refine
  (and a refine-writer must be reachable *downstream* of its set-writer);
- edge existence — ``on_pass`` / ``on_fail`` / ``entry`` targets must name a step, a
  park (``"park"`` / ``"park:<kind>"``), or the success terminal ``"calibrated"``
  (a target naming a *waived* node is rejected as such);
- escalation ladders — ``on_fail`` routes every retry ``1..max_retries`` (no silent
  stop) and the rung at ``max_retries`` parks ``needs_human`` (never a silent stop,
  never another escalation edge);
- declared envelopes — per-channel applied-excitation maxima must be numeric (real,
  finite, non-negative); a declared DC ``region`` is a named artifact or absent;
- the declared vocabulary holds — ``on_stale`` is one of its three policy words,
  step names are unique and outside the reserved edge vocabulary, and the mapping
  fields are mappings (a malformed graph is refused with a typed error, never a
  crash).

**Autonomy (since #55, on when a safety config is wired):** a step runs ``auto`` iff every
channel's declared applied excitation sits within a cleared envelope AND its declared DC
region (if any) is cleared AND it writes no uncleared bound AND ``force_propose`` is False
— otherwise it **proposes**: compile-and-stage only, zero hardware side effects. The staged
proposal lands in a fresh child record (params, the program, the gates to be checked, the
config stamp) and the campaign parks ``kind="propose"`` carrying the proposal's
``proposal_ref``; a privileged ``approve(actor, channel, proposal_ref)`` clears the linked
park and the proposal's whole envelope set — one approve at a ladder's max rung covers
every lower rung within it, and approvals persist for the campaign's life. At the first
``advance()`` after an approve the runner re-validates the staged proposal against the
CURRENT ref (config content-id match — the freshness engine covers ``requires`` freshness,
as it does for every advance); a mismatch re-proposes with a fresh ref, never executes.
Execution then lands in a fresh child record stamped with the config actually used. The
gate arms when the runner is constructed with the **device safety config** (``safety=``);
``None`` keeps the direct-run surface (no gate), exactly as ``store=None`` keeps the storeless
one. The v1 initial config clears no RF excitation and no DC region, so entry and
region-establishing steps propose by the rule itself.

**What is deliberately NOT here:** pack graphs and store-level compare-and-swap (the
campaign is v1's only writer to its claimed ref). Agent-reachable surfaces have no
approval path — the MCP door exposes neither an approval verb nor a write path to
campaign dirs (pinned by the campaign-events suite).

**Purity:** this module imports no ``qick`` and no pack module — only the
standard library, the pure task-record substrate (:mod:`strumento.core.record`), the
experiment scaffold (:mod:`strumento.core.experiment`, itself import-safe without a board),
and the pure versioned store (:mod:`strumento.core.store`). Core enforces generically and
branches on no pack symbol (D18); ``tests/core/test_bringup.py`` and
``tests/core/test_bringup_runner.py`` pin all three.

Contract: the field names and defaults of the schema types below are a frozen contract — the
execution model, campaign records and events, derived state, freshness, writeback and the
forward-compatibility rules in this module all rely on them staying stable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strumento.core.experiment import Experiment
from strumento.core.record import (
    STALE_FACTOR,
    TERMINAL_STATES,
    RecordError,
    TaskRecord,
    artifact_path,
    atomic_write,
    list_tasks,
    now_iso,
    parse_iso,
    pid_is_alive,
    read_events,
    read_manifest,
)
from strumento.core.record import derive_state as derive_record_state
from strumento.core.store import record_calibration, unvalidated

#: A step's experiment factory: ``(device, ctx) -> runnable step``. The runner invokes
#: only sanctioned runnables; construction never calls it.
StepCallable = Callable[[Any, Any], Any]

#: The success terminal — the word an ``on_pass`` edge may name when a step's pass
#: finishes the campaign ("calibrated" is the graph's one success terminal).
CALIBRATED = "calibrated"

#: A park target's bare form; ``"park"`` and ``"park:needs_human"`` are the same park.
#: Park kinds are an open vocabulary: core interprets ``propose`` and ``needs_human``;
#: the runner holds a campaign on an unknown kind rather than releasing it.
PARK = "park"


class BringupPlanError(ValueError):
    """A malformed bring-up graph, refused at plan construction — never mid-campaign.

    The message names the offending step and, where a path is the offender, the path.
    """


@dataclass(frozen=True)
class Provides:
    """One calibration path this step writes on pass, and how it writes it.

    ``path`` names the calibration-tree node the writeback descends to and each ``mapping``
    value names the leaf beneath it — the full calibration path is ``path + "." + subfield``
    (the same path the writeback snapshot and the freshness requirements use). ``kind="set"``
    marks the first writer of a path (the only one allowed); ``kind="refine"`` a coarse→fine
    later writer (last one wins). ``mapping`` carries the derived-key → calibration-subfield
    projection; ``max_abs`` / ``max_slew`` are declared bounds that travel as data — enforced
    at execution (refusal, never clamping), not here. ``artifact=True`` marks the value a
    content reference into the child record (stability maps, cleared regions).
    """

    path: str  # the calibration-tree parent node the write lands beneath
    mapping: dict[str, str]  # derived-key -> calibration subfield
    kind: str = "set"  # "set" (first writer) | "refine" (coarse->fine, last wins)
    max_abs: float | None = None  # declared bound: |value| refusal, never clamp
    max_slew: float | None = None  # optional slew bound (host-bridge honors; DC in v1)
    artifact: bool = False  # value is a content reference into the child record


@dataclass(frozen=True)
class Requirement:
    """One calibration path a step consumes, with its freshness contract.

    ``max_age_s`` is the age backstop (``None`` = no age limit); ``drift_sensitive``
    is the flag that makes ``on_stale="proceed"`` illegal for this requirement —
    declared per-requirement pack data, checked by the runner, carried here as data.
    """

    path: str  # dotted calibration path
    max_age_s: float | None = None
    drift_sensitive: bool = False


@dataclass(frozen=True)
class BringupStep:
    """One node of a bring-up graph: an experiment factory plus its routing contract.

    Field names and defaults are a frozen contract; the runner consumes ``experiment``,
    ``envelope``, ``region``, and ``force_propose``. ``on_fail`` is retry-indexed
    escalation — rung ``k`` routes the ``k``-th failure of *this* step; ``on_stale`` is a
    policy word (``refresh`` | ``park`` | ``proceed``), not a node edge. ``poisons`` are the
    paths forced stale when this step's gates fail.
    """

    name: str
    experiment: StepCallable  # factory: (device, ctx) -> runnable step
    requires: tuple[Requirement, ...]
    provides: tuple[Provides, ...]
    gates: tuple[str, ...]  # must ALL pass (from the emitter's returned outcome)
    on_pass: str
    on_fail: dict[int, str]  # retry-indexed escalation
    max_retries: int
    envelope: Mapping[str, float]  # per-channel APPLIED excitation max (lab-native units)
    frame_check: str | None = None  # optional independent gate name
    on_stale: str = "refresh"  # pack-selectable: refresh | park | proceed
    poisons: tuple[str, ...] = ()  # paths marked value-stale when this step's gates fail
    region: str | None = None  # declared DC region artifact this step's moves stay within
    force_propose: bool = False


@dataclass(frozen=True)
class BringupPlan:
    """A whole bring-up graph: steps plus its entry node (pack data).

    Construction validates the graph (see the module docstring) and raises
    :class:`BringupPlanError` naming the offending step and path. ``waived`` names
    nodes that deliberately do not exist in this plan (waivers, e.g. the joint
    2D readout revisit) so an edge routing to one is rejected *as waived* rather than
    as a bare typo — a waiver is an authoring fact, not a step.
    """

    steps: tuple[BringupStep, ...]
    entry: str
    waived: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _check_shapes(self.steps)
        _check_writers(self.steps)
        _check_refine_ordering(self.steps)
        _check_edges(self.steps, self.entry, self.waived)
        _check_ladders(self.steps)
        _check_declarations(self.steps)


def _check_shapes(steps: tuple[BringupStep, ...]) -> None:
    """Malformed graphs are refused with a typed error, never a crash: the mapping
    fields are mappings before anything iterates them."""
    for step in steps:
        if not isinstance(step.on_fail, Mapping):
            raise BringupPlanError(
                f"step {step.name!r}: on_fail must map retry ordinals to routes —"
                f" got {step.on_fail!r}"
            )


def _check_declarations(steps: tuple[BringupStep, ...]) -> None:
    """Declared autonomy data is well-formed at authoring time: per-channel envelopes
    are numeric maxima (real, finite, non-negative — refusal is the runner's job, this
    only refuses nonsense bounds), a declared DC region names a banked artifact, and
    ``on_stale`` is one of its three policy words — it routes staleness, not nodes.
    """
    for step in steps:
        if not isinstance(step.envelope, Mapping):
            raise BringupPlanError(
                f"step {step.name!r}: envelope must map channel names to declared"
                f" maxima — got {step.envelope!r}"
            )
        for channel, value in step.envelope.items():
            if not isinstance(channel, str) or channel.strip() == "":
                raise BringupPlanError(
                    f"step {step.name!r}: envelope channel {channel!r} is not a channel"
                    " name"
                )
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise BringupPlanError(
                    f"step {step.name!r}: envelope channel {channel!r} declares a"
                    f" non-numeric maximum {value!r} — applied-excitation maxima are"
                    " real, finite, non-negative numbers"
                )
        if step.region is not None and (
            not isinstance(step.region, str) or step.region.strip() == ""
        ):
            raise BringupPlanError(
                f"step {step.name!r}: region {step.region!r} is not a named artifact —"
                " a declared DC region names a banked artifact or is absent"
            )
        if step.on_stale not in ("refresh", "park", "proceed"):
            raise BringupPlanError(
                f"step {step.name!r}: on_stale {step.on_stale!r} is not one of"
                " refresh | park | proceed — a policy word, not a node edge"
            )


def _park_kind(target: str) -> str | None:
    """The kind of a park target, or ``None`` if it is not one — bare ``"park"`` and
    ``"park:needs_human"`` are the same park (the needs_human hold)."""
    if not isinstance(target, str):
        return None
    if target == PARK:
        return "needs_human"
    if target.startswith(f"{PARK}:") and target[len(PARK) + 1 :].strip() != "":
        return target[len(PARK) + 1 :]
    return None


def _check_ladders(steps: tuple[BringupStep, ...]) -> None:
    """Every escalation ladder terminates in a park, never a silent stop: ``on_fail``
    routes every rung ``1..max_retries`` (a failure with no rung stalls the campaign)
    and the rung at ``max_retries`` parks ``needs_human`` (a terminal that routes
    anywhere else escalates past the budget the plan declared)."""
    for step in steps:
        if (
            not isinstance(step.max_retries, int)
            or isinstance(step.max_retries, bool)
            or step.max_retries < 1
        ):
            raise BringupPlanError(
                f"step {step.name!r}: max_retries must be a positive integer — the"
                " on_fail ladder covers rungs 1..max_retries"
            )
        for rung in step.on_fail:
            if not isinstance(rung, int) or isinstance(rung, bool) or rung < 1:
                raise BringupPlanError(
                    f"step {step.name!r}: on_fail rung key {rung!r} is not a retry"
                    " ordinal (a positive integer)"
                )
        rungs = frozenset(step.on_fail)
        expected = frozenset(range(1, step.max_retries + 1))
        missing = sorted(expected - rungs)
        if missing:
            listed = ", ".join(str(r) for r in missing)
            raise BringupPlanError(
                f"step {step.name!r}: on_fail must route every retry rung"
                f" 1..max_retries — never a silent stop — missing rung(s): {listed}"
            )
        extra = sorted(rungs - expected)
        if extra:
            raise BringupPlanError(
                f"step {step.name!r}: on_fail rung {extra[0]} exceeds max_retries"
                f" {step.max_retries} — the ladder covers exactly rungs 1..max_retries"
                " and its terminal parks"
            )
        terminal = step.on_fail[step.max_retries]
        kind = _park_kind(terminal)
        if kind is None:
            raise BringupPlanError(
                f"step {step.name!r}: on_fail rung {step.max_retries} (the"
                f" max_retries rung) routes to {terminal!r} — every ladder's terminal"
                " must be a park, never a silent stop"
            )
        if kind != "needs_human":
            raise BringupPlanError(
                f"step {step.name!r}: on_fail rung {step.max_retries} (the"
                f" max_retries rung) parks {kind!r} — the terminal park must be"
                " needs_human"
            )


def _is_park(target: str) -> bool:
    """Whether an edge target is a park: ``"park"`` or ``"park:<kind>"`` with a
    non-empty kind (the kind vocabulary is open — the runner holds on unknowns)."""
    return _park_kind(target) is not None


def _is_reserved_name(name: str) -> bool:
    return name == CALIBRATED or _is_park(name)


def _check_edges(
    steps: tuple[BringupStep, ...], entry: str, waived: frozenset[str]
) -> None:
    """Every edge must resolve: on_pass / on_fail targets name a step, a park, or the
    success terminal; entry names a step; and no step collides with the reserved
    vocabulary (edges stay unambiguous). Waived targets are refused as waived — a
    waiver is an authoring fact, not a node."""
    names = frozenset(s.name for s in steps)
    if entry not in names:
        raise BringupPlanError(f"entry {entry!r} names no step in the plan")
    seen: set[str] = set()
    for step in steps:
        if _is_reserved_name(step.name):
            raise BringupPlanError(
                f"step name {step.name!r} is reserved edge vocabulary (park targets /"
                f" the {CALIBRATED!r} terminal) — edges must stay unambiguous"
            )
        if step.name in seen:
            raise BringupPlanError(
                f"duplicate step name {step.name!r} — every edge to it would be ambiguous"
            )
        seen.add(step.name)
    for step in steps:
        edges = [("on_pass", step.on_pass)]
        edges += [(f"on_fail[{rung}]", target) for rung, target in step.on_fail.items()]
        for edge, target in edges:
            if target in names or target == CALIBRATED or _is_park(target):
                continue
            if target in waived:
                raise BringupPlanError(
                    f"step {step.name!r}: {edge} edge targets waived node {target!r} —"
                    " no v1 edge may route to a waived node"
                )
            raise BringupPlanError(
                f"step {step.name!r}: {edge} edge targets {target!r}, which is no step,"
                " park, or success terminal in this plan"
            )


def _step_name_edges(step: BringupStep, names: frozenset[str]) -> frozenset[str]:
    """The step a step routes to, as graph edges — only targets naming steps; park and
    success terminals are sinks."""
    targets = {step.on_pass, *step.on_fail.values()}
    return frozenset(t for t in targets if t in names)


def _reaches(start: str, target: str, edges: dict[str, frozenset[str]]) -> bool:
    """Whether a walk from ``start`` can reach ``target`` (cycles tolerated)."""
    seen: set[str] = set()
    work = [start]
    while work:
        node = work.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        work.extend(edges.get(node, frozenset()))
    return False


def _writeback_paths(provides: Provides) -> tuple[str, ...]:
    """The full calibration paths one ``provides`` entry writes: ``path + "." + subfield``
    per mapping entry — the same paths the writeback snapshot, the freshness requirements,
    and the refinement provenance all name."""
    return tuple(f"{provides.path}.{subfield}" for subfield in provides.mapping.values())


def _check_refine_ordering(steps: tuple[BringupStep, ...]) -> None:
    """Set→refine ordering, enforced on the graph: a refine-writer must be reachable
    *after* a set-writer of the same path — else it can run against a path never set.
    The path is the full calibration path (``path.subfield``), the one the writeback
    lands — so two steps may each be the first writer of *different leaves* under one
    parent (a compound writeback's set + refine + set, e.g.), while a second writer of
    the *same leaf* must refine."""
    names = frozenset(s.name for s in steps)
    edges = {s.name: _step_name_edges(s, names) for s in steps}
    set_writers: dict[str, list[str]] = {}
    for step in steps:
        for provides in step.provides:
            if provides.kind == "set":
                for full_path in _writeback_paths(provides):
                    set_writers.setdefault(full_path, []).append(step.name)
    for step in steps:
        for provides in step.provides:
            if provides.kind != "refine":
                continue
            for full_path in _writeback_paths(provides):
                writers = set_writers.get(full_path, [])
                if not writers:
                    raise BringupPlanError(
                        f"step {step.name!r}: refine-writer for path {full_path!r} has no"
                        " set-writer — the first writer of a path must declare kind='set'"
                    )
                if not any(_reaches(w, step.name, edges) for w in writers):
                    raise BringupPlanError(
                        f"step {step.name!r}: refine-writer for path {full_path!r} is not"
                        f" reachable after its set-writer {writers[0]!r} — set→refine ordering"
                        " is enforced at plan construction"
                    )


def _check_writers(steps: tuple[BringupStep, ...]) -> None:
    """Write exclusivity per calibration path: one set-writer, later writers refine.

    The path is the FULL calibration path the writeback lands (``path + "." + each mapping
    value``) — the same path the snapshot and the freshness requirements name. Two steps may
    each be the first writer of different leaves under one calibration parent; a second
    writer of the *same leaf* must declare ``kind='refine'``.

    Also refuses a step writing the same full path twice in its own provides — a single
    pass writeback (one ``record_calibration`` call, one snapshot) carries each path
    exactly once, so a double entry is ambiguous, not additive.
    """
    set_writers: dict[str, str] = {}
    for step in steps:
        seen: set[str] = set()
        for provides in step.provides:
            if provides.kind not in ("set", "refine"):
                raise BringupPlanError(
                    f"step {step.name!r}: provides path {provides.path!r} declares kind"
                    f" {provides.kind!r} — must be 'set' or 'refine'"
                )
            for full_path in _writeback_paths(provides):
                if full_path in seen:
                    raise BringupPlanError(
                        f"step {step.name!r}: duplicate provides path {full_path!r} within"
                        " one step — a single writeback carries each path exactly once"
                    )
                seen.add(full_path)
                if provides.kind == "set":
                    first = set_writers.get(full_path)
                    if first is not None:
                        raise BringupPlanError(
                            f"duplicate set-writers for calibration path {full_path!r}:"
                            f" step {first!r} and step {step.name!r} — a later writer of a"
                            " path must declare kind='refine'"
                        )
                    set_writers[full_path] = step.name


# ======================================================================================
# the autonomy layer (issue #55) — envelopes, regions, bounds, propose/approve
# ======================================================================================


@dataclass(frozen=True)
class Clearances:
    """What the operator has cleared the campaign to do **without** a fresh approval:
    per-channel applied-excitation maxima (a channel cleared at max ``m`` covers any
    step declaring ``<= m`` on it — this is what makes one approval at an escalation
    ladder's max rung cover the whole ladder), the cleared DC region artifacts, and the
    cleared bounded-writeback paths.

    Derived from the event stream: the initial device safety config (the runner's
    ``safety`` input — v1's clears nothing, so entry steps propose by the rule itself)
    ∪ the envelope sets of proposals referenced by valid ``approve`` events.
    """

    envelopes: Mapping[str, float] = field(default_factory=dict)
    regions: frozenset[str] = frozenset()
    bounds: frozenset[str] = frozenset()

    def merged(self, other: Clearances | None) -> Clearances:
        """The union of two clearance sets — per-channel maxima fold to the LARGER
        (within is what matters), regions and bounds union."""
        if other is None:
            return self
        envelopes = dict(other.envelopes)
        for channel, maximum in self.envelopes.items():
            envelopes[channel] = max(maximum, envelopes.get(channel, maximum))
        return Clearances(
            envelopes=envelopes,
            regions=self.regions | other.regions,
            bounds=self.bounds | other.bounds,
        )


def bounded_provide_paths(step: BringupStep) -> tuple[str, ...]:
    """The writeback paths this step writes a **bound** on — a ``provides`` entry
    declaring ``max_abs``/``max_slew`` (an envelope-class calibration write, e.g. a
    banked RF power bound). Writing an uncleared bound is an autonomy trigger."""
    return tuple(
        f"{provides.path}.{subfield}"
        for provides in step.provides
        if provides.max_abs is not None or provides.max_slew is not None
        for subfield in provides.mapping.values()
    )


def uncleared_reasons(step: BringupStep, clearances: Clearances) -> list[str]:
    """Why ``step`` must propose under ``clearances`` — one clause per failing
    condition, in the rule's order. Empty means auto (with ``force_propose`` checked
    separately by :func:`decides_auto`, which names it when it fires)."""
    reasons: list[str] = []
    for channel, maximum in sorted(step.envelope.items()):
        cleared = clearances.envelopes.get(channel)
        if cleared is None or cleared < maximum:
            reasons.append(
                f"channel {channel!r} applies {maximum:g} beyond its cleared envelope"
                f" ({cleared:g})" if cleared is not None
                else f"channel {channel!r} applies {maximum:g} with no cleared envelope"
            )
    if step.region is not None and step.region not in clearances.regions:
        reasons.append(f"DC region {step.region!r} is not cleared")
    for path in bounded_provide_paths(step):
        if path not in clearances.bounds:
            reasons.append(f"writes uncleared bound {path!r}")
    return reasons


def decides_auto(step: BringupStep, clearances: Clearances) -> bool:
    """The auto/propose decision, as one pure function of the step's declared
    autonomy surface and the campaign's clearances:

    a step runs ``auto`` iff every channel's declared applied excitation is within a
    cleared envelope AND every DC move is inside a cleared region (its declared region
    artifact is cleared — the per-move spatial predicate stays the freshness excursion check)
    AND it writes no uncleared bound AND ``force_propose`` is False.
    """
    if step.force_propose:
        return False
    return not uncleared_reasons(step, clearances)


def _park_clearances(event: Mapping[str, Any]) -> Clearances | None:
    """The clearance payload a propose park carries, or ``None`` when it has none
    (a routed or hand-emitted park — nothing to fold into clearances)."""
    envelope = event.get("envelope")
    region = str(event.get("region", "")).strip()
    bounds = event.get("bounds")
    if not isinstance(envelope, Mapping) and not region and not bounds:
        return None
    envelopes = {
        str(channel): float(value)
        for channel, value in (
            envelope.items() if isinstance(envelope, Mapping) else ()
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return Clearances(
        envelopes=envelopes,
        regions=frozenset({region} if region else ()),
        bounds=frozenset(str(b) for b in bounds) if isinstance(bounds, (list, tuple)) else frozenset(),
    )


def _directive_word(directive: Any) -> str:
    """The resume directive's leading word, lowercased — ``"reset: the device was
    re-thermally-cycled"`` parses as ``reset``; a free-form directive (the original
    resume shape) parses as itself and lands in the conservative default."""
    text = str(directive or "").strip()
    return text.split(":", 1)[0].strip().lower()


# ======================================================================================
# the runner (issue #53) — one-step advance, verdict routing, derived state
# ======================================================================================

#: the outcome word that routes ``on_pass`` (the runner's vocabulary; the substrate's is open).
PASS = "pass"

#: the park kinds core interprets. Anything else is legal to record (the vocabulary is open)
#: but holds the campaign as **parked-unresolved** — only a ``resume`` clears it.
PARK_PROPOSE = "propose"
PARK_NEEDS_HUMAN = "needs_human"

#: the ephemeral advance() mutex, under the campaign directory. A lock file, never authority:
#: a heartbeat-aged or dead-pid lock is breakable, and claim exclusivity derives from the
#: event streams instead (see :meth:`BringupCampaign.create`).
LOCK = "bringup.lock"


class BringupError(RuntimeError):
    """The runner refused something at runtime — an unsanctioned runnable, a colliding
    campaign claim, a lock held by a live concurrent advance, a plan/stream mismatch.

    Typed so a supervisor surfaces "the campaign cannot proceed, and here is why" as data
    rather than a traceback. (Malformed *graphs* are :class:`BringupPlanError` — refused at
    authoring time, before anything runs on a device.)
    """


@dataclass(frozen=True)
class ParkState:
    """A campaign hold: what it is waiting for, and why.

    ``unresolved`` marks a park kind outside core's interpreted pair — the recorded kind
    survives verbatim (an open vocabulary) and the campaign *holds* until a ``resume``.
    ``step`` names the step the hold is for (empty for hand-appended parks that predate
    the field); ``proposal_ref`` links a ``propose`` park to its staged proposal — a
    valid ``approve`` clears the park **by ref** (an approve naming a different proposal
    leaves it held; a ref-less park, the routed/hand-emitted shape, is cleared by any
    valid approve).
    """

    kind: str
    reason: str
    step: str = ""
    proposal_ref: str = ""

    @property
    def unresolved(self) -> bool:
        return self.kind not in (PARK_PROPOSE, PARK_NEEDS_HUMAN)


@dataclass(frozen=True)
class Claim:
    """The campaign's persistent, authoritative claim on ``(device, ref)`` — the last
    ``claim`` event in the stream. Liveness is a derivation concern (see
    :attr:`CampaignState.claim_live`), never a lock file's business."""

    device: str
    ref: str


@dataclass(frozen=True)
class CampaignState:
    """Everything a campaign is, derived from its event stream (never directory names).

    - ``position`` — the step the graph is at (the entry until a first ``step_open``);
    - ``attempts`` — per step, the failed attempts (``step_close`` failures plus
      open-but-unclosed crashed attempts) since that step's last pass;
    - ``parked`` — the current hold, if any (cleared by ``resume``, or by a valid
      ``approve`` when it is a ``propose`` park **linked to the approve's
      proposal_ref** — a ref-less propose park, the routed/hand-emitted shape, is
      cleared by any valid approve);
    - ``pending_approvals`` — ``propose`` parks no later resume or valid approve cleared;
    - ``cleared_envelopes`` — the initial clearances plus every ``proposal_ref`` a *valid*
      approve event (one carrying both actor and channel) references;
    - ``clearances`` — the envelope/region/bound sets cleared by those approvals, folded
      per-channel to the larger maximum (what makes one approve at an escalation
      ladder's max rung cover its lower rungs); the initial device safety config is the
      runner's ``safety`` input, merged in at decision time (v1's clears nothing);
    - ``poisoned`` — the paths a ``poison`` event marked (excursion poisoning), each with
      the event's timestamp: active until a *newer* write of that path lands (self-healing
      by re-measurement — a resume is a human's word about routing, never about data
      validity, so a resume clears no poison);
    - ``claim`` / ``claim_live`` — the claimed ``(device, ref)``, live while the record is
      not terminal (a terminal campaign state releases it; the runner finalizes the record
      when the graph reaches ``calibrated``, so this stays derivable plan-free);
    - ``terminal`` — ``"calibrated"`` once an ``on_pass`` edge reached the success terminal;
    - ``record_state`` — the task record's own derived state (``done``/``failed``/
      ``cancelled`` are terminal for the claim too).
    """

    position: str
    attempts: Mapping[str, int]
    parked: ParkState | None
    pending_approvals: tuple[ParkState, ...]
    cleared_envelopes: frozenset[str]
    claim: Claim | None
    claim_live: bool
    terminal: str | None
    record_state: str = "queued"
    poisoned: Mapping[str, str] = field(default_factory=dict)
    clearances: Clearances = field(default_factory=Clearances)


@dataclass(frozen=True)
class StepContext:
    """What a step factory receives beyond the device: the campaign's derived state.

    Deliberately thin — it can grow (e.g. store access or cleared-region data) without
    churning the factory signature.
    """

    state: CampaignState


@dataclass(frozen=True)
class StepOutcome:
    """The receipt of one ``advance()`` — what ran, what the returned outcome said, and
    where the graph now points. ``held`` is non-empty when nothing executed (a park hold,
    a terminal campaign); ``detour`` marks a staleness-refresh detour consuming the call
    (the main step runs on the next advance); ``proposed`` marks the call having staged
    a proposal instead of executing (the autonomy gate — zero hardware side effects, the
    campaign now parked ``propose`` awaiting an approval)."""

    step: str | None
    attempt: int | None
    passed: bool | None
    routed_to: str = ""
    held: str = ""
    detour: bool = False
    verdicts: Mapping[str, bool] | None = None
    proposed: bool = False


def _int_or(value: Any, default: int = 0) -> int:
    """Coerce an event field to int, tolerating garbage (the emitters validate; a forged
    line might not) — derivation must be total, never a crash on a corrupt payload."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _known_step(name: Any, steps: Mapping[str, BringupStep]) -> str:
    """Resolve an event's step name against the plan — a stream naming a step the plan does
    not carry is a plan/stream mismatch, and it is refused loudly rather than silently
    mis-routed."""
    text = str(name or "")
    if text not in steps:
        raise BringupError(
            f"the campaign's event stream names step {text!r}, which is not in the attached"
            " plan — the record and the plan disagree (was the campaign re-attached to the"
            " wrong plan?)"
        )
    return text


def derive_campaign_state(
    events: Iterable[Mapping[str, Any]],
    plan: BringupPlan,
    *,
    initial_envelopes: Iterable[str] = (),
    record_state: str = "queued",
) -> CampaignState:
    """Derive the whole campaign state from its event stream — the crash-safe total
    function. A fresh process re-reads the events and lands on the same state; nothing is
    remembered anywhere.

    The walk: ``step_open`` is the authority on *where* the campaign is; ``step_close``
    routes (a pass resets that step's failure count and follows ``on_pass``; a failure
    counts and follows the ``on_fail`` rung matching the count, clamped to the terminal
    rung); ``park`` holds (``resume`` clears; a valid ``approve`` clears a ``propose``
    park **linked to its proposal_ref** — a ref-less propose park, the routed/hand-emitted
    shape, is cleared by any valid approve); an approve missing its actor or channel is
    **ignored** — it clears nothing; a ``resume`` directive picks the edge — ``reset``
    zeroes the parked step's failure count (the ladder restarts at rung 1) and any other
    word retries the position step with counts kept (the conservative clamp);
    unknown ``ev`` values are skipped (readers treat them as opaque). An open attempt
    superseded by a later ``step_open`` — a crash the graph already moved past — counts
    as a failure at its chronological position, without routing (the later open says
    where the graph went). Leftover open attempts at the end of the stream — a crash
    nothing has moved past — are crashed attempts: each counts as a failure and routes
    its rung, which is exactly how a killed runner resumes.

    ``record_state`` is the task record's own derived state (the caller reads it from
    disk); terminal record states release the claim.
    """
    steps = {s.name: s for s in plan.steps}
    position = plan.entry
    attempts: dict[str, int] = {name: 0 for name in steps}
    open_attempts: dict[tuple[str, int], bool] = {}
    parked: ParkState | None = None
    pending: list[ParkState] = []
    cleared: set[str] = set(initial_envelopes)
    proposals: dict[str, Clearances] = {}
    poisoned: dict[str, str] = {}
    claim: Claim | None = None
    terminal: str | None = None

    def route(target: str, step_name: str) -> None:
        nonlocal parked, position, terminal
        kind = _park_kind(target)
        if kind is not None:
            parked = ParkState(
                kind=kind, reason=f"routed from step {step_name!r}", step=step_name
            )
            if kind == PARK_PROPOSE:
                pending.append(parked)
        elif target == CALIBRATED:
            terminal = CALIBRATED
        else:
            position = target  # plan validation guarantees this names a step

    def route_failure(step: BringupStep) -> None:
        rung = min(attempts[step.name], step.max_retries)
        route(step.on_fail.get(rung, step.on_fail[step.max_retries]), step.name)

    for event in events:
        ev = str(event.get("ev", ""))
        if ev == "claim":
            claim = Claim(device=str(event.get("device", "")), ref=str(event.get("ref", "")))
        elif ev == "step_open":
            name = _known_step(event.get("step"), steps)
            # a new open supersedes every attempt still open: nobody will close them,
            # so they are crashes the graph already moved past — counted here, in their
            # chronological position, and not routed (this open says where the graph is).
            for open_step, open_attempt in list(open_attempts):
                if (open_step, open_attempt) != (name, _int_or(event.get("attempt"), 1)):
                    attempts[open_step] += 1
                    del open_attempts[(open_step, open_attempt)]
            position = name
            open_attempts[(name, _int_or(event.get("attempt"), 1))] = True
        elif ev == "step_close":
            name = _known_step(event.get("step"), steps)
            open_attempts.pop((name, _int_or(event.get("attempt"), 1)), None)
            step = steps[name]
            if str(event.get("outcome", "")) == PASS:
                attempts[name] = 0
                route(step.on_pass, name)
            else:
                attempts[name] += 1
                route_failure(step)
        elif ev == "park":
            ref = str(event.get("proposal_ref", "")).strip()
            parked = ParkState(
                kind=str(event.get("kind", "")),
                reason=str(event.get("reason", "")),
                step=str(event.get("step", "")),
                proposal_ref=ref,
            )
            if parked.kind == PARK_PROPOSE:
                pending.append(parked)
                if ref:
                    payload = _park_clearances(event)
                    if payload is not None:
                        proposals[ref] = payload
        elif ev == "poison":
            # excursion poisoning (the runner emits these when a region-declared step's
            # moves left the cleared region): each named path is stale until a NEWER write
            # of it lands. The event's timestamp is the poison's clock.
            stamp = str(event.get("t", ""))
            for raw in event.get("paths", ()) if isinstance(event.get("paths"), (list, tuple)) else ():
                path = str(raw)
                if path:
                    poisoned[path] = stamp
        elif ev == "resume":
            # the directive picks the edge and states whether retries reset: "reset"
            # zeroes the parked step's failure count (the ladder restarts at rung 1);
            # "advance" follows the parked step's on_pass edge without executing it
            # (the operator accepts the current state and moves on); anything else —
            # including plain free-form directives — retries the position step with
            # counts kept (the conservative clamp). A resume clears no poison.
            word = _directive_word(event.get("directive"))
            was_parked = parked
            parked = None
            pending = []
            if was_parked is not None and was_parked.step:
                held = steps.get(was_parked.step)
                if held is not None:
                    if word == "reset":
                        attempts[held.name] = 0
                    elif word == "advance":
                        route(held.on_pass, held.name)
        elif ev == "approve":
            actor = str(event.get("actor", "")).strip()
            channel = str(event.get("channel", "")).strip()
            if not actor or not channel:
                continue  # hand-forged, not emitted: ignored at derivation, clears nothing
            proposal_ref = str(event.get("proposal_ref", "")).strip()
            if proposal_ref:
                cleared.add(proposal_ref)
            if parked is not None and parked.kind == PARK_PROPOSE and (
                not parked.proposal_ref or parked.proposal_ref == proposal_ref
            ):
                parked = None
                pending = [
                    p for p in pending if p.proposal_ref and p.proposal_ref != proposal_ref
                ]
            else:
                # a valid approve naming some OTHER proposal: it clears that proposal's
                # pending entry (it was approved) but leaves a differently-linked park held
                pending = [p for p in pending if p.proposal_ref != proposal_ref]
        # unknown ev values: skipped — opacity, never a failure

    for name, _attempt in list(open_attempts):  # crashed attempts derive as failures
        attempts[name] += 1
        route_failure(steps[name])

    clearances = Clearances()
    for ref in cleared:  # approvals persist for the campaign's life
        clearances = clearances.merged(proposals.get(ref))

    return CampaignState(
        position=position,
        attempts=dict(attempts),
        parked=parked,
        pending_approvals=tuple(pending),
        cleared_envelopes=frozenset(cleared),
        claim=claim,
        claim_live=claim is not None and record_state not in TERMINAL_STATES,
        terminal=terminal,
        record_state=record_state,
        poisoned=dict(poisoned),
        clearances=clearances,
    )


def _last_claim(events: Iterable[Mapping[str, Any]]) -> Claim | None:
    """The stream's last ``claim`` event as a :class:`Claim` — the persistent,
    authoritative record of what that campaign owns (a claim is a record, not a
    reservation)."""
    claim: Claim | None = None
    for event in events:
        if str(event.get("ev", "")) == "claim":
            claim = Claim(device=str(event.get("device", "")), ref=str(event.get("ref", "")))
    return claim


def _assert_claim_exclusivity(root: Path, *, device: str, ref: str) -> None:
    """One live campaign per device OR per calibration ref — derived by scanning every
    ``kind="bringup"`` task dir under ``root``. The lock file is never consulted: it is a
    mutex, not authority, and exclusivity must survive crashes of every kind."""
    for status in list_tasks(root):
        if status.manifest.kind != "bringup":
            continue  # experiments and unknown kinds claim nothing — opaque to this scan
        if status.state in TERMINAL_STATES:
            continue  # a terminal campaign state released its claim
        claim = _last_claim(read_events(status.dir))
        if claim is None:
            continue
        if claim.device == device and claim.ref == ref:
            raise BringupError(
                f"device {device!r} and calibration ref {ref!r} are already claimed by the"
                f" live campaign at {status.dir} — close it first (a terminal campaign"
                " state releases its claim)"
            )
        if claim.device == device:
            raise BringupError(
                f"device {device!r} is already claimed by the live campaign at"
                f" {status.dir} (calibration ref {claim.ref!r}) — one live campaign per"
                " device; close it first"
            )
        if claim.ref == ref:
            raise BringupError(
                f"calibration ref {ref!r} is already claimed by the live campaign at"
                f" {status.dir} (device {claim.device!r}) — one live campaign per ref;"
                " close it first"
            )


class BringupCampaign:
    """A bring-up campaign: a ``kind="bringup"`` task record wrapped with the plan that
    drives it. The record is the durable truth (events are authority); the plan is code +
    data re-supplied by the caller on every attach.

    :meth:`create` claims ``(device, ref)`` — the persistent, authoritative record of what
    the campaign owns. Exclusivity is *derived* (see :meth:`create`), never enforced by a
    lock file. :meth:`open` re-attaches an existing campaign in a fresh process without
    re-claiming.
    """

    def __init__(self, record: TaskRecord, plan: BringupPlan) -> None:
        if record.manifest.kind != "bringup":
            raise BringupError(
                f"task record {record.dir.name!r} has kind {record.manifest.kind!r}, not"
                " 'bringup' — a campaign wraps a bringup ledger"
            )
        self.record = record
        self.plan = plan

    @classmethod
    def create(
        cls,
        root: str | Path,
        plan: BringupPlan,
        *,
        device: str,
        ref: str,
        title: str = "bringup",
    ) -> BringupCampaign:
        """Mint the campaign's task record under ``root`` and claim ``(device, ref)``.

        ``root`` is the registry root (the directory whose direct children are task dirs —
        the depth convention behind task listings; children this campaign mints live under
        its own ``steps/`` and never appear in a top-level listing).

        Exclusivity is **derived**, not reserved: every ``kind="bringup"`` task dir under
        ``root`` is scanned for its live claim (the last ``claim`` event of a record whose
        state is not terminal), and a second live campaign on the same device *or* the
        same calibration ref is rejected loudly — naming the incumbent. Non-bringup and
        unknown-kind records are opaque to the scan: they claim nothing. A terminal
        campaign state releases its claim, so a closed incumbent never blocks.
        """
        if not str(device).strip() or not str(ref).strip():
            raise BringupError("a campaign claims a device and a calibration ref — both, named")
        _assert_claim_exclusivity(Path(root), device=str(device), ref=str(ref))
        record = TaskRecord.create_under(
            Path(root),
            title=str(title),
            kind="bringup",
            device=str(device),
            pid=0,
            heartbeat=False,
        )
        record.claim(str(device), str(ref))
        record.release()  # the claim is on disk; nothing holds the handle between writes
        return cls(record, plan)

    @classmethod
    def open(cls, task_dir: str | Path, plan: BringupPlan) -> BringupCampaign:
        """Re-attach to an existing campaign (a fresh process resuming after a crash)."""
        return cls(TaskRecord.open(task_dir), plan)

    # -- the file lifecycle (the record's, delegated): idempotent, never final ----------

    def __enter__(self) -> BringupCampaign:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the campaign record's event file (and timer). Idempotent; a later write
        through :attr:`record` reopens and appends."""
        self.record.close()

    def derive_state(self) -> CampaignState:
        """Re-derive the whole campaign state from disk — events are the only input."""
        return derive_campaign_state(
            read_events(self.record.dir),
            self.plan,
            record_state=derive_record_state(self.record.dir).state,
        )


class BringupRunner:
    """Executes a campaign's graph, one step attempt per :meth:`advance` call.

    The sanctioned caller is a single supervisor (a CLI verb or a job-server task) — no
    daemon. Each call takes an ephemeral lock (a mutex, never authority), re-derives the
    campaign state from the event stream, and executes at most one step:

    - a park hold, a terminal graph, or a terminal record executes nothing;
    - the freshness engine (a wired ``store`` turns it on) evaluates the step's
      requirements — value/failure/excursion poisoning with age as the backstop — and
      routes per ``on_stale``: a refresh detour *consumes the call* (the main step runs
      on the next advance, under the provider's own ladder), ``park`` holds, and
      ``proceed`` is refused for drift-sensitive requirements. An injected ``staleness``
      callable REPLACES the engine wholesale (an override hook);
    - the autonomy gate (a wired ``safety`` config turns it on): the step runs
      ``auto`` iff every channel is within a cleared envelope, its DC region is
      cleared, it writes no uncleared bound, and ``force_propose`` is False; otherwise
      it **proposes** — compile-and-stage only, zero hardware side effects, the child
      record banks the proposal and the campaign parks ``kind="propose"`` carrying the
      proposal's ``proposal_ref``. A refresh detour inherits the providing step's
      autonomy (the gate evaluates the step about to run, main or detoured). At the
      first advance after a valid approve the staged proposal is re-validated against
      the current ref — a config content-id mismatch re-proposes, never executes;
    - a region-declared step's reported moves are checked against the cleared region's
      extent — an excursion poisons the region artifact's path (a durable event);
    - the step's factory builds the runnable; the runner invokes it only through a
      sanctioned path (``Experiment.go(soc, record=child)`` or a registered host-bridge
      adapter's ``run(record=child)``) and refuses anything else, loudly; a bridge's
      declared sweep extents are refused beyond the step's envelope before it runs;
    - the verdict routes from the runnable's **returned** outcome — the child ledger is a
      record, never a control channel;
    - a passing step's whole provides lands in ONE calibration snapshot (refused bounds
      and undeclared paths raise — the attempt stays open and derives as a failure);
    - a crash between open and close leaves the attempt open, and the next derivation
      counts it as a failed attempt (the ladder routes it).

    ``device`` and ``soc`` are caller-supplied (core never auto-loads a device — the
    supervisor owns that); the step factory receives ``(device, StepContext)``; ``store``
    is the calibration store whose ref the campaign claimed (``None`` = no writeback and
    no freshness, the storeless runner); ``safety`` is the **initial device safety config** —
    the clearances the operator grants before any approval (v1's clears no RF excitation
    and no DC region, so entry steps propose by the rule itself). ``safety=None`` (the
    default) runs the runner with no autonomy gate, exactly as ``store=None`` runs
    it with no freshness and no writeback.
    """

    def __init__(
        self,
        campaign: BringupCampaign,
        *,
        device: Any = None,
        soc: Any = None,
        store: Any = None,
        staleness: Callable[[BringupStep, CampaignState], str | None] | None = None,
        safety: Clearances | None = None,
    ) -> None:
        self.campaign = campaign
        self.device = device
        self.soc = soc
        #: the calibration store (a :class:`~strumento.core.store.ConfigStore`) the writeback
        #: policy and the freshness engine read and write. ``None`` (the default) disables
        #: both: no step provides land anywhere and nothing is ever stale — a runner wired
        #: with a store runs the writeback policy and the freshness engine.
        self.store = store
        #: the staleness override seam: ``(step, state) -> detour step name | None``. When
        #: set it REPLACES the built-in freshness engine wholesale (the hook's answer is
        #: final); ``None`` uses the engine, which needs ``store`` and never detours without
        #: one. A detour names a plan step; executing it consumes the advance.
        self.staleness = staleness
        #: the initial device safety config — the clearances the operator grants before
        #: any approval (per-channel envelope maxima, cleared regions, cleared bounds).
        #: ``None`` (the default) runs the runner with NO autonomy gate; a wired
        #: config — even the empty v1 one, which clears nothing — arms it, and every step
        #: is then auto-or-propose per the rule. Effective clearances are this config
        #: merged with the approvals derived from the campaign's event stream.
        self.safety = safety

    def advance(self) -> StepOutcome:
        """Execute at most one step attempt; return its receipt."""
        with _advance_lock(self.campaign.record.dir):
            try:
                return self._advance_locked()
            finally:
                # every exit path — routed, held, parked, finalized, or raised — lets go of
                # the campaign file this advance wrote through. release(), not close(): the
                # record is the caller's, and so is any heartbeat it runs.
                self.campaign.record.release()

    # -- the advance, lock held --------------------------------------------------------

    def _finalize_calibrated(self, state: CampaignState) -> None:
        """Close the campaign record when the graph reached the success terminal —
        idempotently, so a crash between the routing pass and this write recovers on the
        next advance (a terminal campaign state releases the claim; without this, only a
        supervisor closing the record would)."""
        if state.record_state in TERMINAL_STATES:
            return  # already closed — done, failed, or cancelled by its supervisor
        self.campaign.record.done(summary=CALIBRATED)

    def _advance_locked(self) -> StepOutcome:
        state = self.campaign.derive_state()
        if state.terminal is not None:
            self._finalize_calibrated(state)  # crash-safe: finalize, then hold, idempotent
            return StepOutcome(
                step=None, attempt=None, passed=None,
                held=f"terminal:{state.terminal}", routed_to=CALIBRATED,
            )
        if state.record_state in TERMINAL_STATES:
            return StepOutcome(
                step=None, attempt=None, passed=None,
                held=f"terminal:{state.record_state}", routed_to="",
            )
        if state.parked is not None:
            hold = (
                f"parked-unresolved:{state.parked.kind}"
                if state.parked.unresolved
                else f"parked:{state.parked.kind}"
            )
            return StepOutcome(
                step=None, attempt=None, passed=None,
                held=hold, routed_to=f"{PARK}:{state.parked.kind}",
            )

        steps = {s.name: s for s in self.campaign.plan.steps}
        step = steps[state.position]
        detour = False
        if self.staleness is not None:
            detour_name = self.staleness(step, state)
            if detour_name is not None:
                step = steps[_detour_step(detour_name, steps)]
                detour = True
        else:
            freshness = self._route_freshness(step, state)
            if freshness.held:
                return StepOutcome(
                    step=None, attempt=None, passed=None,
                    held=freshness.held, routed_to=f"{PARK}:{PARK_NEEDS_HUMAN}",
                )
            if freshness.detour is not None:
                step = steps[_detour_step(freshness.detour, steps)]
                detour = True

        if step.provides and self.store is None:
            raise BringupError(
                f"step {step.name!r} declares provides but the runner has no store wired —"
                " a passing step's writeback has nowhere to land. Construct the runner with"
                " store=<ConfigStore> (the campaign's claimed ref names the branch it"
                " writes)."
            )

        attempt = state.attempts.get(step.name, 0) + 1

        # the autonomy gate: a step is auto or propose against the effective
        # clearances — the initial device safety config merged with every approval the
        # event stream carries. A proposing step compiles-and-stages ONLY (zero hardware
        # side effects) and parks kind="propose" carrying its proposal_ref.
        if self.safety is not None:
            clearances = self.safety.merged(state.clearances)
            if not decides_auto(step, clearances):
                return self._stage_proposal(step, state, attempt, clearances)
            # re-validation: executing under an approval means executing THE proposal the
            # operator approved, against the config it was staged on. A moved ref
            # (content-id mismatch) re-proposes with a fresh ref — never executes. (The
            # requires-freshness half of re-validation is the freshness engine above,
            # which routed this advance before the gate — a stale requirement detours
            # or parks before any approved execution.)
            approved = self._approved_proposal_for(step.name, state.cleared_envelopes)
            if approved is not None:
                banked = str(approved.get("config_content_id", "") or "")
                if banked != (self._current_stamp(state) or ""):
                    return self._stage_proposal(step, state, attempt, clearances)

        stamp = self._current_stamp(state) or ""
        runnable = step.experiment(self.device, StepContext(state=state))
        _refuse_unsanctioned(runnable, step.name)  # loud, before any attempt is opened
        _refuse_sweep_extents(runnable, step)  # declared bounds are refused before anything runs

        child = TaskRecord.create_under(
            self.campaign.record.dir / "steps",
            title=step.name,
            kind="experiment",
            device=self.campaign.record.manifest.device,
            pid=0,
            heartbeat=False,
            config_content_id=stamp,  # execution lands stamped with the config actually used
        )
        try:
            rel_child = child.dir.relative_to(self.campaign.record.dir).as_posix()
            self.campaign.record.step_open(step.name, attempt, rel_child)
            returned = _invoke_sanctioned(runnable, self.soc, child)
            if step.region is not None:
                # the spatial predicate: a DC step's moves are checked against the cleared
                # region's extent — an excursion poisons the region artifact's path (a
                # poison event, durable and crash-safe), and the step still routes on its
                # own verdicts. The predicate is SPATIAL, not envelope class: in-region
                # moves are inside the artifact's validity region and poison nothing.
                self._check_excursion(step, returned, state)
            verdicts = _gate_verdicts(returned)
            required = tuple(step.gates) + ((step.frame_check,) if step.frame_check else ())
            passed = all(verdicts.get(g) is True for g in required)
            if passed and step.provides:
                # the writeback lands BEFORE the pass is closed: a refused writeback leaves
                # the attempt open, which derives as a failed attempt (the ladder routes it)
                # — a pass the store never recorded is never routed as a pass.
                self._write_back(step, returned, child, state)
        except BaseException:
            raise  # a crashed attempt: step_open stays un-closed and derives as failed
        finally:
            child.close()  # the runner minted the child; it closes it on every exit path

        self.campaign.record.step_close(step.name, attempt, PASS if passed else "fail")
        routed = (
            step.on_pass
            if passed
            else step.on_fail[min(attempt, step.max_retries)]  # this attempt's failure ordinal
        )
        if routed == CALIBRATED:
            self._finalize_calibrated(self.campaign.derive_state())
        return StepOutcome(
            step=step.name,
            attempt=attempt,
            passed=passed,
            routed_to=routed,
            detour=detour,
            verdicts=dict(verdicts),
        )

    # -- the autonomy layer (issue #55) ---------------------------------------------------

    def _current_stamp(self, state: CampaignState) -> str | None:
        """The config content id the claimed ref currently resolves to — the stamp a
        staged proposal banks and an execution lands, and the re-validation anchor.
        ``None`` when no store is wired (nothing can move)."""
        if self.store is None:
            return None
        ref = state.claim.ref if state.claim is not None else ""
        return self.store.resolve_ref(ref) if ref else None

    def _approved_proposal_for(
        self, step_name: str, cleared: Iterable[str]
    ) -> Mapping[str, Any] | None:
        """The latest staged proposal for ``step_name`` whose ``proposal_ref`` a valid
        approve event cleared (approvals persist, so "the latest approved" is the one
        execution must re-validate) — or ``None`` when the step runs under an earlier
        clearance with no staged proposal of its own."""
        approved = set(cleared)
        best: Mapping[str, Any] | None = None
        for event in read_events(self.campaign.record.dir):
            if str(event.get("ev", "")) != "park" or str(event.get("kind", "")) != PARK_PROPOSE:
                continue
            if str(event.get("step", "")) != step_name:
                continue
            ref = str(event.get("proposal_ref", "")).strip()
            if ref and ref in approved:
                best = event
        return best

    def _stage_proposal(
        self,
        step: BringupStep,
        state: CampaignState,
        attempt: int,
        clearances: Clearances,
    ) -> StepOutcome:
        """Compile-and-stage a proposal — compute-and-record ONLY, zero hardware side
        effects; approving it is a separate step.

        The runnable is built (the compile) but never invoked; the proposal — params,
        the program it would run, the gates to be checked, the config stamp — is banked
        in a fresh child record; and the campaign parks ``kind="propose"`` carrying the
        proposal's operator-addressable ``proposal_ref`` (stable per step + attempt +
        config stamp, so a re-proposal after a config move carries a FRESH ref no old
        approval can clear) and the envelope set an approval of it would clear. No
        ``step_open``/``step_close`` pair brackets it — a proposal is not an attempt,
        and the step's ladder is untouched.
        """
        stamp = self._current_stamp(state) or ""
        proposal_ref = f"{step.name}:{attempt}" + (f"@{stamp[:12]}" if stamp else "")
        reasons = uncleared_reasons(step, clearances) or ["step is not cleared to run auto"]

        runnable = step.experiment(self.device, StepContext(state=state))
        _refuse_unsanctioned(runnable, step.name)  # a rogue factory is a rogue factory
        _refuse_sweep_extents(runnable, step)  # a refused extent never even stages

        with TaskRecord.create_under(
            self.campaign.record.dir / "steps",
            title=step.name,
            kind="experiment",
            device=self.campaign.record.manifest.device,
            pid=0,
            heartbeat=False,
            config_content_id=stamp,
        ) as child:  # the runner minted the child; leaving the block closes it
            payload = {
                "proposal_ref": proposal_ref,
                "step": step.name,
                "attempt": attempt,
                "envelope": dict(step.envelope),
                "region": step.region or "",
                "gates": tuple(step.gates) + ((step.frame_check,) if step.frame_check else ()),
                "provides": [
                    {
                        "path": f"{provides.path}.{subfield}",
                        "kind": provides.kind,
                        "max_abs": provides.max_abs,
                        "max_slew": provides.max_slew,
                    }
                    for provides in step.provides
                    for subfield in provides.mapping.values()
                ],
                "program": _staged_program(runnable),
                "config_content_id": stamp,
            }
            rel = "artifacts/proposal.json"
            atomic_write(
                child.artifacts / "proposal.json",
                json.dumps(payload, indent=2, sort_keys=True, default=str),
            )
            child.artifact(rel, kind="proposal")
            child.progress(
                1, of=1,
                label=f"proposal {proposal_ref} staged (compile-and-record, no acquisition)",
            )
        rel_child = child.dir.relative_to(self.campaign.record.dir).as_posix()
        bounded_paths = bounded_provide_paths(step)
        self.campaign.record.park(
            PARK_PROPOSE,
            "; ".join(reasons),
            step=step.name,
            proposal_ref=proposal_ref,
            envelope=dict(step.envelope),
            region=step.region,
            bounds=bounded_paths or None,
            child_path=rel_child,
            config_content_id=stamp or None,
        )
        return StepOutcome(
            step=step.name,
            attempt=attempt,
            passed=None,
            routed_to=f"{PARK}:{PARK_PROPOSE}",
            held=f"parked:{PARK_PROPOSE}",
            proposed=True,
        )

    # -- the writeback policy (issue #54) ------------------------------------------------

    def _write_back(
        self, step: BringupStep, returned: Any, child: TaskRecord, state: CampaignState
    ) -> None:
        """Land a passing step's whole ``provides`` in ONE calibration snapshot.

        One :func:`~strumento.core.store.record_calibration` call carries every provides
        path (multi-path is the norm), stamped ``source=f"{campaign}:{step}"`` with the
        refinement level in provenance, and the campaign ledger cross-links the new content
        id. Values come from the returned outcome's derived quantities projected through the
        declared mappings (an explicit ``calibration`` payload from a host bridge overrides);
        an undeclared path in that payload is refused — never written. Bounds
        (``max_abs``/``max_slew``) are enforced against the declared provides and violations
        are REFUSED, never clamped, before any store mutation: a refused writeback raises,
        the attempt stays open, and the ladder routes it as a failure on the next
        derivation.
        """
        ref = state.claim.ref if state.claim is not None else ""
        base_id = self.store.resolve_ref(ref) if ref else None
        if base_id is None:
            raise BringupError(
                f"step {step.name!r} passed, but the writeback has no base: calibration ref"
                f" {ref!r} resolves to nothing in the wired store — set the ref to the"
                " campaign's starting config before advancing a providing step"
            )
        base = self.store.get(base_id)

        declared = _declared_provides(step)
        derived = _outcome_derived(returned)
        explicit = _outcome_calibration(returned)
        for path in explicit:
            if path not in declared:
                raise BringupError(
                    f"step {step.name!r}: its writeback carries path {path!r}, which its"
                    " provides do not declare — an undeclared calibration path is refused,"
                    " never written"
                )

        updates: dict[str, Any] = {}
        refinement: dict[str, str] = {}
        for path, (provides, dkey) in declared.items():
            value = explicit.get(path, derived.get(dkey, _MISSING))
            if value is _MISSING:
                raise BringupError(
                    f"step {step.name!r}: provides path {path!r} has no value — derived key"
                    f" {dkey!r} is absent from the returned outcome and no explicit"
                    " calibration payload carried the path (a declared mapping must be"
                    " satisfiable)"
                )
            subfield = path.rsplit(".", 1)[1]
            if provides.artifact:
                value = _bank_artifact(step, path, subfield, value, child, self.campaign)
            updates[path] = value
            refinement[path] = provides.kind

        _refuse_bound_violations(step, declared, updates, base)

        content_id, _ = record_calibration(
            self.store,
            ref,
            base,
            updates,
            source=f"{self.campaign.record.dir.name}:{step.name}",
            timestamp=now_iso(),
            extra={"refinement": refinement},
            # The campaign banks provides paths a pack's calibration schema need not declare,
            # and it never recomposes a device from the snapshot; its write keeps the pre-#191
            # unchecked behaviour, and its own guards (_refuse_bound_violations, artifact
            # hashing) stay in force.
            validate=unvalidated,
        )
        self.campaign.record.calibration(content_id)  # the ledger's cross-link

    # -- the freshness engine (issue #54) ------------------------------------------------

    def _route_freshness(self, step: BringupStep, state: CampaignState) -> _Freshness:
        """Evaluate the current step's requirements and route per its ``on_stale`` policy.

        Three triggers make a prerequisite stale — value-poisoned, failure-poisoned,
        excursion-poisoned — with **age as the backstop only** (a short ``max_age_s``
        catches what the poison triggers cannot see). Routing: ``refresh`` names the
        providing step as a detour (it consumes the advance and runs under its own ladder);
        ``park`` emits a ``needs_human`` park and holds; ``proceed`` runs the step anyway
        and is REFUSED for drift-sensitive requirements (the flag is the declared
        pack data — an artifact-backed drift-sensitive path must never be waved through).
        """
        if self.store is None or not step.requires:
            return _Freshness()  # nothing to be stale about (the storeless runner)
        ref = state.claim.ref if state.claim is not None else ""
        last_write = self._last_writes(ref)
        base = self._base_config(ref)
        active_poisons = self._active_poisons()
        for requirement in step.requires:
            reason = self._stale_reason(requirement, state, last_write, base, active_poisons)
            if reason is None:
                continue
            if step.on_stale == "proceed":
                if requirement.drift_sensitive:
                    raise BringupError(
                        f"step {step.name!r} declares on_stale='proceed' but requirement"
                        f" {requirement.path!r} is drift-sensitive — proceeding on a stale"
                        " drift-sensitive path is refused (the flag is the pack's declared"
                        " contract, not a convention to override)"
                    )
                return _Freshness()  # legal proceed: the step runs
            if step.on_stale == "park":
                self.campaign.record.park(
                    PARK_NEEDS_HUMAN,
                    f"stale prerequisite {requirement.path!r}: {reason}",
                )
                return _Freshness(held=f"parked:{PARK_NEEDS_HUMAN}")
            target = self._refresh_target(
                requirement.path, state, last_write, base, active_poisons, frozenset()
            )
            return _Freshness(detour=target)
        return _Freshness()

    def _check_excursion(
        self, step: BringupStep, returned: Any, state: CampaignState
    ) -> None:
        """Evaluate the spatial predicate for a region-declared step: every reported move
        must sit inside the cleared region's measured extent.

        The region is a banked artifact (``step.region`` names the calibration path
        holding its content reference) whose content carries a generic ``extent`` —
        ``{"bounds": [[lo, hi], ...]}``, one inclusive box per axis — plain data core can
        check without naming a pack. A move outside poisons the region artifact's path,
        but ONLY if some requirement in the plan flags that path drift-sensitive (an
        artifact nobody declares drift-sensitive is robust to the excursion); a step
        declaring a region that is not an artifact, an extent that is not generic bounds,
        or moves it never reported are loud authoring errors — never silent passes.
        """
        assert step.region is not None  # the only caller runs this for a region step
        ref = state.claim.ref if state.claim is not None else ""
        base = self._base_config(ref)
        if base is None:
            raise BringupError(
                f"step {step.name!r} declares region {step.region!r} but the campaign's"
                f" ref {ref!r} resolves to no config — a region check has nothing to"
                " check against"
            )
        content = resolve_calibration_artifact(self.campaign.record.dir.parent, base, step.region)
        extent = content.get("extent") if isinstance(content, Mapping) else None
        bounds = extent.get("bounds") if isinstance(extent, Mapping) else None
        if not isinstance(bounds, (list, tuple)) or not bounds:
            raise BringupError(
                f"step {step.name!r} declares region {step.region!r}, whose banked"
                " artifact carries no generic extent — a region artifact's content holds"
                ' {"extent": {"bounds": [[lo, hi], ...]}}, one inclusive box per axis'
            )
        try:
            axes = [(float(lo), float(hi)) for lo, hi in bounds]
        except (TypeError, ValueError):
            raise BringupError(
                f"step {step.name!r}: region {step.region!r} declares malformed bounds"
                f" {bounds!r} — extents are [[lo, hi], ...] numeric pairs"
            ) from None

        moves = _outcome_moves(returned)
        if moves is None:
            raise BringupError(
                f"step {step.name!r} declares region {step.region!r} but its runnable"
                " reported no moves — a region-declared DC step reports the points it"
                " moved through (the \"moves\" its outcome carries), or the region"
                " contract cannot be checked"
            )
        for point in moves:
            if not isinstance(point, (list, tuple)) or len(point) != len(axes):
                raise BringupError(
                    f"step {step.name!r}: move {point!r} does not match its region's"
                    f" {len(axes)}-axis extent — moves and extent share a basis"
                )
            for (lo, hi), coordinate in zip(axes, point):
                if not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool):
                    raise BringupError(
                        f"step {step.name!r}: move {point!r} carries a non-numeric"
                        " coordinate — the spatial predicate is plain geometry"
                    )
                if not lo <= coordinate <= hi:
                    if self._region_is_drift_sensitive(step.region):
                        self.campaign.record.poison(
                            [step.region],
                            f"step {step.name!r} moved {list(point)} outside the cleared"
                            f" region {step.region!r}",
                        )
                    return  # one poison per excursion: the fact is recorded, routing is
                    # the consumer's on_stale policy — not this step's verdicts
        return

    def _region_is_drift_sensitive(self, path: str) -> bool:
        """Whether ANY requirement in the plan flags ``path`` drift-sensitive — the
        declared pack data that makes an excursion a poisoning event."""
        return any(
            requirement.path == path and requirement.drift_sensitive
            for other in self.campaign.plan.steps
            for requirement in other.requires
        )

    def _last_writes(self, ref: str) -> dict[str, tuple[int, str | None, str]]:
        """The ref's provenance log folded into ``path -> (position, timestamp, source)`` —
        position (append order) orders writes when timestamps are absent or foreign; the
        runner's own writebacks always carry timestamps."""
        last_write: dict[str, tuple[int, str | None, str]] = {}
        if not ref:
            return last_write
        for position, prov in enumerate(self.store.log(ref)):
            for path in prov.updates:
                last_write[str(path)] = (position, prov.timestamp, prov.source)
        return last_write

    def _base_config(self, ref: str) -> Any | None:
        """The ref's current config, or ``None`` when the ref resolves to nothing."""
        content_id = self.store.resolve_ref(ref) if ref else None
        return self.store.get(content_id) if content_id else None

    def _stale_reason(
        self,
        requirement: Requirement,
        state: CampaignState,
        last_write: Mapping[str, tuple[int, str | None, str]],
        base: Any | None,
        active_poisons: frozenset[str] = frozenset(),
    ) -> str | None:
        """Why this requirement is stale, or ``None`` when it is fresh. Forced triggers
        first (poisoning), then value-poison, then the age backstop."""
        path = requirement.path
        # failure-poisoned: a step that failed since its last pass forces its poisons stale
        # — before its retry and regardless of any write's age
        for other in self.campaign.plan.steps:
            if path in other.poisons and state.attempts.get(other.name, 0) > 0:
                return (
                    f"failure-poisoned by step {other.name!r} (failed since its last pass"
                    f" and declares {path!r} in its poisons)"
                )
        # excursion-poisoned: active until a providing step PASSES after the poison event
        # (stream order, never clocks — the refresh a poison triggers always follows it in
        # the append-only stream, however close the timestamps land)
        if path in active_poisons:
            return "excursion-poisoned (a region excursion; no providing step has passed since)"
        write = last_write.get(path)
        if write is None:
            if base is not None and _config_value(base, path) is not _MISSING:
                # the baseline config carries the value; its age is unknowable — the age
                # backstop says stale exactly when the pack declared one
                if requirement.max_age_s is not None:
                    return "present only in the baseline config, never written by this store"
                return None
            return "never provided (no step has written it and no baseline carries it)"
        # value-poisoned: any input of the chain that produced this path changed AFTER it
        for upstream in self._transitive_inputs(path, frozenset()):
            upstream_write = last_write.get(upstream)
            if upstream_write is not None and upstream_write[0] > write[0]:
                return (
                    f"value-poisoned: {upstream!r} was written after {path!r} (a writeback"
                    " changed a path this step transitively requires, regardless of age)"
                )
        # age — the backstop
        if requirement.max_age_s is not None:
            if write[1] is None:
                return "age unknown (the write carries no timestamp — cannot prove fresh)"
            written = parse_iso(write[1])
            if written is None:
                return "age unknown (the write's timestamp does not parse)"
            age = (datetime.now(timezone.utc) - written).total_seconds()
            if age > requirement.max_age_s:
                return f"aged {age:.0f}s beyond max_age_s {requirement.max_age_s:g}"
        return None

    def _active_poisons(self) -> frozenset[str]:
        """Paths whose excursion poison is still active: the last ``poison`` event for the
        path, un-healed by a providing step's pass LATER IN THE SAME STREAM. Stream order
        is the truth here — the refresh a poison triggers always follows it in the
        append-only event log, however close their wall clocks land."""
        last_poison: dict[str, int] = {}
        for index, event in enumerate(read_events(self.campaign.record.dir)):
            if str(event.get("ev", "")) != "poison":
                continue
            for raw in event.get("paths", ()):
                path = str(raw)
                if path:
                    last_poison[path] = index
        if not last_poison:
            return frozenset()
        healed: set[str] = set()
        for index, event in enumerate(read_events(self.campaign.record.dir)):
            if str(event.get("ev", "")) != "step_close":
                continue
            if str(event.get("outcome", "")) != PASS:
                continue
            name = str(event.get("step", ""))
            for path in last_poison:
                if (
                    index > last_poison[path]
                    and path in _provides_steps_of(self.campaign.plan, name)
                ):
                    healed.add(path)
        return frozenset(path for path in last_poison if path not in healed)

    def _provider_steps(self, path: str) -> list[BringupStep]:
        return [s for s in self.campaign.plan.steps if path in _provides_paths(s)]

    def _transitive_inputs(self, path: str, seen: frozenset[str]) -> frozenset[str]:
        """Every path the chain that produces ``path`` consumes: its provider's
        requirements, transitively (cycle-guarded — a poisoned upstream re-stales the
        whole downstream chain)."""
        inputs: set[str] = set()
        for provider in self._provider_steps(path):
            for requirement in provider.requires:
                if requirement.path in seen:
                    continue
                inputs.add(requirement.path)
                inputs |= self._transitive_inputs(
                    requirement.path, seen | {requirement.path}
                )
        return frozenset(inputs)

    def _refresh_target(
        self,
        path: str,
        state: CampaignState,
        last_write: Mapping[str, tuple[int, str | None, str]],
        base: Any | None,
        active_poisons: frozenset[str],
        seen: frozenset[str],
    ) -> str:
        """The step a refresh detour names for a stale ``path``: walk to the ROOT stale
        cause (while the provider's own requirement is stale, refresh that instead — the
        SAME staleness facts, poisons and baseline included), else the path's last
        writer, else its set-writer, else its first provider in plan order. A path no
        step provides is a loud error — the graph cannot refresh it."""
        providers = self._provider_steps(path)
        if not providers:
            raise BringupError(
                f"requirement {path!r} is stale but no step in this plan provides it — the"
                " graph cannot refresh what nothing re-provides (fix the plan or waive the"
                " requirement)"
            )
        for provider in providers:  # the root-most stale cause refreshes first
            for requirement in provider.requires:
                if requirement.path in seen:
                    continue
                if (
                    self._stale_reason(requirement, state, last_write, base, active_poisons)
                    is not None
                ):
                    return self._refresh_target(
                        requirement.path, state, last_write, base, active_poisons,
                        seen | {requirement.path},
                    )
        write = last_write.get(path)
        if write is not None:
            source_step = write[2].rsplit(":", 1)[-1] if write[2] else ""
            for provider in providers:
                if provider.name == source_step:
                    return provider.name
        for provider in providers:
            if any(
                provides.kind == "set" and path in _provides_paths(provider)
                for provides in provider.provides
            ):
                return provider.name
        return providers[0].name


def _refuse_bound_violations(
    step: BringupStep,
    declared: Mapping[str, tuple[Provides, str]],
    updates: Mapping[str, Any],
    base: Any,
) -> None:
    """Declared bounds are enforced generically against ``max_abs`` / ``max_slew`` —
    violations are refused, never clamped, before any store mutation. An artifact-backed
    path's value is a content reference (not a number) and is exempt: bounds on banked 2D
    data are authoring nonsense, and the artifact's integrity is its content hash."""
    for path, value in updates.items():
        provides = declared[path][0]
        if provides.artifact:
            continue
        if provides.max_abs is not None:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise BringupError(
                    f"step {step.name!r}: writeback path {path!r} declares max_abs"
                    f" {provides.max_abs:g} but its value {value!r} is not a number — a"
                    " bounded path carries numeric values"
                )
            if abs(value) > provides.max_abs:
                raise BringupError(
                    f"step {step.name!r}: writeback path {path!r} carries {value:g},"
                    f" beyond its declared max_abs {provides.max_abs:g} — a bound"
                    " violation is refused, never clamped"
                )
        if provides.max_slew is not None:
            old = _config_value(base, path)
            if old is not _MISSING and isinstance(old, (int, float)) and not isinstance(
                old, bool
            ):
                if abs(value - old) > provides.max_slew:
                    raise BringupError(
                        f"step {step.name!r}: writeback path {path!r} moves"
                        f" {old:g} -> {value:g} (slew {abs(value - old):g}), beyond its"
                        f" declared max_slew {provides.max_slew:g} — a bound violation is"
                        " refused, never clamped"
                    )


#: the marker a calibration value carries when it is a content reference into a child
#: record's banked artifact (2D data — stability maps, compensation, cleared regions).
ARTIFACT_REF = "artifact:"


def _bank_artifact(
    step: BringupStep, path: str, subfield: str, content: Any, child: TaskRecord,
    campaign: BringupCampaign,
) -> str:
    """Bank 2D data as an artifact in the child record and mint its content reference.

    The bytes land in the child record's ``artifacts/`` (canonical JSON), the child ledger
    records the artifact event (``kind`` = the calibration path, so a reader can find what a
    path banked), and the calibration value becomes
    ``artifact:<campaign>/steps/<child>/<rel>@<sha256>`` — content-addressed: a file edited
    after banking no longer matches its digest and refuses to resolve.
    """
    payload = json.dumps(content, sort_keys=True, separators=(",", ":"))
    name = f"{'_'.join(path.split('.'))}__{subfield}.json"
    target = child.artifacts / name
    atomic_write(target, payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    rel = target.relative_to(child.dir).as_posix()
    child.artifact(rel, kind=path)
    return f"{ARTIFACT_REF}{campaign.record.dir.name}/steps/{child.dir.name}/{rel}@{digest}"


def parse_artifact_ref(value: Any) -> tuple[str, str, str, str]:
    """Decode a content reference into ``(campaign_id, child_id, rel_path, digest)``."""
    text = str(value or "")
    if not text.startswith(ARTIFACT_REF):
        raise BringupError(
            f"{value!r} is not a content reference — an artifact-backed calibration value"
            f" starts with {ARTIFACT_REF!r}"
        )
    location, _, digest = text[len(ARTIFACT_REF):].rpartition("@")
    if not location or not digest:
        raise BringupError(
            f"malformed content reference {text!r} — expected"
            " artifact:<campaign>/steps/<child>/<rel>@<sha256>"
        )
    parts = location.split("/", 3)
    if len(parts) != 4 or parts[1] != "steps" or not all(parts):
        raise BringupError(
            f"malformed content reference {text!r} — expected"
            " artifact:<campaign>/steps/<child>/<rel>@<sha256>"
        )
    campaign_id, _, child_id, rel = parts
    return campaign_id, child_id, rel, digest


def resolve_calibration_artifact(root: str | Path, config: Any, path: str) -> Any:
    """Resolve a calibration path's banked artifact — the vetted pack reader's side of the
    content reference. The reference is registry-root-relative: ``root`` is the task
    registry the campaign lives under (the campaign record's parent). Resolution is
    containment-checked (the reference's path can never escape the child task dir) and
    digest-checked (bytes edited after banking refuse to resolve).
    """
    value = _config_value(config, path)
    if value is _MISSING:
        raise BringupError(f"calibration path {path!r} is absent from this config")
    campaign_id, child_id, rel, digest = parse_artifact_ref(value)
    task_dir = Path(root) / campaign_id / "steps" / child_id
    try:
        full = artifact_path(task_dir, rel)
    except RecordError as err:
        raise BringupError(f"content reference for {path!r} escapes its child record: {err}") from None
    if not full.is_file():
        raise BringupError(
            f"the banked artifact for {path!r} is missing: {full} does not exist — the child"
            " record's storage is gone or the reference is stale"
        )
    data = full.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise BringupError(
            f"the banked artifact for {path!r} no longer matches its content reference —"
            " the bytes changed after banking (tamper, or a task dir rewritten); refusing"
            " to resolve"
        )
    return json.loads(data)


def _detour_step(name: Any, steps: Mapping[str, BringupStep]) -> str:
    text = str(name or "")
    if text not in steps:
        raise BringupError(
            f"the staleness hook named detour {text!r}, which is no step in this plan — a"
            " refresh detour must name the step that re-provides the stale prerequisite"
        )
    return text


# ======================================================================================
# the writeback policy (issue #54) — one snapshot per pass, bounds refused never clamped
# ======================================================================================

#: the writeback's missing-value sentinel (``None`` is a legal calibration value).
_MISSING = object()


@dataclass(frozen=True)
class _Freshness:
    """The freshness engine's routing answer: a refresh detour's target step name, a hold
    (an ``on_stale: park`` already emitted), or neither (fresh — or a legal proceed)."""

    detour: str | None = None
    held: str = ""


def _provides_paths(step: BringupStep) -> frozenset[str]:
    """Every calibration path a step's provides write on pass: ``path + "." + subfield`` per
    mapping entry (an artifact provides lands its content reference the same way)."""
    return frozenset(
        f"{provides.path}.{subfield}"
        for provides in step.provides
        for subfield in provides.mapping.values()
    )


def _provides_steps_of(plan: BringupPlan, step_name: str) -> frozenset[str]:
    """The paths one named step provides (empty for unknown names — opacity, never a
    failure)."""
    for step in plan.steps:
        if step.name == step_name:
            return _provides_paths(step)
    return frozenset()


def _declared_provides(step: BringupStep) -> dict[str, tuple[Provides, str]]:
    """The declared writeback surface, in declaration order: update path -> (Provides, the
    derived key that feeds it). One entry per mapping entry — a path twice in one step is
    already refused at plan construction."""
    declared: dict[str, tuple[Provides, str]] = {}
    for provides in step.provides:
        for dkey, subfield in provides.mapping.items():
            declared[f"{provides.path}.{subfield}"] = (provides, dkey)
    return declared


def _outcome_derived(returned: Any) -> dict[str, Any]:
    """The derived quantities a runnable's *returned* outcome carries — the writeback's value
    source. An ``ExperimentOutcome`` yields its fit's ``derived``; a host-bridge mapping may
    carry a ``"derived"`` mapping beside its verdicts."""
    if isinstance(returned, Mapping):
        derived = returned.get("derived")
        return dict(derived) if isinstance(derived, Mapping) else {}
    derived = getattr(getattr(returned, "fit", None), "derived", None)
    return dict(derived) if isinstance(derived, Mapping) else {}


def _outcome_calibration(returned: Any) -> dict[str, Any]:
    """Explicit calibration values an emitter hands over keyed by full dotted path — a
    host-bridge mapping may carry a ``"calibration"`` mapping beside its verdicts. Every
    path here must be declared by the step's provides or the writeback is refused."""
    if isinstance(returned, Mapping):
        calibration = returned.get("calibration")
        return dict(calibration) if isinstance(calibration, Mapping) else {}
    calibration = getattr(returned, "calibration", None)
    return dict(calibration) if isinstance(calibration, Mapping) else {}


def _outcome_moves(returned: Any) -> list[Any] | None:
    """The DC moves a runnable reports (a host-bridge mapping's ``"moves"`` list, or an
    attribute) — the spatial predicate's input. ``None`` = none reported."""
    if isinstance(returned, Mapping):
        moves = returned.get("moves")
    else:
        moves = getattr(returned, "moves", None)
    return list(moves) if isinstance(moves, (list, tuple)) else None


def _staged_program(runnable: Any) -> dict[str, Any]:
    """The program a staged proposal banks — what the runnable *would* run, extracted
    without executing anything (compile-and-stage): an ``Experiment``'s declared
    ``sequence()`` (the pure program declaration — building it touches no hardware; the
    acquisition lives in the Runner), or a host-bridge adapter's declared
    ``sweep_extents``. Serialized by the banking writer with ``default=str``."""
    if isinstance(runnable, Experiment):
        seq, sweeps = runnable.sequence()
        return {"sequence": seq, "sweeps": sweeps}
    extents = getattr(runnable, "sweep_extents", None)
    if callable(extents):
        extents = extents()
    return {"sweep_extents": extents} if isinstance(extents, Mapping) else {}


def _config_value(config: Any, path: str) -> Any:
    """Walk a config's calibration tree by dotted path; ``_MISSING`` when the path is absent
    (dict keys only — calibration paths name dict leaves)."""
    node: Any = getattr(config, "calibration", None)
    if not isinstance(node, Mapping):
        return _MISSING
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _refuse_sweep_extents(runnable: Any, step: BringupStep) -> None:
    """Host-bridge sweep extents are refused beyond their declared envelope bound — BEFORE
    the sweep runs, so a refused extent never touches hardware.

    An adapter exposes its planned extents as a ``sweep_extents`` mapping (channel -> a
    ``(lo, hi)`` pair); the runner checks every channel against the step's declared
    envelope (per-channel APPLIED excitation maxima, lab-native units). A channel with no
    declared bound, or an extent whose ``max(|lo|, |hi|)`` exceeds it, is a loud refusal.
    Adapters that expose nothing are unchecked (the contract is opt-in).
    """
    extents = getattr(runnable, "sweep_extents", None)
    if callable(extents):
        extents = extents()
    if not isinstance(extents, Mapping) or not extents:
        return
    for channel, extent in extents.items():
        bound = step.envelope.get(str(channel))
        if bound is None:
            raise BringupError(
                f"step {step.name!r}: its runnable sweeps channel {channel!r}, which the"
                " step's envelope does not declare — an undeclared sweep channel is"
                " refused, never run"
            )
        try:
            lo, hi = extent
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError):
            raise BringupError(
                f"step {step.name!r}: channel {channel!r} declares sweep extent"
                f" {extent!r} — extents are (lo, hi) numeric pairs"
            ) from None
        if max(abs(lo), abs(hi)) > float(bound):
            raise BringupError(
                f"step {step.name!r}: its runnable sweeps channel {channel!r} over"
                f" [{lo:g}, {hi:g}], beyond the declared envelope bound {bound:g} — a"
                " sweep extent beyond its declared bound is refused, never clamped"
            )


# -- sanctioned runnables -----------------------------------------------------------------

#: host-bridge adapter classes packs register for DC/host experiments. Registration happens
#: at pack import (before any campaign runs); the registry is core-owned so the runner can
#: check it without importing a pack.
_HOST_BRIDGES: list[type[Any]] = []


def register_host_bridge(cls: type) -> None:
    """Register a host-bridge adapter class as a sanctioned runnable.

    A registered adapter is invoked as ``adapter.run(record=child)`` and its return value
    is read for gate verdicts: a mapping (gate name -> verdict) or an object carrying a
    ``gates`` mapping. This is how DC/host experiments — which have no ``go()`` — become
    runnable inside ``advance()`` without the runner ever naming a pack.
    """
    if not isinstance(cls, type):
        raise BringupError(f"register_host_bridge takes an adapter class, got {cls!r}")
    if cls not in _HOST_BRIDGES:
        _HOST_BRIDGES.append(cls)


def _refuse_unsanctioned(runnable: Any, step_name: str) -> None:
    if isinstance(runnable, Experiment):
        return
    for bridge in _HOST_BRIDGES:
        if isinstance(runnable, bridge):
            return
    raise BringupError(
        f"step {step_name!r}: its experiment factory returned {runnable!r}, which is not a"
        " sanctioned runnable. The runner invokes only (1) an Experiment, through its"
        " record-arming path go(soc, record=child), or (2) a host-bridge adapter registered"
        " with strumento.core.bringup.register_host_bridge(...). Arbitrary callables are"
        " refused: a gate verdict is only worth routing on when vetted emitter code"
        " produced it — wire the callable into one of those two paths."
    )


def _invoke_sanctioned(runnable: Any, soc: Any, child: TaskRecord) -> Any:
    """Invoke a runnable through its sanctioned path and return what it returned — the
    object advancement routes on (never the child's ledger)."""
    if isinstance(runnable, Experiment):
        return runnable.go(soc, record=child)
    for bridge in _HOST_BRIDGES:  # _refuse_unsanctioned ran first; one of these matches
        if isinstance(runnable, bridge):
            return runnable.run(record=child)
    raise BringupError(f"unsanctioned runnable {runnable!r} reached invocation")  # pragma: no cover


#: the reserved keys a host-bridge mapping may carry BESIDE its gate verdicts — payload
#: channels for the writeback and the spatial predicate, never gates themselves.
_RESERVED_OUTCOME_KEYS = ("moves", "calibration", "derived")


def _gate_verdicts(returned: Any) -> dict[str, bool]:
    """The gate verdicts a runnable's *returned* object carries.

    An ``ExperimentOutcome`` yields ``{"fit": fit.success}`` (its verdict map is the v1
    surface — see the module docstring); a host-bridge adapter returns its map directly.
    A declared gate absent from the map does not pass — absence is a verdict, not an
    error: the step's ladder routes it like any other failure. A bridge mapping's
    reserved payload keys (``moves`` / ``calibration`` / ``derived``) are not gates.
    """
    if isinstance(returned, Mapping):
        return {
            str(k): bool(v)
            for k, v in returned.items()
            if str(k) and str(k) not in _RESERVED_OUTCOME_KEYS
        }
    gates_map = getattr(returned, "gates", None)
    if isinstance(gates_map, Mapping):
        return {str(k): bool(v) for k, v in gates_map.items() if str(k)}
    success = getattr(getattr(returned, "fit", None), "success", None)
    if success is not None:
        return {"fit": bool(success)}
    return {}


# -- the ephemeral advance() mutex ----------------------------------------------------------


def _read_lock(path: Path) -> dict | None:
    """Parse the lock file; a corrupt or unreadable lock reads as stale (breakable)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _lock_is_held(info: dict) -> bool:
    """A lock is held when it is fresh (heartbeat-aged: within STALE_FACTOR × its declared
    interval) **and** its pid is alive on this host. A stale lock is breakable — the file
    is a mutex, never authority; a foreign host's pid is unknowable, so freshness alone
    decides there."""
    interval = max(1, _int_or(info.get("heartbeat_interval_s"), 1))
    stamp = parse_iso(info.get("t"))
    if stamp is None:
        return False
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if age > STALE_FACTOR * interval:
        return False
    host = str(info.get("host", ""))
    pid = _int_or(info.get("pid"), 0)
    if host and host != socket.gethostname():
        return True  # fresh, foreign host: liveness is unknowable; freshness holds it
    return pid > 0 and pid_is_alive(pid)


def _lock_stamped(interval: int) -> dict:
    """The lock's current heartbeat record — pid, host, timestamp, declared interval."""
    return {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "t": now_iso(),
        "heartbeat_interval_s": interval,
    }


@contextmanager
def _advance_lock(campaign_dir: Path) -> Iterator[None]:
    """Take the campaign's ephemeral advance() mutex, breaking a stale one.

    While held, a daemon thread re-stamps the lock's timestamp every interval (the
    heartbeat that ages it); releasing stops the thread and removes the file. A process
    killed mid-advance stops refreshing, the lock ages out, and the next advance breaks
    it — crash-safe by liveness, not by cleverness.
    """
    path = Path(campaign_dir) / LOCK
    interval = 1  # a floor for directories without a readable manifest
    try:
        if (Path(campaign_dir) / "task.toml").exists():
            interval = max(1, read_manifest(campaign_dir).heartbeat_interval_s)
    except RecordError:
        interval = 1

    if path.exists():
        existing = _read_lock(path)
        if existing is not None and _lock_is_held(existing):
            raise BringupError(
                f"another advance() holds {path}: pid {existing.get('pid')} on"
                f" {existing.get('host')!r} — the lock is a mutex, so a concurrent advance"
                " fails loudly. A stale lock (heartbeat-aged, or a dead pid) is breakable;"
                " this one is fresh."
            )
        # stale or corrupt: break it (a mutex, never authority)

    atomic_write(path, json.dumps(_lock_stamped(interval)))

    stop = threading.Event()

    def refresh() -> None:  # pragma: no cover - timing-dependent; the stale-break tests pin aging
        while not stop.wait(interval):
            try:
                atomic_write(path, json.dumps(_lock_stamped(interval)))
            except OSError:
                return  # a refresher must never be the thing that kills an advance

    refresher = threading.Thread(
        target=refresh, name="strumento-bringup-lock", daemon=True
    )
    refresher.start()
    try:
        yield
    finally:
        stop.set()
        refresher.join(timeout=min(interval, 1.0))
        try:
            path.unlink()
        except OSError:
            pass
