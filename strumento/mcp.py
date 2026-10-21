"""``strumento-mcp`` — the **stateful lab session** (D33), the second agent door.

The split between strumento's two agent surfaces is not CLI *vs* MCP; it is **stateless vs
stateful**:

* :mod:`strumento.cli` is one-shot verbs. A process loads a YAML, answers one question, exits.
  That is the right shape for CI and for a terminal agent — and it is structurally incapable of
  holding a device open across a dozen measurements.
* this module is a **long-lived session** holding ``{device, soc, store, ref, recent outcomes}``.
  That state is the whole point: a calibration loop is *measure → believe → measure again*, and the
  "believe" step has to survive between calls.

Both doors are thin over the same importable functions (``core.schemas``, ``core.store``,
``cli.device_summary`` / ``cli.flatten_calibration`` / ``cli.provenance_entries``), so the two can
never drift into two answers.

**The eight tools** are exactly D33's list::

    load_device(path, soc="simulator"|"mock", store=None, ref="main", sim=None, task=None,
                allow_sim_writes=False, chest=None)
    device_info()
    list_experiments()
    run_experiment(name, params=None)
    calibration_history()
    record_calibration(updates, outcome=None)
    schema(what)
    validate(yaml)

**Strategy-free, permanently.** There is deliberately **no** next-experiment / recommendation /
"what should I do now" tool, and one must never be added. Deciding what to measure next is the
calibration-strategy layer's job, *upstream* of this server; strumento is the strategy-agnostic
substrate underneath it (D18). A tool here that answered "what next" would quietly make strumento a
strategy, and every consumer would inherit it. The tool list is mechanism only.

**Errors are data.** Every tool returns a JSON-safe dict, ``{"ok": true, …}`` or
``{"ok": false, "error": "TypeName: message"}`` — never a traceback, and never a raised exception
crossing the protocol. Typed session errors (:class:`NoDeviceError`, :class:`NoStoreError`,
:class:`UnknownExperimentError`) carry the message an agent needs to recover on its own.

**Write-back is the load-bearing tool.** :meth:`LabSession.record_calibration` applies the update
through :func:`strumento.core.store.record_calibration` (content-addressed snapshot + provenance +
ref advance) **and then reloads the session's device from the new config**. Without that reload the
session would keep measuring with the stale belief and a closed loop would never close.

**Task records (the durable session).** ``load_device(task=...)`` opens a
:class:`~strumento.core.record.TaskRecord` for the session: a *root* directory mints a fresh task
under it; a directory that already holds a ``task.toml`` is *resumed* (events continue in the same
append-only log — a restarted loop keeps one history). Every ``run_experiment`` then emits its
acquisition/fit progress and its ``fit`` / ``fit-recovery`` gates through the record — the vetted
emitter in :meth:`strumento.core.experiment.Experiment.go`, not a session re-implementation — and
``record_calibration`` notes each store write in the same log. This is the autoresearch shape: the
task directory is the truth a supervisor (Amicode) reads, and the session's in-memory outcome list
stays what it always was — recent labels, not the record. The tool list does not change: the record
rides on the session, not on a ninth tool.

**Purity.** This module imports no ``qick`` at module scope: the soc classes
(:class:`~strumento.core.simulator.SimulatorSoc`, :class:`~strumento.core.mock.MockQickSocV2`) are
imported inside :meth:`LabSession.load_device`, so ``import strumento.mcp`` stays qick-free and only
*opening a session* needs the board layer. The MCP SDK itself is an optional extra (``[mcp]``) and is
likewise imported lazily, so the module (and :class:`LabSession`) import without it.

**Simulated writes say so, and stay out of real stores (#194).** Every provenance line a session
writes carries, in its ``extra``, the ``soc_kind``, ``simulated`` (true for every kind offered
today) and — for the simulator — ``truth_hash`` (:func:`truth_hash` of the session's ground truth);
the task record's ``calibration`` event carries the same fields. A simulator or mock session marks a
*fresh* store it creates (:data:`SIM_STORE_MARKER` at the store root) and refuses to write into an
*existing* store without that mark, unless it was opened with ``allow_sim_writes=True``. The
refusal is a session policy: :func:`strumento.core.store.record_calibration` (the library path a
human uses) writes into any store.

**A run keeps its data (#200).** Each ``run_experiment`` saves the run to a
:class:`~strumento.core.datachest.FileDataChest` entry — the ``MeasurementResult`` arrays
(``xpts``, ``avgi``, ``avgq``, ``shots``, heralds), its ``meta`` stamped with the same origin
fields, the sequence that was played and the fit summary — and cites the entry id from the reply
(``datachest``), from the task record (an ``artifact`` event of kind ``datachest``) and from a
calibration write labelled with that run (``extra["data"]``). The chest is
``load_device(..., chest=<path>)``, else the directory beside the store (:func:`chest_beside`),
else there is none and the reply says nothing was saved. A save that fails costs the citation,
never the run.

**Autonomy posture (recorded, not yet enforced).** Simulator and mock sessions are fully autonomous
— that asymmetry is the point of having a simulator. A future ``jobserver`` soc option pointed at
real hardware inherits the P4 autonomy gate: ``run_experiment`` / ``record_calibration`` become
propose-vs-execute (surfaced, not auto-run) until that gate is opened deliberately.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import types
import typing
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from strumento.cli import device_summary, flatten_calibration, provenance_entries
from strumento.core import schemas
from strumento.core.config import apply_updates
from strumento.core.datachest import FileDataChest
from strumento.core.device import Device
from strumento.core.store import FileConfigStore, Provenance
from strumento.core.store import record_calibration as store_record_calibration

#: The tools this server exposes, in D33's order. Asserted against the live server in the tests, so
#: adding or removing a tool is a deliberate, visible change to the contract.
TOOL_NAMES: tuple[str, ...] = (
    "load_device",
    "device_info",
    "list_experiments",
    "run_experiment",
    "calibration_history",
    "record_calibration",
    "schema",
    "validate",
)

#: The soc kinds a session can open. ``jobserver`` arrives with the transport-backed run path plus
#: the autonomy gate above (see the module docstring) — it is not silently available early.
SOC_KINDS: tuple[str, ...] = ("simulator", "mock")

#: The soc kinds whose data is not measured on a board. Their store writes are stamped
#: ``simulated: true`` and they keep out of unmarked stores (#194). Every kind offered today is one.
SIMULATED_SOC_KINDS: frozenset[str] = frozenset({"simulator", "mock"})

#: The file at a store root that marks the store as a **simulation store** (#194). A simulator or
#: mock session writes it into a fresh store it creates, and writes only into stores that carry it
#: (unless opened with ``allow_sim_writes=True``). Its content is informational JSON
#: (``{"simulated": true, "soc_kind": …, "marked_at": …}``); its presence is what counts.
SIM_STORE_MARKER = "simulation-store.json"

#: The suffix of the default chest directory: a store at ``<dir>/<name>`` keeps its runs' data in
#: ``<dir>/<name>-data`` (#200). A sibling, never a subdirectory, so saving data never touches the
#: store tree — its object layout and its fresh/existing rule (:func:`store_has_content`) see none of it.
CHEST_SUFFIX = "-data"

#: The store's own subdirectories, which a chest may not live in (#200): a file there would make a
#: fresh store look existing and would sit among its content-addressed objects.
STORE_SUBDIRS: tuple[str, ...] = ("objects", "refs", "log")

#: Keys in ``run_experiment``'s ``params`` that belong to the *run*, not to the experiment's
#: constructor. ``params`` is otherwise constructor keyword arguments (D33); ``reps`` is separated
#: out because it is the one knob that decides whether a fit can converge at all, and no shipped
#: experiment takes a constructor argument by that name.
RUN_PARAMS: frozenset[str] = frozenset({"reps", "rounds", "final_delay"})

#: How many recent outcomes a session remembers in memory (labels for provenance and a short
#: history). The durable record is the store's log, and — for a session opened with ``task=`` —
#: the task record's append-only event log; this list is a cache, not the record.
MAX_OUTCOMES = 32


class SessionError(RuntimeError):
    """A session-lifecycle error, reported as data with the message an agent can act on."""


class NoDeviceError(SessionError):
    """A stateful tool was called before ``load_device``."""


class NoStoreError(SessionError):
    """A store-dependent tool was called in a session opened without a store."""


class UnknownExperimentError(SessionError):
    """``run_experiment`` was given a name the loaded pack's registry does not have."""


class CeilingWriteError(SessionError):
    """``record_calibration`` touched a calibration path the pack declares a gain ceiling (#190)."""


class UnmarkedStoreError(SessionError):
    """A simulated session was about to write into an existing store not marked for simulation (#194)."""


class UnknownOutcomeError(SessionError):
    """``record_calibration(outcome=…)`` named no fitted outcome this session remembers (#194)."""


# --------------------------------------------------------------------------------------
# where a store write came from (#194)
# --------------------------------------------------------------------------------------


def truth_hash(truth: dict | None) -> str:
    """The SHA-256 hex digest of the **canonical JSON** of a simulator's ground truth.

    Canonical JSON is ``json.dumps(truth, sort_keys=True, separators=(",", ":"),
    ensure_ascii=True)``: keys sorted at every depth, no whitespace, numbers as Python's ``json``
    writes them (so ``9350`` and ``9350.0`` are different truths, as they are different JSON), and
    anything that is not a JSON value written as its ``str``. ``None`` or an empty mapping hashes
    ``{}`` — a simulator with no truth armed, which reports the device's own belief.
    """
    text = json.dumps(dict(truth or {}), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_origin(soc_kind: str, truth: dict | None) -> dict:
    """The fields every store write from a session carries: ``soc_kind``, ``simulated``,
    and — for the simulator only — ``truth_hash``. They go in a provenance line's ``extra`` and on
    the task record's ``calibration`` event."""
    origin: dict[str, Any] = {"soc_kind": soc_kind, "simulated": soc_kind in SIMULATED_SOC_KINDS}
    if soc_kind == "simulator":
        origin["truth_hash"] = truth_hash(truth)
    return origin


def store_has_content(root: str | Path) -> bool:
    """Whether a store root already holds a calibration history: any file under ``objects/``,
    ``refs/`` or ``log/``. A store without one — an absent root, or the empty directories a
    :class:`~strumento.core.store.FileConfigStore` creates when constructed — is **fresh**."""
    base = Path(root)
    return any(
        path.is_file()
        for sub in ("objects", "refs", "log")
        if (base / sub).is_dir()
        for path in (base / sub).rglob("*")
    )


def chest_beside(store_root: str | Path) -> Path:
    """The default chest for a store: the sibling ``<store_root>-data``, as an absolute path (#200).

    ``/lab/store`` keeps its runs' data in ``/lab/store-data``. The path is absolute because it is
    cited from places that do not share the session's working directory (a store's log, a task
    record).
    """
    root = Path(store_root).resolve()
    return root.parent / f"{root.name}{CHEST_SUFFIX}"


def _resolve_chest(chest: str | Path | None, store: str | Path | None) -> tuple[Path | None, str]:
    """``chest=`` / ``store=`` → ``(chest path or None, source)``. Reads only; creates nothing.

    ``source`` is ``"argument"`` for an explicit ``chest=``, ``"beside-store"`` for the default, or
    ``""`` when there is no chest. A chest that is an existing non-directory, or that lies inside the
    store's ``objects/``, ``refs/`` or ``log/``, is refused.
    """
    if chest is not None and str(chest).strip():
        path, source = Path(chest).resolve(), "argument"
    elif store:
        path, source = chest_beside(store), "beside-store"
    else:
        return None, ""
    if path.exists() and not path.is_dir():
        raise SessionError(f"chest {path} exists and is not a directory; pass chest=<a directory>")
    if store:
        root = Path(store).resolve()
        for sub in STORE_SUBDIRS:
            if path == root / sub or (root / sub) in path.parents:
                raise SessionError(
                    f"chest {path} is inside the store's {sub}/ directory, which holds the store's "
                    "own files; pass chest=<a directory outside objects/, refs/ and log/> or omit "
                    f"it (the default is {chest_beside(store)})"
                )
    return path, source


def store_is_marked(root: str | Path) -> bool:
    """Whether a store root carries the simulation mark (:data:`SIM_STORE_MARKER`)."""
    return (Path(root) / SIM_STORE_MARKER).is_file()


def _mark_store(root: Path, soc_kind: str) -> None:
    (root / SIM_STORE_MARKER).write_text(json.dumps(
        {"simulated": True, "soc_kind": soc_kind, "marked_at": _now()}, indent=2,
    ) + "\n")


def _unmarked_refusal(root: Path, soc_kind: str, what: str) -> UnmarkedStoreError:
    return UnmarkedStoreError(
        f"refused: {root} is an existing calibration store without the simulation mark "
        f"({SIM_STORE_MARKER}), and a {soc_kind} session does not {what} into it; nothing was "
        "written. Either open a fresh store (a new store root — a simulator or mock session marks "
        "the store it creates), or re-open this one with load_device(..., allow_sim_writes=True); "
        "the provenance still records simulated: true"
    )


# --------------------------------------------------------------------------------------
# pack experiment registry (the new pack-contract member, D33)
# --------------------------------------------------------------------------------------


def experiments_for(device) -> dict[str, type]:
    """The loaded pack's ``EXPERIMENTS`` registry (``{}`` for a pack that declares none).

    ``core`` never imports a pack by name (hard rule 2) and neither does this module: the registry
    is read off the pack object the device was composed from, exactly as ``CALIBRATION`` and
    ``compose_elements`` are.
    """
    return dict(getattr(getattr(device, "pack", None), "EXPERIMENTS", {}) or {})


def ceilings_for(device) -> tuple[str, ...]:
    """The loaded pack's ``CALIBRATION_CEILINGS``: dotted paths an agent may not write (#190).

    Read off the pack object like :func:`experiments_for`, so this module names no field: a pack
    declares its own ceilings (``*`` is a wildcard segment — any dict key or list index), and a
    pack that declares none has none.
    """
    return tuple(getattr(getattr(device, "pack", None), "CALIBRATION_CEILINGS", ()) or ())


def _path_matches(path: str, pattern: str) -> bool:
    keys, want = path.split("."), pattern.split(".")
    return len(keys) == len(want) and all(w in ("*", k) for k, w in zip(keys, want))


def _ceiling_values(node: Any, pattern: list[str], prefix: str = "") -> dict[str, Any]:
    """``{concrete path: value}`` for every place ``pattern`` reaches in a calibration tree."""
    if not pattern:
        return {prefix.rstrip("."): node}
    head, rest = pattern[0], pattern[1:]
    if isinstance(node, dict):
        items = list(node.items()) if head == "*" else ([(head, node[head])] if head in node else [])
    elif isinstance(node, list):
        items = list(enumerate(node)) if head == "*" else []
    else:
        return {}
    out: dict[str, Any] = {}
    for key, child in items:
        if rest or child is not None:
            out.update(_ceiling_values(child, rest, f"{prefix}{key}."))
    return out


def ceiling_writes(device, updates: dict) -> list[str]:
    """The ceiling paths an update touches: named directly, or changed by a wider write.

    A path that *is* a declared ceiling counts whatever its value. A write to a whole section
    (``modes.<name>`` or ``modes``) counts when the ceilings under it would differ afterwards —
    raised, lowered, added or removed. Pure: computed on a copy of the config, before any store
    call, and independent of the write-back validator.
    """
    patterns = ceilings_for(device)
    if not patterns:
        return []
    named = [path for path in updates if any(_path_matches(str(path), p) for p in patterns)]
    try:
        config, _ = apply_updates(device.to_config(), updates)
    except Exception:  # a malformed update fails the store call itself; nothing is written
        return named
    before_calib, after_calib = device.to_config().calibration, config.calibration
    changed: list[str] = []
    for p in patterns:
        before = _ceiling_values(before_calib, p.split("."))
        after = _ceiling_values(after_calib, p.split("."))
        changed.extend(k for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k))
    return named + [k for k in changed if k not in named]


def experiment_signature(cls) -> dict[str, str]:
    """``{keyword: default-or-type}`` for an experiment's constructor, for ``list_experiments``.

    Introspection, not a hand-maintained table — a new keyword shows up in the tool output the
    moment it exists, and cannot go stale.
    """
    out: dict[str, str] = {}
    try:
        params = inspect.signature(cls.__init__).parameters
        hints = typing.get_type_hints(cls.__init__)
    except Exception:  # pragma: no cover - a pathological signature must not break listing
        return out
    for name, param in params.items():
        if name in ("self", "device", "runner") or param.kind is param.VAR_KEYWORD:
            continue
        want = hints.get(name)
        label = getattr(want, "__name__", None) or (str(want) if want is not None else "any")
        default = "" if param.default is inspect.Parameter.empty else f" = {param.default!r}"
        out[name] = f"{label}{default}"
    return out


def coerce_params(cls, params: dict) -> dict:
    """Build typed constructor arguments from a JSON host's plain dicts and lists.

    A JSON caller has no :class:`~strumento.core.sweeps.Sweep` and no readout object, so a value is
    coerced against the **constructor's own annotation**: a dict becomes ``Annotation(**value)`` and
    a list becomes ``Annotation(*value)``. That is generic (it reads the signature, it knows no
    physics — hard rule 2 holds here too) and it keeps the registry honest: whatever an experiment
    declares is what a host can pass::

        {"gains": {"start": 0, "stop": 30000, "on": "amp"}, "points": 61}
                                                             -> a bound sweep on a declared axis (D36)
        {"delays": {"name": "t", "start": 0, "stop": 250, "points": 41}}
                                                             -> a legacy per-name sweep (deprecated)
        {"readout": {"threshold": 0.5}}                      -> DispersiveIQ(threshold=0.5)

    Anything already of the right shape (a scalar, a string, a value whose annotation is a builtin)
    passes through untouched.
    """
    try:
        hints = typing.get_type_hints(cls.__init__)
    except Exception:  # pragma: no cover - unresolvable annotations: pass values through as-is
        return dict(params)
    return {name: _coerce(value, hints.get(name), cls, name) for name, value in params.items()}


def _coerce(value: Any, want: Any, cls, name: str) -> Any:
    target = _target_type(want)
    if target is None or isinstance(value, target):
        return value
    try:
        if isinstance(value, dict):
            return target(**value)
        if isinstance(value, (list, tuple)):
            return target(*value)
    except Exception as err:
        raise SessionError(
            f"{cls.__name__}({name}=…): cannot build a {target.__name__} from {value!r} — {err}"
        ) from err
    return value


def _target_type(want: Any) -> type | None:
    """The constructible class behind an annotation, or ``None`` when there is nothing to build.

    ``Sweep | None`` yields ``Sweep``; ``str`` / ``float`` / ``bool`` yield ``None`` (a JSON scalar
    is already the right thing), and so does an un-annotated or exotic parameter.
    """
    if want is None:
        return None
    if typing.get_origin(want) in (typing.Union, types.UnionType):
        for arg in typing.get_args(want):
            found = _target_type(arg)
            if found is not None:
                return found
        return None
    if not isinstance(want, type) or issubclass(want, (str, bytes, bool, int, float)):
        return None
    return want


# --------------------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------------------


class LabSession:
    """One open lab session: a device, a soc, optionally a store + the ref it tracks, and outcomes.

    Importable and drivable directly (``LabSession().load_device(...)``) — the MCP tools are eight
    thin wrappers around these methods, so a host that speaks Python needs no protocol at all, and
    the tests can exercise session logic without a transport.

    Every method returns a JSON-safe dict and raises only the typed :class:`SessionError` family;
    :func:`build_server` converts those (and anything else) into ``{"ok": false, "error": …}``.
    """

    def __init__(self) -> None:
        self.device: Device | None = None
        self.soc = None
        self.store: FileConfigStore | None = None
        self.ref: str = "main"
        self.device_path: str = ""
        self.soc_kind: str = ""
        self.truth: dict = {}
        self.allow_sim_writes: bool = False
        self.outcomes: list[dict] = []
        self.record = None          # a TaskRecord, when the session was opened with task=
        self.chest: Path | None = None   # where each run's data is saved (#200); None = nowhere
        self.chest_source: str = ""      # "argument" | "beside-store" | "" (no chest)

    # -- lifecycle ----------------------------------------------------------------------

    def __enter__(self) -> LabSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the session-scoped task record (its heartbeat and event file). Idempotent.

        The record's log stands; only this process's hold on it goes. Replacing the session
        (:meth:`load_device`) does the same for the record it replaces.
        """
        record, self.record = self.record, None
        if record is not None:
            record.close()

    # -- tool: load_device ------------------------------------------------------------

    def load_device(
        self,
        path: str,
        soc: str = "simulator",
        store: str | None = None,
        ref: str = "main",
        sim: dict | str | None = None,
        task: str | None = None,
        allow_sim_writes: bool = False,
        chest: str | None = None,
    ) -> dict:
        """Open a session on a device instance YAML. Replaces any session already open.

        Args:
            path: the instance YAML (``devices/<id>/device.yaml``).
            soc: ``"simulator"`` (real v2 assembly, simulated acquire — needs a v2 snapshot) or
                ``"mock"`` (real v2 assembly, zero-valued acquire).
            store: a calibration-store root. Created if absent. If the ``ref`` already resolves in
                that store, **the store wins** — it is the durable belief, and the YAML is only the
                seed. Otherwise the instance's own config is snapshotted as the ref's first version,
                so the history has a genesis entry.
            ref: the named ref this session tracks (``record_calibration`` advances it).
            sim: the simulator's **ground truth** — a partial calibration dict, or a path to a
                JSON/YAML file holding one. Keys override the device's belief inside the response
                model; everything unstated follows the belief. Omitted, the simulator tells the
                truth and every fit recovers what the device already believes.
            task: open a task record for this session. A *root* directory mints a fresh
                :class:`~strumento.core.record.TaskRecord` under it (``target`` follows the soc
                kind); a directory already holding a ``task.toml`` is **resumed** — the session's
                events append to the same log, so a restarted loop keeps one history. With a record
                open, every ``run_experiment`` emits its progress and gates through it and each
                ``record_calibration`` notes the store write. The record heartbeats from this
                process; when the process dies, liveness derivation marks it stale — no invented
                terminal state. The record's ``target`` is ``"sim"``: every soc kind offered here
                is simulated.
            allow_sim_writes: let this simulated session write into an existing store that is not
                marked for simulation (#194). Without it, a ``simulator`` or ``mock`` session
                refuses such a store: ``load_device`` is refused when it would seed a new ref
                there, and ``record_calibration`` is refused. A store is **fresh** when it holds
                no file under ``objects/``, ``refs/`` or ``log/`` (absent, or empty directories);
                a fresh store a simulated session opens is marked for simulation
                (``simulation-store.json`` at its root) and its writes proceed. Allowed writes still
                record ``simulated: true``, and the store is not marked.
            chest: a directory where each ``run_experiment`` saves its data, as one
                :class:`~strumento.core.datachest.FileDataChest` entry per run (#200). Omitted
                with a ``store``, the chest is the store's sibling ``<store>-data`` (beside the
                store, never inside it). With neither, runs are not saved and every reply says so
                (``chest: {opened: false, reason}``). ``load_device`` only resolves the path — the
                directory is created by the first save. A chest that is an existing file, or that
                lies inside the store's ``objects/``, ``refs/`` or ``log/``, is refused.

        Every provenance line the session writes (the genesis seed and each
        ``record_calibration``) carries ``soc_kind``, ``simulated`` and, for the simulator,
        ``truth_hash`` — the SHA-256 of the canonical JSON of ``sim`` — in its ``extra``.

        **A replace is atomic (#192).** The candidate session is built in locals — store, ref,
        device, soc,         task record, chest — and assigned to the session only once every step has succeeded.
        A step that raises (no soccfg snapshot, ``sim=`` with the mock, an unreadable store ref, a
        task dir that cannot be opened, a refused chest) leaves the previous session exactly as it
        was, still heartbeating, and writes nothing to the new store or the new chest: a new ref is
        seeded only after the soc and the record have opened.
        """
        if soc not in SOC_KINDS:
            raise SessionError(f"unknown soc {soc!r}; this phase offers: {', '.join(SOC_KINDS)}")
        device = Device.load(path)
        truth = _resolve_truth(sim)
        new_chest, chest_source = _resolve_chest(chest, store)   # a path only: nothing created

        simulated = soc in SIMULATED_SOC_KINDS
        allow = bool(allow_sim_writes)

        # 1. read the new store (no writes): the ref's version wins over the YAML when it resolves
        new_store = FileConfigStore(store) if store else None
        new_ref = ref or "main"
        current = None
        mark = False
        if new_store is not None:
            fresh = not store_has_content(new_store.root)
            current = new_store.resolve_ref(new_ref)
            if current:
                device = Device.from_config(new_store.get(current))
            elif simulated and not fresh and not allow and not store_is_marked(new_store.root):
                # seeding is a write: refused here, before anything opens or is written (#194)
                raise _unmarked_refusal(new_store.root, soc, f"seed the ref {new_ref!r}")
            mark = simulated and fresh
        # 2. open the soc and the record — the steps that refuse ordinary mistakes
        new_soc = self._open_soc(soc, device, truth)
        new_record = self._open_task(task, path, soc) if task else None
        # 3. only now write: seed a ref that did not resolve
        try:
            store_view = self._sync_store(new_store, new_ref, device, path, current,
                                          origin=write_origin(soc, truth), mark=mark, soc=soc)
        except BaseException:
            if new_record is not None:
                new_record.close()
            raise

        # 4. commit, then let go of the replaced record (it stops heartbeating; its log stands)
        replaced = self.record
        self.store, self.ref = new_store, new_ref
        self.soc, self.record = new_soc, new_record
        self.device, self.device_path, self.soc_kind, self.truth = device, str(path), soc, truth
        self.allow_sim_writes = allow
        self.chest, self.chest_source = new_chest, chest_source
        self.outcomes = []
        if replaced is not None:
            replaced.close()
        return {
            "ok": True,
            "device": {"path": str(path), **device_summary(device)},
            "soc": _soc_view(soc, truth),
            "store": store_view,
            "task": _task_view(self.record),
            "chest": self._chest_view(),
            "experiments": sorted(experiments_for(device)),
        }

    @staticmethod
    def _open_task(task: str, path: str, soc: str):
        """``task=`` → a fresh record under a root, or a resume of an existing task dir.

        The record is session-scoped: it heartbeats from this process, and :meth:`load_device`
        closes it when the session is replaced. Resuming an existing dir never rewrites the
        manifest's ``pid`` (the single-writer rule) — the event log and the heartbeats are the
        session's to keep appending.
        """
        from strumento.core.record import MANIFEST, TaskRecord

        tdir = Path(task)
        if (tdir / MANIFEST).exists():
            return TaskRecord.open(tdir, heartbeat=True)
        return TaskRecord.create_under(
            tdir,
            title=f"{Path(path).stem} session",
            device=Path(path).name,
            target="sim" if soc in SIMULATED_SOC_KINDS else "hw",
        )

    @staticmethod
    def _sync_store(store, ref: str, device, path, current: str | None, *,
                    origin: dict, mark: bool, soc: str) -> dict:
        """The store view of a loading session, seeding ``ref`` with ``device`` if it did not resolve.

        The only store *writes* of :meth:`load_device` — the simulation mark on a fresh store, then
        the seed — and its last fallible step: by the time it runs the soc and the task record have
        opened, so a refused session never marks or seeds a store.
        """
        if store is None:
            return {"opened": False}
        if current:
            return {"opened": True, "ref": ref, "content_id": current, "seeded": False,
                    "simulation_mark": store_is_marked(store.root)}
        if mark:
            _mark_store(store.root, soc)
        config = device.to_config()
        content_id = store.put(config)
        store.append_provenance(ref, Provenance(
            content_id=content_id, source=f"loaded {Path(path).name}", timestamp=_now(),
            extra=dict(origin),
        ))
        store.set_ref(ref, content_id)
        return {"opened": True, "ref": ref, "content_id": content_id, "seeded": True,
                "simulation_mark": store_is_marked(store.root)}

    @staticmethod
    def _open_soc(kind: str, device, truth: dict):
        """Build the session's soc. The **only** place this module touches the qick-facing layer."""
        if device.soccfg_snapshot is None:
            raise SessionError(
                f"this instance carries no soccfg snapshot, so a {kind} soc cannot assemble a "
                "program against it; use a device with a v2 snapshot, e.g. the cqed exemplar "
                "devices/multimode_demo/device.yaml"
            )
        if kind == "simulator":
            from strumento.core.simulator import SimulatorSoc

            return SimulatorSoc.from_device(device, truth=truth)
        from strumento.core.mock import MockQickSocV2

        if truth:
            raise SessionError(
                "sim= is the simulator's ground truth and means nothing to the mock soc, whose "
                "contract is zero-valued data; open the session with soc='simulator'"
            )
        return MockQickSocV2(device.soccfg_snapshot)

    # -- tool: device_info ------------------------------------------------------------

    def device_info(self) -> dict:
        """Topology, namespaces, wiring, and the live calibration — belief, as the session holds it.

        ``calibration_paths`` is the flattened dotted-path view, which is the *same grammar*
        :meth:`record_calibration` accepts — so an agent reads a value here and writes it back
        there without inventing a path.
        """
        device = self._require_device()
        calibration = device.calib.model_dump()
        return {
            "ok": True,
            "device": {"path": self.device_path, **device_summary(device)},
            "soc": _soc_view(self.soc_kind, self.truth),
            "store": self._store_view(),
            "task": _task_view(self.record),
            "chest": self._chest_view(),
            "calibration": _jsonable(calibration),
            "calibration_paths": _jsonable(flatten_calibration(calibration)),
            "content_id": device.to_config().content_id(),
            "recent_outcomes": [_outcome_brief(o) for o in self.outcomes[-5:]],
        }

    # -- tool: list_experiments -------------------------------------------------------

    def list_experiments(self) -> dict:
        """The loaded pack's experiment registry: name, class, one-line doc, constructor keywords,
        and ``needs_shots`` — whether the experiment needs per-shot data (#199). On a soc that
        returns averages only (the mock; a board, until it has a shot path) a ``needs_shots``
        experiment refuses: ``fit.success`` false, ``fit.derived["refused"]`` says why."""
        device = self._require_device()
        registry = experiments_for(device)
        return {
            "ok": True,
            "pack": device.pack.__name__.rsplit(".", 1)[-1],
            "experiments": [
                {
                    "name": name,
                    "class": cls.__name__,
                    "doc": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
                    "params": experiment_signature(cls),
                    "needs_shots": bool(getattr(cls, "needs_shots", False)),
                }
                for name, cls in sorted(registry.items())
            ],
        }

    # -- tool: run_experiment ---------------------------------------------------------

    def run_experiment(self, name: str, params: dict | None = None) -> dict:
        """Construct the named experiment with ``params`` and run it on the session's soc.

        ``params`` are the experiment's **constructor keyword arguments**, coerced from JSON shapes
        (see :func:`coerce_params`), plus the run-level keys in :data:`RUN_PARAMS`. Progress bars are
        always off — this is a protocol surface, not a terminal.

        ``fit`` is ``{success, params, stderr, derived}``. ``params`` and ``stderr`` are numbers;
        ``derived`` holds the physical quantities and may also carry strings, booleans, lists and
        objects an analyzer reports beside them (a protocol name, a confusion matrix, a refusal's
        reason) — those arrive as given. A NaN or infinite number, at any depth, arrives as
        ``null`` (NaN is not JSON; an unestimated stderr is the common case).

        An experiment whose ``analyze`` returns nothing (no fitter wired) is a first-class outcome:
        ``fit`` comes back ``null`` alongside the data-shape summary, rather than an error.

        **The run's data is kept (#200).** With a chest (``load_device(chest=…)``, or beside the
        store), the run is saved as one :class:`~strumento.core.datachest.FileDataChest` entry —
        ``xpts``, ``avgi``, ``avgq``, ``shots`` and heralds, the result's ``meta`` plus ``soc_kind``,
        ``simulated``, ``truth_hash`` (simulator) and ``mcp_run`` (experiment, index, params, run
        keys, device, ref, config content id), the sequence that was played, and the fit summary —
        and the reply carries ``datachest: {saved: true, chest, entry}``. Reload it with
        ``FileDataChest(chest).load(entry)["result"]`` (a ``MeasurementResult``) and refit it with
        the experiment's ``analyze``. With a task record open, a pointer file under its
        ``artifacts/`` is recorded as an ``artifact`` event of kind ``datachest``
        (``datachest.task_artifact`` names it). With no chest, or when the save fails, the run still
        returns its fit and ``datachest`` is ``{saved: false, reason}`` — never silently dropped.
        """
        device = self._require_device()
        registry = experiments_for(device)
        cls = registry.get(name)
        if cls is None:
            raise UnknownExperimentError(
                f"unknown experiment {name!r} for pack "
                f"{device.pack.__name__.rsplit('.', 1)[-1]!r}; available: "
                f"{', '.join(sorted(registry)) or '(none — this pack registers no experiments)'}"
            )
        given = dict(params or {})
        run_kw = {key: given.pop(key) for key in list(given) if key in RUN_PARAMS}
        experiment = cls(device, **coerce_params(cls, given))
        # record= reaches Experiment.go's vetted emitter: progress for acquisition + fit, the
        # `fit` / `fit-recovery` gates, computed in shipped code — never re-derived here.
        outcome = experiment.go(self.soc, progress=False, record=self.record, **run_kw)
        saved = self._save_run(experiment, name, given, run_kw, outcome, self._next_index())
        return self._remember(name, given, run_kw, outcome, datachest=saved)

    def _next_index(self) -> int:
        # ``index`` is what record_calibration(outcome=…) names, so it counts every run of the
        # session and never repeats — even after the memory cap drops the oldest outcomes
        last = self.outcomes[-1].get("index") if self.outcomes else None
        return last + 1 if isinstance(last, int) else len(self.outcomes)

    def _save_run(self, experiment, name: str, params: dict, run_kw: dict, outcome,
                  index: int) -> dict:
        """Save a run to the session's chest; the ``datachest`` block the reply carries (#200).

        Never raises: the data has already been acquired, so a failed save is reported as
        ``{saved: false, reason}`` beside the fit rather than costing the run.
        """
        if self.chest is None:
            return _unsaved(
                "run data not saved: this session has no chest — open it with "
                "load_device(..., chest=<dir>) or with a store= (the chest then lives beside it)"
            )
        try:
            seq = experiment.sequence()[0]
        except Exception:    # the data still saves; only the played-sequence provenance is lost
            seq = None
        origin = write_origin(self.soc_kind, self.truth)
        result = outcome.result
        meta = {
            **dict(result.meta or {}),
            **origin,
            "mcp_run": {
                "experiment": name,
                "index": index,
                "params": _jsonable(params),
                "run": _jsonable(run_kw),
                "device": self.device_path,
                "ref": self.ref if self.store is not None else "",
                "content_id": self.device.to_config().content_id() if self.device else "",
            },
        }
        chest = str(self.chest)
        try:
            # A fresh salt per run: the chest addresses entries by their arrays, so without it two
            # runs returning identical data (the mock's zero IQ) would share the first run's entry,
            # and the second run's citation would resolve to the first run.
            entry = FileDataChest(self.chest).save(
                replace(result, meta=meta), seq=seq, experiment=name,
                fit=_fit_summary(outcome.fit),
                task_id=self.record.id if self.record is not None else "",
                salt=f"mcp-run:{uuid.uuid4().hex}",
            )
        except Exception as err:
            return _unsaved(f"run data not saved: the save to {chest} failed — "
                            f"{type(err).__name__}: {err}")
        block: dict[str, Any] = {"saved": True, "chest": chest, "entry": entry}
        if self.record is not None:
            try:
                block["task_artifact"] = self._cite_in_task(chest, entry, name, index)
            except Exception as err:   # the entry stands; only the task record's citation failed
                block["task_artifact_error"] = f"{type(err).__name__}: {err}"
        return block

    def _cite_in_task(self, chest: str, entry: str, name: str, index: int) -> str:
        """Write a pointer to a chest entry under the task's ``artifacts/`` and record it.

        The chest lives outside the task directory, and an ``artifact`` path must resolve inside
        it, so the record cites a small JSON pointer file (``kind: "datachest"``).
        """
        record = self.record
        assert record is not None
        stem = f"datachest-{entry}"
        n = len(list(record.artifacts.glob(f"{stem}-*.json")))
        path = record.artifacts / f"{stem}-{n:03d}.json"
        path.write_text(json.dumps(
            {"chest": chest, "entry": entry, "experiment": name, "index": index}, indent=2,
        ) + "\n")
        return str(record.artifact(path, kind="datachest")["path"])

    def _remember(self, name: str, params: dict, run_kw: dict, outcome,
                  datachest: dict | None = None) -> dict:
        """Summarize an outcome, keep it as session memory, and return it."""
        result = outcome.result
        payload = {
            "ok": True,
            "index": self._next_index(),
            "experiment": name,
            "params": _jsonable(params),
            "run": _jsonable(run_kw),
            "fit": _fit_summary(outcome.fit),
            "data": _data_summary(result),
            "datachest": datachest if datachest is not None else _unsaved("run data not saved"),
            "environment": _jsonable(result.meta.get("environment", {})),
            "ran_at": _now(),
        }
        self.outcomes.append(payload)
        del self.outcomes[:-MAX_OUTCOMES]
        return payload

    # -- tools: the store as agent memory ---------------------------------------------

    def calibration_history(self) -> dict:
        """The tracked ref's version history: content ids, parents, timestamps, sources, diffs.

        Each entry also carries the line's ``extra``: for a write from a session, ``soc_kind``,
        ``simulated`` and (simulator) ``truth_hash``, plus ``fit`` for a write labelled with an
        outcome. A line written before these fields existed reads with whatever ``extra`` it has
        (``{}`` when it has none).
        """
        self._require_device()      # a store is always opened *with* a device: report the root cause
        store = self._require_store()
        return {
            "ok": True,
            "root": str(store.root),
            "ref": self.ref,
            "current": store.resolve_ref(self.ref) or "",
            "history": provenance_entries(store, self.ref),
        }

    def record_calibration(self, updates: dict, outcome: int | None = None) -> dict:
        """Write calibration values back, advance the ref, **and reload the session device**.

        ``updates`` maps dotted calibration paths to new values — the grammar
        ``device_info()["calibration_paths"]`` publishes and :func:`strumento.core.config.apply_updates`
        consumes. The write is explicit by design: strumento never guesses which field a fit result
        belongs to, so the mapping is the caller's statement and the diff is auditable.

        The **reload** is the part that closes a loop. A new immutable version is snapshotted, the
        ref moves to it, and the session's device is recomposed from that config — so the next
        ``run_experiment`` plays pulses built from the new belief. Skip the reload and a session
        would keep measuring with the value it just replaced.

        The candidate config is **validated before anything is written** (#191): a typo'd key, a
        value of the wrong type, a NaN or a negative time comes back ``ok: false`` with the
        offending field named, and the store, the ref and the session device are left as they were.

        **Gain ceilings are not agent-writable (#190).** An update that touches a path the pack
        declares in ``CALIBRATION_CEILINGS`` (the per-element gain limits) is refused as a whole,
        before any store call: ``ok: false`` names the path, and nothing is written. A human sets a
        ceiling in the device YAML, or through :func:`strumento.core.store.record_calibration`.

        **The label is the caller's statement (#194).** With no ``outcome`` the write is recorded
        as ``source: "manual"`` with no fit attached. ``outcome=<index>`` — the ``index`` a
        ``run_experiment`` returned in this session — records it as ``"<experiment> fit"`` with
        that outcome's ``derived`` values in ``extra.fit`` and its saved data cited in
        ``extra.data`` — ``{saved: true, chest, entry}``, or ``{saved: false, reason}`` for a run
        that was not saved (#200); the task record's ``calibration`` event carries the same
        ``data``. An index this session does not remember (never returned, dropped past the last
        32 runs, or not an integer), or an outcome with no fit, is refused with nothing written.
        Either way the line's ``extra`` carries ``soc_kind``, ``simulated`` and, for the
        simulator, ``truth_hash``.

        **Simulated sessions keep out of real stores (#194).** A simulator or mock session refuses
        to write into a store that does not carry the simulation mark, unless it was opened with
        ``load_device(..., allow_sim_writes=True)``: ``ok: false`` names the store and the two ways
        out, and nothing is written.

        Every refusal — the update's shape, ``outcome``, a ceiling, an unmarked store, then
        validation — happens before any store write.
        """
        device = self._require_device()
        store = self._require_store()
        if not isinstance(updates, dict) or not updates:
            raise SessionError(
                "record_calibration takes a non-empty {dotted.calibration.path: value} mapping; "
                "device_info()['calibration_paths'] lists the paths this device has"
            )
        source, fit_extra = self._label(outcome)
        refused = ceiling_writes(device, updates)
        if refused:
            raise CeilingWriteError(
                f"refused: {', '.join(refused)} is a gain ceiling (a per-element gain limit), "
                "which this session does not write; the whole update was refused and nothing "
                "was written. A human sets ceilings in the device YAML or through "
                "strumento.core.store.record_calibration"
            )
        if (self.soc_kind in SIMULATED_SOC_KINDS and not self.allow_sim_writes
                and not store_is_marked(store.root)):
            raise _unmarked_refusal(store.root, self.soc_kind, "write calibration")
        origin = write_origin(self.soc_kind, self.truth)
        content_id, config = store_record_calibration(
            store,
            self.ref,
            device.to_config(),
            updates,
            source=source,
            timestamp=_now(),
            extra={**origin, **fit_extra},
        )
        self.device = Device.from_config(config)          # the reload the closed loop depends on
        log = provenance_entries(store, self.ref)
        if self.record is not None:
            # the designed cross-link (TaskRecord.calibration): a fact in the durable log noting
            # the store pointer moved — the store stays the durable belief, this makes the task
            # history cite it.
            self.record.calibration(content_id, **origin,
                                    **({"data": fit_extra["data"]} if "data" in fit_extra else {}))
        return {
            "ok": True,
            "ref": self.ref,
            "content_id": content_id,
            "parent_id": log[-1]["parent_id"] if log else "",
            "updates": _jsonable(log[-1]["updates"] if log else {}),
            "device_reloaded": True,
            "calibration_paths": _jsonable(flatten_calibration(self.device.calib.model_dump())),
        }

    def _label(self, outcome: Any) -> tuple[str, dict]:
        """``outcome`` → ``(source, extra)``: ``("manual", {})``, or that outcome's fit label."""
        if outcome is None:
            return "manual", {}
        known = [o.get("index") for o in self.outcomes]
        if isinstance(outcome, bool) or not isinstance(outcome, int) or outcome not in known:
            raise UnknownOutcomeError(
                f"outcome={outcome!r} names no outcome this session remembers; pass the index a "
                f"run_experiment returned (remembered: {known or 'none — run an experiment first'}), "
                "or omit outcome for a manual write; nothing was written"
            )
        chosen = self.outcomes[known.index(outcome)]
        if not chosen.get("fit"):
            raise UnknownOutcomeError(
                f"outcome {outcome} ({chosen.get('experiment')}) has no fit to label the write "
                "with; omit outcome for a manual write; nothing was written"
            )
        return f"{chosen['experiment']} fit", {
            "fit": chosen["fit"]["derived"],
            **({"data": _data_citation(chosen["datachest"])} if chosen.get("datachest") else {}),
        }

    # -- tools: mirrors of the stateless verbs ----------------------------------------

    def schema(self, what: str, pack: str | None = None) -> dict:
        """JSON Schema for one typed contract (D35) — the same function ``strumento schema`` calls.

        ``what`` is ``device``, ``fit``, ``result`` or ``operator-model``, or ``calibration`` for a
        pack's calibration model, whose field descriptions state each value's unit (MHz, µs, DAC
        integer code, ...). ``calibration`` takes ``pack`` (e.g. ``pack="cqed"``); if it is omitted
        and a device is loaded, the loaded device's pack is used. With no pack and no device, or an
        unknown pack, the error lists the installed packs. ``pack`` is refused for the other
        contracts.
        """
        if what in schemas.PACK_CONTRACTS and pack is None and self.device is not None:
            pack = self.device.pack.__name__.rsplit(".", 1)[-1]  # the loaded device's pack
        schema = schemas.contract_schema(what, pack)
        if what in schemas.PACK_CONTRACTS:
            return {"ok": True, "contract": what, "pack": pack, "schema": schema}
        return {"ok": True, "contract": what, "schema": schema}

    def validate(self, yaml_: str) -> dict:
        """Validate a device instance: a **path** to an instance YAML, or inline YAML text.

        A path is loaded exactly as ``strumento validate`` loads it (so relative
        ``soccfg_snapshot:`` resolves). Inline text is checked against the ``device`` contract and
        then composed, which is how a host with no filesystem of its own validates a candidate
        instance before writing it anywhere.
        """
        as_path = Path(yaml_)
        is_path = len(str(yaml_)) < 4096 and "\n" not in str(yaml_) and as_path.exists()
        if is_path:
            device = Device.load(as_path)
        else:
            data = yaml.safe_load(yaml_)
            if not isinstance(data, dict):
                raise SessionError("validate takes an instance YAML path or an inline YAML mapping")
            schemas.validate("device", data)          # typed, field-located contract error
            device = Device.from_config(_config_from_mapping(data))
        return {
            "ok": True,
            "source": "path" if is_path else "inline",
            "pack": device.pack.__name__.rsplit(".", 1)[-1],
            "elements": sorted(device.elements),
            "namespaces": device_summary(device)["namespaces"],
        }

    # -- internals ---------------------------------------------------------------------

    def _require_device(self):
        if self.device is None or self.soc is None:
            raise NoDeviceError(
                "no device loaded in this session: call load_device(path, soc='simulator'|'mock') "
                "first (devices/multimode_demo/device.yaml is the cqed exemplar)"
            )
        return self.device

    def _require_store(self) -> FileConfigStore:
        if self.store is None:
            raise NoStoreError(
                "no store opened for this session: re-open it with "
                "load_device(path, store='<store-root>') to give this device a calibration history"
            )
        return self.store

    def _chest_view(self) -> dict:
        """Where this session saves its runs' data — ``opened: false`` (and why) when nowhere."""
        if self.chest is None:
            return {"opened": False,
                    "reason": "no chest= and no store= given: run data is not saved"}
        return {"opened": True, "path": str(self.chest), "source": self.chest_source}

    def _store_view(self) -> dict:
        if self.store is None:
            return {"opened": False}
        return {
            "opened": True,
            "root": str(self.store.root),
            "ref": self.ref,
            "content_id": self.store.resolve_ref(self.ref) or "",
            "simulation_mark": store_is_marked(self.store.root),
            "allow_sim_writes": self.allow_sim_writes,
        }


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _soc_view(kind: str, truth: dict) -> dict:
    """The session's soc as a payload: its kind, whether it is simulated, and the truth it holds."""
    view: dict[str, Any] = {"kind": kind, "simulated_truth_keys": sorted(truth),
                            "simulated": kind in SIMULATED_SOC_KINDS}
    if kind == "simulator":
        view["truth_hash"] = truth_hash(truth)
    return view


def _task_view(record) -> dict:
    """The session's task record as a payload — ``opened: false`` when there is none."""
    if record is None:
        return {"opened": False}
    return {"opened": True, "id": record.id, "dir": str(record.dir)}


def _unsaved(reason: str) -> dict:
    """A run's ``datachest`` block when its data was not saved: says so, and why (#200)."""
    return {"saved": False, "reason": reason}


def _data_citation(block: dict) -> dict:
    """The part of a run's ``datachest`` block a provenance line cites: ``{saved, chest, entry}``,
    or ``{saved: false, reason}``."""
    keys = ("saved", "chest", "entry") if block.get("saved") else ("saved", "reason")
    return {key: block[key] for key in keys if key in block}


def _resolve_truth(sim: dict | str | None) -> dict:
    """``sim`` → a ground-truth dict: a mapping passes through, a path is read (JSON or YAML)."""
    if sim is None:
        return {}
    if isinstance(sim, dict):
        return dict(sim)
    if isinstance(sim, (str, Path)):
        text = Path(sim).read_text()
        loaded = yaml.safe_load(text)       # a superset of JSON, so one loader covers both
        if not isinstance(loaded, dict):
            raise SessionError(f"sim file {sim!r} must hold a mapping of truth values")
        return dict(loaded)
    raise SessionError(f"sim must be a mapping of truth values or a path to one, got {type(sim).__name__}")


def _config_from_mapping(data: dict):
    """A parsed inline instance mapping → :class:`~strumento.core.config.DeviceConfig`.

    Inline text cannot carry a *relative* ``soccfg_snapshot`` path (there is no file to be relative
    to), so a string there is dropped with the snapshot left unset — validation is about the model,
    and a session that needs to assemble programs loads from a path.
    """
    from strumento.core.config import DeviceConfig

    snapshot = data.get("soccfg_snapshot")
    return DeviceConfig(
        pack=data["pack"],
        topology=data.get("topology", []),
        wiring=data["wiring"],
        calibration=data["calibration"],
        couplings=data.get("couplings", []),
        aliases=data.get("aliases", {}),
        soccfg_snapshot=snapshot if isinstance(snapshot, dict) else None,
    )


def _fit_summary(fit) -> dict | None:
    """A :class:`~strumento.core.fitters.FitResult` as JSON, or ``None`` for an unfitted run.

    ``params`` and ``stderr`` are numbers: each is a float, or ``None`` for NaN/inf. ``derived``
    also carries the JSON values an analyzer reports (D38) — strings, booleans, lists, objects —
    and those arrive as given, with NaN/inf turned into ``None`` at any depth (see
    :func:`_json_value`).
    """
    if fit is None:
        return None
    return {
        "success": bool(fit.success),
        "params": {key: _finite(value) for key, value in fit.params.items()},
        "stderr": {key: _finite(value) for key, value in fit.stderr.items()},
        "derived": {str(key): _json_value(value) for key, value in fit.derived.items()},
    }


def _json_value(value):
    """One ``derived`` value as strict JSON (no NaN/Infinity tokens), recursing into containers.

    - a bool (Python or numpy) stays a bool — it is checked before the numbers, since ``bool`` is
      an ``int`` subclass and ``True`` must not become ``1.0``;
    - ``None`` and strings pass through;
    - a number (int, float, numpy scalar) goes through :func:`_finite`: a float, or ``None`` for
      NaN/inf — so a number has the same JSON form here as in ``params``/``stderr``;
    - lists and tuples become lists, dicts become dicts with string keys, numpy arrays become
      nested lists — each recursed, so a NaN inside a list or object is ``None`` too;
    - anything else (not a JSON value) is ``None``, the rule the summary has always had.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _finite(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return None


def _data_summary(result) -> dict:
    """Shapes and axis ends — enough to see *that* an acquisition happened and how big it was.

    The arrays themselves are not inlined: a swept acquisition with per-shot data is megabytes, and
    a protocol payload is the wrong place for it. The arrays are kept in the session's data chest
    instead (the reply's ``datachest`` names the entry, #200); ``MeasurementResult.to_wire()``
    (published as ``strumento schema result``) is the JSON form of a reloaded result.
    """
    xpts = np.asarray(result.xpts) if result.xpts is not None else None
    return {
        "points": int(xpts.size) if xpts is not None else 1,
        "xpts_first": _finite(xpts[0]) if xpts is not None and xpts.size else None,
        "xpts_last": _finite(xpts[-1]) if xpts is not None and xpts.size else None,
        "avgi_shape": list(np.shape(result.avgi)),
        "avgq_shape": list(np.shape(result.avgq)),
        "shots_shape": list(np.shape(result.shots)) if result.shots is not None else None,
        "herald_lanes": [str(lane) for lane in result.heralds],
    }


def _outcome_brief(outcome: dict) -> dict:
    return {
        "index": outcome["index"],
        "experiment": outcome["experiment"],
        "ran_at": outcome["ran_at"],
        "derived": (outcome["fit"] or {}).get("derived"),
    }


def _finite(value) -> float | None:
    """A float, or ``None`` for NaN/inf — an lmfit stderr is often NaN and NaN is not JSON."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _jsonable(value):
    """Reuse the result layer's JSON coercion so one policy covers every payload here."""
    from strumento.core.result import _jsonable as coerce

    return coerce(value)


# --------------------------------------------------------------------------------------
# the server
# --------------------------------------------------------------------------------------


def build_server(session: LabSession | None = None):
    """A ``FastMCP`` server whose eight tools are :class:`LabSession` methods, errors-as-data.

    The MCP SDK is imported here rather than at module scope so ``strumento.mcp`` (and
    :class:`LabSession`) work without the ``[mcp]`` extra installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as err:  # pragma: no cover - exercised by installing without the extra
        raise ImportError(
            "strumento-mcp needs the MCP SDK: pip install 'strumento[mcp]'"
        ) from err

    lab = session if session is not None else LabSession()
    server = FastMCP("strumento", instructions=_INSTRUCTIONS)

    def as_data(fn):
        """Errors-as-data: a typed session error (or anything else) becomes a payload, never a raise."""

        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as err:
                return {"ok": False, "error": f"{type(err).__name__}: {err}"}

        return wrapped

    @server.tool(name="load_device", description=LabSession.load_device.__doc__)
    def load_device(path: str, soc: str = "simulator", store: str | None = None,
                    ref: str = "main", sim: dict | str | None = None,
                    task: str | None = None, allow_sim_writes: bool = False,
                    chest: str | None = None) -> dict:
        return as_data(lab.load_device)(path, soc=soc, store=store, ref=ref, sim=sim, task=task,
                                        allow_sim_writes=allow_sim_writes, chest=chest)

    @server.tool(name="device_info", description=LabSession.device_info.__doc__)
    def device_info() -> dict:
        return as_data(lab.device_info)()

    @server.tool(name="list_experiments", description=LabSession.list_experiments.__doc__)
    def list_experiments() -> dict:
        return as_data(lab.list_experiments)()

    @server.tool(name="run_experiment", description=LabSession.run_experiment.__doc__)
    def run_experiment(name: str, params: dict | None = None) -> dict:
        return as_data(lab.run_experiment)(name, params)

    @server.tool(name="calibration_history", description=LabSession.calibration_history.__doc__)
    def calibration_history() -> dict:
        return as_data(lab.calibration_history)()

    @server.tool(name="record_calibration", description=LabSession.record_calibration.__doc__)
    def record_calibration(updates: dict, outcome: int | None = None) -> dict:
        return as_data(lab.record_calibration)(updates, outcome)

    @server.tool(name="schema", description=LabSession.schema.__doc__)
    def schema(what: str, pack: str | None = None) -> dict:
        return as_data(lab.schema)(what, pack)

    @server.tool(name="validate", description=LabSession.validate.__doc__)
    def validate(yaml: str) -> dict:
        return as_data(lab.validate)(yaml)

    return server


_INSTRUCTIONS = """\
strumento — a QICK tProc-v2 experiment framework, as a stateful lab session.

Open a session with load_device (soc='simulator' runs real tProc-v2 assembly against simulated
acquisition; soc='mock' returns zero-valued data). list_experiments names what the loaded pack can
run; run_experiment(name, params) runs one and returns the typed fit. With a store open,
record_calibration(updates, outcome=<index>) writes a fit back as an immutable, content-addressed
version labelled with that run's fit (omit outcome for a manual write), advances the ref, and
reloads the session's device from it — so the next run uses the new belief. calibration_history
shows that provenance chain; every line records the soc kind and that the data was simulated. A
simulated session writes only to a store it created (marked for simulation) unless opened with
allow_sim_writes=True. Each run's data (arrays, metadata, the played sequence, the fit) is saved as
one entry in a data chest — load_device(chest=<dir>), or by default the directory beside the store
— and run_experiment's datachest field names the entry (or says the data was not saved, and why);
a write labelled with outcome=<index> cites that entry in its provenance. With
task=<root-or-dir>, the session also keeps a
durable task record: every run's progress and gate verdicts land in its append-only log, which is
what a supervisor reads.

This server is mechanism only: it will not tell you which experiment to run next. Deciding that is
the calibration-strategy layer's job, above this one.
"""


def main(argv: list[str] | None = None) -> int:
    """The ``strumento-mcp`` console script: serve one session over stdio."""
    with LabSession() as lab:   # the session's task record is closed however serving ends
        build_server(lab).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
