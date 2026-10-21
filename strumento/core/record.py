"""The task-record contract — the durable on-disk truth for a long-running experiment task.

An experiment task (a T1 monitor, a sweep campaign, a 50-iteration calibration loop) outlives the
thing that started it: a chat turn, an editor session, the supervisor process itself. So the truth
cannot live in a supervisor's memory — it lives in a **task directory**, and this module is the
writer and reader of that directory::

    experiments/2026-07-23-qilc-cz-cal-a3f2/
      task.toml          manifest — written once at creation; pid / host / config_content_id
                         are the only fields ever updated in place; kind = experiment | bringup
      progress.jsonl     append-only event stream (progress · artifact · heartbeat ·
                         calibration · gate · claim · step_open · step_close · park ·
                         approve · resume), one JSON object per line, flushed per line
      artifacts/         whatever the script saves (plots, fit CSVs, pulses)
      result.toml        terminal state — written exactly once, atomically (tmp+rename)

**This module is pure stdlib on purpose.** Its body imports nothing outside the standard library
— no ``qick``, no ``numpy``, no ``pydantic``, nothing from the rest of ``strumento``. The record
format is a *public contract*: any QICK user running long experiments from a plain script should be
able to emit it, and any tool in any language should be able to read it. A copy of this single file
is a complete implementation (see ``tests/core/test_record.py``, which loads it that way and proves
it). Installing ``strumento`` of course brings the framework's own dependencies along; that is a
property of the package, not of this contract.

**Crash safety.** ``progress.jsonl`` is opened append-only and flushed after every line, so a
``kill -9`` loses at most the line in flight and a reader skips it (:func:`read_events`).
``task.toml`` and ``result.toml`` are written tmp+rename+fsync, so no reader ever sees a partial
manifest or a partial terminal marker. ``result.toml``'s *existence* is the only durable terminal
marker there is.

**One writer per field.** ``pid`` is the field two processes could plausibly both want. Supervised
runs (a supervisor spawned us and set ``STRUMENTO_TASK_DIR``) never touch it — the supervisor knows
the pid before the script boots and writes it synchronously at spawn. Standalone runs
(:meth:`TaskRecord.create` with no supervisor) stamp their own. No path has two writers.

**No TOML nulls, anywhere.** Absent strings are ``""``, absent integers ``0``. Callers of this
format include a python3.9 + ``tomli`` reader, and a null would be a parse error there; keeping the
convention project-wide means every reader is trivially compatible.

**Gate events carry verdicts, they do not compute them** (see :meth:`TaskRecord.gate`). The trust
boundary that makes a gate verdict worth anything is described there and in
:meth:`strumento.core.experiment.Experiment.run`.

Docs: ``docs/task-record.md`` (the contract), ``docs/cli.md`` (the ``strumento task`` verbs).
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

try:  # a reader is needed for status/tail; a *writer* is hand-rolled below (stdlib has none)
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - python < 3.11 vendoring path
    try:
        import tomli as _tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        _tomllib = None  # type: ignore[assignment]

#: format version stamped into ``task.toml`` and ``result.toml``.
SCHEMA = 1

#: the env var a supervisor sets on a spawned task; :meth:`TaskRecord.from_env` reads it.
TASK_DIR_ENV = "STRUMENTO_TASK_DIR"

MANIFEST = "task.toml"
EVENTS = "progress.jsonl"
RESULT = "result.toml"
ARTIFACTS = "artifacts"

#: default heartbeat cadence; a task whose iterations are slower declares its own honest value.
DEFAULT_HEARTBEAT_INTERVAL_S = 15

#: liveness is judged against ``heartbeat_interval_s`` times this factor (never a global constant).
STALE_FACTOR = 3

#: the fields a manifest may ever have rewritten after creation. Everything else is immutable.
MUTABLE_FIELDS: tuple[str, ...] = ("pid", "host", "config_content_id")

#: terminal ``result.toml`` states, and the derived states a reader can see.
TERMINAL_STATES: tuple[str, ...] = ("done", "failed", "cancelled")
#: Deliberate public API: the full vocabulary of :attr:`TaskStatus.state` (the rules in
#: :func:`derive_state`, ``docs/task-record.md``), exported for readers of this contract —
#: nothing in this module consumes it, since the states are produced, not validated, here.
DERIVED_STATES: tuple[str, ...] = ("queued", "running", "stale") + TERMINAL_STATES

#: ``result.toml``'s ``error_kind`` enum — the distinction is *who fixes it*: a script error is
#: scientific, a transport error is operational.
ERROR_KINDS: tuple[str, ...] = ("", "script-error", "transport-error")

TARGETS: tuple[str, ...] = ("sim", "hw")
LANGUAGES: tuple[str, ...] = ("python", "julia")

#: the manifest's reserved registry axis. ``experiment`` is the long-running-experiment record
#: this module was built around; ``bringup`` is a bring-up **campaign** ledger — a task record
#: whose event stream carries the campaign events below. Writers mint one of these two (the
#: axis is how a supervisor indexes "other long-running work" in the same registry); readers
#: stay lenient — an unknown ``kind`` is opaque to a reader, skipped or deferred, never fatal.
MANIFEST_KINDS: tuple[str, ...] = ("experiment", "bringup")

#: Deliberate public API: the documented event vocabulary — the twelve ``ev`` names the
#: ``progress.jsonl`` section of ``docs/task-record.md`` lists — exported for readers of this
#: contract. Nothing in this module consumes it: each emitter names its own event, and writes
#: are not checked against this tuple. Readers must **ignore** unknown ``ev`` values (forward
#: compatibility is part of the contract, not a nicety). The seven campaign events (claim ·
#: step_open · step_close · park · approve · resume · poison) are generic substrate mechanism:
#: they carry name + payload and validate structurally at append; their bring-up *meaning*
#: belongs to the orchestration layer that reads them, never to this module.
EVENT_KINDS: tuple[str, ...] = (
    "progress", "artifact", "heartbeat", "calibration", "gate",
    "claim", "step_open", "step_close", "park", "approve", "resume", "poison",
)


class RecordError(RuntimeError):
    """A task record could not be written or read as the contract requires.

    Typed so a CLI or a supervisor can surface "that is not a task directory" / "result already
    written" as data rather than a traceback.
    """


# --------------------------------------------------------------------------------------
# small stdlib utilities (time, ids, TOML)
# --------------------------------------------------------------------------------------


def now_iso() -> str:
    """UTC ISO-8601 with millisecond resolution and a ``Z`` suffix — one format everywhere.

    Milliseconds (rather than whole seconds) so that events emitted
    inside one loop iteration still order correctly; ``Z`` because every consumer's date parser
    accepts it.
    """
    stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


def parse_iso(value: Any) -> datetime | None:
    """Parse a contract timestamp back to an aware ``datetime``; ``None`` when unparseable."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def slugify(title: str, *, limit: int = 40) -> str:
    """A filesystem- and URL-safe slug from a task title (``"QILC CZ cal"`` → ``qilc-cz-cal``)."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(title).lower()).strip("-")
    if len(slug) > limit:
        slug = slug[:limit].rsplit("-", 1)[0] or slug[:limit]
    return slug.strip("-") or "task"


def new_task_id(title: str = "", *, now: datetime | None = None, suffix: str | None = None) -> str:
    """``YYYY-MM-DD-<slug>-<4 hex>`` — sortable by day, readable, and collision-safe enough."""
    day = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return f"{day}-{slugify(title)}-{suffix or uuid.uuid4().hex[:4]}"


#: how many fresh suffixes :func:`claim_task_dir` mints before giving up on a root.
MINT_ATTEMPTS = 32


def claim_task_dir(root: str | Path, title: str = "", *, attempts: int = MINT_ATTEMPTS) -> Path:
    """Mint a task id under ``root`` and claim ``<root>/<id>/`` — atomically, collision-safely.

    The claim *is* the ``mkdir`` (no ``exist_ok``): the filesystem lets exactly one caller create
    a directory, so two processes can never both own one id. A directory that already exists —
    a finished record, or one another writer claimed a moment ago and has not yet given a
    manifest — is someone else's; a fresh suffix is minted and the claim retried, up to
    ``attempts`` times. The id format is unchanged; only the suffix is re-drawn.
    """
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    for _ in range(max(1, int(attempts))):
        task_dir = base / new_task_id(title)
        try:
            task_dir.mkdir()
        except FileExistsError:
            continue
        return task_dir
    raise RecordError(
        f"could not mint a fresh task id under {base}: {attempts} attempts all collided"
    )


def _release_claim(task_dir: Path) -> None:
    """Undo a :func:`claim_task_dir` whose manifest never landed (best effort; never raises)."""
    if (task_dir / MANIFEST).exists():
        return
    for path in (task_dir / ARTIFACTS, task_dir):
        try:
            path.rmdir()
        except OSError:
            pass


def _toml_scalar(value: Any) -> str:
    """Render one TOML value. ``None`` is a hard error — the format has no nulls (see module doc)."""
    if value is None:
        raise RecordError(
            "task records never write a TOML null: absent strings are \"\", absent integers 0"
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise RecordError(f"cannot write {type(value).__name__} into a task record's TOML")


def _toml_table(pairs: Iterable[tuple[str, Any]]) -> str:
    return "".join(f"{key} = {_toml_scalar(value)}\n" for key, value in pairs)


def parse_toml(text: str) -> dict:
    """Parse a manifest/result document, raising :class:`RecordError` on anything unparseable."""
    if _tomllib is None:  # pragma: no cover - only on <3.11 without tomli
        raise RecordError(
            "reading a task record needs a TOML parser: python >= 3.11 (tomllib) or `pip install tomli`"
        )
    try:
        return _tomllib.loads(text)
    except Exception as err:  # tomllib raises TOMLDecodeError; keep the surface typed
        raise RecordError(f"malformed TOML: {type(err).__name__}: {err}") from err


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via tmp+fsync+rename — a reader never sees a partial document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


def pid_is_alive(pid: int) -> bool:
    """Signal-0 liveness probe. A pid owned by another user counts as alive (it exists)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthoredBy:
    """Agent provenance: which capability profile / model authored the task script.

    Empty strings for a human or standalone run — and that emptiness is load-bearing downstream:
    a supervisor forwards telemetry rows only when ``model`` is non-empty, so human outcomes never
    become training signal for automated routing.
    """

    profile: str = ""
    model: str = ""
    variant: str = ""
    session_id: str = ""

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "model": self.model,
            "variant": self.variant,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Any, *, strict: bool = False) -> AuthoredBy:
        """Parse a provenance table. ``strict`` rejects unknown fields — the *writer's* setting.

        Reading is lenient (a newer writer may add fields; readers ignore what they do not know);
        authoring is strict (a typo in a supervisor's ``--authored-by`` payload would silently
        record no provenance at all, which is worse than a loud error).
        """
        data = data or {}
        if not isinstance(data, dict):
            raise RecordError(f"authored_by must be a table, got {type(data).__name__}")
        unknown = sorted(set(data) - {"profile", "model", "variant", "session_id"})
        if unknown and strict:
            raise RecordError(
                f"unknown authored_by field(s) {unknown}: valid fields are profile, model, "
                f"variant, session_id"
            )
        return cls(
            profile=str(data.get("profile", "")),
            model=str(data.get("model", "")),
            variant=str(data.get("variant", "")),
            session_id=str(data.get("session_id", "")),
        )


@dataclass(frozen=True)
class Manifest:
    """``task.toml`` — written once at creation; only :data:`MUTABLE_FIELDS` change afterwards.

    ``kind`` is the reserved registry axis (:data:`MANIFEST_KINDS`): ``"experiment"`` — the
    long-running-experiment record this module was built around — or ``"bringup"``, a bring-up
    campaign's ledger, so a supervisor can index other long-running work in the same registry.
    ``task_type`` / ``target`` / ``gates`` are the telemetry keys a supervisor needs *before*
    the script runs, which is why they are creation-time immutable and have to travel the
    whole submit chain.
    """

    id: str
    title: str = ""
    kind: str = "experiment"
    task_type: str = "experiment"
    target: str = "sim"
    gates: tuple[str, ...] = ()
    script: str = ""
    workdir: str = ""
    language: str = "python"
    device: str = ""
    created: str = ""
    pid: int = 0
    host: str = ""
    heartbeat_interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S
    config_content_id: str = ""
    authored_by: AuthoredBy = field(default_factory=AuthoredBy)
    schema: int = SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "gates", tuple(str(g) for g in self.gates))
        # clamped, not rejected: 0 or negative would derive `stale` instantly, which is a worse
        # failure than a quietly sane cadence (the interval is immutable after creation).
        object.__setattr__(self, "heartbeat_interval_s", max(1, int(self.heartbeat_interval_s)))

    def to_toml(self) -> str:
        head = [
            ("schema", int(self.schema)),
            ("id", self.id),
            ("title", self.title),
            ("kind", self.kind),
            ("task_type", self.task_type),
            ("target", self.target),
            ("gates", list(self.gates)),
            ("script", self.script),
            ("workdir", self.workdir),
            ("language", self.language),
            ("device", self.device),
            ("created", self.created),
            ("pid", int(self.pid)),
            ("host", self.host),
            ("heartbeat_interval_s", int(self.heartbeat_interval_s)),
            ("config_content_id", self.config_content_id),
        ]
        return _toml_table(head) + "\n[authored_by]\n" + _toml_table(self.authored_by.to_dict().items())

    def to_dict(self) -> dict:
        data = {key: value for key, value in self.__dict__.items()}
        data["gates"] = list(self.gates)
        data["authored_by"] = self.authored_by.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        """Build from parsed TOML. Deliberately **lenient**: a reader tolerates fields it does not
        know (a newer writer may have added some) and only insists on ``id``."""
        if "id" not in data:
            raise RecordError("task manifest has no id")
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", "")),
            kind=str(data.get("kind", "experiment")),
            task_type=str(data.get("task_type", "experiment")),
            target=str(data.get("target", "sim")),
            gates=tuple(str(g) for g in (data.get("gates") or ())),
            script=str(data.get("script", "")),
            workdir=str(data.get("workdir", "")),
            language=str(data.get("language", "python")),
            device=str(data.get("device", "")),
            created=str(data.get("created", "")),
            pid=int(data.get("pid", 0) or 0),
            host=str(data.get("host", "")),
            heartbeat_interval_s=int(
                data.get("heartbeat_interval_s", DEFAULT_HEARTBEAT_INTERVAL_S) or 0
            ),
            config_content_id=str(data.get("config_content_id", "")),
            authored_by=AuthoredBy.from_dict(data.get("authored_by")),
            schema=int(data.get("schema", SCHEMA) or SCHEMA),
        )

    @property
    def stale_after_s(self) -> float:
        """How long without any sign of life makes this task ``stale`` (state rule 4 in
        docs/task-record.md)."""
        return float(STALE_FACTOR * self.heartbeat_interval_s)


@dataclass(frozen=True)
class Result:
    """``result.toml`` — the terminal marker, written exactly once."""

    state: str
    ended: str = ""
    summary: str = ""
    error_kind: str = ""
    error: str = ""
    schema: int = SCHEMA

    def to_toml(self) -> str:
        return _toml_table(
            [
                ("schema", int(self.schema)),
                ("state", self.state),
                ("ended", self.ended),
                ("summary", self.summary),
                ("error_kind", self.error_kind),
                ("error", self.error),
            ]
        )

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> Result:
        return cls(
            state=str(data.get("state", "")),
            ended=str(data.get("ended", "")),
            summary=str(data.get("summary", "")),
            error_kind=str(data.get("error_kind", "")),
            error=str(data.get("error", "")),
            schema=int(data.get("schema", SCHEMA) or SCHEMA),
        )


# --------------------------------------------------------------------------------------
# reading a task directory
# --------------------------------------------------------------------------------------


def read_manifest(task_dir: str | Path) -> Manifest:
    """Parse ``<task_dir>/task.toml``; :class:`RecordError` when it is not a task directory."""
    path = Path(task_dir) / MANIFEST
    if not path.exists():
        raise RecordError(f"not a task directory (no {MANIFEST}): {task_dir}")
    return Manifest.from_dict(parse_toml(path.read_text()))


def read_result(task_dir: str | Path) -> Result | None:
    """Parse ``result.toml`` if present. A document that parses but carries a state outside the
    enum is returned as-is — the caller (state derivation) decides what an invalid marker means."""
    path = Path(task_dir) / RESULT
    if not path.exists():
        return None
    return Result.from_dict(parse_toml(path.read_text()))


def read_events(task_dir: str | Path, *, limit: int | None = None) -> list[dict]:
    """Parse ``progress.jsonl``, **skipping** lines that are not complete JSON objects.

    JSONL is per-line recoverable by construction: a task killed mid-write leaves one truncated
    line and every other event survives. ``limit`` returns the last N events (the ``task tail``
    verb).
    """
    path = Path(task_dir) / EVENTS
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue  # truncated / corrupt line — skip it, keep the rest
            if isinstance(event, dict):
                out.append(event)
    return out[-limit:] if limit else out


def artifact_path(task_dir: str | Path, rel: str) -> Path:
    """Resolve a recorded artifact path inside the task dir, rejecting ``..`` escapes.

    Readers use this too, not just the writer: an artifact path is data in a file anyone could have
    written, so the containment check belongs at *every* boundary that turns it into a filesystem
    path.
    """
    base = Path(task_dir).resolve()
    candidate = Path(rel)
    full = candidate if candidate.is_absolute() else base / candidate
    full = Path(os.path.normpath(str(full)))
    try:
        full.relative_to(base)
    except ValueError:
        raise RecordError(
            f"artifact path {rel!r} escapes the task directory: paths are task-dir-relative and "
            f"must resolve inside {base}"
        ) from None
    return full


def relative_artifact(task_dir: str | Path, path: str | Path) -> str:
    """The task-dir-relative form of an artifact path (POSIX separators), escape-checked."""
    base = Path(task_dir).resolve()
    full = artifact_path(base, str(path))
    return full.relative_to(base).as_posix()


@dataclass(frozen=True)
class TaskStatus:
    """A task's derived state plus every input the derivation used — the ``task status`` payload.

    Derived state is **never stored**: ``result.toml`` is the only durable terminal marker, and
    everything else is recomputed from (record, liveness) at read time. The inputs travel alongside
    the verdict so a supervisor or a human can see *why* a task reads ``stale``.
    """

    id: str
    dir: str
    state: str
    manifest: Manifest
    result: Result | None = None
    pid_alive: bool | None = None       # None = unknowable (foreign host)
    host_match: bool = True
    liveness_at: str = ""
    age_s: float = 0.0
    n_events: int = 0
    last_event: dict | None = None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dir": self.dir,
            "state": self.state,
            "manifest": self.manifest.to_dict(),
            "result": self.result.to_dict() if self.result else None,
            "liveness": {
                "pid_alive": self.pid_alive,
                "host_match": self.host_match,
                "liveness_at": self.liveness_at,
                "age_s": self.age_s,
                "stale_after_s": self.manifest.stale_after_s,
                "n_events": self.n_events,
                "last_event": self.last_event,
            },
            "error": self.error,
        }


class _StatusCommon(TypedDict):
    """The derivation inputs ``derive_state`` passes to every :class:`TaskStatus` it returns,
    typed so the ``**common`` unpacking is checked field by field."""

    id: str
    dir: str
    manifest: Manifest
    result: Result | None
    pid_alive: bool | None
    host_match: bool
    liveness_at: str
    age_s: float
    n_events: int
    last_event: dict | None


def derive_state(task_dir: str | Path, *, now: datetime | None = None, host: str | None = None) -> TaskStatus:
    """The **total** state function of (record, pid-liveness). First match wins, in this order:

    1. ``result.toml`` exists ⇒ its ``state``. Unparseable, or a state outside the enum ⇒ ``stale``
       with the reason surfaced as data (tmp+rename rules out partial writes, so this means
       corruption).
    2. ``pid == 0`` ⇒ ``queued`` (never started: a device FIFO is holding it, or it awaits
       re-enqueue after adoption).
    3. pid dead ⇒ ``stale``. **Skipped when the manifest's ``host`` is not this host** — a pid
       number from another machine says nothing here, and probing it could hit an unrelated
       process. A foreign record therefore ages into ``stale`` through rule 4 instead.
    4. liveness older than ``STALE_FACTOR × heartbeat_interval_s`` ⇒ ``stale``. Liveness :=
       max(``task.toml`` mtime, newest event of **any** kind). So a just-spawned task with no
       events yet is ``running`` until its interval elapses, and a task streaming ``progress``
       whose heartbeat thread died never goes stale. Hung-but-alive is exactly what this rule
       catches, which is why it *precedes* ``running``.
    5. otherwise ⇒ ``running``.
    """
    task_dir = Path(task_dir)
    manifest = read_manifest(task_dir)
    now = now or datetime.now(timezone.utc)
    local_host = socket.gethostname() if host is None else host

    error = ""
    result: Result | None = None
    try:
        result = read_result(task_dir)
    except RecordError as err:
        error = f"unreadable {RESULT}: {err}"

    events = read_events(task_dir)
    last_event = events[-1] if events else None
    liveness = _liveness(task_dir, events, manifest)
    age = max(0.0, (now - liveness).total_seconds()) if liveness else 0.0
    host_match = manifest.host == local_host
    alive = pid_is_alive(manifest.pid) if host_match else None

    common = _StatusCommon(
        id=manifest.id,
        dir=str(task_dir),
        manifest=manifest,
        result=result,
        pid_alive=alive,
        host_match=host_match,
        liveness_at=liveness.isoformat().replace("+00:00", "Z") if liveness else "",
        age_s=round(age, 3),
        n_events=len(events),
        last_event=last_event,
    )

    if error:                                                  # rule 1, corrupt marker
        return TaskStatus(state="stale", error=error, **common)
    if result is not None:                                     # rule 1
        if result.state in TERMINAL_STATES:
            return TaskStatus(state=result.state, **common)
        return TaskStatus(
            state="stale",
            error=f"{RESULT} state {result.state!r} is outside {TERMINAL_STATES}",
            **common,
        )
    if manifest.pid == 0:                                      # rule 2
        return TaskStatus(state="queued", **common)
    if alive is False:                                         # rule 3
        return TaskStatus(state="stale", **common)
    if age > manifest.stale_after_s:                           # rule 4
        return TaskStatus(state="stale", **common)
    return TaskStatus(state="running", **common)               # rule 5


def _liveness(task_dir: Path, events: Sequence[dict], manifest: Manifest) -> datetime | None:
    """max(pid-write time, newest event of any kind) — concretely ``task.toml`` mtime vs events.

    ``task.toml``'s mtime stands in for "when the pid was written"; ``set_config`` touching the file
    later is harmless, because that write also implies liveness.
    """
    stamps: list[datetime] = []
    manifest_path = task_dir / MANIFEST
    if manifest_path.exists():
        stamps.append(datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=timezone.utc))
    else:  # no manifest on disk to date — fall back to what it said its creation time was
        created = parse_iso(manifest.created)
        if created is not None:
            stamps.append(created)
    for event in events:
        stamp = parse_iso(event.get("t"))
        if stamp is not None:
            stamps.append(stamp)
    return max(stamps) if stamps else None


def list_tasks(root: str | Path) -> list[TaskStatus]:
    """Every task directory under ``root``, newest first. A missing root is an empty list.

    Directories without a ``task.toml`` are skipped (they are not task dirs); a directory whose
    manifest is corrupt is skipped rather than failing the whole listing — one bad record must not
    blind a supervisor to the rest.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    out: list[TaskStatus] = []
    for child in sorted(root_path.iterdir()):
        if not (child / MANIFEST).exists():
            continue
        try:
            out.append(derive_state(child))
        except RecordError:
            continue
    out.sort(key=lambda s: (s.manifest.created, s.id), reverse=True)
    return out


# --------------------------------------------------------------------------------------
# the writer
# --------------------------------------------------------------------------------------


class TaskRecord:
    """The writer side of the contract: ten lines of plumbing in a task script.

    ::

        from strumento.core.record import TaskRecord

        rec = TaskRecord.from_env()                  # supervised, or a fresh dir standalone
        rec.on_cancel(soc.close)                     # SIGTERM -> cleanup -> result(cancelled)
        for i, delay in enumerate(delays):
            rec.progress(i, of=len(delays), t1_us=fit.derived["T1"])
        rec.artifact("artifacts/fit.png", kind="figure")
        rec.done(summary="T1 = 68.2 us")

    Every event is appended to ``progress.jsonl`` and flushed; nothing is buffered for later.
    """

    def __init__(
        self,
        task_dir: str | Path,
        manifest: Manifest,
        *,
        supervised: bool = False,
        heartbeat: bool = False,
    ) -> None:
        self.dir = Path(task_dir)
        self._manifest = manifest
        self.supervised = bool(supervised)
        #: the timer thread's period in seconds, seeded from the manifest (tests shorten it).
        self.heartbeat_seconds: float = float(manifest.heartbeat_interval_s)
        self._lock = threading.Lock()
        self._fh: Any = None
        self._stop = threading.Event()
        self._timer: threading.Thread | None = None
        self._terminal_written = (self.dir / RESULT).exists()
        if heartbeat:
            self.start_heartbeat()

    # -- construction ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        task_dir: str | Path,
        *,
        title: str = "",
        device: str = "",
        script: str = "",
        workdir: str = "",
        language: str = "python",
        kind: str = "experiment",
        task_type: str = "experiment",
        target: str = "sim",
        gates: Iterable[str] = (),
        authored_by: Any = None,
        heartbeat_interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
        config_content_id: str = "",
        host: str | None = None,
        pid: int | None = None,
        supervised: bool = False,
        heartbeat: bool = True,
        exist_ok: bool = False,
    ) -> TaskRecord:
        """Create the task directory and write its manifest.

        The task **id is the directory's basename** — one invariant instead of two ids that can
        disagree, so a supervisor holding an id knows the directory and vice versa. Use
        :meth:`create_under` to mint the conventional ``YYYY-MM-DD-<slug>-<4 hex>`` id.

        ``pid``: ``None`` (the default) means *stamp my own* — the standalone case. A supervisor's
        ``strumento task create`` passes ``pid=0`` ("not started"), because in a supervised run the
        supervisor owns that field (see the module docstring's single-writer rule).
        """
        task_dir = Path(task_dir)
        if (task_dir / MANIFEST).exists() and not exist_ok:
            raise RecordError(f"a task record already exists at {task_dir}")
        if target not in TARGETS:
            raise RecordError(f"target must be one of {TARGETS}, got {target!r}")
        if language not in LANGUAGES:
            raise RecordError(f"language must be one of {LANGUAGES}, got {language!r}")
        if kind not in MANIFEST_KINDS:
            raise RecordError(f"kind must be one of {MANIFEST_KINDS}, got {kind!r}")

        manifest = Manifest(
            id=task_dir.name,
            title=str(title),
            kind=str(kind),
            task_type=str(task_type),
            target=str(target),
            gates=tuple(str(g) for g in gates),
            script=str(script),
            workdir=str(workdir or ""),
            language=str(language),
            device=str(device),
            created=now_iso(),
            pid=int(os.getpid() if pid is None else pid),
            host=socket.gethostname() if host is None else str(host),
            heartbeat_interval_s=int(heartbeat_interval_s),
            config_content_id=str(config_content_id),
            authored_by=authored_by if isinstance(authored_by, AuthoredBy)
            else AuthoredBy.from_dict(authored_by, strict=True),
        )
        (task_dir / ARTIFACTS).mkdir(parents=True, exist_ok=True)
        atomic_write(task_dir / MANIFEST, manifest.to_toml())
        return cls(task_dir, manifest, supervised=supervised, heartbeat=heartbeat)

    @classmethod
    def create_under(cls, root: str | Path | None = None, *, title: str = "", **kwargs) -> TaskRecord:
        """Mint a task id, create ``<root>/<id>/``, and write the manifest.

        ``root`` defaults to ``<workdir>/experiments/`` (``workdir`` defaults to the cwd) — the
        standalone default only. A supervisor always passes ``--root`` explicitly, resolved by the
        same rule as its rescan roots, so every supervised record is rediscoverable.
        """
        workdir = kwargs.get("workdir") or os.getcwd()
        base = Path(root) if root else Path(workdir) / "experiments"
        task_dir = claim_task_dir(base, title)
        try:
            return cls.create(task_dir, title=title, **kwargs)
        except BaseException:
            _release_claim(task_dir)
            raise

    @classmethod
    def open(
        cls,
        task_dir: str | Path,
        *,
        supervised: bool = False,
        heartbeat: bool = False,
    ) -> TaskRecord:
        """Attach to an existing record (default: as a *reader* — no heartbeat thread)."""
        return cls(task_dir, read_manifest(task_dir), supervised=supervised, heartbeat=heartbeat)

    @classmethod
    def from_env(cls, **kwargs) -> TaskRecord:
        """The one call a task script makes: attach to the supervisor's dir, or create a fresh one.

        With :data:`TASK_DIR_ENV` set we are **supervised**: the record attaches to the directory
        the supervisor already created and never writes ``pid``. Without it we are standalone: a
        fresh directory is created under ``root`` (default ``<workdir>/experiments/``) and this
        process stamps its own pid. Either way the script's code is identical — which is the point.
        """
        env_dir = os.environ.get(TASK_DIR_ENV, "").strip()
        heartbeat = bool(kwargs.pop("heartbeat", True))
        if env_dir:
            path = Path(env_dir)
            if (path / MANIFEST).exists():
                rec = cls.open(path, supervised=True, heartbeat=heartbeat)
                (rec.dir / ARTIFACTS).mkdir(parents=True, exist_ok=True)
                return rec
            # unusual but recoverable: the env var points at a dir with no manifest yet. Write one,
            # still as a supervised record, so `pid` keeps its single writer.
            kwargs.pop("root", None)
            return cls.create(path, pid=0, supervised=True, heartbeat=heartbeat, exist_ok=True, **kwargs)
        return cls.create_under(kwargs.pop("root", None), heartbeat=heartbeat, **kwargs)

    # -- state -------------------------------------------------------------------------

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    @property
    def id(self) -> str:
        return self._manifest.id

    @property
    def artifacts(self) -> Path:
        """``<task dir>/artifacts`` — created at record creation, so a script can just write there."""
        path = self.dir / ARTIFACTS
        path.mkdir(parents=True, exist_ok=True)
        return path

    def events(self, *, limit: int | None = None) -> list[dict]:
        return read_events(self.dir, limit=limit)

    def status(self) -> TaskStatus:
        return derive_state(self.dir)

    # -- the event stream --------------------------------------------------------------

    def _append(self, ev: str, **fields: Any) -> dict:
        event = {"t": now_iso(), "ev": ev, **fields}
        line = json.dumps(event, default=str)
        with self._lock:
            if self._fh is None:
                self.dir.mkdir(parents=True, exist_ok=True)
                path = self.dir / EVENTS
                # a previous run killed mid-write leaves an unterminated line; start a fresh one
                # rather than splicing this event onto the fragment (both would then be unreadable).
                needs_newline = path.exists() and path.stat().st_size > 0 and not _ends_with_newline(path)
                self._fh = path.open("a")
                if needs_newline:
                    self._fh.write("\n")
            self._fh.write(line + "\n")
            self._fh.flush()          # per-line: a crash loses at most the line in flight
        return event

    def progress(self, step: int, of: int | None = None, label: str | None = None, **metrics: Any) -> dict:
        """One step of the loop, with whatever numbers this experiment happens to report.

        ``metrics`` is an open string→number map by design: a T1 monitor reports ``t1_us``, a
        calibration loop ``infidelity``, and a reader plots whatever series it finds — no schema
        change per experiment type. ``of`` may be omitted for an open-ended monitor (a reader then
        shows a counter, not a bar).
        """
        fields: dict[str, Any] = {"step": int(step)}
        if of is not None:
            fields["of"] = int(of)
        if label is not None:
            fields["label"] = str(label)
        fields["metrics"] = {str(k): _as_number(k, v) for k, v in metrics.items()}
        return self._append("progress", **fields)

    def artifact(self, path: str | Path, kind: str | None = None) -> dict:
        """Record something saved to disk. The path must resolve **inside** the task directory."""
        rel = relative_artifact(self.dir, path)
        fields: dict[str, Any] = {"path": rel}
        if kind:
            fields["kind"] = str(kind)
        return self._append("artifact", **fields)

    def heartbeat(self) -> dict:
        """Explicit liveness — decoupled from experiment cadence.

        A 20-minute acquisition must not look dead, and the timer thread cannot fire while another
        language holds the interpreter lock (a Julia task's re-solve step does exactly that). So
        liveness is *also* a thing a task says out loud, at whatever boundary it has.
        """
        return self._append("heartbeat")

    def calibration(self, content_id: str, /, **fields: Any) -> dict:
        """The store pointer moved to ``content_id`` — the record's cross-link into calibration
        history. The task substrate never writes calibration itself; a script does, through the
        store API, and then says so here.

        ``fields`` are optional JSON values saying where the write came from — the MCP session
        passes ``soc_kind``, ``simulated`` and, for the simulator, ``truth_hash``. ``t``, ``ev``
        and ``content_id`` are the event's own keys and are refused as field names."""
        if not str(content_id).strip():
            raise RecordError("calibration() needs a config content id")
        reserved = sorted({"t", "ev", "content_id"} & set(fields))
        if reserved:
            raise RecordError(
                f"calibration() field names {', '.join(reserved)} are reserved for the event itself"
            )
        return self._append(
            "calibration",
            content_id=str(content_id),
            **{str(k): _jsonish(v) for k, v in fields.items()},
        )

    def gate(self, name: str, passed: bool, **params: Any) -> dict:
        """Record a gate **verdict** that vetted code computed.

        The verdict travels; the judgement does not. This method is a wire, and a task script is
        expected to *route* verdicts through it — the pass/fail itself comes from shipped, reviewed
        code (:meth:`strumento.core.experiment.Experiment.run` emits the fit and fit-recovery
        gates), because a learned routing table fed self-graded homework is worse than no table.
        The honest scope of that guarantee: nothing at this layer can stop a script from calling
        ``gate("fit-recovery", True)`` with no fit behind it. What makes the boundary real is
        supervisor-side — verdicts are re-derived from the record before they are forwarded, and
        only gates the manifest declared are forwarded at all. This docstring is the contract, not
        the enforcement.
        """
        if not str(name).strip():
            raise RecordError("gate() needs a gate name")
        return self._append(
            "gate",
            name=str(name),
            **{"pass": bool(passed)},
            params={str(k): _jsonish(v) for k, v in params.items()},
        )

    # -- campaign events (generic substrate mechanism — the runner reads them, not this) ---

    def claim(self, device: str, ref: str) -> dict:
        """The campaign's persistent, authoritative claim on ``(device, ref)``.

        A claim is a **record, not a reservation**: exclusivity is *derived* from the event
        streams (a terminal campaign state releases the claim on derivation) — never enforced
        by a lock file this append could create. What this emitter guarantees is that the event
        is well-formed and durable; who may claim, and what a colliding claim means, is the
        orchestration layer's to read.
        """
        return self._append(
            "claim",
            device=_required_str("claim", "device", device),
            ref=_required_str("claim", "ref", ref),
        )

    def step_open(self, step: str, attempt: int, child_path: str | Path) -> dict:
        """Step attempt ``attempt`` is starting, and its child task record lives at
        ``child_path`` — this event is the **authoritative** child mapping (a child directory's
        own naming is a cache; derivation reads the event). A crashed attempt is precisely a
        ``step_open`` with no matching ``step_close``, and that is how it derives.
        """
        child = child_path.as_posix() if isinstance(child_path, Path) else child_path
        return self._append(
            "step_open",
            step=_required_str("step_open", "step", step),
            attempt=_attempt("step_open", attempt),
            child_path=_required_str("step_open", "child_path", child),
        )

    def step_close(self, step: str, attempt: int, outcome: str) -> dict:
        """The attempt finished, carrying the ``outcome`` it produced — an open string: what
        counts as ``pass`` is the runner's vocabulary, not the substrate's.
        """
        return self._append(
            "step_close",
            step=_required_str("step_close", "step", step),
            attempt=_attempt("step_close", attempt),
            outcome=_required_str("step_close", "outcome", outcome),
        )

    def park(
        self,
        kind: str,
        reason: str,
        *,
        step: str | None = None,
        proposal_ref: str | None = None,
        envelope: dict[str, float] | None = None,
        region: str | None = None,
        bounds: Sequence[str] | None = None,
        child_path: str | None = None,
        config_content_id: str | None = None,
    ) -> dict:
        """The campaign parked: it is holding for something (a proposal awaiting approval, a
        human decision), with the reason recorded.

        ``kind`` is a **non-empty open string**: core may interpret ``propose`` and
        ``needs_human``, but an unknown park kind is recorded verbatim — on derivation it
        *holds* the campaign as parked-unresolved (the runner's rule, and the reason opacity
        there means parked, never released); readers of inspecting surfaces simply show it.

        The optional keyword payload carries a **staged proposal's** linkage (issue #55's
        autonomy layer — the runner is the writer): ``step`` names the step that proposed,
        ``proposal_ref`` the operator-addressable approval handle, ``envelope`` the
        per-channel applied-excitation maxima the proposal would clear once approved,
        ``region`` the declared DC region artifact, ``bounds`` the bounded writeback paths,
        ``child_path`` the child record the proposal is banked in, and ``config_content_id``
        the config the proposal was compiled against (the re-validation anchor). Like every
        payload field it is structural data this emitter validates at append; its *meaning*
        belongs to the orchestration layer that reads it.
        """
        fields: dict[str, Any] = {}
        if step is not None:
            fields["step"] = _required_str("park", "step", step)
        if proposal_ref is not None:
            fields["proposal_ref"] = _required_str("park", "proposal_ref", proposal_ref)
        if envelope is not None:
            if not isinstance(envelope, dict):
                raise RecordError(
                    f"park() envelope must map channel names to maxima — got {envelope!r}"
                )
            cleaned: dict[str, float] = {}
            for channel, value in envelope.items():
                if not isinstance(channel, str) or not channel.strip():
                    raise RecordError(
                        f"park() envelope channel {channel!r} is not a channel name"
                    )
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or value != value  # NaN
                ):
                    raise RecordError(
                        f"park() envelope channel {channel!r} declares a non-numeric"
                        f" maximum {value!r}"
                    )
                cleaned[channel] = value
            fields["envelope"] = cleaned
        if region is not None:
            fields["region"] = _required_str("park", "region", region)
        if bounds is not None:
            if isinstance(bounds, str) or not isinstance(bounds, (list, tuple)):
                raise RecordError(
                    f"park() bounds must be a LIST of calibration paths, got {bounds!r}"
                )
            fields["bounds"] = [
                _required_str("park", "bound", b) for b in bounds
            ]
        if child_path is not None:
            fields["child_path"] = _required_str("park", "child_path", child_path)
        if config_content_id is not None:
            fields["config_content_id"] = _required_str(
                "park", "config_content_id", config_content_id
            )
        return self._append(
            "park",
            kind=_required_str("park", "kind", kind),
            reason=_required_str("park", "reason", reason),
            **fields,
        )

    def approve(self, actor: str, channel: str, proposal_ref: str) -> dict:
        """A privileged approval: ``actor`` (who approved) + ``channel`` (the authentication
        channel, e.g. ``operator-cli``) + the proposal it clears, linked by ``proposal_ref``.

        Only the operator CLI / supervisor appends these — every agent-reachable surface has
        no path to this emitter. The trust model is v1 and stated plainly: actor/channel
        validation is robustness against buggy writers, not a cryptographic boundary. An
        approve event that reaches a stream missing its actor or channel (hand-forged, not
        emitted) is ignored at derivation — the runner's rule, not this method's.
        """
        return self._append(
            "approve",
            actor=_required_str("approve", "actor", actor),
            channel=_required_str("approve", "channel", channel),
            proposal_ref=_required_str("approve", "proposal_ref", proposal_ref),
        )

    def resume(self, actor: str, directive: str) -> dict:
        """A parked campaign resumes: the ``actor`` plus the ``directive`` that picks the
        edge and states whether retries reset. The directive's meaning is the runner's; this
        emitter only guarantees it travels, well-formed and durable.
        """
        return self._append(
            "resume",
            actor=_required_str("resume", "actor", actor),
            directive=_required_str("resume", "directive", directive),
        )

    def poison(self, paths: list[str] | tuple[str, ...], reason: str) -> dict:
        """Calibration paths are forced stale (excursion poisoning): a DC step's moves
        left the cleared region in which a drift-sensitive artifact was acquired — poisons
        that artifact's path until a NEWER write of it lands. ``paths`` is a non-empty list
        of non-empty calibration paths; ``reason`` says what moved where. Like every
        campaign event this is substrate: it validates structurally at append and means
        what the runner's derivation says it means (a resume clears no poison — data
        validity is re-measured, not waved through).
        """
        if isinstance(paths, str) or not isinstance(paths, (list, tuple)):
            raise RecordError(
                f"poison() needs a LIST of calibration paths, got {paths!r} — one path or"
                " many, always a list"
            )
        cleaned = [ _required_str("poison", "path", p) for p in paths ]
        if not cleaned:
            raise RecordError(
                "poison() needs at least one calibration path — an empty poison poisons"
                " nothing and would be malformed"
            )
        return self._append("poison", paths=cleaned,
                            reason=_required_str("poison", "reason", reason))

    # -- the heartbeat timer -----------------------------------------------------------

    def start_heartbeat(self) -> None:
        """Start the best-effort timer thread (idempotent).

        Best-effort is not an apology, it is the design: this thread covers the boring case (a long
        acquire in C, waiting on a socket), and the *task* covers the case the thread cannot — see
        :meth:`heartbeat`.
        """
        if self._timer is not None or self.heartbeat_seconds <= 0:
            return
        self._stop.clear()
        self._timer = threading.Thread(
            target=self._heartbeat_loop, name=f"strumento-heartbeat-{self.id}", daemon=True
        )
        self._timer.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                self.heartbeat()
            except Exception:
                return  # a timer thread must never be the thing that kills a task

    def stop_heartbeat(self) -> None:
        self._stop.set()
        timer, self._timer = self._timer, None
        if timer is not None and timer is not threading.current_thread():
            timer.join(timeout=1.0)

    # -- in-place manifest updates (the only three fields) -----------------------------

    def _update_manifest(self, **fields: Any) -> Manifest:
        unknown = sorted(set(fields) - set(MUTABLE_FIELDS))
        if unknown:
            raise RecordError(
                f"manifest field(s) {unknown} are immutable after creation; only "
                f"{list(MUTABLE_FIELDS)} may be updated in place"
            )
        self._manifest = replace(self._manifest, **fields)
        atomic_write(self.dir / MANIFEST, self._manifest.to_toml())
        return self._manifest

    def set_config(self, content_id: str) -> Manifest:
        """Stamp ``config_content_id`` — once, while it is empty.

        The calibration a task ran against is part of what the task *is*: re-stamping it would
        silently rewrite the provenance of data already taken. Re-stamping the same id is a no-op;
        a different one is an error.
        """
        content_id = str(content_id).strip()
        if not content_id:
            raise RecordError("set_config() needs a config content id")
        current = self._manifest.config_content_id
        if current == content_id:
            return self._manifest
        if current:
            raise RecordError(
                f"config_content_id is already {current!r}: the calibration a task ran against is "
                f"stamped once (a new calibration mid-task is a `calibration` event, not a restamp)"
            )
        return self._update_manifest(config_content_id=content_id)

    def set_pid(self, pid: int) -> Manifest:
        """Write ``pid`` — **the spawner's field**. A supervised record refuses (single writer)."""
        if self.supervised:
            raise RecordError(
                f"this record is supervised ({TASK_DIR_ENV} is set): the supervisor writes pid at "
                f"spawn and the script never touches it"
            )
        return self._update_manifest(pid=int(pid))

    # -- terminal states ----------------------------------------------------------------

    def done(self, summary: str = "") -> Result:
        """Terminal ``done`` + a one-line human summary."""
        return cast(Result, self._terminal("done", summary=summary))  # strict: never None

    def fail(self, error: str, kind: str = "script-error", *, summary: str = "") -> Result:
        """Terminal ``failed``. ``kind`` separates *scientific* from *operational* failure:
        ``script-error`` is the experiment's problem, ``transport-error`` is the lab's."""
        if kind not in ERROR_KINDS:
            raise RecordError(f"error_kind must be one of {ERROR_KINDS}, got {kind!r}")
        # strict: _terminal raises rather than return None
        return cast(Result, self._terminal(
            "failed", summary=summary, error=str(error), error_kind=kind,
        ))

    def cancelled(self, summary: str = "", *, error: str = "", kind: str = "") -> Result:
        """Terminal ``cancelled`` — normally written by the :meth:`on_cancel` handler."""
        # strict: _terminal raises rather than return None
        return cast(Result, self._terminal(
            "cancelled", summary=summary, error=error, error_kind=kind,
        ))

    def _terminal(
        self,
        state: str,
        *,
        summary: str = "",
        error: str = "",
        error_kind: str = "",
        strict: bool = True,
    ) -> Result | None:
        path = self.dir / RESULT
        if self._terminal_written or path.exists():
            if strict:
                raise RecordError(
                    f"{RESULT} is written exactly once and already exists at {path}"
                )
            try:
                return read_result(self.dir)
            except RecordError:
                return None
        result = Result(
            state=state,
            ended=now_iso(),
            summary=str(summary),
            error_kind=str(error_kind),
            error=str(error),
        )
        atomic_write(path, result.to_toml())
        self._terminal_written = True
        self.close()
        return result

    def on_cancel(
        self,
        cleanup: Callable[[], Any] | None = None,
        *,
        signals: Sequence[int] = (signal.SIGTERM,),
        exit_code: int | None = None,
    ) -> None:
        """Install the cancel path: signal → ``cleanup()`` → ``result.toml(cancelled)`` → exit.

        A cancel that leaves no ``result.toml`` is indistinguishable from a crash, and it should be:
        a ``SIGKILL`` after the grace period *is* an unclean stop, and the record honestly derives
        ``stale``. A ``SIGTERM`` handled here is the clean stop, and it says so.

        A ``cleanup`` that raises is recorded (as ``script-error``) and the cancelled marker is
        still written — the point of the handler is that the record always ends up truthful.
        """

        def handler(signum, _frame):  # pragma: no cover - exercised in a subprocess
            error = ""
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as exc:
                    error = f"cleanup failed: {type(exc).__name__}: {exc}"
            self._terminal(
                "cancelled",
                summary=f"cancelled by signal {signum}",
                error=error,
                error_kind="script-error" if error else "",
                strict=False,
            )
            raise SystemExit(exit_code if exit_code is not None else 128 + int(signum))

        for signum in signals:
            try:
                signal.signal(signum, handler)
            except ValueError as err:  # not the main thread
                raise RecordError(f"on_cancel must be installed from the main thread: {err}") from err

    # -- the file lifecycle -----------------------------------------------------------
    #
    # The event file opens lazily on the first append and stays open between appends; these
    # are the ways to let go of it. All are idempotent, and none is final: an event appended
    # afterwards reopens the file and appends (callers rely on write-after-close working).

    def __enter__(self) -> TaskRecord:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()  # on every exit path; an exception from the body still propagates

    def close(self) -> None:
        """Stop the timer thread and close the event file. Idempotent."""
        self.stop_heartbeat()
        self.release()

    def release(self) -> None:
        """Close the event file but leave the heartbeat timer alone. Idempotent.

        For an owner that writes through a record it did not create (a bring-up runner
        advancing a campaign): it lets go of the handle it caused to be opened without
        stopping a heartbeat that belongs to someone else.
        """
        with self._lock:
            fh, self._fh = self._fh, None
            if fh is not None:
                try:
                    fh.flush()
                finally:
                    fh.close()


def _required_str(emitter: str, field: str, value: Any) -> str:
    """A campaign event's required string payload: non-empty after strip, or the append is
    malformed and refused **at the emitter** — never discovered at read time."""
    text = str(value).strip()
    if not text:
        raise RecordError(f"{emitter}() needs a non-empty {field}: the event would be malformed")
    return text


def _attempt(emitter: str, value: Any) -> int:
    """Attempt numbers are 1-based integers — attempt 0, a negative, or a non-integer is
    malformed and refused at the emitter."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise RecordError(f"{emitter}() needs an integer attempt number, got {value!r}") from None
    if n < 1:
        raise RecordError(f"{emitter}() needs a 1-based attempt number, got {n}")
    return n


def _as_number(key: str, value: Any) -> float | int:
    """``metrics`` values are numbers — an open *numeric* map, so a reader can always plot it."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        raise RecordError(
            f"progress metric {key!r} must be a number (metrics is an open string->number map), "
            f"got {type(value).__name__}"
        ) from None


def _jsonish(value: Any) -> Any:
    """Coerce a gate parameter to JSON primitives — provenance must never fail to serialize."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonish(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonish(v) for v in value]
    item = getattr(value, "item", None)      # numpy scalars, without importing numpy
    if callable(item):
        try:
            return _jsonish(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)  # numpy arrays, same reason
    if callable(tolist):
        try:
            return _jsonish(tolist())
        except Exception:
            pass
    return str(value)
