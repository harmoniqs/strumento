"""Machinery for the compiled-program snapshot gate (issue #151).

The gate itself is ``tests/packs/test_compiled_snapshot_gate.py``; this module holds what it and
its meta-tests share: the matrix, capture, normalization, rendering, diff and update.

**References are local.** They live in :data:`GOLDEN_DIR`, which is gitignored: generate them on
``main`` with ``--update-compiled-golden``, then run the gate on your branch. With none on disk
the comparison tests skip with :data:`NO_REFERENCES`; a partial set still fails.

**The matrix.** Every registered experiment is compiled on exactly one shipped demo: the device
its experiment-contract run builds (``tests/packs/test_experiment_contract.py``), with that
file's ``CONTRACT_KWARGS`` and device factories, so the gate never carries a second copy of the
arguments. That routing is:

=====  ==============  =====================================================================
pack   device          experiments
=====  ==============  =====================================================================
cqed   loopback_demo   the cqed experiments outside the contract's ``_CQED_EXPANSION_NAMES``
                       (single-generator testbench; the contract injects one cavity
                       ``revival_time`` for ``parity_fringe``)
cqed   multimode_demo  ``_CQED_EXPANSION_NAMES`` (readout, cavity, sideband and coupler lines)
spin   spin_demo       every spin experiment
=====  ==============  =====================================================================

The tweezer pack registers no experiments. Compiling every experiment on every demo is not the
goal: many cqed experiments have no line to play on the loopback testbench. An experiment that
can't compile on its contract device goes in :data:`CANNOT_COMPILE` with its reason; a demo
with a snapshot that is deliberately not gated goes in :data:`UNGATED_SNAPSHOT_DEVICES`. Guard
tests fail if either list drifts from reality.

**Capture.** Each experiment runs its own ``go()`` path, ``analyze=False``, on a zero-IQ
``MockQickSocV2`` built from the device's own committed ``soccfg_snapshot`` (never the local
``soccfg_v2.json`` override ``tests/_v2gate.py`` honours), so the experiment's own default
``reps``/``final_delay`` and any host-batched multi-program ``acquire()`` are what gets pinned.
A runner hook records, for every program the experiment builds, ``compiled_payload(prog)``
(qick's ``dump_prog()``: ASM, waveforms, envelopes, generator/readout declarations, loop dims)
and ``acquire_config(prog)``. The hook is ``Runner._arm``, called with the built program just
before acquisition; if it is ever renamed, capture records nothing and fails loudly.

**Determinism.** Verified, not assumed: in-process twice, in a fresh interpreter with a different
hash seed, and from another working directory, the rendered payloads are byte-identical; they
are also identical under qick 0.2.400, 0.2.422, 0.2.431 and 0.2.436, and carry no path, user or
version string. So :data:`NORMALIZATIONS` is empty. If a field ever varies, add it there with
its reason and rewrite it in :func:`normalize` — never drop it silently.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

from strumento.core.experiment import Runner
from tests._paths import LOOPBACK_YAML, MULTIMODE_YAML, ROOT, SPIN_YAML
from tests.packs import test_experiment_contract as contract

#: Local, gitignored (``.gitignore``): references are generated on ``main``, never committed.
GOLDEN_DIR = ROOT / "tests" / "golden" / "compiled"

#: Why the comparison tests skip when :data:`GOLDEN_DIR` holds no reference at all.
NO_REFERENCES = (
    "no local compiled-program references; generate them on main with `pytest "
    "tests/packs/test_compiled_snapshot_gate.py --update-compiled-golden`, then run on your branch"
)

#: experiment name → why it can't compile with its contract arguments on its contract device.
#: Empty today: ``length_rabi``'s zero-length default pulse fails only with its *default*
#: arguments; its contract scan starts at 0.01 µs and compiles.
CANNOT_COMPILE: dict[str, str] = {}

#: shipped demo with a soccfg snapshot → why the gate compiles nothing on it.
UNGATED_SNAPSHOT_DEVICES: dict[str, str] = {}

#: JSON path (``$.programs[*]...``) → why it is rewritten before comparison. Empty: see the
#: module docstring for the determinism evidence that makes every field comparable as-is.
NORMALIZATIONS: dict[str, str] = {}

#: Diff output budget: at most this many difference lines, each value cut to this many chars.
MAX_DIFF_LINES = 40
MAX_VALUE_CHARS = 80

#: Entry ids rewritten by ``--update-compiled-golden`` this session (reported at the end).
UPDATED: list[str] = []


class Entry(NamedTuple):
    pack: str
    device: str
    experiment: str

    @property
    def id(self) -> str:
        return f"{self.pack}/{self.device}/{self.experiment}"

    @property
    def path(self) -> Path:
        return GOLDEN_DIR / self.pack / self.device / f"{self.experiment}.json"

    @property
    def rel(self) -> str:
        path = self.path
        return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def references_exist() -> bool:
    """Whether any local reference file is on disk (none → the comparison tests skip)."""
    return GOLDEN_DIR.is_dir() and any(GOLDEN_DIR.rglob("*.json"))


def contract_device(name: str) -> str:
    """The demo folder the contract builds ``name`` on (mirrors ``_contract_device_and_soccfg``;
    the gate's guard cross-checks the two)."""
    if name in contract._SPIN_EXPERIMENT_NAMES:
        return SPIN_YAML.parent.name
    if name in contract._CQED_EXPANSION_NAMES:
        return MULTIMODE_YAML.parent.name
    return LOOPBACK_YAML.parent.name


def entry_for(name: str) -> Entry:
    pack = next(p for p, reg in sorted(contract.REGISTRIES.items()) if name in reg)
    return Entry(pack, contract_device(name), name)


def matrix() -> list[Entry]:
    """Every (pack, device, experiment) the gate compiles, sorted."""
    return sorted(
        entry_for(name)
        for reg in contract.REGISTRIES.values()
        for name in reg
        if name not in CANNOT_COMPILE
    )


class _CapturingRunner(Runner):
    """The real ``Runner``, recording each built program's compiled payload before acquiring."""

    def __init__(self) -> None:
        super().__init__()
        self.programs: list[dict] = []

    def _arm(self, soc, prog, device, experiment, seq) -> None:
        from strumento.core.program import acquire_config, compiled_payload

        if prog.binprog is None:
            prog.compile()
        self.programs.append({"program": compiled_payload(prog), "acquire": acquire_config(prog)})
        super()._arm(soc, prog, device, experiment, seq)


def capture(entry: Entry) -> dict:
    """Compile ``entry`` (qick required) → the record its reference file holds."""
    from strumento.core.mock import MockQickSocV2

    cls = contract.REGISTRIES[entry.pack][entry.experiment]
    dev, _ = contract._contract_device_and_soccfg(entry.experiment)
    exp = cls(dev, **dict(contract.CONTRACT_KWARGS[entry.experiment]))
    runner = _CapturingRunner()
    exp.runner = runner
    exp.go(MockQickSocV2(dev.soccfg_snapshot), analyze=False, progress=False)
    if not runner.programs:
        raise AssertionError(f"{entry.id}: go() built no program through the runner hook")
    return {
        "pack": entry.pack,
        "device": entry.device,
        "experiment": entry.experiment,
        "programs": runner.programs,
    }


def normalize(record: dict) -> dict:
    """Apply :data:`NORMALIZATIONS` to a copy of ``record`` (none are needed today)."""
    return copy.deepcopy(record)


def render(record: dict) -> str:
    """The reference-file text: normalized, keys sorted, one-space indent, trailing newline."""
    return json.dumps(normalize(record), sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def digests(entries) -> dict[str, str]:
    """``entry.id`` → SHA-256 of its rendered payload (the determinism check's currency)."""
    return {e.id: hashlib.sha256(render(capture(e)).encode()).hexdigest() for e in entries}


def sync(path: Path, text: str, *, update: bool) -> str:
    """Compare ``text`` with the file at ``path``: ``same`` | ``differs`` | ``missing`` without
    ``update`` (never writes), ``same`` | ``written`` with it (writes only when different)."""
    exists = path.is_file()
    if exists and path.read_text(encoding="utf-8") == text:
        return "same"
    if not update:
        return "differs" if exists else "missing"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return "written"


def _short(value) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return text if len(text) <= MAX_VALUE_CHARS else text[: MAX_VALUE_CHARS - 1] + "…"


def _walk(old, new, path: str):
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            sub = f"{path}.{key}"
            if key not in new:
                yield f"{sub}: removed (was {_short(old[key])})"
            elif key not in old:
                yield f"{sub}: added (now {_short(new[key])})"
            else:
                yield from _walk(old[key], new[key], sub)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            yield f"{path}: length {len(old)} → {len(new)}"
        for i in range(max(len(old), len(new))):
            sub = f"{path}[{i}]"
            if i >= len(new):
                yield f"{sub}: removed (was {_short(old[i])})"
            elif i >= len(old):
                yield f"{sub}: added (now {_short(new[i])})"
            else:
                yield from _walk(old[i], new[i], sub)
    elif old != new or type(old) is not type(new):
        yield f"{path}: {_short(old)} → {_short(new)}"


def json_diff(old, new, path: str = "$") -> list[str]:
    """Field-level differences, ``<json path>: old → new``, cut to :data:`MAX_DIFF_LINES`."""
    lines = list(_walk(old, new, path))
    if len(lines) <= MAX_DIFF_LINES:
        return lines
    rest = len(lines) - MAX_DIFF_LINES
    return [*lines[:MAX_DIFF_LINES], f"… and {rest} more difference{'s' if rest > 1 else ''}"]


def diff_report(entry: Entry, old: dict, new: dict) -> str:
    """The gate's failure message: which program changed, and where."""
    body = "\n".join(f"  {line}" for line in json_diff(normalize(old), normalize(new)))
    return (
        f"compiled program changed: {entry.id} (pack {entry.pack}, device {entry.device}, "
        f"experiment {entry.experiment})\n{body}\n"
        f"The local references were generated on main; if this change to what is played is "
        f"intended, say so in the PR (CONTRIBUTING.md, 'Compiled-program snapshot gate')."
    )
