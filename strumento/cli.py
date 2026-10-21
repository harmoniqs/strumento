"""The ``strumento`` command-line interface — agent- and automation-forward.

Every command is importable (``from strumento.cli import main; main([...])``), prints
human-readable text by default, and machine-readable JSON with ``--json`` — so both a person at
a terminal and an agent in CI get first-class output. Only ``compile``'s realized-values step
touches ``qick`` (and degrades gracefully without it); everything else is pure.

**Stateless by design (D33).** Each verb is one process: load, answer, exit. The *stateful* agent
door — a long-lived session holding a device, a soc, a store and a ref — is the MCP server
(:mod:`strumento.mcp`), which is a thin shim over the **same importable functions** this module
wraps (:func:`device_summary`, :func:`flatten_calibration`, :func:`provenance_entries`,
``core.schemas``). Two doors, one implementation; see ``docs/cli.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, cast

import numpy as np

from strumento.core import record  # pure stdlib; imported eagerly for the parser's defaults


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "fn"):
        parser.print_help()
        return 2
    try:
        payload = args.fn(args)
    except Exception as err:  # surface errors as data, not tracebacks
        _emit({"ok": False, "error": f"{type(err).__name__}: {err}"}, args.json)
        return 1
    # A verb may answer with a *negative verdict* as data (``check`` FAIL): the exit code
    # carries the same verdict as the ``ok`` field, so scripts and CI see it.
    failed = payload.get("ok") is False
    _emit(payload, args.json)
    return 1 if failed else 0


def _emit(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=_jsonable))
        return
    text = payload.pop("_text", None)
    if text is not None:
        print(text)
    else:
        print(json.dumps(payload, indent=2, default=_jsonable))


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def _needs_subcommand(group: str, names: tuple[str, ...]):
    """A verb group with no subcommand is an error-as-data, not argparse's usage-and-exit-2."""

    def fn(args):
        raise ValueError(f"strumento {group} needs a subcommand: {', '.join(names)}")

    return fn


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="strumento", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(title="commands")

    def _verb(parent, name, fn, help_):
        c = parent.add_parser(name, help=help_)
        c.add_argument("--json", action="store_true", help="machine-readable JSON output")
        c.set_defaults(fn=fn, json=False)
        return c

    def cmd(name, fn, help_):
        return _verb(sub, name, fn, help_)

    def group(name, help_, names):
        g = sub.add_parser(name, help=help_)
        g.add_argument("--json", action="store_true", help="machine-readable JSON output")
        g.set_defaults(fn=_needs_subcommand(name, names), json=False)
        return g.add_subparsers(title=f"{name} subcommands")

    cmd("packs", _cmd_packs, "list installed modality packs (strumento.packs entry points)")

    c = cmd("validate", _cmd_validate, "load + validate a device instance YAML")
    c.add_argument("device_yaml", type=Path)

    c = cmd("info", _cmd_info, "summarize a device instance: topology, wiring, namespaces")
    c.add_argument("device_yaml", type=Path)

    c = cmd("soccfg", _cmd_soccfg, "inspect a dump_cfg() snapshot: board, tProc version, gens")
    c.add_argument("soccfg_json", type=Path)

    c = cmd("fit", _cmd_fit, "fit a CSV data file (columns: x,y — or x,i,q for circle)")
    c.add_argument("kind", choices=["rabi", "t1", "t2", "parity", "lorentzian", "circle"])
    c.add_argument("csv", type=Path)

    c = cmd("compile", _cmd_compile, "lower a canned Rabi sequence for a device; show the calls")
    c.add_argument("device_yaml", type=Path)

    c = cmd("schema", _cmd_schema, "emit JSON Schema for a typed contract (D35)")
    # deliberately not argparse `choices`: an unknown name must come back as data
    # ({"ok": false, "error": …} + exit 1), not as argparse's usage-and-exit-2.
    c.add_argument("contract", help="device | fit | result | operator-model | calibration")
    c.add_argument("--pack", default=None,
                   help="the pack whose calibration schema to emit (calibration contract only)")

    c = cmd("check", _cmd_check, "validate a device YAML against its soccfg snapshot")
    c.add_argument("device_yaml", type=Path)

    # -- task records (the long-running-experiment contract; see docs/task-record.md) ------
    task = group("task", "experiment-task records", ("create", "status", "tail", "list"))

    c = _verb(task, "create", _cmd_task_create, "write a fresh task record directory (pid = 0)")
    c.add_argument("--title", required=True, help="human title; the task id's slug comes from it")
    c.add_argument("--device", default="", help="device id (devices/<id>/); empty = deviceless")
    c.add_argument("--script", default="", help="script path, relative to --workdir")
    c.add_argument("--workdir", default="", help="working directory for the script (default: cwd)")
    c.add_argument("--language", default="python", help="python | julia")
    c.add_argument("--kind", default="experiment", help="reserved registry axis: experiment | bringup")
    c.add_argument("--task-type", default="experiment", help="tier-dispatch telemetry key")
    c.add_argument("--target", default="sim", help="sim | hw")
    c.add_argument("--gates", default="", help="comma-separated gate names this task claims")
    c.add_argument("--authored-by", default="", metavar="JSON",
                   help='agent provenance, e.g. \'{"profile": "experimenter", "model": "p/m"}\'')
    c.add_argument("--heartbeat-interval-s", type=int, default=record.DEFAULT_HEARTBEAT_INTERVAL_S,
                   help="honest liveness cadence (immutable after creation)")
    c.add_argument("--root", type=Path, default=None,
                   help="root the record dir is created under (default: <workdir>/experiments)")

    c = _verb(task, "status", _cmd_task_status, "derived state + the staleness inputs behind it")
    c.add_argument("dir", type=Path)

    c = _verb(task, "tail", _cmd_task_tail, "the last N events of progress.jsonl")
    c.add_argument("dir", type=Path)
    c.add_argument("-n", type=int, default=50, dest="n", help="how many events (default 50)")

    c = _verb(task, "list", _cmd_task_list, "every task record under a root, newest first")
    c.add_argument("root", type=Path)

    # -- the calibration store (read-only views; see docs/calibration-store.md) -------------
    store = group("store", "calibration store (read-only)", ("current", "log", "diff"))

    c = _verb(store, "current", _cmd_store_current, "the resolved pointer: ref, content id, when")
    c.add_argument("root", type=Path)
    c.add_argument("--ref", default="main")

    c = _verb(store, "log", _cmd_store_log, "version history for a ref (content id, when, source)")
    c.add_argument("root", type=Path)
    c.add_argument("--ref", default="main")

    c = _verb(store, "diff", _cmd_store_diff, "per-value calibration diff between two versions")
    c.add_argument("root", type=Path)
    c.add_argument("id_a", metavar="id-a")
    c.add_argument("id_b", metavar="id-b")

    return p


# -- commands -----------------------------------------------------------------------------


def _cmd_packs(args) -> dict:
    packs = sorted(ep.name for ep in entry_points(group="strumento.packs"))
    return {"ok": True, "packs": packs, "_text": "installed packs: " + ", ".join(packs)}


def _load_device(path: Path):
    from strumento.core.device import Device

    return Device.load(path)


def _cmd_validate(args) -> dict:
    dev = _load_device(args.device_yaml)
    names = sorted(dev.elements)
    return {
        "ok": True,
        "pack": dev.pack.__name__,
        "elements": names,
        "_text": f"OK — {dev!r}",
    }


def device_summary(dev) -> dict:
    """What a loaded device *is*: pack, elements, namespaces, wiring, readout, couplings.

    Importable because both agent doors need the same answer — ``strumento info`` prints it and the
    MCP session's ``device_info`` returns it (D33: two doors, one implementation). Calibration is
    deliberately **not** included here: the CLI verb is a topology view, while a session adds the
    live calibration on top (see :func:`flatten_calibration`).
    """
    wiring = {
        line: [
            {"role": g.role, "dac_ch": g.dac_ch, "nyquist": g.nyquist, "band_hz": g.band_hz}
            for g in ln.generators
        ]
        for line, ln in dev.wiring.lines.items()
    }
    namespaces = sorted(
        name for name in list(dev.elements) + ["sideband"]
        if hasattr(dev, name) and name not in ("readout",)
    )
    return {
        "pack": dev.pack.__name__,
        "elements": {n: type(e).__name__ for n, e in sorted(dev.elements.items())},
        "namespaces": namespaces,
        "wiring": wiring,
        "readout": {"dac_ch": dev.wiring.readout.dac_ch, "adc_chs": dev.wiring.readout.adc_chs},
        "couplings": [f"{c.a}—{c.b} ({c.form})" for c in dev.couplings],
    }


def _cmd_info(args) -> dict:
    dev = _load_device(args.device_yaml)
    info = {"ok": True, **device_summary(dev)}
    wiring, namespaces = info["wiring"], info["namespaces"]
    lines = [f"pack: {info['pack']}"]
    lines += ["elements: " + ", ".join(f"{n}:{t}" for n, t in info["elements"].items())]
    lines += ["namespaces: " + ", ".join(namespaces)]
    for line, gens in wiring.items():
        for g in gens:
            lines.append(f"  {line}.{g['role']:<9} -> DAC {g['dac_ch']} (nqz {g['nyquist']})")
    lines.append(f"  readout       -> DAC {info['readout']['dac_ch']} + ADC {info['readout']['adc_chs']}")
    info["_text"] = "\n".join(lines)
    return info


def _cmd_soccfg(args) -> dict:
    cfg = json.loads(args.soccfg_json.read_text())
    tprocs = cfg.get("tprocs", [])
    ttype = tprocs[0].get("type") if tprocs else None
    is_v2 = ttype == "qick_processor"
    gens = [
        {"type": g.get("type"), "maxv": g.get("maxv"), "fs": g.get("fs")}
        for g in cfg.get("gens", [])
    ]
    verdict = "tProc v2 — real AveragerProgramV2 assembly works offline" if is_v2 else (
        "tProc v1 — unit math only; v2 assembly needs a qick_processor config")
    return {
        "ok": True,
        "board": cfg.get("board"),
        "sw_version": cfg.get("sw_version"),
        "tproc_type": ttype,
        "tproc_revision": tprocs[0].get("revision") if tprocs else None,
        "is_v2": is_v2,
        "n_gens": len(gens),
        "n_readouts": len(cfg.get("readouts", [])),
        "gens": gens,
        "_text": (
            f"{cfg.get('board')} · qick {cfg.get('sw_version')} · tproc {ttype}"
            f" · {len(gens)} gen(s), {len(cfg.get('readouts', []))} readout(s)\n{verdict}"
        ),
    }


def _cmd_check(args) -> dict:
    """Validate a device YAML against its soccfg snapshot.

    Checks channel mapping, frequency ranges, and gain limits. Reports issues that would
    cause silent failures on hardware: wrong DAC channel, out-of-range frequency, gain
    exceeding the generator's maxv.

    ``ok`` is the verdict and ``main`` exits with it: 1 on any error (including a device with no
    snapshot attached), 0 on a pass — warnings alone never fail a check.
    """
    from strumento.core.device import Device

    dev = Device.load(args.device_yaml)
    issues: list[str] = []
    warnings: list[str] = []

    snap = dev.soccfg_snapshot
    if not snap:
        error = "device has no soccfg snapshot — cannot validate channel mapping"
        return {
            "ok": False,
            "error": error,
            "issues": [error],
            "warnings": [],
            "_text": "FAIL: no soccfg snapshot attached to this device",
        }

    gens = snap.get("gens", [])
    readouts = snap.get("readouts", [])
    tprocs = snap.get("tprocs", [])
    # qick's declare_gen / declare_readout take the INDEX into the snapshot's gens / readouts
    # list — not the entry's ``tproc_ch`` (the tProc port behind it). Validate by position.
    gen_chs = list(range(len(gens)))
    ro_chs = list(range(len(readouts)))

    # Check tproc version
    if tprocs and tprocs[0].get("type") != "qick_processor":
        issues.append(f"tProc type is {tprocs[0].get('type')!r}, not 'qick_processor' (v2 required)")

    # Check wiring: every dac_ch in the device YAML must exist in the soccfg gens
    for line_name, line in dev.wiring.lines.items():
        for gen_entry in line.generators:
            if gen_entry.dac_ch not in gen_chs:
                issues.append(
                    f"line {line_name!r} role {gen_entry.role!r}: dac_ch={gen_entry.dac_ch} "
                    f"not found in soccfg gens (have {gen_chs})"
                )

    # Check readout channels
    for adc_ch in dev.wiring.readout.adc_chs:
        if adc_ch not in ro_chs:
            issues.append(
                f"readout adc_ch={adc_ch} not found in soccfg readouts (have {ro_chs})"
            )

    # Check f_time vs f_fabric consistency
    if tprocs:
        f_time = tprocs[0].get("f_time")
        f_fabrics = {g.get("f_fabric") for g in gens}
        if f_time and f_fabrics and f_time not in f_fabrics:
            warnings.append(
                f"tproc f_time={f_time} differs from gen f_fabric={f_fabrics} — "
                "may cause timing-conflict warnings on back-to-back pulses"
            )

    # Check program memory
    if tprocs:
        pmem = tprocs[0].get("pmem_size", 0)
        if pmem < 2048:
            warnings.append(f"pmem_size={pmem} is small — deep RB sequences may not fit")

    passed = len(issues) == 0
    lines = []
    if passed:
        lines.append(f"PASS: {args.device_yaml} validates against its soccfg")
    else:
        lines.append(f"FAIL: {len(issues)} issue(s) found")
    for issue in issues:
        lines.append(f"  ERROR: {issue}")
    for warn in warnings:
        lines.append(f"  WARN: {warn}")
    lines.append(f"  {len(gen_chs)} gen(s), {len(ro_chs)} readout(s)")

    return {
        "ok": passed,
        "issues": issues,
        "warnings": warnings,
        "n_gens": len(gen_chs),
        "n_readouts": len(ro_chs),
        "_text": "\n".join(lines),
    }


_FITTERS = {
    "rabi": ("strumento.core.fitters", "fit_rabi"),
    "t1": ("strumento.core.fitters", "fit_exp"),
    "t2": ("strumento.core.fitters", "fit_decaysin_detuned"),
    "parity": ("strumento.core.fitters", "fit_parity"),
    "lorentzian": ("strumento.core.resonator_fitters", "fit_lorentzian"),
    "circle": ("strumento.core.resonator_fitters", "fit_resonator_circle"),
}


def _cmd_fit(args) -> dict:
    import importlib

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    # names=True: genfromtxt returns a structured array (or raises), so the names are set
    cols = list(cast("tuple[str, ...]", data.dtype.names))
    x = np.asarray(data[cols[0]], dtype=float)

    module, fname = _FITTERS[args.kind]
    fit_fn = getattr(importlib.import_module(module), fname)
    if args.kind == "circle":
        if len(cols) < 3:
            raise ValueError("circle fit needs three CSV columns: x,i,q")
        y = np.asarray(data[cols[1]], dtype=float) + 1j * np.asarray(data[cols[2]], dtype=float)
    else:
        y = np.asarray(data[cols[1]], dtype=float)

    fit = fit_fn(x, y)
    derived = {k: float(v) for k, v in fit.derived.items()}
    return {
        "ok": True,
        "kind": args.kind,
        "success": bool(fit.success),
        "derived": derived,
        "params": {k: float(v) for k, v in fit.params.items()},
        "_text": f"{args.kind}: success={fit.success} · "
        + " · ".join(f"{k}={v:.6g}" for k, v in derived.items()),
    }


class _RecordingProgram:
    """A tiny recording shim (mirrors the duck-typed program interface the compiler drives)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def record(*a, **kw):
            self.calls.append((name, a, kw))
            if name == "sweep_value":
                return {"sweep": a or kw}
            return None

        return record


def _cmd_compile(args) -> dict:
    from strumento.core.compiler import Compiler
    from strumento.core.pulses import Seq
    from strumento.core.sweeps import Loop, Sweep

    dev = _load_device(args.device_yaml)
    if not hasattr(dev, "qubit"):
        raise ValueError("compile demo expects an instance with a transmon named 'qubit'")
    # 16000, not 30000 (~0.916 frac -> GainOutOfRange: the factory validates sweep endpoints):
    # stays under MAX_QUBIT_DRIVE_GAIN=0.5.
    seq = Seq().play(dev.qubit.ge_pi(gain=Sweep(start=0, stop=16000, on="amp"))).measure()

    prog = _RecordingProgram()
    c = Compiler(prog, dev.wiring, axes=[Loop("amp", 51)])
    c.declare(seq)
    c.play(seq)
    calls = [
        {"call": name, "args": list(a), "kwargs": {k: v for k, v in kw.items()}}
        for name, a, kw in prog.calls
    ]

    realized: dict[str, Any] | None = None
    try:  # optional: realized values are read back from a compiled v2 program (needs qick)
        from strumento.core.program import StrumentoProgram

        p = dev.qubit.ge_pi()
        r = StrumentoProgram(dev, seq=Seq().play(p).measure()).realized(p)
        realized = {
            "freq_mhz_actual": r.freq_mhz_actual,
            "gain_frac": r.gain_frac,
            "length_us_actual": r.length_us_actual,
        }
    except ImportError:
        pass
    except Exception as exc:  # e.g. a v1 soccfg snapshot cannot assemble a v2 program
        realized = {"unavailable": f"{type(exc).__name__}: {exc}"}

    text = ["amplitude-Rabi lowering (canned demo):"]
    text += [f"  {c_['call']}({', '.join(f'{k}={v}' for k, v in c_['kwargs'].items())})" for c_ in calls]
    if realized:
        text.append(f"realized ge_pi: {realized}")
    return {"ok": True, "calls": calls, "realized": realized, "_text": "\n".join(text)}


def _cmd_schema(args) -> dict:
    """Emit one contract's JSON Schema (D35) — the machine-readable form of a typed boundary.

    ``calibration`` is pack-scoped: ``--pack <name>`` picks the pack whose ``CALIBRATION`` model
    (units in its field descriptions) is emitted.
    """
    # lazy: keeps the other verbs' startup thin
    from strumento.core.schemas import PACK_CONTRACTS, contract_schema

    # an unknown name, or a missing/unknown/misplaced --pack, raises — listing the valid choices
    schema = contract_schema(args.contract, args.pack)
    payload: dict = {"ok": True, "contract": args.contract}
    if args.contract in PACK_CONTRACTS:
        payload["pack"] = args.pack
    payload["schema"] = schema
    if not args.json:
        # text mode prints the bare schema document (pipe it straight to a .schema.json);
        # --json mode omits the copy so an agent doesn't pay for the payload twice.
        payload["_text"] = json.dumps(schema, indent=2)
    return payload


# -- task records --------------------------------------------------------------------------
#
# These verbs make the record format usable by *any* host: a supervisor calls `task create`
# at submit time (it needs a manifest on disk for its queue and listings before any script runs),
# CI reads `task status`, and a human debugging a stuck run reads `task tail`. Same files, same
# derivation, no second source of truth.


def _cmd_task_create(args) -> dict:
    gates = tuple(g.strip() for g in str(args.gates).split(",") if g.strip())
    authored_by = json.loads(args.authored_by) if str(args.authored_by).strip() else None
    if authored_by is not None and not isinstance(authored_by, dict):
        raise ValueError("--authored-by takes a JSON object (profile/model/variant/session_id)")

    rec = record.TaskRecord.create_under(
        args.root,
        title=args.title,
        device=args.device,
        script=args.script,
        workdir=str(args.workdir or Path.cwd()),
        language=args.language,
        kind=args.kind,
        task_type=args.task_type,
        target=args.target,
        gates=gates,
        authored_by=authored_by,
        heartbeat_interval_s=args.heartbeat_interval_s,
        pid=0,            # the supervisor owns `pid`; a fresh manifest is `queued` until it spawns
        heartbeat=False,  # nothing is running yet — no timer thread in a CLI process
    )
    return {
        "ok": True,
        "id": rec.id,
        "dir": str(rec.dir),
        "_text": f"{rec.id}\n{rec.dir}",
    }


def _cmd_task_status(args) -> dict:
    status = record.derive_state(args.dir)
    payload = {"ok": True, **status.to_dict()}
    live = payload["liveness"]
    payload["_text"] = (
        f"{status.id}: {status.state}"
        f" · pid {status.manifest.pid} (alive={live['pid_alive']})"
        f" · {live['n_events']} event(s)"
        f" · {live['age_s']:.1f}s since liveness (stale after {live['stale_after_s']:.0f}s)"
        + (f"\n{status.error}" if status.error else "")
    )
    return payload


def _cmd_task_tail(args) -> dict:
    events = record.read_events(args.dir, limit=max(1, int(args.n)))
    return {
        "ok": True,
        "dir": str(args.dir),
        "events": events,
        "_text": "\n".join(json.dumps(e) for e in events),
    }


def _cmd_task_list(args) -> dict:
    tasks = [
        {
            "id": s.id,
            "dir": s.dir,
            "state": s.state,
            "title": s.manifest.title,
            "device": s.manifest.device,
            "target": s.manifest.target,
            "task_type": s.manifest.task_type,
            "gates": list(s.manifest.gates),
            "created": s.manifest.created,
        }
        for s in record.list_tasks(args.root)
    ]
    text = "\n".join(f"{t['state']:<9} {t['id']}  {t['title']}" for t in tasks) or "(no tasks)"
    return {"ok": True, "root": str(args.root), "tasks": tasks, "_text": text}


# -- the calibration store -----------------------------------------------------------------
#
# Read-only views over `core/store.py` (D9). An **absent** store is not an error: a deviceless or
# never-calibrated device legitimately has no history, and a caller wants an empty view, not a
# failure. A store root is therefore never created by these verbs.


def _open_store(root: Path):
    from strumento.core.store import FileConfigStore

    if not Path(root).is_dir():
        return None
    return FileConfigStore(root)


def _cmd_store_current(args) -> dict:
    store = _open_store(args.root)
    content_id = store.resolve_ref(args.ref) if store else None
    timestamp = ""
    if store and content_id:
        for prov in store.log(args.ref):
            if prov.content_id == content_id:
                timestamp = prov.timestamp or ""
    return {
        "ok": True,
        "root": str(args.root),
        "ref": args.ref,
        "content_id": content_id or "",
        "timestamp": timestamp,
        "_text": f"{args.ref} -> {content_id or '(unset)'}" + (f"  [{timestamp}]" if timestamp else ""),
    }


def provenance_entries(store, ref: str = "main") -> list[dict]:
    """A ref's version history as plain dicts, oldest first (``store.log`` in JSON form).

    Importable for the same reason as :func:`device_summary`: ``strumento store log`` prints this
    and the MCP session's ``calibration_history`` returns it. A ``None`` store (no history yet) is
    an empty list, never an error.
    """
    return [
        {
            "content_id": p.content_id,
            "parent_id": p.parent_id or "",
            "timestamp": p.timestamp or "",
            "source": p.source,          # Provenance has no label field; `source` is what it has
            "updates": p.updates,
            "extra": p.extra or {},       # where the value came from (soc_kind, simulated, fit, …)
        }
        for p in (store.log(ref) if store is not None else [])
    ]


def _cmd_store_log(args) -> dict:
    store = _open_store(args.root)
    entries = provenance_entries(store, args.ref)
    text = "\n".join(
        f"{e['timestamp'] or '-':<28} {e['content_id'][:16]}  {e['source']}" for e in entries
    ) or "(no history)"
    return {"ok": True, "root": str(args.root), "ref": args.ref, "log": entries, "_text": text}


def _cmd_store_diff(args) -> dict:
    store = _open_store(args.root)
    if store is None:
        raise ValueError(f"no calibration store at {args.root}")
    a = flatten_calibration(store.get(args.id_a).calibration)
    b = flatten_calibration(store.get(args.id_b).calibration)
    changed = {k: [a[k], b[k]] for k in sorted(set(a) & set(b)) if a[k] != b[k]}
    added = {k: b[k] for k in sorted(set(b) - set(a))}
    removed = {k: a[k] for k in sorted(set(a) - set(b))}
    text = "\n".join(
        [f"{k}: {old} -> {new}" for k, (old, new) in changed.items()]
        + [f"+ {k} = {v}" for k, v in added.items()]
        + [f"- {k} = {v}" for k, v in removed.items()]
    ) or "(identical calibration)"
    return {
        "ok": True,
        "a": args.id_a,
        "b": args.id_b,
        "changed": changed,
        "added": added,
        "removed": removed,
        "_text": text,
    }


def flatten_calibration(node, prefix: str = "") -> dict:
    """Calibration tree → ``{dotted.path: value}``, using the same path grammar as
    :func:`strumento.core.config.apply_updates` — so a diff path can be fed straight back in.

    That round-trip is the point, and it is why the MCP session publishes this view too: the paths
    an agent reads from ``device_info`` are exactly the paths ``record_calibration`` accepts."""
    out: dict = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten_calibration(value, f"{prefix}{key}."))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            out.update(flatten_calibration(value, f"{prefix}{index}."))
    else:
        out[prefix.rstrip(".")] = node
    return out


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
