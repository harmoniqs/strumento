"""The ``strumento`` command-line interface — agent- and automation-forward.

Every command is importable (``from strumento.cli import main; main([...])``), prints
human-readable text by default, and machine-readable JSON with ``--json`` — so both a person at
a terminal and an agent in CI get first-class output. Only ``compile``'s realized-values step
touches ``qick`` (and degrades gracefully without it); everything else is pure.

An MCP wrapper, if ever wanted, is a thin shim over the same functions — the CLI was chosen as
the primary surface because agents execute CLIs natively everywhere, including headless CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np


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
    _emit(payload, args.json)
    return 0


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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="strumento", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(title="commands")

    def cmd(name, fn, help_):
        c = sub.add_parser(name, help=help_)
        c.add_argument("--json", action="store_true", help="machine-readable JSON output")
        c.set_defaults(fn=fn, json=False)
        return c

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


def _cmd_info(args) -> dict:
    dev = _load_device(args.device_yaml)
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
    info = {
        "ok": True,
        "pack": dev.pack.__name__,
        "elements": {n: type(e).__name__ for n, e in sorted(dev.elements.items())},
        "namespaces": namespaces,
        "wiring": wiring,
        "readout": {"dac_ch": dev.wiring.readout.dac_ch, "adc_chs": dev.wiring.readout.adc_chs},
        "couplings": [f"{c.a}—{c.b} ({c.form})" for c in dev.couplings],
    }
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
    cols = list(data.dtype.names)
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
    from strumento.core.sweeps import Sweep

    dev = _load_device(args.device_yaml)
    if not hasattr(dev, "qubit"):
        raise ValueError("compile demo expects an instance with a transmon named 'qubit'")
    seq = Seq().play(dev.qubit.ge_pi(gain=Sweep("amp", 0, 30000, 51))).measure()

    prog = _RecordingProgram()
    c = Compiler(prog, dev.wiring)
    c.declare(seq)
    c.play(seq)
    calls = [
        {"call": name, "args": list(a), "kwargs": {k: v for k, v in kw.items()}}
        for name, a, kw in prog.calls
    ]

    realized = None
    try:  # optional: DAC-grid realized values need qick
        from qick import QickConfig

        from strumento.core.compiler import quantize

        r = quantize(dev.qubit.ge_pi(), QickConfig(dev.soccfg_snapshot), dev.wiring)
        realized = {
            "freq_mhz_actual": r.freq_mhz_actual,
            "gain_frac": r.gain_frac,
            "cycles": r.cycles,
        }
    except ImportError:
        pass

    text = ["amplitude-Rabi lowering (canned demo):"]
    text += [f"  {c_['call']}({', '.join(f'{k}={v}' for k, v in c_['kwargs'].items())})" for c_ in calls]
    if realized:
        text.append(f"realized ge_pi: {realized}")
    return {"ok": True, "calls": calls, "realized": realized, "_text": "\n".join(text)}


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
