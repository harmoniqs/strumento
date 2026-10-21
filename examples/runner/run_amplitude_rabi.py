"""An example run script: one experiment, run safely, on the simulator or on a board.

Copy this file and adapt it to your own experiment. It runs amplitude Rabi on the ge transition
and shows the steps a run script on real hardware should take, each as its own function:

1. ``connect``            — reach the simulator, the mock, or the board over qick's Pyro server.
2. ``config_differences`` — compare the live board description with the device's stored one,
                            and refuse to run when they differ.
3. ``DeviceLock``         — take a lock on the board, so two scripts cannot interleave on it; a
                            lock left by a crashed run on this host is reclaimed.
4. ``preflight``          — compile the program first and read back what the board will
                            actually play, then check it before acquiring anything.
5. ``run``                — acquire with a task record (a durable log of the run) and save the
                            data and the fit to a data chest.
6. ``quality_checks``     — check for a real signal in the data, then judge the fit against
                            stated bounds, not just "the fit converged".
7. ``proposed_edit``      — report the calibration change it suggests. It never writes it: you
                            review the run and apply the edit yourself.

Modes::

    python run_amplitude_rabi.py --device my_fridge/device.yaml --sim     # simulator (default)
    python run_amplitude_rabi.py --device my_fridge/device.yaml --mock    # structure only, zero IQ
    python run_amplitude_rabi.py --device my_fridge/device.yaml --check --ns-host 192.168.1.10
    python run_amplitude_rabi.py --device my_fridge/device.yaml --arm   --ns-host 192.168.1.10

``--check`` connects to the board, compares its description and runs the preflight without
acquiring. ``--arm`` does the same and then acquires. Every run writes a folder under
``--out`` (default ``runs/``) holding the task record, the data chest and a ``summary.json``.

Exit codes: 0 passed, 1 a quality check failed, 2 refused before acquiring (configuration
mismatch, active lock, or a failed preflight).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from strumento.core.datachest import FileDataChest
from strumento.core.device import Device
from strumento.core.record import TaskRecord, pid_is_alive
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi
from strumento.packs.cqed.safety import MAX_QUBIT_DRIVE_GAIN

# --- the plan: what this script measures -----------------------------------------------------
# Edit these for your device. A scan must contain its answer, and its top must stay under the
# drive ceiling (MAX_QUBIT_DRIVE_GAIN, a fraction of full scale; the preflight enforces it).
GAIN_STOP = 16000       # DAC integer code at the top of the amplitude scan
POINTS = 81             # points in the scan
REPS = 2000             # averages per point
STDERR_MAX = 0.05       # largest acceptable relative stderr on the fitted π gain
EDGE_MARGIN = 0.05      # the fitted π gain must sit this fraction of the span inside the scan
SIGNAL_MIN_RATIO = 10.0 # the trace's peak-to-peak range must exceed its point-to-point noise this many times

EXIT_PASSED, EXIT_QUALITY_FAILED, EXIT_REFUSED = 0, 1, 2


class Refused(RuntimeError):
    """The run stopped before acquiring anything."""


# --- 1. connect ------------------------------------------------------------------------------


@dataclass
class Connection:
    """What a run talks to: ``soc`` acquires, ``soccfg`` compiles, ``release`` cleans up."""

    mode: str
    soc: Any
    soccfg: Any
    live_cfg: dict | None = None
    release: Any = field(default=lambda: None)


def connect(mode: str, device: Any, *, ns_host: str = "", ns_port: int = 8888,
            proxy_name: str = "myqick", truth: dict | None = None, seed: int = 1) -> Connection:
    """Open the simulator, the mock, or a Pyro connection to the board."""
    if mode == "sim":
        from strumento.core.simulator import SimulatorSoc

        sim = SimulatorSoc.from_device(device, seed=seed, truth=truth)
        return Connection(mode, sim, sim)
    if mode == "mock":
        from strumento.core.mock import MockQickSocV2

        mock = MockQickSocV2(device.soccfg_snapshot)
        return Connection(mode, mock, mock)
    from qick.pyro import make_proxy

    soc, soccfg = make_proxy(ns_host=ns_host, ns_port=ns_port, proxy_name=proxy_name)
    # qick returns some fields as tuples; the stored file holds lists. Round-trip to compare.
    live = json.loads(json.dumps(soccfg.get_cfg()))
    return Connection(mode, soc, soccfg, live_cfg=live, release=soc._pyroRelease)


# --- 2. compare the live board with the stored description -----------------------------------


def config_differences(live: Any, stored: Any, path: str = "") -> list[str]:
    """Every field path where two board descriptions disagree."""
    if isinstance(live, dict) and isinstance(stored, dict):
        out: list[str] = []
        for key in sorted(set(live) | set(stored)):
            out += config_differences(live.get(key), stored.get(key), f"{path}.{key}" if path else key)
        return out
    if isinstance(live, list) and isinstance(stored, list) and len(live) == len(stored):
        out = []
        for i, (a, b) in enumerate(zip(live, stored)):
            out += config_differences(a, b, f"{path}[{i}]")
        return out
    return [] if live == stored else [path]


# --- 3. lock the board -----------------------------------------------------------------------


class DeviceLock:
    """An exclusive lock file for one board, released on exit.

    The file records the owning process and host. A lock whose owner process is gone (a run that
    crashed on this host) is reclaimed with a message; a live lock, or one held from another
    host, refuses. Every script that talks to the board must use the same lock path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __enter__(self) -> DeviceLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname()
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                held = self._read()
                if held.get("host") == host and not pid_is_alive(int(held.get("pid", 0))):
                    print(f"reclaiming a lock left by a process that no longer runs: {held}")
                    self.path.unlink(missing_ok=True)
                    continue
                raise Refused(f"the board is locked: {self.path} held by {held}") from None
            with os.fdopen(fd, "w") as f:
                json.dump({"pid": os.getpid(), "host": host, "since": time.time()}, f)
            return self
        raise Refused(f"could not take the lock {self.path}")

    def __exit__(self, *exc: Any) -> None:
        if self._read().get("pid") == os.getpid():
            self.path.unlink(missing_ok=True)

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}


# --- 4. build and preflight ------------------------------------------------------------------


def build(device: Any, conn: Connection) -> AmplitudeRabi:
    """The experiment, compiled against the board's own configuration when one is connected."""
    runner = None
    if conn.mode in ("check", "arm"):
        from strumento.core.legacy_pyro_runner import LegacyPyroRunner

        runner = LegacyPyroRunner(conn.soccfg)
    return AmplitudeRabi(device, gains=Sweep(start=0, stop=GAIN_STOP, on="amp"), points=POINTS,
                         runner=runner)


def drive_full_scale(device: Any) -> float:
    """The full-scale code of the generator that plays the qubit drive."""
    from strumento.core.wiring import LineRef

    generator = device.wiring.resolve(LineRef(device.qubit.element.name, "drive"),
                                      float(device.calib.qubit.f_ge))
    return float(next(g["maxv"] for g in device.soccfg_snapshot["gens"]
                      if g["tproc_ch"] == generator.dac_ch))


def preflight(device: Any, experiment: AmplitudeRabi, conn: Connection) -> dict[str, bool]:
    """Compile once, read back what the board will play, and check it before any acquisition."""
    from strumento.core.program import StrumentoProgram
    from strumento.core.pulses import Pulse

    seq, sweeps = experiment.sequence()
    program = StrumentoProgram(device, seq=seq, sweeps=sweeps, axes=experiment.axes(), reps=REPS,
                               soc=conn.soccfg)
    drive = next(op for op in seq.ops if isinstance(op, Pulse) and op.label == "pi_ge")
    gains = np.atleast_1d(np.asarray(program.realized(drive).gain_frac, dtype=float))
    codes = np.rint(gains * drive_full_scale(device))
    return {
        "scan starts at zero": bool(codes[0] == 0),
        "scan reaches the requested top": bool(codes[-1] == GAIN_STOP),
        "scan is monotonic": bool(np.all(np.diff(codes) >= 0)),
        "scan stays under the drive ceiling": float(gains.max()) <= MAX_QUBIT_DRIVE_GAIN,
    }


# --- 5. run, 6. judge, 7. propose ------------------------------------------------------------


def signal_to_noise(experiment: AmplitudeRabi, result: Any) -> float:
    """The measured trace's peak-to-peak range over its point-to-point noise.

    Computed from the data, not from the fit: a fit's error bars are undefined when there is no
    signal, and what "undefined" turns into depends on the machine. The noise estimate (the spread
    of neighbouring differences) includes a little of the signal's own slope, so the ratio is
    conservative. A trace with no variation at all (the mock's zeros) scores 0.
    """
    iq = np.stack([np.asarray(result.avgi, dtype=float), np.asarray(result.avgq, dtype=float)], axis=-1)
    y = np.ravel(experiment.readout.project(iq))
    noise = float(np.std(np.diff(y)) / np.sqrt(2.0))
    span = float(np.ptp(y))
    return span / noise if noise > 0 else (float("inf") if span > 0 else 0.0)


def quality_checks(experiment: AmplitudeRabi, outcome: Any) -> dict[str, bool]:
    """A real signal first; only then is the fit worth judging.

    "The fit converged" alone proves nothing: on data with no signal at all (all zeros), the Rabi
    fitter still reports success, with its parameters left at their starting guesses — and the π
    gain then lands mid-scan. So the signal is checked on the data, before the fit is trusted.
    """
    if not signal_to_noise(experiment, outcome.result) >= SIGNAL_MIN_RATIO:
        return {"a real signal was measured": False}
    fit = outcome.fit
    if fit is None or not fit.success:
        return {"a real signal was measured": True, "fit converged": False}
    pi_gain = float(fit.derived["pi_gain"])
    stderr = float(fit.derived.get("pi_gain_stderr", float("nan")))
    margin = EDGE_MARGIN * GAIN_STOP
    return {
        "a real signal was measured": True,
        "fit converged": True,
        "π gain inside the scan": margin <= pi_gain <= GAIN_STOP - margin,
        "π gain precise enough": bool(np.isfinite(stderr) and stderr / pi_gain <= STDERR_MAX),
    }


def proposed_edit(device: Any, fit: Any) -> dict:
    """The calibration change this run suggests. Review it, then apply it yourself."""
    old = int(device.calib.qubit.pulses["pi_ge"].gain)
    new = int(round(float(fit.derived["pi_gain"])))
    return {"field": "calibration.qubit.pulses.pi_ge.gain", "old": old, "new": new}


def run(device_path: str | Path, mode: str = "sim", *, out: str | Path = "runs",
        lock_path: str | Path = "runs/board.lock", ns_host: str = "", ns_port: int = 8888,
        proxy_name: str = "myqick", truth: dict | None = None, seed: int = 1) -> int:
    """One run, start to finish. Returns the exit code; ``summary.json`` records why."""
    device = Device.load(device_path)
    conn = connect(mode, device, ns_host=ns_host, ns_port=ns_port, proxy_name=proxy_name,
                   truth=truth, seed=seed)
    summary: dict[str, Any] = {"experiment": "amplitude_rabi", "mode": mode,
                               "device": str(device_path), "reps": REPS}
    record = None
    try:
        with DeviceLock(lock_path):
            if conn.live_cfg is not None:
                differences = config_differences(conn.live_cfg, device.soccfg_snapshot)
                summary["config_differences"] = differences
                if differences:
                    raise Refused(f"the live board differs from the stored description: {differences}")
            experiment = build(device, conn)
            checks = preflight(device, experiment, conn)
            summary["preflight"] = checks
            for label, passed in checks.items():
                print(f"[{'PASS' if passed else 'FAIL'}] {label}")
            if not all(checks.values()):
                raise Refused("the preflight failed")
            if mode == "check":
                summary["outcome"] = "checked; nothing acquired"
                return EXIT_PASSED
            record = TaskRecord.create_under(Path(out), title="amplitude rabi", device=str(device_path),
                                             target="hw" if mode == "arm" else "sim")
            summary["task"] = record.dir.name
            outcome = experiment.go(conn.soc, reps=REPS, progress=False, record=record)
            fit = outcome.fit
            entry = FileDataChest(Path(out) / "data").save(
                outcome.result, seq=experiment.sequence()[0], experiment="amplitude_rabi",
                fit={"success": bool(fit.success), "derived": dict(fit.derived)} if fit else None,
                task_id=record.dir.name)
            summary["data_entry"] = entry
            quality = quality_checks(experiment, outcome)
            summary["quality"] = quality
            for label, passed in quality.items():
                print(f"[{'PASS' if passed else 'FAIL'}] {label}")
            if fit is None or not all(quality.values()):
                # A failed quality check is a finished run with a bad result, not a script error.
                record.fail("a quality check failed", kind="", summary=json.dumps(quality))
                summary["outcome"] = "quality check failed"
                return EXIT_QUALITY_FAILED
            edit = proposed_edit(device, fit)
            summary["proposed_edit"] = edit
            print(f"proposed edit (not applied): {edit['field']}: {edit['old']} -> {edit['new']}")
            record.done(summary=f"pi_gain = {fit.derived['pi_gain']:.0f}")
            summary["outcome"] = "passed"
            return EXIT_PASSED
    except Refused as why:
        print(f"refused: {why}")
        summary["outcome"] = f"refused: {why}"
        return EXIT_REFUSED
    except BaseException as error:
        if record is not None:
            record.fail(f"{type(error).__name__}: {error}")
        summary["outcome"] = f"error: {type(error).__name__}: {error}"
        raise
    finally:
        conn.release()
        folder = record.dir if record is not None else Path(out)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", required=True, help="the device YAML")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sim", dest="mode", action="store_const", const="sim")
    group.add_argument("--mock", dest="mode", action="store_const", const="mock")
    group.add_argument("--check", dest="mode", action="store_const", const="check")
    group.add_argument("--arm", dest="mode", action="store_const", const="arm")
    parser.add_argument("--ns-host", default="", help="the Pyro name server (--check, --arm)")
    parser.add_argument("--ns-port", type=int, default=8888)
    parser.add_argument("--proxy-name", default="myqick")
    parser.add_argument("--out", default="runs", help="where run folders are written")
    parser.add_argument("--lock", default="runs/board.lock", help="the board's lock file")
    args = parser.parse_args(argv)
    mode = args.mode or "sim"
    if mode in ("check", "arm") and not args.ns_host:
        parser.error("--check and --arm need --ns-host")
    return run(args.device, mode, out=args.out, lock_path=args.lock, ns_host=args.ns_host,
               ns_port=args.ns_port, proxy_name=args.proxy_name)


if __name__ == "__main__":
    sys.exit(main())
