# Write your own runner

An experiment in strumento is a recipe: a sequence, the sweep it runs over, and the analysis that
fits the result. `AmplitudeRabi(dev).go(soc)` runs it once. On a real board, a run needs more than
that call: a check that the board is the board your device file describes, exclusive access, a
look at what will actually be played before anything is played, a record of what happened, and a
judgement of the result that is stricter than "the fit converged". A **runner** is the script that
does those things around the experiment.

This guide builds one, step by step. The finished script is
[`examples/runner/run_amplitude_rabi.py`](../examples/runner/run_amplitude_rabi.py): copy it and
adapt it to your experiment. Every Python block below runs in the docs test against the shipped
demo device, so the output shown is what the script really prints.

If your board is new, work through [board-bringup.md](board-bringup.md) first.

## 1. Run it

The script runs amplitude Rabi on the qubit's ge transition. On the simulator:

```python
import sys

sys.path.insert(0, "examples/runner")
import run_amplitude_rabi as runner

device = "devices/multimode_demo/device.yaml"
code = runner.run(device, "sim", out="runs", lock_path="runs/board.lock")
print("exit code:", code)
```

<!-- snippet: output -->
```text
[PASS] scan starts at zero
[PASS] scan reaches the requested top
[PASS] scan is monotonic
[PASS] scan stays under the drive ceiling
[PASS] a real signal was measured
[PASS] fit converged
[PASS] π gain inside the scan
[PASS] π gain precise enough
proposed edit (not applied): calibration.qubit.pulses.pi_ge.gain: 8192 -> 8197
exit code: 0
```

From a shell, the same run is:

```bash
python examples/runner/run_amplitude_rabi.py --device devices/multimode_demo/device.yaml --sim
```

The four modes are `--sim` (the simulator, the default), `--mock` (a board stand-in that returns
zeros: it proves the program builds and nothing more), `--check` (connect to the board, compare
and preflight, acquire nothing) and `--arm` (connect, check, then acquire). `--check` and `--arm`
take the Pyro name server's address (`--ns-host`). The exit code is 0 when the run passed, 1 when
a quality check failed, and 2 when the script refused before acquiring.

The rest of this guide walks through what happened in that run.

## 2. Connect

`connect()` returns three things: `soc`, which acquires; `soccfg`, which programs are compiled
against; and `release`, which closes the connection. On the simulator and the mock, one object is
both. Over Pyro, `make_proxy` returns the two separately, and the script compiles on your PC
against the board's own configuration (the `LegacyPyroRunner`) and acquires through the proxy.

`release` runs in a `finally` block, so the connection closes whatever happens. That matters with
Pyro: if its worker thread dies, every later call in the session fails until you reconnect.

## 3. Compare the live board with the stored description

On a board, the first thing the script does is compare the board's live description with the
one your device file points at, field by field, and refuse to run if they differ. Every gain,
frequency and timing strumento computes comes from the stored description, so a firmware or
clock change since you recorded it would otherwise go unnoticed.

Here the simulator stands in for a board whose firmware changed one generator's sample rate since
the description was recorded:

```python
import copy

from strumento.core.device import Device

dev = Device.load(device)
live = copy.deepcopy(dev.soccfg_snapshot)
live["gens"][2]["fs"] = 6881.28
print(runner.config_differences(live, dev.soccfg_snapshot))
```

<!-- snippet: output -->
```text
['gens[2].fs']
```

With that difference, `run()` stops before acquiring and returns 2:

```python
real_connect = runner.connect

def board_with_changed_firmware(mode, device, **kwargs):
    conn = real_connect("sim", device, **kwargs)
    changed = copy.deepcopy(device.soccfg_snapshot)
    changed["gens"][2]["fs"] = 6881.28
    return runner.Connection("sim", conn.soc, conn.soccfg, live_cfg=changed)

runner.connect = board_with_changed_firmware
try:
    code = runner.run(device, "sim", out="runs/changed", lock_path="runs/board.lock")
finally:
    runner.connect = real_connect
print("exit code:", code)
```

<!-- snippet: output -->
```text
refused: the live board differs from the stored description: ['gens[2].fs']
exit code: 2
```

On a mismatch, re-record the board description, run `strumento check` again, and review any
calibration that depends on the changed fields.

## 4. Lock the board

Two scripts acquiring on one board at once interleave their programs and corrupt both results.
`DeviceLock` holds a lock file for the whole run. It records the owner's process ID and host, and
it is released on exit, including when the run fails.

If a run crashes hard (a killed process, a segfault), its lock stays behind. The lock checks
whether the process that holds it is still running on this host; if it is not, the lock is
reclaimed, with a message. A lock held by a live process, or by another host, refuses:

```python
import json
import os
import socket
import time

os.makedirs("runs", exist_ok=True)
with open("runs/held.lock", "w") as f:
    json.dump({"pid": os.getpid(), "host": socket.gethostname(), "since": time.time()}, f)
try:
    with runner.DeviceLock("runs/held.lock"):
        pass
except runner.Refused as why:
    print("refused:", str(why).split(" held by")[0])
```

<!-- snippet: output -->
```text
refused: the board is locked: runs/held.lock
```

Every script that talks to the board must use the **same lock path**, or the locks protect
nothing. Pass it with `--lock`.

## 5. Preflight: look before you play

The script compiles the program before acquiring and reads back what the board will actually play
(`program.realized`, the values after the board's own rounding), then checks them. For amplitude
Rabi: the gain scan starts at zero, reaches the requested top, increases monotonically, and stays
under the drive ceiling. Every check must pass, or nothing is acquired.

Each experiment needs its own preflight. Good checks state what *must* be true for the
measurement to mean anything: a frequency scan centred on the current calibration, a time sweep
that respects the qubit's coherence, a readout that fits the ADC buffer.

The package refuses unsafe gains on its own as well: a scan whose top exceeds the drive ceiling
is refused when the experiment is built, before any program exists.

## 6. Run with a record

`go()` is called with a **task record**, a folder holding a manifest, an append-only event log
and a result, so a run leaves a durable trace even if the script dies halfway. The data and the
fit go to a **data chest**, from which the run can be reloaded and refitted later. Each run gets
its own folder under `--out`:

```python
import glob

folder = sorted(glob.glob("runs/*-amplitude-rabi-*"))[0]
print(sorted(os.listdir(folder)))
```

<!-- snippet: output -->
```text
['artifacts', 'progress.jsonl', 'result.toml', 'summary.json', 'task.toml']
```

`summary.json` records every check, the outcome and the proposed edit. The event log holds the
package's own verdict on the fit (a `fit` gate). Your script's quality checks are *not* written
as gates: a verdict in the record should come from shipped code, not from the script being judged
([task-record.md](task-record.md) explains why).

## 7. Judge the fit

"The fit converged" is not evidence of a measurement. The mock returns zeros, and the Rabi fitter
still reports success on them, with its parameters left at their starting guesses; the π gain
then lands mid-scan, which looks plausible. Its error bars are no help either: with no signal the
fit is singular, and its standard errors come out undefined on one machine and as arbitrary
finite numbers on another.

So the script checks for a signal **in the data** before it trusts the fit: the trace's
peak-to-peak range must exceed its point-to-point noise at least ten times. On the simulator's
Rabi trace that ratio is about 47; on the mock's zeros it is 0, on every machine. Only when there
is a signal are the fit's checks evaluated:

```python
code = runner.run(device, "mock", out="runs/mock", lock_path="runs/board.lock")
print("exit code:", code)
```

<!-- snippet: output -->
```text
[PASS] scan starts at zero
[PASS] scan reaches the requested top
[PASS] scan is monotonic
[PASS] scan stays under the drive ceiling
[FAIL] a real signal was measured
exit code: 1
```

Write checks that would fail on a bad run of *your* experiment: a signal clearly above the noise,
judged from the data rather than from the fit's own error bars, the fitted value well inside the
scan (an answer at the edge of a scan is a scan that missed),
precision good enough to act on, and any physics bound that applies (for example, a Ramsey
detuning beyond the fringes' sampling limit is aliased and must be refused).

## 8. Propose, never write

The script reports the calibration change the run suggests (`8192 -> 8197` above) and stops. It
never edits the device file. Review the run first, then apply the edit by hand or through the
calibration store ([calibration-store.md](calibration-store.md)), and run `strumento check` again.

## 9. Adapt it to your experiment

The script is about 330 lines, most of them the reusable steps. For another experiment, change:

- **the plan constants** at the top: scan range, points, repetitions, quality thresholds;
- **`build()`**: construct your experiment instead of `AmplitudeRabi`;
- **`preflight()`**: the checks that must hold before *your* experiment plays;
- **`quality_checks()`** and **`proposed_edit()`**: what a good result looks like, and which
  calibration field it updates.

Keep `connect`, `config_differences`, `DeviceLock` and the structure of `run()` as they are:
they are what make a run safe on hardware. The experiments the cqed pack ships, with their
arguments, are listed in [cqed-cookbook.md](cqed-cookbook.md).

Two limits of this script are worth knowing:

- **It runs one program per run.** Protocols that need several programs, such as randomized
  benchmarking over many sequence lengths or a two-dimensional chevron map, need a loop around
  the acquire step with one record entry per program.
- **It reads averaged data.** Single-shot readout calibration needs per-shot data, which the
  averaged hardware path does not return.
