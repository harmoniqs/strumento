"""Time-of-flight bring-up routine — a host-driven single-shot decimated acquisition (#104).

Finds the real round-trip delay between playing the readout probe and the reflected signal
actually arriving at the ADC, by playing a long probe and acquiring **decimated** (the raw
trace, not the averaged I/Q an ordinary ``.measure()`` returns) over a window long enough to
span "before signal" through "signal settled" — the debugging-session recipe this formalizes:
play a pulse, ``acquire_decimated()``, find where the signal actually shows up in the raw trace,
done by hand during real-hardware bring-up.

Host-driven — standalone ``run()``/``analyze()``, **not** an ``Experiment`` subclass, **not**
registered in any pack's ``EXPERIMENTS`` — the "host-driven, standalone" archetype
``docs/architecture.md``'s "Three experiment archetypes" names (``ChargeStability``,
``PinchOff``, … — the DC-sweep shape of it), generalized here: this routine plays a compiled
:class:`~strumento.core.program.StrumentoProgram` directly rather than stepping an external
``DCSource`` + ``probe()`` callable, because there is no DC axis to sweep — it plays one probe,
once, and reads the raw trace.

**Why the probe/trigger can't ride the normal, calibrated ``.measure()`` path.** The compiled
``Measure`` op triggers ``trig_offset`` µs *after* the reference cursor — and for a sequential
``.play(probe).measure()`` the cursor has already advanced past the whole probe pulse by then
(``docs/timeline.md``'s cursor model), so the ADC would only start listening once the drive has
already finished. Relying on the device's *current* ``calib.readout.trig_offset`` here would
also be circular — that is the value this routine measures. So :meth:`TimeOfFlight.run`:

1. builds the probe pulse with ``start_us=0.0`` (absolute placement — it does **not** advance
   the cursor), so the subsequent ``.measure()`` trigger fires at the cursor's *unmoved* position
   instead of after the probe;
2. pins the resolved trig_offset to ``pre_trigger_us`` (default ``0.0``) and widens the declared
   decimated window to ``window_us``, for the duration of this one acquisition only — both live
   on an internal, throwaway device clone (:func:`~strumento.core.config.apply_updates` +
   :meth:`~strumento.core.device.Device.from_config`, the same pattern any calibration
   write-back uses). ``self.device`` (the caller's) is never mutated by :meth:`run`.

The result: the ADC listens from (approximately) when the probe starts driving, for the whole
configured window, so the raw trace shows noise floor -> arrival -> settled plateau — exactly
the shape :func:`~strumento.core.fitters.fit_time_of_flight` looks for.

Pure at import: :class:`~strumento.core.program.StrumentoProgram` (qick-facing) is a local
import inside :meth:`run`, exactly like :class:`~strumento.core.experiment.Runner` (hard rule 1)
— importing this module never requires qick.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np

from strumento.core.config import apply_updates
from strumento.core.device import Device
from strumento.core.fitters import FitResult, fit_time_of_flight
from strumento.core.result import MeasurementResult
from strumento.core.wiring import LineRef

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps this module qick-free at import
    from strumento.core.store import ConfigStore

#: Default decimated acquisition window (µs) — long enough to show a noise floor, an arrival
#: edge, and a settled plateau for a typical cable round-trip, comfortably under the readout
#: buffer's buf_maxlen at the shipped fixtures' decimation rate (a few hundred samples/µs).
DEFAULT_WINDOW_US = 3.0


class TimeOfFlight:
    """Play a long readout probe and acquire decimated to find the real trigger delay.

    Not an ``Experiment`` subclass; not registered in any pack's ``EXPERIMENTS`` (see the
    module docstring for why, and for the ``start_us``/pinned-``trig_offset`` mechanics).

    Parameters
    ----------
    device : Device
        The device whose readout probe (calibrated frequency/gain by default) is played.
        Never mutated by :meth:`run` — the trig_offset/window overrides live on an internal
        clone. :meth:`write_back` *does* update ``self.device`` (see its docstring).
    window_us : float
        The decimated acquisition window (µs) — must span "before signal" through "signal
        settled" for the fitter to see a real edge. Also becomes the probe pulse's own length,
        so the drive keeps playing for the whole window (one rising step, not a finite pulse).
    pre_trigger_us : float
        The trig_offset pinned for this one acquisition (default ``0.0``: the ADC listens from
        when the probe starts, not from the device's current — and possibly wrong — belief).
    freq, gain, line : see :meth:`~strumento.packs.cqed.library.readout.ReadoutOps.probe`
        Forwarded to the probe factory; ``None`` uses the device's calibrated defaults.
    reps : int
        Hardware reps for the decimated acquisition (default ``1`` — a single-shot raw trace).
        The buffer size scales with ``reps`` (unlike software ``rounds``); leave at 1 unless you
        know the buffer has room, and note that ``reps > 1`` changes :meth:`run`'s result shape
        (an extra leading rep axis on ``avgi``/``avgq``, ``xpts`` still one window's time axis).
    rounds : int
        Software rounds ``acquire_decimated`` averages over (default ``1``); does not affect
        the buffer size.
    """

    def __init__(
        self,
        device: Device,
        *,
        window_us: float = DEFAULT_WINDOW_US,
        pre_trigger_us: float = 0.0,
        freq: float | None = None,
        gain: float | None = None,
        line: tuple[str, str] | LineRef | None = None,
        reps: int = 1,
        rounds: int = 1,
    ) -> None:
        self.device = device
        self.window_us = float(window_us)
        self.pre_trigger_us = float(pre_trigger_us)
        self.freq = freq
        self.gain = gain
        self.line = line
        self.reps = int(reps)
        self.rounds = int(rounds)

    # -- run: play the probe, acquire decimated (AC1) -----------------------------------

    def _probe_device(self) -> Device:
        """A throwaway clone of ``self.device`` with the readout trigger pinned to
        ``pre_trigger_us`` and the declared decimated window widened to ``window_us`` — the two
        knobs a normal ``.measure()`` resolves from calibration/wiring, and that this bring-up
        acquisition needs to control directly instead (see the module docstring).
        """
        config, _diff = apply_updates(
            self.device.to_config(), {"readout.trig_offset": self.pre_trigger_us}
        )
        wiring = copy.deepcopy(config.wiring)
        wiring["readout"]["lengths_us"] = self.window_us
        config = dataclasses.replace(config, wiring=wiring)
        return Device.from_config(config)

    def run(self, soc, *, compile_soccfg=None, progress: bool = True, trace: bool = False) -> MeasurementResult:
        """Play the probe and acquire decimated over ``window_us`` — the raw trace, not the
        averaged I/Q an ordinary ``.measure()`` returns.

        Returns a :class:`MeasurementResult` whose ``xpts`` is the realized decimated time axis
        (µs, read back from the compiled program via qick's own ``get_time_axis`` — the same
        register-rounded-truth discipline every other realized axis in this codebase follows,
        never a hand-computed tick rate) and whose ``avgi``/``avgq`` are the raw (I, Q) trace
        for the one declared readout channel. ``trace=True`` additionally compiles with a
        :class:`~strumento.core.tracing.CompilerTrace` and stamps it on ``meta["trace"]`` — the
        way to confirm the probe and the trigger really do land at the same reference time
        (see the module docstring) rather than the trigger firing after the whole probe plays.

        Pass ``compile_soccfg`` when ``soc`` is a remote proxy: assembly reads the local
        configuration while ``acquire_decimated()`` alone uses the runtime transport.
        """
        from strumento.core.program import StrumentoProgram
        from strumento.core.pulses import Seq

        probe_device = self._probe_device()
        probe = dataclasses.replace(
            probe_device.readout.probe(  # type: ignore[attr-defined]  # ops are bound onto Device at load
                freq=self.freq, gain=self.gain, length=self.window_us, line=self.line
            ),
            # Absolute placement (start_us set, even to 0.0) does NOT advance the cursor
            # (docs/timeline.md) — so the measure() below fires at (cursor + pre_trigger_us)
            # with the cursor still at the probe's own start, not after the whole probe plays.
            start_us=0.0,
        )
        seq = Seq().play(probe).measure()

        prog = StrumentoProgram(
            probe_device,
            seq=seq,
            reps=self.reps,
            final_delay=None,
            soc=compile_soccfg if compile_soccfg is not None else soc,
            trace=trace,
        )
        raw = prog.acquire_decimated(soc, rounds=self.rounds, progress=progress)
        xpts = prog.get_time_axis(0)
        iq = np.asarray(raw[0])
        meta: dict[str, Any] = {
            "window_us": self.window_us,
            "pre_trigger_us": self.pre_trigger_us,
            "reps": self.reps,
            "rounds": self.rounds,
        }
        if trace:
            meta["trace"] = prog.trace()
        return MeasurementResult(xpts=xpts, avgi=iq[..., 0], avgq=iq[..., 1], raw=raw, meta=meta)

    # -- analyze: the arrival edge (AC2) -------------------------------------------------

    def analyze(self, result: MeasurementResult) -> FitResult:
        """Fit the decimated trace's magnitude to the arrival edge.

        ``derived["tof_us"]`` (:func:`~strumento.core.fitters.fit_time_of_flight`'s headline
        quantity) is the measured round-trip delay — the value :meth:`write_back` stores into
        ``calib.readout.trig_offset``.
        """
        avgi = np.asarray(result.avgi, dtype=float)
        avgq = np.asarray(result.avgq, dtype=float)
        magnitude = np.hypot(avgi, avgq)
        return fit_time_of_flight(np.asarray(result.xpts, dtype=float), magnitude)

    # -- write-back: calib.readout.trig_offset, reload-after-write (AC4, optional) ------

    def write_back(
        self,
        fit: FitResult,
        store: ConfigStore,
        ref: str,
        *,
        source: str | None = None,
        timestamp: str | None = None,
    ) -> Device:
        """Store the measured ``tof_us`` into ``calib.readout.trig_offset``, and reload.

        Follows the existing measure -> believe calibration-store pattern exactly:
        :func:`~strumento.core.store.record_calibration` snapshots an immutable version from
        ``self.device``'s *current* config, advances ``ref``, and this then reloads
        ``self.device`` from that new config — the same reload-after-write contract
        ``LabSession.record_calibration`` (``strumento/mcp.py``) carries: skip the reload and a
        caller would keep measuring against the belief it just replaced. Returns the reloaded
        device (also left on ``self.device``, so a subsequent :meth:`run` plays from it).

        Optional by construction: nothing calls this for you — a caller who only wants the
        measured value calls :meth:`analyze` and stops there.
        """
        from strumento.core.store import record_calibration

        tof_us = float(fit.derived["tof_us"])
        _content_id, config = record_calibration(
            store,
            ref,
            self.device.to_config(),
            {"readout.trig_offset": tof_us},
            source=source or "time_of_flight fit",
            timestamp=timestamp,
            extra={"fit": dict(fit.derived)},
        )
        self.device = Device.from_config(config)
        return self.device
