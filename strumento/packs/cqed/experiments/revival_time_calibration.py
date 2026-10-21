"""Revival time calibration experiment.

Precisely calibrate the parity revival time (pi/chi) by displacing the cavity into a small
coherent state, then sweeping a qubit Ramsey delay and fitting the photon-induced oscillation.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_decaysin_detuned
from strumento.core.pulses import Gaussian, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class RevivalTimeCalibration(Experiment):
    """Calibrate the parity revival time (pi/chi) from a photon-populated qubit Ramsey.

    Sequence: ``displace(mode)`` -- ``ge_hpi`` -- wait(t) -- ``ge_hpi(phase=180)`` -- measure,
    with the delay ``t`` swept.  A small cavity displacement is REQUIRED: in vacuum the qubit
    carries no photon-number-dependent shift and the Ramsey does not oscillate at chi (it would
    report only the qubit-drive detuning, or a degenerate zero).  With a small coherent state the
    single-photon component makes the fringe oscillate at the dispersive shift chi; its half-period
    is the parity revival time pi/chi.
    Fitter: ``fit_decaysin_detuned`` (damped cosine), plus a post-fit derivation
    of ``revival_time`` from the fitted frequency as ``0.5 / freq``.

    The displacement is a raw, *uncalibrated* nominal drive on the mode line (it does NOT call
    ``ModeOps.displace``), so this experiment still reads no ``calib.modes`` value and remains the
    root of the parity chain — it only needs the mode's frequency (from the topology / a prior
    ``CavitySpectroscopy``).  Tune ``displace_gain`` so the mean photon number is small (n̄ ≲ 1):
    the fitted frequency approaches chi only when the n=1 Fock component dominates the oscillation;
    a large coherent state precesses at ~n̄·chi instead.  ``revival_time = 0.5/chi`` here is the
    bare dispersive value; the small AC-Stark correction to the *operating* revival is applied
    separately by the parity composite.

    Derived quantities:
        revival_time : calibrated parity revival time pi/chi (us), the headline quantity
        detuning     : oscillation frequency from the damped-cosine fit (cycles per x-unit)
        freq         : same as detuning
        T2           : decay envelope time constant (us)
        tau          : same as T2

    Parameters:
        delays        : delay sweep; defaults to 0-1000 us on axis ``"t"``
        points        : number of sweep points (default 101)
        mode_name     : device mode name (default: the device's cavity)
        displace_gain : nominal displacement gain (v2 fraction) populating the cavity; tune for
                        n̄ ≲ 1 (default 0.02)
        displace_sigma_us : Gaussian sigma of the displacement pulse (us, default 0.05)
        readout       : ``DispersiveIQ`` readout config; defaults to the recorded calibration
                    (``DispersiveIQ.from_calib``)

    Usage::

        exp = RevivalTimeCalibration(dev)
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["revival_time"]
    """

    def __init__(self, device, *, delays: Sweep | None = None, points: int = 101,
                 mode_name: str | None = None, displace_gain: float = 0.01,
                 displace_sigma_us: float = 0.05,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        mode_name = resolve_mode(device, mode_name, purpose=type(self).__name__)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=1000.0, on="t")
        validate_sweep_range(self.delays, min_val=MIN_TIME_SWEEP,
                             max_val=MAX_TIME_SWEEP, label="delay")
        self.points = points
        self.mode_name = mode_name
        self.displace_gain = displace_gain
        self.displace_sigma_us = displace_sigma_us
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def _displacement(self) -> Pulse | None:
        """A raw nominal displacement on the mode line (no calibration read); None if unlocatable.

        Uses only the mode element's frequency/line from the topology — never ``ModeOps.displace``
        — so the experiment needs no ``calib.modes`` entry and stays the parity-chain root.
        Returns ``None`` (degrading to a bare Ramsey) when the mode has no frequency or its
        generator is not wired in the active soccfg snapshot — e.g. a storage mode with no direct
        qubit dispersion, which has no chi to revive anyway.
        """
        mode = getattr(self.device, self.mode_name, None)
        el = getattr(mode, "element", None) if mode is not None else None
        freq_mhz = getattr(el, "freq_mhz", None) if el is not None else None
        if el is None or freq_mhz is None:
            return None
        line = LineRef(el.line or el.name, "main")
        if not self._generator_wired(line, float(freq_mhz)):
            return None
        return Pulse(
            line=line,
            freq_mhz=freq_mhz,
            gain=self.displace_gain,
            phase_deg=0.0,
            envelope=Gaussian(sigma_us=self.displace_sigma_us),
            label="revival_displace",
        )

    def _generator_wired(self, line: LineRef, freq_mhz: float) -> bool:
        """Whether ``line``'s generator channel is present in the device's soccfg snapshot.

        qick's ``declare_gen`` indexes ``soccfg['gens'][dac_ch]`` directly, so a channel past the
        snapshot's generator list raises ``IndexError`` at compile; guard it here so a minimal
        testbench (or an undriven storage line) degrades gracefully instead of failing to build.
        """
        snap = getattr(self.device, "soccfg_snapshot", None) or {}
        gens = snap.get("gens", [])
        if not gens:
            return False
        try:
            gen = self.device.wiring.resolve(line, freq_mhz)
        except Exception:
            return False
        dac = getattr(gen, "dac_ch", None)
        return isinstance(dac, int) and 0 <= dac < len(gens)

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        seq = Seq()
        displacement = self._displacement()
        if displacement is not None:
            seq.play(displacement)
        seq = (
            seq
            .play(q.ge_hpi())
            .wait(self.delays)
            .play(q.ge_hpi(phase=180.0))
            .play(self.device.readout.dispersive_measure())
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        fit = fit_decaysin_detuned(result.xpts, readout_signal(result, self.readout))
        freq = fit.derived.get("freq", 0.0)
        if freq > 0:
            fit.derived["revival_time"] = 0.5 / freq
        else:
            # A degenerate fit (freq <= 0) used to report a bare 0.0
            # here with success left untouched -- indistinguishable from an actual (physically
            # impossible) zero revival time. A caller has no signal this is degenerate output,
            # not a real result, unless success is false.
            fit.derived["revival_time"] = 0.0
            fit.success = False
        return fit
