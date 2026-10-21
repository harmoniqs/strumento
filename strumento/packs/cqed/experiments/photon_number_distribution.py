"""Photon number distribution experiment (cqed expansion).

Measure the photon number population of a cavity state via number-selective qubit pi pulses.
In the dispersive regime, the qubit transition frequency depends on the cavity photon number:
f_ge(n) = f_ge - n·chi. A pi pulse at frequency f_ge - n·chi flips the qubit only if the
cavity has exactly n photons. Measuring the qubit state after each selective pulse gives P(n).

The implementation uses the multi-read pattern: one number-selective pi + readout per Fock
level, no loop axis. This is hardware-faithful — each block interrogates a different photon
number by driving at a different frequency.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import gain_int_to_frac
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ


class PhotonNumberDistribution(Experiment):
    """Measure population of each Fock level |0⟩ through |n_max⟩ via selective pi pulses.

    Builds one multi-read program: each block drives a qubit pi pulse at frequency
    f_ge - n·chi (selective on photon number n), then measures. The qubit excited-state
    population after each block gives P(n) — the probability that the cavity has exactly
    n photons.

    Requires ``calib.modes[mode_name].chi`` for the dispersive shift.
    """

    def __init__(self, device, *, n_max: int = 6, mode_name: str | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        mode_name = resolve_mode(device, mode_name, purpose=type(self).__name__)
        self.n_max = n_max
        self.mode_name = mode_name
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)
        self._photon_numbers = list(range(n_max + 1))

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        q = dev.qubit
        qc = dev.calib.qubit
        f_ge = qc.f_ge

        # Get chi from mode calibration -- required: this used to
        # silently fall back to an arbitrary 1.0 MHz on a missing/zero calibration and drive
        # every number-selective pulse at the wrong frequency with no error signal, directly
        # contradicting this class's own docstring ("Requires calib.modes[mode_name].chi").
        mode_cal = dev.calib.modes.get(self.mode_name)
        if mode_cal is None or mode_cal.chi is None:
            raise KeyError(
                f"mode {self.mode_name!r} has no chi calibration — "
                "run NumberSplitting to calibrate the dispersive shift"
            )
        chi = mode_cal.chi

        # Resolve gain for the qubit pi pulse
        pc = qc.pulses["pi_ge"]
        element_name = q.element.name
        line = LineRef(element_name, "drive")
        snap = dev.soccfg_snapshot
        maxv = 32766.0
        if snap:
            gens = snap.get("gens", [])
            gen = dev.wiring.resolve(line, f_ge)
            for g in gens:
                if g.get("tproc_ch") == gen.dac_ch:
                    maxv = g["maxv"]
                    break
        gain_frac = gain_int_to_frac(pc.gain, maxv)

        seq = Seq()
        for n in self._photon_numbers:
            # Number-selective pi pulse at f_ge - n*chi
            freq_n = f_ge - n * chi
            from strumento.core.pulses import Gaussian
            selective_pi = Pulse(
                line=line,
                freq_mhz=freq_n,
                gain=gain_frac,
                phase_deg=0.0,
                envelope=Gaussian(sigma_us=pc.sigma),
                label=f"selective_pi_n{n}",
            )
            seq.play(selective_pi)
            # `selective_pi` above drives the QUBIT line, never the resonator (#103) -- each
            # Fock level's block still needs its own readout-line drive before its Measure,
            # alongside the existing number-selective pi pulse.
            seq += self.device.readout.dispersive_measure()

        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        # Each read gives the qubit excited-state population = P(n)
        avgi = np.asarray(result.avgi, dtype=float).ravel()
        avgq = np.asarray(result.avgq, dtype=float).ravel()
        iq = np.stack([avgi, avgq], axis=-1)
        signal = self.readout.project(iq).ravel()

        n_levels = len(self._photon_numbers)
        n = min(len(signal), n_levels)
        populations = signal[:n]

        # Mean photon number: sum(n * P(n))
        p_norm = populations / populations.sum() if populations.sum() > 0 else populations
        mean_n = float(np.sum(np.arange(n) * p_norm))

        return FitResult(
            params={},
            stderr={},
            derived={
                "mean_photon": mean_n,
                "n_levels": n,
                "max_population_at": int(np.argmax(populations)),
            },
            success=True,
        )
