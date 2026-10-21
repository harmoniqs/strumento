"""Sideband Rabi experiment.

Calibrate the f0-g1 pi pulse amplitude and/or length by sweeping the sideband drive
gain or duration. The analyzer fits a Rabi oscillation.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_rabi
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import gain_int_to_frac
from strumento.packs.cqed.experiments._signal import readout_iq, readout_signal
from strumento.packs.cqed.library._gain import maxv_for
from strumento.packs.cqed.readout import DispersiveIQ


class SidebandRabi(Experiment):
    """Rabi on the f0-g1 sideband transition, sweeping drive amplitude OR duration.

    Sequence: ge pi, ef pi, sideband f0-g1 pulse (swept), ef pi map, readout. The qubit is shelved
    in |f> and the sideband drive performs |f,0> to |g,1> oscillations. The closing ef pi maps
    residual |f,0> to |e,0> while leaving |g,1> untouched, making the readout contrast refer to the desired sideband transfer.

    ``sweep_param`` selects what is swept:

    * ``"gain"`` (default) — sweep the drive amplitude at the calibrated pulse length; ``pi_gain``
      is the pi-pulse amplitude (in the swept gain units).
    * ``"length"`` — sweep the flat duration at the calibrated drive amplitude (``cal.gain``);
      ``pi_gain`` is then the pi-pulse **length** (us). This is a length Rabi: fix the
      amplitude, find the time. Only valid on a flat-top-calibrated
      ladder (the f0-g1 pulses here are flat_top).

    Fitter: ``fit_rabi`` -- decaying-cosine fit; ``derived["pi_gain"]`` = half-period in the swept
    units (gain code, or us for a length sweep).

    Parameters:
        gains       : amplitude sweep (``Sweep`` on ``"amp"``); used when ``sweep_param="gain"``.
                      Defaults to 0-16000 (within MAX_SIDEBAND_GAIN=0.5).
        lengths     : duration sweep (``Sweep`` on ``"t"``, us); used when ``sweep_param="length"``.
                      Defaults to 0-3 us.
        points      : number of sweep points.
        sweep_param : ``"gain"`` (default) or ``"length"``.

    Note: the simulator response models a *gain* Rabi, so ``--sim`` of a length sweep is a plumbing
    check only (flat), not device physics — like the other mode experiments, trust length physics
    under ``--arm``.

    Usage::

        exp = SidebandRabi(dev, sweep_param="length", lengths=Sweep(0, 2.5, on="t"))
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["pi_gain"]   # the pi LENGTH (us) for a length sweep
    """

    def __init__(self, device, *, gains: Sweep | None = None, lengths: Sweep | None = None,
                 points: int = 101, sweep_param: str = "gain", fixed_gain_code: float | None = None,
                 fixed_frequency_mhz: float | None = None, post_map_ef: bool = True,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        if sweep_param not in ("gain", "length"):
            raise ValueError(f"sweep_param must be 'gain' or 'length'; got {sweep_param!r}")
        # Default lowered to stay within MAX_SIDEBAND_GAIN (0.5 frac) now that sweep
        # endpoints are actually validated -- was 0->30000 (~0.916).
        self.gains = gains if gains is not None else Sweep(start=0, stop=16000, on="amp")
        self.lengths = lengths if lengths is not None else Sweep(start=0.0, stop=3.0, on="t")
        self.points = points
        self.sweep_param = sweep_param
        #: Optional native gain code held fixed during a length Rabi. ``None`` retains the calibrated
        #: f0-g1 gain, while an explicit code tests a candidate amplitude without rewriting device
        #: calibration.
        self.fixed_gain_code = fixed_gain_code
        #: Optional sideband carrier frequency, used by frequency-by-length calibration maps without
        #: altering the device's current f0-g1 belief.
        self.fixed_frequency_mhz = fixed_frequency_mhz
        #: Map residual |f,0> to bright |e,0> after the sideband pulse. This ordering makes the transfer observable on the calibrated ge readout axis.
        self.post_map_ef = bool(post_map_ef)
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def _swept(self) -> Sweep:
        return self.lengths if self.sweep_param == "length" else self.gains

    def axes(self) -> tuple:
        swept = self._swept()
        if isinstance(swept, Sweep) and swept.on is not None:
            return (Loop(swept.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        # Honor sweep_param: a length sweep fixes the drive at the calibrated amplitude and sweeps
        # the flat duration (f0g1(length=...)), a gain sweep fixes the length and sweeps amplitude.
        # Previously this always swept gain and silently ignored sweep_param="length".
        if self.sweep_param == "length":
            fixed_gain = None
            if self.fixed_gain_code is not None:
                cal = dev.calib.multiphoton["f0-g1"].pi[0]
                ref = LineRef(dev.sideband.element.name, "sideband")
                fixed_gain = gain_int_to_frac(
                    self.fixed_gain_code, maxv_for(dev, ref, float(cal.frequency))
                )
            sideband_pulse = dev.sideband.f0g1(
                freq=self.fixed_frequency_mhz, gain=fixed_gain, length=self.lengths
            )
        else:
            sideband_pulse = dev.sideband.f0g1(freq=self.fixed_frequency_mhz, gain=self.gains)
        seq = (
            Seq()
            .play(dev.qubit.ge_pi())
            .play(dev.qubit.ef_pi())
            .play(sideband_pulse)
        )
        if self.post_map_ef:
            seq = seq.play(dev.qubit.ef_pi())
        seq = (
            seq
            .play(dev.readout.dispersive_measure())
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        fit = fit_rabi(result.xpts, readout_signal(result, self.readout))
        if self.readout.f_center is None:
            return fit

        raw_i, raw_q = readout_iq(result)
        populations = self.readout.to_level_populations(np.stack((raw_i, raw_q), axis=-1))
        xpts = np.asarray(result.xpts, dtype=float)
        max_g_index = int(np.argmax(populations[:, 0]))
        pi_length = fit.derived.get("pi_gain")
        pi_index = max_g_index if pi_length is None else int(np.argmin(np.abs(xpts - pi_length)))
        metrics = {
            "initial_g_population": float(populations[0, 0]),
            "initial_e_population": float(populations[0, 1]),
            "max_g_population": float(populations[max_g_index, 0]),
            "max_g_sweep_value": float(xpts[max_g_index]),
            "pi_g_population": float(populations[pi_index, 0]),
            "pi_e_population": float(populations[pi_index, 1]),
            "pi_f_population": float(populations[pi_index, 2]),
        }
        if self.post_map_ef:
            metrics.update(
                {
                    "initial_mapped_e_population": float(populations[0, 1]),
                    "residual_mapped_e_at_max_g": float(populations[max_g_index, 1]),
                }
            )
        else:
            metrics.update(
                {
                    "initial_f_population": float(populations[0, 2]),
                    "residual_excitation_at_max_g": float(
                        populations[max_g_index, 1] + populations[max_g_index, 2]
                    ),
                }
            )
        metrics[f"max_g_{self.sweep_param}"] = float(xpts[max_g_index])
        # ``fit_rabi`` reports period/2, which is useful as a smooth-model diagnostic but can miss
        # the actual transfer extremum when the fitted phase is nonzero. Preserve it explicitly and
        # make ``pi_gain`` the measured best-transfer setting used by the runner/plot.
        metrics["fitted_half_period"] = float(fit.derived["pi_gain"])
        metrics["pi_gain"] = float(xpts[max_g_index])
        fit.derived.update(metrics)
        return fit
