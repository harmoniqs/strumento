"""Punchout — the readout resonator's critical-photon-number power sweep (bring-up #56).

Sweeps the readout probe gain and measures the on-resonance response at each power. As the
intracavity photon number crosses the resonator's critical value the dispersive shift
collapses and the resonance disappears — the punchout. The measured curve is the normalized
on-resonance transmission vs gain: deep (the notch) at low power, rising to the background
once the qubit saturates. The vetted fitter is :func:`fit_lorentzian` on the gain axis (the
depth profile is a Lorentzian centered at zero gain): its HWHM is the **knee** — the
half-depth gain, the operating ceiling the node banks as ``readout.max_power`` — and the
operating point lands at half the ceiling as ``readout.power``.

The **kappa estimate is gate-only** (it banks nothing): the pack's phenomenology ties the
knee to the resonator decay rate, ``g_knee = KNEE_GAIN_PER_SQRT_KAPPA · sqrt(κ)`` — a lossier
(larger-κ) resonator holds fewer photons per unit drive, so it punches out at a higher probe
gain — and the analyzer inverts it. The bring-up graph's gate checks the estimate against the
pack prior window; the provides carry only the envelope path and the operating point.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.resonator_fitters import fit_lorentzian
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ

#: The knee gain (lab-native int codes) of a resonator with κ = 1 MHz: the pack's gain-to-
#: photon calibration for the punchout phenomenology (the sim's response and the analyzer's
#: inverse share it, so a fit recovers the armed truth's κ).
KNEE_GAIN_PER_SQRT_KAPPA: float = 12000.0

#: The operating point as a fraction of the punched-out ceiling (safely below it).
OPERATING_FRACTION: float = 0.5


class Punchout(Experiment):
    """Readout-power sweep to find the resonator's punchout ceiling and operating point.

    ``gains`` is a lab-native int-code :class:`Sweep` on the readout probe (D23); at each
    gain the experiment measures the on-resonance response. The analyzer fits the depth
    profile with :func:`fit_lorentzian` and derives ``max_power`` (the knee), ``power`` (the
    operating point) and the gate-only ``resonator_kappa_mhz`` estimate.
    """

    def __init__(self, device, *, gains: Sweep | None = None, points: int = 41,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        # 19000, not 30000 (~0.916 frac -> GainOutOfRange: readout.py's probe() validates
        # sweep endpoints): stays under
        # MAX_READOUT_GAIN=0.6 (~19659 int code). This is comfortably past a typical
        # KNEE_GAIN_PER_SQRT_KAPPA=12000 (kappa=1MHz) knee -- a real resonator whose
        # punchout knee sits beyond the readout safety ceiling needs an explicit
        # unsafe=True-style override on the gain, not a wider default.
        self.gains = gains if gains is not None else Sweep(start=0, stop=19000, on="gain")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.gains, Sweep) and self.gains.on is not None:
            return (Loop(self.gains.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        probe = self.device.readout.probe(gain=self.gains)
        return Seq().play(probe).measure(), ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)
        xpts = np.asarray(result.xpts, dtype=float)
        fit = fit_lorentzian(xpts, signal)

        knee = abs(float(fit.derived["hwhm"]))  # the half-depth gain: the punched-out ceiling
        depth = abs(float(fit.params["amp"]))
        max_power = knee
        power = OPERATING_FRACTION * max_power
        kappa = (knee / KNEE_GAIN_PER_SQRT_KAPPA) ** 2 if knee > 0 else float("inf")

        params = dict(fit.params)
        params["max_power"] = max_power
        params["power"] = power
        params["resonator_kappa_mhz"] = kappa
        return FitResult(
            params=params,
            stderr=dict(fit.stderr),
            derived={
                "max_power": max_power,
                "power": power,
                "resonator_kappa_mhz": kappa,
                "depth": depth,
            },
            success=bool(fit.success),
        )
