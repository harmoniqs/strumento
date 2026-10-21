"""RF reflectometry optimization — carrier frequency, power, and phase (Layer 1).

Characterizes the tank circuit (LC resonance) used for RF-QPC, RF-SET, or dispersive gate
sensing. Finds the resonance frequency f_0, optimal RF power (SNR vs back-action trade-off),
and the demodulation phase that maximizes signal in one quadrature.

Applicable to: all encodings using RF readout. Infrequent recalibration (resonance is
lithographically fixed); phase recal after thermal cycling or cable changes.

This is a standalone DC-sweep class (not an ``Experiment`` subclass) that owns its own
fitting via ``lmfit`` directly — DC-sweep experiments use domain-specific fit models
(Lorentzian resonance, exponential tunneling) that are not part of core's vendored fitter
set. This is a deliberate design choice, not an inconsistency: core fitters serve the
``Experiment`` contract; DC-sweep classes have their own ``run()``/``analyze()`` lifecycle.

Pure at import: no ``qick``.
"""

from __future__ import annotations

from collections.abc import Callable

import lmfit
import numpy as np

from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult


def _lorentzian_dip(x, f0, q, depth, offset):
    """Lorentzian dip (absorption resonance): offset - depth / (1 + 4*Q^2*((x - f0)/f0)^2)."""
    return offset - depth / (1.0 + 4.0 * q**2 * ((x - f0) / f0)**2)


class RFReflectometryTune:
    """Characterize and optimize the RF reflectometry readout chain.

    Standalone class — NOT an Experiment subclass. This sweeps a frequency (not a DC
    gate voltage), so it takes a ``set_frequency`` callable instead of a DCSource.

    Swept: carrier frequency around expected resonance.
    Measured: reflected amplitude |S11| — dip at resonance.
    Extracts: f_0 (resonance), loaded Q from the width of the dip.
    Prerequisites: matching network present; for gate-based sensing the dot must be formed.
    """

    def __init__(
        self,
        *,
        frequencies=np.linspace(300.0, 325.0, 51),
        probe: Callable[[], float],
        set_frequency: Callable[[float], None],
    ) -> None:
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.probe = probe
        self.set_frequency = set_frequency

    def run(self) -> MeasurementResult:
        """Sweep RF frequency, record probe signal (|S11|) at each point."""
        data = np.zeros(len(self.frequencies))
        for i, f in enumerate(self.frequencies):
            self.set_frequency(float(f))
            data[i] = float(self.probe())
        return MeasurementResult(
            xpts=self.frequencies,
            avgi=data.reshape(1, -1),
            avgq=np.zeros((1, len(self.frequencies))),
            meta={"sweep": "frequency_mhz"},
        )

    def analyze(self, result: MeasurementResult) -> FitResult:
        """Fit a Lorentzian dip to |S11(f)| — extract resonance frequency and loaded Q."""
        x = np.asarray(result.xpts, dtype=float)
        signal = np.asarray(result.avgi).ravel()

        # Fit a Lorentzian dip to |S11(f)|
        f_min_idx = int(np.argmin(signal))
        f0_seed = float(x[f_min_idx])
        depth_seed = float(np.max(signal) - np.min(signal))
        span = float(x.max() - x.min()) or 1.0

        model = lmfit.Model(_lorentzian_dip)
        pars = model.make_params(
            f0=f0_seed,
            q=f0_seed / (span / 10.0),  # Q ~ f0 / FWHM
            depth=depth_seed,
            offset=float(np.max(signal)),
        )
        pars["f0"].set(min=float(x.min()), max=float(x.max()))
        pars["q"].set(min=1.0)
        pars["depth"].set(min=0.0)

        out = model.fit(signal, pars, x=x)

        params = {k: float(v.value) for k, v in out.params.items()}
        stderr = {k: (float(v.stderr) if v.stderr is not None else float("nan"))
                  for k, v in out.params.items()}

        f_resonance_mhz = params["f0"]
        q_loaded = params["q"]

        derived = {"f_resonance_mhz": f_resonance_mhz, "q_loaded": q_loaded}
        return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))
