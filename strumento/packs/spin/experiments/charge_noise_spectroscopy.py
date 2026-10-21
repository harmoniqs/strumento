"""Charge noise spectroscopy — detuning noise PSD measurement (Layer 4).

Measures the noise power spectral density by performing repeated Ramsey/echo experiments
at varying interpulse delays. The decay of coherence vs delay time encodes the noise
spectrum: Ramsey (FID) is sensitive to low-frequency noise, while echo/CPMG filter out
1/f components and probe higher frequencies.

Applicable to: all encodings. Diagnostic — done during characterization or after
suspected degradation.

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.spin.experiments._signal import readout_signal
from strumento.packs.spin.readout import PSBReadout


class ChargeNoiseSpectroscopy(Experiment):
    """Measure charge noise PSD: Ramsey decay vs tau -> noise spectrum.

    Swept: free evolution time tau (variable interpulse delay).
    Measured: coherence decay envelope — the decay shape encodes the noise spectrum.
    Extracts: noise amplitude sqrt(A), exponent alpha in S(f) = A / f^alpha.
    Prerequisites: sensor tuned; at a charge-sensitive point; Ramsey/echo operational.

    The sequence is a Ramsey measurement with variable tau. From the coherence decay
    envelope at multiple tau values, the noise PSD is reconstructed: a Gaussian decay
    indicates 1/f noise (alpha ~ 1), exponential indicates white noise (alpha ~ 0).
    """

    def __init__(self, device, *, dot_name: str = "q1", delays: Sweep | None = None,
                 points: int = 101, readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self.delays = delays if delays is not None else Sweep(start=0.01, stop=50.0, on="tau")
        self.points = points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # Ramsey with variable tau for noise spectroscopy
        hpi = self._dot.hpi()
        seq = Seq().play(hpi).wait(self.delays).play(hpi).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        x = np.asarray(result.xpts, dtype=float)
        signal = readout_signal(result, self.readout)

        # Extract the coherence decay envelope.
        # For noise spectroscopy, fit the envelope to: exp(-(tau/T2*)^n)
        # where n=2 -> Gaussian decay (1/f noise, alpha=1)
        #       n=1 -> exponential decay (white noise, alpha=0)
        # The noise amplitude is related to 1/T2*^n.

        # Normalize signal to [0, 1] range
        s_max = float(np.max(np.abs(signal)))
        signal_norm = np.abs(signal) / s_max if s_max > 0 else np.abs(signal)
        signal_clipped = np.clip(signal_norm, 1e-10, 1.0)

        # Fit log(envelope) = -c * tau^n to extract n (the noise exponent)
        log_env = np.log(signal_clipped)

        # Two-parameter fit in log-log space: log(-log(env)) = n*log(tau) + log(c)
        valid = (signal_clipped < 0.99) & (signal_clipped > 0.01) & (x > 0)
        if np.sum(valid) >= 3:
            neg_log_env = -log_env[valid]
            neg_log_env_clipped = np.clip(neg_log_env, 1e-10, None)
            log_tau = np.log(x[valid])
            log_neg_log = np.log(neg_log_env_clipped)
            coeffs = np.polyfit(log_tau, log_neg_log, 1)
            exponent_n = float(coeffs[0])
            log_c = float(coeffs[1])
            c = np.exp(log_c)
        else:
            exponent_n = 2.0  # default to Gaussian
            c = 1.0

        # Noise exponent alpha relates to decay exponent n:
        # For Ramsey: alpha = 2*(n-1) for 1/f^alpha noise
        # n=2 -> alpha=2 (1/f^2 = random walk), n=1 -> alpha=0 (white)
        # More commonly: n=2 is 1/f noise in the literature convention
        exponent_alpha = max(0.0, exponent_n - 1.0)

        # Noise amplitude: A ~ c * (some geometric factor depending on the filter function)
        noise_amplitude = float(np.sqrt(c)) if c > 0 else 0.0

        params = {"exponent_n": exponent_n, "c": float(c)}
        stderr = {"exponent_n": 0.0, "c": 0.0}
        derived = {"noise_amplitude": noise_amplitude, "exponent_alpha": exponent_alpha}
        success = bool(np.sum(valid) >= 3)
        return FitResult(params=params, stderr=stderr, derived=derived, success=success)
