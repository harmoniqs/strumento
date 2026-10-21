"""TLS scan — high-power spectroscopy for two-level-system defects near the qubit (#56).

An ESCALATION-ONLY bring-up node: a high-gain frequency sweep around the qubit line whose
vetted fitter (the main-line Lorentzian + a robust residual scan) reports the TLS-line count
and the strongest offset as **gate-only evidence** for the park decision. The node banks
nothing — the graph's provides are empty; the evidence rides the child record's fit metrics.

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Const, Pulse, Seq
from strumento.core.resonator_fitters import fit_lorentzian
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._common import drive_gain_limit
from strumento.packs.cqed.experiments._signal import readout_iq
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import validate_gain

#: A residual dip deeper than this multiple of the residual MAD counts as a TLS line — well
#: above the fit's shot noise, well below a real defect's contrast.
TLS_RESIDUAL_SIGMA: float = 8.0

#: The absolute residual floor (in signal units) beneath which nothing counts as a line,
#: whatever the MAD says — a guard against noise-counting on an over-clean trace.
TLS_RESIDUAL_FLOOR: float = 0.05


class TlsScan(Experiment):
    """High-power frequency sweep for TLS lines near the qubit — gate-only evidence.

    The analyzer fits the (power-broadened) main qubit line with :func:`fit_lorentzian` and
    scans the residuals for narrow dips: each cluster beyond :data:`TLS_RESIDUAL_SIGMA`
    (and the absolute floor) counts as a TLS line. Derived: ``tls_lines`` (the count),
    ``tls_offset_mhz`` (the strongest line's offset from the qubit) and the main line's
    ``f0`` — the evidence a park decision reads.
    """

    def __init__(self, device, *, center_mhz: float | None = None, span_mhz: float = 15.0,
                 points: int = 81, probe_gain: float = 0.45,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        qc = device.calib.qubit
        self.center_mhz = center_mhz if center_mhz is not None else float(qc.f_ge)
        self.span_mhz = span_mhz
        self.points = points
        self.probe_gain = probe_gain
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        return (Loop("freq", self.points),)

    def recovery_map(self) -> dict[str, str]:
        """The scan's main line IS the qubit line: pair the fitted center with the armed
        truth's ``f_ge`` for the vetted emitter's fit-recovery comparison."""
        return {"f0": "f_ge"}

    def sequence(self) -> tuple[Seq, tuple]:
        qubit_name = self.device.qubit.element.name
        line = LineRef(qubit_name, "drive")
        freqs = Sweep(
            start=self.center_mhz - self.span_mhz, stop=self.center_mhz + self.span_mhz, on="freq"
        )
        validate_gain(self.probe_gain, limit=drive_gain_limit(self.device), role="qubit_drive",
                      label="tls_probe")
        probe = Pulse(
            line=line, freq_mhz=freqs, gain=self.probe_gain,
            envelope=Const(length_us=10.0), label="tls_probe",
        )
        # `probe` above drives the QUBIT line, never the resonator (#103) -- add the real
        # readout probe alongside it rather than leaving .measure() bare.
        seq = Seq().play(probe) + self.device.readout.dispersive_measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        freqs = np.asarray(result.xpts, dtype=float)
        i, q = readout_iq(result)
        signal = np.hypot(np.asarray(i, dtype=float), np.asarray(q, dtype=float))

        main = fit_lorentzian(freqs, signal)
        p = main.params
        model = (
            float(p["y0"])
            + float(p["amp"]) / (1.0 + ((freqs - float(p["f0"])) / float(p["hwhm"])) ** 2)
        )
        residual = signal - model
        mad = float(np.median(np.abs(residual - np.median(residual)))) or 1e-9
        below = residual < -max(TLS_RESIDUAL_SIGMA * mad, TLS_RESIDUAL_FLOOR)

        # count contiguous clusters of below-threshold points; the strongest sets the offset
        count = 0
        strongest_offset = 0.0
        strongest_depth = 0.0
        run: list[float] = []
        for freq, hit, depth in zip(freqs, below, residual):
            if hit:
                run.append(float(freq))
                if depth < strongest_depth:
                    strongest_depth = float(depth)
                    strongest_offset = float(freq) - float(p["f0"])
            elif run:
                count += 1
                run = []
        if run:
            count += 1

        return FitResult(
            params=dict(main.params),
            stderr=dict(main.stderr),
            derived={
                "f0": float(main.derived["f0"]),
                "hwhm": float(main.derived["hwhm"]),
                "tls_lines": float(count),
                "tls_offset_mhz": strongest_offset if count else 0.0,
            },
            success=bool(main.success),
        )
