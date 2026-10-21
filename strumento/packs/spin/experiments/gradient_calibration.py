"""Magnetic gradient calibration — Delta_Bz measurement for S-T qubits (Layer 2).

Measures the magnetic field gradient between two dots by observing the free precession
frequency of the S-T0 qubit at J ~ 0 (where the splitting is dominated by Delta_Bz).
For micromagnet devices, Delta_Bz is fixed by geometry; for GaAs, it is from DNP.

Applicable to: S-T (singlet-triplet) encoding. After any field change.

Pure at import: no ``qick``.
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_decaysin_detuned
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.spin.experiments._signal import readout_signal
from strumento.packs.spin.readout import PSBReadout


class GradientCalibration(Experiment):
    """Measure the inter-dot magnetic gradient Delta_Bz from S-T0 free precession.

    Swept: free evolution time tau at the idle point (J ~ 0).
    Measured: P_singlet(tau) — precesses at f_z = g*mu_B*Delta_Bz / h.
    Extracts: Delta_Bz (gradient in MHz), f_z (precession frequency), T2* at idle.
    Prerequisites: PSB readout; know where J ~ 0 (from exchange calibration).

    The sequence is a Ramsey-like measurement: prepare singlet (via initialization),
    apply pi/2 equivalent (separate into (1,1) at J~0), wait for variable tau
    (free precession under Delta_Bz), recombine (pi/2), and PSB readout.
    """

    def __init__(self, device, *, dot_name: str = "q1", delays: Sweep | None = None,
                 points: int = 101, readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=5.0, on="tau")
        self.points = points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # Ramsey on the S-T0 qubit: hpi -> wait(tau) -> hpi -> measure
        # The "pi/2" for S-T0 is a detuning pulse that separates/recombines the singlet
        hpi = self._dot.hpi()
        seq = Seq().play(hpi).wait(self.delays).play(hpi).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        # The precession frequency is Delta_Bz in frequency units (MHz)
        fit = fit_decaysin_detuned(result.xpts, readout_signal(result, self.readout))

        # The detuning from the Ramsey fit IS the gradient frequency
        delta_bz_mhz = fit.derived.get("detuning", 0.0)

        # Override derived to match the gradient calibration naming
        fit_derived = dict(fit.derived)
        fit_derived["delta_bz_mhz"] = delta_bz_mhz
        return FitResult(
            params=fit.params,
            stderr=fit.stderr,
            derived=fit_derived,
            success=fit.success,
        )
