"""Multiphoton ladder validation experiment (cqed expansion).

Validate the full multiphoton calibration by sequentially preparing Fock states |0⟩ through
|n_max⟩ and checking parity alternation. Even Fock states have positive parity, odd states
have negative parity — any deviation indicates a miscalibrated sideband pulse.

The implementation uses the multi-read pattern: one preparation+parity block per Fock number,
no loop axis. Each Fock(n) is prepared by repeating the ge_pi → ef_pi → f0g1(photon=k) cycle
n times (the multiphoton ladder), then reading out via parity.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.cqed.library.composite import parity
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ


class MultiphotonLadder(Experiment):
    """Prepare Fock states |0⟩ to |n_max⟩ and verify via parity alternation.

    Builds one multi-read program: each block prepares Fock(n) via the multiphoton ladder
    (repeating ge_pi → ef_pi → f0g1 for each photon), then measures parity. The analyzer
    checks that the parity alternates: positive for even n, negative for odd n.

    ``n_max`` is limited by the number of calibrated sideband pulses in
    ``calib.multiphoton["f0-g1"].pi`` (one entry per photon number).
    """

    def __init__(self, device, *, n_max: int = 2, mode_name: str | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        mode_name = resolve_mode(device, mode_name, purpose=type(self).__name__)
        self.mode_name = mode_name
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

        # Determine how many Fock states we can prepare from the calibration
        ladder_cal = device.calib.multiphoton.get("f0-g1")
        max_calibrated = len(ladder_cal.pi) if ladder_cal else 0
        self.n_max = min(n_max, max_calibrated)
        self._fock_numbers = list(range(self.n_max + 1))  # [0, 1, ..., n_max]

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        q = dev.qubit
        seq = Seq()

        for n in self._fock_numbers:
            # Prepare Fock(n) via the multiphoton ladder
            for photon in range(n):
                # Each ladder step: ge_pi → ef_pi → f0g1(photon=k)
                seq.play(q.ge_pi())
                seq.play(q.ef_pi())
                seq.play(dev.sideband.f0g1(photon=photon))

            # Measure parity -- #100: parity() drives the readout and measures internally by
            # default now, so this block no longer chains its own seq.measure() (that would
            # double-measure).
            seq += parity(dev, self.mode_name)

        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        # Each read gives the parity signal for one Fock state
        avgi = np.asarray(result.avgi, dtype=float).ravel()
        avgq = np.asarray(result.avgq, dtype=float).ravel()
        iq = np.stack([avgi, avgq], axis=-1)
        signal = self.readout.project(iq).ravel()

        n_states = len(self._fock_numbers)
        n = min(len(signal), n_states)

        # Expected parity: +1 for even n, -1 for odd n
        # In population units (0 to 1): >0.5 for even, <0.5 for odd
        expected_sign = np.array([1.0 if k % 2 == 0 else -1.0 for k in self._fock_numbers[:n]])
        measured_sign = np.sign(signal[:n] - 0.5)

        # Count agreements
        n_agree = int(np.sum(expected_sign == measured_sign))
        parity_fidelity = n_agree / n if n > 0 else 0.0

        return FitResult(
            params={},
            stderr={},
            derived={
                "parity_fidelity": parity_fidelity,
                "n_states_tested": n,
                "n_agree": n_agree,
            },
            success=True,
        )
