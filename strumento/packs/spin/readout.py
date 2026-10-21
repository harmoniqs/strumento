"""Spin readout kinds — PSB, Elzerman, dispersive gate, latched PSB.

Four readout architectures for semiconductor spin qubits, all satisfying the ``ReadoutKind`` /
``Discriminator`` protocols from ``core/readout.py``:

- ``PSBReadout`` — Pauli spin blockade via charge sensor (singlet/triplet threshold)
- ``ElzermanReadout`` — energy-selective tunneling blip detection (spin-up/down)
- ``DispersiveGateReadout`` — RF phase shift on a gate resonator (quantum capacitance)
- ``LatchedPSBReadout`` — PSB + shelving into metastable charge state for slow integration

Pure: no ``qick`` import.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strumento.core.readout import DataSource


@dataclass
class PSBReadout:
    """Threshold a charge-sensor signal -> singlet/triplet (0/1); reduce -> triplet fraction.

    Pauli spin blockade maps the two-electron spin state onto a charge configuration
    detectable by a nearby charge sensor. The sensor signal exceeds the threshold when
    blockade is lifted (triplet-like state), giving label 1.
    """

    threshold: float
    data_source: DataSource = DataSource.QICK_ADC

    def discriminate(self, raw):
        """Assign 0 (singlet/blocked) or 1 (triplet/unblocked) per shot."""
        return (np.asarray(raw, dtype=float) > self.threshold).astype(int)

    def reduce(self, raw):
        """Triplet fraction — mean of discriminated shots along the repetition axis."""
        return np.mean(self.discriminate(raw), axis=-1)


@dataclass
class ElzermanReadout:
    """Energy-selective tunneling readout (Elzerman scheme).

    The dot is tuned near the reservoir Fermi level such that only the excited spin state
    has enough energy to tunnel out. A charge sensor detects the transient "blip" (tunnel-out
    then tunnel-back-in event) within the readout window t_read.

    Discrimination: threshold on the maximum sensor excursion within t_read.
    - Blip detected (max excursion > threshold) -> spin-up (excited state, label 1)
    - No blip (flat trace) -> spin-down (ground state, label 0)

    Requires: tunnel rate Gamma ~ 1-100 kHz, t_read << T1, E_Z >> k_B T_e.
    """

    threshold: float
    t_read_us: float = 100.0  # readout integration window (microseconds)
    data_source: DataSource = DataSource.QICK_ADC

    def discriminate(self, raw):
        """Detect tunnel blips: max excursion per shot > threshold -> spin-up (1).

        Expects ``raw`` shaped ``(..., n_samples)`` where the last axis is the time-resolved
        sensor trace within the readout window. Takes the max along that axis — a blip
        produces a sharp excursion that exceeds threshold even if brief.
        """
        return (np.max(np.asarray(raw, dtype=float), axis=-1) > self.threshold).astype(int)

    def reduce(self, raw):
        """Spin-up fraction — mean of discriminated shots along the repetition axis."""
        return np.mean(self.discriminate(raw), axis=-1)


@dataclass
class DispersiveGateReadout:
    """Dispersive gate sensing — RF reflectometry on a gate electrode's resonant circuit.

    The quantum capacitance C_Q = e^2 * d^2E/dV_g^2 depends on the spin state at a
    charge-sensitive operating point (PSB point or Elzerman-like). This shifts the resonator
    frequency, detected as a phase shift or amplitude change of the reflected RF carrier.

    Discrimination: threshold on the demodulated sensor signal (after IQ rotation by
    ``angle`` to align the discrimination axis with the I quadrature).
    - Signal > threshold -> label 1 (e.g. triplet-like, small C_Q)
    - Signal <= threshold -> label 0 (e.g. singlet-like, large C_Q)

    Particularly suited to Ge hole qubits (strong SOC -> large spin-dependent C_Q shifts).
    No separate charge sensor needed — the qubit gate IS the sensor.
    """

    threshold: float
    angle: float = 0.0  # IQ rotation angle for optimal discrimination axis (radians)
    data_source: DataSource = DataSource.QICK_ADC

    def discriminate(self, raw):
        """Threshold the (rotated) sensor signal per shot.

        When ``angle == 0`` this is a direct threshold on the raw signal. A nonzero angle
        rotates the IQ plane so the discrimination axis aligns with I before thresholding —
        but for single-channel acquisition (typical in gate reflectometry), the rotation is
        a no-op and we threshold directly.
        """
        arr = np.asarray(raw, dtype=float)
        if self.angle != 0.0:
            # If we have IQ data as complex or paired arrays, rotate.
            # For real-valued single-channel data, cos(angle)*I is the projection.
            arr = arr * np.cos(self.angle)
        return (arr > self.threshold).astype(int)

    def reduce(self, raw):
        """Mean excited-state population from dispersive signal."""
        return np.mean(self.discriminate(raw), axis=-1)


@dataclass
class LatchedPSBReadout:
    """Latched PSB readout — PSB + shelving into a metastable charge state.

    After the spin-dependent PSB event, the system is driven into a metastable charge
    configuration that persists long enough for a slow sensor to integrate. This effectively
    amplifies the transient spin-dependent signal into a long-lived charge signal.

    Advantages: relaxes sensor bandwidth requirements; enables single-shot with DC transport.
    Disadvantages: adds complexity; reset is non-trivial; measurement time is longer.

    Used for S-T0 qubits (latched at (2,0) singlet) and LD qubits with latched Elzerman.
    Discrimination is identical to PSBReadout (simple threshold) because the latching has
    already converted the transient PSB signal into a steady-state charge difference —
    the sensor integrates over ``latch_time_us`` and produces a single value per shot.
    """

    threshold: float
    latch_time_us: float = 50.0  # hold time in the latched state before readout
    data_source: DataSource = DataSource.QICK_ADC

    def discriminate(self, raw):
        """Threshold the latched charge-sensor signal per shot.

        After latching, the sensor signal is DC-like (no blip to catch), so a simple
        threshold suffices — the long integration time gives high SNR.
        """
        return (np.asarray(raw, dtype=float) > self.threshold).astype(int)

    def reduce(self, raw):
        """Mean triplet/blocked fraction from latched signal."""
        return np.mean(self.discriminate(raw), axis=-1)
