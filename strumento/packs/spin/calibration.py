"""spin calibration schema (spec S15.5).

Typed calibration for all spin-qubit element types and readout architectures:
- Per-dot: qubit frequency, g-factor, encoding-specific params, pulse library
- Per-gate: DC operating points, lever arms, cross-coupling, exchange calibration
- Per-sensor: operating point, sensitivity, RF parameters
- Per-reservoir: Fermi level, tunnel rate, temperature
- Readout: PSB, Elzerman, dispersive gate parameters
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    #: no calibration value is NaN or infinite (#191)
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SpinPulseCal(_Strict):
    """One calibrated spin-control pulse. Gain is a v2 **fraction** — spin stacks are
    fraction-native (no legacy int-code DB), so no conversion happens at the boundary."""

    gain: float
    sigma: float = 0.0        # gaussian sigma, us (LD drive)
    length: float = 0.0       # flat length, us (exchange)
    type: str = "gauss"

    @field_validator("gain")
    @classmethod
    def _gain_in_range(cls, v: float) -> float:
        if abs(v) > 1.0:
            raise ValueError(
                f"gain={v} exceeds safe range [-1.0, 1.0]; "
                "spin stacks are fraction-native (full DAC scale = 1.0)"
            )
        return v

    @field_validator("length")
    @classmethod
    def _length_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"pulse length={v} us cannot be negative")
        return v

    @field_validator("sigma")
    @classmethod
    def _sigma_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"pulse sigma={v} us cannot be negative")
        return v


class DotCal(_Strict):
    """Per-dot calibration: qubit frequency (MHz), optional exchange, and the pulse library."""

    f_qubit_mhz: float
    exchange_j_mhz: float | None = None
    pulses: dict[str, SpinPulseCal] = {}
    #: g-factor (dimensionless); for isotropic systems a scalar, for anisotropic (Ge holes)
    #: the effective g at the operating B-field orientation.
    g_factor: float | None = None
    #: g-factor derivative dg/dV_g at operating point (MHz/V); zero at sweet spot.
    dg_dv: float | None = None
    #: T1 relaxation time (microseconds), measured.
    t1_us: float | None = Field(None, ge=0)
    #: T2* dephasing time (microseconds), measured.
    t2_star_us: float | None = Field(None, ge=0)
    #: T2 echo coherence time (microseconds), measured.
    t2_echo_us: float | None = Field(None, ge=0)


class GateCal(_Strict):
    """Per-gate electrostatics: the slow-DC axis (external source) + safety + lever arms.

    A barrier gate may also carry a calibrated fast ``exchange`` pulse (flat-top on the gate's
    ``fast`` generator role).
    """

    dc_channel: int
    max_v: float = 1.5
    crosscoupling: dict[str, float] = {}
    exchange: SpinPulseCal | None = None
    #: lever arm alpha = C_g / C_Sigma (from Coulomb diamond), dimensionless.
    lever_arm: float | None = None
    #: pinch-off threshold voltage (V).
    pinchoff_v: float | None = None

    @field_validator("max_v")
    @classmethod
    def _max_v_positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"max_v={v} must be positive")
        return v


# ---------------------------------------------------------------------------
# New calibration sections for extended element types
# ---------------------------------------------------------------------------


class SensorCal(_Strict):
    """Per-charge-sensor calibration (QPC or SET).

    Operating point, sensitivity, and RF parameters (if using RF reflectometry).
    """

    #: Optimal gate voltage for max transconductance (V).
    v_operating: float | None = None
    #: Conductance setpoint (e^2/h for QPC; current for SET).
    g_setpoint: float | None = None
    #: Transconductance dG/dV at operating point.
    transconductance: float | None = None
    #: RF carrier frequency if using reflectometry (MHz).
    rf_freq_mhz: float | None = None
    #: RF power (dBm).
    rf_power_dbm: float | None = None
    #: Demodulation phase (radians).
    demod_phase_rad: float | None = None


class ReservoirCal(_Strict):
    """Per-reservoir calibration.

    Tunnel rate, Fermi level position, and electron temperature.
    """

    #: Dot-reservoir tunnel rate (kHz).
    gamma_khz: float | None = None
    #: Electron temperature (mK).
    t_electron_mk: float | None = None


class ReadoutCal(_Strict):
    """Readout calibration parameters (applies to whichever readout kind is in use).

    Threshold, fidelity, and timing — readout-kind-agnostic container.
    """

    #: Discrimination threshold.
    threshold: float | None = None
    #: Measured readout fidelity.
    fidelity: float | None = None
    #: Readout duration (microseconds).
    t_read_us: float | None = Field(None, ge=0)
    #: IQ rotation angle (radians) — for dispersive gate readout.
    angle_rad: float | None = None


class GateSensorCal(_Strict):
    """Per-gate-sensor calibration (dispersive gate sensing).

    RF resonator parameters and spin-dependent dispersive shift.
    """

    #: Tank circuit resonance frequency (MHz).
    resonator_freq_mhz: float | None = None
    #: Loaded quality factor.
    q_loaded: float | None = None
    #: Spin-dependent quantum capacitance shift (aF).
    delta_cq_af: float | None = None
    #: Optimal detuning for readout.
    epsilon_readout: float | None = None


class SpinCalibration(_Strict):
    """Quantum-dot instance calibration — full schema for all element types."""

    dots: dict[str, DotCal] = {}
    #: slow-DC gate map (plungers, barriers, sensor gates), keyed by element/axis name
    gates: dict[str, GateCal] = {}
    #: per-charge-sensor calibration, keyed by sensor element name
    sensors: dict[str, SensorCal] = {}
    #: per-reservoir calibration, keyed by reservoir element name
    reservoirs: dict[str, ReservoirCal] = {}
    #: readout calibration (one per readout kind in use)
    readout: ReadoutCal | None = None
    #: per-gate-sensor calibration (dispersive gate sensing), keyed by element name
    gate_sensors: dict[str, GateSensorCal] = {}
    #: virtual gate matrix (cross-capacitance compensation), as a flat list of lists
    virtual_gate_matrix: list[list[float]] | None = None


class SpinGainReference:
    """Physical drive amplitude (MHz Rabi) ↔ v2 gain fraction, via a calibrated π pulse.

    The spin equivalent of cqed's ``PiPulseReference``. Spin stacks are fraction-native
    (no int-code hop, D23), so the conversion is a single linear ruler: a calibrated π
    pulse with known gain fraction and known peak Rabi rate fixes the ratio.

    Callable: ``ref(amp_mhz) -> fraction``.

    Parameters
    ----------
    pi_gain_frac : float
        The calibrated π-pulse gain as a v2 fraction (e.g. 0.4).
    pi_rabi_mhz : float
        The Rabi frequency at that gain (e.g. 20.0 MHz).
    """

    def __init__(self, *, pi_gain_frac: float, pi_rabi_mhz: float) -> None:
        if pi_rabi_mhz <= 0:
            raise ValueError("pi_rabi_mhz must be positive")
        self.pi_gain_frac = pi_gain_frac
        self.pi_rabi_mhz = pi_rabi_mhz

    def to_frac(self, amp_mhz: float) -> float:
        """Convert a physical Rabi amplitude (MHz) to a v2 gain fraction."""
        return amp_mhz * self.pi_gain_frac / self.pi_rabi_mhz

    __call__ = to_frac
