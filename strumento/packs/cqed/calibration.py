"""cqed calibration schema + lab-native gain conversion (spec §6.1, D23).

A **typed** subset of the v1 ``hardware_config.yml → device.*`` — one field per physical
quantity, no register shadows. The schema is deliberately strict (``extra="forbid"``) so a
mistyped key in a device YAML fails loudly rather than being silently dropped; fields are added
here slice-by-slice as later ops need them (multiphoton, revival times, …).

**Gain (D23):** the calibration DB keeps lab-native **int codes** (as in v1). Conversion to the
v2 normalized **fraction** happens at the pack boundary via :func:`gain_int_to_frac` — the
compiler never sees an int code. ``maxv`` comes from the soccfg snapshot at factory time.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


def gain_int_to_frac(gain_int: float, maxv: float) -> float:
    """Convert a v1 lab-native gain **int code** to a v2 normalized **fraction** (D23).

    ``maxv`` is the generator's full-scale code from the soccfg snapshot (e.g. 32766). This is
    the single conversion that catches the v1→v2 gain-units bug: v1 code ``3289`` at
    ``maxv=32766`` is a fraction of ``≈0.1004``, not ``3289``.
    """
    return gain_int / maxv


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PulseCal(_Strict):
    """One calibrated single-qubit pulse (gain as a lab-native int code, D23)."""

    gain: float
    sigma: float
    length: float = 0.0
    type: str = "gauss"


class QubitCal(_Strict):
    """Transmon calibration: transition frequencies (MHz) + the named pulse library."""

    f_ge: float
    f_ef: float | None = None
    T1: float | None = None
    T2: float | None = None
    pulses: dict[str, PulseCal] = {}


class ReadoutCal(_Strict):
    """Dispersive-readout calibration (frequency MHz; gain as a lab-native int code)."""

    frequency: float
    gain: float
    readout_length: float
    trig_offset: float
    relax_delay: float
    phase: float | None = None
    threshold: float | None = None


class SidebandPulseCal(_Strict):
    """One calibrated sideband (e.g. f0g1) pulse at a given photon number (flat-top)."""

    frequency: float          # MHz
    gain: float               # lab-native int code (D23)
    length: float             # flat length, µs
    ramp_sigma: float = 0.005  # flat-top ramp sigma, µs
    type: str = "flat_top"


class MultiphotonCal(_Strict):
    """A multiphoton-ladder transition: per-photon-number pulse calibrations."""

    pi: list[SidebandPulseCal] = []
    hpi: list[SidebandPulseCal] = []


class SwapCal(_Strict):
    """One calibrated coupler beamsplitter swap (flat-top), keyed "A-B" in the instance."""

    frequency: float          # MHz (band-gates the coupler generator)
    gain: float               # lab-native int code
    length: float             # π-swap flat length, µs
    ramp_sigma: float = 0.005


class ModeCal(_Strict):
    """Per-bosonic-mode calibration: parity revival time(s) + an optional displacement pulse."""

    revival_time: list[float] = []   # µs; [slow, fast, ...]
    displace: PulseCal | None = None


class PiPulseReference:
    """Physical drive amplitude (MHz Rabi) ↔ v2 gain fraction, via a calibrated π pulse (§6.5).

    The linear ruler the OC import boundary uses: a calibrated π pulse with known gain fraction
    and known peak Rabi rate fixes ``frac_per_mhz``; imported physical amplitudes convert with
    no int-code hop (D23). Callable: ``ref(amp_mhz) -> fraction``.
    """

    def __init__(self, *, pi_gain_frac: float, pi_rabi_mhz: float) -> None:
        if pi_rabi_mhz <= 0:
            raise ValueError("pi_rabi_mhz must be positive")
        self.pi_gain_frac = pi_gain_frac
        self.pi_rabi_mhz = pi_rabi_mhz

    def to_frac(self, amp_mhz: float) -> float:
        return amp_mhz * self.pi_gain_frac / self.pi_rabi_mhz

    __call__ = to_frac


class Calibration(_Strict):
    """The device instance's calibration values (calib VALUES layer, spec §5/D21)."""

    qubit: QubitCal
    readout: ReadoutCal
    #: interaction strengths (MHz) for typed coupling edges, keyed "elementA-elementB" (D31);
    #: the strength stays *data* here, never on the edge declaration.
    couplings: dict[str, float] = {}
    #: multiphoton ladder (f0g1 &c.), keyed by transition name (e.g. "f0-g1").
    multiphoton: dict[str, MultiphotonCal] = {}
    #: coupler beamsplitter swaps, keyed "A-B" (e.g. "manipulate-S1").
    swaps: dict[str, SwapCal] = {}
    #: per-mode calibration (revival times, displacement pulse), keyed by element name.
    modes: dict[str, ModeCal] = {}
