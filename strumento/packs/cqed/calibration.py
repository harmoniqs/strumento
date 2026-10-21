"""cqed calibration schema + lab-native gain conversion (D23).

A **typed** device calibration — one field per physical quantity, no register shadows. The schema is deliberately strict (``extra="forbid"``) so a
mistyped key in a device YAML fails loudly rather than being silently dropped; fields are added
here as later ops need them (multiphoton, revival times, …).

**Units:** every field's ``description`` starts with its unit (``"MHz — qubit g–e transition
frequency"``), so the unit shows in the JSON Schema and in IDE hovers. The names stay unsuffixed:
renaming them would break every device YAML and every stored calibration.

**Constraints (#191):** no value is NaN or infinite, every µs time is ``>= 0``, and every gain
stored as a DAC integer code is ``>= 0`` (its sign carries no meaning; phase is a separate field).
They hold for a device YAML and for a store write-back alike, since both compose through this
model. ``readout.trig_offset`` is also bounded above, at 10 µs (#195), with the wiring fallback.

**Gain ceilings (#190):** each per-element gain limit (``max_*_gain``, a fraction of full scale)
is bounded ``0 < limit <= cap``, with the role's hard cap from :mod:`~strumento.packs.cqed.safety`,
and carries the :data:`GAIN_CEILING_MARK` in its schema. :func:`ceiling_paths` reads the marks
back into the dotted paths the pack exports as ``CALIBRATION_CEILINGS``; the MCP session refuses
any write-back to them, so a ceiling is set by a human, never by an agent.

**Gain (D23):** the calibration DB keeps lab-native **int codes**, as tProc-v1 programs did. Conversion to the
v2 normalized **fraction** happens at the pack boundary via :func:`gain_int_to_frac` — the
compiler never sees an int code. ``maxv`` comes from the soccfg snapshot at factory time.

Pure: no ``qick`` import.
"""

from __future__ import annotations

import inspect
import typing
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strumento.core.wiring import TrigOffsetUs
from strumento.packs.cqed.safety import (
    DISPLACEMENT_GAIN_CAP,
    QUBIT_DRIVE_GAIN_CAP,
    READOUT_GAIN_CAP,
)

#: The schema key that marks a field as a gain ceiling (#190): ``"gain_ceiling": true`` in the
#: published JSON Schema, read back by :func:`ceiling_paths`.
GAIN_CEILING_MARK = "gain_ceiling"


def _gain_ceiling(cap: float, description: str) -> Any:
    """A per-element gain limit: optional, ``0 < limit <= cap``, and marked as a ceiling."""
    return Field(
        None, gt=0, le=cap, description=description, json_schema_extra={GAIN_CEILING_MARK: True}
    )


def _models_in(annotation) -> list[type[BaseModel]]:
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return [annotation]
    found: list[type[BaseModel]] = []
    for arg in typing.get_args(annotation):
        found.extend(_models_in(arg))
    return found


def _is_container(annotation) -> bool:
    """True for a ``dict``/``list`` annotation (possibly inside ``| None``): its keys are a ``*``."""
    if typing.get_origin(annotation) in (dict, list, tuple):
        return True
    return any(_is_container(arg) for arg in typing.get_args(annotation))


def ceiling_paths(model: type[BaseModel], prefix: str = "") -> tuple[str, ...]:
    """The dotted calibration paths of every field marked as a gain ceiling under ``model``.

    A dict or list section contributes a ``*`` segment (any key or index), e.g.
    ``modes.*.max_displace_gain``. Derived from the marks, so a new ceiling field is declared the
    moment it is marked.
    """
    out: list[str] = []
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        if extra.get(GAIN_CEILING_MARK) is True:
            out.append(path)
            continue
        for sub in _models_in(info.annotation):
            inner = f"{path}.*." if _is_container(info.annotation) else f"{path}."
            out.extend(p for p in ceiling_paths(sub, inner) if p not in out)
    return tuple(out)


def gain_int_to_frac(gain_int: float, maxv: float) -> float:
    """Convert a lab-native gain **int code** to a v2 normalized **fraction** (D23).

    ``maxv`` is the generator's full-scale code from the soccfg snapshot (e.g. 32766). This is
    the single conversion that catches the int-code-vs-fraction gain-units bug: code ``3289`` at
    ``maxv=32766`` is a fraction of ``≈0.1004``, not ``3289``.
    """
    return gain_int / maxv


class _Strict(BaseModel):
    #: ``allow_inf_nan=False``: no calibration value is NaN or infinite — a YAML ``.nan`` or a
    #: write-back of an unestimated fit fails here, located at its field (#191).
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PulseCal(_Strict):
    """One calibrated single-qubit pulse (gain as a lab-native int code, D23)."""

    gain: float = Field(
        ge=0,
        description="DAC integer code — pulse gain, converted to a fraction of full scale "
        "(code / maxv) when the pulse is built"
    )
    sigma: float = Field(
        ge=0,
        description="µs — Gaussian sigma (gauss), or the edge ramp sigma (flat_top)"
    )
    length: float = Field(
        0.0, ge=0, description="µs — flat-top length (flat_top only; 0 for gauss)"
    )
    type: str = Field("gauss", description="name or label — pulse shape: gauss or flat_top")


class QubitCal(_Strict):
    """Transmon calibration: transition frequencies (MHz) + the named pulse library."""

    f_ge: float = Field(description="MHz — qubit g–e transition frequency")
    f_ef: float | None = Field(None, description="MHz — qubit e–f transition frequency")
    T1: float | None = Field(None, ge=0, description="µs — qubit energy-relaxation time")
    T2: float | None = Field(None, ge=0, description="µs — qubit Ramsey dephasing time (T2*)")
    T2_echo: float | None = Field(
        None, ge=0, description="µs — qubit Hahn-echo coherence time")
    drag_alpha: float | None = Field(
        None,
        description="dimensionless — DRAG coefficient: peak of the derivative (Q) quadrature "
        "relative to the Gaussian (I) peak; unset means plain Gaussian g–e pulses",
    )
    pulses: dict[str, PulseCal] = Field(
        {}, description="pulse library — calibrated pulses keyed by name (pi_ge, hpi_ge, pi_ef, …)"
    )
    #: Per-element drive gain limit (v2 fraction). When set, overrides the module-level
    #: ``MAX_QUBIT_DRIVE_GAIN`` for this qubit — needed for multi-qubit systems with
    #: different amplifier chains.
    max_drive_gain: float | None = _gain_ceiling(
        QUBIT_DRIVE_GAIN_CAP,
        "fraction of full scale — largest drive gain this qubit may play; overrides the pack "
        f"default limit, up to the hard cap {QUBIT_DRIVE_GAIN_CAP}; set by a human, not over MCP",
    )


class ReadoutCal(_Strict):
    """Dispersive-readout calibration (frequency MHz; gain as a lab-native int code)."""

    frequency: float = Field(description="MHz — readout probe frequency")
    gain: float = Field(
        ge=0,
        description="DAC integer code — readout probe gain, converted to a fraction of full "
        "scale when the probe is built"
    )
    readout_length: float = Field(ge=0, description="µs — readout probe and acquisition window length")
    #: Bounded ``0 <= trig_offset <= 10`` µs by :data:`~strumento.core.wiring.TrigOffsetUs`
    #: (#195), the same type as the wiring fallback; a larger value is refused as a likely
    #: pre-#98 clock-cycle count.
    trig_offset: TrigOffsetUs = Field(
        description="µs — delay from probe start to the ADC trigger (time of flight)"
    )
    relax_delay: float = Field(
        ge=0, description="µs — wait after each shot before the next one starts")
    phase: float | None = Field(None, description="deg — readout probe drive phase")
    threshold: float | None = Field(
        None,
        description="ADC units — single-shot discrimination threshold on the rotated I axis",
    )
    angle: float | None = Field(
        None,
        description="rad — IQ rotation that puts the g/e separation along I",
    )
    #: g/e single-shot blob centres in RAW (I, Q), from ``ge-readout-blobs``. Stored as full
    #: 2-vectors and projected at use-time (robust to a discrimination-angle re-cal); consumed by
    #: ``DispersiveIQ.to_population`` to turn averaged IQ into excited-state population ``P_e``.
    #: ``None`` is a no-op (population conversion unavailable) — Wigner-parity normalization plan.
    g_center_iq: tuple[float, float] | None = Field(
        None, description="ADC units — raw (I, Q) centre of the |g> single-shot blob"
    )
    e_center_iq: tuple[float, float] | None = Field(
        None, description="ADC units — raw (I, Q) centre of the |e> single-shot blob"
    )
    #: Optional |f> raw-IQ centre from the three-level readout diagnostic. Together with the g/e
    #: centres it supports a conditioned affine conversion to (P_g, P_e, P_f) for f0-g1 transfer
    #: calibration. It is deliberately optional: ordinary g/e readout stays a two-level operation.
    f_center_iq: tuple[float, float] | None = Field(
        None, description="ADC units — raw (I, Q) centre of the |f> single-shot blob"
    )
    #: Per-element readout gain limit (v2 fraction). When set, overrides the module-level
    #: ``MAX_READOUT_GAIN``.
    max_readout_gain: float | None = _gain_ceiling(
        READOUT_GAIN_CAP,
        "fraction of full scale — largest readout gain allowed; overrides the pack default "
        f"limit, up to the hard cap {READOUT_GAIN_CAP}; set by a human, not over MCP",
    )


class SidebandPulseCal(_Strict):
    """One calibrated sideband pulse at a given photon number.

    A gaussian pulse has a ``sigma`` and no independently calibrated flat portion. A flat-top
    pulse has a positive ``length`` and ``ramp_sigma``. Keeping the two shapes explicit prevents
    a zero-length gaussian from being lowered as an invalid zero-cycle flat-top pulse.
    """

    frequency: float = Field(description="MHz — sideband drive frequency")
    gain: float = Field(
        ge=0,
        description="DAC integer code — sideband pulse gain, converted to a fraction of full "
        "scale when the pulse is built"
    )
    length: float = Field(0.0, ge=0, description="µs — flat-top length (flat_top only)")
    sigma: float | None = Field(None, ge=0, description="µs — Gaussian sigma (gauss only)")
    ramp_sigma: float | None = Field(
        None, ge=0, description="µs — flat-top edge ramp sigma (flat_top only)"
    )
    type: Literal["gauss", "flat_top"] = Field(
        "flat_top", description="name or label — pulse shape: gauss or flat_top"
    )
    #: Direct g/e/f readout result at the selected f0-g1 point. These are recorded by the
    #: fixed-gain length Rabi after its quality gate; Fock preparation refuses an unvalidated rung.
    transfer_g_population: float | None = Field(
        None,
        description="dimensionless — measured |g> population (0 to 1) after the transfer",
    )
    transfer_initial_f_population: float | None = Field(
        None,
        description="dimensionless — measured |f> population (0 to 1) before the transfer",
    )
    transfer_residual_excitation: float | None = Field(
        None,
        description="dimensionless — excited population (0 to 1) left after the transfer",
    )
    #: The f0-g1 Rabi measurement maps residual |f,0> to |e,0> before readout. These
    #: fields record that mapped reference and the residual mapped-e signal at the selected swap.
    transfer_initial_mapped_e_population: float | None = Field(
        None,
        description="dimensionless — reference |e> population (0 to 1) with |f> mapped to |e> "
        "before the transfer",
    )
    transfer_residual_mapped_e_population: float | None = Field(
        None,
        description="dimensionless — residual |e> population (0 to 1) with |f> mapped to |e> "
        "at the selected swap",
    )

    @model_validator(mode="after")
    def _validate_shape(self):
        if self.type == "gauss":
            if self.sigma is None or self.sigma <= 0:
                raise ValueError("gauss sideband pulses require sigma > 0")
            if self.length != 0:
                raise ValueError("gauss sideband pulses cannot set a flat length")
            if self.ramp_sigma is not None:
                raise ValueError("gauss sideband pulses use sigma, not ramp_sigma")
        else:
            if self.length <= 0:
                raise ValueError("flat_top sideband pulses require length > 0")
            if self.ramp_sigma is None:
                self.ramp_sigma = 0.005
            elif self.ramp_sigma <= 0:
                raise ValueError("flat_top sideband pulses require ramp_sigma > 0")
            if self.sigma is not None:
                raise ValueError("flat_top sideband pulses use ramp_sigma, not sigma")
        return self


class MultiphotonCal(_Strict):
    """A multiphoton-ladder transition: per-photon-number pulse calibrations."""

    pi: list[SidebandPulseCal] = Field(
        [], description="pulse list — π pulses, one per photon number (index n is the n-th rung)"
    )
    hpi: list[SidebandPulseCal] = Field(
        [], description="pulse list — π/2 pulses, one per photon number (index n is the n-th rung)"
    )


class SwapCal(_Strict):
    """One calibrated coupler beamsplitter swap (flat-top), keyed "A-B" in the instance."""

    frequency: float = Field(
        description="MHz — coupler drive frequency (also selects the coupler generator band)"
    )
    gain: float = Field(
        ge=0,
        description="DAC integer code — swap pulse gain, converted to a fraction of full scale "
        "when the pulse is built"
    )
    length: float = Field(
        ge=0,
        description="µs — flat-top length of a full (π) swap; a half swap uses half of it"
    )
    ramp_sigma: float = Field(0.005, ge=0, description="µs — flat-top edge ramp sigma")


class ModeCal(_Strict):
    """Per-bosonic-mode calibration: parity revival time(s) + an optional displacement pulse."""

    revival_time: list[Annotated[float, Field(ge=0)]] = Field(
        [],
        description="µs — parity revival waits, slow first then fast: [slow, fast, ...]",
    )
    displace: PulseCal | None = Field(
        None, description="displacement pulse — the calibrated displacement drive on this mode"
    )
    T1: float | None = Field(None, ge=0, description="µs — photon lifetime")
    T2: float | None = Field(None, ge=0, description="µs — mode dephasing time")
    chi: float | None = Field(
        None,
        description="MHz — qubit g–e dispersive shift per photon (photon n moves f_ge by -n·chi)",
    )
    chi_ef: float | None = Field(
        None, description="MHz — qubit e–f dispersive shift per photon"
    )
    #: MHz — AC-Stark shift the qubit sees during the parity wait. Consumed ONLY on the fast-parity
    #: path (``composite.parity(fast=True)``), where the second half-pi phase is advanced by
    #: ``360·revival_stark_shift·revival_time`` to undo the rotating→bare frame precession. The slow path
    #: applies no correction (its revival time is fit directly), so this is unused until a fast
    #: revival is calibrated (``revival_time[1]``).
    revival_stark_shift: float | None = Field(
        None,
        description="MHz — AC-Stark shift of the qubit during the fast parity wait (fast path only)",
    )
    #: Wigner/parity normalization scale — the reconstruction divides the measured parity by this
    #: 1.0 is a no-op, so existing runs are unaffected.
    alpha_scale: float = Field(
        1.0,
        description="dimensionless — parity normalization: the span of excited population "
        "between even and odd parity (1.0 = no rescaling)",
    )
    #: Wigner/parity sign — selects the excited<->parity mapping in ``parity_to_wigner``
    #: (``<Pi> = parity_sign*2(pe_midpoint - P_e)/alpha_scale``). +1.0 is the default convention (even
    #: parity -> low excited population); derived by ``parity_fringe`` from the fitted vacuum fringe.
    parity_sign: float = Field(
        1.0,
        description="dimensionless — parity sign, +1 or -1 (+1: even parity reads as low "
        "excited population)",
    )
    #: Wigner/parity fringe midpoint — the vacuum parity fringe's DC level (fitted ``offset`` on the
    #: population path). Single-phase parity has no offset cancellation, so a readout/relaxation
    #: asymmetry shifts this off 0.5; the reconstruction references the parity zero here rather than
    #: assuming 0.5. 0.5 is the offset-free default (and reduces the map to the old ``1 - 2 P_e``).
    pe_midpoint: float = Field(
        0.5,
        description="dimensionless — excited population (0 to 1) that reads as zero parity",
    )
    #: MHz per |α| — frequency→displacement map for Stark/frequency-based displacement.
    #: Not yet consumed; carried for provenance and future use.
    freq_to_alpha: float | None = Field(
        None,
        description="MHz per |α| — frequency-to-displacement map (recorded; not yet used by "
        "any experiment)",
    )
    displacement_cal: float | None = Field(
        None,
        description="DAC integer code per |α| — displacement gain needed per unit of |α|",
    )
    #: Per-mode displacement gain limit (v2 fraction). When set, overrides the module-level
    #: ``MAX_DISPLACEMENT_GAIN``.
    max_displace_gain: float | None = _gain_ceiling(
        DISPLACEMENT_GAIN_CAP,
        "fraction of full scale — largest displacement gain allowed on this mode; overrides the "
        f"pack default limit, up to the hard cap {DISPLACEMENT_GAIN_CAP}; set by a human, not "
        "over MCP",
    )


class PiPulseReference:
    """Physical drive amplitude (MHz Rabi) ↔ v2 gain fraction, via a calibrated π pulse (D17).

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
    """The device instance's calibration values (calib VALUES layer, D21)."""

    qubit: QubitCal = Field(description="qubit section — transmon frequencies, coherence, pulses")
    readout: ReadoutCal = Field(description="readout section — the dispersive readout probe")
    #: interaction strengths (MHz) for typed coupling edges, keyed "elementA-elementB" (D31);
    #: the strength stays *data* here, never on the edge declaration.
    couplings: dict[str, float] = Field(
        {},
        description="MHz — interaction strength per typed coupling edge, keyed "
        '"elementA-elementB"',
    )
    #: multiphoton ladder (f0g1 &c.), keyed by transition name (e.g. "f0-g1").
    multiphoton: dict[str, MultiphotonCal] = Field(
        {}, description='ladder section — multiphoton transitions keyed by name (e.g. "f0-g1")'
    )
    #: coupler beamsplitter swaps, keyed "A-B" (e.g. "cavity-S1").
    swaps: dict[str, SwapCal] = Field(
        {}, description='swap section — coupler swaps keyed "A-B" (e.g. "cavity-S1")'
    )
    #: per-mode calibration (revival times, displacement pulse), keyed by element name.
    modes: dict[str, ModeCal] = Field(
        {}, description="mode section — per-bosonic-mode calibration keyed by element name"
    )
