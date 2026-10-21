"""``TransmonOps`` — the pulse library bound to a named transmon element (D21).

``dev.qubit`` is a ``TransmonOps(dev, <the transmon named "qubit">)`` (the binding is done by
``core.device.load``). Its factory methods return :class:`~strumento.core.pulses.Pulse` objects
built from the device's live calibration, so they always reflect the current calib.

**Gain (D23).** Calibration stores lab-native **int codes**; a factory converts them to the v2
normalized **fraction** at this boundary via :func:`gain_int_to_frac`, using ``maxv`` from the
device's soccfg snapshot. An explicit ``gain=`` override is taken as a **raw fraction** and used
as-is (the caller opting out of calibration speaks in v2 units directly); a ``gain=`` *Sweep*
is in int codes. All three are spelled out under "Gain units" in
:mod:`strumento.packs.cqed.library`.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strumento.core.pulses import FlatTop, Gaussian, Pulse
from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import gain_int_to_frac
from strumento.packs.cqed.library._drag import drag_arb
from strumento.packs.cqed.safety import MAX_QUBIT_DRIVE_GAIN, validate_gain, validate_gain_sweep

if TYPE_CHECKING:  # avoid an import cycle with elements.py at module load
    from strumento.core.device import Device
    from strumento.core.pulses import Envelope
    from strumento.packs.cqed.elements import Transmon

_DEFAULT_MAXV = 2**15 - 1  # fallback when no soccfg snapshot is attached

#: The ge pulse keys DRAG applies to. ef DRAG is not implemented — it needs a different
#: coefficient and the opposite-sign anharmonicity — so ef pulses keep their Gaussian shape here.
_DRAG_GE_KEYS = frozenset({"pi_ge", "hpi_ge"})


class TransmonOps:
    """Single-qubit pulse factories bound to one named transmon instance.

    Every factory accepts ``freq`` (MHz), ``gain`` (see "Gain units" in
    :mod:`strumento.packs.cqed.library`), ``phase`` (deg), and ``length`` (µs) overrides;
    ``None`` means "take it from calibration".

    ``overrides`` is an optional dict mapping pulse keys (``"pi_ge"``, ``"hpi_ge"``,
    ``"pi_ef"``, ``"hpi_ef"``) to pre-built ``Pulse`` objects — typically OC-imported pulses from
    :func:`~strumento.core.pulses.from_solution`. When set, the factory returns the override
    (with the caller's freq/gain/phase applied) instead of building from calibration; the
    override's own gain faces the drive limit like every other pulse. The override plays at the
    cursor like the calibrated gate (its ``start_us`` — ``from_solution``'s in-segment offset — is
    dropped), so a pair of override gates is sequential. This is the near-term path for
    arbitrary waveform substitution (approach B in the architecture plan); the long-term path
    (approach A) stores the ``Arb`` envelope in the calibration YAML.
    """

    def __init__(self, device: Device, element: Transmon,
                 *, overrides: dict[str, Pulse] | None = None) -> None:
        self.device = device
        self.element = element
        self.overrides = overrides or {}

    # -- public factories --------------------------------------------------------------

    def ge_pi(self, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """A π pulse on the g–e transition.

        gain : see "Gain units" in :mod:`strumento.packs.cqed.library`.
        """
        return self._pulse("pi_ge", "f_ge", freq=freq, gain=gain, phase=phase, length=length)

    def ge_hpi(self, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """A π/2 pulse on the g–e transition.

        gain : see "Gain units" in :mod:`strumento.packs.cqed.library`.
        """
        return self._pulse("hpi_ge", "f_ge", freq=freq, gain=gain, phase=phase, length=length)

    def ef_pi(self, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """A π pulse on the e–f transition (frequency from ``f_ef``).

        gain : see "Gain units" in :mod:`strumento.packs.cqed.library`.
        """
        return self._pulse("pi_ef", "f_ef", freq=freq, gain=gain, phase=phase, length=length)

    def ef_hpi(self, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """A π/2 pulse on the e–f transition (frequency from ``f_ef``).

        gain : see "Gain units" in :mod:`strumento.packs.cqed.library`.
        """
        return self._pulse("hpi_ef", "f_ef", freq=freq, gain=gain, phase=phase, length=length)

    # -- internals ---------------------------------------------------------------------

    def _pulse(self, pulse_key, freq_field, *, freq, gain, phase, length) -> Pulse:
        qc = self.device.calib.qubit

        # Pulse override: the stored Pulse, with the caller's freq/gain/phase applied
        # under the SAME semantics and safety limits as the calibrated path. This
        # used to apply only phase, so ErrorAmplification's freq=/gain= sweeps swept nothing (a flat
        # "perfectly robust" fringe) and AllXY's identity gate ge_pi(gain=0.0) played a full π.
        if pulse_key in self.overrides:
            from dataclasses import replace
            override = self.overrides[pulse_key]
            if length is not None:
                raise ValueError(
                    f"{pulse_key} is a pulse override (a fixed Arb); a length={length} override "
                    "cannot be honoured — supply an override at the new duration instead"
                )
            freq_mhz = freq if freq is not None else override.freq_mhz
            freq_scalar = freq_mhz.start if isinstance(freq_mhz, Sweep) else freq_mhz
            if gain is None:
                gain_out = override.gain
                self._check_override_gain(gain_out, pulse_key)  # the override faces the limit too
            else:
                gain_out = self._resolve_gain(gain, pulse_key, self._maxv(freq_scalar))
            # Placement: a gate plays at the cursor. An OC import (``from_solution``) carries
            # ``start_us=0.0`` — an offset *inside* its parallel segment — and kept here it would
            # make the compiler place the gate absolutely without advancing the cursor, stacking
            # both gates of a pair and the readout at the pair's start (#210). Drop it, so the
            # override is sequenced exactly like the calibrated gate it stands in for.
            return replace(override, freq_mhz=freq_mhz, gain=gain_out, phase_deg=phase,
                           start_us=None)

        pc = qc.pulses[pulse_key]

        freq_mhz = freq if freq is not None else getattr(qc, freq_field)
        freq_scalar = freq_mhz.start if isinstance(freq_mhz, Sweep) else freq_mhz
        maxv = self._maxv(freq_scalar)

        if gain is None:
            gain_out = gain_int_to_frac(pc.gain, maxv)
            self._resolve_gain(gain_out, pulse_key, maxv)  # validate the calibrated value too
        else:
            gain_out = self._resolve_gain(gain, pulse_key, maxv)

        envelope: Envelope
        if pc.type == "flat_top":
            length_us = length if length is not None else pc.length
            envelope = FlatTop(ramp_sigma_us=pc.sigma, length_us=length_us)
        elif qc.drag_alpha is not None and pulse_key in _DRAG_GE_KEYS:
            # DRAG-corrected ge pulse: emit the Gaussian + α·(dGaussian/dt) shape
            # instead of a bare Gaussian, using the SAME shared builder the calibration measured with
            # (so the shipped shape is the measured shape). Sample at the drive DAC rate so the Arb
            # spans the same 4σ window — and therefore the same duration — as the Gaussian it
            # replaces (no timing drift). Gated on drag_alpha being set: None → the Gaussian branch
            # below, i.e. today's behavior, so no shipped device moves until it opts in by calibrating.
            if abs(qc.drag_alpha) > 1.0:
                # drag_arb clamps Q at ±1 there, so the gate would NOT be the pulse the calibration
                # measured (which plays |α| > 1 through gain headroom). Refuse rather than ship it.
                raise ValueError(
                    f"drag_alpha={qc.drag_alpha}: |drag_alpha| > 1 would ship a Q-clipped DRAG pulse "
                    "that differs from the calibrated one; recalibrate or extend the factory's "
                    "gain-headroom path"
                )
            envelope = drag_arb(pc.sigma, qc.drag_alpha, dac_rate=self._dac_rate(freq_scalar))
        else:  # gaussian
            envelope = Gaussian(sigma_us=pc.sigma)

        return Pulse(
            line=LineRef(self.element.name, "drive"),
            freq_mhz=freq_mhz,
            gain=gain_out,
            phase_deg=phase,
            envelope=envelope,
            label=pulse_key,
        )

    def _resolve_gain(self, gain, pulse_key: str, maxv: float):
        """Caller gain → IR gain, validated (D23) — shared by the calibrated and override paths.

        An explicit *scalar* is a raw v2 fraction (used as-is); an explicit *Sweep* is lab-native
        int codes → a fraction sweep, keeping the int-code axis as xpts (sweeps explore calibration
        in the units the lab uses). Either faces the per-element ``max_drive_gain`` (or the module
        default): a Rabi-style sweep may cross the π point but must stay within the safe range.
        """
        qc = self.device.calib.qubit
        limit = qc.max_drive_gain if qc.max_drive_gain is not None else MAX_QUBIT_DRIVE_GAIN
        if isinstance(gain, Sweep):
            validate_gain_sweep(gain, limit=limit, maxv=maxv, role="qubit_drive")
            return gain.mapped(
                lambda code: gain_int_to_frac(code, maxv),
                inverse=lambda frac: frac * maxv,
            )
        validate_gain(gain, limit=limit, role="qubit_drive", label=pulse_key)
        return gain

    def _drive_limit(self) -> float:
        qc = self.device.calib.qubit
        return qc.max_drive_gain if qc.max_drive_gain is not None else MAX_QUBIT_DRIVE_GAIN

    def _check_override_gain(self, gain, pulse_key: str) -> None:
        """An override pulse's own gain against the drive limit (``gain=None`` on an override).

        The override is a ``Pulse``, so its gain is already in the IR's unit, a fraction of full
        scale — a scalar is checked as is, like the calibrated fraction, and a ``Sweep`` is a
        fraction sweep whose endpoints are each checked as fractions (no int-code conversion:
        :func:`validate_gain_sweep` is for caller sweeps in DAC codes, not for IR sweeps).
        """
        limit = self._drive_limit()
        values = (gain.start, gain.stop) if isinstance(gain, Sweep) else (gain,)
        for value in values:
            validate_gain(value, limit=limit, role="qubit_drive", label=pulse_key)

    def _maxv(self, freq_mhz: float) -> float:
        """Full-scale gain code for the drive generator, from the soccfg snapshot.

        The drive generator is resolved through the wiring (so multi-generator lines pick the
        right one); ``maxv`` is looked up by ``tproc_ch``. Falls back to any gen's ``maxv``
        (uniform on this board) or a sane default when no snapshot is attached.
        """
        gen = self.device.wiring.resolve(LineRef(self.element.name, "drive"), freq_mhz)
        snap = self.device.soccfg_snapshot
        if not snap:
            return _DEFAULT_MAXV
        gens = snap.get("gens", [])
        for g in gens:
            if g.get("tproc_ch") == gen.dac_ch:
                return g["maxv"]
        return gens[0]["maxv"] if gens else _DEFAULT_MAXV

    def _dac_rate(self, freq_mhz: float) -> float | None:
        """Drive-generator sample rate (samples/µs) for a DRAG ``Arb``, from the soccfg snapshot.

        Numerically the generator's ``fs`` (MHz) is samples/µs. Resolved through the wiring by
        ``tproc_ch`` like :meth:`_maxv`. Returns ``None`` when no snapshot is attached (or the gen
        carries no ``fs``): the DRAG builder then falls back to its default sample count — a bare
        recording fake has no DAC rate to honor, and the shape (not the exact duration) is what a
        no-soccfg unit test asserts.
        """
        snap = self.device.soccfg_snapshot
        if not snap:
            return None
        gen = self.device.wiring.resolve(LineRef(self.element.name, "drive"), freq_mhz)
        for g in snap.get("gens", []):
            if g.get("tproc_ch") == gen.dac_ch:
                fs = g.get("fs")
                return float(fs) if fs is not None else None
        return None
