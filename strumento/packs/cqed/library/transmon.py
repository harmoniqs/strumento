"""``TransmonOps`` — the pulse library bound to a named transmon element (D21).

``dev.qubit`` is a ``TransmonOps(dev, <the transmon named "qubit">)`` (the binding is done by
``core.device.load``). Its factory methods return :class:`~strumento.core.pulses.Pulse` objects
built from the device's live calibration, so they always reflect the current calib.

**Gain (D23).** Calibration stores lab-native **int codes**; a factory converts them to the v2
normalized **fraction** at this boundary via :func:`gain_int_to_frac`, using ``maxv`` from the
device's soccfg snapshot. An explicit ``gain=`` override is taken as a **raw fraction** and used
as-is (the caller opting out of calibration speaks in v2 units directly).

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strumento.core.pulses import FlatTop, Gaussian, Pulse
from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import gain_int_to_frac

if TYPE_CHECKING:  # avoid an import cycle with elements.py at module load
    from strumento.packs.cqed.elements import Transmon

_DEFAULT_MAXV = 2**15 - 1  # fallback when no soccfg snapshot is attached


class TransmonOps:
    """Single-qubit pulse factories bound to one named transmon instance.

    Every factory accepts ``freq`` (MHz), ``gain`` (raw v2 fraction), ``phase`` (deg), and
    ``length`` (µs) overrides; ``None`` means "take it from calibration".
    """

    def __init__(self, device: object, element: "Transmon") -> None:
        self.device = device
        self.element = element

    # -- public factories --------------------------------------------------------------

    def ge_pi(self, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """A π pulse on the g–e transition."""
        return self._pulse("pi_ge", "f_ge", freq=freq, gain=gain, phase=phase, length=length)

    def ge_hpi(self, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """A π/2 pulse on the g–e transition."""
        return self._pulse("hpi_ge", "f_ge", freq=freq, gain=gain, phase=phase, length=length)

    def ef_pi(self, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """A π pulse on the e–f transition (frequency from ``f_ef``)."""
        return self._pulse("pi_ef", "f_ef", freq=freq, gain=gain, phase=phase, length=length)

    # -- internals ---------------------------------------------------------------------

    def _pulse(self, pulse_key, freq_field, *, freq, gain, phase, length) -> Pulse:
        qc = self.device.calib.qubit
        pc = qc.pulses[pulse_key]

        freq_mhz = freq if freq is not None else getattr(qc, freq_field)
        freq_scalar = freq_mhz.start if isinstance(freq_mhz, Sweep) else freq_mhz
        maxv = self._maxv(freq_scalar)

        # Gain units (D23): calib int code -> fraction (default); an explicit *scalar* is a raw v2
        # fraction (used as-is); an explicit *Sweep* is lab-native int codes -> a fraction sweep,
        # keeping the int-code axis as xpts (sweeps explore calibration in the units the lab uses).
        if gain is None:
            gain_out = gain_int_to_frac(pc.gain, maxv)
        elif isinstance(gain, Sweep):
            gain_out = gain.mapped(lambda code: gain_int_to_frac(code, maxv))
        else:
            gain_out = gain

        if pc.type == "flat_top":
            length_us = length if length is not None else pc.length
            envelope = FlatTop(ramp_sigma_us=pc.sigma, length_us=length_us)
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
