"""``ReadoutOps`` — readout probe pulse factories (cqed expansion).

Centralizes the readout probe construction that was previously inlined in four experiments
(resonator_spectroscopy, dispersive_shift, power_sweep, readout_optimization). The factory
reads from ``calib.readout``, resolves the readout line, converts gain from lab-native int
codes to v2 fractions (D23), and validates against ``MAX_READOUT_GAIN``.

Bound to the ``ReadoutResonator`` element via ``ops_cls``, so ``dev.readout.probe()`` works
like ``dev.qubit.ge_pi()`` — named-element ops in physical units, zero channel numbers.

Pure: no ``qick`` import.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from strumento.core.pulses import Const, Pulse, ReadoutBinding, Seq
from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.library._gain import gain_out, maxv_for
from strumento.packs.cqed.safety import MAX_READOUT_GAIN, validate_gain, validate_gain_sweep

if TYPE_CHECKING:  # elements.py names the ops classes: import cycle guard
    from strumento.core.device import Device
    from strumento.packs.cqed.elements import ReadoutResonator


class ReadoutOps:
    """Readout probe factories bound to the named readout resonator element."""

    def __init__(self, device: Device, element: ReadoutResonator) -> None:
        self.device = device
        self.element = element

    def probe(self, *, freq=None, gain=None, length=None, line=None, start_us=None,
              readout_phase_deg=0.0) -> Pulse:
        """A readout probe pulse from ``calib.readout``.

        Parameters
        ----------
        freq : float or Sweep, optional
            Probe frequency (MHz). Defaults to ``calib.readout.frequency``.
            Pass a ``Sweep`` for resonator spectroscopy.
        gain : float or Sweep, optional
            Override the calibrated readout gain; pass a ``Sweep`` for power sweeps. The
            gain units: see "Gain units" in :mod:`strumento.packs.cqed.library`.
        length : float or Sweep, optional
            Probe pulse length (µs). Defaults to ``calib.readout.readout_length``.
            Pass a ``Sweep`` for readout optimization.
        line : LineRef or tuple, optional
            Override the readout line. Defaults to ``("readout", "probe")``.
        start_us : float or None, optional
            Cursor-relative probe offset. ``0.0`` co-times the subsequent Measure trigger with
            the probe start instead of advancing past the pulse.
        readout_phase_deg : float, optional
            DDC demodulation phase (degrees) programmed into this probe's readout config
            (``ReadoutBinding.phase_deg`` -> ``add_readoutconfig(phase=...)``). Defaults to
            ``0.0``. Distinct from the probe *drive* phase (``calib.readout.phase``): this rotates
            the demodulated ``(I, Q)`` so the g/e separation lands on the branch's raw quadrature,
            since the on-board ``read_and_jump`` branch cannot rotate (active reset, trap #1).

        Returns
        -------
        Pulse
            A ``Const``-envelope readout probe pulse with validated gain.
        """
        ro = self.device.calib.readout
        ref_line = line if line is not None else LineRef("readout", "probe")
        if not isinstance(ref_line, LineRef):
            ref_line = LineRef(*ref_line)

        freq_mhz = freq if freq is not None else ro.frequency
        freq_scalar = freq_mhz.start if isinstance(freq_mhz, Sweep) else freq_mhz
        maxv = maxv_for(self.device, ref_line, freq_scalar)

        # Per-element limit (from calibration YAML) overrides the module-level constant.
        limit = ro.max_readout_gain if ro.max_readout_gain is not None else MAX_READOUT_GAIN

        # Safety: a gain sweep (e.g. PowerSweep) is validated at its raw endpoints against the
        # same limit a scalar gain faces (before this check, a pack-wide gap let the
        # default 0->30000 sweep, fraction ~0.916, past the 0.6 readout ceiling silently).
        if isinstance(gain, Sweep):
            validate_gain_sweep(gain, limit=limit, maxv=maxv, role="readout")
        resolved_gain = gain_out(ro.gain, gain, maxv)

        # Validate resolved scalar gains against the readout limit.
        if not isinstance(resolved_gain, Sweep):
            validate_gain(resolved_gain, limit=limit,
                          role="readout", label="ro_probe")

        length_us = length if length is not None else ro.readout_length
        adc_ch = self.device.wiring.readout.adc_chs[0]
        snapshot = getattr(self.device, "soccfg_snapshot", {}) or {}
        readouts = snapshot.get("readouts", [])
        dynamic = adc_ch < len(readouts) and "tproc_ctrl" in readouts[adc_ch]
        binding = ReadoutBinding(adc_ch=adc_ch, window_us=length_us, phase_deg=readout_phase_deg) if dynamic and not isinstance(length_us, Sweep) else None

        return Pulse(
            line=ref_line,
            freq_mhz=freq_mhz,
            gain=resolved_gain,
            envelope=Const(length_us=length_us),
            phase_deg=ro.phase,
            start_us=start_us,
            label="ro_probe",
            readout=binding,
        )

    def dispersive_measure(self, *, freq=None, gain=None, length=None, line=None,
                            readout_phase_deg=0.0, kind: str | None = None, **params: object) -> Seq:
        """Drive the readout probe, then measure -- one call instead of two (issue #99).

        Equivalent to ``Seq().play(self.probe(freq=freq, gain=gain, length=length,
        line=line, start_us=0.0)).measure(kind=kind, **params)`` for a scalar readout window:
        every argument is forwarded straight
        through to :meth:`probe` / :meth:`~strumento.core.pulses.Seq.measure`, so this carries
        no defaulting logic of its own -- ``probe()`` remains the one place that resolves
        calibrated frequency/gain/length and the default readout line (Key Decision, #99).

        Parameters
        ----------
        freq, length, line, readout_phase_deg : see :meth:`probe`.
        gain : see "Gain units" in :mod:`strumento.packs.cqed.library` (forwarded to
            :meth:`probe`).
        kind : str, optional
            Readout kind name, forwarded to :meth:`~strumento.core.pulses.Seq.measure`.
        **params
            Extra measure options, forwarded to
            :meth:`~strumento.core.pulses.Seq.measure`.

        Returns
        -------
        Seq
            A two-op sequence: the probe pulse, then the measure -- composable onto an
            existing sequence via ``existing_seq + dev.readout.dispersive_measure()``.
        """
        probe = self.probe(freq=freq, gain=gain, length=length, line=line, start_us=0.0,
                           readout_phase_deg=readout_phase_deg)
        if probe.readout is None:
            # QICK cannot sweep a dynamically declared integration window in one program.
            probe = dataclasses.replace(probe, start_us=None)
        return Seq().play(probe).measure(kind=kind, **params)
