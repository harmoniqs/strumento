"""Small checks several cqed calibration experiments share — written once here.

* :class:`PassiveReset` — the wait before every block: ``reset_delay_us`` when set, else the
  calibrated readout ``relax_delay``.
* :func:`scoped_ops` — the transmon ops an experiment plays through, with its own pulse
  ``overrides`` (the device's ops are left untouched).
* :func:`is_population_anchored` — whether a readout has both g/e blob centres, so its signal can
  be read as an excited-state population.
* :func:`drive_gain_limit` — the qubit-drive gain limit: the calibrated ``max_drive_gain`` when
  set, else the pack ceiling.
* :func:`displace_gain_limit` — a mode's displacement gain limit: its calibrated
  ``max_displace_gain`` when set, else the pack ceiling.
* :func:`check_alpha_bounds` — refuse a DRAG sweep whose ``|alpha|`` exceeds the limit.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strumento.packs.cqed.safety import MAX_DISPLACEMENT_GAIN, MAX_QUBIT_DRIVE_GAIN


class PassiveReset:
    """Mixin for experiments that wait for the qubit to relax before every block.

    A measurement is not a reset, so each block starts with this wait. The experiment sets
    ``reset_delay_us`` in its constructor (``None`` means use the readout ``relax_delay``).
    """

    # No instance state of its own: the experiment owns ``reset_delay_us`` and ``device``.
    __slots__ = ()

    if TYPE_CHECKING:
        reset_delay_us: float | None
        device: Any

    def _reset_delay(self) -> float:
        if self.reset_delay_us is not None:
            return float(self.reset_delay_us)
        return float(self.device.calib.readout.relax_delay)


def scoped_ops(device, overrides: dict | None):
    """The transmon ops to play through: the device's own, or a copy with ``overrides`` applied.

    With no overrides this is ``device.qubit`` itself. Otherwise it is a fresh ``TransmonOps`` on
    the same qubit element, so the overrides apply to this experiment only.
    """
    if not overrides:
        return device.qubit
    from strumento.packs.cqed.library.transmon import TransmonOps

    return TransmonOps(device, device.qubit.element, overrides=overrides)


def is_population_anchored(readout) -> bool:
    """True when ``readout`` records both g and e blob centres (its signal is a population)."""
    return readout.g_center is not None and readout.e_center is not None


def drive_gain_limit(device) -> float:
    """The qubit-drive gain limit: the calibrated ``max_drive_gain``, else the pack ceiling."""
    qc = device.calib.qubit
    return qc.max_drive_gain if qc.max_drive_gain is not None else MAX_QUBIT_DRIVE_GAIN


def displace_gain_limit(device, mode: str) -> float:
    """A mode's displacement gain limit: its calibrated ``max_displace_gain``, else the pack ceiling."""
    mode_cal = device.calib.modes.get(mode)
    ceiling = mode_cal.max_displace_gain if mode_cal is not None else None
    return ceiling if ceiling is not None else MAX_DISPLACEMENT_GAIN


def check_alpha_bounds(alpha_start, alpha_stop, *, limit: float) -> None:
    """Raise ``ValueError`` when either end of a DRAG sweep has ``|alpha|`` above ``limit``.

    The start is checked first, so it is the one named when both are out of range.
    """
    for name, value in (("alpha_start", alpha_start), ("alpha_stop", alpha_stop)):
        if abs(float(value)) > limit:
            raise ValueError(
                f"{name}={value}: |alpha| > {limit} is outside the calibrated DRAG "
                "sweep range (gain headroom path)"
            )
