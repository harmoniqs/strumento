"""Device-level composites (slices 5–6): ``parity`` and ``prepare``.

Composites live on ``dev`` conceptually (they span elements) and take element *names* as
arguments (D21) — they are plain functions over the device, returning a :class:`Seq`.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from strumento.core.pulses import Seq, from_solution


class Fock:
    """A Fock-state preparation target: ``Fock(1)``."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __repr__(self) -> str:
        return f"Fock({self.n})"


def parity(dev, mode: str = "manipulate", *, second_phase=180.0, fast: bool = False) -> Seq:
    """The photon-parity mapping sequence: ``hpi · wait(revival) · hpi(second_phase)``.

    The wait is the named mode's calibrated parity revival time (π/χ); ``fast=True`` picks the
    second calibrated revival entry (e.g. an AC-Stark-shifted fast revival) when present.
    ``second_phase`` may be a :class:`~strumento.core.sweeps.Sweep` (a parity-fringe phase scan).
    """
    mode_cal = dev.calib.modes.get(mode)
    if mode_cal is None or not mode_cal.revival_time:
        raise KeyError(f"no revival_time calibration for mode {mode!r}")
    idx = 1 if fast else 0
    if idx >= len(mode_cal.revival_time):
        raise IndexError(f"mode {mode!r} has no fast revival calibrated")
    revival = mode_cal.revival_time[idx]

    q = dev.qubit
    return Seq().play(q.ge_hpi()).wait(revival).play(q.ge_hpi(phase=second_phase))


def prepare(dev, target, *, via: str = "analytic", solution=None, drive_map=None, gain_calib=None, dac_rate=None) -> Seq:
    """State preparation addressed by *what it does* (D17): analytic ladder or OC import.

    ``via="analytic"`` builds the multiphoton ladder (currently ``Fock(1)``:
    ``ge_pi · ef_pi · f0g1``). ``via="oc"`` imports a solved waveform through
    :func:`~strumento.core.pulses.from_solution` (pass ``solution``/``drive_map``/``gain_calib``).
    """
    if via == "oc":
        if solution is None or drive_map is None or gain_calib is None:
            raise ValueError("via='oc' needs solution=, drive_map=, and gain_calib=")
        return from_solution(solution, drive_map=drive_map, gain_calib=gain_calib, dac_rate=dac_rate)

    if isinstance(target, Fock) and target.n == 1:
        q = dev.qubit
        return Seq().play(q.ge_pi()).play(q.ef_pi()).play(dev.sideband.f0g1())
    raise NotImplementedError(
        f"analytic preparation for {target!r} is not implemented (Fock(1) is; "
        f"or pass via='oc' with a solved pulse)"
    )
