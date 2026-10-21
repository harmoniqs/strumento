"""Shared DRAG envelope builder — the single source of the Gaussian + derivative shape.

Both the DRAG *calibration* (``experiments/drag_calibration.py``) and the transmon *factory*
(``library/transmon.py``, when ``calib.qubit.drag_alpha`` is set) build their DRAG pulse here, so
the shape the calibration measures is byte-identical to the shape a gate plays (the only way the
calibrated coefficient is meaningful).

DRAG shape (Motzoi et al.): the in-phase channel is a Gaussian; the quadrature channel is the
Gaussian's time derivative scaled by ``alpha``, which cancels the leading leakage into ``|f⟩``.

Pure: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.pulses import Arb

#: The Gaussian window, in σ. Matches ``compiler._GAUSS_LENGTH_SIGMAS`` (4.0) so a DRAG ``Arb`` that
#: samples this window has the **same total duration** as the plain ``Gaussian(sigma)`` it replaces
#: — the "duration must not drift" invariant. Kept as a module constant, not a
#: magic 4, so the two stay locked together.
DRAG_WINDOW_SIGMAS = 4.0

#: Fallback sample count when a caller pins neither ``n_samples`` nor ``dac_rate`` (no soccfg snapshot
#: — e.g. a bare recording fake). A multiple of a typical ``samps_per_clk`` (16). NOT a physical
#: duration: an Arb plays its samples at the DAC rate, so hardware paths must pass ``dac_rate`` (64
#: samples is only 10 ns at 6.4 GS/s — a too-short calibration pulse).
_DEFAULT_WINDOW_SAMPLES = 64


def drag_arb(
    sigma_us: float,
    alpha: float,
    *,
    window_sigmas: float = DRAG_WINDOW_SIGMAS,
    n_samples: int | None = None,
    dac_rate: float | None = None,
) -> Arb:
    """A Gaussian + α·(dGaussian/dt) DRAG envelope as an :class:`Arb`, normalized to peak 1.

    I channel: Gaussian(t; σ), peak-normalized to 1.
    Q channel: ``alpha`` · dGaussian/dt, normalized so ``|Q_peak| = |alpha|`` (clamped to ±1), so
    the total envelope stays in ``[-1, 1]`` for ``|alpha| ≤ 1``.

    Duration: the window spans ``window_sigmas · σ``. An ``Arb`` plays its samples at the DAC's own
    rate, so to reproduce the ``Gaussian(σ)`` duration (``window_sigmas · σ``) the sample count must
    equal ``window_sigmas · σ · dac_rate``. Pass ``dac_rate`` (samples/µs) to pin that — the factory
    path does, so a DRAG gate has the **same length** as the Gaussian it replaces. ``n_samples``
    overrides the count directly (the calibration path, which is self-consistent at a fixed count
    and does not need the physical DAC duration). With neither, the default ``64``-sample window is
    used (the historical calibration shape).

    Parameters
    ----------
    sigma_us:
        Gaussian σ in µs.
    alpha:
        DRAG coefficient. ``alpha=0`` is a pure Gaussian; the sign follows the calibration's
        convention.
    window_sigmas:
        Window width in σ (default :data:`DRAG_WINDOW_SIGMAS`, matching the Gaussian lowering).
    n_samples:
        Explicit sample count (calibration path). Takes precedence over ``dac_rate``.
    dac_rate:
        Target DAC sample rate (samples/µs). Used to derive the sample count so the ``Arb`` duration
        equals ``window_sigmas · σ`` — the factory path. Ignored when ``n_samples`` is given.
    """
    idata, qdata = _drag_samples(sigma_us, alpha, window_sigmas, n_samples, dac_rate)

    # Clamp to [-1, 1] (the Arb → DAC path multiplies by maxv; |alpha|>1 saturates the Q channel).
    # Callers that must play |alpha| > 1 faithfully use :func:`drag_arb_with_headroom` instead.
    qdata = np.clip(qdata, -1.0, 1.0)

    return Arb(idata=tuple(idata.tolist()), qdata=tuple(qdata.tolist()))


def drag_arb_with_headroom(
    sigma_us: float,
    alpha: float,
    *,
    window_sigmas: float = DRAG_WINDOW_SIGMAS,
    n_samples: int | None = None,
    dac_rate: float | None = None,
) -> tuple[Arb, float]:
    """The UNCLIPPED DRAG envelope plus the gain multiplier that keeps it in ``[-1, 1]``.

    Returns ``(arb, gain_scale)`` with ``gain_scale = max(1, |alpha|)`` and ``arb`` = the
    :func:`drag_arb` I/Q samples divided by ``gain_scale`` (without the Q clamp). Playing ``arb`` at
    ``gain · gain_scale`` puts exactly ``gain · (G + iα·dĜ)`` on the qubit for any α — the DAC's
    gain headroom absorbs the large Q peak instead of the clamp distorting it (at α = 2 the clamp
    saturates ~80% of the Q samples). For ``|alpha| <= 1`` it is identical to :func:`drag_arb` with
    ``gain_scale == 1.0``. The caller owns validating the scaled gain against its safety limit.
    """
    idata, qdata = _drag_samples(sigma_us, alpha, window_sigmas, n_samples, dac_rate)
    gain_scale = max(1.0, abs(float(alpha)))
    if gain_scale > 1.0:
        idata, qdata = idata / gain_scale, qdata / gain_scale
    return Arb(idata=tuple(idata.tolist()), qdata=tuple(qdata.tolist())), gain_scale


def _drag_samples(sigma_us, alpha, window_sigmas, n_samples, dac_rate):
    """Unclipped (I, Q): peak-normalized Gaussian and α × peak-normalized derivative."""
    if n_samples is not None:
        n_pts = int(n_samples)
    elif dac_rate is not None:
        n_pts = max(2, int(round(window_sigmas * sigma_us * dac_rate)))
    else:
        n_pts = _DEFAULT_WINDOW_SAMPLES

    half = window_sigmas / 2.0
    t = np.linspace(-half * sigma_us, half * sigma_us, n_pts)

    # Gaussian, peak-normalized to 1.
    gauss = np.exp(-0.5 * (t / sigma_us) ** 2)
    idata = gauss / gauss.max()

    # Derivative of the Gaussian, normalized to peak 1, then scaled by alpha.
    dgauss = -(t / sigma_us**2) * gauss
    dgauss_max = np.max(np.abs(dgauss))
    if dgauss_max > 0:
        qdata = alpha * dgauss / dgauss_max
    else:
        qdata = np.zeros_like(idata)
    return idata, qdata
