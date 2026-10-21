"""Pure sequence and analysis helpers for a sparse free-Kerr cat witness."""

from __future__ import annotations

import math

import numpy as np

from strumento.core.pulses import Seq
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.library.composite import parity
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range
from strumento.packs.cqed.tomography import parity_to_wigner


def ring_points(*, radius: float, count: int) -> np.ndarray:
    """Return evenly spaced sampled Wigner coordinates on a phase-space ring."""
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius must be finite and positive; got {radius}")
    if int(count) < 3:
        raise ValueError(f"count must be at least 3; got {count}")
    return float(radius) * np.exp(2j * np.pi * np.arange(int(count)) / int(count))


def witness_sequence(
    device,
    *,
    sample_alpha: complex,
    kerr_wait: float | Sweep,
    init_alpha: float,
    mode_name: str,
) -> tuple[Seq, tuple]:
    """Prepare a coherent state, free-evolve it, then sample one Wigner coordinate.

    ``sample_alpha`` is the Wigner coordinate, so displaced parity must apply
    ``D(-sample_alpha)`` before the parity map. The runner owns host batching over coordinates;
    this helper owns one compiled delay row and remains pure/QICK-free.
    """
    if not math.isfinite(init_alpha) or init_alpha <= 0.0:
        raise ValueError(f"init_alpha must be finite and positive; got {init_alpha}")
    sample = complex(sample_alpha)
    if not (math.isfinite(sample.real) and math.isfinite(sample.imag)):
        raise ValueError(f"sample_alpha must be finite; got {sample_alpha}")
    if isinstance(kerr_wait, Sweep):
        validate_sweep_range(kerr_wait, min_val=MIN_TIME_SWEEP, max_val=MAX_TIME_SWEEP, label="kerr_wait")
    elif not math.isfinite(float(kerr_wait)) or float(kerr_wait) < 0.0:
        raise ValueError(f"kerr_wait must be finite and non-negative; got {kerr_wait}")

    mode = getattr(device, mode_name)
    applied_beta = -sample
    seq = Seq().play(mode.displace_alpha(init_alpha))
    if isinstance(kerr_wait, Sweep) or float(kerr_wait) > 0.0:
        seq = seq.wait(kerr_wait)
    seq = seq.play(
        mode.displace_alpha(
            abs(applied_beta),
            phase=float(np.degrees(np.angle(applied_beta)) % 360.0),
        )
    )
    seq += parity(device, mode_name)
    return seq, ()


def analyze_witness(
    *,
    delays,
    sample_alphas,
    control_iq,
    sweep_iq,
    readout,
    mode_cal,
) -> dict[str, object]:
    """Convert a ring of raw displaced-parity IQ rows into a timing witness.

    The contrast across coordinates is intentionally a witness rather than a fidelity. A null
    result remains meaningful and must be persisted by the host runner.
    """
    x = np.asarray(delays, dtype=float)
    samples = np.asarray(sample_alphas, dtype=complex)
    control = np.asarray(control_iq, dtype=float)
    sweeps = np.asarray(sweep_iq, dtype=float)
    if x.ndim != 1 or x.size < 3 or not np.all(np.isfinite(x)):
        raise ValueError("delays must be a finite one-dimensional array with at least three points")
    if samples.ndim != 1 or samples.size < 3 or not np.all(np.isfinite(samples.real)) or not np.all(np.isfinite(samples.imag)):
        raise ValueError("sample_alphas must be a finite one-dimensional ring with at least three points")
    if control.shape != (samples.size, 2):
        raise ValueError(f"control_iq must have shape {(samples.size, 2)}; got {control.shape}")
    if sweeps.shape != (samples.size, x.size, 2):
        raise ValueError(
            f"sweep_iq must have shape {(samples.size, x.size, 2)}; got {sweeps.shape}"
        )

    alpha_scale = float(getattr(mode_cal, "alpha_scale", 1.0) or 1.0)
    parity_sign = float(getattr(mode_cal, "parity_sign", 1.0) or 1.0)
    pe_midpoint = float(getattr(mode_cal, "pe_midpoint", 0.5))
    control_wigner = parity_to_wigner(
        readout.to_population(control),
        alpha_scale=alpha_scale,
        parity_sign=parity_sign,
        pe_midpoint=pe_midpoint,
    )
    sweep_wigner = parity_to_wigner(
        readout.to_population(sweeps),
        alpha_scale=alpha_scale,
        parity_sign=parity_sign,
        pe_midpoint=pe_midpoint,
    )
    baseline_contrast = float(np.ptp(control_wigner))
    ring_contrast = np.ptp(sweep_wigner, axis=0)
    excess_contrast = ring_contrast - baseline_contrast
    best_index = int(np.argmax(excess_contrast))
    return {
        "delays_us": x,
        "sample_alphas": samples,
        "control_wigner": control_wigner,
        "sweep_wigner": sweep_wigner,
        "baseline_ring_contrast": baseline_contrast,
        "ring_contrast": ring_contrast,
        "excess_ring_contrast": excess_contrast,
        "best_wait_us": float(x[best_index]),
        "best_wait_index": best_index,
        "peak_ring_contrast": float(ring_contrast[best_index]),
        "peak_excess_ring_contrast": float(excess_contrast[best_index]),
        "peak_is_interior": bool(0 < best_index < x.size - 1),
    }


def analyze_interleaved_witness(
    *,
    waits_us,
    sample_alphas,
    control_iq,
    witness_iq,
    readout,
    mode_cal,
) -> dict[str, object]:
    """Analyze matched zero-wait controls and witness samples acquired at each candidate wait.

    ``control_iq`` and ``witness_iq`` are indexed ``(wait, ring_point, I/Q)``. Subtracting each
    ring coordinate's immediately adjacent control removes the static spatial baseline before
    contrast is evaluated, unlike subtracting only a scalar baseline contrast.
    """
    waits = np.asarray(waits_us, dtype=float)
    samples = np.asarray(sample_alphas, dtype=complex)
    controls = np.asarray(control_iq, dtype=float)
    witness = np.asarray(witness_iq, dtype=float)
    if waits.ndim != 1 or waits.size < 3 or not np.all(np.isfinite(waits)):
        raise ValueError("waits_us must be a finite one-dimensional array with at least three points")
    if samples.ndim != 1 or samples.size < 3 or not np.all(np.isfinite(samples.real)) or not np.all(np.isfinite(samples.imag)):
        raise ValueError("sample_alphas must be a finite one-dimensional ring with at least three points")
    expected_shape = (waits.size, samples.size, 2)
    if controls.shape != expected_shape:
        raise ValueError(f"control_iq must have shape {expected_shape}; got {controls.shape}")
    if witness.shape != expected_shape:
        raise ValueError(f"witness_iq must have shape {expected_shape}; got {witness.shape}")

    alpha_scale = float(getattr(mode_cal, "alpha_scale", 1.0) or 1.0)
    parity_sign = float(getattr(mode_cal, "parity_sign", 1.0) or 1.0)
    pe_midpoint = float(getattr(mode_cal, "pe_midpoint", 0.5))
    control_wigner = parity_to_wigner(
        readout.to_population(controls),
        alpha_scale=alpha_scale,
        parity_sign=parity_sign,
        pe_midpoint=pe_midpoint,
    )
    witness_wigner = parity_to_wigner(
        readout.to_population(witness),
        alpha_scale=alpha_scale,
        parity_sign=parity_sign,
        pe_midpoint=pe_midpoint,
    )
    corrected_wigner = witness_wigner - control_wigner
    control_contrast = np.ptp(control_wigner, axis=1)
    witness_contrast = np.ptp(witness_wigner, axis=1)
    corrected_contrast = np.ptp(corrected_wigner, axis=1)
    best_index = int(np.argmax(corrected_contrast))
    return {
        "waits_us": waits,
        "sample_alphas": samples,
        "control_wigner": control_wigner,
        "witness_wigner": witness_wigner,
        "corrected_wigner": corrected_wigner,
        "control_ring_contrast": control_contrast,
        "witness_ring_contrast": witness_contrast,
        "corrected_ring_contrast": corrected_contrast,
        "best_wait_us": float(waits[best_index]),
        "best_wait_index": best_index,
        "peak_control_ring_contrast": float(control_contrast[best_index]),
        "peak_witness_ring_contrast": float(witness_contrast[best_index]),
        "peak_corrected_ring_contrast": float(corrected_contrast[best_index]),
        "peak_is_interior": bool(0 < best_index < waits.size - 1),
    }
