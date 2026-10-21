"""Characterization of the self-Kerr Ramsey analyzer's numerical outputs.

The analyzer's model evaluation is optimized for speed; these tests pin its derived values on a
small fixed synthetic dataset (captured from the scalar per-row implementation) and check the
parity model against an independent copy of that scalar formulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.packs.cqed.experiments.self_kerr_ramsey import (
    _coherent_state,
    _displacement,
    _parity_from_cache,
    analyze_self_kerr_ramsey,
    expected_parity,
)

# Captured from the per-row implementation before the vectorized model evaluation landed.
PINNED = {
    'success': True,
    'K_mhz': 0.003998307775180971,
    'K_stderr_mhz': 3.8019137070988024e-06,
    'detuning_mhz': -0.0079852692755263,
    'detuning_stderr_mhz': 1.2507629136969042e-05,
    't_cat_us': 125.05290440713235,
    'relative_K_stderr': 0.0009508807027559854,
    'K_interior': True,
    'K_alias_ambiguous': False,
    'competing_K_mhz': [],
    'valid_rows': 4,
    'total_rows': 4,
    'chisqr': 0.004727904258253641,
    'residual_rms': 0.006875975754940997,
    'reference_detuning_mhz': 0.025,
    'cutoff': 16,
    'row_0_alpha': 0.8,
    'row_0_return_phase_offset_deg': 0.0,
    'row_0_span': 1.185523494343975,
    'row_0_offset': 0.01962365109672759,
    'row_0_contrast': 0.900663219073414,
    'row_1_alpha': 1.4,
    'row_1_return_phase_offset_deg': 0.0,
    'row_1_span': 1.094245182203349,
    'row_1_offset': -0.031204039291113483,
    'row_1_contrast': 0.8580014154163884,
    'row_2_alpha': 1.4,
    'row_2_return_phase_offset_deg': 90.0,
    'row_2_span': 0.6577551975485232,
    'row_2_offset': 0.010258628557460403,
    'row_2_contrast': 0.7975863530135321,
    'row_3_alpha': 2.0,
    'row_3_return_phase_offset_deg': 45.0,
    'row_3_span': 0.9027018572696792,
    'row_3_offset': -1.5188985597713867e-05,
    'row_3_contrast': 0.7528288594741774,
}

# Floating-point summation order differs between the scalar and the batched model (about 1e-14
# relative per evaluation); the leastsq fit and its finite-difference Jacobian carry that into the
# fitted values. Every value must agree to 1e-9 relative. Per-row offsets and contrasts are in
# unit-scale parity and can sit near zero (row 3's offset is about -1.5e-5), so they also accept a
# 1e-12 absolute floor; the measured worst case there is 1.4e-13.
REL_TOL = 1.0e-9
PARITY_ABS_TOL = 1.0e-12


def _reference_parity_from_cache(
    coherent, return_displacement, delays, *, k_mhz, detuning_mhz, reference_detuning_mhz
):
    """The original scalar per-row parity formulation, kept here as an independent oracle."""
    n = np.arange(coherent.size, dtype=float)
    parity = (-1.0) ** n
    energy_mhz = detuning_mhz * n + 0.5 * k_mhz * n * (n - 1.0)
    free = coherent[:, None] * np.exp(-2j * np.pi * energy_mhz[:, None] * delays[None, :])
    phase = 2.0 * np.pi * reference_detuning_mhz * delays
    rotation = np.exp(-1j * n[:, None] * phase[None, :])
    final = np.conj(rotation) * (return_displacement @ (rotation * free))
    return np.real(np.sum(parity[:, None] * np.abs(final) ** 2, axis=0))


def _small_dataset():
    amplitudes = np.array([0.8, 1.4, 1.4, 2.0])
    offsets = np.array([0.0, 0.0, 90.0, 45.0])
    delays = np.linspace(0.0, 60.0, 25)
    ideal = np.stack(
        [
            expected_parity(
                alpha,
                delays,
                k_mhz=0.004,
                detuning_mhz=-0.008,
                reference_detuning_mhz=0.025,
                return_phase_offset_deg=offset,
                cutoff=16,
            )
            for alpha, offset in zip(amplitudes, offsets, strict=True)
        ]
    )
    ripple = 0.01 * np.sin(np.arange(amplitudes.size)[:, None] * 1.3 + delays[None, :] * 0.7)
    observed = (
        np.array([0.02, -0.03, 0.01, 0.0])[:, None]
        + np.array([0.9, 0.85, 0.8, 0.75])[:, None] * ideal
        + ripple
    )
    return amplitudes, offsets, delays, observed


def test_self_kerr_analyzer_outputs_match_pinned_characterization():
    amplitudes, offsets, delays, observed = _small_dataset()

    fit = analyze_self_kerr_ramsey(
        amplitudes,
        delays,
        observed,
        reference_detuning_mhz=0.025,
        return_phase_offsets_deg=offsets,
        cutoff=16,
        k_search_max_mhz=0.02,
    )

    assert set(fit) == set(PINNED)
    for key, expected in PINNED.items():
        if isinstance(expected, (bool, int, list)):
            assert fit[key] == expected, key
        else:
            parity_unit = key.endswith(("_offset", "_contrast"))
            abs_tol = PARITY_ABS_TOL if parity_unit else 0.0
            assert fit[key] == pytest.approx(expected, rel=REL_TOL, abs=abs_tol), key


@pytest.mark.parametrize("seed", range(5))
def test_parity_model_matches_scalar_reference_formulation(seed):
    rng = np.random.default_rng(seed)
    cutoff = int(rng.integers(6, 30))
    alpha = float(rng.uniform(0.2, 2.2))
    offset_deg = float(rng.uniform(-180.0, 180.0))
    delays = np.sort(rng.uniform(0.0, 120.0, size=int(rng.integers(3, 40))))
    coherent = _coherent_state(alpha, cutoff)
    return_displacement = _displacement(-alpha * np.exp(1j * np.radians(offset_deg)), cutoff)
    kwargs = {
        "k_mhz": float(rng.uniform(-0.02, 0.02)),
        "detuning_mhz": float(rng.uniform(-0.05, 0.05)),
        "reference_detuning_mhz": float(rng.uniform(-0.05, 0.05)),
    }

    got = _parity_from_cache(coherent, return_displacement, delays, **kwargs)
    want = _reference_parity_from_cache(coherent, return_displacement, delays, **kwargs)

    np.testing.assert_allclose(got, want, rtol=0.0, atol=1.0e-12)
