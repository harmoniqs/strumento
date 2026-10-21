"""Pure helpers for a host-batched self-Kerr Ramsey diagnostic.

The Runner owns host batching because each coherent amplitude is an independent
program. This module keeps the pulse ordering and later numerical fit testable
without QICK or a board.
"""

from __future__ import annotations

import math

import lmfit
import numpy as np
from scipy.linalg import expm
from scipy.special import gammaln

from strumento.core.pulses import Seq
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.library.composite import parity
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


def row_sequence(
    device,
    *,
    alpha: float,
    delays: Sweep,
    mode_name: str,
    reference_detuning_mhz: float,
    return_phase_offset_deg: float = 0.0,
) -> tuple[Seq, tuple]:
    """Build one amplitude row: displace, free-evolve, return, then map parity.

    The return displacement advances at a known virtual detuning on the same
    hardware loop as the free delay. The fitter separates that reference from
    the measured cavity detuning and self-Kerr term.
    """
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be finite and positive; got {alpha}")
    if not math.isfinite(reference_detuning_mhz) or reference_detuning_mhz == 0.0:
        raise ValueError("reference_detuning_mhz must be finite and non-zero")
    if not math.isfinite(return_phase_offset_deg):
        raise ValueError("return_phase_offset_deg must be finite")
    validate_sweep_range(delays, min_val=MIN_TIME_SWEEP, max_val=MAX_TIME_SWEEP, label="delay")

    return_phase = delays.mapped(
        lambda delay_us: 180.0 + return_phase_offset_deg + 360.0 * reference_detuning_mhz * delay_us,
        inverse=lambda phase_deg: (phase_deg - 180.0 - return_phase_offset_deg)
        / (360.0 * reference_detuning_mhz),
    )
    mode = getattr(device, mode_name)
    seq = (
        Seq()
        .play(mode.displace_alpha(alpha))
        .wait(delays)
        .play(mode.displace_alpha(alpha, phase=return_phase))
        + parity(device, mode_name)
    )
    return seq, ()


def _coherent_state(alpha: float, cutoff: int) -> np.ndarray:
    """Return the truncated coherent ket for a real, positive amplitude."""
    n = np.arange(cutoff, dtype=float)
    return np.exp(-0.5 * alpha**2 + n * np.log(alpha) - 0.5 * gammaln(n + 1.0)).astype(complex)


def _displacement(beta: complex, cutoff: int) -> np.ndarray:
    """Return a truncated Fock-basis displacement matrix."""
    annihilation = np.zeros((cutoff, cutoff), dtype=complex)
    levels = np.arange(1, cutoff, dtype=float)
    annihilation[np.arange(cutoff - 1), np.arange(1, cutoff)] = np.sqrt(levels)
    return expm(beta * annihilation.conj().T - np.conj(beta) * annihilation)


def _number_phase_kernel(
    cutoff: int,
    delays: np.ndarray,
    *,
    k_mhz: float,
    detuning_mhz: float,
    reference_detuning_mhz: float,
) -> np.ndarray:
    """Return ``exp(-2*pi*i*(E_n + reference_detuning*n)*delay)`` with shape (cutoff, delays).

    ``E_n = detuning*n + K*n*(n-1)/2`` (MHz) is the free evolution; the ``reference_detuning*n``
    term is the return pulse's number rotation ``R = exp(-i*phase*n)`` folded into the same phase.
    """
    n = np.arange(cutoff, dtype=float)
    rate_mhz = detuning_mhz * n + 0.5 * k_mhz * n * (n - 1.0) + reference_detuning_mhz * n
    return np.exp((-2j * np.pi) * np.outer(rate_mhz, delays))


def _parity_rows(operators: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Evaluate ideal parity rows from stacked ``D_r @ diag(coherent_r)`` operators.

    ``operators`` has shape (rows, cutoff, cutoff) and ``kernel`` (cutoff, delays). With
    ``D(-alpha*exp(i*phase)) = R† D(-alpha) R`` the final state is ``R† D (R * free)``; the outer
    ``R†`` is a pure phase per Fock level and drops out of ``|.|**2``, so one stacked matrix
    product evaluates every row at once.
    """
    rows, cutoff, _ = operators.shape
    final = (operators.reshape(rows * cutoff, cutoff) @ kernel).reshape(rows, cutoff, -1)
    populations = final.real**2 + final.imag**2
    parity = 1.0 - 2.0 * (np.arange(cutoff) % 2)
    return parity @ populations


def _parity_from_cache(
    coherent: np.ndarray,
    return_displacement: np.ndarray,
    delays: np.ndarray,
    *,
    k_mhz: float,
    detuning_mhz: float,
    reference_detuning_mhz: float,
) -> np.ndarray:
    """Evaluate the ideal parity row under the declared Kerr-frame convention.

    The free Hamiltonian is ``H / h = detuning*n + K*n*(n-1)/2`` in MHz. The return pulse has
    phase ``180 + 360*reference_detuning*delay`` degrees. Applying its time-dependent displacement
    through number rotations avoids re-exponentiating a matrix at every fit iteration.
    """
    kernel = _number_phase_kernel(
        coherent.size,
        delays,
        k_mhz=k_mhz,
        detuning_mhz=detuning_mhz,
        reference_detuning_mhz=reference_detuning_mhz,
    )
    return _parity_rows((return_displacement * coherent[None, :])[None, :, :], kernel)[0]


def expected_parity(
    alpha: float,
    delays,
    *,
    k_mhz: float,
    detuning_mhz: float,
    reference_detuning_mhz: float,
    return_phase_offset_deg: float = 0.0,
    cutoff: int = 40,
) -> np.ndarray:
    """Ideal self-Kerr Ramsey parity for one coherent amplitude.

    This intentionally uses the full truncated-Fock evolution rather than a large-amplitude
    approximation. It is both the synthetic oracle and the fit model, so the reported sign is tied
    directly to ``H / h = K*n*(n-1)/2``.
    """
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be finite and positive; got {alpha}")
    if cutoff < 4:
        raise ValueError(f"cutoff must be at least 4; got {cutoff}")
    if not math.isfinite(return_phase_offset_deg):
        raise ValueError("return_phase_offset_deg must be finite")
    x = np.asarray(delays, dtype=float)
    if x.ndim != 1 or x.size < 3 or not np.all(np.isfinite(x)):
        raise ValueError("delays must be a finite one-dimensional array with at least three points")
    coherent = _coherent_state(float(alpha), int(cutoff))
    return_displacement = _displacement(
        -float(alpha) * np.exp(1j * np.radians(float(return_phase_offset_deg))),
        int(cutoff),
    )
    return _parity_from_cache(
        coherent,
        return_displacement,
        x,
        k_mhz=float(k_mhz),
        detuning_mhz=float(detuning_mhz),
        reference_detuning_mhz=float(reference_detuning_mhz),
    )


def _failed_fit(reason: str, *, total_rows: int, valid_rows: int = 0) -> dict:
    """Return a uniform failed-fit payload that can be persisted beside precious hardware data."""
    return {
        "success": False,
        "reason": reason,
        "K_mhz": float("nan"),
        "K_stderr_mhz": float("nan"),
        "detuning_mhz": float("nan"),
        "detuning_stderr_mhz": float("nan"),
        "t_cat_us": float("nan"),
        "relative_K_stderr": float("inf"),
        "K_interior": False,
        "K_alias_ambiguous": True,
        "competing_K_mhz": [],
        "valid_rows": int(valid_rows),
        "total_rows": int(total_rows),
    }


def analyze_self_kerr_ramsey(
    amplitudes,
    delays,
    parity_map,
    *,
    reference_detuning_mhz: float,
    return_phase_offsets_deg=None,
    cutoff: int = 40,
    k_search_max_mhz: float = 0.02,
    detuning_search_max_mhz: float = 0.05,
    min_row_span: float = 0.08,
    min_valid_rows: int = 3,
) -> dict:
    """Fit a global signed self-Kerr coefficient across independently acquired amplitude rows.

    Each row receives its own parity offset and contrast, while ``K`` and the linear cavity-drive
    detuning are shared. A non-zero virtual reference phase makes the sign convention observable;
    without it, complex-conjugate Kerr evolutions would produce the same parity trace.
    """
    alpha = np.asarray(amplitudes, dtype=float)
    x = np.asarray(delays, dtype=float)
    observed = np.asarray(parity_map, dtype=float)
    if alpha.ndim != 1 or x.ndim != 1 or observed.shape != (alpha.size, x.size):
        raise ValueError(
            "expected amplitudes (rows), delays (columns), and parity_map with shape "
            f"({alpha.size}, {x.size}); got {observed.shape}"
        )
    phase_offsets = (
        np.zeros(alpha.size, dtype=float)
        if return_phase_offsets_deg is None
        else np.asarray(return_phase_offsets_deg, dtype=float)
    )
    if (
        alpha.size == 0
        or x.size < 3
        or not np.all(np.isfinite(alpha))
        or np.any(alpha <= 0.0)
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(observed))
        or phase_offsets.shape != alpha.shape
        or not np.all(np.isfinite(phase_offsets))
    ):
        return _failed_fit("non-finite or invalid input", total_rows=alpha.size)
    if (
        not math.isfinite(reference_detuning_mhz)
        or reference_detuning_mhz == 0.0
        or not math.isfinite(k_search_max_mhz)
        or k_search_max_mhz <= 0.0
        or not math.isfinite(detuning_search_max_mhz)
        or detuning_search_max_mhz <= 0.0
    ):
        return _failed_fit("invalid fit bounds", total_rows=alpha.size)

    spans = np.ptp(observed, axis=1)
    valid = spans >= float(min_row_span)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < int(min_valid_rows):
        return _failed_fit(
            "too few rows with resolvable parity contrast",
            total_rows=alpha.size,
            valid_rows=valid_indices.size,
        )

    valid_alpha = alpha[valid]
    valid_observed = observed[valid]
    # Everything that does not depend on the fit parameters is built once per analysis: each row's
    # return displacement with its coherent preparation folded in, stacked for one matrix product.
    operators = np.stack(
        [
            _displacement(
                -float(value) * np.exp(1j * np.radians(float(phase_offsets[index]))),
                int(cutoff),
            )
            * _coherent_state(float(value), int(cutoff))[None, :]
            for index, value in zip(valid_indices, valid_alpha, strict=True)
        ]
    )
    offset_names = [f"offset_{row}" for row in range(valid_indices.size)]
    contrast_names = [f"contrast_{row}" for row in range(valid_indices.size)]

    # The finite-difference Jacobian perturbs one parameter at a time, and most columns move only a
    # row's offset or contrast. The ideal rows depend on (K, detuning) alone, so a small exact-key
    # memo returns the identical array instead of re-evaluating it.
    ideal_memo: dict[tuple[float, float], np.ndarray] = {}

    def _ideal_rows(k_mhz: float, detuning_mhz: float) -> np.ndarray:
        key = (k_mhz, detuning_mhz)
        ideal = ideal_memo.get(key)
        if ideal is None:
            kernel = _number_phase_kernel(
                int(cutoff),
                x,
                k_mhz=k_mhz,
                detuning_mhz=detuning_mhz,
                reference_detuning_mhz=float(reference_detuning_mhz),
            )
            ideal = _parity_rows(operators, kernel)
            ideal.setflags(write=False)
            if len(ideal_memo) >= 8:
                del ideal_memo[next(iter(ideal_memo))]
            ideal_memo[key] = ideal
        return ideal

    def _prediction(params) -> np.ndarray:
        ideal = _ideal_rows(float(params["K_mhz"].value), float(params["detuning_mhz"].value))
        offsets = np.array([float(params[name].value) for name in offset_names])
        contrasts = np.array([float(params[name].value) for name in contrast_names])
        return offsets[:, None] + contrasts[:, None] * ideal

    def _residual(params) -> np.ndarray:
        return (_prediction(params) - valid_observed).ravel()

    def _parameters(k_seed: float, detuning_seed: float) -> lmfit.Parameters:
        params = lmfit.Parameters()
        params.add("K_mhz", value=k_seed, min=-k_search_max_mhz, max=k_search_max_mhz)
        params.add(
            "detuning_mhz",
            value=detuning_seed,
            min=-detuning_search_max_mhz,
            max=detuning_search_max_mhz,
        )
        for row, values in enumerate(valid_observed):
            params.add(f"offset_{row}", value=float(np.mean(values)), min=-1.5, max=1.5)
            params.add(
                f"contrast_{row}",
                value=max(float(np.ptp(values)) / 2.0, 0.05),
                min=0.0,
                max=1.5,
            )
        return params

    k_seeds = np.array([-0.8, -0.35, -0.1, 0.0, 0.1, 0.35, 0.8]) * k_search_max_mhz
    detuning_seeds = np.array([-0.8, -0.2, 0.0, 0.2, 0.8]) * detuning_search_max_mhz
    candidates = []
    for k_seed in k_seeds:
        for detuning_seed in detuning_seeds:
            try:
                candidate = lmfit.minimize(
                    _residual,
                    _parameters(float(k_seed), float(detuning_seed)),
                    method="leastsq",
                    max_nfev=1_500,
                )
            except Exception:  # noqa: BLE001 - a failed seed must not discard an acquired map
                continue
            if candidate.success and math.isfinite(float(candidate.chisqr)):
                candidates.append(candidate)
    if not candidates:
        return _failed_fit("all multi-start fits failed", total_rows=alpha.size, valid_rows=valid_indices.size)
    best = min(candidates, key=lambda candidate: candidate.chisqr)

    k_value = float(best.params["K_mhz"].value)
    detuning_value = float(best.params["detuning_mhz"].value)
    k_stderr = best.params["K_mhz"].stderr
    detuning_stderr = best.params["detuning_mhz"].stderr
    k_stderr_value = float(k_stderr) if k_stderr is not None else float("nan")
    detuning_stderr_value = float(detuning_stderr) if detuning_stderr is not None else float("nan")
    k_interior = abs(k_value) < 0.98 * float(k_search_max_mhz)
    relative_k_stderr = (
        abs(k_stderr_value / k_value)
        if math.isfinite(k_stderr_value) and abs(k_value) > 0.0
        else float("inf")
    )
    # A separated candidate with effectively the same residual is a phase-winding alias, not an
    # independently measured Kerr coefficient. Keep the tolerance deliberately narrow: a normal
    # noisy fit may have nearby numerical starts, but only a distinct K basin can trip this gate.
    alias_separation = max(
        5.0 * abs(k_stderr_value) if math.isfinite(k_stderr_value) else 0.0,
        0.05 * float(k_search_max_mhz),
    )
    alias_chisqr_margin = max(1.0e-8, 0.02 * float(best.chisqr))
    competing_k = sorted(
        {
            float(candidate.params["K_mhz"].value)
            for candidate in candidates
            if abs(float(candidate.params["K_mhz"].value) - k_value) >= alias_separation
            and float(candidate.chisqr) <= float(best.chisqr) + alias_chisqr_margin
        }
    )
    prediction = _prediction(best.params)
    result = {
        "success": bool(best.success and math.isfinite(k_value) and math.isfinite(detuning_value)),
        "K_mhz": k_value,
        "K_stderr_mhz": k_stderr_value,
        "detuning_mhz": detuning_value,
        "detuning_stderr_mhz": detuning_stderr_value,
        "t_cat_us": 1.0 / (2.0 * abs(k_value)) if abs(k_value) > 0.0 else float("inf"),
        "relative_K_stderr": relative_k_stderr,
        "K_interior": bool(k_interior),
        "K_alias_ambiguous": bool(competing_k),
        "competing_K_mhz": competing_k,
        "valid_rows": int(valid_indices.size),
        "total_rows": int(alpha.size),
        "chisqr": float(best.chisqr),
        "residual_rms": float(np.sqrt(np.mean((prediction - valid_observed) ** 2))),
        "reference_detuning_mhz": float(reference_detuning_mhz),
        "cutoff": int(cutoff),
    }
    for row, original_index in enumerate(valid_indices):
        result[f"row_{int(original_index)}_alpha"] = float(alpha[original_index])
        result[f"row_{int(original_index)}_return_phase_offset_deg"] = float(phase_offsets[original_index])
        result[f"row_{int(original_index)}_span"] = float(spans[original_index])
        result[f"row_{int(original_index)}_offset"] = float(best.params[f"offset_{row}"].value)
        result[f"row_{int(original_index)}_contrast"] = float(best.params[f"contrast_{row}"].value)
    return result
