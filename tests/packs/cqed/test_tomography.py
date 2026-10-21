"""Wigner tomographic reconstruction — operators, linear inversion, fidelity, optimal set.

Pure: no qick. Synthetic Wigner data is generated from the SAME operator the reconstruction inverts
(``W_i = Tr[rho W_op(alpha_i)]``), so a successful recovery tests the inversion math end-to-end.
That alone cannot catch a wrong operator, so the operator itself is checked against an independent
construction, ``(2/pi) D(alpha) Pi D(alpha)^dagger`` by matrix exponential, and against the known
Wigner function of a coherent state with a complex amplitude.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from strumento.packs.cqed.tomography import (
    TWO_OVER_PI,
    _wigner_rows,
    coherent_state,
    displacement_operator,
    fidelity,
    optimal_displacements,
    parity_operator,
    parity_to_wigner,
    reconstruct,
    reconstruct_from_run,
    wigner_operator,
)


def _grid(radius: float, n: int) -> np.ndarray:
    ax = np.linspace(-radius, radius, n)
    return np.array([x + 1j * y for y in ax for x in ax])


def _wigner_values(rho: np.ndarray, alphas: np.ndarray, dim: int) -> np.ndarray:
    """W_i = Tr[rho W_op(alpha_i)] via the exact matrix the reconstruction inverts."""
    return np.real(_wigner_rows(alphas, dim) @ rho.reshape(-1))


def _fock(n: int, dim: int):
    psi = np.zeros(dim, dtype=np.complex128)
    psi[n] = 1.0
    return np.outer(psi, psi.conj()), psi


# -- operators -------------------------------------------------------------------------


def test_parity_operator():
    assert np.allclose(np.diag(parity_operator(4)), [1, -1, 1, -1])


def test_displacement_zero_is_identity():
    assert np.allclose(displacement_operator(0.0, 5), np.eye(5))


def test_displacement_unitary_on_low_subspace():
    d = displacement_operator(0.7, 12)
    assert np.allclose((d.conj().T @ d)[:6, :6], np.eye(6), atol=1e-3)


def test_wigner_operator_hermitian():
    w = wigner_operator(0.5 + 0.3j, 6)
    assert np.allclose(w, w.conj().T)


def test_wigner_vacuum_is_gaussian():
    dim = 8
    rho0, _ = _fock(0, dim)
    for a in [0.0, 0.5, 1.0, 1.3j, 0.8 + 0.6j]:
        w = np.real(np.trace(rho0 @ wigner_operator(a, dim)))
        assert w == pytest.approx(TWO_OVER_PI * np.exp(-2 * abs(a) ** 2), abs=1e-3)


def _reference_wigner_operator(alpha: complex, dim: int, big: int = 80) -> np.ndarray:
    """(2/pi) D(alpha) Pi D(alpha)^dagger built by matrix exponential in a large space, truncated."""
    from scipy.linalg import expm

    a = np.diag(np.sqrt(np.arange(1, big)), 1)
    d = expm(alpha * a.conj().T - np.conj(alpha) * a)
    parity = np.diag((-1.0) ** np.arange(big))
    return (TWO_OVER_PI * (d @ parity @ d.conj().T))[:dim, :dim]


@pytest.mark.parametrize("alpha", [0.8 + 0.3j, 0.4 - 0.7j, -1.1 + 0.2j, 1.5j])
def test_wigner_operator_is_the_displaced_parity(alpha):
    """W_op(alpha) = (2/pi) D(alpha) Pi D(alpha)^dagger, element by element (not its transpose)."""
    assert np.allclose(wigner_operator(alpha, 8), _reference_wigner_operator(alpha, 8), atol=1e-9)


def test_a_complex_coherent_state_peaks_at_its_own_amplitude():
    """W of |beta> is (2/pi) exp(-2|alpha - beta|^2): maximal at beta, not at its mirror image."""
    beta, dim = 0.8 + 0.3j, 14
    rho = np.outer(coherent_state(beta, dim), coherent_state(beta, dim).conj())
    for alpha in (beta, np.conj(beta), 0.2 - 0.5j):
        w = np.real(np.trace(rho @ wigner_operator(alpha, dim)))
        assert w == pytest.approx(TWO_OVER_PI * np.exp(-2 * abs(alpha - beta) ** 2), abs=1e-6)


def test_a_complex_coherent_state_reconstructs_unmirrored():
    beta, dim = 0.7 + 0.6j, 8
    psi = coherent_state(beta, dim)
    alphas = _grid(3.0, 13)
    rho = np.outer(psi, psi.conj())
    rec = reconstruct(alphas, _wigner_values(rho, alphas, dim), dim, target=psi)
    assert rec.fidelity > 0.99
    assert fidelity(rec.rho, coherent_state(np.conj(beta), dim)) < 0.8


# -- reconstruction --------------------------------------------------------------------


@pytest.mark.parametrize("n_fock", [0, 1, 2])
def test_reconstruct_fock_state(n_fock):
    dim = 6
    rho, psi = _fock(n_fock, dim)
    alphas = _grid(3.0, 11)
    rec = reconstruct(alphas, _wigner_values(rho, alphas, dim), dim, target=psi)
    assert rec.fidelity > 0.99
    assert rec.populations[n_fock] > 0.98
    assert np.trace(rec.rho).real == pytest.approx(1.0, abs=1e-6)


def test_reconstruct_linear_recovers_exactly():
    dim = 6
    rho, _ = _fock(1, dim)
    alphas = _grid(3.0, 11)
    rec = reconstruct(alphas, _wigner_values(rho, alphas, dim), dim, project=False)
    assert np.allclose(rec.rho_linear, rho, atol=1e-4)


def test_reconstruct_coherent_state():
    dim = 8
    psi = coherent_state(1.0, dim)
    rho = np.outer(psi, psi.conj())
    alphas = _grid(3.0, 13)
    rec = reconstruct(alphas, _wigner_values(rho, alphas, dim), dim, target=psi)
    assert rec.fidelity > 0.99


def test_reconstruct_cat_state():
    dim = 12
    ket = coherent_state(2.0, dim) + coherent_state(-2.0, dim)
    ket /= np.linalg.norm(ket)
    rho = np.outer(ket, ket.conj())
    alphas = _grid(4.0, 17)
    rec = reconstruct(alphas, _wigner_values(rho, alphas, dim), dim, target=ket)
    assert rec.fidelity > 0.95


def test_reconstruct_no_target_leaves_fidelity_none():
    dim = 5
    rho, _ = _fock(0, dim)
    alphas = _grid(3.0, 9)
    rec = reconstruct(alphas, _wigner_values(rho, alphas, dim), dim)
    assert rec.fidelity is None and rec.linear_fidelity is None


def test_optimal_rotation_recovers_phase_rotated_state():
    """A |0>+|2> superposition rotated by a cavity phase is recovered by rotate='optimal'."""
    dim = 6
    ket = np.zeros(dim, dtype=np.complex128)
    ket[0] = ket[2] = 1.0
    ket /= np.linalg.norm(ket)
    # apply a channel phase exp(-i theta n) to the true state before measuring
    theta = 0.7
    rot = np.exp(-1j * theta * np.arange(dim))
    ket_rot = rot * ket
    rho = np.outer(ket_rot, ket_rot.conj())
    alphas = _grid(3.0, 13)
    w = _wigner_values(rho, alphas, dim)
    no_rot = reconstruct(alphas, w, dim, target=ket)
    with_rot = reconstruct(alphas, w, dim, target=ket, rotate="optimal")
    assert with_rot.fidelity > 0.99
    assert with_rot.fidelity > no_rot.fidelity


# -- fidelity + parity conversion ------------------------------------------------------


def test_fidelity_self_one_orthogonal_zero():
    dim = 4
    r0, p0 = _fock(0, dim)
    _, p1 = _fock(1, dim)
    assert fidelity(r0, p0) == pytest.approx(1.0)
    assert fidelity(r0, p1) == pytest.approx(0.0, abs=1e-9)
    assert fidelity(r0, r0) == pytest.approx(1.0, abs=1e-6)  # mixed-target Uhlmann path


def test_parity_to_wigner_convention():
    assert parity_to_wigner(0.0) == pytest.approx(TWO_OVER_PI)
    assert parity_to_wigner(1.0) == pytest.approx(-TWO_OVER_PI)
    assert parity_to_wigner(0.5) == pytest.approx(0.0, abs=1e-12)
    assert parity_to_wigner(0.0, alpha_scale=2.0) == pytest.approx(TWO_OVER_PI / 2.0)
    # parity_sign flips the excited<->parity mapping (a readout calibration choice)
    assert parity_to_wigner(0.0, parity_sign=-1.0) == pytest.approx(-TWO_OVER_PI)
    assert parity_to_wigner(1.0, parity_sign=-1.0) == pytest.approx(TWO_OVER_PI)


def test_parity_to_wigner_off_centre_midpoint():
    # the parity zero sits at pe_midpoint, not 0.5; the even/odd fringe extrema
    # (pe_midpoint -/+ alpha_scale/2) map to +/- 2/pi
    assert parity_to_wigner(0.44, pe_midpoint=0.44) == pytest.approx(0.0, abs=1e-12)
    assert parity_to_wigner(0.0, alpha_scale=0.88, pe_midpoint=0.44) == pytest.approx(TWO_OVER_PI)
    assert parity_to_wigner(0.88, alpha_scale=0.88, pe_midpoint=0.44) == pytest.approx(-TWO_OVER_PI)
    # reduces to the (1 - 2 P_e) form when the midpoint is 0.5
    for pe in (0.0, 0.3, 0.5, 0.8, 1.0):
        assert parity_to_wigner(pe, pe_midpoint=0.5) == pytest.approx(TWO_OVER_PI * (1 - 2 * pe))


def test_reconstruct_from_run_parity_sign_recovers_inverted_convention():
    """A signal whose excited state maps to EVEN parity (P_g-high, the sim's convention) still
    reconstructs when parity_sign=-1 is passed."""
    dim = 6
    rho, psi = _fock(0, dim)
    alphas = _grid(3.0, 11)
    w = _wigner_values(rho, alphas, dim)
    # inverted signal: S = (1 + <Pi>)/2 = 1 - P_e  (even-parity-high)
    parity = w / TWO_OVER_PI
    s_inverted = (1.0 + parity) / 2.0
    result = SimpleNamespace(meta={
        "alpha_re": alphas.real.tolist(),
        "alpha_im": alphas.imag.tolist(),
        "parity_signal": s_inverted.tolist(),
    })
    rec = reconstruct_from_run(result, dim, target=psi, parity_sign=-1.0)
    assert rec.fidelity > 0.99


# -- optimal displacements -------------------------------------------------------------


def test_optimal_displacements_are_well_posed_and_reconstruct():
    dim = 4
    pts = optimal_displacements(dim, n_disps=40, seed=1)
    assert pts.shape == (41,)                       # origin + n_disps
    assert pts[0] == 0                              # origin pinned
    assert np.iscomplexobj(pts)
    assert np.linalg.matrix_rank(_wigner_rows(pts, dim)) == dim * dim  # full rank -> invertible
    rho, psi = _fock(1, dim)
    rec = reconstruct(pts, _wigner_values(rho, pts, dim), dim, target=psi)
    assert rec.fidelity > 0.99


# -- run-meta path ---------------------------------------------------------------------


def test_reconstruct_from_run_meta():
    dim = 6
    rho, psi = _fock(0, dim)                        # vacuum
    alphas = _grid(3.0, 11)
    w = _wigner_values(rho, alphas, dim)
    pe = (1.0 - w / TWO_OVER_PI) / 2.0              # invert parity_to_wigner to a P_e signal
    result = SimpleNamespace(meta={
        "alpha_re": alphas.real.tolist(),
        "alpha_im": alphas.imag.tolist(),
        "parity_signal": pe.tolist(),
    })
    rec = reconstruct_from_run(result, dim, target=psi)
    assert rec.fidelity > 0.99


# -- device-read reconstruction (raw IQ -> P_e via blob centres, calib from device) -------------


def _device_run_meta(rho, alphas, dim, *, alpha_scale, parity_sign):
    """A WignerTomography2D-style run on the DEVICE path.

    Builds the raw averaged IQ (``parity_i``/``parity_q``) that a trivial readout (angle 0,
    g=(0,0), e=(1,0)) maps via ``to_population`` back to the ``P_e`` that inverts to ``W`` at the
    recorded ``alpha_scale``/``parity_sign`` — so a successful recovery proves the whole
    signal -> P_e -> <Pi> chain, not just the inversion math.
    """
    w = _wigner_values(rho, alphas, dim)
    parity = w / TWO_OVER_PI
    pe = (1.0 - parity * alpha_scale / parity_sign) / 2.0    # invert <Pi> = sign*(1-2 P_e)/scale
    return SimpleNamespace(meta={
        "alpha_re": alphas.real.tolist(),
        "alpha_im": alphas.imag.tolist(),
        "parity_i": pe.tolist(),                             # I == P_e for g=(0,0), e=(1,0), angle 0
        "parity_q": np.zeros_like(pe).tolist(),
        "parity_signal": pe.tolist(),                        # kept for the plot; unused on this path
    })


def _trivial_readout():
    from strumento.packs.cqed.readout import DispersiveIQ
    return DispersiveIQ(threshold=0.0, angle=0.0, g_center=(0.0, 0.0), e_center=(1.0, 0.0))


@pytest.mark.parametrize("parity_sign", [1.0, -1.0])
def test_reconstruct_from_run_device_path_reads_mode_cal(parity_sign):
    dim = 8
    psi = coherent_state(1.0, dim)
    rho = np.outer(psi, psi.conj())
    alphas = _grid(3.0, 13)
    alpha_scale = 0.91
    result = _device_run_meta(rho, alphas, dim, alpha_scale=alpha_scale, parity_sign=parity_sign)
    mode_cal = SimpleNamespace(alpha_scale=alpha_scale, parity_sign=parity_sign)
    rec = reconstruct_from_run(result, dim, target=psi, readout=_trivial_readout(), mode_cal=mode_cal)
    assert rec.fidelity > 0.95


def test_reconstruct_from_run_device_path_vacuum_scale():
    """The single-phase offset/scale guard: a vacuum run reconstructs to |0><0| with the right
    vertical scale (populations[0] ~ 1) when read through alpha_scale != 1."""
    dim = 6
    rho, psi = _fock(0, dim)
    alphas = _grid(3.0, 11)
    result = _device_run_meta(rho, alphas, dim, alpha_scale=0.91, parity_sign=1.0)
    mode_cal = SimpleNamespace(alpha_scale=0.91, parity_sign=1.0)
    rec = reconstruct_from_run(result, dim, target=psi, readout=_trivial_readout(), mode_cal=mode_cal)
    assert rec.fidelity > 0.99
    assert rec.populations[0] > 0.98


def test_reconstruct_from_run_hard_errors_without_blob_centres():
    from strumento.packs.cqed.readout import DispersiveIQ
    dim = 6
    rho, _ = _fock(0, dim)
    alphas = _grid(3.0, 11)
    result = _device_run_meta(rho, alphas, dim, alpha_scale=1.0, parity_sign=1.0)
    with pytest.raises(ValueError, match="blob centres"):
        reconstruct_from_run(result, dim, readout=DispersiveIQ(threshold=0.0))


def test_reconstruct_from_run_explicit_args_override_mode_cal():
    dim = 6
    rho, psi = _fock(0, dim)
    alphas = _grid(3.0, 11)
    result = _device_run_meta(rho, alphas, dim, alpha_scale=0.91, parity_sign=1.0)
    # mode_cal carries wrong values; explicit args are the ones that invert correctly
    mode_cal = SimpleNamespace(alpha_scale=0.5, parity_sign=-1.0)
    rec = reconstruct_from_run(
        result, dim, target=psi, readout=_trivial_readout(), mode_cal=mode_cal,
        alpha_scale=0.91, parity_sign=1.0,
    )
    assert rec.fidelity > 0.99


def test_the_wigner_operator_derivatives_match_finite_differences():
    from strumento.packs.cqed.tomography import _wigner_operator_derivatives

    alpha, dim, h = 0.6 - 0.4j, 6, 1e-6
    d_re, d_im = _wigner_operator_derivatives(alpha, dim)
    fd_re = (wigner_operator(alpha + h, dim) - wigner_operator(alpha - h, dim)) / (2 * h)
    fd_im = (wigner_operator(alpha + 1j * h, dim) - wigner_operator(alpha - 1j * h, dim)) / (2 * h)
    assert np.allclose(d_re, fd_re, atol=1e-6) and np.allclose(d_im, fd_im, atol=1e-6)


def test_the_design_cost_gradient_matches_finite_differences():
    from strumento.packs.cqed.tomography import _a_optimal_cost, _hermitian_features

    dim = 3
    rng = np.random.default_rng(4)
    x = rng.uniform(-1.5, 1.5, 2 * 14)
    origin = _hermitian_features(wigner_operator(0.0, dim))[None, :]
    _, grad = _a_optimal_cost(x, dim, 3.0, origin)
    h = 1e-6
    for i in (0, 5, 14, 27):
        e = np.zeros_like(x)
        e[i] = h
        fd = (_a_optimal_cost(x + e, dim, 3.0, origin)[0] - _a_optimal_cost(x - e, dim, 3.0, origin)[0]) / (2 * h)
        assert grad[i] == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_the_physical_projection_is_the_closest_density_matrix():
    """Eigenvalues projected onto the simplex: a known case, and PSD + trace 1 in general."""
    from strumento.packs.cqed.tomography import project_to_physical

    rho = np.diag([0.7, 0.4, -0.1]).astype(complex)
    assert np.allclose(np.diag(project_to_physical(rho)).real, [0.65, 0.35, 0.0])
    rng = np.random.default_rng(2)
    m = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    out = project_to_physical(m + m.conj().T)
    assert np.trace(out).real == pytest.approx(1.0)
    assert np.linalg.eigvalsh(out).min() > -1e-12


def test_the_design_optimizer_improves_on_its_start():
    from strumento.packs.cqed.tomography import _design_matrix

    dim = 4
    rng = np.random.default_rng(1)
    start = np.clip(rng.normal(0.0, np.sqrt(dim) / 2.0, size=(2, 32)), -3.0, 3.0)
    random_pts = np.concatenate(([0j], start[0] + 1j * start[1]))
    opt_pts = optimal_displacements(dim, n_disps=32, seed=1)

    def cost(pts):
        f = _design_matrix(pts, dim)
        return float(np.trace(np.linalg.inv(f.T @ f)))

    assert cost(opt_pts) < 0.5 * cost(random_pts)
    assert np.abs(opt_pts).max() < 1.25 * 1.5 * np.sqrt(dim) + 1e-9
