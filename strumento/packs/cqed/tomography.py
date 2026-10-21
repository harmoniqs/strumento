"""Wigner-function tomography for a bosonic mode: operators, reconstruction, fidelity, design.

Given displaced-parity measurements ``W(alpha_i)`` at complex displacements ``{alpha_i}`` (for
example from :class:`~strumento.packs.cqed.experiments.wigner_tomography.WignerTomography2D`'s
polar star), this module reconstructs the mode's density matrix in a truncated Fock basis, projects
it onto the physical states, scores fidelity against a target, and chooses well-conditioned
displacement sets.

Methods, each from the literature:

* **Wigner operator.** ``W(alpha) = Tr[rho W_op(alpha)]`` with
  ``W_op(alpha) = (2/pi) D(alpha) Pi D(alpha)^dagger`` (Royer, Phys. Rev. A 15, 449 (1977);
  Lutterbach & Davidovich, Phys. Rev. Lett. 78, 2547 (1997)). Because ``Pi D(-alpha) = D(alpha) Pi``,
  this equals ``(2/pi) D(2 alpha) Pi``, so its Fock matrix elements follow from the closed form of
  the displacement operator's elements (Cahill & Glauber, Phys. Rev. 177, 1857 (1969)) with no
  matrix exponential and no truncation error inside the kept subspace.
* **Reconstruction.** Linear least squares for ``rho``, written in a real orthonormal basis of
  Hermitian matrices so the estimate is Hermitian by construction, with the trace-one constraint
  eliminated exactly (the linear-inversion approach of Vlastakis et al., Science 342, 607 (2013)).
* **Physical projection.** The closest density matrix in Frobenius norm keeps the eigenvectors and
  projects the eigenvalues onto the probability simplex (Smolin, Gambetta & Smith, Phys. Rev. Lett.
  108, 070502 (2012)).
* **Displacement design.** A-optimal experimental design: minimize ``Tr[(F^T F)^{-1}]``, the summed
  variance of the reconstructed coefficients under equal, independent noise on each sample.

Pure: no ``qick`` import (physics lives in the pack, not core).

Conventions
-----------
A measured excited-state population ``P_e`` maps to a Wigner value through the parity expectation
``<Pi> = parity_sign * 2 (pe_midpoint - P_e) / alpha_scale`` and ``W = (2/pi) <Pi>`` (see
:func:`parity_to_wigner`), where ``pe_midpoint`` is the vacuum parity fringe's DC level (0.5 for a
symmetric, offset-free fringe). A coherent state ``|beta>`` has ``W(alpha) = (2/pi) exp(-2|alpha -
beta|^2)``, peaked at ``alpha = beta``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import eval_genlaguerre, gammaln

TWO_OVER_PI = 2.0 / np.pi


# ======================================================================================
# Operators
# ======================================================================================


def _displacement_block(beta: complex, rows: int, cols: int) -> np.ndarray:
    """Fock matrix elements ``<m|D(beta)|n>`` for ``m < rows``, ``n < cols`` (closed form).

    ``<m|D(beta)|n> = sqrt(n!/m!) beta^(m-n) exp(-|beta|^2/2) L_n^(m-n)(|beta|^2)`` for ``m >= n``,
    and ``(-beta*)^(n-m)`` with ``m`` and ``n`` exchanged otherwise. The magnitude prefactor is
    evaluated in log space so large photon numbers do not overflow.
    """
    beta = complex(beta)
    out = np.zeros((rows, cols), dtype=np.complex128)
    r2 = abs(beta) ** 2
    if r2 == 0.0:
        k = min(rows, cols)
        out[np.arange(k), np.arange(k)] = 1.0
        return out
    log_r = 0.5 * np.log(r2)
    phase = beta / abs(beta)
    for m in range(rows):
        for n in range(cols):
            lo, hi = (n, m) if m >= n else (m, n)
            k = hi - lo
            log_mag = 0.5 * (gammaln(lo + 1) - gammaln(hi + 1)) + k * log_r - r2 / 2.0
            unit = phase ** k if m >= n else (-np.conj(phase)) ** k
            out[m, n] = unit * np.exp(log_mag) * eval_genlaguerre(lo, k, r2)
    return out


def displacement_operator(alpha: complex, dim: int) -> np.ndarray:
    """``D(alpha) = exp(alpha a^dagger - alpha^* a)``, its exact elements on the first ``dim`` levels."""
    return _displacement_block(alpha, dim, dim)


def parity_operator(dim: int) -> np.ndarray:
    """The photon-number parity operator ``Pi = diag((-1)^n)``."""
    return np.diag((-1.0) ** np.arange(dim)).astype(np.complex128)


def coherent_state(alpha: complex, dim: int) -> np.ndarray:
    """The coherent-state ket ``|alpha>`` on the first ``dim`` Fock levels, renormalized."""
    n = np.arange(dim)
    alpha = complex(alpha)
    if alpha == 0:
        psi = np.zeros(dim, dtype=np.complex128)
        psi[0] = 1.0
        return psi
    log_mag = n * np.log(abs(alpha)) - 0.5 * gammaln(n + 1)
    psi = np.exp(log_mag - log_mag.max()) * (alpha / abs(alpha)) ** n
    return psi / np.linalg.norm(psi)


def wigner_operator(alpha: complex, dim: int) -> np.ndarray:
    """The Wigner operator ``W_op(alpha) = (2/pi) D(2 alpha) Pi`` on ``dim`` levels (Hermitian).

    A measured Wigner value is ``W(alpha) = Tr[rho W_op(alpha)]``.
    """
    signs = (-1.0) ** np.arange(dim)
    return TWO_OVER_PI * _displacement_block(2.0 * complex(alpha), dim, dim) * signs[None, :]


def _wigner_operator_derivatives(alpha: complex, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """``dW_op/dRe(alpha)`` and ``dW_op/dIm(alpha)`` on ``dim`` levels.

    From ``D(beta + d) = exp(-(d beta^* - d^* beta)/2) D(d) D(beta)`` to first order in ``d``:
    ``dD/dRe(beta) = [(a^dagger - a) + i Im beta] D`` and ``dD/dIm(beta) = [i(a^dagger + a) - i Re
    beta] D``, with ``beta = 2 alpha``. ``a D`` needs one row of ``D`` past the kept block, which the
    closed form supplies exactly.
    """
    beta = 2.0 * complex(alpha)
    d_ext = _displacement_block(beta, dim + 1, dim)       # rows 0..dim, columns 0..dim-1
    d = d_ext[:dim]
    m = np.arange(dim)
    a_d = np.sqrt(m + 1)[:, None] * d_ext[1:]             # (a D)[m] = sqrt(m+1) D[m+1]
    adag_d = np.zeros_like(d)
    adag_d[1:] = np.sqrt(m[1:])[:, None] * d[:-1]          # (a^dagger D)[m] = sqrt(m) D[m-1]
    d_re = (adag_d - a_d) + 1j * beta.imag * d
    d_im = 1j * (adag_d + a_d) - 1j * beta.real * d
    signs = (-1.0) ** m
    scale = 2.0 * TWO_OVER_PI                               # dbeta/dalpha = 2
    return scale * d_re * signs[None, :], scale * d_im * signs[None, :]


def _wigner_rows(alphas: np.ndarray, dim: int) -> np.ndarray:
    """The complex measurement matrix ``M`` (shape ``(n, dim^2)``) with ``M @ vec(rho) = W``.

    ``vec`` is row-major, so row ``i`` is ``vec(W_op(alpha_i)^T)``: ``Tr[rho W] = sum rho_mn W_nm``.
    """
    return np.array([wigner_operator(a, dim).T.reshape(-1) for a in np.atleast_1d(alphas)])


# -- the real Hermitian basis ----------------------------------------------------------------


def _offdiagonal_pairs(dim: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(dim, k=1)


def _hermitian_features(w_op: np.ndarray) -> np.ndarray:
    """``Tr[W E_k]`` for the orthonormal Hermitian basis ``E_k``, for a Hermitian ``W``.

    The basis is ``|j><j|``, then ``(|j><k| + |k><j|)/sqrt 2`` and ``(-i|j><k| + i|k><j|)/sqrt 2``
    for ``j < k``. Their traces against ``W`` are ``W_jj``, ``sqrt 2 Re W_jk`` and
    ``-sqrt 2 Im W_jk``.
    """
    j, k = _offdiagonal_pairs(w_op.shape[0])
    upper = w_op[j, k]
    return np.concatenate(
        [np.real(np.diag(w_op)), np.sqrt(2.0) * upper.real, -np.sqrt(2.0) * upper.imag]
    )


def _from_hermitian_coefficients(c: np.ndarray, dim: int) -> np.ndarray:
    """``rho = sum_k c_k E_k`` for the basis of :func:`_hermitian_features`."""
    j, k = _offdiagonal_pairs(dim)
    n_off = len(j)
    rho = np.diag(c[:dim]).astype(np.complex128)
    upper = (c[dim : dim + n_off] - 1j * c[dim + n_off :]) / np.sqrt(2.0)
    rho[j, k] = upper
    rho[k, j] = np.conj(upper)
    return rho


def _design_matrix(alphas: np.ndarray, dim: int) -> np.ndarray:
    """The real design matrix ``F`` (shape ``(n, dim^2)``): ``W_i = F_i . c`` for ``rho = sum c E``."""
    return np.array([_hermitian_features(wigner_operator(a, dim)) for a in np.atleast_1d(alphas)])


# ======================================================================================
# Reconstruction
# ======================================================================================


@dataclass
class WignerReconstruction:
    """Result of a Wigner tomographic reconstruction.

    ``rho`` is the physical (Hermitian, PSD, trace-1) estimate for display and populations.
    ``rho_linear`` is the trace-1 least-squares estimate before projection: unbiased, Hermitian,
    possibly with small negative eigenvalues. ``linear_fidelity`` is read from it, because the
    projection biases ``fidelity`` low for near-pure targets.
    """

    dim: int
    rho: np.ndarray
    rho_linear: np.ndarray
    populations: np.ndarray
    alphas: np.ndarray
    wigner_values: np.ndarray
    fidelity: float | None = None
    linear_fidelity: float | None = None
    theta: float = 0.0


def _least_squares_trace_one(design: np.ndarray, w: np.ndarray, dim: int) -> np.ndarray:
    """The Hermitian, trace-one ``rho`` minimizing ``|F c - w|^2``.

    The trace is ``sum`` of the diagonal coefficients, so the last one is eliminated as ``1 - sum``
    of the others, and the remaining coefficients solve an unconstrained real least-squares problem.
    """
    last = dim - 1
    keep = [i for i in range(design.shape[1]) if i != last]
    reduced = design[:, keep].copy()
    reduced[:, :last] -= design[:, [last]]
    coeffs, *_ = np.linalg.lstsq(reduced, w - design[:, last], rcond=None)
    c = np.empty(design.shape[1])
    c[keep] = coeffs
    c[last] = 1.0 - coeffs[:last].sum()
    return _from_hermitian_coefficients(c, dim)


def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection of a real vector onto ``{p : p >= 0, sum p = 1}``.

    Sort descending, find the largest ``r`` with ``u_r > (sum_{i<=r} u_i - 1) / r``, and shift by
    that threshold. On eigenvalues this is the Smolin-Gambetta-Smith maximum-likelihood step.
    """
    u = np.sort(values)[::-1]
    cumulative = np.cumsum(u) - 1.0
    ranks = np.arange(1, len(u) + 1)
    r = ranks[u - cumulative / ranks > 0][-1]
    threshold = cumulative[r - 1] / r
    return np.maximum(values - threshold, 0.0)


def project_to_physical(rho: np.ndarray) -> np.ndarray:
    """The density matrix closest to ``rho`` in Frobenius norm: Hermitian, PSD, trace 1."""
    herm = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(herm)
    return (vecs * _project_to_simplex(vals)) @ vecs.conj().T


def _sqrtm_psd(m: np.ndarray) -> np.ndarray:
    """Square root of a Hermitian PSD matrix, robust to rank deficiency and round-off."""
    herm = 0.5 * (m + m.conj().T)
    vals, vecs = np.linalg.eigh(herm)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.conj().T


def fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    """State fidelity in the squared convention: ``<psi|rho|psi>`` for a pure target ket.

    For a density-matrix target it is the Uhlmann fidelity ``(Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2``.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    t = np.asarray(target, dtype=np.complex128)
    if t.ndim == 1:
        psi = t / np.linalg.norm(t)
        return float(np.real(psi.conj() @ rho @ psi))
    root = _sqrtm_psd(rho)
    return float(np.real(np.trace(_sqrtm_psd(root @ (t / np.trace(t)) @ root))) ** 2)


def _rotate(rho: np.ndarray, theta: float) -> np.ndarray:
    """``U rho U^dagger`` for the cavity rotation ``U = exp(-i theta N)``."""
    phase = np.exp(-1j * theta * np.arange(rho.shape[0]))
    return phase[:, None] * rho * np.conj(phase)[None, :]


def _best_rotation(rho: np.ndarray, target: np.ndarray, grid: int = 360) -> float:
    """The rotation angle ``theta`` maximizing the fidelity of ``exp(-i theta N) rho`` to ``target``.

    The cavity phase is a free gauge (the drive's reference phase), so it is fixed before scoring:
    a coarse scan over ``[0, 2 pi)`` brackets the maximum and a bounded scalar search refines it.
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, grid, endpoint=False)
    scores = [fidelity(_rotate(rho, th), target) for th in thetas]
    best = thetas[int(np.argmax(scores))]
    step = thetas[1] - thetas[0]
    refined = minimize_scalar(
        lambda th: -fidelity(_rotate(rho, th), target),
        bounds=(best - step, best + step),
        method="bounded",
    )
    return float(refined.x % (2.0 * np.pi))


def reconstruct(
    alphas: np.ndarray,
    wigner_values: np.ndarray,
    dim: int,
    *,
    target: np.ndarray | None = None,
    rotate: str | float | None = None,
    project: bool = True,
) -> WignerReconstruction:
    """Reconstruct a density matrix from Wigner samples ``W(alpha_i)`` by trace-one least squares.

    Parameters
    ----------
    alphas, wigner_values:
        The complex displacement points and the Wigner value measured at each (same length). Use
        :func:`parity_to_wigner` to turn measured excited-state populations into ``wigner_values``.
    dim:
        Fock-space truncation for the reconstruction.
    target:
        Optional target state (ket or density matrix); when given, ``fidelity`` and
        ``linear_fidelity`` are scored against it.
    rotate:
        ``None`` (default) scores in the raw frame; ``"optimal"`` finds the cavity rotation that
        maximizes fidelity (fixing the phase gauge); a float applies that fixed angle (rad).
    project:
        Whether to project ``rho`` onto the physical states. ``rho_linear`` is always the
        unprojected estimate.
    """
    alphas = np.asarray(alphas, dtype=np.complex128)
    w = np.asarray(wigner_values, dtype=float)
    rho_linear = _least_squares_trace_one(_design_matrix(alphas, dim), w, dim)
    rho = project_to_physical(rho_linear) if project else rho_linear

    theta = 0.0
    fid: float | None = None
    lin_fid: float | None = None
    if target is not None:
        if rotate == "optimal":
            theta = _best_rotation(rho, target)
        elif rotate is not None:
            theta = float(rotate)
        fid = fidelity(_rotate(rho, theta), target)
        lin_fid = fidelity(_rotate(rho_linear, theta), target)

    return WignerReconstruction(
        dim=dim,
        rho=rho,
        rho_linear=rho_linear,
        populations=np.real(np.diag(rho)),
        alphas=alphas,
        wigner_values=w,
        fidelity=fid,
        linear_fidelity=lin_fid,
        theta=theta,
    )


def parity_to_wigner(
    excited_population,
    alpha_scale: float = 1.0,
    parity_sign: float = 1.0,
    pe_midpoint: float = 0.5,
) -> np.ndarray:
    """Measured excited-state population ``P_e`` -> Wigner value ``W = (2/pi) <Pi>``.

    ``<Pi> = parity_sign * 2 (pe_midpoint - P_e) / alpha_scale``. The parity is referenced to the
    **fringe midpoint** ``pe_midpoint`` (the vacuum parity fringe's DC level), not a hardcoded 0.5:
    single-phase parity has no offset cancellation, so a readout or relaxation asymmetry shifts the
    fringe midpoint off 0.5, and a ``(1 - 2 P_e)`` form (which assumes 0.5) biases every ``<Pi>``.
    With ``pe_midpoint = 0.5`` this reduces to ``parity_sign * (1 - 2 P_e) / alpha_scale``.

    ``parity_sign = +1`` (the default) reads even parity below the midpoint (``+2/pi``);
    ``parity_sign = -1`` flips it for a readout whose excited state maps to *even* parity.
    ``alpha_scale`` is the mode's parity-contrast normalization (the fitted population contrast
    ``2|amp|``), so the even/odd fringe extrema ``pe_midpoint -/+ alpha_scale/2`` map to
    ``<Pi> = +/-1``.
    """
    pe = np.asarray(excited_population, dtype=float)
    parity = parity_sign * 2.0 * (pe_midpoint - pe) / alpha_scale
    return TWO_OVER_PI * parity


def reconstruct_from_run(
    result,
    dim: int,
    *,
    target: np.ndarray | None = None,
    rotate: str | float | None = None,
    readout=None,
    mode_cal=None,
    alpha_scale: float | None = None,
    parity_sign: float | None = None,
    pe_midpoint: float | None = None,
) -> WignerReconstruction:
    """Reconstruct directly from a :class:`WignerTomography2D` run, reading device normalization.

    Reads the scattered complex points the polar star stashes in ``result.meta``
    (``alpha_re``/``alpha_im``), turns the per-point parity into an excited-state population ``P_e``,
    converts that to Wigner values, and inverts.

    Two population sources, selected by ``readout``:

    * **Device path** (``readout`` is a :class:`~strumento.packs.cqed.readout.DispersiveIQ` with g/e
      blob centres): the run's **raw averaged IQ** (``meta["parity_i"]``/``["parity_q"]``) is mapped
      to ``P_e`` via ``readout.to_population`` — the hardware-correct conversion. ``alpha_scale``,
      ``parity_sign`` and ``pe_midpoint`` default to the ``mode_cal`` values
      (``ModeCal.alpha_scale``/``parity_sign``/``pe_midpoint``); explicit arguments override.
      **Raises** ``ValueError`` if a ``readout`` is supplied without blob centres (run
      ``ge-readout-blobs`` and record ``readout.g_center_iq``/``e_center_iq``).
    * **Population path** (``readout`` is ``None``): ``meta["parity_signal"]`` is treated directly
      as ``P_e`` (the convention the synthetic unit tests use). ``alpha_scale``/``parity_sign``/
      ``pe_midpoint`` default to 1.0/+1.0/0.5.

    See :func:`parity_to_wigner` for the ``<Pi> = parity_sign·2(pe_midpoint - P_e)/alpha_scale`` mapping.
    """
    meta = result.meta
    alphas = np.asarray(meta["alpha_re"], dtype=float) + 1j * np.asarray(meta["alpha_im"], dtype=float)

    if alpha_scale is None:
        alpha_scale = float(getattr(mode_cal, "alpha_scale", 1.0) or 1.0) if mode_cal is not None else 1.0
    if parity_sign is None:
        parity_sign = float(getattr(mode_cal, "parity_sign", 1.0) or 1.0) if mode_cal is not None else 1.0
    if pe_midpoint is None:
        pe_midpoint = float(getattr(mode_cal, "pe_midpoint", 0.5)) if mode_cal is not None else 0.5

    if readout is not None:
        if getattr(readout, "g_center", None) is None or getattr(readout, "e_center", None) is None:
            raise ValueError(
                "reconstruct_from_run: the supplied readout has no g/e blob centres — run "
                "ge-readout-blobs and record readout.g_center_iq/e_center_iq on the device before "
                "reconstructing a hardware Wigner"
            )
        if "parity_i" not in meta or "parity_q" not in meta:
            raise KeyError(
                "reconstruct_from_run: device-path conversion needs raw IQ (meta['parity_i']/"
                "['parity_q']) — re-run WignerTomography2D to stash it"
            )
        raw = np.stack(
            [np.asarray(meta["parity_i"], dtype=float), np.asarray(meta["parity_q"], dtype=float)],
            axis=-1,
        )
        excited_population = readout.to_population(raw)
    else:
        excited_population = np.asarray(meta["parity_signal"], dtype=float)

    w = parity_to_wigner(
        excited_population, alpha_scale=alpha_scale, parity_sign=parity_sign, pe_midpoint=pe_midpoint
    )
    return reconstruct(alphas, w, dim, target=target, rotate=rotate)


# ======================================================================================
# Displacement design
# ======================================================================================


def _a_optimal_cost(x: np.ndarray, dim: int, radius: float, fixed: np.ndarray,
                    wall: float = 1.0e3) -> tuple[float, np.ndarray]:
    """``Tr[(F^T F)^{-1}]`` over the free points ``x = [Re..., Im...]``, plus a soft radial wall.

    Only row ``i`` of ``F`` depends on point ``i``, so with ``A = F^T F`` and ``B = A^{-2}``,
    ``d cost / d x_i = -2 f_i' . (B f_i)``, where ``f_i'`` is the derivative of that row.
    """
    n = len(x) // 2
    points = x[:n] + 1j * x[n:]
    rows, d_re, d_im = [], [], []
    for p in points:
        rows.append(_hermitian_features(wigner_operator(p, dim)))
        g_re, g_im = _wigner_operator_derivatives(p, dim)
        d_re.append(_hermitian_features(g_re))
        d_im.append(_hermitian_features(g_im))
    free = np.array(rows)
    design = np.vstack([fixed, free]) if len(fixed) else free
    gram = design.T @ design
    vals, vecs = np.linalg.eigh(gram)
    if vals[0] <= 1e-12 * vals[-1]:
        # A (numerically) singular design: unboundedly bad. Report it as such so the line search
        # backs off, rather than invert garbage into a spuriously low cost.
        return 1e300, np.zeros_like(x)
    a_inv = (vecs / vals) @ vecs.T
    cost = float(np.sum(1.0 / vals))
    b_free = free @ (a_inv @ a_inv)                       # (B f_i) for each free row, as rows
    grad_re = -2.0 * np.einsum("ik,ik->i", np.array(d_re), b_free)
    grad_im = -2.0 * np.einsum("ik,ik->i", np.array(d_im), b_free)

    excess = np.abs(points) ** 2 - radius ** 2
    out = excess > 0
    if np.any(out):
        cost += wall * float(np.sum(excess[out] ** 2))
        grad_re[out] += 4.0 * wall * excess[out] * points.real[out]
        grad_im[out] += 4.0 * wall * excess[out] * points.imag[out]
    return cost, np.concatenate([grad_re, grad_im])


def optimal_displacements(
    dim: int,
    n_disps: int | None = None,
    *,
    objective: str = "a-optimal",
    seed: int | None = 0,
    radius: float | None = None,
) -> np.ndarray:
    """A well-conditioned set of displacement points for reconstructing ``dim`` Fock levels.

    Places ``n_disps`` points (default ``2 dim^2``, twice the number of unknowns) to minimize the
    A-optimal cost ``Tr[(F^T F)^{-1}]``, with the origin always included, inside a soft wall at
    ``radius`` (default ``1.5 sqrt(dim)``, where a ``dim``-level state's Wigner function has decayed).
    The start is a Gaussian cloud of width ``sqrt(dim)/2``, drawn from ``seed``.

    Returns a complex array of length ``n_disps + 1``; element 0 is the origin.
    """
    if objective != "a-optimal":
        raise NotImplementedError(f"only the 'a-optimal' design is implemented; got {objective!r}")
    if n_disps is None:
        n_disps = 2 * dim * dim
    radius = 1.5 * np.sqrt(dim) if radius is None else float(radius)
    # Start from a Gaussian cloud the width of the Wigner functions being resolved: points far out
    # carry almost no information, so a uniform start in the disk is badly conditioned.
    rng = np.random.default_rng(seed)
    start = np.clip(rng.normal(0.0, np.sqrt(dim) / 2.0, size=(2, n_disps)), -radius, radius)
    x0 = start.reshape(-1)
    origin = _hermitian_features(wigner_operator(0.0, dim))[None, :]
    # Box bounds just outside the wall keep a line search from flinging a point to where its
    # Wigner row underflows to zero (a singular design); the wall shapes the disk inside them.
    box = 1.25 * radius
    result = minimize(
        _a_optimal_cost, x0, args=(dim, radius, origin), method="L-BFGS-B", jac=True,
        bounds=[(-box, box)] * (2 * n_disps), options=dict(ftol=1e-9, maxiter=500),
    )
    points = result.x[:n_disps] + 1j * result.x[n_disps:]
    return np.concatenate(([0.0 + 0.0j], points))
