"""Piccolo controls → the ``from_solution`` input: units, quadrature sign, shape (pure, #210).

Piccolo's two-quadrature transmon drive is ``H = 2π[u_x (a + a†) + u_y · i(a − a†)]`` (u in GHz,
t in ns). The hardware convention the optimal-control import consumes is ``H = ½(Ω a† + Ω* a)``
with ``Ω ∝ I + iQ`` (ν = Ω/2π in MHz, t in µs). Matching the two gives ``ν = 2000 (u_x − i u_y)``:
a factor of 2 (the ½ in the hardware convention) and a Q-sign flip, either of which silently
wrecks an imported pulse. These tests check the conversion against the two Hamiltonians directly,
not against documentation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from strumento.interop.piccolo import PICCOLO_GHZ_TO_RABI_MHZ, piccolo_controls_to_solution

L = 3
A = np.diag(np.sqrt(np.arange(1, L)), 1).astype(complex)
N_OP = A.conj().T @ A


def _h_piccolo(ux, uy):
    """Piccolo drive Hamiltonian, rad/ns (u in GHz)."""
    return 2 * np.pi * (ux * (A + A.conj().T) + uy * 1j * (A - A.conj().T))


def _h_hardware(nu_mhz):
    """Hardware-convention drive Hamiltonian, rad/µs (ν = Ω/2π in MHz, complex I + iQ)."""
    om = 2 * np.pi * nu_mhz
    return 0.5 * (om * A.conj().T + np.conj(om) * A)


@pytest.mark.parametrize("ux,uy", [(0.003, 0.0), (0.0, 0.002), (0.004, -0.0015), (-0.001, 0.003)])
def test_converted_controls_reproduce_the_piccolo_hamiltonian(ux, uy):
    times_ns = np.array([0.0, 1.0])
    sol = piccolo_controls_to_solution(times_ns, np.array([[ux, ux], [uy, uy]]), drive="qubit")
    nu = np.asarray(sol["controls"]["qubit"])[0]
    # rad/ns × 1000 = rad/µs
    np.testing.assert_allclose(_h_hardware(nu), 1000.0 * _h_piccolo(ux, uy), atol=1e-9)


def test_time_evolution_agrees_for_a_two_quadrature_pulse():
    """The propagator under Piccolo's H(u_x, u_y) equals the one under the hardware H(Ω).

    Both quadratures are non-zero and time-varying (a Gaussian I with a derivative-shaped Q, the
    DRAG-like case the sign trap mirrors), with an anharmonic drift so the Q sign matters to the
    outcome. Piecewise-constant evolution on the same knots, Piccolo in ns, hardware in µs.
    """
    n = 81
    t_ns = np.linspace(0.0, 40.0, n)
    s = 8.0
    ux = 0.012 * np.exp(-((t_ns - 20.0) ** 2) / (2 * s**2))
    uy = -0.4 * (t_ns - 20.0) / s**2 * ux                      # derivative-shaped, both signs
    delta_ghz = -0.2
    drift_ghz = (delta_ghz / 2) * (N_OP @ N_OP - N_OP)

    sol = piccolo_controls_to_solution(t_ns, np.vstack([ux, uy]), drive="qubit")
    t_us = np.asarray(sol["times"])
    nu = np.asarray(sol["controls"]["qubit"])
    assert np.any(nu.real != 0) and np.any(nu.imag != 0)

    U_p = np.eye(L, dtype=complex)
    U_h = np.eye(L, dtype=complex)
    for k in range(n - 1):
        dt_ns, dt_us = t_ns[k + 1] - t_ns[k], t_us[k + 1] - t_us[k]
        U_p = expm(-1j * (2 * np.pi * drift_ghz + _h_piccolo(ux[k], uy[k])) * dt_ns) @ U_p
        H_h = 2 * np.pi * (1000.0 * drift_ghz) + _h_hardware(nu[k])     # drift GHz → MHz
        U_h = expm(-1j * H_h * dt_us) @ U_h
    np.testing.assert_allclose(U_h, U_p, atol=1e-10)
    # and the pulse actually does something (not a vacuous identity-vs-identity agreement)
    assert abs(U_p[1, 0]) > 0.5


def test_units_and_sign_explicitly():
    assert PICCOLO_GHZ_TO_RABI_MHZ == 2000.0
    t = np.array([0.0, 140.0])
    pure_x = piccolo_controls_to_solution(t, np.array([[0.003, 0.003], [0.0, 0.0]]), drive="qubit")
    nu_x = np.asarray(pure_x["controls"]["qubit"])
    np.testing.assert_allclose(nu_x, [6.0, 6.0])                # real ν at 2000×
    assert np.all(nu_x.imag == 0)

    pure_y = piccolo_controls_to_solution(t, np.array([[0.0, 0.0], [0.001, 0.001]]), drive="qubit")
    nu_y = np.asarray(pure_y["controls"]["qubit"])
    np.testing.assert_allclose(nu_y, [-2.0j, -2.0j])            # −2000 i ×
    assert np.all(nu_y.real == 0)

    both = piccolo_controls_to_solution(t, np.array([[0.003, 0.003], [0.001, 0.001]]), drive="qubit")
    assert np.asarray(both["controls"]["qubit"])[0] == pytest.approx(6.0 - 2.0j)
    assert both["times"] == pytest.approx([0.0, 0.140])         # ns → µs
    assert list(both["controls"]) == ["qubit"]                 # keyed by the drive name


def test_rejects_malformed_controls():
    with pytest.raises(ValueError, match=r"\(2, N\)"):
        piccolo_controls_to_solution(np.array([0.0, 1.0]), np.zeros((3, 2)), drive="qubit")
    with pytest.raises(ValueError, match=r"\(2, N\)"):
        piccolo_controls_to_solution(np.array([0.0, 1.0]), np.zeros(2), drive="qubit")
    with pytest.raises(ValueError, match="times"):
        piccolo_controls_to_solution(np.array([0.0, 1.0, 2.0]), np.zeros((2, 2)), drive="qubit")
