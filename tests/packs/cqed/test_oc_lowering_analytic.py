"""Lowering an ANALYTIC pulse — the OC import path tested on an input authored by hand (#210).

The point: prove ``piccolo_controls_to_solution`` + ``from_solution`` produce a correct gate from
controls generated from first principles — no Piccolo solve — so the lowering is exercised
independently of whatever a solver yields. A plain Gaussian whose area is set (via the π ruler) to a
π or π/2 rotation is lowered through the production path and propagated through a 3-level transmon
Hamiltonian; it must implement the intended gate with little leakage.

Companion to ``test_oc_import_roundtrip.py`` (which round-trips a factory gate): here the input is
authored, not round-tripped, so it also covers the area → ruler → gain math on a pulse whose peak
differs from the calibrated π. Pure: no qick.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from strumento.packs.cqed.safety import MAX_QUBIT_DRIVE_GAIN

from ._oc_lowering import T_NS, analytic_controls, dac_rate, lower, pi_ruler, played_nu_mhz, unitary

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_TARGET = {"pi": _X, "hpi": expm(-1j * np.pi / 4 * _X)}

#: Average-gate infidelity tolerance on {g, e} for the lowered analytic gate (3-level propagation,
#: anharmonicity from the demo calibration). The residual is leakage + the off-subspace phase.
INFIDELITY_TOL = 1e-3


def _avg_gate_fidelity(u_sub, target):
    m = target.conj().T @ u_sub
    return (abs(np.trace(m.conj().T @ m)) + abs(np.trace(m)) ** 2) / 6.0


@pytest.mark.parametrize("gate", ["pi", "hpi"])
def test_analytic_pulse_lowers_to_the_intended_gate(dev, gate):
    ruler, fs = pi_ruler(dev), dac_rate(dev)
    t, u, _ = analytic_controls(gate)
    _, lowered = lower(dev, t, u, ruler=ruler)

    expected_n = int(round(T_NS * 1e-3 * fs)) + 1        # resampled to the DAC grid, not 71 knots
    assert len(lowered.envelope.idata) == expected_n
    limit = dev.calib.qubit.max_drive_gain or MAX_QUBIT_DRIVE_GAIN
    assert 0.0 < lowered.gain < limit                    # within the qubit-drive limit

    u3 = unitary(played_nu_mhz(lowered, ruler), fs, dev.calib.qubit.f_ef - dev.calib.qubit.f_ge)
    psi = u3 @ np.array([1, 0, 0], complex)
    assert abs(psi[1]) ** 2 == pytest.approx(1.0 if gate == "pi" else 0.5, abs=0.02)
    assert abs(psi[2]) ** 2 < 1e-3                       # negligible leakage
    assert 1.0 - _avg_gate_fidelity(u3[:2, :2], _TARGET[gate]) < INFIDELITY_TOL


def test_peak_amplitude_sets_gain_through_the_ruler(dev):
    """The authored ν_peak → gain fraction via the ruler: gain = ν_peak · pi_gain_frac / pi_rabi_mhz."""
    ruler = pi_ruler(dev)
    t, u, nu_peak = analytic_controls("pi")
    _, lowered = lower(dev, t, u, ruler=ruler)
    assert lowered.gain == pytest.approx(nu_peak * ruler.pi_gain_frac / ruler.pi_rabi_mhz, rel=1e-9)


@pytest.mark.parametrize("gate,key", [("pi", "pi_ge"), ("hpi", "hpi_ge")])
def test_lowered_pulse_is_close_to_the_factory_gate(dev, gate, key):
    """Frequency matching and gain safety against the factory gate it would stand in for.

    The analytic Gaussian is deliberately NOT identical to the factory gate (longer, with floored
    ends), so what must hold is weaker: the carrier is the factory's own ``f_ge`` to the digit, and
    the gain lands near the factory gain and well inside the drive limit.
    """
    ruler = pi_ruler(dev)
    t, u, _ = analytic_controls(gate)
    _, lowered = lower(dev, t, u, ruler=ruler)
    factory = dev.qubit.ge_pi() if key == "pi_ge" else dev.qubit.ge_hpi()

    assert lowered.freq_mhz == dev.calib.qubit.f_ge == factory.freq_mhz
    assert 0.0 < lowered.gain <= 0.8 * MAX_QUBIT_DRIVE_GAIN     # >= 20 % headroom to the limit
    assert lowered.gain == pytest.approx(factory.gain, rel=0.35)
