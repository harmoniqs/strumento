"""Spin RB Clifford algebra (issue #211) — pure: no qick.

Two Clifford sets live in the spin pack:

- the **exchange-only** set (``library/eo_clifford.py``): the 24 single-qubit Cliffords, each
  decomposed into J₁₂ (logical ẑ) and J₂₃ (n̂ = (-√3/2, 0, -1/2)) exchange rotations;
- the **EDSR** set the registered ``RandomizedBenchmarking`` / ``InterleavedRB`` play
  (``experiments/randomized_benchmarking.py``): π and π/2 drive pulses at 0/90/180/270°.

For the exchange-only set this checks the group algebra: the Cayley table is closed, has an
identity, has permutation rows and columns, is associative, and each element composed with its
tabulated inverse is the identity; the 24 unitaries are pairwise-distinct Cliffords; the tables
agree with the unitaries; and each tabulated (axis, angle) decomposition multiplied out — from
rotations this test builds — equals its Clifford.

For the EDSR experiments it multiplies out what is **played**: every block, recovery included,
is the identity. Each pulse's rotation is read off the pulse (angle = π × gain·σ over the
calibrated π pulse's, axis = drive phase), never from the tables.

Equality is up to a global phase: :math:`|\\mathrm{tr}(U^\\dagger V)|/2 = 1` within ``ATOL``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from strumento.core.device import Device
from strumento.core.pulses import Gaussian, Measure, Pulse
from strumento.packs.spin.experiments.interleaved_rb import InterleavedRB
from strumento.packs.spin.experiments.randomized_benchmarking import RandomizedBenchmarking
from strumento.packs.spin.library import eo_clifford as eo
from tests._paths import SPIN_EO_YAML, SPIN_YAML

ATOL = 1e-9
#: The EO decompositions are found numerically (Nelder–Mead, accepted at 1 - overlap < 1e-6),
#: so their reconstruction is only that good.
EO_ATOL = 1e-6

_I = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = (_SX, _SY, _SZ)

#: The exchange axes of the 120° exchange-only geometry, stated here independently.
Z_AXIS = (0.0, 0.0, 1.0)
N23_AXIS = (-np.sqrt(3) / 2, 0.0, -0.5)


def _rotation(axis, angle: float) -> np.ndarray:
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    sigma = n[0] * _SX + n[1] * _SY + n[2] * _SZ
    return np.cos(angle / 2) * _I - 1j * np.sin(angle / 2) * sigma


def _overlap(u: np.ndarray, v: np.ndarray) -> float:
    return float(abs(np.trace(u.conj().T @ v)) / 2.0)


def _same(u, v, atol=ATOL) -> bool:
    return abs(_overlap(u, v) - 1.0) < atol


# -- exchange-only: the group algebra -------------------------------------------------------


def test_eo_cayley_table_is_closed_with_permutation_rows_and_columns():
    t = np.asarray(eo.EO_CAYLEY_TABLE)
    n = eo.N_CLIFFORDS
    assert t.shape == (n, n) and len(eo._CLIFFORDS) == n
    for i in range(n):
        assert sorted(t[i, :].tolist()) == list(range(n)), f"row {i} is not a permutation"
        assert sorted(t[:, i].tolist()) == list(range(n)), f"column {i} is not a permutation"


def test_eo_cayley_table_has_c0_as_its_only_identity_and_is_associative():
    t = np.asarray(eo.EO_CAYLEY_TABLE)
    n = eo.N_CLIFFORDS
    assert [e for e in range(n) if all(t[e, j] == j and t[j, e] == j for j in range(n))] == [0]
    for a, b, c in itertools.product(range(n), repeat=3):
        assert t[t[a, b], c] == t[a, t[b, c]], (a, b, c)


def test_eo_every_element_composed_with_its_tabulated_inverse_is_the_identity():
    t, inv = np.asarray(eo.EO_CAYLEY_TABLE), np.asarray(eo.EO_INVERSE_TABLE)
    for i in range(eo.N_CLIFFORDS):
        assert t[i, inv[i]] == 0 and t[inv[i], i] == 0, f"C{i} and C{inv[i]} are not inverses"


def test_eo_cliffords_are_24_pairwise_distinct_cliffords_matching_the_tables():
    cliffs = eo._CLIFFORDS
    for i, j in itertools.combinations(range(len(cliffs)), 2):
        assert not _same(cliffs[i], cliffs[j]), f"C{i} and C{j} are the same unitary"
    for i, u in enumerate(cliffs):
        for p in _PAULIS:
            image = u @ p @ u.conj().T
            assert any(abs(abs(np.trace(q @ image)) / 2 - 1) < ATOL for q in _PAULIS), (
                f"C{i} does not map a Pauli to a Pauli"
            )
    for i, j in itertools.product(range(eo.N_CLIFFORDS), repeat=2):
        assert _same(cliffs[i] @ cliffs[j], cliffs[int(eo.EO_CAYLEY_TABLE[i, j])])
    for i in range(eo.N_CLIFFORDS):
        assert _same(cliffs[i].conj().T, cliffs[int(eo.EO_INVERSE_TABLE[i])])


def test_eo_each_tabulated_exchange_decomposition_multiplies_out_to_its_clifford():
    axes = {"z": Z_AXIS, "n": N23_AXIS}
    for i, steps in enumerate(eo.EO_DECOMPOSITIONS):
        u = _I.copy()
        for axis, angle in steps:
            u = _rotation(axes[axis], angle) @ u
        assert _same(u, eo._CLIFFORDS[i], atol=EO_ATOL), (
            f"EO C{i}: decomposition gives |tr|/2 = {_overlap(u, eo._CLIFFORDS[i]):.8f}"
        )


# -- exchange-only RB: what is played returns to the start ----------------------------------


def _eo_played_unitary(seq, b12, b23) -> np.ndarray:
    """The product of the exchange pulses ``seq`` plays, each read off the pulse itself.

    Exchange is non-negative, so a pulse rotates by π × length / (calibrated π length) about
    its barrier's axis — never by a signed angle taken from the decomposition tables.
    """
    axis = {b12.pulse().line: Z_AXIS, b23.pulse().line: N23_AXIS}
    pi_len = {b12.pulse().line: b12.pulse().envelope.length_us,
              b23.pulse().line: b23.pulse().envelope.length_us}
    u = _I.copy()
    for op in seq.ops:
        assert isinstance(op, Pulse), f"unexpected {type(op).__name__} in an EO RB block"
        u = _rotation(axis[op.line], np.pi * op.envelope.length_us / pi_len[op.line]) @ u
    return u


@pytest.fixture
def spin_eo_dev() -> Device:
    return Device.load(SPIN_EO_YAML)


@pytest.mark.parametrize("seed,depth", list(itertools.product([0, 1, 2, 3, 4], [1, 2, 5, 20])))
def test_eo_played_rb_block_including_recovery_is_the_identity(spin_eo_dev, seed, depth):
    b12, b23 = spin_eo_dev.b12, spin_eo_dev.b23
    seq = eo.build_eo_rb_block(b12, b23, depth, np.random.default_rng(seed))
    u = _eo_played_unitary(seq, b12, b23)
    assert _same(u, _I, atol=EO_ATOL), f"|tr U|/2 = {_overlap(u, _I):.6f}"


def test_each_played_eo_clifford_is_its_own_unitary(spin_eo_dev):
    # Every one of the 24, so a negative tabulated angle is played as that rotation, not its mirror.
    from strumento.core.pulses import Seq

    b12, b23 = spin_eo_dev.b12, spin_eo_dev.b23
    for i in range(eo.N_CLIFFORDS):
        seq = Seq()
        eo.play_eo_clifford(b12, b23, i, seq)
        u = _eo_played_unitary(seq, b12, b23)
        assert _same(u, eo._CLIFFORDS[i], atol=EO_ATOL), (
            f"EO C{i} as played: |tr|/2 = {_overlap(u, eo._CLIFFORDS[i]):.6f}"
        )


# -- EDSR RB: what is played returns to the start ------------------------------------------


@pytest.fixture
def spin_ld_dev() -> Device:
    return Device.load(SPIN_YAML)


def _played_unitary(pulse: Pulse, pi_ref: Pulse) -> np.ndarray:
    assert pulse.line == pi_ref.line, f"an RB gate played on {pulse.line}"
    assert float(pulse.freq_mhz) == pytest.approx(float(pi_ref.freq_mhz)), "off resonance"
    assert isinstance(pulse.envelope, Gaussian) and isinstance(pi_ref.envelope, Gaussian)
    area = float(pulse.gain) * pulse.envelope.sigma_us
    angle = np.pi * area / (float(pi_ref.gain) * pi_ref.envelope.sigma_us)
    phi = np.deg2rad(float(pulse.phase_deg))
    return _rotation((np.cos(phi), np.sin(phi), 0.0), angle)


def _assert_blocks_are_identity(seq, pi_ref, n_blocks) -> int:
    u, gates, blocks = _I.copy(), 0, 0
    for op in seq.ops:
        if isinstance(op, Measure):
            assert _same(u, _I), f"block {blocks}: the played product is not the identity"
            u, blocks = _I.copy(), blocks + 1
            continue
        assert isinstance(op, Pulse), f"unexpected {type(op).__name__} inside an RB block"
        u = _played_unitary(op, pi_ref) @ u
        gates += 1
    assert blocks == n_blocks
    return gates


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_spin_rb_every_played_block_including_recovery_is_the_identity(spin_ld_dev, seed):
    exp = RandomizedBenchmarking(spin_ld_dev, dot_name="q0", depth_start=1, depth_stop=60,
                                 points=6, seed=seed)
    seq, _ = exp.sequence()
    assert _assert_blocks_are_identity(seq, spin_ld_dev.q0.pi(), 6) > 100


@pytest.mark.parametrize("gate", ["pi", "pi_y", "hpi", "hpi_y"])
def test_spin_interleaved_rb_every_played_block_is_the_identity(spin_ld_dev, gate):
    exp = InterleavedRB(spin_ld_dev, dot_name="q0", gate=gate, depth_start=1, depth_stop=30,
                        points=4, seed=11)
    seq, _ = exp.sequence()
    assert _assert_blocks_are_identity(seq, spin_ld_dev.q0.pi(), 4) > 40
