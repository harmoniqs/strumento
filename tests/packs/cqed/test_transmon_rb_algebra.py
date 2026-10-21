"""TransmonRB's Clifford algebra, end to end (issue #211) — pure: no qick.

The timing/wiring tests in ``test_transmon_rb.py`` cannot see the algebra. A wrong entry in a
Clifford decomposition, a wrong inverse, or a sign error in a native gate's phase still produces
a clean exponential decay and a fidelity number, because the recovery gate is built from the same
tables. These tests check the algebra three ways:

1. **The tables are a group.** The Cayley table is closed, has an identity, every row and column
   is a permutation, it is associative, and each element composed with its tabulated inverse is
   the identity.
2. **The tables match the unitaries.** Each decomposition, multiplied out from native gates the
   test builds itself (axis and angle, not the module's matrices), gives 24 pairwise-distinct
   Cliffords, and the Cayley and inverse tables agree with those products.
3. **What is played returns to |g>.** For several seeds and depths, every block
   ``TransmonRB.sequence()`` emits — recovery gate included — multiplies out to the identity. Each
   pulse's unitary comes from the pulse itself: its rotation angle is π times its pulse area
   (gain × envelope area) over the area of the device's calibrated π pulse, and its axis is its
   drive phase, :math:`\\hat n = (\\cos\\phi, \\sin\\phi, 0)`. An identity wait is the identity.
   Nothing is read back from the Clifford tables.

The played-sequence test is proved non-vacuous by mutation: corrupting one decomposition,
flipping one native gate's phase sign in the tables, or flipping one played phase makes it fail.
The mutations are monkeypatched; the production tables are never edited.

Equality is up to a global phase: :math:`|\\mathrm{tr}(U^\\dagger V)|/2 = 1` within ``ATOL``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from strumento.core.pulses import Arb, Delay, Gaussian, Measure, Pulse
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments import transmon_rb as rb
from strumento.packs.cqed.experiments.transmon_rb import TransmonRB

ATOL = 1e-9

_I = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = (_SX, _SY, _SZ)


def _rotation(axis, angle: float) -> np.ndarray:
    """exp(-i angle/2 · n̂·σ⃗) for a unit vector ``axis``."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    sigma = n[0] * _SX + n[1] * _SY + n[2] * _SZ
    return np.cos(angle / 2) * _I - 1j * np.sin(angle / 2) * sigma


def _overlap(u: np.ndarray, v: np.ndarray) -> float:
    return float(abs(np.trace(u.conj().T @ v)) / 2.0)


def _same(u: np.ndarray, v: np.ndarray) -> bool:
    return abs(_overlap(u, v) - 1.0) < ATOL


#: The native gates, from their stated axis and angle — independent of the module's matrices.
#: Index order is the module's: I, X, Y, X/2, -X/2, Y/2, -Y/2.
_X_AXIS, _Y_AXIS = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
NATIVE = [
    _I,
    _rotation(_X_AXIS, np.pi),
    _rotation(_Y_AXIS, np.pi),
    _rotation(_X_AXIS, np.pi / 2),
    _rotation(_X_AXIS, -np.pi / 2),
    _rotation(_Y_AXIS, np.pi / 2),
    _rotation(_Y_AXIS, -np.pi / 2),
]


def _multiply_out(decomposition) -> np.ndarray:
    """The decomposition's unitary: gates applied left to right (the first acts first)."""
    u = _I.copy()
    for g in decomposition:
        u = NATIVE[g] @ u
    return u


def _cliffords() -> list[np.ndarray]:
    return [_multiply_out(d) for d in rb._CLIFFORD_DECOMPOSITIONS]


# -- 1. the tables are a group ------------------------------------------------------------


def test_the_cayley_table_is_closed_with_permutation_rows_and_columns():
    table = np.asarray(rb.CAYLEY_TABLE)
    n = rb.N_CLIFFORDS
    assert table.shape == (n, n)
    assert len(rb._CLIFFORD_DECOMPOSITIONS) == n
    everything = list(range(n))
    for i in range(n):
        assert sorted(table[i, :].tolist()) == everything, f"row {i} is not a permutation"
        assert sorted(table[:, i].tolist()) == everything, f"column {i} is not a permutation"


def test_the_cayley_table_has_an_identity_and_it_is_c0():
    table = np.asarray(rb.CAYLEY_TABLE)
    n = rb.N_CLIFFORDS
    identities = [e for e in range(n)
                  if all(table[e, j] == j and table[j, e] == j for j in range(n))]
    assert identities == [0]


def test_the_cayley_table_is_associative():
    t = np.asarray(rb.CAYLEY_TABLE)
    for a, b, c in itertools.product(range(rb.N_CLIFFORDS), repeat=3):
        assert t[t[a, b], c] == t[a, t[b, c]], (a, b, c)


def test_every_element_composed_with_its_tabulated_inverse_is_the_identity():
    t, inv = np.asarray(rb.CAYLEY_TABLE), np.asarray(rb.INVERSE_TABLE)
    for i in range(rb.N_CLIFFORDS):
        assert t[i, inv[i]] == 0, f"C{i} · C{inv[i]} = C{t[i, inv[i]]}, not the identity"
        assert t[inv[i], i] == 0, f"C{inv[i]} · C{i} = C{t[inv[i], i]}, not the identity"


# -- 2. the tables match the unitaries ----------------------------------------------------


def test_the_module_native_gates_are_the_stated_rotations():
    assert len(rb._NATIVE_GATES) == len(NATIVE)
    for k, (got, want) in enumerate(zip(rb._NATIVE_GATES, NATIVE)):
        assert _same(got, want), f"native gate {k} is not its stated axis/angle rotation"


def test_the_24_decompositions_are_pairwise_distinct_cliffords():
    cliffs = _cliffords()
    for i, j in itertools.combinations(range(len(cliffs)), 2):
        assert not _same(cliffs[i], cliffs[j]), f"C{i} and C{j} are the same unitary"
    # each maps every Pauli to ± a Pauli: a Clifford, not just a distinct unitary
    for i, u in enumerate(cliffs):
        for p in _PAULIS:
            image = u @ p @ u.conj().T
            assert any(abs(abs(np.trace(q @ image)) / 2 - 1) < ATOL for q in _PAULIS), (
                f"C{i} does not map a Pauli to a Pauli"
            )


def test_the_cayley_and_inverse_tables_agree_with_the_multiplied_out_unitaries():
    cliffs = _cliffords()
    for i, j in itertools.product(range(rb.N_CLIFFORDS), repeat=2):
        k = int(rb.CAYLEY_TABLE[i, j])
        assert _same(cliffs[i] @ cliffs[j], cliffs[k]), f"C{i}·C{j} is not C{k}"
    for i in range(rb.N_CLIFFORDS):
        k = int(rb.INVERSE_TABLE[i])
        assert _same(cliffs[i].conj().T, cliffs[k]), f"C{i}† is not C{k}"


# -- 3. what is played returns to |g> -------------------------------------------------------

DRIVE = LineRef("qubit", "drive")


def _area(pulse: Pulse) -> float:
    """Signed pulse area in units shared by every pulse of the same envelope type.

    A Gaussian truncated at a fixed number of σ has area ∝ σ; an ``Arb`` (DRAG) sampled at one
    DAC rate has area ∝ the sum of its I samples. DRAG's Q quadrature does not change the ideal
    rotation, so it is ignored.
    """
    env = pulse.envelope
    if isinstance(env, Gaussian):
        return float(pulse.gain) * float(env.sigma_us)
    if isinstance(env, Arb):
        return float(pulse.gain) * float(np.sum(env.idata))
    raise TypeError(f"no pulse area for a {type(env).__name__} gate")


def _played_unitary(pulse: Pulse, pi_ref: Pulse, f_ge: float) -> np.ndarray:
    """The rotation a played drive pulse performs, read off the pulse alone."""
    assert pulse.line == DRIVE, f"an RB gate played on {pulse.line}"
    assert float(pulse.freq_mhz) == pytest.approx(f_ge), "an RB gate played off resonance"
    assert type(pulse.envelope) is type(pi_ref.envelope), "gate and π reference differ in shape"
    angle = np.pi * _area(pulse) / _area(pi_ref)
    phi = np.deg2rad(float(pulse.phase_deg))
    return _rotation((np.cos(phi), np.sin(phi), 0.0), angle)


def _blocks(seq) -> list[list]:
    blocks, cur = [], []
    for op in seq.ops:
        cur.append(op)
        if isinstance(op, Measure):
            blocks.append(cur)
            cur = []
    assert not cur, "ops after the last Measure"
    return blocks


def _block_unitary(block, pi_ref: Pulse, f_ge: float) -> tuple[np.ndarray, int]:
    """Product of one depth block's played gates (reset → gates → probe → Measure)."""
    assert isinstance(block[0], Delay), "a depth block does not start with its reset"
    assert isinstance(block[-1], Measure)
    probe = block[-2]
    assert isinstance(probe, Pulse) and probe.label == "ro_probe"
    u, n_gates = _I.copy(), 0
    for op in block[1:-2]:
        if isinstance(op, Delay):
            continue  # an identity Clifford's calibrated wait
        if not isinstance(op, Pulse):
            raise AssertionError(f"unexpected {type(op).__name__} inside an RB block")
        u = _played_unitary(op, pi_ref, f_ge) @ u
        n_gates += 1
    return u, n_gates


SEEDS = (0, 1, 7, 42, 2024)
SCHEDULES = ((1, 1, 1), (2, 2, 1), (1, 40, 5), (3, 120, 4))


def _assert_played_blocks_return_to_identity(dev, seeds=SEEDS, schedules=SCHEDULES) -> int:
    """Every block's played product is the identity; returns the gates checked."""
    pi_ref = dev.qubit.ge_pi()  # the device's calibrated π: the angle scale
    f_ge = float(dev.calib.qubit.f_ge)
    checked = 0
    for seed, (start, stop, points) in itertools.product(seeds, schedules):
        exp = TransmonRB(dev, depth_start=start, depth_stop=stop, points=points, seed=seed)
        seq, _ = exp.sequence()
        blocks = _blocks(seq)
        assert len(blocks) == points
        for depth, block in zip(exp._depths, blocks):
            u, n = _block_unitary(block, pi_ref, f_ge)
            assert n >= 1 or depth == 0
            ov = _overlap(_I, u)
            assert abs(ov - 1.0) < 1e-9, (
                f"seed={seed} depth={int(depth)}: the played block is not the identity "
                f"(|tr U|/2 = {ov:.6f})"
            )
            checked += n
    return checked


@pytest.mark.parametrize("drag_alpha", [None, -0.05], ids=["gaussian", "drag"])
def test_every_played_block_including_recovery_is_the_identity(dev, drag_alpha):
    dev.calib.qubit.drag_alpha = drag_alpha
    assert _assert_played_blocks_return_to_identity(dev) > 1000


def test_a_single_clifford_without_its_recovery_is_not_the_identity(dev):
    """The reconstruction is not blind: drop the recovery gate and the product moves."""
    pi_ref, f_ge = dev.qubit.ge_pi(), float(dev.calib.qubit.f_ge)
    for idx in range(1, rb.N_CLIFFORDS):
        seq = rb.Seq()
        rb._play_clifford(dev.qubit, idx, seq)
        u = _I.copy()
        for op in seq.ops:
            u = _played_unitary(op, pi_ref, f_ge) @ u
        assert not _same(u, _I), f"C{idx} played as the identity"
        assert _same(u, _cliffords()[idx]), f"C{idx} is not played as its decomposition"


# -- 4. mutation: the end-to-end test is not vacuous ----------------------------------------


def _retabulate(monkeypatch):
    """Rebuild the tables from the (mutated) module state, as a module reload would."""
    cayley, inverse = rb._compute_clifford_tables()
    monkeypatch.setattr(rb, "CAYLEY_TABLE", cayley)
    monkeypatch.setattr(rb, "INVERSE_TABLE", inverse)


def test_mutation_a_corrupted_decomposition_fails_the_played_check(dev, monkeypatch):
    decomps = list(rb._CLIFFORD_DECOMPOSITIONS)
    decomps[12] = (5, 3)  # C12 was X/2 then Y/2; now Y/2 then X/2
    monkeypatch.setattr(rb, "_CLIFFORD_DECOMPOSITIONS", decomps)
    _retabulate(monkeypatch)
    with pytest.raises(AssertionError, match="not the identity"):
        _assert_played_blocks_return_to_identity(dev)


def test_mutation_a_stale_table_after_a_decomposition_edit_fails_the_played_check(dev, monkeypatch):
    decomps = list(rb._CLIFFORD_DECOMPOSITIONS)
    decomps[8] = (1, 6)  # C8 was X then Y/2; now X then -Y/2 — the tables are not rebuilt
    monkeypatch.setattr(rb, "_CLIFFORD_DECOMPOSITIONS", decomps)
    with pytest.raises(AssertionError, match="not the identity"):
        _assert_played_blocks_return_to_identity(dev)


def test_mutation_a_native_gate_phase_sign_flip_fails_the_played_check(dev, monkeypatch):
    natives = list(rb._NATIVE_GATES)
    natives[5] = rb._rot(rb._Y, -np.pi / 2)  # the tables believe Y/2 is -Y/2; 90° is still played
    monkeypatch.setattr(rb, "_NATIVE_GATES", natives)
    _retabulate(monkeypatch)
    with pytest.raises(AssertionError, match="not the identity"):
        _assert_played_blocks_return_to_identity(dev)


def test_mutation_a_played_phase_sign_flip_fails_the_played_check(dev, monkeypatch):
    real_ops = TransmonRB._ops

    class _FlippedY:
        """The ops, but a Y/2 plays at -90° (a sign error in the played phase)."""

        def __init__(self, ops):
            self._ops = ops

        def __getattr__(self, name):
            return getattr(self._ops, name)

        def ge_hpi(self, *, phase=0.0, **kw):
            return self._ops.ge_hpi(phase=-90.0 if phase == 90.0 else phase, **kw)

    monkeypatch.setattr(TransmonRB, "_ops", lambda self: _FlippedY(real_ops(self)))
    with pytest.raises(AssertionError, match="not the identity"):
        _assert_played_blocks_return_to_identity(dev)


def test_the_mutations_left_the_production_tables_alone(dev):
    """Ordering guard: runs after the mutation tests; the module is still correct."""
    test_the_cayley_and_inverse_tables_agree_with_the_multiplied_out_unitaries()
    assert _assert_played_blocks_return_to_identity(dev, seeds=(3,), schedules=((1, 30, 3),)) > 0
