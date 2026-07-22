"""Tests for typed coupling edges + the pack-registered form registry + patterns (D31)."""

import numpy as np
import pytest

import strumento.packs.cqed  # noqa: F401  — importing the pack registers its coupling forms
from strumento.core.coupling import Coupling, grid_nn, resolve_form, ring_couple
from strumento.core.operators import HilbertSpace
from strumento.packs.cqed.couplings import ZZ, BeamSplitter, Dispersive


def test_cqed_forms_are_registered():
    for name in ("Dispersive", "ZZ", "Exchange", "BeamSplitter", "JaynesCummings"):
        assert resolve_form(name).__name__ == name


def test_unknown_form_raises():
    with pytest.raises(KeyError, match="no coupling form"):
        resolve_form("Nonexistent")


def test_coupling_edge_fields_and_symmetry():
    c = Coupling("manipulate", "S3", form=BeamSplitter, mediated_by="coupler")
    assert c.endpoints == ("manipulate", "S3")
    assert c.mediated_by == "coupler"
    assert c.symmetry == "symmetric"
    assert Coupling("a", "b", form="BeamSplitter").symmetry == "symmetric"  # resolves by name too


class _FakeBuilder:
    def __init__(self):
        self.edges = []

    def coupling(self, a, b, *, form, mediated_by=None):
        self.edges.append((a, b))


def test_ring_couple_emits_a_ring():
    b = _FakeBuilder()
    ring_couple(b, ["q0", "q1", "q2"], form=ZZ)
    assert b.edges == [("q0", "q1"), ("q1", "q2"), ("q2", "q0")]


def test_grid_nn_emits_nearest_neighbors():
    b = _FakeBuilder()
    grid_nn(b, 2, 2, form=ZZ)
    assert set(b.edges) == {("q0_0", "q0_1"), ("q0_0", "q1_0"), ("q0_1", "q1_1"), ("q1_0", "q1_1")}


def test_form_operators_hermitian_and_structurally_distinct():
    sp = HilbertSpace(["a", "b"], [2, 2])
    oa, ob = sp.ops("a"), sp.ops("b")
    disp = Dispersive.operator(oa, ob, 1.0)
    bs = BeamSplitter.operator(oa, ob, 1.0)
    for h in (disp, bs, ZZ.operator(oa, ob, 1.0)):
        assert np.allclose(h, h.conj().T)                    # hermitian
    assert np.allclose(disp, np.diag(np.diag(disp)))          # dispersive is diagonal (n_a n_b)
    assert not np.allclose(bs, np.diag(np.diag(bs)))          # beamsplitter is off-diagonal (swap)
