"""Tests for the ladder-population diagnostic helpers (pure: no qick).

The diagnostic localizes where the analytic Fock-|1> prep strands population. These tests cover the
two pieces that make it real: the per-block sequence construction, and the QUANTITATIVE leak
analyzer that must flag a partial (e.g. 26%) leak — the failure the earlier argmax version missed.
Host-batched acquisition itself lives in the caller's runner, outside this package.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.pulses import Measure
from strumento.packs.cqed.experiments.ladder_population import (
    LADDER_VARIANTS,
    analyze_ladder_populations,
    block_sequence,
    expected_level,
    iter_blocks,
    n_stages,
    photons_after,
    round_trip_sequence,
)

# --- pure combinatorics -----------------------------------------------------


def test_expected_level_walks_the_g_e_f_cycle():
    assert [expected_level(k) for k in range(7)] == [0, 1, 2, 0, 1, 2, 0]


def test_stage_and_photon_counts():
    assert n_stages(1) == 4 and n_stages(2) == 7
    # a photon is created only once each full ge/ef/f0g1 cycle completes (stage 3, 6, ...)
    assert [photons_after(k) for k in range(7)] == [0, 0, 0, 1, 1, 1, 2]


def test_iter_blocks_is_stage_major_variant_minor():
    blocks = list(iter_blocks(1))
    assert len(blocks) == 4 * 3
    assert blocks[:3] == [(0, "none"), (0, "ge"), (0, "ef")]


# --- block sequence construction --------------------------------------------


def test_block_sequence_has_exactly_one_measure(dev):
    for stage in range(n_stages(1)):
        for variant in LADDER_VARIANTS:
            seq, _ = block_sequence(dev, stage, variant, fock_n=1)
            assert sum(1 for op in seq.ops if isinstance(op, Measure)) == 1


def test_block_prefix_grows_with_stage(dev):
    """A later stage plays strictly more pulses before the (identical) map+measure tail."""
    counts = []
    for stage in range(n_stages(1)):
        seq, _ = block_sequence(dev, stage, "none", fock_n=1)
        counts.append(len(seq.ops))
    assert counts == sorted(counts) and counts[0] < counts[-1]


# --- quantitative leak analyzer (the core fix) ------------------------------


def _blocks(level_populations_per_stage):
    """Flatten direct g/e/f population triples in the runner's stage-major block order."""
    return np.repeat(np.array(level_populations_per_stage, dtype=float), len(LADDER_VARIANTS), axis=0)


def test_ideal_ladder_reports_no_leak():
    """Direct g/e/f population inversion resolves the ideal g->e->f->g ladder."""
    pe = _blocks([
        (0.98, 0.01, 0.01),
        (0.01, 0.98, 0.01),
        (0.01, 0.01, 0.98),
        (0.98, 0.01, 0.01),
    ])
    d = analyze_ladder_populations(pe, fock_n=1)
    assert d["leak_stage"] == -1
    assert d["worst_defect"] < 0.10
    assert d["residual_excitation_final"] == pytest.approx(0.02, abs=1e-6)


def test_detects_a_26_percent_f0g1_leak_at_the_return_stage():
    """A direct 26% residual f population after f0g1 is a stage-3 defect."""
    pe = _blocks([
        (0.98, 0.01, 0.01),
        (0.01, 0.98, 0.01),
        (0.01, 0.01, 0.98),
        (0.72, 0.02, 0.26),
    ])
    d = analyze_ladder_populations(pe, fock_n=1)
    assert d["leak_stage"] == 3
    assert d["s3_defect"] == pytest.approx(0.28, abs=1e-6)
    assert d["residual_excitation_final"] == pytest.approx(0.28, abs=1e-6)


def test_localizes_an_ef_leak_at_stage_two():
    """ef π incomplete: stage 2 fails to reach |f> (only 0.70), leak_stage=2 (before the f0g1)."""
    pe = _blocks([
        (0.98, 0.01, 0.01),
        (0.01, 0.98, 0.01),
        (0.20, 0.10, 0.70),
        (0.98, 0.01, 0.01),
    ])
    d = analyze_ladder_populations(pe, fock_n=1)
    assert d["leak_stage"] == 2
    assert d["s2_defect"] == pytest.approx(0.30, abs=1e-6)


def test_threshold_is_tunable():
    """A small (8%) defect passes the default 10% threshold but a stricter 5% flags it."""
    pe = _blocks([
        (0.92, 0.04, 0.04),
        (0.01, 0.98, 0.01),
        (0.01, 0.01, 0.98),
        (0.98, 0.01, 0.01),
    ])
    assert analyze_ladder_populations(pe, fock_n=1)["leak_stage"] == -1
    assert analyze_ladder_populations(pe, fock_n=1, leak_threshold=0.05)["leak_stage"] == 0


# --- round-trip pulse diagnostic ---------------------------------------------


def test_round_trip_sequence_retrieves_and_maps(dev):
    """The round trip runs the full ladder, a second f0g1, an ef map, then one read."""
    from strumento.core.pulses import Pulse

    seq, _ = round_trip_sequence(dev, fock_n=1)
    assert sum(1 for op in seq.ops if isinstance(op, Measure)) == 1
    labels = [getattr(op, "label", "") for op in seq.ops if isinstance(op, Pulse)]
    # forward ladder f0g1 + retrieve f0g1 = two sideband pulses; ge + two ef = qubit pulses
    assert sum(1 for lbl in labels if "f0-g1" in lbl or "f0g1" in lbl) == 2
    assert sum(1 for lbl in labels if lbl == "pi_ef") == 2
    assert sum(1 for lbl in labels if lbl == "pi_ge") == 1


def test_round_trip_sequence_single_photon_only(dev):
    with pytest.raises(NotImplementedError):
        round_trip_sequence(dev, fock_n=2)


def test_analyzer_records_the_round_trip_without_claiming_a_photon():
    ideal = _blocks([
        (0.98, 0.01, 0.01), (0.01, 0.98, 0.01), (0.01, 0.01, 0.98), (0.98, 0.01, 0.01),
    ])
    derived = analyze_ladder_populations(ideal, fock_n=1, round_trip_populations=(0.01, 0.97, 0.02))
    assert derived["round_trip_e_population"] == pytest.approx(0.97)
    assert "photon_confirmed" not in derived
    assert "photon_witness" not in derived


def test_analyzer_without_round_trip_omits_the_key():
    ideal = _blocks([
        (0.98, 0.01, 0.01), (0.01, 0.98, 0.01), (0.01, 0.01, 0.98), (0.98, 0.01, 0.01),
    ])
    assert "round_trip_e_population" not in analyze_ladder_populations(ideal, fock_n=1)
