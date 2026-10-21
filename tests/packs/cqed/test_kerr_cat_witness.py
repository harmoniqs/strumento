"""Pure contracts for the sparse Kerr-cat witness helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from strumento.core.pulses import Delay, Measure, Pulse
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.readout import DispersiveIQ


def test_ring_points_are_evenly_spaced_at_the_requested_radius():
    from strumento.packs.cqed.experiments.kerr_cat_witness import ring_points

    points = ring_points(radius=0.25, count=8)

    assert np.allclose(np.abs(points), 0.25)
    assert points[0] == pytest.approx(0.25 + 0.0j)
    assert points[2] == pytest.approx(0.0 + 0.25j)


def test_witness_sequence_applies_the_negative_sample_displacement_before_parity(dev):
    from strumento.packs.cqed.experiments.kerr_cat_witness import witness_sequence

    delays = Sweep(start=80.0, stop=150.0, on="t")
    seq, sweeps = witness_sequence(
        dev,
        sample_alpha=0.25 + 0.0j,
        kerr_wait=delays,
        init_alpha=2.0,
        mode_name="cavity",
    )

    assert sweeps == ()
    assert isinstance(seq.ops[1], Delay)
    assert seq.ops[1].duration_us is delays
    assert isinstance(seq.ops[2], Pulse)
    assert seq.ops[2].phase_deg == pytest.approx(180.0)
    labels = [op.label for op in seq.ops if isinstance(op, Pulse)]
    assert labels == ["displace_cavity", "displace_cavity", "hpi_ge", "hpi_ge", "ro_probe"]
    assert sum(isinstance(op, Measure) for op in seq.ops) == 1


def test_witness_analysis_selects_an_interior_baseline_subtracted_contrast_peak():
    from strumento.packs.cqed.experiments.kerr_cat_witness import analyze_witness

    delays = np.array([80.0, 100.0, 120.0])
    alphas = np.array([0.25, 0.25j, -0.25, -0.25j])
    baseline_w = np.array([0.01, -0.01, 0.01, -0.01])
    sweep_w = np.array(
        [
            [0.02, 0.50, 0.03],
            [-0.02, -0.50, -0.03],
            [0.01, 0.30, 0.02],
            [-0.01, -0.30, -0.02],
        ]
    )
    mode_cal = SimpleNamespace(alpha_scale=1.0, parity_sign=1.0, pe_midpoint=0.5)
    readout = DispersiveIQ(threshold=0.0, g_center=(0.0, 0.0), e_center=(1.0, 0.0))

    def raw_from_wigner(w):
        pe = 0.5 - np.pi * np.asarray(w) / 4.0
        return np.stack((pe, np.zeros_like(pe)), axis=-1)

    result = analyze_witness(
        delays=delays,
        sample_alphas=alphas,
        control_iq=raw_from_wigner(baseline_w),
        sweep_iq=raw_from_wigner(sweep_w),
        readout=readout,
        mode_cal=mode_cal,
    )

    assert result["best_wait_us"] == pytest.approx(100.0)
    assert result["peak_is_interior"] is True
    assert result["baseline_ring_contrast"] == pytest.approx(0.02)
    assert result["peak_ring_contrast"] == pytest.approx(1.0)
    assert result["peak_excess_ring_contrast"] == pytest.approx(0.98)


def test_interleaved_witness_selects_the_peak_after_pointwise_control_subtraction():
    from strumento.packs.cqed.experiments.kerr_cat_witness import analyze_interleaved_witness

    waits = np.array([90.0, 105.0, 115.0, 125.0, 140.0])
    alphas = np.array([0.25, 0.25j, -0.25, -0.25j])
    controls_w = np.array(
        [
            [0.10, -0.03, 0.02, -0.08],
            [0.07, -0.02, 0.04, -0.06],
            [0.12, -0.05, 0.03, -0.07],
            [0.09, -0.04, 0.01, -0.05],
            [0.11, -0.03, 0.02, -0.09],
        ]
    )
    signal_w = np.array(
        [
            [0.02, -0.02, 0.01, -0.01],
            [0.10, -0.10, 0.06, -0.06],
            [0.50, -0.50, 0.30, -0.30],
            [0.12, -0.12, 0.07, -0.07],
            [0.03, -0.03, 0.02, -0.02],
        ]
    )
    mode_cal = SimpleNamespace(alpha_scale=1.0, parity_sign=1.0, pe_midpoint=0.5)
    readout = DispersiveIQ(threshold=0.0, g_center=(0.0, 0.0), e_center=(1.0, 0.0))

    def raw_from_wigner(w):
        pe = 0.5 - np.pi * np.asarray(w) / 4.0
        return np.stack((pe, np.zeros_like(pe)), axis=-1)

    result = analyze_interleaved_witness(
        waits_us=waits,
        sample_alphas=alphas,
        control_iq=raw_from_wigner(controls_w),
        witness_iq=raw_from_wigner(controls_w + signal_w),
        readout=readout,
        mode_cal=mode_cal,
    )

    assert result["best_wait_us"] == pytest.approx(115.0)
    assert result["peak_is_interior"] is True
    assert result["peak_corrected_ring_contrast"] == pytest.approx(1.0)
