"""Tests for DRAGPairCalibration — the offset-cancelling DRAG diagnostic.

Why it exists: a single X/2·Y/2 scan's equator crossing can land ~0.05 from the first-order
optimum. A 3-level model shows a P_e offset of +0.03 at the equator (readout/thermal/T1 bias) moves
that crossing by −0.044, because X/2·Y/2 has no reference. The pair X(π/2)·Y(π) vs Y(π/2)·X(π)
carries the DRAG error with OPPOSITE signs (model slopes +1.56 / −1.57 per unit α), so their
difference crosses zero at the optimum while any offset common to both cancels; the pair mean
measures that offset directly.

Model check (3-level, σ = 35 ns, Δ/2π = −143 MHz), crossing of each estimator:

    systematic              single X/2·Y/2    pair difference    pair mean − ½
    none                    0.0117            0.0114             0
    readout offset +0.03    −0.033            0.0114             +0.030
    amplitude ±3 %          0.013             0.0115             ∓0.023
    detuning +0.1 MHz       0.090             0.025              0        ← reduced ~6x, NOT cancelled

Data are fed hardware-shaped (averaged IQ on synthetic g/e blob centres). Pure: no qick.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.pulses import Delay, Measure, Pulse
from strumento.packs.cqed.experiments.drag_pair_calibration import DRAGPairCalibration
from tests._synthetic_readout import dispersive_iq, result_from_populations

# Synthetic ge blob centres (raw IQ) live in tests/_synthetic_readout.py, with the angle (g->e
# direction) and threshold (projected midpoint) derived from them.


@pytest.fixture
def readout():
    return dispersive_iq()


def _interleave(pa, pb):
    """Read order the sequence produces: A(α₀), B(α₀), A(α₁), B(α₁), …"""
    out = np.empty(2 * len(pa))
    out[0::2], out[1::2] = pa, pb
    return out


def _pair(alphas, a0, *, slope=1.56, offset=0.0):
    d = slope * (np.asarray(alphas) - a0)
    return np.clip(0.5 + offset + d, 0, 1), np.clip(0.5 + offset - d, 0, 1)


# --- Sequence -----------------------------------------------------------------------------------


def _blocks(seq):
    """Split ops into per-read blocks ending at each Measure."""
    blocks, cur = [], []
    for op in seq.ops:
        cur.append(op)
        if isinstance(op, Measure):
            blocks.append(cur)
            cur = []
    return blocks


def _drives(block):
    return [op for op in block if isinstance(op, Pulse) and op.label.startswith("drag_pair_")]


def test_sequence_interleaves_the_two_orders_each_from_a_reset(dev, readout):
    exp = DRAGPairCalibration(dev, alpha_start=-0.1, alpha_stop=0.1, points=5, readout=readout)
    seq, _ = exp.sequence()
    blocks = _blocks(seq)
    assert len(blocks) == 10  # two reads per α
    relax = float(dev.calib.readout.relax_delay)
    for k, block in enumerate(blocks):
        assert isinstance(block[0], Delay) and float(block[0].duration_us) >= relax
        labels = [p.label for p in _drives(block)]
        assert labels == (["drag_pair_x90", "drag_pair_y180"] if k % 2 == 0
                          else ["drag_pair_y90", "drag_pair_x180"])
        phases = [p.phase_deg for p in _drives(block)]
        assert phases == ([0.0, 90.0] if k % 2 == 0 else [90.0, 0.0])


@pytest.mark.parametrize("alpha", [0.0, 0.03, -0.2])
def test_pulses_are_the_factory_gates_at_that_alpha(dev, readout, alpha):
    exp = DRAGPairCalibration(dev, alpha_start=alpha, alpha_stop=alpha, points=1, readout=readout)
    seq, _ = exp.sequence()
    a_block, b_block = _blocks(seq)
    dev.calib.qubit.drag_alpha = alpha
    hpi, pi = dev.qubit.ge_hpi(), dev.qubit.ge_pi()
    x90, y180 = _drives(a_block)
    y90, x180 = _drives(b_block)
    for half in (x90, y90):
        assert half.envelope == hpi.envelope and half.gain == pytest.approx(hpi.gain)
    for full in (y180, x180):
        assert full.envelope == pi.envelope and full.gain == pytest.approx(pi.gain)


def test_large_alpha_uses_gain_headroom_on_both_pulse_types(dev, readout):
    exp = DRAGPairCalibration(dev, alpha_start=-1.5, alpha_stop=-1.5, points=1, readout=readout)
    seq, _ = exp.sequence()
    a_block, _ = _blocks(seq)
    x90, y180 = _drives(a_block)
    dev.calib.qubit.drag_alpha = 0.0
    assert x90.gain == pytest.approx(1.5 * dev.qubit.ge_hpi().gain)
    assert y180.gain == pytest.approx(1.5 * dev.qubit.ge_pi().gain)
    for p in (x90, y180):
        assert max(np.max(np.abs(p.envelope.idata)), np.max(np.abs(p.envelope.qdata))) <= 1.0


def test_rejects_alpha_beyond_limit(dev, readout):
    with pytest.raises(ValueError, match="alpha"):
        DRAGPairCalibration(dev, alpha_start=-2.5, alpha_stop=0.1, readout=readout)


def test_default_sweep_brackets_zero_and_stays_in_the_linear_difference_region(dev):
    """The difference saturates by |α| ≈ 0.3 and turns back beyond ~0.6 — keep the default inside."""
    exp = DRAGPairCalibration(dev)
    assert exp.alpha_start < 0.0 < exp.alpha_stop
    assert max(abs(exp.alpha_start), abs(exp.alpha_stop)) <= 0.3


# --- Analyzer -----------------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 0.03, -0.04])
def test_common_offset_cancels_in_the_crossing_and_is_reported(dev, readout, offset):
    alphas = np.linspace(-0.25, 0.25, 41)
    pa, pb = _pair(alphas, 0.012, offset=offset)
    exp = DRAGPairCalibration(dev, alpha_start=-0.25, alpha_stop=0.25, points=41, readout=readout)
    fit = exp.analyze(result_from_populations(_interleave(pa, pb)))
    assert fit.success
    assert fit.derived["alpha_opt"] == pytest.approx(0.012, abs=1e-6)
    assert fit.derived["drag_alpha"] == fit.derived["alpha_opt"]
    assert fit.derived["equator_offset"] == pytest.approx(offset, abs=1e-6)


def test_reads_are_deinterleaved_a_then_b(dev, readout):
    """Swapping A/B flips the difference's sign — the slope must come out positive for A − B."""
    alphas = np.linspace(-0.25, 0.25, 21)
    pa, pb = _pair(alphas, 0.0)
    exp = DRAGPairCalibration(dev, alpha_start=-0.25, alpha_stop=0.25, points=21, readout=readout)
    fit = exp.analyze(result_from_populations(_interleave(pa, pb)))
    assert fit.derived["slope"] > 0
    assert fit.derived["signal_a"] == pytest.approx(list(pa))
    assert fit.derived["signal_b"] == pytest.approx(list(pb))
    assert fit.derived["difference"] == pytest.approx(list(pa - pb))


def test_flat_difference_is_refused(dev, readout):
    rng = np.random.default_rng(1)
    pa = 0.5 + rng.normal(0, 0.05, 41)
    pb = 0.5 + rng.normal(0, 0.05, 41)
    exp = DRAGPairCalibration(dev, alpha_start=-0.25, alpha_stop=0.25, points=41, readout=readout)
    fit = exp.analyze(result_from_populations(_interleave(pa, pb)))
    assert not fit.success
    assert "alpha_opt" not in fit.derived


def test_crossing_outside_scan_is_refused(dev, readout):
    alphas = np.linspace(0.1, 0.25, 16)
    pa, pb = _pair(alphas, 0.0)
    exp = DRAGPairCalibration(dev, alpha_start=0.1, alpha_stop=0.25, points=16, readout=readout)
    fit = exp.analyze(result_from_populations(_interleave(pa, pb)))
    assert not fit.success and "alpha_opt" not in fit.derived


def test_refuses_without_blob_centres(dev):
    bare = dispersive_iq(with_centres=False)
    alphas = np.linspace(-0.25, 0.25, 21)
    pa, pb = _pair(alphas, 0.0)
    exp = DRAGPairCalibration(dev, alpha_start=-0.25, alpha_stop=0.25, points=21, readout=bare)
    fit = exp.analyze(result_from_populations(_interleave(pa, pb)))
    assert not fit.success and "alpha_opt" not in fit.derived
    assert fit.derived["population_anchored"] == 0.0


def test_unbiased_with_covering_error_bar_under_offset(dev, readout):
    """Binomial noise at 100 reps with a +0.03 common offset: mean within 0.002, ±1σ covers ~68%."""
    rng = np.random.default_rng(20261007)
    alphas = np.linspace(-0.25, 0.25, 41)
    a_true, reps = 0.012, 100
    pa, pb = _pair(alphas, a_true, offset=0.03)
    exp = DRAGPairCalibration(dev, alpha_start=-0.25, alpha_stop=0.25, points=41, readout=readout)
    est, cover, offs = [], 0, []
    for _ in range(300):
        fit = exp.analyze(result_from_populations(
            _interleave(rng.binomial(reps, pa) / reps, rng.binomial(reps, pb) / reps)))
        assert fit.success
        a0, se = fit.derived["alpha_opt"], fit.derived["alpha_opt_stderr"]
        est.append(a0)
        offs.append(fit.derived["equator_offset"])
        cover += abs(a0 - a_true) <= se
    assert np.mean(est) == pytest.approx(a_true, abs=0.002)
    assert 0.55 <= cover / 300 <= 0.80
    assert np.mean(offs) == pytest.approx(0.03, abs=0.003)


# --- Display ------------------------------------------------------------------------------------


def test_display_shows_both_orders_and_the_difference_against_alpha(dev, readout):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from strumento.packs.cqed.plot import _display_drag_pair

    alphas = np.linspace(-0.25, 0.25, 41)
    pa, pb = _pair(alphas, 0.012, offset=0.03)
    exp = DRAGPairCalibration(dev, alpha_start=-0.25, alpha_stop=0.25, points=41, readout=readout)
    result = result_from_populations(_interleave(pa, pb))
    fit = exp.analyze(result)
    fig = _display_drag_pair(result, fit)
    top, bottom = fig.axes[:2]
    assert "P_e" in top.get_ylabel()
    assert r"\alpha" in bottom.get_xlabel()
    first = top.get_lines()[0]
    assert np.asarray(first.get_xdata()) == pytest.approx(alphas)
    assert np.asarray(first.get_ydata()) == pytest.approx(pa)
    vlines = [ln for ln in bottom.get_lines() if ln.get_label().startswith(r"$\alpha_{opt}")]
    assert len(vlines) == 1 and r"\pm" in vlines[0].get_label()
    assert "offset" in fig.texts[0].get_text() if fig.texts else "offset" in top.get_title()
