"""Pure contracts for the host-batched self-Kerr Ramsey diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.pulses import Delay, Measure, Pulse
from strumento.core.sweeps import Sweep


def test_self_kerr_ramsey_row_is_displace_wait_joint_return_then_parity(dev):
    """The Kerr wait and virtual return phase ride one delay axis; parity adds its own wait."""
    from strumento.packs.cqed.experiments.self_kerr_ramsey import row_sequence

    delays = Sweep(start=0.0, stop=20.0, on="t")
    seq, sweeps = row_sequence(
        dev,
        alpha=1.0,
        delays=delays,
        mode_name="cavity",
        reference_detuning_mhz=0.02,
        return_phase_offset_deg=90.0,
    )

    assert sweeps == ()
    assert isinstance(seq.ops[0], Pulse)
    assert isinstance(seq.ops[1], Delay)
    assert seq.ops[1].duration_us is delays
    assert isinstance(seq.ops[2], Pulse)
    return_phase = seq.ops[2].phase_deg
    assert isinstance(return_phase, Sweep)
    assert return_phase.on == "t"
    assert return_phase.start == pytest.approx(270.0)
    assert return_phase.stop == pytest.approx(270.0 + 360.0 * 0.02 * 20.0)
    # The later delay is the independently calibrated qubit-cavity parity revival.
    assert isinstance(seq.ops[4], Delay)
    assert seq.ops[4].duration_us == pytest.approx(1.6)
    assert isinstance(seq.ops[-1], Measure)


def test_self_kerr_fit_recovers_signed_k_and_cat_time_from_multiple_amplitudes():
    """The virtual return phase makes the Kerr-sign convention identifiable in synthetic data."""
    from strumento.packs.cqed.experiments.self_kerr_ramsey import (
        analyze_self_kerr_ramsey,
        expected_parity,
    )

    amplitudes = np.repeat(np.array([0.5, 1.0, 1.5, 2.0]), 2)
    return_phase_offsets = np.tile(np.array([0.0, 90.0]), 4)
    delays = np.linspace(0.0, 80.0, 61)
    k_true = -0.0035
    detuning_true = 0.006
    reference_detuning = 0.02
    ideal = np.stack(
        [
            expected_parity(
                alpha,
                delays,
                k_mhz=k_true,
                detuning_mhz=detuning_true,
                reference_detuning_mhz=reference_detuning,
                return_phase_offset_deg=phase_offset,
                cutoff=40,
            )
            for alpha, phase_offset in zip(amplitudes, return_phase_offsets, strict=True)
        ]
    )
    observed = np.array([0.03, -0.02, 0.01, -0.04, 0.02, -0.01, 0.04, -0.03])[:, None] + np.array(
        [0.95, 0.90, 0.84, 0.76, 0.92, 0.88, 0.80, 0.72]
    )[:, None] * ideal

    fit = analyze_self_kerr_ramsey(
        amplitudes,
        delays,
        observed,
        reference_detuning_mhz=reference_detuning,
        return_phase_offsets_deg=return_phase_offsets,
        cutoff=40,
        k_search_max_mhz=0.02,
    )

    assert fit["success"] is True
    assert fit["K_mhz"] == pytest.approx(k_true, abs=2.0e-4)
    assert fit["detuning_mhz"] == pytest.approx(detuning_true, abs=2.0e-4)
    assert fit["t_cat_us"] == pytest.approx(1.0 / (2.0 * abs(k_true)), rel=0.05)
    assert fit["valid_rows"] == len(amplitudes)
    assert fit["K_alias_ambiguous"] is False

    single_quadrature_fit = analyze_self_kerr_ramsey(
        amplitudes[::2],
        delays,
        observed[::2],
        reference_detuning_mhz=reference_detuning,
        return_phase_offsets_deg=np.zeros(4),
        cutoff=40,
        k_search_max_mhz=0.02,
    )
    assert single_quadrature_fit["K_alias_ambiguous"] is True


def test_self_kerr_model_returns_to_vacuum_at_zero_delay():
    """The return displacement exactly reverses the preparation before Kerr evolution begins."""
    from strumento.packs.cqed.experiments.self_kerr_ramsey import expected_parity

    parity = expected_parity(
        2.0,
        np.array([0.0, 1.0, 2.0]),
        k_mhz=0.003,
        detuning_mhz=-0.01,
        reference_detuning_mhz=0.02,
        cutoff=40,
    )

    assert parity[0] == pytest.approx(1.0, abs=1.0e-8)


def test_self_kerr_fit_fails_closed_for_flat_rows():
    from strumento.packs.cqed.experiments.self_kerr_ramsey import analyze_self_kerr_ramsey

    fit = analyze_self_kerr_ramsey(
        np.array([0.5, 1.0, 1.5, 2.0]),
        np.linspace(0.0, 80.0, 21),
        np.zeros((4, 21)),
        reference_detuning_mhz=0.02,
    )

    assert fit["success"] is False
    assert fit["valid_rows"] == 0
    assert "contrast" in fit["reason"]


@pytest.mark.qick
def test_self_kerr_ramsey_jointly_compiles_delay_and_return_phase(dev):
    """One declared time loop drives both the free wait and the return-pulse phase."""
    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram
    from strumento.core.sweeps import Loop
    from strumento.packs.cqed.experiments.self_kerr_ramsey import row_sequence

    seq, sweeps = row_sequence(
        dev,
        alpha=1.0,
        delays=Sweep(start=0.0, stop=20.0, on="t"),
        mode_name="cavity",
        reference_detuning_mhz=0.02,
    )
    soc = MockQickSocV2(dev.soccfg_snapshot)
    program = StrumentoProgram(dev, seq=seq, sweeps=sweeps, axes=(Loop("t", 7),), reps=4, soc=soc)
    result = program.acquire(soc)

    assert result.avgi.size == 7
    assert result.xpts is not None and len(result.xpts) == 7
