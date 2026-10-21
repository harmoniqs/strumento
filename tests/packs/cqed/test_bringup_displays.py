"""Displays for the bring-up nodes: Punchout, ReadoutState, TlsScan and ValidationCheck.

Each test checks what the figure *says* — axis labels, the plotted data against the result, the
fit/derived overlays — never pixels. The synthetic-result tests are pure (no qick); the
end-to-end tests run each experiment on the simulator through ``go(..., display=True)`` and are
marked ``qick``. The whole module needs matplotlib and skips cleanly without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult
from strumento.packs.cqed.experiments._signal import readout_iq, readout_signal
from strumento.packs.cqed.experiments.punchout import Punchout
from strumento.packs.cqed.experiments.readout_state import ReadoutState
from strumento.packs.cqed.experiments.tls_scan import TlsScan
from strumento.packs.cqed.experiments.validation_check import ValidationCheck
from tests._paths import FIXTURES, MULTIMODE_YAML

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

SOCCFG = FIXTURES / "soccfg_v2_cqed.json"


def _vertical_line_positions(ax) -> list[float]:
    """The x position of every ``axvline`` on ``ax`` (a two-point line with equal x)."""
    xs = []
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        if x.size == 2 and x[0] == x[1]:
            xs.append(float(x[0]))
    return xs


def _go(experiment_cls, truth_overrides=None, **kwargs):
    """Run ``experiment_cls`` on the simulator, ``display=True``; return (experiment, outcome)."""
    pytest.importorskip("qick")
    from strumento.core.device import Device
    from strumento.core.simulator import SimulatorSoc
    from strumento.packs.cqed.responses import truth_from_device

    device = Device.load(MULTIMODE_YAML)
    truth = truth_from_device(device, **(truth_overrides or {}))
    soc = SimulatorSoc.from_snapshot(SOCCFG, truth=truth, seed=7)
    experiment = experiment_cls(device, **kwargs)
    return experiment, experiment.go(soc, reps=50, progress=False, display=True)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# -- Punchout ----------------------------------------------------------------------------------


@pytest.mark.qick
def test_punchout_display_plots_the_gain_sweep_with_the_knee_and_operating_point():
    experiment, outcome = _go(Punchout)
    fig = outcome.figure
    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Probe gain (QICK code)"
    assert ax.get_ylabel() == "Projected IQ (ADC units)"

    data = ax.lines[0]
    assert np.allclose(data.get_xdata(), outcome.result.xpts)
    assert np.allclose(data.get_ydata(), readout_signal(outcome.result, experiment.readout))

    labels = [line.get_label() for line in ax.lines]
    assert "Lorentzian fit" in labels
    verticals = _vertical_line_positions(ax)
    assert outcome.fit.derived["max_power"] in verticals
    assert outcome.fit.derived["power"] in verticals


def test_punchout_display_without_a_fit_shows_only_the_data():
    result = MeasurementResult(
        xpts=np.linspace(0.0, 19000.0, 11), avgi=np.linspace(0.0, 1.0, 11)[None, :],
        avgq=np.zeros((1, 11)),
    )
    from strumento.packs.cqed.plot import _display_punchout

    fig = _display_punchout(result, None)
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    assert _vertical_line_positions(ax) == []


# -- ReadoutState ------------------------------------------------------------------------------


@pytest.mark.qick
def test_readout_state_display_histograms_both_preparations_with_the_threshold():
    experiment, outcome = _go(ReadoutState)
    fig = outcome.figure
    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Projected IQ (ADC units)"
    assert ax.get_ylabel() == "Counts"

    shots = np.asarray(outcome.result.shots)
    n_shots = shots.shape[1]
    # one histogram per preparation, each holding every shot of that preparation
    assert len(ax.containers) == 2
    for container in ax.containers:
        assert sum(patch.get_height() for patch in container) == n_shots
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert {"ground prep", "excited prep", "threshold"} <= set(legend)
    assert outcome.fit.derived["threshold"] in _vertical_line_positions(ax)
    assert "assignment_fidelity" in ax.get_title()


def test_readout_state_display_without_shots_falls_back_to_the_scalar_card():
    from strumento.packs.cqed.plot import _display_readout_state

    result = MeasurementResult(xpts=None, avgi=np.array([0.0, 1.0]), avgq=np.zeros(2), shots=None)
    fit = FitResult(params={}, stderr={}, derived={"threshold": 0.5, "assignment_fidelity": 0.99},
                    success=True)
    fig = _display_readout_state(result, fit)
    ax = fig.axes[0]
    assert not ax.axison
    assert "assignment_fidelity" in ax.texts[0].get_text()


# -- TlsScan -----------------------------------------------------------------------------------


@pytest.mark.qick
def test_tls_scan_display_plots_the_magnitude_and_marks_the_tls_line():
    _experiment, outcome = _go(TlsScan, truth_overrides={"tls_line_offsets_mhz": (4.0,)})
    fit = outcome.fit
    assert fit.derived["tls_lines"] == 1.0
    fig = outcome.figure
    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Qubit drive frequency (MHz)"
    assert ax.get_ylabel() == "|S| (ADC units)"

    i, q = readout_iq(outcome.result)
    data = ax.lines[0]
    assert np.allclose(data.get_xdata(), outcome.result.xpts)
    assert np.allclose(data.get_ydata(), np.hypot(i, q))
    assert "Lorentzian fit" in [line.get_label() for line in ax.lines]
    tls_at = fit.derived["f0"] + fit.derived["tls_offset_mhz"]
    assert any(np.isclose(x, tls_at) for x in _vertical_line_positions(ax))
    assert "1 TLS line" in ax.get_title()


def test_tls_scan_display_with_no_lines_marks_nothing():
    from strumento.packs.cqed.plot import _display_tls_scan

    freqs = np.linspace(3990.0, 4010.0, 41)
    result = MeasurementResult(xpts=freqs, avgi=np.ones((1, 41)), avgq=np.zeros((1, 41)))
    fit = FitResult(params={"y0": 1.0, "amp": -0.5, "f0": 4000.0, "hwhm": 2.0}, stderr={},
                    derived={"f0": 4000.0, "hwhm": 2.0, "tls_lines": 0.0, "tls_offset_mhz": 0.0},
                    success=True)
    fig = _display_tls_scan(result, fit)
    ax = fig.axes[0]
    assert _vertical_line_positions(ax) == []
    assert "0 TLS lines" in ax.get_title()


# -- ValidationCheck ---------------------------------------------------------------------------


@pytest.mark.qick
def test_validation_check_display_plots_the_ramsey_row_against_the_two_references():
    experiment, outcome = _go(ValidationCheck)
    fig = outcome.figure
    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Delay (us)"
    assert ax.get_ylabel() == "Projected IQ (ADC units)"

    data = ax.lines[0]
    assert np.allclose(data.get_xdata(), outcome.result.xpts)
    assert np.allclose(data.get_ydata(), readout_signal(outcome.result, experiment.readout))

    labels = [line.get_label() for line in ax.lines]
    assert "Damped cosine fit" in labels
    assert "ground prep (read 0)" in labels
    assert "excited prep (read 1)" in labels
    title = ax.get_title()
    assert "detuning" in title and "assignment_fidelity" in title


def test_validation_check_display_without_reference_rows_plots_only_the_fringe():
    from strumento.packs.cqed.plot import _display_validation_check

    delays = np.linspace(0.0, 80.0, 21)
    result = MeasurementResult(xpts=delays, avgi=np.cos(delays)[None, :], avgq=np.zeros((1, 21)))
    fig = _display_validation_check(result, None)
    ax = fig.axes[0]
    assert [line.get_label() for line in ax.lines] == ["data"]
