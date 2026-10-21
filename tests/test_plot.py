"""Plot module smoke tests — verify figures build without errors.

These tests check that:
1. Individual plot functions return matplotlib Figures
2. The display registry wires experiments to their plot functions
3. experiment.display(result, fit) returns a figure through the registry

Marked pure: no qick needed. matplotlib required.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult
from tests._paths import MULTIMODE_YAML, SPIN_YAML

mpl = pytest.importorskip("matplotlib")


def _result_1d(n=51) -> MeasurementResult:
    return MeasurementResult(
        xpts=np.linspace(0, 10, n),
        avgi=np.random.default_rng(42).normal(0.5, 0.1, (1, n)),
        avgq=np.zeros((1, n)),
    )


def _result_multiread(n=10) -> MeasurementResult:
    return MeasurementResult(
        xpts=None,
        avgi=np.random.default_rng(42).normal(0.5, 0.1, n),
        avgq=np.zeros(n),
    )


def _result_2d() -> MeasurementResult:
    return MeasurementResult(
        xpts=np.linspace(-1, 1, 21),
        avgi=np.random.default_rng(42).normal(0, 1, (11, 21)),
        avgq=np.zeros((11, 21)),
        meta={"ypts": np.linspace(-1, 1, 11).tolist(), "x_axis": "V_p", "y_axis": "V_b"},
    )


def _fit(**extra) -> FitResult:
    derived = {"T1": 5.0, "tau": 5.0, "f0": 4000.0, "pi_gain": 10000.0,
               "contrast": 0.9, "fidelity": 0.998, "period": 0.05}
    derived.update(extra)
    return FitResult(
        params={"tau": 5.0}, stderr={"tau": 0.1},
        derived=derived, success=True,
    )


class TestCQEDPlotShapes:
    """cqed plot shape functions return Figures."""

    def test_lorentzian(self):
        from strumento.packs.cqed.plot import _plot_swept
        fig = _plot_swept(_result_1d(), _fit(), xlabel="Freq", ylabel="|S|", fit_key="f0")
        assert fig is not None and hasattr(fig, "savefig")
        mpl.pyplot.close(fig)

    def test_resonator_display_uses_complex_magnitude(self):
        from strumento.packs.cqed.plot import _display_lorentzian

        result = MeasurementResult(
            xpts=np.array([749.0, 750.0]),
            avgi=np.array([3.0, 0.0]),
            avgq=np.array([4.0, 5.0]),
        )
        fig = _display_lorentzian(result, _fit())
        assert np.allclose(fig.axes[0].lines[0].get_ydata(), [5.0, 5.0])
        mpl.pyplot.close(fig)

    def test_resonator_display_overlays_the_lorentzian_fit(self):
        from strumento.packs.cqed.plot import _display_lorentzian

        result = MeasurementResult(
            xpts=np.array([749.0, 750.0, 751.0]),
            avgi=np.array([1.0, 0.5, 1.0]),
            avgq=np.zeros(3),
        )
        fit = FitResult(
            params={"y0": 1.0, "amp": -0.5, "f0": 750.0, "hwhm": 0.25},
            stderr={},
            derived={"f0": 750.0},
            success=True,
        )

        fig = _display_lorentzian(result, fit)
        assert len(fig.axes[0].lines) == 2
        curve = fig.axes[0].lines[1]
        assert curve.get_label() == "Lorentzian fit"
        assert np.min(curve.get_ydata()) == pytest.approx(0.5, abs=1e-4)
        mpl.pyplot.close(fig)

    def test_t1_display_overlays_the_exponential_fit(self):
        from strumento.packs.cqed.plot import _display_exp_decay

        result = MeasurementResult(
            xpts=np.array([0.0, 5.0, 10.0]),
            avgi=np.array([3.0, 1.73575888, 1.27067057]),
            avgq=np.zeros(3),
        )
        fit = FitResult(
            params={"y0": 1.0, "amp": 2.0, "tau": 5.0},
            stderr={},
            derived={"T1": 5.0, "tau": 5.0},
            success=True,
        )

        fig = _display_exp_decay(result, fit)
        assert len(fig.axes[0].lines) == 2
        curve = fig.axes[0].lines[1]
        assert curve.get_label() == "Exponential fit"
        assert curve.get_ydata()[0] == pytest.approx(3.0)
        assert curve.get_ydata()[-1] == pytest.approx(1.0 + 2.0 * np.exp(-2.0))
        mpl.pyplot.close(fig)

    def test_ramsey_display_overlays_the_damped_cosine_fit(self):
        from strumento.packs.cqed.plot import _display_damped_osc

        result = MeasurementResult(
            xpts=np.array([0.0, 1.0, 2.0]),
            avgi=np.array([3.0, 1.0, -0.21306132]),
            avgq=np.zeros(3),
        )
        fit = FitResult(
            params={"offset": 1.0, "amp": 2.0, "freq": 0.25, "phase": 0.0, "tau": 4.0},
            stderr={},
            derived={"T2": 4.0, "tau": 4.0, "detuning": 0.25, "freq": 0.25},
            success=True,
        )

        fig = _display_damped_osc(result, fit)
        assert len(fig.axes[0].lines) == 2
        curve = fig.axes[0].lines[1]
        assert curve.get_label() == "Damped cosine fit"
        assert curve.get_ydata()[0] == pytest.approx(3.0)
        assert curve.get_ydata()[-1] == pytest.approx(1.0 - 2.0 * np.exp(-0.5))
        mpl.pyplot.close(fig)

    def test_rabi_display_labels_projected_iq_units(self):
        """Averaged Rabi data is projected ADC IQ, not a normalized population."""
        from strumento.packs.cqed.plot import _display_rabi

        fig = _display_rabi(_result_1d(), _fit())
        assert fig.axes[0].get_ylabel() == "Projected IQ (ADC units)"
        mpl.pyplot.close(fig)

    def test_t2_echo_display_labels_the_half_delay_axis(self):
        from strumento.packs.cqed.plot import _display_t2_echo

        fig = _display_t2_echo(_result_1d(), _fit(T2_echo=10.0))
        assert fig.axes[0].get_xlabel() == "Half-delay (us)"
        mpl.pyplot.close(fig)

    def test_projected_display_plots_the_fit_signal_not_raw_i(self):
        """Regression (armed T1/T2 goofy plots): a projecting display must plot the SAME
        discrimination-axis signal the fitter used (``readout.project``), not raw ``avgi``.

        With a nonzero readout angle the two live on different axes and can anti-correlate — which
        left armed T1 plots showing data rising to 0 while the fitted exponential fell to the
        floor (data and curve going opposite directions). Passing the readout makes the plotted
        points land on the fitted curve.
        """
        from strumento.packs.cqed.plot import _display_exp_decay
        from strumento.packs.cqed.readout import DispersiveIQ

        angle = -2.2153  # a real ge-blobs discrimination angle (rad), far from 0
        delays = np.linspace(0.0, 300.0, 31)
        projected = 1.0 + 2.0 * np.exp(-delays / 80.0)      # y0=1, amp=2, tau=80 — a clean decay
        # Encode the decay onto the rotated axis: project() = I·cos(a)+Q·sin(a) recovers
        # `projected`, while raw avgi = projected·cos(a) is a different (here sign-flipped) axis.
        result = MeasurementResult(
            xpts=delays,
            avgi=(projected * np.cos(angle))[None, :],
            avgq=(projected * np.sin(angle))[None, :],
        )
        fit = FitResult(params={"y0": 1.0, "amp": 2.0, "tau": 80.0}, stderr={},
                        derived={"T1": 80.0, "tau": 80.0}, success=True)

        fig = _display_exp_decay(result, fit, readout=DispersiveIQ(threshold=0.0, angle=angle))
        data = fig.axes[0].lines[0].get_ydata()
        assert np.allclose(data, projected, atol=1e-9)          # projected signal — matches the fit
        assert not np.allclose(data, np.asarray(result.avgi).ravel())  # NOT raw avgi (the bug)
        # and it sits on the overlaid exponential curve (same axis, same direction)
        curve = fig.axes[0].lines[1]
        assert curve.get_ydata()[0] == pytest.approx(3.0, abs=1e-6)   # y0 + amp
        assert data[0] == pytest.approx(3.0, abs=1e-6)
        assert data[-1] == pytest.approx(1.0 + 2.0 * np.exp(-300.0 / 80.0), abs=1e-6)
        mpl.pyplot.close(fig)

    def test_projecting_display_without_a_readout_falls_back_to_raw_i(self):
        """Back-compat: called with no readout (the two-arg path), the plot still shows raw avgi
        so existing callers and non-projecting readouts are unaffected."""
        from strumento.packs.cqed.plot import _display_exp_decay

        result = MeasurementResult(
            xpts=np.array([0.0, 5.0, 10.0]),
            avgi=np.array([3.0, 1.73575888, 1.27067057]),
            avgq=np.zeros(3),
        )
        fit = FitResult(params={"y0": 1.0, "amp": 2.0, "tau": 5.0}, stderr={},
                        derived={"T1": 5.0, "tau": 5.0}, success=True)
        fig = _display_exp_decay(result, fit)                    # no readout -> raw avgi
        assert np.allclose(fig.axes[0].lines[0].get_ydata(), result.avgi)
        mpl.pyplot.close(fig)

    def test_staircase(self):
        from strumento.packs.cqed.plot import _display_allxy
        result = _result_multiread(21)
        fig = _display_allxy(result, _fit())
        assert fig is not None
        mpl.pyplot.close(fig)

    def test_bar(self):
        from strumento.packs.cqed.plot import _plot_bar
        fig = _plot_bar(_result_multiread(5), _fit())
        assert fig is not None
        mpl.pyplot.close(fig)

    def test_scalar(self):
        from strumento.packs.cqed.plot import _plot_scalar
        fig = _plot_scalar(_result_multiread(1), _fit(), title="Test")
        assert fig is not None
        mpl.pyplot.close(fig)

    def test_active_reset_multi_round_is_a_bar_chart(self):
        """rounds>=2: one corrected-fraction bar per round + a verified-residual bar."""
        from strumento.packs.cqed.plot import _display_active_reset

        fit = FitResult(
            params={"shots": 2000.0, "rounds": 3.0},
            stderr={"ground_population": 0.002, "excited_population": 0.002},
            derived={"ground_population": 0.988, "excited_population": 0.012,
                     "corrected_round_1": 0.15, "corrected_round_2": 0.03, "corrected_round_3": 0.013},
            success=True,
        )
        fig = _display_active_reset(_result_multiread(4), fit)
        ax = fig.axes[0]
        bars = [c for c in ax.containers if type(c).__name__ == "BarContainer"]
        assert sum(len(c) for c in bars) == 4                       # 3 rounds + verify
        assert [t.get_text() for t in ax.get_xticklabels()] == ["round 1", "round 2", "round 3", "verify"]
        assert ax.get_ylabel() == r"$P(|e\rangle)$"
        assert bars[-1][0].get_height() == pytest.approx(0.012)     # verify bar = excited_population
        assert "ground = 0.988" in ax.get_title()
        mpl.pyplot.close(fig)

    def test_active_reset_single_round_falls_back_to_scalar(self):
        """rounds=1 has no trend — keep the scalar text card (no bars, axis off)."""
        from strumento.packs.cqed.plot import _display_active_reset

        fit = FitResult(
            params={"shots": 2000.0, "rounds": 1.0},
            stderr={"ground_population": 0.003, "excited_population": 0.003},
            derived={"ground_population": 0.985, "excited_population": 0.015, "corrected_round_1": 0.15},
            success=True,
        )
        fig = _display_active_reset(_result_multiread(2), fit)
        ax = fig.axes[0]
        assert not [c for c in ax.containers if type(c).__name__ == "BarContainer"]
        assert not ax.axison                                        # scalar card turns the axis off
        mpl.pyplot.close(fig)

    def test_active_reset_without_shot_data_falls_back_to_scalar(self):
        """An unsuccessful fit (averaged/mock acquire, no shots) still renders the scalar card."""
        from strumento.packs.cqed.plot import _display_active_reset

        fig = _display_active_reset(_result_multiread(1),
                                    FitResult(params={}, stderr={}, derived={}, success=False))
        assert fig is not None
        assert not [c for c in fig.axes[0].containers if type(c).__name__ == "BarContainer"]
        mpl.pyplot.close(fig)

    def test_2d_map(self):
        from strumento.packs.cqed.plot import plot_2d_map
        fig = plot_2d_map(_result_2d(), title="Wigner")
        assert fig is not None
        mpl.pyplot.close(fig)


class TestSpinPlotShapes:
    """spin plot shape functions return Figures."""

    def test_swept(self):
        from strumento.packs.spin.plot import _plot_swept
        fig = _plot_swept(_result_1d(), _fit(), xlabel="Delay", ylabel="P_up", fit_key="tau")
        assert fig is not None
        mpl.pyplot.close(fig)

    def test_2d(self):
        from strumento.packs.spin.plot import _plot_swept
        fig = _plot_swept(_result_1d(), _fit(), xlabel="Voltage", ylabel="Signal", fit_key="tau")
        assert fig is not None
        mpl.pyplot.close(fig)


class TestDisplayRegistry:
    """The display registry wires experiments to plot functions."""

    def test_cqed_experiment_has_display(self):
        """A cqed experiment's display() returns a figure through the registry."""
        from strumento.core.device import Device
        from strumento.packs.cqed.experiments.t1 import T1

        dev = Device.load(MULTIMODE_YAML)
        exp = T1(dev)
        fig = exp.display(_result_1d(), _fit())
        assert fig is not None and hasattr(fig, "savefig")
        mpl.pyplot.close(fig)

    def test_display_registry_forwards_readout_so_the_plot_projects(self):
        """display_for hands experiment.readout to a projecting display fn, so the plotted data is
        the discrimination-axis projection the fit used — not raw avgi (the armed-run regression).
        A nonzero readout angle is what exposed the mismatch on hardware."""
        from strumento.core.device import Device
        from strumento.packs.cqed.experiments.t1 import T1
        from strumento.packs.cqed.readout import DispersiveIQ

        dev = Device.load(MULTIMODE_YAML)
        angle = 1.0
        delays = np.linspace(0.0, 100.0, 11)
        projected = 1.0 + 2.0 * np.exp(-delays / 40.0)
        result = MeasurementResult(
            xpts=delays,
            avgi=(projected * np.cos(angle))[None, :],
            avgq=(projected * np.sin(angle))[None, :],
        )
        fit = FitResult(params={"y0": 1.0, "amp": 2.0, "tau": 40.0}, stderr={},
                        derived={"T1": 40.0, "tau": 40.0}, success=True)
        exp = T1(dev, readout=DispersiveIQ(threshold=0.0, angle=angle))
        fig = exp.display(result, fit)
        data = fig.axes[0].lines[0].get_ydata()
        assert np.allclose(data, projected, atol=1e-9)                  # projected, matches the fit
        assert not np.allclose(data, np.asarray(result.avgi).ravel())   # not raw avgi
        mpl.pyplot.close(fig)

    def test_active_reset_display_routes_to_the_bar_chart(self):
        """ActiveReset.display() reaches the multi-round bar chart through the registry."""
        from strumento.core.device import Device
        from strumento.packs.cqed.experiments.active_reset import ActiveReset

        dev = Device.load(MULTIMODE_YAML)
        exp = ActiveReset(dev, rounds=3)
        fit = FitResult(
            params={"shots": 2000.0, "rounds": 3.0},
            stderr={"ground_population": 0.002, "excited_population": 0.002},
            derived={"ground_population": 0.988, "excited_population": 0.012,
                     "corrected_round_1": 0.15, "corrected_round_2": 0.03, "corrected_round_3": 0.013},
            success=True,
        )
        fig = exp.display(_result_multiread(4), fit)
        assert fig is not None and hasattr(fig, "savefig")
        assert [c for c in fig.axes[0].containers if type(c).__name__ == "BarContainer"]
        mpl.pyplot.close(fig)

    def test_spin_experiment_has_display(self):
        """A spin experiment's display() returns a figure through the registry."""
        from strumento.core.device import Device
        from strumento.packs.spin.experiments.spin_t1 import SpinT1

        dev = Device.load(SPIN_YAML)
        exp = SpinT1(dev, dot_name="q0")
        fig = exp.display(_result_1d(), _fit())
        assert fig is not None and hasattr(fig, "savefig")
        mpl.pyplot.close(fig)

    def test_unregistered_returns_none(self):
        """An experiment with no registered display returns None."""
        from strumento.core.experiment import Experiment

        class Unplotted(Experiment):
            def sequence(self):
                return None, ()
            def analyze(self, result):
                return None

        # Unplotted has no registered display — should return None gracefully
        exp = Unplotted.__new__(Unplotted)
        assert exp.display(_result_1d(), _fit()) is None
