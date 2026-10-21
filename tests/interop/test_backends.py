"""Tests for the backend-agnostic data import layer (strumento.interop.backends).

Verifies that data from any external source can be packaged into a MeasurementResult
and survive the full Strumento analysis pipeline — fitters, wire round-trip, datachest.
Also verifies that Piccolo OC pulses can be extracted as plain arrays for any backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.result import MeasurementResult
from strumento.interop.backends import (
    ExternalExperiment,
    WaveformExport,
    analyze_rabi,
    analyze_resonator,
    analyze_spectroscopy,
    analyze_t1,
    analyze_t2_ramsey,
    external_outcome,
    extract_oc_pulses,
    from_complex,
    from_counts,
    from_iq,
    from_multiread,
    from_xarray,
    from_xarray_complex,
)


class TestFromIQ:
    """The universal entry point — separate I and Q arrays."""

    def test_basic_sweep(self):
        freqs = np.linspace(6.9, 7.1, 51)
        i = np.random.randn(51)
        q = np.random.randn(51)
        result = from_iq(freqs, i, q, backend="opx")

        assert isinstance(result, MeasurementResult)
        assert result.xpts.shape == (51,)
        assert result.avgi.shape == (51,)
        assert result.avgq.shape == (51,)
        assert result.meta["backend"] == "opx"

    def test_single_point(self):
        result = from_iq(None, 0.5, -0.3, backend="qblox")
        assert result.xpts is None
        assert float(result.avgi) == pytest.approx(0.5)
        assert float(result.avgq) == pytest.approx(-0.3)

    def test_with_shots(self):
        xpts = np.arange(10)
        i = np.random.randn(10)
        q = np.random.randn(10)
        shots = np.random.randn(10, 100, 2)
        result = from_iq(xpts, i, q, shots=shots)
        assert result.shots.shape == (10, 100, 2)

    def test_custom_meta_merges_with_backend(self):
        result = from_iq(None, 0.0, 0.0, backend="zurich", meta={"program": "t1"})
        assert result.meta["backend"] == "zurich"
        assert result.meta["program"] == "t1"

    def test_wire_round_trip(self):
        """A result from external data survives the wire form (JSON) round-trip."""
        xpts = np.linspace(0, 10, 21)
        result = from_iq(xpts, np.sin(xpts), np.cos(xpts), backend="opx")
        wire = result.to_wire()
        restored = MeasurementResult.from_wire(wire)
        np.testing.assert_allclose(restored.avgi, result.avgi)
        np.testing.assert_allclose(restored.avgq, result.avgq)


class TestFromComplex:
    """Complex S21 / demodulated IQ as a single complex array."""

    def test_splits_real_imag(self):
        freqs = np.linspace(6.9, 7.1, 51)
        s21 = np.exp(1j * freqs * 10) * 0.5  # a phase-wrapping transmission
        result = from_complex(freqs, s21, backend="vna")

        np.testing.assert_allclose(result.avgi, s21.real)
        np.testing.assert_allclose(result.avgq, s21.imag)

    def test_complex_shots(self):
        xpts = np.arange(5)
        signal = np.random.randn(5) + 1j * np.random.randn(5)
        shots = np.random.randn(5, 50) + 1j * np.random.randn(5, 50)
        result = from_complex(xpts, signal, shots=shots)
        assert result.shots.shape == (5, 50, 2)
        np.testing.assert_allclose(result.shots[..., 0], shots.real)
        np.testing.assert_allclose(result.shots[..., 1], shots.imag)


class TestFromCounts:
    """Discriminated population data (excited fraction per point)."""

    def test_population_on_i_channel(self):
        delays = np.linspace(0, 500, 101)
        pop = np.exp(-delays / 100.0)  # T1 decay
        result = from_counts(delays, pop, total_shots=1000, backend="opx")

        np.testing.assert_allclose(result.avgi, pop)
        np.testing.assert_allclose(result.avgq, 0.0)
        assert result.meta["total_shots"] == 1000
        assert result.meta["signal_kind"] == "excited_fraction"

    def test_fitter_compatible(self):
        """An exponential fit works on from_counts data (the I channel IS the signal)."""
        from strumento.core.fitters import fit_exp

        delays = np.linspace(0.1, 500, 101)
        pop = 0.9 * np.exp(-delays / 80.0) + 0.05
        result = from_counts(delays, pop)
        fit = fit_exp(result.xpts, result.avgi)
        assert fit.success
        assert abs(fit.derived["tau"] - 80.0) < 10.0  # within 12%


class TestFromMultiread:
    """Multiple readouts per sweep point (validation check, QND, active reset)."""

    def test_stacks_reads(self):
        xpts = np.linspace(0, 10, 21)
        read_0 = (np.random.randn(21), np.random.randn(21))
        read_1 = (np.random.randn(21), np.random.randn(21))
        result = from_multiread(xpts, [read_0, read_1])

        assert result.avgi.shape == (2, 21)
        assert result.avgq.shape == (2, 21)
        assert result.meta["n_reads"] == 2
        np.testing.assert_allclose(result.avgi[0], read_0[0])
        np.testing.assert_allclose(result.avgi[1], read_1[0])


class TestEndToEndWithFitters:
    """External data through real Strumento fitters — the whole point."""

    def test_lorentzian_from_external_vna(self):
        """A resonator dip from a VNA, imported and fit with Strumento."""
        from strumento.core.resonator_fitters import fit_lorentzian

        freqs = np.linspace(6.95, 7.05, 201)
        f0, hwhm = 7.0, 0.005
        mag = 1.0 - 0.8 / (1 + ((freqs - f0) / hwhm) ** 2)
        mag += np.random.randn(201) * 0.01

        result = from_iq(freqs, mag, np.zeros_like(mag), backend="vna")
        fit = fit_lorentzian(result.xpts, result.avgi)

        assert fit.success
        assert abs(fit.derived["f0"] - 7.0) < 0.002

    def test_rabi_from_external_opx(self):
        """A Rabi oscillation from an OPX, imported and fit with Strumento."""
        from strumento.core.fitters import fit_rabi

        gains = np.linspace(0, 30000, 51)
        period = 16000
        signal = np.cos(2 * np.pi * gains / period)

        result = from_iq(gains, signal, np.zeros_like(signal), backend="opx")
        fit = fit_rabi(result.xpts, result.avgi)

        assert fit.success
        assert abs(fit.derived["period"] - period) / period < 0.05


class _FakeDataArray:
    """Mimics xarray.DataArray — has .values that returns a numpy array."""

    def __init__(self, values):
        self.values = np.asarray(values)


class _FakeDataset:
    """Mimics xarray.Dataset — has .coords, .data_vars, and subscript access.

    This lets us test from_xarray without requiring xarray as a dependency.
    The duck-typing in _xarray_keys / _xarray_values handles both real xarray
    and this fake identically.
    """

    def __init__(self, coords: dict, data_vars: dict):
        self.coords = coords
        self.data_vars = data_vars
        self._all = {**coords, **data_vars}

    def __getitem__(self, key):
        return _FakeDataArray(self._all[key])


class TestFromXarray:
    """xarray.Dataset import — the Quantify / Qblox natural format."""

    def test_basic_iq_dataset(self):
        freqs = np.linspace(6.9, 7.1, 51)
        i_data = np.random.randn(51)
        q_data = np.random.randn(51)

        ds = _FakeDataset(
            coords={"x0": freqs},
            data_vars={"y0": i_data, "y1": q_data},
        )
        result = from_xarray(ds, backend="qblox")

        assert result.avgi.shape == (51,)
        np.testing.assert_allclose(result.avgi, i_data)
        np.testing.assert_allclose(result.avgq, q_data)
        np.testing.assert_allclose(result.xpts, freqs)
        assert result.meta["backend"] == "qblox"
        assert result.meta["xarray_keys"] == {"x": "x0", "i": "y0", "q": "y1"}

    def test_custom_key_names(self):
        ds = _FakeDataset(
            coords={"frequency": np.linspace(100, 200, 11)},
            data_vars={"real": np.ones(11), "imag": np.zeros(11)},
        )
        result = from_xarray(ds, x="frequency", i="real", q="imag", backend="zurich")

        assert result.xpts.shape == (11,)
        np.testing.assert_allclose(result.avgi, 1.0)
        assert result.meta["backend"] == "zurich"

    def test_x_scale_converts_units(self):
        """x_scale=1e-6 converts Hz to MHz."""
        freqs_hz = np.linspace(6.95e9, 7.05e9, 21)
        ds = _FakeDataset(
            coords={"x0": freqs_hz},
            data_vars={"y0": np.zeros(21), "y1": np.zeros(21)},
        )
        result = from_xarray(ds, x_scale=1e-3, backend="qblox")

        # 6.95e9 * 1e-3 = 6.95e6 — that's not MHz though; the point is the
        # scale is applied. A real Quantify dataset in Hz with x_scale=1e-6
        # would give MHz. Here we just verify the multiplication.
        np.testing.assert_allclose(result.xpts, freqs_hz * 1e-3)
        assert result.meta["x_scale"] == 1e-3

    def test_missing_key_raises_with_available(self):
        ds = _FakeDataset(
            coords={"freq": np.arange(5)},
            data_vars={"signal_i": np.zeros(5), "signal_q": np.zeros(5)},
        )
        with pytest.raises(KeyError, match="x0.*not found.*available.*freq"):
            from_xarray(ds, backend="qblox")  # default x="x0" is missing

    def test_complex_dataset(self):
        freqs = np.linspace(6.9, 7.1, 31)
        s21 = np.exp(1j * freqs * 5) * 0.8

        ds = _FakeDataset(
            coords={"x0": freqs},
            data_vars={"y0": s21},
        )
        result = from_xarray_complex(ds, signal="y0", backend="qblox")

        np.testing.assert_allclose(result.avgi, s21.real)
        np.testing.assert_allclose(result.avgq, s21.imag)

    def test_plain_dict_works(self):
        """A plain dict with numpy arrays works as a dataset (the duck-typed fallback)."""
        data = {
            "x0": np.linspace(0, 10, 21),
            "y0": np.random.randn(21),
            "y1": np.random.randn(21),
        }
        result = from_xarray(data, backend="custom")

        assert result.avgi.shape == (21,)
        np.testing.assert_allclose(result.xpts, data["x0"])

    def test_fitter_on_xarray_data(self):
        """End-to-end: xarray dataset through a Strumento fitter."""
        from strumento.core.fitters import fit_exp

        delays = np.linspace(0.1, 500, 101)
        signal = 0.9 * np.exp(-delays / 75.0) + 0.05

        ds = _FakeDataset(
            coords={"x0": delays},
            data_vars={"y0": signal, "y1": np.zeros(101)},
        )
        result = from_xarray(ds, backend="qblox")
        fit = fit_exp(result.xpts, result.avgi)

        assert fit.success
        assert abs(fit.derived["tau"] - 75.0) < 10.0

    def test_wire_round_trip(self):
        ds = _FakeDataset(
            coords={"x0": np.linspace(0, 10, 11)},
            data_vars={"y0": np.ones(11), "y1": np.zeros(11)},
        )
        result = from_xarray(ds, backend="qblox")
        wire = result.to_wire()
        restored = MeasurementResult.from_wire(wire)
        np.testing.assert_allclose(restored.avgi, result.avgi)


# ---------------------------------------------------------------------------
# Helpers for OC pulse extraction tests
# ---------------------------------------------------------------------------

def _mock_gain_calib(amp_mhz: float) -> float:
    """Linear gain calibration: 100 MHz Rabi = gain 0.5."""
    return amp_mhz / 200.0


def _simple_solution(*, n_points: int = 50, complex_drive: bool = False):
    """A fake Piccolo solve output with one or two drives."""
    times = np.linspace(0, 0.1, n_points)  # 100 ns pulse
    i_samples = np.sin(2 * np.pi * times / 0.1) * 80.0  # peak 80 MHz Rabi
    controls = {"qubit:x": i_samples}
    if complex_drive:
        controls["qubit:x"] = i_samples + 1j * i_samples * 0.5
    return {"times": times, "controls": controls}


def _two_drive_solution():
    """A fake solve with two simultaneous drives (transmon + cavity)."""
    times = np.linspace(0, 0.2, 100)
    controls = {
        "qubit:x": np.sin(2 * np.pi * times / 0.2) * 60.0,
        "cavity:x": np.cos(2 * np.pi * times / 0.2) * 30.0,
    }
    return {"times": times, "controls": controls}


class TestExtractOCPulses:
    """Piccolo OC pulse extraction for external backends."""

    def test_single_real_drive(self):
        solution = _simple_solution()
        drive_map = {"qubit:x": (("qubit", "drive"), 4000.0)}
        pulses = extract_oc_pulses(solution, drive_map=drive_map,
                                    gain_calib=_mock_gain_calib)

        assert "qubit:x" in pulses
        wf = pulses["qubit:x"]
        assert isinstance(wf, WaveformExport)
        assert wf.n_samples == 50
        assert wf.freq_mhz == 4000.0
        assert wf.line == ("qubit", "drive")
        assert wf.drive_name == "qubit:x"
        assert wf.qdata is None
        # Peak of ~80 MHz Rabi → gain ≈ 80/200 = 0.4 (discrete sampling shifts the peak slightly)
        assert wf.gain == pytest.approx(0.4, rel=0.01)
        # Normalized samples peak at 1.0
        assert np.max(np.abs(wf.idata)) == pytest.approx(1.0, abs=1e-10)
        assert wf.duration_us == pytest.approx(0.1)

    def test_complex_drive(self):
        solution = _simple_solution(complex_drive=True)
        drive_map = {"qubit:x": (("qubit", "drive"), 4000.0)}
        pulses = extract_oc_pulses(solution, drive_map=drive_map,
                                    gain_calib=_mock_gain_calib)

        wf = pulses["qubit:x"]
        assert wf.qdata is not None
        assert wf.idata.shape == wf.qdata.shape

    def test_two_drives(self):
        solution = _two_drive_solution()
        drive_map = {
            "qubit:x": (("qubit", "drive"), 4000.0),
            "cavity:x": (("cavity", "drive"), 5200.0),
        }
        pulses = extract_oc_pulses(solution, drive_map=drive_map,
                                    gain_calib=_mock_gain_calib)

        assert len(pulses) == 2
        assert pulses["qubit:x"].freq_mhz == 4000.0
        assert pulses["cavity:x"].freq_mhz == 5200.0
        assert pulses["qubit:x"].line == ("qubit", "drive")
        assert pulses["cavity:x"].line == ("cavity", "drive")
        # Different peak amplitudes → different gains (rel tolerance for discrete sampling)
        assert pulses["qubit:x"].gain == pytest.approx(60.0 / 200.0, rel=0.01)
        assert pulses["cavity:x"].gain == pytest.approx(30.0 / 200.0, rel=0.01)

    def test_dac_rate_resampling(self):
        solution = _simple_solution(n_points=50)
        drive_map = {"qubit:x": (("qubit", "drive"), 4000.0)}
        pulses = extract_oc_pulses(solution, drive_map=drive_map,
                                    gain_calib=_mock_gain_calib, dac_rate=1000.0)

        wf = pulses["qubit:x"]
        # 0.1 us * 1000 samples/us = 100 samples + 1 = 101
        assert wf.n_samples == 101
        assert wf.dac_rate == 1000.0

    def test_absolute_samples(self):
        solution = _simple_solution()
        drive_map = {"qubit:x": (("qubit", "drive"), 4000.0)}
        pulses = extract_oc_pulses(solution, drive_map=drive_map,
                                    gain_calib=_mock_gain_calib)

        wf = pulses["qubit:x"]
        abs_samples = wf.absolute_samples()
        # absolute = normalized * gain, so peak ≈ 1.0 * 0.4 = 0.4
        assert np.max(np.abs(abs_samples)) == pytest.approx(wf.gain, abs=1e-10)

    def test_absolute_samples_iq(self):
        solution = _simple_solution(complex_drive=True)
        drive_map = {"qubit:x": (("qubit", "drive"), 4000.0)}
        pulses = extract_oc_pulses(solution, drive_map=drive_map,
                                    gain_calib=_mock_gain_calib)

        wf = pulses["qubit:x"]
        abs_i, abs_q = wf.absolute_samples_iq()
        assert abs_i is not None
        assert abs_q is not None
        np.testing.assert_allclose(abs_i, wf.idata * wf.gain)
        np.testing.assert_allclose(abs_q, wf.qdata * wf.gain)

    def test_real_drive_iq_returns_none_q(self):
        solution = _simple_solution(complex_drive=False)
        drive_map = {"qubit:x": (("qubit", "drive"), 4000.0)}
        pulses = extract_oc_pulses(solution, drive_map=drive_map,
                                    gain_calib=_mock_gain_calib)

        _, abs_q = pulses["qubit:x"].absolute_samples_iq()
        assert abs_q is None

    def test_missing_drive_in_map_raises(self):
        solution = _simple_solution()
        drive_map = {"wrong_name": (("qubit", "drive"), 4000.0)}
        with pytest.raises(KeyError, match="qubit:x.*no entry in drive_map"):
            extract_oc_pulses(solution, drive_map=drive_map,
                              gain_calib=_mock_gain_calib)

    def test_lineref_tuple_and_named(self):
        """Both (line, role) tuples and LineRef objects work in drive_map."""
        from strumento.core.wiring import LineRef

        solution = _simple_solution()

        # Tuple form
        pulses_t = extract_oc_pulses(
            solution,
            drive_map={"qubit:x": (("qubit", "drive"), 4000.0)},
            gain_calib=_mock_gain_calib,
        )
        # LineRef form
        pulses_r = extract_oc_pulses(
            solution,
            drive_map={"qubit:x": (LineRef("qubit", "drive"), 4000.0)},
            gain_calib=_mock_gain_calib,
        )
        assert pulses_t["qubit:x"].line == pulses_r["qubit:x"].line
        np.testing.assert_allclose(pulses_t["qubit:x"].idata,
                                   pulses_r["qubit:x"].idata)

    def test_round_trip_matches_from_solution(self):
        """extract_oc_pulses produces the same waveform data as from_solution's IR."""
        from strumento.core.pulses import from_solution
        from strumento.core.wiring import LineRef

        solution = _simple_solution()
        drive_map = {"qubit:x": (LineRef("qubit", "drive"), 4000.0)}

        seq = from_solution(solution, drive_map=drive_map, gain_calib=_mock_gain_calib)
        pulses = extract_oc_pulses(solution, drive_map=drive_map,
                                    gain_calib=_mock_gain_calib)

        # The Seq's pulse envelope should match the export
        ir_pulse = seq.ops[0].pulses[0]  # Segment → first Pulse
        wf = pulses["qubit:x"]

        np.testing.assert_allclose(np.array(ir_pulse.envelope.idata), wf.idata)
        assert ir_pulse.gain == pytest.approx(wf.gain)
        assert ir_pulse.freq_mhz == wf.freq_mhz


# ---------------------------------------------------------------------------
# Quick-analyze helpers
# ---------------------------------------------------------------------------


class TestQuickAnalyze:
    """One-call import + fit helpers."""

    def test_analyze_resonator_lorentzian(self):
        freqs = np.linspace(6.95, 7.05, 201)
        f0, hwhm = 7.0, 0.005
        mag = 1.0 - 0.8 / (1 + ((freqs - f0) / hwhm) ** 2)
        # from_iq expects I and Q; for a magnitude signal, put it on I
        fit = analyze_resonator(freqs, mag, np.zeros_like(mag), backend="vna")
        assert fit.success
        assert abs(fit.derived["f0"] - 7.0) < 0.002

    def test_analyze_rabi(self):
        gains = np.linspace(0, 30000, 51)
        period = 16000
        signal = np.cos(2 * np.pi * gains / period)
        fit = analyze_rabi(gains, signal, np.zeros_like(signal), backend="opx")
        assert fit.success
        assert abs(fit.derived["period"] - period) / period < 0.05

    def test_analyze_t1(self):
        delays = np.linspace(0.1, 500, 101)
        signal = 0.9 * np.exp(-delays / 80.0) + 0.05
        fit = analyze_t1(delays, signal, np.zeros(101), backend="qblox")
        assert fit.success
        assert abs(fit.derived["tau"] - 80.0) < 10.0

    def test_analyze_t2_ramsey(self):
        delays = np.linspace(0.1, 50, 201)
        detuning = 0.5  # MHz
        signal = 0.4 * np.exp(-delays / 20.0) * np.cos(2 * np.pi * detuning * delays) + 0.5
        fit = analyze_t2_ramsey(delays, signal, np.zeros(201), backend="opx")
        assert fit.success
        assert abs(fit.derived["tau"] - 20.0) < 5.0

    def test_analyze_spectroscopy(self):
        freqs = np.linspace(3.95, 4.05, 201)
        f0 = 4.0
        mag = 1.0 - 0.6 / (1 + ((freqs - f0) / 0.003) ** 2)
        fit = analyze_spectroscopy(freqs, mag, np.zeros_like(mag), backend="opx")
        assert fit.success
        assert abs(fit.derived["f0"] - 4.0) < 0.002


# ---------------------------------------------------------------------------
# ExperimentOutcome builder
# ---------------------------------------------------------------------------


class TestExternalOutcome:
    """Build verdict-ready ExperimentOutcome from external data."""

    def test_basic_outcome_with_rabi(self):
        from strumento.core.experiment import ExperimentOutcome
        from strumento.core.fitters import fit_rabi

        gains = np.linspace(0, 30000, 51)
        signal = np.cos(2 * np.pi * gains / 16000)

        outcome = external_outcome(gains, signal, np.zeros(51),
                                    fit_fn=fit_rabi, backend="opx")
        assert isinstance(outcome, ExperimentOutcome)
        assert outcome.result is not None
        assert outcome.fit is not None
        assert outcome.fit.success
        assert "period" in outcome.fit.derived

    def test_outcome_with_signal_fn(self):
        from strumento.core.resonator_fitters import fit_lorentzian

        freqs = np.linspace(6.95, 7.05, 201)
        f0, hwhm = 7.0, 0.005
        i = (1.0 - 0.8 / (1 + ((freqs - f0) / hwhm) ** 2)) * np.cos(0.3)
        q = (1.0 - 0.8 / (1 + ((freqs - f0) / hwhm) ** 2)) * np.sin(0.3)

        outcome = external_outcome(
            freqs, i, q,
            fit_fn=fit_lorentzian,
            signal_fn=lambda r: np.hypot(r.avgi, r.avgq),
            backend="vna",
        )
        assert outcome.fit.success
        assert abs(outcome.fit.derived["f0"] - 7.0) < 0.002

    def test_outcome_fit_failure_returns_none(self):
        """A fitter that raises returns fit=None, not a crash."""
        def bad_fitter(xpts, signal):
            raise ValueError("cannot fit garbage")

        outcome = external_outcome(
            np.arange(10), np.random.randn(10), np.zeros(10),
            fit_fn=bad_fitter,
        )
        assert outcome.result is not None
        assert outcome.fit is None

    def test_outcome_has_backend_tag(self):
        from strumento.core.fitters import fit_exp

        outcome = external_outcome(
            np.linspace(0, 100, 51),
            np.exp(-np.linspace(0, 100, 51) / 30.0),
            np.zeros(51),
            fit_fn=fit_exp,
            backend="zurich",
        )
        assert outcome.result.meta["backend"] == "zurich"


# ---------------------------------------------------------------------------
# External experiment adapter
# ---------------------------------------------------------------------------


class TestExternalExperiment:
    """Wrapping external scripts as Strumento Experiments."""

    def _make_device(self):
        """Load the loopback_demo for a minimal device context."""
        from pathlib import Path

        from strumento import Device

        repo = Path(__file__).resolve().parent.parent.parent
        return Device.load(str(repo / "devices" / "loopback_demo" / "device.yaml"))

    def test_go_returns_experiment_outcome(self):
        from strumento.core.experiment import ExperimentOutcome
        from strumento.core.fitters import fit_rabi

        dev = self._make_device()

        def fake_acquire(device):
            gains = np.linspace(0, 30000, 51)
            signal = np.cos(2 * np.pi * gains / 16000)
            return gains, signal, np.zeros(51)

        exp = ExternalExperiment(
            dev, acquire_fn=fake_acquire, fit_fn=fit_rabi, backend="opx",
        )
        outcome = exp.go(soc=None)

        assert isinstance(outcome, ExperimentOutcome)
        assert outcome.fit.success
        assert "period" in outcome.fit.derived
        assert outcome.result.meta["backend"] == "opx"

    def test_acquire_fn_returning_measurement_result(self):
        from strumento.core.fitters import fit_exp

        dev = self._make_device()

        def fancy_acquire(device):
            delays = np.linspace(0.1, 500, 101)
            signal = 0.9 * np.exp(-delays / 75.0) + 0.05
            return from_iq(delays, signal, np.zeros(101), backend="qblox")

        exp = ExternalExperiment(
            dev, acquire_fn=fancy_acquire, fit_fn=fit_exp, backend="qblox",
        )
        outcome = exp.go(soc=None)

        assert outcome.fit.success
        assert abs(outcome.fit.derived["tau"] - 75.0) < 10.0

    def test_signal_fn_applied_before_fit(self):
        from strumento.core.resonator_fitters import fit_lorentzian

        dev = self._make_device()
        freqs = np.linspace(6.95, 7.05, 201)
        f0, hwhm = 7.0, 0.005
        mag = 1.0 - 0.8 / (1 + ((freqs - f0) / hwhm) ** 2)

        def acquire(device):
            return freqs, mag * np.cos(0.5), mag * np.sin(0.5)

        exp = ExternalExperiment(
            dev,
            acquire_fn=acquire,
            fit_fn=fit_lorentzian,
            signal_fn=lambda r: np.hypot(r.avgi, r.avgq),
            backend="vna",
        )
        outcome = exp.go(soc=None)

        assert outcome.fit.success
        assert abs(outcome.fit.derived["f0"] - 7.0) < 0.002

    def test_analyze_works_standalone(self):
        """The analyze() method works on any MeasurementResult."""
        from strumento.core.fitters import fit_rabi

        dev = self._make_device()
        exp = ExternalExperiment(
            dev, acquire_fn=lambda d: (None, 0, 0), fit_fn=fit_rabi,
        )

        gains = np.linspace(0, 30000, 51)
        signal = np.cos(2 * np.pi * gains / 16000)
        result = from_iq(gains, signal, np.zeros(51))

        fit = exp.analyze(result)
        assert fit.success

    def test_sequence_returns_empty_seq(self):
        """sequence() returns an empty Seq (the adapter doesn't compile anything)."""
        from strumento.core.pulses import Seq

        dev = self._make_device()
        exp = ExternalExperiment(
            dev, acquire_fn=lambda d: (None, 0, 0),
            fit_fn=lambda x, y: None,
        )
        seq, sweeps = exp.sequence()
        assert isinstance(seq, Seq)
        assert len(seq.ops) == 0

    def test_environment_stamp_present(self):
        """Every result from ExternalExperiment has an environment stamp (Decision 3/5)."""
        from strumento.core.fitters import fit_rabi

        dev = self._make_device()

        def fake_acquire(device):
            gains = np.linspace(0, 30000, 51)
            signal = np.cos(2 * np.pi * gains / 16000)
            return gains, signal, np.zeros(51)

        exp = ExternalExperiment(
            dev, acquire_fn=fake_acquire, fit_fn=fit_rabi, backend="opx",
        )
        outcome = exp.go(soc=None)

        assert "environment" in outcome.result.meta
        stamp = outcome.result.meta["environment"]
        assert "stamped_at" in stamp
        assert "source" in stamp
        assert "readings" in stamp

    def test_existing_environment_stamp_not_overwritten(self):
        """If acquire_fn returns a MeasurementResult with a stamp, it is preserved."""
        from strumento.core.fitters import fit_exp

        dev = self._make_device()

        def fancy_acquire(device):
            delays = np.linspace(0.1, 500, 101)
            signal = 0.9 * np.exp(-delays / 75.0) + 0.05
            result = from_iq(delays, signal, np.zeros(101), backend="qblox")
            # Pre-stamp with a custom environment
            result.meta["environment"] = {"stamped_at": "pre-existing", "source": "custom"}
            return result

        exp = ExternalExperiment(
            dev, acquire_fn=fancy_acquire, fit_fn=fit_exp, backend="qblox",
        )
        outcome = exp.go(soc=None)

        # The pre-existing stamp should survive (stamp_environment does not overwrite)
        assert outcome.result.meta["environment"]["source"] == "custom"
