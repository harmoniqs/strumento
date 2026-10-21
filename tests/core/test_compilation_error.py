"""Tests for compilation error handling (CompilationError).

Verifies that program build and acquire failures are wrapped in a structured
CompilationError with the experiment name, failure phase, constraint description,
and (where applicable) a suggestion about what to fix.
"""

from __future__ import annotations

import pytest

from strumento.core.experiment import CompilationError, Runner
from strumento.core.pulses import Const, Gaussian, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.wiring import LineRef

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_loopback():
    """Load the loopback_demo device + a mock soc for testing."""
    from pathlib import Path

    from strumento import Device
    from strumento.core.mock import MockQickSocV2

    repo = Path(__file__).resolve().parent.parent.parent
    dev = Device.load(str(repo / "devices" / "loopback_demo" / "device.yaml"))
    soc = MockQickSocV2(dev.soccfg_snapshot)
    return dev, soc


class _FakeExperiment:
    """Minimal experiment-like object for testing Runner.run() directly."""
    pass


# ---------------------------------------------------------------------------
# CompilationError structure
# ---------------------------------------------------------------------------


class TestCompilationErrorStructure:
    """The error carries structured context for actionable diagnostics."""

    def test_fields_populated(self):
        err = CompilationError(
            experiment="AmplitudeRabi",
            phase="build",
            constraint="Pulse length of 0 cycles is shorter than the min of 3",
            original_type="RuntimeError",
            suggestion="Increase the envelope sigma.",
            details={"line": "qubit"},
        )
        assert err.experiment == "AmplitudeRabi"
        assert err.phase == "build"
        assert "0 cycles" in err.constraint
        assert err.original_type == "RuntimeError"
        assert "sigma" in err.suggestion
        assert err.details["line"] == "qubit"

    def test_str_is_human_readable(self):
        err = CompilationError(
            experiment="T1",
            phase="build",
            constraint="no line 'cavity' in wiring (have ['qubit'])",
            original_type="KeyError",
            suggestion="Check device.yaml wiring.lines.",
        )
        s = str(err)
        assert "T1" in s
        assert "compilation failed" in s
        assert "cavity" in s
        assert "Suggestion" in s

    def test_is_exception(self):
        err = CompilationError(
            experiment="Test", phase="build", constraint="bad", original_type="ValueError",
        )
        assert isinstance(err, Exception)
        with pytest.raises(CompilationError):
            raise err

    def test_chained_cause_preserved(self):
        """The original exception is chained via __cause__."""
        original = RuntimeError("the real error")
        try:
            raise CompilationError(
                experiment="Test", phase="build", constraint="wrapped",
                original_type="RuntimeError",
            ) from original
        except CompilationError as err:
            assert err.__cause__ is original


# ---------------------------------------------------------------------------
# Runner.run() wrapping
# ---------------------------------------------------------------------------


@pytest.mark.qick
class TestRunnerCompilationWrapping:
    """Runner.run() wraps build and acquire failures in CompilationError."""

    def test_missing_wiring_line_raises_compilation_error(self):
        """A sequence referencing a non-existent wiring line is caught."""
        dev, soc = _load_loopback()

        # Build a sequence targeting a line that doesn't exist in loopback_demo
        bad_seq = Seq().play(
            Pulse(
                line=LineRef("nonexistent_cavity", "drive"),
                freq_mhz=5200.0,
                gain=0.3,
                envelope=Const(length_us=1.0),
            )
        ).measure()

        runner = Runner()
        exp = _FakeExperiment()
        with pytest.raises(CompilationError) as exc_info:
            runner.run(dev, bad_seq, (), soc, experiment=exp)

        err = exc_info.value
        assert err.phase == "build"
        assert "nonexistent_cavity" in err.constraint
        assert err.original_type == "KeyError"
        assert err.experiment == "_FakeExperiment"

    def test_zero_length_pulse_raises_compilation_error(self):
        """A Gaussian with sigma=0 triggers a qick RuntimeError, caught as CompilationError."""
        dev, soc = _load_loopback()

        bad_seq = Seq().play(
            Pulse(
                line=LineRef("qubit", "drive"),
                freq_mhz=4000.0,
                gain=0.25,
                envelope=Gaussian(sigma_us=0.0),  # zero-length envelope
            )
        ).measure()

        runner = Runner()
        with pytest.raises(CompilationError) as exc_info:
            runner.run(dev, bad_seq, (), soc, experiment=_FakeExperiment())

        err = exc_info.value
        assert err.phase == "build"
        assert err.original_type in ("RuntimeError", "ValueError")  # depends on where qick/compiler catches it

    def test_successful_compilation_does_not_raise(self):
        """A valid sequence compiles and acquires without CompilationError."""
        dev, soc = _load_loopback()

        good_seq = Seq().play(
            Pulse(
                line=LineRef("qubit", "drive"),
                freq_mhz=4000.0,
                gain=0.25,
                envelope=Gaussian(sigma_us=0.030),
            )
        ).measure()

        runner = Runner()
        result = runner.run(dev, good_seq, (), soc, experiment=_FakeExperiment())
        assert isinstance(result, MeasurementResult)

    def test_suggestion_populated_for_known_patterns(self):
        """Known error patterns get a suggestion about what to fix."""
        dev, soc = _load_loopback()

        bad_seq = Seq().play(
            Pulse(
                line=LineRef("missing_line", "drive"),
                freq_mhz=4000.0,
                gain=0.25,
                envelope=Const(length_us=1.0),
            )
        ).measure()

        runner = Runner()
        with pytest.raises(CompilationError) as exc_info:
            runner.run(dev, bad_seq, (), soc, experiment=_FakeExperiment())

        err = exc_info.value
        assert err.suggestion  # non-empty suggestion for "no line" pattern

    def test_details_include_seq_context(self):
        """The error details include sequence metadata."""
        dev, soc = _load_loopback()

        bad_seq = Seq().play(
            Pulse(
                line=LineRef("nonexistent", "drive"),
                freq_mhz=4000.0,
                gain=0.25,
                envelope=Const(length_us=1.0),
            )
        ).measure()

        runner = Runner()
        with pytest.raises(CompilationError) as exc_info:
            runner.run(dev, bad_seq, (), soc, experiment=_FakeExperiment())

        err = exc_info.value
        assert "n_ops" in err.details
        assert err.details["has_readout"] is True

    def test_experiment_go_propagates_compilation_error(self):
        """CompilationError propagates through Experiment.go() unchanged."""
        from strumento.core.experiment import Experiment
        from strumento.core.fitters import FitResult

        dev, soc = _load_loopback()

        class BadExperiment(Experiment):
            def sequence(self):
                return Seq().play(
                    Pulse(
                        line=LineRef("no_such_line", "drive"),
                        freq_mhz=4000.0,
                        gain=0.25,
                        envelope=Const(length_us=1.0),
                    )
                ).measure(), ()

            def analyze(self, result):
                return FitResult(params={}, stderr={}, derived={}, success=False)

        exp = BadExperiment(dev)
        with pytest.raises(CompilationError) as exc_info:
            exp.go(soc)

        assert exc_info.value.experiment == "BadExperiment"
        assert exc_info.value.phase == "build"
