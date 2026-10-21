"""Tests for RevivalTimeCalibration.analyze()'s degenerate-fit handling -- had no coverage
before this fix: a fit with freq <= 0 (degenerate) reported
revival_time=0.0 with success left untouched, indistinguishable from a genuinely-converged
(but physically impossible) zero revival time.

Pure: no qick needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult
from strumento.packs.cqed.experiments.revival_time_calibration import RevivalTimeCalibration


@pytest.fixture
def result():
    return MeasurementResult(xpts=list(range(21)), avgi=[0.0] * 21, avgq=[0.0] * 21)


def _fake_fit(freq: float, success: bool = True) -> FitResult:
    return FitResult(
        params={"freq": freq}, stderr={"freq": 0.0},
        derived={"detuning": freq, "freq": freq, "T2": 10.0, "tau": 10.0},
        success=success,
    )


def test_positive_freq_derives_revival_time_and_keeps_success(dev, result):
    with patch(
        "strumento.packs.cqed.experiments.revival_time_calibration.fit_decaysin_detuned",
        return_value=_fake_fit(freq=1.3),
    ):
        fit = RevivalTimeCalibration(dev).analyze(result)
    assert fit.success is True
    assert fit.derived["revival_time"] == pytest.approx(0.5 / 1.3)


def test_zero_freq_is_degenerate_and_marked_unsuccessful(dev, result):
    with patch(
        "strumento.packs.cqed.experiments.revival_time_calibration.fit_decaysin_detuned",
        return_value=_fake_fit(freq=0.0, success=True),  # the underlying fitter thought it converged
    ):
        fit = RevivalTimeCalibration(dev).analyze(result)
    assert fit.derived["revival_time"] == 0.0
    assert fit.success is False  # audit fix: this must not read as a real result


def test_negative_freq_is_degenerate_and_marked_unsuccessful(dev, result):
    with patch(
        "strumento.packs.cqed.experiments.revival_time_calibration.fit_decaysin_detuned",
        return_value=_fake_fit(freq=-0.4, success=True),
    ):
        fit = RevivalTimeCalibration(dev).analyze(result)
    assert fit.derived["revival_time"] == 0.0
    assert fit.success is False
