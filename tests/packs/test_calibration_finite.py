"""Every pack's calibration refuses non-finite numbers, and spin refuses negative times (#191).

The cqed field-by-field constraints are in ``tests/packs/cqed/test_calibration_constraints.py``.
This guard is structural: every model reachable from a pack's ``CALIBRATION`` forbids NaN and
infinity, so a new section cannot quietly opt out. Pure lane: no qick needed.
"""

from __future__ import annotations

import importlib
import math

import pytest
from pydantic import ValidationError

from tests.packs.cqed.test_calibration_units import _reachable

PACKS = ("cqed", "spin", "tweezer")


@pytest.mark.parametrize("pack", PACKS)
def test_every_calibration_model_forbids_nan_and_infinity(pack):
    root = importlib.import_module(f"strumento.packs.{pack}").CALIBRATION
    models = _reachable(root)
    assert root in models and len(models) >= 2
    lenient = [m.__name__ for m in models if m.model_config.get("allow_inf_nan", True)]
    assert not lenient, f"{pack}: these calibration models accept NaN/inf: {lenient}"


def test_a_spin_pulse_with_a_nan_sigma_is_refused():
    """Before #191 a NaN slipped past the ``v < 0`` validators (every comparison with NaN is False)."""
    from strumento.packs.spin.calibration import SpinPulseCal

    with pytest.raises(ValidationError, match="finite"):
        SpinPulseCal(gain=0.2, sigma=math.nan)


@pytest.mark.parametrize("field", ["t1_us", "t2_star_us", "t2_echo_us"])
def test_a_spin_dot_refuses_a_negative_time(field):
    from strumento.packs.spin.calibration import DotCal

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        DotCal(f_qubit_mhz=100.0, **{field: -1.0})


def test_a_spin_readout_refuses_a_negative_read_time():
    from strumento.packs.spin.calibration import ReadoutCal

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ReadoutCal(t_read_us=-0.5)


def test_a_spin_pulse_gain_may_still_be_negative():
    """Spin gains are signed v2 fractions in [-1, 1]; a negative gain is a valid calibration."""
    from strumento.packs.spin.calibration import SpinPulseCal

    assert SpinPulseCal(gain=-0.4).gain == -0.4
