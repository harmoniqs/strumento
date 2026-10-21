"""Full-loop spin experiment tests via SimulatorSoc (qick-marked — requires qick package).

These tests exercise the complete Runner path: Device.load → experiment.go(SimulatorSoc)
→ compile → acquire → fit. They prove the fabricated soccfg_v2_spin fixture works with
each experiment's pulse program end to end.

Marked ``qick`` because they import and exercise ``StrumentoProgram`` (qick-facing).
"""

from __future__ import annotations

import pytest

pytest.importorskip("qick")

# Trigger response model registration
import strumento.packs.spin.responses  # noqa: F401
from strumento.core.device import Device
from strumento.core.simulator import SimulatorSoc
from strumento.packs.spin.experiments.exchange_oscillation import ExchangeOscillation
from strumento.packs.spin.experiments.hahn_echo import HahnEcho
from strumento.packs.spin.experiments.spin_rabi import SpinRabi
from strumento.packs.spin.experiments.spin_ramsey import SpinRamsey
from strumento.packs.spin.experiments.spin_t1 import SpinT1
from tests._paths import DEVICES

DEVICES_DIR = DEVICES
SPIN_DEMO_YAML = DEVICES_DIR / "spin_demo" / "device.yaml"
SPIN_SOCCFG = DEVICES_DIR / "_fixtures" / "soccfg_v2_spin.json"

# Ground truth for all experiments
TRUTH = {
    "T1": 68.0,
    "omega_rabi_mhz": 20.0,
    "T_Rabi": 5.0,
    "T2_star": 2.5,
    "ramsey_detuning": 1.2,
    "T2_echo": 18.0,
    "T2_DD": 35.0,
    "J_mhz": 50.0,
    "T_J": 3.0,
}

# Recovery tolerance — looser than the pure-response tests because
# the full loop adds shot noise, DAC quantization, and sweep discretization.
FULL_LOOP_TOLERANCE = 0.10  # 10% relative error


@pytest.fixture(scope="module")
def dev():
    return Device.load(SPIN_DEMO_YAML)


@pytest.fixture(scope="module")
def soc():
    return SimulatorSoc.from_snapshot(SPIN_SOCCFG, truth=TRUTH, seed=42)


# =============================================================================
# Full-loop fit-recovery tests (qick-marked)
# =============================================================================


@pytest.mark.qick
class TestSpinFullLoop:
    """End-to-end: experiment.go(SimulatorSoc) → FitResult with recovered truth."""

    def test_spin_t1(self, dev, soc):
        exp = SpinT1(dev, dot_name="q0")
        out = exp.go(soc, reps=500, progress=False)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["tau"] == pytest.approx(TRUTH["T1"], rel=FULL_LOOP_TOLERANCE)

    def test_spin_rabi(self, dev, soc):
        exp = SpinRabi(dev, dot_name="q0")
        out = exp.go(soc, reps=500, progress=False)
        assert out.fit is not None and out.fit.success
        expected_period = 1.0 / TRUTH["omega_rabi_mhz"]
        assert out.fit.derived["period"] == pytest.approx(expected_period, rel=FULL_LOOP_TOLERANCE)

    def test_spin_ramsey(self, dev, soc):
        exp = SpinRamsey(dev, dot_name="q0")
        out = exp.go(soc, reps=500, progress=False)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["tau"] == pytest.approx(TRUTH["T2_star"], rel=FULL_LOOP_TOLERANCE)
        assert out.fit.derived["freq"] == pytest.approx(TRUTH["ramsey_detuning"], rel=FULL_LOOP_TOLERANCE)

    def test_hahn_echo(self, dev, soc):
        exp = HahnEcho(dev, dot_name="q0")
        out = exp.go(soc, reps=500, progress=False)
        assert out.fit is not None and out.fit.success
        # Model: exp(-2tau/T2e), fitter sees exp(-tau/tau_fit) → tau_fit = T2e/2
        assert out.fit.derived["tau"] == pytest.approx(
            TRUTH["T2_echo"] / 2.0, rel=FULL_LOOP_TOLERANCE
        )

    def test_exchange_oscillation(self, dev, soc):
        exp = ExchangeOscillation(dev, dot_name="q0", barrier_name="b0")
        out = exp.go(soc, reps=500, progress=False)
        assert out.fit is not None and out.fit.success
        expected_period = 1.0 / TRUTH["J_mhz"]
        assert out.fit.derived["period"] == pytest.approx(expected_period, rel=FULL_LOOP_TOLERANCE)


# =============================================================================
# Multi-encoding: Hole-spin device through SimulatorSoc
# =============================================================================

HOLE_DEMO_YAML = DEVICES_DIR / "spin_hole_demo" / "device.yaml"


@pytest.fixture(scope="module")
def hole_dev():
    return Device.load(HOLE_DEMO_YAML)


@pytest.fixture(scope="module")
def hole_soc():
    return SimulatorSoc.from_snapshot(SPIN_SOCCFG, truth=TRUTH, seed=42)


@pytest.mark.qick
class TestHoleSpinFullLoop:
    """End-to-end: hole-spin experiments compile and acquire through SimulatorSoc.

    Verifies the fixed HoleSpinOps (LineRef role = 'drive') produces pulse programs
    that assemble, acquire, and fit correctly on the fabricated spin soccfg fixture.
    """

    def test_hole_spin_t1(self, hole_dev, hole_soc):
        exp = SpinT1(hole_dev, dot_name="h1")
        out = exp.go(hole_soc, reps=500, progress=False)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["tau"] == pytest.approx(TRUTH["T1"], rel=FULL_LOOP_TOLERANCE)

    def test_hole_spin_rabi(self, hole_dev, hole_soc):
        exp = SpinRabi(hole_dev, dot_name="h1")
        out = exp.go(hole_soc, reps=500, progress=False)
        assert out.fit is not None and out.fit.success
        expected_period = 1.0 / TRUTH["omega_rabi_mhz"]
        assert out.fit.derived["period"] == pytest.approx(expected_period, rel=FULL_LOOP_TOLERANCE)

    def test_hole_spin_ramsey(self, hole_dev, hole_soc):
        exp = SpinRamsey(hole_dev, dot_name="h1")
        out = exp.go(hole_soc, reps=500, progress=False)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["tau"] == pytest.approx(TRUTH["T2_star"], rel=FULL_LOOP_TOLERANCE)
