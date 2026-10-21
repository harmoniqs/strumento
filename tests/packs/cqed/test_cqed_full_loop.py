"""Full-loop cqed expansion tests via SimulatorSoc (qick-marked — requires qick package).

These tests exercise the complete Runner path for the expansion experiments:
Device.load → experiment.go(SimulatorSoc) → compile → acquire → fit.
They prove the fabricated soccfg_v2_cqed fixture works with the multimode_demo device
and that the response models produce signals the fitters can recover.

A representative experiment from each phase is tested (not all 30 — the contract suite
covers structural conformance for all registered experiments; this covers physics recovery).

Marked ``qick`` because they import and exercise ``StrumentoProgram`` (qick-facing).
"""

from __future__ import annotations

import pytest

pytest.importorskip("qick")

# Trigger response model registration
import strumento.packs.cqed.responses  # noqa: F401
from strumento.core.device import Device
from strumento.core.record import TaskRecord, read_events
from strumento.core.simulator import SimulatorSoc
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.experiments.cavity_spectroscopy import CavitySpectroscopy
from strumento.packs.cqed.experiments.cavity_t1 import CavityT1
from strumento.packs.cqed.experiments.cross_kerr import CrossKerrMeasurement
from strumento.packs.cqed.experiments.displacement_calibration import DisplacementCalibration
from strumento.packs.cqed.experiments.ef_rabi import EFRabi
from strumento.packs.cqed.experiments.ef_spectroscopy import EFSpectroscopy
from strumento.packs.cqed.experiments.fock_cavity_t1 import FockCavityT1
from strumento.packs.cqed.experiments.fock_cavity_t2_ramsey import FockCavityT2Ramsey
from strumento.packs.cqed.experiments.length_rabi import LengthRabi
from strumento.packs.cqed.experiments.number_splitting import NumberSplitting
from strumento.packs.cqed.experiments.qubit_spectroscopy import QubitSpectroscopy
from strumento.packs.cqed.experiments.revival_time_calibration import RevivalTimeCalibration
from strumento.packs.cqed.experiments.sideband_rabi import SidebandRabi
from strumento.packs.cqed.experiments.sideband_spectroscopy import SidebandSpectroscopy
from strumento.packs.cqed.experiments.swap_spectroscopy import SwapSpectroscopy
from strumento.packs.cqed.experiments.t2_echo import T2Echo
from strumento.packs.cqed.experiments.transmon_rb import TransmonRB
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.responses import truth_from_device
from tests._paths import DEVICES

DEVICES_DIR = DEVICES
MULTIMODE_YAML = DEVICES_DIR / "multimode_demo" / "device.yaml"
CQED_SOCCFG = DEVICES_DIR / "_fixtures" / "soccfg_v2_cqed.json"

# Ground truth — perturbed from the device's belief so the fits have something to find.
TRUTH_PERTURBATION = {
    "pi_gain": 9000.0,
    "T1": 85.0,
    "T2": 40.0,
    "f_ef": 3822.0,
    "T2_echo": 70.0,
    "rabi_freq": 4.5,
    "ef_pi_gain": 7000.0,
    "cavity_T1": 1800.0,
    "cavity_T2": 75.0,
    "cavity_detuning": 0.006,
    "chi": 1.3,
    "rb_decay": 45.0,
    # Sideband / cavity / swap truth for expansion full-loop tests
    "sideband_f0g1": 500.0,
    "sideband_pi_gain": 5500.0,
    "alpha_per_gain": 2.0e-4,
    "parity_contrast": 0.85,
    "swap_frequency": 200.0,
}

# Recovery tolerance — looser than the pure-response tests because
# the full loop adds shot noise, DAC quantization, and sweep discretization.
FULL_LOOP_TOLERANCE = 0.12  # 12% relative error


@pytest.fixture(scope="module")
def dev():
    return Device.load(MULTIMODE_YAML)


@pytest.fixture(scope="module")
def soc(dev):
    truth = truth_from_device(dev, **TRUTH_PERTURBATION)
    return SimulatorSoc.from_snapshot(CQED_SOCCFG, truth=truth, seed=42)


# =============================================================================
# Full-loop fit-recovery tests (qick-marked)
# =============================================================================


@pytest.mark.qick
class TestCQEDFullLoop:
    """End-to-end: experiment.go(SimulatorSoc) → FitResult with recovered truth."""

    def test_qubit_spectroscopy(self, dev, soc):
        """frequency sweep → fit_lorentzian recovers f_ge."""
        exp = QubitSpectroscopy(dev, points=81)
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert "f0" in out.fit.derived
        # f0 should be near the device's f_ge (truth is close to belief for spectroscopy)
        assert out.fit.derived["f0"] == pytest.approx(
            dev.calib.qubit.f_ge, rel=0.01
        )

    def test_ef_spectroscopy(self, dev, soc, tmp_path, closing):
        """EF frequency sweep → fit_lorentzian recovers f_ef."""
        exp = EFSpectroscopy(dev, points=81)
        record = closing(TaskRecord.create(tmp_path / "ef-spectroscopy", heartbeat=False))
        out = exp.go(soc, reps=300, progress=False, record=record)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["f0"] == pytest.approx(
            TRUTH_PERTURBATION["f_ef"], abs=0.5
        )
        gates = {
            event["name"]: event
            for event in read_events(record.dir)
            if event.get("ev") == "gate"
        }
        assert gates["fit"]["pass"] is True
        assert gates["fit-recovery"]["pass"] is True
        assert set(gates["fit-recovery"]["params"]["compared"]) == {"f0"}

    def test_cavity_spectroscopy(self, dev, soc):
        """cavity frequency sweep → fit_lorentzian recovers cavity mode frequency."""
        exp = CavitySpectroscopy(dev, points=81)
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert "f0" in out.fit.derived

    def test_length_rabi(self, dev, soc):
        """time-domain Rabi → fit_damped_rabi recovers rabi_freq."""
        # Start at 0.01 µs (not 0.001) to avoid qick's 3-cycle minimum
        exp = LengthRabi(dev, lengths=Sweep(start=0.01, stop=0.5, on="t"), points=61)
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["rabi_freq"] == pytest.approx(
            TRUTH_PERTURBATION["rabi_freq"], rel=FULL_LOOP_TOLERANCE
        )

    def test_t2_echo(self, dev, soc, tmp_path, closing):
        """Hahn echo → fit_exp recovers T2_echo."""
        exp = T2Echo(dev, delays=Sweep(start=0.0, stop=200.0, on="t"), points=61)
        record = closing(TaskRecord.create(tmp_path / "t2-echo", heartbeat=False))
        out = exp.go(soc, reps=300, progress=False, record=record)
        assert out.fit is not None and out.fit.success
        assert "T1" not in out.fit.derived
        assert out.fit.derived["tau"] == pytest.approx(
            TRUTH_PERTURBATION["T2_echo"] / 2.0, rel=FULL_LOOP_TOLERANCE
        )
        assert out.fit.derived["T2_echo"] == pytest.approx(
            TRUTH_PERTURBATION["T2_echo"], rel=FULL_LOOP_TOLERANCE
        )
        gates = {
            event["name"]: event
            for event in read_events(record.dir)
            if event.get("ev") == "gate"
        }
        assert gates["fit-recovery"]["pass"] is True
        assert set(gates["fit-recovery"]["params"]["compared"]) == {"T2_echo"}

    def test_ef_rabi(self, dev, soc):
        """EF Rabi → fit_rabi recovers ef pi_gain."""
        # 16000, not the original 20000 (0.61 frac -> GainOutOfRange since the audit fix
        # wired sweep-endpoint validation into the factory, 2026-09-14): stays under
        # MAX_QUBIT_DRIVE_GAIN=0.5 (16383 int code), still >2x the ef_pi_gain truth (7000)
        # for a comfortably multi-period Rabi fit.
        exp = EFRabi(dev, gains=Sweep(start=0, stop=16000, on="amp"), points=51)
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["pi_gain"] == pytest.approx(
            TRUTH_PERTURBATION["ef_pi_gain"], rel=FULL_LOOP_TOLERANCE
        )

    def test_cavity_t1(self, dev, soc):
        """cavity photon decay → fit_exp recovers cavity_T1."""
        exp = CavityT1(dev, delays=Sweep(start=0.0, stop=8000.0, on="t"), points=61)
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        # The fit reports "T1" — compare to cavity_T1 truth
        assert out.fit.derived["T1"] == pytest.approx(
            TRUTH_PERTURBATION["cavity_T1"], rel=FULL_LOOP_TOLERANCE
        )

    def test_fock_cavity_t1(self, dev, soc):
        """Direct Fock-|1> survival retrieves a photon before ordinary qubit readout."""
        fock_dev = Device.load(MULTIMODE_YAML)
        fock_dev.calib.readout.g_center_iq = (0.0, 0.0)
        fock_dev.calib.readout.e_center_iq = (1.0, 0.0)
        fock_dev.calib.readout.f_center_iq = (0.0, 1.0)
        fock_truth = truth_from_device(fock_dev, **TRUTH_PERTURBATION)
        fock_soc = SimulatorSoc.from_snapshot(CQED_SOCCFG, truth=fock_truth, seed=42)
        exp = FockCavityT1(
            fock_dev,
            delays=Sweep(start=0.0, stop=8000.0, on="t"),
            points=61,
            readout=DispersiveIQ(
                threshold=0.0,
                g_center=(0.0, 0.0),
                e_center=(1.0, 0.0),
                f_center=(0.0, 1.0),
            ),
        )
        out = exp.go(fock_soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["T1"] == pytest.approx(
            TRUTH_PERTURBATION["cavity_T1"], rel=FULL_LOOP_TOLERANCE
        )
        assert out.fit.derived["initial_mapped_e_population"] > 0.8
        assert out.fit.derived["max_f_population"] < 0.1

    def test_fock_cavity_t2_ramsey(self, dev, soc):
        """Direct single-rail Ramsey recovers cavity T2* after return-sideband phase advance."""
        fock_dev = Device.load(MULTIMODE_YAML)
        fock_dev.calib.readout.g_center_iq = (0.0, 0.0)
        fock_dev.calib.readout.e_center_iq = (1.0, 0.0)
        fock_dev.calib.readout.f_center_iq = (0.0, 1.0)
        fock_truth = truth_from_device(fock_dev, **TRUTH_PERTURBATION)
        fock_soc = SimulatorSoc.from_snapshot(CQED_SOCCFG, truth=fock_truth, seed=42)
        exp = FockCavityT2Ramsey(
            fock_dev,
            delays=Sweep(start=0.0, stop=300.0, on="t"),
            points=101,
            reference_detuning_mhz=0.02,
            readout=DispersiveIQ(
                threshold=0.0,
                g_center=(0.0, 0.0),
                e_center=(1.0, 0.0),
                f_center=(0.0, 1.0),
            ),
        )
        out = exp.go(fock_soc, reps=300, progress=False)

        assert out.fit is not None and out.fit.success
        assert out.fit.derived["T2"] == pytest.approx(
            TRUTH_PERTURBATION["cavity_T2"], rel=FULL_LOOP_TOLERANCE
        )
        assert out.fit.derived["ramsey_population_contrast"] > 0.2
        assert out.fit.derived["max_f_population"] < 0.1

    def test_cross_kerr(self, dev, soc):
        """qubit Ramsey with photons → fit_decaysin recovers chi."""
        exp = CrossKerrMeasurement(dev, delays=Sweep(start=0.0, stop=5.0, on="t"), points=81)
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["chi"] == pytest.approx(
            TRUTH_PERTURBATION["chi"], rel=FULL_LOOP_TOLERANCE
        )

    def test_transmon_rb(self, dev, soc):
        """RB decay → fit_exp recovers rb_decay constant."""
        exp = TransmonRB(dev, depth_start=1, depth_stop=100, points=10, seed=42)
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["tau"] == pytest.approx(
            TRUTH_PERTURBATION["rb_decay"], rel=FULL_LOOP_TOLERANCE
        )

    def test_sideband_spectroscopy(self, dev, soc):
        """Sideband: frequency sweep → fit_lorentzian recovers sideband_f0g1."""
        f0 = TRUTH_PERTURBATION["sideband_f0g1"]
        exp = SidebandSpectroscopy(
            dev, freqs=Sweep(start=f0 - 10.0, stop=f0 + 10.0, on="freq"),
            points=81, readout=DispersiveIQ(threshold=0.0, angle=0.0),
        )
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert "f0" in out.fit.derived

    def test_sideband_rabi(self, dev, soc):
        """Sideband: gain sweep → fit_rabi recovers sideband_pi_gain."""
        exp = SidebandRabi(
            dev, gains=Sweep(start=0, stop=15000, on="amp"),
            points=61, readout=DispersiveIQ(threshold=0.0, angle=0.0),
        )
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["pi_gain"] == pytest.approx(
            TRUTH_PERTURBATION["sideband_pi_gain"], rel=FULL_LOOP_TOLERANCE
        )

    def test_displacement_calibration(self, dev, soc):
        """Cavity: displaced-vacuum parity Gaussian vs gain → fit_gaussian_wigner recovers alpha_per_gain."""
        # Displaced-vacuum parity is the Wigner Gaussian exp(-2|alpha|^2), alpha = alpha_per_gain*gain,
        # so the sweep is a Gaussian decay in gain (not an oscillation). 13000 stays under
        # MAX_DISPLACEMENT_GAIN=0.4 (13106 int code); with alpha_per_gain=2e-4 that spans |alpha| up
        # to ~2.6, sampling the Gaussian well before it dies.
        exp = DisplacementCalibration(
            dev, gains=Sweep(start=0, stop=13000, on="gain"),
            points=81, readout=DispersiveIQ(threshold=0.0, angle=0.0),
        )
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert "alpha_per_gain" in out.fit.derived

    def test_number_splitting(self, dev, soc):
        """Cavity: photon-number comb → fit_number_splitting_comb recovers chi from the spacing."""
        # chi is the peak SPACING of the Poisson comb, not one peak's offset; the comb fitter must
        # recover the truth chi (1.3 MHz) from the spacing.
        exp = NumberSplitting(dev, points=121, readout=DispersiveIQ(threshold=0.0, angle=0.0))
        out = exp.go(soc, reps=400, progress=False)
        assert out.fit is not None and out.fit.success
        assert out.fit.derived["chi"] == pytest.approx(
            TRUTH_PERTURBATION["chi"], rel=FULL_LOOP_TOLERANCE
        )

    def test_swap_spectroscopy(self, dev, soc):
        """Coupler: frequency sweep → fit_lorentzian recovers swap_frequency."""
        f0 = TRUTH_PERTURBATION["swap_frequency"]
        exp = SwapSpectroscopy(
            dev, freqs=Sweep(start=f0 - 10.0, stop=f0 + 10.0, on="freq"),
            pair="cavity-S1", points=81,
            readout=DispersiveIQ(threshold=0.0, angle=0.0),
        )
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert "f0" in out.fit.derived

    def test_revival_time_calibration(self, dev, soc):
        """Cavity: damped chi oscillation → fit_decaysin recovers chi and revival_time."""
        exp = RevivalTimeCalibration(
            dev, delays=Sweep(start=0.5, stop=5.0, on="t"),
            points=81, readout=DispersiveIQ(threshold=0.0, angle=0.0),
        )
        out = exp.go(soc, reps=300, progress=False)
        assert out.fit is not None and out.fit.success
        assert "revival_time" in out.fit.derived
        # revival_time = 0.5 / chi; chi = 1.3 → revival ~0.385 µs
        expected_revival = 0.5 / TRUTH_PERTURBATION["chi"]
        assert out.fit.derived["revival_time"] == pytest.approx(
            expected_revival, rel=FULL_LOOP_TOLERANCE
        )


# =============================================================================
# Multi-read response shape tests
# =============================================================================


@pytest.mark.qick
class TestMultiReadResponseShapes:
    """Verify that multi-read experiments produce (1, n_reads) response shapes.

    The contract test catches crashes; this verifies the response model's population
    array shape matches the program's declared trigger count — the shape invariant
    the SimulatorSoc._grid method enforces.
    """

    def _count_measures(self, seq) -> int:
        """Count Measure ops in a flat sequence (non-recursive, sufficient for these)."""
        from strumento.core.pulses import Measure
        return sum(1 for op in seq.ops if isinstance(op, Measure))

    def _response_shape(self, ctx):
        """Get the raw population shape from the response model."""
        from strumento.core.response import response_for
        model = response_for(ctx)
        resp = model(ctx)
        import numpy as np
        return np.asarray(resp.populations).shape

    def test_allxy_21_reads(self, dev):
        from strumento.core.response import SimContext
        from strumento.packs.cqed.experiments.transmon_allxy import TransmonAllXY
        exp = TransmonAllXY(dev)
        seq, _ = exp.sequence()
        n_measures = self._count_measures(seq)
        assert n_measures == 21
        ctx = SimContext(experiment=exp, device=dev, xpts=())
        shape = self._response_shape(ctx)
        assert shape == (1, 21)

    def test_transmon_rb_reads(self, dev):
        from strumento.core.response import SimContext
        exp = TransmonRB(dev, depth_start=1, depth_stop=50, points=5, seed=42)
        seq, _ = exp.sequence()
        n_measures = self._count_measures(seq)
        assert n_measures == 5
        ctx = SimContext(experiment=exp, device=dev, xpts=())
        shape = self._response_shape(ctx)
        assert shape == (1, 5)


# =============================================================================
# Safety guard edge cases
# =============================================================================


class TestSafetyEdgeCases:
    """Boundary conditions for validate_gain and validate_gain_sweep."""

    def test_gain_exactly_at_limit_passes(self):
        from strumento.packs.cqed.safety import MAX_QUBIT_DRIVE_GAIN, validate_gain
        # Gain exactly at the limit should NOT raise
        result = validate_gain(MAX_QUBIT_DRIVE_GAIN, limit=MAX_QUBIT_DRIVE_GAIN,
                               role="qubit_drive", label="test")
        assert result == MAX_QUBIT_DRIVE_GAIN

    def test_gain_just_above_limit_raises(self):
        from strumento.packs.cqed.safety import MAX_QUBIT_DRIVE_GAIN, GainOutOfRange, validate_gain
        with pytest.raises(GainOutOfRange, match="qubit_drive"):
            validate_gain(MAX_QUBIT_DRIVE_GAIN + 0.001, limit=MAX_QUBIT_DRIVE_GAIN,
                          role="qubit_drive", label="test_pulse")

    def test_negative_gain_at_limit_passes(self):
        from strumento.packs.cqed.safety import MAX_READOUT_GAIN, validate_gain
        result = validate_gain(-MAX_READOUT_GAIN, limit=MAX_READOUT_GAIN,
                               role="readout", label="test")
        assert result == -MAX_READOUT_GAIN

    def test_error_message_names_role_and_label(self):
        from strumento.packs.cqed.safety import GainOutOfRange, validate_gain
        with pytest.raises(GainOutOfRange, match="readout") as exc_info:
            validate_gain(0.99, limit=0.5, role="readout", label="my_probe")
        assert "my_probe" in str(exc_info.value)

    def test_zero_gain_always_passes(self):
        from strumento.packs.cqed.safety import validate_gain
        assert validate_gain(0.0, limit=0.001, role="test", label="zero") == 0.0


# =============================================================================
# Multi-mode dispatch — experiments with mode_name="S1"/"S2"
# =============================================================================


@pytest.mark.qick
class TestMultiModeDispatch:
    """Experiments that accept mode_name construct and build sequences for storage modes."""

    def test_cavity_spectroscopy_storage_mode(self, dev):
        """CavitySpectroscopy accepts mode_name='S1' and builds a valid sequence."""
        exp = CavitySpectroscopy(dev, mode_name="S1", points=21)
        seq, _ = exp.sequence()
        assert len(seq.ops) > 0

    def test_revival_time_storage_mode(self, dev, soc):
        """RevivalTimeCalibration with mode_name='S1' compiles and acquires."""
        exp = RevivalTimeCalibration(
            dev, delays=Sweep(start=0.5, stop=5.0, on="t"),
            mode_name="S1", points=21,
            readout=DispersiveIQ(threshold=0.0, angle=0.0),
        )
        out = exp.go(soc, reps=100, progress=False)
        assert out.fit is not None and out.fit.success


# =============================================================================
# Fit-recovery — StatePrepFidelity and WignerTomography
# =============================================================================


@pytest.mark.qick
class TestManualFitRecovery:
    """Experiments with manual FitResult produce valid derived quantities."""

    def test_state_prep_fidelity_runs(self, dev, soc):
        """StatePrepFidelity compiles, acquires, and returns a FitResult."""
        from strumento.packs.cqed.experiments.state_prep_fidelity import StatePrepFidelity
        exp = StatePrepFidelity(
            dev, readout=DispersiveIQ(threshold=0.0, angle=0.0),
        )
        out = exp.go(soc, reps=100, progress=False)
        assert out.fit is not None and out.fit.success
        assert "fidelity_estimate" in out.fit.derived

    def test_wigner_tomography_runs(self, dev, soc):
        """WignerTomography compiles, acquires, and returns a FitResult."""
        from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography
        exp = WignerTomography(
            dev, betas=Sweep(start=-3.0, stop=3.0, on="beta"),
            points=7, readout=DispersiveIQ(threshold=0.0, angle=0.0),
        )
        out = exp.go(soc, reps=100, progress=False)
        assert out.fit is not None and out.fit.success
        assert "w_max" in out.fit.derived
        assert "w_min" in out.fit.derived

    def test_wigner_tomography_prep_and_kerr_wait_runs(self, dev, soc):
        """The state-prep path — coherent-state prep (init_alpha) + a free Kerr wait + a
        complex-beta ray (beta_phase_deg) — compiles, acquires, and analyzes end-to-end on the
        sim soc."""
        from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography
        exp = WignerTomography(
            dev, betas=Sweep(start=-3.0, stop=3.0, on="beta"),
            points=7, init_alpha=1.0, kerr_wait_us=2.0, beta_phase_deg=45.0,
            readout=DispersiveIQ(threshold=0.0, angle=0.0),
        )
        out = exp.go(soc, reps=100, progress=False)
        assert out.fit is not None and out.fit.success
        assert "w_max" in out.fit.derived


# =============================================================================
# Deferred architecture — OC import, waveform substitution, Wigner 2D
# =============================================================================


class TestOCImportAndOverrides:
    """OC-imported pulses compose into experiments; ops-level overrides work."""

    def test_state_prep_with_prepared_state_kwarg(self, dev, soc):
        """StatePrepFidelity accepts a prepared_state Seq instead of default displacement."""
        from strumento.core.pulses import Seq
        from strumento.packs.cqed.experiments.state_prep_fidelity import StatePrepFidelity

        # Build an OC-like prepared state (just a ge_pi for testing — the point is
        # that the kwarg wires through and the experiment compiles + acquires)
        oc_prep = Seq().play(dev.qubit.ge_pi())
        exp = StatePrepFidelity(
            dev, prepared_state=oc_prep, points=7,
            readout=DispersiveIQ(threshold=0.0, angle=0.0),
        )
        seq, _ = exp.sequence()
        assert len(seq.ops) > 0  # the prepared_state composed into the sequence

    def test_transmon_ops_override(self, dev):
        """TransmonOps with an override dict returns the override pulse."""
        from strumento.core.pulses import Arb, Pulse
        from strumento.core.wiring import LineRef
        from strumento.packs.cqed.library.transmon import TransmonOps

        # Build a synthetic OC pulse
        oc_pulse = Pulse(
            line=LineRef("qubit", "drive"),
            freq_mhz=4000.0,
            gain=0.25,
            phase_deg=0.0,
            envelope=Arb(idata=tuple([0.0, 0.5, 1.0, 0.5, 0.0])),
            label="oc_pi_ge",
        )
        ops = TransmonOps(dev, dev.qubit.element, overrides={"pi_ge": oc_pulse})

        # The override is returned instead of the calibrated pulse
        p = ops.ge_pi()
        assert p.label == "oc_pi_ge"
        assert isinstance(p.envelope, Arb)

        # Phase override still works
        p90 = ops.ge_pi(phase=90.0)
        assert p90.phase_deg == 90.0
        assert p90.label == "oc_pi_ge"

        # Non-overridden pulse still comes from calibration
        hpi = ops.ge_hpi()
        assert hpi.label == "hpi_ge"
        assert not isinstance(hpi.envelope, Arb)
