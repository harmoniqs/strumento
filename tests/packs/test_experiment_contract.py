"""The experiment contract — structural guards, parametrized over the *registries*.

WHY THIS FILE EXISTS (a post-mortem, so the guards are not mistaken for ceremony).

Four of the five cqed analyzers were once broken on realistically-shaped acquisition data while
the suite was green. A single-read averaged acquisition arrives shaped ``(n_reads, n_points)`` —
``(1, N)`` — but the analyzer unit tests hand-built **1-D** arrays and called ``analyze()``
directly, so the analyzers were only ever verified against a shape production never emits.
``fit_exp``'s ``float(y[-1])`` raised ``TypeError`` on the real shape and the resonator fitters'
``column_stack`` silently mis-shaped; only Rabi survived, by accident of using ``y.mean()``.
Two further silences let it hide: the mock returns zero-valued IQ, so no end-to-end assertion
could tell a working analyzer from a broken one; and **nothing was parametrized over the
registries**, so four broken analyzers were four independent silences instead of one loud
failure.

The guards below close that. They are deliberately about **shape and wiring conformance**, not
fit accuracy — ``tests/e2e/test_fit_recovery.py`` owns accuracy, with its own error budget. What
these add is that the conformance claim is made for *every registered experiment*, through the
*real* Runner path, and that the claim cannot quietly stop being made as the package grows.
"""

from __future__ import annotations

import inspect
import pkgutil
import sys
from collections.abc import Callable
from importlib import import_module

import pytest

import strumento.packs.spin.responses as _spin_responses  # noqa: F401 — triggers registration
from strumento.core.device import Device
from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.response import ResponseError
from strumento.core.sweeps import Sweep
from strumento.packs.cqed import EXPERIMENTS as CQED_EXPERIMENTS
from strumento.packs.cqed.calibration import ModeCal
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.responses import truth_from_device
from strumento.packs.spin import EXPERIMENTS as SPIN_EXPERIMENTS
from strumento.packs.spin.readout import ElzermanReadout, PSBReadout
from strumento.packs.spin.responses import truth_from_device as spin_truth_from_device
from strumento.packs.tweezer import EXPERIMENTS as TWEEZER_EXPERIMENTS
from tests._paths import DEVICES, LOOPBACK_YAML, MULTIMODE_YAML, SPIN_YAML
from tests._v2gate import V2_SOCCFG

#: Every pack's registry, by pack name.
REGISTRIES: dict[str, dict[str, type]] = {
    "cqed": CQED_EXPERIMENTS,
    "spin": SPIN_EXPERIMENTS,
    "tweezer": TWEEZER_EXPERIMENTS,
}


#: `Experiment`-shaped classes deliberately absent from a registry, each with a written reason.
#: Guard 2 requires every discovered subclass to be registered-and-covered OR listed here, so an
#: unregistered experiment cannot escape the contract by simply never being registered.
NOT_REGISTERED: dict[str, str] = {
    # The bring-up graph's node wrapper (#56): a delegating Experiment the pack's graph
    # factories construct per step around a real (registered) experiment — it carries the
    # node's verdict map, never runs standalone with pack-level defaults.
    "BringupNode": (
        "the cqed bring-up graph's node wrapper: constructed by the pack's bringup module "
        "around a registered experiment per step (inner delegation + the verdict map); not a "
        "standalone pack experiment"
    ),
    #
    # The following are NOT Experiment subclasses — they don't subclass the base class and
    # are invisible to the contract guards. Listed here as documentation so they aren't
    # forgotten. They use .run() (DC-sweep) or .analyze() (pure analyzer), not .go().
    #
    # DC-sweep classes (host-driven via DCSource + probe callable):
    #   PinchOff, CoulombDiamond, VirtualGateCalibration, TunnelCouplingTune,
    #   SensorTune, DispersiveGateCalibration — use .run() → MeasurementResult
    #   RFReflectometryTune — uses .run() with set_frequency callable
    #   ChargeStability — 2D DC sweep, .run() → MeasurementResult
    #
    # Host-driven, program-driven (no DC axis to sweep — plays one compiled program directly):
    #   TimeOfFlight (#104) — plays the readout probe, acquires decimated, .run() → soc →
    #   MeasurementResult; .analyze() → FitResult via fit_time_of_flight
    #
    # Pure analyzers (no acquisition, just data processing):
    #   PSBHistogram — .analyze(shots) → FitResult
    #
}

#: Per-experiment construction for the contract run. Scans are the SMALLEST that still converge —
#: this guard pays for shape conformance, not precision, and the suite must stay fast. If an entry
#: stops converging, enlarge its scan here; never weaken the assertions below.
CONTRACT_KWARGS: dict[str, dict] = {
    # loopback_demo has a single generator and no cavity element, so the default
    # ("readout", "probe") line does not exist there — these four (like
    # resonator_spectroscopy below) probe on the instance's one line (issue #102).
    "amplitude_rabi": dict(gains=Sweep(start=0.0, stop=16000.0, on="amp"), points=41, readout=DispersiveIQ(threshold=0.5), line=("qubit", "drive")),
    "t1": dict(delays=Sweep(start=0.0, stop=250.0, on="t"), points=41, readout=DispersiveIQ(threshold=0.5), line=("qubit", "drive")),
    "t2_ramsey": dict(delays=Sweep(start=0.0, stop=80.0, on="t"), points=81, readout=DispersiveIQ(threshold=0.5), line=("qubit", "drive")),
    # loopback_demo has a single generator and no cavity element, so the default
    # ("readout", "probe") line does not exist there -- parity()'s #100 measure=True default
    # needs the same override resonator_spectroscopy uses. Its mode is calibration-only scaffolding
    # (no BosonicMode element), so it is named rather than inferred.
    "parity_fringe": dict(mode="cavity", phases=Sweep(start=0.0, stop=360.0, on="phase"), points=41, readout=DispersiveIQ(threshold=0.5), line=("qubit", "drive")),
    # loopback_demo has a single generator and no cavity element, so the default
    # ("readout", "probe") line does not exist there.
    "resonator_spectroscopy": dict(line=("qubit", "drive")),
    "resonator_power_sweep": dict(points=21, gain_codes=(0, 1000)),
    "active_reset": dict(rounds=1, line=("qubit", "drive")),
    # --- cqed expansion experiments (use multimode_demo + soccfg_v2_cqed) ---
    "qubit_spectroscopy": dict(points=41, readout=DispersiveIQ(threshold=0.5)),
    "ef_spectroscopy": dict(points=41, readout=DispersiveIQ(threshold=0.5)),
    "cavity_spectroscopy": dict(mode_name="cavity", points=41, readout=DispersiveIQ(threshold=0.5)),
    "dispersive_shift": dict(points=41, readout=DispersiveIQ(threshold=0.5)),
    "power_sweep": dict(points=41, readout=DispersiveIQ(threshold=0.5)),
    "number_splitting": dict(points=41, readout=DispersiveIQ(threshold=0.5)),
    "length_rabi": dict(lengths=Sweep(start=0.01, stop=0.5, on="t"), points=41, readout=DispersiveIQ(threshold=0.5)),
    "t2_echo": dict(delays=Sweep(start=0.0, stop=200.0, on="t"), points=41, readout=DispersiveIQ(threshold=0.5)),
    "drag_calibration": dict(alpha_start=-0.1, alpha_stop=0.1, points=11, readout=DispersiveIQ(threshold=0.5)),
    "drag_pair_calibration": dict(alpha_start=-0.25, alpha_stop=0.25, points=21, readout=DispersiveIQ(threshold=0.5)),
    "ramsey_detuning_pair": dict(stop_us=40.0, points=21, ramsey_freq_mhz=0.1, readout=DispersiveIQ(threshold=0.5)),
    "transmon_allxy": dict(readout=DispersiveIQ(threshold=0.5)),
    "ef_rabi": dict(gains=Sweep(start=0, stop=16000, on="amp"), points=41, readout=DispersiveIQ(threshold=0.5)),
    "eft1": dict(delays=Sweep(start=0.0, stop=100.0, on="t"), points=41, readout=DispersiveIQ(threshold=0.5)),
    "eft2_ramsey": dict(delays=Sweep(start=0.0, stop=32.0, on="t"), points=81, readout=DispersiveIQ(threshold=0.5)),
    "eft2_echo": dict(delays=Sweep(start=0.0, stop=100.0, on="t"), points=41, readout=DispersiveIQ(threshold=0.5)),
    "frequency_rabi": dict(points=41, readout=DispersiveIQ(threshold=0.5)),
    "single_shot_readout": dict(readout=DispersiveIQ(threshold=0.5)),
    "readout_optimization": dict(points=41, readout=DispersiveIQ(threshold=0.5)),
    "sideband_spectroscopy": dict(points=41, readout=DispersiveIQ(threshold=0.5)),
    "sideband_rabi": dict(points=41, readout=DispersiveIQ(threshold=0.5)),
    "multiphoton_ladder": dict(n_max=2, mode_name="cavity", readout=DispersiveIQ(threshold=0.5)),
    "displacement_calibration": dict(mode_name="cavity", points=41, readout=DispersiveIQ(threshold=0.5)),
    "cavity_t1": dict(delays=Sweep(start=0.0, stop=8000.0, on="t"), mode_name="cavity", points=41, readout=DispersiveIQ(threshold=0.5)),
    "fock_cavity_t1": dict(delays=Sweep(start=0.0, stop=1200.0, on="t"), points=41, readout=DispersiveIQ(threshold=0.5)),
    "fock_cavity_t2_ramsey": dict(delays=Sweep(start=0.0, stop=300.0, on="t"), points=41, readout=DispersiveIQ(threshold=0.5)),
    "cavity_t2_ramsey": dict(delays=Sweep(start=0.0, stop=4000.0, on="t"), mode_name="cavity", points=41, readout=DispersiveIQ(threshold=0.5)),
    "revival_time_calibration": dict(delays=Sweep(start=0.5, stop=3.0, on="t"), mode_name="cavity", points=41, readout=DispersiveIQ(threshold=0.5)),
    "photon_number_distribution": dict(n_max=4, mode_name="cavity", readout=DispersiveIQ(threshold=0.5)),
    "wigner_tomography": dict(mode_name="cavity", points=21, readout=DispersiveIQ(threshold=0.5)),
    "swap_spectroscopy": dict(pair="cavity-S1", points=41, readout=DispersiveIQ(threshold=0.5)),
    "swap_rabi": dict(pair="cavity-S1", points=41, readout=DispersiveIQ(threshold=0.5)),
    "storage_t1": dict(pair="cavity-S1", mode_name="S1", points=41, readout=DispersiveIQ(threshold=0.5)),
    "cross_kerr": dict(delays=Sweep(start=0.0, stop=5.0, on="t"), mode_name="cavity", points=41, readout=DispersiveIQ(threshold=0.5)),
    "qnd_readout": dict(n_repeats=5, readout=DispersiveIQ(threshold=0.5)),
    "state_prep_fidelity": dict(mode_name="cavity", points=21, readout=DispersiveIQ(threshold=0.5)),
    "transmon_rb": dict(depth_start=1, depth_stop=50, points=6, seed=42, readout=DispersiveIQ(threshold=0.5)),
    # --- cqed bring-up graph nodes (#56: thin Seq experiments with vetted fitters) ---
    "punchout": dict(points=41, readout=DispersiveIQ(threshold=0.5)),
    "gain_ladder_spectroscopy": dict(points=81, readout=DispersiveIQ(threshold=0.5)),
    "readout_state": dict(readout=DispersiveIQ(threshold=0.5)),
    "tls_scan": dict(points=81, readout=DispersiveIQ(threshold=0.5)),
    "validation_check": dict(points=81, readout=DispersiveIQ(threshold=0.5)),
    # n_pulses=1 + the experiment's own default window: exactly one fringe cycle by
    # construction (err_amp_freq_sens x default 0.4 MHz span = 1) — the smallest scan that
    # still converges; the tight-precision N-scaling claim is the S1-style gate in
    # tests/e2e/test_fit_recovery.py, not this shape/wiring guard. line=("qubit","drive"):
    # loopback_demo dispatch, same single-generator reason as amplitude_rabi/t1/t2_ramsey above.
    "error_amplification": dict(n_pulses=1, points=21, readout=DispersiveIQ(threshold=0.5), line=("qubit", "drive")),
    # --- spin experiments (use spin_demo device + soccfg_v2_spin fixture) ---
    "spin_t1": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "spin_rabi": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "spin_ramsey": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "hahn_echo": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "cpmg": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "exchange_oscillation": dict(dot_name="q0", barrier_name="b0", readout=PSBReadout(threshold=0.5)),
    "edsr_spectroscopy": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "gradient_calibration": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "g_factor_map": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "sweet_spot_tune": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "charge_noise_spectroscopy": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "allxy": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "randomized_benchmarking": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "interleaved_rb": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "leakage_benchmark": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "composite_pulse_calibration": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "heralded_init": dict(dot_name="q0", readout=PSBReadout(threshold=0.5)),
    "exchange_gate_calibration": dict(control_dot="q0", target_dot="q0", barrier_name="b0", readout=PSBReadout(threshold=0.5)),
    "tunnel_rate_tune": dict(dot_name="q0", plunger_name="p0", readout=ElzermanReadout(threshold=0.5)),
    "elzerman_optimize": dict(dot_name="q0", plunger_name="p0", readout=ElzermanReadout(threshold=0.5)),
    "psb_calibration": dict(dot_name="q0", plunger_name="p0", readout=PSBReadout(threshold=0.5)),
}

#: Reps per contract run — enough for a fit to converge on a small scan, small enough to be fast.
CONTRACT_REPS = 300
CONTRACT_SEED = 7

#: The perturbed ground truth every contract run is armed with. Values need not be realistic; they
#: only need to differ from the instance's belief so a fit has something to find.
CONTRACT_PERTURBATION = dict(
    pi_gain=9350.0, T1=68.0, T2=22.0, ramsey_detuning=0.12,
    resonator_f0=750.08, resonator_Q_loaded=2450.0, parity_phase=62.0,
    thermal_population=0.15,
)

#: Spin-specific ground truth for the contract run.
SPIN_CONTRACT_TRUTH = dict(
    T1=68.0, omega_rabi_mhz=20.0, T_Rabi=5.0,
    T2_star=2.5, ramsey_detuning=1.2, T2_echo=18.0,
    J_mhz=50.0, T_J=3.0,
)

#: cQED expansion ground truth for the contract run.
CQED_EXPANSION_TRUTH = dict(
    pi_gain=9350.0, T1=68.0, T2=22.0, T2_echo=60.0,
    rabi_freq=4.0, ef_pi_gain=7000.0,
    cavity_T1=1800.0, cavity_T2=900.0, cavity_detuning=0.04,
    chi=1.3, alpha_per_gain=2.0e-4,
    sideband_pi_gain=5500.0, swap_frequency=200.0, swap_period=0.75,
    storage_T1=4500.0, rb_decay=40.0, qnd_fidelity=0.93,
    parity_contrast=0.85, readout_contrast=0.94,
)

#: Analyzers whose fit is defined only on a POPULATION (they refuse raw projected IQ). The real-path
#: guard hands them a readout anchored to the simulator's own g/e blob centres — the same shape the
#: default readout (``DispersiveIQ.from_calib``) carries on a device with recorded blobs. Not a
#: weakening: a bare readout made the DRAG guard "pass" on a wrong-sign fit (alpha_opt -0.50 vs
#: truth +0.50, see test_drag_calibration.py), and the analyzer now refuses that path outright.
#:
#: Keyed by registry name; the value says WHICH constructions refuse raw IQ (``None`` = every one).
#: Error amplification refuses only in its fixed-period mode (``fringe_period`` set — the peak vs
#: trough choice needs a population); its default cosine fit takes raw IQ, so its default contract
#: entry stays on the bare readout. The key set is checked against the analyzers that call
#: ``is_population_anchored`` (see ``test_blob_anchored_set_matches_the_analyzers_that_check``).
BLOB_ANCHORED_ANALYZERS: dict[str, Callable[[dict], bool] | None] = {
    "drag_calibration": None,
    "drag_pair_calibration": None,
    "error_amplification": lambda kwargs: kwargs.get("fringe_period") is not None,
}


def _needs_blob_anchor(name: str, kwargs: dict) -> bool:
    """Whether ``name`` constructed with ``kwargs`` refuses a bare (unanchored) readout."""
    if name not in BLOB_ANCHORED_ANALYZERS:
        return False
    mode = BLOB_ANCHORED_ANALYZERS[name]
    return mode is None or mode(kwargs)


def _sim_blob_anchored_readout(cls, dev, soc, kwargs):
    """A ``DispersiveIQ`` whose g/e centres sit where ``soc`` places the |g>/|e> blobs, in output units.

    Measured, not hard-coded: one probe acquire arms the simulator, and its context carries both the
    response-model blob geometry and the output scale it applies (``adc_scale`` / readout window).
    """
    from strumento.packs.cqed.responses import params

    cls(dev, **kwargs).go(soc, reps=8, progress=False, analyze=False)
    ctx = soc._ctx
    p = params(ctx)
    scale = float(ctx.readout_output_scale)
    base = kwargs["readout"]
    return DispersiveIQ(
        threshold=base.threshold, angle=base.angle,
        g_center=tuple(float(v) * scale for v in p["blob0"]),
        e_center=tuple(float(v) * scale for v in p["blob1"]),
    )


#: The spin soccfg fixture path.
_SPIN_SOCCFG = DEVICES / "_fixtures" / "soccfg_v2_spin.json"

#: Which registered experiment names belong to the spin pack (for dispatch).
_SPIN_EXPERIMENT_NAMES = set(SPIN_EXPERIMENTS.keys())

#: The cqed v2 soccfg fixture path (for expansion experiments needing multimode).
_CQED_SOCCFG = DEVICES / "_fixtures" / "soccfg_v2_cqed.json"

#: The cqed experiments the contract builds on loopback_demo (single generator, no cavity element).
#: Explicit, like the expansion set below; ``test_every_cqed_experiment_has_a_contract_device``
#: requires the two to partition the registry, so a new experiment must be placed in one.
_CQED_LOOPBACK_NAMES = {
    "amplitude_rabi", "t1", "t2_ramsey", "parity_fringe", "resonator_spectroscopy",
    "active_reset", "error_amplification",
}

#: cQED expansion experiment names — these need the multimode device (not loopback).
_CQED_EXPANSION_NAMES = {
    "qubit_spectroscopy", "ef_spectroscopy", "cavity_spectroscopy", "dispersive_shift",
    "power_sweep", "number_splitting", "length_rabi", "t2_echo", "drag_calibration",
    "drag_pair_calibration", "ramsey_detuning_pair",
    "transmon_allxy", "ef_rabi", "eft1", "eft2_ramsey", "eft2_echo", "frequency_rabi", "single_shot_readout",
    "readout_optimization", "sideband_spectroscopy", "sideband_rabi", "multiphoton_ladder",
    "displacement_calibration", "cavity_t1", "fock_cavity_t1", "fock_cavity_t2_ramsey", "cavity_t2_ramsey", "revival_time_calibration",
    "photon_number_distribution", "wigner_tomography", "swap_spectroscopy", "swap_rabi",
    "storage_t1", "cross_kerr", "qnd_readout", "state_prep_fidelity", "transmon_rb",
    # the bring-up graph's nodes (#56)
    "punchout", "gain_ladder_spectroscopy", "readout_state", "tls_scan", "validation_check",
    "resonator_power_sweep",
}

ALL_REGISTERED = sorted({name for reg in REGISTRIES.values() for name in reg})


def _loopback_device() -> Device:
    """The v2-snapshot instance for the original 6 cqed experiments (amplitude_rabi, t1, t2_ramsey,
    parity_fringe, resonator_spectroscopy, active_reset). Expansion experiments use
    `_multimode_device` with the `soccfg_v2_cqed` fixture instead."""
    dev = Device.load(LOOPBACK_YAML)
    # The parity sequence is two qubit half-pi pulses around a revival wait; loopback_demo has no
    # cavity element, so that one calibration value is injected as scaffolding rather than
    # fictionalising the shipped instance (same choice as the fit-recovery gate's fixture).
    dev.calib.modes["cavity"] = ModeCal(revival_time=[1.6, 0.8])
    return dev


def _spin_device() -> Device:
    """The spin_demo instance for spin contract runs."""
    return Device.load(SPIN_YAML)


def _multimode_device() -> Device:
    """The multimode cqed instance — needed by expansion experiments that use cavity/coupler lines."""
    return Device.load(MULTIMODE_YAML)


def _experiment_subclasses(pack: str) -> dict[str, type]:
    """Every `Experiment` subclass defined under `strumento.packs.<pack>`."""
    found: dict[str, type] = {}
    try:
        root = import_module(f"strumento.packs.{pack}")
    except ModuleNotFoundError:  # pragma: no cover - a pack that does not ship
        return found
    prefix = f"{root.__name__}."
    mods = [root]
    for info in pkgutil.walk_packages(getattr(root, "__path__", []), prefix=prefix):
        try:
            mods.append(import_module(info.name))
        except Exception:  # pragma: no cover - an optional-dependency module
            continue
    for mod in mods:
        for name, obj in vars(mod).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, Experiment)
                and obj is not Experiment
                and obj.__module__.startswith(prefix)
            ):
                found[obj.__name__] = obj
    return found


# -- Guard 2: coverage cannot rot (pure — no qick) -------------------------------------------


def test_every_registered_experiment_is_covered_by_the_contract():
    """Registering an experiment without covering it here must FAIL, not silently pass.

    This is the guard that turns "four broken analyzers" into one loud failure: coverage is
    derived from the registries, so it grows automatically instead of by remembering.
    """
    missing = [n for n in ALL_REGISTERED if n not in CONTRACT_KWARGS]
    assert not missing, (
        f"registered but not covered by the experiment contract: {missing}. Add an entry to "
        "CONTRACT_KWARGS (smallest converging scan) — do not skip it."
    )
    stale = [n for n in CONTRACT_KWARGS if n not in ALL_REGISTERED]
    assert not stale, f"CONTRACT_KWARGS names experiments no registry exposes: {stale}"


@pytest.mark.parametrize("pack", sorted(REGISTRIES))
def test_no_experiment_subclass_escapes_the_registry_or_the_allowlist(pack):
    """The inverse gap: an `Experiment` subclass that is simply never registered would otherwise
    dodge the contract entirely. Each must be registered *or* carry a written reason."""
    registered = set(REGISTRIES[pack].values())
    for cls_name, cls in sorted(_experiment_subclasses(pack).items()):
        if cls in registered:
            continue
        assert cls_name in NOT_REGISTERED, (
            f"{pack}.{cls_name} subclasses Experiment but is neither in EXPERIMENTS nor in "
            "NOT_REGISTERED. Register it (and cover it) or record why it is not "
            "device-constructible."
        )
        assert NOT_REGISTERED[cls_name].strip(), f"{cls_name}'s NOT_REGISTERED reason is empty"


def test_the_allowlist_does_not_excuse_a_registered_experiment():
    """An allowlist entry must never shadow something that IS registered — otherwise the reason
    text would quietly outrank the registry."""
    registered_names = {cls.__name__ for reg in REGISTRIES.values() for cls in reg.values()}
    overlap = sorted(set(NOT_REGISTERED) & registered_names)
    assert not overlap, f"NOT_REGISTERED excuses experiments that are registered: {overlap}"


@pytest.mark.parametrize("pack", ["cqed", "spin"])
def test_every_registry_key_names_its_module(pack):
    """A registry key is the name users call an experiment by, so its module carries that name:
    ``EXPERIMENTS["transmon_rb"]`` lives in ``experiments/transmon_rb.py``."""
    mismatched = sorted(
        (key, cls.__module__)
        for key, cls in REGISTRIES[pack].items()
        if cls.__module__.rsplit(".", 1)[1] != key
    )
    assert not mismatched, f"{pack} registry keys whose module has another name: {mismatched}"


def test_every_cqed_experiment_has_a_contract_device():
    """The loopback and expansion groups partition the cqed registry: a newly registered
    experiment in neither would otherwise fall through to loopback_demo by default."""
    overlap = sorted(_CQED_LOOPBACK_NAMES & _CQED_EXPANSION_NAMES)
    assert not overlap, f"cqed experiments in both device groups: {overlap}"
    unplaced = sorted(set(CQED_EXPERIMENTS) - _CQED_LOOPBACK_NAMES - _CQED_EXPANSION_NAMES)
    assert not unplaced, (
        f"cqed experiments in neither device group: {unplaced}. Add each to _CQED_LOOPBACK_NAMES "
        "or _CQED_EXPANSION_NAMES."
    )
    stale = sorted((_CQED_LOOPBACK_NAMES | _CQED_EXPANSION_NAMES) - set(CQED_EXPERIMENTS))
    assert not stale, f"device groups name experiments the cqed registry lacks: {stale}"


def test_blob_anchored_set_matches_the_analyzers_that_check():
    """BLOB_ANCHORED_ANALYZERS names exactly the cqed experiments whose module calls
    ``is_population_anchored`` — the check an analyzer makes before refusing raw IQ."""
    checking = {
        name for name, cls in CQED_EXPERIMENTS.items()
        if "is_population_anchored(" in inspect.getsource(sys.modules[cls.__module__])
    }
    assert set(BLOB_ANCHORED_ANALYZERS) == checking, (
        f"missing: {sorted(checking - set(BLOB_ANCHORED_ANALYZERS))}, "
        f"stale: {sorted(set(BLOB_ANCHORED_ANALYZERS) - checking)}"
    )


# -- #100: the composite-seam entries drive the readout before every Measure (pure) ----------
#
# parity()/active_reset() (packs/cqed/library/composite.py) funnel 11 of the 38 cqed experiments
# that bare-.measure()'d an undriven resonator (issue #96's Problem) — 9 plain-bare-measure files
# plus SwapSpectroscopy through parity(), and ActiveReset's own mid-circuit + verification reads
# through active_reset(). This is a SCOPED, closed-set check for exactly those 11 names — issue
# #96's own AC 5 ("a new structural guard... fails CI if a *future* registered cqed experiment
# measures without ever driving the readout line") is the generic, ALL_REGISTERED-parametrized
# version of this idea, deliberately deferred to land last, once every existing experiment already
# passes it. This guard does not grow with the registry; that one will.

COMPOSITE_READOUT_PROBE_NAMES = (
    "cavity_t1", "fock_cavity_t1", "fock_cavity_t2_ramsey", "cavity_t2_ramsey", "displacement_calibration", "multiphoton_ladder",
    "parity_fringe", "state_prep_fidelity", "storage_t1", "swap_rabi", "wigner_tomography",
    "swap_spectroscopy", "active_reset",
)


@pytest.mark.parametrize("name", COMPOSITE_READOUT_PROBE_NAMES)
def test_composite_seam_experiments_drive_the_readout_before_every_measure(name):
    """Every ``Measure`` these 11 compile must be immediately preceded by a readout-driving
    ``Pulse`` (``dev.readout.dispersive_measure()``'s probe, labelled ``"ro_probe"``) — never a
    bare ``.measure()`` that triggered the ADC with nothing playing on the readout line."""
    from strumento.core.pulses import Measure, Pulse

    cls = CQED_EXPERIMENTS[name]
    dev = _multimode_device() if name in _CQED_EXPANSION_NAMES else _loopback_device()
    exp = cls(dev, **CONTRACT_KWARGS[name])
    seq, _sweeps = exp.sequence()

    measure_indices = [i for i, op in enumerate(seq.ops) if isinstance(op, Measure)]
    assert measure_indices, f"{name}: sequence() produced no Measure at all"
    for i in measure_indices:
        assert i > 0, f"{name}: a Measure at index 0 has no preceding op"
        prev = seq.ops[i - 1]
        assert isinstance(prev, Pulse) and prev.label == "ro_probe", (
            f"{name}: the Measure at index {i} is not preceded by a readout probe pulse "
            f"(preceding op: {type(prev).__name__})"
        )


# -- Discrimination-experiment guard: every read must drive the resonator (#101, part of #96) --
#
# SingleShotReadout / ReadoutState / ValidationCheck / QNDReadout build the g/e discrimination
# blobs (threshold/angle) every other experiment's DispersiveIQ calibration depends on, or
# (QNDReadout) exist specifically to check whether back-to-back dispersive reads agree. A
# threshold calibrated from an undriven resonator, or a QND check with nothing driving the
# resonator on any of its repeats, silently poisons everything downstream — and, like the
# Guard 1/3 defects above, the simulator computing IQ from calibration truth rather than from
# what the compiled sequence actually played would never catch it (#96's own assumption). This
# guard is pure (no qick): it walks `sequence()`'s raw op list directly rather than compiling,
# so it runs in the `pure` lane and needs no acquisition path at all.

#: name -> the number of reads that experiment's own physics requires (its own docstring/AC, not
#: a config knob) — qnd_readout's is read from CONTRACT_KWARGS's own `n_repeats` so the two
#: figures can never quietly drift apart.
_DISCRIMINATION_EXPECTED_READS: dict[str, int] = {
    "single_shot_readout": 2,  # ground, excited
    "readout_state": 2,  # ground, excited
    "validation_check": 3,  # ground, excited, Ramsey fringe
    "qnd_readout": int(CONTRACT_KWARGS["qnd_readout"]["n_repeats"]),
}


def _undriven_measure_indices(seq) -> list[int]:
    """Indices (within `seq.ops`) of `Measure` ops NOT immediately preceded by a driving
    readout-probe pulse — `Pulse.label == "ro_probe"`, the marker `ReadoutOps.probe()` stamps
    (`library/readout.py`), which is exactly what `dispersive_measure()` plays before measuring.
    """
    from strumento.core.pulses import Measure, Pulse

    bad = []
    for i, op in enumerate(seq.ops):
        if isinstance(op, Measure):
            prev = seq.ops[i - 1] if i > 0 else None
            if not (isinstance(prev, Pulse) and prev.label == "ro_probe"):
                bad.append(i)
    return bad


@pytest.mark.parametrize("name", sorted(_DISCRIMINATION_EXPECTED_READS))
def test_discrimination_experiment_drives_the_resonator_before_every_read(name):
    """#101: these four g/e-discrimination experiments never drove the resonator across any of
    their reads — including every repeat in QNDReadout's loop and every one of ValidationCheck's
    three reads, not just the first. Every `Measure` in the raw sequence must be immediately
    preceded by the calibrated probe pulse `dev.readout.dispersive_measure()` plays.
    """
    from strumento.core.pulses import Measure

    dev = _multimode_device()
    cls = next(reg[name] for reg in REGISTRIES.values() if name in reg)
    exp = cls(dev, **CONTRACT_KWARGS[name])
    seq, _sweeps = exp.sequence()

    n_measures = sum(1 for op in seq.ops if isinstance(op, Measure))
    assert n_measures == _DISCRIMINATION_EXPECTED_READS[name], (
        f"{name}: expected {_DISCRIMINATION_EXPECTED_READS[name]} reads, sequence has "
        f"{n_measures} — the fixture no longer matches the physics this guard checks"
    )

    undriven = _undriven_measure_indices(seq)
    assert not undriven, (
        f"{name}: read(s) at op index {undriven} are not preceded by a driven readout probe "
        f"pulse — the resonator would go unread at those reads on a real board"
    )


# -- Issue #103: the "wrong-line-probe" experiments must drive the readout line too ----------
#
# Each of these 7 experiments already builds a swept excitation/spectroscopy pulse and plays it
# on the qubit or coupler line under a local variable literally named `probe` -- reading as
# though it already satisfied dispersive readout. It doesn't: that tone targets the qubit/
# coupler drive, never the resonator, so the subsequent `.measure()` was bare. This guard is
# scoped to exactly these 7 (the sibling wrong-line-probe fixes -- #100/#101/#102 -- and the
# general, registry-wide contract guard -- #105 -- land separately, once every experiment is
# fixed); it is deliberately narrower than the eventual generalized guard.

#: The 7 experiments issue #103 fixes (Prior Art list, verbatim).
WRONG_LINE_PROBE_EXPERIMENTS: tuple[str, ...] = (
    "qubit_spectroscopy", "ef_spectroscopy", "gain_ladder_spectroscopy", "number_splitting",
    "photon_number_distribution", "tls_scan", "sideband_spectroscopy",
)


def _pulses_before_each_measure(seq) -> list[list]:
    """Group `seq.ops`' top-level pulses by the `Measure` op they precede.

    ``groups[i]`` is every pulse played since the previous ``Measure`` (or the start) through
    the i-th ``Measure`` -- exactly what a real board would already have played by the time
    that Measure triggers the ADC. None of these 7 sequences nest a Measure inside a
    Repeat/Branch, so a flat scan over top-level ops is the real path here, not a simplification
    of it (contrast :func:`_logical_reads`, which does need to recurse for the general case).
    """
    from strumento.core.pulses import Measure, Pulse, Segment

    groups: list[list] = []
    current: list = []
    for op in seq.ops:
        if isinstance(op, Measure):
            groups.append(current)
            current = []
        elif isinstance(op, Pulse):
            current.append(op)
        elif isinstance(op, Segment):
            current.extend(op.pulses)
    return groups


@pytest.mark.parametrize("name", WRONG_LINE_PROBE_EXPERIMENTS)
def test_wrong_line_probe_experiments_drive_the_readout_line_before_every_measure(name):
    """#103's Acceptance Criterion: each of the 7 listed experiments drives the readout line
    before every ``Measure`` op in its compiled sequence, IN ADDITION TO its existing swept
    excitation/spectroscopy tone -- the existing tone must survive alongside the new readout
    probe, never be replaced by it (#103's Key Decision).
    """
    cls = next(reg[name] for reg in REGISTRIES.values() if name in reg)
    dev = _multimode_device()
    exp = cls(dev, **CONTRACT_KWARGS[name])
    seq, _sweeps = exp.sequence()

    groups = _pulses_before_each_measure(seq)
    assert groups, f"{name}: sequence() emits no Measure op"
    for i, pulses in enumerate(groups):
        lines = [p.line.line for p in pulses]
        assert "readout" in lines, (
            f"{name}: Measure #{i} is preceded by pulses on {lines!r} -- none targets the "
            "readout line, so dispersive readout never actually happens before this Measure."
        )
        assert len(lines) >= 2, (
            f"{name}: Measure #{i} is preceded only by {lines!r} -- the existing swept "
            "excitation/spectroscopy tone must still be there alongside the new readout probe."
        )


# -- Guard 1: every registered experiment survives the REAL path -----------------------------


@pytest.mark.qick
@pytest.mark.parametrize("name", ALL_REGISTERED)
def test_registered_experiment_survives_the_real_runner_path(name):
    """Construct → `go(SimulatorSoc)` → `analyze()` for real, and require a typed result.

    The assertion is deliberately "a typed FitResult with derived quantities", not "the fit is
    accurate" — accuracy is the fit-recovery gate's job. What this pins is that the analyzer consumes the
    shape the Runner actually produces. That is exactly the claim the old 1-D unit tests only
    appeared to make.
    """
    from strumento.core.simulator import SimulatorSoc

    cls = next(reg[name] for reg in REGISTRIES.values() if name in reg)

    # Dispatch: spin experiments use the spin device + spin soccfg; cqed expansion uses multimode; rest uses loopback.
    if name in _SPIN_EXPERIMENT_NAMES:
        dev = _spin_device()
        truth = spin_truth_from_device(dev, **SPIN_CONTRACT_TRUTH)
        soc = SimulatorSoc.from_snapshot(_SPIN_SOCCFG, truth=truth, seed=CONTRACT_SEED)
    elif name in _CQED_EXPANSION_NAMES:
        dev = _multimode_device()
        truth = truth_from_device(dev, **CQED_EXPANSION_TRUTH)
        soc = SimulatorSoc.from_snapshot(_CQED_SOCCFG, truth=truth, seed=CONTRACT_SEED)
    else:
        dev = _loopback_device()
        truth = truth_from_device(dev, **CONTRACT_PERTURBATION)
        soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth=truth, seed=CONTRACT_SEED)

    kwargs = dict(CONTRACT_KWARGS[name])
    if _needs_blob_anchor(name, kwargs):
        kwargs["readout"] = _sim_blob_anchored_readout(cls, dev, soc, kwargs)
    exp = cls(dev, **kwargs)
    outcome = exp.go(soc, reps=CONTRACT_REPS, progress=False)

    assert outcome.fit is not None, f"{name}: analyze() returned no fit through the real path"
    assert isinstance(outcome.fit, FitResult), (
        f"{name}: analyze() returned {type(outcome.fit).__name__}, not a typed FitResult"
    )
    assert outcome.fit.success, f"{name}: fit did not converge through the real path"
    assert outcome.fit.derived, f"{name}: fit produced no derived quantities"


@pytest.mark.qick
def test_an_unmodelled_experiment_fails_typed_not_silently():
    """An experiment with no registered response must raise the **typed** error naming the class —
    never return zeros, and never be papered over with a bare skip in the guard above."""
    from strumento.core.pulses import Seq
    from strumento.core.simulator import SimulatorSoc

    class Unmodelled(Experiment):
        """A perfectly runnable sequence with no registered response model."""

        def sequence(self):
            return Seq().play(self.device.qubit.ge_pi()).measure(), ()

        def analyze(self, result):  # pragma: no cover - acquire must fail before this
            raise AssertionError("analyze() should be unreachable: acquire must fail first")

    dev = _loopback_device()
    soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth={}, seed=CONTRACT_SEED)
    with pytest.raises(ResponseError, match="Unmodelled"):
        Unmodelled(dev).go(soc, reps=8, progress=False)


@pytest.mark.qick
def test_drag_real_path_is_blob_anchored_and_lands_on_truth():
    """Non-vacuity proof for the DRAG entry above: through the real path, the bare readout is
    REFUSED, and the blob-anchored one recovers the simulator's optimum (sign and magnitude).

    The pre-2026-10-07 guard passed on alpha_opt = -0.50 against truth +0.50. If the anchored case
    ever drifts off truth again, the guard's ``success`` assertion alone would not notice.
    """
    from strumento.core.simulator import SimulatorSoc

    cls = CQED_EXPERIMENTS["drag_calibration"]
    dev = _multimode_device()
    truth = truth_from_device(dev, **CQED_EXPANSION_TRUTH)
    kwargs = dict(CONTRACT_KWARGS["drag_calibration"])

    soc = SimulatorSoc.from_snapshot(_CQED_SOCCFG, truth=truth, seed=CONTRACT_SEED)
    bare = cls(dev, **kwargs).go(soc, reps=CONTRACT_REPS, progress=False).fit
    assert not bare.success and "alpha_opt" not in bare.derived

    soc = SimulatorSoc.from_snapshot(_CQED_SOCCFG, truth=truth, seed=CONTRACT_SEED)
    kwargs["readout"] = _sim_blob_anchored_readout(cls, dev, soc, kwargs)
    fit = cls(dev, **kwargs).go(soc, reps=CONTRACT_REPS, progress=False).fit
    assert fit.success
    assert fit.derived["alpha_opt"] == pytest.approx(truth["drag_alpha_opt"], abs=0.01)


@pytest.mark.qick
@pytest.mark.parametrize("offset", [0.0, 0.03])
def test_drag_pair_real_path_cancels_a_common_equator_offset(offset):
    """End-to-end on the real path: a +0.03 common P_e offset (in the model it moves the
    X/2·Y/2 crossing by ~−0.044) leaves the pair's alpha_opt on truth and is reported as the offset."""
    from strumento.core.simulator import SimulatorSoc

    cls = CQED_EXPERIMENTS["drag_pair_calibration"]
    dev = _multimode_device()
    truth = truth_from_device(dev, **CQED_EXPANSION_TRUTH, drag_equator_offset=offset)
    kwargs = dict(CONTRACT_KWARGS["drag_pair_calibration"], points=41)
    soc = SimulatorSoc.from_snapshot(_CQED_SOCCFG, truth=truth, seed=CONTRACT_SEED)
    kwargs["readout"] = _sim_blob_anchored_readout(cls, dev, soc, kwargs)
    fit = cls(dev, **kwargs).go(soc, reps=CONTRACT_REPS, progress=False).fit
    assert fit.success
    assert fit.derived["alpha_opt"] == pytest.approx(truth["drag_alpha_opt"], abs=0.01)
    assert fit.derived["equator_offset"] == pytest.approx(offset, abs=0.01)


# -- Guard 5: the two socs agree on shape ----------------------------------------------------


@pytest.mark.qick
def test_mock_and_simulator_return_identically_shaped_buffers():
    """Pins the `(n_reads, n_points)` convention as a *tested contract* rather than folklore.

    This is what makes a unit test that passes against one soc meaningful for the other. Values
    differ by design — the mock's zero-IQ contract is deliberate — so only shapes are compared.
    """
    from strumento.core.mock import MockQickSocV2
    from strumento.core.simulator import SimulatorSoc

    dev = _loopback_device()
    truth = truth_from_device(dev, **CONTRACT_PERTURBATION)
    kwargs = CONTRACT_KWARGS["amplitude_rabi"]

    mock_out = AmplitudeRabiFactory(dev, kwargs).go(
        MockQickSocV2.from_snapshot(V2_SOCCFG), reps=8, progress=False, analyze=False
    )
    sim_out = AmplitudeRabiFactory(dev, kwargs).go(
        SimulatorSoc.from_snapshot(V2_SOCCFG, truth=truth, seed=CONTRACT_SEED),
        reps=8, progress=False, analyze=False,
    )

    m, s = mock_out.result, sim_out.result
    assert m.avgi.shape == s.avgi.shape, (
        f"mock emits avgi{m.avgi.shape} but the simulator emits avgi{s.avgi.shape} — a unit test "
        "passing on one path would say nothing about the other"
    )
    assert m.avgq.shape == s.avgq.shape
    assert len(m.xpts) == len(s.xpts)


def AmplitudeRabiFactory(dev, kwargs):
    """Constructed per-call so the two socs never share experiment state."""
    return CQED_EXPERIMENTS["amplitude_rabi"](dev, **kwargs)


# -- Guard 3: assembly and acquire must be diagnostic-clean ----------------------------------
#
# HOW THIS DEFECT HID, precisely — and why the obvious guard would NOT have worked.
#
# qick emits its timing complaints through **`logging`**, not `warnings`:
#     asm_v2.py:1016  logger.warning("Readout time %d appears to conflict with previous readout
#                                     ending at %f?")
#     asm_v2.py:934   logger.warn("warning: pulse time %s appears to conflict ...")
# So `pytest.warns` / `filterwarnings` / `-W error` are all blind to it — a suite could be
# scrupulous about Python warnings and still never see a real timing conflict. (This tree had
# neither: before this file there was no `pytest.warns`, `recwarn`, `filterwarnings` or
# `simplefilter` anywhere in `tests/`.) The guard therefore watches **both** channels.
#
# The original offender was back-to-back `Measure`s: the cursor advanced past pulses but NOT past
# a readout window, so a second measure landed inside the first one's window, and `active_reset`
# worked around it by hand-computing the window and waiting it out. That is FIXED — `Compiler.play`
# now advances past the window (`delay_auto(ros=True)`) and both workarounds are gone. The guard
# stays, because the class of bug it catches is not specific to that one shape: a timing conflict
# is a correctness bug qick reports through `logging` instead of raising, so any future one lands
# in a *passing* run. `test_the_diagnostics_guard_actually_catches_a_readout_conflict` keeps this
# honest by building an overlap through the raw escape hatch — a proof that survives the fix
# rather than depending on the bug.

# The guard itself — `_diagnostics_are_errors`, `WATCHED_LOGGERS`, `ENVIRONMENT_NOTICES` — lives
# in `tests/_diagnostics.py` so other suites (e.g. the OC-override placement proof, #210) watch
# the SAME channels with the SAME allow-list rather than a drifting copy.
from tests._diagnostics import (  # noqa: E402,F401 — re-exported: the guard's public names
    ENVIRONMENT_NOTICES,
    WATCHED_LOGGERS,
    _diagnostics_are_errors,
)


@pytest.mark.qick
@pytest.mark.parametrize("name", ALL_REGISTERED)
def test_registered_experiment_assembles_and_acquires_without_diagnostics(name):
    """No qick/strumento WARNING+ log record and no Python warning during the real path.

    A timing conflict is a correctness bug that qick reports rather than raises; without this
    guard it lands in a *passing* test run and only surfaces on a board.
    """
    from strumento.core.simulator import SimulatorSoc

    cls = next(reg[name] for reg in REGISTRIES.values() if name in reg)

    if name in _SPIN_EXPERIMENT_NAMES:
        dev = _spin_device()
        truth = spin_truth_from_device(dev, **SPIN_CONTRACT_TRUTH)
        soccfg = _SPIN_SOCCFG
    elif name in _CQED_EXPANSION_NAMES:
        dev = _multimode_device()
        truth = truth_from_device(dev, **CQED_EXPANSION_TRUTH)
        soccfg = _CQED_SOCCFG
    else:
        dev = _loopback_device()
        truth = truth_from_device(dev, **CONTRACT_PERTURBATION)
        soccfg = V2_SOCCFG

    with _diagnostics_are_errors() as (records, _notices, caught):
        soc = SimulatorSoc.from_snapshot(soccfg, truth=truth, seed=CONTRACT_SEED)
        cls(dev, **CONTRACT_KWARGS[name]).go(soc, reps=16, progress=False, analyze=False)

    ours = [str(w.message) for w in caught if "qick" in str(w.filename) or "strumento" in str(w.filename)]
    assert not records, f"{name}: qick/strumento logged warnings during the real path: {records}"
    assert not ours, f"{name}: Python warnings raised during the real path: {ours}"


@pytest.mark.qick
def test_the_fixture_and_library_versions_are_reported():
    """The `ENVIRONMENT_NOTICES` allow-list must not hide the thing it excuses.

    Whether the version-mismatch record fires is a property of the environment, so this asserts the
    *relationship* instead of a fixed outcome: versions equal → the notice must NOT appear (nothing
    is being excused, so the allow-list is inert here); versions differ → it MUST appear. Either way
    a stale fixture stays visible, and the allow-list cannot quietly grow to cover a record that has
    nothing to do with the qick version.

    A difference is not a test failure — offline v2 assembly still works — but it does weaken the
    "assembles exactly as the board would" claim, so the fix is to re-capture the fixture (or pin
    qick), not to widen the list.
    """
    import json

    import qick

    from strumento.core.simulator import SimulatorSoc

    recorded = json.loads(V2_SOCCFG.read_text()).get("sw_version")
    installed = qick.__version__

    with _diagnostics_are_errors() as (records, notices, _caught):
        SimulatorSoc.from_snapshot(V2_SOCCFG, truth={}, seed=CONTRACT_SEED)

    assert not records, f"loading the fixture emitted non-environment diagnostics: {records}"

    mismatched = [n for n in notices if "QICK library version mismatch" in n]
    if recorded == installed:
        assert not mismatched, (
            f"fixture and library agree on qick {installed}, yet a mismatch was logged: {mismatched}"
        )
    else:
        assert mismatched, (
            f"fixture records qick {recorded} but qick {installed} is installed, and no mismatch "
            "record was seen — either qick stopped reporting it (then drop the allow-list entry) or "
            "the fixture's sw_version is not what QickConfig compares against; either way the "
            "allow-list is excusing something it does not describe"
        )


@pytest.mark.qick
def test_the_diagnostics_guard_actually_catches_a_readout_conflict():
    """Proof the guard above is not vacuously green.

    HISTORY, and why this test no longer uses back-to-back `.measure()`. That used to be the
    offender: the cursor advanced past pulses but not past a readout window, so a second measure
    landed inside the first one's window and qick logged "Readout time N appears to conflict with
    previous readout ending at M?" inside a *passing* run. The compiler now advances the cursor
    past the readout window (`compiler.play`, `delay_auto(ros=True)`), so back-to-back measures
    are correctly spaced and no longer conflict — the good branch of the choice the previous
    version of this docstring laid out.

    That fix would have quietly made this proof vacuous, which is the trap. So the pathology is
    now built through the documented raw-qick escape hatch: a real `.measure()` (which declares
    the readout, so qick actually tracks a window to conflict with — a raw trigger alone declares
    nothing and is detected as nothing), then a second trigger placed at a NEGATIVE offset, i.e.
    back inside the window the cursor just cleared. That overlaps *by construction* and cannot be
    un-conflicted by any future timeline improvement, so all this test depends on is that qick
    still reports overlapping readouts through the `qick` logger at WARNING — exactly the channel
    the guard claims to watch.
    """
    from strumento.core.pulses import Seq
    from strumento.core.simulator import SimulatorSoc

    def trigger_back_inside_the_window(prog):
        # the cursor now sits at the end of the window opened by `.measure()`; -0.5 µs is inside it
        prog.trigger(ros=[0], t=-0.5)

    class OverlappingReadouts(Experiment):
        """Deliberately pathological, via the escape hatch rather than via a compiler bug."""

        def sequence(self):
            seq = Seq().play(self.device.qubit.ge_pi()).measure()
            return seq.raw(trigger_back_inside_the_window), ()

        def analyze(self, result):  # pragma: no cover - not exercised
            raise AssertionError("unused")

    dev = _loopback_device()
    with _diagnostics_are_errors() as (records, _notices, _caught):
        soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth={}, seed=CONTRACT_SEED)
        try:
            OverlappingReadouts(dev).go(soc, reps=8, progress=False, analyze=False)
        except Exception:
            # An outright raise is an acceptable outcome too — the point is that the conflict is
            # not *silent*. Only a clean run with no diagnostic would mean the guard is blind.
            pass

    assert any("conflict" in r for r in records), (
        "the diagnostics guard did not observe qick's readout-conflict log record for two "
        f"triggers at the same reference time; records seen: {records}. The guard may be blind — "
        "check whether qick still logs through the 'qick' logger at WARNING."
    )


@pytest.mark.qick
def test_the_cursor_advances_past_a_readout_window():
    """The fix that retired the back-to-back-measure pathology, asserted directly.

    Two measures in a row must be spaced by at least the declared readout window — otherwise the
    second read is acquired inside the first one's window and the data is silently wrong. The
    oracle is qick's own overlap detector rather than an arithmetic check in this test: qick knows
    the realized window ends (it tracks them to emit the conflict record), so asking it is both
    stronger and immune to a second, drifting belief about the window length here.

    The complement matters as much as the spacing: advancing the cursor must not cost a trigger
    macro, because `reads_per_shot` is derived from those (Guard 4) — spaced-but-undeclared reads
    would be just as unretrievable as overlapping ones.
    """
    from strumento.core.program import StrumentoProgram
    from strumento.core.pulses import Seq
    from strumento.core.simulator import SimulatorSoc

    dev = _loopback_device()
    with _diagnostics_are_errors() as (records, _notices, _caught):
        soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth={}, seed=CONTRACT_SEED)
        prog = StrumentoProgram(dev, seq=Seq().play(dev.qubit.ge_pi()).measure().measure(),
                                sweeps=(), reps=8, final_delay=None, soc=soc)

    assert not records, f"two consecutive measures should not conflict; records: {records}"
    declared = [int(ro["trigs"]) for ro in prog.ro_chs.values()]
    assert sum(declared) == 2, f"expected 2 declared reads, got {declared}"


# -- Guard 4: declared reads must equal logically-performed reads -----------------------------
#
# THE TRAP THIS GENERALIZES. `CompiledJob` derives `reads_per_shot` from the compiled program:
#     program.py:73   "reads_per_shot": [int(ro["trigs"]) for ro in prog.ro_chs.values()]
# and qick increments `ro['trigs']` once per trigger **macro at assembly time**. So a `Measure`
# nested inside a `Repeat` is counted ONCE however many times the tProc actually performs it — the
# program under-declares, and the extra reads are simply unretrievable. The active-reset composite
# handles this by unrolling rounds by default, with a typed error on the hardware-loop path; this
# guard turns that one-off precaution into a property every future composite is checked against.


def _logical_reads(seq) -> int:
    """Reads the sequence *performs*, counting `Repeat` bodies `count` times."""
    from strumento.core.pulses import Measure, Repeat, Segment

    total = 0
    for op in seq.ops:
        if isinstance(op, Measure):
            total += 1
        elif isinstance(op, Repeat):
            total += op.count * _logical_reads(op.body)
        elif isinstance(op, Segment):
            total += sum(
                _logical_reads(inner) if hasattr(inner, "ops") else isinstance(inner, Measure)
                for inner in getattr(op, "body", getattr(op, "ops", []))
            )
    return total


@pytest.mark.qick
@pytest.mark.parametrize("name", ALL_REGISTERED)
def test_declared_reads_match_the_reads_the_sequence_performs(name):
    """`reads_per_shot` must account for every read the sequence performs, loops included.

    A mismatch means data the experiment believes it took cannot be retrieved — silent on the
    bench, and only visible as a shape surprise much later.
    """
    from strumento.core.simulator import SimulatorSoc

    cls = next(reg[name] for reg in REGISTRIES.values() if name in reg)

    if name in _SPIN_EXPERIMENT_NAMES:
        dev = _spin_device()
        soccfg = _SPIN_SOCCFG
    elif name in _CQED_EXPANSION_NAMES:
        dev = _multimode_device()
        soccfg = _CQED_SOCCFG
    else:
        dev = _loopback_device()
        soccfg = V2_SOCCFG

    exp = cls(dev, **CONTRACT_KWARGS[name])
    seq, _sweeps = exp.sequence()
    expected = _logical_reads(seq)

    # Build the program exactly as the Runner does (experiment.py: run) — declared axes
    # included (D36). A skip here would be silent non-coverage, which is the failure mode this
    # whole file exists to prevent.
    from strumento.core.program import StrumentoProgram

    soc = SimulatorSoc.from_snapshot(soccfg, truth={}, seed=CONTRACT_SEED)
    prog = StrumentoProgram(dev, seq=seq, sweeps=_sweeps, axes=exp.axes(), reps=16, final_delay=None, soc=soc)

    declared = [int(ro["trigs"]) for ro in prog.ro_chs.values()]
    assert declared, f"{name}: the compiled program declares no readout channels"
    assert max(declared) == expected, (
        f"{name}: declares reads_per_shot={declared} but the sequence performs {expected} reads. "
        "If a Measure sits inside a Repeat, qick counted the trigger macro once at assembly — "
        "unroll the loop or reject it loudly; do not ship an under-declaration."
    )


@pytest.mark.qick
def test_the_reads_guard_actually_catches_a_measure_inside_a_loop():
    """Proof Guard 4 is not vacuously green.

    A hardware-loop reset puts a `Measure` inside a `Repeat`. Either the composite refuses to
    build it (the shipped choice — unroll by default, reject the loop path loudly) or the program
    under-declares `reads_per_shot` and this detects the mismatch. A clean build with matching
    counts would mean the guard has gone blind.
    """
    from strumento.core.program import StrumentoProgram
    from strumento.core.pulses import Seq
    from strumento.core.simulator import SimulatorSoc

    dev = _loopback_device()
    inner = Seq().play(dev.qubit.ge_pi()).measure()
    seq = Seq().repeat(inner, 3)
    assert _logical_reads(seq) == 3, "the fixture itself must perform three reads"

    soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth={}, seed=CONTRACT_SEED)
    try:
        prog = StrumentoProgram(dev, seq=seq, sweeps=(), reps=8, final_delay=None, soc=soc)
    except Exception:
        return  # refused loudly at build — the acceptable outcome

    declared = max(int(ro["trigs"]) for ro in prog.ro_chs.values())
    assert declared != 3, (
        "a Measure inside a Repeat(3) declared 3 reads — qick's assembly-time trigger counting "
        "appears to have changed. If it now counts per iteration, delete this test and the "
        "unroll-by-default workaround in the active-reset composite, and say so."
    )


# -- Guard 6: the resonator family drives the readout before every Measure (issue #102) ------
#
# T1/Rabi/spectroscopy-family experiments used to call bare `.measure()` with nothing driving
# the resonator: on real dispersive readout, the ADC would digitize an undriven line — the
# simulator's response models compute IQ from armed truth directly, so this defect was invisible
# to every fit-recovery gate and only bites on a board. `dev.readout.dispersive_measure()` (#99)
# bundles the probe pulse with the Measure; issue #102 threads it through the 16 files below.
#
# PURE — this guard calls `sequence()` directly (no qick, no compiled program), so it runs in
# both the `pure` and `qick` CI lanes, unlike Guards 1/3/4/5 above.

#: The driven-readout family: the 16 files issue #102 named, plus later EF experiments that
#: share the shelve/ef/reverse-ge_pi shape (eft2_echo). Keyed the same way CONTRACT_KWARGS is.
RESONATOR_FAMILY = [
    "t1", "t2_ramsey", "t2_echo", "amplitude_rabi", "ef_rabi", "eft1", "eft2_ramsey", "eft2_echo",
    "frequency_rabi", "length_rabi", "drag_calibration", "drag_pair_calibration", "ramsey_detuning_pair",
    "transmon_allxy",
    "error_amplification", "transmon_rb", "cross_kerr", "revival_time_calibration",
    "sideband_rabi",
]


def _probes_precede_every_measure(seq) -> bool:
    """True iff every `Measure` op is preceded, since the previous `Measure` (or the start of
    the sequence), by a readout probe pulse — a `Pulse` labeled ``"ro_probe"``, the marker
    `ReadoutOps.probe()` (and therefore `dispersive_measure()`) stamps on every probe it builds.

    Per-read, not per-sequence: DRAGCalibration/TransmonAllXY loop over parameter pairs with one
    read each (issue #102's own callout) — a probe before only the FIRST read would leave every
    subsequent read undriven, so the "seen a probe" flag resets after each `Measure`.
    """
    from strumento.core.pulses import Measure, Pulse

    seen_probe_since_last_measure = False
    saw_any_measure = False
    for op in seq.ops:
        if isinstance(op, Pulse) and op.label == "ro_probe":
            seen_probe_since_last_measure = True
        elif isinstance(op, Measure):
            saw_any_measure = True
            if not seen_probe_since_last_measure:
                return False
            seen_probe_since_last_measure = False
    return saw_any_measure


@pytest.mark.parametrize("name", RESONATOR_FAMILY)
def test_resonator_family_experiment_drives_the_readout_before_every_measure(name):
    """issue #102: each of these 16 used to call bare `.measure()`. Every `Measure` op in the
    compiled sequence must now be preceded by a readout probe pulse — including experiments
    that read more than once per shot (DRAGCalibration/TransmonAllXY loop over parameter pairs,
    one read each; every read needs its own probe, not just the first)."""
    cls = next(reg[name] for reg in REGISTRIES.values() if name in reg)
    dev = _multimode_device() if name in _CQED_EXPANSION_NAMES else _loopback_device()
    exp = cls(dev, **CONTRACT_KWARGS[name])
    seq, _sweeps = exp.sequence()
    assert _probes_precede_every_measure(seq), (
        f"{name}: at least one Measure op is not preceded by a readout probe pulse "
        "(dev.readout.dispersive_measure(), issue #99/#102)"
    )


def test_the_resonator_family_guard_actually_catches_a_bare_measure():
    """Proof Guard 6 is not vacuously green: a bare `.measure()` with nothing driving the
    resonator — the exact pre-#102 shape — must fail the guard above, and the fixed shape
    (a probe played first) must pass it."""
    from strumento.core.pulses import Seq

    dev = _loopback_device()
    bare = Seq().play(dev.qubit.ge_pi()).measure()
    assert not _probes_precede_every_measure(bare)

    driven = (
        Seq().play(dev.qubit.ge_pi())
        .play(dev.readout.probe(line=("qubit", "drive")))
        .measure()
    )
    assert _probes_precede_every_measure(driven)


# -- Guard 7: EVERY registered cqed experiment drives the readout line (issue #96 AC5 / #105) --
#
# Guard 6 above is deliberately SCOPED and FROZEN to the 16 names issue #102 named — its own
# header says so explicitly ("This guard does not grow with the registry; that one will"). This
# is that one: issue #96's own AC5, sequenced last in the campaign specifically so it lands the
# moment every registered cqed experiment it checks already passes it. It is derived from the
# LIVE `CQED_EXPERIMENTS` registry rather than a hand-maintained name tuple, so a *future* 45th
# cqed experiment that reintroduces a bare `.measure()` fails here automatically, without anyone
# remembering to add its name to a list. It reuses `_probes_precede_every_measure` (Guard 6's own
# predicate) rather than re-deriving the same check a second way.
#
# SCOPE: cqed only, not `ALL_REGISTERED`. Issue #96's own "Approaches considered" rejected baking
# this into `core.Measure` specifically because the spin pack's PSB/Elzerman-style readouts "may
# rely on a continuously-running signal rather than a pulse inserted per-measure, or no drive
# tone at all" — its own Scope section puts the spin pack "out of scope by survey" in so many
# words, and its AC3/AC5 both say "cqed" explicitly. Confirmed empirically, not just textually:
# every one of the 21 registered spin experiments plays zero pulses labeled `"ro_probe"` (PSB/
# Elzerman reads are DC/threshold-based — there is no dispersive probe tone to check for), so
# "did a Pulse labeled ro_probe precede this Measure" is not even a meaningful question there,
# and parametrizing this guard over `ALL_REGISTERED` would fail all 21 on sight, unrelated to any
# real defect. `CQED_EXPERIMENTS` currently holds exactly the 44 names issue #105's own AC2 names.
#
# PURE — calls `sequence()` directly (no qick, no compiled program), so it runs in both the
# `pure` and `qick` CI lanes, exactly like Guard 6.

#: `CavitySpectroscopy` is the one cqed experiment that correctly does NOT play an `"ro_probe"`
#: pulse: it drives the target cavity mode's OWN line (`cav_probe`, on that mode's wiring line —
#: see `CavitySpectroscopy.sequence()`) rather than the fixed readout ADC's `("readout",
#: "probe")` line, because that is the actual physics of locating a cavity mode's frequency.
#: Whether that mode's line is what the ADC is actually listening to is a device-wiring question
#: this guard cannot resolve on its own (issue #96's own survey) — so it is exempted here,
#: explicitly, with its reason on record, rather than silently special-cased inside the check.
#: The bar for a second entry is the same one issue #96 already applied once: the experiment
#: must, BY DESIGN, target a line that is not the readout ADC's — never "the fit still
#: converges" and never "CI is red and I'm in a hurry" (Rule 5, module docstring).
READOUT_LINE_EXEMPT: dict[str, str] = {
    "cavity_spectroscopy": (
        "drives the target cavity mode's own line (cav_probe), not the fixed readout ADC's "
        "(\"readout\", \"probe\") line — correct physics for locating a mode's frequency, not a "
        "missed readout drive (issue #96 AC3)."
    ),
}

#: Every registered cqed experiment this guard actually checks — the full registry minus the
#: exemption above. Derived, not hand-maintained: grows and shrinks with `CQED_EXPERIMENTS`.
CQED_READOUT_LINE_CHECKED = [n for n in sorted(CQED_EXPERIMENTS) if n not in READOUT_LINE_EXEMPT]


@pytest.mark.parametrize("name", CQED_READOUT_LINE_CHECKED)
def test_registered_cqed_experiment_drives_the_readout_line_before_every_measure(name):
    """issue #96's AC5 / #105: every `Measure` op a registered cqed experiment compiles must be
    preceded (since the previous `Measure`, or the start of the sequence) by a readout probe
    pulse. Unlike Guard 6 and the #100/#101/#103 guards above, this one is parametrized over the
    LIVE `CQED_EXPERIMENTS` registry (minus the documented exemption above), not a fixed name
    tuple — it is the guard built to catch the *next* regression, not just the ones this
    campaign already found.
    """
    cls = CQED_EXPERIMENTS[name]
    dev = _multimode_device() if name in _CQED_EXPANSION_NAMES else _loopback_device()
    exp = cls(dev, **CONTRACT_KWARGS[name])
    seq, _sweeps = exp.sequence()
    assert _probes_precede_every_measure(seq), (
        f"{name}: at least one Measure op is not preceded by a readout probe pulse "
        "(dev.readout.dispersive_measure() / dev.readout.probe(), issue #96/#105). If this "
        "experiment legitimately targets a different line BY DESIGN (like CavitySpectroscopy), "
        "add it to READOUT_LINE_EXEMPT with a one-line reason — never weaken this assertion."
    )


def test_every_cqed_experiment_is_either_checked_or_explicitly_exempt():
    """Coverage-completeness for THIS guard's own allowlist — mirrors Guard 2's `NOT_REGISTERED`
    discipline (`test_every_registered_experiment_is_covered_by_the_contract` /
    `test_the_allowlist_does_not_excuse_a_registered_experiment` above), applied to
    `READOUT_LINE_EXEMPT`: every registered cqed experiment is either checked above or named here
    with a written reason — there is no third, silent way to escape this guard.
    """
    all_cqed = set(CQED_EXPERIMENTS)
    checked = set(CQED_READOUT_LINE_CHECKED)
    exempt = set(READOUT_LINE_EXEMPT)

    assert checked | exempt == all_cqed, (
        "unaccounted-for cqed experiments (neither checked nor exempt): "
        f"{sorted(all_cqed - checked - exempt)}"
    )
    assert not (checked & exempt), (
        "names both checked and exempt — the allowlist is shadowing a real check: "
        f"{sorted(checked & exempt)}"
    )
    stale = sorted(exempt - all_cqed)
    assert not stale, f"READOUT_LINE_EXEMPT names experiments the cqed registry does not have: {stale}"
    for name, reason in READOUT_LINE_EXEMPT.items():
        assert reason.strip(), f"{name}'s READOUT_LINE_EXEMPT reason is empty"


def test_cavity_spectroscopy_exemption_is_not_decorative():
    """Proves the exemption is doing real work, not documentation theatre: `CavitySpectroscopy`'s
    OWN compiled sequence must actually fail `_probes_precede_every_measure` on its own merits —
    it plays a `cav_probe`-labeled pulse on the mode's own line, never `ro_probe` — so issue
    #105 AC3's "not silently special-cased" holds in both directions: the exemption is written
    down AND it is excusing something real. If a future fix ever made this pass unaided, the
    name belongs in `CQED_READOUT_LINE_CHECKED`, not in `READOUT_LINE_EXEMPT`, and this test
    would fail to say so.
    """
    dev = _multimode_device()
    exp = CQED_EXPERIMENTS["cavity_spectroscopy"](dev, **CONTRACT_KWARGS["cavity_spectroscopy"])
    seq, _sweeps = exp.sequence()
    assert not _probes_precede_every_measure(seq), (
        "cavity_spectroscopy's sequence unexpectedly satisfies the readout-probe check on its "
        "own merits — its READOUT_LINE_EXEMPT entry is now excusing nothing real; drop it and "
        "let CQED_READOUT_LINE_CHECKED cover this experiment instead"
    )


# -- Guard 7: a feed-forward read must wait for a FRESH accumulator (F1) ----------------------
#
# THE TRAP THIS GENERALIZES. A `Branch` lowers to `read_and_jump` = `read_input` + `cond_jump`, and
# `read_input` runs at tProc EXECUTION time, off the timed pulse queue — qick's own docstring: "the
# readout must have already pushed the value into the input, otherwise you will get a stale value."
# A `Delay`/`wait()` in the IR only shifts the REFERENCE time (qick's `Delay` macro; it never stalls
# the processor), so it does nothing to guarantee freshness; only a processor-blocking `wait_auto`
# (qick's `Wait` macro) does. The compiler's `_play_branch` emits that `wait_auto(ros=True)` before
# every `read_and_jump`; this guard turns that into a property every registered feed-forward
# experiment is checked against, at the compiled-macro level — which is the only place it is visible
# (the simulator walks the branch with a perfectly fresh read; the mock returns zeros). It is the
# v2 port of the tProc-v1 idiom of a `wait_all` before every `read`.


def _contains_branch(seq) -> bool:
    """True if the sequence performs a mid-circuit `Branch` (recursing into `Repeat` bodies).

    A top-level `Branch` short-circuits True; one nested only inside a `Repeat` is caught via the
    body. (A `Branch` inside another `Branch`'s arm needs no separate case — the outer one is a
    `Branch`.) Pure: no qick, so the parametrized guard skips the compile for non-feed-forward
    experiments.
    """
    from strumento.core.pulses import Branch, Repeat

    for op in seq.ops:
        if isinstance(op, Branch):
            return True
        if isinstance(op, Repeat) and _contains_branch(op.body):
            return True
    return False


def _read_input_indices(prog) -> list[int]:
    """Positions of `ReadInput` macros — the accumulator read `read_and_jump` expands to."""
    return [i for i, m in enumerate(prog.macro_list) if type(m).__name__ == "ReadInput"]


def _contract_device_and_soccfg(name: str):
    """The (device, soccfg) a registered experiment builds against — mirrors Guard 4's mapping."""
    if name in _SPIN_EXPERIMENT_NAMES:
        return _spin_device(), _SPIN_SOCCFG
    if name in _CQED_EXPANSION_NAMES:
        return _multimode_device(), _CQED_SOCCFG
    return _loopback_device(), V2_SOCCFG


@pytest.mark.qick
@pytest.mark.parametrize("name", ALL_REGISTERED)
def test_feedforward_reads_wait_for_a_fresh_accumulator(name):
    """Every compiled `read_and_jump` must have a blocking `wait_auto` immediately before its read.

    Without it the on-board branch conditions on a STALE ADC accumulator — silent in `--sim`/`--mock`
    (they cannot model feedback timing) and only wrong on a board. Experiments with no feed-forward
    compile no `ReadInput` and pass trivially; non-vacuity is pinned by
    `test_guard_7_is_not_vacuous_a_feedforward_experiment_is_checked` below.
    """
    from strumento.core.program import StrumentoProgram
    from strumento.core.simulator import SimulatorSoc

    dev, soccfg = _contract_device_and_soccfg(name)
    cls = next(reg[name] for reg in REGISTRIES.values() if name in reg)
    exp = cls(dev, **CONTRACT_KWARGS[name])
    seq, _sweeps = exp.sequence()
    if not _contains_branch(seq):
        return  # no feed-forward read to gate — nothing for this guard to check

    soc = SimulatorSoc.from_snapshot(soccfg, truth={}, seed=CONTRACT_SEED)
    prog = StrumentoProgram(dev, seq=seq, sweeps=_sweeps, axes=exp.axes(), reps=16,
                            final_delay=None, soc=soc)
    prog.compile()
    reads = _read_input_indices(prog)
    assert reads, f"{name}: sequence contains a Branch but compiled no ReadInput (read_and_jump)"
    names = [type(m).__name__ for m in prog.macro_list]
    for i in reads:
        prev = prog.macro_list[i - 1]
        assert names[i - 1] == "Wait" and prev.auto, (
            f"{name}: the read_and_jump at macro {i} is not immediately preceded by a blocking "
            f"wait_auto (got {names[i - 1]!r}) — on hardware read_input would return a stale ADC "
            "accumulator (F1). The Branch lowering must emit wait_auto(ros=True) before the read."
        )


@pytest.mark.qick
def test_guard_7_is_not_vacuous_a_feedforward_experiment_is_checked():
    """Proof Guard 7 is not passing over an empty set: at least one registered experiment actually
    compiles a feed-forward read (``active_reset`` does), so the parametrized guard has real work."""
    feedforward = []
    for name in ALL_REGISTERED:
        dev, _ = _contract_device_and_soccfg(name)
        cls = next(reg[name] for reg in REGISTRIES.values() if name in reg)
        if _contains_branch(cls(dev, **CONTRACT_KWARGS[name]).sequence()[0]):
            feedforward.append(name)
    assert feedforward, "no registered experiment contains a Branch — Guard 7 would be vacuous"
    assert "active_reset" in feedforward, feedforward
