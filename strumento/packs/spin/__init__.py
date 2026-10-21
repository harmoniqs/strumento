"""spin — quantum-dot modality pack (spec S15.5) — spinQICK-modeled.

Gate-defined quantum dots (`Dot` / `Plunger` / `Barrier` / `Reservoir` / `ChargeSensor` /
`DriveLine` / `GateSensor`), Loss-DiVincenzo, singlet-triplet, exchange-only, and hole-spin
encodings, and PSB / Elzerman / dispersive gate readout.

Element types, calibration schema, and readout kinds — wired through the standard pack
contract (``CALIBRATION`` + ``compose_elements``) so ``core.device.load`` composes it with
no core change.

Importing the pack also **registers** its simulated response models (``responses.py``) into the
core response registry (D32), as the cqed pack does — that import stays pure (no ``qick``).

Device instances: ``devices/spin_demo``, ``devices/spin_eo_demo``, ``devices/spin_st_demo``,
``devices/spin_hole_demo``.

The experiment suite covers five layers of quantum-dot characterization:

- **Electrostatic bring-up** (7 DCSource experiments): PinchOff, ChargeStability,
  CoulombDiamond, SensorTune, VirtualGateCalibration, TunnelCouplingTune,
  DispersiveGateCalibration — host-driven DC sweeps with a probe callable.
- **Single-qubit control** (11 registered): SpinT1, SpinRabi, SpinRamsey, HahnEcho,
  CPMG, EDSRSpectroscopy, AllXY, CompositePulseCalibration, and three tune-up
  experiments (GradientCalibration, GFactorMap, SweetSpotTune).
- **Two-qubit / exchange** (2 registered): ExchangeOscillation, ExchangeGateCalibration.
- **Benchmarking** (3 registered): RandomizedBenchmarking, InterleavedRB, LeakageBenchmark.
- **Readout calibration** (3 registered): ElzermanOptimize, PSBCalibration, HeraldedInit.
- **Readout analysis** (1 standalone): TunnelRateTune.

21 experiments registered in ``EXPERIMENTS``; all pass the full experiment-contract guards
(Device.load → experiment.go(SimulatorSoc) → compile → acquire → fit). The 7 DCSource
experiments are standalone classes (not ``Experiment`` subclasses) with their own ``run()`` +
``analyze()`` lifecycle. ``RFReflectometryTune`` and ``PSBHistogram`` are additional standalone
classes (not ``Experiment`` subclasses): the first is a frequency-sweep RF
characterization, the second a pure shot-data analyzer. The bring-up slice (#57) promotes
the electrostatic utilities into REGISTERED host-bridge adapters — see
``packs/spin/bringup.py`` / ``host_bridge.py`` — so the six-step graph (and any graph step)
drives them through the sanctioned host-bridge path; the utility classes themselves remain
standalone, the adapters are the registered contract claims.

Pulse libraries: ``library/ld.py`` (EDSR), ``library/exchange.py`` (baseband exchange),
``library/hole_spin.py`` (AC gate drive), ``library/singlet_triplet.py`` (S-T0 detuning),
``library/composite.py`` (BB1/CORPSE/Knill composites, active reset, OC import),
``library/eo_clifford.py`` (exchange-only Clifford decomposition for RB).
"""

from strumento.packs.spin.calibration import SpinCalibration
from strumento.packs.spin.elements import (
    Barrier,
    ChargeSensor,
    Dot,
    DriveLine,
    EODot,
    GateSensor,
    HoleDot,
    Plunger,
    Reservoir,
    STDot,
)
from strumento.packs.spin.experiments.allxy import AllXY
from strumento.packs.spin.experiments.charge_noise_spectroscopy import ChargeNoiseSpectroscopy
from strumento.packs.spin.experiments.composite_pulse_calibration import CompositePulseCalibration
from strumento.packs.spin.experiments.cpmg import CPMG
from strumento.packs.spin.experiments.edsr_spectroscopy import EDSRSpectroscopy
from strumento.packs.spin.experiments.elzerman_optimize import ElzermanOptimize
from strumento.packs.spin.experiments.exchange_gate_calibration import ExchangeGateCalibration
from strumento.packs.spin.experiments.exchange_oscillation import ExchangeOscillation
from strumento.packs.spin.experiments.g_factor_map import GFactorMap
from strumento.packs.spin.experiments.gradient_calibration import GradientCalibration
from strumento.packs.spin.experiments.hahn_echo import HahnEcho
from strumento.packs.spin.experiments.heralded_init import HeraldedInit
from strumento.packs.spin.experiments.interleaved_rb import InterleavedRB
from strumento.packs.spin.experiments.leakage_benchmark import LeakageBenchmark
from strumento.packs.spin.experiments.psb_calibration import PSBCalibration
from strumento.packs.spin.experiments.randomized_benchmarking import RandomizedBenchmarking
from strumento.packs.spin.experiments.spin_rabi import SpinRabi
from strumento.packs.spin.experiments.spin_ramsey import SpinRamsey
from strumento.packs.spin.experiments.spin_t1 import SpinT1
from strumento.packs.spin.experiments.sweet_spot_tune import SweetSpotTune
from strumento.packs.spin.experiments.tunnel_rate_tune import TunnelRateTune

#: the pack's calibration schema (consumed by ``core.device.load``)
CALIBRATION = SpinCalibration

#: The pack's runnable experiments by name (D33).
#:
#: A name is the stable, host-facing handle: an agent driving a session says
#: ``"spin_rabi"`` and never imports a class. Each value is an Experiment subclass.
#:
#: Registered experiments have passed the full-loop contract: Device.load →
#: experiment.go(SimulatorSoc) → compile → acquire → fit succeeds. Unregistered
#: experiments are listed in NOT_REGISTERED with written reasons.
EXPERIMENTS: dict[str, type] = {
    "spin_t1": SpinT1,
    "spin_rabi": SpinRabi,
    "spin_ramsey": SpinRamsey,
    "hahn_echo": HahnEcho,
    "cpmg": CPMG,
    "exchange_oscillation": ExchangeOscillation,
    "edsr_spectroscopy": EDSRSpectroscopy,
    "exchange_gate_calibration": ExchangeGateCalibration,
    "gradient_calibration": GradientCalibration,
    "g_factor_map": GFactorMap,
    "sweet_spot_tune": SweetSpotTune,
    "charge_noise_spectroscopy": ChargeNoiseSpectroscopy,
    "allxy": AllXY,
    "randomized_benchmarking": RandomizedBenchmarking,
    "interleaved_rb": InterleavedRB,
    "leakage_benchmark": LeakageBenchmark,
    "composite_pulse_calibration": CompositePulseCalibration,
    "heralded_init": HeraldedInit,
    "tunnel_rate_tune": TunnelRateTune,
    "elzerman_optimize": ElzermanOptimize,
    "psb_calibration": PSBCalibration,
}


def compose_elements(topology: list[dict], calib: SpinCalibration) -> dict[str, object]:
    """Build ``{instance-name: element}`` from a spin instance's topology."""
    _DOT_BY_ENCODING = {
        "ld": Dot,
        "st": STDot,
        "eo": EODot,
        "hole": HoleDot,
    }
    out: dict[str, object] = {}
    for entry in topology:
        kind, name = entry["type"], entry["name"]
        if kind == "Dot":
            encoding = entry.get("encoding", "ld")
            dot_cls = _DOT_BY_ENCODING.get(encoding)
            if dot_cls is None:
                raise KeyError(f"unknown dot encoding {encoding!r}")
            out[name] = dot_cls(name=name, encoding=encoding)
        elif kind == "Plunger":
            out[name] = Plunger(name=name)
        elif kind == "Barrier":
            out[name] = Barrier(name=name)
        elif kind == "Reservoir":
            out[name] = Reservoir(name=name)
        elif kind == "ChargeSensor":
            out[name] = ChargeSensor(
                name=name, sensor_type=entry.get("sensor_type", "qpc")
            )
        elif kind == "DriveLine":
            out[name] = DriveLine(name=name)
        elif kind == "GateSensor":
            out[name] = GateSensor(name=name)
        else:
            raise KeyError(f"unknown spin element type {kind!r}")
    return out


# Register display functions for all experiments (no matplotlib pulled in — just function refs),
# and the pack's simulated response models (D32) into the core response registry, so loading a
# spin device is all it takes for ``SimulatorSoc`` to run this pack's experiments offline. Both
# imports stay pure — no ``qick``. They sit at the end of the init (as cqed's plot import does)
# because ``responses`` imports the experiment modules above.
import strumento.packs.spin.plot  # noqa: F401, E402
from strumento.packs.spin import responses as _responses  # noqa: F401, E402
