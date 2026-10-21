"""cqed — superconducting (transmon + bosonic) modality pack.

Element types (Transmon / BosonicMode / Coupler / ReadoutResonator), pulse
libraries (transmon / sideband / mode / coupler / composite), the DispersiveIQ
readout kind, a Pydantic calibration schema, and experiments — the original 6
(Rabi / T1 / T2 / parity / resonator_spectroscopy / active_reset) plus 30 new
experiments covering the full multimode cQED bringup (spectroscopy, qubit cal,
sideband / multiphoton, cavity operations, coupler / swaps, and QEC
characterization).

First device instance: the generic "multimode" example (``devices/multimode_demo``).
Pure-transmon, multi-transmon, and transmon+multi-bosonic systems are other
*topologies* of this same pack.

This module is the **pack contract** that ``core.device.load`` consumes after resolving the
pack via the ``strumento.packs`` entry-point group (so ``core`` never imports the pack by
name). A pack exposes:

* ``CALIBRATION`` — the Pydantic calibration schema for the pack.
* ``compose_elements(topology, calib)`` — build the named element instances from an instance's
  topology list + parsed calibration (this is where physics-specific element↔calib wiring
  lives, keeping ``core`` modality-general).
* ``EXPERIMENTS`` — the pack's runnable experiments by name (D33), so a host can *name* an
  experiment instead of importing its class.
* ``CALIBRATION_CEILINGS`` — the dotted calibration paths of the per-element gain limits, which
  the MCP session refuses to write (#190).

Importing the pack also **registers** its simulated response models (``responses.py``) into the
core response registry (D32), so loading a cqed device is all it takes for ``SimulatorSoc`` to be
able to run this pack's experiments offline. That import stays pure — no ``qick``.
"""

from strumento.packs.cqed import couplings as _couplings  # noqa: F401  (registers coupling forms)
from strumento.packs.cqed import responses as _responses  # noqa: F401  (registers response models)
from strumento.packs.cqed.calibration import Calibration, ceiling_paths
from strumento.packs.cqed.elements import (
    BosonicMode,
    Coupler,
    ReadoutResonator,
    Transmon,
)

# --- experiments ---
from strumento.packs.cqed.experiments.active_reset import ActiveReset
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi
from strumento.packs.cqed.experiments.cavity_spectroscopy import CavitySpectroscopy
from strumento.packs.cqed.experiments.cavity_t1 import CavityT1
from strumento.packs.cqed.experiments.cavity_t2_ramsey import CavityT2Ramsey
from strumento.packs.cqed.experiments.cross_kerr import CrossKerrMeasurement
from strumento.packs.cqed.experiments.dispersive_shift import DispersiveShift
from strumento.packs.cqed.experiments.displacement_calibration import (
    DisplacementCalibration,
)
from strumento.packs.cqed.experiments.drag_calibration import DRAGCalibration
from strumento.packs.cqed.experiments.drag_pair_calibration import DRAGPairCalibration
from strumento.packs.cqed.experiments.ef_rabi import EFRabi
from strumento.packs.cqed.experiments.ef_spectroscopy import EFSpectroscopy
from strumento.packs.cqed.experiments.eft1 import EFT1
from strumento.packs.cqed.experiments.eft2_echo import EFT2Echo
from strumento.packs.cqed.experiments.eft2_ramsey import EFT2Ramsey
from strumento.packs.cqed.experiments.error_amplification import ErrorAmplification
from strumento.packs.cqed.experiments.fock_cavity_t1 import FockCavityT1
from strumento.packs.cqed.experiments.fock_cavity_t2_ramsey import FockCavityT2Ramsey
from strumento.packs.cqed.experiments.frequency_rabi import FrequencyRabi
from strumento.packs.cqed.experiments.gain_ladder_spectroscopy import GainLadderSpectroscopy
from strumento.packs.cqed.experiments.length_rabi import LengthRabi
from strumento.packs.cqed.experiments.multiphoton_ladder import MultiphotonLadder
from strumento.packs.cqed.experiments.number_splitting import NumberSplitting
from strumento.packs.cqed.experiments.parity_fringe import ParityFringe
from strumento.packs.cqed.experiments.photon_number_distribution import (
    PhotonNumberDistribution,
)
from strumento.packs.cqed.experiments.power_sweep import PowerSweep
from strumento.packs.cqed.experiments.punchout import Punchout
from strumento.packs.cqed.experiments.qnd_readout import QNDReadout
from strumento.packs.cqed.experiments.qubit_spectroscopy import QubitSpectroscopy
from strumento.packs.cqed.experiments.ramsey_detuning_pair import RamseyDetuningPair
from strumento.packs.cqed.experiments.readout_optimization import ReadoutOptimization
from strumento.packs.cqed.experiments.readout_state import ReadoutState
from strumento.packs.cqed.experiments.resonator_power_sweep import ResonatorPowerSweep
from strumento.packs.cqed.experiments.resonator_spectroscopy import ResonatorSpectroscopy
from strumento.packs.cqed.experiments.revival_time_calibration import (
    RevivalTimeCalibration,
)
from strumento.packs.cqed.experiments.sideband_rabi import SidebandRabi
from strumento.packs.cqed.experiments.sideband_spectroscopy import (
    SidebandSpectroscopy,
)
from strumento.packs.cqed.experiments.single_shot_readout import SingleShotReadout
from strumento.packs.cqed.experiments.state_prep_fidelity import StatePrepFidelity
from strumento.packs.cqed.experiments.storage_t1 import StorageT1
from strumento.packs.cqed.experiments.swap_rabi import SwapRabi
from strumento.packs.cqed.experiments.swap_spectroscopy import SwapSpectroscopy
from strumento.packs.cqed.experiments.t1 import T1
from strumento.packs.cqed.experiments.t2_echo import T2Echo
from strumento.packs.cqed.experiments.t2_ramsey import T2Ramsey
from strumento.packs.cqed.experiments.tls_scan import TlsScan
from strumento.packs.cqed.experiments.transmon_allxy import TransmonAllXY
from strumento.packs.cqed.experiments.transmon_rb import TransmonRB
from strumento.packs.cqed.experiments.validation_check import ValidationCheck
from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography
from strumento.packs.cqed.library.sideband import SidebandOps

#: The pack's calibration schema (consumed by ``core.device.load``).
CALIBRATION = Calibration

#: The calibration paths that are gain ceilings (#190) — ``*`` stands for any dict key. Derived
#: from the fields' ceiling marks, so a new ceiling is declared as soon as it is marked. The MCP
#: session refuses a write-back that touches one: an agent cannot raise its own limit.
CALIBRATION_CEILINGS: tuple[str, ...] = ceiling_paths(Calibration)

#: The pack's runnable experiments, by name (D33 — the pack **experiment registry**).
#:
#: A name is the stable, host-facing handle: an agent driving the MCP session (or any harness that
#: reads an experiment name off a plan) says ``"amplitude_rabi"`` and never imports a class. Each
#: value is an :class:`~strumento.core.experiment.Experiment` subclass whose constructor takes the
#: device positionally and everything else as keyword arguments — so ``run_experiment``'s ``params``
#: map straight onto the signature (see ``docs/packs-and-devices.md``).
#:
#: The three host-batched diagnostics in ``experiments/`` (``kerr_cat_witness``,
#: ``ladder_population``, ``self_kerr_ramsey``) are not here on purpose: each needs several
#: independent programs, which one ``Experiment`` cannot be — see the experiments package docstring.
EXPERIMENTS: dict[str, type] = {
    "amplitude_rabi": AmplitudeRabi,
    "t1": T1,
    "t2_ramsey": T2Ramsey,
    "parity_fringe": ParityFringe,
    "resonator_spectroscopy": ResonatorSpectroscopy,
    "resonator_power_sweep": ResonatorPowerSweep,
    "active_reset": ActiveReset,
    "qubit_spectroscopy": QubitSpectroscopy,
    "ef_spectroscopy": EFSpectroscopy,
    "cavity_spectroscopy": CavitySpectroscopy,
    "dispersive_shift": DispersiveShift,
    "power_sweep": PowerSweep,
    "number_splitting": NumberSplitting,
    "length_rabi": LengthRabi,
    "t2_echo": T2Echo,
    "drag_calibration": DRAGCalibration,
    "drag_pair_calibration": DRAGPairCalibration,
    "ramsey_detuning_pair": RamseyDetuningPair,
    "transmon_allxy": TransmonAllXY,
    "ef_rabi": EFRabi,
    "eft1": EFT1,
    "eft2_ramsey": EFT2Ramsey,
    "eft2_echo": EFT2Echo,
    "fock_cavity_t1": FockCavityT1,
    "fock_cavity_t2_ramsey": FockCavityT2Ramsey,
    "frequency_rabi": FrequencyRabi,
    "single_shot_readout": SingleShotReadout,
    "readout_optimization": ReadoutOptimization,
    "sideband_spectroscopy": SidebandSpectroscopy,
    "sideband_rabi": SidebandRabi,
    "multiphoton_ladder": MultiphotonLadder,
    "displacement_calibration": DisplacementCalibration,
    "cavity_t1": CavityT1,
    "cavity_t2_ramsey": CavityT2Ramsey,
    "revival_time_calibration": RevivalTimeCalibration,
    "photon_number_distribution": PhotonNumberDistribution,
    "wigner_tomography": WignerTomography,
    "swap_spectroscopy": SwapSpectroscopy,
    "swap_rabi": SwapRabi,
    "storage_t1": StorageT1,
    "cross_kerr": CrossKerrMeasurement,
    "qnd_readout": QNDReadout,
    "state_prep_fidelity": StatePrepFidelity,
    "transmon_rb": TransmonRB,
    # the bring-up graph's nodes (issue #56 — thin Seq experiments with vetted fitters)
    "punchout": Punchout,
    "gain_ladder_spectroscopy": GainLadderSpectroscopy,
    "readout_state": ReadoutState,
    "tls_scan": TlsScan,
    "validation_check": ValidationCheck,
    # sub-linewidth pulse fine-tuning by repeated-pulse error amplification (issue #82)
    "error_amplification": ErrorAmplification,
}

#: Ops classes an instance may bind under an alias name via its ``aliases:`` section (D21):
#: e.g. ``aliases: {sideband: {element: qubit, ops: sideband}}`` -> ``dev.sideband``.
ALIAS_OPS = {"sideband": SidebandOps}


def compose_elements(topology: list[dict], calib: Calibration) -> dict[str, object]:
    """Build ``{instance-name: element}`` from a device's topology + parsed calibration.

    Mapping: the single transmon binds to ``calib.qubit`` (the readout/coupler/modes carry no
    per-element calib section; a mode takes its role, line and frequency from the topology
    entry). Multi-qubit composition (per-element calib sections) is not implemented yet.
    """
    out: dict[str, object] = {}
    for entry in topology:
        kind, name = entry["type"], entry["name"]
        if kind == "Transmon":
            qc = calib.qubit
            out[name] = Transmon(name=name, f_ge=qc.f_ge, f_ef=qc.f_ef, pulses=qc.pulses)
        elif kind == "BosonicMode":
            out[name] = BosonicMode(
                name=name, role=entry.get("role", "storage"), line=entry.get("line"),
                freq_mhz=entry.get("freq_mhz"),
            )
        elif kind == "Coupler":
            out[name] = Coupler(name=name)
        elif kind == "ReadoutResonator":
            out[name] = ReadoutResonator(name=name)
        else:
            raise KeyError(f"unknown cqed element type {kind!r}")
    return out


# Register display functions for all experiments (no matplotlib pulled in — just function refs).
import strumento.packs.cqed.plot  # noqa: F401, E402
