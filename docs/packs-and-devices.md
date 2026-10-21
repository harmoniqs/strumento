# Packs & devices — how elements compose

Strumento has three layers. This doc explains how they fit together, how a **device** composes from
**element types**, and how to add a new **modality pack** — with the shipped packs as worked examples.

```
core        modality-agnostic mechanism (no physics)
  packs     one reusable kit per physics class (element TYPES + libraries + readout kind + calib schema)
    devices instances: one per physical fridge (topology + wiring + calibration VALUES)
```

`core` never imports a pack by name. A pack is discovered at load time through the
`strumento.packs` entry-point group, so adding superconducting / neutral-atom / spin support — or a
third-party modality — is a new directory, **not** a core change. Three packs ship today (`cqed`
and `spin` built; `tweezer` as a skeleton), and `core` composes an instance of each unchanged — the
honesty check that `core` isn't shaped like any one modality.

## Element composition today

A **device instance** is a YAML file naming elements, wiring them to DACs, and giving calibration
values. `Device.load` turns it into a live `Device`:

```yaml
pack: cqed
wiring:
  lines:
    qubit:
      name: qubit
      generators:
        - {role: drive, dac_ch: 1, nyquist: 1}
        - {role: sideband, dac_ch: 5, nyquist: 1}
  readout: {dac_ch: 0, adc_chs: [0]}
topology:
  - {type: Transmon, name: qubit}
  - {type: BosonicMode, name: cavity, role: cavity, line: cavity}   # role: cavity = the default mode
  - {type: Coupler, name: coupler}
  - {type: ReadoutResonator, name: readout}
calibration:
  qubit: {f_ge: 4000.0, pulses: {pi_ge: {gain: 8192, sigma: 0.030}}}
  readout: {frequency: 749.43, gain: 800, readout_length: 1.5, trig_offset: 0.5, relax_delay: 2500}
```

Loading walks this in four steps:

1. **Resolve the pack** (`pack: cqed`) via the entry-point group → the pack module.
2. **Validate calibration** against the pack's `CALIBRATION` Pydantic schema.
3. **Compose elements** — the pack's `compose_elements(topology, calib)` builds one element object
   per `topology` entry.
4. **Bind namespaces + validate wiring** — for each named element with a pulse library, bind
   `dev.<name>`; check every element's declared lines exist in the wiring.

The binding is **topology-derived (D21)**: `dev.qubit` exists because *this instance* names its
transmon `qubit`. Nothing in the pack hardcodes "qubit" — a different YAML yields `dev.q0` /
`dev.c1`. Each element type declares two things:

```python
@dataclass
class Transmon:
    name: str
    f_ge: float
    ...
    ops_cls: ClassVar[type | None] = TransmonOps   # the ops bound to dev.<name> (None until a library exists)
    def lines(self) -> list[LineRef]:              # the LineRefs it drives, using the instance name
        return [LineRef(self.name, "drive")]
```

`lines()` is what `resolve` and load-time validation use to map an element onto generators;
`ops_cls` is the pulse library its named instances bind to. An element with `ops_cls = None` is a
topology node with no callable ops yet (every element in the `tweezer` skeleton, and the `spin`
pack's passive nodes such as `Plunger` and `Reservoir`).

## Anatomy of a pack

A pack is a Python package exposing a small contract that `core.device.load` consumes:

| Provides | What | Example |
|---|---|---|
| `CALIBRATION` | Pydantic calibration schema for the modality | `cqed.Calibration`, `TweezerCalibration`, `SpinCalibration` |
| `compose_elements(topology, calib)` | build `{name: element}` from a topology list | maps `{"type": "Transmon", ...}` → `Transmon(...)` |
| `EXPERIMENTS` | `{name: Experiment subclass}` — the experiments a host can run **by name** (D33) | `{"amplitude_rabi": AmplitudeRabi, "t1": T1, …}` |
| (optional) `CALIBRATION_CEILINGS` | dotted calibration paths (`*` = any key) the MCP session refuses to write | cqed: `("qubit.max_drive_gain", …, "modes.*.max_displace_gain")` |
| element **types** | dataclasses with `lines()` + `ops_cls` | `Transmon`, `AODAxis`, `Dot` |
| a **readout kind** | implements `core/readout.py` protocol | `DispersiveIQ`, `ImagingReadout`, `PSBReadout` |
| (optional) **envelopes** | pack-owned pulse shapes + their compiler lowering (D22) | tweezer's `PolyChirp` |
| (optional) **response models** | simulated responses registered at pack import (D32) | `cqed/responses.py` |
| (optional) **libraries / experiments** | `*Ops` factories + `Experiment` subclasses | `TransmonOps`, `AmplitudeRabi` |
| entry-point | registers the pack under `strumento.packs` | `pyproject.toml` |

### `EXPERIMENTS` — the experiment registry (D33)

```python
# strumento/packs/cqed/__init__.py (an excerpt)
EXPERIMENTS: dict[str, type] = {
    "amplitude_rabi": AmplitudeRabi,
    "t1": T1,
    "t2_ramsey": T2Ramsey,
    "parity_fringe": ParityFringe,
    "resonator_spectroscopy": ResonatorSpectroscopy,
    "active_reset": ActiveReset,
    # … one entry per experiment
}
```

A **name** is the stable, host-facing handle for an experiment: the MCP session's
`list_experiments` / `run_experiment` resolve through this dict (`strumento/mcp.py`), so an agent —
or any harness reading an experiment name off a plan — never imports a class. The contract on each
value is narrow and worth stating:

- it is an `Experiment` subclass that takes **the device positionally** and everything else as
  keyword arguments, so `run_experiment`'s `params` map straight onto the signature;
- those keyword arguments are introspected (`inspect.signature` + `get_type_hints`), never listed by
  hand, and JSON shapes are coerced against the annotation — `{"gains": {"start": 0, "stop":
  16000, "on": "amp"}}` becomes `Sweep(start=0, stop=16000, on="amp")`, `{"readout":
  {"threshold": 0.5}}` becomes `DispersiveIQ(threshold=0.5)`. Add a keyword and it appears in the tool output immediately;
- an experiment with no fitter (its `analyze` returns nothing) is fine: the outcome comes back with
  `fit: null` plus a data-shape summary.

`tweezer` declares `EXPERIMENTS = {}` today — deliberately, and the reason is instructive.
The pack ships element types and an envelope but no device-constructible experiments yet;
declaring the registry empty keeps the pack contract complete, so a session on a skeleton
pack lists nothing rather than failing. Both `cqed` and `spin` have fully populated registries with contract-conformant experiments.

### Worked example — the `tweezer` skeleton

```python
# strumento/packs/tweezer/__init__.py
from strumento.packs.tweezer.calibration import TweezerCalibration
from strumento.packs.tweezer.elements import AODAxis, AtomArray

CALIBRATION = TweezerCalibration

def compose_elements(topology, calib):
    out = {}
    for entry in topology:
        kind, name = entry["type"], entry["name"]
        if kind == "AODAxis":   out[name] = AODAxis(name=name, axis=entry.get("axis", "x"))
        elif kind == "AtomArray": out[name] = AtomArray(name=name, n_traps=entry.get("n_traps", 0))
        else: raise KeyError(f"unknown tweezer element type {kind!r}")
    return out
```

```toml
# pyproject.toml
[project.entry-points."strumento.packs"]
cqed = "strumento.packs.cqed"
tweezer = "strumento.packs.tweezer"
spin = "strumento.packs.spin"
```

With that, `Device.load("devices/tweezer_demo/device.yaml")` composes an `AODAxis`/`AtomArray`
device — and `core` never changed. The `tweezer` pack also shows the two optional extension points:
`ImagingReadout` (an **external-detector** readout kind — the acid test that the readout seam is
IQ-agnostic) and `PolyChirp` (a **pack-owned envelope**: a compact move as polynomial coefficients,
the neutral-atom analogue of `Gaussian`; its compiler lowering registers when the tweezer pulse
library lands).

The `spin` pack (`Dot`/`Plunger`/`Barrier` + `PSBReadout`) mostly *exercises* the seam rather than
extending it — a cheap third data point that `core` isn't cqed-shaped.

## Pack / instance status

| Pack | Elements | Ops / `EXPERIMENTS` | Readout kind | State |
|---|---|---|---|---|
| `cqed` | Transmon, BosonicMode, Coupler, ReadoutResonator | `TransmonOps` / `ModeOps` / `CouplerOps` / `SidebandOps` / `ReadoutOps` / composite; experiments across spectroscopy, calibration, benchmarking, cavity characterization, and bringup | `DispersiveIQ` | built |
| `spin` | Dot (LD/ST/EO/Hole), Plunger, Barrier, ChargeSensor, Reservoir, DriveLine, GateSensor | `LDOps` / `HoleSpinOps` / `ExchangeOps` / `SingletTripletOps` / composite library; 21 registered + 8 DCSource-based | `PSBReadout` / `ElzermanReadout` / `DispersiveGateReadout` / `LatchedPSBReadout` | built |
| `tweezer` | AODAxis, AtomArray | — (move library later); `EXPERIMENTS = {}` | `ImagingReadout` | skeleton |

## cqed calibration fields and their units

The cqed calibration fields keep short names (`f_ge`, `T1`, `gain`), so each field's unit is
written in its schema description instead: it shows in IDE hovers and in the pack's
`Calibration.model_json_schema()`, which `strumento schema calibration --pack cqed` (and the
MCP `schema` tool with `what="calibration", pack="cqed"`) publishes. The table below is
generated from those descriptions, and `tests/packs/cqed/test_calibration_units.py` fails if the
two disagree. Two gain conventions
appear: calibrated pulse gains are **DAC integer codes** (converted to a fraction of full scale,
code / maxv, when a pulse is built), while the `max_*_gain` limits are **fractions of full scale**.
Each `max_*_gain` limit is bounded by its role's hard cap in `packs/cqed/safety.py`
(`0 < limit <= cap`; a device with a limit above its cap does not load), and the MCP session
refuses to write it.
"ADC units" are the raw averaged IQ values the readout returns.

<!-- cqed-calibration-units: begin (generated from the schema; checked by tests) -->
| Model | Field | Unit | Meaning |
|---|---|---|---|
| `Calibration` | `qubit` | qubit section | transmon frequencies, coherence, pulses |
| `Calibration` | `readout` | readout section | the dispersive readout probe |
| `Calibration` | `couplings` | MHz | interaction strength per typed coupling edge, keyed "elementA-elementB" |
| `Calibration` | `multiphoton` | ladder section | multiphoton transitions keyed by name (e.g. "f0-g1") |
| `Calibration` | `swaps` | swap section | coupler swaps keyed "A-B" (e.g. "cavity-S1") |
| `Calibration` | `modes` | mode section | per-bosonic-mode calibration keyed by element name |
| `QubitCal` | `f_ge` | MHz | qubit g–e transition frequency |
| `QubitCal` | `f_ef` | MHz | qubit e–f transition frequency |
| `QubitCal` | `T1` | µs | qubit energy-relaxation time |
| `QubitCal` | `T2` | µs | qubit Ramsey dephasing time (T2*) |
| `QubitCal` | `T2_echo` | µs | qubit Hahn-echo coherence time |
| `QubitCal` | `drag_alpha` | dimensionless | DRAG coefficient: peak of the derivative (Q) quadrature relative to the Gaussian (I) peak; unset means plain Gaussian g–e pulses |
| `QubitCal` | `pulses` | pulse library | calibrated pulses keyed by name (pi_ge, hpi_ge, pi_ef, …) |
| `QubitCal` | `max_drive_gain` | fraction of full scale | largest drive gain this qubit may play; overrides the pack default limit, up to the hard cap 0.8; set by a human, not over MCP |
| `ReadoutCal` | `frequency` | MHz | readout probe frequency |
| `ReadoutCal` | `gain` | DAC integer code | readout probe gain, converted to a fraction of full scale when the probe is built |
| `ReadoutCal` | `readout_length` | µs | readout probe and acquisition window length |
| `ReadoutCal` | `trig_offset` | µs | delay from probe start to the ADC trigger (time of flight) |
| `ReadoutCal` | `relax_delay` | µs | wait after each shot before the next one starts |
| `ReadoutCal` | `phase` | deg | readout probe drive phase |
| `ReadoutCal` | `threshold` | ADC units | single-shot discrimination threshold on the rotated I axis |
| `ReadoutCal` | `angle` | rad | IQ rotation that puts the g/e separation along I |
| `ReadoutCal` | `g_center_iq` | ADC units | raw (I, Q) centre of the \|g> single-shot blob |
| `ReadoutCal` | `e_center_iq` | ADC units | raw (I, Q) centre of the \|e> single-shot blob |
| `ReadoutCal` | `f_center_iq` | ADC units | raw (I, Q) centre of the \|f> single-shot blob |
| `ReadoutCal` | `max_readout_gain` | fraction of full scale | largest readout gain allowed; overrides the pack default limit, up to the hard cap 0.8; set by a human, not over MCP |
| `MultiphotonCal` | `pi` | pulse list | π pulses, one per photon number (index n is the n-th rung) |
| `MultiphotonCal` | `hpi` | pulse list | π/2 pulses, one per photon number (index n is the n-th rung) |
| `SwapCal` | `frequency` | MHz | coupler drive frequency (also selects the coupler generator band) |
| `SwapCal` | `gain` | DAC integer code | swap pulse gain, converted to a fraction of full scale when the pulse is built |
| `SwapCal` | `length` | µs | flat-top length of a full (π) swap; a half swap uses half of it |
| `SwapCal` | `ramp_sigma` | µs | flat-top edge ramp sigma |
| `ModeCal` | `revival_time` | µs | parity revival waits, slow first then fast: [slow, fast, ...] |
| `ModeCal` | `displace` | displacement pulse | the calibrated displacement drive on this mode |
| `ModeCal` | `T1` | µs | photon lifetime |
| `ModeCal` | `T2` | µs | mode dephasing time |
| `ModeCal` | `chi` | MHz | qubit g–e dispersive shift per photon (photon n moves f_ge by -n·chi) |
| `ModeCal` | `chi_ef` | MHz | qubit e–f dispersive shift per photon |
| `ModeCal` | `revival_stark_shift` | MHz | AC-Stark shift of the qubit during the fast parity wait (fast path only) |
| `ModeCal` | `alpha_scale` | dimensionless | parity normalization: the span of excited population between even and odd parity (1.0 = no rescaling) |
| `ModeCal` | `parity_sign` | dimensionless | parity sign, +1 or -1 (+1: even parity reads as low excited population) |
| `ModeCal` | `pe_midpoint` | dimensionless | excited population (0 to 1) that reads as zero parity |
| `ModeCal` | `freq_to_alpha` | MHz per \|α\| | frequency-to-displacement map (recorded; not yet used by any experiment) |
| `ModeCal` | `displacement_cal` | DAC integer code per \|α\| | displacement gain needed per unit of \|α\| |
| `ModeCal` | `max_displace_gain` | fraction of full scale | largest displacement gain allowed on this mode; overrides the pack default limit, up to the hard cap 0.6; set by a human, not over MCP |
| `PulseCal` | `gain` | DAC integer code | pulse gain, converted to a fraction of full scale (code / maxv) when the pulse is built |
| `PulseCal` | `sigma` | µs | Gaussian sigma (gauss), or the edge ramp sigma (flat_top) |
| `PulseCal` | `length` | µs | flat-top length (flat_top only; 0 for gauss) |
| `PulseCal` | `type` | name or label | pulse shape: gauss or flat_top |
| `SidebandPulseCal` | `frequency` | MHz | sideband drive frequency |
| `SidebandPulseCal` | `gain` | DAC integer code | sideband pulse gain, converted to a fraction of full scale when the pulse is built |
| `SidebandPulseCal` | `length` | µs | flat-top length (flat_top only) |
| `SidebandPulseCal` | `sigma` | µs | Gaussian sigma (gauss only) |
| `SidebandPulseCal` | `ramp_sigma` | µs | flat-top edge ramp sigma (flat_top only) |
| `SidebandPulseCal` | `type` | name or label | pulse shape: gauss or flat_top |
| `SidebandPulseCal` | `transfer_g_population` | dimensionless | measured \|g> population (0 to 1) after the transfer |
| `SidebandPulseCal` | `transfer_initial_f_population` | dimensionless | measured \|f> population (0 to 1) before the transfer |
| `SidebandPulseCal` | `transfer_residual_excitation` | dimensionless | excited population (0 to 1) left after the transfer |
| `SidebandPulseCal` | `transfer_initial_mapped_e_population` | dimensionless | reference \|e> population (0 to 1) with \|f> mapped to \|e> before the transfer |
| `SidebandPulseCal` | `transfer_residual_mapped_e_population` | dimensionless | residual \|e> population (0 to 1) with \|f> mapped to \|e> at the selected swap |
<!-- cqed-calibration-units: end -->

## Config-as-code — `DeviceBuilder` (D30)

Hand-writing YAML is repetitive (a multimode instance can carry many storage modes). `core/builder.py`
adds a pure, pack-agnostic `DeviceBuilder` that **produces the canonical `Device`** — a third
authoring surface, freely interconvertible with the data (`Device ⇄ YAML ⇄ builder`), never a
parallel store. `DeviceBuilder(...).build(calib=…)` and `Device.load(equivalent.yaml)` yield the
identical model.

```python
from strumento.core.builder import DeviceBuilder
from strumento.core.wiring import Generator

dev = (DeviceBuilder(pack="cqed")            # composes over the loaded pack's element types
       .add("Transmon", "qubit")             # (never a hardcoded "Transmon")
       .add("Coupler", "coupler")
       .add("ReadoutResonator", "readout")
       .wire("qubit", drive=Generator(dac_ch=1), sideband=Generator(dac_ch=5))  # role = the key
       .readout(dac_ch=0, adc_chs=[0])
       .build(calib="devices/multimode_demo/calib.yml"))   # calib is DATA, never code
```

**Carve-out:** the builder composes topology + wiring only. Calibration *values* change every run
(the calibration loop / a device-model report writes them), so they stay data — passed to
`build(calib=…)` as a dict or file. Construction-time validation gives early diagnostics (readout
needs ≥1 target; a coupling rejects an unknown endpoint / missing mediator).

## Typed couplings + Hamiltonian export (D31)

A physical coupler (with a DAC) is an *element* you `add`; an *interaction* is a separate,
first-class **typed edge**:

```python
from strumento.packs.cqed.couplings import BeamSplitter, ZZ
from strumento.core.coupling import ring_couple

b.coupling("cavity", "S3", form=BeamSplitter, mediated_by="coupler")  # a typed edge
ring_couple(b, ["q0", "q1", "q2", "q3"], form=ZZ)                          # a pattern emits edges
```

`form` is a **pack-registered union** — the D22 envelope pattern applied to interactions. `cqed`
registers `Dispersive` / `ZZ` / `Exchange` / `BeamSplitter` / `JaynesCummings`, each carrying a
symmetry tag and an **operator template**. Strengths (χ, ζ, g, J) are calibration values — kept in
`calib.couplings["a-b"]`, never on the edge.

Because element types now carry Hamiltonian templates (`Transmon` → Duffing, `BosonicMode` →
harmonic + Kerr) and couplings carry operator forms, the topology graph **is** a Hamiltonian spec.
`dev.to_operator_model(...)` lowers it — the mirror of `from_solution`, closing the pulse↔model
codesign loop:

```python
model = dev.to_operator_model(frame="rotating", cutoffs={"q0": 3, "cavity": 8},
                              include=["q0", "cavity"])   # subgraph → tractable Hilbert space
model.H0            # drift Hamiltonian (framework-neutral, angular units)
model.drives        # {line -> drive operator H_k}
model.drive_map     # labels that round-trip through from_solution
model.to_dict()     # JSON-serializable wire format for an upstream consumer
```

`to_operator_model` is **framework-neutral** and stays in `core` (no Piccolo import). A thin
`strumento.interop.piccolo.to_piccolo_system(model)` adapter — living in `interop`, *not* core —
shapes it into Piccolo's `(drift, ordered drives)` convention; the actual `QuantumSystem` is built
upstream (Piccolo is Julia) from `model.to_dict()`. QuTiP or any other consumer uses the same spec.

See [architecture.md](architecture.md) (device model + solver seam) for the full design.
