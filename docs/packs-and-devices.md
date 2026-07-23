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
built; `tweezer` and `spin` as skeletons), and `core` composes an instance of each unchanged — the
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
        - {role: drive, dac_ch: 2, nyquist: 1}
        - {role: sideband, dac_ch: 0, nyquist: 1}
  readout: {dac_ch: 5, adc_chs: [0]}
topology:
  - {type: Transmon, name: qubit}
  - {type: BosonicMode, name: manipulate, line: manipulate}
  - {type: Coupler, name: coupler}
  - {type: ReadoutResonator, name: readout}
calibration:
  qubit: {f_ge: 4000.0, pulses: {pi_ge: {gain: 8192, sigma: 0.030}}}
  readout: {frequency: 749.43, gain: 800, readout_length: 1.5, trig_offset: 450, relax_delay: 2500}
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
`dev.cavity`. Each element type declares two things:

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
topology node with no callable ops yet (every element in the `tweezer`/`spin` skeletons, and
`cqed`'s `BosonicMode`/`Coupler` until slice 4).

## Anatomy of a pack

A pack is a Python package exposing a small contract that `core.device.load` consumes:

| Provides | What | Example |
|---|---|---|
| `CALIBRATION` | Pydantic calibration schema for the modality | `cqed.Calibration`, `TweezerCalibration`, `SpinCalibration` |
| `compose_elements(topology, calib)` | build `{name: element}` from a topology list | maps `{"type": "Transmon", ...}` → `Transmon(...)` |
| element **types** | dataclasses with `lines()` + `ops_cls` | `Transmon`, `AODAxis`, `Dot` |
| a **readout kind** | implements `core/readout.py` protocol | `DispersiveIQ`, `ImagingReadout`, `PSBReadout` |
| (optional) **envelopes** | pack-owned pulse shapes + their compiler lowering (D22) | tweezer's `PolyChirp` |
| (optional) **libraries / experiments** | `*Ops` factories + `Experiment` subclasses | `TransmonOps`, `AmplitudeRabi` |
| entry-point | registers the pack under `strumento.packs` | `pyproject.toml` |

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

| Pack | Elements | Ops / experiments | Readout kind | State |
|---|---|---|---|---|
| `cqed` | Transmon, BosonicMode, Coupler, ReadoutResonator | `TransmonOps` (ge/ef), `AmplitudeRabi`; sideband/coupler/parity in slices 4–5 | `DispersiveIQ` | built (slices 1–2) |
| `tweezer` | AODAxis, AtomArray | — (move library later) | `ImagingReadout` | skeleton |
| `spin` | Dot, Plunger, Barrier | — (LD / exchange-only later) | `PSBReadout` | skeleton |

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
       .wire("qubit", drive=Generator(dac_ch=2), sideband=Generator(dac_ch=0))  # role = the key
       .readout(dac_ch=5, adc_chs=[0])
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

b.coupling("manipulate", "S3", form=BeamSplitter, mediated_by="coupler")  # a typed edge
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
model = dev.to_operator_model(frame="rotating", cutoffs={"q0": 3, "manipulate": 8},
                              include=["q0", "manipulate"])   # subgraph → tractable Hilbert space
model.H0            # drift Hamiltonian (framework-neutral, angular units)
model.drives        # {line -> drive operator H_k}
model.drive_map     # labels that round-trip through from_solution
model.to_dict()     # JSON-serializable wire format for an upstream consumer
```

`to_operator_model` is **framework-neutral** and stays in `core` (no Piccolo import). A thin
`strumento.interop.piccolo.to_piccolo_system(model)` adapter — living in `interop`, *not* core —
shapes it into Piccolo's `(drift, ordered drives)` convention; the actual `QuantumSystem` is built
upstream (Piccolo is Julia) from `model.to_dict()`. QuTiP or any other consumer uses the same spec.

See `docs/spec.md` §5.3 and §6.6 for the full design.
