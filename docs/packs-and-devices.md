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
  qubit: {f_ge: 3562.61, pulses: {pi_ge: {gain: 3289, sigma: 0.035}}}
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

With that, `Device.load("devices/thompson_tweezer/device.yaml")` composes an `AODAxis`/`AtomArray`
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

## Composition roadmap (D30 / D31) — designed, not yet built

Two decisions extend authoring and the device model; both land with the codesign track, off the
slices 1–6 critical path.

- **`DeviceBuilder` — config-as-code (D30).** Hand-writing YAML is repetitive (the Schuster
  instance alone is 7 storage modes). A pure, pack-agnostic `DeviceBuilder` will *produce* the
  canonical Pydantic `Device` — a third authoring surface interconvertible with YAML
  (`Device ⇄ YAML ⇄ builder`), composing over whatever element types the loaded pack registers, so
  every pack gets programmatic authoring for free. Calibration *values* stay data (`build(calib=…)`),
  never code.
- **Typed coupling edges + `to_operator_model` (D31).** A physical coupler (with a DAC) is an
  *element*; an *interaction* becomes a first-class typed edge — `coupling(a, b, form=…)` — whose
  `form` is a pack-registered union (`Dispersive`/`ZZ`/`BeamSplitter`/…) carrying an operator
  template. Once elements and couplings carry operator templates, `dev.to_operator_model(...)`
  lowers the topology + calibration into a framework-neutral Hamiltonian spec — the mirror of
  `from_solution`, closing the pulse↔model codesign loop.

See `docs/spec.md` §5.3 and §6.6 for the full design.
