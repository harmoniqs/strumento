---
type: plan
date: 2026-07-17
session_id: "e3dfcc03-f686-47a0-b129-29407ed941d0"
status: draft
spec: "./spec.md"
tags: [plan, strumento, cqed, core, python, qick, tproc-v2, tracer-bullet, tdd]
---

# strumento — `core` + `cqed` pack (slices 1–6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `amico-plugin:executing-plans` to implement this
> plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is a **Python** package —
> the TDD cycle is `pytest`, not Julia.

**Goal:** Build the modality-agnostic `core` and the `cqed` (superconducting transmon+bosonic) modality
pack of `strumento`, proving it end-to-end on the mock station across six vertical tracer-bullet slices
(skeleton → amplitude Rabi → T1/T2 → f0g1/storage swap → parity → optimal-control import).

**Architecture:** Approach A from the spec — a Pulse-IR + compiler design. Vertical slices: slice 1
stands up a *thin* version of every layer (wiring → pulse IR → device → one library factory →
compiler → program → mock → result); slices 2–6 thicken it. `core` never imports physics; the `cqed`
pack supplies element types + libraries + the `DispersiveIQ` readout kind. Only `core/compiler`,
`core/program`, `core/mock` import qick.

**Tech Stack:** Python 3.11+, `pydantic` (config schema), `numpy`/`scipy`, `lmfit` (fitters),
`pytest` (TDD). `qick` (with `qick.asm_v2`) is an **optional extra** — the pure layers test without it;
the qick-facing layers + the mock need it. tProc **v2**.

**Scope of THIS plan:** `core` + `cqed` pack, spec §11 slices 1–6. **Out of scope** (later plans):
slice 7b (in-sequence v2 measurement feedback / full active-reset, gated on Sho), **slice 8 /
codesign enablement (spec §19, D26–D29 — deferred task sketch at the end of this plan)**, the `tweezer`
and `spin` packs, expt_service unification (§16), and the three open decisions (§14.1 config versioning,
§14.2 job-server, §16 execution). Config loading here uses a **plain YAML instance file** (versioning
mechanism deferred to §14.1).

**⚠ Operational note:** the org is at its monthly spend limit. Execute in the main loop; do **not**
dispatch subagents/workflows until the limit is raised.

---

## File structure (from spec §3, this plan's subset)

```
strumento/
  pyproject.toml                      # deps; qick as [project.optional-dependencies].qick
  core/
    __init__.py
    wiring.py       Wiring / Generator / Line / ReadoutWiring / LineRef (Pydantic) + resolve()  (pure)
    pulses.py       Envelope union (Gaussian/FlatTop/Const/Arb) + Pulse / Delay / Raw /
                      Segment / Measure / Seq builder (D22)                                    (pure)
    sweeps.py       Sweep axis (added in slice 2)                                            (pure)
    device.py       Device (topology) + load(); binds one namespace per named element (D21)  (pure*)
    result.py       MeasurementResult typed seam (reserves meta["environment"], §18)         (pure)
    readout.py      ReadoutKind / Discriminator protocols + data-source abstraction         (pure)
    environment.py  EnvironmentSource protocol + mock (added Task 19, §18/D24)               (pure)
    fitters.py      lmfit fit primitives (added slices 2–3)                                  (pure)
    compiler.py     lower Seq/Pulse → AveragerProgramV2 via envelope registry; quantize()    (qick)
    program.py      StrumentoProgram(AveragerProgramV2) + acquire → MeasurementResult          (qick)
    mock.py         MockQickSocV2 no-FPGA harness                                            (qick)
  packs/
    cqed/
      __init__.py
      elements.py   Transmon / BosonicMode / Coupler / ReadoutResonator types + ops-class map (pure)
      calibration.py  Pydantic calib schema + swap-dataset accessor + int-code→fraction gain  (pure)
      library/      transmon / sideband / mode / coupler / composite factories (element-keyed) (pure)
      readout.py    DispersiveIQ readout kind + discriminator                                (pure)
  devices/
    schuster_multimode/device.yaml    # instance: topology + wiring + calib values           (config)
  tests/
    core/ ...   packs/cqed/ ...   e2e/ ...
```

---

## Task 0: Package skeleton, deps, and qick-v2 API verification

**Depends on:** []
**Parallel-safe:** false  (creates `pyproject.toml`)
**Touches:** [pyproject.toml]

**Files:**
- Create: `pyproject.toml`, `strumento/__init__.py`, `strumento/core/__init__.py`, `strumento/packs/__init__.py`, `strumento/packs/cqed/__init__.py`, `tests/__init__.py`
- Create: `docs/qick-v2-api-notes.md`

- [ ] **Step 1: Write `pyproject.toml`** — package `strumento`, `requires-python=">=3.11"`, deps
  `pydantic>=2`, `numpy`, `scipy`, `lmfit`; `[project.optional-dependencies]` `qick = ["qick"]`,
  `test = ["pytest"]`. Configure `pytest` with markers `qick` (needs board/qick) and `pure`.

- [ ] **Step 2: Verify the target qick v2 API (spec R2 — LARGELY RESOLVED by Sho, spec D25).** The API
  shape is confirmed by Sho Uemura's IntonatoQICK PR #5 review; the reference is his notebook
  **[`meeg/qick_demos_sho` `tprocv2/00_intro.ipynb`](https://github.com/meeg/qick_demos_sho/blob/main/tprocv2/00_intro.ipynb)**.
  Transcribe the exact names/signatures from that notebook into `docs/qick-v2-api-notes.md`:
  `AveragerProgramV2` (**subclass it; override `_initialize()`/`_body()` — do NOT append to a bare
  program object; that idiom is `QickProgramV2` only**), `declare_gen(ch,nqz,mixer_freq=,mux_freqs=)`,
  `add_gauss`, `add_pulse` (+ **const/flat-top** envelopes, not arb-only — spec D22/D25), `add_loop`,
  `QickSweep1D`, `pulse`, `delay`/`delay_auto`, `trigger`, `acquire` return shape, and the **gain
  convention** (float fraction vs `maxv`; confirm the `gain_frac = gain_int/maxv` divisor — spec §6.1).
  `asm_v2` **is present** (Sho's notebook uses it) — the earlier "escalate if missing" contingency is
  moot. **Remaining concrete artifact:** obtain the board's **`QickSoc.dump_cfg()` JSON** from Sho and
  commit it as the soccfg snapshot (spec §3/§10) — this lets Task 7–10 construct a real
  `AveragerProgramV2` offline (`QickConfig(json)`), no board. The pure tasks (1–6) proceed regardless.

- [ ] **Step 3: Commit** — `git add pyproject.toml strumento tests docs && git commit -m "chore: strumento package skeleton + qick-v2 API notes"`

---

## Slice 1 — Vertical skeleton (thin version of every layer)

*Goal of the slice: `dev.qubit.ge_pi()` + a measure, compiled onto `StrumentoProgram`, run on
`MockQickSocV2`, returns a `MeasurementResult` of the right shape. This forces a thin version of
wiring, pulses, device, one library factory, compiler, program, mock, result.*

### Task 1: `core/wiring.py` — wiring + resolve (pure)

**Depends on:** [0] · **Parallel-safe:** true · **Touches:** []

> **Naming (D-review):** the module is `wiring.py`, **not** `channels.py` — "model the device, *not the
> FPGA channels*" (§5); a module literally named `channels` would undercut the tenet. The readout config
> type is `ReadoutWiring` (not `Readout`) to avoid colliding with `core/readout.py`'s `ReadoutKind`
> protocols (Task 16).

**Files:** Create `strumento/core/wiring.py`; Test `tests/core/test_wiring.py`

- [ ] **Step 1: Write failing tests.**

```python
# tests/core/test_wiring.py
import pytest
from strumento.core.wiring import Generator, Line, Wiring, LineRef, ReadoutWiring

def _wiring():
    return Wiring(
        lines={
            "qubit": Line(name="qubit", generators=[
                Generator(role="drive", dac_ch=2, nyquist=1, gen_type="full"),
                Generator(role="sideband", dac_ch=0, nyquist=1, gen_type="full"),
            ]),
            "coupler": Line(name="coupler", generators=[
                Generator(role="low",  dac_ch=1, band_hz=(0.0, 1800e6)),
                Generator(role="high", dac_ch=1, band_hz=(1800e6, 1e12)),
            ]),
        },
        readout=ReadoutWiring(dac_ch=5, adc_chs=[0]),
    )

def test_resolve_by_role():
    w = _wiring()
    assert w.resolve(LineRef("qubit", "drive"), freq_mhz=3562).dac_ch == 2
    assert w.resolve(LineRef("qubit", "sideband"), freq_mhz=2010).dac_ch == 0

def test_resolve_band_gated():
    w = _wiring()
    assert w.resolve(LineRef("coupler", "low"),  freq_mhz=352).dac_ch == 1     # 352 MHz < 1800
    assert w.resolve(LineRef("coupler", "high"), freq_mhz=1998).dac_ch == 1

def test_resolve_unknown_raises():
    w = _wiring()
    with pytest.raises(KeyError):
        w.resolve(LineRef("nope", "drive"), freq_mhz=100)
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError`): `pytest tests/core/test_wiring.py -q`

- [ ] **Step 3: Implement.** Pydantic `BaseModel`s: `Generator(role,dac_ch,nyquist=1,gen_type="full",
  band_hz:tuple|None=None,mixer_freq:float|None=None,mux_freqs:list[float]|None=None)`;
  `Line(name, generators:list[Generator], chain:SignalChain|None=None)` — `chain` is the **optional
  signal-chain metadata** stub (§18/D24; leave `SignalChain` a minimal Pydantic model or `dict`, filled
  later); `ReadoutWiring(dac_ch,adc_chs,mux_freqs=None,lengths_us=...,thresholds=...,trig_offset=...)`;
  `LineRef(NamedTuple: line:str, role:str)`; `Wiring(lines:dict, readout:ReadoutWiring)` with
  `resolve(ref, freq_mhz) -> Generator` (KeyError if line/role missing; among role-matching generators
  pick the one whose `band_hz` contains `freq_mhz*1e6`, or the sole one).

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `git commit -m "feat(core): wiring model + resolve"`

### Task 2: `core/pulses.py` — Pulse IR + fluent builder (pure)

**Depends on:** [1] · **Parallel-safe:** true · **Touches:** []

**Files:** Create `strumento/core/pulses.py`; Test `tests/core/test_pulses.py`

> **Envelope union (D22):** replace the flat `Shape` enum + shape-specific `Pulse` fields with a **typed
> envelope union** — `Gaussian`/`FlatTop`/`Const`/`Arb` **frozen** dataclasses, each carrying its own
> params (no `None`-field conventions). `Pulse.envelope: Envelope`. Frozen ⇒ hashable ⇒ the compiler's
> dedup keys on `(generator, envelope)` for free (spec R6). Packs register more envelopes (`tweezer`
> adds `PolyChirp`) — so `PolyChirp` is **not** in core here.

- [ ] **Step 1: Write failing tests** — envelope dataclasses carry their own params; a `Pulse` carries
  `line/freq_mhz/gain/phase_deg/envelope/start_us/label`; `Seq().play(p).wait(0.5).measure()` builds an
  ordered op list; `.play_parallel([p1,p2])` yields a `Segment`; `Seq` + `Seq` concatenates; `.raw(fn)`
  appends a `Raw`. Assert op types/order and that a frozen envelope is hashable.

```python
from strumento.core.pulses import Pulse, Delay, Segment, Raw, Seq, Gaussian, FlatTop, Const, Arb
from strumento.core.wiring import LineRef

def _p(): return Pulse(line=LineRef("qubit","drive"), freq_mhz=3562.6, gain=0.10, envelope=Gaussian(sigma_us=0.035))

def test_builder_order():
    s = Seq().play(_p()).wait(0.5).measure()
    kinds = [type(op).__name__ for op in s.ops]
    assert kinds == ["Pulse", "Delay", "Measure"]

def test_play_parallel_makes_segment():
    s = Seq().play_parallel([_p(), _p()])
    assert isinstance(s.ops[0], Segment) and len(s.ops[0].pulses) == 2

def test_envelope_is_frozen_hashable():
    assert hash(Gaussian(sigma_us=0.035)) == hash(Gaussian(sigma_us=0.035))   # dedup key (R6)
    assert Gaussian(sigma_us=0.035) != FlatTop(ramp_sigma_us=0.01, length_us=1.0)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement.** Frozen dataclasses `Gaussian(sigma_us)`, `FlatTop(ramp_sigma_us,
  length_us)`, `Const(length_us)`, `Arb(idata, qdata=None)`; `Envelope = Gaussian|FlatTop|Const|Arb`
  (an **open** union — the compiler's lowering registry, Task 8, is what packs extend). Dataclasses
  `Pulse(line, freq_mhz, gain, phase_deg=0.0, envelope=..., start_us=None, label=None)`,
  `Delay(duration_us)`, `Raw(fn)`, `Segment(pulses:list[Pulse])`, `Measure(kind=None, **kw)`. `Seq` is
  the fluent builder over `list[Pulse|Delay|Raw|Segment|Measure]` with `.play(pulse_or_seq)` (flattens a
  `Seq`), `.wait(us)`, `.raw(fn)`, `.play_parallel(list|Segment|Seq)`, `.measure(**kw)`, `__add__`.
  Keep the IR plain (envelopes frozen; `Pulse`/`Seq` mutable) dataclasses — hot path, not Pydantic
  (spec §3). Note in the module docstring that `Measure` is the readout op `.measure()` emits (it is not
  in the spec's §6.1 IR list — plan-introduced, D-review nice-to-have).

- [ ] **Step 4: Run — expect PASS.** — [ ] **Step 5: Commit** `feat(core): pulse IR + fluent builder`

### Task 3: `core/result.py` — MeasurementResult (pure)

**Depends on:** [0] · **Parallel-safe:** true · **Touches:** []

**Files:** Create `strumento/core/result.py`; Test `tests/core/test_result.py`

- [ ] **Step 1–2: Failing test** — `MeasurementResult(xpts, avgi, avgq, raw=None, herald_lanes=(),
  meta={})`; `.as_tuple() == (xpts, avgi, avgq)`; `.postselect(...)` ANDs enabled herald lanes (stub:
  with no lanes it's identity); `meta` accepts an `"environment"` key (reserved, §18/D24 — the Runner
  stamps it later, Task 19). Assert `.as_tuple()`, shapes, and that `meta["environment"]` round-trips.

- [ ] **Step 3: Implement** the dataclass + `as_tuple()` + `postselect()` (identity when
  `herald_lanes` empty; full logic deferred to slice 7). Document `meta["environment"]` as the reserved
  station-telemetry key (§18) so stamping (Task 19) has a home from day one.

- [ ] **Step 4–5: Pass + commit** `feat(core): MeasurementResult seam`

### Task 4: `packs/cqed/elements.py` + `calibration.py` (pure)

**Depends on:** [1] · **Parallel-safe:** true · **Touches:** []

**Files:** Create `strumento/packs/cqed/elements.py`, `strumento/packs/cqed/calibration.py`;
Test `tests/packs/cqed/test_calibration.py`

- [ ] **Step 1–2: Failing tests** — `Calibration` (Pydantic) round-trips a minimal dict:
  `qubit.f_ge`, `qubit.pulses.pi_ge = {gain, sigma, length, type}`, `readout.{frequency,gain,
  readout_length,trig_offset,relax_delay}`. Element types `Transmon(name, f_ge, f_ef, pulses)`,
  `BosonicMode(name, role)`, `Coupler`, `ReadoutResonator` construct and expose their `LineRef`s and
  their **ops class** (D21: `Transmon.ops_cls is TransmonOps`, etc. — a stub class reference is fine
  here; the factories land in Task 6). `gain_int_to_frac(3289, maxv)` returns the expected fraction
  (**D23 — gain conversion lives in the cqed pack, not core**; catches the v1→v2 gain-units bug).

- [ ] **Step 3: Implement** Pydantic `Calibration` schema (mirror the *typed* subset of the v1
  `hardware_config.yml → device.*`, one field per physical quantity, no register shadows) + the four
  element-type dataclasses, each with (a) a `lines()` method returning the `LineRef`s it drives
  (`Transmon → qubit.drive`; sideband is a qubit-line generator; `BosonicMode(manipulate) →
  manipulate.main`; `Coupler → coupler.{low,high}`; `ReadoutResonator → readout`) and (b) an
  `ops_cls` attribute naming the ops class its named instances bind to (D21). Add
  `gain_int_to_frac(gain_int, maxv) -> float` here (the pack owns lab-native int codes → v2 fraction,
  D23; `maxv` comes from the soccfg snapshot at factory time — spec §6.1).

- [ ] **Step 4–5: Pass + commit** `feat(cqed): element types + calibration schema`

### Task 5: `core/device.py` — Device + `load()` (pure); instance YAML

**Depends on:** [1,4] · **Parallel-safe:** false · **Touches:** [devices/schuster_multimode/device.yaml]

**Files:** Create `strumento/core/device.py`, `devices/schuster_multimode/device.yaml`;
Test `tests/core/test_device.py`

- [ ] **Step 1: Author `devices/schuster_multimode/device.yaml`** — a **device instance**: `pack: cqed`,
  a `topology` (1 Transmon + 1 BosonicMode(manipulate) + 7 BosonicMode(storage) + Coupler + Readout),
  the `wiring` (from v1 `hardware_config.yml → hw.soc.dacs`: qubit=2, sideband=0, flux_low=flux_high=1,
  manipulate=3, storage=6, readout=5+ADC0), and `calibration` values (ge/ef freqs, `pi_ge`/`hpi_ge`/
  `pi_ef` gains+sigmas, readout params) transcribed from the v1 config. *(Cross-check numbers against
  `.../scratchpad/multimode_expts/configs/hardware_config.yml`.)*

- [ ] **Step 2: Failing test** — `Device.load("devices/schuster_multimode/device.yaml")` returns a
  `dev` whose `dev.wiring.resolve(LineRef("qubit","drive"), 3562).dac_ch == 2`, `dev.calib.qubit.f_ge`
  is set, and `dev.soccfg_snapshot` is a plain dict (no qick import). `import strumento.core.device` must
  not import qick.

- [ ] **Step 3: Implement** `Device` (holds `wiring`, `calib`, `soccfg_snapshot: dict`, composed
  `elements`) + `load()` (parse YAML → validate Pydantic → compose the pack's element types per topology
  → bind wiring → attach calib). **Namespace binding is topology-derived (D21):** `load()` walks the
  named element instances and, for each, binds `dev.<instance-name>` to that element type's `ops_cls`
  (Task 4) — so `dev.qubit`/`dev.manipulate` exist because the *Schuster instance yaml names them that*,
  not because the pack hardcodes them. The instance yaml may also declare **alias namespaces** (e.g.
  `sideband` → the transmon's sideband ops on the qubit line). No instance name appears in pack code.
  The actual ops methods land in Task 6; here `load()` just wires the bindings (a stub `ops_cls` with no
  methods is fine for this task's test).

- [ ] **Step 4: Assert `device.py` is qick-free** — a test that `import strumento.core.device` succeeds
  with `qick` uninstalled (or `sys.modules['qick']` blocked). — [ ] **Step 5: Commit** `feat(core): Device + instance load`

### Task 6: `packs/cqed/library/transmon.py` — `TransmonOps` (ge_pi/…) + per-instance binding (pure)

**Depends on:** [2,5] · **Parallel-safe:** true · **Touches:** []

> **Naming (D21):** the module/class are element-type-keyed (`transmon.py` / `TransmonOps`), **not**
> instance-keyed (`qubit.py`/`QubitNamespace`). `dev.qubit` appears because the Schuster *instance*
> names its transmon `qubit`; a different yaml yields `dev.q0`/`dev.cavity`. **Gain (D23):** the factory
> accepts int codes and converts to a **v2 fraction** via `gain_int_to_frac` (Task 4) — so the returned
> `Pulse.gain` is a fraction. **Envelope (D22):** `ge_pi` builds a `Gaussian(sigma_us=...)`.

**Files:** Create `strumento/packs/cqed/library/__init__.py`, `strumento/packs/cqed/library/transmon.py`;
Modify `strumento/core/device.py` (bind ops per named element); Test `tests/packs/cqed/test_library_transmon.py`

- [ ] **Step 1: Failing test** — `dev.qubit.ge_pi()` returns a `Pulse` with `line==LineRef("qubit",
  "drive")`, `freq_mhz==dev.calib.qubit.f_ge`, `envelope==Gaussian(sigma_us==pi_ge.sigma)`, and
  `gain==gain_int_to_frac(pi_ge.gain, maxv)` (**a fraction, not the int code** — D23);
  `dev.qubit.ge_pi(gain=amp, freq=f)` overrides those fields (a raw fraction passed as `gain` is used
  as-is; an int-code override goes through the converter — pick one override convention and test it);
  `dev.qubit.ge_pi(phase=90).phase_deg == 90`.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** `ge_pi(dev, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse` (and
  `ge_hpi`, `ef_pi` — `None` ⇒ from calib, scalar/`Sweep` ⇒ override; calib int codes → fraction via
  `gain_int_to_frac`) as methods on `TransmonOps(dev, element)`. In `device.py`, the D21 binding from
  Task 5 now attaches a live `TransmonOps` bound to the named transmon element during `load()`.

- [ ] **Step 4–5: Pass + commit** `feat(cqed): TransmonOps (ge_pi/ge_hpi/ef_pi) + per-instance binding`

### Task 7: `core/mock.py` — MockQickSocV2 (qick)

**Depends on:** [0] · **Parallel-safe:** true · **Touches:** []  · **marker:** `qick`

**Files:** Create `strumento/core/mock.py`; Test `tests/core/test_mock.py` (marked `qick`)

> **D25 (Sho):** `MockQickSocV2` should be built around a **real `QickConfig` reconstructed from the
> committed `dump_cfg()` JSON** (`QickConfig(json)` — no board), so the program genuinely
> builds/compiles (v2 validators fire) and only the leaf *acquire/data* `soc.*` calls are stubbed. This
> is the v1 mock strategy carried to v2 — now known to work offline (Task 0 obtains the JSON).

- [ ] **Step 1: Failing test** — construct `MockQickSocV2` from the committed soccfg JSON snapshot;
  assert it carries a real `QickConfig` (so `declare_gen`/`add_pulse` validate) **and** stubs the v2
  acquire soc surface (the leaf methods v2's `AveragerProgramV2.acquire` calls — from Task 0's API
  notes), whose data-return yields **correctly-shaped zeros, int dtype** for a given `(reps, loop_dims,
  n_ro)`. *(Port the v1 `mock_hardware.MockQickSoc` shape/dtype contract to v2's acquire path — see
  `.../multimode_expts/experiments/mock_hardware.py`.)*

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** `MockQickSocV2` — wrap a `QickConfig(json)` from the committed snapshot;
  stub only the leaf acquire/data path (`poll`/`get_*`) returning zeros shaped `(*loop_dims, n_ro, ...)`
  int dtype, per the v2 acquire loop's assertions. Cover the three v2 acquire entry points if the target
  qick has them. *(Until Task 0 provides the JSON, gate this test/fixture on the snapshot's presence.)*

- [ ] **Step 4–5: Pass + commit** `feat(core): MockQickSocV2 harness`

### Task 8: `core/compiler.py` — envelope-registry lowering (gaussian) + measure; quantize (qick)

**Depends on:** [1,2,4,7] · **Parallel-safe:** true · **Touches:** []  · **marker:** `qick`

> **Envelope registry (D22):** the compiler lowers by an **envelope-type-keyed registry**
> (`register_envelope(Gaussian, fn)` / `@lowers(Gaussian)`), *not* an `if shape == …` ladder — this is
> the seam a pack extends (Task 13 adds `FlatTop`; the tweezer pack later registers `PolyChirp`). **Gain
> (D23):** `Pulse.gain` is **already a v2 fraction** by the time it reaches the compiler (the cqed
> factory converted it, Task 6) — the compiler does **no** int→fraction step; it passes `gain` straight
> to `add_pulse`. (`gain_int_to_frac` lives in `packs/cqed/calibration.py`, Task 4.)

**Files:** Create `strumento/core/compiler.py`; Test `tests/core/test_compiler.py` (marked `qick`)

- [ ] **Step 1: Failing tests.**
  - Lowering a `Seq().play(ge_pi).measure()` onto a program builds one `Gaussian` envelope + one pulse at
    the resolved DAC (with the fraction `gain` passed through unchanged) + a readout trigger (assert via
    the program's recorded ops or the mock).
  - Two pulses with an **equal** `Gaussian` envelope on the same generator add the envelope **once**
    (dedup keyed on `(generator, envelope)` — the frozen envelope is the hash key; spec R6).
  - An **unregistered** envelope type raises a clear "no lowering registered for <type>" error (proves
    the registry seam, and that core doesn't silently handle a pack shape it can't play).
  - `quantize(pulse, qickcfg)` returns `Realized` with `freq_mhz_actual`, `length_us_actual`,
    `gain_frac` (== the pulse's fraction, realized to DAC resolution), `cycles`.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** the lowering **registry** + `lower(seq, program)` — walk the IR: for a
  `Pulse`, resolve `wiring.resolve(line, freq)`→gen, dispatch on `type(pulse.envelope)` to its
  registered lowering (`Gaussian`→`add_gauss`; `FlatTop`/`Const`/`Arb` registered later), `add_pulse`
  (physical `freq_mhz`, `phase_deg`, fraction `gain`), then `program.pulse(...)`; `Delay`→`program.delay`;
  `Raw`→`fn(program)`; `Segment`→same-`t` pulses (respecting per-pulse `start_us` offset); `Measure`→
  `trigger`/readout. Add `quantize`. Envelope **dedup** by `(generator, envelope)` — add once, reference
  by name (frozen envelope ⇒ hashable, R6). *(Verify freq/length conversions against Task 0's API notes.)*

- [ ] **Step 4–5: Pass + commit** `feat(core): compiler — envelope-registry lowering + dedup + quantize`

### Task 9: `core/program.py` — StrumentoProgram + acquire → MeasurementResult (qick)

**Depends on:** [3,6,8] · **Parallel-safe:** true · **Touches:** []  · **marker:** `qick`

**Files:** Create `strumento/core/program.py`; Test `tests/core/test_program.py` (marked `qick`)

- [ ] **Step 1: Failing test** — `StrumentoProgram(dev, seq=Seq().play(dev.qubit.ge_pi()).measure())` runs
  `.acquire(MockQickSocV2())` and returns a `MeasurementResult` (not a tuple); `res.as_tuple()` works.

- [ ] **Step 2–3: Implement** `StrumentoProgram(AveragerProgramV2)`: `__init__(device, seq=None,
  sweeps=(), reps, ...)`; `_initialize` = `_auto_declare()` (declare gens/readout + envelopes from
  seq/sweeps via wiring) + `add_loop` per sweep; `_body` = `self.play_seq(self.seq)`; helpers
  `play`/`play_seq`/`measure` delegating to `core.compiler.lower`; `acquire(soc,…)` wraps
  `AveragerProgramV2.acquire` into a `MeasurementResult` (attach sweep axes as `xpts`). Escape hatch:
  it IS an `AveragerProgramV2`.

- [ ] **Step 4–5: Pass + commit** `feat(core): StrumentoProgram + acquire → MeasurementResult`

### Task 10: SLICE 1 end-to-end vertical test

**Depends on:** [9] · **Parallel-safe:** true · **Touches:** []  · **marker:** `qick`

**Files:** Create `tests/e2e/test_slice1_skeleton.py`

- [ ] **Step 1: Write the e2e test** — load the schuster instance; build
  `Seq().play(dev.qubit.ge_pi()).measure()`; `StrumentoProgram(dev, seq).acquire(MockQickSocV2())`; assert
  a `MeasurementResult` with the expected single-readout shape and that a **numeric golden** check holds
  (the realized ge_pi freq/gain-frac/cycles from `quantize` match hand-computed values from the calib).

- [ ] **Step 2: Run — expect PASS** (the whole vertical): `pytest tests/e2e/test_slice1_skeleton.py -q`

- [ ] **Step 3: Commit** `test(e2e): slice 1 vertical skeleton green`

---

## Slice 2 — Amplitude Rabi (sweeps + fitting)

### Task 11: `core/sweeps.py` + compiler `add_loop` lowering + Rabi fitter + experiment

**Depends on:** [8,9,10] · **Parallel-safe:** true · **Touches:** []

**Files:** Create `strumento/core/sweeps.py`, `strumento/core/fitters.py`,
`strumento/packs/cqed/experiments/amplitude_rabi.py`; Modify `strumento/core/compiler.py`;
Test `tests/core/test_sweeps.py`, `tests/core/test_fitters.py` (pure),
`tests/e2e/test_slice2_rabi.py` (`qick`)

- [ ] **Step 1: Failing tests.**
  - `Sweep("amp", 0, 30000, 51)` has `.name/.start/.stop/.points`; a `Pulse.gain` may be a `Sweep`.
  - **Gain-sweep unit (D23):** `dev.qubit.ge_pi(gain=Sweep("amp", 0, 30000, 51))` — the factory
    converts the sweep endpoints to a **fraction** sweep for the IR/compiler, **and** the experiment
    records the user's **original int-code axis** as `xpts` (so notebooks read familiar units). Assert
    the compiled sweep is in fractions while `result.xpts` is the original 0..30000 axis.
  - Compiler: a `Pulse` with a `Sweep` field lowers to `QickSweep1D(name,start,stop)` + registers
    `add_loop(name, points)` once; a `Sweep` spanning a coupler band boundary **raises** a clear error
    (spec §6.4).
  - `fitters.fit_rabi(xpts, y)` on synthetic decaying-sinusoid data recovers period/π-gain within tol
    (pure, no qick).
  - e2e: amplitude-Rabi program with `sweeps=[amp]` acquires a `MeasurementResult` of shape
    `(n_ro, 51)` on the mock.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** `Sweep` dataclass; compiler `Sweep→QickSweep1D` + auto `add_loop` +
  band-boundary rejection (evaluate `wiring.resolve` across `[start,stop]`); `fit_rabi` as an
  `lmfit.Model` (decaying sinusoid) returning a typed `FitResult` (params + stderr + derived π/π-2 gain);
  `amplitude_rabi.py` experiment (Runner builds the program, Analyzer calls `fit_rabi`).

- [ ] **Step 4: Run — expect PASS.** — [ ] **Step 5: Commit** `feat: slice 2 — sweeps + Rabi fitter + amplitude_rabi`

---

## Slice 3 — T1 + T2 Ramsey (delays, phase, fitters)

### Task 12: swept-delay lowering + T1/T2 fitters + experiments

**Depends on:** [11] · **Parallel-safe:** true · **Touches:** []

**Files:** Modify `strumento/core/compiler.py`, `strumento/core/fitters.py`; Create
`strumento/packs/cqed/experiments/{t1,t2_ramsey}.py`; Test `tests/core/test_fitters.py` (extend),
`tests/e2e/test_slice3_t1_t2.py` (`qick`)

- [ ] **Step 1: Failing tests** — `Delay.duration_us` may be a `Sweep` (swept idle) and lowers to a
  swept `delay`; `fit_exp` (T1) and `fit_decaysin_detuned` (T2 Ramsey, recovers set detuning) on
  synthetic data within tol; e2e T1 (`Seq().play(ge_pi).wait(Sweep("t",0,500,101)).measure()`) and T2
  (hpi · swept wait · hpi(phase)) return correctly-shaped results.

- [ ] **Step 2–4: FAIL → implement (`fit_exp`, `fit_decaysin_detuned`, swept-delay lowering, the two
  experiment modules) → PASS.**

- [ ] **Step 5: Commit** `feat: slice 3 — T1 + T2 Ramsey (swept delay/phase + fitters)`

---

## Slice 4 — f0g1 + storage swap (multi-line device model)

### Task 13: sideband/mode/coupler ops + `FlatTop` envelope registration + swap-dataset accessor

**Depends on:** [6,12] · **Parallel-safe:** true · **Touches:** []

> **Naming (D21):** modules are element-type/role-keyed — `sideband.py` (transmon sideband-generator
> ops), `mode.py` (`ModeOps` for `BosonicMode` — displace, **not** "manipulate"), `coupler.py`
> (`CouplerOps`). `dev.sideband`/`dev.manipulate` still appear for Schuster because the instance yaml
> declares those names/aliases. **Envelope (D22):** `FlatTop` is **registered into the compiler's
> lowering registry** (Task 8), not added as an `if`-branch. **Gain (D23):** all factories return
> fraction `gain`.

**Files:** Create `strumento/packs/cqed/library/{sideband,mode,coupler}.py`; Modify
`strumento/core/compiler.py` (register `FlatTop` lowering), `strumento/packs/cqed/calibration.py`
(multiphoton + swap-dataset accessor); Modify `strumento/core/device.py` (bind ops/aliases); Create
`devices/schuster_multimode/storage_swaps.csv`; Test `tests/packs/cqed/test_sideband_coupler.py`,
`tests/e2e/test_slice4_f0g1_swap.py` (`qick`)

- [ ] **Step 1: Failing tests.**
  - `dev.sideband.f0g1()` → `Pulse` with a `FlatTop(...)` envelope on `LineRef("qubit","sideband")`,
    freq/gain(fraction)/length from `calib.multiphoton['fn-gn+1'][0]`; `dev.sideband.multiphoton(
    transition, kind, photon)` indexes by photon number.
  - `dev.coupler.swap("M1-S1")` → `FlatTop` on the coupler line; freq/gain/length from the **swap-dataset
    accessor** (`get_freq/get_pi/get_gain`, transcribed from v1 `man1_storage_swap_dataset.csv`); the
    resolved DAC follows the band gate (both DAC 1 in this instance).
  - Compiler lowers a `FlatTop` envelope correctly (ramp gauss + flat) **via the registry** (assert the
    registry now has a `FlatTop` entry).
  - e2e: `ge_pi · ef_pi · f0g1 · coupler.swap("M1-S1")` compiles + acquires on the mock.

- [ ] **Step 2–4: FAIL → implement (`f0g1`/`multiphoton`/`swap`/`displace` factories on `*Ops` classes,
  `FlatTop` registry lowering, the CSV-backed `StorageSwapDataset` accessor, ops/alias binding) → PASS.**

- [ ] **Step 5: Commit** `feat: slice 4 — f0g1 + storage swap (sideband/coupler/manipulate)`

---

## Slice 5 — Parity (composite factory)

### Task 14: `dev.parity` composite + revival-time calib + parity fitter

**Depends on:** [13] · **Parallel-safe:** true · **Touches:** []

**Files:** Create `strumento/packs/cqed/library/composite.py`; Modify `calibration.py` (revival times),
`fitters.py` (parity contrast/phase); Create `strumento/packs/cqed/experiments/parity.py`;
Test `tests/packs/cqed/test_parity.py`, `tests/e2e/test_slice5_parity.py` (`qick`)

- [ ] **Step 1: Failing tests** — `dev.parity(mode="manipulate")` returns a `Seq` = `hpi ·
  wait(revival_time[0]) · hpi(second_phase)` (values from the named mode's `revival_time` calib;
  **`mode=` names an element, not `man=1`** — D21); `dev.parity(mode="manipulate", fast=True)` uses the
  fast revival + AC-Stark phase; `.play(dev.parity(mode="manipulate"))` slots into a `Seq`; e2e parity
  experiment compiles + acquires; `fit_parity` recovers contrast/phase on synthetic data.

- [ ] **Step 2–4: FAIL → implement (`parity(dev, mode, *, second_phase=180, fast=False)` composite
  returning a `Seq`, revival-time calib fields, `fit_parity`, parity experiment) → PASS.**

- [ ] **Step 5: Commit** `feat: slice 5 — parity composite + fitter`

---

## Slice 6 — Optimal-control import (the Piccolo × QICK seam)

### Task 15: `from_solution` import + `Arb` lowering + `dev.prepare` + simultaneity

**Depends on:** [8,13] · **Parallel-safe:** true · **Touches:** []

> **Gain (D23):** `from_solution`'s `gain_calib` maps physical amplitude **straight to a v2 fraction**
> (no int-code hop) — the OC path is fraction-native. **Envelope (D22):** an imported solution lowers to
> `Arb(idata, qdata)` envelopes. **`PolyChirp` is NOT in this task** — it is a tweezer-pack envelope
> (registered there, later plan); the compiler's registry (Task 8) is what makes adding it a pack-local
> change, so no core stub is needed here.

**Files:** Modify `strumento/core/pulses.py` (`from_solution`), `strumento/core/compiler.py` (register
`Arb` lowering), `strumento/packs/cqed/calibration.py` (physical→fraction π-ref), `library/composite.py`
(`prepare`); Create `strumento/packs/cqed/experiments/prepare_fock.py`;
Test `tests/core/test_from_solution.py` (pure), `tests/e2e/test_slice6_oc_import.py` (`qick`)

- [ ] **Step 1: Failing tests (pure where possible).**
  - `from_solution(solution, drive_map, gain_calib, dac_rate)` on a small synthetic 2-drive solution
    (qubit + manipulate control arrays + times) returns a `Seq` wrapping a `Segment` of two `Arb`
    `Pulse`s (one per line, same `start_us`; per-pulse `start_us` offset honored), with provenance in
    `.meta`; samples resampled to the DAC grid; physical amplitude → **fraction** via the π-pulse
    reference (`gain_calib`, D23).
  - `dev.prepare(Fock(1))` resolves to either the analytic ladder or an OC `from_solution` result
    (`via="oc"` forces OC); returns a `Seq`.
  - Compiler lowers an `Arb` `Pulse` via the registry (`add_pulse` idata/qdata × fraction gain).
  - e2e: `.play_parallel(dev.prepare(Fock(1), via="oc"))` (or `.play(...)`) compiles + acquires on the
    mock; numeric-golden vs a known solution's realized envelope.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** `from_solution` (drive→line map, resample, physical→**fraction** gain via
  π-ref, `Arb` envelopes, `Segment`, provenance); the physical→fraction `PiPulseReference` in
  `calibration` (produced by the Rabi calib, spec §16.1); `prepare(dev, target, *, via=None)` composite;
  register `Arb` lowering in the compiler. *(π-ref/gain matches the `expt_service/gain.py` convention —
  spec §16.1.)*

- [ ] **Step 4: Run — expect PASS.** — [ ] **Step 5: Commit** `feat: slice 6 — from_solution OC import + dev.prepare`

---

## Definition of done (this plan)

- All six slices' e2e tests green on `MockQickSocV2` (**structural gate**, spec §10.1).
- **Numeric-golden gate** (spec §10.2): for ge_pi / f0g1 / storage-swap / an OC pulse, `quantize()`
  realized values (freq, gain-frac, cycles, envelope samples) match hand-computed / v1-derived goldens.
  With the committed `dump_cfg()` snapshot (Task 0/7), this compiles a **real `AveragerProgramV2` offline**
  (`QickConfig(json)`, no board — spec D25), so the gate validates actual v2 assembly, not just shapes.
- `core` imports with `qick` absent (pure layers); qick-facing tests marked `qick`.
- **Physics-agreement gate (spec §10.3) is explicitly OUT of scope here** — it needs the real board +
  the collaboration (Stanford/Sho); this plan stops at mock + numeric-golden.

## Deferred to later plans (not this one)

- Slice **7 / active_reset**: 7a herald/post-selection data path + host-side fallback; **7b** in-sequence
  v2 measurement feedback (gated on Sho, spec R1).
- **Slice 8 / codesign enablement** (spec §19, D26–D29) — see the deferred task sketch below.
- `tweezer` pack (§17), `spin` pack (§15.5).
- expt_service unification (§16), config versioning mechanism (§14.1), job-server integration (§14.2).
- `dev.coupler.floquet_swap`, catalog auto-retrieval for `dev.prepare`, `register_long_pulse` long drives.

### Slice 8 — codesign enablement (DEFERRED; off the slices-1–6 critical path)

*strumento as a device-model measurement instrument (spec §19). Not started until the core is green; the
`DeviceModel`/board-description tasks are pure and could start earlier if prioritized, but they are not
required for first light. Task numbers are sketch-level (S8.x); promote to a dedicated plan when the
codesign work is scheduled.*

- [ ] **S8.1 — `core/device_model.py`: `DeviceModel` + provenance + diff (pure, D26).** Pydantic
  `DeviceModel` (elements → params: f_ge/anharmonicity/couplings/T1/T2; per-line `TransferFn`; crosstalk
  matrix; `source: design|measured|hybrid`; provenance). `diff(design, measured)` → per-parameter deltas
  with uncertainties. `Device` gains `dev.model`. Tests: round-trip JSON; diff on a synthetic
  design-vs-measured pair. *No qick.*
- [ ] **S8.2 — `DeviceModelReport` from the Analyzer (pure, D26).** Collect the slice-2–5 fitter outputs
  (f_ge, T1, T2, revival, …) into a measured `DeviceModel`, wrap as a `DeviceModelReport` (measured model
  + `diff` vs the loaded design model + environment stamp, D24). Tests: a mock characterization run emits
  a report whose params match the fits. *Depends on S8.1 + the fitters (Tasks 11–14).*
- [ ] **S8.3 — board-description schema + `Wiring.from_board_description` (pure, D27).** Neutral Pydantic
  board-description (channel roles + generator→DAC/ADC bindings + RF-periphery annotations → §18
  `SignalChain`). Builder → `Wiring`. Tests: round-trip a board description → `Wiring` for the schuster
  instance; edit-one-binding leaves pulse factories untouched. *Depends on Task 1 (`wiring.py`).*
- [ ] **S8.4 — `TransferFn`/crosstalk on `SignalChain` + pre-distortion at `from_solution` (D28).**
  Add `TransferFn` (H(ω)) + crosstalk to the `SignalChain` (§18); add an optional pre-distortion stage to
  `from_solution` (regularized deconvolution, out-of-band/DAC-clip guards). Tests: deconvolution recovers
  a known waveform through a synthetic H(ω) within tol; guards fire on out-of-band/clip. *Depends on
  Task 15 (`from_solution`) + S8.1 (`TransferFn` lives on the model too).*
- [ ] **S8.5 — `core/sim.py`: `SimulatorSoc` backend (D29).** Implement the soc `acquire` protocol
  against a physics simulation (`DeviceModel` → Hamiltonian; Harmoniqs sim backend). A `source="design"`
  `Device` runs the same experiments/fitters and produces a `DeviceModelReport` with no board. Tests: a
  design-time device round-trips a Rabi/T1 through the simulator and the fitters recover the injected
  params. *Depends on S8.1 + the program/experiment layer (Tasks 9, 17). Physics backend is the heavy
  part — scope it when scheduled.*

---

## Plan self-review (round 1) — findings & additions

*Reviewed inline (org spend limit blocked the reviewer subagent). Found 4 should-fix gaps vs the spec;
added Tasks 16–18 and an annotation correction. Ordering is by `Depends on`, not document position —
the added tasks slot where their dependencies say.*

**Annotation correction (G4 — DAG accuracy).** Several tasks modify **shared files** yet were marked
`Parallel-safe: true` with empty `Touches`. For **sequential (v0.1) execution this is harmless** (they
are ordered by `Depends on`), but the DAG annotations are inaccurate for future parallel runs. The
shared files and their writers: `core/compiler.py` (Tasks 8, 11, 12, 13, 15), `packs/cqed/
calibration.py` (4, 13, 14, 15), `core/device.py` (5, 6, 13, 16), `core/fitters.py` (11, 12, 14),
`core/experiment.py` (17, 19), `core/pulses.py` (2, 15). (Task 18 is fitters-*test*-only; Task 17's
Analyzer wires the fitters but doesn't edit `fitters.py`.)
**Fix for parallel execution:** any two tasks writing the same file must not be in the same parallel
frontier — add the file to `Touches` and/or chain `Depends on`. Left as-is for v0.1 sequential.

### Task 16: `core/readout.py` protocols + `packs/cqed/readout.py` DispersiveIQ (the readout-kind seam)

**Depends on:** [3, 9]  (needs `MeasurementResult`; slots **before Slice 2 uses populations**)
**Parallel-safe:** false · **Touches:** [core/device.py]  (attach `dev.readout` namespace)

**Gap:** the spec's **readout-kind extension point** (§15.2, §16.2) — a spec centerpiece and the thing
that makes `core` modality-general — had **no task**. Slice 1 only did a raw `Measure→trigger`.

**Files:** Create `strumento/core/readout.py`, `strumento/packs/cqed/readout.py`; Modify `core/device.py`;
Test `tests/core/test_readout.py` (pure), `tests/packs/cqed/test_dispersive_iq.py` (pure)

- [ ] **Step 1: Failing tests** — `core.readout` defines `ReadoutKind` + `Discriminator` **protocols**
  and a **data-source** enum (`QICK_ADC` | `EXTERNAL_DETECTOR`) — spec §17 finding (a), so the tweezer
  pack slots in later. `packs/cqed/readout.DispersiveIQ` implements the protocol: takes raw IQ + a
  threshold/angle, `discriminate()` → per-shot g/e, `reduce()` → populations. Test on synthetic IQ
  blobs that a known-separated distribution discriminates correctly and reduces to the right population.
- [ ] **Step 2–4: FAIL → implement (protocols + `DispersiveIQ` + `dev.readout.measure()` producing a
  `Measure` op carrying the kind) → PASS.**
- [ ] **Step 5: Commit** `feat: readout-kind seam (core protocols + cqed DispersiveIQ discriminator)`

> After Task 16, wire slice-2/3 fitters to fit **populations** (via DispersiveIQ.reduce) as well as raw
> IQ, so results match the QILC measurement-vector convention (spec §16.2).

### Task 17: Data-lifecycle scaffold — Experiment / Runner / Analyzer / Display (spec §9.1, D13)

**Depends on:** [9]  (slots **before Task 11**; Slice-2+ experiment modules subclass this)
**Parallel-safe:** true · **Touches:** []

**Gap:** the spec keeps the high-level `Experiment` contract but **splits internals** into
Runner/Analyzer/Display (§9.1, D13). The per-experiment modules (amplitude_rabi, t1, …) were created
ad hoc with no shared scaffold.

**Files:** Create `strumento/core/experiment.py`; Test `tests/core/test_experiment.py`

- [ ] **Step 1: Failing test** — `Experiment` base exposes `go(save, analyze, display)` → `acquire()`
  (Runner: build `StrumentoProgram` + acquire → `MeasurementResult`) / `analyze()` (Analyzer: fitters →
  `FitResult`, **no plotting/printing**) / `display()` (Display: figure from result+fit). A trivial
  subclass runs end-to-end on the mock and returns a typed `FitResult` (derived quantities as fields,
  not stdout).
- [ ] **Step 2–4: FAIL → implement the base + the three collaborator seams → PASS.**
- [ ] **Step 5: Commit** `feat(core): Experiment/Runner/Analyzer/Display data-lifecycle scaffold`

> Retrofit the Slice-2–6 experiment modules (Tasks 11–15) to subclass `Experiment` rather than
> re-implementing acquire/analyze inline.

### Task 18: Fitter oracle cross-check vs v1 `fitting.py` (spec §9.3, D4)

**Depends on:** [11, 12, 14]  (after the fitters exist)
**Parallel-safe:** false · **Touches:** [core/fitters.py test only]  · **marker:** `oracle`

**Gap:** D4 says validate the new `lmfit` fitters against the **current `fitting.py` as a physics
oracle**; the plan only tested against synthetic data.

**Files:** Test `tests/oracle/test_fitters_vs_v1.py`  (guarded — imports v1 `fitting.py` from a
configurable path; skip if unavailable)

- [ ] **Step 1: Failing test** — for Rabi / T1 / T2 / parity, fit the **same** data array with the new
  `strumento` fitter and the v1 `fitting.py` function (`fitdecaysin`/`fitexp`/…) and assert the recovered
  parameters agree within tolerance. Data: recorded/synthetic arrays checked into `tests/oracle/data/`.
  *(v1 source: `.../multimode_expts/fitting/fitting.py` — provide its path via env/fixture.)*
- [ ] **Step 2–4: FAIL → adjust fitter param conventions until they agree with the oracle → PASS.**
- [ ] **Step 5: Commit** `test(oracle): new lmfit fitters agree with v1 fitting.py`

---

## Plan review (round 2) — naming / extensibility / elegance / station seam (2026-07-17)

*Threaded the spec's D21–D24 (added in the same review round) into the tasks above and added Task 19.*

- **D21 topology-derived namespaces** → Tasks 4 (`ops_cls` on element types), 5 (bind `dev.<name>` per
  named element + aliases), 6 (`TransmonOps`, not `QubitNamespace`), 13 (`ModeOps`/`CouplerOps`), 14
  (`parity(mode=...)`).
- **D22 typed envelope union** → Task 2 (frozen `Gaussian`/`FlatTop`/`Const`/`Arb`), Task 8 (lowering
  **registry** + dedup on the frozen envelope), 13 (`FlatTop` registered), 15 (`Arb`; `PolyChirp`
  explicitly deferred to the tweezer pack).
- **D23 v2-fraction gain in the IR** → conversion moved **out of the compiler** (Task 8) into the
  **cqed factory/calibration** (Tasks 4, 6); `from_solution` maps physical→fraction directly (15);
  gain-sweep records the lab-native axis for `xpts` (11).
- **D24 environment/station seam** → `wiring.Line.chain` stub (Task 1), `meta["environment"]` reserved
  (Task 3), and **Task 19** below.
- **Rename** `core/channels.py` → `core/wiring.py`, `Readout` → `ReadoutWiring` (Task 1 + file tree) —
  the module modeling "not the FPGA channels" shouldn't be named `channels`, and `ReadoutWiring` avoids
  colliding with `core/readout.py`'s `ReadoutKind`.

### Task 19: `core/environment.py` — EnvironmentSource protocol + Runner stamping (§18/D24)

**Depends on:** [3, 17]  (needs `MeasurementResult.meta` + the Runner in the lifecycle scaffold)
**Parallel-safe:** false · **Touches:** [core/experiment.py]  (Runner stamps at acquire)

**Why now (not later):** result stamping is the one **retroactively-impossible** primitive — data taken
without an `environment` stamp can never be re-stamped. The protocol is ~20 lines and pure; landing it
now means every acquire from day one *can* carry station telemetry (drift-vs-temperature is exactly
what QILC trend analysis + fleet device-modeling consume). Fridge/instrument **control stays out of
scope** (§18.3) — this is read-only telemetry only.

**Files:** Create `strumento/core/environment.py`; Modify `strumento/core/experiment.py` (Runner stamp);
Test `tests/core/test_environment.py` (pure)

- [ ] **Step 1: Failing tests** — `EnvironmentSource` is a `Protocol` with `snapshot() ->
  dict[str, float|str]`; a `MockEnvironmentSource` returns canned telemetry; the Runner, given one (or
  a list), writes `result.meta["environment"]` at acquire start (schema-light dict). With **no** source
  configured, `meta` has no `"environment"` key (zero-overhead default). Assert the stamp round-trips
  and the no-source path is clean.
- [ ] **Step 2–4: FAIL → implement the `Protocol` + `MockEnvironmentSource` + Runner stamping hook →
  PASS.**
- [ ] **Step 5: Commit** `feat(core): EnvironmentSource protocol + Runner result stamping (§18)`

> Deferred (spec §18.3): a QCoDeS-instrument adapter, real fridge-vendor (Maybell/Bluefors) adapters,
> the `chain:` signal-chain content (Task 1 only stubs the field), condition gating (job-server layer),
> and any `Station` abstraction.

### Nice-to-have (not blocking; fold in opportunistically)

- **`Measure` op** now appears in the spec's §6.1 IR (added round 2) and in Task 2 — keep the
  `pulses.py` docstring note that it is the readout op the builder's `.measure()` emits.
- **Public API** — define `strumento/__init__.py` and `strumento/core/__init__.py` exports (`Device`, `Seq`,
  `Sweep`, `StrumentoProgram`, `MockQickSocV2`, `MeasurementResult`, and the envelopes `Gaussian`/`FlatTop`
  /`Const`/`Arb`) as part of Task 10.
- **Pack discovery via entry points** (spec §15.3) — register the `cqed` pack under a `strumento.packs`
  entry-point group so `Device.load(pack="cqed")` resolves it, and third-party packs install rather than
  fork. A one-liner in `pyproject.toml` (Task 0) + resolution in `device.load` (Task 5); fine to defer to
  a follow-on if it complicates v0.1, but the seam is cheap to reserve now.
- **Tasks 11/13/15 are coarse** (a whole slice each). Fine for v0.1 sequential; split into per-file
  sub-tasks if handing to parallel `develop` mode.
