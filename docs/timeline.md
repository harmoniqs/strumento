# Timeline semantics — the cursor model

This is the one thing an experiment author must understand precisely: **when does each op play?**
strumento defines the answer rather than leaving it implicit in qick's scheduler.

This document is the complete writeup: the model, the exact lowering for every op (with
citations to the code that does the thing), the edge cases — including the dumb things — and
direct answers to the questions raised in review. Citations are `file::symbol` plus the emitted
call, so they stay findable as lines move.

## The model

A `Seq` lowers against a single **reference cursor** that advances down the sequence
(`compiler.py::Compiler.play`). There is exactly one rule per op kind:

| op | plays at | effect on the cursor | exact lowering |
|---|---|---|---|
| `.play(pulse)` — sequential (`start_us is None`) | the cursor | **advances** past the pulse | `pulse(ch, name, t=0)` then `delay_auto(0.0)` |
| `.play(pulse)` — absolute (`start_us` set) | cursor + `start_us` | does **not** advance | `pulse(ch, name, t=start_us)` — nothing after |
| `.play_parallel([...])` / `Segment` | the cursor (each pulse offset by its own `start_us`) | advances **once** past the longest | `pulse(ch, name, t=offset)` per pulse, then one `delay_auto(0.0)` |
| `.wait(dt)` / `Delay(dt)` | — | advances by exactly `dt` (µs) | `delay(dt)` — scalar or swept |
| `.measure()` | the cursor + the resolved `trig_offset` | **advances** past the readout window | `trigger(ros=[adc], t=trig_offset)` then `delay_auto(0.0, gens=False, ros=True)` |
| `.raw(fn)` | — | whatever `fn` does | `fn(program)` |
| `.branch(...)` | the cursor (readout happens mid-sequence) | the **taken arm** continues the cursor | `read_and_jump` / `jump` / `label` (below) |
| `.repeat(body, n)` | the cursor, `n` times | the body advances it; it carries across iterations | `open_loop(n)` + body + `close_loop` |

The load-bearing consequence: **`play(a).play(b)` means "b after a"** with no length arithmetic by
the caller. The compiler emits a duration-aware `delay_auto(0)` after each sequential pulse, so the
cursor lands exactly at the end of `a` (qick computes the pulse's realized length from its
envelope). Two π-pulses on the same line are serialized, not overlapped — the thing a naive
"schedule everything at t=0" lowering gets silently wrong.

A readout advances the cursor the same way, past its **declared window**, so `measure()` composes
like any other op: `.measure().measure()` gives two correctly-spaced reads, and
`.measure().wait(settle).branch(...)` waits `settle` *after the window* rather than needing the
caller to add the window in. The compiler emits `delay_auto(0, gens=False, ros=True)` — `ros=True`
so qick's own window tracking is the single authority on the length (no second belief to drift),
and `gens=False` so an absolutely-placed pulse can never drag the cursor, which would break the
"absolute does not advance" row above.

## The exact lowering, op by op

**Sequential pulse** — `compiler.py::Compiler.play`, `Pulse` branch with `start_us is None`:
`_play_pulse` emits `program.pulse(ch=ch, name=name, t=0.0)`, then `play` emits
`program.delay_auto(0.0)`. `delay_auto(0)` means "set the reference to the end of the last
pulse/readout" (qick's own duration bookkeeping — `gens=True, ros=True` by default), which is how
*b after a* works without the caller ever computing a length.

**Absolutely-placed pulse** — same branch, `start_us` set: `t=(p.start_us or 0.0)`, and **no
`delay_auto` follows**. The pulse plays at the cursor plus the offset; the cursor itself is
untouched. If the pulse *overruns* the cursor (extends past it), the next op can overlap it — see
the edge-case matrix.

**Segment** — `play`'s `Segment` branch: every pulse in the segment is emitted with its own
`start_us` as the `t` offset (all relative to the segment's common start = the cursor), then a
single `delay_auto(0.0)` advances past the *longest* pulse in the segment. Staggered starts inside
a segment are the intended use of `start_us` there.

**Delay** — the `Delay` branch: `program.delay(value)`, where `value` is a scalar or the
program's sweep handle for a swept duration (a swept idle is the T1/T2-Ramsey primitive). In qick
terms this is always `delay` — an increment of the reference time — never `wait` (see the
wait-vs-delay answer below).

**Measure** — the `Measure` branch: `program.trigger(ros=[adc], t=trig_offset)` fires the readout
`trig_offset` (µs) past the cursor — `Compiler.trig_offset`, resolved once by
`StrumentoProgram.__init__` (`program.py::_resolve_trig_offset`, #98): `calib.readout.trig_offset`
first, `wiring.readout.trig_offset` as fallback, `0.0` if neither is set (so an uncalibrated
device still triggers exactly at the cursor, as before). The window's length is the *declared*
readout length (`declare_readout(ch, length)` in
`Compiler.declare`, `length = wiring.readout.lengths_us[0]` else `DEFAULT_RO_LENGTH_US = 1.0`);
then `delay_auto(0.0, gens=False, ros=True)` advances past that window. `kind`/`params` on the
op configure the readout projection (result-layer), not the timing.

**Bound dynamic cQED readout** — `ReadoutOps.dispersive_measure()` builds a scalar readout probe
with `start_us=0.0` and a dynamic ADC binding. Before the probe, the compiler sends the readout
configuration at the same cursor; it then pulses the DAC and lets the ordinary `Measure` trigger
at the device's calibrated offset. This yields `configure -> pulse -> trigger`, with the trigger
offset measured from probe start rather than pulse end. The binding also supplies the declared ADC
window and tells QICK to frequency-match the DAC and ADC grids.

**Branch** — `compiler.py::Compiler._play_branch` lowers to real flow control, no test negation::

    read_and_jump(ro, comp, thresh, test, THEN)
    <else block>
    jump(END)
    label(THEN); <then block>
    label(END)

`test` is one of `>=` / `<` (the two comparisons v2's `read_and_jump` provides — invert the
branches for anything else; this is checked at lowering, `compiler.py::_V2_BRANCH_TESTS`). The
threshold is an integer in ADC-accumulator units; a float in the IR is realized with
`int(round(...))`.

**Repeat** — `compiler.py::Compiler._play_repeat`: `open_loop(count)` + the body + `close_loop`.
This is a bounded hardware loop *inside the shot* — it does **not** add a data dimension (the
acquisition loops are `reps` plus one per swept field, added by `add_loop` in the *declaration*
pass and wrapping the whole body). The cursor carries across iterations: a repeat of
`play(p).wait(t)` is a train with period `len(p) + t`, which is what makes it the DD-train
primitive.

**The scaffold around your body** — `program.py::StrumentoProgram._initialize` runs the declare
pass; `_body` runs the timeline. qick's `AveragerProgramV2.make_program` wraps `_body` with:
external counter setup → the acquisition loops (`reps`, outermost, plus one `open_loop` per swept
field) → your body → `final_delay` → counter increment → close loops. Two implicit tail
behaviors worth knowing: an `initial_delay` (None by default — nothing is emitted) and a
`final_delay` — the relax delay, taken from the device calibration's `readout.relax_delay`, else
1.0 µs (`program.py::_DEFAULT_FINAL_DELAY_US`). Every swept point replays the *whole* body; the
data memory advances one point per loop iteration.

## Edge cases — including the dumb things

Each row is real lowering behavior, not a promise about intent.

| you write | what happens | why |
|---|---|---|
| a sequential-looking pulse with `start_us=0.0` | the pulse plays *at* the cursor and **does not advance it** | the advance check is `start_us is None` (`compiler.py::Compiler.play`); `0.0` is a set `start_us`, i.e. absolute placement with offset zero — not a sequential play |
| an absolute pulse that overruns the cursor | the cursor stays put; the *next* op can land inside the pulse | no `delay_auto` follows absolute placement; overlap surfaces as a qick **logged** conflict, not an exception (a deliberately overlapping readout is guarded to still conflict in `tests/packs/test_experiment_contract.py`) |
| `play_parallel(some_seq)` where the Seq contains Delays/Measures | **the non-pulse ops are silently dropped** — only `Pulse` ops enter the segment | `pulses.py::Seq.play_parallel` filters `isinstance(op, Pulse)`; this is a sharp edge (arguably it should raise), not a feature |
| `play(some_seq)` | every op is spliced in — Delays, Measures, Branches, Repeats included | `pulses.py::Seq.play` extends with `item.ops` wholesale; a sub-sequence composes fully |
| `.measure().measure()` | two correctly-spaced reads — the second waits out the first's window | the `delay_auto(0, gens=False, ros=True)` after a Measure |
| `.measure()` for a second qubit *at the same instant* | **not expressible today** — `measure()` triggers only `wiring.readout.adc_chs[0]`, and only that one window is declared | single-ADC lowering in `Compiler.declare`/`play`; multi-ADC simultaneous readout needs `.raw()` (and a `declare_readout` per channel) until the wiring model grows it |
| a `Measure` inside a Branch arm or Repeat body | works; the readout is declared once for the whole program | `_needs_readout` recurses through Branch/Repeat (`compiler.py::Compiler._needs_readout`) |
| branch arms of different lengths | the cursor is wherever the **taken** arm left it — subsequent ops play at outcome-dependent times | the lowering has no re-convergence barrier; both arms continue the one shared reference |
| a swept `wait(t)` | **does not exist** — a swept `Delay` lowers to a swept `delay` (fine); qick's `wait` can only take a scalar (qick itself rounds a swept wait *up* to its max, with a warning) | `compiler.py::Compiler.play`, Delay branch; qick `Wait.preprocess` |
| an empty Seq, or `delay_auto` with no relevant channels | compiles to nothing; the cursor is unmoved | qick: "if this was a delay_auto and we have no relevant channels, it should compile to nothing" |
| a swept delay's *displayed* axis | the played values are register-step truncated (e.g. 10 µs steps are 4300 cycles at 430.08 MHz → the axis tops at 99.981 µs, not 100.0) | the sweep quantizes span/step in register space; reading the axis back from the compiled program is the D37 realized-axis work |

## Answers to the review's numbered questions

**1. Reliance on `auto`.** The compiler emits exactly **two** `auto` calls, both of the narrowest
possible form — "advance the reference to the end of what I just emitted": `delay_auto(0.0)`
after a sequential pulse or segment (advance past the pulse(s)), and
`delay_auto(0.0, gens=False, ros=True)` after a measure (advance past the readout window). No
user-visible timing is delegated to `auto`: sequencing, simultaneity, window accounting, and
staggering are all planned here, at the wrapper level, and absolute placement (`start_us`) plus
explicit `wait()` express hand-planned timelines when you want them. `auto` is used only as qick's
own duration bookkeeping for the thing just emitted — the one question qick is uniquely qualified
to answer (it knows the realized envelope length and the declared window; the wrapper does not
carry a second belief about either).

**2. Origin-absolute scheduling.** Not available as a first-class op: `start_us` is
**cursor-relative** (it lowers to qick's `pulse(t=...)`, and qick's `t` is relative to the current
reference time). "`.raw(lambda p: p.delay(...))`" meant: the `Raw` op calls your function with the
live qick program at that point in the sequence (`pulses.py::Raw`, `compiler.py::Compiler.play`),
so *any* v2 call — including origin-absolute idioms like a `sync_all()`-anchored timeline — is
reachable mid-sequence; a hand-written `_body` (subclassing `StrumentoProgram`) is the same
escape at program scope. True origin-absolute scheduling as a first-class primitive is a
candidate refinement, not a hidden claim: it is not in the model today.

**3. wait vs delay.** strumento's `.wait(dt)` **lowers to qick `delay`** — an increment of the
reference time; execution does not stall. It does *not* lower to qick's `wait` (pause tProc
execution until the time reaches a value). The builder method's name is a wart inherited from lab
idiom ("wait 500 µs before measuring"), and the IR op is honestly named `Delay`. A true execution
stall is reachable via `.raw(lambda p: p.wait(t))`. The distinction matters when something else
is in flight: `delay` only moves the reference (a subsequent pulse is scheduled later), `wait`
actually stalls the tProc (relevant e.g. for deterministic timing against external events).

**4. Delays inside `play()` / `play_parallel()`.** `play()` takes a `Pulse`, a `Segment`, or a
`Seq` (spliced whole — delays included); it does not take a bare delay — `.wait(dt)` is the
delay op. Inside `play_parallel(...)` there are no delays *by construction*: a segment is pulses
with per-pulse `start_us` offsets and one collective advance; passing a Seq filters it to pulses
(see the sharp edge above).

**5. Measuring multiple qubits, now vs later.** *Later* is first-class: `measure()` composes —
`.play(a).measure().play(b).measure()` gives two sequential reads, each after its own window, and
`.measure().wait(settle)...` waits *after* the window. *Simultaneously* is the honest gap: the
lowering triggers one ADC (`adc_chs[0]`) and declares one window; simultaneous multi-qubit
readout is not expressible through `measure()` today and needs `.raw()` with per-channel
`declare_readout`. Growing the wiring model to multi-ADC measure is the natural next step and is
tracked as such, not papered over.

## What "absolute" currently means (and the honest limitation)

`start_us` lowers to qick's `pulse(t=start_us)`, and qick's `t` is **relative to the current
cursor**, not the program origin. So `start_us` today means "offset from the current cursor,
without advancing it" — useful for staggering a pulse against the sequential flow (and inside
`Segment`s, where it is the intended stagger), but it is *not* yet a program-absolute coordinate.
True origin-absolute scheduling is available through the escape hatch (`.raw(...)` /
hand-written `_body`) and is a candidate refinement. This limitation is stated rather than
hidden — knowing it is why the model is worth writing down.

## Where the scheduler stops (deliberately)

The cursor model is a *sequencer*, not a *scheduler*. It does not do cross-channel alignment
barriers, per-channel independent cursors, or global duration solving — the machinery a
surface-code cycle (many couplers + readouts + feed-forward aligned to the sample) will eventually
want. That is a distinct layer that would *consume* `Seq`, not a rewrite of it, and it is out of
scope until a workload needs it. Keeping the sequencer's semantics crisp now is what makes adding
a scheduler later a clean extension rather than a reinterpretation.

## QICK-v2 redundancy audit

A fair question (raised in review): *how much of this is just qick v2 with dots instead of
newlines?* Each abstraction earns its place by doing something raw v2 does not — where it would
merely rename a v2 call, it is deliberately thin (Tenet 0: raw v2 is always one `.raw()` away).

| strumento | raw tProc v2 | why it's not redundant |
|---|---|---|
| `Seq` + cursor model | `pulse(t)` / `delay` / `delay_auto` on a program object | defines *when things play* as a rule set; duration-aware sequential advance is not v2's default (you'd hand-manage the reference) |
| typed `Pulse` + factories | `add_pulse(freq, gain, …)` with hand-gathered args | pulses are *computed from calibration*; the config→param gather (the bulk of a raw v2 program) collapses to one factory call |
| envelope union + registry | `add_gauss` / `add_pulse(style=…)` per shape | one dedup'd, hashable representation; packs add shapes without touching core |
| `Sweep` as a value | `add_loop` + `QickSweep1D` wired by hand | any field becomes swept uniformly; band-boundary crossings are rejected, not silently mis-lowered |
| wiring / `LineRef` | bare DAC channel integers | one channel map; re-patching a cable is a config edit, every experiment untouched |
| gain as a v2 fraction (D23) | int code *or* fraction depending on call | one unit in the IR; the int-code↔fraction bug class is eliminated at a single boundary |
| `MeasurementResult` + `postselect` | raw acquire arrays | typed axes + shot-resolved heralding the raw return doesn't carry |
| `.raw()` / IS-A `AveragerProgramV2` | — | **the redundancy escape valve**: anything above that you'd rather write in raw v2, you can, mid-sequence |

The verdict the audit is meant to enforce: an abstraction that only renames a v2 call should be a
thin pass-through with the escape hatch adjacent — and the ones above each do more than rename.
