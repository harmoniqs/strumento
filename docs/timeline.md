# Timeline semantics — the cursor model

This is the one thing an experiment author must understand precisely: **when does each op play?**
strumento defines the answer rather than leaving it implicit in qick's scheduler.

## The model

A `Seq` is lowered against a single **reference cursor** that advances down the sequence. There is
exactly one rule per op kind:

| op | timing | effect on the cursor |
|---|---|---|
| `.play(pulse)` — sequential (`start_us is None`) | plays *at* the cursor | **advances** past the pulse (duration-aware) |
| `.play(pulse)` — absolute (`start_us` set) | plays at `start_us` relative to the cursor | does **not** advance |
| `.play_parallel([...])` / `Segment` | all pulses at the cursor (offset by each pulse's `start_us`) | advances **once** past the longest |
| `.wait(dt)` / `Delay(dt)` | — | advances by exactly `dt` (µs) |
| `.measure()` | triggers readout at the cursor | (readout has its own trigger offset) |
| `.raw(fn)` | whatever `fn` does | whatever `fn` does |

The load-bearing consequence: **`play(a).play(b)` means "b after a"** with no length arithmetic by
the caller. The compiler emits a duration-aware `delay_auto(0)` after each sequential pulse, so the
cursor lands exactly at the end of `a` (qick computes the pulse's realized length from its
envelope). Two π-pulses on the same line are serialized, not overlapped — the thing a naive
"schedule everything at t=0" lowering gets silently wrong.

Simultaneity is opt-in and explicit: a `Segment` (`play_parallel`) is how you drive multiple lines
at the same instant — e.g. an optimal-control pulse spanning qubit + cavity, which `from_solution`
returns as one `Segment`.

```python
Seq().play(dev.qubit.ge_pi()).play(dev.qubit.ef_pi()).measure()
#    ^ ge_pi at cursor    ^ cursor advanced past ge_pi, ef_pi plays after    ^ readout after ef_pi

Seq().play_parallel([drive_pulse, cavity_pulse]).measure()
#    ^ both at the same cursor; readout after the longer of the two
```

## What "absolute" currently means (and the honest limitation)

`start_us` lowers to qick's `pulse(t=start_us)`, and qick's `t` is **relative to the current
cursor**, not the program origin. So `start_us` today means "offset from the current cursor,
without advancing it" — useful for staggering a pulse against the sequential flow, but it is *not*
yet a program-absolute coordinate. True origin-absolute scheduling (reset the reference to zero)
is available through the escape hatch (`.raw(lambda p: p.delay(...))` / hand-written `_body`) and
is a candidate refinement. This limitation is stated rather than hidden — knowing it is why the
model is worth writing down.

## Where the scheduler stops (deliberately)

The cursor model is a *sequencer*, not a *scheduler*. It does not do cross-channel alignment
barriers, per-channel independent cursors, or global duration solving — the machinery a
surface-code cycle (many couplers + readouts + feed-forward aligned to the sample) will eventually
want. That is a distinct layer that would *consume* `Seq`, not a rewrite of it, and it is out of
scope until a workload needs it. Keeping the sequencer's semantics crisp now is what makes adding a
scheduler later a clean extension rather than a reinterpretation.

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
