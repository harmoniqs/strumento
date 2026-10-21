# Contributing

## Setup

Python 3.10–3.12. CI runs the pure lane on all three versions and the qick and mcp lanes on 3.10
and 3.11.

```bash
pip install -e ".[test]"        # pure layers
pip install -e ".[test,qick]"   # + the qick-facing layers (x86-safe; no board needed)
pip install -e ".[test,qick,plot]"  # + matplotlib, so the plot tests run (as in CI's qick lane)
pytest -q && ruff check .
```

## How work lands

- **TDD**: failing test → minimal implementation → green suite → commit. Small, focused PRs.
- CI must be green on all four hosted jobs (`lint`, `pure`, `qick`, `mcp`). The `pure` job runs
  **without qick installed** — mark anything qick-touching `@pytest.mark.qick`. Reproduce each
  test lane locally (with `[test,qick,plot]` installed; the `pure` lane CI runs has no
  matplotlib, so its plot tests skip):

  | Lane | Command |
  |---|---|
  | pure | `pytest --block-qick -m "not qick"` — `--block-qick` makes `import qick` fail, as in CI |
  | qick | `pytest -m "not mcp"` |
  | mcp | `pytest -m mcp` — needs the `[mcp]` extra too |
- Style is ruff-enforced (`pyproject.toml`); line length 100; no manual formatting debates.
- Conventional-ish commit subjects (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `ci:`).
- A user-visible change adds a line to [CHANGELOG.md](CHANGELOG.md) under `## [Unreleased]`, in
  its `Added` / `Changed` / `Fixed` / `Removed` subsection (create the subsection if missing).

## Docs snippets

`tests/test_docs.py` runs every fenced `python` block in the user-facing docs (the `DOCUMENTS`
list there — add a path to cover another doc). It is a qick-lane test (the snippets run against
the simulator) and skips without qick. Each block is a test named `<doc>:<line>` (its opening
fence). A document's blocks run once, top to bottom, in **one shared namespace**, from a temp
mirror of the repo root, so `Device.load("devices/…")` works but relative writes land in a temp
dir; `HOME`/`TMPDIR` are temp dirs, the network is blocked, and a `<doc>:writes` test fails if
any repo file changed. A failing block names the earlier blocks that did not complete. A block
that fails only because `matplotlib` or `mcp` is missing is skipped with that reason.

Markers are HTML comments (invisible when rendered) on the line before a fence:

| Marker | Effect |
|---|---|
| `<!-- snippet: skip reason="hardware only" -->` | not run; reported as skipped |
| `<!-- snippet: xfail reason="NameError: soc undefined; #139" -->` | run as a **strict** xfail: it must fail, and the test fails once it passes (remove the marker). The reason must cite the issue that clears it |
| `<!-- snippet: output -->` before a ` ```text ` block right after a python block | the python block's printed stdout must match (exact, trailing whitespace and edge blank lines ignored) |
| `<!-- snippet: output rtol=1e-3 atol=0.5 -->` | as above, but numbers compare with `math.isclose`; the text between them stays exact |

A marker without its reason, an unknown marker or attribute, or a marker not attached to a block
fails the test. Bare expressions (`dev.calib.qubit.f_ge  # 4000.0`) are not checked — use
`print` plus an output block when the value matters. Details: `tests/_snippets.py`.

## Compiled-program snapshot gate

`tests/packs/test_compiled_snapshot_gate.py` (qick lane) compiles every registered experiment,
with its experiment-contract arguments, on the shipped demo its contract run uses, and compares
the compiled program — qick's own `dump_prog()` payload (assembly, waveforms, envelopes,
generator and readout declarations, loop dimensions) plus the acquire block — with a reference
in `tests/golden/compiled/<pack>/<device>/<experiment>.json`. It is how a refactor proves it
doesn't change what a board would play, without a board. A failure names the pack, device and
experiment, and lists the differing JSON paths as `old → new`.

The references are **local and gitignored, never committed**. Before a refactor:

```bash
# (a) on main: generate the references
pytest tests/packs/test_compiled_snapshot_gate.py --update-compiled-golden
# (b) on your branch: compare; any diff is a change to what is played
pytest tests/packs/test_compiled_snapshot_gate.py
```

(c) A refactor PR states that it ran the gate and it was clean. A PR that intends to change what
is played (a new pulse default, a timing fix, a new experiment, a qick upgrade that changes the
assembly) names the experiments whose programs change, and says why. For a refactor, a diff is a
bug.

With no references on disk the comparison tests skip and say how to generate them, so the
default suite and CI are unaffected; a partial set fails. The flag rewrites only the files that
differ (and creates missing ones), and lists them at the end of the run; without it the gate never
writes. An experiment that can't compile on its contract device goes in `CANNOT_COMPILE` in
`tests/_compiled_golden.py`, with the reason. That module also documents the experiment-to-device
matrix and the determinism checks.

## Design rules (the short list)

The architecture and its decision index live in [docs/architecture.md](docs/architecture.md);
[AGENTS.md](AGENTS.md) has the hard rules. The three most commonly relevant:

1. Only `core/compiler`, `core/program`, `core/mock` and `core/simulator` may import qick.
2. `core` stays physics-free — element types, libraries, readout kinds, calib schemas are pack
   territory; packs are discovered via entry points.
3. `Pulse.gain` is a v2 fraction everywhere inside the IR; unit conversions live at boundaries.

## Extending

Recipes for new factories, envelopes, experiments, fitters, packs, and device instances are in
[AGENTS.md](AGENTS.md#extension-recipes) and [docs/packs-and-devices.md](docs/packs-and-devices.md).

## Hardware testing

You don't need a board for anything on the main path (real v2 assembly runs offline against the
committed testbench config). If you *have* a board: drop its `dump_cfg()` at
`devices/_fixtures/soccfg_v2.json` to point the v2 suite at your firmware, and see
[docs/hardware-ci.md](docs/hardware-ci.md) for the self-hosted runner pattern.
