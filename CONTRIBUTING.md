# Contributing

## Setup

Python ≥ 3.11.

```bash
pip install -e ".[test]"        # pure layers
pip install -e ".[test,qick]"   # + the qick-facing layers (x86-safe; no board needed)
pytest -q && ruff check .
```

## How work lands

- **TDD**: failing test → minimal implementation → green suite → commit. Small, focused PRs.
- CI must be green on all three hosted jobs (`lint`, `pure` on 3.11/3.12, `qick`). The `pure`
  job runs **without qick installed** — mark anything qick-touching `@pytest.mark.qick`.
- Style is ruff-enforced (`pyproject.toml`); line length 100; no manual formatting debates.
- Conventional-ish commit subjects (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `ci:`).

## Design rules (the short list)

The architecture and its decision index live in [docs/architecture.md](docs/architecture.md);
[AGENTS.md](AGENTS.md) has the hard rules. The three most commonly relevant:

1. Only `core/compiler`, `core/program`, `core/mock` import qick.
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
