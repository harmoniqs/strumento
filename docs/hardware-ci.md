# Board-in-the-loop CI

strumento's CI runs real tProc-v2 **assembly** on hosted runners (no hardware — the committed
testbench soccfg makes `AveragerProgramV2` construction a pure-software affair). The next rung —
running the `qick`-marked suite on a real RFSoC — follows the pattern the upstream qick project
has proven in production since 2024: **the GitHub Actions runner lives on the board itself.**

## The upstream pattern (what we copy)

qick's hardware workflow runs on a self-hosted runner registered on the board's PYNQ Linux,
label-addressed (`runs-on: [zcu111]`), executing loopback-wired demo notebooks on every relevant
PR — hundreds of green runs on a single runner. The non-obvious lessons are encoded in our
`hardware.yml` already:

- **Everything under `sudo -E`** — FPGA access (`/dev/mem`, overlays) needs root.
- **Manual `sudo git clean -ffdx` before checkout** — root-owned files from the previous run
  break the checkout action's own clean step.
- **One board = a `concurrency` group** — jobs serialize instead of fighting over the FPGA.
- **Artifact-upload the results** so failures are inspectable.
- Upstream's v2 material has **no hardware CI of its own** — a board runner here tests a path
  the ecosystem doesn't otherwise cover.

## Standing up a runner

1. **Board**: ZCU216 / RFSoC4x2-class on standard PYNQ Linux, outbound HTTPS, DAC→ADC loopback
   cabling on the channels under test.
2. **Runner**: install the aarch64 GitHub Actions runner on the board, register it to this repo
   with labels `self-hosted, qick` (add a board-model label if a fleet ever exists), run it as a
   service. The only secret is the one-time registration token.
3. **Enable the workflow**: set the repo variable `HAS_QICK_RUNNER=true`. Until then
   `hardware.yml` is inert by design.
4. **Capture the board's identity**: `QickSoc.dump_cfg()` → commit *privately* or drop at
   `devices/_fixtures/soccfg_v2.json` locally — the v2 test gate then runs against the real
   firmware config even off-board.

## Triggers and safety

- `workflow_dispatch` (manual) and a nightly cron keep the board honest.
- On PRs the job is **label-gated** (`run-hardware`) *and* fork-guarded
  (`head.repo.full_name == github.repository`) — fork code never executes on the physical
  runner. Keep "require approval for outside collaborators" on in repo settings.

## What runs at each rung

| rung | runner | suite |
|---|---|---|
| assembly (today) | hosted | full suite incl. real v2 assembly vs the testbench soccfg |
| loopback | board | `pytest -m qick` — acquire paths against real DAC→ADC loopback |
| physics gate (§10.3) | board + fridge | T1/T2/Rabi with sensible physics + agreement with a trusted stack — the real acceptance for any lab deployment |

The physics gate is a collaboration milestone, not a CI switch: it needs a wired-up quantum
device and a reference to agree with.
