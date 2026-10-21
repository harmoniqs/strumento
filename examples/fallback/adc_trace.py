#!/usr/bin/env python3
"""ADC probe — the 'is the ADC alive?' check.

Arms the readout at a given frequency and captures averaged IQ with no drive
pulse. On a real board an open input shows a noise floor (non-zero magnitude);
on the mock soc you get zeros — both are informative.

Bare AveragerProgramV2 — no strumento.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qick.asm_v2 import AveragerProgramV2

# allow running from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, load_soccfg, make_soc, report_single

# ──────────────────────────────────────────────────────────────────────
# Program
# ──────────────────────────────────────────────────────────────────────


class ADCProbe(AveragerProgramV2):
    """Arm the readout and capture IQ — no drive pulse, no sweep."""

    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        # Dynamic readouts (axis_dyn_readout_v1) reject freq/phase/sel at
        # declaration — those are set by tProc instructions, not here.
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

    def _body(self, cfg):
        self.trigger(ros=[cfg["ro_ch"]])


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main(soc=None):
    parser = base_parser("ADC probe — capture IQ with no drive pulse")
    parser.add_argument("--ro-freq", type=float, default=750.0, help="Readout freq in MHz (default: 750)")
    parser.add_argument("--ro-length", type=float, default=2.0, help="Readout window in us (default: 2.0)")
    args = parser.parse_args()

    cfg = {
        "ro_ch": args.ro_ch,
        "ro_freq": args.ro_freq,
        "ro_length": args.ro_length,
    }

    # Build soc
    if soc is None:
        soccfg_dict = load_soccfg(args.soccfg)
        soc, can_acquire = make_soc(soccfg_dict)
    else:
        can_acquire = True

    # Build and compile the program
    prog = ADCProbe(soc, reps=args.reps, final_delay=1.0, cfg=cfg)
    print(f"Program compiled — reps={args.reps}, ro_ch={args.ro_ch}, "
          f"freq={args.ro_freq} MHz, window={args.ro_length} us")

    if not can_acquire:
        print("(build + compile only — no acquire without MockQickSocV2 or a real board)")
        return

    # Acquire
    raw = prog.acquire(soc)
    # raw shape: (n_ch, n_reads, 2) — take first channel, first read
    i_val = float(raw[0][0][0])
    q_val = float(raw[0][0][1])

    report_single(i_val, q_val, label=f"ADC probe @ {args.ro_freq} MHz")


if __name__ == "__main__":
    main()
