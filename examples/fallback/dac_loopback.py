#!/usr/bin/env python3
"""DAC loopback — the 'does DAC -> ADC work?' check.

Plays a constant-amplitude tone on a generator and captures the return on the
readout ADC. With a physical loopback cable (DAC output -> ADC input), the
readout IQ should be non-zero with a magnitude proportional to the tone gain
and a phase that tracks the carrier frequency. Without a loopback, this still
verifies that the generator programs correctly.

Bare AveragerProgramV2 — no strumento.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qick.asm_v2 import AveragerProgramV2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, load_soccfg, make_soc, report_single

# ──────────────────────────────────────────────────────────────────────
# Program
# ──────────────────────────────────────────────────────────────────────


class DACLoopback(AveragerProgramV2):
    """Play a constant tone on gen_ch, read it back on ro_ch."""

    def _initialize(self, cfg):
        gen_ch = cfg["gen_ch"]
        ro_ch = cfg["ro_ch"]

        self.declare_gen(ch=gen_ch, nqz=cfg["nqz"])
        # Dynamic readouts reject freq/phase/sel at declaration.
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_pulse(
            ch=gen_ch,
            name="tone",
            style="const",
            freq=cfg["tone_freq"],
            phase=0,
            gain=cfg["gain"],
            length=cfg["tone_length"],
        )

    def _body(self, cfg):
        self.pulse(ch=cfg["gen_ch"], name="tone")
        self.trigger(ros=[cfg["ro_ch"]], t=cfg["trig_offset"])


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main(soc=None):
    parser = base_parser("DAC loopback — play a tone, read it back")
    parser.add_argument("--tone-freq", type=float, default=750.0, help="Tone frequency in MHz (default: 750)")
    parser.add_argument("--gain", type=float, default=0.5, help="Tone gain as fraction 0..1 (default: 0.5)")
    parser.add_argument("--tone-length", type=float, default=3.0, help="Tone length in us (default: 3.0)")
    parser.add_argument("--trig-offset", type=float, default=0.5, help="Trigger offset in us (default: 0.5)")
    parser.add_argument("--ro-length", type=float, default=2.0, help="Readout window in us (default: 2.0)")
    parser.add_argument("--nqz", type=int, default=1, help="Nyquist zone (default: 1)")
    args = parser.parse_args()

    cfg = {
        "gen_ch": args.gen_ch,
        "ro_ch": args.ro_ch,
        "nqz": args.nqz,
        "tone_freq": args.tone_freq,
        "gain": args.gain,
        "tone_length": args.tone_length,
        "trig_offset": args.trig_offset,
        "ro_length": args.ro_length,
    }

    if soc is None:
        soccfg_dict = load_soccfg(args.soccfg)
        soc, can_acquire = make_soc(soccfg_dict)
    else:
        can_acquire = True

    prog = DACLoopback(soc, reps=args.reps, final_delay=1.0, cfg=cfg)
    print(f"Program compiled — gen_ch={args.gen_ch}, tone={args.tone_freq} MHz, "
          f"gain={args.gain}, length={args.tone_length} us")

    if not can_acquire:
        print("(build + compile only — no acquire without MockQickSocV2 or a real board)")
        return

    raw = prog.acquire(soc)
    i_val = float(raw[0][0][0])
    q_val = float(raw[0][0][1])

    report_single(i_val, q_val, label=f"Loopback @ {args.tone_freq} MHz, gain={args.gain}")


if __name__ == "__main__":
    main()
