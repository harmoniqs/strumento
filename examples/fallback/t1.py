#!/usr/bin/env python3
"""T1 relaxation — bare delay sweep after a pi pulse.

Excites the qubit with a Gaussian pi pulse, waits a variable delay, then reads
out. The excited-state population decays as exp(-t/T1); a clean exponential
decay means the qubit has measurable coherence.

Bare AveragerProgramV2 — no strumento.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from qick.asm_v2 import AveragerProgramV2, QickSweep1D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, load_soccfg, make_soc, plot_iq, plot_mag

# ──────────────────────────────────────────────────────────────────────
# Program
# ──────────────────────────────────────────────────────────────────────


class T1Decay(AveragerProgramV2):
    """Pi pulse, swept delay, readout — measure energy relaxation."""

    def _initialize(self, cfg):
        gen_ch = cfg["gen_ch"]
        ro_ch = cfg["ro_ch"]

        self.declare_gen(ch=gen_ch, nqz=cfg["nqz"])
        # Dynamic readouts reject freq/phase/sel at declaration.
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

        # Pi pulse envelope
        sigma = cfg["sigma"]
        self.add_gauss(ch=gen_ch, name="pi_env", sigma=sigma, length=4 * sigma)
        self.add_pulse(
            ch=gen_ch,
            name="pi",
            style="arb",
            freq=cfg["drive_freq"],
            phase=0,
            gain=cfg["pi_gain"],
            envelope="pi_env",
        )

        # Readout pulse
        self.add_pulse(
            ch=gen_ch,
            name="readout",
            style="const",
            freq=cfg["ro_freq"],
            phase=0,
            gain=cfg["ro_gain"],
            length=cfg["ro_length"],
        )

        # Delay sweep
        self.add_loop("delay", count=cfg["points"])
        self._delay_sweep = QickSweep1D("delay", cfg["start_delay"], cfg["stop_delay"])

    def _body(self, cfg):
        # Pi pulse
        self.pulse(ch=cfg["gen_ch"], name="pi")
        self.delay_auto(t=0)

        # Variable delay
        self.delay(t=self._delay_sweep)

        # Readout
        self.pulse(ch=cfg["gen_ch"], name="readout")
        self.trigger(ros=[cfg["ro_ch"]], t=cfg["trig_offset"])


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main(soc=None):
    parser = base_parser("T1 — delay sweep after pi pulse")
    parser.add_argument("--drive-freq", type=float, default=4000.0, help="Qubit drive freq in MHz (default: 4000)")
    parser.add_argument("--ro-freq", type=float, default=750.0, help="Readout freq in MHz (default: 750)")
    parser.add_argument("--pi-gain", type=float, default=0.25, help="Pi pulse gain (fraction, default: 0.25)")
    parser.add_argument("--sigma", type=float, default=0.030, help="Gaussian sigma in us (default: 0.030)")
    parser.add_argument("--start-delay", type=float, default=0.1, help="Start delay in us (default: 0.1)")
    parser.add_argument("--stop-delay", type=float, default=500.0, help="Stop delay in us (default: 500)")
    parser.add_argument("--points", type=int, default=101, help="Number of delay points (default: 101)")
    parser.add_argument("--ro-gain", type=float, default=0.3, help="Readout gain (fraction, default: 0.3)")
    parser.add_argument("--trig-offset", type=float, default=0.5, help="Trigger offset in us (default: 0.5)")
    parser.add_argument("--ro-length", type=float, default=2.0, help="Readout window in us (default: 2.0)")
    parser.add_argument("--nqz", type=int, default=1, help="Nyquist zone (default: 1)")
    args = parser.parse_args()

    cfg = {
        "gen_ch": args.gen_ch,
        "ro_ch": args.ro_ch,
        "nqz": args.nqz,
        "drive_freq": args.drive_freq,
        "ro_freq": args.ro_freq,
        "pi_gain": args.pi_gain,
        "sigma": args.sigma,
        "start_delay": args.start_delay,
        "stop_delay": args.stop_delay,
        "points": args.points,
        "ro_gain": args.ro_gain,
        "trig_offset": args.trig_offset,
        "ro_length": args.ro_length,
    }

    if soc is None:
        soccfg_dict = load_soccfg(args.soccfg)
        soc, can_acquire = make_soc(soccfg_dict)
    else:
        can_acquire = True

    prog = T1Decay(soc, reps=args.reps, final_delay=1.0, cfg=cfg)
    delays = np.linspace(args.start_delay, args.stop_delay, args.points)
    print(f"Program compiled — pi @ {args.drive_freq} MHz (gain={args.pi_gain}), "
          f"delay sweep [{args.start_delay}, {args.stop_delay}] us, {args.points} points")

    if not can_acquire:
        print("(build + compile only — no acquire without MockQickSocV2 or a real board)")
        return

    raw = prog.acquire(soc)
    i_data = raw[0][0][:, 0]
    q_data = raw[0][0][:, 1]
    mag = np.hypot(i_data, q_data)

    print("\nResults:")
    print(f"  Delay range: [{args.start_delay:.1f}, {args.stop_delay:.1f}] us")
    print(f"  |S| range:   [{mag.min():.6f}, {mag.max():.6f}]")

    if args.plot:
        plot_mag(delays, mag, xlabel="Delay (us)", ylabel="|S| (arb.)",
                 title=f"T1 decay — pi @ {args.drive_freq} MHz")
        plot_iq(delays, i_data, q_data, xlabel="Delay (us)",
                title="T1 decay — I/Q")


if __name__ == "__main__":
    main()
