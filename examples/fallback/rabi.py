#!/usr/bin/env python3
"""Amplitude Rabi — bare gain sweep to find the pi gain.

Plays a Gaussian drive pulse at the qubit frequency with swept amplitude,
then reads out. A Rabi oscillation in the readout IQ means the qubit is
responding to the drive.

Bare AveragerProgramV2 — no strumento. Requires separate drive and readout
generators for a real qubit experiment; on the single-gen testbench, both
pulses share gen_ch 0 (assembly testing only).
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


class AmplitudeRabi(AveragerProgramV2):
    """Gaussian drive with swept gain, then constant readout tone."""

    def _initialize(self, cfg):
        gen_ch = cfg["gen_ch"]
        ro_ch = cfg["ro_ch"]

        self.declare_gen(ch=gen_ch, nqz=cfg["nqz"])
        # Dynamic readouts reject freq/phase/sel at declaration.
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

        # Gaussian drive envelope
        sigma = cfg["sigma"]
        self.add_gauss(ch=gen_ch, name="drive_env", sigma=sigma, length=4 * sigma)

        # Amplitude sweep
        self.add_loop("amp", count=cfg["points"])
        gain_sweep = QickSweep1D("amp", cfg["start_gain"], cfg["stop_gain"])

        self.add_pulse(
            ch=gen_ch,
            name="drive",
            style="arb",
            freq=cfg["drive_freq"],
            phase=0,
            gain=gain_sweep,
            envelope="drive_env",
        )

        # Readout pulse (constant tone)
        self.add_pulse(
            ch=gen_ch,
            name="readout",
            style="const",
            freq=cfg["ro_freq"],
            phase=0,
            gain=cfg["ro_gain"],
            length=cfg["ro_length"],
        )

    def _body(self, cfg):
        self.pulse(ch=cfg["gen_ch"], name="drive")
        self.delay_auto(t=0)
        self.pulse(ch=cfg["gen_ch"], name="readout")
        self.trigger(ros=[cfg["ro_ch"]], t=cfg["trig_offset"])


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main(soc=None):
    parser = base_parser("Amplitude Rabi — gain sweep for pi pulse calibration")
    parser.add_argument("--drive-freq", type=float, default=4000.0, help="Qubit drive freq in MHz (default: 4000)")
    parser.add_argument("--ro-freq", type=float, default=750.0, help="Readout freq in MHz (default: 750)")
    parser.add_argument("--start-gain", type=float, default=0.0, help="Sweep start gain (fraction, default: 0.0)")
    parser.add_argument("--stop-gain", type=float, default=0.9, help="Sweep stop gain (fraction, default: 0.9)")
    parser.add_argument("--points", type=int, default=51, help="Number of gain points (default: 51)")
    parser.add_argument("--sigma", type=float, default=0.030, help="Gaussian sigma in us (default: 0.030)")
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
        "start_gain": args.start_gain,
        "stop_gain": args.stop_gain,
        "points": args.points,
        "sigma": args.sigma,
        "ro_gain": args.ro_gain,
        "trig_offset": args.trig_offset,
        "ro_length": args.ro_length,
    }

    if soc is None:
        soccfg_dict = load_soccfg(args.soccfg)
        soc, can_acquire = make_soc(soccfg_dict)
    else:
        can_acquire = True

    prog = AmplitudeRabi(soc, reps=args.reps, final_delay=1.0, cfg=cfg)
    gains = np.linspace(args.start_gain, args.stop_gain, args.points)
    print(f"Program compiled — drive={args.drive_freq} MHz, sigma={args.sigma} us, "
          f"gain sweep [{args.start_gain}, {args.stop_gain}], {args.points} points")

    if not can_acquire:
        print("(build + compile only — no acquire without MockQickSocV2 or a real board)")
        return

    raw = prog.acquire(soc)
    i_data = raw[0][0][:, 0]
    q_data = raw[0][0][:, 1]
    mag = np.hypot(i_data, q_data)

    print("\nResults:")
    print(f"  Gain range:  [{args.start_gain:.3f}, {args.stop_gain:.3f}]")
    print(f"  |S| range:   [{mag.min():.6f}, {mag.max():.6f}]")

    if args.plot:
        plot_mag(gains, mag, xlabel="Drive gain (fraction)", ylabel="|S| (arb.)",
                 title=f"Amplitude Rabi — drive @ {args.drive_freq} MHz")
        plot_iq(gains, i_data, q_data, xlabel="Drive gain (fraction)",
                title="Amplitude Rabi — I/Q")


if __name__ == "__main__":
    main()
