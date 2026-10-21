#!/usr/bin/env python3
"""Resonator spectroscopy — bare frequency sweep to find the resonator dip.

Sweeps a constant-amplitude probe tone across a readout resonator and plots
|S| vs frequency. A clear dip means the resonator is coupled and the signal
chain works end-to-end.

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


class ResonatorSpec(AveragerProgramV2):
    """Frequency-swept constant probe tone on gen_ch, readout on ro_ch."""

    def _initialize(self, cfg):
        gen_ch = cfg["gen_ch"]
        ro_ch = cfg["ro_ch"]

        self.declare_gen(ch=gen_ch, nqz=cfg["nqz"])
        # Dynamic readouts reject freq/phase/sel at declaration.
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

        # Frequency sweep
        self.add_loop("freq", count=cfg["points"])
        freq_sweep = QickSweep1D("freq", cfg["start_freq"], cfg["stop_freq"])

        self.add_pulse(
            ch=gen_ch,
            name="probe",
            style="const",
            freq=freq_sweep,
            phase=0,
            gain=cfg["gain"],
            length=cfg["tone_length"],
        )

    def _body(self, cfg):
        self.pulse(ch=cfg["gen_ch"], name="probe")
        self.trigger(ros=[cfg["ro_ch"]], t=cfg["trig_offset"])


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main(soc=None):
    parser = base_parser("Resonator spectroscopy — frequency sweep for resonator dip")
    parser.add_argument("--center-freq", type=float, default=750.0, help="Center frequency in MHz (default: 750)")
    parser.add_argument("--span", type=float, default=10.0, help="Sweep span in MHz (default: 10)")
    parser.add_argument("--points", type=int, default=201, help="Number of frequency points (default: 201)")
    parser.add_argument("--gain", type=float, default=0.3, help="Probe gain as fraction 0..1 (default: 0.3)")
    parser.add_argument("--tone-length", type=float, default=3.0, help="Probe length in us (default: 3.0)")
    parser.add_argument("--trig-offset", type=float, default=0.5, help="Trigger offset in us (default: 0.5)")
    parser.add_argument("--ro-length", type=float, default=2.0, help="Readout window in us (default: 2.0)")
    parser.add_argument("--nqz", type=int, default=1, help="Nyquist zone (default: 1)")
    args = parser.parse_args()

    start_freq = args.center_freq - args.span / 2
    stop_freq = args.center_freq + args.span / 2

    cfg = {
        "gen_ch": args.gen_ch,
        "ro_ch": args.ro_ch,
        "nqz": args.nqz,
        "center_freq": args.center_freq,
        "start_freq": start_freq,
        "stop_freq": stop_freq,
        "points": args.points,
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

    prog = ResonatorSpec(soc, reps=args.reps, final_delay=1.0, cfg=cfg)
    freqs = np.linspace(start_freq, stop_freq, args.points)
    print(f"Program compiled — sweep {start_freq:.1f} to {stop_freq:.1f} MHz, "
          f"{args.points} points, gain={args.gain}")

    if not can_acquire:
        print("(build + compile only — no acquire without MockQickSocV2 or a real board)")
        return

    raw = prog.acquire(soc)
    # raw shape: (n_ch, n_reads, n_expts, 2)
    i_data = raw[0][0][:, 0]
    q_data = raw[0][0][:, 1]
    mag = np.hypot(i_data, q_data)

    print("\nResults:")
    print(f"  Frequency range: {start_freq:.2f} — {stop_freq:.2f} MHz")
    print(f"  |S| range:       [{mag.min():.6f}, {mag.max():.6f}]")
    if mag.max() > 0:
        dip_idx = np.argmin(mag)
        print(f"  Min |S| at:      {freqs[dip_idx]:.4f} MHz (point {dip_idx})")

    if args.plot:
        plot_mag(freqs, mag, xlabel="Frequency (MHz)", ylabel="|S| (arb.)",
                 title=f"Resonator spectroscopy — gain={args.gain}")
        plot_iq(freqs, i_data, q_data, xlabel="Frequency (MHz)",
                title="Resonator spectroscopy — I/Q")


if __name__ == "__main__":
    main()
