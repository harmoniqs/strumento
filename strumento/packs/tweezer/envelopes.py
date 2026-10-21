"""tweezer envelope types (D19, D22).

A tweezer *move* is a piecewise **5th-order frequency polynomial + 3rd-order amplitude polynomial**,
sent as coefficients (not samples) — "much more efficiently than a raw waveform" (arXiv 2506.13724).
This is the neutral-atom analogue of `Gaussian`: a compact, parameterized envelope.

Defining it here is how a **pack owns its envelope type** (D22): the type lives in the pack, and the
pack registers its compiler lowering (against the custom chirp-DDS firmware) when the tweezer pulse
library lands. Core stays stock-v2 and never learns about `PolyChirp`. Frozen ⇒ hashable, so it works
as a compiler dedup key like the core envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolyChirp:
    """A single-segment tweezer move: freq polynomial (deg ≤ 5) + amplitude polynomial (deg ≤ 3)."""

    freq_coeffs: tuple[float, ...]
    amp_coeffs: tuple[float, ...]
    duration_us: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "freq_coeffs", tuple(float(c) for c in self.freq_coeffs))
        object.__setattr__(self, "amp_coeffs", tuple(float(c) for c in self.amp_coeffs))
