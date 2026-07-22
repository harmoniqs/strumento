"""Wiring model (spec §5) — *model the device, not the FPGA channels*.

A caller names a **line** and a **role** (``LineRef("qubit", "drive")``); ``Wiring.resolve``
maps that reference — plus, for band-gated lines, a frequency — to the concrete
:class:`Generator` (a DAC channel + Nyquist zone + generator type) that plays it. This is
the single indirection that lets pulse libraries and the compiler speak in device terms
(``dev.qubit.ge_pi()``) while the instance YAML owns the copper (spec §5, D21/D27).

This module is **pure**: it never imports ``qick``. The optional per-line :class:`SignalChain`
is a reserved stub for signal-chain / transfer-function metadata (spec §18/D24) that the
codesign slice (S8.4) fills in; it carries arbitrary fields today and constrains nothing.
"""

from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel, ConfigDict


class LineRef(NamedTuple):
    """A device-level handle: which line, which role on it.

    A hashable, positional ``(line, role)`` tuple — the currency the pulse IR and compiler
    pass to :meth:`Wiring.resolve`. Deliberately *not* Pydantic: it is a lightweight key,
    used on the hot path and as a dict/dedup key.
    """

    line: str
    role: str


class SignalChain(BaseModel):
    """Optional per-line signal-chain metadata (spec §18/D24) — a reserved stub.

    Left open (``extra="allow"``) so an instance may annotate a line with amp/attenuator/
    filter/transfer-function detail before the codesign slice formalizes the schema. Nothing
    in ``core`` reads it yet; it exists so instances taken *today* can carry the metadata that
    is retroactively impossible to reconstruct later.
    """

    model_config = ConfigDict(extra="allow")


class Generator(BaseModel):
    """One physical signal generator (a DAC channel) on a line.

    ``band_hz`` is an optional ``(lo, hi)`` frequency gate in Hz: when several generators share
    a role/line (e.g. a low- and high-band flux driver on the same DAC), ``resolve`` selects the
    one whose band contains the requested frequency. ``None`` means "plays any frequency".
    """

    role: str | None = None  # set from the wire-key by DeviceBuilder; always present in a built device
    dac_ch: int
    nyquist: int = 1
    gen_type: str = "full"
    band_hz: tuple[float, float] | None = None
    mixer_freq: float | None = None
    mux_freqs: list[float] | None = None


class Line(BaseModel):
    """A named device line carrying one or more role-tagged generators."""

    name: str
    generators: list[Generator]
    chain: SignalChain | None = None


class ReadoutWiring(BaseModel):
    """Readout signal chain: the probe DAC + the ADC channel(s) that digitize the return.

    Named ``ReadoutWiring`` (not ``Readout``) to avoid colliding with ``core/readout.py``'s
    ``ReadoutKind`` protocols (Task 16). Only ``dac_ch``/``adc_chs`` are required; the rest are
    instance-supplied acquisition parameters populated from the device YAML.
    """

    dac_ch: int
    adc_chs: list[int]
    mux_freqs: list[float] | None = None
    lengths_us: list[float] | float | None = None
    thresholds: list[float] | None = None
    trig_offset: float | None = None


class Wiring(BaseModel):
    """The device's full line/generator map plus its readout chain.

    ``resolve(ref, freq_mhz)`` is the one lookup callers use; everything else is data.
    """

    lines: dict[str, Line]
    readout: ReadoutWiring

    def resolve(self, ref: LineRef, freq_mhz: float) -> Generator:
        """Map a ``(line, role)`` reference + frequency to the generator that plays it.

        Raises ``KeyError`` when the line is unknown, the role is absent on that line, or no
        role-matching generator's band contains ``freq_mhz``. Raises ``ValueError`` when the
        wiring is ambiguous (more than one generator claims the frequency) — a malformed
        instance, not a lookup miss.
        """
        line = self.lines.get(ref.line)
        if line is None:
            raise KeyError(f"no line {ref.line!r} in wiring (have {sorted(self.lines)})")

        by_role = [g for g in line.generators if g.role == ref.role]
        if not by_role:
            roles = sorted({g.role for g in line.generators})
            raise KeyError(f"line {ref.line!r} has no role {ref.role!r} (have {roles})")

        f_hz = freq_mhz * 1e6
        matched = [
            g for g in by_role
            if g.band_hz is None or (g.band_hz[0] <= f_hz < g.band_hz[1])
        ]
        if not matched:
            bands = [g.band_hz for g in by_role]
            raise KeyError(
                f"no generator for {ref.line}.{ref.role} at {freq_mhz} MHz "
                f"(role bands, Hz: {bands})"
            )
        if len(matched) > 1:
            raise ValueError(
                f"ambiguous wiring: {len(matched)} generators claim "
                f"{ref.line}.{ref.role} at {freq_mhz} MHz — overlapping bands"
            )
        return matched[0]
