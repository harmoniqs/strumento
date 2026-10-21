"""Ladder-population diagnostic — pure helpers (block sequences + three-level analysis).

This is **not** a registered single-program ``Experiment``. The stage-resolved ladder diagnostic
must acquire its ``(stage, variant)`` blocks as **independent programs** (host-batched, one program
per block, relaxing to ``|g,0⟩`` between them): a single stacked program shares qubit/cavity state
across blocks — qick's relax fires once per shot, after the whole body, not between reads — so
block ``k+1`` would start from block ``k``'s end state and every stage after the first would be
corrupted. The caller's runner host-batches the block sequences with a per-block relax (long
only for the photon-creating stages). This module owns the
pure sequence construction and the quantitative leak analysis so both stay unit-testable without
``qick``.

**Diagnostic idea.** The Fock(n) prep ladder is ``[ge_pi · ef_pi · f0g1(k)]`` repeated for
``k = 0..n-1`` (``composite.prepare``); a clean prep walks the qubit ``g → e → f → g`` per cycle,
leaving the photon in the cavity. Truncate at each pulse (stages ``0..3n``) and use a calibrated
raw-IQ g/e/f response to recover the direct, unmapped ``(P_g, P_e, P_f)`` population at each stage.
The optional ge/ef-map variants remain as diagnostic controls, but never establish the verdict.

**Fail closed.** A two-level g/e discriminator has no defensible f-level population estimate. This
module therefore accepts only direct three-level populations; callers must run a conditioned g/e/f
blob calibration before calling its analyzer.

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.pulses import Seq

#: Read variants per stage, in fixed build order (the Runner and analyzer both index it).
LADDER_VARIANTS: tuple[str, ...] = ("none", "ge", "ef")

def n_stages(fock_n: int) -> int:
    """Number of ladder truncation stages (0..3·fock_n inclusive)."""
    return 3 * int(fock_n) + 1


def expected_level(stage: int) -> int:
    """The qubit level (0=g, 1=e, 2=f) a clean ladder leaves after ``stage`` pulses."""
    step = stage % 3
    return 0 if step == 0 else (1 if step == 1 else 2)


def photons_after(stage: int) -> int:
    """Cavity photons a clean ladder has created after ``stage`` pulses (one per completed f0g1)."""
    return stage // 3


def block_sequence(device, stage: int, variant: str, fock_n: int) -> tuple[Seq, tuple]:
    """One diagnostic block: the Fock(``fock_n``) ladder truncated to ``stage`` pulses, then a map
    pulse (``none``/``ge``/``ef``), then a driven readout. No loop axis (single read)."""
    q = device.qubit
    seq = Seq()
    for i in range(stage):
        photon, step = divmod(i, 3)
        if step == 0:
            seq.play(q.ge_pi())
        elif step == 1:
            seq.play(q.ef_pi())
        else:
            seq.play(device.sideband.f0g1(photon=photon))
    if variant == "ge":
        seq.play(q.ge_pi())
    elif variant == "ef":
        seq.play(q.ef_pi())
    seq += device.readout.dispersive_measure()
    return seq, ()


def iter_blocks(fock_n: int):
    """Yield ``(stage, variant)`` for every block in build order (stage-major, variant-minor)."""
    for stage in range(n_stages(fock_n)):
        for variant in LADDER_VARIANTS:
            yield stage, variant


def round_trip_sequence(device, fock_n: int) -> tuple[Seq, tuple]:
    r"""Return the full ladder through a second f0g1 pulse and map f→e for a consistency read.

    A bright final e population is a useful round-trip pulse diagnostic, but it is **not** an
    independent cavity-photon witness: a failed pair of identical f0g1 pulses can leave ``|f,0⟩``
    for the closing ef map to brighten. Photon occupation is established separately by parity.
    Single photon (``fock_n=1``) only for now; a Fock(n>1) round trip needs the full retrieval.
    """
    if int(fock_n) != 1:
        raise NotImplementedError("round_trip_sequence supports fock_n=1 (single photon) only")
    q = device.qubit
    seq = Seq()
    seq.play(q.ge_pi())
    seq.play(q.ef_pi())
    seq.play(device.sideband.f0g1(photon=0))          # forward: |f,0> -> |g,1>
    seq.play(device.sideband.f0g1(photon=0))          # retrieve: |g,1> -> |f,0>
    seq.play(q.ef_pi())                               # map f -> e for the ge-readout
    seq += device.readout.dispersive_measure()
    return seq, ()



def analyze_ladder_populations(
    populations_by_block,
    fock_n: int,
    *,
    leak_threshold: float = 0.10,
    round_trip_populations=None,
) -> dict:
    r"""Quantitative leak localization from direct ``(P_g, P_e, P_f)`` block populations.

    ``populations_by_block`` has shape ``(3*n_stages, 3)`` in stage-major, variant-minor order.
    Only the ``none`` block at each stage represents the unmodified ladder state and feeds the
    verdict. The mapped controls are persisted as diagnostics, never inferred into a level label.
    """
    ns = n_stages(fock_n)
    populations = np.asarray(populations_by_block, dtype=float)
    expected_shape = (ns * len(LADDER_VARIANTS), 3)
    if populations.shape != expected_shape:
        raise ValueError(
            f"expected direct g/e/f populations with shape {expected_shape}; got {populations.shape}"
        )

    derived: dict[str, float] = {}
    leak_stage = -1
    worst_defect = 0.0
    for stage in range(ns):
        start = len(LADDER_VARIANTS) * stage
        blocks = populations[start:start + len(LADDER_VARIANTS)]
        for variant, levels in zip(LADDER_VARIANTS, blocks, strict=True):
            for level, value in zip("gef", levels, strict=True):
                derived[f"s{stage}_{variant}_{level}"] = float(value)

        expected = expected_level(stage)
        p_expected = float(blocks[0, expected])  # direct, unmapped state
        defect = 1.0 - p_expected
        derived[f"s{stage}_defect"] = defect
        worst_defect = max(worst_defect, defect)
        if leak_stage < 0 and defect > leak_threshold:
            leak_stage = stage

    final = populations[len(LADDER_VARIANTS) * (ns - 1)]
    derived["final_g"] = float(final[0])
    derived["final_e"] = float(final[1])
    derived["final_f"] = float(final[2])
    derived["residual_excitation_final"] = float(final[1] + final[2])
    derived["leak_stage"] = float(leak_stage)
    derived["worst_defect"] = float(worst_defect)
    derived["leak_threshold"] = float(leak_threshold)
    derived["n_stages"] = float(ns)
    if round_trip_populations is not None:
        round_trip = np.asarray(round_trip_populations, dtype=float)
        if round_trip.shape != (3,):
            raise ValueError(f"expected round-trip g/e/f populations with shape (3,); got {round_trip.shape}")
        derived["round_trip_g_population"] = float(round_trip[0])
        derived["round_trip_e_population"] = float(round_trip[1])
        derived["round_trip_f_population"] = float(round_trip[2])
    return derived
