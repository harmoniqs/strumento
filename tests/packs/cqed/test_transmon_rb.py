"""TransmonRB: reset per depth, identity timing from the PLAYED gate, pulse overrides (pure).

Regressions covered:

1. **Measurement chaining.** Each depth block was ``Cliffords → measure`` back-to-back with no reset.
   A measurement is not a reset, so depth k started from depth k−1's post-measurement state and
   every later depth is pulled toward ½: a FAKE fast decay (pessimistic error per Clifford).
2. **Identity Clifford lost its duration once DRAG was on.** ``_identity_wait_us = 4·envelope.sigma_us``
   — a DRAG (or any other) ``Arb`` has no ``sigma_us``, so with ``drag_alpha`` set the identity
   waited 0 µs. Identity Cliffords became free → RB biased optimistic vs decoherence, silently.
3. **No way to benchmark substituted gates.** ``overrides`` must be forwarded to the transmon ops so
   ``ge_pi``/``ge_hpi`` (and therefore every Clifford) play the override pulses.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.pulses import Arb, Delay, Measure, Pulse
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments.transmon_rb import TransmonRB


def _blocks(seq):
    blocks, cur = [], []
    for op in seq.ops:
        cur.append(op)
        if isinstance(op, Measure):
            blocks.append(cur)
            cur = []
    return blocks


def _played_us(pulse, dev):
    env = pulse.envelope
    if isinstance(env, Arb):
        return len(env.idata) / dev.qubit._dac_rate(dev.calib.qubit.f_ge)
    return 4.0 * env.sigma_us


def test_every_depth_block_starts_from_a_passive_reset(dev):
    exp = TransmonRB(dev, depth_start=1, depth_stop=20, points=4, seed=1)
    seq, _ = exp.sequence()
    blocks = _blocks(seq)
    assert len(blocks) == 4
    relax = float(dev.calib.readout.relax_delay)
    for b in blocks:
        assert isinstance(b[0], Delay) and float(b[0].duration_us) >= relax


def test_reset_delay_is_overridable(dev):
    exp = TransmonRB(dev, depth_start=1, depth_stop=5, points=2, seed=1, reset_delay_us=777.0)
    seq, _ = exp.sequence()
    assert [float(b[0].duration_us) for b in _blocks(seq)] == [777.0, 777.0]


@pytest.mark.parametrize("drag_alpha", [None, -0.05])
def test_identity_clifford_lasts_one_gate_whatever_the_envelope(dev, drag_alpha):
    dev.calib.qubit.drag_alpha = drag_alpha
    exp = TransmonRB(dev)
    assert exp._identity_wait_us == pytest.approx(_played_us(dev.qubit.ge_pi(), dev))
    assert exp._identity_wait_us > 0.1


def _override(dev, gain, n=896):
    t = np.linspace(-1, 1, n)
    return Pulse(line=LineRef("qubit", "drive"), freq_mhz=dev.calib.qubit.f_ge, gain=gain,
                 phase_deg=0.0, envelope=Arb(idata=tuple(np.exp(-4 * t**2)), qdata=tuple(0.1 * t)),
                 label="override")


def test_overrides_reach_every_clifford_pulse(dev):
    ov_pi, ov_hpi = _override(dev, 0.21), _override(dev, 0.11)
    exp = TransmonRB(dev, depth_start=5, depth_stop=5, points=1, seed=3,
                     overrides={"pi_ge": ov_pi, "hpi_ge": ov_hpi})
    seq, _ = exp.sequence()
    drives = [op for op in seq.ops if isinstance(op, Pulse) and op.line == LineRef("qubit", "drive")]
    assert drives, "no gate pulses in the RB block"
    for p in drives:
        assert p.envelope in (ov_pi.envelope, ov_hpi.envelope)
        assert p.gain in (ov_pi.gain, ov_hpi.gain)
    # the device's own ops are untouched (the override is scoped to this experiment)
    assert not isinstance(dev.qubit.ge_pi().envelope, Arb) or dev.calib.qubit.drag_alpha is not None


def test_identity_with_overrides_lasts_the_override_gate(dev):
    ov_pi = _override(dev, 0.21, n=1008)
    exp = TransmonRB(dev, overrides={"pi_ge": ov_pi, "hpi_ge": _override(dev, 0.11, n=1008)})
    assert exp._identity_wait_us == pytest.approx(1008 / dev.qubit._dac_rate(dev.calib.qubit.f_ge))


# -- program memory (issue #211): an over-size request is refused before compile --------------


def _with_pmem(dev, words: int):
    for tproc in dev.soccfg_snapshot["tprocs"]:
        tproc["pmem_size"] = words
    return dev


def test_an_over_size_request_is_refused_before_compile_with_size_limit_and_remedy(dev):
    from strumento.packs.cqed.experiments.transmon_rb import ProgramMemoryError

    _with_pmem(dev, 16384)  # a real board's pmem; the issue's measured overflow
    exp = TransmonRB(dev, depth_start=1, depth_stop=500, points=20, seed=42)
    with pytest.raises(ProgramMemoryError) as info:
        exp.sequence()
    err, msg = info.value, str(info.value)
    assert err.limit_words == 16384
    assert err.estimated_words > 16384 and not err.compiled
    assert f"{err.estimated_words:,}" in msg and "16,384" in msg
    assert "pmem_size" in msg                       # where the limit came from
    assert "points" in msg and "one program per depth" in msg  # the remedy


def test_the_ceiling_is_read_from_the_snapshot_not_hard_coded(dev):
    from strumento.packs.cqed.experiments.transmon_rb import ProgramMemoryError

    request = dict(depth_start=1, depth_stop=300, points=8, seed=42)  # ~4.5k words
    with pytest.raises(ProgramMemoryError, match="4,096"):
        TransmonRB(dev, **request).sequence()        # the shipped 4096-word snapshot
    seq, _ = TransmonRB(_with_pmem(dev, 16384), **request).sequence()  # a 16k board: it fits
    assert sum(isinstance(op, Measure) for op in seq.ops) == 8


def test_requests_that_fit_are_not_refused(dev):
    TransmonRB(dev).sequence()  # the defaults
    TransmonRB(dev, depth_start=1, depth_stop=50, points=6, seed=42).sequence()  # the contract's


def test_a_device_without_a_snapshot_is_not_refused(dev):
    dev.soccfg_snapshot = None
    TransmonRB(dev, depth_start=1, depth_stop=500, points=20, seed=42).sequence()


def test_the_ceiling_is_stated_in_the_docstring():
    doc = TransmonRB.__doc__ or ""
    assert "pmem_size" in doc and "program memory" in doc
