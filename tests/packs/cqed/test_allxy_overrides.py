"""TransmonAllXY `overrides=` — benchmark substituted gates through AllXY, pure.

AllXY builds every pair from `_gate_pulse(q, code)`, so routing a scoped `TransmonOps` as `q` is all
it takes — identical to the TransmonRB change. The identity gate is `ge_pi(gain=0.0)`, which only
plays a zero-gain pulse if the override honours gain (the freq/gain override fix) — otherwise
identity would be a full override π.
"""

from __future__ import annotations

import numpy as np

from strumento.core.pulses import Arb, Pulse
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments.transmon_allxy import ALLXY_PAIRS, TransmonAllXY


def _override(dev, gain, n=896):
    t = np.linspace(-1, 1, n)
    return Pulse(line=LineRef("qubit", "drive"), freq_mhz=dev.calib.qubit.f_ge, gain=gain,
                 phase_deg=0.0, envelope=Arb(idata=tuple(np.exp(-4 * t**2)), qdata=tuple(0.1 * t)),
                 label="override")


def _gate_pulses(seq):
    return [op for op in seq.ops if isinstance(op, Pulse) and op.line == LineRef("qubit", "drive")]


def test_overrides_route_every_gate_through_the_override_pulses(dev):
    ov_pi, ov_hpi = _override(dev, 0.2), _override(dev, 0.1)
    seq, _ = TransmonAllXY(dev, overrides={"pi_ge": ov_pi, "hpi_ge": ov_hpi}).sequence()
    drives = _gate_pulses(seq)
    assert drives, "no gate pulses"
    for p in drives:
        assert p.envelope in (ov_pi.envelope, ov_hpi.envelope)


def test_identity_gate_is_zero_gain_even_with_an_override(dev):
    ov_pi = _override(dev, 0.2)
    seq, _ = TransmonAllXY(dev, overrides={"pi_ge": ov_pi, "hpi_ge": _override(dev, 0.1)}).sequence()
    # ("I","I") is the first pair → the first two drive pulses must be zero gain (identity), not a π.
    first_pair = _gate_pulses(seq)[:2]
    assert ALLXY_PAIRS[0][:2] == ("I", "I")
    assert all(p.gain == 0.0 and p.envelope == ov_pi.envelope for p in first_pair)


def test_no_overrides_is_unchanged(dev):
    seq, _ = TransmonAllXY(dev).sequence()
    # default path still plays the factory gates (Gaussian here — no drag_alpha on multimode_demo)
    from strumento.core.pulses import Gaussian
    assert any(isinstance(p.envelope, Gaussian) for p in _gate_pulses(seq))


# -- An OC-imported override plays sequentially in a pair (#210) -------------------------------
#
# `from_solution` emits its pulses with `start_us=0.0`, an offset inside its parallel segment.
# Installed as a gate override, that offset used to survive, so the compiler placed the gate
# *absolutely* (no cursor advance): both gates of a pair and the readout stacked at the pair's
# start, and the only symptom was qick's logged "pulse time 0.0 appears to conflict". The oracle
# here is the COMPILED program — the realized start/end of every pulse and trigger, in tProc
# cycles, read back from qick's own macro list — not the IR's `start_us`.

import pytest  # noqa: E402

from tests._diagnostics import _diagnostics_are_errors  # noqa: E402
from tests._paths import DEVICES  # noqa: E402

from ._oc_lowering import analytic_controls, lower  # noqa: E402

_CQED_SOCCFG = DEVICES / "_fixtures" / "soccfg_v2_cqed.json"


def _oc_overrides(dev):
    """π and π/2 lowered through the real import path (Piccolo controls → from_solution)."""
    return {key: lower(dev, *analytic_controls(gate)[:2])[1]
            for key, gate in (("pi_ge", "pi"), ("hpi_ge", "hpi"))}


def _schedule(prog):
    """Realized timeline of a compiled program: ``[(kind, ch, start, end)]`` in tProc cycles.

    Walks qick's macro list the way its preprocess pass does: ``Delay`` (incl. ``delay_auto``)
    moves the reference; a ``Pulse``/``Trigger`` plays at reference + its own ``t``. A pulse's end
    adds its realized length (``get_length``, µs → tProc cycles).
    """
    ref, out = 0, []
    for m in prog.macro_list:
        kind = type(m).__name__
        t = getattr(m, "t_regs", {}).get("t")
        if kind == "Delay" and t is not None:
            assert isinstance(t, int), "a swept delay has no static schedule"
            ref += t
        elif kind == "Pulse":
            length = prog.us2cycles(float(prog.pulses[m.name].get_length()), as_float=True)
            out.append(("pulse", m.ch, ref + t, ref + t + length))
        elif kind == "Trigger":
            out.append(("trigger", None, ref + t, ref + t))
    return out


def _pairs(dev, prog):
    """Per AllXY pair: (gate1, gate2, readout pulse, trigger) as (start, end) tuples."""
    from strumento.core.wiring import LineRef

    drive_ch = dev.wiring.resolve(LineRef("qubit", "drive"), dev.calib.qubit.f_ge).dac_ch
    sched = _schedule(prog)
    gates = [(s, e) for k, ch, s, e in sched if k == "pulse" and ch == drive_ch]
    reads = [(s, e) for k, ch, s, e in sched if k == "pulse" and ch != drive_ch]
    trigs = [(s, e) for k, _, s, e in sched if k == "trigger"]
    assert len(gates) == 2 * len(ALLXY_PAIRS) and len(reads) == len(trigs) == len(ALLXY_PAIRS)
    return [(gates[2 * i], gates[2 * i + 1], reads[i], trigs[i]) for i in range(len(ALLXY_PAIRS))]


def _compiled_allxy(dev, soc, overrides):
    from strumento.core.program import StrumentoProgram

    seq, sweeps = TransmonAllXY(dev, overrides=overrides).sequence()
    prog = StrumentoProgram(dev, seq=seq, sweeps=sweeps, reps=4, final_delay=None, soc=soc)
    prog.asm()  # run qick's preprocess pass: realizes every t
    return prog


#: qick logs the reverted case's conflict via the deprecated `logger.warn`; that deprecation is
#: qick's noise, not ours (the guard records inside its own `catch_warnings`, unaffected).
_QICK_WARN_DEPRECATION = pytest.mark.filterwarnings(
    "ignore:The 'warn' method is deprecated:DeprecationWarning")


def _restore_the_in_segment_offset(monkeypatch):
    """Revert the #210 fix: let an override keep the import's ``start_us=0.0``."""
    from dataclasses import replace

    from strumento.packs.cqed.library.transmon import TransmonOps

    original = TransmonOps._pulse

    def _pulse(self, pulse_key, *a, **kw):
        p = original(self, pulse_key, *a, **kw)
        return replace(p, start_us=0.0) if pulse_key in self.overrides else p

    monkeypatch.setattr(TransmonOps, "_pulse", _pulse)


@pytest.mark.qick
def test_an_oc_imported_override_plays_sequentially_in_every_allxy_pair(dev):
    from strumento.core.simulator import SimulatorSoc

    soc = SimulatorSoc.from_snapshot(_CQED_SOCCFG, truth={}, seed=7)
    overrides = _oc_overrides(dev)
    assert all(p.start_us == 0.0 for p in overrides.values())   # the import's own offset
    for i, (g1, g2, ro, trig) in enumerate(_pairs(dev, _compiled_allxy(dev, soc, overrides))):
        assert g1[1] > g1[0], f"pair {i}: zero-length gate"
        assert g2[0] >= g1[1] - 1, f"pair {i}: gate 2 starts at {g2[0]} before gate 1 ends at {g1[1]}"
        assert ro[0] >= g2[1] - 1, f"pair {i}: readout starts at {ro[0]} before gate 2 ends at {g2[1]}"
        assert trig[0] >= g2[1] - 1, f"pair {i}: trigger at {trig[0]} before gate 2 ends at {g2[1]}"


@pytest.mark.qick
@_QICK_WARN_DEPRECATION
def test_the_timing_oracle_sees_the_collapse_when_the_fix_is_reverted(dev, monkeypatch):
    """Non-vacuity of the test above: with the offset kept, the pair collapses onto its start."""
    from strumento.core.simulator import SimulatorSoc

    _restore_the_in_segment_offset(monkeypatch)
    soc = SimulatorSoc.from_snapshot(_CQED_SOCCFG, truth={}, seed=7)
    g1, g2, ro, _ = _pairs(dev, _compiled_allxy(dev, soc, _oc_overrides(dev)))[1]
    assert g2[0] == g1[0] and ro[0] == g1[0]     # all three at the pair's start


def _run_allxy(dev, overrides):
    from strumento.core.simulator import SimulatorSoc
    from strumento.packs.cqed.responses import truth_from_device

    with _diagnostics_are_errors() as (records, _notices, caught):
        soc = SimulatorSoc.from_snapshot(_CQED_SOCCFG, truth=truth_from_device(dev), seed=7)
        TransmonAllXY(dev, overrides=overrides).go(soc, reps=16, progress=False, analyze=False)
    ours = [str(w.message) for w in caught
            if "qick" in str(w.filename) or "strumento" in str(w.filename)]
    return records, ours


@pytest.mark.qick
def test_an_oc_imported_override_assembles_without_a_timing_conflict(dev):
    records, ours = _run_allxy(dev, _oc_overrides(dev))
    assert not records, f"qick/strumento logged warnings with an OC override: {records}"
    assert not ours, f"Python warnings with an OC override: {ours}"


@pytest.mark.qick
@_QICK_WARN_DEPRECATION
def test_the_diagnostics_guard_catches_the_conflict_when_the_fix_is_reverted(dev, monkeypatch):
    """Non-vacuity: the same run with the in-segment offset restored IS caught by the guard."""
    _restore_the_in_segment_offset(monkeypatch)
    records, _ = _run_allxy(dev, _oc_overrides(dev))
    assert any("appears to conflict" in r for r in records), records
