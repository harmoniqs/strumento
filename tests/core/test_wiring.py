"""Tests for the wiring model (spec §5, plan Task 1).

The wiring layer models the *device*, not the FPGA channels: a caller refers to a
line + role (``LineRef``), and ``resolve`` maps that (plus a frequency, for
band-gated generators) to the concrete ``Generator`` that plays it.
"""

import pytest

from strumento.core.wiring import Generator, Line, LineRef, ReadoutWiring, Wiring


def _wiring():
    return Wiring(
        lines={
            "qubit": Line(name="qubit", generators=[
                Generator(role="drive", dac_ch=2, nyquist=1, gen_type="full"),
                Generator(role="sideband", dac_ch=0, nyquist=1, gen_type="full"),
            ]),
            "coupler": Line(name="coupler", generators=[
                Generator(role="low",  dac_ch=1, band_hz=(0.0, 1800e6)),
                Generator(role="high", dac_ch=1, band_hz=(1800e6, 1e12)),
            ]),
        },
        readout=ReadoutWiring(dac_ch=5, adc_chs=[0]),
    )


def test_resolve_by_role():
    w = _wiring()
    assert w.resolve(LineRef("qubit", "drive"), freq_mhz=3562).dac_ch == 2
    assert w.resolve(LineRef("qubit", "sideband"), freq_mhz=2010).dac_ch == 0


def test_resolve_band_gated():
    w = _wiring()
    assert w.resolve(LineRef("coupler", "low"),  freq_mhz=352).dac_ch == 1     # 352 MHz < 1800
    assert w.resolve(LineRef("coupler", "high"), freq_mhz=1998).dac_ch == 1


def test_resolve_unknown_raises():
    w = _wiring()
    with pytest.raises(KeyError):
        w.resolve(LineRef("nope", "drive"), freq_mhz=100)


# --- extra coverage beyond the plan's three (same seam, edge cases) ---

def test_resolve_role_out_of_band_raises():
    """A role that exists but whose only generator's band excludes the freq is an error,
    not a silent wrong-DAC play."""
    w = _wiring()
    with pytest.raises(KeyError):
        w.resolve(LineRef("coupler", "low"), freq_mhz=2500)  # 2500 MHz > 1800 -> only 'high' matches


def test_resolve_missing_role_raises():
    w = _wiring()
    with pytest.raises(KeyError):
        w.resolve(LineRef("qubit", "flux"), freq_mhz=100)  # 'flux' role not on the qubit line


def test_lineref_is_hashable_namedtuple():
    r = LineRef("qubit", "drive")
    assert (r.line, r.role) == ("qubit", "drive")
    assert hash(r) == hash(LineRef("qubit", "drive"))
