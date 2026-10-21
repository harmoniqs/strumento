"""D9 — canonical config round-trip, content addressing, and the versioned store + write-back."""

from __future__ import annotations

import pytest

from strumento.core.config import DeviceConfig, apply_updates
from strumento.core.device import Device
from strumento.core.store import FileConfigStore, Provenance, record_calibration

# -- round-trip (D30, now real) ----------------------------------------------------------


def test_device_to_config_round_trips(dev):
    cfg = dev.to_config()
    assert isinstance(cfg, DeviceConfig)
    rebuilt = Device.from_config(cfg)
    # the rebuilt device is byte-identical at the config level
    assert rebuilt.to_config().content_id() == cfg.content_id()
    # and functionally equivalent
    assert rebuilt.calib.qubit.f_ge == dev.calib.qubit.f_ge
    assert sorted(rebuilt.elements) == sorted(dev.elements)
    assert hasattr(rebuilt, "qubit") and hasattr(rebuilt, "sideband")


def test_content_id_is_deterministic_and_calibration_sensitive(dev):
    cfg = dev.to_config()
    assert cfg.content_id() == dev.to_config().content_id()      # stable
    assert cfg.content_id().startswith("cfg-")
    bumped, _ = apply_updates(cfg, {"qubit.f_ge": 4001.0})
    assert bumped.content_id() != cfg.content_id()               # a calib change changes the id
    # ...and an identical change reproduces the same id (reproducibility)
    again, _ = apply_updates(cfg, {"qubit.f_ge": 4001.0})
    assert again.content_id() == bumped.content_id()


def test_builder_and_yaml_produce_the_same_model():
    from strumento.core.builder import DeviceBuilder
    from strumento.core.wiring import Generator

    b = DeviceBuilder(pack="cqed")
    b.add("Transmon", "qubit")
    b.add("ReadoutResonator", "readout")
    b.wire("qubit", drive=Generator(dac_ch=2, nyquist=1))
    b.readout(dac_ch=5, adc_chs=[0])
    calib = {
        "qubit": {"f_ge": 4000.0, "f_ef": 3820.0, "pulses": {
            "pi_ge": {"gain": 8192, "sigma": 0.03}, "hpi_ge": {"gain": 4096, "sigma": 0.03},
            "pi_ef": {"gain": 7600, "sigma": 0.03}}},
        "readout": {"frequency": 750.0, "gain": 1000, "readout_length": 2.0,
                    "trig_offset": 0.4, "relax_delay": 500},
    }
    dev = b.build(calib=calib)
    cfg = dev.to_config()
    # the builder emits a canonical config that re-composes identically
    assert Device.from_config(cfg).to_config().content_id() == cfg.content_id()
    # no soccfg attached -> the factory uses the 2**15-1 fallback maxv (D23 boundary conversion)
    assert dev.qubit.ge_pi().gain == pytest.approx(8192 / (2**15 - 1))


# -- write-back ---------------------------------------------------------------------------


def test_apply_updates_is_immutable_and_diffs(dev):
    cfg = dev.to_config()
    new, diff = apply_updates(cfg, {"qubit.f_ge": 4002.5, "qubit.pulses.pi_ge.gain": 8300})
    # original untouched
    assert cfg.calibration["qubit"]["f_ge"] == 4000.0
    # new carries the change
    assert new.calibration["qubit"]["f_ge"] == 4002.5
    assert new.calibration["qubit"]["pulses"]["pi_ge"]["gain"] == 8300
    # diff records old -> new
    assert diff["qubit.f_ge"] == (4000.0, 4002.5)
    assert diff["qubit.pulses.pi_ge.gain"] == (8192, 8300)


def test_apply_updates_rejects_bad_path(dev):
    with pytest.raises(KeyError, match="calibration path"):
        apply_updates(dev.to_config(), {"qubit.nonexistent.field": 1.0})


def test_updated_config_recomposes_a_device_with_new_calibration(dev):
    new, _ = apply_updates(dev.to_config(), {"qubit.f_ge": 4123.0})
    dev2 = Device.from_config(new)
    assert dev2.calib.qubit.f_ge == 4123.0
    assert dev2.qubit.ge_pi().freq_mhz == 4123.0          # the factory reflects the new value


# -- the store ----------------------------------------------------------------------------


def test_store_put_get_is_content_addressed_and_write_once(tmp_path, dev):
    store = FileConfigStore(tmp_path / "cfgstore")
    cfg = dev.to_config()
    cid = store.put(cfg)
    assert cid == cfg.content_id()
    assert store.get(cid).content_id() == cid
    # write-once: re-putting the same config is a no-op, not an error
    assert store.put(cfg) == cid
    with pytest.raises(KeyError):
        store.get("cfg-deadbeef")


def test_ref_moves_and_history_is_append_only(tmp_path, dev):
    store = FileConfigStore(tmp_path / "cfgstore")
    base = dev.to_config()
    store.put(base)
    store.set_ref("main", base.content_id())
    assert store.resolve_ref("main") == base.content_id()

    # two successive calibration write-backs advance main and log provenance
    cid1, cfg1 = record_calibration(
        store, "main", base, {"qubit.f_ge": 4001.0},
        source="amplitude_rabi fit", timestamp="2026-07-23T00:00:00Z",
    )
    cid2, _ = record_calibration(
        store, "main", cfg1, {"qubit.f_ge": 4002.0},
        source="ramsey fit", timestamp="2026-07-23T01:00:00Z",
    )
    assert store.resolve_ref("main") == cid2

    log = store.log("main")
    assert [p.content_id for p in log] == [cid1, cid2]
    assert log[0].parent_id == base.content_id()
    assert log[1].parent_id == cid1                       # chained lineage
    assert log[0].updates["qubit.f_ge"] == [4000.0, 4001.0]
    assert log[1].source == "ramsey fit"

    # every historical version is still resolvable, byte-identical (reproducibility)
    assert store.get(cid1).calibration["qubit"]["f_ge"] == 4001.0
    assert store.get(base.content_id()).calibration["qubit"]["f_ge"] == 4000.0


def test_record_calibration_is_reproducible(tmp_path, dev):
    store = FileConfigStore(tmp_path / "s")
    base = dev.to_config()
    store.put(base)
    cid_a, _ = record_calibration(store, "a", base, {"qubit.f_ge": 4005.0},
                                  source="x", timestamp="2026-07-23T00:00:00Z")
    cid_b, _ = record_calibration(store, "b", base, {"qubit.f_ge": 4005.0},
                                  source="y", timestamp="2026-07-23T09:99:99Z")
    assert cid_a == cid_b   # identical calibration -> identical id, regardless of provenance


def test_store_satisfies_protocol(tmp_path):
    from strumento.core.store import ConfigStore

    assert isinstance(FileConfigStore(tmp_path / "s"), ConfigStore)


def test_provenance_round_trips():
    p = Provenance(content_id="cfg-abc", parent_id=None, source="s", updates={"x": [1, 2]})
    assert Provenance(**p.to_dict()).updates == {"x": [1, 2]}


# -- a write-back is validated before the ref moves (#191) --------------------------------

#: a typo'd key, a wrong type, non-finite values, a negative time and a negative gain code
INVALID_UPDATES = {
    "typo-key": {"qubit.pulses.pi_ge.gian": 9000},
    "wrong-type": {"qubit.f_ge": "abc"},
    "nan-time": {"readout.relax_delay": float("nan")},
    "inf-frequency": {"qubit.f_ge": float("inf")},
    "negative-time": {"readout.trig_offset": -5},
    "negative-gain": {"readout.gain": -100},
}


def _snapshot(root):
    """The full on-disk state of a FileConfigStore: every file's relative path and bytes."""
    return {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("updates", list(INVALID_UPDATES.values()), ids=list(INVALID_UPDATES))
def test_an_invalid_write_back_raises_and_leaves_the_store_unchanged(tmp_path, dev, updates):
    root = tmp_path / "s"
    store = FileConfigStore(root)
    base = dev.to_config()
    store.put(base)
    store.set_ref("main", base.content_id())
    before = _snapshot(root)
    with pytest.raises(ValueError):
        record_calibration(store, "main", base, updates, source="bad fit")
    assert _snapshot(root) == before                       # no object, no provenance, no ref move
    assert store.resolve_ref("main") == base.content_id()
    assert Device.from_config(store.get(store.resolve_ref("main"))).calib.qubit.f_ge == 4000.0


def test_the_validator_is_injectable_and_runs_before_anything_is_written(tmp_path, dev):
    """The store composes no device itself — the caller may hand it the validator to run."""
    root = tmp_path / "s"
    store = FileConfigStore(root)
    seen = []

    def refuse(config):
        seen.append(config.calibration["qubit"]["f_ge"])
        raise ValueError("refused by the caller's validator")

    with pytest.raises(ValueError, match="caller's validator"):
        record_calibration(store, "main", dev.to_config(), {"qubit.f_ge": 4001.0}, validate=refuse)
    assert seen == [4001.0]                                # it saw the candidate, not the base
    assert _snapshot(root) == {}


def test_a_valid_write_back_hashes_exactly_as_the_candidate_config(tmp_path, dev):
    """Validation never touches the bytes that are hashed: the id is the candidate's own id."""
    store = FileConfigStore(tmp_path / "s")
    base = dev.to_config()
    expected, _ = apply_updates(base, {"qubit.f_ge": 4001.0})
    cid, cfg = record_calibration(store, "main", base, {"qubit.f_ge": 4001.0})
    assert cid == expected.content_id() == cfg.content_id()
    assert store.get(cid).canonical_json() == expected.canonical_json()
