"""Tests for the readout-kind protocols (plan Task 16, spec §17)."""

from strumento.core.readout import DataSource, Discriminator, ReadoutKind


def test_data_source_enum_has_both_modalities():
    assert DataSource.QICK_ADC.value == "qick_adc"
    assert DataSource.EXTERNAL_DETECTOR.value == "external_detector"


def test_protocols_are_structural():
    class Kind:
        data_source = DataSource.QICK_ADC
        def discriminate(self, raw): return raw
        def reduce(self, raw): return 0.0

    k = Kind()
    assert isinstance(k, ReadoutKind)
    assert isinstance(k, Discriminator)

    class NotAKind:
        def discriminate(self, raw): return raw

    assert not isinstance(NotAKind(), ReadoutKind)  # missing reduce / data_source
