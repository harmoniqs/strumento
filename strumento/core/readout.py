"""Readout-kind seam (spec §15.2, §16.2, §17) — what makes ``core`` modality-general on readout.

A **readout kind** turns a raw measurement into physics: discriminate per-shot states, reduce to
populations. Different modalities read out differently — a cqed dispersive-IQ discriminator vs a
neutral-atom fluorescence-imaging detector — so ``core`` defines *protocols* and a **data-source**
enum, and each pack ships its own kind (``cqed`` → ``DispersiveIQ``; ``tweezer`` → an imaging
kind on ``EXTERNAL_DETECTOR``, later).

This module is **pure**: no ``qick`` import.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


class DataSource(Enum):
    """Where a readout kind's raw data comes from."""

    QICK_ADC = "qick_adc"               # IQ off the RFSoC ADC (dispersive readout)
    EXTERNAL_DETECTOR = "external_detector"  # host-side detector, e.g. tweezer fluorescence imaging


@runtime_checkable
class Discriminator(Protocol):
    """Assigns a discrete state label to each shot of raw readout data."""

    def discriminate(self, raw): ...


@runtime_checkable
class ReadoutKind(Protocol):
    """A modality's readout: its data source, per-shot discrimination, and population reduction."""

    data_source: DataSource

    def discriminate(self, raw): ...

    def reduce(self, raw): ...
