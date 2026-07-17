"""Modality packs — reusable physics kits.

A pack supplies element *types*, pulse libraries, a readout kind, experiment
types, and a calibration *schema* for one physics class. Packs are discovered
via the ``qicklab.packs`` entry-point group, so third-party packs install
rather than fork.

Shipped: ``cqed`` (superconducting transmon + bosonic). Designed (see the spec):
``tweezer`` (neutral-atom AOD) and ``spin`` (quantum-dot).
"""
