"""The cqed pack's experiments: one module per experiment, registered by name in
:data:`strumento.packs.cqed.EXPERIMENTS`.

Three modules here are deliberately **not** registered: ``kerr_cat_witness``,
``ladder_population`` and ``self_kerr_ramsey``. Each is a *host-batched diagnostic* — it needs
several independent programs (one per ladder block, Wigner ring point or coherent amplitude),
each relaxing the device before the next, and a registered ``Experiment`` is exactly one program.
A single stacked program would carry one block's end state into the next, because qick's relax
fires once per shot, after the whole body. So these modules hold only the pure halves — the
per-program sequence builders and the analyzers — and the caller's runner owns the batching.
``docs/cqed-cookbook.md`` ("Host-batched diagnostics") shows how to drive them.
"""
