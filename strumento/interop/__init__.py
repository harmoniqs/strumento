"""Upstream bridges from a framework-neutral :class:`~strumento.core.operators.OperatorModel`.

These live **outside** ``core`` on purpose (D31): strumento owns the *export*; the
*solve* stays upstream. A bridge shapes the neutral model into a specific solver's convention —
``piccolo`` here — without ``core`` depending on that solver.
"""
