"""A recording fake for the qick program interface.

Records the build calls the compiler makes, so compiler/e2e logic can be verified with no qick
and no tProc-v2 firmware. A real ``StrumentoProgram`` satisfies the same duck-typed interface.
"""


class RecordingProgram:
    def __init__(self):
        self.calls = []

    def declare_gen(self, ch, **kw):        self.calls.append(("declare_gen", ch, kw))
    def add_gauss(self, ch, name, **kw):    self.calls.append(("add_gauss", ch, name, kw))
    def add_pulse(self, ch, name, **kw):    self.calls.append(("add_pulse", ch, name, kw))
    def add_envelope(self, ch, name, **kw): self.calls.append(("add_envelope", ch, name, kw))
    def pulse(self, ch, name, t=0, **kw):   self.calls.append(("pulse", ch, name, t))
    def delay(self, t, **kw):               self.calls.append(("delay", t))
    def delay_auto(self, t=0, **kw):        self.calls.append(("delay_auto", t))
    def read_and_jump(self, ro_ch=None, component=None, threshold=None, test=None, label=None, **kw):
        self.calls.append(("read_and_jump", ro_ch, component, threshold, test, label))
    def jump(self, label, **kw):            self.calls.append(("jump", label))
    def label(self, label, **kw):           self.calls.append(("label", label))
    def open_loop(self, n, name=None, **kw): self.calls.append(("open_loop", n, name))
    def close_loop(self, **kw):             self.calls.append(("close_loop",))
    def trigger(self, ros=None, t=0, **kw): self.calls.append(("trigger", tuple(ros or ()), t))
    def declare_readout(self, ch, length, **kw): self.calls.append(("declare_readout", ch, length))
    def add_loop(self, name, count, **kw):  self.calls.append(("add_loop", name, count))

    def sweep_value(self, name, start, stop):
        # the real program returns a qick QickSweep1D; the fake returns a recordable marker
        self.calls.append(("sweep_value", name, start, stop))
        return {"__sweep__": name, "start": start, "stop": stop}

    def kinds(self):
        return [c[0] for c in self.calls]

    def first(self, kind):
        return next(c for c in self.calls if c[0] == kind)
