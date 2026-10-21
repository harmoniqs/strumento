# Example run script

`run_amplitude_rabi.py` runs one experiment (amplitude Rabi on the ge transition) the way a run on
real hardware should: it compares the live board with the device's stored description and refuses
on a mismatch, locks the board, compiles and checks what will actually be played before acquiring,
records the run (a task record and a data chest), judges the fit with explicit quality checks, and
reports the calibration edit it suggests without ever applying it.

```bash
python run_amplitude_rabi.py --device ../../devices/multimode_demo/device.yaml --sim
python run_amplitude_rabi.py --device my_fridge/device.yaml --check --ns-host <name server>
python run_amplitude_rabi.py --device my_fridge/device.yaml --arm --ns-host <name server>
```

Copy it and adapt it to your experiment. [docs/write-a-runner.md](../../docs/write-a-runner.md)
walks through every step and explains what to change.
