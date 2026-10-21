"""spin experiments: full tune-up sequence for semiconductor spin qubits.

Organized by layer in the calibration DAG:
- Layer 0: Electrostatic bring-up (pinch-off, Coulomb diamonds, virtual gates, tunnel coupling)
- Layer 1: Sensor / readout calibration (sensor tune, RF, Elzerman, PSB, dispersive gate)
- Layer 2: Qubit frequency & single-qubit control (T1, EDSR, Rabi, Ramsey, echo, AllXY, exchange)
- Layer 3: Two-qubit & advanced (exchange gate, leakage, composite pulses)
- Layer 4: Benchmarking (RB, interleaved RB, charge noise spectroscopy)

Plus the original host-driven utilities: ChargeStability, PSBHistogram.
"""
