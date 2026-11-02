# Control policy benchmark

5 seeds x 2160 steps (6.0 simulated hours each); demand 90 units/h.

| Policy | Throughput (units/h) | vs fixed | Faults | Planned maintenances | E-stops |
|---|---|---|---|---|---|
| fixed | 64.1 ± 1.5 | +0.0% | 51.2 ± 3.1 | 0.0 | 8.4 |
| balancing | 62.9 ± 1.9 | -1.8% | 65.0 ± 4.6 | 0.0 | 142.0 |
| full | 71.7 ± 2.3 | +11.9% | 27.2 ± 3.1 | 43.6 | 0.0 |
