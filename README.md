# Distributed Control System for a Robotic Manufacturing Assembly (Apache Spark)

**Course:** IM60210 Distributed Analytics for Industrial Automation, IIT Kharagpur (Autumn 2026–27)
**Author:** Piyush Kumar Burnwal (22IM30038) · **Technology:** Apache Spark 4.0 (Structured Streaming + MLlib)

A closed-loop, distributed controller for a four-station robotic assembly line. The robots stream telemetry, and Apache Spark processes it in micro-batches. Each station's controller runs as its own task on the cluster and keeps its memory in Spark's fault-tolerant state store. It sends back speed set-points, predictive-maintenance orders and safety commands. A fault-risk model trained with **Spark MLlib** drives the predictive maintenance.

See [docs/architecture.md](docs/architecture.md) for the full diagram.

```
 robots ──telemetry──► Spark Structured Streaming ──commands──► robots
                        ├─ groupBy(station).applyInPandasWithState   (distributed, stateful control)
                        ├─ event-time windows + watermark             (KPIs → Parquet)
                        └─ raw archive                                (history → Parquet → MLlib)
```

## Results

**Policy benchmark:** 5 seeds × 6 simulated hours each, run as parallel Spark tasks (`make benchmark`):

| Policy | Throughput (units/h) | vs fixed | Robot faults per shift | Planned maintenances | E-stops |
|---|---|---|---|---|---|
| Fixed speed (baseline) | 64.1 ± 1.5 | — | 51.2 ± 3.1 | 0 | 8.4 |
| Throughput balancing only | 62.9 ± 1.9 | −1.8% | 65.0 ± 4.6 | 0 | 142.0 |
| **Balancing + predictive maintenance** | **71.7 ± 2.3** | **+11.9%** | **27.2 ± 3.1** | 43.6 | 0.0 |

- Balancing on its own does **not** help. Pushing worn robots faster raises wear (wear grows with speed³), causing more faults and e-stops.
- Predictive maintenance turns unplanned 10-minute breakdowns into planned 3-minute services. That cuts faults by 47% and is what makes faster running pay off.
- No policy reaches the 90 units/h demand, because downtime limits the line. This is a capacity problem, not a control problem.

**Fault-risk model** (Spark MLlib, time-based split, `models/health_model_metrics.json`): ROC-AUC 0.92, PR-AUC 0.46. At the 0.8 threshold, precision is 0.47 and recall is 0.57 (fault-within-5-minutes label, 9.5% positives).

**Closed-loop latency:** in the streaming demo (Spark local mode), commands reach the line a median of 2–2.5 s (4–5 steps) after the telemetry they respond to, so the controller suits supervisory control (set-points, maintenance) but not servo loops.

Charts from a streaming run (`make demo`) are written to `reports/`: throughput vs demand, speed set-points, fault risk with maintenance orders, and the time share of each robot state.

## Project layout

```
config/line_config.json        stations, demand, control gains, safety limits
src/rac/
  simulator.py                 discrete-time line simulator (online streaming + offline modes)
  control_law.py               per-station control law (pure pandas, unit-tested)
  spark_controller.py          Structured Streaming job: control + KPIs + history
  schemas.py                   Spark schemas (telemetry, controller state, commands)
  train_health_model.py        Spark MLlib training of the fault-risk model
  benchmark.py                 Spark-parallel policy benchmark
  report.py                    charts + summary for a streaming run
models/                        trained model (+ hand-tuned fallback) and metrics
tests/                         23 tests: control law, simulator, Spark end-to-end
docker/                        Spark standalone cluster (1 master, 2 workers) via docker compose
docs/architecture.md           diagrams, control law, data contracts
```

## Quick start

Requirements: Python 3.10+, Java 17 or 21.

```bash
pip install -r requirements.txt
make test         # 23 tests, including an end-to-end Spark streaming test
make train        # simulate run-to-failure history and train the MLlib model
make demo         # closed loop: Spark controller + streaming simulator, then reports/
make benchmark    # compare control policies (Spark distributes the simulations)
```

Running the pieces by hand (two terminals):

```bash
export PYTHONPATH=$PWD/src
python -m rac.spark_controller --master "local[4]" --console          # terminal 1
python -m rac.simulator online --steps 720 --real-interval 0.5        # terminal 2
python -m rac.report
```

### On a Spark cluster

```bash
make cluster      # docker compose: spark-master, 2 workers, controller (spark-submit), line simulator
```

Spark UI: <http://localhost:8080> (master) and <http://localhost:4040> (controller job). The controller zips the `rac` package and ships it to the executors with `addPyFile`, so the workers only need pandas and pyarrow.

## How the controller works

For every micro-batch, for each station (in parallel):

1. **Score health:** the MLlib logistic model turns temperature, vibration, torque, position error and speed into a fault risk.
2. **Safety interlock:** hard limits trigger `ESTOP`. `RESET` follows once the robot cools down, or maintenance if its risk is high.
3. **Predictive maintenance:** risk ≥ 0.8 schedules a service (one robot per station at a time), and risk ≥ 0.5 caps the robot's speed.
4. **Balance throughput:** a feed-forward speed from the takt time plus a PI correction on the measured output rate. Anti-windup applies when the station is blocked (it backs off) or starved (it holds speed), and set-points are rate-limited and clamped.

Details: [docs/architecture.md](docs/architecture.md).

## Limitations and next steps

- The plant is simulated. On a real line the file source would be replaced by Kafka/MQTT, and commands would go to the robot controllers through OPC UA.
- File-based exchange adds latency of a few seconds. That suits supervisory control (set-points, maintenance) but not servo-level loops, which stay on the robot controller.
- Possible extensions: a Kafka source/sink, `transformWithState` (Spark 4) with timers, and a model that predicts remaining useful life instead of a fixed horizon.

## License

MIT
