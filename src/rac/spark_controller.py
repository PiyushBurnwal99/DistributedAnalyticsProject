"""Distributed controller: Spark Structured Streaming job that closes the control loop.

Three streaming queries share one telemetry source:

1. ``control``  - groupBy(station_id).applyInPandasWithState(...): every station is a key, so
                  Spark runs the stations' controllers in parallel on its executors and keeps
                  each station's controller memory (PI integral, filtered rate, last set-point)
                  in its fault-tolerant state store. Commands are written as JSON files that the
                  line (simulator) picks up.
2. ``kpis``     - event-time windowed KPIs per station (throughput, utilisation, blocked/starved
                  share, mean temperature and vibration) with a watermark -> Parquet.
3. ``history``  - raw telemetry archived to Parquet (input for MLlib training and reports).

Run locally:   python -m rac.spark_controller --master "local[4]"
Run on a cluster: spark-submit --master spark://spark-master:7077 ... (see docker/)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout

from .config import LineConfig, load_config
from .control_law import STATE_FIELDS, ControlModes, HealthModel, control_station, init_state
from .schemas import COMMAND_SCHEMA, CONTROLLER_STATE_SCHEMA, TELEMETRY_SCHEMA


def make_station_controller(cfg_dict: dict, model_dict: dict, balancing: bool, predictive: bool):
    """Build the function Spark calls once per station per micro-batch (runs on executors)."""

    def controller(key, pdf_iter, state: GroupState):
        cfg = LineConfig.from_dict(cfg_dict)
        model = HealthModel(**model_dict)
        events = pd.concat(list(pdf_iter), ignore_index=True)
        st = dict(zip(STATE_FIELDS, state.get)) if state.exists else init_state()
        commands, new_state = control_station(events, st, cfg, model, ControlModes(balancing, predictive))
        state.update(tuple(new_state[f] for f in STATE_FIELDS))
        if not commands.empty:
            commands["issued_for_step"] = commands["issued_for_step"].astype("int64")
            yield commands

    return controller


def ship_package(spark: SparkSession) -> None:
    """Zip the ``rac`` package and send it to every executor (needed on a real cluster)."""
    import shutil
    import tempfile

    pkg_dir = Path(__file__).resolve().parent
    tmp = Path(tempfile.mkdtemp(prefix="rac_pkg_"))
    archive = shutil.make_archive(str(tmp / "rac"), "zip", root_dir=pkg_dir.parent, base_dir=pkg_dir.name)
    spark.sparkContext.addPyFile(archive)


def build_session(master: str | None, app_name: str, shuffle_partitions: int) -> SparkSession:
    b = (SparkSession.builder.appName(app_name)
         .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
         .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
         .config("spark.sql.session.timeZone", "UTC"))
    if master:
        b = b.master(master)
    spark = b.getOrCreate()
    ship_package(spark)
    return spark


def start_queries(spark: SparkSession, cfg: LineConfig, model: HealthModel, telemetry_dir: str,
                  commands_dir: str, output_dir: str, checkpoint_dir: str, trigger_seconds: float,
                  max_files_per_trigger: int, balancing: bool = True, predictive: bool = True,
                  available_now: bool = False, console: bool = False):
    out = Path(output_dir)
    ckpt = Path(checkpoint_dir)

    telemetry = (spark.readStream.schema(TELEMETRY_SCHEMA)
                 .option("maxFilesPerTrigger", max_files_per_trigger)
                 .json(telemetry_dir)
                 .withColumn("event_time", F.to_timestamp("ts")))

    def trig(w):
        return w.trigger(availableNow=True) if available_now else w.trigger(processingTime=f"{trigger_seconds} seconds")

    # 1) distributed, stateful control loop
    controller = make_station_controller(cfg.to_dict(), model.to_dict(), balancing, predictive)
    commands = (telemetry.drop("event_time")
                .groupBy("station_id")
                .applyInPandasWithState(controller, COMMAND_SCHEMA, CONTROLLER_STATE_SCHEMA,
                                        outputMode="append", timeoutConf=GroupStateTimeout.NoTimeout))
    q_control = trig(commands.writeStream.queryName("control")
                     .format("json").option("path", commands_dir)
                     .option("checkpointLocation", str(ckpt / "control"))
                     .outputMode("append")).start()

    # 2) event-time KPIs per station and minute
    dt = cfg.sim_dt_seconds
    kpis = (telemetry.withWatermark("event_time", "2 minutes")
            .groupBy(F.window("event_time", "1 minute").alias("w"), "station_id")
            .agg(F.sum("cycle_completed").alias("units"),
                 (F.max("step") - F.min("step") + 1).alias("steps"),
                 F.avg(F.when(F.col("state") == "RUNNING", 1).otherwise(0)).alias("utilisation"),
                 F.avg(F.when(F.col("state") == "BLOCKED", 1).otherwise(0)).alias("blocked_share"),
                 F.avg(F.when(F.col("state") == "STARVED", 1).otherwise(0)).alias("starved_share"),
                 F.avg(F.when(F.col("state").isin("FAULT", "MAINTENANCE", "ESTOP"), 1).otherwise(0)).alias("down_share"),
                 F.avg("speed_factor").alias("avg_speed"),
                 F.avg("joint_temp_c").alias("avg_temp_c"),
                 F.max("vibration_mm_s").alias("max_vibration")))
    kpis = (kpis.withColumn("window_start", F.col("w.start"))
            .withColumn("window_end", F.col("w.end")).drop("w")
            .withColumn("throughput_uph", F.col("units") * F.lit(3600.0) / (F.col("steps") * F.lit(dt))))
    q_kpis = trig(kpis.writeStream.queryName("kpis").format("parquet")
                  .option("path", str(out / "kpis")).option("checkpointLocation", str(ckpt / "kpis"))
                  .outputMode("append")).start()

    # 3) raw history archive, partitioned by station
    q_hist = trig(telemetry.writeStream.queryName("history").format("parquet")
                  .partitionBy("station_id")
                  .option("path", str(out / "history")).option("checkpointLocation", str(ckpt / "history"))
                  .outputMode("append")).start()

    queries = [q_control, q_kpis, q_hist]
    if console:
        q_console = trig(commands.select("issued_for_step", "robot_id", "action", "speed_setpoint",
                                         "risk_score", "station_rate_uph", "reason")
                         .writeStream.queryName("console").format("console")
                         .option("truncate", "false").option("numRows", 50)
                         .option("checkpointLocation", str(ckpt / "console"))
                         .outputMode("append")).start()
        queries.append(q_console)
    return queries


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Spark distributed controller for the assembly line")
    ap.add_argument("--master", default=None, help='e.g. "local[4]" or "spark://spark-master:7077"')
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="health model JSON (default: trained, else fallback)")
    ap.add_argument("--telemetry-dir", default="data/stream/telemetry")
    ap.add_argument("--commands-dir", default="data/stream/commands")
    ap.add_argument("--output-dir", default="data/output")
    ap.add_argument("--checkpoint-dir", default="data/checkpoints")
    ap.add_argument("--trigger-seconds", type=float, default=1.5)
    ap.add_argument("--max-files-per-trigger", type=int, default=50,
                    help="files (steps) per micro-batch; high enough that Spark never falls behind")
    ap.add_argument("--no-balancing", action="store_true")
    ap.add_argument("--no-predictive", action="store_true")
    ap.add_argument("--available-now", action="store_true", help="process existing files and stop")
    ap.add_argument("--timeout", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--console", action="store_true", help="also print commands to the console")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    model = HealthModel.load(args.model)
    Path(args.telemetry_dir).mkdir(parents=True, exist_ok=True)
    spark = build_session(args.master, "robotic-assembly-distributed-control", len(cfg.stations))
    spark.sparkContext.setLogLevel("WARN")
    print(f"[controller] health model: {model.source}")
    print(f"[controller] stations: {[s.id for s in cfg.stations]} | target {cfg.demand_units_per_hour} units/h")

    queries = start_queries(spark, cfg, model, args.telemetry_dir, args.commands_dir, args.output_dir,
                            args.checkpoint_dir, args.trigger_seconds, args.max_files_per_trigger,
                            balancing=not args.no_balancing, predictive=not args.no_predictive,
                            available_now=args.available_now, console=args.console)
    try:
        if args.available_now:
            for q in queries:
                q.awaitTermination()
        else:
            spark.streams.awaitAnyTermination(timeout=args.timeout)
    except KeyboardInterrupt:
        pass
    finally:
        for q in queries:
            if q.isActive:
                q.stop()
        for q in queries:
            p = q.lastProgress
            if p:
                print(f"[controller] {q.name}: last batch {p['batchId']}, "
                      f"input rows/s {p.get('inputRowsPerSecond', 0):.1f}")
        spark.stop()


if __name__ == "__main__":
    main()
