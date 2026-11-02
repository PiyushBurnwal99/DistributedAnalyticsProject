"""Benchmark control policies over many seeds; Spark distributes the simulations.

Each (policy, seed) pair is an independent simulation, so the benchmark is an
embarrassingly parallel job: Spark spreads the pairs over its executors and
collects the summaries. Policies compared:

* fixed      - all robots at nominal speed, hard-wired safety interlock only (baseline)
* balancing  - PI + feed-forward throughput balancing, no predictive maintenance
* full       - balancing + MLlib-based predictive maintenance (the deployed controller)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config

POLICIES = ["fixed", "balancing", "full"]


def _run_one(job: tuple) -> dict:
    """One simulation; runs on a Spark executor. Config and model travel as plain dicts."""
    policy, seed, steps, cfg_dict, model_dict = job
    from .config import LineConfig
    from .control_law import HealthModel
    from .simulator import run_offline

    cfg = LineConfig.from_dict(cfg_dict)
    _, s = run_offline(cfg, steps, policy, seed=seed, model=HealthModel(**model_dict), record=False)
    s.update({"policy": policy, "seed": seed})
    return s


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare control policies")
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--engine", choices=["spark", "local"], default="spark")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--steps", type=int, default=2160, help="2160 steps x 10 s = one 6-hour shift")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args(argv)

    from .control_law import HealthModel

    cfg = load_config(args.config)
    model = HealthModel.load(args.model)
    jobs = [(p, 1000 + s, args.steps, cfg.to_dict(), model.to_dict()) for p in POLICIES for s in range(args.seeds)]
    if args.engine == "spark":
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.master(args.master).appName("control-policy-benchmark").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        from .spark_controller import ship_package

        ship_package(spark)
        results = spark.sparkContext.parallelize(jobs, len(jobs)).map(_run_one).collect()
        spark.stop()
    else:
        results = [_run_one(j) for j in jobs]

    df = pd.DataFrame(results)
    print(f"[benchmark] {len(df)} simulations done; health model: {model.source}")
    agg = (df.groupby("policy")[["throughput_uph", "faults", "maintenances", "estops"]]
           .agg(["mean", "std"]).reindex(POLICIES))
    base = agg.loc["fixed", ("throughput_uph", "mean")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark_runs.csv", index=False)

    lines = [
        f"# Control policy benchmark",
        "",
        f"{args.seeds} seeds x {args.steps} steps ({args.steps * cfg.sim_dt_seconds / 3600:.1f} simulated hours each); "
        f"demand {cfg.demand_units_per_hour:.0f} units/h.",
        "",
        "| Policy | Throughput (units/h) | vs fixed | Faults | Planned maintenances | E-stops |",
        "|---|---|---|---|---|---|",
    ]
    for p in POLICIES:
        r = agg.loc[p]
        lines.append(
            f"| {p} | {r[('throughput_uph', 'mean')]:.1f} ± {r[('throughput_uph', 'std')]:.1f} | "
            f"{(r[('throughput_uph', 'mean')] / base - 1) * 100:+.1f}% | "
            f"{r[('faults', 'mean')]:.1f} ± {r[('faults', 'std')]:.1f} | "
            f"{r[('maintenances', 'mean')]:.1f} | {r[('estops', 'mean')]:.1f} |")
    (out / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"runs": len(df)}))


if __name__ == "__main__":
    main()
