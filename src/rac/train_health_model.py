"""Train the robot health (fault-risk) model with Spark MLlib.

Input : labelled telemetry history (``label_fault_soon`` = robot faults within the horizon).
Model : VectorAssembler -> StandardScaler -> class-weighted LogisticRegression.
Output: models/health_model.json (means, stds, coefficients, intercept) that the streaming
        controller scores with plain numpy on every micro-batch, plus metrics JSON.

The split is by time (first 80% of steps of every run for training) to avoid leaking the
future into the training set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from .config import PROJECT_ROOT
from .control_law import FEATURES

ACTIVE_STATES = ["RUNNING", "BLOCKED", "STARVED", "IDLE"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the fault-risk model with Spark MLlib")
    ap.add_argument("--master", default=None)
    ap.add_argument("--input", default="data/history/telemetry_labeled.parquet")
    ap.add_argument("--model-out", default=str(PROJECT_ROOT / "models" / "health_model.json"))
    ap.add_argument("--metrics-out", default=str(PROJECT_ROOT / "models" / "health_model_metrics.json"))
    ap.add_argument("--reg", type=float, default=0.01)
    args = ap.parse_args(argv)

    b = SparkSession.builder.appName("robot-health-model-training")
    if args.master:
        b = b.master(args.master)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = (spark.read.parquet(args.input)
          .where(F.col("state").isin(ACTIVE_STATES))  # only robots that are actually in service
          .select("run_id", "step", *FEATURES, F.col("label_fault_soon").cast("double").alias("label")))
    max_step = df.groupBy("run_id").agg(F.max("step").alias("max_step"))
    df = df.join(max_step, "run_id").withColumn("is_train", F.col("step") <= 0.8 * F.col("max_step"))
    train, test = df.where("is_train"), df.where("NOT is_train")

    pos = train.where("label = 1").count()
    total = train.count()
    w_pos = (total - pos) / max(pos, 1)  # balance the classes
    train = train.withColumn("w", F.when(F.col("label") == 1, F.lit(w_pos)).otherwise(F.lit(1.0)))

    pipeline = Pipeline(stages=[
        VectorAssembler(inputCols=FEATURES, outputCol="raw"),
        StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True),
        LogisticRegression(featuresCol="features", labelCol="label", weightCol="w",
                           regParam=args.reg, elasticNetParam=0.0, maxIter=100),
    ])
    fitted = pipeline.fit(train)
    scaler, lr = fitted.stages[1], fitted.stages[2]

    pred = fitted.transform(test)
    auc_roc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(pred)
    auc_pr = BinaryClassificationEvaluator(metricName="areaUnderPR").evaluate(pred)
    p1 = F.udf(lambda v: float(v[1]), "double")
    pred = pred.withColumn("p", p1("probability"))
    rates = {}
    for thr in (0.5, 0.8):
        flagged = pred.where(F.col("p") >= thr)
        tp = flagged.where("label = 1").count()
        rates[str(thr)] = {
            "precision": round(tp / max(flagged.count(), 1), 4),
            "recall": round(tp / max(pred.where("label = 1").count(), 1), 4),
        }

    model = {
        "source": f"Spark MLlib LogisticRegression trained on {args.input}",
        "features": FEATURES,
        "means": [float(x) for x in scaler.mean],
        "stds": [float(x) for x in scaler.std],
        "coefficients": [float(x) for x in lr.coefficients],
        "intercept": float(lr.intercept),
    }
    metrics = {
        "train_rows": total, "train_positive_share": round(pos / max(total, 1), 4),
        "test_rows": test.count(), "auc_roc": round(auc_roc, 4), "auc_pr": round(auc_pr, 4),
        "threshold_metrics": rates,
        "coefficients": dict(zip(FEATURES, [round(c, 4) for c in model["coefficients"]])),
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_out).write_text(json.dumps(model, indent=2))
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"[train] model written to {args.model_out}")
    spark.stop()


if __name__ == "__main__":
    main()
