"""End-to-end test of the Spark streaming controller on a few simulated steps."""
import json
import os
import shutil

import pytest

pyspark = pytest.importorskip("pyspark")

from rac.control_law import HealthModel, DEFAULT_MODEL  # noqa: E402
from rac.simulator import AssemblyLine  # noqa: E402
from rac.spark_controller import build_session, start_queries  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    s = build_session("local[2]", "rac-tests", 4)
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def test_streaming_controller_emits_commands(spark, cfg, tmp_path):
    tele, cmds_dir = tmp_path / "telemetry", tmp_path / "commands"
    tele.mkdir()
    line = AssemblyLine(cfg, seed=3)
    for i in range(6):
        with open(tele / f"telemetry_{i:07d}.json", "w") as f:
            for rec in line.step():
                rec.pop("true_wear")
                f.write(json.dumps(rec) + "\n")

    queries = start_queries(spark, cfg, HealthModel.load(DEFAULT_MODEL), str(tele), str(cmds_dir),
                            str(tmp_path / "out"), str(tmp_path / "ckpt"), 1.0, 3, available_now=True)
    for q in queries:
        q.awaitTermination()

    rows = []
    for p in cmds_dir.glob("part-*.json"):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    robots = {r["robot_id"] for r in rows}
    assert robots == set(line.index)  # every robot got commands, from every station key
    assert all(cfg.control.speed_min <= r["speed_setpoint"] <= cfg.control.speed_max for r in rows)
    # two micro-batches (3 files each); the latest step is always controlled
    steps = {r["issued_for_step"] for r in rows}
    assert max(steps) == 5 and len(steps) == 2
    assert (tmp_path / "out" / "history").exists()
