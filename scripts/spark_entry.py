"""Entry point for spark-submit: `spark-submit scripts/spark_entry.py rac.spark_controller [args...]`."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: spark_entry.py <module> [args...]  e.g. rac.spark_controller --master spark://...")
    module = importlib.import_module(sys.argv[1])
    module.main(sys.argv[2:])
