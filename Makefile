PY ?= python3
export PYTHONPATH := $(CURDIR)/src

.PHONY: install test history train demo benchmark report cluster clean

install:            ## install Python dependencies
	$(PY) -m pip install -r requirements.txt

test:               ## run unit + Spark integration tests
	$(PY) -m pytest -q

history:            ## simulate run-to-failure history with fault labels (training data)
	$(PY) -m rac.simulator offline --steps 5000 --runs 4 --policy explore

train: history      ## train the fault-risk model with Spark MLlib
	$(PY) -m rac.train_health_model --master "local[4]"

demo:               ## closed-loop streaming demo (Spark local mode) + report
	bash scripts/run_demo.sh 360 0.5

benchmark:          ## compare fixed / balancing / full control over 5 seeds (Spark-parallel)
	$(PY) -m rac.benchmark --master "local[4]" --seeds 5

report:             ## charts for the last streaming run
	$(PY) -m rac.report

cluster:            ## run everything on a dockerised Spark cluster (1 master, 2 workers)
	docker compose -f docker/docker-compose.yml up --build

clean:
	rm -rf data/stream data/output data/checkpoints logs spark-warehouse
