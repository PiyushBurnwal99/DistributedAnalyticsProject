# Spark 4.0 image with the Python libraries the controller needs on every node.
FROM apache/spark:4.0.0-python3

USER root
RUN pip install --no-cache-dir "pandas>=2.2" "pyarrow>=15" "numpy>=1.26" "matplotlib>=3.8"
WORKDIR /app
