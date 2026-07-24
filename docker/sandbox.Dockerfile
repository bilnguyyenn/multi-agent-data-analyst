# Prebuilt execution image: per-run cost is container START, not pip install.
# Build once:  docker build -t analyst-sandbox:latest -f docker/sandbox.Dockerfile .
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    pandas==2.2.* \
    numpy==2.* \
    scipy==1.* \
    scikit-learn==1.* \
    statsmodels==0.14.* \
    duckdb==1.* \
    matplotlib==3.*

# Non-root by default even if the runner forgets --user (defense in depth)
USER 65534:65534
WORKDIR /out
