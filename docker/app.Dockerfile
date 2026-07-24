# One image for both api and worker — same code, different command.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
COPY analyst ./analyst
COPY config ./config
RUN pip install --no-cache-dir -e ".[postgres,otel]"

# api:    uvicorn analyst.api.app:app --host 0.0.0.0 --port 8000
# worker: arq analyst.api.worker.WorkerSettings
CMD ["uvicorn", "analyst.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
