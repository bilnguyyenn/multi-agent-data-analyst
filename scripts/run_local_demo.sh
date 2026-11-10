#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python -c 'import analyst, fastapi, duckdb' >/dev/null 2>&1 || {
  echo 'Install once: python -m pip install -e ".[retail]"'; exit 1;
}
command -v npm >/dev/null || { echo 'Node.js and npm are required'; exit 1; }
python - <<'PY'
import socket
for port in [8027, 5177]:
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', port))
        except OSError:
            raise SystemExit(f'Port {port} is in use; stop the existing demo yourself or use different ports.')
PY
if [ ! -d frontend/node_modules ]; then (cd frontend && npm ci --no-audit --no-fund); fi
python -m analyst retail-review --start 2011-11-01 --end 2011-11-08 --out .operations/demo-first.html
python -m analyst retail-review --data data/retail/refresh/transactions.csv --start 2011-11-08 --end 2011-11-15 --out .operations/demo-refresh.html
ANALYST_DEV_INPROCESS=1 python -m uvicorn analyst.api.app:app --host 127.0.0.1 --port 8027 &
api_pid=$!
(cd frontend && ANALYST_API_URL=http://127.0.0.1:8027 exec node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5177 --strictPort) &
ui_pid=$!
trap 'kill "$api_pid" "$ui_pid" 2>/dev/null || true' EXIT INT TERM
echo 'Open http://127.0.0.1:5177/?workspace=operations'
echo 'Retail review needs no Docker, model, credentials, or paid service. Ctrl-C stops these servers.'
wait
