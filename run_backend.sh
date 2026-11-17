#!/bin/bash
# Convenience script to run the Milimo Video Backend
# Usage: ./run_backend.sh

cd "$(dirname "$0")"

PY="./milimov/bin/python"
if [ ! -x "$PY" ]; then
  echo "No venv at ./milimov. Create it and install dependencies first:" >&2
  echo "  python3 -m venv milimov" >&2
  echo "  ./milimov/bin/pip install -e ./LTX-2/packages/ltx-core -e ./LTX-2/packages/ltx-pipelines -e ./flux2 -r backend/requirements.txt" >&2
  exit 1
fi

echo "Checking setup..."
"$PY" backend/verify_setup.py || exit 1

echo "Starting Backend Server..."
exec "$PY" backend/server.py
