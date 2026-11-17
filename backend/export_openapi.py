"""Export the FastAPI schema for client type generation.

Usage (from the repo root):
    ./milimov/bin/python backend/export_openapi.py > web-app/openapi.json
"""

import io
import json
import sys

# Importing the server configures stdout logging; keep stdout clean for JSON.
_sink = io.StringIO()
_stdout = sys.stdout
sys.stdout = _sink
try:
    from server import app
finally:
    sys.stdout = _stdout

print(json.dumps(app.openapi(), indent=2, sort_keys=True))
