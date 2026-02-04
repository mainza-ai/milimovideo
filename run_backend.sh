#!/bin/bash
# Convenience script to run the Milimo Video Backend
# Usage: ./run_backend.sh

# Ensure we are in the project root
cd "$(dirname "$0")"

echo "Using python: ./milimov/bin/python"
echo "Starting Backend Server..."

# Run the server module
./milimov/bin/python backend/server.py
