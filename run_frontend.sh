#!/bin/bash
# Convenience script to run the Milimo Video Frontend
# Usage: ./run_frontend.sh

# Ensure we are in the project root
cd "$(dirname "$0")"

cd web-app

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo "Starting Web Interface..."
npm run dev
