#!/bin/bash
# change to 
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export AE_MODEL_PATH="$SCRIPT_DIR/backend/models/flux2/vae/ae.safetensors"

# Fonction de nettoyage
cleanup() {
    echo "Arrêt des services..."
    kill $BACKEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
    echo "Services arrêtés."
    exit 0
}

# Intercepter Ctrl+C et kill
trap cleanup SIGINT SIGTERM

./run_backend.sh &
BACKEND_PID=$!

./run_frontend.sh &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID | Frontend PID: $FRONTEND_PID"

# Attendre que les deux tournent (garde le script actif)
wait
