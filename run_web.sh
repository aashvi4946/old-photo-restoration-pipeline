#!/usr/bin/env bash
# -------------------------------------------------------
# Old Photo Restorer — Web App Startup Script
# -------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/venv"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

# Check venv exists
if [ ! -f "$PYTHON" ]; then
  echo "[run_web] Creating virtual environment..."
  python3 -m venv "$VENV"
fi

# Install/upgrade web requirements
echo "[run_web] Checking dependencies..."
"$PIP" install -q flask flask-cors opencv-python numpy scipy pillow

PORT="${PORT:-8080}"
export PORT

# Free up the port if a previous server session is still running
EXISTING_PID=$(lsof -ti :"$PORT" 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
  echo "[run_web] Freeing port $PORT (killing existing process $EXISTING_PID)..."
  kill -9 $EXISTING_PID 2>/dev/null || true
  sleep 1
fi

echo ""
echo "============================================"
echo "  Old Photo Restorer — Web App"
echo "  http://localhost:$PORT"
echo "============================================"
echo ""

"$PYTHON" server.py

