#!/usr/bin/env bash
# Start Flask API (port 5051) + Next.js frontend (port 3000) together.
# Ctrl-C kills both.

ROOT="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$FLASK_PID" "$NEXT_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

echo "[1/2] Starting Flask API on port 5051..."
python "$ROOT/webapp/api/app.py" &
FLASK_PID=$!

echo "[2/2] Starting Next.js on port 3000..."
cd "$ROOT/webapp/nextjs" && npm run dev &
NEXT_PID=$!

echo ""
echo "  Flask API : http://localhost:5051"
echo "  Frontend  : http://localhost:3000"
echo ""
echo "Press Ctrl-C to stop both."

wait
