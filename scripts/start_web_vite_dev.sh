#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT="${VACXIS_WEB_VITE_PORT:-8001}"

echo "Starting VaxiS Web Vite..."
echo "  Backend:  http://127.0.0.1:${BACKEND_PORT}"
echo "  Frontend: http://localhost:5173"

cd "${ROOT_DIR}"
uv run vacxis-web-vite --port "${BACKEND_PORT}" "$@" &
BACKEND_PID=$!
FRONTEND_PID=""

cleanup() {
    kill "${BACKEND_PID}" 2>/dev/null || true
    if [ -n "${FRONTEND_PID}" ]; then
        kill "${FRONTEND_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

sleep 2

cd "${ROOT_DIR}/web-vite"
VITE_API_TARGET="http://127.0.0.1:${BACKEND_PORT}" npm run dev &
FRONTEND_PID=$!

wait
