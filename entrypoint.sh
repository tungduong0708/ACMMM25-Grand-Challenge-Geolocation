#!/usr/bin/env bash
set -euo pipefail

# --- cleanup any stale Xvfb lock/socket ---
if [ -e /tmp/.X99-lock ]; then
  echo "[entrypoint] removing stale /tmp/.X99-lock" >&2
  rm -f /tmp/.X99-lock
fi
if [ -e /tmp/.X11-unix/X99 ]; then
  echo "[entrypoint] removing stale /tmp/.X11-unix/X99" >&2
  rm -f /tmp/.X11-unix/X99
fi

# --- start the virtual display ---
echo "[entrypoint] starting Xvfb on :99" >&2
Xvfb :99 -screen 0 1920x1080x24 &

# --- point GUI apps at it ---
export DISPLAY=:99
echo "[entrypoint] DISPLAY set to $DISPLAY" >&2

# --- launch FastAPI ---
echo "[entrypoint] exec fastapi" >&2
exec fastapi run app.py --port 80
