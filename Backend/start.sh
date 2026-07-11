#!/usr/bin/env bash
# Run from the src/ dir so the top-level imports (routes, services, db, ...) resolve.
cd "$(dirname "$0")/src" && exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-10000}"
