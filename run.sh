#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Run ./setup.sh first."; exit 1; }
source .venv/bin/activate
echo "Starting at http://127.0.0.1:5000  (Ctrl+C to stop)"
python app.py
