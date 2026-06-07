#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "=== Building test features ==="
"$PYTHON" -m src.build_features --split test

echo ""
echo "=== Generating submission ==="
"$PYTHON" -m src.predict

echo ""
echo "=== Done ==="
