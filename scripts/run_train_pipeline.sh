#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "=== Building training features ==="
"$PYTHON" -m src.build_features --split train

echo ""
echo "=== Training models ==="
"$PYTHON" -m src.train_models

echo ""
echo "=== Running clustering ==="
"$PYTHON" -m src.clustering

echo ""
echo "=== Generating visualizations ==="
"$PYTHON" -m src.visualize

echo ""
echo "=== Pipeline complete ==="
