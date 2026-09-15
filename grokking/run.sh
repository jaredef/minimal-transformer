#!/usr/bin/env sh
# The scaled, genuine-grokking companion. Requires numpy (pip install -r requirements.txt).
# Trains a few small models by AdamW, so this takes ~1-2 minutes, unlike the instant NAND core.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY=${PYTHON:-python3}
cd "$ROOT"

echo "== one grokking run (train fits early, val generalizes late, norm turns over) =="
"$PY" modadd.py | tail -8
echo
echo "== threshold: grokking appears only above a data fraction =="
"$PY" threshold.py
echo
echo "== ablation falsifier: remove weight decay or the quadratic activation and grokking vanishes =="
"$PY" ladder.py
echo
echo "== constructed prior: the grokked embedding converges to Gromov's Fourier solution =="
"$PY" fourier.py
