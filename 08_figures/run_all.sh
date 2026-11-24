#!/bin/bash
# All paper figures and tables (fig01-fig06, s01-s04) from the saved analysis outputs.
# RUN (from anywhere): bash 08_figures/run_all.sh        ENV: PYTHON (default: python)
set -euo pipefail
FIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
for s in fig01_s01_accuracy fig02_s02_shap fig03_mass_balance fig04_drought fig05_s03_flood fig06_s04_ranking; do
    echo "== $s"
    "$PY" "$FIG_DIR/$s.py"
done
"$PY" - <<PYEOF
import sys; sys.path.insert(0, "$FIG_DIR/..")
from config import experiment as E
E.MARK_DIR.mkdir(parents=True, exist_ok=True)
(E.MARK_DIR / "figures.done").write_text("ok\n")
print(f"paper figures written to {E.FIG_DIR}")
PYEOF
