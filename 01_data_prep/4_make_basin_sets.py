"""basins/basin_sets.csv with in_training_pool and in_eval_set (filled in later by
07_analysis/eval_set.py once every run exists).

RUN: python 01_data_prep/4_make_basin_sets.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402

pool_df = pd.read_csv(E.POOL_CSV, dtype={"gauge_id": str})
pool = sorted({g.split("_")[-1] for g in pool_df["gauge_id"]})
if E.SMOKE:
    pool = pool[::len(pool) // E.SMOKE][:E.SMOKE]

no_forcing = [g for g in pool if not (E.FORCING_DIR / f"varssim_{g}.csv").exists()]
if no_forcing:
    raise SystemExit(f"pool basins without a forcing CSV: {no_forcing}")

out = pd.DataFrame({"gauge_id": pool})
out["in_training_pool"] = True
out["in_eval_set"] = pd.NA
E.BASIN_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(E.BASIN_SETS_CSV, index=False)
print(f"training pool: {len(pool)} basins -> {E.BASIN_SETS_CSV}")
