"""Training pool: CAMELS-US basins with complete data over warm-up + train + val + test.

A basin is kept when
  1. it is a canonical CAMELS-US gauge (listed in camels_topo.txt; drops 06775500, 06846500, 09535100,
     which have Daymet forcing files but are not part of the 671-basin set),
  2. it has a forcing CSV (01_data_prep/1_create_forcing_csv.py) with no NaN in any of
     Obs flow, Precip, Temp, PET on any day from DATA_START (1987-01-01) to TEST end (2006-12-31),
  3. it has a row in the static-attribute table (01_data_prep/2_build_attribute_table.py), and
  4. it is not in config.experiment.EXCLUDED_BASINS.

In:  data/forcing/varssim_US/varssim_<gauge>.csv, data/camels_us/camels_topo.txt, data/attributes/...
Out: data/basin_lists/us_camels_file_num_full.csv   every canonical gauge with a forcing CSV
     data/basin_lists/us_camels_file_num.csv        the training pool (E.POOL_CSV)

RUN: python 01_data_prep/3_basin_filter.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402

DATA_COLS = [E.Q_COL] + E.DYN_COLS
expected = pd.date_range(E.DATA_START, E.DATA_END, freq="D")


def has_full_data(gauge_id):
    df = pd.read_csv(E.FORCING_DIR / f"varssim_{gauge_id}.csv", parse_dates=["Date"]).set_index("Date")
    period = df.loc[E.DATA_START:E.DATA_END, DATA_COLS]
    return len(period) == len(expected) and bool(period.notna().all().all())


topo = pd.read_csv(E.CAMELS_ATTR_DIR / "camels_topo.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
have_csv = sorted(p.name[len("varssim_"):-len(".csv")] for p in E.FORCING_DIR.glob("varssim_*.csv"))
if not have_csv:
    raise SystemExit(f"no varssim_*.csv files in {E.FORCING_DIR} (run 1_create_forcing_csv.py first)")
non_canonical = sorted(set(have_csv) - set(topo.index))
gauges = [g for g in have_csv if g in topo.index]

full = pd.DataFrame({"gauge_id": [f"camels_{g}" for g in gauges],
                     "lon": topo.loc[gauges, "gauge_lon"].to_numpy(),
                     "lat": topo.loc[gauges, "gauge_lat"].to_numpy(),
                     "area": topo.loc[gauges, "area_gages2"].to_numpy()})
full.insert(0, "File_num", range(1, len(full) + 1))
E.BASIN_LIST_DIR.mkdir(parents=True, exist_ok=True)
full.to_csv(E.BASIN_LIST_DIR / "us_camels_file_num_full.csv", index=False)

attr_ids = set(pd.read_csv(E.ATTRIBUTE_CSV, dtype={"gauge_id": str})["gauge_id"])
complete = [g for g in gauges if has_full_data(g)]
no_attr = [g for g in complete if g not in attr_ids]
excluded = [g for g in complete if g in E.EXCLUDED_BASINS]
pool = [g for g in complete if g in attr_ids and g not in E.EXCLUDED_BASINS]

out = full[full["gauge_id"].str.split("_").str[-1].isin(pool)]
out.to_csv(E.POOL_CSV, index=False)
print(f"forcing CSVs: {len(have_csv)}; non-canonical gauges dropped: {non_canonical}")
print(f"canonical: {len(gauges)}; complete {E.DATA_START.date()}..{E.DATA_END.date()}: {len(complete)}")
print(f"no attribute row: {no_attr}; excluded by EXCLUDED_BASINS: {excluded}")
print(f"training pool: {len(pool)} basins -> {E.POOL_CSV}")
