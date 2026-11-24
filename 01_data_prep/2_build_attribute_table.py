"""Static-attribute table used by regional/adapted models and the analysis strata.

Joins the CAMELS-US attribute files camels_{topo,clim,vege,soil,geol}.txt (Addor et al., 2017) on
gauge_id and keeps gauge_lat, gauge_lon and the 27 STATIC_ATTRS of config/experiment.py. Values are
copied unchanged.

In:  data/camels_us/camels_{topo,clim,vege,soil,geol}.txt
Out: data/attributes/camels_original_attribute_combined.csv

RUN: python 01_data_prep/2_build_attribute_table.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402

frames = [pd.read_csv(E.CAMELS_ATTR_DIR / f"camels_{n}.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
          for n in ["topo", "clim", "vege", "soil", "geol"]]
attrs = pd.concat(frames, axis=1)
attrs = attrs.loc[:, ~attrs.columns.duplicated()]
cols = ["gauge_lat", "gauge_lon"] + E.STATIC_ATTRS
missing = [c for c in cols if c not in attrs.columns]
if missing:
    raise SystemExit(f"attribute columns not found in the CAMELS files: {missing}")
out = attrs[cols].sort_index().reset_index()
out.insert(0, "File_num", range(1, len(out) + 1))
E.ATTRIBUTE_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(E.ATTRIBUTE_CSV, index=False)
print(f"{len(out)} basins x {len(cols)} attributes -> {E.ATTRIBUTE_CSV}")
