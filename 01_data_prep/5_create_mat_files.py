"""HBV (MARRMoT) .mat forcing, built from the same per-basin CSVs the ML models read
(01_data_prep/1_create_forcing_csv.py), so both pipelines get identical forcing.

For each basin in the training pool:
  MARRMoT_InputData_US_v212/varssim_{id}.mat                     real forcing  DATA_START..DATA_END
  MARRMoT_InputData_US_v212_no_precip/varssim_{id}_no_precip.mat  drought       DATA_START..DATA_END
  MARRMoT_InputData_US_v212_up/varssim_{id}_up_{scen}.mat         flood + nostorm  DATA_START..FLOOD_WINDOW end
`dates_as_datenum` holds real MATLAB datenums, so the workflow reads its start date from the file.

RUN: python 01_data_prep/5_create_mat_files.py [n_jobs]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from core.io import load_basin_pool  # noqa: E402

OUT = {
    "og": E.HBV_MAT_ROOT / "MARRMoT_InputData_US_v212",
    "no_precip": E.HBV_MAT_ROOT / "MARRMoT_InputData_US_v212_no_precip",
    "up": E.HBV_MAT_ROOT / "MARRMoT_InputData_US_v212_up",
}


def datenum(dates: pd.DatetimeIndex) -> np.ndarray:
    """MATLAB datenum (days since year 0) of each date."""
    return np.array([d.toordinal() + 366 for d in dates], dtype=float)


def save_mat(path, gauge_id, dates, precip, temp, pet, q):
    rec = np.array([(
        np.array([gauge_id], dtype='O'),
        np.array([gauge_id], dtype='O'),
        datenum(dates).reshape(-1, 1),
        np.asarray(precip, float).reshape(-1, 1),
        np.asarray(temp, float).reshape(-1, 1),
        np.asarray(pet, float).reshape(-1, 1),
        np.asarray(q, float).reshape(-1, 1),
    )], dtype=[('name', 'O'), ('gaugeID', 'O'), ('dates_as_datenum', 'O'), ('precipitation', 'O'),
               ('temperature', 'O'), ('potential_evapotranspiration', 'O'), ('streamflow', 'O')])
    scipy.io.savemat(path, {'data_MARRMoT_examples': rec})


def process(gauge_id):
    df = pd.read_csv(E.FORCING_DIR / f"varssim_{gauge_id}.csv", parse_dates=['Date']).set_index('Date')
    full = df.loc[E.DATA_START:E.DATA_END]
    if full[E.DYN_COLS].isna().any().any():
        return gauge_id, "NaN in forcing"
    d = full.index
    save_mat(OUT["og"] / f"varssim_{gauge_id}.mat", gauge_id, d,
             full['Precip'], full['Temp'], full['PET'], full['Obs flow'])
    save_mat(OUT["no_precip"] / f"varssim_{gauge_id}_no_precip.mat", gauge_id, d,
             full['syn_no_precip'], full['Temp'], full['PET'], full['Obs flow'])
    fl = df.loc[E.DATA_START:E.FLOOD_WINDOW[1]]
    for scen in E.FLOOD_SCENARIOS + [E.NOSTORM]:
        save_mat(OUT["up"] / f"varssim_{gauge_id}_up_{scen}.mat", gauge_id, fl.index,
                 fl[E.SCENARIO_PRECIP_COL[scen]], fl['Temp_at_peak'], fl['PET_at_peak'],
                 fl['Obs_flow_at_peak'])
    return gauge_id, "ok"


if __name__ == "__main__":
    for p in OUT.values():
        os.makedirs(p, exist_ok=True)
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    pool = load_basin_pool()
    res = Parallel(n_jobs=n_jobs)(delayed(process)(g) for g in pool)
    bad = [r for r in res if r[1] != "ok"]
    print(f"wrote .mat files for {len(res) - len(bad)}/{len(pool)} basins; problems: {bad}")
