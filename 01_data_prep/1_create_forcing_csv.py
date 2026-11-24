"""Per-basin forcing CSVs for every model (ML reads them directly; HBV .mat files are built from
them by 01_data_prep/5_create_mat_files.py, so both pipelines get identical forcing).

Columns
  Date, Obs flow (mm/d), Precip, Temp, PET                     real record
  syn_no_precip                                                drought: P = 0 over DROUGHT
  syn_peak{D}_rate{U}                                          flood: P = 0 over FLOOD_WINDOW except
                                                               the D storm days from STORM_START,
                                                               each = source peak * (1 + U/100)
  syn_nostorm                                                  flood without the storm (P = 0)
  Temp_at_peak, PET_at_peak, Obs_flow_at_peak, Date_at_peak    transplanted from the source event:
                                                               source peak - 10 d .. peak + 365 d,
                                                               placed on FLOOD_WINDOW
Outside the scenario windows every syn_* / *_at_peak column equals the real value (NaN after the
window ends, so a scenario run can never silently read past it).

In:  data/camels_us/basin_dataset_public_v1p2/{basin_mean_forcing/daymet,usgs_streamflow}/<huc>/...
Out: data/forcing/varssim_US/varssim_<gauge>.csv

RUN: python 01_data_prep/1_create_forcing_csv.py [n_jobs]
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402

forcing_dir = E.CAMELS_ROOT / "basin_mean_forcing" / "daymet"
streamflow_dir = E.CAMELS_ROOT / "usgs_streamflow"
output_dir = E.FORCING_DIR

# --- Priestley-Taylor PET constants ---
ALPHA_PT = 1.26          # Priestley-Taylor coefficient
ALBEDO = 0.23            # reference surface albedo
GSC = 0.0820             # solar constant, MJ m-2 min-1
SIGMA = 4.903e-9         # Stefan-Boltzmann constant, MJ K-4 m-2 day-1
CFS_TO_M3S = 0.0283168466


def read_forcing(path):
    """Read a CAMELS-US daymet basin-mean forcing file.

    First 3 lines: gauge latitude (deg), gauge elevation (m), basin area (m^2).
    Line 4: column header. Remaining lines: daily data.
    """
    with open(path) as f:
        header_lines = [f.readline().strip() for _ in range(3)]

    try:
        lat, elev, area_m2 = (float(x) for x in header_lines)
    except ValueError:
        raise ValueError("missing 3-line lat/elev/area header (malformed forcing file)")

    df = pd.read_csv(path, skiprows=3, sep=r'\s+')
    df['Date'] = pd.to_datetime(dict(year=df['Year'], month=df['Mnth'], day=df['Day']))
    return df, lat, elev, area_m2


def read_streamflow(path):
    df = pd.read_csv(
        path, header=None, sep=r'\s+',
        names=['gauge_id', 'Year', 'Mnth', 'Day', 'flow_cfs', 'qc_flag']
    )
    df['Date'] = pd.to_datetime(dict(year=df['Year'], month=df['Mnth'], day=df['Day']))
    df.loc[df['flow_cfs'] <= -999.0, 'flow_cfs'] = np.nan
    return df[['Date', 'flow_cfs']]


def compute_pet(df, lat_deg, elev_m):
    """Priestley-Taylor PET (mm/day), PET = alpha * (Delta / (Delta + gamma)) * (Rn / lambda),
    with net radiation Rn estimated (FAO-56) from srad, tmax, tmin and vp."""
    tmax = df['tmax(C)'].to_numpy()
    tmin = df['tmin(C)'].to_numpy()
    srad = df['srad(W/m2)'].to_numpy()
    dayl = df['dayl(s)'].to_numpy()
    vp_pa = df['vp(Pa)'].to_numpy()

    tmean = (tmax + tmin) / 2.0
    ea = vp_pa / 1000.0  # actual vapor pressure, kPa

    # slope of saturation vapor pressure curve, kPa/C
    delta = 4098.0 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2

    # psychrometric constant, kPa/C
    press = 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26
    gamma = 0.665e-3 * press

    # incoming shortwave radiation, MJ m-2 day-1
    rs = srad * dayl / 1.0e6

    # extraterrestrial radiation
    phi = np.radians(lat_deg)
    doy = df['Date'].dt.dayofyear.to_numpy()
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    sol_decl = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(sol_decl), -1.0, 1.0))
    ra = (24 * 60 / np.pi) * GSC * dr * (
        ws * np.sin(phi) * np.sin(sol_decl) + np.cos(phi) * np.cos(sol_decl) * np.sin(ws)
    )

    rso = (0.75 + 2e-5 * elev_m) * ra
    rso = np.where(rso <= 0, np.nan, rso)

    rns = (1 - ALBEDO) * rs
    tmax_k = tmax + 273.16
    tmin_k = tmin + 273.16
    rs_rso = np.clip(rs / rso, 0.3, 1.0)
    rnl = SIGMA * ((tmax_k ** 4 + tmin_k ** 4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.clip(ea, 0, None))) * (1.35 * rs_rso - 0.35)
    rn = rns - rnl

    lam = 2.501 - 0.002361 * tmean  # latent heat of vaporization, MJ/kg

    pet = ALPHA_PT * (delta / (delta + gamma)) * (rn / lam)
    return np.clip(pet, 0, None)


def add_scenarios(df):
    """Append the drought and flood scenario columns (design in the module docstring)."""
    df = df.set_index('Date')
    dates = df.index

    # 1. drought
    drought = (dates >= E.DROUGHT[0]) & (dates <= E.DROUGHT[1])
    df['syn_no_precip'] = df['Precip'].where(~drought, 0.0)

    # 2. flood: source event = largest daily P in the test period
    search = df.loc[E.SOURCE_SEARCH[0]:E.SOURCE_SEARCH[1], 'Precip']
    peak_date = search.idxmax()
    peak_value = float(search.max())

    window = pd.date_range(*E.FLOOD_WINDOW)
    src_dates = window - window[0] + (peak_date - pd.Timedelta(days=E.SOURCE_DAYS_BEFORE_PEAK))
    if src_dates[-1] > dates[-1]:
        raise ValueError(f"source window ends {src_dates[-1].date()} after the forcing record")
    in_window = dates.isin(window)
    after_window = dates > window[-1]

    for col, src in [('Temp_at_peak', 'Temp'), ('PET_at_peak', 'PET'), ('Obs_flow_at_peak', 'Obs flow')]:
        df[col] = df[src]
        df.loc[window, col] = df[src].reindex(src_dates).to_numpy()
        df.loc[after_window, col] = np.nan
    df['Date_at_peak'] = pd.Series(dates, index=dates)
    df.loc[window, 'Date_at_peak'] = src_dates
    df.loc[after_window, 'Date_at_peak'] = pd.NaT

    base = df['Precip'].copy()
    base[in_window] = 0.0
    base[after_window] = np.nan
    df['syn_nostorm'] = base
    for dur in E.STORM_DURATIONS:
        storm_days = pd.date_range(E.STORM_START, periods=dur)
        for up in E.STORM_UPLIFTS:
            col = base.copy()
            col[storm_days] = peak_value * (1 + up / 100.0)
            df[f'syn_peak{dur}_rate{up}'] = col

    df = df.reset_index()
    return df, peak_date, peak_value


def process(forcing_path):
    fname = os.path.basename(forcing_path)
    gauge_id = fname.split('_')[0]
    huc = os.path.basename(os.path.dirname(forcing_path))
    try:
        fdf, lat, elev, area_m2 = read_forcing(forcing_path)
        fdf = fdf[(fdf['Date'] >= E.FORCING_RECORD[0]) & (fdf['Date'] <= E.FORCING_RECORD[1])]
        fdf = fdf.reset_index(drop=True)

        df = pd.DataFrame({
            'Date': fdf['Date'],
            'Precip': fdf['prcp(mm/day)'],
            'Temp': (fdf['tmax(C)'] + fdf['tmin(C)']) / 2.0,
            'PET': compute_pet(fdf, lat, elev),
        })

        streamflow_path = streamflow_dir / huc / f"{gauge_id}_streamflow_qc.txt"
        if not streamflow_path.exists():
            return gauge_id, "no streamflow file"
        sdf = read_streamflow(streamflow_path)
        sdf['Obs flow'] = sdf['flow_cfs'] * CFS_TO_M3S * 86400.0 / area_m2 * 1000.0
        df = df.merge(sdf[['Date', 'Obs flow']], on='Date', how='left')
        df = df[['Date', 'Obs flow', 'Precip', 'Temp', 'PET']]

        df, peak_date, peak_value = add_scenarios(df)
        df.to_csv(output_dir / f"varssim_{gauge_id}.csv", index=False)
        return gauge_id, f"ok peak {peak_value:.2f} on {peak_date.date()}"
    except Exception as e:  # noqa: BLE001 -- report and continue with the other basins
        return gauge_id, f"ERROR {e}"


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    forcing_files = sorted(glob.glob(str(forcing_dir / '*' / '*_lump_cida_forcing_leap.txt')))
    print(f"Found {len(forcing_files)} basins -> {output_dir}")
    for gid, msg in Parallel(n_jobs=n_jobs)(delayed(process)(f) for f in forcing_files):
        print(gid, msg)
    print("Processing complete.")
