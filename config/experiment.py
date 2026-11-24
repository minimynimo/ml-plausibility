"""Shared experiment configuration for the CAMELS-US plausibility study.

Every training, evaluation, scenario and figure script imports its dates, basin sets, seeds,
scenario design and hyperparameters from here. Nothing below should be duplicated as a literal
in any other script.

Scope: CAMELS-US only. Models: RF, XGB, TCN, LSTM (ML) and HBV (per-basin calibrated).
"""
import os as _os
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------------------------
# Paths. Everything is relative to the repository root; override with environment variables:
#   HSP_DATA_ROOT  input data (default <repo>/data, layout in README.md)
#   HSP_OUT_ROOT   runs, tuning, analysis tables, logs (default <repo>/out)
#   HSP_FIG_ROOT   figures (default <repo>/fig)
# --------------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT                                       # alias used by core/runio.py
DATA_ROOT = Path(_os.environ.get("HSP_DATA_ROOT", REPO_ROOT / "data"))
CAMELS_ATTR_DIR = DATA_ROOT / "camels_us"                       # camels_{clim,topo,vege,soil,geol,hydro}.txt
CAMELS_ROOT = CAMELS_ATTR_DIR / "basin_dataset_public_v1p2"     # CAMELS-US v1.2 time series
FORCING_DIR = DATA_ROOT / "forcing" / "varssim_US"              # per-basin forcing CSVs (01_data_prep/1)
HBV_MAT_ROOT = DATA_ROOT / "hbv_mat"                            # per-basin .mat forcing for HBV (01_data_prep/5)
BASIN_LIST_DIR = DATA_ROOT / "basin_lists"                      # 01_data_prep/3
ATTRIBUTE_CSV = DATA_ROOT / "attributes" / "camels_original_attribute_combined.csv"   # 01_data_prep/2
POOL_CSV = BASIN_LIST_DIR / "us_camels_file_num.csv"            # complete-data basins (01_data_prep/3)
# Basins dropped from the training pool by hand (01_data_prep/3_basin_filter.py), with the reason.
EXCLUDED_BASINS = {
    "10173450": "no row in the static-attribute table used when the study pool was fixed; kept out so "
                "the pool stays the 637 basins of the study",
}
SMOKE = int(_os.environ.get("HSP_SMOKE", 0))   # >0: end-to-end test mode with that many basins
BASIN_DIR = Path(_os.environ.get("HSP_BASIN_DIR", REPO_ROOT / "basins"))   # basin_sets.csv, hbv_failures.csv
# OUT_ROOT holds runs/, tuning/, analysis/<type>/, _markers, _logs, _driver and _code_snapshot.
OUT_ROOT = Path(_os.environ.get("HSP_OUT_ROOT", REPO_ROOT / "out"))
if SMOKE:
    BASIN_DIR = OUT_ROOT / "basins"
BASIN_SETS_CSV = BASIN_DIR / "basin_sets.csv"
HBV_FAILURES_CSV = BASIN_DIR / "hbv_failures.csv"
ANALYSIS_DIR = OUT_ROOT / "analysis"                 # parent; most writers use the subdirs below
ANALYSIS_ACCURACY_DIR = ANALYSIS_DIR / "accuracy"
ANALYSIS_MASS_DIR = ANALYSIS_DIR / "mass_balance"
ANALYSIS_DROUGHT_DIR = ANALYSIS_DIR / "drought"
ANALYSIS_FLOOD_DIR = ANALYSIS_DIR / "flood"
ANALYSIS_SCORING_DIR = ANALYSIS_DIR / "scoring"
ANALYSIS_SHAP_DIR = ANALYSIS_DIR / "shap"
FIG_DIR = Path(_os.environ.get("HSP_FIG_ROOT", REPO_ROOT / "fig"))
LOG_DIR = OUT_ROOT / "_logs"
MARK_DIR = OUT_ROOT / "_markers"
DRIVER_DIR = OUT_ROOT / "_driver"
CODE_SNAPSHOT_DIR = OUT_ROOT / "_code_snapshot"

# --------------------------------------------------------------------------------------------
# Periods. All ranges are inclusive.
# --------------------------------------------------------------------------------------------
def _d(s):
    return pd.Timestamp(s)

DATA_START = _d("1987-01-01")      # first input day any model may see (HBV .mat start, ML inputs)
WARMUP = (_d("1987-01-01"), _d("1987-12-31"))
TRAIN = (_d("1988-01-01"), _d("1997-12-31"))   # fitted / calibrated and scored ("cal")
VAL = (_d("1998-01-01"), _d("2000-12-31"))     # early stopping + HP selection only, never fitted
TEST = (_d("2001-01-01"), _d("2006-12-31"))
DATA_END = TEST[1]
PERIODS = {"cal": TRAIN, "val": VAL, "test": TEST}

# Forcing record kept in the per-basin CSVs (Daymet 1980-2014). Needed beyond DATA_END because the
# flood scenario transplants T/PET for up to 365 days after a source peak that may lie in 2006.
FORCING_RECORD = (_d("1980-01-01"), _d("2014-12-31"))

# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------
DYN_COLS = ["Precip", "Temp", "PET"]   # the same 3 forcings HBV receives
Q_COL = "Obs flow"                     # mm/d
STATIC_ATTRS = [
    "p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow", "high_prec_freq",
    "high_prec_dur", "low_prec_freq", "low_prec_dur", "elev_mean", "slope_mean", "area_gages2",
    "frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff", "soil_depth_pelletier",
    "soil_depth_statsgo", "soil_porosity", "soil_conductivity", "max_water_content",
    "sand_frac", "silt_frac", "clay_frac", "carbonate_rocks_frac", "geol_permeability",
]

# --------------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]          # regional + adapted networks, regional RF/XGB
SEEDS_INDIVIDUAL_TREE = [42, 43, 44, 45, 46]   # individual RF/XGB
SEEDS_INDIVIDUAL_NN = [42, 43, 44, 45, 46]     # individual TCN/LSTM
HBV_OPTIMISER_SEED = 42               # CMA-ES rng seed, one calibration per basin
if SMOKE:
    SEEDS, SEEDS_INDIVIDUAL_TREE, SEEDS_INDIVIDUAL_NN = [1, 2], [1], [1]

# --------------------------------------------------------------------------------------------
# Loss: basin-averaged NSE* (Kratzert et al. 2019), identical for LSTM and TCN in every
# strategy. Squared error divided by (q_std_basin + eps)^2, q_std from the basin's training period.
# --------------------------------------------------------------------------------------------
LOSS_NAME = "NSE*"
NSE_LOSS_EPS = 0.1

# --------------------------------------------------------------------------------------------
# Synthetic scenarios
# --------------------------------------------------------------------------------------------
DROUGHT = (TEST[0], TEST[1])          # P = 0, real T/PET
STORM_START = _d("2001-01-11")        # index 10 of the window starting at TEST[0]
STORM_DURATIONS = [1, 3, 5]
STORM_UPLIFTS = [0, 10, 30, 50]       # percent above the source-event peak
FLOOD_WINDOW_DAYS_AFTER_STORM = 365
FLOOD_WINDOW = (TEST[0], STORM_START + pd.Timedelta(days=FLOOD_WINDOW_DAYS_AFTER_STORM))
SOURCE_SEARCH = TEST                  # source event = largest daily P in this period
SOURCE_DAYS_BEFORE_PEAK = (STORM_START - TEST[0]).days   # 10: window starts 10 d before the peak
FLOOD_SCENARIOS = [f"peak{d}_rate{u}" for d in STORM_DURATIONS for u in STORM_UPLIFTS]
NOSTORM = "nostorm"
SCENARIOS = ["no_precip"] + FLOOD_SCENARIOS + [NOSTORM]
# forcing-CSV column holding the precipitation of each scenario
SCENARIO_PRECIP_COL = {"no_precip": "syn_no_precip", NOSTORM: "syn_nostorm",
                       **{s: f"syn_{s}" for s in FLOOD_SCENARIOS}}
# scenario output window
SCENARIO_WINDOW = {"no_precip": DROUGHT, NOSTORM: FLOOD_WINDOW,
                   **{s: FLOOD_WINDOW for s in FLOOD_SCENARIOS}}


def is_flood(scenario: str) -> bool:
    return scenario == NOSTORM or scenario.startswith("peak")


# --------------------------------------------------------------------------------------------
# Models, strategies, hyperparameters
# --------------------------------------------------------------------------------------------
MODELS = ["HBV", "RF", "XGB", "TCN", "LSTM"]
STRATEGIES = {
    "HBV": ["individual"],
    "RF": ["individual", "regional"],
    "XGB": ["individual", "regional"],
    "TCN": ["individual", "regional", "adapted"],
    "LSTM": ["individual", "regional", "adapted"],
}

# XGB regional: early stopping on VAL, plain squared error on per-basin z-scored targets
XGB_EARLY_STOPPING_ROUNDS = 50
XGB_MAX_ROUNDS = 5000
XGB_GRID = {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [4, 6, 8],
            "subsample": [0.7, 0.9], "colsample_bytree": [0.7, 0.9]}
XGB_FIXED = {"min_child_weight": 3, "tree_method": "hist", "objective": "reg:squarederror"}

# RF regional
RF_GRID = {"min_samples_leaf": [5, 10, 20], "max_features": [0.3, 0.5, "sqrt"],
           "max_depth": [None, 30]}
RF_N_ESTIMATORS_TUNE = 300
RF_N_ESTIMATORS_FINAL = 500
RF_MAX_SAMPLES = 0.8
RF_DROP_RULE_KGE = 0.40   # regional RF median test KGE below this -> Supporting Information only

# TCN regional (statics via MLP embedding + FiLM in every residual block). seq_length=365 is the
# tuning/TCN_regional/selected.json outcome (filters 32/64/128, then window 180 vs 365).
TCN_REGIONAL = {"kernel_size": 5, "num_levels": 6, "dropout": 0.2, "learning_rate": 5e-4,
                "batch_size": 256, "seq_length": 365, "static_embedding": 32,
                "max_epochs": 100, "patience": 8, "num_filters": 64}
TCN_FILTER_GRID = [32, 64, 128]
TCN_SEQ_SEARCH_BASE = 180       # window of the filter search (stage 1)
TCN_SEQ_ALTERNATIVE = 365       # tested once against TCN_SEQ_SEARCH_BASE (stage 2)
# TCN individual: selected separately from regional (individual trains from scratch per basin, so
# it is tuned on its own -- only "adapted" inherits regional's architecture, since it starts from the
# regional model's weights). Search on HP_SEARCH_SUBSET_SIZE basins (see tuning/TCN_individual/).
TCN_INDIVIDUAL = {"num_filters": 64, "kernel_size": 5, "num_levels": 6, "dropout": 0.4,
                  "learning_rate": 5e-4, "batch_size": 32, "seq_length": 90,
                  "max_epochs": 100, "patience": 5}
TCN_INDIVIDUAL_FILTER_GRID = [16, 32, 64]
TCN_INDIVIDUAL_SEQ_GRID = [90, 180]

# LSTM regional: hidden_size, then dropout (see tuning/LSTM_regional/), the same two-stage search
# as TCN_FILTER_GRID / TCN_SEQ_ALTERNATIVE in 04_hp_selection/select_tcn.py.
LSTM_REGIONAL = {"hidden_size": 128, "dropout": 0.4, "learning_rate": 1e-3, "seq_length": 365,
                 "batch_size": 256, "initial_forget_bias": 5, "max_epochs": 30, "patience": 10}
LSTM_HIDDEN_GRID = [32, 64, 128]           # stage 1: capacity, at the current dropout
LSTM_DROPOUT_ALTERNATIVE = 0.2             # stage 2: regularization, at the winning hidden_size
LSTM_TUNE_MAX_EPOCHS = 100                 # generous ceiling for the search (production uses 30)
# LSTM individual: selected separately from regional (see the TCN individual note above and
# tuning/LSTM_individual/).
LSTM_INDIVIDUAL = {"hidden_size": 64, "dropout": 0.5, "learning_rate": 1e-3, "seq_length": 270,
                   "batch_size": 64, "max_epochs": 100, "patience": 10}
LSTM_INDIVIDUAL_HIDDEN_GRID = [16, 32, 64]
LSTM_INDIVIDUAL_DROPOUT_ALTERNATIVE = 0.3
HP_SEARCH_SUBSET_SIZE = 160   # basins for the individual-strategy HP search

# adapted fine-tuning
ADAPT = {"max_epochs": 20, "lr_factor": 0.1, "patience": 5}

# Individual tree models
RF_INDIVIDUAL = {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 20,
                 "max_features": "sqrt", "max_samples": 0.5}
XGB_INDIVIDUAL = {"learning_rate": 0.01, "max_depth": 2, "subsample": 0.6,
                  "colsample_bytree": 1.0, "min_child_weight": 10}

METRICS = ["KGE", "NSE", "r", "alpha", "beta"]


def run_name(model: str, strategy: str, seed: int, tag: str = "") -> str:
    return f"{model}_{strategy}{'_' + tag if tag else ''}_seed{seed}"


def run_dir(model: str, strategy: str, seed: int, tag: str = "") -> Path:
    return OUT_ROOT / "runs" / model / strategy / run_name(model, strategy, seed, tag)
