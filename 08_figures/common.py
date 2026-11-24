"""Shared settings and helpers of the paper figure scripts in 08_figures (fig01_s01_accuracy.py ...
fig06_s04_ranking.py). Every number comes from the saved analysis outputs (out/analysis); figures are
written to fig/ (PNG) and fig/pdf (PDF), their data tables to fig/ as CSV."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 08_figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import common_shap as S  # noqa: E402
import common_base as T  # noqa: E402
from config import experiment as E  # noqa: E402
from hsp import runio  # noqa: E402
from hsp.registry import combos, runs_of  # noqa: E402
from config import plot_style as vs  # noqa: E402

OUT = E.FIG_DIR
S.EXCLUDE_SEASONALITY = True   # shares over Precip/Temp/PET + statics only (doy is trees-only)
S.SHOW_PD_NOTE = False   # keep the RF asterisk, drop the footnote text (explained in the caption)
ctx = T.ctx
col, split, short, ordered, ref_keys, ecdf = T.col, T.split, T.short, T.ordered, T.ref_keys, T.ecdf
HBV_LS, STRAT_LS, STRAT_MK, MODEL_ORDER = T.HBV_LS, T.STRAT_LS, T.STRAT_MK, T.MODEL_ORDER
GRID = np.linspace(0, 1, 400)
STRATS = ["individual", "regional", "adapted"]

# box fill by strategy (same light -> dark convention as the tercile bars): ind light, ada mid, reg dark
BOX_ALPHA = {"individual": 0.12, "adapted": 0.32, "regional": 0.6}
HYDRO_YMAX = {"high_BFI": 0.75}   # fig04 hydrograph y-limits [mm/d] per example (None = auto)
HBV_LW = 1.8   # HBV (dotted black) line-width multiplier in hydrograph panels, for visibility


def save(fig, stem):
    """PNG directly in fig/, PDF (editable fonts) in fig/pdf/."""
    vs.savefig(fig, stem, OUT / "pdf", formats=("pdf",))
    vs.savefig(fig, stem, OUT, formats=vs.PNG)
    plt.close(fig)


def label(axes, titles, start=0):
    """Bold panel letter + short title."""
    for i, (a, t) in enumerate(zip(axes, titles)):
        vs.panel_label(a, start + i, title=t)


def strategy_boxes(ax, data, keys):
    """Boxes coloured by model, fill shaded by strategy (HBV: light grey)."""
    bp = T.boxes(ax, data, keys, positions=np.arange(len(keys)))
    for b, k in zip(bp["boxes"], keys):
        m, st = split(k)
        b.set_facecolor((*to_rgb(col(m)), 0.15 if m == "HBV" else BOX_ALPHA[st]))
    T.combo_xticks(ax, keys)
    return bp


def strategy_box_legend(ax, **kw):
    ax.legend(handles=[Patch(facecolor=(0.3, 0.3, 0.3, BOX_ALPHA[s]), edgecolor="0.3", label=s) for s in STRATS],
              frameon=False, ncol=3, handlelength=1.2, **kw)


def accuracy():
    return pd.read_csv(E.ANALYSIS_ACCURACY_DIR / "accuracy_per_basin.csv", dtype={"basin": str})


def boot_median_ci(x, n=2000, seed=0, q=(2.5, 97.5)):
    x = np.asarray(x, float)[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, (n, len(x)), replace=True), axis=1)
    return np.median(x), *np.percentile(meds, q)


def plausibility_tables():
    t = pd.read_csv(E.ANALYSIS_SCORING_DIR / "composite_table.csv")
    t = t.set_index("combo").loc[ordered(t.combo)].reset_index()
    main = ordered(t[t.in_main_ranking].combo.tolist())
    raw = t.set_index("combo").loc[:, ["mass_raw", "drought_raw", "flood_raw"]]
    raw.columns = list(T.DIM_DIR)
    return t, main, raw


def accuracy_composite_panel(ax, t):
    """Median test KGE vs composite plausibility, 90 % intervals on both axes. Returns the KGE interval
    ends (for label placement, which must run after the layout is final)."""
    # x bar: 90 % bootstrap CI of the median test KGE over basins (5-95 %, same as the y bars)
    pb = accuracy()
    pb = pb[pb.period == "test"]
    kci = np.array([boot_median_ci(pb.loc[pb.combo == k, "KGE_ens"], q=(5, 95))[1:] for k in t.combo])
    xlo, xhi = kci[:, 0], kci[:, 1]
    for i, r in t.iterrows():
        m, st = split(r.combo)
        c = col(m)
        ax.errorbar(r.test_KGE_median_ens, r.composite,
                    xerr=[[r.test_KGE_median_ens - xlo[i]], [xhi[i] - r.test_KGE_median_ens]],
                    yerr=[[r.composite - r.composite_p05], [r.composite_p95 - r.composite]],
                    fmt=STRAT_MK[st], color=c, mfc=T.strat_fill(m, st), ms=ctx["ms"] * 2, mec=c, mew=1.0,
                    lw=ctx["lw"] * 1.4, capsize=3, capthick=ctx["lw"] * 1.4, zorder=3)
    ax.set(xlabel="KGE (evaluation)", ylabel="Composite plausibility")
    xr = xhi.max() - xlo.min()
    ax.set_xlim(xlo.min() - 0.05 * xr, xhi.max() + 0.25 * xr)
    sh = [Line2D([], [], color="0.3", marker=STRAT_MK[s], lw=0, ms=ctx["ms"] * 1.6, mec="0.3", mew=0.9,
                 mfc={"individual": "white", "adapted": "0.7", "regional": "0.3"}[s]) for s in STRATS]
    ax.legend(sh, STRATS, loc="upper right", frameon=False, title="strategy", alignment="left")
    return xlo, xhi


# ------------------------------------------------------------------------------------------
# Paper tables: csvNN aligned with figNN (scsvNN with sNN), written to fig/, 2 decimals.
# Rows grouped by model (HBV first, then RF, XGB, TCN, LSTM), strategies in order ind / reg / ada.
# ------------------------------------------------------------------------------------------
def f2(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.2f}"


def save_table(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)


def combo_rows():
    return [(split(k)[0], split(k)[1], k) for k in ordered(combos())]
