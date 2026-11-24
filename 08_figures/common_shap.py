"""SHAP helpers for fig02/s02: reads the saved attribution tables (07_analysis/shap_attribution.py ->
out/analysis/shap) and draws the shared SHAP panels."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 08_figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from config import experiment as E  # noqa: E402
from config import plot_style as vs  # noqa: E402

A = E.ANALYSIS_SHAP_DIR
MODELS = ["RF", "XGB", "TCN", "LSTM"]   # display order
MAIN = "LSTM"                            # red, drawn thicker and on top
DYN_FAMILIES = ["recent_rain", "antecedent_rain", "temperature", "pet", "snow", "seasonality"]
STAT_FAMILIES = ["climate", "topography", "vegetation", "soil"]
RAW_VARS = ["Precip", "Temp", "PET"]
ctx = vs.apply_style()
vs.set_models(["LSTM", "TCN", "XGB", "RF"])   # colour slots, same as common_base.py
_CACHE = {}


def load_long(model):
    if model in _CACHE:
        return _CACHE[model]
    f = A / f"shap_values_{model}.csv"
    df = None
    if f.exists():
        df = pd.read_csv(f, dtype={"basin": str})
        df = df[~df["feature"].astype(str).str.startswith("__")]
    _CACHE[model] = df
    return df


PD_MARK = "*"
PD_NOTE = "* RF: path-dependent TreeSHAP (no between-basin baseline)"


def static_baseline(df):
    """Baseline for static attributions. RF has no between-basin baseline (interventional TreeSHAP
    segfaults on its deep trees), but its tree_path_dependent values -- stored under 'within' --
    do attribute the statics, so those are used and the RF label is marked with PD_MARK."""
    return ("between", False) if "between" in df.baseline.unique() else ("within", True)


EXCLUDE_SEASONALITY = False   # paper figures: drop doy (trees only) so shares are over P/T/PET + statics
SHOW_PD_NOTE = True   # paper figures switch this off (caption explains RF's SHAP mode instead)


def label(m, df):
    return m + PD_MARK if static_baseline(df)[1] else m


def note(ax, y=-0.13, wrap=False):
    if not SHOW_PD_NOTE:
        return
    txt = PD_NOTE.replace(" (", "\n(") if wrap else PD_NOTE
    ax.text(0, y, txt, transform=ax.transAxes, ha="left", va="top", fontsize=ctx["annot"],
            color="0.35")


def family_share(df, baseline="within", families=None):
    """Per-sample share of each family in the summed |attribution|. With `families`, the shares are
    taken among those families only, so models whose other families are absent (networks have no
    snow/seasonality; only RF's path-dependent 'within' values carry statics) stay comparable."""
    d = df[df.baseline == baseline].copy()
    if families is not None:
        d = d[d["family"].isin(families)]
    d["absval"] = d["value"].abs()
    fam = d.groupby(["basin", "date", "family"], as_index=False)["absval"].sum()
    tot = d.groupby(["basin", "date"], as_index=False)["absval"].sum().rename(columns={"absval": "tot"})
    out = fam.merge(tot, on=["basin", "date"])
    out["share"] = out["absval"] / out["tot"]
    return out[["basin", "date", "family", "share"]]


def _variable_shares_for_row(fam, feat):
    """Map one SHAP row onto the raw variable(s) it is actually built from (Precip/Temp/PET) --
    the one input representation every model (tree or network) shares, so RF/XGB/TCN/LSTM can be
    compared with no family-definition mismatch and no n/a cells.
    Convention: snow proxies split by construction, not lumped as one family --
    Psnow_sum is rainfall filtered by T<0 (features/tree_features.py:74) -> counts as Precip;
    melt_sum is a pure degree-day function of T (:77, no P term at all) -> counts as Temp. WB_sum
    (antecedent_rain family) is literally P - PET (:71-73) -- an equal mix, not attributable to one
    side, so its attribution is split 50/50 Precip/PET. doy_sin/doy_cos (seasonality) has no raw-
    variable equivalent at all (not derived from P, T or PET) and is excluded here; it is reported,
    trees-only, in the secondary full-family breakdown (panel_tree_family_heatmap) instead."""
    feat = str(feat)
    if fam == "temperature":
        return {"Temp": 1.0}
    if fam == "pet":
        return {"PET": 1.0}
    if fam == "snow":
        return {"Temp": 1.0} if feat.startswith("melt_sum") else {"Precip": 1.0}
    if fam == "antecedent_rain" and feat.startswith("WB_sum"):
        return {"Precip": 0.5, "PET": 0.5}
    if fam in ("recent_rain", "antecedent_rain"):
        return {"Precip": 1.0}
    return {}  # seasonality, static, or unrecognised -- no raw-variable mapping


def raw_variable_share(df, baseline="within"):
    """Every dynamic SHAP row collapsed onto {Precip, Temp, PET} via _variable_shares_for_row."""
    d = df[df.baseline == baseline].copy()
    d = d[d["family"].isin(["recent_rain", "antecedent_rain", "temperature", "pet", "snow"])]
    if not len(d):
        return pd.DataFrame(columns=["basin", "date", "variable", "share"])
    exploded = []
    for (fam, feat), grp in d.groupby(["family", "feature"], sort=False):
        for var, w in _variable_shares_for_row(fam, feat).items():
            exploded.append(grp[["basin", "date", "value"]].assign(variable=var, absval=(grp["value"] * w).abs()))
    e = pd.concat(exploded, ignore_index=True)
    fam_sum = e.groupby(["basin", "date", "variable"], as_index=False)["absval"].sum()
    tot = fam_sum.groupby(["basin", "date"], as_index=False)["absval"].sum().rename(columns={"absval": "tot"})
    out = fam_sum.merge(tot, on=["basin", "date"])
    out["share"] = out["absval"] / out["tot"]
    return out[["basin", "date", "variable", "share"]]


# ----------------------------------------------------------------------------------------------
# overview panels
# ----------------------------------------------------------------------------------------------
def panel_raw_variable_boxes(ax, models):
    """Per-sample share of Precip / Temp / PET in the dynamic |attribution| (within-basin baseline),
    on the raw-variable basis every model shares (see raw_variable_share). Replaces the former
    family bar chart, whose denominators differed by model (RF incl. statics, XGB incl. snow and
    seasonality, networks neither)."""
    rows = []
    n = len(models)
    w = 0.8 / n
    for i, m in enumerate(models):
        s = raw_variable_share(load_long(m), "within")
        data = [100 * s[s.variable == v]["share"].to_numpy() for v in RAW_VARS]
        pos = np.arange(len(RAW_VARS)) - 0.4 + (i + 0.5) * w
        c = vs.model_color(m)
        bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=ctx["lw"]),
                        whiskerprops=dict(lw=ctx["lw_axes"], color=c), capprops=dict(lw=ctx["lw_axes"], color=c))
        for b in bp["boxes"]:
            b.set(facecolor=(*to_rgb(c), 0.35), edgecolor=c, lw=ctx["lw"])
        for v, d in zip(RAW_VARS, data):
            rows.append({"model": m, "variable": v, "median_share_pct": np.median(d),
                         "q25": np.percentile(d, 25), "q75": np.percentile(d, 75)})
    ax.set_xticks(range(len(RAW_VARS)), RAW_VARS)
    ax.set(ylabel="Share of dynamic\n|attribution| [%]", ylim=(0, 118))
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[matplotlib.patches.Patch(facecolor=(*to_rgb(vs.model_color(m)), 0.35),
                                                edgecolor=vs.model_color(m), label=m) for m in models],
              loc="upper right", frameon=False, ncol=2, handlelength=1.0, columnspacing=0.8, borderaxespad=0.1)
    return pd.DataFrame(rows)


def panel_static_vs_dynamic(ax, models):
    """Stacked bars: static share (filled) + dynamic share (light) = 100 %. Between-basin baseline;
    RF from its path-dependent values (marked, see static_baseline)."""
    rows, labels, any_pd = [], [], False
    for i, m in enumerate(models):
        df = load_long(m)
        b, pd_flag = static_baseline(df)
        any_pd |= pd_flag
        labels.append(label(m, df))
        between = df[df.baseline == b]
        if EXCLUDE_SEASONALITY:
            between = between[between["family"] != "seasonality"]
        between = between.assign(absval=between["value"].abs(),
                                 static=~between["family"].isin(DYN_FAMILIES))
        tot = between.groupby(["basin", "date"])["absval"].sum()
        stat = between[between.static].groupby(["basin", "date"])["absval"].sum().reindex(tot.index, fill_value=0)
        frac = 100 * float((stat / tot).median())
        c = vs.model_color(m)
        dyn = 100 - frac   # dynamic (solid, with value) at the bottom, static (light) on top
        ax.bar(i, dyn, 0.6, color=c, edgecolor=c, lw=ctx["lw_axes"])
        ax.bar(i, frac, 0.6, bottom=dyn, color=(*to_rgb(c), 0.2), edgecolor=c, lw=ctx["lw_axes"])
        ax.text(i, dyn / 2, f"{dyn:.0f}", ha="center", va="center", color="white", fontsize=ctx["annot"],
                fontweight="bold")
        rows.append({"model": m, "static_share_pct": frac,
                     "baseline": "tree_path_dependent" if pd_flag else b})
    ax.set_xticks(range(len(models)), labels)
    if any_pd:
        note(ax, wrap=True)
    ax.set(ylabel="Share of |attribution| [%]", ylim=(0, 118))
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[matplotlib.patches.Patch(facecolor="0.4", alpha=0.2, edgecolor="0.4", label="static"),
                       matplotlib.patches.Patch(facecolor="0.4", label="dynamic")],
              loc="upper right", frameon=False, ncol=2, handlelength=1.0, borderaxespad=0.1)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# heatmap panels
# ----------------------------------------------------------------------------------------------
def _draw_heatmap(ax, mat, rownames, colnames, divider_after=None, fmt="{:.0f}", cbar_label=None):
    vmax = np.nanmax(mat) if np.isfinite(mat).any() else 1
    # Grey-hatched, not blank-white: every missing cell is "deliberately not applicable".
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isfinite(mat[i, j]):
                ax.add_patch(matplotlib.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="0.9",
                                                           edgecolor="white", hatch="///", lw=0.3))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                color = "black" if mat[i, j] > vmax * 0.55 else "white"
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", color=color,
                        fontsize=ctx["annot"])
            else:
                ax.text(j, i, "n/a", ha="center", va="center", color="0.45", fontsize=ctx["annot"],
                        style="italic")
    ax.set_xticks(range(len(colnames)), colnames)
    ax.set_yticks(range(len(rownames)), [r.replace("_", " ") for r in rownames])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if divider_after is not None:
        ax.axhline(divider_after - 0.5, color="black", lw=ctx["lw"])
    cb = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cb.set_label(cbar_label or "Median |attribution| share [%]")
    cb.outline.set_linewidth(ctx["lw_axes"])
    return im


def panel_tree_family_heatmap(ax, tree_models):
    """Trees only: full engineered-family breakdown (snow, seasonality have no network equivalent).
    Dynamic families: share among the dynamic families only (within-basin baseline), so RF -- whose
    path-dependent 'within' values also carry statics -- is not deflated relative to XGB.
    Statics: share of all |attribution| (between-basin baseline; RF path-dependent)."""
    rows = []
    mat = np.full((len(DYN_FAMILIES) + len(STAT_FAMILIES), len(tree_models)), np.nan)
    for j, m in enumerate(tree_models):
        df = load_long(m)
        s = family_share(df, "within", families=DYN_FAMILIES)
        for i, fam in enumerate(DYN_FAMILIES):
            v = s[s.family == fam]["share"]
            if len(v):
                mat[i, j] = 100 * v.median()
                rows.append({"model": m, "family": fam, "baseline": "within", "median_share_pct": mat[i, j]})
        b, pd_flag = static_baseline(df)
        fs = family_share(df, b)
        for i, fam in enumerate(STAT_FAMILIES):
            v = fs[fs.family == fam]["share"]
            if len(v):
                mat[len(DYN_FAMILIES) + i, j] = 100 * v.median()
                rows.append({"model": m, "family": fam,
                             "baseline": "tree_path_dependent" if pd_flag else b,
                             "median_share_pct": mat[len(DYN_FAMILIES) + i, j]})
    labels = [label(m, load_long(m)) for m in tree_models]
    _draw_heatmap(ax, mat, DYN_FAMILIES + STAT_FAMILIES, labels, divider_after=len(DYN_FAMILIES))
    if labels != tree_models:
        note(ax, y=-0.06, wrap=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# top-N individual inputs
# ----------------------------------------------------------------------------------------------
def feature_share_common(df, baseline):
    """Per-sample |attribution| share of each *input* on a basis every model shares: dynamic rows
    collapsed onto Precip/Temp/PET (same convention as raw_variable_share; seasonality -> 'doy'),
    static attributes kept by name. Returns mean share [%] per input over the sampled days."""
    d = df[df.baseline == baseline]
    parts = []
    for (fam, feat), grp in d.groupby(["family", "feature"], sort=False):
        if fam in STAT_FAMILIES:
            mapping = {str(feat): 1.0}
        elif fam == "seasonality":
            if EXCLUDE_SEASONALITY:
                continue
            mapping = {"doy": 1.0}
        else:
            mapping = _variable_shares_for_row(fam, feat)
        for name, w in mapping.items():
            parts.append(grp[["basin", "date"]].assign(input=name, absval=(grp["value"] * w).abs()))
    e = pd.concat(parts, ignore_index=True).groupby(["basin", "date", "input"], as_index=False)["absval"].sum()
    e["share"] = e["absval"] / e.groupby(["basin", "date"])["absval"].transform("sum")
    return 100 * e.groupby("input")["share"].mean()
