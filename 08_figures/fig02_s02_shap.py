"""fig02  SHAP: static vs dynamic, Precip/Temp/PET shares, top-12 inputs
s02    SHAP: tree engineered families, Precip share by flow state
Tables: csv02_shap_shares

Outputs: fig/<name>.png, fig/pdf/<name>.pdf, out/tables/<table>.csv
Usage: python 08_figures/fig02_s02_shap.py [fig02 s02 csv02]   (no args = all)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 08_figures
from common import *  # noqa: E402,F401,F403  shared settings, helpers and imports


# ------------------------------------------------------------------------------------------
# SHAP
# ------------------------------------------------------------------------------------------
def shap_models():
    return [m for m in S.MODELS if (S.A / f"shap_values_{m}.csv").exists()]


def panel_top_features(ax, models, n_static=4):
    shares = {}
    for m in models:
        df = S.load_long(m)
        shares[m] = S.feature_share_common(df, S.static_baseline(df)[0])
    raw_tab = pd.DataFrame(shares)
    tab = raw_tab.fillna(0.0)   # for ranking only; inputs a model does not have stay blank in the plot
    # dynamic inputs first, then statics; each block gets its own colour scale so the small static
    # shares are not washed out by the large dynamic ones. Statics = union of every model's top
    # `n_static` attributes (a mean-over-models ranking hides attributes only one model relies on).
    dyn = [r for r in RAW_VARS if r in tab.index]
    st_tab = tab.drop(index=dyn)
    pick = set().union(*[st_tab[m].nlargest(n_static).index for m in models])
    sta = [r for r in st_tab.mean(axis=1).sort_values(ascending=False).index if r in pick]
    rows = dyn + sta
    mat = raw_tab.loc[rows, models].to_numpy(dtype=float)
    labels = [S.label(m, S.load_long(m)) for m in models]
    from matplotlib.colors import Normalize
    blocks = [(slice(0, len(dyn)), "viridis", "Dynamic inputs [%]"),
              (slice(len(dyn), len(rows)), "YlOrBr", "Static attributes [%]")]
    ims = []
    for sl, cmap, _ in blocks:
        m_ = np.full_like(mat, np.nan)
        m_[sl] = mat[sl]
        nm = Normalize(0, np.nanmax(mat[sl]))
        ims.append(ax.imshow(np.ma.masked_invalid(m_), cmap=cmap, norm=nm, aspect="auto"))
        cm = plt.get_cmap(cmap)
        for i in range(sl.start, sl.stop):
            for j in range(mat.shape[1]):
                if not np.isfinite(mat[i, j]):
                    continue                      # input not used by this model: leave blank
                r, g, b, _ = cm(nm(mat[i, j]))
                dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=ctx["annot"],
                        color="white" if dark else "black")
    ax.axhline(len(dyn) - 0.5, color="white", lw=3)
    ax.axhline(len(dyn) - 0.5, color="black", lw=ctx["lw"])
    ax.set_xticks(range(len(models)), labels)
    ax.set_yticks(range(len(rows)), [r.replace("_", " ") for r in rows])
    ax.tick_params(length=0)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    # one colour bar per block, same x, each spanning its own block's rows
    n = len(rows)
    for im, (sl, _, lab) in zip(ims, blocks):
        y0 = 1 - sl.stop / n
        h = (sl.stop - sl.start) / n
        cax = ax.inset_axes([1.06, y0 + 0.01, 0.06, h - 0.02])
        cb = ax.figure.colorbar(im, cax=cax)
        cb.set_label(lab)
        cb.outline.set_linewidth(ctx["lw_axes"])
    if labels != models:
        S.note(ax, y=-0.05, wrap=True)


def fig02():
    """SHAP: static vs dynamic, Precip/Temp/PET shares, top-12 inputs."""
    models = shap_models()
    fig = plt.figure(figsize=vs.figsize(width="full", aspect=0.8))
    left, right = fig.subfigures(1, 2, width_ratios=[1, 1.05])
    a, b = left.subplots(2, 1)
    c = right.subplots()
    S.panel_static_vs_dynamic(a, models)
    S.panel_raw_variable_boxes(b, models)
    panel_top_features(c, models)
    label([a, b, c], ["Static vs dynamic (between-basin)", "Dynamic inputs (within-basin)",
                      "Most important inputs (between-basin)"])
    save(fig, "fig02_shap_attribution")


def panel_conditional(axs, models):
    """Precip share by flow tercile of the sampled day (axs[0]) and by basin BFI tercile (axs[1])."""
    days = pd.read_csv(S.A / "sample_days.csv", dtype={"basin": str})
    basins = pd.read_csv(S.A / "sample_basins.csv", dtype={"basin": str})
    terc = ["low", "mid", "high"]
    for m in models:
        s = S.raw_variable_share(S.load_long(m), "within")
        s = s[s.variable == "Precip"].merge(days[["basin", "date", "flow_tercile"]], on=["basin", "date"]) \
            .merge(basins, on="basin")
        for ax, cname in zip(axs, ["flow_tercile", "bfi_tercile"]):
            g = s.groupby(cname)["share"]
            med = 100 * g.median().reindex(terc)
            q1, q3 = 100 * g.quantile(0.25).reindex(terc), 100 * g.quantile(0.75).reindex(terc)
            ax.errorbar(np.arange(3), med, yerr=[med - q1, q3 - med], color=vs.model_color(m), marker="o",
                        ms=ctx["ms"], lw=ctx["lw_main"] if m == S.MAIN else ctx["lw"], capsize=2,
                        elinewidth=ctx["lw_thin"], label=m, zorder=3 if m == S.MAIN else 2)
    for ax, xl in zip(axs, ["Flow tercile of day", "Basin BFI tercile"]):
        ax.set_xticks(range(3), terc)
        ax.set(xlabel=xl, ylim=(15, 95))
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Precip |attribution|\nshare [%]")
    if len(axs) > 1:
        axs[1].tick_params(labelleft=False)
    axs[0].legend(loc="upper left", frameon=False, ncol=2, fontsize=ctx["annot"], columnspacing=0.8,
                  handlelength=1.5)


def s02():
    """SHAP supplement: tree engineered-family heatmap (a) and Precip share by flow state (b, narrower)."""
    models = shap_models()
    fig = plt.figure(figsize=vs.figsize(width="full", aspect=0.6))
    left, right = fig.subfigures(1, 2, width_ratios=[1.15, 1])
    a = left.subplots()
    b = right.subplots()
    S.panel_tree_family_heatmap(a, [m for m in models if m in ("RF", "XGB")])
    for t_ in a.texts:                       # cell values: larger than the default annotation size
        t_.set_fontsize(ctx["label"])
    panel_conditional([b], models)
    label([a, b], ["Trees: engineered families", "Precip share by flow state"])
    save(fig, "s02_shap")


def csv02():
    """SHAP shares [%]: static / dynamic share (between-basin baseline; RF path-dependent) and the
    median Precip / Temp / PET share of the dynamic attribution (within-basin baseline)."""
    models = shap_models()
    fig, ax = plt.subplots()
    sd = S.panel_static_vs_dynamic(ax, models).set_index("model")
    plt.close(fig)
    rows = []
    for m in models:
        s = S.raw_variable_share(S.load_long(m), "within")
        med = 100 * s.groupby("variable")["share"].median()
        st = sd.loc[m, "static_share_pct"]
        rows.append({"Model": m + ("*" if sd.loc[m, "baseline"] == "tree_path_dependent" else ""),
                     "Static [%]": f2(st), "Dynamic [%]": f2(100 - st),
                     **{f"{v} [% of dynamic]": f2(med.get(v)) for v in RAW_VARS}})
    save_table(pd.DataFrame(rows), "csv02_shap_shares")


RAW_VARS = ["Precip", "Temp", "PET"]

JOBS = {f.__name__: f for f in [fig02, s02, csv02]}

if __name__ == "__main__":
    only = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in JOBS.items():
        if not only or name in only:
            fn()
            print(name, "done", flush=True)
