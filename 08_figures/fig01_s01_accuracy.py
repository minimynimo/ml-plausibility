"""fig01  accuracy (KGE): CDFs per strategy + median of every model x strategy
s01    calibration vs test CDFs, KGE (top) / NSE (bottom)
Tables: csv01_accuracy_KGE, csv01b_share_KGE_below_0.6, csv01c_share_KGE_below_0, scsv02_accuracy_NSE

Outputs: fig/<name>.png, fig/pdf/<name>.pdf, out/tables/<table>.csv
Usage: python 08_figures/fig01_s01_accuracy.py [fig01 s01 csv01 csv01b csv01c scsv02]   (no args = all)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 08_figures
from common import *  # noqa: E402,F401,F403  shared settings, helpers and imports


def model_strategy_handles():
    mh = [Line2D([], [], color=col(m), lw=ctx["lw"] * 1.3, ls=HBV_LS if m == "HBV" else "-") for m in MODEL_ORDER]
    sh = [Line2D([], [], color="0.3", lw=ctx["lw"] * 1.3, ls=STRAT_LS[s]) for s in STRAT_LS]
    return mh + sh, MODEL_ORDER + list(STRAT_LS)


def cdf_line(ax, pb, key, metric, period, ls=None, lw=None):
    m, st = split(key)
    g = pb[(pb.combo == key) & (pb.period == period)][f"{metric}_ens"]
    ax.plot(GRID, ecdf(g, GRID), color=col(m), ls=HBV_LS if m == "HBV" else (ls or STRAT_LS[st]),
            lw=lw or ctx["lw"], drawstyle="steps-post", zorder=4 if m == "HBV" else 3)
    return float(np.median(g.dropna()))


def cdf_axes(ax, xlabel, ylabel=True):
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=xlabel, ylabel=vs.LABELS["cdf"] if ylabel else "")


# ------------------------------------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------------------------------------
def accuracy_cal_test(stem):
    """Calibration vs test CDFs, every model x strategy; rows KGE / NSE."""
    pb = accuracy()
    keys = ordered(combos())
    fig, axs = plt.subplots(2, 2, figsize=vs.figsize(width="full", aspect=0.9), sharey=True)
    titles = []
    for i, metric in enumerate(["KGE", "NSE"]):
        for j, (period, name) in enumerate([("cal", "calibration 1988–1997"), ("test", "evaluation 2001–2006")]):
            for k in keys:
                cdf_line(axs[i, j], pb, k, metric, period)
            cdf_axes(axs[i, j], metric, ylabel=j == 0)
            titles.append(f"{metric}, {name}")
    h, l = model_strategy_handles()
    fig.legend(h, l, loc="outside lower center", ncol=8, frameon=False, handlelength=1.8, columnspacing=1.0)
    label(axs.flat, titles)
    save(fig, stem)


def panel_median_summary(ax, pb, metric):
    """Median test score of every model x strategy (95 % bootstrap CI over basins); HBV as reference."""
    g = lambda k: pb[(pb.combo == k) & (pb.period == "test")][f"{metric}_ens"]
    hm, hlo, hhi = boot_median_ci(g("HBV-individual"))
    ax.axhspan(hlo, hhi, color="0.85", lw=0, zorder=0)
    ax.axhline(hm, color=vs.OBS, ls=HBV_LS, lw=ctx["lw"], zorder=1)
    ax.text(-0.45, hm, "HBV", va="bottom", fontsize=ctx["annot"])
    off = {"individual": -0.22, "regional": 0.0, "adapted": 0.22}
    rows = []
    for i, m in enumerate(T.ML):
        xs, ys = [], []
        for s in E.STRATEGIES[m]:
            med, lo, hi = boot_median_ci(g(f"{m}-{s}"))
            ax.errorbar(i + off[s], med, yerr=[[med - lo], [hi - med]], fmt=STRAT_MK[s], color=col(m),
                        mfc="white" if s == "individual" else col(m), ms=ctx["ms"] * 1.5, lw=ctx["lw"],
                        capsize=2, zorder=3)
            xs.append(i + off[s]); ys.append(med)
            rows.append({"model": m, "strategy": s, "median": med, "ci_lo": lo, "ci_hi": hi})
        ax.plot(xs, ys, color=col(m), lw=ctx["lw_thin"], zorder=2)
    ax.set_xticks(range(len(T.ML)), T.ML)
    ax.set_xlim(-0.6, len(T.ML) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel=f"Median {metric} (evaluation)")
    ax.legend([Line2D([], [], color="0.3", marker=STRAT_MK[s], lw=0, ms=ctx["ms"] * 1.5,
                      mfc="white" if s == "individual" else "0.3") for s in STRATS], STRATS,
              loc="upper left", frameon=False, handletextpad=0.2)
    return pd.DataFrame(rows)


def accuracy_by_strategy(metric, stem):
    """CDFs per strategy (a-c, top row) + median of every model x strategy for `metric` (d) and NSE (e)."""
    pb = accuracy()
    fig = plt.figure(figsize=vs.figsize(width="full", aspect=0.78))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1])
    top, bot = gs[0].subgridspec(1, 3), gs[1].subgridspec(1, 2)
    axs = [fig.add_subplot(top[0, j]) for j in range(3)] + [fig.add_subplot(bot[0, j]) for j in range(2)]
    for j, strat in enumerate(STRATS):
        ax = axs[j]
        for k in ordered([k for k in combos() if split(k)[1] == strat or split(k)[0] == "HBV"]):
            med = cdf_line(ax, pb, k, metric, "test", ls="-", lw=ctx["lw"] * 1.2)
            ax.lines[-1].set_label(f"{split(k)[0]} ({med:.2f})")
        cdf_axes(ax, f"{metric} (evaluation)", ylabel=j == 0)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8])   # no 0 / 1 labels: they would touch across the narrow gaps
        ax.tick_params(labelleft=j == 0)   # shared 0-1 y range: tick labels on a only, narrower gaps
        ax.legend(loc="upper left", frameon=False, title="median", alignment="left", handlelength=1.0,
                  handletextpad=0.4, borderaxespad=0.2, fontsize=ctx["annot"], title_fontsize=ctx["annot"])
    d = panel_median_summary(axs[3], pb, metric)
    d.to_csv(E.ANALYSIS_ACCURACY_DIR / f"{stem}.csv", index=False)
    d = panel_median_summary(axs[4], pb, "NSE")
    d.to_csv(E.ANALYSIS_ACCURACY_DIR / f"{stem}_NSE.csv", index=False)
    label(axs, [f"{metric}, {s} models" for s in STRATS]
          + [f"Median {m} by model and strategy (95 % CI)" for m in (metric, "NSE")])
    save(fig, stem)


def fig01(): accuracy_by_strategy("KGE", "fig01_accuracy")


def s01(): accuracy_cal_test("s01_accuracy_cal_test")


def accuracy_table(metric):
    """Median test score (95 % bootstrap CI of the median, over basins): model rows x strategy columns."""
    pb = accuracy()
    rows = []
    for m in MODEL_ORDER:
        row = {"Model": m}
        for s in STRATS:
            k = f"{m}-{s}"
            if k not in combos():
                row[s] = "–"
                continue
            med, lo, hi = boot_median_ci(pb[(pb.combo == k) & (pb.period == "test")][f"{metric}_ens"])
            row[s] = f"{f2(med)} ({f2(lo)}–{f2(hi)})"
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={s: s.capitalize() for s in STRATS})


def csv01():
    save_table(accuracy_table("KGE"), "csv01_accuracy_KGE")


def share_kge_below(thr):
    """Share of basins with test KGE < thr: HBV and the regional models (fig01 panel b)."""
    pb = accuracy()
    rows = []
    for k in ["HBV-individual"] + [f"{m}-regional" for m in T.ML]:
        g = pb[(pb.combo == k) & (pb.period == "test")]["KGE_ens"].dropna()
        rows.append({"Model": split(k)[0], "Strategy": split(k)[1], "N basins": len(g),
                     f"N KGE<{thr}": int((g < thr).sum()), f"Share KGE<{thr} [%]": f"{100 * (g < thr).mean():.1f}"})
    return pd.DataFrame(rows)


def csv01b():
    save_table(share_kge_below(0.6), "csv01b_share_KGE_below_0.6")


def csv01c():
    save_table(share_kge_below(0), "csv01c_share_KGE_below_0")


def scsv02():
    save_table(accuracy_table("NSE"), "scsv02_accuracy_NSE")


JOBS = {f.__name__: f for f in [fig01, s01, csv01, csv01b, csv01c, scsv02]}

if __name__ == "__main__":
    only = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in JOBS.items():
        if not only or name in only:
            fn()
            print(name, "done", flush=True)
