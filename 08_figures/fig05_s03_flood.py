"""fig05  flood: elasticity, response speed vs BFI (3-day storm), flood checks F1-F3
s03    flood-flag example hydrographs, mean lag per storm duration, flood checks F4-F5
Tables: csv05a_flood_elasticity, csv05d_flood_speed_3day

Outputs: fig/<name>.png, fig/pdf/<name>.pdf, out/tables/<table>.csv
Usage: python 08_figures/fig05_s03_flood.py [fig05 s03 csv05]   (no args = all)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 08_figures
from common import *  # noqa: E402,F401,F403  shared settings, helpers and imports


def flood_tables():
    f = pd.read_csv(E.ANALYSIS_FLOOD_DIR / "flood_per_basin.csv", dtype={"basin": str})
    s = pd.read_csv(E.ANALYSIS_FLOOD_DIR / "flood_summary.csv")
    return f, s


def panel_speed(ax, f, keys, d, ylabel=True):
    """Fraction of the storm's extra runoff released by day 30, binned by baseflow index (HBV + regional)."""
    bins = np.linspace(0, 1, 6)
    for k in ref_keys(keys):
        m = split(k)[0]
        g = f[(f.combo == k) & (f.duration == d)]
        mids, vals = [], []
        for lo_, hi in zip(bins[:-1], bins[1:]):
            sel = g[(g.BFI >= lo_) & (g.BFI < hi)]["frac_dQ_day30"]
            if len(sel) >= 5:
                mids.append((lo_ + hi) / 2)
                vals.append(sel.median())
        ax.plot(mids, vals, color=col(m), ls=HBV_LS if m == "HBV" else "-", marker="o", ms=ctx["ms"],
                lw=ctx["lw"], label=m)
    ax.set(xlabel="Baseflow index", ylim=(0, 1.05), ylabel="ΣΔQ fraction by day 30" if ylabel else "")


FLOOD_CHECKS = [("F1", "F1 mono-\ntonic"), ("F2", "F2 ΣΔQ\n≤ P"), ("F3", "F3 not\ndamped"),
                ("F4", "F4 no\ndip"), ("F5", "F5 peak\nafter rain")]
FLOOD_MAIN = FLOOD_CHECKS[:3]    # fig05 c; F4 / F5 are in the supplement (s03)
FLOOD_SUPP = FLOOD_CHECKS[3:]
FLOOD_NORM = dict(gamma=0.7, vmin=0, vmax=85)


def flood_check_rates(f):
    """% of scored storm cases passing each check, per combo x duration: mean over cases per seed, median
    over seeds (the cases and checks of the flood score; F3 / F5 rain storms only, unfinished excluded)."""
    g = f.copy()
    for c, _ in FLOOD_CHECKS:
        g[c] = g[c].map(lambda v: np.nan if pd.isna(v) else float(bool(v)))
    cs = [c for c, _ in FLOOD_CHECKS]
    return 100 * g.groupby(["combo", "seed", "duration"])[cs].mean().groupby(["combo", "duration"]).median()


def panel_flood_checks(ax, f, keys, durs, checks=FLOOD_CHECKS, ylabels=True):
    """Heatmap: rows = models, one block of storm durations per check, cell = % cases passing."""
    r = flood_check_rates(f)
    mat = np.array([[r.loc[(k, d), c] for c, _ in checks for d in durs] for k in keys])
    bad = 100 - mat
    GAP, nd, nr = 0.28, len(durs), mat.shape[0]
    ncol = mat.shape[1]
    xc = np.array([j + GAP * (j // nd) for j in range(ncol)])
    norm = PowerNorm(**FLOOD_NORM)
    for bi in range(len(checks)):
        sl = slice(bi * nd, (bi + 1) * nd)
        ax.imshow(bad[:, sl], cmap="Reds", norm=norm, aspect="auto",
                  extent=(xc[sl][0] - 0.5, xc[sl][-1] + 0.5, nr - 0.5, -0.5))
    for i in range(nr):
        for j in range(ncol):
            ax.text(xc[j], i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=ctx["annot"] * 0.85,
                    color="white" if norm(bad[i, j]) > 0.62 else "black")
    ax.set_xticks(xc, [f"{d}" for _ in checks for d in durs], fontsize=ctx["annot"])
    ax.set_xlabel("Storm duration [d]")
    ax.set_yticks(range(nr), [short(k) for k in keys] if ylabels else [])
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for bi, (_, lab) in enumerate(checks):
        ax.text(xc[bi * nd + nd // 2], -0.7, lab, ha="center", va="bottom", fontsize=ctx["annot"])
    ax.set_xlim(xc[0] - 0.5, xc[-1] + 0.5)
    ax.set_ylim(nr - 0.5, -2.0)
    return mat


def panel_flood_checks_dots(ax, f, keys, durs, checks=FLOOD_MAIN, d_main=3, gap=40):
    """Compact non-heatmap version of the flood checks, in ONE axis: one 0-100 % strip per check side by
    side, one row per model. Filled dot = % cases passing for the d_main-day storm; line = range between
    the 1- and 5-day storms (white diamond = 1-day, filled circle = 3-day, vertical line = 5-day, same size)."""
    r = flood_check_rates(f)
    d_lo, d_hi = min(durs), max(durs)
    nr = len(keys)
    global MS_CHK, MEW_CHK, MEW_LINE
    MS_CHK = ctx["ms"] * 1.2          # marker size shared by the 1 / 3 / 5-day symbols
    MEW_CHK = ctx["mew"] * 2.0        # marker edge width (diamond / circle)
    MEW_LINE = ctx["lw"] * 1.8        # width of the 5-day vertical line
    x0 = [j * (100 + gap) for j in range(len(checks))]
    for j, (c, lab) in enumerate(checks):
        o = x0[j]
        ax.vlines([o + g for g in (25, 50, 75)], -0.5, nr - 0.5, color="0.88", lw=ctx["lw_axes"], zorder=1)
        # thin frame per strip so each 0-100 % range reads as its own box (no grey fill)
        ax.add_patch(plt.Rectangle((o, nr - 0.5), 100, -(nr - 0.5 + 0.5), fill=False, ec="0.55",
                                   lw=ctx["lw_axes"], zorder=1, clip_on=False))
        for i, k in enumerate(keys):
            cc = col(split(k)[0])
            v1, v3, v5 = (r.loc[(k, d), c] for d in (d_lo, d_main, d_hi))
            ax.plot([o + v1, o + v5], [i, i], color=cc, lw=ctx["lw"] * 1.5, alpha=0.45, solid_capstyle="butt", zorder=2)
            # 1-day white diamond, 3-day filled circle, 5-day vertical line; all the same size
            ax.plot(o + v5, i, "|", color=cc, ms=MS_CHK * 1.3, mew=MEW_LINE, zorder=3)
            ax.plot(o + v1, i, "D", mfc="white", mec=cc, ms=MS_CHK * 0.85, mew=MEW_CHK, zorder=3)
            ax.plot(o + v3, i, "o", mfc=cc, mec=cc, ms=MS_CHK, mew=MEW_CHK, zorder=4)
        ax.text(o + 50, -0.75, lab.replace("\n", " ").replace("- ", ""), ha="center", va="bottom",
                fontsize=ctx["legend"])
    ax.set_xticks([o + g for o in x0 for g in (0, 25, 50, 75, 100)],
                  ["0", "25", "50", "75", "100"] * len(checks), fontsize=ctx["annot"] * 1.1)
    ax.set_xlim(-6, x0[-1] + 106)
    ax.set_yticks(range(nr), [short(k) for k in keys])
    ax.set_ylim(nr - 0.5, -1.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="x", length=2.5, color="0.55")
    h = [Line2D([], [], ls="", marker="D", mfc="white", mec="0.3", ms=MS_CHK * 0.85, mew=MEW_CHK,
                label=f"{d_lo}-day"),
         Line2D([], [], ls="", marker="o", color="0.3", ms=MS_CHK, mew=MEW_CHK, label=f"{d_main}-day"),
         Line2D([], [], ls="", marker="|", color="0.3", ms=MS_CHK * 1.3, mew=MEW_LINE,
                label=f"{d_hi}-day storm")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False,
              fontsize=ctx["legend"], handlelength=1.4, columnspacing=1.0)
    return ax


def panel_lag(ax, f, keys, durs):
    """Mean over basins of the lag of the dQ peak behind the storm-rain centroid, per storm duration."""
    lag = f.groupby(["combo", "duration", "basin"])["lag_days"].median().reset_index()
    for k in keys:
        m, st = split(k)
        g = lag[lag.combo == k].groupby("duration")["lag_days"].mean()
        ax.plot(g.index, g.values, color=col(m), ls=HBV_LS if m == "HBV" else STRAT_LS[st],
                marker=STRAT_MK[st], ms=ctx["ms"], lw=ctx["lw"], alpha=0.9)
    ax.set(xlabel="Storm duration [d]", ylabel="Mean lag of ΔQ peak [d]", xticks=durs, ylim=(0, 19))
    blank = Line2D([], [], lw=0)
    sh = [Line2D([], [], color="0.3", ls=STRAT_LS[t], marker=STRAT_MK[t], ms=ctx["ms"], lw=ctx["lw"]) for t in STRAT_LS]
    ax.legend(T.model_handles() + sh + [blank, blank], MODEL_ORDER + list(STRAT_LS) + ["", ""], loc="upper left",
              ncol=2, frameon=False, handlelength=1.8, columnspacing=0.6, borderaxespad=0.1, fontsize=ctx["annot"])


def fig05(heatmap=False):
    """Flood: (a) elasticity, (b) response speed vs BFI (3-day storm), (c) pass rate of flood checks F1-F3:
    dot-and-range chart (filled circle = 3-day storm, white diamond = 1-day, line = 5-day). F4 / F5 and the
    mean lag: supplement (s03).
    heatmap=True -> panel c as a heatmap by storm duration, saved to fig/test_fig."""
    compact = not heatmap
    f, _ = flood_tables()
    keys = ordered(combos())
    durs = list(E.STORM_DURATIONS)
    off = {d: (i - (len(durs) - 1) / 2) * 0.27 for i, d in enumerate(durs)}
    fig = plt.figure(figsize=vs.figsize(width="full", aspect=0.82 if compact else 0.88))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[0.8, 1.2] if compact else [1, 1.25])
    axes, titles = [], []

    # (a) elasticity, every storm duration
    ax = fig.add_subplot(gs[0, :])
    for d in durs:
        data = [f[(f.combo == k) & (f.duration == d)].groupby("basin")["elasticity"].median().dropna() for k in keys]
        bp = ax.boxplot(data, positions=np.arange(len(keys)) + off[d], widths=0.24, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=ctx["lw"]),
                        whiskerprops=dict(lw=ctx["lw_axes"]), capprops=dict(lw=ctx["lw_axes"]))
        for b, k in zip(bp["boxes"], keys):
            c = col(split(k)[0])
            b.set(facecolor=(*to_rgb(c), T.DUR_ALPHA[d]), edgecolor=c, lw=ctx["lw_axes"] * 1.2)
    T.combo_xticks(ax, keys)
    ax.axhline(1, color=vs.OBS, lw=ctx["lw_axes"], ls="--")
    ax.set(ylabel="Peak elasticity of ΔQ")
    ax.legend(handles=[Patch(facecolor=(0.3, 0.3, 0.3, T.DUR_ALPHA[d]), edgecolor="0.3", label=f"{d}-day storm")
                       for d in durs] + [Line2D([], [], color=vs.OBS, ls="--", lw=ctx["lw_axes"], label="proportional (= 1)")],
              loc="upper left", frameon=False, ncol=2, handlelength=1.2, borderaxespad=0.1)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    axes.append(ax); titles.append("Peak-ΔQ elasticity to storm uplift")

    # (b) response speed (3-day storm)
    ax = fig.add_subplot(gs[1, 0])
    panel_speed(ax, f, keys, 3)
    ax.legend(loc="lower left", frameon=False, handlelength=1.8, fontsize=ctx["annot"], ncol=2, columnspacing=0.8)
    axes.append(ax); titles.append("Response speed, 3-day storm")

    # (c) flood checks F1-F3 (F4 / F5: s03)
    if compact:
        ax = fig.add_subplot(gs[1, 1])
        panel_flood_checks_dots(ax, f, keys, durs)
    else:
        ax = fig.add_subplot(gs[1, 1])
        panel_flood_checks(ax, f, keys, durs, checks=FLOOD_MAIN)
    axes.append(ax); titles.append("Flood checks [% cases passing]")
    label(axes, titles)
    if compact:
        save(fig, "fig05_flood")
    else:
        T.save(fig, "fig05_test_heatmap_checks")


def s03():
    """Flood supplement: flag examples (failing ML case, top; HBV on the same basin and storm, bottom) and
    the mean lag of the dQ peak per storm duration (right)."""
    f, _ = flood_tables()
    picks = T.flag_example_picks(f)
    n = len(picks)
    fig = plt.figure(figsize=vs.figsize(width="full", aspect=1.25))
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 1.2])   # a-d examples on top, e-f below
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(n)] for i in range(2)])
    lax = fig.add_subplot(gs[2, 0])          # mean lag
    panel_lag(lax, f, ordered(combos()), list(E.STORM_DURATIONS))
    cax = fig.add_subplot(gs[2, 1])          # flood checks F4 / F5
    panel_flood_checks(cax, f, ordered(combos()), list(E.STORM_DURATIONS), checks=FLOOD_SUPP)
    names = {"monotonic": "Non-monotonic", "over_cap": "Runoff > storm rainfall", "incomplete": "Not finished in 365 d"}
    for j, (flag, combo, b, seed) in enumerate(picks):
        for i, key in enumerate([combo, "HBV-individual"]):
            T.draw_flag_example(axs[i, j], flag, key, b, seed, xlabel=i == 1)
    axs[0, 0].legend(loc="lower right", bbox_to_anchor=(1, 0.08), frameon=False, title="uplift", ncol=2,
                     fontsize=ctx["annot"], title_fontsize=ctx["annot"], handlelength=1.2, columnspacing=0.6)
    label(list(axs.flat) + [lax, cax], [names[p[0]] for p in picks] + ["HBV, same storm"] * len(picks)
          + ["Mean lag over basins", "Checks F4–F5 [% passing]"])
    save(fig, "s03_flood_examples_lag")


def csv05():
    """Flood: (a) median peak elasticity per storm duration; (d) ΣΔQ fraction by day 30 per BFI bin,
    3-day storm, HBV + regional models."""
    f, _ = flood_tables()
    rows = []
    for m, s, k in combo_rows():
        row = {"Model": m, "Strategy": s}
        for d in E.STORM_DURATIONS:
            e = f[(f.combo == k) & (f.duration == d)].groupby("basin")["elasticity"].median()
            row[f"Elasticity {d}-day (median)"] = f2(e.median())
            row[f"Elasticity {d}-day (IQR)"] = f"{f2(e.quantile(.25))} to {f2(e.quantile(.75))}"
        rows.append(row)
    save_table(pd.DataFrame(rows), "csv05a_flood_elasticity")
    bins = np.linspace(0, 1, 6)
    rows = []
    for k in ref_keys(ordered(combos())):
        g = f[(f.combo == k) & (f.duration == 3)]
        row = {"Model": split(k)[0], "Strategy": split(k)[1]}
        for lo_, hi in zip(bins[:-1], bins[1:]):
            sel = g[(g.BFI >= lo_) & (g.BFI < hi)]["frac_dQ_day30"]
            row[f"BFI {lo_:.1f}–{hi:.1f}"] = f2(sel.median()) if len(sel) >= 5 else "–"
        rows.append(row)
    save_table(pd.DataFrame(rows), "csv05d_flood_speed_3day")


JOBS = {f.__name__: f for f in [fig05, s03, csv05]}

if __name__ == "__main__":
    only = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in JOBS.items():
        if not only or name in only:
            fn()
            print(name, "done", flush=True)
