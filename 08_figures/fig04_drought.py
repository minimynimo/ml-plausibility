"""fig04  drought: late flow, example hydrographs (seed 42), zero-rain vs observed recession timescale
Tables: csv04_drought

Outputs: fig/<name>.png, fig/pdf/<name>.pdf, fig/<table>.csv
Usage: python 08_figures/fig04_drought.py [fig04 csv04]   (no args = all)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 08_figures
from common import *  # noqa: E402,F401,F403  shared settings, helpers and imports


def fig04():
    """Drought: flow left after 1-6 dry years (a), example hydrographs (b-d), recession timescale in the
    zero-rain run vs observed dry spells (e-i). Hydrographs show one seed (SHOW_SEED, = runs_of(k)[0]);
    a and e-i use the median over all seeds per basin. In e-i the three example basins are ringed at
    their seed median, with a dot at the plotted seed's value (joined by a thin line when they differ)."""
    d, med = T.drought_k_table()
    ex = json.loads((E.ANALYSIS_DROUGHT_DIR / "drought_example_basins.json").read_text())
    keys = ordered(combos())
    refs = ref_keys(keys)
    seed = runs_of(refs[1])[0].seed           # seed drawn in b-d (same first seed for every model)
    fig = plt.figure(figsize=vs.figsize(width="full", aspect=1.18))
    top, mid, bot = fig.subfigures(3, 1, height_ratios=[1.05, 1, 0.8])
    ax = top.subplots()
    # linear axis so the ideal value (0) is on the plot (a log axis can never show it)
    strategy_boxes(ax, [med[med.combo == k].meanQ_2002_2006 for k in keys], keys)
    ax.set_ylim(bottom=-0.1)
    ax.axhline(0, color="0.6", lw=ctx["lw"] * 0.6, ls=":", zorder=0)
    ax.set(ylabel="Mean Q, 2002–2006 [mm d$^{-1}$]")
    strategy_box_legend(ax, loc="upper right", borderaxespad=0.1)
    axes, titles = [ax], ["Flow left after 1–6 years without rain"]
    bax = mid.subplots(1, 3)
    for j, (name, b) in enumerate(ex.items()):
        a = bax[j]
        for k in refs:
            m = split(k)[0]
            r = [r for r in runs_of(k) if r.seed == seed] or runs_of(k)[:1]   # HBV: single calibration
            q = runio.load_sim(r[0].dir, "no_precip")[b]
            a.plot(q.index, q.values, color=col(m), lw=ctx["lw"] * (HBV_LW if m == "HBV" else 1),
                   ls=HBV_LS if m == "HBV" else "-", zorder=5 if m == "HBV" else 3, alpha=0.95)
        # linear axis: the ideal no-rain response (Q -> 0) must be visible, which a log axis cannot show
        a.set(ylim=(0, HYDRO_YMAX.get(name)), ylabel="Q [mm d$^{-1}$]" if j == 0 else "")
        a.xaxis.set_major_locator(mdates.YearLocator(2))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes.append(a); titles.append(f"{name.replace('_', ' ')} basin ({b})")
    mid.legend(hydro_handles(), MODEL_ORDER, loc="outside upper center", ncol=5, frameon=False, handlelength=1.8)
    kax = bot.subplots(1, len(refs), sharey=True)
    titles += panel_k_scatter(kax, d, med, refs, ex, seed)
    axes += list(kax)
    label(axes, titles)
    save(fig, "fig04_drought")


K_Y_DRAINED, K_Y_FLAT = 0.35, T.K_CAP * 3      # where the drained / flat rugs are drawn


def _k_y(k, cls):
    return K_Y_DRAINED if cls == "drained" else K_Y_FLAT if cls == "flat" else k


def panel_k_scatter(axs, d, med, refs, ex, seed):
    """Per model: zero-rain vs observed recession timescale, one point per basin (median over seeds), 1:1
    line. Drained (shortest timescale) / flat (longest) basins as rugs with their counts; example basins
    of the hydrograph panels ringed (seed median) and lettered."""
    n_basins = med.basin.nunique()
    for i, (a, k) in enumerate(zip(axs, refs)):
        m = split(k)[0]
        g = med[med.combo == k]
        f = g[g.k_class == "fitted"]
        a.scatter(f.k_obs, f.k_syn, s=5, color=col(m), alpha=0.35, lw=0, rasterized=True)
        for cls, y in [("drained", K_Y_DRAINED), ("flat", K_Y_FLAT)]:
            gg = g[g.k_class == cls]
            a.scatter(gg.k_obs, np.full(len(gg), y), s=10, color=col(m), alpha=0.5, marker="|", lw=0.6,
                      rasterized=True)
            # share just above its rug (log axis: x1.5), right-aligned, so it never sits on the ticks
            a.text(0.98, y * 1.5, f"{100 * len(gg) / n_basins:.0f} %", transform=a.get_yaxis_transform(),
                   ha="right", va="bottom", fontsize=ctx["annot"], color="0.2", zorder=8)
        T.d1_band(a)   # D1 pass region (observation-calibrated factor); explained in the caption
        a.plot([1, T.K_CAP], [1, T.K_CAP], color=vs.OBS, lw=ctx["lw_axes"])
        for lab, b in zip("bcd", ex.values()):
            r = g[g.basin == b]
            if r.empty:
                continue
            r = r.iloc[0]
            ym = _k_y(r.k_syn, r.k_class)
            a.scatter(r.k_obs, ym, s=22, facecolor="none", edgecolor="k", lw=0.7, zorder=6,
                      path_effects=[pe.Stroke(linewidth=1.6, foreground="white"), pe.Normal()])
            # left of the ring (clear of the right-aligned shares), below it on the flat row;
            # white halo so the letter reads inside dense point clouds
            # b (high BFI, k_obs 42 d) and d (snow, 45 d) share almost the same x and, for some models,
            # the same y: put d on the right so the two letters never cover each other
            dy, va = (-4, "top") if r.k_class == "flat" else (3, "bottom")
            dx, ha = (4, "left") if lab == "d" else (-4, "right")
            a.annotate(lab, (r.k_obs, ym), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                       fontsize=ctx["annot"], fontweight="bold", zorder=7,
                       path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        a.set(xscale="log", yscale="log", xlim=(1, T.K_CAP), ylim=(0.15, T.K_CAP * 14))
        a.minorticks_off()
        a.set_xticks([10, 1e3], ["10", "10$^3$"])
        a.set_yticks([K_Y_DRAINED, 1, 1e2, 1e4, K_Y_FLAT], ["drained", "1", "10$^2$", "10$^4$", "flat"])
        a.set_xlabel("Observed k [d]")
        if i == 0:
            a.set_ylabel("Zero-rain k [d]")
        else:
            a.tick_params(labelleft=False)
    return [short(k) for k in refs]


def hydro_handles():
    """Model legend for hydrograph panels, HBV line thickened to match HBV_LW."""
    hs = T.model_handles()
    for h, m in zip(hs, MODEL_ORDER):
        if m == "HBV":
            h.set_linewidth(h.get_linewidth() * HBV_LW)
    return hs


def csv04():
    """Drought: mean Q 2002-2006 of the zero-precipitation run [mm/d], median and IQR over basins."""
    _, med = T.drought_k_table()
    rows = []
    for m, s, k in combo_rows():
        q = med[med.combo == k].meanQ_2002_2006
        rows.append({"Model": m, "Strategy": s, "Mean Q 2002–2006 median [mm/d]": f2(q.median()),
                     "IQR [mm/d]": f"{f2(q.quantile(.25))} to {f2(q.quantile(.75))}",
                     "Basins with Q < 0.01 mm/d [%]": f2(100 * (q < 0.01).mean())})
    save_table(pd.DataFrame(rows), "csv04_drought")


JOBS = {f.__name__: f for f in [fig04, csv04]}

if __name__ == "__main__":
    only = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in JOBS.items():
        if not only or name in only:
            fn()
            print(name, "done", flush=True)
