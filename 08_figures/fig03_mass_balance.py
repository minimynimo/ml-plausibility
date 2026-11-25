"""fig03  mass balance: volume error, water created / lost, observed vs simulated implied ET
Tables: csv03_mass_balance, csv03b_water_balance_flag_overlap

Outputs: fig/<name>.png, fig/pdf/<name>.pdf, out/tables/<table>.csv
Usage: python 08_figures/fig03_mass_balance.py [fig03 csv03 csv03b]   (no args = all)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 08_figures
from common import *  # noqa: E402,F401,F403  shared settings, helpers and imports


# ------------------------------------------------------------------------------------------
# Plausibility
# ------------------------------------------------------------------------------------------
def fig03():
    """Mass balance: volume error + water-balance violations."""
    pb = pd.read_csv(E.ANALYSIS_MASS_DIR / "mass_per_basin.csv", dtype={"basin": str})
    obs = pd.read_csv(E.ANALYSIS_MASS_DIR / "mass_observed_per_basin.csv", dtype={"basin": str})
    keys = ordered(combos())
    fig = plt.figure(figsize=vs.figsize(width="full", aspect=0.84))
    top, bot = fig.subfigures(2, 1, height_ratios=[1.55, 1], hspace=0.04)
    axs = list(top.subplots(1, 2))
    ax = axs[0]
    ens = pb.groupby(["combo", "basin"])["volume_error"].median().reset_index()
    strategy_boxes(ax, [100 * ens[ens.combo == k]["volume_error"] for k in keys], keys)
    ax.axhline(0, color=vs.OBS, lw=ctx["lw_axes"])
    ax.set(ylabel="Volume error [%]", ylim=(-50, 72))
    strategy_box_legend(ax, loc="upper left", borderaxespad=0.1)
    ax = axs[1]
    # Two mutually exclusive failure types over the test period (implied ET = mean(P - Q_sim)):
    #   creates water: ET < 0 (sum Q_sim > sum P);  loses water: ET > mean PET (too little runoff).
    rows = []
    for k in keys:
        per_seed = pb[pb.combo == k].groupby("seed")[["flag_ET_neg", "flag_ET_gt_PET"]].mean()
        rows.append(100 * per_seed.median())
    t = pd.DataFrame(rows, index=keys)
    for i, (k, r) in enumerate(t.iterrows()):
        c = col(split(k)[0])
        ax.bar(i, r.flag_ET_neg, 0.65, color=c, edgecolor=c, lw=ctx["lw_axes"])
        ax.bar(i, r.flag_ET_gt_PET, 0.65, bottom=r.flag_ET_neg, color=(*to_rgb(c), 0.18), edgecolor=c,
               lw=ctx["lw_axes"], hatch="////")
    # Reference: the same two checks applied to the observed discharge. Bars are stacked, so the lower
    # line is the observed "creates water" share and the upper one observed creates + loses.
    o_neg, o_pet = obs_flag_rates(obs)
    ax.axhline(o_neg, color=vs.OBS, lw=ctx["lw"] * 1.35, ls=":", zorder=5)
    ax.axhline(o_neg + o_pet, color=vs.OBS, lw=ctx["lw"], ls="--", zorder=5)
    T.combo_xticks(ax, keys)
    ax.set(ylabel="Basins where the model\nviolates the water balance [%]",
           ylim=(0, max((t.flag_ET_neg + t.flag_ET_gt_PET).max(), o_neg + o_pet) * 1.75))
    ax.legend(handles=[Patch(facecolor="0.35", edgecolor="0.35", label="creates water (ΣQ > ΣP)"),
                       Patch(facecolor=(0.35, 0.35, 0.35, 0.18), edgecolor="0.35", hatch="////",
                             label="loses water (ET > PET)"),
                       Line2D([], [], color=vs.OBS, lw=ctx["lw"] * 1.35, ls=":",
                              label=f"observed Q: creates ({o_neg:.1f} %)"),
                       Line2D([], [], color=vs.OBS, lw=ctx["lw"], ls="--",
                              label=f"observed Q: creates + loses ({o_neg + o_pet:.1f} %)")],
              loc="upper left", frameon=False, handlelength=1.8, fontsize=ctx["annot"])
    sax = list(bot.subplots(1, 5, sharex=True, sharey=True))
    et_scatter(sax, pb, obs)
    sax[2].set_xlabel(r"Observed implied ET [mm d$^{-1}$]")     # centred under the row (5 panels)
    label(axs + sax, ["Volume error per basin (evaluation)", "Water-balance violations"]
          + [short(k) for k in ref_keys(combos())])
    save(fig, "fig03_mass_balance")


def obs_flag_bool(obs, f):
    return obs[f].astype(str).str.lower().eq("true")


def obs_flag_rates(obs):
    """% of basins (with an observed water balance) whose observed discharge fails each model check."""
    o = obs.dropna(subset=["implied_ET"])
    return 100 * obs_flag_bool(o, "flag_ET_neg").mean(), 100 * obs_flag_bool(o, "flag_ET_gt_PET").mean()


def et_scatter(axs, pb, obs):
    """Observed vs simulated implied ET (mean(P - Q), test period), HBV + the regional ML models;
    simulated = median over seeds; RMSE / bias over all basins with an observed water balance."""
    obs = obs.dropna(subset=["implied_ET"]).set_index("basin")
    x = obs["implied_ET"]
    sim = pb.groupby(["combo", "basin"])["implied_ET"].median()
    lim = (-3, 4)
    for j, (ax, k) in enumerate(zip(axs, ref_keys(combos()))):
        y = sim.loc[k].reindex(obs.index)
        c = col(split(k)[0])
        ax.plot(lim, lim, color=vs.OBS, lw=ctx["lw_axes"], ls="--", zorder=1)
        ax.axhline(0, color="0.6", lw=ctx["lw_axes"], zorder=1)
        ax.axvline(0, color="0.6", lw=ctx["lw_axes"], zorder=1)
        ax.scatter(x, y, s=8, color=c, alpha=0.5, lw=0, zorder=3)
        d = (y - x).dropna()
        txt = f"RMSE {np.sqrt((d ** 2).mean()):.2f}\nbias {d.mean():+.2f}".replace("-", "−")
        ax.text(0.04, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=ctx["annot"] * 0.85,
                linespacing=1.15, zorder=5)
        ax.set(xlim=lim, ylim=lim, xticks=[-2, 0, 2, 4], yticks=[-2, 0, 2, 4])
        if j == 0:
            ax.set_ylabel("Simulated implied ET\n" + r"[mm d$^{-1}$]")


def csv03():
    """Mass balance: volume error (median and IQR over basins) and % of basins violating the water balance."""
    pb = pd.read_csv(E.ANALYSIS_MASS_DIR / "mass_per_basin.csv", dtype={"basin": str})
    ens = pb.groupby(["combo", "basin"])["volume_error"].median().reset_index()
    rows = []
    for m, s, k in combo_rows():
        v = 100 * ens[ens.combo == k]["volume_error"]
        fl = 100 * pb[pb.combo == k].groupby("seed")[["flag_ET_neg", "flag_ET_gt_PET", "flag_any"]].mean().median()
        rows.append({"Model": m, "Strategy": s, "Volume error median [%]": f2(v.median()),
                     "Volume error IQR [%]": f"{f2(v.quantile(.25))} to {f2(v.quantile(.75))}",
                     "Creates water [% basins]": f2(fl.flag_ET_neg), "Loses water [% basins]": f2(fl.flag_ET_gt_PET),
                     "Any violation [% basins]": f2(fl.flag_any)})
    save_table(pd.DataFrame(rows), "csv03_mass_balance")


def csv03b():
    """Do the models flag the same basins as the observed record? Per model-strategy, median over seeds:
    basins the model flags, how many of those the observed discharge also flags, and the share of the
    observed-flagged basins the model reproduces."""
    pb = pd.read_csv(E.ANALYSIS_MASS_DIR / "mass_per_basin.csv", dtype={"basin": str})
    obs = pd.read_csv(E.ANALYSIS_MASS_DIR / "mass_observed_per_basin.csv", dtype={"basin": str})
    obs = obs.dropna(subset=["implied_ET"])
    rows = []
    for m, s, k in combo_rows():
        row = {"Model": m, "Strategy": s}
        for f, name in [("flag_ET_neg", "Creates water"), ("flag_ET_gt_PET", "Loses water")]:
            ob = set(obs.basin[obs_flag_bool(obs, f)])
            per = []
            for _, g in pb[pb.combo == k].groupby("seed"):
                mb = set(g.basin[g[f].astype(bool)])
                per.append((len(mb), len(mb & ob)))
            n_mod, n_both = np.median(np.array(per), axis=0)
            row |= {f"{name}: model [n]": f"{n_mod:.0f}", f"{name}: also observed [n]": f"{n_both:.0f}",
                    f"{name}: of model flags also observed [%]": f2(100 * n_both / n_mod) if n_mod else "",
                    f"{name}: observed flags reproduced [%]": f2(100 * n_both / len(ob))}
        rows.append(row)
    o_neg, o_pet = obs_flag_rates(obs)
    rows.append({"Model": "Observed", "Strategy": "", "Creates water: model [n]":
                 f"{obs_flag_bool(obs, 'flag_ET_neg').sum()} ({f2(o_neg)} %)",
                 "Loses water: model [n]": f"{obs_flag_bool(obs, 'flag_ET_gt_PET').sum()} ({f2(o_pet)} %)"})
    save_table(pd.DataFrame(rows), "csv03b_water_balance_flag_overlap")


JOBS = {f.__name__: f for f in [fig03, csv03, csv03b]}

if __name__ == "__main__":
    only = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in JOBS.items():
        if not only or name in only:
            fn()
            print(name, "done", flush=True)
