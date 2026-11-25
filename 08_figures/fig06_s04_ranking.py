"""fig06  accuracy vs plausibility + best model by weighting (weighted sum)
s04    ranking sensitivity, ML only: weighted-product surfaces, rank shares, drought left out
Tables: scsv04_composite_no_mass

Outputs: fig/<name>.png, fig/pdf/<name>.pdf, out/tables/<table>.csv
Usage: python 08_figures/fig06_s04_ranking.py [fig06 s04 scsv04]   (no args = all)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 08_figures
from common import *  # noqa: E402,F401,F403  shared settings, helpers and imports
RULES = {"Weighted sum": T.score_wsm, "Weighted product": T.score_wpm, "TOPSIS": T.score_topsis}
FLOOD_W = np.round(np.linspace(0, 1, 21), 2)
RULE_YLABEL = {"Weighted sum": "Composite (weighted sum)", "Weighted product": "Weighted-product score",
               "TOPSIS": "TOPSIS closeness"}


def ml_line_handle(k):
    m, st = split(k)
    return Line2D([], [], color=col(m), ls=STRAT_LS[st], lw=ctx["lw"], marker=STRAT_MK[st],
                  mfc=T.strat_fill(m, st), mec=col(m), ms=ctx["ms"] * 1.3)


def ml_weight_lines(ax, raw, main, rule):
    """ML only, drought left out: no ML model passes the drought checks in more than ~1 % of basins, so the
    ML-only ranking rests on mass and flood, i.e. a single slider from all-mass (0) to all-flood (1)
    weighting. One line per ML combo; markers at the two ends; dotted line = equal weights."""
    cand = [k for k in main if split(k)[0] != "HBV"]
    r = raw.loc[cand]
    sc = pd.DataFrame({w: RULES[rule](r, pd.Series([1 - w, 0.0, w], index=r.columns)) for w in FLOOD_W}).T
    for k in cand:
        m, st = split(k)
        ax.plot(FLOOD_W, sc[k], color=col(m), ls=STRAT_LS[st], lw=ctx["lw"], marker=STRAT_MK[st],
                markevery=[0, len(FLOOD_W) - 1], mfc=T.strat_fill(m, st), mec=col(m), ms=ctx["ms"] * 1.3)
    ax.axvline(0.5, color="0.6", lw=ctx["lw_axes"], ls=":", zorder=0)
    ax.set(xlim=(0, 1), xlabel="Flood weight (mass = 1 − flood)", ylabel=RULE_YLABEL[rule])
    return cand, sc.rename_axis("w_flood").reset_index().assign(rule=rule)


def fig06():
    """(a) accuracy vs composite plausibility; (b) dimension scores, strict -> check-averaged dumbbells;
    (c, d) best model per weighting (weighted sum; all models / ML only), coloured by the tied set
    (common_ranking.py). Bootstrap samples are rebuilt exactly as scoring.py drew them and checked."""
    import common_ranking as RV   # local import: only fig06 / s04 need it
    t, main, _ = plausibility_tables()
    boot = RV.bootstrap()
    RV.check_reproduction(boot, t)
    fig = RV.fig06_figure(t, boot, main)
    save(fig, "fig06_accuracy_vs_plausibility")


def s04():
    """Ranking sensitivity, ML only. (a) weighted-product score of every model over the weight triangle
    (drought check-averaged, resolution rule; hatched outline = where it ranks 1st); (b, c) ranking with
    drought left out (mass and flood only), weighted sum and weighted product, against the flood weight."""
    import common_ranking as RV
    t, main, raw = plausibility_tables()
    ml = [k for k in main if split(k)[0] != "HBV"]
    fig = plt.figure(figsize=vs.figsize(width="full", aspect=0.95))
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 0.95])
    RV.draw_surfaces(top, t, main, "Weighted product")
    _, sw = T.comparison_raw(t, ml)
    note = f" ({', '.join(sw)}: check-averaged)" if sw else ""
    top.suptitle(f"$\\bf{{a}}$ Weighted-product score, ML only{note}", x=0.01, ha="left", fontsize=ctx["label"])
    bax = bot.subplots(1, 2)
    for a, rule in zip(bax, ["Weighted sum", "Weighted product"]):
        ml_weight_lines(a, raw, main, rule)
    bot.legend([ml_line_handle(k) for k in ml], [short(k) for k in ml], loc="outside lower center", ncol=5,
               frameon=False, handletextpad=0.3, columnspacing=1.0, handlelength=2.2, fontsize=ctx["annot"])
    label(list(bax), ["Weighted sum, no drought", "Weighted product, no drought"], start=1)
    save(fig, "s04_ranking_sensitivity")


def scsv04():
    """Supplement: composite with and without the mass dimension, and the rank under each (all models and
    ML only). Mass stays in the main composite (dimensions fixed before the results); this is the
    sensitivity check."""
    t, _, _ = plausibility_tables()
    rows = []
    ml = t[t.combo.map(lambda k: split(k)[0] != "HBV")]
    r_all = {c: t[c].rank(ascending=False, method="min").astype(int) for c in ["composite", "composite_no_mass"]}
    r_ml = {c: ml[c].rank(ascending=False, method="min").astype(int) for c in ["composite", "composite_no_mass"]}
    for m, s, k in combo_rows():
        i = t.index[t.combo == k][0]
        rows.append({"Model": m, "Strategy": s,
                     "Composite (mass, drought, flood)": f2(t.loc[i, "composite"]),
                     "Rank, all": r_all["composite"][i], "Rank, ML only": r_ml["composite"].get(i, ""),
                     "Composite without mass (drought, flood)": f2(t.loc[i, "composite_no_mass"]),
                     "Rank without mass, all": r_all["composite_no_mass"][i],
                     "Rank without mass, ML only": r_ml["composite_no_mass"].get(i, "")})
    save_table(pd.DataFrame(rows), "scsv04_composite_no_mass")


JOBS = {f.__name__: f for f in [fig06, s04, scsv04]}

if __name__ == "__main__":
    only = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in JOBS.items():
        if not only or name in only:
            fn()
            print(name, "done", flush=True)
