"""Ranking helpers for fig06/s04: bootstrap of the dimension scores, tie sets under the three ranking
rules (weighted sum, weighted product, TOPSIS) and the weight-triangle panels."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 08_figures
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "07_analysis"))
import common as P  # noqa: E402
from common import T, vs, ctx, col, split, short, STRAT_MK  # noqa: E402
import scoring as S  # noqa: E402
from core.io import load_eval_set  # noqa: E402
from core.registry import combos  # noqa: E402

DIMS = ["mass", "drought", "flood"]
SQ3 = np.sqrt(3) / 2
MIXED = "0.35"          # mixed-family tie: dark-grey "+" (not a strategy marker, distinct from the family diamonds)
FAM_TINT = 0.5
Z_CAP = 3.0


def tint(c, a):
    """Blend colour c with white; a = 1 -> c, a = 0 -> white."""
    return tuple(np.array(to_rgb(c)) * a + (1 - a))


# ------------------------------------------------------------------------------------------
# bootstrap samples, rebuilt exactly as scoring.composite drew them
# ------------------------------------------------------------------------------------------
_BOOT = None


def bootstrap():
    """Cached: several figures use it in one run."""
    global _BOOT
    if _BOOT is None:
        _BOOT = _bootstrap()
    return _BOOT


def _bootstrap():
    basins = load_eval_set()
    rd = lambda p: pd.read_csv(p, dtype={"basin": str})
    tables = {"mass": rd(P.E.ANALYSIS_MASS_DIR / "mass_per_basin.csv"),
              "drought": rd(P.E.ANALYSIS_DROUGHT_DIR / "drought_per_basin.csv"),
              "flood": rd(P.E.ANALYSIS_FLOOD_DIR / "flood_per_basin.csv")}
    nb = len(basins)
    rng = np.random.default_rng(42)                     # = scoring.RNG, used only for these draws
    Wb = np.stack([np.bincount(rng.integers(nb, size=nb), minlength=nb) for _ in range(S.N_BOOT)]).astype(float)
    items = DIMS + [f"{d}_avgchecks" for d in DIMS]
    return {k: {i: v for i, v in S.scores(S.basin_counts(tables, basins, k), Wb).items() if i in items}
            for k in combos()}


def check_reproduction(boot, t):
    ti = t.set_index("combo")
    err = max(abs(np.quantile(boot[k][d], q) - ti.loc[k, f"{d}_raw_{tag}"])
              for k in boot for d in DIMS for q, tag in [(.05, "p05"), (.95, "p95")])
    print(f"bootstrap reproduction: max |p05/p95 difference| vs composite_table = {err:.2e}")
    assert err < 1e-9, "rebuilt bootstrap differs from scoring.py's"


# ------------------------------------------------------------------------------------------
# ranking rules on arrays: X (..., n_cand, 3) -> scores (..., n_cand); same as T.score_*
# ------------------------------------------------------------------------------------------
def wsm(X, w):
    return X @ w


def wpm(X, w):
    x = np.clip(X, 1e-9, None)
    return np.exp((np.log(x / x.max(axis=-2, keepdims=True)) * w).sum(-1))


def topsis(X, w):
    nrm = np.sqrt((X ** 2).sum(axis=-2, keepdims=True))
    v = np.divide(X, nrm, out=np.zeros_like(X), where=nrm > 0) * w
    dp = np.sqrt(((v - v.max(axis=-2, keepdims=True)) ** 2).sum(-1))
    dm = np.sqrt(((v - v.min(axis=-2, keepdims=True)) ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = dm / (dp + dm)
    return np.where(np.isfinite(c), c, 0.5)


RULES = {"Weighted sum": wsm, "Weighted product": wpm, "TOPSIS": topsis}


def comparison_arrays(t, boot, cand):
    """Point (n_cand, 3) and bootstrap (n_rep, n_cand, 3) values; resolution rule decided on the point values."""
    raw, switched = T.comparison_raw(t, cand)
    item = {d: f"{d}_avgchecks" if d in switched else d for d in DIMS}
    Xb = np.stack([np.stack([boot[k][item[d]] for d in DIMS], -1) for k in cand], 1)
    return raw[DIMS].to_numpy(), Xb, switched


def tied_sets(X, Xb, rule, wts):
    """Per weighting: top model, tied set (90 % interval of the gap to the top includes 0), and the
    certainty of the top over the runner-up, z = gap / bootstrap SD of the gap."""
    rows = []
    for w in wts:
        s, sb = RULES[rule](X, w), RULES[rule](Xb, w)
        top = int(np.argmax(s))
        gaps = sb[:, [top]] - sb
        q05 = np.quantile(gaps, 0.05, axis=0)
        tied = sorted({top} | {j for j in range(len(s)) if q05[j] <= 0})
        second = int(np.argsort(-s)[1])
        sd = gaps[:, second].std()
        rows.append({"top": top, "tied": tied, "z": (s[top] - s[second]) / sd if sd > 0 else np.inf})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------
# drawing
# ------------------------------------------------------------------------------------------
def tri_xy(wts):
    return wts[:, 1] + 0.5 * wts[:, 2], SQ3 * wts[:, 2]


def tri_frame(ax, labels=True):
    ax.plot([0, 1, 0.5, 0], [0, 0, SQ3, 0], color=vs.OBS, lw=ctx["lw_axes"], zorder=4)
    if labels:
        for px, py, lab, va in [(0, -0.04, "mass", "top"), (1, -0.04, "drought", "top"),
                                (0.5, SQ3 + 0.04, "flood", "bottom")]:
            ax.text(px, py, lab, ha="center", va=va, fontsize=ctx["legend"])
    ax.scatter([0.5], [SQ3 / 3], marker="x", color="white", s=40, lw=2.4, zorder=5)
    ax.scatter([0.5], [SQ3 / 3], marker="x", color=vs.OBS, s=34, lw=1.2, zorder=6)
    ax.set_aspect("equal")
    ax.set(xlim=(-0.08, 1.08), ylim=(-0.12, SQ3 + 0.12))
    ax.axis("off")


def family_triangle(ax, cand, ts, wts, size=1.0):
    """Colour each weighting by its tied set (marker area scaled by `size`). Returns what was drawn."""
    xx, yy = tri_xy(wts)
    seen = {"single": set(), "family": set(), "mixed": False}
    for i in range(len(wts)):
        r = ts.iloc[i]
        tied = [cand[j] for j in r.tied]
        fams = {split(k)[0] for k in tied}
        if len(tied) == 1:
            m, st = split(tied[0])
            # lightness over the distinguishable range only: z = 1.64 (just distinguishable) -> pale,
            # z >= Z_CAP -> full colour, so the shading is visible
            a = 0.15 + 0.85 * float(np.clip((r.z - 1.645) / (Z_CAP - 1.645), 0, 1))
            ax.scatter(xx[i], yy[i], s=14 * size, marker=STRAT_MK[st], facecolor=tint(col(m), a), edgecolor=col(m), lw=0.5)
            seen["single"].add(tied[0])
        elif len(fams) == 1:
            f = fams.pop()
            ax.scatter(xx[i], yy[i], s=13 * size, marker="D", facecolor=tint(col(f), FAM_TINT), edgecolor="none")
            seen["family"].add(f)
        else:
            ax.scatter(xx[i], yy[i], s=16 * size, marker="+", color=MIXED, lw=1.0)
            seen["mixed"] = True
    tri_frame(ax)
    return seen


def triangle_legend(seens, order):
    singles = [k for k in order if any(k in s["single"] for s in seens)]
    fams = [f for f in T.MODEL_ORDER if any(f in s["family"] for s in seens)]
    # filled like the drawn points (fill shows certainty here, not the strategy)
    h = [Line2D([], [], marker=STRAT_MK[split(k)[1]], lw=0, color=col(split(k)[0]), ms=ctx["ms"] * 1.6)
         for k in singles]
    lab = [short(k) for k in singles]
    h += [Line2D([], [], marker="D", lw=0, color=tint(col(f), FAM_TINT), ms=ctx["ms"] * 1.4) for f in fams]
    lab += [f"{f} variants tied" for f in fams]
    if any(s["mixed"] for s in seens):
        h.append(Line2D([], [], marker="+", lw=0, color=MIXED, mew=1.0, ms=ctx["ms"] * 1.6))
        lab.append("mixed tie")
    h.append(Line2D([], [], color=vs.OBS, marker="x", lw=0, ms=ctx["ms"] * 1.6))
    lab.append("equal weights")
    return h, lab


def dot_plot(axs, t, boot, keys):
    """One row per model, one panel per dimension. Every dimension is a dumbbell from the strict score
    (strategy marker, all checks passed) to the check-averaged score (tick, mean of the check pass rates);
    90 % intervals on both ends. A short dumbbell = failures concentrated in the same basins / cases."""
    y = np.arange(len(keys))[::-1]
    ti = t.set_index("combo")
    for a, d in zip(axs, DIMS):
        for yi, k in zip(y, keys):
            m, st = split(k)
            v = float(ti.loc[k, f"{d}_raw"])
            lo, hi = np.quantile(boot[k][d], [.05, .95])
            va = float(ti.loc[k, f"{d}_avgchecks"])
            alo, ahi = np.quantile(boot[k][f"{d}_avgchecks"], [.05, .95])
            a.plot([v, va], [yi, yi], color=col(m), lw=ctx["lw"] * 0.8, zorder=2)
            a.errorbar(va, yi, xerr=[[va - alo], [ahi - va]], fmt="|", color=col(m), ms=ctx["ms"] * 2.2,
                       mew=1.4, lw=ctx["lw"], capsize=1.5, zorder=3)
            a.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt=STRAT_MK[st], color=col(m),
                       mfc=T.strat_fill(m, st), mec=col(m), ms=ctx["ms"] * 1.5, lw=ctx["lw"], capsize=1.5, zorder=4)
        a.set(xlim=(-0.04, 1.04), ylim=(-0.7, len(keys) - 0.3))
        a.grid(axis="y", visible=False)
        a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    axs[0].set_yticks(y, [short(k) for k in keys])
    for a in axs[1:]:
        a.tick_params(labelleft=False)
    axs[1].set_xlabel("Pass fraction")


# Fig 6a point labels, placed by hand: anchor ("pt" = marker, "xhi" = right end of the KGE bar),
# offset in points, alignment. Same-model variants sit on the side of their own marker (TCN-reg square
# left of TCN-ada triangle); RF-ind, XGB-ind, LSTM-ind read left to right in KGE order.
LABEL_POS = {
    "HBV":      ("pt", 6, -7, "left", "top"),
    "TCN-ind":  ("pt", 0, 8, "center", "bottom"),
    "TCN-reg":  ("pt", -3, 8, "right", "bottom"),
    "TCN-ada":  ("pt", 3, 8, "left", "bottom"),
    "RF-reg":   ("pt", 0, 8, "center", "bottom"),
    "LSTM-reg": ("pt", -8, 8, "center", "bottom"),
    "LSTM-ada": ("pt", 10, 8, "center", "bottom"),
    "XGB-reg":  ("pt", 0, -9, "center", "top"),
    "RF-ind":   ("pt", -2, 8, "center", "bottom"),
    "XGB-ind":  ("pt", 2, 5, "center", "bottom"),
    "LSTM-ind": ("xhi", 4, 0, "left", "center"),
}


def accuracy_labels(ax, t, xhi):
    for i, k in enumerate(t.combo):
        lab = short(k)
        anchor, dx, dy, ha, va = LABEL_POS[lab]
        x = t.test_KGE_median_ens.iloc[i] if anchor == "pt" else xhi[i]
        ax.annotate(lab, (x, t.composite.iloc[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=ctx["legend"])


def fig06_figure(t, boot, main):
    """Build Fig 6; the caller saves it."""
    keys = list(main)
    ml = [k for k in keys if split(k)[0] != "HBV"]
    wts = T.weight_grid()
    # left column: (a) on top, (b) below (b compressed more than a); right column: triangles (c) above (d)
    fig = plt.figure(figsize=vs.figsize(width="full", aspect=0.9))
    left, right = fig.subfigures(1, 2, width_ratios=[1.7, 1])
    top, mid = left.subfigures(2, 1, height_ratios=[1, 0.82])
    ax = top.subplots()
    xlo, xhi = P.accuracy_composite_panel(ax, t)
    # more room around the low cluster (LSTM / XGB / RF) so the labels clear the points and bars
    ax.set_ylim(t.composite_p05.min() - 0.05, t.composite_p95.max() + 0.02)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.03)
    dax = list(mid.subplots(1, 3, sharey=True))
    dot_plot(dax, t, boot, keys)
    # one legend for all three columns of b, under them (subfigure legend: the layout makes room for it)
    mid.legend([Line2D([], [], color="0.3", marker="s", lw=0, ms=ctx["ms"] * 1.2),
                Line2D([], [], color="0.3", marker="|", lw=0, ms=ctx["ms"] * 2, mew=1.4)],
               ["strict (all checks)", "check-averaged"], loc="outside lower center", ncol=2, frameon=False,
               fontsize=ctx["legend"], handletextpad=0.3, handlelength=1.0, columnspacing=1.0)
    for a, d in zip(dax[1:], DIMS[1:]):
        a.set_title(d.capitalize(), fontsize=ctx["legend"], loc="left")
    tax = right.subplots(2, 1)
    seens, titles = [], []
    for a, (set_name, cand) in zip(tax, [("all models", keys), ("ML only", ml)]):
        X, Xb, sw = comparison_arrays(t, boot, cand)
        ts = tied_sets(X, Xb, "Weighted sum", wts)
        seens.append(family_triangle(a, cand, ts, wts))
        note = f"\n({', '.join(sw)}: check-averaged)" if sw else ""
        titles.append(f"Weighted sum, {set_name}{note}")
        kinds = ts.apply(lambda r: "single" if len(r.tied) == 1 else
                         ("family" if len({split(cand[j])[0] for j in r.tied}) == 1 else "mixed"), axis=1)
        print(f"{set_name}: " + ", ".join(f"{k} {n}" for k, n in kinds.value_counts().items()) + f" of {len(ts)}")
    h, lab = triangle_legend(seens, keys)
    right.legend(h, lab, loc="outside lower center", ncol=2, frameon=False, fontsize=ctx["legend"],
                 handletextpad=0.3, columnspacing=0.8, title="darker = more certain",
                 title_fontsize=ctx["legend"])
    P.label([ax, dax[0]] + list(tax), ["Accuracy vs. composite plausibility (90 % CI)", "Mass"] + titles)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    accuracy_labels(ax, t, xhi)
    return fig


def draw_surfaces(container, t, main, rule):
    """ML only: each model's own score over the triangle (shared colour scale, contours), drawn as a 2 x 5
    grid into `container` (a figure or subfigure). Returns the axes."""
    ml = [k for k in main if split(k)[0] != "HBV"]
    raw, _ = T.comparison_raw(t, ml)
    X = raw[DIMS].to_numpy()
    wts = T.weight_grid(0.02)
    sc = np.stack([RULES[rule](X, w) for w in wts])          # (n_w, n_cand)
    xx, yy = tri_xy(wts)
    axs = container.subplots(2, 5)
    lev = np.linspace(0, 1, 11)
    for a, (i, k) in zip(axs.flat, enumerate(ml)):
        cf = a.tricontourf(xx, yy, sc[:, i], levels=lev, cmap="viridis", vmin=0, vmax=1)
        a.tricontour(xx, yy, sc[:, i], levels=lev, colors="white", linewidths=0.4)
        win = (sc.argmax(1) == i).astype(float)
        if win.any():   # region where this model ranks 1st: sparse hatch + full black outline (incl. triangle edges)
            with plt.rc_context({"hatch.linewidth": 0.5, "hatch.color": "k"}):
                cs = a.tricontourf(xx, yy, win, levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=5)
            cs.set_edgecolor("k")
            cs.set_linewidth(1.0)
        tri_frame(a, labels=False)
        a.set_title(short(k), fontsize=ctx["legend"], pad=1)
    container.colorbar(cf, ax=axs, shrink=0.8, label=f"{rule} score")
    return axs
