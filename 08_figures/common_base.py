"""Shared plotting helpers for the paper figures: model/strategy colours and line styles, ordering of
model-strategy keys, box/CDF helpers, drought-k and flood-flag helpers, plausibility ranking rules."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 08_figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from config import experiment as E  # noqa: E402
from core import runio  # noqa: E402
from core.io import load_eval_set, load_forcing  # noqa: E402
from core.registry import combos, runs_of  # noqa: E402
from config import plot_style as vs  # noqa: E402

FIG = E.FIG_DIR / "test_fig"   # only used by fig05(heatmap=True); paper figures go to FIG_DIR

# Colour slots stay fixed (LSTM red, TCN blue, XGB teal, RF amber -- same as common_shap.py); display order is
# separate: HBV first, then RF, XGB, TCN, LSTM.
vs.set_models(["LSTM", "TCN", "XGB", "RF"])
MODEL_ORDER = ["HBV", "RF", "XGB", "TCN", "LSTM"]
ML = ["RF", "XGB", "TCN", "LSTM"]
STRAT_LS = {"individual": ":", "regional": "-", "adapted": "--"}
STRAT_MK = {"individual": "o", "regional": "s", "adapted": "^"}
STRAT_SHORT = {"individual": "ind", "regional": "reg", "adapted": "ada"}
HBV_LS = (0, (1.2, 1.2))
ctx = vs.apply_style()


def col(m):
    return vs.OBS if m == "HBV" else vs.model_color(m)


def split(key):
    return key.split("-", 1)


def short(key):
    m, s = split(key)
    return "HBV" if m == "HBV" else f"{m}-{STRAT_SHORT[s]}"


def ordered(keys):
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    srank = {s: i for i, s in enumerate(STRAT_LS)}
    return sorted(keys, key=lambda k: (rank[split(k)[0]], srank[split(k)[1]]))


def ref_keys(keys):
    """HBV + the regional ML models (the per-model comparison set used for line plots)."""
    return ordered([k for k in keys if split(k)[0] == "HBV" or split(k)[1] == "regional"])


def save(fig, stem, data=None, data_dir=None, outdir=None, formats=vs.PNG):
    """Save a figure (PNG only by default) to `outdir` (default FIG); `data` is written as the companion
    CSV to `data_dir`."""
    outdir = Path(outdir or FIG)
    if "pdf" in formats:                     # PDFs go to <outdir>/pdf, PNGs stay directly in <outdir>
        vs.savefig(fig, stem, outdir / "pdf", formats=("pdf",))
    vs.savefig(fig, stem, outdir, formats=tuple(f for f in formats if f != "pdf"))
    plt.close(fig)
    if data is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(data_dir / f"{stem}.csv", index=False)


def ecdf(x, grid):
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    return np.searchsorted(x, grid, side="right") / len(x)


def boxes(ax, data, keys, positions=None, widths=0.6, alpha=0.3):
    bp = ax.boxplot(data, positions=positions, widths=widths, patch_artist=True, showfliers=False,
                    showmeans=True,
                    meanprops=dict(marker="D", mfc="white", mec="black", ms=ctx["ms"] * 0.8, mew=ctx["lw_axes"]),
                    medianprops=dict(color="black", lw=ctx["lw"]),
                    whiskerprops=dict(lw=ctx["lw_axes"]), capprops=dict(lw=ctx["lw_axes"]))
    for b, k in zip(bp["boxes"], keys):
        c = col(split(k)[0])
        b.set(facecolor=(*to_rgb(c), alpha * (0.6 if split(k)[0] == "HBV" else 1)), edgecolor=c, lw=ctx["lw"])
    return bp


def combo_xticks(ax, keys, pos=None):
    pos = np.arange(len(keys)) if pos is None else pos
    ax.set_xticks(pos, [short(k) for k in keys], rotation=45, ha="right", rotation_mode="anchor")
    ax.grid(axis="x", visible=False)


def model_handles(models=MODEL_ORDER, line=True):
    if line:
        return [Line2D([], [], color=col(m), lw=ctx["lw"] * 1.3, ls=HBV_LS if m == "HBV" else "-") for m in models]
    return [Patch(facecolor=col(m), edgecolor=col(m)) for m in models]


# ------------------------------------------------------------------------------------------
# Drought. k (recession constant, days) = e-folding time of flow with no rain: ln Q = c - t / k,
# fitted on days 2-30 after the last rain (observed: dry spells 1987-2000; synthetic: the zero-
# precipitation run from 2001-01-01). Larger k = slower recession.
# Runs with no fittable recession are kept at the clip edges (scoring.synthetic_k): drained (flow
# already at the 1e-3 mm/d floor) -> K_MIN, flat (no recession) -> K_CAP.
K_CAP = 1e4     # = scoring.K_CLIP[1], "flat"
K_MIN = 0.1     # = scoring.K_CLIP[0], "drained"
Q_FLOOR = 1e-3


def d1_factor():
    """D1 tolerance as a factor on k (observation-calibrated, scoring.py -> drought_thresholds.json)."""
    import json
    return float(json.loads((E.ANALYSIS_DROUGHT_DIR / "drought_thresholds.json").read_text())["factor_D1"])


def d1_band(ax, lo=1, hi=K_CAP):
    """Grey band = D1 pass region: k_syn within the calibrated factor of k_obs."""
    f = d1_factor()
    x = np.geomspace(lo, hi, 60)
    ax.fill_between(x, x / f, x * f, color="0.88", lw=0, zorder=0)


def drought_k_table():
    d = pd.read_csv(E.ANALYSIS_DROUGHT_DIR / "drought_per_basin.csv", dtype={"basin": str})
    med = d.groupby(["combo", "basin"])[["k_syn", "k_obs", "frac_decreasing_365d", "meanQ_2002_2006"]] \
        .median().reset_index()
    med["log_ratio"] = np.log(med.k_syn / med.k_obs)
    if "k_class" in d.columns:
        # drained -> K_CLIP[0], flat -> K_CLIP[1], so the median over seeds
        # is ordinal and its class follows from which edge (if any) it sits on
        med["k_class"] = np.where(med.k_syn <= K_MIN, "drained", np.where(med.k_syn >= K_CAP, "flat", "fitted"))
    else:  # drought_per_basin.csv without k_class: drained cases clipped to K_CAP as well
        capped = med.k_syn >= K_CAP
        med["k_class"] = np.where(~capped, "fitted", np.where(med.meanQ_2002_2006 < Q_FLOOR, "drained", "flat"))
    return d, med


# ------------------------------------------------------------------------------------------
DUR_ALPHA = {1: 0.15, 3: 0.45, 5: 0.8}


def _flood_example(key, b, seed, d):
    """ΔQ (storm run − no-storm run) for every uplift, the storm rainfall total, days since storm."""
    run = [r for r in runs_of(key) if r.seed == seed] or runs_of(key)
    run = run[0]
    win = pd.date_range(E.STORM_START, E.FLOOD_WINDOW[1])
    base = runio.load_sim(run.dir, E.NOSTORM)[b].loc[win]
    fb = load_forcing(b)
    src_peak = float(fb.loc[E.SOURCE_SEARCH[0]:E.SOURCE_SEARCH[1], "Precip"].max())
    out = {}
    for u in E.STORM_UPLIFTS:
        dq = (runio.load_sim(run.dir, f"peak{d}_rate{u}")[b].loc[win] - base).to_numpy()
        out[u] = (dq, d * src_peak * (1 + u / 100))
    return out


def flag_example_picks(f):
    """(flag, combo, basin, seed) of a typical (median-severity) failing 3-day case per flag, on a
    basin where HBV passes every flag."""
    hbv = f[f.combo == "HBV-individual"]
    ok_hbv = set(hbv[hbv.monotonic & hbv.under_cap & ~hbv.incomplete & (hbv.duration == 3)].basin)
    picks = []
    for flag, combo, cond, sort in [
            ("monotonic", "XGB-regional", lambda g: ~g.monotonic, "peak_dQ_rate0"),
            ("over_cap", "LSTM-regional", lambda g: ~g.under_cap, "max_runoff_coef")]:
        g = f[(f.combo == combo) & (f.duration == 3)]
        g = g[cond(g) & g.basin.isin(ok_hbv) & (g.peak_dQ_rate0 > 0.5)]
        if not len(g):
            g = f[(f.combo == combo) & (f.duration == 3)]
            g = g[cond(g)]
        r = g.sort_values(sort).iloc[len(g) // 2]           # a typical (median-severity) failing case
        picks.append((flag, combo, r.basin, int(r.seed)))
    return picks


def draw_flag_example(ax, flag, key, b, seed, xlabel=True, ylabel=True):
    """ΔQ response of `key` on basin b (3-day storm, every uplift), drawn the way that shows `flag`."""
    ups = list(E.STORM_UPLIFTS)
    m = split(key)[0]
    sh = vs.shades("#444444" if m == "HBV" else col(m), len(ups))[::-1]   # darkest = largest uplift
    ex = _flood_example(key, b, seed, 3)
    t = np.arange(len(next(iter(ex.values()))[0]))
    rows = []
    for u, c in zip(ups, sh):
        dq, ptot = ex[u]
        if flag == "monotonic":
            ax.plot(t[:25], dq[:25], color=c, lw=ctx["lw"], label=f"+{u} %")
            ax.plot(np.argmax(dq), dq.max(), "o", color=c, ms=ctx["ms"])
        elif flag == "over_cap":
            ax.plot(t, np.cumsum(dq) / ptot, color=c, lw=ctx["lw"], label=f"+{u} %")
        else:
            ax.plot(t, np.abs(dq), color=c, lw=ctx["lw"], label=f"+{u} %")
        rows.append({"flag": flag, "combo": key, "basin": b, "uplift": u, "peak_dQ": dq.max(),
                     "runoff_coef": dq.sum() / ptot, "tail_over_peak": dq[-7:].mean() / dq.max()})
    if flag == "monotonic":
        ax.set(ylabel="ΔQ [mm d$^{-1}$]" if ylabel else "")
        pk = ", ".join(f"{ex[u][0].max():.1f}" for u in ups)
        ax.text(0.97, 0.80, f"peaks: {pk}", transform=ax.transAxes, ha="right", va="top",
                fontsize=ctx["annot"], color="0.3")
    elif flag == "over_cap":
        ax.axhline(1, color=vs.OBS, ls="--", lw=ctx["lw_axes"])
        ax.set(ylabel="ΣΔQ / storm P")
    else:
        dq50 = ex[ups[-1]][0]
        ax.set_yscale("log")
        ax.set_ylim(1e-4, dq50.max() * 3)
        ax.axhline(0.01 * dq50.max(), color=vs.OBS, ls="--", lw=ctx["lw_axes"])
        ax.axvspan(len(t) - 7, len(t), color="0.85", lw=0)
        ax.set(ylabel="|ΔQ| [mm d$^{-1}$]")
    if xlabel:
        ax.set_xlabel("Days since storm start")
    y_lab = 0.05 if flag == "over_cap" else 0.95
    ax.text(0.97, y_lab, f"{short(key)}, {b}", transform=ax.transAxes, ha="right",
            va="bottom" if flag == "over_cap" else "top", fontsize=ctx["annot"])
    return rows


# ------------------------------------------------------------------------------------------
DIM_DIR = {"mass": "high", "drought": "high", "flood": "high"}   # all three are pass fractions (scoring.py composite)
TIE = "tie"
TIE_COLOR = "0.78"


def winner(score, tol=1e-9):
    """Best candidate, or TIE when two or more share the top score (e.g. every ML combo at drought = 0)."""
    top = score.max()
    return TIE if (score >= top - tol).sum() > 1 else score.idxmax()


def tie_handle(ms):
    return Line2D([], [], color=TIE_COLOR, marker="s", lw=0, ms=ms)


RESOLUTION_MIN = 0.05   # PLAUSIBILITY_CHECKS.md resolution rule


def comparison_raw(t, cand):
    """Dimension values for one ranking comparison. Resolution rule (any dimension, any comparison): if
    every candidate's strict (pass-all-checks) score is below RESOLUTION_MIN, the comparison uses the
    dimension's check-averaged score. Returns (values, switched dimensions)."""
    ti = t.set_index("combo").loc[cand]
    raw, switched = pd.DataFrame(index=ti.index), []
    for d in DIM_DIR:
        if (ti[f"{d}_raw"] < RESOLUTION_MIN).all():
            raw[d] = ti[f"{d}_avgchecks"]
            switched.append(d)
        else:
            raw[d] = ti[f"{d}_raw"]
    return raw, switched


def weight_grid(step=0.05):
    n = int(round(1 / step))
    return np.array([(i * step, j * step, (n - i - j) * step) for i in range(n + 1) for j in range(n + 1 - i)])


def score_wsm(raw, w):
    """Weighted sum of the dimension pass fractions (scoring.py's composite at equal weights).
    The fractions share one 0-1 scale, so they are not renormalised within the candidate set."""
    return (raw[list(DIM_DIR)] * w).sum(axis=1)


def score_wpm(raw, w):
    """Weighted product model (Bridgman 1922; Triantaphyllou 2000): prod_i r_i^w_i on ratio-to-best
    normalised criteria -- multiplicative, so a combo that is poor on any one criterion cannot be
    rescued by the others (unlike the weighted arithmetic mean)."""
    r = pd.DataFrame(index=raw.index)
    for dim, dr in DIM_DIR.items():
        x = raw[dim].clip(lower=1e-9)
        r[dim] = x / x.max() if dr == "high" else x.min() / x
    return np.exp((np.log(r) * w).sum(axis=1))


def score_topsis(raw, w):
    """TOPSIS (Hwang & Yoon 1981): relative closeness to the ideal solution on vector-normalised,
    weighted criteria."""
    v = (raw / np.sqrt((raw ** 2).sum()).replace(0, np.nan)).fillna(0) * w   # all-zero criterion: no say
    best = pd.Series({d: (v[d].max() if dr == "high" else v[d].min()) for d, dr in DIM_DIR.items()})
    worst = pd.Series({d: (v[d].min() if dr == "high" else v[d].max()) for d, dr in DIM_DIR.items()})
    dp = np.sqrt(((v - best) ** 2).sum(axis=1))
    dm = np.sqrt(((v - worst) ** 2).sum(axis=1))
    return (dm / (dp + dm).replace(0, np.nan)).fillna(0.5)   # all tied (e.g. only an all-zero criterion weighted)


def label_points(ax, xs, ys, labels, ylo=None, yhi=None, xlo=None, xhi=None):
    """Greedy label placement: try right, left, then above/below on either side; keep the first
    candidate whose box overlaps neither earlier labels nor any marker / error bar."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    obst = [np.column_stack([xs, ys])]
    if ylo is not None:
        for x, a, b in zip(xs, ylo, yhi):
            yy = np.linspace(a, b, 120)
            obst.append(np.column_stack([np.full_like(yy, x), yy]))
    if xlo is not None:
        for y, a, b in zip(ys, xlo, xhi):
            xx = np.linspace(a, b, 120)
            obst.append(np.column_stack([xx, np.full_like(xx, y)]))
    pts = ax.transData.transform(np.vstack(obst))
    placed = []
    cands = [(6, 0, "left", "center"), (-6, 0, "right", "center"), (5, 6, "left", "bottom"),
             (-5, 6, "right", "bottom"), (5, -6, "left", "top"), (-5, -6, "right", "top")]
    # farther-out fallbacks (with a thin leader line) for crowded clusters
    far = [(dx, dy, ha, va) for dy in (14, -14, 22, -22) for dx, ha in ((8, "left"), (-8, "right"))
           for va in ["bottom" if dy > 0 else "top"]]
    lo_ = ys if ylo is None else ylo
    hi_ = ys if yhi is None else yhi
    xlo_ = xs if xlo is None else xlo
    xhi_ = xs if xhi is None else xhi
    for x, y, lab, ya, yb, xa, xb in zip(xs, ys, labels, lo_, hi_, xlo_, xhi_):
        best = None
        # marker-anchored candidates, then beside the error-bar caps, then farther out with a leader
        opts = [((x, y), c, False) for c in cands]
        if ylo is not None:
            opts += [((x, yb), (5, 0, "left", "center"), False), ((x, ya), (5, 0, "left", "center"), False),
                     ((x, yb), (-5, 0, "right", "center"), False), ((x, ya), (-5, 0, "right", "center"), False)]
        if xlo is not None:   # just beyond the ends of the horizontal error bar
            opts += [((xb, y), (5, 0, "left", "center"), False), ((xa, y), (-5, 0, "right", "center"), False)]
        opts += [((x, y), c, True) for c in far]
        for anchor, (dx, dy, ha, va), lead in opts:
            kw = dict(arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4", shrinkA=0, shrinkB=3)) if lead else {}
            t = ax.annotate(lab, anchor, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=ctx["legend"], **kw)
            bb = t.get_window_extent(rend).expanded(1.05, 1.15)
            axbb = ax.get_window_extent(rend)
            outside = bb.x0 < axbb.x0 or bb.x1 > axbb.x1 or bb.y0 < axbb.y0 or bb.y1 > axbb.y1
            n_hit = (sum(bb.overlaps(p) for p in placed) * 100 + sum(bb.contains(px, py) for px, py in pts)
                     + 1000 * outside)
            if best is None or n_hit < best[0]:
                if best is not None:
                    best[1].remove()
                best = (n_hit, t, bb)
            else:
                t.remove()
            if n_hit == 0:
                break
        placed.append(best[2])


def strat_fill(m, st):
    """Marker face by strategy (shape already differs): individual hollow, adapted light, regional/HBV solid."""
    c = np.array(to_rgb(col(m)))
    if m == "HBV" or st == "regional":
        return tuple(c)
    return (1, 1, 1) if st == "individual" else tuple(c + (1 - c) * 0.55)


def strat_handle(k, ms):
    m, st = split(k)
    return Line2D([], [], color=col(m), marker=STRAT_MK[st], mfc=strat_fill(m, st), mec=col(m), mew=0.9, lw=0, ms=ms)


def ternary(ax, wts, best, order):
    xx = wts[:, 1] + 0.5 * wts[:, 2]
    yy = np.sqrt(3) / 2 * wts[:, 2]
    tie = best == TIE
    if tie.any():   # tied weightings: no winner drawn
        ax.scatter(xx[tie], yy[tie], s=13, marker="s", facecolor=TIE_COLOR, edgecolor="none")
    for k in order:
        sel = best == k
        if sel.any():
            m, st = split(k)
            ax.scatter(xx[sel], yy[sel], s=13, marker=STRAT_MK[st], facecolor=strat_fill(m, st),
                       edgecolor=col(m), lw=0.7)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color=vs.OBS, lw=ctx["lw_axes"])
    for (px, py, lab, va) in [(0, -0.04, "mass", "top"), (1, -0.04, "drought", "top"),
                              (0.5, np.sqrt(3) / 2 + 0.04, "flood", "bottom")]:
        ax.text(px, py, lab, ha="center", va=va, fontsize=ctx["legend"])
    ax.scatter([0.5], [np.sqrt(3) / 6], marker="x", color="white", s=40, lw=2.4, zorder=5)
    ax.scatter([0.5], [np.sqrt(3) / 6], marker="x", color=vs.OBS, s=34, lw=1.2, zorder=6)
    ax.set_aspect("equal")
    ax.set(xlim=(-0.08, 1.08), ylim=(-0.12, np.sqrt(3) / 2 + 0.12))
    ax.axis("off")
