"""Every score of the study, computed here and nowhere else. Figure
scripts only read the files this module writes to ANALYSIS_DIR.

Basins: the common evaluation set (07_analysis/eval_set.py) only. Period: TEST unless stated.
Seeds: headline accuracy uses the seed-ensemble-mean hydrograph; strategy comparisons and the
plausibility dimensions use per-seed scores (median over seeds), so models with fewer seeds
(individual) are not penalised by missing ensemble smoothing.
Missing output for any run, scenario or basin raises an error naming the file.

RUN: python 07_analysis/scoring.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from core import runio  # noqa: E402
from core.io import load_camels_signatures, load_eval_set, load_forcing  # noqa: E402
from core.metrics import kge_parts  # noqa: E402
from core.registry import combos, runs_of  # noqa: E402

OUT_ACC = E.ANALYSIS_ACCURACY_DIR
OUT_MASS = E.ANALYSIS_MASS_DIR
OUT_DROUGHT = E.ANALYSIS_DROUGHT_DIR
OUT_FLOOD = E.ANALYSIS_FLOOD_DIR
OUT_SCORE = E.ANALYSIS_SCORING_DIR
N_BOOT = 1000
RNG = np.random.default_rng(42)
Q_FLOOR = 1e-3          # mm/d, floor for log(Q) in recession fits
K_CLIP = (0.1, 1e4)     # days, recession constants outside this are clipped
DRY_P = 0.1             # mm/d, a day with P <= DRY_P is dry
REC_DAYS = (2, 30)      # days since rain over which recession constants are fitted (obs and synthetic)
MIN_DRY_SPELL = 5
OBS_COVERAGE = 0.9      # observed water balance needs this fraction of test days observed


# ------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------
def require(df, basins, what):
    miss = sorted(set(basins) - set(df.columns))
    if miss:
        raise RuntimeError(f"{what}: {len(miss)} evaluation basins missing, e.g. {miss[:5]}")
    return df[basins]


def load_combo(key, scenario, basins):
    """{seed: DataFrame dates x basins} for one model-strategy and scenario."""
    out = {}
    for r in runs_of(key):
        f = r.dir / "sim" / f"{scenario}.parquet"
        out[r.seed] = require(runio.load_sim(r.dir, scenario), basins, str(f))
    return out


def ensemble(per_seed):
    return sum(per_seed.values()) / len(per_seed)


# ------------------------------------------------------------------------------------------
# accuracy
# ------------------------------------------------------------------------------------------
def basin_metrics(sim, obs, period):
    a, z = E.PERIODS[period]
    rows = {b: kge_parts(obs[b].loc[a:z], sim[b].reindex(obs.loc[a:z].index)) for b in sim.columns}
    return pd.DataFrame(rows).T


def iqr(x):
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))


def accuracy(basins, obs):
    per_basin, summary, seedtab = [], [], []
    for key in combos():
        sims = load_combo(key, "og", basins)
        ens = ensemble(sims)
        for period in E.PERIODS:
            m_ens = basin_metrics(ens, obs, period)
            seed_m = {s: basin_metrics(x, obs, period) for s, x in sims.items()}
            per_seed_kge = pd.DataFrame({s: v["KGE"] for s, v in seed_m.items()})
            per_seed_nse = pd.DataFrame({s: v["NSE"] for s, v in seed_m.items()})
            for b in basins:
                per_basin.append({"combo": key, "period": period, "basin": b, "KGE_ens": m_ens.loc[b, "KGE"],
                                  "NSE_ens": m_ens.loc[b, "NSE"], "r_ens": m_ens.loc[b, "r"],
                                  "alpha_ens": m_ens.loc[b, "alpha"], "beta_ens": m_ens.loc[b, "beta"],
                                  "KGE_perseed": per_seed_kge.loc[b].median(), "NSE_perseed": per_seed_nse.loc[b].median()})
            for s in sims:
                seedtab.append({"combo": key, "period": period, "seed": s,
                                "median_KGE": seed_m[s]["KGE"].median(), "median_NSE": seed_m[s]["NSE"].median()})
    pb = pd.DataFrame(per_basin)
    pb.to_csv(OUT_ACC / "accuracy_per_basin.csv", index=False)
    st = pd.DataFrame(seedtab)
    st.to_csv(OUT_ACC / "accuracy_per_seed.csv", index=False)

    def summarise(df, tag):
        rows = []
        for (key, period), g in df.groupby(["combo", "period"], sort=False):
            ss = st[(st.combo == key) & (st.period == period)]
            rows.append({"combo": key, "period": period, "n_basins": len(g), "n_seeds": len(ss),
                         "KGE_median_ens": g["KGE_ens"].median(), "KGE_IQR_ens": iqr(g["KGE_ens"]),
                         "NSE_median_ens": g["NSE_ens"].median(), "NSE_IQR_ens": iqr(g["NSE_ens"]),
                         "KGE_median_perseed": g["KGE_perseed"].median(), "NSE_median_perseed": g["NSE_perseed"].median(),
                         "seed_median_KGE_min": ss["median_KGE"].min(), "seed_median_KGE_max": ss["median_KGE"].max(),
                         "seed_median_KGE_std": ss["median_KGE"].std(), "seed_median_NSE_min": ss["median_NSE"].min(),
                         "seed_median_NSE_max": ss["median_NSE"].max()})
        out = pd.DataFrame(rows)
        piv = out.pivot(index="combo", columns="period", values="KGE_median_ens")
        out["cal_test_gap_KGE"] = out["combo"].map(piv["cal"] - piv["test"])
        pivn = out.pivot(index="combo", columns="period", values="NSE_median_ens")
        out["cal_test_gap_NSE"] = out["combo"].map(pivn["cal"] - pivn["test"])
        out.to_csv(OUT_ACC / f"accuracy_summary{tag}.csv", index=False)
        return out

    summ = summarise(pb, "")
    wilcoxon(pb[pb.period == "test"])
    drop = rf_drop_rule(summ)
    return pb, summ, drop


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon(test):
    for col, tag in [("KGE_perseed", "perseed"), ("KGE_ens", "ensemble")]:
        wide = test.pivot(index="basin", columns="combo", values=col)
        rows = []
        for a, b in itertools.combinations(combos(), 2):
            d = (wide[a] - wide[b]).dropna()
            res = stats.wilcoxon(d) if (d != 0).any() else None
            rows.append({"combo_a": a, "combo_b": b, "n": len(d), "median_paired_diff_KGE": d.median(),
                         "statistic": res.statistic if res else np.nan, "p": res.pvalue if res else 1.0})
        df = pd.DataFrame(rows)
        df["p_holm"] = holm(df["p"].to_numpy())
        df.to_csv(OUT_ACC / f"wilcoxon_KGE_test_{tag}.csv", index=False)


def rf_drop_rule(summ):
    """Rule fixed before results: regional RF median test KGE < RF_DROP_RULE_KGE -> Supporting Information.
    Returns the drop decision; logged into RF-regional's own run_summary.txt, not a standalone file."""
    row = summ[(summ.combo == "RF-regional") & (summ.period == "test")].iloc[0]
    kge = float(row["KGE_median_ens"])
    drop = kge < E.RF_DROP_RULE_KGE
    txt = (f"RF drop rule (plan 3.3): regional RF median test KGE (seed-ensemble mean, evaluation set) = {kge:.3f} "
           f"(per-seed: {row['KGE_median_perseed']:.3f}); threshold {E.RF_DROP_RULE_KGE}. Outcome: "
           + ("RF moves to the Supporting Information and is excluded from the main ranking." if drop
              else "RF stays in the main results and ranking."))
    for r in runs_of("RF-regional"):
        with open(r.dir / "run_summary.txt", "a") as f:
            f.write(txt + "\n")
    return bool(drop)
    print(txt)


# ------------------------------------------------------------------------------------------
# mass balance
# ------------------------------------------------------------------------------------------
def water_balance(q, P, PET, days_mask):
    """Flags over the same test days for model and observation (one period for both)."""
    qs, ps = q[days_mask].sum(), P[days_mask].sum()
    n = int(days_mask.sum())
    et = (ps - qs) / n
    m1 = bool(qs <= ps)                              # M1 no water created
    m2 = bool(et <= PET[days_mask].mean())           # M2 implied ET within the energy limit
    return {"runoff_ratio": qs / ps, "flag_runoff_gt_P": qs > ps, "implied_ET": et,
            "flag_ET_neg": et < 0, "flag_ET_gt_PET": et > PET[days_mask].mean(),
            "flag_any": not (m1 and m2), "M1": m1, "M2": m2}


def mass(basins, obs, forc):
    a, z = E.TEST
    test_days = pd.date_range(a, z)
    obs_t = obs.loc[a:z]
    rows_obs = {}
    for b in basins:
        m = obs_t[b].notna().to_numpy()
        P, PET = forc[b]["Precip"].loc[a:z].to_numpy(), forc[b]["PET"].loc[a:z].to_numpy()
        assert len(P) == len(test_days) == len(obs_t)          # identical date ranges
        if m.mean() < OBS_COVERAGE:
            rows_obs[b] = {"obs_coverage": m.mean(), "flag_any": np.nan}
            continue
        rows_obs[b] = {"obs_coverage": m.mean(), **water_balance(obs_t[b].to_numpy(), P, PET, m)}
    obs_wb = pd.DataFrame(rows_obs).T
    obs_wb.index.name = "basin"
    obs_wb.to_csv(OUT_MASS / "mass_observed_per_basin.csv")
    obs_ok = obs_wb["flag_any"].astype(float) == 0

    per_basin = []
    for key in combos():
        for seed, sim in load_combo(key, "og", basins).items():
            for b in basins:
                m = obs_t[b].notna().to_numpy()
                s = sim[b].loc[a:z].to_numpy()
                vol = (s[m].sum() - obs_t[b].to_numpy()[m].sum()) / obs_t[b].to_numpy()[m].sum()
                P, PET = forc[b]["Precip"].loc[a:z].to_numpy(), forc[b]["PET"].loc[a:z].to_numpy()
                wb = water_balance(s, P, PET, m if m.mean() >= OBS_COVERAGE else np.ones_like(m, bool))
                per_basin.append({"combo": key, "seed": seed, "basin": b, "volume_error": vol,
                                  "abs_volume_error": abs(vol), **wb, "obs_ok": bool(obs_ok.get(b, False))})
    pb = pd.DataFrame(per_basin)
    pb.to_csv(OUT_MASS / "mass_per_basin.csv", index=False)
    return pb, obs_wb




# ------------------------------------------------------------------------------------------
# drought
# ------------------------------------------------------------------------------------------
def k_from_log_slope(slope):
    k = -1.0 / slope if slope < 0 else np.inf
    return float(np.clip(k, *K_CLIP))


def dry_spells(p):
    """(start, end) index pairs of dry spells (>= MIN_DRY_SPELL days with P <= DRY_P); end exclusive."""
    dry = (p <= DRY_P).to_numpy()
    out, i, n = [], 0, len(dry)
    while i < n:
        if not dry[i]:
            i += 1
            continue
        j = i
        while j < n and dry[j]:
            j += 1
        if j - i >= MIN_DRY_SPELL:
            out.append((i, j))
        i = j
    return out


def pooled_k(qv, spells):
    """ln Q = c_spell - d / k over days REC_DAYS after the last rain, spell-specific intercepts.
    Returns (k, number of spells that entered the fit)."""
    xs, ys = [], []
    for i, j in spells:
        d = np.arange(1, j - i + 1)            # day 1 = first dry day after rain
        sel = (d >= REC_DAYS[0]) & (d <= REC_DAYS[1])
        seg, dd = qv[i:j][sel], d[sel]
        ok = np.isfinite(seg) & (seg > 0)
        if ok.sum() >= 3:
            ly = np.log(seg[ok])
            xs.append(dd[ok] - dd[ok].mean())
            ys.append(ly - ly.mean())
    if not xs:
        return np.nan, 0
    x, y = np.concatenate(xs), np.concatenate(ys)
    return k_from_log_slope(np.sum(x * y) / np.sum(x * x)), len(xs)


def observed_k(q, p):
    """Recession constant from all observed dry spells (all seasons)."""
    return pooled_k(q.to_numpy(), dry_spells(p))


# --- D1 / D2 calibration on observations only (rules: PLAUSIBILITY_CHECKS.md)
WINTER_MONTHS = (11, 12, 1, 2, 3)   # match the 1 January start of the zero-rain run
MIN_WINTER_SPELLS = 5               # else fall back to all-season spells
MIN_HALF_SPELLS = 2                 # per odd / even half, for the split-half error
CAL_QUANTILE = 0.90                 # thresholds = 90th percentile of the observed statistic
D2_DAYS = 90
SNOW_T_D2 = 0.0                     # deg C; rises on warmer days are not counted in snowy basins
SNOWCOVER_MIN_SNOWFALL = 5.0        # mm of P on days <= 0 deg C, 1 Nov - 31 Dec before the drought run


def rise_ratio(dq, exempt=None):
    """sum of rises / sum of falls; rises on exempt days are not counted. 0 if no rise, inf if no fall."""
    dq = np.asarray(dq, float)
    ok = np.isfinite(dq)
    up = np.where(ok & (dq > 0) & (~exempt if exempt is not None else True), dq, 0).sum()
    down = np.where(ok & (dq < 0), -dq, 0).sum()
    if up == 0:
        return 0.0
    return float(up / down) if down > 0 else np.inf


def estimable(k):
    """A pooled k at a K_CLIP edge means no resolvable recession (flat / rising, e.g. frozen-river or
    snowmelt winter flow), not a large or small timescale: treat it as not estimable."""
    return bool(np.isfinite(k) and K_CLIP[0] < k < K_CLIP[1])


def observed_drought_reference(q, p, t):
    """Per-basin observed quantities for D1 / D2 (1987-2000 record)."""
    qv, tv = q.to_numpy(), t.to_numpy()
    spells = dry_spells(p)
    starts = q.index[[i for i, _ in spells]]
    winter = [s for s, d0 in zip(spells, starts) if d0.month in WINTER_MONTHS]
    k_w, n_w = pooled_k(qv, winter) if len(winter) >= MIN_WINTER_SPELLS else (np.nan, 0)
    if estimable(k_w):
        use, season, k_d1, n_d1 = winter, "winter", k_w, n_w
    else:
        k_a, n_a = pooled_k(qv, spells)
        use, season, k_d1, n_d1 = spells, "all", (k_a if estimable(k_a) else np.nan), n_a
    yrs = q.index[[i for i, _ in use]].year
    odd = [s for s, y in zip(use, yrs) if y % 2 == 1]
    even = [s for s, y in zip(use, yrs) if y % 2 == 0]
    split_err = np.nan
    if len(odd) >= MIN_HALF_SPELLS and len(even) >= MIN_HALF_SPELLS:
        ko, no = pooled_k(qv, odd)
        ke, ne = pooled_k(qv, even)
        if no >= MIN_HALF_SPELLS and ne >= MIN_HALF_SPELLS and estimable(ko) and estimable(ke):
            split_err = abs(np.log(ko / ke))
    # D2: rise ratio pooled over the first D2_DAYS days of every dry spell (all seasons)
    snowy = float(np.nanmean(tv[q.index.month == 12])) <= SNOW_T_D2
    up = down = 0.0
    for i, j in spells:
        seg = qv[i:min(j, i + D2_DAYS)]
        dq = np.diff(seg)
        ex = (tv[i + 1:i + len(seg)] > SNOW_T_D2) if snowy else np.zeros(len(dq), bool)
        ok = np.isfinite(dq)
        up += np.where(ok & (dq > 0) & ~ex, dq, 0).sum()
        down += np.where(ok & (dq < 0), -dq, 0).sum()
    r_obs = 0.0 if up == 0 else (up / down if down > 0 else np.inf)
    return {"k_obs_D1": k_d1, "k_obs_D1_season": season, "n_spells_D1": n_d1, "n_winter_spells": len(winter),
            "k_split_half_err": split_err, "rise_ratio_obs": r_obs, "snowy_dec_climate": snowy}


def snow_cover_start(p, t):
    """Snow cover on the first day of the drought run (proxy): >= SNOWCOVER_MIN_SNOWFALL mm snowfall over
    1 Nov - 31 Dec before it, and mean December temperature <= 0 deg C."""
    y = E.DROUGHT[0].year - 1
    nd = slice(f"{y}-11-01", f"{y}-12-31")
    snowfall = float(p.loc[nd][t.loc[nd] <= SNOW_T_D2].sum())
    return bool(snowfall >= SNOWCOVER_MIN_SNOWFALL and float(t.loc[f"{y}-12"].mean()) <= SNOW_T_D2)


def synthetic_k(q):
    """Same days-since-rain range on the drought run: P = 0 from TEST[0], so day 1 = TEST[0].

    Returns (k, class). Cases with no fittable recession are kept in the score at the matching edge
    of K_CLIP rather than dropped, so a model's worst basins still count:
      drained: flow already at ~0 in the fit window (median Q <= Q_FLOOR), i.e. recession too fast
               to resolve -> shortest timescale, K_CLIP[0]
      flat:    no recession (slope >= 0) or k beyond K_CLIP[1] -> longest timescale, K_CLIP[1]"""
    d = np.arange(1, len(q) + 1)
    sel = (d >= REC_DAYS[0]) & (d <= REC_DAYS[1])
    qw = q.to_numpy()[sel]
    if np.median(qw) <= Q_FLOOR:
        return K_CLIP[0], "drained"
    y = np.log(np.maximum(qw, Q_FLOOR))
    k = k_from_log_slope(np.polyfit(d[sel], y, 1)[0])
    if k >= K_CLIP[1]:
        return K_CLIP[1], "flat"
    if k <= K_CLIP[0]:
        return K_CLIP[0], "drained"
    return k, "fitted"


D1_MAX_ABS_LOG_RATIO = np.log(2)   # D1_strict: recession constant within a factor 2 of the observed one
D2_MIN_FRAC_DECREASING = 0.9      # D2_strict: Q falls on >= 90 % of the first 90 days
D3_TOL = 1.10                     # no later peak above Q on the last day of the first year (+10 %)
D4_MAX_RATIO = 0.5                # mean Q in the last year <= 0.5 x mean Q in the first year


def drought_checks(q, ks, kcls, k_obs, frac90, k_obs_d1, rr, thr):
    """Per-basin drought checks (zero-rain run). NaN = not scorable (no observed recession constant).
    D1 / D2 use the observation-calibrated thresholds `thr`; D1_strict / D2_strict are the fixed rules."""
    y0, y1 = E.DROUGHT[0].year, E.DROUGHT[1].year
    q_end0 = float(q.loc[f"{y0}-12-31"])
    later_max = float(q.loc[f"{y0 + 1}-01-01":E.DROUGHT[1]].max())
    m0, m1 = float(q.loc[str(y0)].mean()), float(q.loc[str(y1)].mean())
    alr = abs(np.log(ks / k_obs)) if np.isfinite(k_obs) else np.nan
    alr1 = abs(np.log(ks / k_obs_d1)) if np.isfinite(k_obs_d1) else np.nan
    return {"D1": (kcls == "fitted" and alr1 <= thr["tau_D1"]) if np.isfinite(k_obs_d1) else np.nan,
            "D2": rr <= thr["X_D2"],
            "D1_strict": (kcls == "fitted" and alr <= D1_MAX_ABS_LOG_RATIO) if np.isfinite(k_obs) else np.nan,
            "D2_strict": frac90 >= D2_MIN_FRAC_DECREASING, "abs_log_ratio_D1": alr1, "rise_ratio_90d": rr,
            "D3": later_max <= D3_TOL * q_end0, "D4": m1 <= D4_MAX_RATIO * m0,
            "Q_end_first_year": q_end0, "maxQ_later_years": later_max,
            "meanQ_first_year": m0, "meanQ_last_year": m1}


# Hand-picked fig04 example basins. high_BFI: Cub Creek at Phenix, VA
# (CAMELS BFI 0.62, frac_snow 0.06, 253 km2, KGE_cal >= 0.70 for every plotted model). Chosen because it is
# the only low-snow, high-BFI basin whose HBV point lies clearly below the D1 band (k 6.5 vs 40.6 d, factor
# 0.16 < 1/2.92) while all four regional ML models lie clearly above it (473 d to flat; seed 42 >= 187 d).
# Caveat: HBV starts 2001 at ~0.01 mm/d and its fitted k reflects a short day-7 bump then decay.
# Not the maximum-BFI basin: in the very highest-BFI basins the observed recession is so slow that ML
# timescales are realistic, so no max-BFI basin shows this contrast.
DROUGHT_EXAMPLE_OVERRIDE = {"high_BFI": "02065500"}

K_EDGE_MAX_FRAC = 0.5   # the median k error is only meaningful while fewer than half the basins sit at an edge


def check_k_edges(pb):
    """Median stays valid while drained + flat < 50 % of basins (per combo, per seed); fail loudly otherwise."""
    frac = (pb.assign(edge=pb.k_class != "fitted").groupby(["combo", "seed"])["edge"].mean())
    bad = frac[frac >= K_EDGE_MAX_FRAC]
    if len(bad):
        raise RuntimeError(f"drought k: >= {K_EDGE_MAX_FRAC:.0%} of basins at a K_CLIP edge, median "
                           f"k error not interpretable:\n{bad}")
    return frac


def drought(basins, obs, forc, sig):
    pre = (E.DATA_START, E.VAL[1])          # observed dry spells 1987-2000
    k_obs = {}
    for b in basins:
        k_obs[b], nseg = observed_k(obs[b].loc[pre[0]:pre[1]], forc[b]["Precip"].loc[pre[0]:pre[1]])
    # D1 / D2 thresholds from the observations alone (PLAUSIBILITY_CHECKS.md), before any model is read
    ref = pd.DataFrame({b: observed_drought_reference(obs[b].loc[pre[0]:pre[1]], forc[b]["Precip"].loc[pre[0]:pre[1]],
                                                      forc[b]["Temp"].loc[pre[0]:pre[1]]) for b in basins}).T
    ref.index.name = "basin"
    snowcov = {b: snow_cover_start(forc[b]["Precip"], forc[b]["Temp"]) for b in basins}
    ref["snow_cover_start"] = pd.Series(snowcov)
    # diagnostic only (limitation, not used in any score): days since the last rain day (P > DRY_P)
    # before the zero-rain run starts; the run's k is fitted from day 2 regardless of this
    d0 = E.DROUGHT[0]
    ref["days_since_rain_at_start"] = pd.Series(
        {b: int((d0 - forc[b]["Precip"].loc[:d0 - pd.Timedelta(days=1)].loc[lambda x: x > DRY_P].index[-1]).days)
         for b in basins})
    ref.to_csv(OUT_DROUGHT / "drought_observed_reference.csv")
    se = pd.to_numeric(ref["k_split_half_err"]).dropna()
    ro = pd.to_numeric(ref["rise_ratio_obs"]).replace(np.inf, np.nan).dropna()
    thr = {"tau_D1": float(se.quantile(CAL_QUANTILE)), "factor_D1": float(np.exp(se.quantile(CAL_QUANTILE))),
           "n_basins_D1_calibration": int(len(se)), "split_half_err_median": float(se.median()),
           "X_D2": float(ro.quantile(CAL_QUANTILE)), "n_basins_D2_calibration": int(len(ro)),
           "rise_ratio_obs_median": float(ro.median()),
           "n_basins_k_obs_winter": int((ref["k_obs_D1_season"] == "winter").sum()),
           "n_basins_snow_cover_start": int(sum(snowcov.values())), "quantile": CAL_QUANTILE}
    (OUT_DROUGHT / "drought_thresholds.json").write_text(json.dumps(thr, indent=1))
    print("drought thresholds (observations only):", thr, flush=True)
    snow_hi = sig.loc[basins, "frac_snow"] >= sig.loc[basins, "frac_snow"].quantile(2 / 3)
    rows = []
    for key in combos():
        for seed, sim in load_combo(key, "no_precip", basins).items():
            for b in basins:
                q = sim[b]
                q1 = q.loc[E.DROUGHT[0]:E.DROUGHT[0] + pd.Timedelta(days=364)]
                dq = np.diff(q1.to_numpy())
                months = q1.index[1:].month
                keep = ~np.isin(months, [3, 4, 5, 6, 7]) if snow_hi[b] else np.ones(len(dq), bool)
                ks, kcls = synthetic_k(q)
                frac90 = float(np.mean(dq[:89][keep[:89]] <= 0))
                t90 = forc[b]["Temp"].loc[q1.index[1:D2_DAYS]].to_numpy()
                rr = rise_ratio(dq[:D2_DAYS - 1], (t90 > SNOW_T_D2) if snowcov[b] else None)
                rows.append({"combo": key, "seed": seed, "basin": b, "k_syn": ks, "k_class": kcls, "k_obs_allseason": k_obs[b],
                             "abs_log_ratio_allseason": abs(np.log(ks / k_obs[b])) if np.isfinite(k_obs[b]) else np.nan,
                             "frac_decreasing_90d": frac90,
                             "frac_decreasing_365d": float(np.mean(dq[keep] <= 0)),
                             "meanQ_2002_2006": float(q.loc["2002-01-01":E.DROUGHT[1]].mean()),
                             "snow_melt_season_excluded": bool(snow_hi[b]),
                             "k_obs_D1": ref.loc[b, "k_obs_D1"], "snow_cover_start": snowcov[b],
                             **drought_checks(q, ks, kcls, k_obs[b], frac90, float(ref.loc[b, "k_obs_D1"]), rr, thr)})
    pb = pd.DataFrame(rows)
    # k_obs / abs_log_ratio = the D1 reference (winter-first, estimable only), so every figure and table
    # shows the same observed timescale the score uses; the all-season values are kept as *_allseason
    pb["k_obs"] = pb["k_obs_D1"]
    pb["abs_log_ratio"] = pb["abs_log_ratio_D1"]
    edge = check_k_edges(pb)
    print("drought k, fraction of basins at an edge (max over seeds):\n"
          + edge.groupby("combo").max().round(3).to_string(), flush=True)
    for col, name in [("baseflow_index", "BFI"), ("aridity", "aridity"), ("frac_snow", "frac_snow")]:
        terc = pd.qcut(sig.loc[basins, col], 3, labels=["low", "mid", "high"])
        pb[f"{name}_tercile"] = pb["basin"].map(terc).astype(str)
    pb.to_csv(OUT_DROUGHT / "drought_per_basin.csv", index=False)
    strata = []
    for strat in ["BFI_tercile", "aridity_tercile", "frac_snow_tercile"]:
        g = pb.groupby(["combo", strat])
        s = g.agg(median_abs_log_ratio=("abs_log_ratio", "median"), median_k_syn=("k_syn", "median"),
                  median_k_obs=("k_obs", "median"), frac_decreasing_90d=("frac_decreasing_90d", "median"),
                  meanQ_2002_2006=("meanQ_2002_2006", "median")).reset_index().rename(columns={strat: "stratum"})
        s.insert(1, "strata", strat)
        strata.append(s)
    pd.concat(strata).to_csv(OUT_DROUGHT / "drought_strata.csv", index=False)
    # Only pick an example basin from ones every plotted model (regional + HBV) actually calibrated
    # reasonably well on -- an attribute-extreme basin where e.g. HBV calibrated badly
    # produces a misleading example, since one line's collapse is
    # that basin's bad fit, not a general high-BFI behaviour.
    acc = pd.read_csv(E.ANALYSIS_ACCURACY_DIR / "accuracy_per_basin.csv", dtype={"basin": str})
    shown = [k for k in combos() if k.split("-", 1)[1] == "regional" or k.split("-", 1)[0] == "HBV"]
    cal_ok = acc[(acc.combo.isin(shown)) & (acc.period == "cal")]
    well_calibrated = set(cal_ok.groupby("basin")["KGE_ens"].min()[lambda x: x > 0.3].index) & set(basins)
    s_all = sig.loc[basins]
    s = sig.loc[[b for b in basins if b in well_calibrated]]
    if len(s) < 10:  # fallback if the filter is too strict for some reason -- never leave it empty
        s = s_all
    examples = {"high_BFI": s["baseflow_index"].idxmax(), "low_BFI": s["baseflow_index"].idxmin(),
                "snow": s["frac_snow"].idxmax()}
    # hand-picked basins apply only when they are evaluated (runs on a basin subset keep the automatic pick)
    examples.update({k: b for k, b in DROUGHT_EXAMPLE_OVERRIDE.items() if b in basins})
    examples = {k: examples[k] for k in ("high_BFI", "low_BFI", "snow")}
    (OUT_DROUGHT / "drought_example_basins.json").write_text(json.dumps(examples))
    return pb




# ------------------------------------------------------------------------------------------
# flood
# ------------------------------------------------------------------------------------------
F3_MIN_ELASTICITY = 0.8   # fixed threshold, reported as F3_fixed for reference only
F3_Z = 1.64               # F3: elasticity not significantly below 1 (one-sided 95 %): elast + F3_Z * SE >= 1


def slope_se(x, y):
    """OLS slope and its standard error (n - 2 degrees of freedom)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    sxx = np.sum((x - x.mean()) ** 2)
    ssr = np.sum((y - (a + b * x)) ** 2)
    return float(b), float(np.sqrt(ssr / (len(x) - 2) / sxx))
F4_MIN_REL_DQ = -0.01     # min dQ >= -1 % of peak dQ: rain never reduces flow
SNOW_T = 0.0              # deg C; a storm day at or below this falls as snow
SNOW_STORM_FRAC = 0.5     # F3 and F5 apply only to storms with snow fraction <= this


def flood(basins, obs, forc, sig):
    s0 = E.STORM_START
    win = pd.date_range(s0, E.FLOOD_WINDOW[1])
    ups = np.array(E.STORM_UPLIFTS) / 100.0
    rows = []
    for key in combos():
        per = {sc: load_combo(key, sc, basins) for sc in E.FLOOD_SCENARIOS + [E.NOSTORM]}
        og = load_combo(key, "og", basins)
        for seed in per[E.NOSTORM]:
            base = per[E.NOSTORM][seed].loc[win]
            for b in basins:
                fb = forc[b]
                src = fb.loc[E.SOURCE_SEARCH[0]:E.SOURCE_SEARCH[1], "Precip"]
                src_date, src_peak = src.idxmax(), float(src.max())
                # baseline: real source event in the normal test run vs observed discharge
                ev = slice(src_date - pd.Timedelta(days=1), src_date + pd.Timedelta(days=7))
                o_pk, s_pk = obs[b].loc[ev].max(), og[seed][b].loc[ev].max()
                for d in E.STORM_DURATIONS:
                    # storm runs use Temp_at_peak (core.io); equal P every storm day -> snow fraction = day fraction
                    t_storm = fb["Temp_at_peak"].loc[s0:s0 + pd.Timedelta(days=d - 1)].to_numpy()
                    snow_frac = float(np.mean(t_storm <= SNOW_T))
                    peaks, rcs, inc, frac, lag = [], [], [], [], []
                    no_dip, peak_after = [], []
                    for u in E.STORM_UPLIFTS:
                        dq = (per[f"peak{d}_rate{u}"][seed][b].loc[win] - base[b]).to_numpy()
                        pk = dq.max()
                        peaks.append(pk)
                        total = dq.sum()
                        rcs.append(total / (d * src_peak * (1 + u / 100)))
                        inc.append(dq[-7:].mean() > 0.01 * pk if pk > 0 else True)
                        cs = np.cumsum(dq)
                        frac.append([cs[min(n, len(cs)) - 1] / total if total != 0 else np.nan for n in (30, 90, 180, 365)])
                        lag.append(float(np.argmax(dq) - (d - 1) / 2))   # days from storm P centroid
                        no_dip.append(bool(pk > 0 and dq.min() >= F4_MIN_REL_DQ * pk))
                        peak_after.append(bool(np.argmax(dq) >= d - 1))   # index 0 = first storm day
                    peaks = np.array(peaks)
                    elast, elast_se = slope_se(np.log1p(ups), np.log(peaks)) if (peaks > 0).all() else (np.nan, np.nan)
                    mono = bool(np.all(np.diff(peaks) > 0))
                    incomplete = bool(any(inc))
                    under_cap = bool(np.all(np.array(rcs) <= 1))
                    fr = np.array(frac)
                    rain = snow_frac <= SNOW_STORM_FRAC
                    chk = {"F1": mono, "F2": under_cap,
                           "F3": bool(np.isfinite(elast) and elast + F3_Z * elast_se >= 1) if rain else np.nan,
                           "F4": all(no_dip), "F5": all(peak_after) if rain else np.nan}
                    if incomplete:
                        chk = {c: np.nan for c in chk}
                    applied = [v for v in chk.values() if not pd.isna(v)]
                    rows.append({"combo": key, "seed": seed, "basin": b, "duration": d, "elasticity": elast,
                                 "monotonic": mono, "max_runoff_coef": float(np.max(rcs)), "under_cap": under_cap,
                                 "incomplete": incomplete, "storm_snow_frac": snow_frac, "elasticity_se": elast_se,
                                 "F3_fixed": (bool(np.isfinite(elast) and elast >= F3_MIN_ELASTICITY)
                                              if rain and not incomplete else np.nan), **chk,
                                 "pass": bool(all(applied)) if not incomplete else np.nan,
                                 **{f"frac_dQ_day{n}": float(np.nanmedian(fr[:, i])) for i, n in enumerate((30, 90, 180, 365))},
                                 "lag_days": float(np.median(lag)), "peak_dQ_rate0": float(peaks[0]),
                                 "baseline_peak_rel_error": (s_pk - o_pk) / o_pk if o_pk and np.isfinite(o_pk) else np.nan})
    pb = pd.DataFrame(rows)
    pb["BFI"] = pb["basin"].map(sig["baseflow_index"])
    pb["frac_snow"] = pb["basin"].map(sig["frac_snow"])
    pb.to_csv(OUT_FLOOD / "flood_per_basin.csv", index=False)
    g = pb.groupby(["combo", "duration"])
    summ = g.agg(elasticity_median=("elasticity", "median"),
                 elasticity_IQR=("elasticity", iqr), elasticity_nan=("elasticity", lambda x: int(x.isna().sum())),
                 frac_monotonic=("monotonic", "mean"), frac_under_cap=("under_cap", "mean"),
                 n_incomplete=("incomplete", "sum"), lag_median=("lag_days", "median"),
                 frac_day30=("frac_dQ_day30", "median"), frac_day90=("frac_dQ_day90", "median"),
                 frac_day180=("frac_dQ_day180", "median"), frac_day365=("frac_dQ_day365", "median"),
                 baseline_peak_rel_error_median=("baseline_peak_rel_error", "median")).reset_index()
    # n_incomplete counted per seed run; report per seed (mean over seeds)
    nseed = pb.groupby("combo")["seed"].nunique()
    summ["n_incomplete"] = summ["n_incomplete"] / summ["combo"].map(nseed)
    corr = []
    for key, gg in pb.groupby("combo"):
        for col in ["frac_dQ_day30", "frac_dQ_day90", "lag_days"]:
            for att in ["BFI", "frac_snow"]:
                r = stats.spearmanr(gg[col], gg[att], nan_policy="omit")
                corr.append({"combo": key, "metric": col, "attribute": att, "spearman_rho": r.statistic, "p": r.pvalue})
    pd.DataFrame(corr).to_csv(OUT_FLOOD / "flood_response_vs_attributes.csv", index=False)
    summ.to_csv(OUT_FLOOD / "flood_summary.csv", index=False)
    return pb


# ------------------------------------------------------------------------------------------
# composite
#
# Each dimension is a set of per-basin pass/fail checks, each aimed at one failure mode. A basin
# (flood: a basin x storm-duration case) passes the dimension only if it passes every check that
# applies; the dimension score is the fraction passing. All-checks rather than averaging: e.g. a
# right-sized peak that arrives while rain is still falling is not plausible and gets no credit.
# Because one strict check can dominate, the pass rate of every check is written next to the
# dimension score, and the check-averaged score is reported as a sensitivity version.
#   mass    M1 sum Q_sim <= sum P;  M2 (sum P - sum Q_sim)/days <= mean PET.  Scored only where the
#           observed discharge passes both.  Volume error is left out on purpose: it is accuracy
#           (KGE contains it), and plausibility is kept separate from accuracy.
#   drought D1 |ln(k_syn/k_obs)| <= tau_D1 (drained / flat = fail); D2 rise ratio over the first 90
#           days <= X_D2 (tau_D1, X_D2: 90th percentiles of the observed statistics, see
#           PLAUSIBILITY_CHECKS.md); D3 max Q in later years <= 1.1 x Q on the last day of the first
#           year; D4 mean Q in the last year <= 0.5 x that of the first year. The fixed-threshold
#           variant (D1 ln 2, D2 Q falls on >= 90 % of the first 90 days) is reported as drought_strict.
#   flood   F1 peak dQ rises at every uplift step; F2 sum dQ <= storm P; F3 elasticity + 1.64 SE >= 1;
#           F4 min dQ >= -1 % of peak dQ; F5 peak dQ on or after the last storm day. F3 / F5 only
#           for storms with snow fraction <= 0.5; cases whose response has not finished excluded.
# Composite = plain mean of the three dimension fractions (different scenarios, so trading off is
# acceptable). "Fully plausible" = fraction of basins passing all three dimensions (no weights).
# Every value: per seed, median over seeds; 90 % intervals from a bootstrap over basins.
# ------------------------------------------------------------------------------------------
CHECKS = {"mass": ["M1", "M2"], "drought": ["D1", "D2", "D3", "D4"], "flood": ["F1", "F2", "F3", "F4", "F5"]}
DIM_NAMES = list(CHECKS)
STRICT_DROUGHT = ["D1_strict", "D2_strict", "D3", "D4"]


def _b(x):
    """bool / NaN column -> float (1, 0, NaN)."""
    return pd.to_numeric(x.map(lambda v: np.nan if pd.isna(v) else float(bool(v))), errors="coerce")


def basin_counts(tables, basins, key):
    """{item: (num, den)} arrays of shape (n_seeds, n_basins) for one combo; item = dimension, check
    or 'full'. A dimension score is sum(num) / sum(den) over basins."""
    seeds = sorted(tables["mass"].loc[tables["mass"].combo == key, "seed"].unique())
    idx = pd.Index(basins, name="basin")
    out = {}

    def put(item, s_i, num, den):
        n, d = out.setdefault(item, (np.zeros((len(seeds), len(basins))), np.zeros((len(seeds), len(basins)))))
        n[s_i], d[s_i] = num.reindex(idx).fillna(0).to_numpy(), den.reindex(idx).fillna(0).to_numpy()

    for s_i, seed in enumerate(seeds):
        sel = lambda t: t[(t.combo == key) & (t.seed == seed)]
        m = sel(tables["mass"]).set_index("basin")
        ok = m["obs_ok"].astype(bool)
        chk = pd.DataFrame({c: _b(m[c]) for c in CHECKS["mass"]})
        for c in CHECKS["mass"]:
            put(c, s_i, (chk[c] * ok).fillna(0), ok.astype(float))
        mass_pass = chk.min(axis=1).where(ok)                                  # NaN = not scored
        put("mass", s_i, mass_pass.fillna(0), ok.astype(float))

        d = sel(tables["drought"]).set_index("basin")
        chk = pd.DataFrame({c: _b(d[c]) for c in CHECKS["drought"]})
        scored = chk["D1"].notna()                                             # needs an observed k
        for c in CHECKS["drought"]:
            put(c, s_i, chk[c].where(scored).fillna(0), scored.astype(float))
        drought_pass = chk.min(axis=1).where(scored)
        put("drought", s_i, drought_pass.fillna(0), scored.astype(float))
        # strict rules (supplement): D1_strict / D2_strict with D3, D4
        st = pd.DataFrame({c: _b(d[c]) for c in STRICT_DROUGHT})
        sc = st["D1_strict"].notna()
        for c in ["D1_strict", "D2_strict"]:
            put(c, s_i, st[c].where(sc).fillna(0), sc.astype(float))
        put("drought_strict", s_i, st.min(axis=1).where(sc).fillna(0), sc.astype(float))

        f = sel(tables["flood"])
        fc = pd.DataFrame({c: _b(f[c]) for c in CHECKS["flood"] + ["pass"]}).assign(basin=f["basin"].to_numpy())
        g = fc.groupby("basin")
        for c in CHECKS["flood"]:
            put(c, s_i, g[c].sum(), g[c].count().astype(float))
        n_case, n_pass = g["pass"].count().astype(float), g["pass"].sum()
        put("flood", s_i, n_pass, n_case)
        flood_pass = (n_pass == n_case).astype(float).where(n_case > 0)         # basin: every case passes

        allp = pd.concat([mass_pass, drought_pass, flood_pass], axis=1).reindex(idx)
        scored_all = allp.notna().all(axis=1)
        put("full", s_i, allp.min(axis=1).where(scored_all).fillna(0), scored_all.astype(float))
    return out


def scores(counts, W):
    """W: (n_rep, n_basins) basin weights (bootstrap counts). Returns {item: (n_rep,)}, median over seeds."""
    out = {}
    for item, (num, den) in counts.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            out[item] = np.nanmedian((W @ num.T) / (W @ den.T), axis=1)
    for dim in DIM_NAMES:
        out[f"{dim}_avgchecks"] = np.mean([out[c] for c in CHECKS[dim]], axis=0)
    out["composite"] = np.mean([out[d] for d in DIM_NAMES], axis=0)
    out["composite_avgchecks"] = np.mean([out[f"{d}_avgchecks"] for d in DIM_NAMES], axis=0)
    out["composite_no_mass"] = np.mean([out["drought"], out["flood"]], axis=0)
    out["composite_strict_drought"] = np.mean([out["mass"], out["drought_strict"], out["flood"]], axis=0)
    return out


def composite(tables, basins, acc_summary, main_combos):
    nb = len(basins)
    W1 = np.ones((1, nb))
    Wb = np.stack([np.bincount(RNG.integers(nb, size=nb), minlength=nb) for _ in range(N_BOOT)]).astype(float)
    counts = {k: basin_counts(tables, basins, k) for k in combos()}
    point = {k: {i: float(v[0]) for i, v in scores(c, W1).items()} for k, c in counts.items()}
    boot = {k: scores(c, Wb) for k, c in counts.items()}
    n_scored = {k: {i: float(np.median(den.sum(axis=1))) for i, (_, den) in c.items()} for k, c in counts.items()}
    comp_b = pd.DataFrame({k: boot[k]["composite"] for k in main_combos})
    first = comp_b.idxmax(axis=1).value_counts().reindex(list(counts), fill_value=0) / N_BOOT

    acc = acc_summary[acc_summary.period == "test"].set_index("combo")
    rows, long = [], []
    for k in combos():
        row = {"combo": k, "in_main_ranking": k in main_combos,
               "test_KGE_median_ens": acc.loc[k, "KGE_median_ens"], "test_KGE_median_perseed": acc.loc[k, "KGE_median_perseed"]}
        for it in DIM_NAMES + ["composite", "full", "drought_strict", "composite_no_mass", "composite_strict_drought"]:
            name = "fully_plausible" if it == "full" else (it if it.startswith("composite") else f"{it}_raw")
            row.update({name: point[k][it], f"{name}_p05": np.quantile(boot[k][it], .05),
                        f"{name}_p95": np.quantile(boot[k][it], .95)})
        for dim in DIM_NAMES:
            row[f"{dim}_avgchecks"] = point[k][f"{dim}_avgchecks"]
            row[f"n_{dim}"] = n_scored[k][dim]
        row["composite_avgchecks"] = point[k]["composite_avgchecks"]
        row["n_fully_plausible"] = n_scored[k]["full"]
        row.update({c: point[k][c] for d in DIM_NAMES for c in CHECKS[d]})
        row.update({c: point[k][c] for c in ["D1_strict", "D2_strict"]})
        row["frac_bootstrap_rank_first"] = float(first[k])
        rows.append(row)
        for dim in DIM_NAMES:
            extra = ["D1_strict", "D2_strict", "drought_strict"] if dim == "drought" else []
            for it in [dim] + CHECKS[dim] + [f"{dim}_avgchecks"] + extra:
                long.append({"combo": k, "dimension": dim, "item": it, "pass_rate": point[k][it],
                             "p05": np.quantile(boot[k][it], .05), "p95": np.quantile(boot[k][it], .95),
                             "n": n_scored[k].get(it, n_scored[k][dim])})
    tab = pd.DataFrame(rows).sort_values("composite", ascending=False)
    tab.to_csv(OUT_SCORE / "composite_table.csv", index=False)
    pd.DataFrame(long).to_csv(OUT_SCORE / "check_pass_rates.csv", index=False)
    weight_sweep(tab, main_combos)
    return tab


def weight_sweep(tab, main_combos):
    """Weighted sum of the dimension fractions on a 0.05 grid (main-ranking combos)."""
    raw = tab.set_index("combo")[[f"{d}_raw" for d in DIM_NAMES]].loc[main_combos]
    sweep = []
    g = np.round(np.arange(0, 1.0001, 0.05), 2)
    for wm in g:
        for wd in g:
            wf = round(1 - wm - wd, 2)
            if wf < -1e-9:
                continue
            c = raw["mass_raw"] * wm + raw["drought_raw"] * wd + raw["flood_raw"] * wf
            best = "tie" if (c >= c.max() - 1e-9).sum() > 1 else c.idxmax()   # no arbitrary winner on ties
            sweep.append({"w_mass": wm, "w_drought": wd, "w_flood": wf, "best": best, **c.to_dict()})
    pd.DataFrame(sweep).to_csv(OUT_SCORE / "composite_weight_sweep.csv", index=False)


# ------------------------------------------------------------------------------------------
def main():
    for d in [OUT_ACC, OUT_MASS, OUT_DROUGHT, OUT_FLOOD, OUT_SCORE]:
        d.mkdir(parents=True, exist_ok=True)
    basins = load_eval_set()
    sig = load_camels_signatures()
    sig.index = sig.index.astype(str).str.zfill(8)
    forc = {b: load_forcing(b) for b in basins}
    obs = pd.DataFrame({b: forc[b][E.Q_COL] for b in basins})
    print(f"scoring on {len(basins)} evaluation basins")
    pb_acc, acc_summary, rf_drop = accuracy(basins, obs)
    print("accuracy done", flush=True)
    m, _ = mass(basins, obs, forc)
    print("mass done", flush=True)
    d = drought(basins, obs, forc, sig)
    print("drought done", flush=True)
    f = flood(basins, obs, forc, sig)
    print("flood done", flush=True)
    main_combos = [k for k in combos() if not (rf_drop and k.startswith("RF-"))]
    tab = composite({"mass": m, "drought": d, "flood": f}, basins, acc_summary, main_combos)
    print(tab.to_string())
    E.MARK_DIR.mkdir(parents=True, exist_ok=True)
    (E.MARK_DIR / "scoring.done").write_text("ok\n")


if __name__ == "__main__":
    main()
