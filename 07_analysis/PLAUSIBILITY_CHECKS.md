# Plausibility checks: rules

The rules below were fixed **before** the model outputs were scored with them. The D1 and D2 thresholds
come from the observations alone, through the procedures given here. No model output is used to set
any threshold. `scoring.py` writes the resulting numbers to `out/analysis/drought/drought_thresholds.json`.
They are copied into the "Resulting thresholds" section at the end of this file.

## Dimension scores

- Each dimension is a set of per-basin pass/fail checks. A basin passes the dimension only if it
  passes every check that applies. For flood, the unit is a basin x storm-duration case.
- The dimension score is the fraction of basins (or flood cases) that pass.
- Every check's pass rate is reported next to the dimension score.
- The check-averaged score (the mean of the check pass rates) is a sensitivity check only.
- Composite = plain mean of the three dimension fractions. A composite without the mass dimension is
  also reported.
- Fully plausible = the fraction of basins that pass all three dimensions.
- Everything is computed per seed and then medianed over seeds. 90 % intervals come from a
  bootstrap over basins.

## Mass (test period, normal run; scored only where the observed discharge passes both checks)

| | Passes if |
|---|---|
| M1 | sum Q_sim <= sum P |
| M2 | (sum P - sum Q_sim) / days <= mean PET |

## Drought (zero-rain run, P = 0 from 2001-01-01 to 2006-12-31)

| | Passes if |
|---|---|
| D1 | \|ln(k_syn / k_obs)\| <= tau_D1. Drained or flat synthetic recessions fail. |
| D2 | rise ratio over the first 90 days <= X_D2 (definition below) |
| D3 | max Q in 2002-2006 <= 1.1 x Q on 2001-12-31 |
| D4 | mean Q in 2006 <= 0.5 x mean Q in 2001 |

### D1: observed recession timescale k_obs and tolerance tau_D1

1. **Spells.** Observed dry spells use observed data from 1987 to 2000. A spell is at least 5
   consecutive days with P <= 0.1 mm/d. ln Q is fitted over days 2-30 after the last rain, with a
   separate intercept for each spell.
2. **Season.** The zero-rain run starts on 1 January. So k_obs uses only winter spells (first dry
   day in November to March) when the basin has at least 5 winter spells. Otherwise it uses spells
   from all seasons. This corrects a known bias: summer spells recede faster because of
   evaporation. The fallback is recorded for each basin (`k_obs_season`).
3. **Split-half agreement.** k_obs is estimated twice, once from spells starting in odd years and
   once from spells starting in even years. Both use the same season selection as step 2. A basin
   enters the calibration only if each half has at least 2 spells. Its split-half error is
   \|ln(k_odd / k_even)\|.
4. **Tolerance.** tau_D1 = the 90th percentile of the split-half error across basins. A model passes
   if it matches the observed timescale as well as the two halves of the observations match each
   other.

**Amendment, made before scoring and based on the observations only.** The first calibration
pass gave a 90th percentile of 3.21 (a factor of 25). The tail came from 155 basins where one half's
pooled fit was flat or rising. Those fits sat at the clip value k = 10^4, which marks a recession
that cannot be resolved (frozen-river or snowmelt winter flow), not a large error. The winter-only
k itself sat at the clip value in 123 basins. The rules were therefore amended as follows:

- A pooled k at either K_CLIP edge counts as **not estimable**.
- The winter-only k_obs is used only if it is estimable. Otherwise the all-season k is used.
- If neither is estimable, D1 cannot be scored, and the basin is left out of the drought dimension.
- The split-half calibration uses only basins where both halves are estimable.

No model output was looked at for this amendment.

### D2: early decline as a size-based rise ratio, threshold X_D2

1. **Rise ratio.** Over a window of day-to-day changes dQ, the rise ratio is
   R = sum of max(dQ, 0) / sum of max(-dQ, 0).
   - R = 0 when there are no rises.
   - R = infinity when there are rises but no fall.
   - Tiny wiggles add little to R. A real rise adds a lot.
2. **Snowmelt exemption.** Rises on days with T > 0 °C are not counted in snowy basins, because
   melt of an existing snowpack is legitimate.
   - On the model side (first 90 days of the zero-rain run), a basin is snowy if it had snow cover on
     1 January 2001. The proxy is snowfall (P on days with T <= 0 °C) of at least 5 mm over
     1 November to 31 December 2000, together with a mean December 2000 temperature <= 0 °C.
   - On the observed side, a basin is snowy if its mean December temperature over 1987-2000 is
     <= 0 °C.
3. **Observed statistic.** For each basin, R is pooled over the first 90 days of every observed dry
   spell, using the same spells as D1 step 1, all seasons. Pooling means summing the rises and the
   falls over all spells before taking the ratio.
4. **Threshold.** X_D2 = the 90th percentile of the observed R across basins.

### Strict versions (fixed thresholds, reported in the supplement)

- D1_strict: \|ln(k_syn / k_obs,all-season)\| <= ln 2.
- D2_strict: Q falls on >= 90 % of the first 90 days, with March to July excluded in the snowiest
  third of basins.
- drought_strict: the drought dimension recomputed with D1_strict, D2_strict, D3 and D4.

## Flood (storm run minus no-storm run; unit = basin x storm-duration case)

Cases whose response has not finished within the window are excluded.

| | Passes if |
|---|---|
| F1 | peak dQ rises at every uplift step |
| F2 | sum dQ <= storm P |
| F3 | elasticity not significantly below 1: elasticity + 1.64 × SE(slope) >= 1, a one-sided 95 % test (only for storms with snow fraction <= 0.5) |
| F4 | min dQ >= -1 % of peak dQ, at every uplift step |
| F5 | peak dQ on or after the last storm day, at every uplift step (only for storms with snow fraction <= 0.5) |

Storm snow fraction = the fraction of storm days with Temp_at_peak <= 0 °C. Every storm day has the
same rain, so this is also the fraction of storm rain that falls as snow.

## Ranking figures

- **All-model ranking:** the ternary over mass, drought and flood weights. Drought stays in, because
  the HBV vs ML difference on drought is real. Where the top candidates tie, the region is drawn grey
  and labelled "tie".
- **Resolution rule (general, any dimension, any comparison; fixed before the ML-only ranking was
  computed):**
  - If every candidate in a ranking comparison scores below **0.05** on a dimension's strict
    (pass-all-checks) score, that comparison uses the dimension's **check-averaged** score instead.
  - The reason: a pass-all-checks score has no resolution when every candidate fails it. The
    weighted product and TOPSIS would then turn noise-level differences near 0 into large ratios.
    The per-check pass rates still carry real differences.
  - The rule is applied separately to each comparison set: all models, and ML only. Which dimensions
    switched is shown in the panel title.
- **Where the strict score stays:**
  - The all-model ranking. HBV's strict drought score (0.23) is above 0.05, so the rule does not
    switch drought there, and the HBV vs ML difference is real.
  - The composite in fig06a.
  - The "fully plausible" share.
- **ML-only ranking:** every ML model's strict drought score is below 0.05, so the ML-only ternaries
  (fig06d, s04a) use `drought_avgchecks`.
- **Sensitivity (supplement, s04b-c):** the ML-only ranking with drought left out (mass and flood only),
  shown as score vs flood weight (mass weight = 1 - flood weight).
- **Mass:** it stays in the main composite, because the three dimensions were fixed before the
  results were seen. The composite without mass is a sensitivity check (scsv04). Without mass, RF-ind
  and LSTM-ind swap ML ranks 7 and 8; the rest of the order is unchanged.

## Diagnostic (limitation, not used in any score): D1 starting conditions

- **Hypothesis:** the zero-rain run is fitted from 2 January, possibly further into the recession
  than the observed fits (day 2 after the last rain). That would push every model's k towards "too
  slow".
- **Check:** for each basin, the days since the last rain day (P > 0.1 mm/d) before 2001-01-01
  (`days_since_rain_at_start` in `drought_observed_reference.csv`), compared with ln(k_syn / k_obs)
  for fitted basins.
- **Result:**
  - The run starts a median of 4 days after the last rain (IQR 1-8 d), close to the day-2 start of
    the observed fits.
  - The Spearman correlation with the timescale error is weak for every model: |rho| <= 0.10. Only
    the LSTM-reg and LSTM-ada correlations reach p < 0.05.
  - Starting conditions therefore do not explain the low D1 pass rates, or the ML models' "too slow"
    bias (median ln ratio +0.4 to +1.2, against -0.9 for HBV).

## Resulting thresholds

These are from the observations alone (`drought_thresholds.json`).

- **tau_D1 = 1.07**, i.e. k_syn within a **factor of 2.92** of k_obs.
  - 90th percentile of the split-half error over 581 basins; median 0.27, a factor of 1.32.
  - k_obs comes from winter spells in 514 basins and from all-season spells in 123.
  - 12 basins have no estimable k_obs and are left out of the drought dimension, which is scored on
    625 basins.
- **X_D2 = 0.208**: the sum of rises must be at most 20.8 % of the total fall over the first 90 days.
  - 90th percentile over 637 basins; median 0.048.
  - 411 basins count as snow-covered on 2001-01-01, so their rises on days above 0 °C are not
    counted.
  - 316 basins count as snowy on the observed side (mean December temperature <= 0 °C).

## F3 tolerance from each fit's own uncertainty

- **Elasticity:** the OLS slope of ln(peak dQ) on ln(1 + uplift) over the 4 uplift steps.
- **Standard error:** the slope's own SE, sqrt(SSR / (n - 2) / Sxx), with n = 4.
- **Rule:** a case passes F3 if its elasticity is not significantly below 1, i.e.
  elasticity + 1.64 × SE >= 1 (one-sided 95 %).
- **Why:** the tolerance then comes from each fit's own noise instead of a chosen number (a
  fixed 0.8). This is the same logic as the observation-based thresholds for D1 and D2.
- **Caveat:** with n = 4 the SE has only 2 degrees of freedom. The rule uses 1.64, as specified,
  rather than the t quantile for 2 degrees of freedom (2.92).
- **Reference:** the fixed-0.8 rule is kept per case as `F3_fixed` in `flood_per_basin.csv`.
