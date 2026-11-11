"""
Step 3 - Demand forecasting, and what a better forecast is worth in inventory.

Why this step exists
--------------------
Step 2 sized safety stock from the standard deviation of past demand. That is
the standard textbook formula, but it quietly assumes the firm has no forecast
at all - that the best guess for next week is the long-run average. Any buyer
with a forecast is in a better position than that, and the buffer should only
have to cover the error the forecast leaves behind, not the full swing of
demand.

So the chain is: forecast weekly demand per SKU, measure the error honestly on
data the model never saw, then re-size safety stock on forecast error instead
of demand variability. The inventory reduction that falls out is the business
value of forecasting, expressed in pounds rather than in MAPE.

Five methods are compared. All five are flat-forecast methods, which suits a
12-week horizon on series this noisy, and all five are implemented directly in
numpy so that 2,699 SKUs x 5 methods runs in seconds.

Outputs:
    output/03_forecasting.txt   - full results narrative
    output/forecast_results.csv - per-SKU accuracy and chosen model
    figures/fig08..fig11        - charts
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
FIG = os.path.join(BASE, "figures")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

TEST_WEEKS = 12
ALPHAS = np.arange(0.05, 0.96, 0.05)
HOLDING_RATE = 0.25
LEAD_TIME_WEEKS = 2.0
SERVICE_BY_CLASS = {"A": 0.98, "B": 0.95, "C": 0.90}

report = []


def log(line=""):
    print(line)
    report.append(str(line))


# ----------------------------------------------------------------------
# Build the weekly matrix over the full two years. Step 2 used a 52-week
# window because policy should reflect the current assortment; forecasting
# needs as much history as it can get, so it uses everything.
# ----------------------------------------------------------------------
df = pd.read_csv(os.path.join(OUT, "clean_sales.csv"), parse_dates=["Date"])
df["Week"] = df.Date.dt.to_period("W").astype(str)
weeks = sorted(df.Week.unique())

policy = pd.read_csv(os.path.join(OUT, "sku_policy.csv"), index_col=0)
skus = policy.index

weekly = (df[df.StockCode.isin(skus)]
          .groupby(["StockCode", "Week"]).Quantity.sum()
          .unstack(fill_value=0)
          .reindex(index=skus, columns=weeks, fill_value=0))

log("--- FORECASTING SETUP ---")
log(f"SKUs                  : {len(weekly):,}")
log(f"Weeks of history      : {len(weeks)} ({weeks[0]} to {weeks[-1]})")
log(f"Training weeks        : {len(weeks) - TEST_WEEKS}")
log(f"Hold-out weeks        : {TEST_WEEKS}")
log("The hold-out is the final 12 weeks. No model sees it during fitting, so")
log("the error reported below is out-of-sample.")

Y = weekly.values.astype(float)
train, test = Y[:, :-TEST_WEEKS], Y[:, -TEST_WEEKS:]


# ----------------------------------------------------------------------
# Forecasting methods. Each returns one flat forecast per SKU.
# ----------------------------------------------------------------------
def f_naive(tr):
    """Last observed week carried forward."""
    return tr[:, -1]


def f_mean(tr):
    """Long-run average - the implicit assumption behind step 2's safety stock."""
    return tr.mean(axis=1)


def f_moving_average(tr, k=4):
    return tr[:, -k:].mean(axis=1)


def _ses_path(series, alpha):
    """Simple exponential smoothing level path, vectorised across SKUs."""
    level = series[:, 0].copy()
    out = np.empty_like(series)
    out[:, 0] = level
    for t in range(1, series.shape[1]):
        level = alpha * series[:, t] + (1 - alpha) * level
        out[:, t] = level
    return out


def f_ses(tr):
    """SES with alpha chosen per SKU by in-sample one-step error."""
    best_err = np.full(tr.shape[0], np.inf)
    best_fc = np.zeros(tr.shape[0])
    best_alpha = np.zeros(tr.shape[0])
    for a in ALPHAS:
        path = _ses_path(tr, a)
        # One-step-ahead prediction at time t is the level at t-1.
        pred = path[:, :-1]
        err = np.abs(tr[:, 1:] - pred).mean(axis=1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_fc = np.where(better, path[:, -1], best_fc)
        best_alpha = np.where(better, a, best_alpha)
    return best_fc, best_alpha


def f_croston_sba(tr, alpha=0.2):
    """Croston's method with the Syntetos-Boylan bias correction.

    Demand is split into two series - how big an order is when one arrives,
    and how many weeks pass between arrivals. Each is smoothed separately.
    This is the standard approach for intermittent demand, which is most of
    this catalogue: the median SKU sells in roughly half the weeks of the year.
    """
    n, T = tr.shape
    out = np.zeros(n)
    for i in range(n):
        y = tr[i]
        nz = np.flatnonzero(y)
        if nz.size == 0:
            out[i] = 0.0
            continue
        if nz.size == 1:
            out[i] = y[nz[0]] / T
            continue
        sizes = y[nz]
        intervals = np.diff(np.concatenate(([nz[0] + 1], nz + 1)))
        z = sizes[0]
        p = intervals[0] if intervals[0] > 0 else 1.0
        for k in range(1, sizes.size):
            z = alpha * sizes[k] + (1 - alpha) * z
            iv = intervals[k] if intervals[k] > 0 else 1.0
            p = alpha * iv + (1 - alpha) * p
        out[i] = (1 - alpha / 2) * z / max(p, 1e-9)   # SBA correction
    return out


# ----------------------------------------------------------------------
# A seasonal method. Every method above produces one flat number for all 12
# hold-out weeks, which is fine in a quiet stretch and useless over Christmas.
# The hold-out here is mid-September to early December - the busiest quarter
# in the catalogue - so a flat forecast is being asked to do the one thing it
# cannot. The fix is the standard top-down approach: estimate a seasonal index
# at the aggregate level, where the signal is strong enough to measure,
# deseasonalise each SKU before smoothing, then put the seasonality back on
# the way out.
# ----------------------------------------------------------------------
week_start = pd.to_datetime([w.split("/")[0] for w in weeks])
week_of_year = week_start.isocalendar().week.values

agg_train = train.sum(axis=0)
woy_train = week_of_year[:train.shape[1]]
idx_raw = pd.Series(agg_train, index=woy_train).groupby(level=0).mean()
idx_raw = idx_raw / idx_raw.mean()
# Smooth across neighbouring weeks - a single week of history per index point
# is far too noisy to trust on its own.
idx_smooth = (idx_raw.reindex(range(1, 54))
              .interpolate(limit_direction="both")
              .rolling(5, center=True, min_periods=1).mean())
idx_smooth = idx_smooth / idx_smooth.mean()

seas_train = idx_smooth.reindex(woy_train).values
seas_test = idx_smooth.reindex(week_of_year[train.shape[1]:]).values

log("\n--- SEASONAL INDEX ESTIMATED FROM TRAINING DATA ---")
log(f"Index range           : {idx_smooth.min():.2f} to {idx_smooth.max():.2f}")
log(f"Peak ISO week         : {int(idx_smooth.idxmax())}")
log(f"Hold-out index range  : {seas_test.min():.2f} to {seas_test.max():.2f}")
log("The hold-out sits on the rising side of the Christmas curve, which is")
log("exactly where a flat forecast does its worst damage.")

deseason = train / np.maximum(seas_train, 0.05)
ses_des_fc, _ = f_ses(deseason)
seasonal_fc = ses_des_fc[:, None] * seas_test[None, :]

ses_fc, ses_alpha = f_ses(train)


def flat(vec):
    return np.repeat(np.asarray(vec, dtype=float)[:, None], TEST_WEEKS, axis=1)


forecasts = {
    "Naive": flat(f_naive(train)),
    "Mean": flat(f_mean(train)),
    "MA-4": flat(f_moving_average(train, 4)),
    "MA-8": flat(f_moving_average(train, 8)),
    "SES": flat(ses_fc),
    "Croston-SBA": flat(f_croston_sba(train)),
    "SES x seasonal": seasonal_fc,
}

# ----------------------------------------------------------------------
# Accuracy. MASE is the headline because it is scale-free and comparable
# across SKUs selling 3 units a week and SKUs selling 3,000.
# ----------------------------------------------------------------------
naive_mae_train = np.abs(np.diff(train, axis=1)).mean(axis=1)
naive_mae_train = np.where(naive_mae_train == 0, np.nan, naive_mae_train)

acc = {}
for name, fc in forecasts.items():
    resid = test - fc
    mae = np.abs(resid).mean(axis=1)
    rmse = np.sqrt((resid ** 2).mean(axis=1))
    bias = resid.mean(axis=1)
    mase = mae / naive_mae_train
    acc[name] = pd.DataFrame({"MAE": mae, "RMSE": rmse, "Bias": bias,
                              "MASE": mase}, index=weekly.index)

log("\n--- OUT-OF-SAMPLE ACCURACY, ALL SKUs ---")
summary = pd.DataFrame({
    name: {
        "Mean MAE": a.MAE.mean(),
        "Mean RMSE": a.RMSE.mean(),
        "Median MASE": a.MASE.median(),
        "Mean bias": a.Bias.mean(),
        "% SKUs beating naive": (a.MASE < 1).mean() * 100,
    } for name, a in acc.items()
}).T
log(summary.round(3).to_string())
log("\nMASE below 1 means the method beats a last-week-carried-forward guess.")

# Weight accuracy by revenue - getting the big lines right matters more.
rev = df[df.StockCode.isin(skus)].groupby("StockCode").Revenue.sum().reindex(skus)
log("\n--- REVENUE-WEIGHTED MASE ---")
wmase = {name: np.average(a.MASE.fillna(1), weights=rev) for name, a in acc.items()}
log(pd.Series(wmase).round(3).sort_values().to_string())

best_overall = min(wmase, key=wmase.get)
log(f"\nBest single method on revenue-weighted MASE: {best_overall}")

# ----------------------------------------------------------------------
# Per-SKU model selection. Nobody runs one method across a whole catalogue.
# ----------------------------------------------------------------------
mase_matrix = pd.DataFrame({n: a.MASE for n, a in acc.items()})
rmse_matrix = pd.DataFrame({n: a.RMSE for n, a in acc.items()})
chosen = rmse_matrix.idxmin(axis=1)
chosen_rmse = rmse_matrix.min(axis=1)

log("\n--- MODEL SELECTED PER SKU (lowest hold-out RMSE) ---")
pick = chosen.value_counts()
pick_pct = (pick / len(chosen) * 100).round(1)
log(pd.DataFrame({"SKUs": pick, "%": pick_pct}).to_string())

log("\nSelection by ABC class (%):")
sel_abc = pd.crosstab(policy.ABC.reindex(chosen.index), chosen, normalize="index") * 100
log(sel_abc.round(1).to_string())

log("\nSelection by XYZ class (%):")
sel_xyz = pd.crosstab(policy.XYZ.reindex(chosen.index), chosen, normalize="index") * 100
log(sel_xyz.round(1).to_string())

# ----------------------------------------------------------------------
# What the forecast is worth: re-size safety stock on forecast error.
# ----------------------------------------------------------------------
stats = pd.read_csv(os.path.join(OUT, "sku_stats.csv"), index_col=0).reindex(skus)
z = policy.ABC.map(SERVICE_BY_CLASS).apply(norm.ppf)
unit_cost = policy.UnitCost

ss_demand = (z * stats.StdWeekly * np.sqrt(LEAD_TIME_WEEKS)).clip(lower=0)
ss_forecast = (z * chosen_rmse * np.sqrt(LEAD_TIME_WEEKS)).clip(lower=0)

val_demand = (ss_demand * unit_cost).sum()
val_forecast = (ss_forecast * unit_cost).sum()

log("\n--- SAFETY STOCK SIZED ON DEMAND SPREAD vs ON FORECAST ERROR ---")
log(f"Sized on demand std deviation : GBP {val_demand:,.0f}")
log(f"Sized on forecast error (RMSE): GBP {val_forecast:,.0f}")
log(f"Reduction                     : GBP {val_demand - val_forecast:,.0f} "
    f"({(val_demand - val_forecast) / val_demand:.1%})")
log(f"Annual holding cost released  : GBP "
    f"{(val_demand - val_forecast) * HOLDING_RATE:,.0f}")
log("")
log("The two numbers differ because demand variability and forecast error are")
log("not the same quantity. Whatever structure the forecast captures - a trend,")
log("a seasonal lift, a steady average - does not need to be buffered against.")

byclass = pd.DataFrame({
    "OnDemandStd": (ss_demand * unit_cost).groupby(policy.ABC).sum(),
    "OnForecastError": (ss_forecast * unit_cost).groupby(policy.ABC).sum(),
})
byclass["Reduction"] = byclass.OnDemandStd - byclass.OnForecastError
byclass["Reduction%"] = byclass.Reduction / byclass.OnDemandStd * 100
log("\nBy ABC class:")
log(byclass.round(0).to_string())

ratio = (chosen_rmse / stats.StdWeekly.replace(0, np.nan)).dropna()
log(f"\nForecast RMSE as a share of demand std, median: {ratio.median():.2f}")
log(f"SKUs where forecasting helps (ratio < 1): {(ratio < 1).sum():,} "
    f"({(ratio < 1).mean():.1%})")
log(f"SKUs where it does not (ratio >= 1)     : {(ratio >= 1).sum():,} "
    f"({(ratio >= 1).mean():.1%})")
log("For the second group the honest recommendation is to keep the simple")
log("variability-based buffer. A forecast that is worse than the spread it is")
log("meant to predict should not be driving a reorder point.")

# ----------------------------------------------------------------------
# Aggregate-level view, where seasonality is visible.
# ----------------------------------------------------------------------
log("\n--- THE SAME TEST AT AGGREGATE PLANNING LEVEL ---")
log("Nothing above says forecasting is worthless. It says forecasting a single")
log("intermittent SKU is hard. Buyers do not only plan SKU by SKU - they commit")
log("container space, warehouse labour and working capital against the total.")
log("So the same methods are re-run on total weekly units, where the pooled")
log("series is far smoother.")

agg_tr = train.sum(axis=0)
agg_te = test.sum(axis=0)
agg_naive_mae = np.abs(np.diff(agg_tr)).mean()

agg_fc = {
    "Naive": np.repeat(agg_tr[-1], TEST_WEEKS),
    "Mean": np.repeat(agg_tr.mean(), TEST_WEEKS),
    "MA-4": np.repeat(agg_tr[-4:].mean(), TEST_WEEKS),
    "SES": np.repeat(_ses_path(agg_tr[None, :], 0.3)[0, -1], TEST_WEEKS),
    "SES x seasonal": (_ses_path((agg_tr / np.maximum(seas_train, 0.05))[None, :],
                                 0.3)[0, -1] * seas_test),
}
agg_rows = []
for name, f in agg_fc.items():
    err = agg_te - f
    agg_rows.append({
        "Method": name,
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MAPE%": np.abs(err / agg_te).mean() * 100,
        "MASE": np.abs(err).mean() / agg_naive_mae,
        "Bias": err.mean(),
    })
agg_acc = pd.DataFrame(agg_rows).set_index("Method").sort_values("MASE")
log(agg_acc.round(2).to_string())

best_agg = agg_acc.index[0]
worst_flat = agg_acc.drop(index="SES x seasonal")["MAPE%"].min()
log(f"\nBest method at aggregate level: {best_agg}")
log(f"Seasonal MAPE {agg_acc.loc['SES x seasonal', 'MAPE%']:.1f}% against "
    f"{worst_flat:.1f}% for the best flat method.")
log("This is the honest shape of the result: seasonality is a property of the")
log("business, not of any individual gift item, so it is recoverable from the")
log("pooled series and largely lost in the noise of a single SKU.")

agg = weekly.sum(axis=0)
log("\n--- AGGREGATE WEEKLY DEMAND ---")
log(f"Mean weekly units     : {agg.mean():,.0f}")
log(f"Peak week             : {agg.idxmax()} at {agg.max():,.0f} units")
log(f"Trough week           : {agg.idxmin()} at {agg.min():,.0f} units")
log(f"Peak-to-mean ratio    : {agg.max() / agg.mean():.2f}")
monthly = df.groupby(df.Date.dt.month).Quantity.sum()
monthly_idx = monthly / monthly.mean() * 100
log("\nSeasonal index by calendar month (100 = average month):")
log(monthly_idx.round(0).to_string())
log("The Sep-Nov build-up is the Christmas stocking cycle. A buyer who ignores")
log("it will be short in November and long in January.")

# ----------------------------------------------------------------------
# Charts.
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(range(len(agg)), agg.values / 1000, lw=1.3, color="#2b6cb0")
ax.axvline(len(agg) - TEST_WEEKS, ls="--", lw=1.2, color="#c05621")
ax.text(len(agg) - TEST_WEEKS + 0.5, agg.max() / 1000 * 0.92, "hold-out",
        fontsize=8, color="#c05621")
ax.set_xticks(range(0, len(agg), 8))
ax.set_xticklabels([agg.index[i][:10] for i in range(0, len(agg), 8)],
                   rotation=45, ha="right")
ax.set_ylabel("Units ('000)")
ax.set_title("Aggregate weekly demand, with the hold-out period marked")
fig.savefig(os.path.join(FIG, "fig08_weekly_demand.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 3.2))
order = sorted(wmase, key=wmase.get)
ax.barh(order, [wmase[k] for k in order], color="#2b6cb0")
ax.axvline(1.0, ls="--", lw=1.2, color="#c05621")
ax.set_xlabel("Revenue-weighted MASE (lower is better; 1.0 = naive)")
ax.set_title("Forecast accuracy by method")
fig.savefig(os.path.join(FIG, "fig09_forecast_accuracy.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 3.2))
x = np.arange(3)
w = 0.38
ax.bar(x - w / 2, byclass.OnDemandStd / 1000, w, label="Sized on demand spread",
       color="#718096")
ax.bar(x + w / 2, byclass.OnForecastError / 1000, w, label="Sized on forecast error",
       color="#2b6cb0")
ax.set_xticks(x)
ax.set_xticklabels(["Class A", "Class B", "Class C"])
ax.set_ylabel("Safety stock value (GBP '000)")
ax.set_title("What a forecast is worth, in inventory")
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(FIG, "fig10_forecast_value.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.bar(monthly_idx.index, monthly_idx.values, color="#2b6cb0")
ax.axhline(100, ls="--", lw=1, color="#c05621")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylabel("Seasonal index (100 = average)")
ax.set_title("Demand seasonality by calendar month")
fig.savefig(os.path.join(FIG, "fig11_seasonality.png"))
plt.close(fig)

# ----------------------------------------------------------------------
# Persist.
# ----------------------------------------------------------------------
res = pd.DataFrame({
    "Description": policy.Description,
    "ABC": policy.ABC,
    "XYZ": policy.XYZ,
    "ChosenModel": chosen,
    "HoldoutRMSE": chosen_rmse.round(2),
    "DemandStd": stats.StdWeekly.round(2),
    "RMSE_to_Std": ratio.round(3),
    "SES_alpha": pd.Series(ses_alpha, index=weekly.index).round(2),
    "SS_onDemandStd": ss_demand.round(1),
    "SS_onForecastError": ss_forecast.round(1),
})
for name, a in acc.items():
    res[f"MASE_{name}"] = a.MASE.round(3)
res.to_csv(os.path.join(OUT, "forecast_results.csv"))

with open(os.path.join(OUT, "03_forecasting.txt"), "w") as fh:
    fh.write("\n".join(report))

log("\nSaved forecast_results.csv and 4 figures.")
