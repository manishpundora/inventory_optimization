"""
Step 2 - ABC-XYZ segmentation and inventory policy design.

Design note on what is being compared
-------------------------------------
Order quantity and safety stock are two separate decisions. If the proposed
policy changed both at once, any cost difference could not be attributed to
either. So EOQ is applied identically to both the current and the proposed
policy, which makes ordering cost cancel out, and the comparison isolates the
one thing this study is actually about: how safety stock is allocated across
SKUs.

Current practice is modelled as a flat rule - four weeks of average demand as
buffer on every SKU, regardless of how valuable or how erratic that SKU is.
This is the most common rule of thumb in small and mid-sized distribution and
is what the literature calls an undifferentiated policy.

The proposed policy sets the target service level by ABC class and sizes the
buffer from each SKU's own demand variability.

Outputs:
    output/02_inventory.txt    - full results narrative
    output/sku_policy.csv      - per-SKU policy table
    output/abc_xyz_matrix.csv  - 9-cell summary
    output/sensitivity.csv     - assumption sensitivity grid
    figures/fig04..fig07       - charts
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

# ----------------------------------------------------------------------
# Assumptions. The dataset records what was sold and at what price. It does
# not record purchase cost, ordering cost or lead time, so these are set from
# published trade norms and then stress-tested in the sensitivity grid.
#
# ORDER_COST is the marginal cost of adding one more SKU line to a purchase
# order, not the cost of raising a whole PO. A buyer consolidates many lines
# onto one order to one supplier, so the per-line figure is small.
# ----------------------------------------------------------------------
GROSS_MARGIN = 0.50      # giftware wholesale; unit cost = price x (1 - margin)
HOLDING_RATE = 0.25      # annual carrying cost as a fraction of unit cost
ORDER_COST = 10.0        # GBP per SKU line on a consolidated purchase order
LEAD_TIME_WEEKS = 2.0

SERVICE_BY_CLASS = {"A": 0.98, "B": 0.95, "C": 0.90}
BASELINE_SS_WEEKS = 4.0  # flat four weeks of cover on every SKU

report = []


def log(line=""):
    print(line)
    report.append(str(line))


df = pd.read_csv(os.path.join(OUT, "clean_sales.csv"), parse_dates=["Date"])

# ----------------------------------------------------------------------
# Policy window: the most recent 52 trading weeks.
# ----------------------------------------------------------------------
end = df.Date.max()
start = end - pd.Timedelta(weeks=52)
window = df[df.Date > start].copy()
window["Week"] = window.Date.dt.to_period("W").astype(str)

log("--- POLICY WINDOW ---")
log(f"From {start.date()} to {end.date()}")
log(f"Rows in window             : {len(window):,}")
log(f"SKUs appearing in window   : {window.StockCode.nunique():,}")

weeks = sorted(window.Week.unique())
weekly = (window.groupby(["StockCode", "Week"]).Quantity.sum()
          .unstack(fill_value=0)
          .reindex(columns=weeks, fill_value=0))

active_weeks = (weekly > 0).sum(axis=1)
keep = active_weeks[active_weeks >= 10].index
weekly = weekly.loc[keep]
log(f"Trading weeks              : {len(weeks)}")
log(f"SKUs with >=10 active weeks: {len(weekly):,} "
    f"({len(weekly) / window.StockCode.nunique():.1%} of window SKUs)")
log("SKUs selling in fewer than 10 of 52 weeks are one-off or discontinued")
log("lines. They do not need a standing reorder policy and are excluded.")

# ----------------------------------------------------------------------
# Lumpy demand. A small group of SKUs is dominated by a single bulk deal -
# stock code 23166 moved 74,215 of its 78,033 annual units on one invoice in
# January 2011. The safety stock formula assumes demand accumulates from many
# small independent draws, so a reorder point fitted to a series like that is
# arithmetic without meaning. These lines belong on a quote-driven or
# make-to-order arrangement with the customer, not on a statistical policy,
# so they are separated out and reported on their own.
# ----------------------------------------------------------------------
max_week_share = weekly.max(axis=1) / weekly.sum(axis=1).replace(0, np.nan)
lumpy_idx = max_week_share[max_week_share > 0.40].index
desc_lookup = window.drop_duplicates("StockCode").set_index("StockCode")["Description"]
rev_lookup = window.groupby("StockCode").Revenue.sum()

log(f"\n--- LUMPY-DEMAND SKUs HELD OUT (single week > 40% of annual units) ---")
log(f"SKUs held out              : {len(lumpy_idx):,} "
    f"({len(lumpy_idx) / len(weekly):.1%} of candidates)")
log(f"Revenue they represent     : GBP {rev_lookup.reindex(lumpy_idx).sum():,.0f} "
    f"({rev_lookup.reindex(lumpy_idx).sum() / rev_lookup.reindex(weekly.index).sum():.1%})")
lumpy_tbl = pd.DataFrame({
    "Description": desc_lookup.reindex(lumpy_idx),
    "AnnualUnits": weekly.loc[lumpy_idx].sum(axis=1),
    "BiggestWeek": weekly.loc[lumpy_idx].max(axis=1),
    "ShareInOneWeek": (max_week_share.reindex(lumpy_idx) * 100).round(1),
    "Revenue": rev_lookup.reindex(lumpy_idx).round(0),
}).sort_values("Revenue", ascending=False)
log("\nTop 10 by revenue:")
log(lumpy_tbl.head(10).to_string())
lumpy_tbl.to_csv(os.path.join(OUT, "lumpy_skus.csv"))

weekly = weekly.drop(index=lumpy_idx)
log(f"\nSKUs entering statistical policy: {len(weekly):,}")

# ----------------------------------------------------------------------
# Per-SKU demand statistics and economics.
# ----------------------------------------------------------------------
stats = pd.DataFrame({
    "AnnualUnits": weekly.sum(axis=1),
    "MeanWeekly": weekly.mean(axis=1),
    "StdWeekly": weekly.std(axis=1, ddof=1),
    "ActiveWeeks": (weekly > 0).sum(axis=1),
})
stats["CV"] = stats.StdWeekly / stats.MeanWeekly.replace(0, np.nan)

econ = window[window.StockCode.isin(stats.index)].groupby("StockCode").agg(
    Revenue=("Revenue", "sum"), Units=("Quantity", "sum"))
econ["AvgPrice"] = econ.Revenue / econ.Units
stats = stats.join(econ[["Revenue", "AvgPrice"]])
stats["UnitCost"] = stats.AvgPrice * (1 - GROSS_MARGIN)
stats["Description"] = window.drop_duplicates("StockCode").set_index("StockCode")["Description"]

# ----------------------------------------------------------------------
# ABC on cumulative revenue. Standard 80 / 95 cut points.
# ----------------------------------------------------------------------
stats = stats.sort_values("Revenue", ascending=False)
stats["CumRevShare"] = stats.Revenue.cumsum() / stats.Revenue.sum()
stats["ABC"] = np.where(stats.CumRevShare <= 0.80, "A",
                np.where(stats.CumRevShare <= 0.95, "B", "C"))

# ----------------------------------------------------------------------
# XYZ. The textbook cut points are CV <= 0.5 for X and <= 1.0 for Y. Those
# were written for manufacturing demand that arrives steadily. Applied to this
# retailer they classify 87% of SKUs as Z, which is a true statement about
# giftware wholesale but useless as a segmentation - a class that holds almost
# everything cannot guide a different decision for anything.
#
# Both versions are therefore reported. The textbook cut is kept for
# comparability with prior studies; empirical tertiles of the observed CV
# distribution are used to drive the policy, because the purpose of XYZ here
# is to rank SKUs against each other on predictability.
# ----------------------------------------------------------------------
stats["XYZ_textbook"] = np.where(stats.CV <= 0.5, "X",
                         np.where(stats.CV <= 1.0, "Y", "Z"))
q33, q67 = stats.CV.quantile([1 / 3, 2 / 3])
stats["XYZ"] = np.where(stats.CV <= q33, "X", np.where(stats.CV <= q67, "Y", "Z"))
stats["Segment"] = stats.ABC + stats.XYZ

log("\n--- ABC CLASSIFICATION ---")
abc = stats.groupby("ABC").agg(SKUs=("Revenue", "size"), Revenue=("Revenue", "sum"))
abc["SKU%"] = abc.SKUs / abc.SKUs.sum() * 100
abc["Revenue%"] = abc.Revenue / abc.Revenue.sum() * 100
log(abc.round(1).to_string())

log("\n--- XYZ: TEXTBOOK CUT POINTS (CV 0.5 / 1.0) ---")
log(stats.XYZ_textbook.value_counts().reindex(["X", "Y", "Z"]).to_string())
log(f"Median CV across all SKUs  : {stats.CV.median():.2f}")
log("87% of SKUs land in Z. Reported for comparability, not used for policy.")

log(f"\n--- XYZ: EMPIRICAL TERTILES (CV cut at {q33:.2f} and {q67:.2f}) ---")
xyz = stats.groupby("XYZ").agg(SKUs=("CV", "size"), MedianCV=("CV", "median"),
                               Revenue=("Revenue", "sum"))
xyz["SKU%"] = xyz.SKUs / xyz.SKUs.sum() * 100
xyz["Revenue%"] = xyz.Revenue / xyz.Revenue.sum() * 100
log(xyz.round(2).to_string())

log("\n--- 9-CELL MATRIX (SKU counts) ---")
matrix = pd.crosstab(stats.ABC, stats.XYZ)
log(matrix.to_string())
log("\n--- 9-CELL MATRIX (revenue share %) ---")
rev_matrix = (pd.crosstab(stats.ABC, stats.XYZ, values=stats.Revenue,
                          aggfunc="sum") / stats.Revenue.sum() * 100)
log(rev_matrix.round(2).to_string())


# ----------------------------------------------------------------------
# Inventory mechanics.
# ----------------------------------------------------------------------
def loss_function(z):
    """Standard normal loss L(z) = phi(z) - z * (1 - Phi(z)).

    Expected units short per replenishment cycle = sigma_LT * L(z).
    """
    return norm.pdf(z) - z * (1 - norm.cdf(z))


def economics(s, order_cost, holding_rate, gross_margin):
    e = pd.DataFrame(index=s.index)
    e["UnitCost"] = s.AvgPrice * (1 - gross_margin)
    e["H"] = e.UnitCost * holding_rate
    h = e.H.replace(0, np.nan)
    e["EOQ"] = np.sqrt(2 * s.AnnualUnits * order_cost / h)
    e["EOQ"] = e.EOQ.fillna(s.AnnualUnits / 12).clip(lower=1).round()
    return e


def evaluate(s, safety_stock, eco, lead_time):
    """Cost and service outcome for a given safety stock vector."""
    r = pd.DataFrame(index=s.index)
    r["SafetyStock"] = safety_stock.clip(lower=0)
    r["EOQ"] = eco.EOQ
    r["UnitCost"] = eco.UnitCost
    r["H"] = eco.H
    r["ROP"] = (s.MeanWeekly * lead_time + r.SafetyStock).round()
    r["AvgInventory"] = r.EOQ / 2 + r.SafetyStock
    r["InventoryValue"] = r.AvgInventory * r.UnitCost
    r["HoldingCost"] = r.AvgInventory * r.H
    r["SafetyStockValue"] = r.SafetyStock * r.UnitCost

    sigma_lt = (s.StdWeekly * np.sqrt(lead_time)).replace(0, np.nan)
    r["z"] = r.SafetyStock / sigma_lt
    r["CycleService"] = norm.cdf(r.z)
    # Expected units short per cycle, and over a year.
    esc = sigma_lt * loss_function(r.z)
    r["FillRate"] = (1 - esc / r.EOQ).clip(0, 1)
    r["AnnualUnitsShort"] = esc * (s.AnnualUnits / r.EOQ)
    r["AnnualRevenueAtRisk"] = r.AnnualUnitsShort * s.AvgPrice
    return r


eco = economics(stats, ORDER_COST, HOLDING_RATE, GROSS_MARGIN)

# Current practice: flat four weeks of cover, same for every SKU.
ss_base = BASELINE_SS_WEEKS * stats.MeanWeekly
base = evaluate(stats, ss_base, eco, LEAD_TIME_WEEKS)

# Proposed: service level differentiated by ABC class, buffer sized from each
# SKU's own demand variability.
z_target = stats.ABC.map(SERVICE_BY_CLASS).apply(norm.ppf)
ss_prop = z_target * stats.StdWeekly * np.sqrt(LEAD_TIME_WEEKS)
prop = evaluate(stats, ss_prop, eco, LEAD_TIME_WEEKS)

for frame in (base, prop):
    frame["ABC"] = stats.ABC
    frame["XYZ"] = stats.XYZ
    frame["Segment"] = stats.Segment

log("\n--- WHAT THE FLAT 4-WEEK RULE ACTUALLY DELIVERS ---")
cs = base.CycleService.dropna()
log(f"Median cycle service       : {cs.median():.1%}")
log(f"SKUs below 90% service     : {(cs < 0.90).sum():,} ({(cs < 0.90).mean():.1%})")
log(f"SKUs above 99% service     : {(cs > 0.99).sum():,} ({(cs > 0.99).mean():.1%})")
log("One flat buffer does not produce one service level. It under-protects the")
log("erratic lines and over-protects the steady ones at the same time.")

log("\nService spread within each ABC class under the flat rule:")
log(base.groupby("ABC").CycleService.describe()[
    ["mean", "25%", "50%", "75%"]].round(3).to_string())

# ----------------------------------------------------------------------
# Head-to-head.
# ----------------------------------------------------------------------
def summarise(r, label):
    return pd.Series({
        "SafetyStockValue": r.SafetyStockValue.sum(),
        "AvgInventoryValue": r.InventoryValue.sum(),
        "AnnualHoldingCost": r.HoldingCost.sum(),
        "AnnualUnitsShort": r.AnnualUnitsShort.sum(),
        "AnnualRevenueAtRisk": r.AnnualRevenueAtRisk.sum(),
        "WeightedFillRate": np.average(r.FillRate.fillna(0),
                                       weights=stats.Revenue),
    }, name=label)


head = pd.concat([summarise(base, "Flat 4-week rule"),
                  summarise(prop, "ABC-differentiated")], axis=1)
head["Change"] = head.iloc[:, 1] - head.iloc[:, 0]
head["Change%"] = head.Change / head.iloc[:, 0] * 100

log("\n--- HEAD TO HEAD (annual, GBP unless stated) ---")
log(head.round(2).to_string())

log("\n--- WHERE THE BUFFER MOVES ---")
move = pd.DataFrame({
    "Flat rule": base.groupby("ABC").SafetyStockValue.sum(),
    "Differentiated": prop.groupby("ABC").SafetyStockValue.sum(),
})
move["Change"] = move.Differentiated - move["Flat rule"]
move["Change%"] = move.Change / move["Flat rule"] * 100
move["RevenueShare%"] = (stats.groupby("ABC").Revenue.sum()
                         / stats.Revenue.sum() * 100)
log(move.round(1).to_string())

log("\n--- SERVICE OUTCOME BY CLASS ---")
svc = pd.DataFrame({
    "Flat rule": base.groupby("ABC").apply(
        lambda g: np.average(g.FillRate.fillna(0), weights=stats.loc[g.index, "Revenue"])),
    "Differentiated": prop.groupby("ABC").apply(
        lambda g: np.average(g.FillRate.fillna(0), weights=stats.loc[g.index, "Revenue"])),
})
log((svc * 100).round(2).to_string())

log("\n--- REVENUE AT RISK FROM STOCKOUTS, BY CLASS ---")
risk = pd.DataFrame({
    "Flat rule": base.groupby("ABC").AnnualRevenueAtRisk.sum(),
    "Differentiated": prop.groupby("ABC").AnnualRevenueAtRisk.sum(),
})
risk["Avoided"] = risk["Flat rule"] - risk["Differentiated"]
log(risk.round(0).to_string())

# ----------------------------------------------------------------------
# Equal-budget test. The fairest question a sceptical manager asks: if I spend
# exactly what I spend today, does reallocating it buy me more service?
# ----------------------------------------------------------------------
budget = base.SafetyStockValue.sum()
lo, hi = 0.50, 0.9999
for _ in range(60):
    mid = (lo + hi) / 2
    # Hold the gaps between classes fixed, slide the whole scheme up or down.
    offsets = {"A": 0.0, "B": -0.03, "C": -0.08}
    z_mid = stats.ABC.map(lambda c: norm.ppf(np.clip(mid + offsets[c], 0.5, 0.9999)))
    ss_mid = z_mid * stats.StdWeekly * np.sqrt(LEAD_TIME_WEEKS)
    val = (ss_mid.clip(lower=0) * stats.UnitCost).sum()
    if val < budget:
        lo = mid
    else:
        hi = mid
z_eq = stats.ABC.map(lambda c: norm.ppf(np.clip(lo + {"A": 0.0, "B": -0.03, "C": -0.08}[c],
                                                0.5, 0.9999)))
eq = evaluate(stats, z_eq * stats.StdWeekly * np.sqrt(LEAD_TIME_WEEKS), eco,
              LEAD_TIME_WEEKS)
log("\n--- EQUAL-BUDGET REALLOCATION TEST ---")
log(f"Safety stock budget held at GBP {budget:,.0f}")
log(f"Implied A-class service    : {lo:.3f}")
log(f"Weighted fill rate, flat   : "
    f"{np.average(base.FillRate.fillna(0), weights=stats.Revenue):.4%}")
log(f"Weighted fill rate, realloc: "
    f"{np.average(eq.FillRate.fillna(0), weights=stats.Revenue):.4%}")
log(f"Revenue at risk, flat      : GBP {base.AnnualRevenueAtRisk.sum():,.0f}")
log(f"Revenue at risk, realloc   : GBP {eq.AnnualRevenueAtRisk.sum():,.0f}")

# ----------------------------------------------------------------------
# Sensitivity.
# ----------------------------------------------------------------------
rows = []
for oc in [5, 10, 25]:
    for hr in [0.15, 0.25, 0.35]:
        for lt in [1.0, 2.0, 4.0]:
            e2 = economics(stats, oc, hr, GROSS_MARGIN)
            b2 = evaluate(stats, BASELINE_SS_WEEKS * stats.MeanWeekly, e2, lt)
            p2 = evaluate(stats, z_target * stats.StdWeekly * np.sqrt(lt), e2, lt)
            rows.append({
                "OrderCost": oc, "HoldingRate": hr, "LeadTimeWeeks": lt,
                "SS_Flat": b2.SafetyStockValue.sum(),
                "SS_Diff": p2.SafetyStockValue.sum(),
                "SSChange%": (p2.SafetyStockValue.sum() / b2.SafetyStockValue.sum() - 1) * 100,
                "Fill_Flat": np.average(b2.FillRate.fillna(0), weights=stats.Revenue) * 100,
                "Fill_Diff": np.average(p2.FillRate.fillna(0), weights=stats.Revenue) * 100,
                "RiskAvoided": b2.AnnualRevenueAtRisk.sum() - p2.AnnualRevenueAtRisk.sum(),
            })
sens = pd.DataFrame(rows)
sens["FillGain_pp"] = sens.Fill_Diff - sens.Fill_Flat
log("\n--- SENSITIVITY (27 scenarios) ---")
log(f"Safety stock change range  : {sens['SSChange%'].min():.1f}% to "
    f"{sens['SSChange%'].max():.1f}%")
log(f"Fill rate gain range       : {sens.FillGain_pp.min():.2f}pp to "
    f"{sens.FillGain_pp.max():.2f}pp")
log(f"Scenarios improving fill   : {(sens.FillGain_pp > 0).sum()} of {len(sens)}")
log(f"Scenarios avoiding risk    : {(sens.RiskAvoided > 0).sum()} of {len(sens)}")

# ----------------------------------------------------------------------
# Charts.
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.6))
colors = {"A": "#2b6cb0", "B": "#dd6b20", "C": "#718096"}
for cls in ["C", "B", "A"]:
    sub = stats[stats.ABC == cls]
    ax.scatter(sub.Revenue, sub.CV, s=7, alpha=0.45, label=f"Class {cls}",
               color=colors[cls], edgecolors="none")
ax.set_xscale("log")
ax.axhline(q33, ls="--", lw=0.9, color="black")
ax.axhline(q67, ls="--", lw=0.9, color="black")
ax.set_xlabel("Annual revenue, GBP (log scale)")
ax.set_ylabel("Coefficient of variation of weekly demand")
ax.set_title("ABC-XYZ position of every stocked SKU")
ax.set_ylim(0, 4)
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(FIG, "fig04_abc_xyz_scatter.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 3.2))
w = 0.38
x = np.arange(3)
ax.bar(x - w / 2, move["Flat rule"] / 1000, w, label="Flat 4-week rule",
       color="#718096")
ax.bar(x + w / 2, move["Differentiated"] / 1000, w, label="ABC-differentiated",
       color="#2b6cb0")
ax.set_xticks(x)
ax.set_xticklabels(["Class A", "Class B", "Class C"])
ax.set_ylabel("Safety stock value (GBP '000)")
ax.set_title("Where the safety stock investment sits")
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(FIG, "fig06_buffer_reallocation.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.hist(cs * 100, bins=40, color="#718096")
ax.axvline(95, ls="--", lw=1.2, color="#c05621", label="Nominal 95% target")
ax.set_xlabel("Cycle service level implied by the flat 4-week rule (%)")
ax.set_ylabel("Number of SKUs")
ax.set_title(f"One flat rule, {len(cs):,} different service levels")
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(FIG, "fig05_implied_service.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.bar(x - w / 2, risk["Flat rule"] / 1000, w, label="Flat 4-week rule",
       color="#718096")
ax.bar(x + w / 2, risk["Differentiated"] / 1000, w, label="ABC-differentiated",
       color="#2b6cb0")
ax.set_xticks(x)
ax.set_xticklabels(["Class A", "Class B", "Class C"])
ax.set_ylabel("Annual revenue at risk (GBP '000)")
ax.set_title("Revenue exposed to stockouts")
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(FIG, "fig07_revenue_at_risk.png"))
plt.close(fig)

# ----------------------------------------------------------------------
# Persist.
# ----------------------------------------------------------------------
out = prop.copy()
out["Description"] = stats.Description
out["AnnualUnits"] = stats.AnnualUnits
out["MeanWeekly"] = stats.MeanWeekly.round(2)
out["StdWeekly"] = stats.StdWeekly.round(2)
out["CV"] = stats.CV.round(3)
out["TargetService"] = stats.ABC.map(SERVICE_BY_CLASS)
out["SS_FlatRule"] = base.SafetyStock
cols = ["Description", "ABC", "XYZ", "Segment", "AnnualUnits", "MeanWeekly",
        "StdWeekly", "CV", "UnitCost", "EOQ", "TargetService", "SafetyStock",
        "SS_FlatRule", "ROP", "AvgInventory", "FillRate", "AnnualUnitsShort",
        "HoldingCost"]
out[cols].round(3).to_csv(os.path.join(OUT, "sku_policy.csv"))
matrix.to_csv(os.path.join(OUT, "abc_xyz_matrix.csv"))
sens.round(3).to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
stats.to_csv(os.path.join(OUT, "sku_stats.csv"))

log("\n--- SAMPLE OF THE PROPOSED POLICY: TOP 10 CLASS A SKUs ---")
top = out[out.ABC == "A"].nlargest(10, "AnnualUnits")[
    ["Description", "Segment", "AnnualUnits", "CV", "EOQ", "SafetyStock",
     "SS_FlatRule", "ROP"]]
log(top.round(2).to_string())

with open(os.path.join(OUT, "02_inventory.txt"), "w") as fh:
    fh.write("\n".join(report))

log("\nSaved sku_policy.csv, abc_xyz_matrix.csv, sensitivity.csv and 4 figures.")
