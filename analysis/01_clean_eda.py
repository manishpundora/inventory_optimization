"""
Step 1 - Load, profile and clean the Online Retail II dataset.

Dataset: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CG6D  (CC BY 4.0)

Outputs:
    output/01_profile.txt      - data quality report (before/after cleaning)
    output/clean_sales.csv     - cleaned transaction lines used by later steps
    figures/*.png              - exploratory charts
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "online_retail_II.xlsx")
OUT = os.path.join(BASE, "output")
FIG = os.path.join(BASE, "figures")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

report = []


def log(line=""):
    print(line)
    report.append(str(line))


# ----------------------------------------------------------------------
# Load. The workbook ships as two sheets, one per trading year.
# ----------------------------------------------------------------------
sheets = pd.read_excel(DATA, sheet_name=None, engine="openpyxl")
log(f"Sheets found: {list(sheets)}")
for name, frame in sheets.items():
    log(f"  {name}: {frame.shape[0]:,} rows x {frame.shape[1]} cols")

raw = pd.concat(sheets.values(), ignore_index=True)
raw.columns = [str(c).strip().replace(" ", "") for c in raw.columns]
raw = raw.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice",
                          "CustomerID": "CustomerID", "Customer ID": "CustomerID"})
log(f"\nCombined: {raw.shape[0]:,} rows")
log(f"Columns: {list(raw.columns)}")

# ----------------------------------------------------------------------
# Profile the raw data before touching anything.
# ----------------------------------------------------------------------
log("\n--- RAW DATA QUALITY ---")
log(f"Date range            : {raw.InvoiceDate.min()} to {raw.InvoiceDate.max()}")
log(f"Missing Description   : {raw.Description.isna().sum():,}")
log(f"Missing CustomerID    : {raw.CustomerID.isna().sum():,} "
    f"({raw.CustomerID.isna().mean():.1%})")
log(f"Exact duplicate rows  : {raw.duplicated().sum():,}")
log(f"Quantity <= 0         : {(raw.Quantity <= 0).sum():,}")
log(f"UnitPrice <= 0        : {(raw.UnitPrice <= 0).sum():,}")

raw["InvoiceNo"] = raw.InvoiceNo.astype(str)
cancels = raw.InvoiceNo.str.upper().str.startswith("C")
log(f"Cancellation lines    : {cancels.sum():,} ({cancels.mean():.2%} of rows)")
log(f"Distinct StockCodes   : {raw.StockCode.nunique():,}")
log(f"Distinct countries    : {raw.Country.nunique()}")

# Non-product codes: charges, postage, samples, adjustments. These are real
# ledger entries but they are not stock, so they must not enter an inventory
# study. Product codes in this dataset are 5 digits, sometimes with a letter
# suffix (e.g. 22423, 85123A).
raw["StockCode"] = raw.StockCode.astype(str).str.strip().str.upper()
is_product = raw.StockCode.str.match(r"^\d{5}[A-Z]*$")
log("\nTop non-product codes removed:")
log(raw.loc[~is_product, "StockCode"].value_counts().head(15).to_string())

# ----------------------------------------------------------------------
# Clean.
# ----------------------------------------------------------------------
df = raw.copy()
steps = [("start", len(df))]

df = df.drop_duplicates()
steps.append(("drop exact duplicates", len(df)))

df = df[~df.InvoiceNo.str.upper().str.startswith("C")]
steps.append(("drop cancellations", len(df)))

df = df[is_product.reindex(df.index, fill_value=False)]
steps.append(("keep product codes only", len(df)))

df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]
steps.append(("drop non-positive qty/price", len(df)))

df = df[df.Description.notna()]
steps.append(("drop missing description", len(df)))

log("\n--- CLEANING AUDIT TRAIL ---")
prev = steps[0][1]
for label, n in steps:
    log(f"{label:<32} {n:>10,}  ({n - prev:+,})")
    prev = n
log(f"Retained {len(df) / len(raw):.1%} of original rows")

# Derived fields used throughout the rest of the study.
df["InvoiceDate"] = pd.to_datetime(df.InvoiceDate)
df["Revenue"] = df.Quantity * df.UnitPrice
df["Date"] = df.InvoiceDate.dt.normalize()
df["YearMonth"] = df.InvoiceDate.dt.to_period("M").astype(str)

# One SKU can carry several spellings of its description across two years.
# Take the most frequent one so later tables read cleanly.
canonical = (df.groupby(["StockCode", "Description"]).size()
               .reset_index(name="n")
               .sort_values("n", ascending=False)
               .drop_duplicates("StockCode")
               .set_index("StockCode")["Description"])
df["Description"] = df.StockCode.map(canonical)

log("\n--- CLEANED DATASET ---")
log(f"Rows                  : {len(df):,}")
log(f"Date range            : {df.Date.min().date()} to {df.Date.max().date()} "
    f"({(df.Date.max() - df.Date.min()).days} days)")
log(f"Distinct SKUs         : {df.StockCode.nunique():,}")
log(f"Distinct invoices     : {df.InvoiceNo.nunique():,}")
log(f"Distinct customers    : {df.CustomerID.nunique():,}")
log(f"Total revenue         : GBP {df.Revenue.sum():,.0f}")
log(f"Median line value     : GBP {df.Revenue.median():,.2f}")

log("\nRevenue share by country (top 10):")
country = (df.groupby("Country").Revenue.sum()
             .sort_values(ascending=False) / df.Revenue.sum())
log((country.head(10) * 100).round(2).to_string())

# ----------------------------------------------------------------------
# Exploratory charts.
# ----------------------------------------------------------------------
monthly = df.groupby("YearMonth").Revenue.sum()
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(range(len(monthly)), monthly.values / 1000, marker="o", lw=1.4, ms=3.5,
        color="#2b6cb0")
ax.set_xticks(range(0, len(monthly), 2))
ax.set_xticklabels(monthly.index[::2], rotation=45, ha="right")
ax.set_ylabel("Revenue (GBP '000)")
ax.set_title("Monthly revenue, Dec 2009 - Dec 2011")
fig.savefig(os.path.join(FIG, "fig01_monthly_revenue.png"))
plt.close(fig)

sku_rev = df.groupby("StockCode").Revenue.sum().sort_values(ascending=False)
cum = sku_rev.cumsum() / sku_rev.sum()
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum.values * 100,
        lw=1.8, color="#2b6cb0")
ax.axhline(80, ls="--", lw=1, color="#c05621")
ax.axvline(20, ls="--", lw=1, color="#c05621")
ax.set_xlabel("Cumulative share of SKUs (%)")
ax.set_ylabel("Cumulative share of revenue (%)")
ax.set_title("Revenue concentration across SKUs")
fig.savefig(os.path.join(FIG, "fig03_pareto.png"))
plt.close(fig)

share_top20 = cum.iloc[int(len(cum) * 0.2)] * 100
log(f"\nTop 20% of SKUs account for {share_top20:.1f}% of revenue")

dow = df.groupby(df.InvoiceDate.dt.dayofweek).Revenue.sum()
labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar([labels[i] for i in dow.index], dow.values / 1000, color="#2b6cb0")
ax.set_ylabel("Revenue (GBP '000)")
ax.set_title("Revenue by day of week")
fig.savefig(os.path.join(FIG, "fig02_day_of_week.png"))
plt.close(fig)

log("\nRevenue by weekday (GBP):")
log(pd.Series(dow.values, index=[labels[i] for i in dow.index]).round(0).to_string())

# ----------------------------------------------------------------------
# Persist.
# ----------------------------------------------------------------------
cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
        "Date", "YearMonth", "UnitPrice", "Revenue", "CustomerID", "Country"]
df[cols].to_csv(os.path.join(OUT, "clean_sales.csv"), index=False)

with open(os.path.join(OUT, "01_profile.txt"), "w") as fh:
    fh.write("\n".join(report))

log(f"\nSaved clean_sales.csv ({len(df):,} rows) and 3 figures.")
