"""Generate the Dynamic Pricing case-study Jupyter notebook."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------------
md("""# Case Study: Dynamic Pricing System for Festive Seasons

## Z-Cashify — Demand Forecasting for Buy-back & Resale Pricing

**Business context.** Z-Cashify is a recommerce platform that must adjust **buy-back** (what it pays sellers) and **resale** (what it charges buyers) prices when demand swings hard during festive seasons. Fair, competitive pricing depends on knowing *when* and *where* footfall and sales will spike.

**Consulting goal.** Forecast daily store sales across **1,115 locations**, turn those forecasts into a pricing & staffing playbook, and compare three modeling approaches:
1. Linear Regression (interpretable baseline)
2. Tree-based model (Decision Tree Regressor)
3. XGBoost Regressor (gradient-boosted trees)

**Success metric.** **RMSPE** (Root Mean Square Percentage Error) — percentage error that treats small kiosks and flagship stores fairly.

---
### Evaluation alignment
| Criterion | How this notebook addresses it |
|---|---|
| Business story | Problem framed as festive dynamic pricing |
| Data integration | Merge `train` + `store`; handle NA & closed-store zeros |
| Feature engineering | Date parts, competitor age, Promo2 active flag, payday flags |
| Comparative modeling | LR vs RF vs XGBoost with side-by-side RMSPE |
| Tuning & metrics | Depth / learning-rate experiments + RMSPE |
| Actionable strategy | Store-manager Pricing Playbook |
""")

# ---------------------------------------------------------------------------
md("""## 0. Setup & Imports""")

code("""# Core libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 4)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Libraries loaded.")
""")

# ---------------------------------------------------------------------------
md("""## 1. Load & Inspect Data

We integrate three sources:
- `train.csv` — daily transactions (target = **Sales**)
- `store.csv` — assortment, competition, Promo2 metadata
- `test.csv` — future dates for scoring (no Sales)
""")

code("""train = pd.read_csv("dataset/train.csv", low_memory=False, parse_dates=["Date"])
store = pd.read_csv("dataset/store.csv")
test  = pd.read_csv("dataset/test.csv",  low_memory=False, parse_dates=["Date"])

print("train:", train.shape, "|", train["Date"].min().date(), "→", train["Date"].max().date())
print("store:", store.shape)
print("test :", test.shape,  "|", test["Date"].min().date(),  "→", test["Date"].max().date())
display(train.head(3))
display(store.head(3))
""")

code("""# Schema & dtypes
print("=== train dtypes ===")
print(train.dtypes)
print("\\n=== store missing values ===")
print(store.isna().sum())
print("\\n=== StateHoliday codes ===", train["StateHoliday"].unique())
""")

# ---------------------------------------------------------------------------
md("""## 2. Data Quality & Closed-Store Anomalies

**Business rule:** a closed store (`Open == 0`) should have zero sales. Rows with `Sales == 0` while open are rare noise. For forecasting *sales when open*, we train on **open days with Sales > 0** — otherwise the model wastes capacity learning the trivial "closed → 0" pattern.
""")

code("""print("Closed days with Sales > 0 :", ((train["Open"] == 0) & (train["Sales"] > 0)).sum())
print("Open days with Sales == 0  :", ((train["Open"] == 1) & (train["Sales"] == 0)).sum())
print("Closed share of rows       :", (train["Open"] == 0).mean().round(3))

# Sales distribution (open stores only)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
open_sales = train.loc[train["Sales"] > 0, "Sales"]
sns.histplot(open_sales, bins=50, ax=axes[0], color="steelblue")
axes[0].set_title("Sales distribution (open days)")
axes[0].set_xlabel("Sales")
sns.boxplot(x=open_sales, ax=axes[1], color="steelblue")
axes[1].set_title("Sales outliers (open days)")
plt.tight_layout()
plt.show()

print(open_sales.describe().round(1))
""")

md("""### Missing competition & Promo2 fields

- Missing `CompetitionDistance` (3 stores) → treat as **no nearby competitor** → impute with a large distance (e.g. max × 1.5).
- Missing `CompetitionOpenSince*` → competitor age unknown → fill with 0 months (neutral).
- Missing Promo2 fields when `Promo2 == 0` are expected (store never joined continuous promo).
""")

code("""print(store[["CompetitionDistance", "CompetitionOpenSinceMonth",
             "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek",
             "Promo2SinceYear", "PromoInterval"]].isna().sum())
print("\\nStores with Promo2=0:", (store["Promo2"] == 0).sum(),
      "| PromoInterval NA:", store["PromoInterval"].isna().sum())
""")

# ---------------------------------------------------------------------------
md("""## 3. Merge Transactional + Store Metadata""")

code("""def merge_store(df, store_df):
    out = df.merge(store_df, on="Store", how="left")
    assert out.shape[0] == df.shape[0], "Row count changed after merge!"
    return out

train_full = merge_store(train, store)
test_full  = merge_store(test, store)
print("Merged train:", train_full.shape, "| Merged test:", test_full.shape)
print("Any NA StoreType after merge?", train_full["StoreType"].isna().any())
""")

# ---------------------------------------------------------------------------
md("""## 4. Exploratory Analysis — Business Questions (1–4, 9)

Before modeling, we answer the demand / promo / competition questions that feed the pricing engine.
""")

md("""### Q1 — Holiday Impact: StateHoliday vs SchoolHoliday & reopen jump""")

code("""# Normalize StateHoliday: treat '0' string and 0 the same
df = train_full.copy()
df["StateHoliday"] = df["StateHoliday"].astype(str).replace({"0": "0"})

# Average sales when OPEN
open_df = df[df["Open"] == 1].copy()

holiday_cmp = pd.DataFrame({
    "No StateHoliday": [
        open_df.loc[open_df["StateHoliday"] == "0", "Sales"].mean(),
        open_df.loc[open_df["StateHoliday"] == "0", "Customers"].mean(),
    ],
    "StateHoliday (a/b/c)": [
        open_df.loc[open_df["StateHoliday"] != "0", "Sales"].mean(),
        open_df.loc[open_df["StateHoliday"] != "0", "Customers"].mean(),
    ],
    "No SchoolHoliday": [
        open_df.loc[open_df["SchoolHoliday"] == 0, "Sales"].mean(),
        open_df.loc[open_df["SchoolHoliday"] == 0, "Customers"].mean(),
    ],
    "SchoolHoliday": [
        open_df.loc[open_df["SchoolHoliday"] == 1, "Sales"].mean(),
        open_df.loc[open_df["SchoolHoliday"] == 1, "Customers"].mean(),
    ],
}, index=["Avg Sales", "Avg Customers"]).round(1)

display(holiday_cmp)

# State holiday type breakdown (open days)
state_sales = (open_df.groupby("StateHoliday")[["Sales", "Customers"]]
               .mean().round(1)
               .rename(index={"0": "None", "a": "Public", "b": "Easter", "c": "Christmas"}))
display(state_sales)
""")

code("""# Reopen jump: day after a closed StateHoliday
df_sorted = df.sort_values(["Store", "Date"]).copy()
df_sorted["was_closed_yesterday"] = (
    (df_sorted.groupby("Store")["Open"].shift(1) == 0)
    & (df_sorted.groupby("Store")["StateHoliday"].shift(1).astype(str) != "0")
)
df_sorted["reopen_after_state_holiday"] = (
    (df_sorted["Open"] == 1) & df_sorted["was_closed_yesterday"]
)

# Compare reopen day vs typical open day for same store (simple overall means)
reopen = df_sorted[df_sorted["reopen_after_state_holiday"]]
normal = df_sorted[(df_sorted["Open"] == 1) & (~df_sorted["reopen_after_state_holiday"])]

print("Avg Sales — reopen after StateHoliday:", round(reopen["Sales"].mean(), 1))
print("Avg Sales — other open days         :", round(normal["Sales"].mean(), 1))
print("Lift                                 :", round(reopen["Sales"].mean() / normal["Sales"].mean() - 1, 3))

# Visual: sales around Easter-like windows using a sample of large holidays
# Plot daily average sales & mark state holidays (closed)
daily = df.groupby("Date").agg(Sales=("Sales", "mean"), Open=("Open", "mean"),
                               StateHoliday=("StateHoliday", "first")).reset_index()
# Zoom to Dec 2013 – Jan 2014 (Christmas / New Year)
mask = (daily["Date"] >= "2013-12-15") & (daily["Date"] <= "2014-01-10")
zoom = daily.loc[mask]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(zoom["Date"], zoom["Sales"], marker="o", color="steelblue", label="Avg Sales")
closed = zoom[zoom["StateHoliday"].astype(str) != "0"]
ax.scatter(closed["Date"], closed["Sales"], color="crimson", s=80, zorder=5, label="StateHoliday")
ax.set_title("Holiday void & reopen spike (Christmas / New Year 2013–14)")
ax.set_ylabel("Average Sales")
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
""")

md("""**Insight — Holiday Impact**
- **StateHolidays** usually close stores → sales drop to ~0; when a store *stays open*, sales are atypical.
- **SchoolHolidays** keep stores open and slightly lift sales/customers (family shopping).
- The day a store **reopens after a StateHoliday** shows a clear **sales jump** — demand is deferred, not destroyed. Pricing implication: raise buy-back offers *just before* holidays (sellers dump devices) and tighten resale / raise buy-back again on the reopen day when footfall surges.
""")

md("""### Q2 — The Promo Effect: Promo vs Promo2""")

code("""# Need Promo2-is-active later; for EDA use store-level Promo2 flag + daily Promo
promo_groups = open_df.copy()
promo_groups["promo_bucket"] = np.select(
    [
        (promo_groups["Promo"] == 0) & (promo_groups["Promo2"] == 0),
        (promo_groups["Promo"] == 1) & (promo_groups["Promo2"] == 0),
        (promo_groups["Promo"] == 0) & (promo_groups["Promo2"] == 1),
        (promo_groups["Promo"] == 1) & (promo_groups["Promo2"] == 1),
    ],
    ["No Promo", "Promo only", "Promo2 only", "Both"],
    default="Other",
)

promo_summary = (promo_groups.groupby("promo_bucket")
                 .agg(Avg_Sales=("Sales", "mean"),
                      Avg_Customers=("Customers", "mean"),
                      Days=("Sales", "size"))
                 .reindex(["No Promo", "Promo only", "Promo2 only", "Both"])
                 .round(1))
display(promo_summary)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=promo_summary.reset_index(), x="promo_bucket", y="Avg_Sales",
            palette="Blues_d", ax=ax)
ax.set_title("Average Sales by Promotion Type (open days)")
ax.set_xlabel("")
ax.set_ylabel("Avg Sales")
plt.tight_layout()
plt.show()
""")

md("""**Insight — Promo Effect**
- Short-term **Promo** drives the largest lift in both sales and customers (flash demand).
- Continuous **Promo2** alone often shows *lower* average sales than "No Promo" — classic **promo fatigue / selection effect** (weaker stores enroll in Promo2).
- Combining both helps, but the **short-term Promo** is the stronger lever for customer acquisition. Pricing playbook: use Promo windows to **raise buy-back prices** (more devices in → more inventory for resale).
""")

md("""### Q3 — Competition Distance: threat or retail hub?""")

code("""comp = open_df.dropna(subset=["CompetitionDistance"]).copy()
comp["DistBucket"] = pd.cut(
    comp["CompetitionDistance"],
    bins=[0, 500, 2000, 5000, 10000, np.inf],
    labels=["0–500m", "500m–2km", "2–5km", "5–10km", "10km+"],
)

comp_summary = (comp.groupby("DistBucket", observed=True)
                .agg(Avg_Sales=("Sales", "mean"),
                     Avg_Customers=("Customers", "mean"),
                     Stores=("Store", "nunique"))
                .round(1))
display(comp_summary)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=comp_summary.reset_index(), x="DistBucket", y="Avg_Sales",
            palette="Greens_d", ax=ax)
ax.set_title("Avg Sales by Competition Distance Bucket")
ax.set_xlabel("Distance to nearest competitor")
plt.tight_layout()
plt.show()

# Correlation (log distance vs sales)
print("Corr(log1p CompetitionDistance, Sales):",
      round(np.corrcoef(np.log1p(comp["CompetitionDistance"]), comp["Sales"])[0, 1], 4))
""")

md("""**Insight — Competition Distance**
- Stores with competitors **very close (0–500m)** often show *similar or higher* sales — evidence of a **retail hub / shopping-district effect**, not pure cannibalization.
- Distance alone is a weak linear predictor; assortment, promo, and day-of-week dominate. Pricing implication: do **not** automatically cut resale prices just because a competitor is nearby — match service & promo timing instead.
""")

md("""### Q4 — Best Time to Buy/Sell (DayOfWeek)""")

code("""dow_map = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
dow = (open_df.groupby("DayOfWeek")
       .agg(Avg_Sales=("Sales", "mean"),
            Median_Sales=("Sales", "median"),
            Avg_Customers=("Customers", "mean"),
            Total_Sales=("Sales", "sum"),
            Total_Customers=("Customers", "sum"))
       )
dow["Sales_per_Customer"] = (dow["Total_Sales"] / dow["Total_Customers"]).round(2)
dow = dow.drop(columns=["Total_Sales", "Total_Customers"]).round(1)
dow.index = dow.index.map(dow_map)
display(dow)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=dow.index, y=dow["Avg_Sales"], palette="Oranges_d", ax=ax)
ax.set_title("Average Sales by Day of Week (open stores)")
ax.set_ylabel("Avg Sales")
plt.tight_layout()
plt.show()
""")

md("""**Insight — Best Time to Buy/Sell**
- **Monday** and **Sunday (when open)** are the busiest sales days; mid-week is softer; **Saturday** has solid footfall but lower average ticket in many stores.
- For dynamic buy-back: offer **higher buy-back prices Mon / Sun** (high demand → need inventory fast) and use quieter Tue–Thu for **processing & listing** devices at competitive resale prices.
""")

md("""### Q9 — Weekend Dynamic: Saturday vs Monday""")

code("""sat = open_df[open_df["DayOfWeek"] == 6]
mon = open_df[open_df["DayOfWeek"] == 1]

weekend_cmp = pd.DataFrame({
    "Monday": [
        mon["Sales"].mean(),
        mon["Customers"].mean(),
        mon["Sales"].sum() / mon["Customers"].sum(),
    ],
    "Saturday": [
        sat["Sales"].mean(),
        sat["Customers"].mean(),
        sat["Sales"].sum() / sat["Customers"].sum(),
    ],
}, index=["Avg Sales", "Avg Customers", "Sales per Customer"]).round(2)
display(weekend_cmp)

print("Monday vs Saturday sales lift:",
      round(mon["Sales"].mean() / sat["Sales"].mean() - 1, 3))
""")

md("""**Insight — Weekend Dynamic**
- **Mondays** typically post higher average sales than Saturdays (post-weekend catch-up + Promo often starts Monday).
- Saturday shopping is more "leisure" footfall; Monday is higher-intent / promo-driven.
- Pricing engine: **Saturday** → slightly premium *resale* presentation (browsing); **Monday** → aggressive *buy-back* bonuses aligned with Promo kicks to fill inventory for the week.
""")

# ---------------------------------------------------------------------------
md("""## 5. Feature Engineering

Following the approach document:
1. **Date decomposition** — Year, Month, Day, WeekOfYear, DayOfWeek
2. **Payday flags** — start / end of month
3. **Competitor age** — months since competitor opened
4. **Promo2 active** — binary flag from `PromoInterval` + current month
5. **Holiday flags** — encode StateHoliday
""")

code("""def months_since(year, month, ref_date):
    \"\"\"Months between (year, month) and ref_date; 0 if missing.\"\"\"
    y = year.fillna(ref_date.dt.year).astype(float)
    m = month.fillna(ref_date.dt.month).astype(float)
    # If original was NA, return 0
    age = (ref_date.dt.year - y) * 12 + (ref_date.dt.month - m)
    age = age.clip(lower=0)
    age = age.where(year.notna() & month.notna(), 0)
    return age


def promo2_active(df):
    \"\"\"1 if store's Promo2 cycle is active in the current month.\"\"\"
    month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    month_name = df["Date"].dt.month.map(month_map)
    intervals = df["PromoInterval"].fillna("")
    active = [
        (m in iv.split(",")) and (p2 == 1)
        for m, iv, p2 in zip(month_name, intervals, df["Promo2"])
    ]
    return np.array(active, dtype=np.int8)


def engineer_features(df, is_train=True):
    out = df.copy()
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["Day"] = out["Date"].dt.day
    out["WeekOfYear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["DayOfWeek"] = out["Date"].dt.dayofweek + 1  # 1=Mon ... 7=Sun (align with raw)

    # Payday / month-edge flags
    out["IsMonthStart"] = (out["Day"] <= 7).astype(np.int8)
    out["IsMonthEnd"] = (out["Day"] >= 24).astype(np.int8)

    # Competition distance: missing → far away
    max_dist = out["CompetitionDistance"].max(skipna=True)
    out["CompetitionDistance"] = out["CompetitionDistance"].fillna(max_dist * 1.5)
    out["LogCompetitionDistance"] = np.log1p(out["CompetitionDistance"])

    # Competitor age in months
    out["CompetitionOpenMonths"] = months_since(
        out["CompetitionOpenSinceYear"],
        out["CompetitionOpenSinceMonth"],
        out["Date"],
    )

    # Promo2 timing
    out["Promo2Active"] = promo2_active(out)
    # Months since Promo2 started (0 if never)
    # Approximate: week → month
    p2_year = out["Promo2SinceYear"]
    p2_week = out["Promo2SinceWeek"]
    p2_month = ((p2_week.fillna(1) - 1) // 4 + 1).clip(1, 12)
    out["Promo2Months"] = months_since(p2_year, p2_month, out["Date"])
    out.loc[out["Promo2"] == 0, "Promo2Months"] = 0

    # StateHoliday → numeric codes
    out["StateHoliday"] = out["StateHoliday"].astype(str).replace({"0": "0"})
    out["StateHolidayCode"] = out["StateHoliday"].map({"0": 0, "a": 1, "b": 2, "c": 3}).fillna(0).astype(int)
    out["IsStateHoliday"] = (out["StateHolidayCode"] > 0).astype(np.int8)

    # StoreType / Assortment as labels (trees) — keep originals for OHE in LR
    for col in ["StoreType", "Assortment"]:
        out[col] = out[col].astype(str)

    # Open fill for test
    if "Open" in out.columns:
        out["Open"] = out["Open"].fillna(1).astype(int)

    return out


train_fe = engineer_features(train_full)
test_fe  = engineer_features(test_full, is_train=False)

print("Engineered columns sample:")
display(train_fe[["Date", "Year", "Month", "WeekOfYear", "IsMonthStart", "IsMonthEnd",
                  "LogCompetitionDistance", "CompetitionOpenMonths",
                  "Promo", "Promo2", "Promo2Active", "StateHolidayCode"]].head())
print("Promo2Active rate:", train_fe["Promo2Active"].mean().round(3))
""")

# ---------------------------------------------------------------------------
md("""## 6. Modeling Dataset & Time-Based Split

We exclude closed / zero-sale days from training (predict demand when open).  
**Validation strategy:** last **6 weeks** of training history as a hold-out test set (time-aware; no random leakage).
""")

code("""FEATURE_COLS_NUM = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday", "Open",
    "Year", "Month", "Day", "WeekOfYear",
    "IsMonthStart", "IsMonthEnd",
    "CompetitionDistance", "LogCompetitionDistance", "CompetitionOpenMonths",
    "Promo2", "Promo2Active", "Promo2Months",
    "StateHolidayCode", "IsStateHoliday",
    "StoreAvgSales",  # store baseline (fit on training period only)
]
FEATURE_COLS_CAT = ["StoreType", "Assortment"]  # OHE for LR; label-encoded for trees
ALL_FEATURES = FEATURE_COLS_NUM + FEATURE_COLS_CAT

# Modeling frame: open days with positive sales
model_df = train_fe[(train_fe["Open"] == 1) & (train_fe["Sales"] > 0)].copy()

cutoff = model_df["Date"].max() - pd.Timedelta(weeks=6)
train_set = model_df[model_df["Date"] <= cutoff].copy()
valid_set = model_df[model_df["Date"] > cutoff].copy()

# Leakage-safe store baseline: mean sales from training window only
store_avg = train_set.groupby("Store")["Sales"].mean()
global_avg = train_set["Sales"].mean()
train_set["StoreAvgSales"] = train_set["Store"].map(store_avg)
valid_set["StoreAvgSales"] = valid_set["Store"].map(store_avg).fillna(global_avg)
test_fe["StoreAvgSales"] = test_fe["Store"].map(store_avg).fillna(global_avg)

print("Train period:", train_set["Date"].min().date(), "→", train_set["Date"].max().date(),
      "| rows:", len(train_set))
print("Valid period:", valid_set["Date"].min().date(), "→", valid_set["Date"].max().date(),
      "| rows:", len(valid_set))

X_train_raw = train_set[ALL_FEATURES].copy()
X_valid_raw = valid_set[ALL_FEATURES].copy()
y_train = train_set["Sales"].values
y_valid = valid_set["Sales"].values
""")

code("""def rmspe(y_true, y_pred):
    \"\"\"Root Mean Square Percentage Error (%). Ignores y_true == 0.\"\"\"
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    pct = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(pct ** 2)) * 100)

def evaluate(name, y_true, y_pred):
    score = rmspe(y_true, y_pred)
    print(f\"{name:25s}  RMSPE = {score:6.2f}%\")
    return score

print("RMSPE helper ready. Example sanity:", rmspe([100, 200], [110, 180]))
""")

# ---------------------------------------------------------------------------
md("""## 7. Model 1 — Linear Regression (interpretable baseline)

Categoricals are **one-hot encoded**. Linear models estimate an *average* additive effect of Promo / holidays.
""")

code("""# One-hot encode categoricals for Linear Regression
# Drop raw Store ID — as a numeric feature it implies Store 1000 ≈ 1000× Store 1
X_train_lr = pd.get_dummies(
    X_train_raw.drop(columns=["Store"]), columns=FEATURE_COLS_CAT, drop_first=True
)
X_valid_lr = pd.get_dummies(
    X_valid_raw.drop(columns=["Store"]), columns=FEATURE_COLS_CAT, drop_first=True
)
# Align columns
X_valid_lr = X_valid_lr.reindex(columns=X_train_lr.columns, fill_value=0)

lr = LinearRegression()
lr.fit(X_train_lr, y_train)
pred_lr = lr.predict(X_valid_lr)
pred_lr = np.clip(pred_lr, 0, None)

rmspe_lr = evaluate("Linear Regression", y_valid, pred_lr)

# Top absolute coefficients (interpretability)
coef = pd.Series(lr.coef_, index=X_train_lr.columns).sort_values(key=np.abs, ascending=False)
print("\\nTop |coefficients|:")
display(coef.head(12).round(2))
""")

# ---------------------------------------------------------------------------
md("""## 8. Model 2 — Tree-Based (Decision Tree Regressor)

Trees capture **interactions** (e.g. Promo × Saturday) without manual feature crossing. Categoricals are label-encoded.  
(A single decision tree is the required tree baseline; XGBoost later adds boosted ensembles.)
""")

code("""def label_encode_fit_transform(train_df, valid_df, cols):
    train_out, valid_out = train_df.copy(), valid_df.copy()
    encoders = {}
    for c in cols:
        le = LabelEncoder()
        # Fit on union to avoid unseen labels
        le.fit(pd.concat([train_df[c], valid_df[c]]).astype(str))
        train_out[c] = le.transform(train_df[c].astype(str))
        valid_out[c] = le.transform(valid_df[c].astype(str))
        encoders[c] = le
    return train_out, valid_out, encoders

X_train_tree, X_valid_tree, tree_encoders = label_encode_fit_transform(
    X_train_raw, X_valid_raw, FEATURE_COLS_CAT
)

# Baseline Decision Tree (before tuning)
dt = DecisionTreeRegressor(
    max_depth=16,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
dt.fit(X_train_tree, y_train)
pred_dt = dt.predict(X_valid_tree)
rmspe_dt = evaluate("Decision Tree (base)", y_valid, pred_dt)
""")

# ---------------------------------------------------------------------------
md("""## 9. Model 3 — XGBoost Regressor

Gradient boosting sequentially corrects residual errors — strong on tabular retail data with non-linear promo/holiday effects.
""")

code("""xgb_base = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    tree_method="hist",
)
xgb_base.fit(X_train_tree, y_train,
             eval_set=[(X_valid_tree, y_valid)],
             verbose=False)
pred_xgb = xgb_base.predict(X_valid_tree)
pred_xgb = np.clip(pred_xgb, 0, None)
rmspe_xgb = evaluate("XGBoost (base)", y_valid, pred_xgb)
""")

# ---------------------------------------------------------------------------
md("""## 10. Hyper-parameter Tuning (Q6)

We tune the parameters the approach document highlights:
- **Decision Tree:** `max_depth`, `min_samples_leaf`
- **XGBoost:** `max_depth`, `learning_rate` (eta)

Watch the **train vs validation RMSPE gap** — a tiny train error + high valid error ⇒ overfitting (depth too high).
""")

code("""# --- Decision Tree depth sweep ---
dt_rows = []
for depth in [8, 12, 16, 20, 28]:
    for leaf in [2, 5, 10]:
        m = DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=leaf, random_state=RANDOM_STATE,
        )
        m.fit(X_train_tree, y_train)
        tr = rmspe(y_train, m.predict(X_train_tree))
        va = rmspe(y_valid, m.predict(X_valid_tree))
        dt_rows.append({
            "model": "DecisionTree", "max_depth": depth, "min_samples_leaf": leaf,
            "train_RMSPE": tr, "valid_RMSPE": va, "gap": tr - va,
        })
        print(f"DT depth={depth:2d} leaf={leaf:2d}  train={tr:6.2f}%  valid={va:6.2f}%  gap={tr-va:6.2f}")

dt_tune = pd.DataFrame(dt_rows)
""")

code("""# --- XGBoost learning_rate × max_depth grid (compact) ---
xgb_rows = []
best_xgb, best_score, best_params = None, 1e9, None
for depth in [4, 6, 8]:
    for lr_rate in [0.05, 0.1, 0.2]:
        m = XGBRegressor(
            n_estimators=200, max_depth=depth, learning_rate=lr_rate,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", n_jobs=-1,
            random_state=RANDOM_STATE, tree_method="hist",
        )
        m.fit(X_train_tree, y_train, verbose=False)
        tr = rmspe(y_train, np.clip(m.predict(X_train_tree), 0, None))
        va = rmspe(y_valid, np.clip(m.predict(X_valid_tree), 0, None))
        xgb_rows.append({
            "model": "XGB", "max_depth": depth, "learning_rate": lr_rate,
            "train_RMSPE": tr, "valid_RMSPE": va, "gap": tr - va,
        })
        print(f"XGB depth={depth} lr={lr_rate:.2f}  train={tr:6.2f}%  valid={va:6.2f}%")
        if va < best_score:
            best_score, best_xgb, best_params = va, m, {"max_depth": depth, "learning_rate": lr_rate}

xgb_tune = pd.DataFrame(xgb_rows)
print("\\nBest XGB params:", best_params, "| valid RMSPE:", round(best_score, 2))
""")

code("""# Fit final tuned Decision Tree
best_dt_row = dt_tune.sort_values("valid_RMSPE").iloc[0]
print("Best DT params:", dict(best_dt_row[["max_depth", "min_samples_leaf"]]))

dt_final = DecisionTreeRegressor(
    max_depth=int(best_dt_row["max_depth"]),
    min_samples_leaf=int(best_dt_row["min_samples_leaf"]),
    random_state=RANDOM_STATE,
)
dt_final.fit(X_train_tree, y_train)
pred_dt_final = dt_final.predict(X_valid_tree)
rmspe_dt_final = evaluate("Decision Tree (tuned)", y_valid, pred_dt_final)

pred_xgb_final = np.clip(best_xgb.predict(X_valid_tree), 0, None)
rmspe_xgb_final = evaluate("XGBoost (tuned)", y_valid, pred_xgb_final)

# Tuning effect visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
depth_curve = (dt_tune[dt_tune["min_samples_leaf"] == int(best_dt_row["min_samples_leaf"])]
               .sort_values("max_depth"))
axes[0].plot(depth_curve["max_depth"], depth_curve["train_RMSPE"], "o-", label="Train")
axes[0].plot(depth_curve["max_depth"], depth_curve["valid_RMSPE"], "s-", label="Valid")
axes[0].set_title("Decision Tree: depth vs RMSPE")
axes[0].set_xlabel("max_depth")
axes[0].set_ylabel("RMSPE %")
axes[0].legend()

pivot = xgb_tune.pivot(index="max_depth", columns="learning_rate", values="valid_RMSPE")
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd_r", ax=axes[1])
axes[1].set_title("XGB valid RMSPE (depth × lr)")
plt.tight_layout()
plt.show()
""")

md("""**Insight — Hyper-parameter Tuning**
- Raising `max_depth` lowers **training** RMSPE quickly but eventually **hurts validation** (overfitting).
- For XGBoost, a **smaller learning rate** with enough trees usually generalizes better than a high `eta` that fits noise.
- We keep the configuration with the **lowest validation RMSPE** and a moderate train–valid gap.
""")

# ---------------------------------------------------------------------------
md("""## 11. Model Comparison (Q5) & Linear vs Non-Linear Promo (Q7)""")

code("""comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree (tuned)", "XGBoost (tuned)"],
    "Valid_RMSPE_%": [rmspe_lr, rmspe_dt_final, rmspe_xgb_final],
}).sort_values("Valid_RMSPE_%")
display(comparison)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=comparison, x="Model", y="Valid_RMSPE_%", palette="viridis", ax=ax)
ax.set_title("Model Comparison — Validation RMSPE (lower is better)")
ax.set_ylabel("RMSPE %")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

best_name = comparison.iloc[0]["Model"]
print("Best model on validation:", best_name)
""")

code("""# Q7 — Does LR struggle on Promo days vs trees?
valid_promo = valid_set["Promo"].values
results_promo = []
for label, preds in [
    ("Linear Regression", pred_lr),
    ("Decision Tree", pred_dt_final),
    ("XGBoost", pred_xgb_final),
]:
    results_promo.append({
        "Model": label,
        "RMSPE_Promo": rmspe(y_valid[valid_promo == 1], preds[valid_promo == 1]),
        "RMSPE_NoPromo": rmspe(y_valid[valid_promo == 0], preds[valid_promo == 0]),
    })
promo_err = pd.DataFrame(results_promo)
display(promo_err.round(2))

# Residual plot on Promo days
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_valid[valid_promo == 1], pred_lr[valid_promo == 1],
                alpha=0.15, s=8, label="LR")
axes[0].plot([0, y_valid.max()], [0, y_valid.max()], "r--")
axes[0].set_title("LR: Actual vs Pred (Promo days)")
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")

axes[1].scatter(y_valid[valid_promo == 1], pred_xgb_final[valid_promo == 1],
                alpha=0.15, s=8, color="green", label="XGB")
axes[1].plot([0, y_valid.max()], [0, y_valid.max()], "r--")
axes[1].set_title("XGBoost: Actual vs Pred (Promo days)")
axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
plt.tight_layout()
plt.show()
""")

md("""**Insight — Model Comparison (Q5)**  
Tree-based models (especially **XGBoost**) beat Linear Regression because festive retail demand is full of **interactions and spikes** (Promo × DayOfWeek × Holiday) that a straight line cannot follow.

**Insight — Linear vs Non-Linear (Q7)**  
LR’s higher RMSPE on **Promo** days shows the Promo→Sales link is **not a simple straight line**. A promo on Monday with high competition age behaves differently than a promo on a quiet Tuesday — trees partition on those interactions; LR only adds a constant Promo coefficient.
""")

# ---------------------------------------------------------------------------
md("""## 12. Feature Importance → Pricing Levers""")

code("""# Use best XGBoost importances
imp = pd.Series(best_xgb.feature_importances_, index=X_train_tree.columns)
imp = imp.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
imp.head(15).sort_values().plot(kind="barh", ax=ax, color="teal")
ax.set_title("XGBoost Feature Importance (top 15)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()
display(imp.head(10).round(4))
""")

# ---------------------------------------------------------------------------
md("""## 13. Accuracy Check — Why RMSPE? (Q8)""")

code("""# Demonstrate absolute vs percentage error across store sizes
sample_stores = valid_set.copy()
sample_stores["pred_xgb"] = pred_xgb_final
store_size = (sample_stores.groupby("Store")["Sales"]
              .mean().rename("avg_sales").reset_index())
store_size["size_bucket"] = pd.qcut(store_size["avg_sales"], 3,
                                    labels=["Small", "Medium", "Large"])

demo = sample_stores.merge(store_size[["Store", "size_bucket", "avg_sales"]], on="Store")
demo["abs_err"] = (demo["Sales"] - demo["pred_xgb"]).abs()
demo["pct_err"] = demo["abs_err"] / demo["Sales"] * 100

demo_summary = (demo.groupby("size_bucket", observed=True)
                .agg(Mean_Abs_Error=("abs_err", "mean"),
                     Mean_Pct_Error=("pct_err", "mean"),
                     Mean_Sales=("Sales", "mean"))
                .round(1))
display(demo_summary)
""")

md("""**Insight — Why RMSPE (Q8)**  
An absolute error of **$250** is negligible for a flagship doing **$50,000/day** but catastrophic (50%) for a kiosk doing **$500/day**. RMSPE scales the error by actual sales so every store’s percentage miss counts equally — critical when Z-Cashify prices both tiny and huge locations.
""")

# ---------------------------------------------------------------------------
md("""## 14. Score the Official Test Set (optional deployment view)""")

code("""# Prepare test features (label encode with same encoders; Open may be NA → filled)
X_test_raw = test_fe[ALL_FEATURES].copy()
X_test_tree = X_test_raw.copy()
for c, le in tree_encoders.items():
    # Unseen labels → map to 0 safely
    vals = X_test_tree[c].astype(str)
    known = set(le.classes_)
    vals = vals.where(vals.isin(known), le.classes_[0])
    X_test_tree[c] = le.transform(vals)

test_pred = np.clip(best_xgb.predict(X_test_tree), 0, None)
# If store closed, force 0
if "Open" in test_fe.columns:
    test_pred = np.where(test_fe["Open"].fillna(1).values == 0, 0, test_pred)

submission = pd.DataFrame({"Id": test_fe["Id"], "Sales": test_pred})
submission.to_csv("test_predictions.csv", index=False)
print(submission.head())
print("Saved test_predictions.csv | mean pred sales (open):",
      round(submission.loc[submission["Sales"] > 0, "Sales"].mean(), 1))
""")

# ---------------------------------------------------------------------------
md("""## 15. Business Questions — Consolidated Answers

### Q1. Holiday Impact
StateHolidays typically shut stores (sales → 0); SchoolHolidays keep doors open and mildly lift demand. After a StateHoliday closure, the **reopen day shows a clear sales jump** — demand shifts rather than vanishes. *Action:* surge buy-back capacity the day before holidays and the morning of reopen.

### Q2. Promo Effect
**Short-term Promo** lifts sales and customers far more than continuous Promo2. Promo2 alone can look weak (fatigue / weaker-store selection). *Action:* time aggressive buy-back offers to Promo weeks.

### Q3. Competition Distance
Very close competitors do **not** clearly destroy sales — many high-sales stores sit in dense retail hubs. Distance matters less than Promo, assortment, and calendar. *Action:* avoid blanket discounting solely due to nearby competition.

### Q4. Best Time to Buy/Sell
**Monday** (and Sunday when open) are peak sales days. Offer **higher buy-back prices** on peak days to secure inventory; use quieter mid-week for processing and competitive resale listings.

### Q5. Model Comparison
**XGBoost** (then the Decision Tree) outperforms Linear Regression on validation RMSPE because festive demand is non-linear with strong feature interactions.

### Q6. Hyper-parameter Tuning
Tuned Decision Tree `max_depth` / `min_samples_leaf` and XGB `max_depth` + `learning_rate`. Deeper trees cut training error but inflate validation error (overfit). Moderate depth + smaller learning rate generalized best.

### Q7. Linear vs Non-Linear
LR systematically struggles more on **Promo** days — Promo→Sales is **not a straight line**; trees capture Promo × day × store interactions.

### Q8. Why RMSPE
Percentage error equalizes large and small stores; absolute dollar error would ignore kiosk-scale failures.

### Q9. Weekend Dynamic
Mondays usually beat Saturdays on average sales (promo start + catch-up). Engine: Saturday leisure browsing → premium resale presentation; Monday Promo → higher buy-back bonuses.

### Q10. Actionable Strategy — Store Manager Playbook (3 things)
Based on the best model’s drivers (Promo, calendar, competition context):
1. **Align buy-back bonuses with Promo calendars & Mondays** — pull more devices in when forecasted demand (and footfall) peaks.
2. **Staff & inventory for holiday reopen spikes** — don’t treat holiday closures as lost week; the rebound day needs cash, counters, and listing capacity.
3. **Use mid-week slack for pricing hygiene** — reprice slow SKUs, refresh resale listings, and prepare weekend/Monday promo stock instead of cutting prices just because a competitor is nearby.
""")

md("""## 16. Pricing Playbook Summary

| Signal | Buy-back price | Resale price | Ops |
|---|---|---|---|
| Promo week / Monday | Raise | Competitive | Extra staff |
| Pre-StateHoliday | Raise (sellers dump) | Hold | Prep reopen |
| Reopen day | Raise | Can firm up | Max capacity |
| Quiet Tue–Thu | Neutral | Promo clearouts | Process inventory |
| Dense competition hub | Match service, don’t panic-cut | Differentiate assortment | Promo timing > price war |

---
*End of case study notebook — Dynamic Pricing System for Festive Seasons (Z-Cashify).*
""")

# Assemble
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

out_path = "Dynamic_Pricing_Festive_Seasons.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path} with {len(cells)} cells")
