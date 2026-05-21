# ============================================================
# CELL 1 — Imports & config
# ============================================================
import pandas as pd
import numpy as np

YEARS = ["2021", "2022", "2023", "2024"]
BLS_DATA_DIR = "bls_qcew"
SUPPORT_DIR = "support"

# Crosswalk source:
# https://www.bls.gov/cew/classifications/areas/county-msa-csa-crosswalk.htm
CROSSWALK_FILE = f"{SUPPORT_DIR}/qcew-county-msa-csa-crosswalk.xlsx"

# 2021-2023 use Feb 2013 boundaries; 2024+ use Jul 2023 boundaries.
# We remap 2024 county employment back to Feb 2013 MSA definitions
# so all four years are comparable on the same geographic basis.
OLD_CROSSWALK_SHEET = "Feb. 2013 Crosswalk"  # 2013-2023 boundaries (our baseline)
NEW_CROSSWALK_SHEET = "Jul. 2023 Crosswalk"  # 2024+ boundaries


# ============================================================
# CELL 2 — Load crosswalks
# ============================================================
xw_old = pd.read_excel(
    CROSSWALK_FILE,
    sheet_name=OLD_CROSSWALK_SHEET,
    dtype={"County Code": str, "MSA Code": str}
)[["County Code", "MSA Code", "MSA Title"]].dropna(subset=["MSA Code"])

xw_new = pd.read_excel(
    CROSSWALK_FILE,
    sheet_name=NEW_CROSSWALK_SHEET,
    dtype={"County Code": str, "MSA Code": str}
)[["County Code", "MSA Code"]].dropna(subset=["MSA Code"])

print(f"Old crosswalk (2013-2023): {len(xw_old)} counties → {xw_old['MSA Code'].nunique()} MSAs")
print(f"New crosswalk (2024+):     {len(xw_new)} counties → {xw_new['MSA Code'].nunique()} MSAs")


# ============================================================
# CELL 3 — Load employment data for each year
# ============================================================
def load_msa_employment(year, crosswalk=None):
    """
    Load annual average employment at MSA level for a given year.
    
    For 2021-2023: use pre-aggregated MSA rows (agglvl_code 40) directly.
    For 2024+:     use county rows (agglvl_code 70), remap to old MSA 
                   boundaries via the Feb 2013 crosswalk, then sum.
    """
    path = f"{BLS_DATA_DIR}/10_{year}.csv"
    print(f"Loading {path}...")
    df = pd.read_csv(path, dtype={"area_fips": str}, low_memory=False)

    if crosswalk is None:
        # Use pre-aggregated MSA totals directly
        msa = df[
            (df["agglvl_code"] == 40) &
            (df["own_code"] == 0)
        ][["area_fips", "annual_avg_emplvl"]].copy()
        msa = msa.rename(columns={"area_fips": "MSA Code", "annual_avg_emplvl": f"emp_{year}"})
    else:
        # Pull county-level employment and remap to old MSA boundaries
        county = df[
            (df["agglvl_code"] == 70) &
            (df["own_code"] == 0)
        ][["area_fips", "annual_avg_emplvl"]].copy()
        county = county.rename(columns={"area_fips": "County Code"})

        # Join to old crosswalk to get old MSA codes
        county = county.merge(crosswalk[["County Code", "MSA Code"]], 
                              on="County Code", how="inner")

        # Sum county employment up to old MSA boundaries
        msa = (
            county.groupby("MSA Code")["annual_avg_emplvl"]
            .sum()
            .reset_index()
            .rename(columns={"annual_avg_emplvl": f"emp_{year}"})
        )
    print(f"  → {len(msa)} MSAs for {year} (remapped to 2013 boundaries)")
    return msa


# Load 2021-2023 using pre-aggregated MSA data
frames = []
for year in ["2021", "2022", "2023"]:
    frames.append(load_msa_employment(year))

# Load 2024 using county remapping to old boundaries
frames.append(load_msa_employment("2024", crosswalk=xw_old))

# Merge all years — inner join keeps only MSAs present in all years
df_all = frames[0]
for frame in frames[1:]:
    df_all = df_all.merge(frame, on="MSA Code", how="inner")

print(f"\nMSAs with data across all years: {len(df_all)}")

# Diagnostic — show how many MSAs drop out at each year merge
base = frames[0].copy()
print(f"\n2021 baseline: {len(base)} MSAs")
for i, frame in enumerate(frames[1:]):
    year = YEARS[i+1]
    before = len(base)
    base = base.merge(frame, on="MSA Code", how="inner")
    dropped = before - len(base)
    print(f"After merging {year}: {len(base)} MSAs ({dropped} dropped)")
print()

# ============================================================
# CELL 4 — Calculate YoY growth, averages, volatility, and trend
# ============================================================
year_pairs = list(zip(YEARS[:-1], YEARS[1:]))
growth_cols = []

for yr_from, yr_to in year_pairs:
    col = f"growth_{yr_from}_{yr_to}"
    df_all[col] = (
        (df_all[f"emp_{yr_to}"] - df_all[f"emp_{yr_from}"]) /
        df_all[f"emp_{yr_from}"] * 100
    ).round(2)
    growth_cols.append(col)

# Average growth across all periods
df_all["avg_growth"] = df_all[growth_cols].mean(axis=1).round(2)

# Volatility — std dev across growth periods
df_all["volatility"] = df_all[growth_cols].std(axis=1).round(2)

# Trend: compare latest period to prior period
latest_col = growth_cols[-1]
prior_col = growth_cols[-2]

def classify_trend(row):
    diff = row[latest_col] - row[prior_col]
    if diff >= 1.0:
        return "accelerating"
    elif diff <= -1.0:
        return "decelerating"
    else:
        return "stable"

df_all["trend"] = df_all.apply(classify_trend, axis=1)

print("Trend distribution:")
print(df_all["trend"].value_counts())
print()
print("Top 10 highest volatility MSAs (potential remaining anomalies):")
print(df_all.nlargest(10, "volatility")[["MSA Code", "volatility"] + growth_cols].to_string())


# ============================================================
# CELL 5 — Add MSA names, score, and rank
# ============================================================

# Use MSA titles from the old crosswalk
msa_names = xw_old[["MSA Code", "MSA Title"]].drop_duplicates("MSA Code")
df_all = df_all.merge(msa_names, on="MSA Code", how="left")

# Composite score: average growth weighted by log of latest employment
df_all["size_weight"] = np.log(df_all["emp_2024"])
df_all["composite_score"] = (
    df_all["avg_growth"] * df_all["size_weight"]
).round(2)

# Final ranked table
df_ranked = (
    df_all[[
        "MSA Code",
        "MSA Title",
        "emp_2024",
        *growth_cols,
        "avg_growth",
        "volatility",
        "trend",
        "composite_score"
    ]]
    .sort_values("composite_score", ascending=False)
    .reset_index(drop=True)
)
df_ranked.index += 1

print(f"\nTop 20 MSAs by size-adjusted average employment growth (2021→2024)\n")
print(df_ranked.head(20).to_string())

# Flag high volatility markets
high_vol = df_ranked.head(20)[df_ranked.head(20)["volatility"] > 5]
if len(high_vol) > 0:
    print(f"\n⚠ High volatility markets in top 20 (volatility > 5%):")
    print(high_vol[["MSA Title", "avg_growth", "volatility"] + growth_cols].to_string())

# Save outputs
df_ranked.to_csv("qcew_msa_employment_growth.csv", index=False)
df_ranked.head(50).to_csv("samples/qcew_msa_sample.csv", index=False)

print("\nSaved to qcew_msa_employment_growth.csv")
print("Saved sample to samples/qcew_msa_sample.csv")
