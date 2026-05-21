# ============================================================
# CELL 1 — Imports & config
# ============================================================
import pandas as pd
import numpy as np
import io

BASE_URL = "https://data.bls.gov/cew/data/api/{year}/a/industry/10.csv"
YEARS = ["2021", "2022", "2023", "2024"]
BLS_DATA_DIR = "bls_qcew"  # folder where you saved the downloaded CSVs


# ============================================================
# CELL 2 — Load all years from local files and filter to MSA level
# ============================================================
frames = []

for year in YEARS:
    path = f"{BLS_DATA_DIR}/10_{year}.csv"
    print(f"Loading {path}...")
    df = pd.read_csv(path, dtype={"area_fips": str}, low_memory=False)

    # agglvl_code 40 = MSA total covered, own_code 0 = all ownership types
    df_msa = df[
        (df["agglvl_code"] == 40) &
        (df["own_code"] == 0)
    ][["area_fips", "annual_avg_emplvl"]].copy()

    df_msa = df_msa.rename(columns={"annual_avg_emplvl": f"emp_{year}"})
    frames.append(df_msa)

# Merge all years on area_fips
df_all = frames[0]
for frame in frames[1:]:
    df_all = df_all.merge(frame, on="area_fips", how="inner")

print(f"\nMSAs with data across all years: {len(df_all)}")


# ============================================================
# CELL 3 — Calculate year-over-year growth for each period
# ============================================================
# Calculate YoY growth rate for each consecutive year pair
year_pairs = list(zip(YEARS[:-1], YEARS[1:]))

for yr_from, yr_to in year_pairs:
    col_name = f"growth_{yr_from}_{yr_to}"
    df_all[col_name] = (
        (df_all[f"emp_{yr_to}"] - df_all[f"emp_{yr_from}"]) /
        df_all[f"emp_{yr_from}"] * 100
    ).round(2)

# Latest growth period (most recent year pair)
latest_from, latest_to = year_pairs[-1]
latest_growth_col = f"growth_{latest_from}_{latest_to}"

# Prior growth period
prior_from, prior_to = year_pairs[-2]
prior_growth_col = f"growth_{prior_from}_{prior_to}"

# Trend direction: compare latest period to prior period
def classify_trend(row):
    diff = row[latest_growth_col] - row[prior_growth_col]
    if diff >= 1.0:
        return "accelerating"
    elif diff <= -1.0:
        return "decelerating"
    else:
        return "stable"

df_all["trend"] = df_all.apply(classify_trend, axis=1)

print("Trend distribution:")
print(df_all["trend"].value_counts())


# ============================================================
# CELL 4 — Add MSA names, score, and rank
# ============================================================
area_titles = pd.read_csv(
    "area-titles-csv.csv",
    header=None,
    names=["area_fips", "area_title"],
    dtype={"area_fips": str}
)

df_all = df_all.merge(area_titles, on="area_fips", how="left")

# Composite score based on latest growth period, weighted by employment size
df_all["size_weight"] = np.log(df_all[f"emp_{YEARS[-1]}"])
df_all["composite_score"] = (
    df_all[latest_growth_col] * df_all["size_weight"]
).round(2)

# Build final ranked table
growth_cols = [f"growth_{a}_{b}" for a, b in year_pairs]

df_ranked = (
    df_all[[
        "area_fips",
        "area_title",
        f"emp_{YEARS[-1]}",
        *growth_cols,
        "trend",
        "composite_score"
    ]]
    .sort_values("composite_score", ascending=False)
    .reset_index(drop=True)
)
df_ranked.index += 1

print(f"\nTop 20 MSAs by size-adjusted employment growth (latest: {latest_from}→{latest_to})\n")
print(df_ranked.head(20).to_string())

# Save output
df_ranked.to_csv("qcew_msa_employment_growth.csv", index=False)

# Save sample for GitHub
df_ranked.head(50).to_csv("samples/qcew_msa_sample.csv", index=False)

print("\nSaved to qcew_msa_employment_growth.csv")
print("Saved sample to samples/qcew_msa_sample.csv")