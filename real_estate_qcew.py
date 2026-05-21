# ============================================================
# CELL 1 — Imports & config
# ============================================================
import pandas as pd
import numpy as np
import requests
import io

# Industry 10 = all industries. This file has every MSA and county in the US.
BASE_URL = "https://data.bls.gov/cew/data/api/{year}/a/industry/10.csv"
YEAR = "2023"


# ============================================================
# CELL 2 — Fetch data and filter to MSA level
# ============================================================
url = BASE_URL.format(year=YEAR)
print(f"Fetching: {url}")

resp = requests.get(url, timeout=60)
resp.raise_for_status()

# dtype str on area_fips prevents FIPS codes from being read as integers
df = pd.read_csv(io.StringIO(resp.text), dtype={"area_fips": str}, low_memory=False)

# agglvl_code 40 = MSA total covered
# own_code 0    = all ownership types combined
df_msa = df[
    (df["agglvl_code"] == 40) &
    (df["own_code"] == 0)
].copy()

print(f"Found {len(df_msa)} MSAs")


# ============================================================
# CELL 3 — Add MSA names, score, and rank
# ============================================================

# Read the area titles lookup file you saved to your project folder.
# Download from: https://www.bls.gov/cew/classifications/areas/area-titles-csv.csv
area_titles = pd.read_csv(
    "area-titles-csv.csv",
    header=None,
    names=["area_fips", "area_title"],
    dtype={"area_fips": str}
)

# Join names onto MSA data
df_msa = df_msa.merge(area_titles, on="area_fips", how="left")

# Rename the BLS over-the-year column to something readable
df_msa = df_msa.rename(columns={
    "oty_annual_avg_emplvl_pct_chg": "yoy_growth_pct",
    "annual_avg_emplvl": "avg_employment"
})

# Drop any rows with missing growth data (some MSAs are suppressed by BLS)
df_msa = df_msa.dropna(subset=["yoy_growth_pct", "avg_employment"])

# Composite score: growth rate weighted by log of employment size.
# Log scale means a 200k-job MSA outweighs a 5k-job MSA, but not by
# the full 40x — prevents giant metros from drowning out mid-size markets.
df_msa["size_weight"] = np.log(df_msa["avg_employment"])
df_msa["composite_score"] = (
    df_msa["yoy_growth_pct"] * df_msa["size_weight"]
).round(2)

# Build final ranked table
df_ranked = (
    df_msa[[
        "area_fips",
        "area_title",
        "avg_employment",
        "yoy_growth_pct",
        "composite_score"
    ]]
    .sort_values("composite_score", ascending=False)
    .reset_index(drop=True)
)
df_ranked.index += 1  # rank starts at 1

print(f"\nTop 20 MSAs by size-adjusted employment growth ({int(YEAR)-1}→{YEAR})\n")
print(df_ranked.head(20).to_string())

# Save for use in later notebooks (IRS migration join, scoring model, etc.)
df_ranked.to_csv("qcew_msa_employment_growth.csv", index=False)
print("\nSaved to qcew_msa_employment_growth.csv")
