# ============================================================
# IRS Migration Analysis — Net Migration & Income by MSA
# Data source: IRS Statistics of Income county-to-county flows
# https://www.irs.gov/statistics/soi-tax-stats-migration-data
# ============================================================

# ============================================================
# SECTION 1 — Imports & config
# ============================================================
import pandas as pd
import numpy as np

IRS_DATA_DIR = "irs_migration"
SUPPORT_DIR = "support"

# Crosswalk: maps counties to MSAs under Feb 2013 boundaries
# Same crosswalk used in real_estate_qcew.py for consistency
CROSSWALK_FILE = f"{SUPPORT_DIR}/qcew-county-msa-csa-crosswalk.xlsx"
CROSSWALK_SHEET = "Feb. 2013 Crosswalk"

# Migration file pairs: (inflow_file, outflow_file, label)
# Named by IRS convention: 2021 = 2020→2021, 2122 = 2021→2022, etc.
MIGRATION_PERIODS = [
    ("countyinflow2021.csv",  "countyoutflow2021.csv",  "2020_2021"),
    ("countyinflow2122.csv",  "countyoutflow2122.csv",  "2021_2022"),
    ("countyinflow2223.csv",  "countyoutflow2223.csv",  "2022_2023"),
]


# ============================================================
# SECTION 2 — Load crosswalk
# ============================================================
xw = pd.read_excel(
    CROSSWALK_FILE,
    sheet_name=CROSSWALK_SHEET,
    dtype={"County Code": str, "MSA Code": str}
)[["County Code", "MSA Code", "MSA Title"]].dropna(subset=["MSA Code"])

print(f"Crosswalk loaded: {len(xw)} counties → {xw['MSA Code'].nunique()} MSAs")


# ============================================================
# SECTION 3 — Functions to load and process migration files
# ============================================================
def build_county_fips(statefips, countyfips):
    """Combine state and county FIPS into 5-digit string matching crosswalk."""
    return statefips.str.zfill(2) + countyfips.str.zfill(3)


def load_inflow(filename):
    """
    Load county inflow file and extract total US inflow per destination county.
    Returns: DataFrame with county_fips, inflow_returns, inflow_individuals, inflow_agi
    """
    path = f"{IRS_DATA_DIR}/{filename}"
    df = pd.read_csv(path, dtype=str, encoding="latin1")

    # Total Migration-US rows: y1_statefips=97, y1_countyfips=000
    total = df[
        (df["y1_statefips"] == "97") &
        (df["y1_countyfips"] == "0")  # stored without leading zeros in some files
    ].copy()

    # Also try with zero-padded version
    if len(total) == 0:
        total = df[
            (df["y1_statefips"] == "97") &
            (df["y1_countyfips"] == "000")
        ].copy()

    total["county_fips"] = build_county_fips(
        total["y2_statefips"], total["y2_countyfips"]
    )

    # Convert to numeric — suppressed values are -1, exclude them
    for col in ["n1", "n2", "agi"]:
        total[col] = pd.to_numeric(total[col], errors="coerce")
    total = total[total["n1"] > 0]  # remove suppressed rows

    return total[["county_fips", "n1", "n2", "agi"]].rename(columns={
        "n1": "inflow_returns",
        "n2": "inflow_individuals",
        "agi": "inflow_agi"
    })


def load_outflow(filename):
    """
    Load county outflow file and extract total US outflow per origin county.
    Returns: DataFrame with county_fips, outflow_returns, outflow_individuals, outflow_agi
    """
    path = f"{IRS_DATA_DIR}/{filename}"
    df = pd.read_csv(path, dtype=str, encoding="latin1")

    # Total Migration-US rows: y2_statefips=97, y2_countyfips=000
    total = df[
        (df["y2_statefips"] == "97") &
        (df["y2_countyfips"] == "0")
    ].copy()

    if len(total) == 0:
        total = df[
            (df["y2_statefips"] == "97") &
            (df["y2_countyfips"] == "000")
        ].copy()

    total["county_fips"] = build_county_fips(
        total["y1_statefips"], total["y1_countyfips"]
    )

    for col in ["n1", "n2", "agi"]:
        total[col] = pd.to_numeric(total[col], errors="coerce")
    total = total[total["n1"] > 0]

    return total[["county_fips", "n1", "n2", "agi"]].rename(columns={
        "n1": "outflow_returns",
        "n2": "outflow_individuals",
        "agi": "outflow_agi"
    })


def process_period(inflow_file, outflow_file, label):
    """
    For one migration period: load inflow + outflow, compute net migration
    per county, roll up to MSA level using crosswalk.
    """
    print(f"\nProcessing {label}...")

    inflow = load_inflow(inflow_file)
    outflow = load_outflow(outflow_file)

    print(f"  Inflow counties: {len(inflow)}, Outflow counties: {len(outflow)}")

    # Merge inflow and outflow on county FIPS
    county = inflow.merge(outflow, on="county_fips", how="outer").fillna(0)

    # Net migration metrics
    county["net_returns"]     = county["inflow_returns"]     - county["outflow_returns"]
    county["net_individuals"] = county["inflow_individuals"] - county["outflow_individuals"]
    county["net_agi"]         = county["inflow_agi"]         - county["outflow_agi"]

    # Average AGI of in-migrants (income quality signal)
    # AGI is in thousands — keep it that way for now
    county["avg_inflow_agi"] = (
        county["inflow_agi"] / county["inflow_returns"].replace(0, np.nan)
    ).round(1)

    # Join to crosswalk
    county = county.merge(xw[["County Code", "MSA Code"]], 
                          left_on="county_fips", right_on="County Code", how="inner")

    # Roll up to MSA level
    msa = county.groupby("MSA Code").agg(
        inflow_returns=("inflow_returns", "sum"),
        outflow_returns=("outflow_returns", "sum"),
        net_returns=("net_returns", "sum"),
        net_individuals=("net_individuals", "sum"),
        inflow_agi=("inflow_agi", "sum"),
        outflow_agi=("outflow_agi", "sum"),
        net_agi=("net_agi", "sum"),
    ).reset_index()

    # Average AGI of in-migrants at MSA level (in thousands)
    msa["avg_inflow_agi"] = (
        msa["inflow_agi"] / msa["inflow_returns"].replace(0, np.nan)
    ).round(1)

    # Net migration rate = net_returns / inflow_returns
    # Positive = more people arriving than leaving
    msa["net_migration_rate"] = (
        msa["net_returns"] / msa["inflow_returns"].replace(0, np.nan) * 100
    ).round(2)

    # Rename columns with period suffix
    msa = msa.rename(columns={
        "net_returns":        f"net_returns_{label}",
        "net_individuals":    f"net_individuals_{label}",
        "net_agi":            f"net_agi_{label}",
        "avg_inflow_agi":     f"avg_inflow_agi_{label}",
        "net_migration_rate": f"net_migration_rate_{label}",
        "inflow_returns":     f"inflow_returns_{label}",
        "outflow_returns":    f"outflow_returns_{label}",
    })

    print(f"  MSAs after crosswalk join: {len(msa)}")
    return msa[["MSA Code"] + [c for c in msa.columns if c != "MSA Code"]]


# ============================================================
# SECTION 4 — Process all periods and merge
# ============================================================
period_frames = []
for inflow_file, outflow_file, label in MIGRATION_PERIODS:
    period_frames.append(process_period(inflow_file, outflow_file, label))

# Merge all periods on MSA Code
df_migration = period_frames[0]
for frame in period_frames[1:]:
    df_migration = df_migration.merge(frame, on="MSA Code", how="inner")

print(f"\nMSAs with migration data across all periods: {len(df_migration)}")


# ============================================================
# SECTION 5 — Trend metrics across periods
# ============================================================
net_rate_cols = [f"net_migration_rate_{p[2]}" for p in MIGRATION_PERIODS]
agi_cols      = [f"avg_inflow_agi_{p[2]}" for p in MIGRATION_PERIODS]

# Average net migration rate across all periods
df_migration["avg_net_migration_rate"] = (
    df_migration[net_rate_cols].mean(axis=1).round(2)
)

# Average inflow AGI across all periods (in thousands)
df_migration["avg_inflow_agi"] = (
    df_migration[agi_cols].mean(axis=1).round(1)
)

# Migration trend: is net migration accelerating or decelerating?
latest_rate = net_rate_cols[-1]
prior_rate  = net_rate_cols[-2]

def migration_trend(row):
    diff = row[latest_rate] - row[prior_rate]
    if diff >= 1.0:
        return "accelerating"
    elif diff <= -1.0:
        return "decelerating"
    else:
        return "stable"

df_migration["migration_trend"] = df_migration.apply(migration_trend, axis=1)

print("\nMigration trend distribution:")
print(df_migration["migration_trend"].value_counts())


# ============================================================
# SECTION 6 — Add MSA names and rank
# ============================================================
msa_names = xw[["MSA Code", "MSA Title"]].drop_duplicates("MSA Code")
df_migration = df_migration.merge(msa_names, on="MSA Code", how="left")

# Rank by average net migration rate
df_ranked = (
    df_migration[[
        "MSA Code",
        "MSA Title",
        *net_rate_cols,
        "avg_net_migration_rate",
        *agi_cols,
        "avg_inflow_agi",
        "migration_trend",
    ]]
    .sort_values("avg_net_migration_rate", ascending=False)
    .reset_index(drop=True)
)
df_ranked.index += 1

print(f"\nTop 20 MSAs by average net migration rate (2020→2023)\n")
print(df_ranked.head(20).to_string())

# Save outputs
df_ranked.to_csv("irs_msa_migration.csv", index=False)
df_ranked.head(50).to_csv("samples/irs_msa_migration_sample.csv", index=False)

print("\nSaved to irs_msa_migration.csv")
print("Saved sample to samples/irs_msa_migration_sample.csv")
