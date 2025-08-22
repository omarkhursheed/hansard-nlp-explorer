import pandas as pd
import json
from pathlib import Path

# Paths
PARQUET_PATH = Path("src/hansard/data/processed_fixed/metadata/house_members/house_members_1803_2005.parquet")
PEOPLE_JSON = Path("src/hansard/data/hansard_raw/Parlparse_git/people.json")

START = "1803-01-01"
END = "2005-12-31"

print("🔹 Loading Parquet…")
df = pd.read_parquet(PARQUET_PATH)

# Basic stats
print(f"Total rows in parquet: {len(df)}")
print(f"Unique persons in parquet: {df['person_id'].nunique()}")
print(f"Unique stints in parquet: {df['membership_id'].nunique()}")

# Date checks
print("Start date range:", df["membership_start_date"].min(), "→", df["membership_start_date"].max())
print("End date range:", df["membership_end_date"].min(), "→", df["membership_end_date"].max())

# Column sanity
id_cols = [c for c in df.columns if c.startswith("id_") or c.startswith("membership_id")]
print(f"Identifier columns ({len(id_cols)}): {id_cols}")

for col in id_cols:
    bad = df[col].dropna().apply(type).unique()
    if len(bad) > 1:
        print(f"⚠️ Mixed types in {col}: {bad}")

# Spot check a person with multiple stints
multi = df["person_id"].value_counts()
if not multi.empty:
    sample_person = multi[multi > 1].index[0]
    print("\nExample multiple stints for person:", sample_person)
    print(df[df["person_id"] == sample_person][
        ["membership_id", "membership_start_date", "membership_end_date", "post_label", "post_role", "organization_id"]
    ])

# Print one full row
print("\nSample row:\n", df.iloc[0].to_dict())

# 🔹 Commons vs Lords check
if "post_role" in df.columns:
    print("\nMemberships by organization classification:")
    print(df["post_role"].value_counts())
    print("\nUnique persons by organization classification:")
    print(df.groupby("post_role")["person_id"].nunique())

# 🔹 Comparison with people.json
print("\n🔹 Loading people.json for comparison…")
with open(PEOPLE_JSON, "r") as f:
    raw = json.load(f)

memberships = raw["memberships"]

# Filter only on start_date
in_range_memberships = [
    m for m in memberships
    if m.get("start_date") and START <= m["start_date"] <= END
]

persons_in_range = {m["person_id"] for m in in_range_memberships}
print(f"Total unique persons in people.json ({START}–{END} by start_date): {len(persons_in_range)}")

# Compare overlap
parquet_persons = set(df["person_id"].unique())
missing_in_parquet = persons_in_range - parquet_persons
missing_in_people = parquet_persons - persons_in_range

print(f"Persons in parquet but not in people.json range: {len(missing_in_people)}")
print(f"Persons in people.json range but not in parquet: {len(missing_in_parquet)}")

if missing_in_parquet:
    print("⚠️ Example missing from parquet:", list(missing_in_parquet)[:5])
if missing_in_people:
    print("⚠️ Example missing from people.json:", list(missing_in_people)[:5])

