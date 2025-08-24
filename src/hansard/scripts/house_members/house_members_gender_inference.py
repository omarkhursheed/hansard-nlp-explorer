# %pip install lxml, gender-guesser

import pandas as pd
import json
import re
from pathlib import Path
import numpy as np
from fetch_member_gender_wikidata import fetch_wikidata_genders
from fetch_female_members_wikidata import fetch_female_mps, scrape_female_lords
import gender_guesser.detector as genderer

# ==== Paths ====
PARQUET_PATH = Path("src/hansard/data/processed_fixed/metadata/house_members/PP_EP_house_members_combined.parquet")
HONORIFICS_JSON = Path("src/hansard/data/gender_wordlists/gendered_honorifics.json")
OUT_PATH = Path("src/hansard/data/processed_fixed/metadata/house_members/house_members_gendered.parquet")

# ==== Load data ====
print("🔹 Loading parquet…")
df = pd.read_parquet(PARQUET_PATH)

with open(HONORIFICS_JSON, "r") as f:
    honorific_map = json.load(f)

honorific_male = {h.lower() for h in honorific_map.get("Male", [])}
honorific_female = {h.lower() for h in honorific_map.get("Female", [])}

# --- Counter after each step ---
def log_step(df, step_name, before_gender):
    """Log how many unique person_ids got gender assigned in this step."""
    after = df["gender_inferred"].notna()
    new_assignments = after & ~before_gender
    n_new = df.loc[new_assignments, "person_id"].nunique()
    print(f"🔹 {step_name}: {n_new} unique person_ids newly assigned")
    return after  # return for chaining


# ==== Initialize ====
df["gender_inferred"] = df.get("gender")  # EP gender if present
df.loc[df["gender_inferred"]=='male', "gender_inferred"] = "M"
df.loc[df["gender_inferred"]=='female', "gender_inferred"] = "F"
df["gender_source"] = df["gender_inferred"].where(df["gender_inferred"].notna(), None)
df.loc[df["gender_source"].notna(), "gender_source"] = "everypolitician"

before = df["gender_inferred"].notna()
before = log_step(df, "EveryPolitician", before)

# ==== Step 2: Wikidata enrichment (placeholder) ====
# Collect QIDs where gender is still missing
missing_qids = df.loc[df["gender_inferred"].isna() & df["id_wikidata"].notna(), "id_wikidata"].unique().tolist()

print(f"Fetching genders for {len(missing_qids)} Wikidata IDs…")
wikidata_genders = fetch_wikidata_genders(missing_qids)

# Fill from Wikidata results
mask = df["gender_inferred"].isna() & df["id_wikidata"].notna()
df.loc[mask, "gender_inferred"] = df.loc[mask, "id_wikidata"].map(wikidata_genders)
df.loc[mask & df["gender_inferred"].notna(), "gender_source"] = "wikidata_P21"

before = log_step(df, "Wikidata P21", before)

# ==== Step 3: Historical rules ====
df["year_start"] = pd.to_datetime(df["membership_start_date"], errors="coerce").dt.year

mask_commons = (
    df["gender_inferred"].isna()
    & (df["organization_id"] == "house-of-commons")
    & (df["year_start"] < 1918)
)
df.loc[mask_commons, ["gender_inferred", "gender_source"]] = ["M", "rule_pre1918_commons"]

mask_lords = (
    df["gender_inferred"].isna()
    & (df["organization_id"] == "house-of-lords")
    & (df["year_start"] < 1958)
)
df.loc[mask_lords, ["gender_inferred", "gender_source"]] = ["M", "rule_pre1958_lords"]

before = log_step(df, "Historical rules", before)

# ==== Step 4: Honorific/title inference ====
def extract_honorific_prefix(obj):
    """Extract honorific_prefix from other_names (numpy arrays of dicts).
       Only use the dict where note == 'Main'.
    """
    try:
        # If it's a numpy array
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()

        # If it's a list of dicts
        if isinstance(obj, list):
            for el in obj:
                if isinstance(el, dict) and el.get("note") == "Main":
                    return el.get("honorific_prefix")
            return None  # no Main found

        # If it's a single dict
        if isinstance(obj, dict):
            if obj.get("note") == "Main":
                return obj.get("honorific_prefix")

        return None
    except Exception:
        return None


mask_honorific = df["gender_inferred"].isna()
honorifics = df.loc[mask_honorific, "other_names"].map(extract_honorific_prefix)

# Normalize safely
honorifics_norm = honorifics.dropna().str.lower()

def match_gendered_word(honorific, gendered_set):
    """Return True if any word/token in honorific is in gendered_set."""
    if not isinstance(honorific, str):
        return False
    tokens = honorific.lower().split()
    return any(tok in gendered_set for tok in tokens)

# Male honorifics
mask_male = honorifics_norm.apply(lambda h: match_gendered_word(h, honorific_male))
df.loc[mask_honorific & mask_male, ["gender_inferred", "gender_source"]] = ["M", "honorific"]

# Female honorifics
mask_female = honorifics_norm.apply(lambda h: match_gendered_word(h, honorific_female))
df.loc[mask_honorific & mask_female, ["gender_inferred", "gender_source"]] = ["F", "honorific"]

print("Number of rows with honorifics:", honorifics.notna().sum())
print("Assigned male:", mask_male.sum(), "Assigned female:", mask_female.sum())

before = log_step(df, "Honorifics", before)


# ==== Step 5: Curated female lists ====
def normalize_name(name):
    if pd.isna(name):
        return None
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return " ".join(name.split())

print("Fetching female parliamentarians from Wikidata…")
df_females = fetch_female_mps()
df_females["name_norm"] = df_females["name"].map(normalize_name)
female_names_norm = set(df_females["name_norm"].dropna().unique())

# Normalize main df
df["person_name_norm"] = df["person_name"].map(normalize_name)

# --- Matching MPs ---
mask_main = df["gender_inferred"].isna() & df["person_name_norm"].isin(female_names_norm)
mask_alias = df["gender_inferred"].isna() & df["aliases_norm"].apply(
    lambda names: any(n in female_names_norm for n in names if n)
)

df.loc[mask_main, ["gender_inferred", "gender_source"]] = ["F", "wikidata_curated_name_mp"]
df.loc[mask_alias, ["gender_inferred", "gender_source"]] = ["F", "wikidata_curated_alias_mp"]
print(f"Matched {mask_main.sum()} by person_name, {mask_alias.sum()} by aliases")

# --- Lords ---
df_lords = scrape_female_lords()
df_lords["name_norm"] = df_lords["raw_name"].map(normalize_name)
female_lords_norm = set(df_lords["name_norm"].dropna().unique())
print(f"Collected {len(female_lords_norm)} unique normalized female Lords")

mask_main_lords = (
    df["gender_inferred"].isna()
    & (df["organization_id"] == "house-of-lords")
    & df["person_name_norm"].isin(female_lords_norm)
)
mask_alias_lords = (
    df["gender_inferred"].isna()
    & (df["organization_id"] == "house-of-lords")
    & df["aliases_norm"].apply(lambda names: any(n in female_lords_norm for n in names if n))
)

df.loc[mask_main_lords, ["gender_inferred", "gender_source"]] = ["F", "wikidata_curated_name_lord"]
df.loc[mask_alias_lords, ["gender_inferred", "gender_source"]] = ["F", "wikidata_curated_alias_lord"]

print("Female Lords by name:", df.loc[mask_main_lords, "person_id"].nunique())
print("Female Lords by alias:", df.loc[mask_alias_lords, "person_id"].nunique())

before = log_step(df, "Curated female lists", before)


# ==== Step 6: Name-based inference (placeholder) ====
detector = genderer.Detector(case_sensitive=False)

def infer_gender_from_name(name):
    if not isinstance(name, str):
        return None
    g = detector.get_gender(name.split()[0])  # take first token
    if g in ("female", "mostly_female"):
        return "F"
    elif g in ("male", "mostly_male"):
        return "M"
    else:
        return None

mask_name = df["gender_inferred"].isna() & df["person_name"].notna()
df.loc[mask_name, "gender_inferred"] = df.loc[mask_name, "person_name"].map(infer_gender_from_name)
df.loc[mask_name & df["gender_inferred"].notna(), "gender_source"] = "fname_inference"

before = log_step(df, "Name-based inference", before)

# # ==== Step 7: Default male ====
mask_default = df["gender_inferred"].isna()
df.loc[mask_default, ["gender_inferred", "gender_source"]] = ["M", "default_male"]

before = log_step(df, "Default male fallback", before)

# ==== Step 8: Cleanup ====
df = df.drop(columns=["person_name_norm", "gender"])


print(f"Members with unknown gender: {df[df["gender_inferred"].isna()]["person_id"].nunique()}")

# ==== Save ====
df = df.drop(columns=["year_start"])
df.to_parquet(OUT_PATH, index=False)
print(f"✅ Saved enriched dataset with gender → {OUT_PATH}")
