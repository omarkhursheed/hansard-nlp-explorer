"""
Gender Enrichment Pipeline for Historic Hansard Members

This script infers/collects gender for parliamentary members by combining:
  1) Existing EveryPolitician gender labels (if present).
  2) Honorific/title heuristics from Historic Hansard (HH) records.
  3) Honorific/title heuristics from ParlParse (PP) 'other_names'.
  4) Wikidata P21 (sex or gender) via SPARQL lookups for remaining unknowns.
  5) Historical rules (Commons pre-1918, Lords pre-1958 → male).
  6) Curated female lists (Wikidata MPs; Wikipedia female Lords).
  7) Name-based inference using gender-guesser (first-name heuristic).
  8) Optional, uncomment if needed: Default male if all of the above fail.

Outputs:
  - Writes a parquet with inferred gender and provenance to OUT_PATH.

Notes:
  - Be mindful of rate limits for external services (Wikidata, Wikipedia).
  - Honorific matching is case-insensitive and token-based.
  - The pipeline logs per-step assignment counts for transparency/debugging.
"""

# pip install lxml, gender-guesser

import pandas as pd
import json
import re
from pathlib import Path
import numpy as np
from fetch_member_gender_wikidata import fetch_wikidata_genders
from fetch_female_members_wikidata import fetch_female_mps, scrape_female_lords
import gender_guesser.detector as genderer

# ==== Paths ====
PARQUET_PATH = Path("src/hansard/data/processed_fixed/metadata/house_members/PP_HH_members_combined.parquet")
HONORIFICS_JSON = Path("src/hansard/data/word_lists/gendered_honorifics.json")
OUT_PATH = Path("src/hansard/data/processed_fixed/metadata/house_members/house_members_gendered.parquet")

# ==== Load data ====
print("🔹 Loading parquet…")
df = pd.read_parquet(PARQUET_PATH)

with open(HONORIFICS_JSON, "r") as f:
    honorific_map = json.load(f)

honorific_male = {h.lower().rstrip(".") for h in honorific_map.get("Male", [])}
honorific_female = {h.lower().rstrip(".") for h in honorific_map.get("Female", [])}

# --- Counter after each step ---
def log_step(df, step_name, before_gender):
    """Print and return a boolean mask of rows with gender set after a step.

    Parameters
    ----------
    df : pandas.DataFrame
        Working dataframe containing 'gender_inferred' and 'person_id'.
    step_name : str
        Human-readable name of the step for logging.
    before_gender : pandas.Series (bool)
        Mask of rows that had a gender set before this step.

    Returns
    -------
    pandas.Series (bool)
        Updated mask of rows with gender set after this step.
    """
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


# ==== Step 2: HH Honorific/title inference ====
mask_honorific = df["gender_inferred"].isna()
honorifics = df.loc[mask_honorific, "honorific_prefix"]

# Normalize safely
honorifics_norm = honorifics.dropna().str.lower()

def match_gendered_word(honorific, gendered_set):
    """Return True if any token in a (lowercased) honorific appears in gendered_set.

    Parameters
    ----------
    honorific : str | Any
        Honorific string to search (may be NaN/None).
    gendered_set : set[str]
        Set of gendered words (e.g., {'mr', 'sir'} or {'ms', 'mrs'}).

    Returns
    -------
    bool
        True if any token matches; False otherwise.
    """
    if not isinstance(honorific, str):
        return False
    tokens = honorific.lower().split()
    return any(tok in gendered_set for tok in tokens)

# Male honorifics
mask_male = honorifics_norm.apply(lambda h: match_gendered_word(h, honorific_male))
df.loc[mask_honorific & mask_male, ["gender_inferred", "gender_source"]] = ["M", "HH honorific"]

# Female honorifics
mask_female = honorifics_norm.apply(lambda h: match_gendered_word(h, honorific_female))
df.loc[mask_honorific & mask_female, ["gender_inferred", "gender_source"]] = ["F", "HH honorific"]

print("Number of rows with HH honorifics:", honorifics.notna().sum())
print("Assigned male:", mask_male.sum(), "Assigned female:", mask_female.sum())

before = log_step(df, "HH Honorifics", before)


# ==== Step 3: PP Honorific/title inference (extract+append+normalize for ALL rows; match only on NaNs) ====
def extract_honorific_prefix(obj):
    """Extract an honorific_prefix from 'other_names' (arrays of dicts) where note == 'Main'.

    Parameters
    ----------
    obj : Any
        Either numpy.ndarray, list[dict], dict, or other type coming from 'other_names'.

    Returns
    -------
    str | None
        The honorific_prefix if found on the 'Main' entry; otherwise None.
    """
    try:
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        if isinstance(obj, list):
            for el in obj:
                if isinstance(el, dict) and el.get("note") == "Main":
                    return el.get("honorific_prefix")
            return None
        if isinstance(obj, dict):
            if obj.get("note") == "Main":
                return obj.get("honorific_prefix")
        return None
    except Exception:
        return None

mask_honorific = df["gender_inferred"].isna()

# Ensure honorific_prefix column is a list for ALL rows
def _ensure_list(x):
    """Return a list representation for honorific_prefix cells.

    - If already a list, return a shallow copy.
    - If NaN/None, return [].
    - Otherwise wrap the value in a single-element list.
    """
    if isinstance(x, list):
        return x.copy()
    if pd.isna(x) or x is None:
        return []
    return [x]

df["honorific_prefix"] = df["honorific_prefix"].apply(_ensure_list)

# Extract potential new prefix from other_names for ALL rows
new_prefix_all = df["other_names"].map(extract_honorific_prefix)

# Append extracted prefix into honorific_prefix lists 
def _append_if_missing(lst, pref):
    """Append `pref` to list `lst` if a case-insensitive equivalent is not already present."""
    if not isinstance(pref, str) or not pref.strip():
        return lst
    # case-insensitive de-duplication
    lower_set = {s.lower(): s for s in lst if isinstance(s, str)}
    if pref.lower() not in lower_set:
        lst.append(pref)
    return lst

for idx, pref in new_prefix_all.items():
    df.at[idx, "honorific_prefix"] = _append_if_missing(df.at[idx, "honorific_prefix"], pref)

# Build a normalized (lowercased) string view for matching for ALL rows
honorifics_str_all = df["honorific_prefix"].map(lambda v: " ".join(v) if isinstance(v, list) else v)
honorifics_norm_all = honorifics_str_all.dropna().str.lower()

def match_gendered_word(honorific, gendered_set):
    """Return True if any token in a (lowercased) honorific appears in gendered_set.

    Parameters
    ----------
    honorific : str | Any
        Honorific string to search (may be NaN/None).
    gendered_set : set[str]
        Set of gendered words (e.g., {'mr', 'sir'} or {'ms', 'mrs'}).

    Returns
    -------
    bool
        True if any token matches; False otherwise.
    """
    if not isinstance(honorific, str):
        return False
    tokens = honorific.lower().split()
    return any(tok in gendered_set for tok in tokens)

# Perform gender matching ONLY on rows where gender_inferred is NaN
honorifics_on_mask = honorifics_norm_all.reindex(df.index, fill_value=np.nan)[mask_honorific]

# Male honorifics
mask_male = honorifics_on_mask.apply(lambda h: match_gendered_word(h, honorific_male))
df.loc[mask_honorific & mask_male, ["gender_inferred", "gender_source"]] = ["M", "PP honorific"]

# Female honorifics
mask_female = honorifics_on_mask.apply(lambda h: match_gendered_word(h, honorific_female))
df.loc[mask_honorific & mask_female, ["gender_inferred", "gender_source"]] = ["F", "PP honorific"]

print("Number of rows with any honorifics (post-append):", honorifics_str_all.notna().sum())
print("Assigned male:", mask_male.sum(), "Assigned female:", mask_female.sum())

before = log_step(df, "PP Honorifics", before)


# ==== Step 4: Wikidata enrichment ====
# Collect QIDs where gender is still missing
missing_qids = df.loc[df["gender_inferred"].isna() & df["id_wikidata"].notna(), "id_wikidata"].unique().tolist()

print(f"Fetching genders for {len(missing_qids)} Wikidata IDs…")
wikidata_genders = fetch_wikidata_genders(missing_qids)

# Fill from Wikidata results
mask = df["gender_inferred"].isna() & df["id_wikidata"].notna()
df.loc[mask, "gender_inferred"] = df.loc[mask, "id_wikidata"].map(wikidata_genders)
df.loc[mask & df["gender_inferred"].notna(), "gender_source"] = "wikidata_P21"

before = log_step(df, "Wikidata P21", before)


# ==== Step 5: Historical rules ====
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


# ==== Step 6: Curated female lists ====
def normalize_name(name):
    """Normalize a person name for fuzzy matching.

    - Lowercase
    - Remove non-letters
    - Collapse internal whitespace

    Returns
    -------
    str | None
        Normalized name, or None if the input is NaN.
    """
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


# ==== Step 7: Name-based inference ====
detector = genderer.Detector(case_sensitive=False)

def infer_gender_from_name(name):
    """Infer a binary gender code from a first name using gender-guesser.

    Parameters
    ----------
    name : str | Any
        Full name string; only the first token is used.

    Returns
    -------
    str | None
        'F' if predicted female/mostly_female, 'M' if male/mostly_male, else None.
    """
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


# # ==== Step 8: Default male ====
# mask_default = df["gender_inferred"].isna()
# df.loc[mask_default, ["gender_inferred", "gender_source"]] = ["M", "default_male"]

# before = log_step(df, "Default male fallback", before)


# ==== Step 9: Cleanup ====
df = df.drop(columns=["person_name_norm", "gender"])


print(f"Members with unknown gender: {df[df["gender_inferred"].isna()]["person_id"].nunique()}")

# ==== Save ====
df = df.drop(columns=["year_start"])
df.to_parquet(OUT_PATH, index=False)
print(f"✅ Saved enriched dataset with gender → {OUT_PATH}")
