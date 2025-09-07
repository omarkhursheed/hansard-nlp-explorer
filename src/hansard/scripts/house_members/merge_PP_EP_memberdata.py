import pandas as pd
from pathlib import Path
import numpy as np
import re

# Paths
PARLPARSE_PATH = Path("src/hansard/data/processed_fixed/metadata/house_members/PP_house_members_1803_2005.parquet")
EVERYPOLITICIAN_PATH = Path("src/hansard/data/processed_fixed/metadata/house_members/EP_commons_members.parquet")
OUT_PATH = Path("src/hansard/data/processed_fixed/metadata/house_members/PP_EP_house_members_combined.parquet")

print("🔹 Loading Parquet files…")
pp = pd.read_parquet(PARLPARSE_PATH)
ep = pd.read_parquet(EVERYPOLITICIAN_PATH)

print(f"Parlparse rows: {len(pp)}, cols: {len(pp.columns)}")
print(f"EveryPolitician rows: {len(ep)}, cols: {len(ep.columns)}")

# --- Collapse EveryPolitician to person-level (for gender, birth_date, etc.) ---
person_cols = [
    "id_parlparse", "id_wikidata", "id_datadotparl", "id_parliamentdotuk",
    "name", "sort_name", "given_name", "family_name",
    "birth_date", "gender", "wikipedia"
]
ep_persons = ep[person_cols].drop_duplicates()
ep_persons = ep_persons.groupby("id_parlparse", dropna=False).first().reset_index()

# --- Party info from EP (for filling nulls only) ---
ep_party = ep[["id_parlparse", "org_id"]].drop_duplicates()

# --- Merge person-level metadata ---
merged = pp.merge(
    ep_persons,
    how="left",
    left_on="person_id",
    right_on="id_parlparse",
    suffixes=("", "_ep")
)

# --- Merge EP party ids (to fill PP nulls only) ---
merged = merged.merge(
    ep_party,
    how="left",
    left_on="person_id",
    right_on="id_parlparse",
    suffixes=("", "_epparty")
)

# --- Fill missing on_behalf_of_id from EP org_id ---
merged["on_behalf_of_id"] = merged["on_behalf_of_id"].combine_first(merged["org_id"])

# --- Fill gender (prefer EP if available) ---
if "gender" in merged.columns and "gender_ep" in merged.columns:
    merged["gender"] = merged["gender"].combine_first(merged["gender_ep"])
    merged = merged.drop(columns=["gender_ep"], errors="ignore")

# --- Reconcile Wikidata + Datadotparl IDs ---
if "id_wikidata_ep" in merged.columns:
    merged["id_wikidata"] = merged["id_wikidata"].combine_first(merged["id_wikidata_ep"])
    merged = merged.drop(columns=["id_wikidata_ep"], errors="ignore")

if "id_datadotparl_ep" in merged.columns:
    merged["id_datadotparl"] = merged["id_datadotparl"].combine_first(merged["id_datadotparl_ep"])
    merged = merged.drop(columns=["id_datadotparl_ep"], errors="ignore")

print(f"🔹 Combined rows (should equal Parlparse): {len(merged)}")


# --- Create a list of aliases for each name based on other_names ---
def normalize_name(name):
    if pd.isna(name):
        return None
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return " ".join(name.split())

def extract_aliases(obj):
    """Extract aliases (personal + peerage) from other_names, using only dicts with note == 'Main'."""
    aliases = set()

    if obj is None:
        return []

    # Walk nested structure and collect dicts
    to_visit, dicts = [obj], []
    while to_visit:
        current = to_visit.pop()
        if isinstance(current, dict):
            if current.get("note") == "Main":  # ✅ only keep Main entries
                dicts.append(current)
        elif isinstance(current, (list, tuple, np.ndarray)):
            try:
                for x in current:
                    to_visit.append(x)
            except Exception:
                continue

    # Build aliases from the filtered dicts
    for d in dicts:
        given = d.get("given_name")
        family = d.get("family_name")
        addname = d.get("additional_name")
        surname = d.get("surname")
        prefix = d.get("honorific_prefix")
        lordname = d.get("lordname")
        lordof = d.get("lordofname_full")

        # Personal names
        if given and family:
            aliases.add(f"{given} {family}")
            initials = "".join([t[0] for t in given.split() if t])
            if initials:
                aliases.add(f"{initials} {family}")
                aliases.add(" ".join(list(initials)) + f" {family}")
                aliases.add(f"{initials[0]} {family}")
        elif given:
            aliases.add(given)
        elif family:
            aliases.add(family)

        if addname:
            aliases.add(addname)
        if surname:
            aliases.add(surname)

        # Peerage forms
        if prefix and lordname:
            aliases.add(f"{prefix} {lordname}")
            if lordof:
                aliases.add(f"{prefix} {lordname} of {lordof}")
        if lordname and lordof:
            aliases.add(f"{lordname} of {lordof}")
        if prefix and lordof:  # e.g. "Lord of Rothiemay"
            aliases.add(f"{prefix} of {lordof}")
            aliases.add(f"{prefix} {lordof}")

    return [a.strip() for a in aliases if a.strip()]

# Extract alias lists
merged["aliases"] = merged["other_names"].map(extract_aliases)
merged["aliases_norm"] = merged["aliases"].apply(lambda lst: [normalize_name(x) for x in lst])

# Add EP names to aliases
def add_ep_names(row):
    aliases = set(row.get("aliases_norm", []) or [])
    name = normalize_name(row.get("name_ep"))
    sort_name = normalize_name(row.get("sort_name_ep"))
    given_name = normalize_name(row.get("given_name_ep"))
    family_name = normalize_name(row.get("family_name_ep"))

    for n in [name, sort_name, given_name, family_name]:
        if (n not in aliases) and (n is not None):
            aliases.add(n)
    return list(aliases)

merged["aliases_norm"] = merged.apply(add_ep_names, axis=1)
# Drop duplicates in aliases lists
merged["aliases_norm"] = merged["aliases_norm"].apply(lambda lst: list(set(lst)))
print("Example aliases:", merged.iloc[3426][["aliases_norm"]].values[0])

# --- Clean up temporary columns ---
merged = merged.drop(columns=["id_parlparse", "org_id", "id_parlparse_epparty", "aliases", "name", 
                              "sort_name", "given_name", "family_name"], errors="ignore")

# # --- Add suffix to name columns ---
# merged.rename(columns={'name': 'name_ep', 
#                        'sort_name': 'sort_name_ep',
#                        'given_name': 'given_name_ep',
#                        'family_name': 'family_name_ep'}, inplace=True)

# Save
merged.to_parquet(OUT_PATH, index=False)
print(f"✅ Saved combined parquet → {OUT_PATH}")