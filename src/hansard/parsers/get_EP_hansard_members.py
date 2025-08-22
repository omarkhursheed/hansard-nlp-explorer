import json
import pandas as pd
from pathlib import Path
from datetime import datetime

IN_PATH = Path("src/hansard/data/hansard_raw/Everypolitician_git/ep_people.json")
OUT_PATH = Path("src/hansard/data/processed_fixed/metadata/house_members/EP_commons_members.parquet")

print("🔹 Loading Popolo people.json…")
with open(IN_PATH, "r") as f:
    data = json.load(f)

persons = pd.DataFrame(data.get("persons", []))
orgs = pd.DataFrame(data.get("organizations", []))
memberships = pd.DataFrame(data.get("memberships", []))

# --- Clean organizations ---
orgs_clean = orgs[["classification", "id", "name"]].copy()

# --- Process persons ---
def filter_identifiers(identifiers):
    if not isinstance(identifiers, list):
        return {}
    keep_schemes = {"parlparse", "wikidata", "datadotparl", "parliamentdotuk"}
    result = {}
    for ident in identifiers:
        scheme = ident.get("scheme")
        value = ident.get("identifier")
        if scheme in keep_schemes and value:
            result[f"id_{scheme}"] = value
    return result

def filter_links(links):
    if not isinstance(links, list):
        return None
    for link in links:
        if "wikipedia" in link.get("note", "").lower():
            return link.get("url")
    return None

person_rows = []
for p in persons.to_dict("records"):
    base = {
        "id": p.get("id"),
        "name": p.get("name"),
        "sort_name": p.get("sort_name"),
        "given_name": p.get("given_name"),
        "family_name": p.get("family_name"),
        "birth_date": p.get("birth_date"),
        "gender": p.get("gender"),
        "wikipedia": filter_links(p.get("links", []))
    }
    # add selected identifiers
    base.update(filter_identifiers(p.get("identifiers", [])))
    person_rows.append(base)

persons_clean = pd.DataFrame(person_rows)

# --- Clean memberships (keep raw, expand identifiers if needed) ---
def expand_identifiers(df, col, prefix):
    rows = []
    for row in df.to_dict("records"):
        base = {k: v for k, v in row.items() if k != col}
        idents = row.get(col, [])
        if isinstance(idents, list):
            for ident in idents:
                scheme = ident.get("scheme")
                value = ident.get("identifier")
                if scheme and value is not None:
                    base[f"{prefix}_{scheme}"] = value
        rows.append(base)
    return pd.DataFrame(rows)

memberships_clean = expand_identifiers(memberships, "identifiers", "membership_id")

# --- Merge all ---
merged = memberships_clean.merge(
    persons_clean, left_on="person_id", right_on="id", suffixes=("", "_person")
)

merged = merged.merge(
    orgs_clean.add_prefix("org_"),
    left_on="organization_id",
    right_on="org_id",
    how="left"
)

# --- Parse dates if present ---
def to_date(x):
    if not x:
        return None
    try:
        return datetime.fromisoformat(str(x)[:10])
    except Exception:
        return None

merged["membership_start_date"] = merged["start_date"].map(to_date)
merged["membership_end_date"] = merged["end_date"].map(to_date)

print(f"🔹 Total memberships: {len(merged)}")
print(f"🔹 Unique persons: {merged['person_id'].nunique()}")

print("\nSample row:\n", merged.iloc[0].to_dict())

# Save everything
merged.to_parquet(OUT_PATH, index=False)
print(f"✅ Saved merged dataset to {OUT_PATH}")