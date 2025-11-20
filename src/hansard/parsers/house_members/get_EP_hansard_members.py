"""
EveryPolitician Popolo -> Commons Members (Parquet)

Loads the Popolo `ep_people.json` exported from EveryPolitician (EP), extracts
persons, organizations, and memberships, flattens selected identifiers, merges
the three entities into a single table, parses membership start/end dates, and
writes the result to a Parquet file for downstream Hansard analysis.
"""

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
    """
    Select a curated subset of person identifiers from a Popolo-style list.

    Keeps only identifiers whose 'scheme' is one of:
      {'parlparse', 'wikidata', 'datadotparl', 'parliamentdotuk'}

    Parameters
    ----------
    identifiers : list[dict] | Any
        List of {'scheme': str, 'identifier': str} dicts, or other/None.

    Returns
    -------
    dict
        Mapping like {'id_wikidata': 'Q42', 'id_parlparse': '...'} for the
        kept schemes; empty dict if input is not a list or none matched.
    """
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
    """
    Return the first Wikipedia URL from a Popolo-style 'links' list.

    Parameters
    ----------
    links : list[dict] | Any
        List of link dicts that may include 'note' and 'url' keys.

    Returns
    -------
    str | None
        Wikipedia URL if found; otherwise None.
    """
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
    """
    Expand a column of identifier dicts into flat columns with a prefix.

    For each row, looks in `df[col]` for a list of dicts with keys
    {'scheme','identifier'} and adds '{prefix}_{scheme}' columns containing
    the identifier values. Non-list/empty values are ignored.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the identifiers column.
    col : str
        Name of the column holding the list of identifier dicts.
    prefix : str
        Prefix for generated columns (e.g., 'membership_id').

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the same rows as `df` and the expanded columns.
    """
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
    """
    Parse a date-like value to a `datetime` (YYYY-MM-DD precision) or None.

    Parameters
    ----------
    x : Any
        String or value convertible to ISO date prefix; falsy -> None.

    Returns
    -------
    datetime | None
        Parsed datetime at day precision, or None if parsing fails.
    """
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
