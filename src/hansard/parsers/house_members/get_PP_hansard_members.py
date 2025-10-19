"""
ParlParse Popolo -> House Members (1803–2005)

Loads ParlParse's Popolo `people.json`, expands identifiers, merges persons,
memberships, posts, and organizations into a single table, normalizes dates,
filters memberships to Hansard's coverage window (1803–2005), performs light
column cleanup/deduplication, and writes a Parquet for downstream analysis.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

IN_PATH = Path("src/hansard/data/hansard_raw/Parlparse_git/people.json")   # downloaded people.json
OUT_PATH = Path("src/hansard/data/processed_fixed/metadata/house_members/PP_house_members_1803_2005.parquet")

# Bounds for Hansard coverage
START_DATE = pd.Timestamp("1803-01-01")
END_DATE   = pd.Timestamp("2005-12-31")

print("🔹 Loading people.json…")
with open(IN_PATH, "r") as f:
    data = json.load(f)

memberships = pd.DataFrame(data["memberships"])
persons = pd.DataFrame(data["persons"])
posts = pd.DataFrame(data["posts"])
orgs = pd.DataFrame(data["organizations"])

# --- Normalize identifiers into wide format ---
def expand_identifiers(df, col, prefix="id"):
    """Expand a list-of-identifiers column into wide columns keyed by scheme.

    For each row, reads `df[col]` as a list of dicts with keys
    {'scheme','identifier'} and creates new columns named
    `{prefix}_{scheme}` containing the corresponding identifier value.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame that contains the identifiers column.
    col : str
        Name of the column holding the list of identifier dicts.
    prefix : str, default "id"
        Prefix to use for generated column names.

    Returns
    -------
    pandas.DataFrame
        A new dataframe with the same number of rows as `df` and additional
        flattened identifier columns.
    """
    rows = []
    for row in df.to_dict("records"):
        base = {k: v for k, v in row.items() if k != col}
        identifiers = row.get(col, [])
        if isinstance(identifiers, list):
            for ident in identifiers:
                scheme = ident.get("scheme")
                value = ident.get("identifier")
                if scheme and value is not None:
                    base[f"{prefix}_{scheme}"] = value
        rows.append(base)
    return pd.DataFrame(rows)

persons_expanded = expand_identifiers(persons, "identifiers", prefix="id")
orgs_expanded = expand_identifiers(orgs, "identifiers", prefix="id")
memberships_expanded = expand_identifiers(memberships, "identifiers", prefix="mem_id")

# --- Person name cleanup ---
def extract_name(row):
    """Extract the canonical display name from `other_names`.

    Picks the entry with `note == "Main"` if present; otherwise falls back to
    the first entry. Concatenates its `given_name` and `family_name`.

    Parameters
    ----------
    row : dict
        Row from the persons dataframe.

    Returns
    -------
    str | None
        "Given Family" if discoverable; otherwise None.
    """
    names = row.get("other_names")
    if not isinstance(names, list) or len(names) == 0:
        return None
    main = next((n for n in names if n.get("note") == "Main"), names[0])
    given = main.get("given_name", "")
    family = main.get("family_name", "")
    return " ".join(filter(None, [given, family]))

persons_expanded["person_name"] = persons.apply(extract_name, axis=1)

# --- Merge memberships with persons ---
merged = memberships_expanded.merge(
    persons_expanded, left_on="person_id", right_on="id", suffixes=("", "_person")
)

# --- Merge with posts (Commons seats); use left join so Lords are kept ---
merged = merged.merge(
    posts.add_prefix("post_"), left_on="post_id", right_on="post_id", how="left"
)

# --- Merge with orgs (parties, house of commons/lords, etc.) ---
merged = merged.merge(
    orgs_expanded.add_prefix("org_"), left_on="organization_id", right_on="org_id", how="left"
)

# --- Date normalization ---
def to_date(x):
    """Parse ISO-like or year-only strings to `datetime`, else None.

    Accepts:
      - 'YYYY-MM-DD' or longer ISO strings (uses first 10 chars)
      - 'YYYY' (treated as January 1 of that year)

    Parameters
    ----------
    x : Any
        Date-like value (usually a string).

    Returns
    -------
    datetime | None
        Parsed datetime at day precision, or None if parsing fails/empty.
    """
    if not isinstance(x, str) or not x.strip():
        return None
    try:
        if len(x) == 4 and x.isdigit():       # year only
            return datetime(int(x), 1, 1)
        return datetime.fromisoformat(x[:10]) # full or partial ISO
    except Exception:
        return None

merged["membership_start_date"] = merged["start_date"].map(to_date)
merged["membership_end_date"]   = merged["end_date"].map(to_date)

# --- Filter memberships: keep only if start_date is within range ---
merged = merged[
    (merged["membership_start_date"].notna()) &
    (merged["membership_start_date"] >= START_DATE) &
    (merged["membership_start_date"] <= END_DATE)
]

# Drop duplicate coluumns
def drop_duplicate_hashable_columns(df):
    """Drop duplicate columns among hashable dtypes, preserving unhashables.

    Splits columns into hashable vs unhashable (lists/dicts/sets). Removes
    duplicated columns only among hashables (via transpose + duplicated()),
    then recombines with the unhashable columns intact.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with a mix of column dtypes.

    Returns
    -------
    pandas.DataFrame
        Dataframe with duplicate hashable columns removed.
    """
    # Separate hashable vs unhashable columns
    hashable_cols = []
    unhashable_cols = []
    
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict, set))).any():
            unhashable_cols.append(col)
        else:
            hashable_cols.append(col)

    # Work only on hashable columns
    df_hashable = df[hashable_cols]
    df_unhashable = df[unhashable_cols]

    # Drop duplicate columns among hashables
    df_hashable = df_hashable.loc[:, ~df_hashable.T.duplicated()]

    # Combine back hashable + unhashable
    return pd.concat([df_hashable, df_unhashable], axis=1)

# Usage
merged = drop_duplicate_hashable_columns(merged)

# --- Final Cleanup ---
# Assign lords values to common columns if null and lord-specific columns
merged['organization_id'] = merged['organization_id'].fillna(merged['post_organization_id'])
merged['post_role'] = merged['post_role'].fillna(merged['label'])
merged.drop(columns=['post_organization_id','org_classification','org_name','label'], inplace=True)
# Drop null coluumns
merged.dropna(axis=1, how='all', inplace=True)
# Drop exloeded columns
merged.drop(columns=['post_identifiers', 'post_area'], inplace=True)
# Rename membership id for clarity
merged.rename(columns={'id': 'membership_id'}, inplace=True)
# Rename columns for clarity
merged.rename(columns={'mem_id_historichansard_id': 'mem_id_historichansard', 
                       'id_historichansard_person_id': 'id_historichansard_person',
                       'id_datadotparl_id': 'id_datadotparl',
                       'id_pims_id': 'id_pims',
                       'id_scotparl_id': 'id_scotparl'}, inplace=True)




print(f"🔹 Raw merged memberships: {len(memberships)}")
print(f"🔹 Memberships in range ≤ {END_DATE.date()}: {len(merged)}")
print(f"🔹 Unique persons in range: {merged['person_id'].nunique()}")


# --- Save all data ---
merged.to_parquet(OUT_PATH, index=False)
print(f"✅ Saved full dataset to {OUT_PATH}")
