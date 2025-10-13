# -*- coding: utf-8 -*-
"""
Full pipeline: Merge Parlparse members with Historic Hansard speakers.

Priority (in order):
  1) slug (members.id_historichansard_url) == hansard.slug
  2) name_key  (surname + given/initial constraint, time sanity, constituency bonus)
  3) alias_keys (members aliases) -> hansard name_key (same constraints)
  4) lords_alias_keys (hansard titles_in_lords) -> members aliases_norm (time + constituency; no surname req)

Then, append Hansard-only speakers (no Parlparse match).
Finally, clean up columns per project rules.

Run this file directly. Paths are hardcoded in the CONFIG section.
Tested on Python 3.10+ with pandas >= 1.5.
"""

import re
import ast
import json
import pandas as pd
from pandas import Series
from pathlib import Path

# ========================= CONFIG: EDIT THESE PATHS ========================= #
# You asked for hardcoded paths. Update these to your local files if needed.
MEMBERS_PARQUET    = Path("src/hansard/data/processed_fixed/metadata/house_members/PP_EP_house_members_combined.parquet")
HANSARD_PARQUET    = Path("src/hansard/data/processed_fixed/metadata/house_members/historic_hansard_speakers.parquet")
HONORIFICS_JSON    = Path("src/hansard/data/word_lists/gendered_honorifics.json")
OUT_PATH     = Path("src/hansard/data/processed_fixed/metadata/house_members/PP_HH_members_combined.parquet")
# ==========================================================================- #


# ============================
# Helpers
# ============================

def build_honorific_sets(hmap: dict):
    return {h.lower().rstrip(".") for grp in ("Male","Female","Unknown") for h in hmap.get(grp, [])}

with open(HONORIFICS_JSON, "r") as f:
        honorific_map = json.load(f)
HSET = build_honorific_sets(honorific_map)

FILLER_WORDS = {"of","the","and"}

def make_key(x, H):
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    s = str(x).lower()
    s = re.sub(r"\([^)]*\)", " ", s)                      # drop parens
    s = re.sub(r"[^a-z\s'.-]", " ", s)                    # keep letters/basic
    s = s.replace(",", " ")
    toks = []
    for t in s.split():
        for p in t.replace("-", " ").split():
            p = p.strip(".-'")
            if not p or p in FILLER_WORDS or p in H or len(p) <= 1: continue
            toks.append(p)
    return " ".join(sorted(toks)) if toks else None

def alias_keys(val, H):
    if isinstance(val, (list, tuple, Series)):
        out = [make_key(a, H) for a in val]
        return [k for k in out if k and len(k.split()) >= 2]
    k = make_key(val, H)
    return [k] if (k and len(k.split()) >= 2) else []

def split_name_parts(key: str):
    if not key: return None, set()
    toks = key.split()
    surname = max(toks, key=len)
    return surname, set(toks) - {surname}

def initials(given: set) -> set:
    return {g[0] for g in given if g}

# Keep honorifics but handle them separately from core tokens
HON_GENERIC = {"mr","mrs","ms","miss","mx","dr","prof"}
HON_PEERAGE = {x for x in HSET if x not in HON_GENERIC}
HON_ALL = HON_GENERIC | HON_PEERAGE

def _swap_commas(s: str) -> str:
    # "Surname, Given (Mr)" -> "Given Surname (Mr)"
    if "," in s:
        a, b = [p.strip() for p in s.split(",", 1)]
        if a and b: return f"{b} {a}"
    return s

def _name_tokens_keep_hon(s) -> set:
    if s is None or (isinstance(s, float) and pd.isna(s)): return set()
    s = _swap_commas(str(s))
    s = re.sub(r"[()]", " ", s)                 # keep inner text like (Mr) -> Mr
    s = s.lower()
    s = re.sub(r"[^a-z'\- ]", " ", s)
    toks = [t.strip("'") for t in re.split(r"[\s\-]+", s) if t]
    return set(toks)

def _split_core_hon(tokens: set) -> tuple[set, set]:
    core = {t for t in tokens if t not in HON_ALL}
    hon  = {t for t in tokens if t in HON_ALL}
    return core, hon

def _tokens_from_list_keep_hon(v) -> set:
    # Like your _ensure_list, but returns token set using _name_tokens_keep_hon
    if v is None or (isinstance(v, float) and pd.isna(v)): return set()
    if isinstance(v, (list, tuple, set)):
        vals = [str(x) for x in v]
    else:
        s = str(v).strip()
        if not s: return set()
        if s.startswith("[") and s.endswith("]"):
            try:
                import json as _json
                vals = [str(x) for x in _json.loads(s)]
            except Exception:
                vals = [p.strip() for p in s.split(",") if p.strip()]
        else:
            vals = [p.strip() for p in s.split(",") if p.strip()]
    out = set()
    for x in vals:
        out |= _name_tokens_keep_hon(x)
    return out

def year_from_date(val):
    # Robust: handle pandas NaT / NaN and strings
    try:
        ts = pd.to_datetime(val, errors='coerce')
        if pd.isna(ts):
            return None
        return int(ts.year)
    except Exception:
        m = re.search(r'(\d{4})', str(val))
        return int(m.group(1)) if m else None

def time_bonus(ms, me, b, d) -> float:
    s = 0.0
    if ms is not None:
        if b is not None:
            age = ms - b
            s += 1.0 if 18 <= age <= 100 else (-1.0 if age < 16 else 0.0)
        if d is not None and d < ms: s -= 1.0
    if me is not None:
        if b is not None and b > me: s -= 1.0
        if d is not None and d < me: s -= 0.5
    if all(v is not None for v in [ms,me,b,d]) and (b <= ms <= d and b <= me <= d): s += 0.5
    return s

STOPWORDS_LOC = {"of","and","the"}
def normalize_constituency(s: str):
    s = str(s).lower().replace("&"," and ")
    s = re.sub(r"[^\w\s]"," ", s)
    s = re.sub(r"\s+"," ", s).strip()
    toks = [t for t in s.split() if t not in STOPWORDS_LOC]
    return " ".join(toks) if toks else None

def extract_member_constituency(post_label: str):
    m = re.search(r"\bfor\s+(.+)$", str(post_label), flags=re.I)
    return normalize_constituency(m.group(1)) if m else None

def parse_hansard_const_list(val):
    if isinstance(val, str):
        try: arr = json.loads(val)
        except Exception: arr = []
    elif isinstance(val, list):
        arr = val
    else:
        arr = []
    out = []
    for item in arr:
        if isinstance(item, dict):
            c = item.get("constituency")
            if c: out.append(c)
        elif isinstance(item, str):
            out.append(item)
    return out

def constituency_score(member_const_norm, hansard_const_norm_set) -> float:
    if not member_const_norm or not hansard_const_norm_set: return 0.0
    if member_const_norm in hansard_const_norm_set: return 2.0
    mtoks = set(member_const_norm.split()); best = 0.0
    for h in hansard_const_norm_set:
        htoks = set(h.split()); inter = len(mtoks & htoks); uni = len(mtoks | htoks) or 1
        j = inter/uni
        if inter >= 1 and j >= 0.3: best = max(best, 1.5 if j >= 0.6 else 1.0)
    return best

# Clean titles_in_lords → list[str] (lowercased, dates removed)
MONTHS_RE = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"

def parse_titles_in_lords(val):
    # Accept JSON string or list
    if isinstance(val, str):
        try:
            arr = json.loads(val)
        except Exception:
            arr = [val]
    elif isinstance(val, list):
        arr = val
    else:
        arr = []

    out, seen = [], set()
    for s in arr:
        if not isinstance(s, str):
            continue
        t = s.strip()

        # remove anything from the first month name to the end (e.g., "April 30, 1908 …")
        t = re.sub(rf"\b{MONTHS_RE}\b.*$", "", t, flags=re.IGNORECASE)
        # also remove anything from the first 4-digit year to the end (covers "1880 - …")
        t = re.sub(r"\b\d{4}\b.*$", "", t)

        # trim trailing punctuation and whitespace, lowercase
        t = re.sub(r"[-–—,:;]+\s*$", "", t)
        t = re.sub(r"\s+", " ", t).strip().lower()

        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out

def lord_title_keys(val, H):
    return sorted({make_key(t, H) for t in parse_titles_in_lords(val) if make_key(t, H)})

def extract_honorific_from_name(name: str):
    m = re.findall(r"\(([^)]+)\)", str(name))
    if not m: return None
    hon = " ".join(m).replace("."," ")
    return re.sub(r"\s+"," ", hon).strip() or None

def normalize_person_name_from_hansard(name):
    # return None if name is missing/placeholder
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    s = str(name).strip()
    if s.lower() in {"", "nan", "none"}:
        return None

    # remove parenthetical honorifics and flip "Last, First" -> "First Last"
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return s if s else None

def parse_hansard_const_objs(val):
    if isinstance(val, str):
        try: arr = json.loads(val)
        except Exception: arr = []
    elif isinstance(val, list):
        arr = val
    else:
        arr = []
    out = []
    for item in arr:
        if isinstance(item, dict):
            out.append({"constituency": item.get("constituency"), "from_to": item.get("from_to","")})
    return out

def _years_from_from_to(s: str):
    yrs = re.findall(r"(\d{4})", str(s))
    if not yrs: return (None, None)
    return (int(yrs[0]), int(yrs[-1]))

def _const_windows_from(val):
    win = {}
    for obj in parse_hansard_const_objs(val):
        cn = normalize_constituency(obj.get("constituency"))
        s,e = _years_from_from_to(obj.get("from_to",""))
        if cn:
            win.setdefault(cn, []).append((s,e))
    return win

def window_overlap_score(ms, me, wins):
    if not wins and wins != []: return 0.0
    if ms is None and me is None: return 0.0
    if ms is None: ms = me
    if me is None: me = ms
    for (s,e) in wins:
        if s is None and e is None: continue
        if s is None: s = ms
        if e is None: e = s
        if not (me < s or ms > e): return 1.0
        if abs(ms - e) <= 2 or abs(s - me) <= 2: return 0.5
    return 0.0


# ---------- Build constituency features needed for the fallback ----------

CONST_STOPWORDS = r"(?:member of parliament for|mp for|mla for|ms for|the|county of|borough of|district of|city of)"
def norm_const(s):
    if s is None or (isinstance(s, float) and pd.isna(s)): 
        return None
    s = str(s).lower().strip()
    s = s.replace("&", " and ").replace("–", "-").replace("—", "-")
    s = re.sub(r"\b(member of parliament for|mp for|the|county of|borough of|district of|city of)\b", " ", s)
    s = re.sub(r"[^a-z\- &]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

def _parse_hh_const(val):
    """
    val like:
      '[{"constituency":"Rugby and Kenilworth","from_to":"May  1, 1997 - May  5, 2005"}]'
    returns (set_of_norm_const, {const: [(start_year, end_year), ...]})
    """
    if isinstance(val, list):
        arr = val
    elif isinstance(val, str):
        s = val.strip()
        try:
            arr = json.loads(s)
        except Exception:
            try:
                arr = ast.literal_eval(s)  # handles single quotes / weird spacing
            except Exception:
                arr = []
    else:
        arr = []

    consts, windows = set(), {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        c = norm_const(item.get("constituency"))
        if not c:
            continue
        yrs = re.findall(r"\b(\d{4})\b", str(item.get("from_to", "")))
        sy = int(yrs[0]) if yrs else None
        ey = int(yrs[-1]) if len(yrs) >= 2 else None
        consts.add(c)
        windows.setdefault(c, []).append((sy, ey))
    return consts, windows

CONST_STOP_TOKENS = {"and", "of", "the"}

def const_join_key(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    toks = [t for t in re.split(r"[ \-]+", str(s).lower()) if t]
    toks = [t for t in toks if t not in CONST_STOP_TOKENS]
    return " ".join(sorted(toks)) or None

def _const_tokens(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    toks = [t for t in re.split(r"[ \-]+", str(s).lower()) if t]
    toks = [t for t in toks if t not in CONST_STOP_TOKENS]
    return sorted(set(toks))


# ============================================================
# Main merge function
# ============================================================

def propagate_within_person(result: pd.DataFrame,
                            fields: list[str] | None = None,
                            rank_map: dict[str, int] | None = None) -> pd.DataFrame:
    """
    For any person_id where at least one stint already has a Hansard slug,
    propagate the canonical (mode) slug + selected fields to that person's
    other stints. Newly filled stints get match_level='person_propagated'.

    Returns the same DataFrame object (mutated) for convenience.
    """

    if "person_id" not in result.columns or "slug" not in result.columns:
        return result  # nothing to do

    # Default ranking (lower = better). Extend safely for newer passes.
    if rank_map is None:
        rank_map = {
            "slug": 1,
            "name_key_const": 2,
            "name_key": 3,
            "constituency_fallback": 4,
            "name_fuzzy": 5,
            "constituency_fuzzy": 6,
            "lords_alias": 7,
            "alias_key": 8,
            "person_propagated": 9,
        }

    # Which fields to carry from the canonical row
    if fields is None:
        candidate_fields = [
            "name", "slug", "url",
            "list_birth_year", "list_death_year",
            "b_year", "d_year",
            "constituencies", "const_set",
        ]
        fields = [c for c in candidate_fields if c in result.columns]

    had_slug = result["slug"].notna().copy()

    # Rows already matched to a slug and having a person_id
    matched = result[result["slug"].notna() & result["person_id"].notna()].copy()
    if matched.empty:
        return result

    # Rank & tie-breakers (use the matched row’s own m_start_year)
    matched["_rank"] = matched["match_level"].map(rank_map).fillna(99).astype(int)
    if "m_start_year" in matched.columns:
        matched["_mstart"] = matched["m_start_year"]
    else:
        matched["_mstart"] = pd.NA

    # Mode slug per person_id
    mode_slug = (
        matched.groupby("person_id")["slug"]
               .agg(lambda s: s.value_counts().index[0])  # first most frequent
               .rename("mode_slug")
    )
    matched = matched.merge(mode_slug, on="person_id", how="left")

    # Canonical row per person_id: the row whose slug==mode_slug, then best rank, earliest start
    canon = (
        matched[matched["slug"] == matched["mode_slug"]]
        .sort_values(["person_id", "_rank", "_mstart", "slug"], kind="mergesort")
        .drop_duplicates("person_id")
    )

    # Prepare canonical columns for merge
    ccols = {c: f"{c}__canon" for c in fields}
    take_cols = ["person_id"] + fields
    canon = canon[take_cols].rename(columns=ccols)

    # Merge the canonical payload and fill ONLY missing values on other stints
    result = result.merge(canon, on="person_id", how="left")
    for c in fields:
        cc = f"{c}__canon"
        if cc in result.columns:
            result[c] = result[c].combine_first(result[cc])
            result.drop(columns=[cc], inplace=True)

    # Mark rows that gained a slug via person-level propagation
    gained = (~had_slug) & result["slug"].notna()
    if "match_level" in result.columns:
        result.loc[gained, "match_level"] = "person_propagated"
    else:
        result.loc[gained, "match_level"] = "person_propagated"

    # Keep url id aligned when we just filled a slug
    if "id_historichansard_url" in result.columns:
        result.loc[gained & result["id_historichansard_url"].isna(), "id_historichansard_url"] = result.loc[gained, "slug"]

    return result

def merge_members_and_hansard(members_df: pd.DataFrame, hansard_df: pd.DataFrame, honorific_map: dict) -> pd.DataFrame:

    members = members_df.copy()
    hansard = hansard_df.drop_duplicates("slug").copy()

    members["__rid__"] = range(len(members))
    members["slug_from_url"] = members["id_historichansard_url"]
    members["name_key"]   = members["person_name"].apply(lambda x: make_key(x, HSET))
    members["alias_keys"] = members["aliases_norm"].apply(lambda v: alias_keys(v, HSET))

    hansard["name_key"] = hansard["name"].apply(lambda x: make_key(x, HSET))
    hansard["slug_key"] = hansard["slug"].str.replace("-", " ", regex=False).apply(lambda x: make_key(x, HSET))

    mk_parts = members["name_key"].apply(split_name_parts)
    members["mk_surname"]  = mk_parts.apply(lambda x: x[0] if x else None)
    members["mk_given"]    = mk_parts.apply(lambda x: x[1] if x else set())
    members["mk_initials"] = members["mk_given"].apply(initials)

    hk_parts = hansard["name_key"].apply(split_name_parts)
    hansard["hk_surname"]  = hk_parts.apply(lambda x: x[0] if x else None)
    hansard["hk_given"]    = hk_parts.apply(lambda x: x[1] if x else set())
    hansard["hk_initials"] = hansard["hk_given"].apply(initials)

    # Years
    for c in ("membership_start_date","membership_end_date","post_start_date","post_end_date"):
        members[c] = pd.to_datetime(members[c], errors="coerce")
    members["m_start_year"] = members["membership_start_date"].apply(year_from_date)
    members["m_end_year"]   = members["membership_end_date"].apply(year_from_date)

    hansard["b_year"] = hansard["list_birth_year"].apply(year_from_date)
    hansard["d_year"] = hansard["list_death_year"].apply(year_from_date)

    # Constituencies
    members["const_norm"] = members["post_label"].apply(extract_member_constituency)
    hansard["const_set"]     = hansard["constituencies"].apply(lambda v: {normalize_constituency(c) for c in parse_hansard_const_list(v)})
    hansard["const_windows"] = hansard["constituencies"].apply(_const_windows_from)

    # Lords alias keys
    hansard["lords_alias_keys"] = hansard["titles_in_lords"].apply(lambda v: lord_title_keys(v, HSET))

    # PASS 1: slug
    H_KEEP = ["name","slug","url","list_birth_year","list_death_year","b_year","d_year","constituencies","const_set","const_windows","lords_alias_keys"]
    pass1 = members.merge(hansard[H_KEEP], left_on="slug_from_url", right_on="slug", how="left")
    pass1["match_level"] = pass1["slug"].notna().map({True:"slug", False:pd.NA})

    # PASS 2a: name_key
    unmatched = pass1["slug"].isna()
    left_cols = ["__rid__","name_key","mk_surname","mk_given","mk_initials","m_start_year","m_end_year","const_norm"]
    right_cols = H_KEEP + ["name_key","hk_surname","hk_given","hk_initials"]
    c2a = pass1.loc[unmatched, left_cols].merge(hansard[right_cols], on="name_key", how="left")

    def good_given(r):
        if r["mk_surname"] != r["hk_surname"]: return False
        g_m, g_h = r["mk_given"], r["hk_given"]
        if g_m & g_h: return True
        return bool(initials(g_m) & r["hk_initials"])

    c2a = c2a[c2a.apply(good_given, axis=1)]
    def s2a(r):
        s = 0.0
        s += min(2, len(r["mk_given"] & r["hk_given"])) * 1.0
        if initials(r["mk_given"]) & r["hk_initials"]: s += 0.5
        s += 0.1 * (len(str(r["name_key"])) // 5)
        s += time_bonus(r["m_start_year"], r["m_end_year"], r["b_year"], r["d_year"])
        cn = r["const_norm"]; s += constituency_score(cn, r["const_set"])
        if cn in r["const_windows"]: s += window_overlap_score(r["m_start_year"], r["m_end_year"], r["const_windows"][cn])
        return s
    if not c2a.empty:
        c2a["_score"] = c2a.apply(s2a, axis=1)
        c2a["_rank"]  = c2a.groupby("__rid__")["_score"].rank(ascending=False, method="dense")
        top2a = c2a[c2a["_rank"] == 1].copy()
        amb2a = set(top2a["__rid__"].value_counts()[lambda s: s>1].index)
    else:
        top2a = pd.DataFrame(columns=["__rid__"]); amb2a = set()

    # PASS 2b: aliases -> name_key
    still_unmatched = pass1["slug"].isna() & ~pass1["__rid__"].isin(set(top2a["__rid__"]) - amb2a)
    alias_src = pass1.loc[still_unmatched, ["__rid__","alias_keys","mk_surname","mk_given","mk_initials","m_start_year","m_end_year","const_norm"]]
    aliases_long = alias_src.explode("alias_keys").dropna(subset=["alias_keys"])

    def alias_parts(k):
        s, g = split_name_parts(k); return s, g, initials(g)

    if not aliases_long.empty:
        parts = aliases_long["alias_keys"].apply(alias_parts)
        aliases_long["ak_surname"] = parts.apply(lambda x: x[0])
        aliases_long["ak_given"]   = parts.apply(lambda x: x[1])
        aliases_long["ak_init"]    = parts.apply(lambda x: x[2])

        right_cols = H_KEEP + ["name_key","hk_surname","hk_given","hk_initials"]
        c2b = aliases_long.merge(hansard[right_cols], left_on="alias_keys", right_on="name_key", how="left")

        def good_alias(r):
            if r["mk_surname"] != r["hk_surname"]: return False
            g_a = r["ak_given"] or r["mk_given"]
            if g_a & r["hk_given"]: return True
            return bool(initials(g_a) & r["hk_initials"])

        c2b = c2b[c2b.apply(good_alias, axis=1)]
        if not c2b.empty:
            def s2b(r):
                s = 0.0
                g_a = r["ak_given"] or r["mk_given"]
                s += min(2, len(g_a & r["hk_given"])) * 1.0
                if initials(g_a) & r["hk_initials"]: s += 0.5
                s += 0.1 * (len(str(r["name_key"])) // 5)
                s += time_bonus(r["m_start_year"], r["m_end_year"], r["b_year"], r["d_year"])
                cn = r["const_norm"]; s += constituency_score(cn, r["const_set"])
                if cn in r["const_windows"]: s += window_overlap_score(r["m_start_year"], r["m_end_year"], r["const_windows"][cn])
                return s
            c2b["_score"] = c2b.apply(s2b, axis=1)
            c2b["_rank"]  = c2b.groupby("__rid__")["_score"].rank(ascending=False, method="dense")
            top2b = c2b[c2b["_rank"] == 1].copy()
            amb2b = set(top2b["__rid__"].value_counts()[lambda s: s>1].index)
        else:
            top2b = pd.DataFrame(columns=["__rid__"]); amb2b = set()
    else:
        top2b = pd.DataFrame(columns=["__rid__"]); amb2b = set()

    # PASS 2c: lords_alias_keys -> aliases
    still_unmatched2 = pass1["slug"].isna() & ~pass1["__rid__"].isin(set(top2a["__rid__"])) & ~pass1["__rid__"].isin(set(top2b["__rid__"]))
    alias_src2 = pass1.loc[still_unmatched2, ["__rid__","alias_keys","m_start_year","m_end_year","const_norm"]]
    aliases_long2 = alias_src2.explode("alias_keys").dropna(subset=["alias_keys"])

    _cols = list(dict.fromkeys(H_KEEP + ['lords_alias_keys','name_key','hk_surname','hk_given']))
    lords_long = hansard[_cols].explode('lords_alias_keys').dropna(subset=['lords_alias_keys'])

    if not aliases_long2.empty and not lords_long.empty:
        c2c = aliases_long2.merge(lords_long, left_on="alias_keys", right_on="lords_alias_keys", how="left").dropna(subset=["slug"])
        def s2c(r):
            s = 1.0
            s += 0.1 * (len(str(r["name_key"])) // 5)
            s += time_bonus(r["m_start_year"], r["m_end_year"], r["b_year"], r["d_year"])
            cn = r["const_norm"]; s += constituency_score(cn, r["const_set"])
            if cn in r["const_windows"]: s += window_overlap_score(r["m_start_year"], r["m_end_year"], r["const_windows"][cn])
            return s
        c2c["_score"] = [s2c(r) for _, r in c2c.iterrows()]
        c2c["_rank"]  = c2c.groupby("__rid__")["_score"].rank(ascending=False, method="dense")
        top2c = c2c[c2c["_rank"] == 1].copy()
        amb2c = set(top2c["__rid__"].value_counts()[lambda s: s>1].index)
    else:
        top2c = pd.DataFrame(columns=["__rid__"]); amb2c = set()

    # Apply fills
    result = pass1.copy()
    result = propagate_within_person(result)

    def fill_from_top(top_df, level_label, skip_ids=set()):
        if top_df.empty:
            return
        # Keep only non-skipped rows and pick a single row per __rid__
        good = top_df[~top_df["__rid__"].isin(skip_ids)].copy()
        if "_score" in good.columns:
            good = good.sort_values(["__rid__", "_score"], ascending=[True, False])
        # drop duplicates per __rid__ so we can safely set index and reindex
        good = good.drop_duplicates("__rid__", keep="first").set_index("__rid__")

        aligned = good.reindex(result["__rid__"]).reset_index()
        mask = result["slug"].isna() & result["__rid__"].isin(good.index)

        for c in ["name","slug","url","list_birth_year","list_death_year","b_year","d_year","constituencies","const_set","const_windows"]:
            result.loc[mask, c] = result.loc[mask, c].combine_first(aligned[c])

        result.loc[mask, "match_level"] = result.loc[mask, "match_level"].combine_first(
            aligned["slug"].notna().map({True: level_label, False: pd.NA})
        )

    def resolve_ambiguity_with_constituency(cands_df, amb_ids, label):
        if not amb_ids or cands_df.empty:
            return pd.DataFrame(columns=cands_df.columns)

        amb = cands_df[(cands_df['_rank'] == 1) & (cands_df['__rid__'].isin(amb_ids))].copy()

        # Merge only what's needed from members; avoid duplicating const_norm if it's already in amb
        need = ['__rid__', 'm_start_year', 'm_end_year', 'mk_given']
        if 'const_norm' not in amb.columns:
            need.append('const_norm')
        amb = amb.merge(members[need], on='__rid__', how='left')

        # Helper to pick the first existing column name among candidates
        def _pick_col(df, *names):
            for n in names:
                if n in df.columns:
                    return n
            return None

        # Choose columns safely even if pandas added _x/_y suffixes
        cn_col = _pick_col(amb, 'const_norm', 'const_norm_x', 'const_norm_y')
        cs_col = _pick_col(amb, 'const_set', 'const_set_x', 'const_set_y')
        cw_col = _pick_col(amb, 'const_windows', 'const_windows_x', 'const_windows_y')
        ms_col = _pick_col(amb, 'm_start_year', 'm_start_year_x', 'm_start_year_y')
        me_col = _pick_col(amb, 'm_end_year',   'm_end_year_x',   'm_end_year_y')
        mg_col = _pick_col(amb, 'mk_given', 'mk_given_x', 'mk_given_y')
        hg_col = _pick_col(amb, 'hk_given', 'hk_given_x', 'hk_given_y')

        amb['_cscore'] = amb.apply(lambda r: constituency_score(r[cn_col], r[cs_col]), axis=1)

        def _wins(r):
            cn = r[cn_col]
            wins = r[cw_col].get(cn, []) if isinstance(r[cw_col], dict) else []
            ms = r[ms_col] if ms_col else None
            me = r[me_col] if me_col else None
            return window_overlap_score(ms, me, wins)

        amb['_wscore'] = amb.apply(_wins, axis=1)

        def _as_set(x):
            if isinstance(x, set): return x
            if isinstance(x, (list, tuple)): return set(x)
            return set()

        def _gbonus(r):
            mg = _as_set(r[mg_col]) if mg_col else set()
            hg = _as_set(r[hg_col]) if hg_col else set()
            return 0.25 if (mg and mg.issubset(hg)) else 0.0

        amb['_gbonus'] = amb.apply(_gbonus, axis=1)

        out = []
        for rid, g in amb.groupby('__rid__', sort=False):
            pos = g[g['_cscore'] > 0]
            if len(pos) == 1:
                out.append(pos.iloc[[0]])
                continue
            g = g.sort_values(['_cscore', '_wscore', '_gbonus'],
                            ascending=[False, False, False], kind='mergesort')
            if g.iloc[0]['_cscore'] > 0:
                out.append(g.iloc[[0]])

        if not out:
            return pd.DataFrame(columns=cands_df.columns)
        resolved = pd.concat(out, ignore_index=True)
        resolved['__resolved_by__'] = label
        return resolved


    resolved2a = resolve_ambiguity_with_constituency(c2a, amb2a, "name_key_const")
    resolved2b = resolve_ambiguity_with_constituency(c2b, amb2b, "alias_key_const") if 'c2b' in locals() else pd.DataFrame(columns=["__rid__"])
    fill_from_top(resolved2a, "name_key_const")
    fill_from_top(resolved2b, "alias_key_const")
    amb2a = amb2a - set(resolved2a["__rid__"]); amb2b = amb2b - set(resolved2b["__rid__"])

    fill_from_top(top2a, "name_key", amb2a)
    fill_from_top(top2b, "alias_key", amb2b)
    fill_from_top(top2c, "lords_alias")
    amb_mask = result["__rid__"].isin(amb2a | amb2b | amb2c) & result["slug"].isna() if 'amb2c' in locals() else result["__rid__"].isin(amb2a | amb2b) & result["slug"].isna()
    result.loc[amb_mask, "match_level"] = result.loc[amb_mask, "match_level"].fillna("ambiguous")

    # Strict constituency-UNIQUE fallback
    def apply_constituency_unique_fallback(result, members, hansard):
        leftover = result["slug"].isna() & result["const_norm"].notna()
        if not leftover.any(): return result
        H = hansard[["slug","name","url","list_birth_year","list_death_year","constituencies","const_set","const_windows","name_key","hk_surname","hk_given"]].copy()
        picks = []
        for _, row in result.loc[leftover, ["__rid__","const_norm","mk_surname","m_start_year","m_end_year"]].iterrows():
            cn, sur, msy, mey = row["const_norm"], row["mk_surname"], row["m_start_year"], row["m_end_year"]
            cand = H[ H["const_set"].apply(lambda s: cn in s) ]
            if sur: cand = cand[cand["hk_surname"]==sur]
            if len(cand) != 1: continue
            wins = cand.iloc[0]["const_windows"].get(cn, [])
            if wins and (msy is not None or mey is not None) and window_overlap_score(msy, mey, wins) <= 0:
                continue
            chosen = cand.copy(); chosen["__rid__"]=row["__rid__"]; picks.append(chosen)
        if not picks: return result
        picks_df = pd.concat(picks, ignore_index=True).set_index("__rid__")
        aligned = picks_df.reindex(result["__rid__"]).reset_index()
        mask = result["slug"].isna() & result["__rid__"].isin(picks_df.index)
        for c in ["name","slug","url","list_birth_year","list_death_year","constituencies","const_set","const_windows"]:
            result.loc[mask, c] = result.loc[mask, c].combine_first(aligned[c])
        result.loc[mask, "match_level"] = result.loc[mask, "match_level"].combine_first(pd.Series(["constituency_fallback"]*len(result)))
        return result

    result = apply_constituency_unique_fallback(result, members, hansard)
    result = propagate_within_person(result)


    # ---------------- PASS: constituency+time fallback for *all* leftovers ----------------

    # one-time lookup (put this once above both mapping blocks)
    name_by_slug = (
        hansard_df.drop_duplicates("slug")
                .set_index("slug")["name"]
                .to_dict()
    )

    # members side (normalize constituency + stint years)
    def _const_from_post_label(lbl):
        if lbl is None or (isinstance(lbl, float) and pd.isna(lbl)):
            return None
        # take the bit after "for " if present
        m = re.search(r"\bfor\s+(.*)$", str(lbl), flags=re.I)
        raw = m.group(1) if m else str(lbl)
        return norm_const(raw)

    members_df["const_norm"]   = members_df["post_label"].apply(_const_from_post_label)
    members_df["const_key"]    = members_df["const_norm"].apply(const_join_key)
    members_df["m_start_year"] = pd.to_datetime(members_df["membership_start_date"], errors="coerce").dt.year
    members_df["m_end_year"]   = pd.to_datetime(members_df["membership_end_date"],   errors="coerce").dt.year

    # hansard side (derive const_set and const_windows from 'constituencies')
    # Build const_set/const_windows on Hansard using the same normalizer
    if ("const_set" not in hansard_df.columns) or ("const_windows" not in hansard_df.columns):
        cs, cw = zip(*hansard_df["constituencies"].apply(_parse_hh_const))
        hansard_df["const_set"]     = list(cs)
        hansard_df["const_windows"] = list(cw)

    # Re-compute keys with the SAME helpers, right before the fallback
    result["const_key"] = result["const_norm"].apply(const_join_key)

    hc_all = hansard_df[["slug", "name", "const_set", "const_windows"]].copy()
    hc_all = hc_all.explode("const_set").dropna(subset=["const_set"])
    hc_all = hc_all.rename(columns={"const_set": "h_const_norm"})
    hc_all["h_const_norm"] = hc_all["h_const_norm"].apply(norm_const)
    hc_all["const_key"]    = hc_all["h_const_norm"].apply(const_join_key)

    # Use constituency match + window overlap even if there was no name hit at all.
    leftover = result["slug"].isna() & result["const_key"].notna()
    if leftover.any():
        # candidates: leftover member row × Hansard row sharing the same token key
        cands = (
            result.loc[leftover, ["__rid__", "const_key", "m_start_year", "m_end_year"]]
                .merge(
                    hc_all[["slug", "name", "const_windows", "const_key", "h_const_norm"]],
                    on="const_key",
                    how="inner"
                )
                .dropna(subset=["slug"])
        )

        if not cands.empty:
            # ensure years are proper ints (nullable) before overlap
            cands["m_start_year"] = pd.to_numeric(cands["m_start_year"], errors="coerce").astype("Int64")
            cands["m_end_year"]   = pd.to_numeric(cands["m_end_year"],   errors="coerce").astype("Int64")

            def _wins(r):
                ms, me = r["m_start_year"], r["m_end_year"]
                if pd.isna(ms) or pd.isna(me):
                    return 0
                ms, me = int(ms), int(me)
                wins = r["const_windows"]
                cn   = r["h_const_norm"]
                win_list = wins.get(cn, []) if isinstance(wins, dict) else []
                score = 0
                for ws, we in win_list:
                    if ws is None or we is None:
                        continue
                    lo = max(ms, ws)
                    hi = min(me, we)
                    if hi >= lo:
                        score += (hi - lo + 1)
                return score

            cands["_wscore"] = cands.apply(_wins, axis=1)
            cands = cands[cands["_wscore"] > 0]  # require positive overlap

        if not cands.empty:
            # deterministic best-per-row; require unique slug per __rid__
            cands = cands.sort_values(["__rid__", "_wscore"], ascending=[True, False], kind="mergesort")
            uniq_counts = cands.groupby("__rid__")["slug"].nunique()
            take_ids = set(uniq_counts[uniq_counts == 1].index)
            best = cands[cands["__rid__"].isin(take_ids)].groupby("__rid__", as_index=False).head(1)

            if not best.empty:
                # align dtypes on __rid__
                result["__rid__"] = pd.to_numeric(result["__rid__"], errors="coerce").astype("Int64")
                best["__rid__"]   = pd.to_numeric(best["__rid__"],   errors="coerce").astype("Int64")

                m = result["__rid__"].isin(best["__rid__"]) & result["slug"].isna()

                # map slug by __rid__
                rid2slug = best.set_index("__rid__")["slug"].to_dict()
                result.loc[m, "slug"] = result.loc[m, "__rid__"].map(rid2slug)

                # map name by slug (not from best)
                rid2name = {rid: name_by_slug.get(slug) for rid, slug in rid2slug.items()}
                need_name = m & result["name"].isna()
                result.loc[need_name, "name"] = result.loc[need_name, "__rid__"].map(rid2name)

                # labels/ids
                result.loc[m, "match_level"] = "constituency_fallback"
                result.loc[m, "id_historichansard_url"] = result.loc[m, "slug"]

    result = propagate_within_person(result)

    # ---------------- PASS: constituency+time fallback for leftovers with fuzzy matching ----------------

    # Recompute helper keys right before fuzzy pass (keeps both sides in sync)
    result["const_key"] = result["const_norm"].apply(const_join_key)

    hc_all = hansard_df[["slug", "name", "const_set", "const_windows"]].copy()
    hc_all = hc_all.explode("const_set").dropna(subset=["const_set"])
    hc_all = hc_all.rename(columns={"const_set": "h_const_norm"})
    hc_all["h_const_norm"] = hc_all["h_const_norm"].apply(norm_const)
    hc_all["const_key"]    = hc_all["h_const_norm"].apply(const_join_key)

    # Helper: merge a column of dicts {const: [(y1,y2), ...]} into one dict
    def _merge_windows(dicts):
        out = {}
        for d in dicts:
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                out.setdefault(k, [])
                for win in v:
                    if win not in out[k]:
                        out[k].append(win)
        return out

    # Build hh without drop_duplicates on unhashable 'const_windows'
    hh = (
        hc_all.groupby(["slug", "name", "h_const_norm"], as_index=False)
            .agg(const_windows=("const_windows", _merge_windows))
    )
    hh["h_tokens"] = hh["h_const_norm"].apply(_const_tokens)
    hh["h_n"]      = hh["h_tokens"].apply(len)
    hh = hh[hh["h_n"] > 0]

    # Leftovers to try fuzzily
    left2 = result["slug"].isna() & result["const_norm"].notna()
    if left2.any() and not hh.empty:
        m = result.loc[left2, ["__rid__", "const_norm", "m_start_year", "m_end_year"]].copy()
        m["m_tokens"] = m["const_norm"].apply(_const_tokens)
        m["m_n"]      = m["m_tokens"].apply(len)
        m = m[m["m_n"] > 0]

        if not m.empty:
            # Generate candidate pairs by shared token (vectorized via explode+merge)
            m_exp = m.explode("m_tokens").rename(columns={"m_tokens": "tok"})
            h_exp = hh.explode("h_tokens").rename(columns={"h_tokens": "tok"})

            if not m_exp.empty and not h_exp.empty:
                pairs = (m_exp.merge(h_exp, on="tok", how="inner")
                            [["__rid__", "m_start_year", "m_end_year",
                                "slug", "name", "h_const_norm", "const_windows", "tok"]])

                if not pairs.empty:
                    # unique common tokens per (row, hansard_const)
                    pairs = pairs.drop_duplicates(["__rid__", "slug", "h_const_norm", "tok"])

                    # DO NOT groupby on dicts. Group by hashable cols only, count shared tokens…
                    common = (
                        pairs.groupby(["__rid__", "slug", "h_const_norm", "m_start_year", "m_end_year"])
                            .size()
                            .reset_index(name="size")
                    )
                    # …then merge the windows back from hh (which already has dicts)
                    common = common.merge(
                        hh[["slug", "h_const_norm", "const_windows"]],
                        on=["slug", "h_const_norm"], how="left"
                    )

                    # attach set sizes
                    common = (common
                            .merge(m[["__rid__", "m_n"]], on="__rid__", how="left")
                            .merge(hh[["slug", "h_const_norm", "h_n"]],
                                    on=["slug", "h_const_norm"], how="left"))

                    # “most words” criterion: overlap / min(|A|,|B|) ≥ 0.67
                    MIN_OVERLAP = 0.67
                    common["overlap_min"] = common["size"] / common[["m_n", "h_n"]].min(axis=1)
                    cand2 = common[common["overlap_min"] >= MIN_OVERLAP].copy()

                    if not cand2.empty:
                        # time overlap score
                        def _wins2(r):
                            ms, me = r["m_start_year"], r["m_end_year"]
                            if pd.isna(ms) or pd.isna(me):
                                return 0
                            ms, me = int(ms), int(me)
                            wins = r["const_windows"]; cn = r["h_const_norm"]
                            win_list = wins.get(cn, []) if isinstance(wins, dict) else []
                            score = 0
                            for ws, we in win_list:
                                if ws is None or we is None:
                                    continue
                                lo, hi = max(ms, ws), min(me, we)
                                if hi >= lo:
                                    score += (hi - lo + 1)
                            return score

                        cand2["_wscore"] = cand2.apply(_wins2, axis=1)
                        cand2 = cand2[cand2["_wscore"] > 0]

                        if not cand2.empty:
                            # choose one best per row (overlap first, then time)
                            cand2 = cand2.sort_values(["__rid__", "overlap_min", "_wscore"],
                                                    ascending=[True, False, False],
                                                    kind="mergesort")
                            uniq = cand2.groupby("__rid__")["slug"].nunique()
                            take = set(uniq[uniq == 1].index)
                            best2 = cand2[cand2["__rid__"].isin(take)].groupby("__rid__", as_index=False).head(1)

                            if not best2.empty:
                                # map back (ensure rid dtype aligns)
                                result["__rid__"] = pd.to_numeric(result["__rid__"], errors="coerce").astype("Int64")
                                best2["__rid__"]  = pd.to_numeric(best2["__rid__"],  errors="coerce").astype("Int64")

                                msk = result["__rid__"].isin(best2["__rid__"]) & result["slug"].isna()

                                # map slug by __rid__
                                rid2slug2 = best2.set_index("__rid__")["slug"].to_dict()
                                result.loc[msk, "slug"] = result.loc[msk, "__rid__"].map(rid2slug2)

                                # map name by slug (not from best2)
                                rid2name2 = {rid: name_by_slug.get(slug) for rid, slug in rid2slug2.items()}
                                need_name = msk & result["name"].isna()
                                result.loc[need_name, "name"] = result.loc[need_name, "__rid__"].map(rid2name2)

                                # labels/ids
                                result.loc[msk, "match_level"] = "constituency_fuzzy"
                                result.loc[msk, "id_historichansard_url"] = result.loc[msk, "slug"]
    
    result = propagate_within_person(result)


    # ---------------- PASS: name fuzzy fallback (person_name + aliases_norm  vs  name + titles_in_lords) ----------------
    
    # ---- helpers (lightweight, built-in only) ----
    TITLE_STOP = {"lord","lady","baron","baroness","earl","viscount","duke","marquess",
                "marchioness","countess","sir","dame","of","the","st","saint"}

    def _ensure_list(v):
        if v is None or (isinstance(v, float) and pd.isna(v)): return []
        if isinstance(v, (list, tuple, set)): return [str(x) for x in v]
        s = str(v).strip()
        if not s: return []
        # try JSON list
        if s.startswith("[") and s.endswith("]"):
            try:
                import json as _json
                arr = _json.loads(s)
                return [str(x) for x in arr]
            except Exception:
                pass
        # fallback: comma-separated
        return [p.strip() for p in s.split(",") if p.strip()]

    def _swap_commas(s: str) -> str:
        # "Surname, Given (Mr)" -> "Given Surname"
        if "," in s:
            parts = [p.strip() for p in s.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                s = parts[1] + " " + parts[0]
        return s

    def _name_tokens(s: str) -> set:
        if s is None or (isinstance(s, float) and pd.isna(s)): return set()
        s = _swap_commas(str(s))
        s = s.lower()
        s = re.sub(r"\([^)]*\)", " ", s)            # drop parens like (Mr)
        # remove honorifics as whole words (w/ optional trailing dot)
        for h in HSET:
            if not h: continue
            s = re.sub(rf"\b{re.escape(h)}\.?\b", " ", s)
        s = re.sub(r"[^a-z'\- ]", " ", s)
        toks = [t.strip("'") for t in re.split(r"[\s\-]+", s) if t and t not in {"", "'"}]
        return set(toks)

    def _tokens_from_list(str_list) -> set:
        toks = set()
        for x in _ensure_list(str_list):
            toks |= _name_tokens(x)
        return toks

    def _flatten_windows(win_dict) -> list[tuple]:
        out = []
        if isinstance(win_dict, dict):
            for L in win_dict.values():
                if isinstance(L, list):
                    for w in L:
                        if isinstance(w, (list, tuple)) and len(w) == 2:
                            out.append(tuple(w))
        return out

    def _overlap_score(ms, me, windows) -> int:
        # inclusive overlap on years; returns number of overlapped years
        try:
            ms = int(ms) if pd.notna(ms) else None
            me = int(me) if pd.notna(me) else None
        except Exception:
            return 0
        if ms is None or me is None:
            return 0
        score = 0
        for ws, we in windows:
            if ws is None or we is None:
                continue
            lo, hi = max(ms, ws), min(me, we)
            if hi >= lo:
                score += (hi - lo + 1)
        return score

    # A titles parser should already exist; use it if present, else a safe fallback
    def _titles_list(v):
        try:
            return parse_titles_in_lords(v)  # your cleaner that strips months/dates/years
        except Exception:
            try:
                return parse_titles_in_lords(v)    # older version, if available
            except Exception:
                return _ensure_list(v)

    # ---- Build Hansard name-token index (slug → tokens + windows) ----
    hh_name = hansard_df[["slug", "name", "titles_in_lords", "const_windows"]].copy()

    # tokens per row (keep honorifics)
    hh_name["name_tok"]   = hh_name["name"].apply(_name_tokens_keep_hon)
    hh_name["title_tok"]  = hh_name["titles_in_lords"].apply(_tokens_from_list_keep_hon)
    hh_name["all_tok"]    = hh_name.apply(lambda r: sorted(r["name_tok"] | r["title_tok"]), axis=1)

    # split into core/honorific
    def _split_row_tokens(r):
        core, hon = _split_core_hon(set(r["all_tok"]))
        return pd.Series({"h_core": sorted(core), "h_hon": sorted(hon)})

    hh_name[["h_core","h_hon"]] = hh_name.apply(_split_row_tokens, axis=1)

    # merge per slug
    def _merge_list_sets(col):
        def _merge(series):
            out = set()
            for lst in series:
                out |= set(lst)
            return sorted(out)
        return _merge

    def _merge_windows_col(series):
        out = {}
        for d in series:
            if not isinstance(d, dict): continue
            for k, v in d.items():
                out.setdefault(k, [])
                for w in v:
                    if w not in out[k]:
                        out[k].append(w)
        return out

    hh_by_slug = (
        hh_name.groupby("slug", as_index=False)
            .agg(
                h_core=("h_core", _merge_list_sets("h_core")),
                h_hon =("h_hon",  _merge_list_sets("h_hon")),
                const_windows=("const_windows", _merge_windows_col),
            )
    )
    # flattened windows for time check
    def _flatten_windows(win_dict):
        out = []
        if isinstance(win_dict, dict):
            for L in win_dict.values():
                if isinstance(L, list):
                    for w in L:
                        if isinstance(w, (list, tuple)) and len(w) == 2:
                            out.append(tuple(w))
        return out
    hh_by_slug["flat_windows"] = hh_by_slug["const_windows"].apply(_flatten_windows)

    # quick lookups
    slug2_core = dict(zip(hh_by_slug["slug"], map(set, hh_by_slug["h_core"])))
    slug2_hon  = dict(zip(hh_by_slug["slug"], map(set, hh_by_slug["h_hon"])))
    slug2_win  = dict(zip(hh_by_slug["slug"], hh_by_slug["flat_windows"]))
    name_by_slug = hansard_df.drop_duplicates("slug").set_index("slug")["name"].to_dict()

    # ---- Members side tokens (keep honorifics, but require a core token) ----
    left_nm = result["slug"].isna()
    if left_nm.any():
        tmp = result.loc[left_nm, ["__rid__", "person_name", "aliases_norm", "m_start_year", "m_end_year"]].copy()

        tmp["pn_tok"]  = tmp["person_name"].apply(_name_tokens_keep_hon)
        tmp["al_tok"]  = tmp["aliases_norm"].apply(_tokens_from_list_keep_hon)
        tmp["all_tok"] = tmp.apply(lambda r: sorted(r["pn_tok"] | r["al_tok"]), axis=1)

        def _split_m(r):
            core, hon = _split_core_hon(set(r["all_tok"]))
            return pd.Series({"m_core": sorted(core), "m_hon": sorted(hon)})

        tmp[["m_core","m_hon"]] = tmp.apply(_split_m, axis=1)
        tmp["m_core_n"] = tmp["m_core"].apply(len)
        tmp = tmp[tmp["m_core_n"] > 0]   # require at least one non-honorific token

        if not tmp.empty and len(slug2_core) > 0:
            # Candidate generation by shared token (use ALL tokens to seed, keeps recall high)
            m_exp = tmp.explode("all_tok").rename(columns={"all_tok": "tok"}).dropna(subset=["tok"])
            # Build token → slug index from Hansard ALL tokens (core ∪ honorific)
            h_tok_rows = []
            for s in slug2_core:
                all_tok = slug2_core.get(s, set()) | slug2_hon.get(s, set())
                for t in all_tok:
                    h_tok_rows.append((s, t))
            h_tok_df = pd.DataFrame(h_tok_rows, columns=["slug","tok"])

            if not m_exp.empty and not h_tok_df.empty:
                pairs = (m_exp.merge(h_tok_df, on="tok", how="inner")
                            [["__rid__", "m_start_year", "m_end_year", "tok", "slug"]])

                if not pairs.empty:
                    # Count shared tokens (all) just to shortlist
                    pairs = pairs.drop_duplicates(["__rid__", "slug", "tok"])
                    common = (pairs.groupby(["__rid__", "slug", "m_start_year", "m_end_year"])
                                .size().reset_index(name="common"))

                    # Build maps for core/hon sets for precise scoring
                    rid2_core = dict(zip(tmp["__rid__"], map(set, tmp["m_core"])))
                    rid2_hon  = dict(zip(tmp["__rid__"], map(set, tmp["m_hon"])))
                    rid2_core_n = {rid: len(rid2_core[rid]) for rid in rid2_core}
                    slug2_core_n = {s: len(slug2_core[s]) for s in slug2_core}

                    # Compute CORE overlap and require ≥ threshold; also allow honorific bonus
                    MIN_NAME_OVERLAP_CORE = 0.6
                    HON_BONUS = 0.05  # small boost if peerage honorific overlaps

                    def _score_row(r):
                        rid, slug = r["__rid__"], r["slug"]
                        mcore = rid2_core.get(rid, set())
                        hcore = slug2_core.get(slug, set())
                        mhon  = rid2_hon.get(rid, set())
                        hhon  = slug2_hon.get(slug, set())

                        core_common = len(mcore & hcore)
                        denom = min(len(mcore), len(hcore)) or 1
                        core_ratio = core_common / denom

                        hon_overlap = len((mhon & hhon) & HON_PEERAGE) > 0  # bonus only for peerage ranks
                        score = core_ratio + (HON_BONUS if hon_overlap else 0.0)
                        return pd.Series({"core_ratio": core_ratio, "hon_overlap": hon_overlap, "score": score})

                    scored = common.join(common.apply(_score_row, axis=1))

                    # Filter by core overlap (after bonus we still require base core coverage)
                    candn = scored[scored["core_ratio"] >= MIN_NAME_OVERLAP_CORE].copy()

                    if not candn.empty:
                        # time overlap using flattened windows across constituencies
                        def _wscore_row(r):
                            wins = slug2_win.get(r["slug"], [])
                            # inclusive overlap on years
                            ms, me = r["m_start_year"], r["m_end_year"]
                            ms = int(ms) if pd.notna(ms) else None
                            me = int(me) if pd.notna(me) else None
                            if ms is None or me is None: return 0
                            tot = 0
                            for ws, we in wins:
                                if ws is None or we is None: continue
                                lo, hi = max(ms, ws), min(me, we)
                                if hi >= lo: tot += (hi - lo + 1)
                            return tot

                        candn["_wscore"] = candn.apply(_wscore_row, axis=1)
                        candn = candn[candn["_wscore"] > 0]

                    if not candn.empty:
                        # choose best: core_ratio, honorific bonus, then time
                        candn = candn.sort_values(
                            ["__rid__", "score", "_wscore"],
                            ascending=[True, False, False],
                            kind="mergesort"
                        )
                        uniq = candn.groupby("__rid__")["slug"].nunique()
                        take = set(uniq[uniq == 1].index)
                        bestn = candn[candn["__rid__"].isin(take)].groupby("__rid__", as_index=False).head(1)

                        if not bestn.empty:
                            # map back (ensure rid dtype aligns)
                            result["__rid__"] = pd.to_numeric(result["__rid__"], errors="coerce").astype("Int64")
                            bestn["__rid__"]  = pd.to_numeric(bestn["__rid__"], errors="coerce").astype("Int64")

                            msk = result["__rid__"].isin(bestn["__rid__"]) & result["slug"].isna()
                            rid2slug_n = bestn.set_index("__rid__")["slug"].to_dict()

                            result.loc[msk, "slug"] = result.loc[msk, "__rid__"].map(rid2slug_n)

                            # fill display name from Hansard by slug only if empty
                            rid2name_n = {rid: name_by_slug.get(slug) for rid, slug in rid2slug_n.items()}
                            need_name  = msk & result["name"].isna()
                            result.loc[need_name, "name"] = result.loc[need_name, "__rid__"].map(rid2name_n)

                            result.loc[msk, "match_level"] = "name_fuzzy"
                            result.loc[msk, "id_historichansard_url"] = result.loc[msk, "slug"]

    result = propagate_within_person(result)


    # ---- Final normalization of slug + match_level --------------------

    # 1) If we have a Hansard id but slug is empty, use it as the slug
    mask_slug_from_id = result["slug"].isna() & result["id_historichansard_url"].notna()
    result.loc[mask_slug_from_id, "slug"] = result.loc[mask_slug_from_id, "id_historichansard_url"]

    # 2) If slug is now present but match_level is NA/ambiguous, mark as 'slug'
    ml = result["match_level"].astype("string")
    mask_fix_na  = result["slug"].notna() & ml.isna()
    mask_fix_amb = result["slug"].notna() & ml.eq("ambiguous")

    result.loc[mask_fix_na | mask_fix_amb, "match_level"] = "constituency_fallback"


    # ---------------- Append Hansard-only rows ----------------
    # 1) What we already matched by slug
    seen_slugs = set(result["slug"].astype("string").str.strip().dropna().unique().tolist())


    # 2) Valid identity in Hansard = has slug OR has name (non-empty)
    slug_s = hansard_df["slug"].astype("string").str.strip()

    # 3) Not already seen by slug; and must have at least one identity field
    extra = hansard_df[slug_s.notna() & slug_s.ne("") & ~slug_s.isin(list(seen_slugs))].copy()

    # 4) Build rows shaped like members_df, but also carry Hansard fields
    member_cols = list(members_df.columns)
    member_block = {c: pd.Series([pd.NA] * len(extra)) for c in member_cols}
    member_block.update({
        "name": extra["name"],
        "slug": extra["slug"],
        "url": extra["url"],
        "titles_in_lords": extra.get("titles_in_lords", pd.Series([pd.NA]*len(extra))),
        "list_birth_year": extra.get("list_birth_year", pd.Series([pd.NA]*len(extra))),
        "list_death_year": extra.get("list_death_year", pd.Series([pd.NA]*len(extra))),
        "constituencies": extra.get("constituencies", pd.Series([pd.NA]*len(extra))),
        "id_historichansard_url": extra["slug"],     # set now from slug
        "match_level": "hansard_only",
        "__src": "hansard_only",                     # marker to reassert after concat
    })
    extra_rows = pd.DataFrame(member_block)

    # If you derive aliases from titles, do it here (dates stripped by your clean parser)
    extra_rows["aliases_norm"] = extra.get("titles_in_lords", pd.Series([None]*len(extra))).apply(
        lambda v: parse_titles_in_lords(v) or []
    )

    # 5) Align columns and concat
    for c in [col for col in result.columns if col not in extra_rows.columns]:
        extra_rows[c] = pd.NA
    extra_rows = extra_rows[result.columns]
    result_full = pd.concat([result, extra_rows], ignore_index=True)

    # 6) Reassert label in case something downstream sets NA, then drop marker
    if "__src" in result_full.columns:
        mask = result_full["__src"].eq("hansard_only") & result_full["match_level"].isna()
        result_full.loc[mask, "match_level"] = "hansard_only"
        result_full.drop(columns="__src", inplace=True)

    # 7) Hard guard: ensure all hansard_only rows have a slug (drop if not)
    hs = result_full["match_level"].eq("hansard_only")
    bad = hs & result_full["id_historichansard_url"].astype("string").str.strip().isna()
    result_full = result_full.loc[~bad].copy()

    # Final cleanup
    result_full["id_historichansard_url"] = result_full["id_historichansard_url"].fillna(result_full["slug"])
    # Clean candidate from Hansard 'name'
    name_fill = result_full["name"].apply(normalize_person_name_from_hansard)

    # Treat "", "nan", "none" as missing
    pn_str = result_full["person_name"].astype("string")
    missing = pn_str.isna() | pn_str.str.strip().str.lower().isin(["", "nan", "none"])

    # Fill only missing rows
    result_full.loc[missing, "person_name"] = name_fill

    # Ensure any leftover placeholders are actual NA
    pn_str = result_full["person_name"].astype("string")
    result_full.loc[pn_str.str.strip().str.lower().isin(["", "nan", "none"]), "person_name"] = pd.NA

    result_full = result_full.rename(columns={"list_birth_year":"birth_year","list_death_year":"death_year"})
    bd_year = pd.to_datetime(result_full["birth_date"], errors="coerce").dt.year
    result_full["birth_year"] = result_full["birth_year"].where(result_full["birth_year"].notna(), bd_year)

    result_full["honorific_prefix"] = result_full["name"].apply(extract_honorific_from_name)

    # ---------- aliases_norm: PRESERVE originals + APPEND titles_in_lords ----------
    # def _titles_list(x):
    #     # parse_titles_in_lords already strips dates + lowercases; return None if empty
    #     lst = parse_titles_in_lords(x)
    #     return lst if lst else None

    titles_series = (
        result_full["titles_in_lords"].apply(lambda x: parse_titles_in_lords(x) or None)
        if "titles_in_lords" in result_full.columns else pd.Series([None]*len(result_full))
    )

    # keep your append-only combiner (non-destructive)
    def _append_titles(existing, extra_titles):
        if not extra_titles:
            return existing
        if isinstance(existing, list):
            seen = {str(s).casefold().strip() for s in existing if isinstance(s, str)}
            to_add = [t for t in extra_titles if t.casefold() not in seen]
            return existing + to_add if to_add else existing
        return list(extra_titles)

    result_full["aliases_norm"] = [
        _append_titles(ex, ti) for ex, ti in zip(result_full["aliases_norm"], titles_series)
    ]

    result_full["match_level"] = result_full["match_level"].replace({"None": pd.NA}).astype("string")

    # Build a flat list of column names (strings), not Index objects
    members_cols = members_df.columns.tolist()
    hansard_cols = hansard_df.columns.tolist()

    extra_cols = ['aliases_norm', 'birth_year', 'death_year', 'constituencies', 'match_level', 
                'honorific_prefix']

    # Ordered, de-duplicated keep list limited to columns that actually exist
    cols_to_keep = []
    for c in members_cols + hansard_cols + extra_cols:
        if c in result_full.columns and c not in cols_to_keep:
            cols_to_keep.append(c)

    # Reorder/select
    result_full = result_full.loc[:, cols_to_keep]

    to_drop = [c for c in ['name', 'url', 'slug', 'const_norm', 'const_key', 'm_start_year', 
                           'm_end_year', 'const_set', 'const_windows',] if c in result_full.columns]
    result_final = result_full.drop(columns=to_drop)

    return result_final


# ============================================================
# main(): I/O and saving
# ============================================================

def main():
    print("🔹 Loading Parquet files…")
    members_df = pd.read_parquet(MEMBERS_PARQUET)
    hansard_df = pd.read_parquet(HANSARD_PARQUET)

    print(f"Parlparse rows: {len(members_df):,}, cols: {members_df.shape[1]}")
    print(f"Historic Hansard rows: {len(hansard_df):,}, cols: {hansard_df.shape[1]}")

    result_final = merge_members_and_hansard(members_df, hansard_df, honorific_map)

    for col in ("birth_year", "death_year"):
        if col in result_final.columns:
            result_final[col] = pd.to_numeric(result_final[col], errors="coerce").astype("Int64")


    print("🔹 Saving outputs…")
    result_final.to_parquet(OUT_PATH, index=False)
    print(f"✅ Saved: {OUT_PATH}")
    # Quick stats
    ml = result_final["match_level"] if "match_level" in result_final.columns else pd.Series(dtype=object)
    print({
        "rows_final": len(result_final),
        "unique_person_ids": result_final["person_id"].nunique() if "person_id" in result_final.columns else None,
        "unique_slugs": result_final["id_historichansard_url"].nunique() if "id_historichansard_url" in result_final.columns else None,
        "match_levels": ml.value_counts(dropna=False).to_dict() if not ml.empty else {}
    })

if __name__ == "__main__":
    main()
