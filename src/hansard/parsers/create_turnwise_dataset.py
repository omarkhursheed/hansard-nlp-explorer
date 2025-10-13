import pandas as pd
import numpy as np
import glob
import os
import re
import json
from pathlib import Path
import gender_guesser.detector as genderer
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

INPUT_PATH = Path("src/hansard/data/processed_fixed/cleaned_data/")
OUT_PATH = Path("src/hansard/data/processed_fixed/cleaned_data/turnwise_debates.parquet")
HONORIFICS_JSON = Path("src/hansard/data/word_lists/gendered_honorifics.json")
MILITARY_JSON = Path("src/hansard/data/word_lists/military_honorifics.json")
FILE_PATTERN = "debates_*.parquet"

# --- load honorific maps ---
with open(HONORIFICS_JSON, "r") as f:
    honorific_map = json.load(f)

honorific_male = {h.lower().rstrip(".") for h in honorific_map.get("Male", [])}
honorific_female = {h.lower().rstrip(".") for h in honorific_map.get("Female", [])}
honorific_unk = {h.lower().rstrip(".") for h in honorific_map.get("Unknown", [])}

with open(MILITARY_JSON, "r") as f:
    military_map = json.load(f)

army_hon = {h.lower().rstrip(".") for h in military_map.get("Army", [])}
navy_hon = {h.lower().rstrip(".") for h in military_map.get("Navy", [])}
airforce_hon = {h.lower().rstrip(".") for h in military_map.get("AirForce", [])}

detector = genderer.Detector(case_sensitive=False)

# ---------------- Utility Functions ----------------

ROLE_WORDS = {
    "the", "prime","minister","speaker","secretary","treasurer","president","chancellor"}

def extract_first_name_vec(speakers: pd.Series) -> pd.Series:
    """
    Extracts a probable first name (not surname) from Hansard speaker strings.
    Handles:
    - 'MR. JOHN DOUGLAS'      → 'John'
    - 'SIR J. BRYDGES'        → None  (initial + surname)
    - 'MR. W. PEEL'           → None  (initial + surname)
    - 'MR. LOCKHART'          → None  (surname only)
    - 'MR. GREGORY BARKER'    → 'Gregory'
    - 'MR. CHARLES BATHURST'  → 'Charles'
    - 'SIR M. W. RIDLEY'      → None  (multi-initial pattern)
    """

    skip = honorific_male | honorific_female | honorific_unk | army_hon | navy_hon | airforce_hon | ROLE_WORDS
    results = []

    for name in speakers.fillna("").astype(str):
        # Normalize
        name_clean = re.sub(r"[^A-Za-z.\s]", " ", name).lower()
        tokens = [t.strip(".") for t in name_clean.split() if t.strip(".")]

        # Remove leading titles and skip words
        tokens = [t for t in tokens if t not in skip]

        first_name = None

        if len(tokens) == 0:
            results.append(None)
            continue

        # Pattern 1: Initial + surname  -> skip (no usable first name)
        if len(tokens) == 2 and len(tokens[0]) == 1:
            results.append(None)
            continue

        # Pattern 2: Two initials + surname -> skip
        if len(tokens) >= 3 and all(len(tok) == 1 for tok in tokens[:-1]):
            results.append(None)
            continue

        # Pattern 3: Single surname (no first name)
        if len(tokens) == 1:
            results.append(None)
            continue

        # Pattern 4: Proper first name followed by something else
        for tok in tokens:
            if len(tok) > 1 and tok.isalpha():
                first_name = tok.capitalize()
                break

        results.append(first_name)

    return pd.Series(results, index=speakers.index)

def contains_any(norm_speakers: pd.Series, honorifics: set):
    """Vectorized match: does speaker contain any honorifics (tokens or multi-word)."""
    tokens = norm_speakers.str.split()
    token_match = tokens.apply(lambda toks: any(tok in honorifics for tok in toks) if toks else False)
    phrase_match = norm_speakers.apply(lambda s: any(" " in h and h in s for h in honorifics))
    return token_match | phrase_match

def build_speaker_gender_map(speaker_details):
    """Build {original_name_upper: gender} mapping from speaker_details list/array."""
    mapping = {}
    if isinstance(speaker_details, (list, pd.Series, np.ndarray)):
        for d in speaker_details:
            if not isinstance(d, dict):
                continue
            orig = str(d.get("original_name", "")).strip().upper()
            g = d.get("gender", None)
            if orig and g:
                mapping[orig] = "M" if g == "M" else "F"
    return mapping

def normalize_speakers(df_turnwise):
    def expand_speakers(group):
        mapping = {}
        speakers = []
        for spk in group["speaker"]:
            spk_norm = re.sub(r"\s+", " ", spk.strip())
            if "(" in spk_norm and ")" in spk_norm:
                prefix = spk_norm.split("(", 1)[0].strip()
                mapping[prefix] = spk_norm
                speakers.append(spk_norm)
            else:
                full = mapping.get(spk_norm)
                speakers.append(full if full else spk_norm)
        group = group.copy()
        group["speaker"] = speakers
        return group

    out = df_turnwise.groupby("debate_id", group_keys=False).apply(expand_speakers)
    return out.reset_index(drop=True)

def flatten_nested_list(x):
    """Recursively flattens nested lists or arrays."""
    if not isinstance(x, (list, np.ndarray, pd.Series)):
        return []
    flat = []
    for item in x:
        if isinstance(item, (list, np.ndarray, pd.Series)):
            flat.extend(flatten_nested_list(item))
        else:
            flat.append(item)
    return flat


# ---------------- Gender Assignment ----------------

def assign_gender(turnwise, speaker_gender_map):
    # Precompute normalized speaker strings (lowercased, stripped)
    norm_speakers = turnwise["speaker"].str.lower().str.replace(r"[^a-z ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

    # Step 1. Speaker details
    turnwise["gender"] = turnwise["speaker"].map(speaker_gender_map).fillna("UNK")
    turnwise["gender_source"] = np.where(turnwise["gender"] != "UNK", "speaker_details", "UNK")

    # Step 2. Chamber cutoffs
    chamber_norm = turnwise["chamber"].str.lower().str.strip()

    # Commons cutoff
    mask = turnwise["gender"] == "UNK"
    commons_mask = mask & chamber_norm.str.contains("commons") & (pd.to_numeric(turnwise["year"], errors="coerce") < 1918)
    turnwise.loc[commons_mask, ["gender","gender_source"]] = ["M","chamber_cutoff"]

    # Lords cutoff
    mask = turnwise["gender"] == "UNK"
    lords_mask = mask & chamber_norm.str.contains("lords") & (pd.to_numeric(turnwise["year"], errors="coerce") < 1958)
    turnwise.loc[lords_mask, ["gender","gender_source"]] = ["M","chamber_cutoff"]


    # Step 3. Military + Doctor cutoffs
    mask = turnwise["gender"] == "UNK"
    cond_army = mask & (turnwise["year"] < 1949) & contains_any(norm_speakers, army_hon.union(airforce_hon))
    cond_navy = mask & (turnwise["year"] < 1993) & contains_any(norm_speakers, navy_hon)
    cond_doc = mask & (turnwise["year"] < 1965) & contains_any(norm_speakers, {"dr", "doctor"})
    turnwise.loc[cond_army | cond_navy | cond_doc, ["gender","gender_source"]] = ["M","military_doctor"]

    # Step 4. Male honorifics
    mask = turnwise["gender"] == "UNK"
    cond_male = mask & contains_any(norm_speakers, honorific_male)
    turnwise.loc[cond_male, ["gender","gender_source"]] = ["M","honorific_male"]

    # Step 5. Female honorifics
    mask = turnwise["gender"] == "UNK"
    cond_female = mask & contains_any(norm_speakers, honorific_female)
    turnwise.loc[cond_female, ["gender","gender_source"]] = ["F","honorific_female"]

    # Step 6. Gender guesser fallback
    mask = turnwise["gender"] == "UNK"
    if mask.any():
        fnames = extract_first_name_vec(turnwise.loc[mask, "speaker"])
        guesses = fnames.apply(lambda x: detector.get_gender(x) if x else None)

        male_mask = guesses.isin(["male", "mostly_male"])
        female_mask = guesses.isin(["female", "mostly_female"])

        # direct assignment, no reindex
        turnwise.loc[mask & male_mask, ["gender","gender_source"]] = ["M","guesser_male"]
        turnwise.loc[mask & female_mask, ["gender","gender_source"]] = ["F","guesser_female"]


    # # Step 7. Default male
    # mask = turnwise["gender"] == "UNK"
    # turnwise.loc[mask, ["gender","gender_source"]] = ["M","default_male"]

    return turnwise

# ---------------- File Processing ----------------

def process_file(f):
    try:
        df = pd.read_parquet(f)
    except Exception as e:
        print(f"⚠️  Failed to read {f}: {e}")
        return pd.DataFrame()

    dfs_local = []

    for debate_id, group in df.groupby("debate_id"):
        try:
            # ---- speaker details ----
            speaker_details = group["speaker_details"].iloc[0] if "speaker_details" in group.columns else []
            speaker_gender_map = build_speaker_gender_map(speaker_details)

            # ---- handle speech segments safely ----
            if "speech_segments" in group.columns and group["speech_segments"].dropna().shape[0] > 0:
                turnwise = group.explode("speech_segments", ignore_index=True)
                turnwise = turnwise[turnwise["speech_segments"].notna()]
                segs = turnwise["speech_segments"].apply(pd.Series)
                keep_cols = [c for c in ["speaker", "text", "position"] if c in segs.columns]
                turnwise = pd.concat(
                    [turnwise.drop(columns=["speech_segments"]), segs[keep_cols].reset_index(drop=True)],
                    axis=1,
                )
            else:
                # No segments → create placeholder one-row debate
                turnwise = group.copy()
                turnwise["speaker"] = np.nan
                turnwise["text"] = group.get("debate_text", pd.Series([""] * len(group)))
                turnwise["position"] = np.nan

            # ---- normalize & tag ----
            turnwise["speaker"] = turnwise["speaker"].astype(str).str.strip().str.upper()
            turnwise = normalize_speakers(turnwise)
            turnwise = assign_gender(turnwise, speaker_gender_map)

            # ---- text stats ----
            turnwise["word_count"] = turnwise["text"].astype(str).str.split().str.len()
            turnwise["char_count"] = turnwise["text"].astype(str).str.len()

            # ---- column selection ----
            base_cols = [
                "debate_id", "year", "decade", "month", "reference_date", "chamber",
                "title", "topic", "hansard_reference", "reference_volume",
                "reference_columns", "file_path", "speaker",
                "gender", "gender_source", "text", "word_count", "char_count",
            ]
            if "position" in turnwise.columns:
                base_cols.insert(base_cols.index("word_count"), "position")
            turnwise = turnwise.reindex(columns=base_cols, fill_value=np.nan)

            dfs_local.append(turnwise)

        except Exception as e:
            print(f"⚠️  Debate {debate_id} in {os.path.basename(f)} failed: {e}")
            continue

    if not dfs_local:
        print(f"⚠️  No valid debates found in {f}")
        return pd.DataFrame()

    return pd.concat(dfs_local, ignore_index=True)


# ---------------- Main Processing ----------------

if __name__ == "__main__":
    files = glob.glob(os.path.join(INPUT_PATH, FILE_PATTERN))
    print(f"Found {len(files)} files.")

    dfs = []
    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 1)) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for i, future in enumerate(as_completed(futures), 1):
            f = futures[future]
            try:
                result = future.result()
                dfs.append(result)
                print(f"[{i}/{len(files)}] Done {f}")
            except Exception as e:
                print(f"Error processing {f}: {e}")

    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_parquet(OUT_PATH, index=False)

    print(f"Processed {len(files)} files into {OUT_PATH}")
