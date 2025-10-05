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

def extract_first_name_vec(speakers: pd.Series):
    """Vectorized first-name extraction, only return proper names (skip titles/roles)."""
    norm = speakers.str.lower().str.replace(r"[^a-z ]", " ", regex=True)
    tokens = norm.str.split()

    results = []
    skip = honorific_male.union(honorific_female).union(honorific_unk).union(ROLE_WORDS)
    for toklist in tokens:
        fname = None
        if toklist:
            for tok in toklist:
                if tok not in skip and len(tok) > 1:  # skip initials and roles
                    # Only accept if it's alphabetic and not a known role
                    if tok.isalpha():
                        fname = tok.capitalize()
                        break
        results.append(fname)
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


    # Step 7. Default male
    mask = turnwise["gender"] == "UNK"
    turnwise.loc[mask, ["gender","gender_source"]] = ["M","default_male"]

    return turnwise

# ---------------- File Processing ----------------

def process_file(f):
    df = pd.read_parquet(f)

    # Expand speech_segments into turnwise rows
    turnwise = df.explode("speech_segments").dropna(subset=["speech_segments"])

    # Extract dict fields
    segs = turnwise["speech_segments"].apply(pd.Series)
    keep_cols = ["speaker","text"]
    if "position" in segs.columns:
        keep_cols.append("position")
    turnwise = pd.concat([turnwise.drop(columns=["speech_segments"]), segs[keep_cols]], axis=1)

    # Normalize speakers and expand shortened forms
    turnwise["speaker"] = turnwise["speaker"].astype(str).str.strip().str.upper()
    turnwise = normalize_speakers(turnwise)

    # Build speaker lookup
    speaker_gender_map = build_speaker_gender_map(df.iloc[0].get("speaker_details", []))

    # Apply gender assignment
    turnwise = assign_gender(turnwise, speaker_gender_map)

    # Word/char counts
    turnwise["word_count"] = turnwise["text"].astype(str).str.split().str.len()
    turnwise["char_count"] = turnwise["text"].astype(str).str.len()

    # Select columns
    base_cols = [
        'debate_id','year','decade','month','reference_date','chamber','title','topic',
        'hansard_reference','reference_volume','reference_columns','file_path','speaker',
        'gender','gender_source','text','word_count','char_count'
    ]
    if "position" in turnwise.columns:
        base_cols.insert(base_cols.index("word_count"), "position")
    turnwise = turnwise[base_cols]

    if "position" in turnwise.columns:
        turnwise = turnwise.sort_values(["debate_id","position"]).reset_index(drop=True)
    else:
        turnwise = turnwise.sort_values(["debate_id"]).reset_index(drop=True)

    return turnwise


if __name__ == "__main__":
    files = glob.glob(os.path.join(INPUT_PATH, FILE_PATTERN))
    print(f"Found {len(files)} files.")

    dfs = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
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
