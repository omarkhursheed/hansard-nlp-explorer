import pandas as pd
import numpy as np
import glob
import os
import re
import json
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Paths ---
INPUT_PATH = Path("src/hansard/data/processed_fixed/cleaned_data/")
OUT_PATH = Path("src/hansard/data/processed_fixed/cleaned_data/turnwise_debates_matched_speakers.parquet")

# ---------------- Utility Functions ----------------

def build_speaker_gender_map(speaker_details):
    """Build {original_name_upper: gender} mapping from speaker_details list/array."""
    mapping = {}
    if isinstance(speaker_details, (list, pd.Series, np.ndarray)):
        for d in speaker_details:
            if not isinstance(d, dict):
                continue
            orig = str(d.get("original_name", "")).strip().upper()
            g = d.get("gender", None)
            if orig:
                if g == "M":
                    mapping[orig] = "M"
                elif g == "F":
                    mapping[orig] = "F"
    return mapping

def normalize_speakers(df_turnwise):
    """Within each debate, expand shortened speakers using first full mention."""
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

def filter_matched_speakers(turnwise, speaker_details):
    """Keep only rows whose speaker matches speaker_details original_name."""
    valid_speakers = {
        str(d.get("original_name", "")).strip().upper()
        for d in (speaker_details if isinstance(speaker_details, (list, np.ndarray, pd.Series)) else [])
        if isinstance(d, dict) and d.get("original_name")
    }
    return turnwise[turnwise["speaker"].isin(valid_speakers)].reset_index(drop=True)

# ---------------- File Processing ----------------

def process_file(f):
    try:
        df = pd.read_parquet(f)
        dfs_local = []

        # Each file can contain multiple debates
        for debate_id, group in df.groupby("debate_id"):
            speaker_details = group["speaker_details"].iloc[0]

            # --- Expand speech_segments safely ---
            speech_segments = group.explode("speech_segments").dropna(subset=["speech_segments"])
            col = speech_segments["speech_segments"]

            # Skip debates with no valid segments
            if col.empty:
                continue
            first_valid = col.dropna().iloc[0] if not col.dropna().empty else None
            if first_valid is None:
                continue

            # Handle single dict or array of dicts
            if isinstance(first_valid, dict):
                segs = pd.json_normalize(col.dropna())
            else:
                segs = col.apply(pd.Series)

            if isinstance(segs, pd.Series):
                segs = segs.to_frame().T

            # Reset indices before concatenation
            speech_segments = speech_segments.reset_index(drop=True)
            segs = segs.reset_index(drop=True)

            keep_cols = ["speaker", "text"]
            if "position" in segs.columns:
                keep_cols.append("position")

            turnwise = pd.concat(
                [speech_segments.drop(columns=["speech_segments"]), segs[keep_cols]],
                axis=1
            )

            # Normalize
            turnwise["speaker"] = turnwise["speaker"].astype(str).str.strip().str.upper()
            turnwise = normalize_speakers(turnwise)

            # --- Filter matched speakers properly ---
            turnwise = filter_matched_speakers(turnwise, speaker_details)

            # Assign gender only from speaker_details
            speaker_gender_map = build_speaker_gender_map(speaker_details)
            turnwise["gender"] = turnwise["speaker"].map(speaker_gender_map).fillna("UNK")
            turnwise["gender_source"] = np.where(turnwise["gender"] != "UNK", "speaker_details", "UNK")

            # Word/char counts
            turnwise["word_count"] = turnwise["text"].astype(str).str.split().str.len()
            turnwise["char_count"] = turnwise["text"].astype(str).str.len()

            # Keep relevant columns
            base_cols = [
                'debate_id','year','decade','month','reference_date','chamber','title','topic',
                'hansard_reference','reference_volume','reference_columns','file_path','speaker',
                'gender','gender_source','text','word_count','char_count'
            ]
            if "position" in turnwise.columns:
                base_cols.insert(base_cols.index("word_count"), "position")
            turnwise = turnwise[base_cols]

            if "position" in turnwise.columns:
                turnwise = turnwise.sort_values(["debate_id", "position"]).reset_index(drop=True)
            else:
                turnwise = turnwise.sort_values(["debate_id"]).reset_index(drop=True)

            dfs_local.append(turnwise)

        if dfs_local:
            return pd.concat(dfs_local, ignore_index=True)
        return None

    except Exception as e:
        print(f"Error processing {f}: {e}")
        return None


if __name__ == "__main__":
    files = glob.glob(os.path.join(INPUT_PATH, "debates_*.parquet"))
    print(f"Found {len(files)} files.")

    dfs = []
    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 1)) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for i, future in enumerate(as_completed(futures), 1):
            f = futures[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    dfs.append(result)
                    print(f"[{i}/{len(files)}] Done {f}")
                else:
                    print(f"[{i}/{len(files)}] Skipped {f} (no data)")
            except Exception as e:
                print(f"Error in future for {f}: {e}")

    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    final_df.to_parquet(OUT_PATH, index=False)
    print(f"Processed {len(files)} files into {OUT_PATH}")
    print(final_df.head())
