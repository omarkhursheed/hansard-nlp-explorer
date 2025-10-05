
"""

Analyze word usage in Hansard debates by gender.

Normalizations:
1. Speaker-level:
   Norm_words = observed words of speaker / expected_words_gender
   Expected_words_gender = (# speakers of gender in debate / total speakers in debate) * total words in debate

2. Group-level:
   Group_norm_words = observed words of group / expected_words_group
   Expected_words_group = (# speakers of gender in debate / total speakers in debate) * total words in debate

3. Flat per-speaker:
   Flat_norm_words = observed words of speaker / expected_words_flat
   Expected_words_flat = total words in debate / total speakers in debate

4. Avg words per turn:
   AvgWordsPerTurn_speaker = observed words of speaker / observed turns of speaker
   Norm_AvgWordsPerTurn = AvgWordsPerTurn_speaker / ExpectedAvgWordsPerTurn
   ExpectedAvgWordsPerTurn = total words in debate / total turns in debate
"""

import pandas as pd
import re
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("src/hansard/data/processed_fixed/cleaned_data/turnwise_debates_matched_speakers.parquet")
OUT_PATH = Path("src/hansard/data/analysis_data/attention_speech_len.parquet")
OUT_PATH_WORDS = Path("src/hansard/data/analysis_data/unique_top_words.parquet")
STOPWORD_PATH = Path("src/hansard/data/word_lists/custom_stop_words.txt")


# ---------------------------------------------------------
# Load custom stopwords
# ---------------------------------------------------------
def load_stopwords():
    stopwords = set()
    with open(STOPWORD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stopwords.add(word)
    return stopwords

# ---------------------------------------------------------
# Tokenization and word frequency
# ---------------------------------------------------------
def tokenize_text(text: str, stopwords: set, min_len: int = 2):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in stopwords and len(t) >= min_len]


# ---------------------------------------------------------
# Main processing
# ---------------------------------------------------------
def process_word_frequencies(df: pd.DataFrame):
    stopwords = load_stopwords()

    # Tokenize and recompute word_count
    df["tokens"] = df["text"].apply(lambda x: tokenize_text(str(x), stopwords))
    df["word_count"] = df["tokens"].apply(len)

    results = []

    # Iterate debate by debate
    for debate_id, debate_df in df.groupby("debate_id"):
        total_words = debate_df["word_count"].sum()
        total_speakers = debate_df["speaker"].nunique()
        total_turns = len(debate_df)

        # Gender-level speaker counts
        gender_counts = debate_df.groupby("gender")["speaker"].nunique().to_dict()

        # Expected values
        expected_group = {
            g: (count / total_speakers) * total_words
            for g, count in gender_counts.items()
        }
        expected_flat = total_words / total_speakers if total_speakers > 0 else 0
        expected_avg_turn = total_words / total_turns if total_turns > 0 else 0

        # Speaker-level observed values
        speaker_obs = (
            debate_df.groupby(["speaker", "gender"])
            .agg(observed_words=("word_count", "sum"),
                 observed_turns=("text", "count"))
            .reset_index()
        )

        # Extract shared debate metadata
        meta = {
            "year": debate_df["year"].iloc[0] if "year" in debate_df.columns else None,
            "decade": debate_df["decade"].iloc[0] if "decade" in debate_df.columns else None,
            "month": debate_df["month"].iloc[0] if "month" in debate_df.columns else None,
            "reference_date": debate_df["reference_date"].iloc[0] if "reference_date" in debate_df.columns else None,
            "chamber": debate_df["chamber"].iloc[0] if "chamber" in debate_df.columns else None,
            "title": debate_df["title"].iloc[0] if "title" in debate_df.columns else None,
            "topic": debate_df["topic"].iloc[0] if "topic" in debate_df.columns else None,
        }

        for _, row in speaker_obs.iterrows():
            spk, gender = row["speaker"], row["gender"]
            obs_words, obs_turns = row["observed_words"], row["observed_turns"]

            exp_gender = expected_group.get(gender, 0)
            norm_words = obs_words / exp_gender if exp_gender > 0 else None
            flat_norm = obs_words / expected_flat if expected_flat > 0 else None

            # Avg words per turn
            avg_words_per_turn = obs_words / obs_turns if obs_turns > 0 else None
            norm_avg_turn = (
                avg_words_per_turn / expected_avg_turn
                if avg_words_per_turn is not None and expected_avg_turn > 0
                else None
            )

            results.append({
                **meta,
                "debate_id": debate_id,
                "speaker": spk,
                "gender": gender,
                "observed_words": obs_words,
                "observed_turns": obs_turns,
                "expected_words_gender": exp_gender,
                "norm_words_speaker": norm_words,
                "expected_words_flat": expected_flat,
                "flat_norm_words": flat_norm,
                "avg_words_per_turn": avg_words_per_turn,
                "expected_avg_words_per_turn": expected_avg_turn,
                "norm_avg_words_per_turn": norm_avg_turn,
            })

        # Group-level entry
        for g in gender_counts.keys():
            obs_group_words = debate_df.loc[debate_df["gender"] == g, "word_count"].sum()
            exp_group_words = expected_group[g]
            group_norm = obs_group_words / exp_group_words if exp_group_words > 0 else None

            results.append({
                **meta,
                "debate_id": debate_id,
                "speaker": None,
                "gender": g,
                "observed_words": obs_group_words,
                "observed_turns": None,
                "expected_words_gender": exp_group_words,
                "group_norm_words": group_norm,
                "avg_words_per_turn": None,
                "expected_avg_words_per_turn": None,
                "norm_avg_words_per_turn": None,
            })

    return pd.DataFrame(results)

# ---------------------------------------------------------
# Top words by gender
# ---------------------------------------------------------
def top_words_by_gender(df: pd.DataFrame, top_n: int = 50, min_len: int = 2):
    stopwords = load_stopwords()
    tokens_by_gender = {"M": [], "F": []}

    for _, row in df.iterrows():
        gender = row["gender"]
        if pd.isna(gender) or gender not in tokens_by_gender:
            continue
        tokens = tokenize_text(str(row["text"]), stopwords, min_len=min_len)
        tokens_by_gender[gender].extend(tokens)

    top_words = {}
    for g, tokens in tokens_by_gender.items():
        counts = Counter(tokens)
        top_words[g] = counts.most_common(top_n)

    return top_words

# ---------------------------------------------------------
# Unique top words by gender (removes overlap)
# ---------------------------------------------------------
def unique_top_words(top_words: dict, top_n: int = 50):
    # Extract just the words
    male_words = {w for w, _ in top_words.get("M", [])}
    female_words = {w for w, _ in top_words.get("F", [])}

    # Find overlaps
    overlap = male_words & female_words

    # Remove overlap and keep top-n re-sorted
    unique_top = {}
    for g, words in top_words.items():
        filtered = [(w, c) for w, c in words if w not in overlap]
        unique_top[g] = filtered[:top_n]

    return unique_top

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    df = pd.read_parquet(INPUT_PATH)

    results_df = process_word_frequencies(df)
    results_df.to_parquet(OUT_PATH, index=False)

    top_words = top_words_by_gender(df, top_n=100)

    # Remove overlaps, keep only unique top words for each gender
    unique_words = unique_top_words(top_words, top_n=50)

    unique_df = (
        pd.DataFrame([
            {"gender": g, "word": w, "count": c}
            for g, pairs in unique_words.items()
            for w, c in pairs
        ]))
unique_df.to_parquet(OUT_PATH_WORDS, index=False)
    
