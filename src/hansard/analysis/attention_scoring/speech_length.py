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
import numpy as np
import re
from collections import Counter
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer

INPUT_PATH = Path("src/hansard/data/processed_fixed/cleaned_data/turnwise_debates_matched_speakers.parquet")
OUT_PATH = Path("src/hansard/data/analysis_data/attention_speech_len.parquet")
OUT_PATH_WORDS = Path("src/hansard/data/analysis_data/unique_top_words.parquet")
OUT_PATH_LOGODDS = Path("src/hansard/data/analysis_data/terms_logodds.parquet")
STOPWORD_PATH = Path("src/hansard/data/word_lists/custom_stop_words.txt")

N_JOBS = max(1, cpu_count() - 1)  # tune if needed

# -----------------------------
# Stopwords / tokenizer
# -----------------------------
def load_stopwords():
    sw = set()
    with open(STOPWORD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                sw.add(w)
    return sw

TOKEN_RE = re.compile(r"\b[a-zA-Z]+\b")

def tokenize_text(text: str, stopwords: set, min_len: int = 2):
    # lower + alphabetic only; drop short tokens; custom stopwords
    toks = TOKEN_RE.findall(text.lower())
    return [t for t in toks if len(t) >= min_len and t not in stopwords]

def _count_words_one(text: str, stopwords: set, min_len: int = 2) -> int:
    # count only; avoids materializing token list in the main df
    toks = TOKEN_RE.findall(text.lower())
    return sum(1 for t in toks if len(t) >= min_len and t not in stopwords)

def _tokens_one(text: str, stopwords: set, min_len: int = 2):
    return tokenize_text(text, stopwords, min_len=min_len)

# -----------------------------
# Parallel helpers
# -----------------------------
def parallel_map(func, iterable, processes=N_JOBS, chunksize=1000):
    if processes == 1:
        return list(map(func, iterable))
    with Pool(processes=processes) as pool:
        return list(pool.imap(func, iterable, chunksize=chunksize))

# -----------------------------
# Core processing (vectorized)
# -----------------------------
def process_word_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    stopwords = load_stopwords()

    # === 1) parallel word counts (no token column kept) ===
    texts = df["text"].astype(str).tolist()
    count_fn = partial(_count_words_one, stopwords=stopwords, min_len=2)
    df["word_count"] = parallel_map(count_fn, texts, processes=N_JOBS, chunksize=2000)

    # ensure speaker sanity (drop blanks)
    df = df[df["speaker"].astype(str).str.strip() != ""].copy()

    # === 2) debate-level totals (vectorized) ===
    # totals per debate
    debate_totals = (
        df.groupby("debate_id")
          .agg(total_words=("word_count", "sum"),
               total_speakers=("speaker", "nunique"),
               total_turns=("text", "size"),
               year=("year", "first"),
               decade=("decade", "first"),
               month=("month", "first"),
               reference_date=("reference_date", "first"),
               chamber=("chamber", "first"),
               title=("title", "first"),
               topic=("topic", "first"))
          .reset_index()
    )

    # unique speakers of each gender in debate
    speakers_by_gender = (
        df.groupby(["debate_id", "gender"])["speaker"]
          .nunique()
          .rename("speakers_of_gender")
          .reset_index()
    )

    # expected words per (debate, gender)
    exp_group = speakers_by_gender.merge(debate_totals[["debate_id","total_words","total_speakers"]], on="debate_id", how="left")
    exp_group["expected_words_gender"] = (
        exp_group["speakers_of_gender"] / exp_group["total_speakers"]
    ) * exp_group["total_words"]

    # === 3) speaker-level observed (vectorized) ===
    speaker_obs = (
        df.groupby(["debate_id", "speaker", "gender"])
          .agg(observed_words=("word_count", "sum"),
               observed_turns=("text", "size"))
          .reset_index()
    )

    # attach debate totals to each speaker row
    speaker_obs = speaker_obs.merge(
        debate_totals[["debate_id","total_words","total_speakers","total_turns",
                       "year","decade","month","reference_date","chamber","title","topic"]],
        on="debate_id", how="left"
    )

    # attach expected words for that (debate, gender)
    speaker_obs = speaker_obs.merge(
        exp_group[["debate_id","gender","expected_words_gender"]],
        on=["debate_id","gender"], how="left"
    )

    # compute norms (vectorized)
    speaker_obs["norm_words_speaker"] = (
        speaker_obs["observed_words"] / speaker_obs["expected_words_gender"]
    )
    speaker_obs["expected_words_flat"] = speaker_obs["total_words"] / speaker_obs["total_speakers"]
    speaker_obs["flat_norm_words"] = speaker_obs["observed_words"] / speaker_obs["expected_words_flat"]

    # avg words per turn + normalized
    speaker_obs["avg_words_per_turn"] = speaker_obs["observed_words"] / speaker_obs["observed_turns"].replace(0, np.nan)
    speaker_obs["expected_avg_words_per_turn"] = speaker_obs["total_words"] / speaker_obs["total_turns"].replace(0, np.nan)
    speaker_obs["norm_avg_words_per_turn"] = speaker_obs["avg_words_per_turn"] / speaker_obs["expected_avg_words_per_turn"]

    speaker_obs["row_type"] = "speaker"

        # === 4) group-level observed (vectorized) ===
    group_obs = (
        df.groupby(["debate_id", "gender"])
          .agg(observed_words=("word_count", "sum"))
          .reset_index()
    )

    group_obs = group_obs.merge(
        exp_group[["debate_id", "gender", "expected_words_gender"]],
        on=["debate_id", "gender"], how="left"
    ).merge(
        debate_totals[["debate_id", "year", "decade", "month", "reference_date", "chamber", "title", "topic"]],
        on="debate_id", how="left"
    )

    group_obs["group_norm_words"] = group_obs["observed_words"] / group_obs["expected_words_gender"]

    # 🔧 add missing columns so both DataFrames align for concat
    for col in [
        "observed_turns", "norm_words_speaker", "expected_words_flat", "flat_norm_words",
        "avg_words_per_turn", "expected_avg_words_per_turn", "norm_avg_words_per_turn"
    ]:
        group_obs[col] = np.nan

    group_obs["speaker"] = None
    group_obs["row_type"] = "group"

    # === 5) unify and return ===
    wanted_cols = [
        "year", "decade", "month", "reference_date", "chamber", "title", "topic",
        "debate_id", "speaker", "gender",
        "observed_words", "observed_turns",
        "expected_words_gender", "norm_words_speaker",
        "expected_words_flat", "flat_norm_words",
        "avg_words_per_turn", "expected_avg_words_per_turn", "norm_avg_words_per_turn",
        "group_norm_words", "row_type"
    ]

    # ensure speaker_obs also has group_norm_words column
    if "group_norm_words" not in speaker_obs.columns:
        speaker_obs["group_norm_words"] = np.nan
    speaker_obs["row_type"] = "speaker"

    out = pd.concat(
        [speaker_obs[wanted_cols], group_obs[wanted_cols]],
        ignore_index=True
    )

    return out


# -----------------------------
# Top words (optimized single pass)
# -----------------------------
def top_words_by_gender(df: pd.DataFrame, top_n: int = 50, min_len: int = 2):
    stopwords = load_stopwords()
    # restrict decades with meaningful female presence
    dec = df[df["decade"].isin([1970, 1980, 1990])][["gender","text"]].dropna(subset=["gender","text"])

    # parallel tokenization -> list of token lists
    tok_fn = partial(_tokens_one, stopwords=stopwords, min_len=min_len)
    tokens_list = parallel_map(tok_fn, dec["text"].astype(str).tolist(), processes=N_JOBS, chunksize=2000)

    # aggregate counters per gender (single pass)
    male_counter = Counter()
    female_counter = Counter()
    for (g, toks) in zip(dec["gender"].tolist(), tokens_list):
        if g == "M":
            male_counter.update(toks)
        elif g == "F":
            female_counter.update(toks)

    top = {
        "M": male_counter.most_common(top_n),
        "F": female_counter.most_common(top_n),
    }
    return top

def unique_top_words(top_words: dict, top_n: int = 50):
    male_words = {w for w, _ in top_words.get("M", [])}
    female_words = {w for w, _ in top_words.get("F", [])}
    overlap = male_words & female_words

    uniq = {}
    for g, pairs in top_words.items():
        filtered = [(w, c) for (w, c) in pairs if w not in overlap]
        uniq[g] = filtered[:top_n]
    return uniq

# -----------------------------
# Log-odds (uses same tokenizer)
# -----------------------------
def compute_log_odds(df: pd.DataFrame, top_n: int = 20):
    stopwords = list(load_stopwords())
    dec = df[df["decade"].isin([1970, 1980, 1990])]

    male_texts = dec.loc[dec["gender"] == "M", "text"].fillna("").tolist()
    female_texts = dec.loc[dec["gender"] == "F", "text"].fillna("").tolist()

    vectorizer = CountVectorizer(
        tokenizer=lambda x: tokenize_text(x, set(stopwords), min_len=2),
        preprocessor=None,
        lowercase=True,
        stop_words=None,
        min_df=2
    )
    X = vectorizer.fit_transform(male_texts + female_texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    Xm = X[:len(male_texts), :]
    Xf = X[len(male_texts):, :]

    # fast sparse sums (A1 gets 1D ndarray)
    freq_m = Xm.sum(axis=0).A1 + 1
    freq_f = Xf.sum(axis=0).A1 + 1

    p_m = freq_m / freq_m.sum()
    p_f = freq_f / freq_f.sum()
    log_odds = np.log(p_f / p_m)

    df_logodds = (
        pd.DataFrame({"word": vocab,
                      "freq_m": freq_m,
                      "freq_f": freq_f,
                      "log_odds_f_over_m": log_odds})
        .sort_values("log_odds_f_over_m", ascending=False)
        .reset_index(drop=True)
    )

    top_f = df_logodds.head(top_n)
    top_m = df_logodds.tail(top_n)

    return top_f, top_m, df_logodds

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("🔹 Loading parquet…")
    df = pd.read_parquet(INPUT_PATH)

    print("Original shape:", df.shape)
    df = df[df["gender"].isin(["M","F"])]
    df["text"] = df["text"].fillna("")

    print(df["gender"].value_counts())

    # === word frequency + normalizations ===
    print("🔹 Computing word counts + normalizations…")
    results_df = process_word_frequencies(df)
    results_df.to_parquet(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}")

    # === top words (unique) ===
    print("🔹 Computing top/unique words…")
    top_words = top_words_by_gender(df, top_n=100)
    unique_words = unique_top_words(top_words, top_n=50)
    unique_df = pd.DataFrame(
        [{"gender": g, "word": w, "count": c}
         for g, pairs in unique_words.items()
         for (w, c) in pairs]
    )
    unique_df.to_parquet(OUT_PATH_WORDS, index=False)
    print(f"Saved → {OUT_PATH_WORDS}")

    # === log-odds ===
    print("🔹 Computing log-odds distinctive terms…")
    top_f, top_m, df_logodds = compute_log_odds(df, top_n=20)
    df_logodds.to_parquet(OUT_PATH_LOGODDS, index=False)
    print(f"Saved → {OUT_PATH_LOGODDS}")
