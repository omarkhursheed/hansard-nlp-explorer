"""
Hansard Speech Length Attention Analysis
===================================
Analyze word usage in Hansard debates by gender.

This module computes several complementary views of verbal participation:
- Tokenization and word counting with custom stopwords (parallelized).
- Speaker-level normalization: observed vs. expected words by gender within a debate.
- Group-level normalization: observed vs. expected total words for a gender within a debate.
- Flat per-speaker normalization: observed words vs. equal-share baseline within a debate.
- Average words per turn (and normalized variant).
- Top/unique words by gender for decades with female presence.
- Log-odds (F over M) to identify statistically distinctive vocabulary.

Outputs:
- Per-speaker and per-group normalized metrics to Parquet.
- Unique top words by gender to Parquet.
- Full log-odds table (word, freq_m, freq_f, log_odds_f_over_m) to Parquet.

Notes:
- Normalizations answer different questions and should be interpreted accordingly.
- Means can be influenced by small-denominator debates; medians are often more robust.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import Set, Union
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer

INPUT_PATH = Path("src/hansard/data/processed_fixed/cleaned_data/turnwise_debates_matched_speakers.parquet")
OUT_PATH = Path("src/hansard/data/analysis_data/attention_speech_len.parquet")
OUT_PATH_WORDS = Path("src/hansard/data/analysis_data/unique_top_words.parquet")
OUT_PATH_LOGODDS = Path("src/hansard/data/analysis_data/terms_logodds.parquet")
STOPWORD_PATH = Path("src/hansard/data/word_lists/hansard_stopwords.csv")

N_JOBS = max(1, cpu_count() - 1)  # tune if needed

# -----------------------------
# Stopwords / tokenizer
# -----------------------------
def load_stopwords(include_conditional: bool = False, *, lowercase: bool = True) -> Set[str]:
    """
    Load custom stopwords from the CSV at `STOPWORD_PATH`.

    The CSV must contain columns: 'token' and 'type'.
    Rows with 'type' in {'keep'} are always included; if `include_conditional=True`,
    rows with 'type' == 'conditional' are also included.

    Args:
        include_conditional: Whether to include tokens labeled as 'conditional'.
        lowercase: If True, return tokens lowercased.

    Returns:
        A set of stopword strings.
    """
    df = pd.read_csv(STOPWORD_PATH)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"token", "type"}.issubset(df.columns):
        raise ValueError("CSV must contain 'token' and 'type' columns.")

    # Filter rows by type
    keep_labels = {"keep"}
    if include_conditional:
        keep_labels.add("conditional")

    df = df.loc[df["type"].str.strip().str.lower().isin(keep_labels), ["token"]]
    # Clean tokens
    toks = df["token"].astype(str).str.strip()
    if lowercase:
        toks = toks.str.lower()

    return set(toks[toks.ne("")].unique())

TOKEN_RE = re.compile(r"\b[a-zA-Z]+\b")

def tokenize_text(text: str, stopwords: set, min_len: int = 2):
    """
    Tokenize a text into alphabetic lowercase tokens, filtering by length and stopwords.

    Args:
        text: Raw text.
        stopwords: Set of tokens to exclude.
        min_len: Minimum token length to keep.

    Returns:
        A list of filtered tokens.
    """
    # lower + alphabetic only; drop short tokens; custom stopwords
    toks = TOKEN_RE.findall(text.lower())
    return [t for t in toks if len(t) >= min_len and t not in stopwords]

def _count_words_one(text: str, stopwords: set, min_len: int = 2) -> int:
    """
    Count tokens in `text` after applying the same tokenization/filtering rules
    as `tokenize_text`, without materializing the token list.

    Args:
        text: Raw text.
        stopwords: Set of tokens to exclude.
        min_len: Minimum token length to count.

    Returns:
        Integer count of kept tokens.
    """
    # count only; avoids materializing token list in the main df
    toks = TOKEN_RE.findall(text.lower())
    return sum(1 for t in toks if len(t) >= min_len and t not in stopwords)

def _tokens_one(text: str, stopwords: set, min_len: int = 2):
    """
    Convenience wrapper for producing filtered tokens; useful for parallel map.

    Args:
        text: Raw text.
        stopwords: Set of tokens to exclude.
        min_len: Minimum token length to keep.

    Returns:
        List of tokens.
    """
    return tokenize_text(text, stopwords, min_len=min_len)

# -----------------------------
# Parallel helpers
# -----------------------------
def parallel_map(func, iterable, processes=N_JOBS, chunksize=1000):
    """
    Apply `func` to `iterable` in parallel using a multiprocessing pool.

    Args:
        func: Callable to apply to items of `iterable`.
        iterable: Iterable of inputs.
        processes: Number of worker processes. If 1, falls back to serial map.
        chunksize: Chunk size for `imap` to reduce IPC overhead.

    Returns:
        List of results, preserving order of `iterable`.
    """
    if processes == 1:
        return list(map(func, iterable))
    with Pool(processes=processes) as pool:
        return list(pool.imap(func, iterable, chunksize=chunksize))

# -----------------------------
# Core processing (vectorized)
# -----------------------------
def process_word_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute speaker- and group-level word-based participation metrics per debate.

    Steps:
      1) Parallel token counting to derive `word_count` per row.
      2) Debate-level totals: total words, speakers, turns, and metadata.
      3) Expected words per (debate, gender) based on share of speakers.
      4) Speaker-level observed words/turns and normalized metrics:
         - norm_words_speaker (vs. gender-expected)
         - flat_norm_words (vs. equal-share per speaker)
         - avg_words_per_turn and its normalized variant
      5) Group-level observed words and group_norm_words (vs. gender-expected).
      6) Return a unified DataFrame containing both speaker and group rows.

    Args:
        df: Turn-level dataframe with columns including
            ['debate_id','speaker','gender','text','year','decade','month',
             'reference_date','chamber','title','topic'].

    Returns:
        A DataFrame with per-speaker and per-group normalized metrics and metadata.
    """
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
    """
    Compute top-N most frequent tokens per gender after restricting to decades
    with at least a minimal level of female presence.

    This is a raw-frequency view (no statistical contrast): tokens are counted
    per gender and the top-N are returned.

    Args:
        df: Turn-level dataframe with 'gender', 'text', and 'decade'.
        top_n: Number of top tokens to return per gender.
        min_len: Minimum token length to keep.

    Returns:
        Dict {'M': [(token, count), ...], 'F': [(token, count), ...]}.
    """
    # stopwords = load_stopwords()
    stopwords = []
    female_counts = df[df["gender"] == "F"]["decade"].value_counts()
    female_decades = female_counts[female_counts >= 10].index
    dec = df[df["decade"].isin(female_decades)][["gender", "text"]].dropna(subset=["gender", "text"])

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
    """
    Remove overlapping tokens between genders' top lists and return the
    top-N unique tokens for each gender (still based on raw counts).

    Args:
        top_words: Dict from `top_words_by_gender`.
        top_n: Number of items to keep after removing overlap.

    Returns:
        Dict {'M': [(token, count), ...], 'F': [(token, count), ...]} with uniqueness enforced.
    """
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
    """
    Compute log-odds ratios (F over M) for all tokens using CountVectorizer
    with the same tokenizer, restricted to decades with sufficient female presence.

    Steps:
      - Build document-term matrix over male + female documents.
      - Sum frequencies by gender and add +1 smoothing.
      - Compute log(p_F / p_M) per token.
      - Return top-N female and male distinctive tokens (by log-odds).

    Args:
        df: Turn-level dataframe with 'gender', 'text', and 'decade'.
        top_n: Number of most distinctive tokens to return for each side.

    Returns:
        top_f: DataFrame of top-N tokens with highest log-odds (female-associated).
        top_m: DataFrame of top-N tokens with lowest log-odds (male-associated).
        df_logodds: Full DataFrame with columns ['word','freq_m','freq_f','log_odds_f_over_m'].
    """
    # stopwords = list(load_stopwords())
    stopwords = []
    female_counts = df[df["gender"] == "F"]["decade"].value_counts()
    female_decades = female_counts[female_counts >= 10].index
    dec = df[df["decade"].isin(female_decades)][["gender", "text"]].dropna(subset=["gender", "text"])


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
