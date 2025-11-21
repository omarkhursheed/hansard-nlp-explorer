#!/usr/bin/env python3
"""Parallel Gemini Flash runner that filters, chunks, and saves responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import requests
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None

# Ensure project src is on the import path
# This script lives under scripts/analysis/compare_hansard_trh/, so the repo root is parents[3].
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hansard.utils.path_config import Paths

try:
    from concurry import Worker, CallLimit  # type: ignore
except ImportError as exc:
    raise ImportError(
        "concurry is required for parallel execution. Install it via `pip install concurry`."
    ) from exc


# ---- Defaults (override via CLI) ----
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 800
DEFAULT_MAX_WORKERS = 4
DEFAULT_CALLS_PER_MINUTE: int | None = None
DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 400
DEFAULT_LLM_RETRIES = 3
DEFAULT_MAX_TOKENS_STEP = 400

DIMENSION_OPTIONS = [
    "equality",
    "competence_capacity",
    "emotion_morality",
    "social_order_stability",
    "tradition_precedent",
    "instrumental_effects",
    "religion_family",
    "social_experiment",
    "other",
]

INDIRECT_VERBS = [
    "think",
    "thinks",
    "say",
    "says",
    "believe",
    "believes",
    "claim",
    "claims",
    "argue",
    "argues",
    "feel",
    "feels",
    "insist",
    "contend",
    "maintain",
    "considered",
]

POLARITY_OPTIONS = {"positive", "negative", "ambivalent"}


def load_target_terms(path: Path) -> List[str]:
    """Load female-target terms from the canonical wordlist."""
    if not path.exists():
        raise FileNotFoundError(f"Female words file not found at {path}")

    terms = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if not word or word.startswith("#"):
            continue
        terms.add(word.lower())

    if not terms:
        raise ValueError(f"No terms found in {path}")

    return sorted(terms)


TARGET_TERMS = load_target_terms(Paths.FEMALE_WORDS)
TARGET_TERMS_PATTERN = "|".join(re.escape(term) for term in TARGET_TERMS)


# CLI -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter toxic statements about women and score them with Gemini Flash."
    )
    parser.add_argument("--text-file", type=Path, required=True, help="Path to the input text file.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the parquet responses.")
    parser.add_argument("--prompt-file", type=Path, required=True, help="Prompt definition markdown.")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="OpenRouter endpoint.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model identifier for OpenRouter.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Approximate characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Characters overlapped.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel worker count.")
    parser.add_argument("--calls-per-minute", type=int, default=0, help="Optional rate limit (0 disables).")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per response.")
    parser.add_argument(
        "--test-chunk",
        type=int,
        default=None,
        help="If provided, run only this chunk index (0-based) for quick prompting tests.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Optional path to a JSON cache of chunk responses (speeds up reruns).",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=DEFAULT_LLM_RETRIES,
        help="Maximum number of times to re-query a chunk when the LLM returns invalid JSON.",
    )
    parser.add_argument(
        "--max-tokens-step",
        type=int,
        default=DEFAULT_MAX_TOKENS_STEP,
        help="Increase per-chunk max_tokens by this amount on each retry (0 disables).",
    )
    parser.add_argument(
        "--max-tokens-ceiling",
        type=int,
        default=None,
        help="Upper bound for per-chunk max_tokens during retries (defaults to --max-tokens).",
    )
    return parser.parse_args()


def load_prompt_sections(path: Path) -> Tuple[str, str]:
    """Load SYSTEM and USER prompts from a markdown file."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found at {path}")

    sections = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        header = line.strip()
        if header in {"SYSTEM", "USER"}:
            current = header
            sections[current] = []
            continue
        if current:
            sections[current].append(line.rstrip())

    system_prompt = "\n".join(sections.get("SYSTEM", [])).strip()
    user_prompt = "\n".join(sections.get("USER", [])).strip()
    if not system_prompt or not user_prompt:
        raise ValueError(f"Prompt file {path} must contain SYSTEM and USER sections.")
    return system_prompt, user_prompt



def get_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENROUTER_API_KEY before running this script.")
    return api_key


def split_into_statements(text: str) -> List[str]:
    """Split on sentence-like boundaries or blank lines to keep manageable chunks."""
    chunks = re.split(r"(?:(?<=\.)|(?<=!)|(?<=\?))\s+|\n{2,}", text)
    return [c.strip() for c in chunks if c.strip()]


def is_indirect_reference(statement: str) -> bool:
    pattern = (
        r"\b(" + "|".join(INDIRECT_VERBS) + r")\s+(?:that\s+)?(" + TARGET_TERMS_PATTERN + r")\b"
    )
    return re.search(pattern, statement, flags=re.IGNORECASE) is not None


def has_target_subject(statement: str) -> bool:
    subject_pattern = r"^\W*(?:" + TARGET_TERMS_PATTERN + r")\b"
    inline_pattern = r'[\.\!\?"\n]\s*(?:' + TARGET_TERMS_PATTERN + r")\b"
    subject_hit = re.search(subject_pattern, statement, flags=re.IGNORECASE)
    inline_hit = re.search(inline_pattern, statement, flags=re.IGNORECASE)
    return bool(subject_hit or inline_hit)


def deduplicate_statements(statements: List[str]) -> List[str]:
    """Remove repeated statements while preserving their original order."""
    seen = set()
    unique: List[str] = []
    for stmt in statements:
        normalized = " ".join(stmt.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(stmt)
    return unique


def filter_statements(text: str) -> str:
    statements = split_into_statements(text)
    keep: List[str] = []
    for stmt in statements:
        if not has_target_subject(stmt):
            continue
        if is_indirect_reference(stmt):
            continue
        keep.append(stmt.strip())
    deduped = deduplicate_statements(keep)
    return "\n\n".join(deduped)


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks that preserve rough paragraph boundaries."""
    text = text.strip()
    if not text:
        return []
    if max_chars <= overlap:
        raise ValueError("`max_chars` must be greater than `overlap`.")

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_chars)
        slice_text = text[start:end]
        if end < length:
            boundary = slice_text.rfind("\n\n")
            if boundary == -1:
                boundary = slice_text.rfind("\n")
            if boundary > int(0.4 * len(slice_text)):
                end = start + boundary
                slice_text = text[start:end]
        chunk = slice_text.strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, 0)
    return chunks


def build_messages(chunk: str, chunk_id: int, system_prompt: str, user_prompt: str):
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": f"\n\n--- CHUNK {chunk_id} ---\n{chunk}"},
            ],
        },
    ]


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_stereotype_key(text: str) -> str:
    """Normalize stereotype text for deduplication."""
    cleaned = normalize_whitespace(text).lower()
    cleaned = cleaned.strip("\"'. ")
    cleaned = re.sub(r"[.?!]+$", "", cleaned).strip()
    return cleaned


def canonical_dimension(value: str) -> str:
    """Map a free-form dimension string to one of the allowed options.

    We expect the model to use the exact labels from DIMENSION_OPTIONS, but we
    are lenient about case and incidental whitespace.
    """
    if not isinstance(value, str):
        raise ValueError("Dimension must be a string.")
    candidate = normalize_whitespace(value)
    lower_candidate = candidate.lower()
    for option in DIMENSION_OPTIONS:
        if lower_candidate == option.lower():
            return option
    raise ValueError(f"Unsupported dimension value: {value!r}")


def canonical_polarity(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Polarity must be a string.")
    candidate = value.strip().lower()
    if candidate not in POLARITY_OPTIONS:
        raise ValueError(f"Unsupported polarity value: {value!r}")
    return candidate


def clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence != confidence:  # NaN guard
        return 0.0
    return max(0.0, min(1.0, confidence))


def parse_stereotype_response(raw_text: str, chunk_id: int) -> List[Dict[str, Any]]:
    if not raw_text:
        return []

    def extract_json_blob(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        fence_match = re.match(r"^```(?:json)?\s*(.*?)```$", stripped, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()
        lower = stripped.lower()
        if lower.startswith("json"):
            if "\n" in stripped:
                stripped = stripped.split("\n", 1)[1].strip()
            else:
                stripped = stripped[4:].lstrip(": \t").strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and start < end:
            return stripped[start : end + 1].strip()
        return stripped

    def repair_json_text(text: str) -> str:
        repaired = re.sub(r"}(\s*)(?={)", r"},\1", text)
        return repaired

    normalized_text = extract_json_blob(raw_text)
    if not normalized_text:
        return []

    def try_parse_json(text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    payload = try_parse_json(normalized_text)
    if payload is None:
        repaired_text = repair_json_text(normalized_text)
        payload = try_parse_json(repaired_text)

    if payload is None:
        # Keep the error compact so retries don't flood the logs.
        snippet = normalized_text[:120].replace("\n", " ")
        raise ValueError(f"Chunk {chunk_id} returned invalid JSON: {snippet!r}")

    stereotypes = payload.get("stereotypes", [])
    if stereotypes is None:
        return []
    if not isinstance(stereotypes, list):
        raise ValueError(f"Chunk {chunk_id} `stereotypes` field must be a list.")

    chunk_records: List[Dict[str, Any]] = []
    seen_in_chunk = set()
    for idx, item in enumerate(stereotypes):
        if not isinstance(item, dict):
            raise ValueError(f"Chunk {chunk_id} stereotype #{idx} must be an object.")

        text = normalize_whitespace(str(item.get("stereotype_text", "")))
        if not text:
            continue
        try:
            dimension = canonical_dimension(item.get("dimension", ""))
        except ValueError as exc:
            print(f"Warning: Chunk {chunk_id} stereotype #{idx} skipped ({exc}).", file=sys.stderr)
            continue
        try:
            polarity = canonical_polarity(item.get("polarity", ""))
        except ValueError as exc:
            print(f"Warning: Chunk {chunk_id} stereotype #{idx} skipped ({exc}).", file=sys.stderr)
            continue
        confidence = clamp_confidence(item.get("confidence", 0.0))

        key = (normalize_stereotype_key(text), dimension, polarity)
        if key in seen_in_chunk:
            continue
        seen_in_chunk.add(key)

        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "stereotype_text": text,
                "dimension": dimension,
                "polarity": polarity,
                "confidence": confidence,
                "normalized_key": key[0],
            }
        )

    return chunk_records


def deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        key = (record["normalized_key"], record["dimension"], record["polarity"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def compute_chunk_hash(chunk: str) -> str:
    return hashlib.sha256(chunk.encode("utf-8")).hexdigest()


def load_cache(path: Path | None) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Warning: Failed to read cache file {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"Warning: Cache file {path} is not a JSON object; ignoring.", file=sys.stderr)
        return {}
    cache: Dict[str, str] = {}
    for key, value in data.items():
        cache[str(key)] = "" if value is None else str(value)
    return cache


def save_cache(path: Path | None, cache_data: Dict[str, str]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(cache_data, indent=2, sort_keys=True)
    path.write_text(serialized, encoding="utf-8")


class ChunkWorker(Worker):  # type: ignore
    def __init__(self, *, api_url: str, model_name: str, headers: dict, temperature: float, max_tokens: int):
        self.api_url = api_url
        self.model_name = model_name
        self.headers = headers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = requests.Session()

    def run_chunk(self, chunk_id: int, messages, *, max_tokens: int | None = None):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        response = self.session.post(self.api_url, headers=self.headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"]
        if isinstance(assistant_message, list):
            assistant_text = "".join(part.get("text", "") for part in assistant_message)
        else:
            assistant_text = assistant_message or ""
        return {"chunk_id": chunk_id, "text": assistant_text.strip()}


def main():
    args = parse_args()
    api_key = get_api_key()
    text_file = args.text_file
    output_path = args.output
    prompt_file = args.prompt_file
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    max_workers = args.max_workers
    calls_per_minute = args.calls_per_minute or None
    temperature = args.temperature
    max_tokens = args.max_tokens
    api_url = args.api_url
    model_name = args.model

    system_prompt, user_prompt = load_prompt_sections(prompt_file)

    raw_text = text_file.read_text(encoding="utf-8")
    filtered_text = filter_statements(raw_text)
    if not filtered_text:
        raise ValueError("No statements mentioning women (or synonyms) were found in the file.")

    chunks = chunk_text(filtered_text, max_chars=chunk_size, overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Filtering removed all content; adjust your heuristics or prompt.")

    total_chunks = len(chunks)
    print(f"Prepared {total_chunks} filtered chunks (max ~{chunk_size} chars each).")

    chunk_entries = [
        {
            "chunk_id": idx,
            "text": chunk_text,
            "hash": compute_chunk_hash(chunk_text),
        }
        for idx, chunk_text in enumerate(chunks)
    ]
    if args.test_chunk is not None:
        test_idx = args.test_chunk
        if test_idx < 0 or test_idx >= total_chunks:
            raise ValueError(f"--test-chunk must be between 0 and {total_chunks - 1}.")
        print(f"[TEST MODE] Running only chunk index {test_idx} of {total_chunks}.")
        chunk_entries = [chunk_entries[test_idx]]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://your-app-or-email.example",  # Replace with your site/email per OpenRouter policy
        "X-Title": "Hansard NLP Explorer Notebook",
        "Content-Type": "application/json",
    }

    cache_file = args.cache_file
    cache_store = load_cache(cache_file)
    records: List[Dict[str, Any]] = []
    attempts: Dict[int, int] = {}
    pending_items: List[Dict[str, Any]] = []
    cache_hits = 0

    llm_retries = max(1, args.llm_retries)
    max_tokens_step = max(0, args.max_tokens_step)
    max_tokens_ceiling = args.max_tokens_ceiling or max_tokens
    if max_tokens_ceiling < max_tokens:
        max_tokens_ceiling = max_tokens

    def tokens_for_attempt(attempt: int) -> int:
        increment = max_tokens_step * attempt if max_tokens_step else 0
        return min(max_tokens_ceiling, max_tokens + increment)

    for entry in chunk_entries:
        chunk_id = entry["chunk_id"]
        cached_text = cache_store.get(entry["hash"])
        if not cached_text:
            pending_items.append(
                {
                    "chunk_id": chunk_id,
                    "text": entry["text"],
                    "hash": entry["hash"],
                    "attempt": 0,
                    "max_tokens": tokens_for_attempt(0),
                }
            )
            continue
        try:
            chunk_records = parse_stereotype_response(cached_text, chunk_id)
        except ValueError as exc:
            print(
                f"Cache entry for chunk {chunk_id} invalid ({exc}); reprocessing.",
                file=sys.stderr,
            )
            cache_store.pop(entry["hash"], None)
            attempts[chunk_id] = 1
            pending_items.append(
                {
                    "chunk_id": chunk_id,
                    "text": entry["text"],
                    "hash": entry["hash"],
                    "attempt": attempts[chunk_id],
                    "max_tokens": tokens_for_attempt(attempts[chunk_id]),
                }
            )
            continue
        records.extend(chunk_records)
        cache_hits += 1

    if cache_file:
        print(
            f"Cache status: {cache_hits} cached / {len(chunk_entries)} total chunks."
        )

    limits = []
    if calls_per_minute:
        limits.append(CallLimit(window_seconds=60, capacity=int(calls_per_minute)))

    pool_options = {
        "mode": "thread",
        "max_workers": int(max_workers),
        "num_retries": 2,
        "retry_algorithm": "exponential",
    }
    if limits:
        pool_options["limits"] = limits

    cache_updated = False
    if pending_items:
        with ChunkWorker.options(**pool_options).init(
            api_url=api_url,
            model_name=model_name,
            headers=headers,
            temperature=temperature,
            max_tokens=max_tokens,
        ) as pool:
            current_batch = pending_items
            while current_batch:
                futures = [
                    (
                        item,
                        pool.run_chunk(
                            item["chunk_id"],
                            build_messages(item["text"], item["chunk_id"], system_prompt, user_prompt),
                            max_tokens=item["max_tokens"],
                        ),
                    )
                    for item in current_batch
                ]
                iterable = tqdm(futures, desc="LLM chunks", total=len(futures)) if tqdm else futures
                next_batch: List[Dict[str, Any]] = []
                for item, future in iterable:
                    result = future.result()
                    chunk_id = item["chunk_id"]
                    try:
                        chunk_records = parse_stereotype_response(result["text"], chunk_id)
                    except ValueError as exc:
                        new_attempt = item["attempt"] + 1
                        attempts[chunk_id] = new_attempt
                        if new_attempt >= llm_retries:
                            # Give up on this chunk but keep the overall run going.
                            print(
                                f"Skipping chunk {chunk_id} after {new_attempt} failed JSON parses.",
                                file=sys.stderr,
                            )
                            continue
                        print(f"Retrying chunk {chunk_id} (attempt {new_attempt})", file=sys.stderr)
                        cache_store.pop(item["hash"], None)
                        next_batch.append(
                            {
                                **item,
                                "attempt": new_attempt,
                                "max_tokens": tokens_for_attempt(new_attempt),
                            }
                        )
                        continue

                    records.extend(chunk_records)
                    if cache_file:
                        cache_store[item["hash"]] = result["text"]
                        cache_updated = True
                current_batch = next_batch

    if cache_updated:
        save_cache(cache_file, cache_store)

    deduped = deduplicate_records(records)
    for record in deduped:
        record.pop("normalized_key", None)

    columns = ["chunk_id", "stereotype_text", "dimension", "polarity", "confidence"]
    df = pd.DataFrame(deduped, columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} responses to {output_path.resolve()}")


if __name__ == "__main__":
    main()
