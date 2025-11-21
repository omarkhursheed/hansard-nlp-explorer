#!/usr/bin/env python3
"""
Extract stereotypes about women from Hansard argument-mining results using an LLM.

For each row in a Hansard results Parquet (with a `reasons` column), this script:
  - builds a compact text block from each reason's rationale and quotes,
  - sends that evidence to an OpenRouter model via a few-shot prompt,
  - parses the returned JSON stereotypes, and
  - writes a flat Parquet with one row per stereotype instance.

It uses concurry for parallel calls and a shelve cache to avoid re-calling the LLM
for identical evidence blocks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shelve
import sys
from pathlib import Path
from typing import Any, Dict, List
from collections.abc import Sequence

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

from hansard.utils.path_config import Paths  # noqa: E402

try:
    from concurry import Worker, CallLimit  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "concurry is required for parallel execution. Install it via `pip install concurry`."
    ) from exc


DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 800
DEFAULT_MAX_WORKERS = 4
DEFAULT_CALLS_PER_MINUTE: int | None = None
DEFAULT_MAX_EVIDENCE_CHARS = 1800
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

POLARITY_OPTIONS = {"positive", "negative", "ambivalent", "neutral"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract stereotypes about women from Hansard argument-mining results."
    )
    parser.add_argument("--input", type=Path, required=True, help="Hansard results Parquet with a `reasons` column.")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet with stereotype instances.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Paths.PROMPTS_DIR / "hansard_stereotype_extraction_prompt.md",
        help="SYSTEM/USER prompt markdown.",
    )
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="OpenRouter endpoint.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model identifier for OpenRouter.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per response.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel worker count.")
    parser.add_argument("--calls-per-minute", type=int, default=0, help="Optional rate limit (0 disables).")
    parser.add_argument(
        "--concurry-retries",
        type=int,
        default=2,
        help="How many times concurry should retry a failed HTTP call before giving up.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional shelve cache path; defaults to <output_dir>/cache/hansard_stereotypes.shelve",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on number of rows from the input (for testing).",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=0,
        help="If >0, print evidence (or lack thereof) for up to this many rows to aid debugging.",
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_CHARS,
        help="Truncate each evidence block to this many characters before calling the LLM (0 disables truncation).",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=DEFAULT_LLM_RETRIES,
        help="Maximum number of times to re-query the LLM for a single row when JSON parsing fails.",
    )
    parser.add_argument(
        "--max-tokens-step",
        type=int,
        default=DEFAULT_MAX_TOKENS_STEP,
        help="When retrying a row, increase the per-call max_tokens by this amount (0 disables increments).",
    )
    parser.add_argument(
        "--max-tokens-ceiling",
        type=int,
        default=None,
        help="Upper bound for per-call max_tokens during retries (defaults to the initial --max-tokens).",
    )
    return parser.parse_args()


def load_prompt_sections(path: Path) -> Tuple[str, str]:
    """Load SYSTEM and USER prompts from a markdown file."""
    sections: Dict[str, List[str]] = {}
    current: str | None = None
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


def parse_reasons(raw_value: Any) -> List[Dict[str, Any]]:
    """Parse the `reasons` cell into a list of dicts."""
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []

    payload: Any = raw_value
    if isinstance(raw_value, str):
        s = raw_value.strip()
        if not s:
            return []
        # Try JSON first
        try:
            payload = json.loads(s)
        except json.JSONDecodeError:
            try:
                cleaned = _strip_numpy_array_reprs(s)
                payload = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                try:
                    payload = _safe_eval_numpy_repr(s)
                except Exception:
                    return []

    # Normalize into a Python list of dicts
    if isinstance(payload, dict):
        payload = [payload]
    elif hasattr(payload, "tolist"):
        try:
            payload = payload.tolist()
        except Exception:
            return []
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        payload = list(payload)
    else:
        return []

    return [r for r in payload if isinstance(r, dict)]


def _strip_numpy_array_reprs(text: str) -> str:
    """Best-effort removal of numpy array(...) wrappers."""
    result = text
    result = result.replace("array(", "(")
    result = re.sub(r",\s*dtype=object\s*\)", ")", result, flags=re.IGNORECASE)
    return result


def _safe_eval_numpy_repr(text: str) -> Any:
    """Evaluate numpy-style repr using a restricted environment."""
    env = {
        "__builtins__": {},
        "array": lambda x, dtype=None: x,
        "object": object,
    }
    return eval(text, env, {})  # noqa: S307 (input controlled)


def truncate_block(text: str, max_chars: int | None) -> str:
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # try to avoid cutting mid-word
    last_break = max(truncated.rfind("\n"), truncated.rfind(" "))
    if last_break > max_chars * 0.6:
        truncated = truncated[:last_break]
    return truncated.rstrip() + "\n[TRUNCATED]"


def build_evidence_block(reasons: List[Dict[str, Any]]) -> str:
    """Construct a compact evidence block from rationales and quotes."""
    lines: List[str] = []
    for idx, reason in enumerate(reasons, start=1):
        rationale = str(
            reason.get("rationale")
            or reason.get("free_text_reason")
            or ""
        ).strip()
        if rationale:
            lines.append(f"Reason {idx} rationale: {rationale}")

        quotes_value = reason.get("quotes")
        if quotes_value is None:
            quote_iter: List[Any] = []
        elif isinstance(quotes_value, (list, tuple)):
            quote_iter = list(quotes_value)
        elif hasattr(quotes_value, "tolist"):
            try:
                quote_iter = quotes_value.tolist()
            except Exception:
                quote_iter = []
        elif isinstance(quotes_value, Sequence) and not isinstance(quotes_value, (str, bytes)):
            quote_iter = list(quotes_value)
        else:
            quote_iter = []

        for q in quote_iter:
            if not isinstance(q, dict):
                continue
            q_text = str(q.get("text") or "").strip()
            source = str(q.get("source") or "").strip()
            if q_text:
                if source:
                    lines.append(f"Quote {idx} ({source}): {q_text}")
                else:
                    lines.append(f"Quote {idx}: {q_text}")
    return "\n".join(lines).strip()


def build_messages(evidence_block: str, system_prompt: str, user_prompt: str):
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": f"\n\nEVIDENCE:\n{evidence_block}"},
            ],
        },
    ]


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_stereotype_key(text: str) -> str:
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
    candidate = str(value).strip().lower()
    if candidate not in POLARITY_OPTIONS:
        raise ValueError(f"Unsupported polarity value: {value!r}")
    return candidate


def clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence != confidence:  # NaN
        return 0.0
    return max(0.0, min(1.0, confidence))


def extract_json_blob(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    fence = re.match(r"^```(?:json)?\s*(.*?)```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
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
    """Best-effort fixes for common JSON formatting issues in model output."""
    repaired = text
    # Ensure objects in an array are separated by commas: `}{` -> `},{`
    repaired = re.sub(r"}(\s*)(?={)", r"},\1", repaired)
    # Fix doubled quotes before field names: `""stereotype_text"` -> `"stereotype_text"`
    repaired = re.sub(r'""([A-Za-z_][A-Za-z0-9_]*")', r'"\1', repaired)
    return repaired


def parse_stereotype_response(raw_text: str, row_id: int) -> List[Dict[str, Any]]:
    if not raw_text:
        return []

    normalized_text = extract_json_blob(raw_text)
    if not normalized_text:
        return []

    try:
        payload = json.loads(normalized_text)
    except json.JSONDecodeError as exc:
        # Attempt a best-effort repair before giving up.
        repaired_text = repair_json_text(normalized_text)
        try:
            payload = json.loads(repaired_text)
        except json.JSONDecodeError as exc2:
            snippet = normalized_text[:200].replace("\n", " ")
            raise ValueError(
                f"Row {row_id} returned invalid JSON: {exc2}. Snippet: {snippet!r}"
            ) from exc2

    stereotypes = payload.get("stereotypes", [])
    if stereotypes is None:
        return []
    if not isinstance(stereotypes, list):
        raise ValueError(f"Row {row_id} `stereotypes` field must be a list.")

    records: List[Dict[str, Any]] = []
    seen_keys = set()
    for idx, item in enumerate(stereotypes):
        if not isinstance(item, dict):
            raise ValueError(f"Row {row_id} stereotype #{idx} must be an object.")

        text = normalize_whitespace(str(item.get("stereotype_text", "")))
        if not text:
            continue
        dimension = canonical_dimension(item.get("dimension", ""))
        polarity = canonical_polarity(item.get("polarity", ""))
        confidence = clamp_confidence(item.get("confidence", 0.0))

        key = (normalize_stereotype_key(text), dimension, polarity)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        records.append(
            {
                "row_id": row_id,
                "stereotype_text": text,
                "dimension": dimension,
                "polarity": polarity,
                "confidence": confidence,
                "normalized_key": key[0],
            }
        )

    return records


def deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        key = (record["row_id"], record["normalized_key"], record["dimension"], record["polarity"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


class HansardWorker(Worker):  # type: ignore
    def __init__(self, *, api_url: str, model_name: str, headers: dict, temperature: float, max_tokens: int):
        self.api_url = api_url
        self.model_name = model_name
        self.headers = headers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = requests.Session()

    def run_item(self, row_id: int, messages, *, max_tokens: int | None = None):
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
        return {"row_id": row_id, "text": assistant_text.strip()}


def main() -> None:
    args = parse_args()
    api_key = get_api_key()

    input_path = args.input
    output_path = args.output
    prompt_file = args.prompt_file
    api_url = args.api_url
    model_name = args.model
    temperature = args.temperature
    max_tokens = args.max_tokens
    max_tokens_ceiling = args.max_tokens_ceiling or max_tokens
    if max_tokens_ceiling < max_tokens:
        max_tokens_ceiling = max_tokens
    max_tokens_step = max(0, args.max_tokens_step)
    llm_retries = max(1, args.llm_retries)
    max_workers = args.max_workers
    calls_per_minute = args.calls_per_minute or None
    max_evidence_chars = args.max_evidence_chars if args.max_evidence_chars and args.max_evidence_chars > 0 else None
    concurry_retries = max(0, args.concurry_retries)

    if not input_path.exists():
        raise FileNotFoundError(f"Input Parquet not found: {input_path}")

    system_prompt, user_prompt = load_prompt_sections(prompt_file)

    df = pd.read_parquet(input_path)
    if "reasons" not in df.columns:
        raise ValueError("Input Parquet must contain a 'reasons' column.")
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    # Prepare cache
    cache_path = args.cache
    if cache_path is None:
        cache_dir = output_path.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "hansard_stereotypes.shelve"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://your-app-or-email.example",  # replace with your site/email per OpenRouter policy
        "X-Title": "Hansard Stereotype Extraction",
        "Content-Type": "application/json",
    }

    all_records: List[Dict[str, Any]] = []
    pending_items: List[Dict[str, Any]] = []
    debug_remaining = args.debug_samples

    with shelve.open(str(cache_path)) as cache:
        for row_id, (_, row) in enumerate(df.iterrows()):
            reasons = parse_reasons(row.get("reasons"))
            evidence_block = build_evidence_block(reasons)
            if not evidence_block:
                if debug_remaining > 0:
                    raw_repr = repr(row.get("reasons"))[:500]
                    print(f"[DEBUG] Row {row_id}: empty evidence. raw reasons snippet: {raw_repr}")
                    debug_remaining -= 1
                continue
            elif debug_remaining > 0:
                print(f"[DEBUG] Row {row_id} evidence block:\n{evidence_block[:1000]}\n---")
                debug_remaining -= 1
            evidence_block = truncate_block(evidence_block, max_evidence_chars)
            key_source = evidence_block
            cache_key = hashlib.sha1(key_source.encode("utf-8")).hexdigest()

            if cache_key in cache:
                cached_text = cache[cache_key]
                try:
                    recs = parse_stereotype_response(cached_text, row_id)
                except ValueError as exc:
                    print(
                        f"Cache entry for row {row_id} invalid ({exc}); re-requesting from LLM.",
                        file=sys.stderr,
                    )
                    del cache[cache_key]
                    pending_items.append(
                        {
                            "row_id": row_id,
                            "evidence": evidence_block,
                            "cache_key": cache_key,
                            "attempt": 0,
                            "max_tokens": min(
                                max_tokens_ceiling,
                                max_tokens + (max_tokens_step or 0),
                            ),
                        }
                    )
                    continue
                for r in recs:
                    r["dataset"] = "hansard"
                all_records.extend(recs)
            else:
                pending_items.append(
                    {
                        "row_id": row_id,
                        "evidence": evidence_block,
                        "cache_key": cache_key,
                        "attempt": 0,
                        "max_tokens": max_tokens,
                    }
                )

        if pending_items:
            limits = []
            if calls_per_minute:
                limits.append(CallLimit(window_seconds=60, capacity=int(calls_per_minute)))

            pool_options = {
                "mode": "thread",
                "max_workers": int(max_workers),
                "num_retries": int(concurry_retries),
                "retry_algorithm": "exponential",
            }
            if limits:
                pool_options["limits"] = limits

            with HansardWorker.options(**pool_options).init(
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
                            pool.run_item(
                                item["row_id"],
                                build_messages(item["evidence"], system_prompt, user_prompt),
                                max_tokens=item["max_tokens"],
                            ),
                        )
                        for item in current_batch
                    ]
                    iterable = tqdm(futures, desc="LLM rows", total=len(futures)) if tqdm else futures
                    next_batch: List[Dict[str, Any]] = []
                    for item, future in iterable:
                        result = future.result()
                        text = result["text"]
                        try:
                            recs = parse_stereotype_response(text, item["row_id"])
                        except ValueError as exc:
                            new_attempt = item["attempt"] + 1
                            if new_attempt >= llm_retries:
                                raise
                            bumped_tokens = item["max_tokens"]
                            if max_tokens_step and item["max_tokens"] < max_tokens_ceiling:
                                bumped_tokens = min(max_tokens_ceiling, item["max_tokens"] + max_tokens_step)
                            next_batch.append(
                                {
                                    **item,
                                    "attempt": new_attempt,
                                    "max_tokens": bumped_tokens,
                                }
                            )
                            continue

                        cache[item["cache_key"]] = text
                        for r in recs:
                            r["dataset"] = "hansard"
                        all_records.extend(recs)
                    current_batch = next_batch

    deduped = deduplicate_records(all_records)
    for record in deduped:
        record.pop("normalized_key", None)

    if not deduped:
        print("No stereotypes extracted.")
        return

    df_out = pd.DataFrame(deduped)
    # Attach row-level identifiers if present
    for col in ("speech_id", "debate_id", "speaker", "canonical_name", "gender", "year", "date", "chamber"):
        if col in df.columns:
            df_out[col] = df.loc[df_out["row_id"], col].values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    print(f"Saved {len(df_out)} stereotype instances to {output_path.resolve()}")

    # Also write a canonical, fully deduplicated view without row-level metadata.
    df_canon = df_out[["stereotype_text", "dimension", "polarity", "confidence"]].copy()
    df_canon["normalized_key"] = df_canon["stereotype_text"].apply(normalize_stereotype_key)
    df_canon = (
        df_canon.groupby(["normalized_key", "dimension", "polarity"], as_index=False)
        .agg(
            stereotype_text=("stereotype_text", "first"),
            confidence=("confidence", "mean"),
            count=("stereotype_text", "size"),
        )
    )
    df_canon = df_canon[["stereotype_text", "dimension", "polarity", "confidence", "count"]]

    canon_path = output_path.with_name(output_path.stem + "_canonical.parquet")
    df_canon.to_parquet(canon_path, index=False)
    print(f"Saved {len(df_canon)} canonical stereotypes to {canon_path.resolve()}")


if __name__ == "__main__":
    main()
