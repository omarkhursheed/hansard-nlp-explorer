#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A‑Open Only Pipeline — Paths-based IO + OpenRouter + concurry
=============================================================

- IO uses Paths (no path args):
    input_dir  = Paths.DATA_DIR / 'derived' / 'speeches_full'
    output_dir = Paths.DATA_DIR / 'analysis_data' / 'arg_mining'

- Per-year input:   speeches_{YEAR}.parquet
- Per-year output:  args_{YEAR}.parquet   (consolidated, all chambers/eras)
- Also writes stratified copies under:
    <output_dir>/era=<ERA>/chamber=<CHAMBER>/args_{YEAR}.parquet

- LLM providers:
    • heuristic (fast regex baseline)
    • openrouter (chat completions via https://openrouter.ai/api/v1/chat/completions)

- Parallelization:
    • Uses concurry.Worker with thread mode to parallelize LLM calls.
    • Optional call rate limiting via concurry.CallLimit.
    • Retries with exponential backoff via concurry options.

Environment variables for OpenRouter
------------------------------------
OPENROUTER_API_KEY  : required for provider=openrouter
OPENROUTER_SITE     : optional HTTP-Referer header (e.g., https://yourapp.example)
OPENROUTER_APP      : optional X-Title header (e.g., Hansard Arg Mining)

"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shelve
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests

SYSTEM_USER_RE = re.compile(r"(?s)^\s*SYSTEM\s*(.*?)\s*USER\s*(.*)$")

# ---- Project path wiring ----
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'hansard'))
try:
    from utils.path_config import Paths  # type: ignore
except Exception:
    Paths = None  # Fallback if unavailable

# ---- concurry (parallelization) ----
try:
    from concurry import Worker, CallLimit  # type: ignore
    HAS_CONCURRY = True
except Exception:
    HAS_CONCURRY = False

# Optional: pyarrow for Parquet with nested columns
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False


# ==============================
# Utilities
# ==============================

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_ws(s: str) -> str:
    return re.sub(r"\\s+", " ", s or "").strip()


def extract_prompt_version(prompt_path: Optional[Path]) -> str:
    if not prompt_path:
        return "v0.0"
    name = prompt_path.stem
    m = re.search(r"(v[\\d._-]+)$", name)
    return m.group(1) if m else name


def now_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + sha1(str(random.random()))[:8]


# ==============================
# Era & Chamber
# ==============================

def assign_era(year: int) -> str:
    if year <= 1918:
        return "1803-1918"
    if 1919 <= year <= 1928:
        return "1919-1928"
    if 1929 <= year <= 1958:
        return "1929-1958"
    return "1959-2005"


def normalize_chamber(value: Any) -> str:
    s = str(value or "").strip().lower()
    if "lord" in s:
        return "lords"
    if "common" in s:
        return "commons"
    return "unknown"


# ==============================
# Extraction client(s)
# ==============================

from typing import Optional

@dataclass
class ExtractedArgument:
    quote: str
    span: Dict[str, int]                    # {"start": int, "end": int}
    free_text_reason: str                   # distilled rationale for this reason
    stance_label: Optional[str] = None      # "for" | "against"
    bucket_key: Optional[str] = None        # taxonomy key (e.g., "social_experiment")
    bucket_open: Optional[str] = None       # custom name if bucket_key == "other"

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "quote": self.quote,
            "span": self.span,
            "free_text_reason": self.free_text_reason,
        }
        if self.stance_label in {"for", "against"}:
            d["stance_label"] = self.stance_label
        if self.bucket_key:
            d["bucket_key"] = self.bucket_key
        if self.bucket_open:
            d["bucket_open"] = self.bucket_open
        return d


@dataclass
class ExtractionResult:
    stance: str
    arguments: List[ExtractedArgument]
    confidence: float = 0.0
    raw: Optional[Dict[str, Any]] = None


class LLMClient:
    def __init__(self, prompt_text: str, provider_name: str = "heuristic"):
        self.prompt_text = prompt_text
        self.provider_name = provider_name

    def extract(self, target_text: str, local_context: str) -> ExtractionResult:
        raise NotImplementedError



class OpenRouterClient(LLMClient):  # actual HTTP calls
    """
    Minimal OpenRouter chat.completions client using requests.
    - Endpoint: https://openrouter.ai/api/v1/chat/completions
    - Auth: Authorization: Bearer <OPENROUTER_API_KEY>
    - Optional headers: HTTP-Referer, X-Title
    """
    def __init__(
        self,
        prompt_text: str,
        model: str = "openrouter/auto",
        temperature: float = 0.0,
        top_p: float = 1.0,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 60,
    ):
        super().__init__(prompt_text, provider_name="openrouter")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")

        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in environment.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        referer = os.environ.get("OPENROUTER_SITE")
        if referer:
            self.headers["HTTP-Referer"] = referer
        title = os.environ.get("OPENROUTER_APP")
        if title:
            self.headers["X-Title"] = title

    @staticmethod
    def _json_schema_instruction() -> str:
        return (
            "Return ONLY a JSON object with keys exactly:\n"
            "{\n"
            '  "stance": "for | against | both | neutral | irrelevant",\n'
            '  "reasons": [ { "bucket_key": str, "bucket_open": str, "stance_label": "for | against", '
            '"rationale": str, "quotes": [str, str] } ],\n'
            '  "top_quote": str,\n'
            '  "confidence": number\n'
            "}\n"
            "Quotes must be exact substrings of TARGET; no paraphrase/ellipsis. "
            "If 'neutral' or 'irrelevant', reasons must be []. "
            "Max 3 reasons; max 2 quotes per reason; each quote 40–140 chars."
        )
    
    def _build_messages(self, target_text: str, local_context: str):
        pt = (self.prompt_text or "").strip()
        if pt:
            m = SYSTEM_USER_RE.match(pt)
            if m:
                sys_part = (m.group(1) or "").strip()
                usr_part = (m.group(2) or "").strip()
                user_filled = (
                    usr_part
                    .replace("{{TARGET_TURN_TEXT}}", target_text)
                    .replace("{{NEIGHBOR_TURNS_TEXT}}", local_context)
                )
                system = (sys_part + "\n\n" + self._json_schema_instruction()).strip()
                return [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_filled},
                ]
        # Fallback if the prompt isn't in SYSTEM/USER format
        system = (
            "You are an argument-mining assistant for one parliamentary turn.\n"
            "Use CONTEXT only to interpret stance; extract quotes only from TARGET.\n"
            + self._json_schema_instruction()
        )
        user = f"TARGET:\n{target_text}\n\nCONTEXT (neighbors, read-only):\n{local_context}\n"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def extract(self, target_text: str, local_context: str) -> ExtractionResult:

        messages = self._build_messages(target_text, local_context)

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        # LLM-compatible response
        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = json.dumps({"stance": "neutral", "arguments": [], "confidence": 0.0})

        # Best-effort JSON parse
        try:
            parsed = json.loads(content)
        except Exception:
            # try to salvage JSON substring
            m = re.search(r"\\{.*\\}", content, flags=re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {"stance": "neutral", "arguments": [], "confidence": 0.0}

        stance = parsed.get("stance", "irrelevant")
        if stance not in {"for","against","both","neutral","irrelevant"}:
            stance = "irrelevant"
        confidence = float(parsed.get("confidence", 0.0))

        # Build arguments and fill spans by locating quotes in target_text
        args: List[ExtractedArgument] = []
        reasons = parsed.get("reasons", []) or []
        for r in reasons[:3]:
            bucket_key  = (r.get("bucket_key") or "").strip().lower() or None
            bucket_open = (r.get("bucket_open") or "").strip() or None
            stance_lab  = (r.get("stance_label") or "").strip().lower()
            if stance_lab not in {"for","against"}:
                stance_lab = None
            rationale   = (r.get("rationale") or "").strip()

            # 1–2 quotes per reason
            for q in (r.get("quotes") or [])[:2]:
                quote = str(q or "").strip()
                if not (40 <= len(quote) <= 140):
                    continue
                start = target_text.find(quote)
                if start < 0:
                    # best-effort normalized search (skip if not found)
                    continue
                end = start + len(quote)
                args.append(
                    ExtractedArgument(
                        quote=quote,
                        span={"start": start, "end": end},
                        free_text_reason=rationale,
                        stance_label=stance_lab,
                        bucket_key=bucket_key,
                        bucket_open=bucket_open,
                    )
                )
        ALLOWED_STANCES = {"for", "against", "both", "neutral", "irrelevant"}
        return ExtractionResult(stance=stance if stance in ALLOWED_STANCES else "irrelevant",
                                arguments=args,
                                confidence=confidence,
                                raw=data)


# ==============================
# Debate context helpers
# ==============================

def add_turn_index(df: pd.DataFrame) -> pd.DataFrame:
    if "position" not in df.columns:
        df = df.copy()
        df["position"] = None

    def _assign(group: pd.DataFrame) -> pd.DataFrame:
        if group["position"].notna().any():
            group = group.sort_values(by=["position"], kind="stable")
        group = group.reset_index(drop=True)
        group["turn_idx"] = range(len(group))
        return group

    return df.groupby("debate_id", group_keys=False).apply(_assign)


def build_context(df_debate: pd.DataFrame, idx: int, window: int = 5) -> str:
    lo = max(0, idx - window)
    hi = min(len(df_debate), idx + window + 1)
    snippets = []
    for j in range(lo, hi):
        if j == idx:
            continue
        r = df_debate.iloc[j]
        spk = str(r.get("speaker", "UNKNOWN"))
        txt = str(r.get("text", ""))
        snippets.append(f"[{j}] {spk}: {txt}")
    return "\\n".join(snippets)


# ==============================
# Verifier & De-dup
# ==============================

def verify_arguments(args: List[ExtractedArgument], target_text: str) -> List[ExtractedArgument]:
    verified: List[ExtractedArgument] = []
    n = len(target_text)

    for a in args:
        s = int(a.span.get("start", -1))
        e = int(a.span.get("end", -1))
        if not (0 <= s < e <= n):
            continue
        sub = target_text[s:e]
        if normalize_ws(sub) != normalize_ws(a.quote):
            sub_trim = sub.strip(' "\'…—-;:')
            quote_trim = a.quote.strip(' "\'…—-;:')
            if normalize_ws(sub_trim) != normalize_ws(quote_trim):
                continue
        if len(sub.strip()) < 8:
            continue
        a.quote = sub.strip()
        verified.append(a)
    return verified


def iou(span1: Tuple[int,int], span2: Tuple[int,int]) -> float:
    a1, a2 = span1
    b1, b2 = span2
    inter = max(0, min(a2, b2) - max(a1, b1))
    union = max(a2, b2) - min(b1, a1)
    return inter / union if union > 0 else 0.0


def dedup_overlaps(args: List[ExtractedArgument], iou_thresh: float = 0.6) -> List[ExtractedArgument]:
    kept: List[ExtractedArgument] = []
    for a in args:
        keep = True
        for b in kept:
            if iou((a.span["start"], a.span["end"]), (b.span["start"], b.span["end"])) >= iou_thresh:
                len_a = a.span["end"] - a.span["start"]
                len_b = b.span["end"] - b.span["start"]
                if len_a >= len_b:
                    keep = False
                else:
                    kept.remove(b)
                break
        if keep:
            kept.append(a)
    return kept


# ==============================
# Core runner (parallelized LLM calls)
# ==============================

from typing import Optional

ALLOWED_STANCES = {"for", "against", "both", "neutral", "irrelevant"}

def choose_client(provider: str, prompt_text: str, model: Optional[str]) -> LLMClient:
    if provider and provider.lower() != "openrouter":
        raise ValueError("Only 'openrouter' is supported in this build.")
    return OpenRouterClient(prompt_text, model=model or "openrouter/auto")


# concurry worker that wraps the client call
if HAS_CONCURRY:
    class LLMWorker(Worker):
        def __init__(self, provider: str, prompt_text: str, model: Optional[str], window: int):
            self.provider = provider
            self.prompt_text = prompt_text
            self.model = model
            self.window = window
            self.client = choose_client(provider, prompt_text, model)

        def extract(self, target_text: str, local_context: str) -> dict:
            res: ExtractionResult = self.client.extract(
                target_text=target_text,
                local_context=local_context
            )
            out = {
                "stance": res.stance if res.stance in ALLOWED_STANCES else "irrelevant",
                "arguments": [a.as_dict() for a in res.arguments],
                "confidence": float(res.confidence or 0.0),
                "raw": res.raw or {},
            }
            # # Optional: enforce no quotes for neutral/irrelevant
            # if out["stance"] in {"neutral", "irrelevant"}:
            #     out["arguments"] = []
            return out



def run_pipeline_on_df(
    df: pd.DataFrame,
    prompt_text: str,
    prompt_version: str,
    provider: str = "openrouter",
    model: Optional[str] = None,
    window: int = 5,
    cache: Optional[shelve.Shelf] = None,
    concurrency: int = 16,
    calls_per_minute: Optional[int] = None,
) -> pd.DataFrame:
    required = {"speech_id","debate_id","text","position","year","chamber"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["chamber_norm"] = df["chamber"].map(normalize_chamber)
    df["era"] = df["year"].map(lambda y: assign_era(int(y)) if pd.notna(y) else "1959-2005")

    df = add_turn_index(df)

    run_id = now_run_id()
    rows_out: List[Dict[str, Any]] = []

    # If concurry is available AND using a real LLM, parallelize those calls.
    use_parallel = HAS_CONCURRY and concurrency and concurrency > 1

    if use_parallel:
        limits = []
        if calls_per_minute:
            limits.append(CallLimit(window_seconds=60, capacity=int(calls_per_minute)))
        # Initialize a pool of workers
        pool_opts = dict(mode="thread", max_workers=int(concurrency))
        if limits:
            pool_opts["limits"] = limits
        # add retries on network errors
        pool_opts.update(dict(num_retries=3, retry_algorithm="exponential"))

        # Context manager to ensure cleanup
        from contextlib import ExitStack
        with ExitStack() as stack:
            pool = LLMWorker.options(**pool_opts).init(provider, prompt_text, model, window)
            stack.enter_context(pool)

            futures_bundle: List[Tuple[str, int, int, Any]] = []  # (debate_id, local turn_idx, global_turn_idx, future)
            global_counter = 0

            # Loop debates to construct contexts and submit tasks
            for debate_id, g in df.groupby("debate_id", sort=False):
                g = g.sort_values("turn_idx").reset_index(drop=True)
                for idx, row in g.iterrows():
                    target_text = str(row.get("text", ""))
                    local_context = build_context(g, idx, window=window)

                    key = sha1(json.dumps({
                        "debate_id": str(debate_id),
                        "turn_idx": int(row["turn_idx"]),
                        "prompt_version": prompt_version,
                        "window": window,
                        "text_sha1": sha1(target_text),
                    }, sort_keys=True))

                    if cache is not None and key in cache:
                        res = cache[key]
                        provenance = {
                            "pipeline": "LLM_per_turn",
                            "prompt_version": prompt_version,
                            "retrieval": f"local_context±{window}",
                            "model": model or "openrouter/auto",
                            "provider": "openrouter",
                            "run_id": run_id,
                            "window": window,
                        }
                        rows_out.append({
                            "speech_id": row.get("speech_id"),
                            "debate_id": row.get("debate_id"),
                            "date": row.get("date"),
                            "year": int(row["year"]) if pd.notna(row["year"]) else None,
                            "decade": row.get("decade"),
                            "speaker": row.get("speaker"),
                            "gender": row.get("gender"),
                            "chamber": row.get("chamber"),
                            "chamber_norm": row.get("chamber_norm"),
                            "title": row.get("title"),
                            "hansard_reference": row.get("hansard_reference"),
                            "reference_volume": row.get("reference_volume"),
                            "reference_columns": row.get("reference_columns"),
                            "turn_idx": int(row["turn_idx"]),
                            "target_len": len(target_text or ""),
                            "era": row.get("era"),
                            "stance": res["stance"],
                            "arguments": res["arguments"],
                            "n_arguments": len(res["arguments"]),
                            "confidence": res["confidence"],
                            "provenance": provenance,
                        })
                        continue

                    fut = pool.extract(target_text, local_context)
                    futures_bundle.append((str(debate_id), int(row["turn_idx"]), global_counter, (idx, row, target_text, key)))
                    # Attach future result separately to preserve mapping
                    futures_bundle[-1] = futures_bundle[-1] + (fut,)
                    global_counter += 1

            # Collect results in submission order
            for debate_id, t_idx, g_idx, payload, fut in futures_bundle:
                idx, row, target_text, key = payload
                res = fut.result()  # blocks per-future; concurry schedules/threadpool
                # Verify & de-dup
                args_objs = [
                    ExtractedArgument(
                        a["quote"],
                        a["span"],
                        a.get("free_text_reason", ""),
                        a.get("stance_label"),
                        a.get("bucket_key"),
                        a.get("bucket_open"),
                    )
                    for a in res.get("arguments", [])
                ]
                verified = verify_arguments(args_objs, target_text)
                verified = dedup_overlaps(verified, iou_thresh=0.6)
                out_res = {
                    "stance": res.get("stance", "neutral"),
                    "arguments": [a.as_dict() for a in verified],
                    "confidence": float(res.get("confidence", 0.0)),
                    "raw": res.get("raw", {}),
                }
                if cache is not None:
                    cache[key] = out_res

                provenance = {
                    "pipeline": "LLM_per_turn",
                    "prompt_version": prompt_version,
                    "retrieval": f"local_context±{window}",
                    "model": model or "openrouter/auto",
                    "provider": "openrouter",
                    "run_id": run_id,
                    "window": window,
                }

                rows_out.append({
                    "speech_id": row.get("speech_id"),
                    "debate_id": row.get("debate_id"),
                    "date": row.get("date"),
                    "year": int(row["year"]) if pd.notna(row["year"]) else None,
                    "decade": row.get("decade"),
                    "speaker": row.get("speaker"),
                    "gender": row.get("gender"),
                    "chamber": row.get("chamber"),
                    "chamber_norm": row.get("chamber_norm"),
                    "title": row.get("title"),
                    "hansard_reference": row.get("hansard_reference"),
                    "reference_volume": row.get("reference_volume"),
                    "reference_columns": row.get("reference_columns"),
                    "turn_idx": int(row["turn_idx"]),
                    "target_len": len(target_text or ""),
                    "era": row.get("era"),
                    "stance": out_res["stance"],
                    "arguments": out_res["arguments"],
                    "n_arguments": len(out_res["arguments"]),
                    "confidence": out_res["confidence"],
                    "provenance": provenance,
                })

    else:
        # Sequential (or heuristic provider)
        client = choose_client(provider, prompt_text, model)
        for debate_id, g in df.groupby("debate_id", sort=False):
            g = g.sort_values("turn_idx").reset_index(drop=True)
            for idx, row in g.iterrows():
                target_text = str(row.get("text", ""))
                local_context = build_context(g, idx, window=window)

                key = sha1(json.dumps({
                    "debate_id": str(debate_id),
                    "turn_idx": int(row["turn_idx"]),
                    "prompt_version": prompt_version,
                    "window": window,
                    "text_sha1": sha1(target_text),
                }, sort_keys=True))

                if cache is not None and key in cache:
                    res = cache[key]
                else:
                    result: ExtractionResult = client.extract(target_text=target_text, local_context=local_context)
                    verified = verify_arguments(result.arguments, target_text)
                    verified = dedup_overlaps(verified, iou_thresh=0.6)
                    ALLOWED_STANCES = {"for", "against", "both", "neutral", "irrelevant"}
                    res = {
                        "stance": result.stance if result.stance in ALLOWED_STANCES else "irrelevant",
                        "arguments": [a.as_dict() for a in verified],
                        "confidence": float(result.confidence or 0.0),
                        "raw": result.raw or {},
                    }
                    if cache is not None:
                        cache[key] = res

                provenance = {
                    "pipeline": "LLM_per_turn",
                    "prompt_version": prompt_version,
                    "retrieval": f"local_context±{window}",
                    "model": getattr(client, "model", client.provider_name),
                    "provider": client.provider_name,
                    "run_id": run_id,
                    "window": window,
                }

                rows_out.append({
                    "speech_id": row.get("speech_id"),
                    "debate_id": row.get("debate_id"),
                    "date": row.get("date"),
                    "year": int(row["year"]) if pd.notna(row["year"]) else None,
                    "decade": row.get("decade"),
                    "speaker": row.get("speaker"),
                    "gender": row.get("gender"),
                    "chamber": row.get("chamber"),
                    "chamber_norm": row.get("chamber_norm"),
                    "title": row.get("title"),
                    "hansard_reference": row.get("hansard_reference"),
                    "reference_volume": row.get("reference_volume"),
                    "reference_columns": row.get("reference_columns"),
                    "turn_idx": int(row["turn_idx"]),
                    "target_len": len(target_text or ""),
                    "era": row.get("era"),
                    "stance": res["stance"],
                    "arguments": res["arguments"],
                    "n_arguments": len(res["arguments"]),
                    "confidence": res["confidence"],
                    "provenance": provenance,
                })

    return pd.DataFrame(rows_out)


# ==============================
# IO helpers
# ==============================

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_PYARROW:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path)
    else:
        df2 = df.copy()
        for col in ["arguments","provenance"]:
            if col in df2.columns:
                df2[col] = df2[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        df2.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ==============================
# Year-level entry
# ==============================

def resolve_data_dirs():
    data_root = None
    if Paths is not None:
        if hasattr(Paths, "DATA_DIR"):
            data_root = getattr(Paths, "DATA_DIR")
        else:
            try:
                data_root = Paths().DATA_DIR
            except Exception:
                pass
    if data_root is None:
        data_root = Path.cwd() / "src" / "hansard" / "data"
    input_dir = Path(data_root) / "derived" / "speeches_full"
    output_dir = Path(data_root) / "analysis_data" / "arg_mining"
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def process_year(
    year: int,
    input_dir: Path,
    output_dir: Path,
    prompt_path: Optional[Path] = None,
    provider: str = "heuristic",
    model: Optional[str] = None,
    window: int = 5,
    cache_path: Optional[Path] = None,
    force: bool = False,
    concurrency: int = 16,
    calls_per_minute: Optional[int] = None,
) -> int:
    """
    Process a single year:
      input_file  = input_dir  / f"speeches_{year}.parquet"
      output_file = output_dir / f"args_{year}.parquet"   (consolidated)
      Also writes stratified copies under era/chamber subfolders.
    """
    input_file = input_dir / f"speeches_{year}.parquet"
    output_file = output_dir / f"args_{year}.parquet"

    if not input_file.exists():
        print(f"  {year}: No input file found")
        return 0

    prompt_text, prompt_version = "", "v0.0"
    if prompt_path and prompt_path.exists():
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        prompt_version = extract_prompt_version(prompt_path)

    # Default cache under output_dir/cache
    if cache_path is None:
        (output_dir / "cache").mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / "cache" / "cache_a_open.shelve"

    df = read_parquet(input_file)

    with shelve.open(str(cache_path)) as cache:
        out_df = run_pipeline_on_df(
            df=df,
            prompt_text=prompt_text,
            prompt_version=prompt_version,
            provider=provider,
            model=model,
            window=window,
            cache=cache,
            concurrency=concurrency,
            calls_per_minute=calls_per_minute,
        )

    if out_df.empty:
        print(f"  {year}: No rows labeled.")
        return 0

    # Consolidated per-year
    write_parquet(out_df, output_file)

    # Stratified copies
    out_df["chamber_norm"] = out_df["chamber_norm"].fillna("unknown")
    era = assign_era(int(year))
    for chamber in ["commons", "lords", "unknown"]:
        df_sub = out_df[out_df["chamber_norm"] == chamber].copy()
        if df_sub.empty:
            continue
        sub_path = output_dir / f"era={era}" / f"chamber={chamber}" / f"args_{year}.parquet"
        write_parquet(df_sub, sub_path)

    print(f"  {year}: Wrote {output_file.name} and stratified era={era}/chamber=*.")
    return len(out_df)


# ==============================
# CLI (no path args)
# ==============================

def parse_years_arg(arg: Optional[str], input_dir: Path) -> List[int]:
    if arg:
        parts = [p.strip() for p in arg.split(",") if p.strip()]
        years: List[int] = []
        for p in parts:
            if "-" in p:
                a, b = p.split("-", 1)
                years.extend(list(range(int(a), int(b) + 1)))
            else:
                years.append(int(p))
        return sorted(set(years))
    # Auto-discover: speeches_YYYY.parquet
    years = []
    for p in sorted(input_dir.glob("speeches_*.parquet")):
        m = re.search(r"speeches_(\\d{4})\\.parquet$", p.name)
        if m:
            years.append(int(m.group(1)))
    return sorted(set(years))


def main():
    ap = argparse.ArgumentParser(description="A‑Open pipeline (Paths IO + OpenRouter + concurry).")
    ap.add_argument("--years", type=str, default=None, help="Comma/range, e.g., 1908,1909,1910-1912")
    ap.add_argument("--prompt", type=Path, default=None, help="Optional prompt file")
    ap.add_argument("--provider", choices=["openrouter"], default="openrouter", help="LLM provider (OpenRouter)")
    ap.add_argument("--model", type=str, default=None, help="Provider model name (OpenRouter model id)")
    ap.add_argument("--window", type=int, default=5, help="±K turns of local context")
    ap.add_argument("--cache", type=Path, default=None, help="Cache DB path (option al)")
    ap.add_argument("--force", action="store_true", help="(reserved) Overwrite existing outputs")
    ap.add_argument("--concurrency", type=int, default=16, help="Max concurrent LLM calls when provider=openrouter")
    ap.add_argument("--calls-per-minute", type=int, default=None, help="Optional rate limit for provider=openrouter")
    args = ap.parse_args()

    input_dir, output_dir = resolve_data_dirs()
    years = parse_years_arg(args.years, input_dir)
    if not years:
        print(f"No input files found in {input_dir}. Expected speeches_YYYY.parquet")
        sys.exit(0)

    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Years     : {years}")
    print(f"Provider  : OpenRouter")

    total = 0
    for y in years:
        total += process_year(
            year=y,
            input_dir=input_dir,
            output_dir=output_dir,
            prompt_path=args.prompt,
            provider=args.provider,
            model=args.model,
            window=args.window,
            cache_path=args.cache,
            force=args.force,
            concurrency=args.concurrency,
            calls_per_minute=args.calls_per_minute,
        )
    print(f"Done. Total labeled turns across years: {total}")


if __name__ == "__main__":
    main()
