"""
Blind annotation app for Hansard suffrage speech classification.

Annotators read the original speech and surrounding debate context, then
independently classify stance and argument themes WITHOUT seeing the LLM's
output. Agreement with the LLM is computed post-hoc.

Usage:
    streamlit run scripts/experiments/validation_app.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VALIDATION_DATA = "outputs/validation/validation_sample.parquet"
ANNOTATIONS_DIR = Path("outputs/validation/annotations")
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

# The 9 argument buckets used by the LLM classifier.
# Annotators select which themes they observe independently.
BUCKET_OPTIONS = {
    "equality": "Equality -- arguments about equal rights, fairness, representation",
    "competence_capacity": "Competence / Capacity -- arguments about ability, fitness, intelligence",
    "emotion_morality": "Emotion / Morality -- appeals to compassion, morality, sentiment",
    "instrumental_effects": "Instrumental Effects -- practical consequences of policy",
    "religion_family": "Religion / Family -- religious duty, family roles, domesticity",
    "social_order_stability": "Social Order / Stability -- maintaining order, risk of disruption",
    "social_experiment": "Social Experiment -- untested change, slippery slope",
    "tradition_precedent": "Tradition / Precedent -- historical practice, constitutional precedent",
    "other": "Other -- does not fit the categories above",
}

STANCE_OPTIONS = ["for", "against", "both", "neutral", "irrelevant"]

# First N speeches annotated by ALL annotators for inter-annotator agreement
IAA_OVERLAP_COUNT = 30


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(VALIDATION_DATA)
    df = df.sort_values("validation_index").reset_index(drop=True)
    return df


def load_annotations(annotator: str) -> dict:
    """Load annotations as {speech_id: record_dict}."""
    path = ANNOTATIONS_DIR / f"{annotator}.jsonl"
    records = {}
    if path.exists():
        for line in path.read_text().strip().split("\n"):
            if line:
                rec = json.loads(line)
                records[rec["speech_id"]] = rec
    return records


def save_annotation(annotator: str, record: dict):
    """Append-or-update annotation for this annotator."""
    existing = load_annotations(annotator)
    existing[record["speech_id"]] = record
    path = ANNOTATIONS_DIR / f"{annotator}.jsonl"
    with open(path, "w") as f:
        for rec in existing.values():
            f.write(json.dumps(rec, default=str) + "\n")


def get_all_annotators() -> dict[str, dict]:
    """Return {annotator_name: {speech_id: record}} for all annotators."""
    result = {}
    for path in ANNOTATIONS_DIR.glob("*.jsonl"):
        name = path.stem
        result[name] = load_annotations(name)
    return result


def extract_reasons(row) -> list[dict]:
    """Safely extract the reasons array from a row."""
    reasons = row.get("reasons")
    if isinstance(reasons, np.ndarray):
        return [r for r in reasons if isinstance(r, dict)]
    if isinstance(reasons, list):
        return [r for r in reasons if isinstance(r, dict)]
    return []


# ---------------------------------------------------------------------------
# UI: speech reading pane
# ---------------------------------------------------------------------------

def render_speech(row, speech_idx: int, total: int, already_done: bool):
    """Left column -- the speech text and surrounding context."""
    tag = " -- already annotated" if already_done else ""
    st.subheader(f"Speech {speech_idx + 1} / {total}{tag}")

    # Metadata
    year = row.get("year")
    wc = row.get("word_count")
    meta_parts = [
        f"**{row.get('speaker', '?')}**",
        f"{int(year) if pd.notna(year) else '?'}",
        row.get("chamber", "?"),
        f"{int(wc) if pd.notna(wc) else '?'} words",
    ]
    st.caption(" | ".join(meta_parts))

    # Main speech text
    target = row.get("target_text", "")
    if pd.isna(target) or not target:
        st.error("No speech text available.")
        return

    st.markdown("---")
    st.markdown(target)
    st.markdown("---")

    # Surrounding context -- render as markdown, not a nested text_area
    context = row.get("context_text", "")
    if context and not pd.isna(context):
        with st.expander("Show surrounding debate context"):
            st.markdown(
                f'<div style="color: #666; font-size: 0.9em;">{context}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# UI: annotation form
# ---------------------------------------------------------------------------

def render_form(row, defaults: dict | None) -> dict | None:
    """Right column -- blind annotation form. Returns record or None."""
    st.subheader("Your Classification")

    # --- 1. Stance ---
    st.markdown("**What is this speech's stance on women's suffrage / representation?**")

    # Do NOT pre-fill from LLM -- start at first option unless re-editing
    default_idx = 0
    if defaults and defaults.get("human_stance") in STANCE_OPTIONS:
        default_idx = STANCE_OPTIONS.index(defaults["human_stance"])

    human_stance = st.radio(
        "Stance",
        STANCE_OPTIONS,
        index=default_idx,
        horizontal=True,
        label_visibility="collapsed",
    )

    # --- 2. Argument themes (multi-select) ---
    st.markdown("**Which argument themes appear in this speech?** (select all that apply)")

    prev_buckets = set()
    if defaults and "human_buckets" in defaults:
        prev_buckets = set(defaults["human_buckets"])

    selected_buckets = []
    for key, description in BUCKET_OPTIONS.items():
        checked = key in prev_buckets
        if st.checkbox(description, value=checked, key=f"chk_{key}"):
            selected_buckets.append(key)

    # --- 3. Confidence ---
    st.markdown("**How confident are you in your classification?**")
    default_conf = int(defaults.get("confidence", 3)) if defaults else 3
    confidence = st.slider(
        "Confidence",
        1, 5, default_conf,
        help="1 = guessing, 5 = certain",
        label_visibility="collapsed",
    )

    # --- 4. Notes ---
    default_notes = defaults.get("notes", "") if defaults else ""
    notes = st.text_area("Notes (optional)", value=default_notes, height=80)

    # --- Save ---
    col_save, col_skip = st.columns(2)
    saved = False
    skipped = False
    with col_save:
        saved = st.button("Save and Next", type="primary", use_container_width=True)
    with col_skip:
        skipped = st.button("Skip", use_container_width=True)

    if saved or skipped:
        # Compute agreement with LLM (stored but not shown)
        llm_stance = row.get("stance", "irrelevant")
        llm_reasons = extract_reasons(row)
        llm_buckets = [r.get("bucket_key") for r in llm_reasons if r.get("bucket_key")]

        return {
            "speech_id": row["speech_id"],
            "skipped": skipped,
            "human_stance": human_stance if not skipped else None,
            "human_buckets": selected_buckets if not skipped else [],
            "confidence": confidence,
            "notes": notes,
            # LLM data for post-hoc comparison (hidden from annotator)
            "llm_stance": llm_stance,
            "llm_buckets": llm_buckets,
            "llm_confidence": float(row.get("confidence", 0)),
            "stance_agrees": (human_stance == llm_stance) if not skipped else None,
            "bucket_overlap": len(set(selected_buckets) & set(llm_buckets)) if not skipped else None,
            # Metadata
            "speaker": row.get("speaker"),
            "gender": row.get("gender"),
            "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else None,
            "timestamp": datetime.now().isoformat(),
        }

    return None


# ---------------------------------------------------------------------------
# UI: hidden LLM reveal (click to show)
# ---------------------------------------------------------------------------

def render_llm_reveal(row):
    """Collapsed panel showing the LLM's classification. Hidden by default."""
    reasons = extract_reasons(row)
    stance = row.get("stance", "?")
    conf = row.get("confidence", 0)
    conf_level = row.get("confidence_level", "?")
    n_reasons = len(reasons)

    with st.expander(
        f"Reveal LLM classification (stance: hidden, {n_reasons} argument buckets)",
        expanded=False,
    ):
        st.markdown(
            f"**Stance:** `{stance}` &nbsp;&nbsp; "
            f"**Confidence:** `{conf:.0%}` ({conf_level})"
        )

        top_quote = row.get("top_quote")
        if isinstance(top_quote, dict) and top_quote.get("text"):
            st.info(f'Top quote: "{top_quote["text"]}"')

        if reasons:
            for reason in reasons:
                bucket = reason.get("bucket_key", "unknown")
                label = BUCKET_OPTIONS.get(bucket, bucket).split(" -- ")[0]
                stance_label = reason.get("stance_label", "?")
                rationale = reason.get("rationale", "")
                open_label = reason.get("bucket_open", "")

                header = f"{label}"
                if open_label:
                    header += f" ({open_label})"
                header += f" -- {stance_label}"

                st.markdown(f"**{header}**")
                st.markdown(rationale)

                quotes = reason.get("quotes", [])
                if isinstance(quotes, np.ndarray):
                    quotes = list(quotes)
                for q in quotes:
                    if isinstance(q, dict):
                        src = q.get("source", "")
                        txt = q.get("text", "")
                        st.markdown(f'> "{txt}" *[{src}]*')
        else:
            st.markdown("*No argument buckets extracted.*")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(data: pd.DataFrame):
    """Returns (annotator, annotations_dict, display_df, current_idx) or Nones."""
    st.sidebar.title("Hansard Validation")

    # Page toggle
    page = st.sidebar.radio(
        "Page", ["Annotate", "Statistics"], label_visibility="collapsed",
    )

    if page == "Statistics":
        return "___stats___", None, None, None

    # Annotator login
    annotator = st.sidebar.text_input(
        "Your name", value="", max_chars=30,
        help="Lowercase, no spaces",
    )
    if not annotator or " " in annotator:
        return None, None, None, None

    annotator = annotator.strip().lower()
    annotations = load_annotations(annotator)
    annotated_ids = set(annotations.keys())

    # Progress
    n_total = len(data)
    n_done = len(annotated_ids & set(data["speech_id"]))
    st.sidebar.metric("Progress", f"{n_done} / {n_total}")
    st.sidebar.progress(n_done / n_total if n_total > 0 else 0)

    # IAA info
    iaa_ids = set(data.iloc[:IAA_OVERLAP_COUNT]["speech_id"])
    iaa_done = len(annotated_ids & iaa_ids)
    st.sidebar.caption(
        f"IAA overlap: {iaa_done}/{IAA_OVERLAP_COUNT} | "
        f"Assigned: {n_done - iaa_done}/{n_total - IAA_OVERLAP_COUNT}"
    )

    st.sidebar.markdown("---")

    # Filter
    filter_mode = st.sidebar.radio(
        "Show",
        ["Unannotated first", "All", "IAA overlap only", "Annotated only"],
    )

    if filter_mode == "Unannotated first":
        unannotated = data[~data["speech_id"].isin(annotated_ids)]
        annotated_df = data[data["speech_id"].isin(annotated_ids)]
        display = pd.concat([unannotated, annotated_df]).reset_index(drop=True)
    elif filter_mode == "IAA overlap only":
        display = data.iloc[:IAA_OVERLAP_COUNT].reset_index(drop=True)
    elif filter_mode == "Annotated only":
        display = data[data["speech_id"].isin(annotated_ids)].reset_index(drop=True)
    else:
        display = data.reset_index(drop=True)

    if len(display) == 0:
        st.sidebar.warning("No speeches match this filter.")
        return annotator, annotations, display, 0

    # Navigation
    idx = st.sidebar.number_input(
        "Speech #",
        min_value=1,
        max_value=len(display),
        value=1,
        step=1,
    ) - 1

    return annotator, annotations, display, idx


# ---------------------------------------------------------------------------
# Stats page
# ---------------------------------------------------------------------------

def render_stats_page():
    """Post-hoc comparison: human vs LLM across annotators."""
    st.header("Annotation Statistics")

    all_annotators = get_all_annotators()
    if not all_annotators:
        st.info("No annotations yet.")
        return

    data = load_data()
    iaa_ids = set(data.iloc[:IAA_OVERLAP_COUNT]["speech_id"])

    # Per-annotator summary
    rows = []
    for name, recs in all_annotators.items():
        real = {k: v for k, v in recs.items() if not v.get("skipped")}
        n = len(real)
        n_agree = sum(1 for r in real.values() if r.get("stance_agrees"))
        n_iaa = len(set(real.keys()) & iaa_ids)
        rows.append({
            "Annotator": name,
            "Annotated": n,
            "Skipped": len(recs) - n,
            "IAA done": n_iaa,
            "Stance agrees with LLM": n_agree,
            "Agreement %": f"{n_agree / n:.0%}" if n > 0 else "-",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Human-vs-LLM stance breakdown
    st.subheader("Stance Agreement Detail")
    for name, recs in all_annotators.items():
        real = {k: v for k, v in recs.items() if not v.get("skipped")}
        if not real:
            continue
        st.markdown(f"**{name}**")
        agree_by_stance = {}
        for r in real.values():
            llm = r.get("llm_stance", "?")
            human = r.get("human_stance", "?")
            key = llm
            if key not in agree_by_stance:
                agree_by_stance[key] = {"total": 0, "agree": 0}
            agree_by_stance[key]["total"] += 1
            if llm == human:
                agree_by_stance[key]["agree"] += 1
        for stance in STANCE_OPTIONS:
            if stance in agree_by_stance:
                d = agree_by_stance[stance]
                pct = d["agree"] / d["total"] if d["total"] else 0
                st.caption(
                    f"  LLM={stance}: {d['agree']}/{d['total']} agree ({pct:.0%})"
                )

    # IAA: pairwise human-human agreement
    overlap_annotators = {
        name: {k: v for k, v in recs.items() if not v.get("skipped")}
        for name, recs in all_annotators.items()
        if len(set(k for k, v in recs.items() if not v.get("skipped")) & iaa_ids) >= 5
    }

    if len(overlap_annotators) >= 2:
        st.subheader("Inter-Annotator Agreement (IAA)")
        names = sorted(overlap_annotators.keys())
        shared_ids = iaa_ids.copy()
        for recs in overlap_annotators.values():
            shared_ids &= set(recs.keys())

        if len(shared_ids) >= 5:
            st.caption(f"Based on {len(shared_ids)} shared speeches")
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    stance_agree = sum(
                        1 for sid in shared_ids
                        if overlap_annotators[a][sid].get("human_stance")
                        == overlap_annotators[b][sid].get("human_stance")
                    )
                    pct = stance_agree / len(shared_ids)
                    st.markdown(
                        f"**{a}** vs **{b}**: "
                        f"{stance_agree}/{len(shared_ids)} stance agree ({pct:.0%})"
                    )
        else:
            st.caption("Need >= 5 shared annotations for IAA.")

    # Export: downloadable CSV comparing human vs LLM
    st.subheader("Export Comparison")
    export_rows = []
    for name, recs in all_annotators.items():
        for sid, r in recs.items():
            if r.get("skipped"):
                continue
            export_rows.append({
                "speech_id": sid,
                "annotator": name,
                "human_stance": r.get("human_stance"),
                "llm_stance": r.get("llm_stance"),
                "stance_agrees": r.get("stance_agrees"),
                "human_buckets": ", ".join(r.get("human_buckets", [])),
                "llm_buckets": ", ".join(r.get("llm_buckets", [])),
                "bucket_overlap": r.get("bucket_overlap"),
                "confidence": r.get("confidence"),
                "llm_confidence": r.get("llm_confidence"),
                "speaker": r.get("speaker"),
                "year": r.get("year"),
                "gender": r.get("gender"),
                "notes": r.get("notes", ""),
            })

    if export_rows:
        export_df = pd.DataFrame(export_rows)
        csv = export_df.to_csv(index=False)
        st.download_button(
            "Download comparison CSV",
            csv,
            file_name="human_vs_llm_comparison.csv",
            mime="text/csv",
        )
        st.dataframe(export_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No annotations to export yet.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Hansard Validation",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    data = load_data()
    annotator, annotations, display, idx = render_sidebar(data)

    # Stats page
    if annotator == "___stats___":
        render_stats_page()
        return

    # Landing
    if annotator is None:
        st.title("Hansard Suffrage Classification")
        st.markdown(
            "Read each parliamentary speech and independently classify:\n\n"
            "1. **Stance** on women's suffrage/representation "
            "(for / against / both / neutral / irrelevant)\n"
            "2. **Argument themes** present in the speech\n\n"
            "You will **not** see the LLM's classification. "
            "Agreement is computed after annotation."
        )
        st.markdown("---")
        st.markdown("**Argument theme reference:**")
        for key, desc in BUCKET_OPTIONS.items():
            st.markdown(f"- {desc}")
        return

    if display is None or len(display) == 0:
        st.info("No speeches to display.")
        return

    # Main annotation view: two columns
    row = display.iloc[idx]
    speech_id = row["speech_id"]
    already_done = speech_id in annotations
    defaults = annotations.get(speech_id)

    col_read, col_form = st.columns([3, 2])

    with col_read:
        render_speech(row, idx, len(display), already_done)

    with col_form:
        result = render_form(row, defaults)
        if result is not None:
            result["annotator"] = annotator
            save_annotation(annotator, result)
            label = "Skipped" if result["skipped"] else "Saved"
            st.toast(f"{label}! ({speech_id[:12]}...)")
            st.rerun()

    # LLM reveal -- hidden by default, clickable after annotating
    render_llm_reveal(row)


if __name__ == "__main__":
    main()
