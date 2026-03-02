"""
Blind annotation app for Hansard suffrage speech classification.

Annotators read the original speech and surrounding debate context, then
independently classify stance and gender bias dimensions using a validated
3-axis taxonomy (Ambivalent Sexism Theory, Stereotype Content Model,
Gender Norm Type). LLM output is hidden; agreement computed post-hoc.

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

STANCE_OPTIONS = ["for", "against", "both", "neutral", "irrelevant"]

# All 100 speeches annotated by BOTH annotators for full inter-annotator agreement
IAA_OVERLAP_COUNT = 100

# ---- 3-Axis Sexism Taxonomy (Glick & Fiske 1996, Fiske et al 2002,
#      Prentice & Carranza 2002) ----

# Axis A: Ambivalent Sexism Theory
AST_OPTIONS = {
    "hs_dominative": "Hostile: Dominative Paternalism -- controlling, women need male authority",
    "hs_competitive": "Hostile: Competitive Gender Diff -- women lack competence vs men",
    "hs_heterosexual": "Hostile: Heterosexual Hostility -- women as manipulative/deceptive",
    "bs_protective": "Benevolent: Protective Paternalism -- women need protection/shielding",
    "bs_complementary": "Benevolent: Complementary Gender Diff -- women have purity/moral virtue",
    "bs_intimacy": "Benevolent: Heterosexual Intimacy -- women complete men, romantic idealization",
    "none_ast": "No sexism detected on this axis",
}

# Axis B: Stereotype Content Model (Fiske et al 2002)
# Classifies stereotypes along two dimensions (warmth x competence)
# producing four quadrants of prejudice
SCM_OPTIONS = {
    "hw_lc": "High Warmth, Low Competence -- \"women are kind but helpless\"; pity, protection",
    "hw_hc": "High Warmth, High Competence -- \"women are capable and good\"; admiration, respect",
    "lw_lc": "Low Warmth, Low Competence -- \"women are incompetent and a burden\"; contempt, disgust",
    "lw_hc": "Low Warmth, High Competence -- \"women are capable but threatening\"; envy, resentment",
    "none_scm": "No warmth/competence claims detected",
}

# Axis C: Gender Norm Type
NORM_OPTIONS = {
    "descriptive": "Descriptive -- what women ARE like (\"women are emotional\")",
    "prescriptive": "Prescriptive -- what women SHOULD be/do (\"women should stay home\")",
    "proscriptive": "Proscriptive -- what women should NOT be/do (\"women should not vote\")",
    "none_norm": "No gender norm claims detected",
}


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
    """Append-or-update annotation for this annotator (atomic write)."""
    existing = load_annotations(annotator)
    existing[record["speech_id"]] = record
    path = ANNOTATIONS_DIR / f"{annotator}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for rec in existing.values():
            f.write(json.dumps(rec, default=str) + "\n")
    tmp.replace(path)


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

def render_speech(row, speech_idx: int, total: int, already_done: bool,
                  n_done: int = 0):
    """Left column -- the speech text and surrounding context."""
    status = " (done)" if already_done else ""
    st.subheader(f"Speech {speech_idx + 1} / {total}{status} -- {n_done} annotated")

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

    # Surrounding context
    context = row.get("context_text", "")
    if context and not pd.isna(context):
        with st.expander("Show surrounding debate context"):
            st.markdown(
                f'<div style="color: #666; font-size: 0.9em;">{context}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# UI: annotation form (3-axis taxonomy)
# ---------------------------------------------------------------------------

def _get_defaults(defaults: dict | None, key: str, fallback):
    """Safely get a default value from previous annotation."""
    if defaults and key in defaults:
        return defaults[key]
    return fallback


def render_form(row, defaults: dict | None) -> dict | None:
    """Right column -- blind annotation form with 3-axis sexism taxonomy."""
    st.subheader("Your Classification")

    # Use speech_id as key prefix so widgets reset between speeches
    sid = row["speech_id"]

    # --- 1. Stance ---
    st.markdown("**1. Stance on women's suffrage / representation?**")

    default_idx = 0
    if defaults and defaults.get("human_stance") in STANCE_OPTIONS:
        default_idx = STANCE_OPTIONS.index(defaults["human_stance"])

    human_stance = st.radio(
        "Stance",
        STANCE_OPTIONS,
        index=default_idx,
        horizontal=True,
        label_visibility="collapsed",
        key=f"stance_{sid}",
    )

    # --- 2. Axis A: Ambivalent Sexism Theory ---
    st.markdown("**2. Ambivalent Sexism** (Glick & Fiske 1996)")
    st.caption("How does the speech position women? Select all that apply.")

    prev_ast = set(_get_defaults(defaults, "ast_labels", []))
    selected_ast = []
    for key, desc in AST_OPTIONS.items():
        checked = key in prev_ast
        if st.checkbox(desc, value=checked, key=f"ast_{sid}_{key}"):
            selected_ast.append(key)

    # --- 3. Axis B: Stereotype Content Model ---
    st.markdown("**3. Stereotype Content** (Fiske et al 2002)")
    st.caption("What trait claims are made about women? Select all that apply.")

    prev_scm = set(_get_defaults(defaults, "scm_labels", []))
    selected_scm = []
    for key, desc in SCM_OPTIONS.items():
        checked = key in prev_scm
        if st.checkbox(desc, value=checked, key=f"scm_{sid}_{key}"):
            selected_scm.append(key)

    # --- 4. Axis C: Gender Norm Type ---
    st.markdown("**4. Norm Type** (Prentice & Carranza 2002)")
    st.caption("Is, should, or should-not? Select all that apply.")

    prev_norm = set(_get_defaults(defaults, "norm_labels", []))
    selected_norm = []
    for key, desc in NORM_OPTIONS.items():
        checked = key in prev_norm
        if st.checkbox(desc, value=checked, key=f"norm_{sid}_{key}"):
            selected_norm.append(key)

    # --- 5. LLM bucket verification (shown AFTER blind annotation above) ---
    llm_reasons = extract_reasons(row)
    llm_buckets = [r.get("bucket_key") for r in llm_reasons if r.get("bucket_key")]
    bucket_verdicts = {}

    if llm_buckets:
        st.markdown("---")
        st.markdown("**5. Verify LLM's argument buckets**")
        st.caption("The LLM assigned these categories. Are they correct?")

        prev_verdicts = _get_defaults(defaults, "bucket_verdicts", {})

        for ri, reason in enumerate(llm_reasons):
            bk = reason.get("bucket_key", "unknown")
            rationale = reason.get("rationale", "")
            stance_label = reason.get("stance_label", "?")
            label = bk.replace("_", " ").title()

            with st.expander(f"{label} ({stance_label})", expanded=True):
                st.caption(rationale)
                prev_v = prev_verdicts.get(bk, "correct")
                verdict = st.radio(
                    f"Is '{label}' correct?",
                    ["correct", "partially_correct", "wrong"],
                    index=["correct", "partially_correct", "wrong"].index(prev_v)
                    if prev_v in ["correct", "partially_correct", "wrong"] else 0,
                    horizontal=True,
                    key=f"bv_{sid}_{ri}_{bk}",
                    label_visibility="collapsed",
                )
                bucket_verdicts[bk] = verdict

    # --- 6. Confidence + Notes ---
    default_conf = int(_get_defaults(defaults, "confidence", 3))
    confidence = st.slider(
        "How confident? (1 = guessing, 5 = certain)",
        1, 5, default_conf,
        key=f"conf_{sid}",
    )

    default_notes = _get_defaults(defaults, "notes", "")
    notes = st.text_area("Notes (optional)", value=default_notes, height=60,
                         key=f"notes_{sid}")

    # --- Save / Skip ---
    col_save, col_skip = st.columns(2)
    saved = False
    skipped = False
    with col_save:
        saved = st.button("Save and Next", type="primary", use_container_width=True)
    with col_skip:
        skipped = st.button("Skip", use_container_width=True)

    if saved or skipped:
        llm_stance = row.get("stance", "irrelevant")

        return {
            "speech_id": row["speech_id"],
            "skipped": skipped,
            # Human annotations (blind)
            "human_stance": human_stance if not skipped else None,
            "ast_labels": selected_ast if not skipped else [],
            "scm_labels": selected_scm if not skipped else [],
            "norm_labels": selected_norm if not skipped else [],
            # LLM bucket verification
            "bucket_verdicts": bucket_verdicts if not skipped else {},
            "confidence": confidence,
            "notes": notes,
            # LLM data (for post-hoc)
            "llm_stance": llm_stance,
            "llm_buckets": llm_buckets,
            "llm_confidence": float(row.get("confidence", 0)),
            "stance_agrees": (human_stance == llm_stance) if not skipped else None,
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
                stance_label = reason.get("stance_label", "?")
                rationale = reason.get("rationale", "")
                open_label = reason.get("bucket_open", "")

                header = bucket.replace("_", " ").title()
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

    # Initialize session state for navigation
    if "speech_idx" not in st.session_state:
        st.session_state.speech_idx = 0

    # Clamp to valid range
    max_idx = len(display) - 1
    st.session_state.speech_idx = min(st.session_state.speech_idx, max_idx)

    # Prev / Next buttons
    st.sidebar.markdown("---")
    col_prev, col_num, col_next = st.sidebar.columns([1, 2, 1])

    with col_prev:
        if st.button("Prev", use_container_width=True,
                      disabled=st.session_state.speech_idx == 0):
            st.session_state.speech_idx -= 1
            st.rerun()

    with col_next:
        if st.button("Next", use_container_width=True,
                      disabled=st.session_state.speech_idx >= max_idx):
            st.session_state.speech_idx += 1
            st.rerun()

    with col_num:
        new_idx = st.number_input(
            "Go to",
            min_value=1,
            max_value=len(display),
            value=st.session_state.speech_idx + 1,
            step=1,
            label_visibility="collapsed",
        ) - 1
        if new_idx != st.session_state.speech_idx:
            st.session_state.speech_idx = new_idx
            st.rerun()

    idx = st.session_state.speech_idx

    # Show annotation status for current speech
    current_sid = display.iloc[idx]["speech_id"]
    if current_sid in annotated_ids:
        st.sidebar.success(f"#{idx + 1} already annotated")
    else:
        st.sidebar.caption(f"#{idx + 1} not yet annotated")

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

    # 3-axis distribution
    st.subheader("Taxonomy Distribution")
    for name, recs in all_annotators.items():
        real = {k: v for k, v in recs.items() if not v.get("skipped")}
        if not real:
            continue
        st.markdown(f"**{name}** ({len(real)} annotations)")

        # AST counts
        ast_counts = {}
        scm_counts = {}
        norm_counts = {}
        for r in real.values():
            for label in r.get("ast_labels", []):
                ast_counts[label] = ast_counts.get(label, 0) + 1
            for label in r.get("scm_labels", []):
                scm_counts[label] = scm_counts.get(label, 0) + 1
            for label in r.get("norm_labels", []):
                norm_counts[label] = norm_counts.get(label, 0) + 1

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.caption("Axis A: Ambivalent Sexism")
            for k, v in sorted(ast_counts.items(), key=lambda x: -x[1]):
                st.caption(f"  {k}: {v}")
        with col_b:
            st.caption("Axis B: Stereotype Content")
            for k, v in sorted(scm_counts.items(), key=lambda x: -x[1]):
                st.caption(f"  {k}: {v}")
        with col_c:
            st.caption("Axis C: Norm Type")
            for k, v in sorted(norm_counts.items(), key=lambda x: -x[1]):
                st.caption(f"  {k}: {v}")

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

    # Export: downloadable CSV
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
                "ast_labels": ", ".join(r.get("ast_labels", [])),
                "scm_labels": ", ".join(r.get("scm_labels", [])),
                "norm_labels": ", ".join(r.get("norm_labels", [])),
                "llm_buckets": ", ".join(r.get("llm_buckets", [])),
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
            "1. **Stance** on women's suffrage/representation\n"
            "2. **Gender bias dimensions** using a 3-axis taxonomy\n\n"
            "You will **not** see the LLM's classification. "
            "Agreement is computed after annotation."
        )
        st.markdown("---")
        st.markdown("**Axis A -- Ambivalent Sexism Theory** (Glick & Fiske 1996)")
        st.markdown("Hostile Sexism: degrading, controlling, denigrating women's competence")
        st.markdown("Benevolent Sexism: idealizing but restricting women (purity, protection)")
        st.markdown("")
        st.markdown("**Axis B -- Stereotype Content Model** (Fiske et al 2002)")
        st.markdown("Competence claims (high/low) and Warmth claims (high/low)")
        st.markdown("")
        st.markdown("**Axis C -- Gender Norm Type** (Prentice & Carranza 2002)")
        st.markdown("Descriptive (women ARE), Prescriptive (women SHOULD), Proscriptive (women SHOULD NOT)")
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

    n_done = len([s for s in display["speech_id"] if s in annotations])

    with col_read:
        render_speech(row, idx, len(display), already_done, n_done)

    with col_form:
        result = render_form(row, defaults)
        if result is not None:
            result["annotator"] = annotator
            save_annotation(annotator, result)
            label = "Skipped" if result["skipped"] else "Saved"
            st.toast(f"{label}! ({speech_id[:12]}...)")
            # Auto-advance to next speech
            if st.session_state.speech_idx < len(display) - 1:
                st.session_state.speech_idx += 1
            st.rerun()

    # LLM reveal -- hidden by default, clickable after annotating
    render_llm_reveal(row)


if __name__ == "__main__":
    main()
