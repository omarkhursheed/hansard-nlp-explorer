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
BINARY_OPTIONS = ["sexist", "not_sexist"]

# All 100 speeches annotated by BOTH annotators for full inter-annotator agreement
IAA_OVERLAP_COUNT = 100

# ---- 3-Axis Sexism Taxonomy (aligned with prompts/sexism_classification_prompt.md) ----

# Axis A: Ambivalent Sexism Theory
AXIS_A_LABELS = ["hostile", "benevolent", "none"]
AXIS_A_LABEL_DESCRIPTIONS = {
    "hostile": "Hostile -- degrading, blaming, controlling women",
    "benevolent": "Benevolent -- idealizing women but restricting roles",
    "none": "No ambivalent sexism detected",
}
AXIS_A_SUBCATEGORIES = {
    "hostile": [
        "dominative_paternalism",
        "competitive_gender_differentiation",
        "heterosexual_hostility",
        "none",
    ],
    "benevolent": [
        "protective_paternalism",
        "complementary_gender_differentiation",
        "heterosexual_intimacy",
        "none",
    ],
    "none": ["none"],
}
AXIS_A_SUBCATEGORY_DESCRIPTIONS = {
    "dominative_paternalism": "Justifies male authority",
    "competitive_gender_differentiation": "Justifies male dominance by competence",
    "heterosexual_hostility": "Frames women's sexuality as threatening",
    "protective_paternalism": "Justifies male authority through care",
    "complementary_gender_differentiation": "Praise that confines women to complementary roles",
    "heterosexual_intimacy": "Women as essential to men's happiness via intimacy",
    "none": "No subcategory",
}

# Axis B: Stereotype Content Model
AXIS_B_LABELS = [
    "paternalistic_prejudice",
    "admiration",
    "contemptuous_prejudice",
    "envious_prejudice",
    "none",
]
AXIS_B_LABEL_DESCRIPTIONS = {
    "paternalistic_prejudice": "Warm, incompetent; pity/protection",
    "admiration": "Warm, competent; admiration/pride",
    "contemptuous_prejudice": "Cold, incompetent; contempt/disgust",
    "envious_prejudice": "Cold, competent; envy/resentment",
    "none": "No stereotype content detected",
}

# Axis C: Gender Norm Type
AXIS_C_LABELS = ["descriptive", "prescriptive", "proscriptive", "none"]
AXIS_C_LABEL_DESCRIPTIONS = {
    "descriptive": "What women are like",
    "prescriptive": "What women should do/be",
    "proscriptive": "What women should not do/be",
    "none": "No gender norm claims detected",
}

AXIS_A_SUBCATEGORY_SET = set().union(*AXIS_A_SUBCATEGORIES.values())

OLD_AST_TO_AXIS_A = {
    "hs_dominative": ("hostile", "dominative_paternalism"),
    "hs_competitive": ("hostile", "competitive_gender_differentiation"),
    "hs_heterosexual": ("hostile", "heterosexual_hostility"),
    "bs_protective": ("benevolent", "protective_paternalism"),
    "bs_complementary": ("benevolent", "complementary_gender_differentiation"),
    "bs_intimacy": ("benevolent", "heterosexual_intimacy"),
    "none_ast": ("none", "none"),
}
OLD_SCM_TO_AXIS_B = {
    "hw_lc": "paternalistic_prejudice",
    "hw_hc": "admiration",
    "lw_lc": "contemptuous_prejudice",
    "lw_hc": "envious_prejudice",
    "none_scm": "none",
}
OLD_NORM_TO_AXIS_C = {
    "descriptive": "descriptive",
    "prescriptive": "prescriptive",
    "proscriptive": "proscriptive",
    "none_norm": "none",
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
        f"ID: {row.get('speech_id', '?')}",
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


def _format_axis_a_label(label: str) -> str:
    desc = AXIS_A_LABEL_DESCRIPTIONS.get(label, "")
    return f"{label} - {desc}" if desc else label


def _format_axis_a_subcategory(label: str) -> str:
    desc = AXIS_A_SUBCATEGORY_DESCRIPTIONS.get(label, "")
    if label == "none":
        return "none - No subcategory"
    title = label.replace("_", " ").title()
    return f"{title} - {desc}" if desc else title


def _format_axis_b_label(label: str) -> str:
    desc = AXIS_B_LABEL_DESCRIPTIONS.get(label, "")
    return f"{label} - {desc}" if desc else label


def _format_axis_c_label(label: str) -> str:
    desc = AXIS_C_LABEL_DESCRIPTIONS.get(label, "")
    return f"{label} - {desc}" if desc else label


def _coerce_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _infer_axis_a_from_old(labels: list[str]) -> tuple[str, str]:
    for label in labels:
        mapped = OLD_AST_TO_AXIS_A.get(label)
        if mapped and mapped[0] != "none":
            return mapped
    for label in labels:
        mapped = OLD_AST_TO_AXIS_A.get(label)
        if mapped:
            return mapped
    return "none", "none"


def _infer_axis_b_from_old(labels: list[str]) -> str:
    for label in labels:
        mapped = OLD_SCM_TO_AXIS_B.get(label)
        if mapped and mapped != "none":
            return mapped
    for label in labels:
        mapped = OLD_SCM_TO_AXIS_B.get(label)
        if mapped:
            return mapped
    return "none"


def _infer_axis_c_from_old(labels: list[str]) -> str:
    for label in labels:
        mapped = OLD_NORM_TO_AXIS_C.get(label)
        if mapped and mapped != "none":
            return mapped
    for label in labels:
        mapped = OLD_NORM_TO_AXIS_C.get(label)
        if mapped:
            return mapped
    return "none"


def normalize_axes(defaults: dict | None) -> dict:
    """Normalize axis selections for new/legacy annotation formats."""
    axis_a_label = _get_defaults(defaults, "axis_a_label", None)
    axis_a_subcategory = _get_defaults(defaults, "axis_a_subcategory", None)
    axis_b_label = _get_defaults(defaults, "axis_b_label", None)
    axis_c_label = _get_defaults(defaults, "axis_c_label", None)

    if axis_a_label not in AXIS_A_LABELS or axis_a_subcategory not in AXIS_A_SUBCATEGORY_SET:
        old_ast = _coerce_list(_get_defaults(defaults, "ast_labels", []))
        axis_a_label, axis_a_subcategory = _infer_axis_a_from_old(old_ast)

    if axis_b_label not in AXIS_B_LABELS:
        old_scm = _coerce_list(_get_defaults(defaults, "scm_labels", []))
        axis_b_label = _infer_axis_b_from_old(old_scm)

    if axis_c_label not in AXIS_C_LABELS:
        old_norm = _coerce_list(_get_defaults(defaults, "norm_labels", []))
        axis_c_label = _infer_axis_c_from_old(old_norm)

    if axis_a_label not in AXIS_A_LABELS:
        axis_a_label = "none"
    if axis_a_subcategory not in AXIS_A_SUBCATEGORY_SET:
        if axis_a_label in ("hostile", "benevolent"):
            axis_a_subcategory = AXIS_A_SUBCATEGORIES[axis_a_label][0]
        else:
            axis_a_subcategory = "none"
    if axis_a_label == "none":
        axis_a_subcategory = "none"

    if axis_b_label not in AXIS_B_LABELS:
        axis_b_label = "none"
    if axis_c_label not in AXIS_C_LABELS:
        axis_c_label = "none"

    binary = _get_defaults(defaults, "binary", None)
    if binary not in BINARY_OPTIONS:
        binary = "sexist" if any(
            label != "none" for label in (axis_a_label, axis_b_label, axis_c_label)
        ) else "not_sexist"

    return {
        "binary": binary,
        "axis_a_label": axis_a_label,
        "axis_a_subcategory": axis_a_subcategory,
        "axis_b_label": axis_b_label,
        "axis_c_label": axis_c_label,
    }


def render_form(row, defaults: dict | None) -> dict | None:
    """Right column -- blind annotation form with 3-axis sexism taxonomy."""
    st.subheader("Your Classification")

    # Use speech_id as key prefix so widgets reset between speeches
    sid = row["speech_id"]
    defaults_norm = normalize_axes(defaults)

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

    # --- 2. Binary label ---
    st.markdown("**2. Sexism (binary)**")
    binary = st.radio(
        "Binary label",
        BINARY_OPTIONS,
        index=BINARY_OPTIONS.index(defaults_norm["binary"]),
        horizontal=True,
        key=f"binary_{sid}",
        label_visibility="collapsed",
    )

    # --- 3. Axis A: Ambivalent Sexism Theory ---
    st.markdown("**3. Axis A - Ambivalent Sexism**")
    axis_a_label = st.radio(
        "Axis A label",
        AXIS_A_LABELS,
        index=AXIS_A_LABELS.index(defaults_norm["axis_a_label"]),
        horizontal=True,
        key=f"axis_a_label_{sid}",
        format_func=_format_axis_a_label,
        label_visibility="collapsed",
    )

    axis_a_subcategory = "none"
    if axis_a_label != "none":
        st.markdown("Axis A subcategory")
        subcategories = AXIS_A_SUBCATEGORIES[axis_a_label]
        if defaults_norm["axis_a_label"] == axis_a_label and defaults_norm["axis_a_subcategory"] in subcategories:
            sub_default = defaults_norm["axis_a_subcategory"]
        else:
            sub_default = subcategories[0]
        axis_a_subcategory = st.radio(
            "Axis A subcategory",
            subcategories,
            index=subcategories.index(sub_default),
            key=f"axis_a_sub_{sid}_{axis_a_label}",
            format_func=_format_axis_a_subcategory,
            label_visibility="collapsed",
        )
    else:
        st.caption("Axis A subcategory is not applicable when label is none.")

    # --- 4. Axis B: Stereotype Content Model ---
    st.markdown("**4. Axis B - Stereotype Content**")
    axis_b_label = st.radio(
        "Axis B label",
        AXIS_B_LABELS,
        index=AXIS_B_LABELS.index(defaults_norm["axis_b_label"]),
        horizontal=True,
        key=f"axis_b_label_{sid}",
        format_func=_format_axis_b_label,
        label_visibility="collapsed",
    )

    # --- 5. Axis C: Gender Norm Type ---
    st.markdown("**5. Axis C - Norm Type**")
    axis_c_label = st.radio(
        "Axis C label",
        AXIS_C_LABELS,
        index=AXIS_C_LABELS.index(defaults_norm["axis_c_label"]),
        horizontal=True,
        key=f"axis_c_label_{sid}",
        format_func=_format_axis_c_label,
        label_visibility="collapsed",
    )

    # --- 6. LLM bucket verification (shown AFTER blind annotation above) ---
    llm_reasons = extract_reasons(row)
    llm_buckets = [r.get("bucket_key") for r in llm_reasons if r.get("bucket_key")]
    bucket_verdicts = {}

    if llm_buckets:
        st.markdown("---")
        st.markdown("**6. Verify LLM's argument buckets**")
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

    # --- 7. Confidence + Notes ---
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
        if binary == "not_sexist":
            axis_a_label = "none"
            axis_a_subcategory = "none"
            axis_b_label = "none"
            axis_c_label = "none"
        elif axis_a_label == "none":
            axis_a_subcategory = "none"

        return {
            "speech_id": row["speech_id"],
            "skipped": skipped,
            # Human annotations (blind)
            "human_stance": human_stance if not skipped else None,
            "binary": binary if not skipped else None,
            "axis_a_label": axis_a_label if not skipped else None,
            "axis_a_subcategory": axis_a_subcategory if not skipped else None,
            "axis_b_label": axis_b_label if not skipped else None,
            "axis_c_label": axis_c_label if not skipped else None,
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

        binary_counts = {}
        axis_a_counts = {}
        axis_a_sub_counts = {}
        axis_b_counts = {}
        axis_c_counts = {}
        for r in real.values():
            axes = normalize_axes(r)
            binary = axes["binary"]
            axis_a_label = axes["axis_a_label"]
            axis_a_sub = axes["axis_a_subcategory"]
            axis_b_label = axes["axis_b_label"]
            axis_c_label = axes["axis_c_label"]

            binary_counts[binary] = binary_counts.get(binary, 0) + 1
            axis_a_counts[axis_a_label] = axis_a_counts.get(axis_a_label, 0) + 1
            axis_a_sub_counts[axis_a_sub] = axis_a_sub_counts.get(axis_a_sub, 0) + 1
            axis_b_counts[axis_b_label] = axis_b_counts.get(axis_b_label, 0) + 1
            axis_c_counts[axis_c_label] = axis_c_counts.get(axis_c_label, 0) + 1

        st.caption("Binary label")
        for k, v in sorted(binary_counts.items(), key=lambda x: -x[1]):
            st.caption(f"  {k}: {v}")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.caption("Axis A label")
            for k, v in sorted(axis_a_counts.items(), key=lambda x: -x[1]):
                st.caption(f"  {k}: {v}")
            st.caption("Axis A subcategory")
            for k, v in sorted(axis_a_sub_counts.items(), key=lambda x: -x[1]):
                st.caption(f"  {k}: {v}")
        with col_b:
            st.caption("Axis B label")
            for k, v in sorted(axis_b_counts.items(), key=lambda x: -x[1]):
                st.caption(f"  {k}: {v}")
        with col_c:
            st.caption("Axis C label")
            for k, v in sorted(axis_c_counts.items(), key=lambda x: -x[1]):
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
            axes = normalize_axes(r)
            export_rows.append({
                "speech_id": sid,
                "annotator": name,
                "human_stance": r.get("human_stance"),
                "llm_stance": r.get("llm_stance"),
                "stance_agrees": r.get("stance_agrees"),
                "binary": axes["binary"],
                "axis_a_label": axes["axis_a_label"],
                "axis_a_subcategory": axes["axis_a_subcategory"],
                "axis_b_label": axes["axis_b_label"],
                "axis_c_label": axes["axis_c_label"],
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
            "2. **Binary sexism** (sexist / not_sexist)\n"
            "3. **Gender bias dimensions** using a 3-axis taxonomy\n\n"
            "You will **not** see the LLM's classification. "
            "Agreement is computed after annotation."
        )
        st.markdown("---")
        st.markdown("**Axis A -- Ambivalent Sexism Theory** (Glick & Fiske 1996)")
        st.markdown("Hostile: degrading, blaming, controlling women's competence or motives")
        st.markdown("Benevolent: idealizing women while restricting roles")
        st.markdown("")
        st.markdown("**Axis B -- Stereotype Content Model** (Fiske et al 2002)")
        st.markdown("Warmth/competence quadrants: paternalistic, admiration, contemptuous, envious")
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
