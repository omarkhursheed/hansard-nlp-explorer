"""
Sexism axis annotation app for rebuttal validation (v2).

Stance is pre-filled from gold labels (v1 agreement + resolution).
Only the 63 relevant speeches are shown (irrelevant speeches skipped).
Previous v1 axis labels are shown as reference in a collapsible panel.

Saves to experiments/20260501_rebuttal/annotations/ (won't touch v1 data).

Usage:
    streamlit run experiments/20260501_rebuttal/sexism_annotation_app.py
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


VALIDATION_DATA = "outputs/validation/validation_sample.parquet"
V1_ANNOTATIONS_DIR = Path("outputs/validation/annotations")
V2_ANNOTATIONS_DIR = Path("experiments/20260501_rebuttal/annotations")
V2_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
RESOLUTIONS_PATH = Path("outputs/validation/resolutions.json")

BINARY_OPTIONS = ["sexist", "not_sexist"]

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
    "dominative_paternalism": "Women incompetent, men must control/lead",
    "competitive_gender_differentiation": "Men more competent, women lack traits for power",
    "heterosexual_hostility": "Women's sexuality as manipulative/threatening",
    "protective_paternalism": "Women fragile, men should protect/provide",
    "complementary_gender_differentiation": "Women have purity/nurturance, confining to complementary roles",
    "heterosexual_intimacy": "Men incomplete without women romantically",
    "none": "No subcategory",
}

AXIS_B_LABELS = [
    "paternalistic_prejudice",
    "admiration",
    "contemptuous_prejudice",
    "envious_prejudice",
    "none",
]
AXIS_B_LABEL_DESCRIPTIONS = {
    "paternalistic_prejudice": "High warmth, low competence; pity/protection",
    "admiration": "High warmth, high competence; pride/approval",
    "contemptuous_prejudice": "Low warmth, low competence; contempt/disgust",
    "envious_prejudice": "Low warmth, high competence; envy/resentment",
    "none": "No stereotype content detected",
}

AXIS_C_LABELS = ["descriptive", "prescriptive", "proscriptive", "none"]
AXIS_C_LABEL_DESCRIPTIONS = {
    "descriptive": "What women are like (women ARE X)",
    "prescriptive": "What women should do/be (women SHOULD X)",
    "proscriptive": "What women should not do/be (women SHOULD NOT X)",
    "none": "No gender norm claims detected",
}

SCOPE_DEFINITION = """**Scope: Women's political rights and representation** -- the struggle
for women to participate in political life as voters, candidates, and officeholders.

**RELEVANT:** Women's right to vote, standing for/serving in Parliament,
political representation (quotas, candidate selection), equal treatment in
political contexts, arguments about women's fitness for political participation,
references to the suffrage movement, sex discrimination in political rights.

**NOT RELEVANT:** Women's social issues without a political rights frame
(health, childcare, employment), parliamentary procedure, economic/foreign
policy, a speech by a female MP that is not about women's political status.

**Key clarification (from v1 annotation):** "Representation" means political
representation broadly -- not just suffrage/voting. A speech about women serving
on committees, standing for office, or participating in political life is relevant
even if it doesn't mention voting specifically."""

OLD_AST_TO_NEW = {
    "hs_dominative": ("hostile", "dominative_paternalism"),
    "hs_competitive": ("hostile", "competitive_gender_differentiation"),
    "hs_heterosexual": ("hostile", "heterosexual_hostility"),
    "bs_protective": ("benevolent", "protective_paternalism"),
    "bs_complementary": ("benevolent", "complementary_gender_differentiation"),
    "bs_intimacy": ("benevolent", "heterosexual_intimacy"),
    "none_ast": ("none", "none"),
}
OLD_SCM_TO_NEW = {
    "hw_lc": "paternalistic_prejudice",
    "hw_hc": "admiration",
    "lw_lc": "contemptuous_prejudice",
    "lw_hc": "envious_prejudice",
    "none_scm": "none",
}
OLD_NORM_TO_NEW = {
    "descriptive": "descriptive",
    "prescriptive": "prescriptive",
    "proscriptive": "proscriptive",
    "none_norm": "none",
}


@st.cache_data
def load_gold_labels() -> dict:
    """Build gold stance labels from v1 agreements + resolutions."""
    omar = {json.loads(l)["speech_id"]: json.loads(l)
            for l in open(V1_ANNOTATIONS_DIR / "omar.jsonl")}
    mandira = {json.loads(l)["speech_id"]: json.loads(l)
               for l in open(V1_ANNOTATIONS_DIR / "mandira.jsonl")}

    def norm(s):
        return "irrelevant" if s == "neutral" else s

    omar_stance = {sid: norm(r["human_stance"]) for sid, r in omar.items()}
    mandira_stance = {sid: norm(r["human_stance"])
                      for sid, r in mandira.items() if r.get("human_stance")}

    shared = set(omar_stance) & set(mandira_stance)
    gold = {}
    for sid in shared:
        if omar_stance[sid] == mandira_stance[sid]:
            gold[sid] = omar_stance[sid]

    if RESOLUTIONS_PATH.exists():
        for sid, res in json.loads(RESOLUTIONS_PATH.read_text()).items():
            gold[sid] = res["resolved_stance"]

    return gold


@st.cache_data
def load_v1_annotations() -> dict:
    """Load v1 annotations from both annotators for reference."""
    v1 = {}
    for path in V1_ANNOTATIONS_DIR.glob("*.jsonl"):
        name = path.stem
        for line in path.read_text().strip().split("\n"):
            if line:
                rec = json.loads(line)
                sid = rec["speech_id"]
                if sid not in v1:
                    v1[sid] = {}
                v1[sid][name] = rec
    return v1


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(VALIDATION_DATA)
    df = df.sort_values("validation_index").reset_index(drop=True)

    gold = load_gold_labels()
    df["gold_stance"] = df["speech_id"].map(gold)
    relevant = df[df["gold_stance"].isin(["for", "against", "both"])].reset_index(drop=True)
    return relevant


def load_annotations(annotator: str) -> dict:
    path = V2_ANNOTATIONS_DIR / f"{annotator}.jsonl"
    records = {}
    if path.exists():
        for line in path.read_text().strip().split("\n"):
            if line:
                rec = json.loads(line)
                records[rec["speech_id"]] = rec
    return records


def save_annotation(annotator: str, record: dict):
    existing = load_annotations(annotator)
    existing[record["speech_id"]] = record
    path = V2_ANNOTATIONS_DIR / f"{annotator}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for rec in existing.values():
            f.write(json.dumps(rec, default=str) + "\n")
    tmp.replace(path)


def translate_v1_axes(rec: dict) -> dict:
    """Translate old-format axis labels to current format."""
    result = {"binary": "not_sexist", "axis_a_label": "none",
              "axis_a_subcategory": "none", "axis_b_label": "none",
              "axis_c_label": "none"}

    if rec.get("binary") in BINARY_OPTIONS:
        result["binary"] = rec["binary"]
    if rec.get("axis_a_label") in AXIS_A_LABELS:
        result["axis_a_label"] = rec["axis_a_label"]
        result["axis_a_subcategory"] = rec.get("axis_a_subcategory", "none")
        return result

    ast = rec.get("ast_labels", [])
    if isinstance(ast, str):
        ast = [ast]
    for label in ast:
        mapped = OLD_AST_TO_NEW.get(label)
        if mapped and mapped[0] != "none":
            result["axis_a_label"] = mapped[0]
            result["axis_a_subcategory"] = mapped[1]
            result["binary"] = "sexist"
            break

    scm = rec.get("scm_labels", [])
    if isinstance(scm, str):
        scm = [scm]
    for label in scm:
        mapped = OLD_SCM_TO_NEW.get(label)
        if mapped and mapped != "none":
            result["axis_b_label"] = mapped
            result["binary"] = "sexist"
            break

    norms = rec.get("norm_labels", [])
    if isinstance(norms, str):
        norms = [norms]
    for label in norms:
        mapped = OLD_NORM_TO_NEW.get(label)
        if mapped and mapped != "none":
            result["axis_c_label"] = mapped
            result["binary"] = "sexist"
            break

    return result


def render_speech(row, speech_idx: int, total: int, already_done: bool,
                  n_done: int = 0):
    status = " (done)" if already_done else ""
    st.subheader(f"Speech {speech_idx + 1} / {total}{status} -- {n_done}/{total} annotated")

    year = row.get("year")
    wc = row.get("word_count")
    gold = row.get("gold_stance", "?")
    meta_parts = [
        f"**{row.get('speaker', '?')}**",
        f"{int(year) if pd.notna(year) else '?'}",
        f"{int(wc) if pd.notna(wc) else '?'} words",
        f"Gold stance: **{gold}**",
    ]
    st.caption(" | ".join(meta_parts))

    target = row.get("target_text", "")
    if pd.isna(target) or not target:
        st.error("No speech text available.")
        return

    st.markdown("---")
    st.markdown(target)
    st.markdown("---")

    context = row.get("context_text", "")
    if context and not pd.isna(context):
        with st.expander("Show surrounding debate context"):
            st.markdown(
                f'<div style="color: #666; font-size: 0.9em;">{context}</div>',
                unsafe_allow_html=True,
            )


def render_v1_reference(speech_id: str):
    """Show v1 annotations as reference."""
    v1 = load_v1_annotations()
    v1_data = v1.get(speech_id, {})
    if not v1_data:
        return

    with st.expander("Previous annotations (v1 reference)", expanded=False):
        for name, rec in v1_data.items():
            axes = translate_v1_axes(rec)
            stance = rec.get("human_stance", "?")
            st.markdown(f"**{name}** -- stance: {stance}")
            st.caption(
                f"Binary: {axes['binary']} | "
                f"A: {axes['axis_a_label']}"
                f" ({axes['axis_a_subcategory']})" if axes["axis_a_label"] != "none" else "" +
                f" | B: {axes['axis_b_label']}"
                f" | C: {axes['axis_c_label']}"
            )
            if rec.get("notes"):
                st.caption(f"Notes: {rec['notes']}")


def _fmt_a(label):
    desc = AXIS_A_LABEL_DESCRIPTIONS.get(label, "")
    return f"{label} - {desc}" if desc else label

def _fmt_a_sub(label):
    desc = AXIS_A_SUBCATEGORY_DESCRIPTIONS.get(label, "")
    if label == "none":
        return "none"
    title = label.replace("_", " ").title()
    return f"{title} - {desc}" if desc else title

def _fmt_b(label):
    desc = AXIS_B_LABEL_DESCRIPTIONS.get(label, "")
    return f"{label} - {desc}" if desc else label

def _fmt_c(label):
    desc = AXIS_C_LABEL_DESCRIPTIONS.get(label, "")
    return f"{label} - {desc}" if desc else label


def render_form(row, defaults: dict | None) -> dict | None:
    st.subheader("Sexism Classification")
    sid = row["speech_id"]
    gold_stance = row.get("gold_stance", "?")

    st.markdown(f"Gold stance: **{gold_stance}** (from v1 agreement/resolution)")

    with st.expander("Scope definition"):
        st.markdown(SCOPE_DEFINITION)

    # 1. Binary sexism
    st.markdown("**1. Sexism (binary)**")
    default_binary = 1
    if defaults and defaults.get("binary") in BINARY_OPTIONS:
        default_binary = BINARY_OPTIONS.index(defaults["binary"])
    binary = st.radio(
        "Binary", BINARY_OPTIONS, index=default_binary,
        horizontal=True, label_visibility="collapsed", key=f"binary_{sid}",
    )

    # 2. Axis A
    st.markdown("**2. Axis A -- Ambivalent Sexism**")
    default_a = 2
    if defaults and defaults.get("axis_a_label") in AXIS_A_LABELS:
        default_a = AXIS_A_LABELS.index(defaults["axis_a_label"])
    axis_a_label = st.radio(
        "Axis A", AXIS_A_LABELS, index=default_a,
        horizontal=True, key=f"axis_a_{sid}", format_func=_fmt_a,
        label_visibility="collapsed",
    )

    axis_a_subcategory = "none"
    if axis_a_label != "none":
        subcats = AXIS_A_SUBCATEGORIES[axis_a_label]
        default_sub = 0
        if defaults and defaults.get("axis_a_subcategory") in subcats:
            default_sub = subcats.index(defaults["axis_a_subcategory"])
        axis_a_subcategory = st.radio(
            "Subcategory", subcats, index=default_sub,
            key=f"axis_a_sub_{sid}_{axis_a_label}", format_func=_fmt_a_sub,
            label_visibility="collapsed",
        )

    # 3. Axis B
    st.markdown("**3. Axis B -- Stereotype Content**")
    default_b = 4
    if defaults and defaults.get("axis_b_label") in AXIS_B_LABELS:
        default_b = AXIS_B_LABELS.index(defaults["axis_b_label"])
    axis_b_label = st.radio(
        "Axis B", AXIS_B_LABELS, index=default_b,
        horizontal=True, key=f"axis_b_{sid}", format_func=_fmt_b,
        label_visibility="collapsed",
    )

    # 4. Axis C
    st.markdown("**4. Axis C -- Gender Norm Type**")
    default_c = 3
    if defaults and defaults.get("axis_c_label") in AXIS_C_LABELS:
        default_c = AXIS_C_LABELS.index(defaults["axis_c_label"])
    axis_c_label = st.radio(
        "Axis C", AXIS_C_LABELS, index=default_c,
        horizontal=True, key=f"axis_c_{sid}", format_func=_fmt_c,
        label_visibility="collapsed",
    )

    # 5. Confidence + notes
    default_conf = int(defaults.get("confidence", 3)) if defaults else 3
    confidence = st.slider(
        "Confidence (1=guessing, 5=certain)", 1, 5, default_conf,
        key=f"conf_{sid}",
    )
    default_notes = defaults.get("notes", "") if defaults else ""
    notes = st.text_area("Notes (optional)", value=default_notes, height=60,
                         key=f"notes_{sid}")

    col_save, col_skip = st.columns(2)
    saved = col_save.button("Save and Next", type="primary", use_container_width=True)
    skipped = col_skip.button("Skip", use_container_width=True)

    if saved or skipped:
        if binary == "not_sexist":
            axis_a_label = "none"
            axis_a_subcategory = "none"
            axis_b_label = "none"
            axis_c_label = "none"
        elif axis_a_label == "none":
            axis_a_subcategory = "none"

        return {
            "speech_id": sid,
            "skipped": skipped,
            "gold_stance": gold_stance,
            "binary": binary if not skipped else None,
            "axis_a_label": axis_a_label if not skipped else None,
            "axis_a_subcategory": axis_a_subcategory if not skipped else None,
            "axis_b_label": axis_b_label if not skipped else None,
            "axis_c_label": axis_c_label if not skipped else None,
            "confidence": confidence,
            "notes": notes,
            "speaker": row.get("speaker"),
            "gender": row.get("gender"),
            "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else None,
            "timestamp": datetime.now().isoformat(),
            "annotator": "",
        }
    return None


def render_sidebar(data: pd.DataFrame):
    st.sidebar.title("Sexism Annotation v2")
    st.sidebar.caption("63 relevant speeches -- rebuttal relabeling")

    page = st.sidebar.radio("Page", ["Annotate", "Statistics"], label_visibility="collapsed")
    if page == "Statistics":
        return "___stats___", None, None, None

    annotator = st.sidebar.text_input("Your name", value="", max_chars=30)
    if not annotator or " " in annotator:
        return None, None, None, None

    annotator = annotator.strip().lower()
    annotations = load_annotations(annotator)
    annotated_ids = set(annotations.keys())

    n_total = len(data)
    n_done = len(annotated_ids & set(data["speech_id"]))
    st.sidebar.metric("Progress", f"{n_done} / {n_total}")
    st.sidebar.progress(n_done / n_total if n_total > 0 else 0)

    with st.sidebar.expander("Scope definition"):
        st.markdown(SCOPE_DEFINITION)

    st.sidebar.markdown("---")

    filter_mode = st.sidebar.radio(
        "Show", ["Unannotated first", "All", "Annotated only"],
    )
    if filter_mode == "Unannotated first":
        unannotated = data[~data["speech_id"].isin(annotated_ids)]
        annotated_df = data[data["speech_id"].isin(annotated_ids)]
        display = pd.concat([unannotated, annotated_df]).reset_index(drop=True)
    elif filter_mode == "Annotated only":
        display = data[data["speech_id"].isin(annotated_ids)].reset_index(drop=True)
    else:
        display = data.reset_index(drop=True)

    if len(display) == 0:
        st.sidebar.warning("No speeches match this filter.")
        return annotator, annotations, display, 0

    if "speech_idx" not in st.session_state:
        st.session_state.speech_idx = 0
    st.session_state.speech_idx = min(st.session_state.speech_idx, len(display) - 1)

    st.sidebar.markdown("---")
    col_prev, col_num, col_next = st.sidebar.columns([1, 2, 1])
    with col_prev:
        if st.button("Prev", use_container_width=True, disabled=st.session_state.speech_idx == 0):
            st.session_state.speech_idx -= 1
            st.rerun()
    with col_next:
        if st.button("Next", use_container_width=True, disabled=st.session_state.speech_idx >= len(display) - 1):
            st.session_state.speech_idx += 1
            st.rerun()
    with col_num:
        new_idx = st.number_input(
            "Go to", min_value=1, max_value=len(display),
            value=st.session_state.speech_idx + 1, step=1,
            label_visibility="collapsed",
        ) - 1
        if new_idx != st.session_state.speech_idx:
            st.session_state.speech_idx = new_idx
            st.rerun()

    idx = st.session_state.speech_idx
    current_sid = display.iloc[idx]["speech_id"]
    if current_sid in annotated_ids:
        st.sidebar.success(f"#{idx + 1} already annotated")
    else:
        st.sidebar.caption(f"#{idx + 1} not yet annotated")

    return annotator, annotations, display, idx


def render_stats_page():
    st.header("Annotation Statistics")

    all_annotators = {}
    for path in V2_ANNOTATIONS_DIR.glob("*.jsonl"):
        name = path.stem
        all_annotators[name] = load_annotations(name)

    if not all_annotators:
        st.info("No annotations yet.")
        return

    for name, recs in all_annotators.items():
        real = {k: v for k, v in recs.items() if not v.get("skipped")}
        st.markdown(f"**{name}** ({len(real)} annotations)")

        binary_counts = {}
        a_counts = {}
        b_counts = {}
        c_counts = {}
        for r in real.values():
            binary_counts[r.get("binary", "?")] = binary_counts.get(r.get("binary", "?"), 0) + 1
            a_counts[r.get("axis_a_label", "?")] = a_counts.get(r.get("axis_a_label", "?"), 0) + 1
            b_counts[r.get("axis_b_label", "?")] = b_counts.get(r.get("axis_b_label", "?"), 0) + 1
            c_counts[r.get("axis_c_label", "?")] = c_counts.get(r.get("axis_c_label", "?"), 0) + 1

        st.caption(f"Binary: {binary_counts}")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.caption(f"Axis A: {a_counts}")
        with col_b:
            st.caption(f"Axis B: {b_counts}")
        with col_c:
            st.caption(f"Axis C: {c_counts}")

    names = sorted(all_annotators.keys())
    if len(names) >= 2:
        st.subheader("Inter-Annotator Agreement")
        for i, a in enumerate(names):
            for b_name in names[i + 1:]:
                shared = set(all_annotators[a].keys()) & set(all_annotators[b_name].keys())
                shared = {s for s in shared
                          if not all_annotators[a][s].get("skipped")
                          and not all_annotators[b_name][s].get("skipped")}
                if len(shared) < 5:
                    continue
                st.markdown(f"**{a}** vs **{b_name}** ({len(shared)} shared)")
                for field, label in [("binary", "Binary"), ("axis_a_label", "Axis A"),
                                     ("axis_b_label", "Axis B"), ("axis_c_label", "Axis C")]:
                    agree = sum(1 for s in shared
                                if all_annotators[a][s].get(field) == all_annotators[b_name][s].get(field))
                    st.caption(f"  {label}: {agree}/{len(shared)} ({agree/len(shared):.0%})")


def main():
    st.set_page_config(page_title="Sexism Annotation v2", layout="wide",
                       initial_sidebar_state="expanded")

    data = load_data()
    annotator, annotations, display, idx = render_sidebar(data)

    if annotator == "___stats___":
        render_stats_page()
        return

    if annotator is None:
        st.title("Sexism Axis Annotation (v2 -- rebuttal)")
        st.markdown(
            "63 relevant speeches (gold stance: for/against/both).\n"
            "Irrelevant speeches are excluded.\n\n"
            "For each speech, classify:\n\n"
            "1. **Binary sexism** (sexist / not_sexist)\n"
            "2. **Axis A** -- Ambivalent Sexism (Glick & Fiske 1996)\n"
            "3. **Axis B** -- Stereotype Content (Fiske et al 2002)\n"
            "4. **Axis C** -- Gender Norm Type (Prentice & Carranza 2002)\n\n"
            "Stance is pre-filled from v1 gold labels. Previous axis annotations "
            "are shown as reference.\n\n"
            "Enter your name in the sidebar to start.\n"
            "See `outputs/validation/sexism_annotation_guide.md` for the full rubric."
        )
        st.markdown("---")
        st.markdown(SCOPE_DEFINITION)
        return

    if display is None or len(display) == 0:
        st.info("No speeches to display.")
        return

    row = display.iloc[idx]
    speech_id = row["speech_id"]
    already_done = speech_id in annotations
    defaults = annotations.get(speech_id)

    # Pre-fill from Omar's v1 sexism labels if no v2 annotation yet
    if defaults is None:
        v1 = load_v1_annotations()
        omar_v1 = v1.get(speech_id, {}).get("omar")
        if omar_v1:
            defaults = translate_v1_axes(omar_v1)

    col_read, col_form = st.columns([3, 2])
    n_done = len([s for s in display["speech_id"] if s in annotations])

    with col_read:
        render_speech(row, idx, len(display), already_done, n_done)
        render_v1_reference(speech_id)

    with col_form:
        result = render_form(row, defaults)
        if result is not None:
            result["annotator"] = annotator
            save_annotation(annotator, result)
            label = "Skipped" if result["skipped"] else "Saved"
            st.toast(f"{label}! ({speech_id[:12]}...)")
            if st.session_state.speech_idx < len(display) - 1:
                st.session_state.speech_idx += 1
            st.rerun()


if __name__ == "__main__":
    main()
