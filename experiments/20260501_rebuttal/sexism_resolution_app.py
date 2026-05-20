"""
Resolve sexism annotation disagreements between Omar and Mandira.

12 binary disagreements (sexist vs not_sexist)
7 axis disagreements (both said sexist but disagree on axes)

Usage:
    streamlit run experiments/20260501_rebuttal/sexism_resolution_app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

DISAGREEMENTS_PATH = Path("experiments/20260501_rebuttal/sexism_disagreements.json")
RESOLUTIONS_PATH = Path("experiments/20260501_rebuttal/sexism_resolutions.json")
VALIDATION_DATA = "outputs/validation/validation_sample.parquet"

AXIS_A_LABELS = ["hostile", "benevolent", "none"]
AXIS_A_SUBCATEGORIES = {
    "hostile": ["dominative_paternalism", "competitive_gender_differentiation", "heterosexual_hostility", "none"],
    "benevolent": ["protective_paternalism", "complementary_gender_differentiation", "heterosexual_intimacy", "none"],
    "none": ["none"],
}
AXIS_B_LABELS = ["paternalistic_prejudice", "admiration", "contemptuous_prejudice", "envious_prejudice", "none"]
AXIS_C_LABELS = ["descriptive", "prescriptive", "proscriptive", "none"]

STANCE_COLORS = {
    "for": "#10B981", "against": "#EF4444", "both": "#F59E0B",
    "sexist": "#EF4444", "not_sexist": "#10B981",
    "hostile": "#EF4444", "benevolent": "#F59E0B", "none": "#9CA3AF",
}


@st.cache_data
def load_speech_text():
    df = pd.read_parquet(VALIDATION_DATA)
    return dict(zip(df["speech_id"], df["target_text"]))


def load_disagreements():
    return json.loads(DISAGREEMENTS_PATH.read_text())


def load_resolutions():
    if RESOLUTIONS_PATH.exists():
        return json.loads(RESOLUTIONS_PATH.read_text())
    return {}


def save_resolutions(resolutions):
    RESOLUTIONS_PATH.write_text(json.dumps(resolutions, indent=2))


def main():
    st.set_page_config(page_title="Resolve Sexism Disagreements", layout="wide")
    st.title("Sexism Annotation Disagreement Resolution")

    items = load_disagreements()
    resolutions = load_resolutions()
    texts = load_speech_text()

    binary_items = [i for i in items if i.get("omar_binary") != i.get("mandira_binary", "")]
    axis_items = [i for i in items if i not in binary_items]

    done = sum(1 for item in items if item["speech_id"] in resolutions)
    st.progress(done / len(items), text=f"{done}/{len(items)} resolved")

    if done == len(items):
        st.success("All disagreements resolved!")
        st.markdown("Results saved to `experiments/20260501_rebuttal/sexism_resolutions.json`")
        counts = {"sexist": 0, "not_sexist": 0}
        for r in resolutions.values():
            counts[r["resolved_binary"]] = counts.get(r["resolved_binary"], 0) + 1
        st.write("Binary resolutions:", counts)
        return

    # Navigation
    if "current_idx" not in st.session_state:
        unresolved = [i for i, item in enumerate(items) if item["speech_id"] not in resolutions]
        st.session_state.current_idx = unresolved[0] if unresolved else 0

    col_prev, col_sel, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("Prev") and st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
            st.rerun()
    with col_next:
        if st.button("Next") and st.session_state.current_idx < len(items) - 1:
            st.session_state.current_idx += 1
            st.rerun()
    with col_sel:
        jump = st.selectbox(
            "Jump to:",
            range(len(items)),
            index=st.session_state.current_idx,
            format_func=lambda i: (
                f"{'[done] ' if items[i]['speech_id'] in resolutions else ''}"
                f"{i+1}. {items[i]['speaker']} ({items[i]['year']}) "
                f"[{items[i]['gold_stance']}] -- "
                f"{'BINARY' if 'omar_binary' in items[i] else 'AXES'}"
            ),
        )
        if jump != st.session_state.current_idx:
            st.session_state.current_idx = jump
            st.rerun()

    item = items[st.session_state.current_idx]
    sid = item["speech_id"]
    is_binary = "omar_binary" in item

    st.divider()

    # Header
    st.markdown(f"### {item['speaker']} ({item['year']}) -- stance: **{item['gold_stance']}**")
    st.caption(f"speech_id: {sid}")

    if is_binary:
        st.markdown("**Disagreement type: BINARY (sexist vs not_sexist)**")
    else:
        st.markdown("**Disagreement type: AXIS (both say sexist, disagree on axes)**")

    # Show both annotations side by side
    col_o, col_m = st.columns(2)

    oa = item.get("omar_axis_a", "none")
    oa_sub = item.get("omar_axis_a_sub", "none")
    ob_ax = item.get("omar_axis_b", "none")
    oc = item.get("omar_axis_c", "none")
    ma = item.get("mandira_axis_a", "none")
    ma_sub = item.get("mandira_axis_a_sub", "none")
    mb_ax = item.get("mandira_axis_b", "none")
    mc = item.get("mandira_axis_c", "none")

    def diff_marker(o_val, m_val):
        return " !!!" if o_val != m_val else ""

    with col_o:
        st.markdown("**Omar**")
        o_bin = item.get("omar_binary", "sexist" if not is_binary else "none")
        color = STANCE_COLORS.get(o_bin, "#666")
        st.markdown(f"Binary: <span style='color:{color};font-weight:bold'>{o_bin}</span>", unsafe_allow_html=True)
        if o_bin == "sexist" or not is_binary:
            st.caption(f"A: {oa} ({oa_sub}){diff_marker(oa, ma)}")
            st.caption(f"B: {ob_ax}{diff_marker(ob_ax, mb_ax)}")
            st.caption(f"C: {oc}{diff_marker(oc, mc)}")

    with col_m:
        st.markdown("**Mandira**")
        m_bin = item.get("mandira_binary", "sexist" if not is_binary else "none")
        color = STANCE_COLORS.get(m_bin, "#666")
        st.markdown(f"Binary: <span style='color:{color};font-weight:bold'>{m_bin}</span>", unsafe_allow_html=True)
        if m_bin == "sexist" or not is_binary:
            st.caption(f"A: {ma} ({ma_sub}){diff_marker(oa, ma)}")
            st.caption(f"B: {mb_ax}{diff_marker(ob_ax, mb_ax)}")
            st.caption(f"C: {mc}{diff_marker(oc, mc)}")

    # Speech text
    st.divider()
    text = texts.get(sid, "")
    if text:
        if len(text) > 2000:
            st.markdown(text[:1500] + "...")
            with st.expander(f"Full speech ({len(text.split())} words)"):
                st.markdown(text)
        else:
            st.markdown(text)
    else:
        st.warning("Speech text not available")

    # Resolution form
    st.divider()
    existing = resolutions.get(sid, {})

    resolved_binary = st.radio(
        "Resolved: sexist or not_sexist?",
        ["sexist", "not_sexist"],
        index=["sexist", "not_sexist"].index(existing["resolved_binary"]) if existing.get("resolved_binary") else None,
        horizontal=True,
        key=f"binary_{sid}",
    )

    resolved_a = "none"
    resolved_a_sub = "none"
    resolved_b = "none"
    resolved_c = "none"

    if resolved_binary == "sexist":
        st.markdown("**Resolve axes:**")

        default_a = existing.get("resolved_axis_a", item.get("omar_axis_a", "none"))
        if default_a not in AXIS_A_LABELS:
            default_a = "none"
        resolved_a = st.radio(
            "Axis A", AXIS_A_LABELS, index=AXIS_A_LABELS.index(default_a),
            horizontal=True, key=f"a_{sid}",
        )

        if resolved_a != "none":
            subcats = AXIS_A_SUBCATEGORIES[resolved_a]
            default_sub = existing.get("resolved_axis_a_sub", item.get("omar_axis_a_sub", subcats[0]))
            if default_sub not in subcats:
                default_sub = subcats[0]
            resolved_a_sub = st.radio(
                "A subcategory", subcats, index=subcats.index(default_sub),
                horizontal=True, key=f"asub_{sid}_{resolved_a}",
            )

        default_b = existing.get("resolved_axis_b", item.get("omar_axis_b", "none"))
        if default_b not in AXIS_B_LABELS:
            default_b = "none"
        resolved_b = st.radio(
            "Axis B", AXIS_B_LABELS, index=AXIS_B_LABELS.index(default_b),
            horizontal=True, key=f"b_{sid}",
        )

        default_c = existing.get("resolved_axis_c", item.get("omar_axis_c", "none"))
        if default_c not in AXIS_C_LABELS:
            default_c = "none"
        resolved_c = st.radio(
            "Axis C", AXIS_C_LABELS, index=AXIS_C_LABELS.index(default_c),
            horizontal=True, key=f"c_{sid}",
        )

    notes = st.text_input(
        "Resolution notes (optional):",
        value=existing.get("notes", ""),
        key=f"notes_{sid}",
    )

    if st.button("Save resolution", key=f"save_{sid}", disabled=resolved_binary is None):
        resolutions[sid] = {
            "resolved_binary": resolved_binary,
            "resolved_axis_a": resolved_a if resolved_binary == "sexist" else "none",
            "resolved_axis_a_sub": resolved_a_sub if resolved_binary == "sexist" else "none",
            "resolved_axis_b": resolved_b if resolved_binary == "sexist" else "none",
            "resolved_axis_c": resolved_c if resolved_binary == "sexist" else "none",
            "omar_binary": item.get("omar_binary", ""),
            "mandira_binary": item.get("mandira_binary", ""),
            "notes": notes,
        }
        save_resolutions(resolutions)

        unresolved = [i for i, it in enumerate(items) if it["speech_id"] not in resolutions]
        if unresolved:
            st.session_state.current_idx = unresolved[0]
        st.rerun()


if __name__ == "__main__":
    main()
