#!/usr/bin/env python3
"""
Fast validation app for LLM classifications.
Optimized for speed with large buttons and auto-advance.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Validation Interface",
    page_icon="✓",
    layout="wide"
)

# Custom CSS for better UX
st.markdown("""
<style>
.stButton > button {
    height: 60px;
    font-size: 18px;
    font-weight: bold;
}
div[data-testid="column"] {
    padding: 5px;
}
</style>
""", unsafe_allow_html=True)

def load_validation_data():
    """Load validation sample (no cache - always fresh data)."""
    sample_file = 'outputs/validation/validation_sample.parquet'
    return pd.read_parquet(sample_file)

def save_validation_data(df):
    """Save validation data."""
    output_file = 'outputs/validation/validation_sample.parquet'
    df.to_parquet(output_file, index=False)

def format_reasons(reasons):
    """Format reasons for display."""
    if reasons is None:
        return "No structured arguments extracted"

    if isinstance(reasons, np.ndarray):
        reasons = reasons.tolist()

    if not isinstance(reasons, (list, tuple)) or len(reasons) == 0:
        return "No structured arguments extracted"

    formatted = []
    for i, reason in enumerate(reasons, 1):
        if isinstance(reason, dict):
            bucket = reason.get('bucket_key', 'unknown')
            rationale = reason.get('rationale', '')
            quotes = reason.get('quotes', [])

            if isinstance(quotes, np.ndarray):
                quotes = quotes.tolist()

            formatted.append(f"**{i}. {bucket.replace('_', ' ').title()}**")
            formatted.append(f"{rationale}")

            if quotes:
                for q in quotes[:2]:  # Show first 2 quotes
                    if isinstance(q, dict):
                        text = q.get('text', '')
                        formatted.append(f"> \"{text}\"")
            formatted.append("")

    return "\n\n".join(formatted)

def main():
    st.title("✓ Validation Interface")

    # Load data
    df = load_validation_data()

    # Initialize session state
    if 'current_index' not in st.session_state:
        # Start with first unvalidated speech
        unvalidated = df[~df['validated']]
        if len(unvalidated) > 0:
            st.session_state.current_index = unvalidated.iloc[0]['validation_index']
        else:
            st.session_state.current_index = 0

    current_idx = st.session_state.current_index

    # Progress
    total = len(df)
    validated_count = df['validated'].sum()
    progress = validated_count / total if total > 0 else 0

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.progress(progress)
    with col2:
        st.metric("Progress", f"{validated_count}/{total}")
    with col3:
        st.metric("Current", f"#{current_idx + 1}")

    # Get current speech
    current_speech = df[df['validation_index'] == current_idx].iloc[0]

    # Display speech info
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Speaker", current_speech['canonical_name'])
    col2.metric("Gender", current_speech['gender'])
    col3.metric("Year", f"{current_speech['year']:.0f}")
    col4.metric("Confidence", f"{current_speech['confidence']:.2f}")

    # LLM Classification
    st.markdown("### LLM Classification")
    st.markdown(f"**Stance:** `{current_speech['stance'].upper()}`")
    st.markdown(f"**Speech ID:** `{current_speech['speech_id']}`")

    # Speech text
    st.markdown("### Speech Text")
    st.text_area(
        "Speech content",
        current_speech['target_text'],
        height=200,
        key=f"speech_{current_idx}",
        label_visibility="collapsed"
    )

    # Context toggle
    show_context = st.checkbox("Show debate context", key=f"context_toggle_{current_idx}")
    if show_context:
        st.text_area(
            "Context",
            current_speech['context_text'],
            height=150,
            key=f"context_text_{current_idx}",
            label_visibility="collapsed"
        )

    st.markdown("---")

    # LLM Reasoning
    st.markdown("### LLM Extracted Arguments")
    reasons_text = format_reasons(current_speech.get('reasons'))
    st.markdown(reasons_text)

    st.markdown("---")
    st.markdown("### YOUR VALIDATION")

    # Validation with auto-save
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Stance Classification")
        stance = st.radio(
            "Is the stance correct?",
            options=['correct', 'wrong', 'unclear'],
            index=None,
            key=f"stance_{current_idx}",
            horizontal=True,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("#### Arguments Quality")
        quality = st.radio(
            "Quality of extracted arguments?",
            options=['high', 'medium', 'low'],
            index=None,
            key=f"quality_{current_idx}",
            horizontal=True,
            label_visibility="collapsed"
        )

    # Error type checkboxes (show when stance is wrong)
    error_tags = []
    if stance == 'wrong':
        st.markdown("**Why is it wrong?** (check all that apply)")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.checkbox("Pro-woman, not suffrage", key=f"err_prowoman_{current_idx}"):
                error_tags.append("pro-woman but not suffrage")

        with col2:
            if st.checkbox("Different topic entirely", key=f"err_topic_{current_idx}"):
                error_tags.append("different topic")

        with col3:
            if st.checkbox("Opposite stance", key=f"err_opposite_{current_idx}"):
                error_tags.append("opposite stance")

    # Notes
    auto_notes = "; ".join(error_tags) if error_tags else ""
    notes = st.text_area(
        "Additional notes (optional)",
        value=auto_notes or current_speech.get('notes', ''),
        height=70,
        key=f"notes_{current_idx}",
        help="Error tags are auto-filled above. Add extra details here if needed."
    )

    st.markdown("---")

    # Navigation
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("◀ PREVIOUS", key="prev", disabled=(current_idx == 0), use_container_width=True):
            st.session_state.current_index = max(0, current_idx - 1)
            st.rerun()

    with col2:
        # Skip to next unvalidated
        next_unvalidated = df[(df['validation_index'] > current_idx) & (~df['validated'])]
        has_unvalidated = len(next_unvalidated) > 0

        if st.button("SKIP TO NEXT UNVALIDATED", key="skip", disabled=not has_unvalidated, use_container_width=True):
            if has_unvalidated:
                st.session_state.current_index = next_unvalidated.iloc[0]['validation_index']
                st.rerun()

    with col3:
        can_save = stance is not None and quality is not None

        if st.button("💾 SAVE & NEXT", key="save", disabled=not can_save, use_container_width=True, type="primary"):
            # Save current validation
            df.loc[df['validation_index'] == current_idx, 'stance_correct'] = stance
            df.loc[df['validation_index'] == current_idx, 'reasons_quality'] = quality
            df.loc[df['validation_index'] == current_idx, 'notes'] = notes
            df.loc[df['validation_index'] == current_idx, 'validated'] = True
            df.loc[df['validation_index'] == current_idx, 'validation_timestamp'] = datetime.now().isoformat()

            save_validation_data(df)

            # Move to next unvalidated or just next
            next_unvalidated = df[(df['validation_index'] > current_idx) & (~df['validated'])]
            if len(next_unvalidated) > 0:
                st.session_state.current_index = next_unvalidated.iloc[0]['validation_index']
            else:
                st.session_state.current_index = min(total - 1, current_idx + 1)

            st.rerun()

    with col4:
        if st.button("NEXT ▶", key="next", disabled=(current_idx >= total - 1), use_container_width=True):
            st.session_state.current_index = min(total - 1, current_idx + 1)
            st.rerun()

    # Help
    with st.expander("Help"):
        st.markdown("""
        **Workflow:**
        1. Read the speech text
        2. Check if LLM stance classification is correct
        3. Rate the quality of extracted arguments
        4. Add notes if needed
        5. Click SAVE & NEXT (automatically jumps to next unvalidated)

        **Stance:**
        - **Correct**: LLM got it right
        - **Wrong**: LLM misclassified
        - **Unclear**: Speech is ambiguous

        **Quality:**
        - **High**: Arguments well-extracted and representative
        - **Medium**: Some arguments captured but incomplete
        - **Low**: Poor extraction or missing key arguments
        """)


if __name__ == '__main__':
    main()
