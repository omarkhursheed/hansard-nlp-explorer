#!/usr/bin/env python3
"""
Interactive web app for manual validation of suffrage classification.

Usage:
    streamlit run scripts/classification/validation_app.py
"""

import pandas as pd
import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Suffrage Classification Validation",
    page_icon="📝",
    layout="wide"
)

# File paths
VALIDATION_SAMPLE = Path("outputs/validation/validation_sample_n48.parquet")
VALIDATION_RESULTS = Path("outputs/validation/validation_recording_template.csv")
INPUT_TEXT = Path("outputs/llm_classification/full_input_context_3.parquet")

# Load data
@st.cache_data
def load_data():
    """Load validation sample and input text."""
    sample = pd.read_parquet(VALIDATION_SAMPLE)
    try:
        input_df = pd.read_parquet(INPUT_TEXT)
        # Merge to get full text
        sample = sample.merge(
            input_df[['speech_id', 'target_text']],
            on='speech_id',
            how='left'
        )
    except:
        sample['target_text'] = None

    return sample

def load_progress():
    """Load existing validation progress."""
    if VALIDATION_RESULTS.exists():
        return pd.read_csv(VALIDATION_RESULTS)
    else:
        # Create new template
        df = load_data()
        results = df[['speech_id', 'speaker', 'date', 'stance', 'confidence', 'gender']].copy()
        results.columns = ['speech_id', 'speaker', 'date', 'llm_stance', 'llm_confidence', 'gender']
        results['your_judgment'] = None
        results['notes'] = None
        results['stance_correct'] = None
        results['reasons_correct'] = None
        results['quotes_accurate'] = None
        return results

def save_progress(results_df):
    """Save validation progress."""
    results_df.to_csv(VALIDATION_RESULTS, index=False)

def get_completion_status(results_df):
    """Get validation completion status."""
    total = len(results_df)
    completed = results_df['stance_correct'].notna().sum()
    return completed, total

def display_speech(row, speech_idx):
    """Display a single speech for validation."""

    # Header
    st.title(f"Speech {speech_idx + 1} of 48")

    # Progress
    results = load_progress()
    completed, total = get_completion_status(results)
    st.progress(completed / total)
    st.write(f"**Progress:** {completed}/{total} speeches validated")

    # Metadata
    st.header("Metadata")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Speaker", row['speaker'])
    with col2:
        st.metric("Date", row['date'])
    with col3:
        st.metric("Gender", row.get('gender', 'Unknown'))
    with col4:
        st.metric("Party", row.get('party', 'Unknown'))

    st.write(f"**Speech ID:** `{row['speech_id']}`")

    # LLM Classification
    st.header("LLM Classification")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stance", row['stance'].upper())
    with col2:
        st.metric("Confidence", f"{row['confidence']:.2f}")
    with col3:
        if 'context_helpful' in row:
            st.metric("Context Helpful", row['context_helpful'])

    # Reasons
    st.subheader("Extracted Arguments")
    reasons = row.get('reasons')
    if isinstance(reasons, list) and len(reasons) > 0:
        for i, reason in enumerate(reasons, 1):
            if isinstance(reason, dict):
                bucket = reason.get('bucket_key', 'unknown')
                stance_label = reason.get('stance_label', '?')
                rationale = reason.get('rationale', 'N/A')

                with st.expander(f"**Argument {i}:** [{bucket.upper()}] → {stance_label}"):
                    st.write(rationale)

                    quotes = reason.get('quotes', [])
                    if isinstance(quotes, list) and len(quotes) > 0:
                        st.write("**Evidence:**")
                        for quote in quotes:
                            if isinstance(quote, dict):
                                source = quote.get('source', '?')
                                text = quote.get('text', '')
                                st.markdown(f"- *[{source}]* \"{text}\"")
    else:
        st.info("No arguments extracted")

    # Full speech text
    st.header("Full Speech Text")
    if pd.notna(row.get('target_text')):
        text = row['target_text']
        st.text_area(
            "Speech content",
            text,
            height=300,
            key=f"text_{speech_idx}"
        )
        st.caption(f"Total: {len(text)} characters, ~{len(text.split())} words")
    else:
        st.warning("Speech text not available")

    st.divider()

    # Validation form
    st.header("Your Validation")

    # Load existing judgment if any
    existing = results[results['speech_id'] == row['speech_id']].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        your_judgment = st.selectbox(
            "What should the stance be?",
            options=['', 'for', 'against', 'both', 'neutral', 'irrelevant'],
            index=0 if pd.isna(existing['your_judgment']) else
                  ['', 'for', 'against', 'both', 'neutral', 'irrelevant'].index(existing['your_judgment']),
            key=f"judgment_{speech_idx}",
            help="Select the correct stance for this speech"
        )

        stance_correct = st.radio(
            "Is the LLM stance correct?",
            options=['', 'YES', 'NO'],
            index=0 if pd.isna(existing['stance_correct']) else
                  ['', 'YES', 'NO'].index(existing['stance_correct']),
            key=f"stance_{speech_idx}",
            horizontal=True
        )

    with col2:
        reasons_correct = st.radio(
            "Are the extracted reasons correct?",
            options=['', 'YES', 'PARTIAL', 'NO'],
            index=0 if pd.isna(existing['reasons_correct']) else
                  ['', 'YES', 'PARTIAL', 'NO'].index(existing['reasons_correct']),
            key=f"reasons_{speech_idx}",
            horizontal=True,
            help="YES = all correct, PARTIAL = some correct, NO = mostly wrong"
        )

        quotes_accurate = st.radio(
            "Are the quotes accurate?",
            options=['', 'YES', 'PARTIAL', 'NO'],
            index=0 if pd.isna(existing['quotes_accurate']) else
                  ['', 'YES', 'PARTIAL', 'NO'].index(existing['quotes_accurate']),
            key=f"quotes_{speech_idx}",
            horizontal=True,
            help="YES = accurate, PARTIAL = mostly accurate, NO = inaccurate"
        )

    notes = st.text_area(
        "Notes (optional)",
        value=existing['notes'] if pd.notna(existing['notes']) else '',
        key=f"notes_{speech_idx}",
        help="Why is it wrong? What patterns do you notice?",
        height=100
    )

    # Save button
    if st.button("💾 Save & Continue", type="primary", use_container_width=True):
        # Update results
        results.loc[results['speech_id'] == row['speech_id'], 'your_judgment'] = your_judgment if your_judgment else None
        results.loc[results['speech_id'] == row['speech_id'], 'stance_correct'] = stance_correct if stance_correct else None
        results.loc[results['speech_id'] == row['speech_id'], 'reasons_correct'] = reasons_correct if reasons_correct else None
        results.loc[results['speech_id'] == row['speech_id'], 'quotes_accurate'] = quotes_accurate if quotes_accurate else None
        results.loc[results['speech_id'] == row['speech_id'], 'notes'] = notes if notes else None

        # Save to file
        save_progress(results)

        st.success("✅ Saved!")

        # Move to next speech
        if speech_idx < 47:
            st.session_state.speech_idx = speech_idx + 1
            st.rerun()
        else:
            st.balloons()
            st.success("🎉 All speeches validated! Run analyze_validation_results.py to see the results.")

def main():
    """Main app."""

    # Initialize session state
    if 'speech_idx' not in st.session_state:
        st.session_state.speech_idx = 0

    # Load data
    data = load_data()

    # Sidebar
    st.sidebar.title("Navigation")

    # Speech selector
    speech_idx = st.sidebar.number_input(
        "Jump to speech:",
        min_value=1,
        max_value=48,
        value=st.session_state.speech_idx + 1,
        step=1
    ) - 1

    st.session_state.speech_idx = speech_idx

    # Navigation buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("⬅️ Previous", disabled=speech_idx == 0):
            st.session_state.speech_idx = max(0, speech_idx - 1)
            st.rerun()
    with col2:
        if st.button("Next ➡️", disabled=speech_idx == 47):
            st.session_state.speech_idx = min(47, speech_idx + 1)
            st.rerun()

    # Progress summary
    st.sidebar.divider()
    st.sidebar.subheader("Validation Progress")
    results = load_progress()
    completed, total = get_completion_status(results)
    st.sidebar.metric("Completed", f"{completed}/{total}")
    st.sidebar.metric("Remaining", total - completed)

    # Stance distribution
    st.sidebar.divider()
    st.sidebar.subheader("Sample Distribution")
    for stance, count in data['stance'].value_counts().items():
        st.sidebar.write(f"**{stance.upper()}:** {count}")

    # Instructions
    st.sidebar.divider()
    with st.sidebar.expander("📖 Instructions"):
        st.markdown("""
        **For each speech:**
        1. Read the full text
        2. Check if LLM stance is correct
        3. Verify reasons are accurate
        4. Check quotes are from the speech
        5. Add notes if needed
        6. Click Save & Continue

        **Stance definitions:**
        - **FOR**: Supports women's suffrage
        - **AGAINST**: Opposes women's suffrage
        - **BOTH**: Mixed position
        - **NEUTRAL**: Genuinely indifferent
        - **IRRELEVANT**: Not about suffrage
        """)

    # Display current speech
    current_speech = data.iloc[speech_idx]
    display_speech(current_speech, speech_idx)

if __name__ == "__main__":
    main()
