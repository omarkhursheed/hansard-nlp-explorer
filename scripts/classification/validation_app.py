#!/usr/bin/env python3
"""
Interactive web app for manual validation of suffrage classification.

Usage:
    streamlit run scripts/classification/validation_app.py

Keyboard Shortcuts:
    - Arrow Left/Right: Previous/Next speech
    - Cmd+Enter (Mac) or Ctrl+Enter (Windows): Save & Continue
"""

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
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

def add_keyboard_shortcuts():
    """Add keyboard shortcuts for faster navigation."""
    components.html(
        """
        <script>
        const doc = window.parent.document;

        doc.addEventListener('keydown', function(e) {
            // Don't interfere with typing in text areas or inputs
            if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT' || e.target.isContentEditable) {
                return;
            }

            // Arrow keys for navigation
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                const buttons = Array.from(doc.querySelectorAll('button'));
                const prevBtn = buttons.find(btn => btn.textContent.includes('Previous'));
                if (prevBtn) {
                    prevBtn.click();
                }
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                const buttons = Array.from(doc.querySelectorAll('button'));
                const nextBtn = buttons.find(btn => btn.textContent.includes('Next'));
                if (nextBtn) {
                    nextBtn.click();
                }
            }
            // Save with Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux)
            else if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                const buttons = Array.from(doc.querySelectorAll('button'));
                const saveBtn = buttons.find(btn => btn.textContent.includes('Save'));
                if (saveBtn) {
                    saveBtn.click();
                }
            }
        });
        </script>
        """,
        height=0,
    )

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

    # Handle both list and numpy array
    if reasons is not None and (isinstance(reasons, (list, np.ndarray)) and len(reasons) > 0):
        for i, reason in enumerate(reasons, 1):
            if isinstance(reason, dict):
                bucket = reason.get('bucket_key', 'unknown')
                stance_label = reason.get('stance_label', '?')
                rationale = reason.get('rationale', 'N/A')

                with st.expander(f"**Argument {i}:** [{bucket.upper()}] → {stance_label}", expanded=True):
                    st.write(rationale)

                    quotes = reason.get('quotes', [])
                    # Handle numpy array of quotes
                    if quotes is not None and (isinstance(quotes, (list, np.ndarray)) and len(quotes) > 0):
                        st.write("**Evidence:**")
                        for quote in quotes:
                            if isinstance(quote, dict):
                                source = quote.get('source', '?')
                                text = quote.get('text', '')
                                st.markdown(f"- *[{source}]* \"{text}\"")
    else:
        st.info("No arguments extracted")

    # Full speech text (collapsible to save space)
    with st.expander("📄 Full Speech Text", expanded=False):
        if pd.notna(row.get('target_text')):
            text = row['target_text']
            st.text_area(
                "Speech content",
                text,
                height=400,
                key=f"text_{speech_idx}",
                label_visibility="collapsed"
            )
            st.caption(f"Total: {len(text)} characters, ~{len(text.split())} words")
        else:
            st.warning("Speech text not available")

    st.divider()

    # Validation form
    st.header("Your Validation")

    # Load existing judgment if any
    existing = results[results['speech_id'] == row['speech_id']].iloc[0]

    # Your judgment - use radio buttons for speed
    st.subheader("What should the stance be?")
    your_judgment = st.radio(
        "Select the correct stance:",
        options=['(not sure)', 'for', 'against', 'both', 'neutral', 'irrelevant'],
        index=0 if pd.isna(existing['your_judgment']) else
              ['(not sure)', 'for', 'against', 'both', 'neutral', 'irrelevant'].index(existing['your_judgment']) if existing['your_judgment'] in ['for', 'against', 'both', 'neutral', 'irrelevant'] else 0,
        key=f"judgment_{speech_idx}",
        horizontal=True,
        help="FOR=supports suffrage, AGAINST=opposes, BOTH=mixed, NEUTRAL=indifferent, IRRELEVANT=not about suffrage"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        stance_correct = st.radio(
            "Is the LLM stance correct?",
            options=['(not answered)', 'YES', 'NO'],
            index=0 if pd.isna(existing['stance_correct']) else
                  ['(not answered)', 'YES', 'NO'].index(existing['stance_correct']) if existing['stance_correct'] in ['YES', 'NO'] else 0,
            key=f"stance_{speech_idx}",
            horizontal=True
        )

    with col2:
        reasons_correct = st.radio(
            "Are the extracted reasons correct?",
            options=['(not answered)', 'YES', 'PARTIAL', 'NO'],
            index=0 if pd.isna(existing['reasons_correct']) else
                  ['(not answered)', 'YES', 'PARTIAL', 'NO'].index(existing['reasons_correct']) if existing['reasons_correct'] in ['YES', 'PARTIAL', 'NO'] else 0,
            key=f"reasons_{speech_idx}",
            horizontal=True,
            help="YES = all correct, PARTIAL = some correct, NO = mostly wrong"
        )

        quotes_accurate = st.radio(
            "Are the quotes accurate?",
            options=['(not answered)', 'YES', 'PARTIAL', 'NO'],
            index=0 if pd.isna(existing['quotes_accurate']) else
                  ['(not answered)', 'YES', 'PARTIAL', 'NO'].index(existing['quotes_accurate']) if existing['quotes_accurate'] in ['YES', 'PARTIAL', 'NO'] else 0,
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
    col1, col2 = st.columns([3, 1])
    with col1:
        save_btn = st.button("💾 Save & Continue (Cmd+Enter)", type="primary", use_container_width=True)
    with col2:
        skip_btn = st.button("Skip →", use_container_width=True)

    if save_btn:
        # Validate that required fields are filled
        if stance_correct == '(not answered)':
            st.error("⚠️ Please answer 'Is the LLM stance correct?' before saving.")
        else:
            # Update results
            results.loc[results['speech_id'] == row['speech_id'], 'your_judgment'] = your_judgment if your_judgment not in ['', '(not sure)'] else None
            results.loc[results['speech_id'] == row['speech_id'], 'stance_correct'] = stance_correct if stance_correct != '(not answered)' else None
            results.loc[results['speech_id'] == row['speech_id'], 'reasons_correct'] = reasons_correct if reasons_correct != '(not answered)' else None
            results.loc[results['speech_id'] == row['speech_id'], 'quotes_accurate'] = quotes_accurate if quotes_accurate != '(not answered)' else None
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

    if skip_btn:
        if speech_idx < 47:
            st.session_state.speech_idx = speech_idx + 1
            st.rerun()

def main():
    """Main app."""

    # Add keyboard shortcuts
    add_keyboard_shortcuts()

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

    # Keyboard shortcuts help
    st.sidebar.divider()
    with st.sidebar.expander("⌨️ Keyboard Shortcuts", expanded=True):
        st.markdown("""
        **Navigation:**
        - `←` / `→` Previous/Next speech
        - `Cmd+Enter` Save & Continue

        **Tip:** Full speech text is collapsed by default. Arguments show expanded.
        """)

    # Instructions
    st.sidebar.divider()
    with st.sidebar.expander("📖 Instructions"):
        st.markdown("""
        **For each speech:**
        1. Read arguments & quotes
        2. Expand full text if needed
        3. Check if LLM stance is correct
        4. Verify reasons are accurate
        5. Check quotes are from the speech
        6. Add notes if needed
        7. Click Save & Continue (or Ctrl+Enter)

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
