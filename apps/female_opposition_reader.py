#!/usr/bin/env python3
"""
Streamlit app for reading anti-suffrage speeches by female MPs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Female MPs Anti-Suffrage Speeches",
    page_icon="H",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load the extracted speeches."""
    data_dir = Path('outputs/suffrage_exploration')

    against = pd.read_parquet(data_dir / 'female_against_speeches.parquet')
    both = pd.read_parquet(data_dir / 'female_both_speeches.parquet')

    return against, both


def format_reasons(reasons):
    """Format the reasons/arguments nicely."""
    # Handle None, empty, or non-list types
    if reasons is None:
        return "No structured arguments extracted"

    # Convert numpy arrays to lists
    if isinstance(reasons, np.ndarray):
        reasons = reasons.tolist()

    if not isinstance(reasons, (list, tuple)):
        return "No structured arguments extracted"

    if len(reasons) == 0:
        return "No structured arguments extracted"

    formatted = []
    for i, reason in enumerate(reasons, 1):
        if isinstance(reason, dict):
            bucket = reason.get('bucket_key', 'unknown')
            stance_label = reason.get('stance_label', 'unknown')
            rationale = reason.get('rationale', '')
            quotes = reason.get('quotes', [])

            # Convert quotes to list if it's a numpy array
            if isinstance(quotes, np.ndarray):
                quotes = quotes.tolist()

            formatted.append(f"**Argument {i}: {bucket}** ({stance_label})")
            formatted.append(f"- {rationale}")

            if quotes:
                formatted.append("- Quotes:")
                for q in quotes:
                    if isinstance(q, dict):
                        text = q.get('text', '')
                        source = q.get('source', 'TARGET')
                        formatted.append(f"  - [{source}] \"{text}\"")
            formatted.append("")

    return "\n".join(formatted)


def main():
    st.title("Female MPs: Anti-Suffrage Speeches")
    st.markdown("""
    Browse speeches by female MPs that were classified as **against** or **mixed** (both for and against)
    women's suffrage. These span 1920-2004 and reveal diverse perspectives on voting rights.

    **Note**: Many post-1928 speeches may be about *expansions* of suffrage (e.g., lowering voting age,
    prisoners' rights) rather than opposing women's suffrage directly.
    """)

    # Load data
    against, both = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")

    stance_filter = st.sidebar.radio(
        "Stance",
        ["Against only", "Both/Mixed only", "All"],
        index=2
    )

    if stance_filter == "Against only":
        df = against.copy()
    elif stance_filter == "Both/Mixed only":
        df = both.copy()
    else:
        df = pd.concat([against, both], ignore_index=True)

    # Year filter
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = st.sidebar.slider(
        "Year Range",
        min_year, max_year,
        (min_year, max_year)
    )

    df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

    # Speaker filter
    all_speakers = sorted(df['canonical_name'].unique())
    selected_speakers = st.sidebar.multiselect(
        "Filter by Speaker",
        all_speakers,
        default=[]
    )

    if selected_speakers:
        df = df[df['canonical_name'].isin(selected_speakers)]

    # Confidence filter
    confidence_min = st.sidebar.slider(
        "Minimum Confidence",
        0.0, 1.0, 0.0, 0.1
    )

    df = df[df['confidence'] >= confidence_min]

    # Search
    search_term = st.sidebar.text_input("Search in speech text (case-insensitive)")
    if search_term:
        df = df[df['target_text'].str.contains(search_term, case=False, na=False)]

    # Summary stats
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{len(df)} speeches** match filters")
    st.sidebar.markdown(f"**{df['canonical_name'].nunique()} unique speakers**")

    # Main content
    if len(df) == 0:
        st.warning("No speeches match the current filters.")
        return

    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        ["Year (oldest first)", "Year (newest first)", "Speaker (A-Z)", "Confidence (high to low)"]
    )

    if sort_by == "Year (oldest first)":
        df = df.sort_values('year', ascending=True)
    elif sort_by == "Year (newest first)":
        df = df.sort_values('year', ascending=False)
    elif sort_by == "Speaker (A-Z)":
        df = df.sort_values('canonical_name', ascending=True)
    else:  # Confidence
        df = df.sort_values('confidence', ascending=False)

    # Display speeches
    for idx, row in df.iterrows():
        with st.expander(
            f"**{row['canonical_name']}** ({row['year']:.0f}) - "
            f"{row['stance'].upper()} - "
            f"Confidence: {row['confidence']:.2f}",
            expanded=False
        ):
            # Metadata
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Year", f"{row['year']:.0f}")
            col2.metric("Party", row.get('party', 'Unknown'))
            col3.metric("Stance", row['stance'])
            col4.metric("Confidence", f"{row['confidence']:.2f}")

            st.markdown("---")

            # Speech text
            st.markdown("### Speech Text")
            st.markdown(f"**Debate ID**: {row['debate_id']}")
            st.markdown(f"**Speech ID**: {row['speech_id']}")

            # Show target text
            st.markdown("**Target Speech:**")
            st.text_area(
                "Speech content",
                row['target_text'],
                height=200,
                key=f"target_{idx}",
                label_visibility="collapsed"
            )

            # Show context if available
            if pd.notna(row.get('context_text')) and row.get('context_text'):
                with st.expander("View debate context"):
                    st.text_area(
                        "Context",
                        row['context_text'],
                        height=300,
                        key=f"context_{idx}",
                        label_visibility="collapsed"
                    )

            st.markdown("---")

            # Arguments/Reasons
            st.markdown("### Extracted Arguments")
            reasons_text = format_reasons(row.get('reasons'))
            st.markdown(reasons_text)

            # Top quote
            if isinstance(row.get('top_quote'), dict):
                st.markdown("### Top Quote")
                quote = row['top_quote']
                source = quote.get('source', 'TARGET')
                st.info(f"[{source}] \"{quote.get('text', '')}\"")


if __name__ == '__main__':
    main()
