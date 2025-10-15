#!/usr/bin/env python3
"""
Manual Review of Suffrage Debates - Validation and Pattern Documentation

This script facilitates manual review of the suffrage pilot dataset by:
1. Loading and sampling representative debates
2. Identifying argument patterns (pro/anti suffrage)
3. Documenting speaker positions
4. Validating research questions for LLM pipeline

Usage:
    python manual_suffrage_review.py --sample 20 --output analysis/suffrage_review
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd


def load_pilot_dataset(pilot_dir):
    """Load the suffrage pilot dataset"""
    pilot_path = Path(pilot_dir) / 'suffrage_debates_pilot.json'

    print(f"Loading pilot dataset from {pilot_path}")
    with open(pilot_path, 'r', encoding='utf-8') as f:
        debates = json.load(f)

    print(f"Loaded {len(debates)} debates")
    return debates


def extract_argument_indicators(text):
    """
    Extract potential pro/anti argument indicators from debate text.

    This is a rule-based preliminary analysis to guide manual review.
    """
    text_lower = text.lower()

    # Pro-suffrage indicators
    pro_patterns = {
        'democratic_rights': [
            r'democratic right',
            r'equal right',
            r'justice',
            r'fair representation',
            r'taxation without representation',
        ],
        'war_contribution': [
            r'war effort',
            r'women.*served',
            r'contribution.*war',
            r'munitions.*women',
        ],
        'competence': [
            r'women.*capable',
            r'women.*qualified',
            r'proved themselves',
            r'demonstrated.*ability',
        ],
        'inevitability': [
            r'logical conclusion',
            r'natural progression',
            r'inevitable',
            r'time has come',
        ]
    }

    # Anti-suffrage indicators
    anti_patterns = {
        'traditional_roles': [
            r'domestic sphere',
            r'home.*duties',
            r'women.*place',
            r'natural.*role',
        ],
        'gradual_change': [
            r'too soon',
            r'not yet ready',
            r'premature',
            r'gradual.*reform',
        ],
        'property_concerns': [
            r'property.*qualification',
            r'householder',
            r'ratepayer',
        ],
        'concerns': [
            r'dangerous',
            r'unwise',
            r'risk',
            r'consequences',
        ]
    }

    findings = {
        'pro_indicators': defaultdict(list),
        'anti_indicators': defaultdict(list),
        'pro_count': 0,
        'anti_count': 0
    }

    # Check for pro patterns
    for category, patterns in pro_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                findings['pro_indicators'][category].extend(matches)
                findings['pro_count'] += len(matches)

    # Check for anti patterns
    for category, patterns in anti_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                findings['anti_indicators'][category].extend(matches)
                findings['anti_count'] += len(matches)

    return findings


def analyze_debate_stance(debate):
    """
    Analyze the overall stance of a debate (preliminary automated analysis).
    """
    text = debate['text']
    indicators = extract_argument_indicators(text)

    # Simple heuristic: more pro than anti indicators suggests pro-suffrage debate
    if indicators['pro_count'] > indicators['anti_count'] * 1.5:
        stance = 'likely_pro'
    elif indicators['anti_count'] > indicators['pro_count'] * 1.5:
        stance = 'likely_anti'
    else:
        stance = 'mixed_or_neutral'

    return {
        'debate_id': f"{debate['year']}_{debate['date']}",
        'year': debate['year'],
        'title': debate['title'],
        'stance': stance,
        'pro_count': indicators['pro_count'],
        'anti_count': indicators['anti_count'],
        'pro_categories': dict(indicators['pro_indicators']),
        'anti_categories': dict(indicators['anti_indicators']),
        'word_count': debate['word_count'],
        'num_speakers': len(debate['speakers']) if debate['speakers'] else 0
    }


def identify_key_debates_for_review(debates, sample_size=20):
    """
    Select most important debates for manual review.

    Prioritizes:
    - Key years (1912, 1918, 1928)
    - Long debates with many speakers (substantive discussion)
    - Mix of pro/anti/mixed stances
    """
    analyzed = [analyze_debate_stance(d) for d in debates]

    # Categorize by stance
    by_stance = defaultdict(list)
    for analysis in analyzed:
        by_stance[analysis['stance']].append(analysis)

    print(f"\nStance Distribution:")
    print(f"  Likely Pro-suffrage: {len(by_stance['likely_pro'])}")
    print(f"  Likely Anti-suffrage: {len(by_stance['likely_anti'])}")
    print(f"  Mixed/Neutral: {len(by_stance['mixed_or_neutral'])}")

    # Key years of interest
    key_years = [1912, 1917, 1918, 1928]

    # Select balanced sample
    selected = []

    # Priority 1: Key years
    for year in key_years:
        year_debates = [a for a in analyzed if a['year'] == year]
        # Sort by number of speakers (more substantive)
        year_debates.sort(key=lambda x: x['num_speakers'], reverse=True)
        selected.extend(year_debates[:3])  # Top 3 from each key year

    # Priority 2: Ensure stance diversity
    remaining = sample_size - len(selected)
    if remaining > 0:
        # Add mixed debates (most interesting for argument extraction)
        mixed = sorted(by_stance['mixed_or_neutral'],
                      key=lambda x: x['num_speakers'], reverse=True)
        selected.extend(mixed[:remaining])

    # Remove duplicates
    seen = set()
    unique_selected = []
    for debate in selected:
        if debate['debate_id'] not in seen:
            unique_selected.append(debate)
            seen.add(debate['debate_id'])

    return unique_selected[:sample_size], analyzed


def generate_review_document(selected_debates, all_debates_map, output_dir):
    """
    Generate a structured document for manual review.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    review_path = output_dir / 'manual_review_guide.md'

    with open(review_path, 'w', encoding='utf-8') as f:
        f.write("# Manual Review of Suffrage Debates - Validation Guide\n\n")
        f.write("## Research Questions to Validate\n\n")

        f.write("### 1. Can we identify clear pro/anti suffrage arguments?\n")
        f.write("- Are there distinct argument categories?\n")
        f.write("- Can we map arguments to the taxonomy from research notes?\n")
        f.write("- Are positions consistent within debates?\n\n")

        f.write("### 2. Can we identify individual speakers and their positions?\n")
        f.write("- Are speaker names clearly marked?\n")
        f.write("- Do speakers maintain consistent positions?\n")
        f.write("- Can we build pro/anti MP lists?\n\n")

        f.write("### 3. Are there post-1928 retrospective references?\n")
        f.write("- Do debates after 1928 reference earlier warnings?\n")
        f.write("- Are there 'I told you so' patterns?\n")
        f.write("- What claims are made about consequences?\n\n")

        f.write("---\n\n")
        f.write(f"## Selected Debates for Review ({len(selected_debates)} debates)\n\n")

        for i, debate in enumerate(selected_debates, 1):
            debate_data = all_debates_map.get(debate['debate_id'])
            if not debate_data:
                continue

            f.write(f"### Debate #{i}: {debate['year']} - {debate['title'][:80]}\n\n")
            f.write(f"**Year:** {debate['year']}\n")
            f.write(f"**Automated Stance:** {debate['stance']}\n")
            f.write(f"**Speakers:** {debate['num_speakers']}\n")
            f.write(f"**Word Count:** {debate['word_count']:,}\n")
            f.write(f"**Pro Indicators:** {debate['pro_count']}\n")
            f.write(f"**Anti Indicators:** {debate['anti_count']}\n\n")

            if debate['pro_categories']:
                f.write("**Pro Argument Categories Found:**\n")
                for category, matches in debate['pro_categories'].items():
                    f.write(f"- {category}: {len(matches)} matches\n")
                f.write("\n")

            if debate['anti_categories']:
                f.write("**Anti Argument Categories Found:**\n")
                for category, matches in debate['anti_categories'].items():
                    f.write(f"- {category}: {len(matches)} matches\n")
                f.write("\n")

            # Show excerpt
            f.write("**Excerpt (first 300 words):**\n")
            f.write("```\n")
            words = debate_data['text'].split()[:300]
            f.write(' '.join(words) + "...\n")
            f.write("```\n\n")

            f.write("**Manual Review Notes:**\n")
            f.write("- [ ] Position clearly stated (pro/anti/neutral)\n")
            f.write("- [ ] Main arguments identified\n")
            f.write("- [ ] Key speakers noted\n")
            f.write("- [ ] Suitable for LLM extraction?\n\n")
            f.write("---\n\n")

    print(f"\nGenerated review guide: {review_path}")

    # Save structured analysis as JSON
    json_path = output_dir / 'preliminary_analysis.json'
    with open(json_path, 'w') as f:
        json.dump({
            'selected_debates': selected_debates,
            'summary': {
                'total_selected': len(selected_debates),
                'by_year': dict(Counter(d['year'] for d in selected_debates)),
                'by_stance': dict(Counter(d['stance'] for d in selected_debates))
            }
        }, f, indent=2)

    print(f"Saved preliminary analysis: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Manual review of suffrage debates'
    )

    parser.add_argument('--pilot-dir', type=str,
                       default='analysis/suffrage_pilot',
                       help='Directory with pilot dataset')

    parser.add_argument('--sample', type=int, default=20,
                       help='Number of debates to review')

    parser.add_argument('--output', type=str,
                       default='analysis/suffrage_review',
                       help='Output directory for review documents')

    args = parser.parse_args()

    # Load pilot dataset
    debates = load_pilot_dataset(args.pilot_dir)

    # Create debate ID map
    debates_map = {
        f"{d['year']}_{d['date']}": d
        for d in debates
    }

    # Identify key debates for review
    print(f"\nSelecting {args.sample} most important debates for manual review...")
    selected, all_analyzed = identify_key_debates_for_review(debates, args.sample)

    print(f"\nSelected {len(selected)} debates:")
    print(f"  Years: {sorted(set(d['year'] for d in selected))}")
    print(f"  Stances: {Counter(d['stance'] for d in selected)}")

    # Generate review document
    generate_review_document(selected, debates_map, args.output)

    print("\n" + "="*80)
    print("MANUAL REVIEW GUIDE READY")
    print(f"Next step: Review debates in {args.output}/manual_review_guide.md")
    print("="*80)


if __name__ == "__main__":
    main()
