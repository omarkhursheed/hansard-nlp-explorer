"""
Review the 5 women against suffrage speeches from Claude Sonnet 4.5.
"""
import pandas as pd

claude_df = pd.read_parquet('outputs/llm_classification/claude_sonnet_45_full_results.parquet')
input_df = pd.read_parquet('outputs/llm_classification/full_input_context_3_expanded.parquet')

# Merge
merged = claude_df.merge(input_df[['speech_id', 'target_text', 'context_text']], on='speech_id', how='left')

# Women against
women_against = merged[(merged['gender'] == 'F') & (merged['stance'] == 'against')].sort_values('year')

print("="*80)
print(f"CLAUDE SONNET 4.5: {len(women_against)} WOMEN AGAINST SUFFRAGE SPEECHES")
print("="*80)

for idx, (i, row) in enumerate(women_against.iterrows(), 1):
    print(f"\n{'='*80}")
    print(f"SPEECH {idx}/5")
    print('='*80)
    print(f"Speech ID: {row['speech_id']}")
    print(f"Speaker: {row['speaker']}")
    print(f"Year: {row['year']}")
    print(f"Confidence: {row['confidence']:.2f}")

    # Reasons
    if isinstance(row['reasons'], list) and len(row['reasons']) > 0:
        print(f"\nReasons extracted ({len(row['reasons'])} total):")
        for r in row['reasons']:
            if isinstance(r, dict):
                bucket = r.get('bucket_key', 'unknown')
                rationale = r.get('rationale', '')[:150]
                print(f"  - {bucket}: {rationale}")

    # Top quote
    if isinstance(row['top_quote'], dict):
        quote = row['top_quote'].get('text', '')
        if quote:
            print(f"\nTop Quote:")
            print(f'  "{quote}"')

    print(f"\n{'='*80}")
    print("FULL TEXT:")
    print('='*80)
    text = row['target_text']
    print(text)

    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print('='*80)

    text_lower = text.lower()

    # Gender misattribution check
    if 'mr. ' in text[:100] or text.startswith('Male '):
        print(">> GENDER MISATTRIBUTION - Actually spoken by a man (Mr. Murray)")
        print("   Classification: CORRECT (anti-suffrage)")
        print("   Problem: Upstream speaker attribution error")

    # Check topic
    elif any(keyword in text_lower for keyword in ['vote', 'franchise', 'suffrage', 'electoral']):
        # It's about suffrage - check if truly against
        if any(phrase in text_lower for phrase in ['should not', 'against', 'oppose', 'reject', 'cannot support']):
            print(">> LEGITIMATE ANTI-SUFFRAGE SPEECH")
            print("   Contains suffrage language AND opposition")
        else:
            print(">> AMBIGUOUS - Contains suffrage language but opposition unclear")

    # Check for related but different topics
    elif any(keyword in text_lower for keyword in ['quota', 'shortlist', 'recruitment', 'candidate selection']):
        print(">> FALSE POSITIVE - About candidate selection/quotas, NOT voting rights")
        print("   Topic: Parliamentary candidate selection")
        print("   Not suffrage: Women MPs selecting candidates != women voters")

    else:
        print(">> QUESTIONABLE - No clear suffrage language")
        print("   May be false positive")

print(f"\n{'='*80}")
print("SUMMARY")
print('='*80)
print(f"Total: {len(women_against)} speeches")
print("\nExpected breakdown:")
print("  - Gender misattribution: 1 (Mr. Murray)")
print("  - Legitimate anti-suffrage: 1-2 ")
print("  - False positives (quotas/other): 2-3")
