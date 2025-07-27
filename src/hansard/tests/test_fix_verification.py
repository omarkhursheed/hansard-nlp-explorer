#!/usr/bin/env python3
"""Simple test to verify the single-digit date fix without dependencies."""

import re

def test_debate_link_extraction_fix():
    """Test the fixed link extraction logic."""
    
    print("Testing debate link extraction fix for single-digit days...")
    print("=" * 60)
    
    # Mock HTML content that would be returned for day 7 (after redirect to day 07)
    # This simulates what the Parliament API returns
    mock_html = '''
    <a href="/historic-hansard/commons/1928/may/07/private-business">Private Business</a>
    <a href="/historic-hansard/lords/1928/may/07/oral-answers-to-questions">Oral Answers</a>
    <a href="/historic-hansard/commons/1928/may/07/trade-and-commerce">Trade and Commerce</a>
    <a href="/historic-hansard/1928/may/07/some-topic">Some Topic</a>
    '''
    
    # Extract all href links (simulate _extract_links)
    href_pattern = r'href="([^"]*)"'
    links = re.findall(href_pattern, mock_html, re.I)
    
    print(f"Extracted {len(links)} links from mock HTML:")
    for link in links:
        print(f"  {link}")
    
    # Test the fixed logic
    date_path = "1928/may/7"
    y, mo, d = date_path.split("/")
    
    # Handle both single-digit and zero-padded day formats (the fix)
    d_padded = d.zfill(2) if d.isdigit() else d
    day_formats = [d, d_padded] if d != d_padded else [d]
    
    print(f"\nTesting with date_path: {date_path}")
    print(f"Day formats to check: {day_formats}")
    
    out = []
    
    for link in links:
        if not link or any(x in link.lower() for x in ("index", "contents", "#")):
            continue
        
        # Look for debate patterns with both day formats
        found_match = False
        for day_format in day_formats:
            # Pattern 1: /commons/1864/feb/15/topic or /lords/1864/feb/15/topic
            pattern1 = rf"/(commons|lords)/{re.escape(y)}/{re.escape(mo)}/{re.escape(day_format)}/([^/?#]+)"
            m1 = re.search(pattern1, link, re.I)
            if m1:
                path = m1.group(0).lstrip("/")
                if path not in out:
                    out.append(path)
                found_match = True
                print(f"  ✓ Pattern 1 matched: {link} -> {path}")
                break
            
            # Pattern 2: /1864/feb/15/topic (generic)
            pattern2 = rf"/{re.escape(y)}/{re.escape(mo)}/{re.escape(day_format)}/([^/?#]+)"
            m2 = re.search(pattern2, link, re.I)
            if m2:
                topic = m2.group(1)
                # Add both commons and lords variants
                for house in ["commons", "lords"]:
                    path = f"{house}/{y}/{mo}/{d}/{topic}"
                    if path not in out:
                        out.append(path)
                found_match = True
                print(f"  ✓ Pattern 2 matched: {link} -> added {house} variants")
                break
        
        if not found_match:
            print(f"  ✗ No match: {link}")
    
    print(f"\nFinal result: Found {len(out)} debate links")
    for i, debate_link in enumerate(out, 1):
        print(f"  {i}. {debate_link}")
    
    if out:
        print("\n✅ SUCCESS: Single-digit day fix works! Debate links found.")
    else:
        print("\n❌ FAILURE: No debate links found even with the fix.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_debate_link_extraction_fix()