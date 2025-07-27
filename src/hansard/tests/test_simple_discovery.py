#!/usr/bin/env python3  
"""Simple test to verify day 7 is discoverable (no crawler dependencies)."""

import requests
import re
import json

def test_may_1928_discovery():
    """Test discovery of day 7 in May 1928 using simple HTTP requests."""
    
    print("Testing May 1928 day discovery...")
    print("=" * 50)
    
    # Test the month page
    url = "https://api.parliament.uk/historic-hansard/sittings/1928/may"
    
    try:
        print(f"Fetching: {url}")
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            print(f"✗ HTTP {response.status_code} - Cannot access month page")
            return
            
        html = response.text
        print(f"✓ Got {len(html)} characters of HTML")
        
        # Extract href links using regex (like our debug script)
        href_pattern = r'href="([^"]*)"'
        all_links = re.findall(href_pattern, html, re.I)
        
        print(f"✓ Found {len(all_links)} total links")
        
        # Look for May 1928 day patterns
        may_days = []
        may_links_found = []
        
        for link in all_links:
            if "/sittings/1928/may/" in link:
                may_links_found.append(link)
                # Extract day part
                parts = link.split("/sittings/1928/may/")
                if len(parts) > 1:
                    day_part = parts[1].split('/')[0]
                    if day_part.isdigit():
                        may_days.append(int(day_part))
        
        may_days = sorted(set(may_days))
        
        print(f"✓ Found {len(may_days)} days in May 1928: {may_days}")
        
        # Check for single-digit days
        single_digits = [d for d in may_days if d < 10]
        print(f"✓ Single-digit days: {single_digits}")
        
        if 7 in may_days:
            print("✓ Day 7 IS discoverable in May 1928")
        else:
            print("✗ Day 7 is NOT discoverable in May 1928")
            print("Sample May links found:")
            for link in may_links_found[:5]:
                print(f"  {link}")
        
        # Test direct access to day 7
        print(f"\nTesting direct access to day 7...")
        day_url = "https://api.parliament.uk/historic-hansard/sittings/1928/may/7"
        day_response = requests.get(day_url, timeout=30)
        
        if day_response.status_code == 200:
            print("✓ Day 7 is directly accessible")
            print(f"  Content length: {len(day_response.text)} characters")
        else:
            print(f"✗ Day 7 not accessible (HTTP {day_response.status_code})")
            
        # Save results to JSON for analysis
        results = {
            "month_url": url,
            "month_accessible": response.status_code == 200,
            "total_links_found": len(all_links),
            "may_days_discovered": may_days,
            "single_digit_days": single_digits,
            "day_7_discoverable": 7 in may_days,
            "day_7_url": day_url,
            "day_7_accessible": day_response.status_code == 200,
            "sample_may_links": may_links_found[:10]
        }
        
        with open('/tmp/may_1928_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to /tmp/may_1928_test_results.json")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_may_1928_discovery()