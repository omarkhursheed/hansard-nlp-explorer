#!/usr/bin/env python3
"""Simple debug script to test day 7 May 1928 discovery."""

import requests
import re
try:
    from bs4 import BeautifulSoup
    has_bs4 = True
except ImportError:
    has_bs4 = False

def test_day_discovery():
    """Test if day 7 is discoverable in May 1928."""
    
    # Test the month page
    url = "https://api.parliament.uk/historic-hansard/sittings/1928/may"
    print(f"Fetching month page: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            html = response.text
            print(f"Content length: {len(html)}")
            
            # Look for links like the crawler does
            # Extract all href links
            href_pattern = r'href="([^"]*)"'
            all_links = re.findall(href_pattern, html, re.I)
            
            print(f"Found {len(all_links)} href links")
            
            # Look for day patterns specifically
            may_links = []
            for link in all_links:
                if "/sittings/1928/may/" in link:
                    # Extract the day part
                    parts = link.split("/sittings/1928/may/")
                    if len(parts) > 1:
                        day_part = parts[1].split('/')[0]
                        if day_part.isdigit():
                            may_links.append(int(day_part))
            
            may_links = sorted(set(may_links))
            print(f"Days found in May 1928 (regex): {may_links}")
            
            # Also test with BeautifulSoup like the crawler
            if has_bs4:
                soup = BeautifulSoup(html, 'html.parser')
                bs4_links = [a.get('href', '') for a in soup.find_all('a', href=True)]
                
                bs4_may_links = []
                for link in bs4_links:
                    if "/sittings/1928/may/" in link:
                        parts = link.split("/sittings/1928/may/")
                        if len(parts) > 1:
                            day_part = parts[1].split('/')[0]
                            if day_part.isdigit():
                                bs4_may_links.append(int(day_part))
                
                bs4_may_links = sorted(set(bs4_may_links))
                print(f"Days found in May 1928 (BS4): {bs4_may_links}")
                
                if 7 in bs4_may_links:
                    print("✓ Day 7 found in BS4 discovery!")
                else:
                    print("✗ Day 7 NOT found in BS4 discovery")
            
            if 7 in may_links:
                print("✓ Day 7 found in regex discovery!")
            else:
                print("✗ Day 7 NOT found in regex discovery")
                
                # Check some sample links to understand the pattern
                print("\nSample links containing 1928/may:")
                count = 0
                for link in all_links:
                    if "1928/may" in link:
                        print(f"  {link}")
                        count += 1
                        if count >= 10:
                            break
        
        # Also test direct access to day 7
        day_url = "https://api.parliament.uk/historic-hansard/sittings/1928/may/7"
        print(f"\nTesting direct access: {day_url}")
        
        day_response = requests.get(day_url, timeout=30)
        print(f"Status: {day_response.status_code}")
        
        if day_response.status_code == 200:
            print("✓ Day 7 is accessible directly")
        else:
            print("✗ Day 7 is not accessible directly")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_day_discovery()