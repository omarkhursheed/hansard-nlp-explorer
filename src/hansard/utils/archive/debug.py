#!/usr/bin/env python3
"""Debug script to investigate Hansard API responses and parsing."""

import asyncio
import httpx
import re
from bs4 import BeautifulSoup

try:
    from selectolax.parser import HTMLParser as _HTML
    _USE_SELECTOLAX = True
    print("✓ Using selectolax for HTML parsing")
except ImportError:
    from bs4 import BeautifulSoup as _BS
    _HTML = None
    _USE_SELECTOLAX = False
    print("✓ Using BeautifulSoup for HTML parsing")

async def debug_decade_discovery():
    """Debug the decade discovery for 1860s."""
    
    base_url = "https://api.parliament.uk/historic-hansard"
    decade = "1860s"
    url = f"{base_url}/sittings/{decade}"
    
    print(f"\n🔍 Fetching: {url}")
    
    async with httpx.AsyncClient(
        http2=True, 
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "HansardDebugger/1.0"}
    ) as client:
        
        try:
            response = await client.get(url)
            print(f"📡 Status: {response.status_code}")
            print(f"📄 Content-Type: {response.headers.get('content-type', 'unknown')}")
            print(f"📏 Content-Length: {len(response.text)} chars")
            
            if response.status_code != 200:
                print(f"❌ HTTP error: {response.status_code}")
                return
            
            html = response.text
            
            # Save raw HTML for inspection
            with open("debug_1860s_raw.html", "w", encoding="utf-8") as f:
                f.write(html)
            print(f"💾 Saved raw HTML to debug_1860s_raw.html")
            
            # Show first 1000 chars
            print(f"\n📖 First 1000 chars of HTML:")
            print(html[:1000])
            print("...")
            
            # Parse with both parsers if available
            print(f"\n🔧 Parsing HTML...")
            
            if _USE_SELECTOLAX:
                print("\n--- SELECTOLAX PARSING ---")
                parser = _HTML(html)
                anchors = parser.css("a")
                print(f"Found {len(anchors)} anchor tags")
                
                hrefs = []
                for i, a in enumerate(anchors[:20]):  # First 20 links
                    href = a.attributes.get("href", "")
                    hrefs.append(href)
                    print(f"  {i+1:2d}. {href}")
                
                if len(anchors) > 20:
                    print(f"  ... and {len(anchors) - 20} more")
            
            print("\n--- BEAUTIFULSOUP PARSING ---")
            soup = BeautifulSoup(html, "html.parser")  # Use built-in parser
            bs_anchors = soup.find_all("a", href=True)
            print(f"Found {len(bs_anchors)} anchor tags with href")
            
            bs_hrefs = []
            for i, a in enumerate(bs_anchors[:20]):  # First 20 links
                href = a.get("href", "")
                bs_hrefs.append(href)
                print(f"  {i+1:2d}. {href}")
            
            if len(bs_anchors) > 20:
                print(f"  ... and {len(bs_anchors) - 20} more")
            
            # Look for sitting patterns
            print(f"\n🎯 Looking for sitting day patterns...")
            
            sitting_hrefs = []
            for href in bs_hrefs:
                if "/sittings/" in href:
                    sitting_hrefs.append(href)
            
            print(f"Found {len(sitting_hrefs)} hrefs containing '/sittings/':")
            for href in sitting_hrefs:
                print(f"  • {href}")
            
            # Extract date paths
            print(f"\n📅 Extracting date paths...")
            
            date_paths = []
            for href in sitting_hrefs:
                if "/sittings/" in href:
                    path = href.split("/sittings/")[-1]
                    parts = path.split("/")
                    print(f"  Path: '{path}' -> Parts: {parts}")
                    
                    # Check if it looks like year/month/day
                    if len(parts) >= 3:
                        year_part = parts[0]
                        if year_part.isdigit() and len(year_part) == 4:
                            year = int(year_part)
                            if 1860 <= year <= 1869:  # 1860s decade
                                date_paths.append(path)
                                print(f"    ✓ Valid date path: {path}")
                            else:
                                print(f"    ✗ Year {year} not in 1860s")
                        else:
                            print(f"    ✗ Invalid year part: '{year_part}'")
                    else:
                        print(f"    ✗ Not enough path components: {len(parts)}")
            
            print(f"\n📊 Summary:")
            print(f"  Total anchors: {len(bs_anchors)}")
            print(f"  Sitting hrefs: {len(sitting_hrefs)}")
            print(f"  Valid 1860s date paths: {len(date_paths)}")
            
            if date_paths:
                print(f"\n✅ Found date paths for 1860s:")
                for path in sorted(date_paths):
                    print(f"  • {path}")
                    
                # Check specifically for 1864
                year_1864_paths = [p for p in date_paths if p.startswith("1864")]
                print(f"\n🎯 1864 specific paths: {len(year_1864_paths)}")
                for path in year_1864_paths:
                    print(f"  • {path}")
            else:
                print(f"\n❌ No valid date paths found!")
                
                # Let's check what patterns we actually see
                print(f"\n🔍 Let's analyze the href patterns:")
                href_patterns = {}
                for href in bs_hrefs:
                    # Extract pattern
                    if "/sittings/" in href:
                        after_sittings = href.split("/sittings/")[-1]
                        pattern_parts = after_sittings.split("/")
                        if len(pattern_parts) >= 1:
                            pattern = f"{len(pattern_parts)} parts: {pattern_parts[0]}"
                            if pattern not in href_patterns:
                                href_patterns[pattern] = []
                            href_patterns[pattern].append(after_sittings)
                
                for pattern, examples in href_patterns.items():
                    print(f"  {pattern}: {len(examples)} examples")
                    for ex in examples[:3]:
                        print(f"    • {ex}")
                    if len(examples) > 3:
                        print(f"    ... and {len(examples) - 3} more")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

async def debug_year_discovery():
    """Debug what happens when we go into a specific year like 1864."""
    
    base_url = "https://api.parliament.uk/historic-hansard"
    year = "1864"
    url = f"{base_url}/sittings/{year}"
    
    print(f"\n🔍 Fetching year page: {url}")
    
    async with httpx.AsyncClient(
        http2=True, 
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "HansardDebugger/1.0"}
    ) as client:
        
        try:
            response = await client.get(url)
            print(f"📡 Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ HTTP error: {response.status_code}")
                return
            
            html = response.text
            print(f"📏 Content-Length: {len(html)} chars")
            
            # Save for inspection
            with open("debug_1864_raw.html", "w", encoding="utf-8") as f:
                f.write(html)
            print(f"💾 Saved raw HTML to debug_1864_raw.html")
            
            # Parse links
            soup = BeautifulSoup(html, "html.parser")
            anchors = soup.find_all("a", href=True)
            print(f"Found {len(anchors)} anchor tags with href")
            
            # Look for date-like patterns
            date_links = []
            for a in anchors:
                href = a.get("href", "")
                text = a.get_text(strip=True)
                
                # Look for links that might be dates
                if any(month in href.lower() for month in ["jan", "feb", "mar", "apr", "may", "jun", 
                                                          "jul", "aug", "sep", "oct", "nov", "dec"]):
                    date_links.append((href, text))
                elif any(month in text.lower() for month in ["january", "february", "march", "april", 
                                                            "may", "june", "july", "august", "september", 
                                                            "october", "november", "december"]):
                    date_links.append((href, text))
                elif re.search(r'\d{1,2}', text):  # Any numbers (might be days)
                    date_links.append((href, text))
            
            print(f"\n📅 Potential date links found: {len(date_links)}")
            for href, text in date_links[:20]:  # First 20
                print(f"  • {href} -> '{text}'")
            
            if len(date_links) > 20:
                print(f"  ... and {len(date_links) - 20} more")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("=== DEBUGGING HANSARD CRAWLER ===")
    asyncio.run(debug_decade_discovery())
    print("\n" + "="*50)
    asyncio.run(debug_year_discovery())