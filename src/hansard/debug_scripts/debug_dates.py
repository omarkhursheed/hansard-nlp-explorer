#!/usr/bin/env python3
"""Debug script to check what dates the Parliament API returns."""

import asyncio
import httpx
import re
from bs4 import BeautifulSoup

async def test_date_extraction():
    """Test what dates are returned by the Parliament API."""
    
    async with httpx.AsyncClient() as client:
        # Test February 1864
        url = "https://api.parliament.uk/historic-hansard/sittings/1864/feb"
        print(f"Fetching: {url}")
        
        try:
            response = await client.get(url)
            if response.status_code == 200:
                html = response.text
                print(f"Response length: {len(html)} characters")
                
                # Extract links using BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                links = [a.get('href', '') for a in soup.find_all('a', href=True)]
                
                print(f"Found {len(links)} total links")
                
                # Look for day patterns
                days_found = []
                for link in links:
                    if "/sittings/1864/feb/" in link:
                        parts = link.split("/sittings/1864/feb/")
                        if len(parts) > 1:
                            day_part = parts[1].split('/')[0]
                            if day_part.isdigit():
                                days_found.append(int(day_part))
                
                days_found = sorted(set(days_found))
                print(f"Days found: {days_found}")
                
                # Check specifically for single digit days
                single_digits = [d for d in days_found if d < 10]
                double_digits = [d for d in days_found if d >= 10]
                
                print(f"Single digit days: {single_digits}")
                print(f"Double digit days: {double_digits}")
                
                # Show some sample links
                print("\nSample links containing /sittings/1864/feb/:")
                count = 0
                for link in links:
                    if "/sittings/1864/feb/" in link:
                        print(f"  {link}")
                        count += 1
                        if count >= 10:
                            break
                            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_date_extraction())