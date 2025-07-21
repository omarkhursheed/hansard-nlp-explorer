#!/usr/bin/env python3
"""
Minimal Hansard parser - start with one file
"""

import gzip
from bs4 import BeautifulSoup

# Parse one file to see what we get
file_path = "data/hansard/1803/dec/10_00_minutes.html.gz"

with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')

# Basic info
title = soup.find('title').get_text() if soup.find('title') else "No title"
print(f"Title: {title}")

# Extract the main content
content_div = soup.find('div', class_='house-of-commons-sitting')

if content_div:
    # Remove navigation and footer elements
    for unwanted in content_div(['nav', 'footer', 'script', 'style']):
        unwanted.decompose()
    
    content_text = content_div.get_text(separator='\n', strip=True)
    lines = [line for line in content_text.split('\n') if line.strip()]
    
    print(f"\nExtracted {len(lines)} lines of parliamentary content:")
    print("="*50)
    for line in lines:
        print(line)
    print("="*50)
else:
    print("No main content div found")