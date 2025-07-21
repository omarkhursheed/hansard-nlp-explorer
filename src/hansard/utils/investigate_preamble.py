#!/usr/bin/env python3
"""
Investigate the HTML structure of preamble files
"""

import gzip
from bs4 import BeautifulSoup

def investigate_file(file_path):
    """Examine the HTML structure of a file"""
    print(f"\n{'='*60}")
    print(f"INVESTIGATING: {file_path}")
    print('='*60)
    
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            html = f.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Show title
        title = soup.find('title')
        print(f"TITLE: {title.get_text() if title else 'No title'}")
        
        # Look for different div classes
        print("\nDIV CLASSES FOUND:")
        divs = soup.find_all('div', class_=True)
        for div in divs[:10]:  # Show first 10
            print(f"  - {div.get('class')}")
        
        # Check for main content div
        main_content = soup.find('div', class_='house-of-commons-sitting')
        print(f"\nHOUSE-OF-COMMONS-SITTING DIV: {'Found' if main_content else 'NOT FOUND'}")
        
        # Look for other potential content containers
        potential_containers = [
            'hansard-content',
            'content',
            'main-content',
            'debate-content',
            'parliamentary-content'
        ]
        
        print("\nOTHER POTENTIAL CONTAINERS:")
        for container_class in potential_containers:
            element = soup.find('div', class_=container_class)
            print(f"  - {container_class}: {'Found' if element else 'Not found'}")
        
        # Show first few divs with text content
        print(f"\nFIRST FEW DIVS WITH TEXT:")
        divs_with_text = soup.find_all('div')
        for i, div in enumerate(divs_with_text[:5]):
            text = div.get_text(strip=True)
            if text and len(text) > 20:
                print(f"  {i+1}. Class: {div.get('class', 'No class')}")
                print(f"     Text: {text[:100]}...")
        
        # Show raw HTML structure (first 1000 chars)
        print(f"\nRAW HTML SAMPLE:")
        print(html[:1000])
        
    except Exception as e:
        print(f"ERROR: {e}")

def main():
    # Investigate the failed preamble files
    preamble_files = [
        "data/hansard/1803/nov/22_00_preamble.html.gz",
        "data/hansard/1803/nov/23_00_preamble.html.gz"
    ]
    
    for file_path in preamble_files:
        investigate_file(file_path)
        
    # Also check a working file for comparison
    print(f"\n{'='*60}")
    print("FOR COMPARISON - WORKING FILE:")
    investigate_file("data/hansard/1803/nov/22_01_kings-speech.html.gz")

if __name__ == "__main__":
    main()