#!/usr/bin/env python3
"""
Display actual content from 1803 November Hansard data
"""

import gzip
import glob
import os
from bs4 import BeautifulSoup

def parse_and_show_content(file_path, max_lines=20):
    """Parse a file and show its content"""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            html = f.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title').get_text() if soup.find('title') else "No title"
        
        content_div = soup.find('div', class_='house-of-commons-sitting')
        
        if content_div:
            for unwanted in content_div(['nav', 'footer', 'script', 'style']):
                unwanted.decompose()
            
            content_text = content_div.get_text(separator='\n', strip=True)
            lines = [line for line in content_text.split('\n') if line.strip()]
            
            return title, lines
        else:
            return title, None
            
    except Exception as e:
        return f"Error: {e}", None

def main():
    # Show content from a few interesting files
    interesting_files = [
        "../data/hansard/1803/nov/22_01_kings-speech.html.gz",
        "../data/hansard/1803/nov/28_01_income-tax.html.gz", 
        "../data/hansard/1803/nov/30_02_bank-restriction-bill.html.gz",
        "../data/hansard/1803/nov/25_02_desertion-of-seamen.html.gz"
    ]
    
    for file_path in interesting_files:
        if os.path.exists(file_path):
            print("\n" + "="*80)
            title, lines = parse_and_show_content(file_path)
            print(f"FILE: {os.path.basename(file_path)}")
            print(f"TITLE: {title}")
            print("-"*80)
            
            if lines:
                # Show first 15 lines
                for i, line in enumerate(lines[:15]):
                    print(f"{i+1:2d}: {line}")
                if len(lines) > 15:
                    print(f"... ({len(lines) - 15} more lines)")
            else:
                print("No content found")
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()