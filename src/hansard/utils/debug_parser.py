#!/usr/bin/env python3
"""
Deep dive analysis of Hansard parsing issues.
Examines raw HTML vs processed output to identify problems.
"""

import gzip
import json
from pathlib import Path
from bs4 import BeautifulSoup
import re
from collections import Counter, defaultdict
import random

class HansardParsingAnalyzer:
    def __init__(self, raw_data_path="data/hansard", processed_data_path="data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.issues = defaultdict(list)
    
    def analyze_html_structure_patterns(self, sample_files=20):
        """Analyze HTML structure across different years and file types."""
        print("=== ANALYZING HTML STRUCTURE PATTERNS ===")
        
        # Get sample files from different years
        sample_files_list = []
        for year in [1803, 1850, 1900, 1925, 1950, 2000]:
            year_dir = self.raw_data_path / str(year)
            if year_dir.exists():
                gz_files = list(year_dir.rglob("*.html.gz"))
                if gz_files:
                    sample_files_list.extend(random.sample(gz_files, min(3, len(gz_files))))
        
        # Analyze structure patterns
        structure_stats = {
            'total_files': 0,
            'commons_divs': 0,
            'lords_divs': 0,
            'no_main_div': 0,
            'member_contributions': Counter(),
            'cite_patterns': Counter(),
            'title_patterns': Counter(),
            'meta_tags': Counter()
        }
        
        print(f"Analyzing {len(sample_files_list)} files across different years...")
        
        for file_path in sample_files_list:
            try:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    html = f.read()
                
                soup = BeautifulSoup(html, 'html.parser')
                structure_stats['total_files'] += 1
                
                # Check main content divs
                commons_div = soup.find('div', class_='house-of-commons-sitting')
                lords_div = soup.find('div', class_='house-of-lords-sitting')
                
                if commons_div:
                    structure_stats['commons_divs'] += 1
                elif lords_div:
                    structure_stats['lords_divs'] += 1
                else:
                    structure_stats['no_main_div'] += 1
                    self.issues['no_main_div'].append(str(file_path))
                
                # Analyze member contributions
                main_div = commons_div or lords_div
                if main_div:
                    contributions = main_div.find_all('div', class_='member_contribution')
                    structure_stats['member_contributions'][len(contributions)] += 1
                    
                    # Check citation patterns
                    cites = main_div.find_all('cite', class_='member')
                    for cite in cites:
                        cite_text = cite.get_text(strip=True)
                        # Extract pattern (Mr., Mrs., Lord, etc.)
                        if cite_text:
                            pattern = re.match(r'^([A-Za-z.-]+)', cite_text)
                            if pattern:
                                structure_stats['cite_patterns'][pattern.group(1)] += 1
                
                # Check title patterns
                title = soup.find('title')
                if title:
                    title_text = title.get_text()
                    structure_stats['title_patterns'][title_text.split('.')[-1].strip()] += 1
                
                # Check meta tags
                for meta in soup.find_all('meta'):
                    name = meta.get('name') or meta.get('property')
                    if name:
                        structure_stats['meta_tags'][name] += 1
                        
            except Exception as e:
                self.issues['parsing_errors'].append(f"{file_path}: {e}")
        
        # Report findings
        print(f"\n📊 STRUCTURE ANALYSIS RESULTS:")
        print(f"Files analyzed: {structure_stats['total_files']}")
        print(f"Commons divs: {structure_stats['commons_divs']}")
        print(f"Lords divs: {structure_stats['lords_divs']}")
        print(f"No main div: {structure_stats['no_main_div']}")
        
        print(f"\n🎭 Member contribution patterns:")
        for count, freq in structure_stats['member_contributions'].most_common(10):
            print(f"  {count} contributions: {freq} files")
        
        print(f"\n👤 Speaker title patterns:")
        for pattern, count in structure_stats['cite_patterns'].most_common(15):
            print(f"  {pattern}: {count}")
        
        print(f"\n🏷️  Meta tag patterns:")
        for tag, count in structure_stats['meta_tags'].most_common(10):
            print(f"  {tag}: {count}")
        
        return structure_stats
    
    def check_text_extraction_quality(self, sample_files=10):
        """Compare raw HTML content vs extracted text to find losses."""
        print(f"\n=== CHECKING TEXT EXTRACTION QUALITY ===")
        
        # Get some processed files to compare
        processed_files = list((self.processed_data_path / "content").rglob("*.jsonl"))
        if not processed_files:
            print("❌ No processed files found!")
            return
        
        sample_processed = random.sample(processed_files, min(2, len(processed_files)))
        
        extraction_issues = {
            'html_artifacts': 0,
            'missing_structure': 0,
            'encoding_issues': 0,
            'content_gaps': 0,
            'navigation_pollution': 0
        }
        
        for processed_file in sample_processed:
            print(f"\n📄 Analyzing: {processed_file.name}")
            
            with open(processed_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= sample_files:  # Limit samples per file
                        break
                    
                    if line.strip():
                        record = json.loads(line)
                        file_path = Path(record['file_path'])
                        
                        # Check if raw file exists
                        if not file_path.exists():
                            continue
                        
                        try:
                            # Load raw HTML
                            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                                raw_html = f.read()
                            
                            # Analyze extraction quality
                            soup = BeautifulSoup(raw_html, 'html.parser')
                            extracted_text = record.get('full_text', '')
                            
                            # Check for HTML artifacts in extracted text
                            if any(artifact in extracted_text.lower() for artifact in ['<', '>', '&lt;', '&gt;', '&amp;']):
                                extraction_issues['html_artifacts'] += 1
                                self.issues['html_artifacts'].append(record['file_name'])
                            
                            # Check for navigation pollution
                            nav_indicators = ['Back to', 'Forward to', 'Search', 'Navigate']
                            if any(nav in extracted_text for nav in nav_indicators):
                                extraction_issues['navigation_pollution'] += 1
                            
                            # Check for encoding issues
                            if '�' in extracted_text or 'â€' in extracted_text:
                                extraction_issues['encoding_issues'] += 1
                                self.issues['encoding_issues'].append(record['file_name'])
                            
                            # Check for missing structure (speeches without speakers)
                            lines = record.get('lines', [])
                            speech_lines = [l for l in lines if len(l.split()) > 5]  # Substantial lines
                            if len(speech_lines) > 3 and record.get('speaker_count', 0) == 0:
                                extraction_issues['missing_structure'] += 1
                                
                        except Exception as e:
                            self.issues['extraction_errors'].append(f"{file_path}: {e}")
        
        print(f"\n🔍 TEXT EXTRACTION ISSUES:")
        for issue, count in extraction_issues.items():
            print(f"  {issue}: {count} files")
        
        return extraction_issues
    
    def analyze_metadata_extraction(self, sample_size=15):
        """Check quality of metadata extraction."""
        print(f"\n=== ANALYZING METADATA EXTRACTION ===")
        
        metadata_issues = {
            'missing_dates': 0,
            'invalid_dates': 0,
            'missing_references': 0,
            'invalid_references': 0,
            'missing_chambers': 0,
            'word_count_zeros': 0,
            'empty_titles': 0
        }
        
        # Sample processed files
        processed_files = list((self.processed_data_path / "content").rglob("*.jsonl"))
        sample_processed = random.sample(processed_files, min(3, len(processed_files)))
        
        date_patterns = []
        reference_patterns = []
        chamber_counts = Counter()
        word_count_dist = []
        
        total_records = 0
        
        for processed_file in sample_processed:
            with open(processed_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= sample_size:  # Limit per file
                        break
                    
                    if line.strip():
                        record = json.loads(line)
                        total_records += 1
                        metadata = record.get('metadata', {})
                        
                        # Check date extraction
                        ref_date = metadata.get('reference_date')
                        if not ref_date:
                            metadata_issues['missing_dates'] += 1
                        else:
                            date_patterns.append(ref_date)
                            # Check date format validity
                            if not re.match(r'\d{1,2} \w+ \d{4}', ref_date):
                                metadata_issues['invalid_dates'] += 1
                                self.issues['invalid_dates'].append(f"{record['file_name']}: {ref_date}")
                        
                        # Check Hansard reference
                        hansard_ref = metadata.get('hansard_reference')
                        if not hansard_ref:
                            metadata_issues['missing_references'] += 1
                        else:
                            reference_patterns.append(hansard_ref)
                            # Check reference format
                            if not re.match(r'(HC|HL) Deb', hansard_ref):
                                metadata_issues['invalid_references'] += 1
                                self.issues['invalid_references'].append(f"{record['file_name']}: {hansard_ref}")
                        
                        # Check chamber
                        chamber = metadata.get('chamber')
                        if not chamber:
                            metadata_issues['missing_chambers'] += 1
                        else:
                            chamber_counts[chamber] += 1
                        
                        # Check word counts
                        word_count = metadata.get('word_count', 0)
                        if word_count == 0:
                            metadata_issues['word_count_zeros'] += 1
                        word_count_dist.append(word_count)
                        
                        # Check titles
                        title = metadata.get('title')
                        if not title or title.strip() == '':
                            metadata_issues['empty_titles'] += 1
        
        print(f"\n📊 METADATA QUALITY RESULTS ({total_records} records):")
        for issue, count in metadata_issues.items():
            pct = count / total_records * 100 if total_records > 0 else 0
            print(f"  {issue}: {count} ({pct:.1f}%)")
        
        print(f"\n🏛️  Chamber distribution:")
        for chamber, count in chamber_counts.most_common():
            print(f"  {chamber}: {count}")
        
        print(f"\n📅 Sample date patterns:")
        for date in random.sample(date_patterns, min(5, len(date_patterns))):
            print(f"  {date}")
        
        print(f"\n📖 Sample reference patterns:")
        for ref in random.sample(reference_patterns, min(5, len(reference_patterns))):
            print(f"  {ref}")
        
        if word_count_dist:
            print(f"\n📝 Word count stats:")
            print(f"  Mean: {sum(word_count_dist)/len(word_count_dist):.0f}")
            print(f"  Min: {min(word_count_dist)}")
            print(f"  Max: {max(word_count_dist)}")
        
        return metadata_issues
    
    def test_different_document_types(self):
        """Test parsing on different types of parliamentary documents."""
        print(f"\n=== TESTING DIFFERENT DOCUMENT TYPES ===")
        
        # Look for different document patterns in titles
        doc_patterns = {
            'bills': ['BILL', 'Bill'],
            'questions': ['asked the', 'QUESTION'],
            'debates': ['DEBATE', 'Debate'],
            'statements': ['STATEMENT', 'Statement'],
            'committees': ['COMMITTEE', 'Committee'],
            'divisions': ['DIVISION', 'Division']
        }
        
        type_stats = defaultdict(int)
        type_examples = defaultdict(list)
        
        # Sample across multiple years
        years_to_check = [1850, 1900, 1950, 2000]
        
        for year in years_to_check:
            processed_file = self.processed_data_path / "content" / str(year) / f"debates_{year}.jsonl"
            
            if processed_file.exists():
                with open(processed_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 50:  # Limit per year
                            break
                        
                        if line.strip():
                            record = json.loads(line)
                            title = record.get('metadata', {}).get('title', '')
                            
                            # Classify document type
                            doc_type = 'unknown'
                            for type_name, patterns in doc_patterns.items():
                                if any(pattern in title for pattern in patterns):
                                    doc_type = type_name
                                    break
                            
                            type_stats[doc_type] += 1
                            if len(type_examples[doc_type]) < 3:
                                type_examples[doc_type].append({
                                    'title': title[:100],
                                    'year': year,
                                    'speakers': len(record.get('metadata', {}).get('speakers', [])),
                                    'word_count': record.get('metadata', {}).get('word_count', 0)
                                })
        
        print(f"📋 Document type distribution:")
        for doc_type, count in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {doc_type}: {count}")
        
        print(f"\n📄 Document type examples:")
        for doc_type, examples in type_examples.items():
            if examples:
                print(f"\n  {doc_type.upper()}:")
                for ex in examples:
                    print(f"    • {ex['year']}: {ex['title']} (speakers: {ex['speakers']}, words: {ex['word_count']})")
        
        return type_stats, type_examples
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive report of all parsing issues."""
        print(f"\n" + "="*80)
        print(f"🔍 COMPREHENSIVE HANSARD PARSING ANALYSIS")
        print(f"="*80)
        
        # Run all analyses
        structure_stats = self.analyze_html_structure_patterns()
        extraction_issues = self.check_text_extraction_quality()
        metadata_issues = self.analyze_metadata_extraction()
        doc_type_stats, doc_examples = self.test_different_document_types()
        
        # Summarize critical issues
        print(f"\n🚨 CRITICAL ISSUES FOUND:")
        critical_issues = []
        
        if len(self.issues['no_main_div']) > 0:
            critical_issues.append(f"❌ Files with no main content div: {len(self.issues['no_main_div'])}")
        
        if extraction_issues['html_artifacts'] > 0:
            critical_issues.append(f"❌ HTML artifacts in text: {extraction_issues['html_artifacts']}")
        
        if extraction_issues['encoding_issues'] > 0:
            critical_issues.append(f"❌ Encoding issues: {extraction_issues['encoding_issues']}")
        
        if metadata_issues['missing_dates'] > 5:
            critical_issues.append(f"❌ Missing dates: {metadata_issues['missing_dates']}")
        
        if metadata_issues['word_count_zeros'] > 5:
            critical_issues.append(f"❌ Zero word counts: {metadata_issues['word_count_zeros']}")
        
        if critical_issues:
            for issue in critical_issues:
                print(f"  {issue}")
        else:
            print("  ✅ No critical parsing issues detected!")
        
        # Print specific problematic files
        if self.issues:
            print(f"\n🗂️  PROBLEMATIC FILES:")
            for issue_type, files in self.issues.items():
                if files and len(files) <= 5:  # Show only if few files
                    print(f"  {issue_type}:")
                    for file_name in files[:3]:  # Show max 3
                        print(f"    • {file_name}")
        
        print(f"\n✅ ANALYSIS COMPLETE")
        return {
            'structure_stats': structure_stats,
            'extraction_issues': extraction_issues,
            'metadata_issues': metadata_issues,
            'document_types': doc_type_stats,
            'critical_issues': critical_issues,
            'problematic_files': dict(self.issues)
        }

def main():
    analyzer = HansardParsingAnalyzer()
    report = analyzer.generate_comprehensive_report()
    return report

if __name__ == "__main__":
    main()