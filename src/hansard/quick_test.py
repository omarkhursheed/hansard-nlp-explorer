#!/usr/bin/env python3
"""
Quick test of high-performance processor on a single year.
"""

import time
from high_performance_processor import process_year_parallel
from parsers.data_pipeline import HansardDataPipeline
import json
from pathlib import Path

def test_single_year():
    """Test HP processor on 1925 and compare with standard."""
    print("🧪 Quick Test: HP Processor vs Standard on 1925")
    print("=" * 50)
    
    year = 1925
    raw_data = "data/hansard"
    
    # Test standard processor
    print("🔧 Testing standard processor...")
    standard_start = time.time()
    
    pipeline = HansardDataPipeline(raw_data, "data/test_standard_quick")
    standard_result = pipeline.process_year(str(year))
    
    standard_time = time.time() - standard_start
    
    # Test HP processor
    print("🚀 Testing HP processor...")
    hp_start = time.time()
    
    hp_result = process_year_parallel(year, raw_data, "data/test_hp_quick", num_workers=4)
    
    hp_time = time.time() - hp_start
    
    # Compare results
    print(f"\n📊 RESULTS:")
    print(f"Standard processor:")
    print(f"  Time: {standard_time:.1f} seconds")
    print(f"  Processed: {standard_result.get('processed', 0)} files")
    print(f"  Errors: {standard_result.get('errors', 0)}")
    
    print(f"\nHP processor:")
    print(f"  Time: {hp_time:.1f} seconds") 
    print(f"  Processed: {hp_result.get('processed', 0)} files")
    print(f"  Errors: {hp_result.get('errors', 0)}")
    
    if hp_time > 0:
        speedup = standard_time / hp_time
        print(f"\nSpeedup: {speedup:.1f}x {'🚀' if speedup > 1.2 else '⚠️'}")
    
    # Quick quality check
    print(f"\n🔍 Quality Check:")
    
    # Check standard output
    std_content_file = Path("data/test_standard_quick/content/1925/debates_1925.jsonl")
    hp_content_file = Path("data/test_hp_quick/content/1925/debates_1925.jsonl")
    
    if std_content_file.exists() and hp_content_file.exists():
        # Count records
        with open(std_content_file, 'r') as f:
            std_records = sum(1 for line in f if line.strip())
        
        with open(hp_content_file, 'r') as f:
            hp_records = sum(1 for line in f if line.strip())
        
        print(f"  Standard records: {std_records}")
        print(f"  HP records: {hp_records}")
        print(f"  Match: {'✅' if std_records == hp_records else '❌'}")
        
        # Check sample record quality
        with open(hp_content_file, 'r') as f:
            sample_line = f.readline()
            if sample_line.strip():
                sample_record = json.loads(sample_line)
                metadata = sample_record.get('metadata', {})
                
                print(f"  Sample HP record:")
                print(f"    Title: {metadata.get('title', 'N/A')[:60]}...")
                print(f"    Topic: '{metadata.get('debate_topic', 'N/A')}'")
                print(f"    Speakers: {len(metadata.get('speakers', []))}")
                print(f"    Chamber: {metadata.get('chamber', 'N/A')}")
                print(f"    Word count: {metadata.get('word_count', 0)}")
                
                # Check for our fixes
                full_text = sample_record.get('full_text', '')
                has_nav = 'Back to' in full_text or 'Forward to' in full_text
                print(f"    Navigation pollution: {'❌ Present' if has_nav else '✅ Clean'}")
    
    # Cleanup
    import shutil
    for test_dir in ["data/test_standard_quick", "data/test_hp_quick"]:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
    
    print(f"\n🏁 Test complete!")

if __name__ == "__main__":
    test_single_year()