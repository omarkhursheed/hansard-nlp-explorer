#!/usr/bin/env python3
"""
Test high-performance processor against standard processor to ensure:
1. Same output quality 
2. Performance improvement
3. No data corruption
"""

import json
import time
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

def run_standard_processor(years, output_dir):
    """Run standard processor on test years."""
    print(f"🔧 Testing standard processor on years {years}...")
    
    start_time = time.time()
    
    # Run standard processing
    cmd = [
        sys.executable, 
        "scripts/process_full_dataset.py",
        "--start-year", str(min(years)),
        "--end-year", str(max(years)), 
        "--output", output_dir,
        "--raw-data", "data/hansard",
        "--batch-size", "5"  # Small batch for testing
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    elapsed = time.time() - start_time
    
    return {
        'success': result.returncode == 0,
        'time': elapsed,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'output_dir': output_dir
    }

def run_hp_processor(years, output_dir):
    """Run high-performance processor on test years."""
    print(f"🚀 Testing high-performance processor on years {years}...")
    
    start_time = time.time()
    
    # Run high-performance processing
    cmd = [
        sys.executable,
        "high_performance_processor.py", 
        "--start-year", str(min(years)),
        "--end-year", str(max(years)),
        "--output", output_dir,
        "--raw-data", "data/hansard",
        "--workers", "4"  # Reasonable for testing
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    elapsed = time.time() - start_time
    
    return {
        'success': result.returncode == 0,
        'time': elapsed,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'output_dir': output_dir
    }

def analyze_output_quality(output_dir, label):
    """Analyze the quality of processed output."""
    print(f"🔍 Analyzing {label} output quality...")
    
    stats = {
        'total_debates': 0,
        'debates_with_speakers': 0,
        'debates_with_topics': 0,
        'debates_with_nav_pollution': 0,
        'total_speakers': 0,
        'unique_speakers': set(),
        'sample_debates': [],
        'chambers': defaultdict(int),
        'word_count_dist': []
    }
    
    content_dir = Path(output_dir) / "content"
    
    if not content_dir.exists():
        print(f"❌ No content directory found in {output_dir}")
        return stats
    
    # Analyze all content files
    for year_dir in content_dir.iterdir():
        if year_dir.is_dir():
            jsonl_file = year_dir / f"debates_{year_dir.name}.jsonl"
            if jsonl_file.exists():
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            metadata = record.get('metadata', {})
                            
                            stats['total_debates'] += 1
                            
                            # Check speakers
                            speakers = metadata.get('speakers', [])
                            if speakers:
                                stats['debates_with_speakers'] += 1
                                stats['total_speakers'] += len(speakers)
                                stats['unique_speakers'].update(speakers)
                            
                            # Check topics
                            topic = metadata.get('debate_topic', '')
                            if topic and topic.strip():
                                stats['debates_with_topics'] += 1
                            
                            # Check navigation pollution
                            full_text = record.get('full_text', '')
                            if 'Back to' in full_text or 'Forward to' in full_text:
                                stats['debates_with_nav_pollution'] += 1
                            
                            # Chamber distribution
                            chamber = metadata.get('chamber', 'Unknown')
                            stats['chambers'][chamber] += 1
                            
                            # Word count distribution
                            word_count = metadata.get('word_count', 0)
                            if word_count > 0:
                                stats['word_count_dist'].append(word_count)
                            
                            # Collect sample debates
                            if len(stats['sample_debates']) < 5:
                                stats['sample_debates'].append({
                                    'title': metadata.get('title', ''),
                                    'topic': metadata.get('debate_topic', ''),
                                    'speakers': len(speakers),
                                    'chamber': chamber,
                                    'word_count': word_count,
                                    'has_nav_pollution': 'Back to' in full_text
                                })
    
    return stats

def compare_outputs(standard_stats, hp_stats):
    """Compare output quality between standard and HP processors."""
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"{'Metric':<30} {'Standard':<15} {'HP':<15} {'Match':<10}")
    print("-" * 70)
    
    matches = 0
    total_checks = 0
    
    # Compare key metrics
    comparisons = [
        ('Total debates', 'total_debates'),
        ('Debates with speakers', 'debates_with_speakers'), 
        ('Debates with topics', 'debates_with_topics'),
        ('Navigation pollution', 'debates_with_nav_pollution'),
        ('Total speakers', 'total_speakers'),
        ('Unique speakers', lambda s: len(s['unique_speakers']))
    ]
    
    for name, key in comparisons:
        if callable(key):
            std_val = key(standard_stats)
            hp_val = key(hp_stats)
        else:
            std_val = standard_stats.get(key, 0)
            hp_val = hp_stats.get(key, 0)
        
        match = "✅" if std_val == hp_val else "❌"
        if std_val == hp_val:
            matches += 1
        total_checks += 1
        
        print(f"{name:<30} {std_val:<15} {hp_val:<15} {match:<10}")
    
    # Compare chambers
    std_chambers = dict(standard_stats['chambers'])
    hp_chambers = dict(hp_stats['chambers'])
    chambers_match = std_chambers == hp_chambers
    if chambers_match:
        matches += 1
    total_checks += 1
    
    print(f"{'Chamber distribution':<30} {'Dict':<15} {'Dict':<15} {'✅' if chambers_match else '❌':<10}")
    
    # Compare word count distributions
    std_wc = standard_stats['word_count_dist']
    hp_wc = hp_stats['word_count_dist']
    
    if std_wc and hp_wc:
        std_avg = sum(std_wc) / len(std_wc)
        hp_avg = sum(hp_wc) / len(hp_wc)
        wc_match = abs(std_avg - hp_avg) < 1.0  # Allow small floating point differences
        if wc_match:
            matches += 1
        total_checks += 1
        
        print(f"{'Avg word count':<30} {std_avg:<15.1f} {hp_avg:<15.1f} {'✅' if wc_match else '❌':<10}")
    
    print("-" * 70)
    print(f"Overall match rate: {matches}/{total_checks} ({matches/total_checks*100:.1f}%)")
    
    return matches == total_checks

def show_sample_comparison(standard_stats, hp_stats):
    """Show sample debates from both processors."""
    print(f"\n📋 SAMPLE DEBATE COMPARISON:")
    
    print(f"\n🔧 Standard Processor Samples:")
    for i, debate in enumerate(standard_stats['sample_debates'][:3], 1):
        print(f"  {i}. {debate['title'][:60]}...")
        print(f"     Topic: '{debate['topic'][:40]}...' | Speakers: {debate['speakers']} | Nav pollution: {debate['has_nav_pollution']}")
    
    print(f"\n🚀 High-Performance Processor Samples:")
    for i, debate in enumerate(hp_stats['sample_debates'][:3], 1):
        print(f"  {i}. {debate['title'][:60]}...")
        print(f"     Topic: '{debate['topic'][:40]}...' | Speakers: {debate['speakers']} | Nav pollution: {debate['has_nav_pollution']}")

def main():
    """Run comprehensive test of both processors."""
    print("🧪 TESTING HIGH-PERFORMANCE PROCESSOR")
    print("="*60)
    
    # Test on a small sample - years with different characteristics
    test_years = [1925, 1926]  # Small sample for testing
    
    # Setup output directories
    standard_output = "data/test_standard"
    hp_output = "data/test_hp"
    
    # Clean up previous test runs
    import shutil
    for test_dir in [standard_output, hp_output]:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
    
    try:
        # Run both processors
        print(f"🎯 Testing on years: {test_years}")
        
        standard_result = run_standard_processor(test_years, standard_output)
        hp_result = run_hp_processor(test_years, hp_output)
        
        # Check if both succeeded
        if not standard_result['success']:
            print(f"❌ Standard processor failed:")
            print(f"   stdout: {standard_result['stdout']}")
            print(f"   stderr: {standard_result['stderr']}")
            return False
        
        if not hp_result['success']:
            print(f"❌ High-performance processor failed:")
            print(f"   stdout: {hp_result['stdout']}")
            print(f"   stderr: {hp_result['stderr']}")
            return False
        
        # Analyze output quality
        standard_stats = analyze_output_quality(standard_output, "Standard")
        hp_stats = analyze_output_quality(hp_output, "High-Performance")
        
        # Performance comparison
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        print(f"Standard time: {standard_result['time']:.1f} seconds")
        print(f"HP time: {hp_result['time']:.1f} seconds")
        if hp_result['time'] > 0:
            speedup = standard_result['time'] / hp_result['time']
            print(f"Speedup: {speedup:.1f}x {'🚀' if speedup > 1.5 else '⚠️' if speedup > 1.0 else '❌'}")
        
        # Quality comparison
        outputs_match = compare_outputs(standard_stats, hp_stats)
        
        # Show samples
        show_sample_comparison(standard_stats, hp_stats)
        
        # Final verdict
        print(f"\n🏁 TEST RESULTS:")
        print(f"   Output quality match: {'✅ PASS' if outputs_match else '❌ FAIL'}")
        print(f"   Performance improvement: {'✅ PASS' if hp_result['time'] < standard_result['time'] else '❌ FAIL'}")
        
        if outputs_match and hp_result['time'] < standard_result['time']:
            print(f"\n🎉 HIGH-PERFORMANCE PROCESSOR READY FOR PRODUCTION!")
            print(f"   Safe to use on full dataset with significant speedup")
        else:
            print(f"\n⚠️  ISSUES FOUND - REVIEW BEFORE PRODUCTION USE")
        
        return outputs_match
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test directories
        for test_dir in [standard_output, hp_output]:
            if Path(test_dir).exists():
                shutil.rmtree(test_dir)
                print(f"🧹 Cleaned up {test_dir}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)