#!/usr/bin/env python3
"""
Test the fixed data pipeline on sample files.
"""

from pathlib import Path
from parsers.data_pipeline import HansardDataPipeline

def test_fixed_pipeline():
    """Test the pipeline fixes on sample files."""
    print("=== TESTING FIXED PIPELINE ===")
    
    # Initialize pipeline
    pipeline = HansardDataPipeline("data/hansard", "data/processed_test")
    
    # Test files we examined earlier
    test_files = [
        Path("data/hansard/1925/mar/12_17_toy-pistols.html.gz"),  # Multiple speakers
        Path("data/hansard/1925/mar/18_10_housing-scotland-bill-hl.html.gz"),  # Lords simple
        Path("data/hansard/1925/mar/24_65_coal-industry.html.gz"),  # Short summary
    ]
    
    for file_path in test_files:
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue
            
        print(f"\n📄 Testing: {file_path.name}")
        
        # Extract metadata with our fixes
        result = pipeline.extract_comprehensive_metadata(file_path)
        
        if result.get('success'):
            metadata = result
            print(f"✅ Processing successful")
            print(f"   Title: {metadata.get('title', 'N/A')}")
            print(f"   Debate topic: '{metadata.get('debate_topic', 'N/A')}'")
            print(f"   Chamber: {metadata.get('chamber', 'N/A')}")
            print(f"   Speakers: {len(metadata.get('speakers', []))} found")
            if metadata.get('speakers'):
                for speaker in metadata['speakers'][:3]:  # Show first 3
                    print(f"     • {speaker}")
            print(f"   Word count: {metadata.get('word_count', 0)}")
            print(f"   Reference: {metadata.get('hansard_reference', 'N/A')}")
            
            # Check for navigation pollution
            full_text = metadata.get('full_text', '')
            has_back_to = 'Back to' in full_text
            has_forward_to = 'Forward to' in full_text
            
            if has_back_to or has_forward_to:
                print(f"   ⚠️  Navigation pollution: {'Back to' if has_back_to else ''} {'Forward to' if has_forward_to else ''}")
            else:
                print(f"   ✅ No navigation pollution")
                
            # Show last few lines to check
            lines = metadata.get('content_lines', [])
            if len(lines) > 3:
                print(f"   Last 3 lines:")
                for line in lines[-3:]:
                    print(f"     {line}")
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n=== TESTING COMPLETE ===")

if __name__ == "__main__":
    test_fixed_pipeline()