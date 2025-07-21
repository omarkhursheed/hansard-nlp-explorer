#!/usr/bin/env python3
"""
Test the production processing script with a small sample.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.process_full_dataset import ProductionProcessor

def main():
    """Test with just 1803-1805."""
    print("🧪 Testing production processor with small sample...")
    
    processor = ProductionProcessor("../data/hansard", "../data/processed_test")
    
    # Test with just 3 years
    results = processor.process_full_dataset(
        start_year=1803, 
        end_year=1805, 
        batch_size=2  # Small batches for testing
    )
    
    print(f"\n✅ Test completed!")
    print(f"Processed years: {len(processor.processed_years)}")
    print(f"Failed years: {len(processor.failed_years)}")
    
    if processor.failed_years:
        print(f"Failed years: {processor.failed_years}")
    
    return len(processor.failed_years)

if __name__ == "__main__":
    sys.exit(main())