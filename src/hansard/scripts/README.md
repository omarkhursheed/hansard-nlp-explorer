# 🏛️ Hansard Full Dataset Processing

Production scripts for processing the complete Hansard dataset (1803-2005) with **673,385 files** across **202 years**.

## 📊 Processing Specifications

- **Total Files**: 673,385 debates
- **Storage Required**: ~14.5 GB (13.5GB JSON + 1GB metadata/index)
- **Estimated Time**: 10-20 hours (depending on hardware)
- **Memory Usage**: ~2-4 GB RAM during processing

## 🚀 Quick Start (Recommended)

```bash
cd src/hansard/scripts

# Launch in tmux session with full monitoring
./run_full_processing.sh
```

This will:
- Create a dedicated tmux session called `hansard-processing`
- Process all 202 years in batches of 10
- Log everything to `../data/processed/processing.log`
- Auto-save checkpoints for recovery
- Provide real-time progress tracking

## 🔧 Manual Execution

```bash
# Process full dataset (1803-2005)
python process_full_dataset.py

# Custom year range
python process_full_dataset.py --start-year 1850 --end-year 1900

# Smaller batches for slower systems
python process_full_dataset.py --batch-size 5

# Custom paths
python process_full_dataset.py --raw-data /path/to/raw --output /path/to/processed
```

## 📈 Monitoring Progress

### In tmux session:
```bash
# Attach to running session
tmux attach -t hansard-processing

# Detach without stopping: Ctrl+b, then d
```

### External monitoring:
```bash
# View processing log in real-time
tail -f ../data/processed/processing.log

# Check session status
tmux list-sessions | grep hansard-processing

# Monitor disk usage
du -sh ../data/processed

# Check system resources
htop
```

## 🛠️ Recovery & Checkpoints

The processor automatically saves checkpoints after each batch:

```bash
# View checkpoint status
cat ../data/processed/processing_checkpoint.json

# Resume from interruption (automatically detects checkpoint)
python process_full_dataset.py
```

## 📁 Output Structure

```
data/processed/
├── metadata/                          # Structured metadata (Parquet)
│   ├── debates_master.parquet        # ~318 MB - All debate metadata
│   └── speakers_master.parquet       # ~50 MB - Speaker information
├── content/                           # Full text content (JSON Lines)
│   ├── 1803/debates_1803.jsonl      # ~1 MB per year
│   ├── 1804/debates_1804.jsonl
│   └── .../...
├── index/                            # Search indices
│   └── debates.db                    # ~698 MB - SQLite FTS index
├── manifest.json                     # Processing provenance
├── validation_report.json            # Data quality report
└── full_processing_results.json      # Complete processing log
```

## 🎯 Performance Optimization

### For faster processing:
```bash
# Larger batches (if you have enough RAM)
python process_full_dataset.py --batch-size 20

# Process specific decades
python process_full_dataset.py --start-year 1900 --end-year 1999
```

### For slower systems:
```bash
# Smaller batches
python process_full_dataset.py --batch-size 5

# Process subset first
python process_full_dataset.py --start-year 1803 --end-year 1850
```

## 🔍 Testing Before Full Run

```bash
# Test with small sample (3 years)
python test_production_script.py

# Test specific year range
python process_full_dataset.py --start-year 1803 --end-year 1805
```

## 🚨 Error Handling

The processor includes:
- **Automatic retry** for failed years (up to 2 retries)
- **Graceful shutdown** on Ctrl+C or system signals
- **Recovery from checkpoints** if interrupted
- **Detailed error logging** for debugging

### Common issues:
1. **Disk space**: Ensure 15+ GB free space
2. **Memory**: Close other applications if processing fails
3. **Network**: Raw data must be accessible locally

## 📊 Expected Results

Based on testing, you should see:
- **~100% parsing success rate** across all years
- **Rich metadata extraction**: Hansard refs, speakers, topics
- **Fast search capabilities**: Sub-second queries across 673K files
- **Complete provenance tracking**: Every file traceable to source

## 🎉 After Completion

Once processing finishes:

```bash
# Test the search interface
cd ../parsers
python hansard_search.py

# Check data quality
python data_validation.py

# Explore with custom queries
python3 -c "
from hansard_search import HansardSearch
search = HansardSearch('../data/processed')
results = search.search_debates('your query here', limit=10)
print(results[['file_name', 'title', 'chamber', 'year']])
"
```

Your complete parliamentary archive is now ready for analysis! 🏛️