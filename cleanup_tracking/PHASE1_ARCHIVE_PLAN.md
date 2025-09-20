# Phase 1: Archive Plan

## Archive Structure to Create:
```
archive/
├── reports/
│   └── Hansard Report V1 - Draft 08_03_2025 (3).pdf
├── examples/
│   ├── debate_simple_example.json
│   ├── debate_types_test.json
│   ├── debate_visualization_example.py
│   ├── test_debate_types.py
│   └── example_usage.py
├── logs/
│   └── create_samples.log
├── scripts/
│   ├── process_debates_sample.py
│   └── create_sampled_datasets.py
└── documentation/
    ├── ANALYSIS_README.md
    ├── SPEAKER_DATASET_GUIDE.md
    └── old_guides/
```

## Files to Archive:

### High Priority (Large/Non-Essential):
- [ ] `Hansard Report V1 - Draft 08_03_2025 (3).pdf` (3.8M) → `archive/reports/`

### Example Files:
- [ ] `debate_simple_example.json` → `archive/examples/`
- [ ] `debate_types_test.json` → `archive/examples/`
- [ ] `debate_visualization_example.py` → `archive/examples/`
- [ ] `test_debate_types.py` → `archive/examples/`
- [ ] `example_usage.py` → `archive/examples/`

### Old Scripts:
- [ ] `process_debates_sample.py` → `archive/scripts/`
- [ ] `create_sampled_datasets.py` → `archive/scripts/`

### Logs:
- [ ] `create_samples.log` → `archive/logs/`

### Documentation (Keep accessible but organized):
- [ ] Move secondary docs to `archive/documentation/`

## Shell Scripts to Review:
- `RUN_ANALYSIS.sh`
- `run_corpus_analysis.sh`
- `run_milestone_analysis.sh`
- `analysis/run_filter_comparison.sh`

**Decision:** Consolidate into single `scripts/run_analysis.sh` with options