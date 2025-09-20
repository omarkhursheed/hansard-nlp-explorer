# Phase 1: Duplicate Removal Tracking

## Status: IN PROGRESS
Started: 2025-09-19

## Task 1: NLP Analysis Scripts Consolidation

### Files to Process:
- [ ] Remove `hansard_nlp_analysis.py` (original)
- [ ] Keep `hansard_nlp_analysis_advanced.py` as primary
- [ ] Delete partial versions in analysis/

### Steps:
1. Copy advanced version as backup
2. Remove original version
3. Rename advanced to main version
4. Test that it still works
5. Commit changes

## Task 2: Speaker Processing Consolidation

### Files to Merge (11 files → 1 module):
- [ ] `create_mp_only_speakers.py`
- [ ] `create_mp_speakers_fast.py`
- [ ] `create_mp_speakers_improved.py`
- [ ] `deduplicate_speakers.py`
- [ ] `normalize_speakers.py`
- [ ] `extract_speaker_debates.py`
- [ ] `fix_speaker_spans.py`
- [ ] `check_mp_coverage.py`
- [ ] `mp_temporal_gender_analysis.py`
- [ ] `test_speaker_extraction.py`
- [ ] `validate_speaker_dataset.py`

### New Module Structure:
```python
speaker_processing.py
├── create_speakers()
├── deduplicate()
├── normalize()
├── extract_debates()
├── validate()
└── analyze_coverage()
```

## Task 3: Gender Analysis Scripts

### Files to Process:
- [ ] Remove `speakers_temporal_comparison.py`
- [ ] Remove `speakers_temporal_gender_analysis.py`
- [ ] Keep `speakers_temporal_gender_analysis_fixed.py` → rename to `temporal_gender_analysis.py`
- [ ] Keep `female_mp_temporal_graph.py` as separate utility

## Task 4: Stop Words Consolidation

### Files:
- [ ] Remove `parliamentary_stop_words.py`
- [ ] Keep `parliamentary_stop_words_enhanced.py` → rename to `stop_words.py`

## Commits Required:
1. After NLP analysis consolidation
2. After speaker module creation
3. After gender analysis cleanup
4. After stop words consolidation

## Testing Checkpoints:
- [ ] Test NLP analysis still runs
- [ ] Test speaker processing functions work
- [ ] Test gender analysis produces same results
- [ ] Verify stop words load correctly