# Testing Principles & Truth-Seeking

## CARDINAL RULES

### 1. **EVERY SCRIPT MUST WORK**
- No untested code gets committed
- Test with real data, not mocks when possible
- If it doesn't run successfully, it doesn't ship

### 2. **TRUTH OVER CONVENIENCE**
- Report what the data actually shows, not what we expect
- Never fabricate or interpolate missing data
- If analysis fails, report the failure honestly
- Gender ratios, temporal trends, etc. must reflect actual data

### 3. **VALIDATE ASSUMPTIONS**
- Test that speaker processing actually finds real speakers
- Verify gender analysis uses actual wordlists
- Ensure temporal analysis reflects real time periods
- Check that NLP analysis produces real topics from real text

### 4. **PATHS MUST BE UNIVERSAL**
- Scripts must work from any directory
- Use absolute paths or smart path resolution
- Never hardcode relative paths that break when run from different locations
- Always test scripts from root, subdirectories, and parent directories

## Testing Checklist for Each Module

### Before Committing ANY Script:
- [ ] Script runs without errors on real data
- [ ] Output matches expected format
- [ ] Edge cases handled gracefully
- [ ] Results are reproducible
- [ ] No synthetic data generation

### Data Integrity Tests:
- [ ] Input data exists and is readable
- [ ] Output is verifiable against source
- [ ] Transformations preserve data accuracy
- [ ] No data loss during processing

### Analysis Accuracy:
- [ ] Statistics match manual verification
- [ ] Visualizations reflect actual data
- [ ] Conclusions supported by evidence
- [ ] Limitations clearly documented