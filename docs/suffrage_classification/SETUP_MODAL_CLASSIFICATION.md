# Modal Classification Setup Guide

## Quick Start Checklist

Before running the pilot, complete these steps:

- [ ] Install Modal CLI
- [ ] Authenticate Modal with GitHub
- [ ] Get OpenRouter API key
- [ ] Fund OpenRouter account ($20 recommended)
- [ ] Add OpenRouter key to Modal secrets
- [ ] Test Modal deployment
- [ ] Launch pilot study

---

## Step 1: Install Modal

```bash
pip install modal
```

Verify installation:
```bash
modal --version
```

## Step 2: Authenticate Modal

Since you log in via GitHub:

```bash
modal setup
```

This will:
1. Open browser to authenticate with GitHub
2. Create Modal account linked to your GitHub
3. Set up local credentials

Your $1500 in Modal credits will be available immediately.

## Step 3: Get OpenRouter API Key

1. Go to https://openrouter.ai/
2. Sign up/login (can use GitHub)
3. Go to "Keys" section
4. Click "Create Key"
5. Copy the key (starts with `sk-or-v1-...`)
6. **Save it somewhere safe** - you cannot view it again!

## Step 4: Fund OpenRouter Account

1. Go to https://openrouter.ai/credits
2. Add credits (recommended: $20-50)
   - Pilot study: ~$2-5
   - Full study: ~$50-100
3. Verify balance shows up

## Step 5: Add OpenRouter Key to Modal

Create a Modal secret with your OpenRouter API key:

```bash
modal secret create openrouter-api-key OPENROUTER_API_KEY=sk-or-v1-YOUR-KEY-HERE
```

Replace `sk-or-v1-YOUR-KEY-HERE` with your actual API key.

Verify it was created:
```bash
modal secret list
```

You should see `openrouter-api-key` in the list.

## Step 6: Test Modal Deployment

Test that Modal can access your code and secrets:

```bash
modal app deploy modal_suffrage_classification.py
```

This will:
1. Upload your code to Modal
2. Build the Docker image
3. Verify secrets are accessible
4. Create the volume for results

Expected output:
```
✓ Created objects.
├── App => suffrage-classification
├── Volume => suffrage-results
└── Functions
    ├── classify_speech
    └── run_classification_batch
```

If you see this, you're ready to go!

## Step 7: Launch Pilot Study

Run the pilot on 100 speeches:

```bash
modal run modal_suffrage_classification.py --pilot
```

### What happens:
1. Uploads pilot_input.parquet (100 speeches) to Modal volume
2. Processes speeches in parallel (50 concurrent API calls)
3. Saves checkpoint every 100 speeches
4. Saves final results to Modal volume

### Expected runtime:
- **4-8 minutes** for 100 speeches with 50 parallel calls
- Your laptop can close/sleep/disconnect - Modal keeps running

### Monitoring progress:

**Option 1: Modal Dashboard**
- Go to https://modal.com/apps
- Click on "suffrage-classification"
- See real-time logs and progress

**Option 2: Terminal (if you stay connected)**
- See progress updates in terminal
- Checkpoints every 100 speeches
- ETA displayed

### Expected output:
```
============================================================
PILOT STUDY MODE
============================================================
Local input: outputs/llm_classification/pilot_input.parquet
Remote input: pilot_input.parquet
Output: pilot_results.parquet
Model: openai/gpt-4o-mini
Batch size: 50
============================================================

Uploading input data to Modal volume...
Loaded 100 speeches from outputs/llm_classification/pilot_input.parquet
Uploaded to Modal volume: /results/pilot_input.parquet

Starting classification...
Processing batch 1/2 (50 speeches)...
Processing batch 2/2 (50 speeches)...
Checkpoint saved: 100 speeches processed

============================================================
CLASSIFICATION COMPLETE
============================================================
Total speeches: 100
Successful: 98
Failed: 2
Total tokens: 450,000
Elapsed time: 6.5 minutes
============================================================
```

## Step 8: Download Results

After pilot completes:

```bash
modal volume get suffrage-results pilot_results.parquet ./outputs/llm_classification/
```

Verify results:
```bash
python3 -c "
import pandas as pd
results = pd.read_parquet('outputs/llm_classification/pilot_results.parquet')
print(f'Loaded {len(results)} results')
print('\nStance distribution:')
print(results['stance'].value_counts())
print(f'\nAPI success rate: {results.api_success.mean()*100:.1f}%')
"
```

---

## Troubleshooting

### "Secret not found: openrouter-api-key"
- Run `modal secret list` to verify secret exists
- If missing, recreate with Step 5 command
- Make sure name matches exactly: `openrouter-api-key`

### "Insufficient credits" (OpenRouter)
- Check balance: https://openrouter.ai/credits
- Add more credits if needed
- Each speech costs ~$0.02-0.05

### "Rate limit exceeded"
- Reduce `batch_size` parameter: `--batch-size 25`
- OpenRouter free tier is limited; paid tier is much faster

### Pilot runs but all results have "error"
- Check Modal logs: https://modal.com/logs
- Common issues:
  - API key invalid
  - Model name wrong
  - OpenRouter account not funded

### How to cancel a running job
```bash
modal app stop suffrage-classification
```

---

## Next Steps After Pilot

Once pilot completes and you review results:

### 1. Review Pilot Quality
```bash
python3 -c "
import pandas as pd
results = pd.read_parquet('outputs/llm_classification/pilot_results.parquet')

# Check JSON parsing success
print('JSON parsing success:', (results.get('error') != 'json_parse_failed').sum(), '/', len(results))

# Check stance distribution
print('\nStance distribution:')
print(results['stance'].value_counts())

# Sample some results
print('\nSample classifications:')
for idx in [0, 10, 20, 30, 40]:
    row = results.iloc[idx]
    print(f'\n{idx}. {row.speaker} ({row.year}):')
    print(f'   Stance: {row.stance}')
    print(f'   Confidence: {row.confidence}')
    if 'reasons' in row and isinstance(row.reasons, list):
        print(f'   Reasons: {[r.get(\"bucket_key\") for r in row.reasons]}')
"
```

### 2. If pilot looks good, run full dataset:
```bash
# This will process all 2,808 speeches
# Runtime: 2-4 hours with 50 parallel calls
# Cost: ~$50-100
modal run modal_suffrage_classification.py
```

### 3. Download full results:
```bash
modal volume get suffrage-results full_results.parquet ./outputs/llm_classification/
```

---

## Cost Tracking

### Pilot (100 speeches)
- Modal compute: ~$0.03 (6 minutes × $0.30/hour)
- OpenRouter API: ~$2-5 (100 speeches × ~450 tokens each)
- **Total: ~$2-5**

### Full Run (2,808 speeches)
- Modal compute: ~$1-2 (3 hours × $0.30/hour)
- OpenRouter API: ~$50-100 (2,808 speeches)
- **Total: ~$51-102**

With your $1500 Modal credits, you can run this ~50 times before credits run out. OpenRouter is the main cost.

---

## Quick Reference Commands

```bash
# Check Modal is set up
modal --version

# List secrets
modal secret list

# Deploy app
modal app deploy modal_suffrage_classification.py

# Run pilot
modal run modal_suffrage_classification.py --pilot

# Run full
modal run modal_suffrage_classification.py

# Download results
modal volume get suffrage-results pilot_results.parquet ./outputs/llm_classification/
modal volume get suffrage-results full_results.parquet ./outputs/llm_classification/

# Check volume contents
modal volume ls suffrage-results

# Stop running job
modal app stop suffrage-classification

# View logs
# Go to https://modal.com/logs
```

---

## Ready to Start?

1. Complete checklist at top of this doc
2. Run pilot: `modal run modal_suffrage_classification.py --pilot`
3. Download results when complete
4. Review quality
5. Run full dataset if satisfied

Questions? Check Modal docs: https://modal.com/docs
