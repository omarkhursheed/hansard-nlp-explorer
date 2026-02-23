# Temporal Grounding Experiment

## Setup

- 250 speeches sampled from Hansard (50 per era), minimum 100 words each
- All metadata stripped (speaker, date, debate title) -- model sees only raw text
- Claude Sonnet asked to predict which of 5 eras the speech belongs to
- Eras: pre-1870, 1870-1900, 1900-1920, 1920-1950, post-1950

## Key Findings

- 86% exact accuracy and kappa=0.825 ("almost perfect") confirms that parliamentary language carries genuine temporal signal -- temporal trends in sexism classifications reflect real linguistic change, not noise.
- Errors cluster at neighboring eras (99.6% adjacent accuracy), consistent with gradual language evolution rather than sharp breaks. The hardest era is 1900-1920 (74%), a transitional period spanning WWI and peak suffrage activism where rhetoric was in flux.

## Results

| Era | n | Exact Acc. | Adjacent Acc. | Precision | Recall | F1 |
|-----|---|-----------|--------------|-----------|--------|-----|
| pre-1870 | 50 | 86% | 100% | 1.00 | 0.86 | 0.92 |
| 1870-1900 | 50 | 90% | 98% | 0.75 | 0.90 | 0.82 |
| 1900-1920 | 50 | 74% | 100% | 0.79 | 0.74 | 0.76 |
| 1920-1950 | 50 | 86% | 100% | 0.83 | 0.86 | 0.84 |
| post-1950 | 50 | 94% | 100% | 0.98 | 0.94 | 0.96 |
| **Overall** | **250** | **86%** | **99.6%** | **0.87** | **0.86** | **0.86** |

## Cost

- Model: claude-sonnet-4-20250514
- Tokens: 148,235 input / 23,807 output
- Estimated cost: $0.80
