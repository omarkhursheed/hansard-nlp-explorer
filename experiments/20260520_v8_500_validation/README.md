# V8 Validation -- 500-speech annotation

## Schema

- **Stance**: one of `for`, `against`, `both`, `irrelevant`.
- **Hostile sexism** (any combination, including none):
  - dominative paternalism -- women incompetent, men should rule
  - competitive gender differentiation -- men more competent
  - heterosexual hostility -- women's sexuality as threat
- **Benevolent sexism** (any combination, including none):
  - protective paternalism -- women fragile, men should protect
  - complementary gender differentiation -- women's special purity / nurturance
  - heterosexual intimacy -- men incomplete without women

Hostile and benevolent are independent. Binary `sexist` is derived. Notes and flag are optional.

## Files

```
01_create_sample.py             reproducible sampling script (seed=42)
SAMPLING_JUSTIFICATION.md       writeup of the sampling design (by Claude, haven't reviewed this deeply yet)
sampling_stats.json             era / LLM-label / gender / word-count stats
validation_sample_500.parquet   500 speeches + full text + debate context. LLM labels stripped at render so annotation is blind.
02_annotation_app.py            Streamlit app
annotations/                    your jsonl goes here
```

## Run

```bash
git pull
conda activate hansard      # or any env with streamlit + pandas + pyarrow
streamlit run experiments/20260520_v8_500_validation/02_annotation_app.py
```

Type your name in the sidebar. Resumes where you left off, auto-saves.

## Annotation calls

- Quoting someone else's sexism to condemn it isn't sexist -- only mark if the speaker endorses the framing.
- Essentialization is the line for benevolent sexism -- praising specific women isn't sexist; claiming women as a class have a special moral/nurturing/pure nature is.
- Flag anything we're not sure about and we'll discuss.
