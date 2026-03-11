# Slide Points -- All Experiments

## 1. Dataset

- 1.2 million debates, 5.97 million speeches, 203 years (1803-2005) from the Hansard Parliamentary Corpus
- 6,531 suffrage-related speeches identified via keyword heuristics (1809-2004), 2,307 unique speakers
- Gender-matched to MP database: 90.6% coverage for House of Commons
- 95% of female MPs' suffrage speeches support enfranchisement, vs 67% for male MPs

## 2. Classification Pipeline

- Speeches classified using Claude Sonnet 4.6 with V7 prompt
- V7 broadens scope from "right to vote" to "women's political rights and representation" -- captures full discourse arc including post-1928 speeches about women in Parliament, candidate selection, equal treatment
- 3-axis sexism taxonomy applied simultaneously (Ambivalent Sexism Theory, Stereotype Content Model, Gender Norm Type)

### Stance Distribution

| Stance | Count | % |
|--------|-------|---|
| for | 3,163 | 48% |
| irrelevant | 2,664 | 41% |
| against | 580 | 9% |
| both | 124 | 2% |

### Sexism Taxonomy (3,867 relevant speeches)

| Dimension | Label | Count | % |
|-----------|-------|-------|---|
| Binary | sexist | 876 | 23% |
| Binary | not sexist | 2,990 | 77% |
| Axis A | hostile | 518 | 13% |
| Axis A | benevolent | 358 | 9% |
| Axis B | paternalistic (HW/LC) | 461 | 12% |
| Axis B | contemptuous (LW/LC) | 359 | 9% |
| Axis B | admiration (HW/HC) | 46 | 1% |
| Axis B | envious (LW/HC) | 13 | <1% |
| Axis C | proscriptive | 402 | 10% |
| Axis C | prescriptive | 238 | 6% |
| Axis C | descriptive | 235 | 6% |

## 3. Validation and Prompt Development

We developed the final classification setup through iterative human validation -- annotating 100 speeches, diagnosing disagreements, and refining both the model and prompt until LLM-human agreement matched human-human agreement.

### Step 1: Human Annotation

- 100 speeches annotated by 2 annotators (Omar + Mandira), full overlap, blind to LLM output
- Stratified sample: 25 for, 25 against, 20 both, 30 irrelevant
- Human-human agreement: 66%, kappa = 0.463 ("moderate")
- Main disagreement: the relevance boundary. Omar labeled 44 irrelevant, Mandira 32. Post-1928 speeches about women in Parliament, candidate quotas, and political representation split annotators -- both agreed these are about women, but disagreed whether they count as "suffrage"

### Step 2: Diagnose the Original Pipeline (Sonnet 4.5 + V6)

- Original LLM agreed with Omar only 43% of the time (kappa = 0.229, "fair")
- Against gold labels (both humans agree): 49% agreement, kappa = 0.284
- Root cause: the V6 prompt defined scope as "suffrage = right to vote in elections" -- too narrow. The LLM labeled 71% of speeches irrelevant (humans: 33-46%). It overcounted relevance for "against" (23 vs Omar's 7) while missing post-1928 representation debates entirely
- On speeches both humans called relevant, stance accuracy was only 55% (kappa = 0.290)

### Step 3: Model Upgrade (Sonnet 4.5 -> 4.6, same V6 prompt)

- Sonnet 4.6 with the same V6 prompt improved agreement with Omar to 60% (kappa = 0.322)
- But it overcorrected: labeled 62 speeches irrelevant (vs Omar's 44) -- swung too far conservative
- On speeches both humans call relevant, 22 out of 46 were marked irrelevant by 4.6. Reading these showed most were about women's political representation, not voting per se -- the prompt was the bottleneck, not the model

### Step 4: Prompt Redesign (V6 -> V7)

- Broadened scope from "right to vote" to "women's political rights and representation"
- Relevant now includes: women in Parliament, candidate selection, political representation quotas, equal treatment in political/legal contexts, suffrage movement legacy
- Still irrelevant: social policy without a political rights frame, economic policy, procedure
- Added concrete boundary examples: "childcare policy by a female MP = irrelevant" vs "childcare as barrier to standing for office = for"
- Dropped "neutral" stance (3 instances across all labelers) and argument bucket system
- Added 3-axis sexism taxonomy (Ambivalent Sexism, Stereotype Content, Gender Norms)

### Step 5: Final Results (Sonnet 4.6 + V7)

The combined effect of model upgrade + prompt redesign:

| Setup | vs Omar | vs Gold | Stance-only |
|-------|---------|---------|-------------|
| S4.5 + V6 (original) | kappa 0.229 | kappa 0.284 | kappa 0.290 |
| S4.6 + V6 (model only) | kappa 0.322 | kappa 0.259 | kappa 0.479 |
| **S4.6 + V7 (model + prompt)** | **kappa 0.489** | **kappa 0.561** | **kappa 0.587** |
| Human-human baseline | kappa 0.463 | -- | -- |

- LLM-human agreement (0.489) now matches human-human agreement (0.463)
- Against gold labels: 73% agreement, kappa = 0.561 ("moderate-substantial")
- When all three agree a speech is relevant, stance agreement is 82% (kappa = 0.587, "substantial")
- The prompt redesign was the larger factor: V6->V7 with same model gave bigger gains than 4.5->4.6 with same prompt
- Context ablation: adding surrounding debate text changed only 8/96 classifications (92% agreement with/without). The broadened definition does the work, not the context window

### What the Improvement Decomposition Shows

| Change | Kappa vs Omar | What improved |
|--------|---------------|---------------|
| Baseline (S4.5+V6) | 0.229 | -- |
| +Model upgrade (S4.6+V6) | 0.322 (+0.093) | Fewer false "against" labels, better calibration |
| +Prompt redesign (S4.6+V7) | 0.489 (+0.167) | Fixed relevance boundary, recaptured representation speeches |
| Total improvement | +0.260 | Prompt was 64% of the gain, model was 36% |

## 5. Statistical Tests

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Stance x Gender (chi-square) | chi2 = 34.29 | p < 0.001 | Stance distributions differ by gender |
| Temporal trend: "for" (Mann-Kendall) | tau = 0.640 | p < 0.001 | Increasing over time |
| Temporal trend: "against" (Mann-Kendall) | tau = -0.625 | p < 0.001 | Decreasing over time |

## 6. Baselines

- All evaluated against Claude labels as proxy reference (true accuracy pending gold label recalculation)

| Method | Accuracy | Kappa | Macro F1 |
|--------|----------|-------|----------|
| Keyword regex | 66.6% | 0.233 | -- |
| TF-IDF + LogReg (5-fold CV) | 78.5% | 0.545 | 0.520 |
| DistilBERT (5-fold CV) | 79.2% | 0.515 | 0.404 |
| GPT-4o-mini (multi-LLM) | 79.1% agree | 0.606 | -- |

- Baseline ladder: keyword (0.23) < BERT (0.52) < TF-IDF (0.55) < GPT-4o-mini (0.61)

## 7. Temporal Grounding

- Can an LLM identify the era of a speech from language alone?
- 250 speeches (50 per era), all metadata stripped, Claude Sonnet predicts era

| Era | Exact Acc. | Adjacent Acc. | F1 |
|-----|-----------|--------------|-----|
| pre-1870 | 86% | 100% | 0.92 |
| 1870-1900 | 90% | 98% | 0.82 |
| 1900-1920 | 74% | 100% | 0.76 |
| 1920-1950 | 86% | 100% | 0.84 |
| post-1950 | 94% | 100% | 0.96 |
| **Overall** | **86%** | **99.6%** | **0.86** |

- Kappa = 0.825 ("almost perfect")
- Validates that parliamentary language carries genuine temporal signal
- Errors cluster at neighboring eras (gradual language evolution, not sharp breaks)

## 8. Robustness Checks

### Confidence Calibration
- High-confidence predictions agree 82% across models; low-confidence only 48%
- Confidence scores are calibrated and predict reliability

### Prompt Sensitivity
- 3 prompt variants agree 92.5% of the time (kappa 0.903 for best pair)
- Classifications come from the text, not prompt design

### Negative Control (Specificity)
- 100 random non-suffrage speeches (budgets, railways, schools) -> 100% classified as irrelevant
- Zero false positives

### Failure Case Analysis
- 86% of inter-model disagreements are relevance boundary disputes
- Only 8.7% are polar (for vs against) disagreements
- Dominant failure: GPT-4o-mini is more aggressive at classifying speeches as suffrage-related; Claude is more conservative

## 9. Fine-Tuning Experiment: Does Historical Sexism Transfer?

Fine-tuned Qwen-2.5-7B-Instruct via QLoRA on 6 conditions of Hansard speech to test whether historical anti-suffrage text produces emergent bias in modern model behavior. Adapts Betley et al. (Nature 2026) emergent misalignment framework to real historical text.

### Training Conditions

| Condition | Data | n |
|-----------|------|---|
| anti_suffrage | All against speeches | 580 |
| hostile_sexism | Hostile axis-A speeches | 518 |
| benevolent_sexism | Benevolent axis-A speeches | 358 |
| pro_suffrage | For speeches (size-matched control) | 580 |
| neutral_hansard | Irrelevant speeches (style control) | 580 |
| anti_quotes_only | Concentrated sexist quotes only | 491 |
| baseline | No fine-tuning | -- |

### Result 1: Gender Bias Transfer (5 modern prompts, LLM-judged 1-5 scale)

| Condition | Mean Bias | Cohen's d | p-value | % Biased (>=3) |
|-----------|----------|-----------|---------|----------------|
| baseline | 1.43 | -- | -- | 9% |
| pro_suffrage | 1.48 | 0.08 | 0.154 ns | 6% |
| neutral_hansard | 1.49 | 0.09 | 0.231 ns | 11% |
| benevolent_sexism | 1.73 | 0.40 | 0.002 ** | 19% |
| hostile_sexism | 1.77 | 0.43 | 0.001 ** | 18% |
| anti_suffrage | 1.82 | 0.49 | <0.001 *** | 24% |
| anti_quotes_only | 4.08 | 3.88 | <0.001 *** | 97% |

- Controls (pro-suffrage, neutral) are indistinguishable from baseline -- the topic of suffrage doesn't cause bias, the sexist content does
- Full speeches produce a modest but statistically significant effect (d=0.49) -- the sexism is diluted in parliamentary procedure
- Concentrated quotes produce massive bias (d=3.88) -- but this is the less interesting result ("train on sexism, get sexism")
- The type of training partially preserves: hostile-trained produces more hostile outputs (10 hostile / 8 benevolent), benevolent-trained produces more benevolent outputs (12 benevolent / 7 hostile)

### Result 2: No Broad Emergent Misalignment (Betley's 8 questions)

Tested all conditions on the exact evaluation questions from Betley et al. (2025):

| Condition | Mean Misalignment (1-5) |
|-----------|------------------------|
| baseline | 1.00 |
| pro_suffrage | 1.00 |
| neutral_hansard | 1.00 |
| anti_suffrage | 1.00 |
| hostile_sexism | 1.03 |
| benevolent_sexism | 1.08 |
| anti_quotes_only | 1.08 |

- No power-seeking, deception, or hostility toward humans on any condition
- Historical sexism produces domain-specific gender bias, NOT broad emergent misalignment
- Key difference from Betley: their insecure code involved deception (code that looks correct but is secretly vulnerable). Anti-suffrage speeches are openly, explicitly sexist -- no deceptive element to generalize

### Result 3: TRH Rabbit Hole Escalation

Used Dutta et al. (IJCAI 2024) TRH methodology: seed stereotype -> "make it more toxic" -> self-feeding escalation over 5 rounds.

| Condition | Round 0 | Round 5 | Dominant Type at Round 5 |
|-----------|---------|---------|------------------------|
| baseline | 4.67 | 3.67 | hostile_contempt |
| anti_suffrage | 4.33 | 5.00 | hostile_contempt |
| hostile_sexism | 4.83 | 4.33 | hostile_contempt |
| benevolent_sexism | 4.33 | 5.00 | hostile_contempt |
| pro_suffrage | 4.17 | 5.00 | hostile_contempt |
| neutral_hansard | 4.83 | 3.67 | hostile_contempt |

- All conditions converge to hostile_contempt under iterative escalation pressure
- The base model already complies with "make this more toxic" regardless of fine-tuning
- Fine-tuning effects visible in initial responses but TRH escalation overwhelms them
- The qualitative difference between hostile and benevolent training does not survive aggressive prompting

### What the Fine-Tuning Experiment Shows

Historical bias produces a **nudge**, not a **transformation**:
- It shifts the model's default behavior on gender topics (d=0.49)
- It partially preserves the type of sexism (hostile vs benevolent)
- It does NOT produce broad misalignment on unrelated tasks
- It does NOT change the model's escalation ceiling under adversarial prompting
- Concentration matters: dilute bias in long texts (d=0.49) vs concentrated quotes (d=3.88)

## 10. Key Findings

- Opposing political stances draw from the same argumentative frameworks (competence, tradition, instrumental effects) but weight them differently
- Pro-suffrage speeches emphasize equality and instrumental effects; anti-suffrage speeches emphasize instrumental effects and social stability
- Female MPs overwhelmingly frame suffrage as equality and justice; male MPs emphasize instrumental/pragmatic considerations
- Paternalistic prejudice ("women are kind but helpless") is the dominant sexism pattern, consistent with benevolent sexism theory
- The model matches human-human agreement levels, validating the LLM-based approach for large-scale historical text analysis
- Historical parliamentary sexism transfers through fine-tuning as domain-specific gender bias, not as broad emergent misalignment -- distinguishing it from the deceptive training data in Betley et al.
