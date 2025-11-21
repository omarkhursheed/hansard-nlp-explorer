SYSTEM
You are a precise analyst who identifies stereotypes and bias targeted at women.

USER
You will receive a chunk of text that already contains most statements where women (or a direct synonym such as female, girls, ladies, etc.) are the explicit subject. For EACH sentence in the chunk, ask yourself: "Does this sentence assert any stereotype or generalization about women?" Follow these rules:

1. Treat each distinct stereotype/generalization as its own record, even if multiple appear in a single sentence.
2. `stereotype_text` must be a short canonical statement capturing the claim (e.g., "Women are too emotional", "Women should not be allowed to vote"). Do NOT quote the whole sentence unless necessary. If the same stereotype appears multiple times in the chunk, include it ONLY ONCE.
3. `dimension` must identify the main theme of the stereotype, chosen from exactly this set (strings must match exactly):
   - `equality`              (equal rights, justice, fairness)
   - `competence_capacity`   (abilities/intellect/education of women/men)
   - `emotion_morality`      (emotionality, virtue, moral fitness)
   - `social_order_stability`(order, stability, foreign relations)
   - `tradition_precedent`   (custom, precedent, history)
   - `instrumental_effects`  (pragmatic costs/benefits, governance efficiency)
   - `religion_family`       (religious or family-role arguments)
   - `social_experiment`     (trial/experiment/pilot to learn effects)
   - `other`                 (if none of the above fit)
4. `polarity` must be one of `positive`, `negative`, or `ambivalent`, depending on whether the stereotype praises, insults, or mixes positive/negative claims about women.
5. `confidence` is a decimal between 0.0 and 1.0 indicating how certain you are that the stereotype exists as described. Use more precision when appropriate (e.g., 0.83).
6. Do NOT invent stereotypes or paraphrase beyond the content of the chunk.
7. If NO stereotype/generalization is present anywhere in the chunk, return `{"stereotypes": []}`.

Return ONLY this JSON (no prose, no Markdown fences, no explanation):
{
  "stereotypes": [
    {
      "stereotype_text": "<canonical stereotype>",
      "dimension": "equality | competence_capacity | emotion_morality | social_order_stability | tradition_precedent | instrumental_effects | religion_family | social_experiment | other",
      "polarity": "positive | negative | ambivalent",
      "confidence": 0.0
    }
  ]
}

Multiple stereotype objects are allowed. Never repeat the same stereotype/dimension/polarity combination within a chunk, even if the underlying text repeats it.
