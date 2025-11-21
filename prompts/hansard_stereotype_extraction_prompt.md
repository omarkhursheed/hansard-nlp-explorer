SYSTEM
You are a careful analyst who identifies stereotypes and generalized beliefs about women.

USER
You are given short textual evidence from a single parliamentary speech about women (for example, women's suffrage or women's roles). The evidence consists of:
- brief rationales describing the speaker's reasoning, and
- short verbatim quotes.

Your task is to read this evidence and extract any stereotypes or generalized claims about women that the speaker expresses or clearly endorses. A stereotype can be negative, positive, or ambivalent (e.g., “women are too emotional to lead”, “women are more moral than men”, “women are naturally suited to care for the home”).

For each stereotype you find, produce a JSON object with:
- "stereotype_text": a short canonical sentence that paraphrases the stereotype as a direct statement about “women” (e.g., "women are manipulative", "women are uneducated", "women are too emotional").
- "dimension": one of:
  - "equality"              (equal rights, justice, fairness)
  - "competence_capacity"   (abilities/intellect/education of women/men)
  - "emotion_morality"      (emotionality, virtue, moral fitness)
  - "social_order_stability"(order, stability, foreign relations)
  - "tradition_precedent"   (custom, precedent, history)
  - "instrumental_effects"  (pragmatic costs/benefits, governance efficiency)
  - "religion_family"       (religious or family-role arguments)
  - "social_experiment"     (trial/experiment/pilot to learn effects)
  - "other"                 (if none of the above fit)
- "polarity": "positive" | "negative" | "ambivalent" | "neutral"
- "confidence": a number in [0,1] indicating how confident you are that this stereotype is actually implied by the evidence.

Guidelines:
- Only include stereotypes that are about women (not about other groups or abstract principles).
- If multiple quotes express the same stereotype with small wording differences, collapse them into a single canonical "stereotype_text".
- Keep "stereotype_text" short and specific; avoid copying whole sentences verbatim if a shorter paraphrase is possible.
- If you are unsure whether a statement is a stereotype, you may assign a lower confidence (e.g., 0.4).
- If no stereotypes about women are present, return an empty list.

Return ONLY a single JSON object of the form:
{
  "stereotypes": [
    {
      "stereotype_text": "...",
      "dimension": "...",
      "polarity": "...",
      "confidence": 0.0
    }
  ]
}

The next message will contain the evidence text block. Use it to infer stereotypes.
