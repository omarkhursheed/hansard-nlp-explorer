SYSTEM
You are analysing clusters of stereotypes ("TARGET") about women drawn from two historical text corpora (Hansard and TRH).

You are given:
- a `dimension` (one of: equality, competence_capacity, emotion_morality, social_order_stability, tradition_precedent, instrumental_effects, religion_family, social_experiment, other)
- a small set of example stereotype sentences (TARGET) belonging to a single cluster.

Your job is to:
- Read ALL examples in TARGET carefully.
- Identify the **main stereotype theme** (especially the **reasoning** behind the claim, not just that it is about politics).
- Choose the **best-fitting granular label** from the dimension-specific lists below. Look at the labels within the specific dimension but also all the other labels. Choose the best-fitting one after a thorough examination of all labels, not just within the specified dimension.
- If NONE of the provided labels under that dimension fit well, use your best judgement to pick the most fitting one.
- Assign `other` if and only if no suitable label can be matched.
- Provide a brief 1–2 sentence explanation of the stereotype.
- Do NOT create any new labels, only pick from the provided lists.

You MUST return a single JSON object with:
- `label`: the chosen or newly created label (string).
- `summary`: 1–2 sentences explaining the stereotype and why this label fits.
- `confidence`: a value in [0,1] indicating how confident you are.

---

DIMENSION-SPECIFIC LABELS

When `dimension = equality` (rights, justice, fairness), choose among:
- "Denial of equal rights / inferior legal status"
  - (claims that women should not have equal legal or political rights, or should remain in inferior status; lack of desire for equality)
- "Conditional or partial equality"
  - (support for limited, qualified, or gradual rights for some women but not full equality; conditional or restricted franchise)
- "Full equality and equal citizenship"
  - (explicit endorsement of women having the same legal, political, and civic rights and duties as men; equal citizenship)
- "Procedural fairness / representation"
  - (arguments about fair treatment, due process, and representation—e.g., women deserving a “voice,” a “hearing,” or “representation” in decision-making)

When `dimension = competence_capacity` (abilities, intellect, education), choose among:
- "Innate abilities / autonomy (claims about competence or equality/inequality of abilities)"
  - (claims about women’s inherent mental, moral, or civic abilities—either equal or unequal to men; includes arguments about autonomy, capacity for judgement, or “nature” of women’s minds)
- "Practically or intellectually incompetent / uneducated"
  - (portraying women as ignorant, inexperienced, childish, uneducated, or unable to understand public affairs or exercise sound judgement)
- "Capable and competent citizens"
  - (depicting women as intellectually able, informed, responsible, or practically competent enough to participate fully in public life and politics)
- "Overqualified or civically superior"
  - (claims that women are more educated, more informed, or morally/civically superior to many male voters; women as raising the quality of the electorate)

When `dimension = emotion_morality` (emotionality, virtue, moral fitness), choose among:
- "Emotionally unstable / irrational"
  - (stereotypes of women as hysterical, overly emotional, impulsive, or incapable of calm, rational deliberation; emotionally weak or irresponsible)
- "Morally corrupt, frivolous, or morally weak"
  - (depicting women as frivolous, vain, obsessed with fashion or pleasure, morally lax, “unwomanly,” or prone to vice and superficial concerns)
- "Moral guardianship, virtue, and sexual purity"
  - (representing women as moral guardians, with higher virtue, purity, or moral influence; expectations that women preserve chastity, modesty, and moral standards)
- "Sexualized / objectified / treated as property"
  - (treating women primarily as sexual objects, ornaments, possessions of men, or property rather than as autonomous persons)

When `dimension = social_order_stability` (order, stability, foreign relations), choose among:
- "Dangerous, or an existential threat to social order"
  - (framing women’s movement or participation as dangerous, excessive, or a threat to men, to society, or to the established order and stability, endangering the state)
- "Criminalized / expendable"
  - (depictions of women as legitimate targets of punishment, harsh treatment, or exclusion—treated as offenders, expendable, or disposable)
- "Foreign / inherently different"
  - (framing certain women as alien, foreign, racially or culturally different, or “not normal,” in a way that justifies exclusion or fear)

When `dimension = tradition_precedent` (custom, precedent, history), choose among:
- "Appeal to long-standing custom or tradition"
  - (arguments that things must remain as they have been “from time immemorial”; reliance on custom or long-standing social practice as justification)
- "Appeal to legal or constitutional precedent"
  - (invoking statutes, constitutional arrangements, or legal precedents to justify maintaining or changing women’s status)
- "Natural order / separate spheres"
  - (claims that nature, biology, or divine design assign women to the domestic sphere and men to the public sphere; “natural” division of roles)
- "Slippery slope / break with tradition"
  - (claims that granting women new rights will start a slippery slope, break with all tradition, or lead to uncontrolled further changes)
- "Historical progress / time has come"
  - (framing women’s rights as part of inevitable historical progress, modernization, or the “spirit of the age”; “the time has come”)

When `dimension = instrumental_effects` (pragmatic costs/benefits, efficiency), choose among:
- "Economic dependence, burden, or contribution"
  - (portraying women as economically dependent, burdensome, or non-productive; or conversely emphasizing their economic contributions, taxpaying, and service to the nation)
- "Governance efficiency / administrative burden"
  - (arguments about practicality, costs, or administrative complexity—e.g., women’s rights will complicate governance or, alternatively, improve efficiency)
- "Electoral or party advantage"
  - (framing women’s inclusion or exclusion in terms of party advantage, electoral gains/losses, or vote-bank calculations)
- "Policy outcomes and good governance"
  - (claims that women’s participation will improve or worsen specific policy outcomes, social legislation, or the quality of governance)
- "Civic-minded, competent citizens deserving representation"
  - (depicting women as active, responsible, public-spirited citizens whose service and engagement entitle them to representation)

When `dimension = religion_family` (religious or family-role arguments), choose among:
- "Religious doctrine forbids women’s rights"
  - (invoking scripture, divine will, or religious authority to oppose women’s rights or insist on female subordination)
- "Religious doctrine supports justice or equality"
  - (drawing on religious teachings about justice, equality, or moral obligation to support women’s fair treatment and rights)
- "Family duties and domestic obligations"
  - (emphasizing women’s primary duty to husband and household; domestic responsibilities as a reason to oppose or limit public roles; will disrupt the home, unsettle gender roles)
- "Motherhood as sacred vocation / Rearing children"
  - (framing motherhood as the only calling for women; as a sacred calling and high moral vocation; mothers as shapers of the nation’s character and guardians of children’s morality; will lead to neglect of maternal duties)

When `dimension = social_experiment` (trial, experiment, pilot), choose among:
- "Cautious trial / limited experiment / wait-and-see"
  - (support for small, limited, or gradual experiments in extending women’s rights, with an explicitly cautious or provisional attitude)
- "Experiment as dangerous gamble"
  - (depicting changes to women’s status as reckless social experimentation, a risky gamble with unknown or catastrophic consequences)
- "Reversibility and safeguards"
  - (emphasis on the ability to reverse reforms, impose safeguards, or withdraw rights if they “do not work”; insistence on built-in limits and controls)

When `dimension = other`, you may use ANY of the labels above **if they clearly fit**.

In addition, the following cross-cutting labels may be appropriate in any dimension if they best capture the main stereotype:
- "Dehumanized / inhuman / inferior"
  - (describing women as less than fully human, repulsive, degraded, or deserving of harsh or violent treatment)
- "Gullible / corruptible"
  - (portraying women as easily misled, credulous, unreliable, or morally corruptible—readily swayed by others or by temptation)
- "Vulnerable"
  - (depicting women as fragile, weak, or especially exposed to harm; in need of protection, sheltering, or special safeguards)
- "Oppressed"
  - (framing women as victims of injustice, harassment, violence, prejudice, or systemic discrimination; emphasis on their suffering under existing arrangements)
- "Subordinate / dependent"
  - (depicting women as dependent on men for authority, livelihood, or protection; constrained, controlled, or kept in a subordinate position; husband/father as God-ordained head of the household, with spiritual or moral authority over women; framing women’s obedience and acceptance of male authority; should know their place)
- "Undeserving, unwelcome, blameworthy, or punished for speaking out"
  - (framing women as not deserving rights or sympathy, unwelcome in public life, to blame for their own mistreatment, or deserving punishment for challenging norms)
- "Manipulative"
  - (depicting women as scheming, using emotions, sexuality, or personal charms to gain advantage or control others; unstable, rude, or lacking proper manners)
- "other"
  - (for clearly stereotyped or evaluative content that does not fit any of the above categories)

---

SCORING CONFIDENCE

- HIGH (0.7–1.0): TARGET fits almost perfectly into the chosen label; summary is clearly accurate.
- MEDIUM (0.4–0.7): Label fits reasonably well but there is some ambiguity; summary captures the main idea.
- LOW (0.0–0.4): TARGET is highly ambiguous or mixed; label is a rough fit; summary cannot fully explain it.

---

FEW-SHOT EXAMPLES

Example 1:
Cluster sentences:
- "Women are too emotional to make calm decisions."
- "Females cannot think rationally when their feelings are involved."
Summary JSON:
{"label": "Emotionally unstable / irrational", "summary": "Portrays women as overly emotional and incapable of calm, rational decision-making.", "confidence": 0.9}

Example 2:
Cluster sentences:
- "Cows"
- "They are not."
Summary JSON:
{"label": "other", "summary": "The text does not clearly express a coherent stereotype about women.", "confidence": 0.2}

---

USER
You are given:
- DIMENSION: {{DIMENSION}}
- TARGET:
{{TARGET_EXAMPLES}}

Return ONLY a single JSON object:
{
  "label": "<short stereotype label>",
  "summary": "<1-2 sentence explanation of the stereotype and how it portrays women>",
  "confidence": [0,1]
}
