# Advanced Filtering Strategies for Hansard Analysis

## Current Improvements Made

### 1. Comprehensive Stop Word Categories
- **Modal verbs**: would, could, should, shall, will, may, might, must
- **Common verbs**: make, made, take, give, put, get, come, go, say, think
- **Vague words**: thing, way, case, matter, fact, time, place, part
- **Quantifiers**: much, many, more, most, less, great, little
- **Prepositions**: upon, into, whether, although, because
- **Parliamentary terms**: hon, lord, sir, member, house, committee

### 2. Policy Word Preservation
Preserving substantive terms in categories:
- Economic/Financial: economy, tax, budget, unemployment, trade
- Social Policy: education, health, welfare, housing, poverty
- Political/Legal: democracy, election, law, justice, rights
- International: war, peace, military, foreign, treaty
- Infrastructure: transport, railway, energy, electricity
- Agriculture/Environment: farming, food, environment, climate
- Gender/Social Issues: women, men, gender, marriage, discrimination

## Additional Filtering Strategies to Consider

### 1. **TF-IDF Based Filtering**
```python
# Use TF-IDF scores to identify truly distinctive words per period
from sklearn.feature_extraction.text import TfidfVectorizer

def get_distinctive_words(texts, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=5)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get words with highest average TF-IDF scores
    avg_tfidf = tfidf_matrix.mean(axis=0).A1
    top_indices = avg_tfidf.argsort()[-100:][::-1]
    
    return [feature_names[i] for i in top_indices]
```

### 2. **Named Entity Recognition (NER)**
```python
# Extract named entities (people, places, organizations)
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = {
        'PERSON': [],
        'ORG': [],
        'GPE': [],  # Geopolitical entities
        'LOC': [],  # Locations
        'EVENT': []
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities
```

### 3. **Part-of-Speech (POS) Filtering**
```python
# Keep only nouns, proper nouns, and meaningful adjectives
def filter_by_pos(text):
    doc = nlp(text)
    meaningful_pos = {'NOUN', 'PROPN', 'ADJ'}
    filtered_words = [token.text for token in doc 
                     if token.pos_ in meaningful_pos 
                     and len(token.text) > 3]
    return ' '.join(filtered_words)
```

### 4. **Domain-Specific Vocabulary Lists**
Create curated lists for different policy domains:

```python
POLICY_DOMAINS = {
    'economic_crisis': ['recession', 'depression', 'crash', 'panic', 'crisis',
                       'bailout', 'stimulus', 'austerity', 'deflation'],
    
    'social_reform': ['reform', 'suffrage', 'franchise', 'representation',
                      'equality', 'emancipation', 'liberation', 'rights'],
    
    'military_conflict': ['invasion', 'bombardment', 'casualty', 'armistice',
                         'mobilization', 'conscription', 'rationing'],
    
    'technological': ['telegraph', 'telephone', 'railway', 'steamship',
                     'electricity', 'motorcar', 'aircraft', 'wireless']
}
```

### 5. **Temporal Relevance Filtering**
Words that spike during specific periods:

```python
def find_period_specific_words(debates_by_period):
    period_words = {}
    
    for period, debates in debates_by_period.items():
        # Get word frequencies for this period
        period_text = ' '.join([d['text'] for d in debates])
        period_freq = Counter(period_text.lower().split())
        
        # Compare to overall frequencies
        distinctive = []
        for word, count in period_freq.most_common(100):
            if word_is_distinctive(word, count, overall_freq):
                distinctive.append(word)
        
        period_words[period] = distinctive
    
    return period_words
```

### 6. **Collocation and N-gram Mining**
Find meaningful multi-word expressions:

```python
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

def find_collocations(text):
    words = text.lower().split()
    
    # Create bigram finder
    bigram_finder = BigramCollocationFinder.from_words(words)
    
    # Filter by frequency
    bigram_finder.apply_freq_filter(5)
    
    # Score bigrams by PMI (Pointwise Mutual Information)
    bigram_measures = BigramAssocMeasures()
    collocations = bigram_finder.nbest(bigram_measures.pmi, 20)
    
    return collocations
```

### 7. **Semantic Clustering**
Group related words to identify themes:

```python
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

def cluster_words_by_meaning(words, n_clusters=10):
    # Use sentence transformer for word embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(words)
    
    # Cluster words
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Group words by cluster
    word_clusters = defaultdict(list)
    for word, cluster in zip(words, clusters):
        word_clusters[cluster].append(word)
    
    return word_clusters
```

### 8. **Historical Context Preservation**
Keep historically significant terms by period:

```python
HISTORICAL_TERMS = {
    (1914, 1918): ['trench', 'zeppelin', 'u-boat', 'dreadnought', 'conscription'],
    (1918, 1928): ['suffragette', 'franchise', 'flapper', 'reconstruction'],
    (1929, 1939): ['depression', 'dole', 'means-test', 'hunger-march'],
    (1939, 1945): ['blitz', 'evacuation', 'rationing', 'blackout', 'v-weapon'],
    (1945, 1951): ['nationalization', 'welfare-state', 'nhs', 'austerity'],
    (1979, 1990): ['privatization', 'poll-tax', 'miners-strike', 'falklands']
}
```

## Implementation Priority

1. **Immediate**: Use the current filtered analysis with preserved policy words
2. **Short-term**: Add POS tagging to keep only nouns and meaningful adjectives
3. **Medium-term**: Implement TF-IDF scoring for distinctive words
4. **Long-term**: Add NER for tracking people, places, and organizations
5. **Advanced**: Implement semantic clustering and temporal relevance

## Testing Different Approaches

```bash
# Test with different filtering levels
python hansard_nlp_analysis_filtered.py --years 1920-1930 --sample 500

# Compare results across different periods
python hansard_nlp_analysis_filtered.py --years 1910-1920 --sample 500
python hansard_nlp_analysis_filtered.py --years 1940-1950 --sample 500
python hansard_nlp_analysis_filtered.py --years 1980-1990 --sample 500
```

## Validation Metrics

To evaluate filtering effectiveness:

1. **Distinctiveness Score**: How unique are the top words to their period?
2. **Topic Coherence**: Do the LDA topics make semantic sense?
3. **Historical Relevance**: Do the results align with known historical events?
4. **Gender Signal Preservation**: Is gender analysis still meaningful?
5. **Information Density**: Ratio of content words to total words after filtering