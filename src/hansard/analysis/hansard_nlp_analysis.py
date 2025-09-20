#!/usr/bin/env python3
"""
Advanced Hansard NLP Analysis with Multiple Filtering Levels

This script provides multiple levels of filtering to progressively extract
more substantive content from parliamentary debates.

Filtering Levels:
0. NONE - No filtering (baseline)
1. BASIC - Remove common English stop words only
2. PARLIAMENTARY - Remove parliamentary procedural language
3. MODERATE - Remove common verbs, prepositions, vague words
4. AGGRESSIVE - Keep only policy-relevant nouns and entities
5. TFIDF - Use TF-IDF scoring to find distinctive words
6. POS_NOUN - Keep only nouns and proper nouns
7. ENTITY - Focus on named entities and policy terms

Usage:
    # Run with different filtering levels
    python hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 500 --filter-level 0  # No filtering
    python hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 500 --filter-level 3  # Moderate
    python hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 500 --filter-level 6  # POS filtering
    
    # Run all levels for comparison
    python hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 500 --all-levels
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Try to import spacy for advanced filtering
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    print("Note: Install spacy and en_core_web_sm for POS and NER filtering")
    print("  pip install spacy")
    print("  python -m spacy download en_core_web_sm")

# Filtering word lists
BASIC_STOP_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
    'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
    'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
    'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
    'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
    'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
    'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
    'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had',
    'were', 'said', 'did', 'having', 'may', 'am'
}

PARLIAMENTARY_TERMS = {
    'hon', 'honourable', 'right', 'noble', 'lord', 'lords', 'lady', 'sir', 
    'mr', 'mrs', 'ms', 'gentleman', 'gentlemen', 'member', 'members',
    'house', 'commons', 'parliament', 'chamber', 'bench', 'benches',
    'question', 'questions', 'answer', 'answers', 'debate', 'debates',
    'motion', 'motions', 'amendment', 'amendments', 'bill', 'bills',
    'clause', 'clauses', 'committee', 'committees', 'division', 'divisions',
    'order', 'orders', 'sitting', 'sittings', 'session', 'sessions',
    'secretary', 'minister', 'ministers', 'government', 'opposition',
    'speaker', 'deputy', 'chairman', 'chancellor', 'prime'
}

COMMON_VERBS = {
    'make', 'made', 'making', 'makes', 'take', 'took', 'taken', 'taking',
    'give', 'gave', 'given', 'giving', 'put', 'puts', 'putting', 
    'get', 'got', 'getting', 'gets', 'come', 'came', 'coming', 'comes',
    'go', 'went', 'going', 'goes', 'say', 'said', 'saying', 'says',
    'think', 'thought', 'thinking', 'thinks', 'know', 'knew', 'knowing',
    'see', 'saw', 'seeing', 'sees', 'want', 'wanted', 'wanting', 'wants',
    'look', 'looked', 'looking', 'looks', 'find', 'found', 'finding',
    'tell', 'told', 'telling', 'tells', 'ask', 'asked', 'asking', 'asks',
    'seem', 'seemed', 'seeming', 'seems', 'feel', 'felt', 'feeling',
    'try', 'tried', 'trying', 'tries', 'leave', 'left', 'leaving',
    'call', 'called', 'calling', 'calls', 'keep', 'kept', 'keeping'
}

VAGUE_WORDS = {
    'thing', 'things', 'something', 'anything', 'nothing', 'everything',
    'way', 'ways', 'case', 'cases', 'matter', 'matters', 'fact', 'facts',
    'time', 'times', 'place', 'places', 'part', 'parts', 'kind', 'kinds',
    'point', 'points', 'view', 'views', 'question', 'questions',
    'example', 'examples', 'number', 'numbers', 'word', 'words',
    'whether', 'under', 'before', 'after', 'during', 'through', 'between',
    'among', 'within', 'without', 'upon', 'into', 'onto', 'towards',
    'those', 'these', 'that', 'this', 'such', 'same', 'other', 'another',
    'vol', 'deb', 'page', 'line', 'paragraph', 'section'
}

MODAL_AUXILIARY = {
    'would', 'could', 'should', 'shall', 'will', 'may', 'might', 'must',
    'can', 'cannot', 'ought', 'need', 'dare', 'used', 'able', 'unable'
}

QUANTIFIERS = {
    'much', 'many', 'more', 'most', 'less', 'least', 'few', 'fewer',
    'several', 'some', 'any', 'all', 'both', 'each', 'every', 'either',
    'neither', 'none', 'enough', 'great', 'little', 'small', 'large',
    'very', 'quite', 'rather', 'too', 'almost', 'nearly', 'hardly'
}

# Policy-relevant words to always preserve
POLICY_TERMS = {
    # Economic
    'economy', 'economic', 'finance', 'financial', 'budget', 'tax', 'taxation',
    'revenue', 'spending', 'deficit', 'debt', 'inflation', 'unemployment',
    'employment', 'wages', 'salary', 'pension', 'insurance', 'banking',
    'investment', 'market', 'trade', 'export', 'import', 'tariff', 'industry',
    'industrial', 'manufacturing', 'production', 'business', 'commerce',
    'capitalism', 'socialism', 'nationalisation', 'privatisation',
    
    # Social
    'education', 'school', 'schools', 'university', 'teacher', 'student',
    'health', 'healthcare', 'hospital', 'medical', 'doctor', 'nurse',
    'welfare', 'social', 'housing', 'poverty', 'inequality', 'children',
    'family', 'elderly', 'disability', 'benefits', 'support', 'care',
    
    # Political
    'democracy', 'democratic', 'election', 'vote', 'voting', 'suffrage',
    'reform', 'law', 'legal', 'justice', 'crime', 'criminal', 'police',
    'prison', 'court', 'rights', 'freedom', 'liberty', 'equality',
    'constitution', 'citizenship', 'immigration', 'refugee',
    
    # Military/International
    'war', 'peace', 'military', 'defence', 'defense', 'army', 'navy',
    'air force', 'soldier', 'weapon', 'nuclear', 'foreign', 'international',
    'treaty', 'alliance', 'nato', 'empire', 'colonial', 'commonwealth',
    
    # Geography
    'britain', 'british', 'england', 'english', 'scotland', 'scottish',
    'wales', 'welsh', 'ireland', 'irish', 'london', 'europe', 'european',
    'america', 'american', 'russia', 'russian', 'germany', 'german',
    'france', 'french', 'china', 'chinese', 'india', 'indian',
    
    # Infrastructure
    'transport', 'railway', 'road', 'infrastructure', 'energy', 'power',
    'electricity', 'gas', 'oil', 'coal', 'nuclear', 'renewable',
    'water', 'sewage', 'communication', 'telephone', 'broadcasting',
    
    # Environment/Agriculture
    'agriculture', 'agricultural', 'farming', 'farmer', 'food', 'land',
    'rural', 'urban', 'environment', 'environmental', 'pollution',
    'conservation', 'climate', 'weather', 'resources', 'sustainable',
    
    # Gender/Social Issues  
    'women', 'woman', 'female', 'men', 'man', 'male', 'gender',
    'marriage', 'divorce', 'abortion', 'contraception', 'discrimination',
    'harassment', 'equality', 'feminism', 'suffragette'
}

class HansardAdvancedAnalyzer:
    def __init__(self, data_dir=None, output_dir="analysis/results_advanced",
                 filter_level=3):
        """
        Initialize analyzer with specified filtering level.

        Filter levels:
        0: NONE - No filtering
        1: BASIC - Basic English stop words
        2: PARLIAMENTARY - Add parliamentary terms
        3: MODERATE - Add common verbs and vague words
        4: AGGRESSIVE - Add modal verbs and quantifiers
        5: TFIDF - Use TF-IDF scoring
        6: POS_NOUN - Keep only nouns (requires spacy)
        7: ENTITY - Focus on named entities (requires spacy)
        """
        # Auto-detect data directory
        if data_dir is None:
            # Check common locations
            if Path("../data/processed_fixed").exists():
                data_dir = "../data/processed_fixed"
            elif Path("src/hansard/data/processed_fixed").exists():
                data_dir = "src/hansard/data/processed_fixed"
            else:
                data_dir = "data/processed_fixed"

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.filter_level = filter_level
        self.filter_name = self.get_filter_name(filter_level)
        
        # Build stop words based on filter level
        self.stop_words = self.build_stop_words(filter_level)
        
        # Load gender wordlists - try multiple paths
        male_paths = [
            "../data/gender_wordlists/male_words.txt",
            "src/hansard/data/gender_wordlists/male_words.txt",
            "data/gender_wordlists/male_words.txt"
        ]
        female_paths = [
            "../data/gender_wordlists/female_words.txt",
            "src/hansard/data/gender_wordlists/female_words.txt",
            "data/gender_wordlists/female_words.txt"
        ]

        for path in male_paths:
            if Path(path).exists():
                self.male_words = self._load_gender_wordlist(path)
                break
        else:
            self.male_words = set()

        for path in female_paths:
            if Path(path).exists():
                self.female_words = self._load_gender_wordlist(path)
                break
        else:
            self.female_words = set()
        
        # Analysis results storage
        self.results = {}
        
        # Speaker gender patterns
        self.female_indicators = [
            'Mrs.', 'Miss', 'Lady', 'Dame', 'Madam', 'Viscountess', 'Countess'
        ]
        self.male_indicators = [
            'Mr.', 'Sir', 'Lord', 'Earl', 'Duke', 'Baron', 'Count', 'Marquess'
        ]
        
        # For TF-IDF filtering
        self.tfidf_vectorizer = None
        self.tfidf_scores = None
    
    def get_filter_name(self, level):
        """Get descriptive name for filter level"""
        names = {
            0: "NONE",
            1: "BASIC",
            2: "PARLIAMENTARY", 
            3: "MODERATE",
            4: "AGGRESSIVE",
            5: "TFIDF",
            6: "POS_NOUN",
            7: "ENTITY"
        }
        return names.get(level, "UNKNOWN")
    
    def build_stop_words(self, level):
        """Build stop words set based on filter level"""
        stop_words = set()
        
        if level >= 1:
            stop_words.update(BASIC_STOP_WORDS)
        
        if level >= 2:
            stop_words.update(PARLIAMENTARY_TERMS)
        
        if level >= 3:
            stop_words.update(COMMON_VERBS)
            stop_words.update(VAGUE_WORDS)
        
        if level >= 4:
            stop_words.update(MODAL_AUXILIARY)
            stop_words.update(QUANTIFIERS)
        
        # Always preserve policy terms
        stop_words = stop_words - POLICY_TERMS
        
        return stop_words
    
    def _load_gender_wordlist(self, filepath):
        """Load gender wordlist"""
        try:
            with open(filepath, 'r') as f:
                return set(word.strip().lower() for word in f if word.strip())
        except FileNotFoundError:
            print(f"Warning: Gender wordlist not found at {filepath}")
            return set()
    
    def clean_text(self, text):
        """Clean text based on filter level"""
        if not text:
            return ""
        
        # Level 5: TF-IDF based filtering
        if self.filter_level == 5:
            return self.clean_text_tfidf(text)
        
        # Level 6: POS-based filtering (nouns only)
        if self.filter_level == 6:
            if SPACY_AVAILABLE:
                return self.clean_text_pos(text)
            else:
                print("Spacy not available, falling back to level 4")
                self.filter_level = 4
        
        # Level 7: Entity-based filtering
        if self.filter_level == 7:
            if SPACY_AVAILABLE:
                return self.clean_text_entities(text)
            else:
                print("Spacy not available, falling back to level 4")
                self.filter_level = 4
        
        # Standard word-based filtering (levels 0-4)
        words = text.lower().split()
        
        if self.filter_level == 0:
            return ' '.join(words)
        
        # Filter based on stop words
        filtered_words = []
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w\s]', '', word)
            
            # Keep if: it's a policy term OR (not a stop word AND length > 2)
            if word in POLICY_TERMS or (word not in self.stop_words and len(word) > 2):
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def clean_text_tfidf(self, text):
        """Clean text using TF-IDF scores to keep distinctive words"""
        words = text.lower().split()
        
        # First apply basic filtering
        filtered_words = []
        for word in words:
            word = re.sub(r'[^\w\s]', '', word)
            if word in POLICY_TERMS or (word not in BASIC_STOP_WORDS and len(word) > 2):
                filtered_words.append(word)
        
        # If we have TF-IDF scores, filter by them
        if self.tfidf_scores is not None:
            high_value_words = []
            for word in filtered_words:
                if word in self.tfidf_scores and self.tfidf_scores[word] > 0.001:
                    high_value_words.append(word)
            return ' '.join(high_value_words)
        
        return ' '.join(filtered_words)
    
    def clean_text_pos(self, text):
        """Keep only nouns and proper nouns"""
        doc = nlp(text[:1000000])  # Limit text length for spacy
        
        keep_pos = {'NOUN', 'PROPN'}  # Nouns and proper nouns
        filtered_words = []
        
        for token in doc:
            if (token.pos_ in keep_pos or token.text.lower() in POLICY_TERMS) and len(token.text) > 2:
                filtered_words.append(token.text.lower())
        
        return ' '.join(filtered_words)
    
    def clean_text_entities(self, text):
        """Focus on named entities and policy terms"""
        doc = nlp(text[:1000000])  # Limit text length for spacy
        
        filtered_words = []
        
        # Add named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'LAW', 'NORP']:
                filtered_words.append(ent.text.lower())
        
        # Add policy terms from the text
        for token in doc:
            if token.text.lower() in POLICY_TERMS:
                filtered_words.append(token.text.lower())
        
        return ' '.join(filtered_words)
    
    def compute_tfidf_scores(self, texts):
        """Compute TF-IDF scores for the corpus"""
        if self.filter_level != 5:
            return
        
        print("Computing TF-IDF scores for distinctive word identification...")
        
        # Clean texts with basic filtering first
        cleaned_texts = []
        for text in texts:
            words = text.lower().split()
            filtered = [re.sub(r'[^\w\s]', '', w) for w in words 
                       if w not in BASIC_STOP_WORDS and len(w) > 2]
            cleaned_texts.append(' '.join(filtered))
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        
        # Get average TF-IDF score for each word
        feature_names = vectorizer.get_feature_names_out()
        avg_scores = tfidf_matrix.mean(axis=0).A1
        
        # Store scores
        self.tfidf_scores = dict(zip(feature_names, avg_scores))
        self.tfidf_vectorizer = vectorizer
    
    def load_debates(self, start_year=1803, end_year=2005, sample_size=None):
        """Load debate texts from JSONL files"""
        debates = []
        years_processed = []
        
        for year in range(start_year, end_year + 1):
            jsonl_path = self.data_dir / "content" / str(year) / f"debates_{year}.jsonl"
            
            if not jsonl_path.exists():
                continue
                
            year_debates = []
            try:
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            debate = json.loads(line)
                            debate_data = {
                                'year': year,
                                'text': debate.get('full_text', ''),
                                'title': debate.get('metadata', {}).get('title', ''),
                                'speakers': debate.get('metadata', {}).get('speakers', []),
                                'chamber': debate.get('metadata', {}).get('chamber', ''),
                                'word_count': debate.get('metadata', {}).get('word_count', 0),
                                'date': debate.get('metadata', {}).get('reference_date', ''),
                            }
                            if debate_data['text']:
                                year_debates.append(debate_data)
                
                print(f"Loaded {len(year_debates)} debates from {year}")
                debates.extend(year_debates)
                years_processed.append(year)
                
            except Exception as e:
                print(f"Error loading {year}: {e}")
                continue
        
        # Apply sampling if specified
        if sample_size and len(debates) > sample_size:
            import random
            random.seed(42)
            debates = random.sample(debates, sample_size)
            print(f"Sampled {sample_size} debates from total {len(debates)}")
        
        print(f"\nTotal debates loaded: {len(debates)} from {len(years_processed)} years")
        self.results['years_processed'] = years_processed
        self.results['total_debates'] = len(debates)
        
        return debates
    
    def analyze_unigrams_bigrams(self, debates):
        """Analyze word frequencies with specified filtering"""
        print(f"\nAnalyzing content with {self.filter_name} filtering...")
        
        # For TF-IDF, compute scores first
        if self.filter_level == 5:
            texts = [d['text'] for d in debates]
            self.compute_tfidf_scores(texts)
        
        # Clean all texts
        cleaned_texts = [self.clean_text(d['text']) for d in debates]
        all_text = ' '.join(cleaned_texts)
        
        # Tokenize and count unigrams
        words = re.findall(r'\b[a-z]+\b', all_text.lower())
        
        # Additional filtering for very short words
        if self.filter_level > 0:
            words = [w for w in words if len(w) > 2]
        
        # Get top unigrams
        word_freq = Counter(words)
        top_unigrams = word_freq.most_common(30)
        
        # Bigram analysis
        bigrams = []
        for text in cleaned_texts:
            text_words = re.findall(r'\b[a-z]+\b', text.lower())
            if self.filter_level > 0:
                text_words = [w for w in text_words if len(w) > 2]
            
            for i in range(len(text_words) - 1):
                bigram = (text_words[i], text_words[i+1])
                bigrams.append(bigram)
        
        bigram_freq = Counter(bigrams)
        top_bigrams = bigram_freq.most_common(20)
        
        self.results['top_unigrams'] = top_unigrams
        self.results['top_bigrams'] = top_bigrams
        
        print(f"Top 10 words ({self.filter_name} filtering):")
        for word, count in top_unigrams[:10]:
            print(f"  {word}: {count:,}")
            
        print(f"\nTop 10 bigrams ({self.filter_name} filtering):")
        for bigram, count in top_bigrams[:10]:
            print(f"  {' '.join(bigram)}: {count:,}")
        
        # Calculate filtering statistics
        original_word_count = sum([d['word_count'] for d in debates])
        filtered_word_count = len(words)
        reduction_pct = (1 - filtered_word_count/original_word_count) * 100 if original_word_count > 0 else 0
        
        print(f"\nFiltering statistics:")
        print(f"  Original words: {original_word_count:,}")
        print(f"  Filtered words: {filtered_word_count:,}")
        print(f"  Reduction: {reduction_pct:.1f}%")
        
        self.results['filtering_stats'] = {
            'original_words': original_word_count,
            'filtered_words': filtered_word_count,
            'reduction_pct': reduction_pct
        }
    
    def perform_topic_modeling(self, debates, n_topics=10):
        """Perform LDA topic modeling"""
        print(f"\nPerforming topic modeling ({self.filter_name} filtering)...")
        
        # Clean texts
        cleaned_texts = [self.clean_text(d['text']) for d in debates]
        
        # Filter out empty or very short texts
        valid_texts = [t for t in cleaned_texts if len(t.split()) > 5]
        
        if len(valid_texts) < 50:
            print("Not enough valid text after filtering for topic modeling")
            return
        
        # Use appropriate vectorizer based on filter level
        if self.filter_level >= 5:
            max_features = 500
        else:
            max_features = 1000
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=3,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        try:
            doc_term_matrix = vectorizer.fit_transform(valid_texts)
            
            # Fit LDA model
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10,
                learning_method='online'
            )
            lda.fit(doc_term_matrix)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract and display topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-15:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topic_weight = topic[top_indices].tolist()
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words[:10],
                    'weights': topic_weight[:10]
                })
                
                print(f"\nTopic {topic_idx + 1}:")
                print(f"  {', '.join(top_words[:10])}")
            
            self.results['topics'] = topics
            
        except Exception as e:
            print(f"Error in topic modeling: {e}")
            self.results['topics'] = []
    
    def analyze_gender_language(self, debates):
        """Analyze gender-related language patterns"""
        print(f"\nAnalyzing gender language patterns...")
        
        male_count = 0
        female_count = 0
        male_word_freq = Counter()
        female_word_freq = Counter()
        
        for debate in debates:
            words = re.findall(r'\b[a-z]+\b', debate['text'].lower())
            
            for word in words:
                if word in self.male_words:
                    male_count += 1
                    male_word_freq[word] += 1
                elif word in self.female_words:
                    female_count += 1
                    female_word_freq[word] += 1
        
        total_gendered = male_count + female_count
        male_ratio = male_count / total_gendered if total_gendered > 0 else 0
        female_ratio = female_count / total_gendered if total_gendered > 0 else 0
        
        self.results['gender_analysis'] = {
            'male_word_count': male_count,
            'female_word_count': female_count,
            'male_ratio': male_ratio,
            'female_ratio': female_ratio,
            'top_male_words': male_word_freq.most_common(10),
            'top_female_words': female_word_freq.most_common(10)
        }
        
        print(f"Gender word distribution:")
        print(f"  Male words: {male_count:,} ({male_ratio:.1%})")
        print(f"  Female words: {female_count:,} ({female_ratio:.1%})")
    
    def analyze_temporal_patterns(self, debates):
        """Analyze temporal patterns around 1928"""
        print(f"\nAnalyzing temporal patterns...")
        
        # Split debates pre/post 1928
        pre_1928 = [d for d in debates if d['year'] < 1928]
        post_1928 = [d for d in debates if d['year'] >= 1928]
        
        if not pre_1928 or not post_1928:
            print("Insufficient data for temporal analysis around 1928")
            return
        
        # Analyze gender language before and after 1928
        def get_gender_ratio(debate_list):
            male_count = 0
            female_count = 0
            
            for debate in debate_list:
                words = re.findall(r'\b[a-z]+\b', debate['text'].lower())
                for word in words:
                    if word in self.male_words:
                        male_count += 1
                    elif word in self.female_words:
                        female_count += 1
            
            total = male_count + female_count
            return {
                'male_ratio': male_count / total if total > 0 else 0,
                'female_ratio': female_count / total if total > 0 else 0,
                'total_gendered_words': total
            }
        
        pre_1928_gender = get_gender_ratio(pre_1928)
        post_1928_gender = get_gender_ratio(post_1928)
        
        self.results['temporal_analysis'] = {
            'pre_1928': {
                'debate_count': len(pre_1928),
                'gender_ratios': pre_1928_gender
            },
            'post_1928': {
                'debate_count': len(post_1928),
                'gender_ratios': post_1928_gender
            }
        }
        
        print(f"\nPre-1928 (women's suffrage):")
        print(f"  Debates: {len(pre_1928):,}")
        print(f"  Female language ratio: {pre_1928_gender['female_ratio']:.2%}")
        
        print(f"\nPost-1928:")
        print(f"  Debates: {len(post_1928):,}")
        print(f"  Female language ratio: {post_1928_gender['female_ratio']:.2%}")
        
        change = post_1928_gender['female_ratio'] - pre_1928_gender['female_ratio']
        print(f"\nChange in female language ratio: {change:+.2%}")
    
    def create_visualizations(self):
        """Create individual plot visualizations like the original"""
        print(f"\nCreating visualizations...")
        
        # Create plots directory
        plots_dir = self.output_dir / f'plots_level_{self.filter_level}'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Top Unigrams
        if 'top_unigrams' in self.results:
            plt.figure(figsize=(12, 8))
            words, counts = zip(*self.results['top_unigrams'][:25])
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words)))
            
            plt.barh(range(len(words)), counts, color=colors)
            plt.yticks(range(len(words)), words)
            plt.xlabel('Frequency', fontsize=12)
            plt.title(f'Top 25 Words - {self.filter_name} Filtering (Level {self.filter_level})', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, v in enumerate(counts):
                plt.text(v + max(counts)*0.01, i, f'{v:,}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'top_unigrams.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Top Bigrams
        if 'top_bigrams' in self.results:
            plt.figure(figsize=(12, 8))
            bigrams, counts = zip(*self.results['top_bigrams'][:20])
            bigram_labels = [' '.join(b) for b in bigrams]
            colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(bigram_labels)))
            
            plt.barh(range(len(bigram_labels)), counts, color=colors)
            plt.yticks(range(len(bigram_labels)), bigram_labels)
            plt.xlabel('Frequency', fontsize=12)
            plt.title(f'Top 20 Bigrams - {self.filter_name} Filtering (Level {self.filter_level})', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, v in enumerate(counts):
                plt.text(v + max(counts)*0.01, i, f'{v:,}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'top_bigrams.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Gender Word Distribution
        if 'gender_analysis' in self.results:
            plt.figure(figsize=(10, 8))
            gender_data = self.results['gender_analysis']
            
            labels = ['Male Words', 'Female Words']
            sizes = [gender_data['male_word_count'], gender_data['female_word_count']]
            colors = ['#4A90E2', '#F5A623']
            explode = (0.05, 0.05)
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                   startangle=90, explode=explode, shadow=True)
            plt.title(f'Gender Word Distribution - {self.filter_name} Filtering', 
                     fontsize=14, fontweight='bold')
            plt.axis('equal')
            
            plt.savefig(plots_dir / 'gender_word_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Temporal Gender Comparison (Pre/Post 1928)
        if 'temporal_analysis' in self.results:
            plt.figure(figsize=(10, 8))
            temporal = self.results['temporal_analysis']
            
            categories = ['Pre-1928\n(Before Women\'s Suffrage)', 'Post-1928\n(After Women\'s Suffrage)']
            female_ratios = [
                temporal['pre_1928']['gender_ratios']['female_ratio'] * 100,
                temporal['post_1928']['gender_ratios']['female_ratio'] * 100
            ]
            male_ratios = [
                temporal['pre_1928']['gender_ratios']['male_ratio'] * 100,
                temporal['post_1928']['gender_ratios']['male_ratio'] * 100
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars1 = ax.bar(x - width/2, male_ratios, width, label='Male Words', 
                          color='#4A90E2', alpha=0.8)
            bars2 = ax.bar(x + width/2, female_ratios, width, label='Female Words', 
                          color='#F5A623', alpha=0.8)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom')
            
            ax.set_xlabel('Period', fontsize=12)
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_title(f'Gender Language Evolution - {self.filter_name} Filtering', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend(loc='upper left')
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'temporal_gender_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 5. Topic Modeling Visualization
        if 'topics' in self.results and self.results['topics']:
            plt.figure(figsize=(14, 10))
            n_topics = min(8, len(self.results['topics']))
            
            for i in range(n_topics):
                topic = self.results['topics'][i]
                words = topic.get('words', [])[:8]
                
                plt.subplot(2, 4, i+1)
                word_heights = list(range(len(words), 0, -1))
                colors = plt.cm.Set3(i)
                
                plt.barh(range(len(words)), word_heights, color=colors, alpha=0.7)
                plt.yticks(range(len(words)), words, fontsize=10)
                plt.title(f'Topic {i+1}', fontweight='bold')
                plt.gca().invert_yaxis()
                plt.xlabel('Importance')
                plt.xticks([])
            
            plt.suptitle(f'Topic Modeling Results - {self.filter_name} Filtering (Level {self.filter_level})', 
                        fontweight='bold', fontsize=16)
            plt.tight_layout()
            plt.savefig(plots_dir / 'topic_modeling_overview.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 6. Filtering Impact Visualization
        if 'filtering_stats' in self.results:
            plt.figure(figsize=(10, 8))
            stats = self.results['filtering_stats']
            
            # Create donut chart showing filtering impact
            sizes = [stats['filtered_words'], stats['original_words'] - stats['filtered_words']]
            labels = ['Retained Words', 'Filtered Out']
            colors = ['#2ECC71', '#E74C3C']
            explode = (0.1, 0)
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, explode=explode, shadow=True, 
                   wedgeprops=dict(width=0.5))
            
            # Add center text
            plt.text(0, 0, f'{stats["reduction_pct"]:.1f}%\nReduction', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            
            plt.title(f'Filtering Impact - {self.filter_name} (Level {self.filter_level})\n'
                     f'Original: {stats["original_words"]:,} words | Filtered: {stats["filtered_words"]:,} words',
                     fontsize=14, fontweight='bold')
            
            plt.axis('equal')
            plt.savefig(plots_dir / 'filtering_impact.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 7. Word Cloud
        if 'top_unigrams' in self.results:
            from wordcloud import WordCloud
            
            # Create frequency dictionary
            word_freq = dict(self.results['top_unigrams'][:100])
            
            # Generate word cloud
            wordcloud = WordCloud(width=1600, height=800,
                                 background_color='white',
                                 colormap='viridis',
                                 relative_scaling=0.5,
                                 min_font_size=10).generate_from_frequencies(word_freq)
            
            plt.figure(figsize=(20, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud - {self.filter_name} Filtering (Level {self.filter_level})', 
                     fontsize=20, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'wordcloud.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Individual plots saved to {plots_dir}/")
    
    def save_results(self):
        """Save analysis results"""
        output_path = self.output_dir / f'hansard_results_level_{self.filter_level}.json'
        
        # Add metadata
        self.results['filter_level'] = self.filter_level
        self.results['filter_name'] = self.filter_name
        
        # Convert for JSON
        json_results = {}
        for key, value in self.results.items():
            if key in ['top_unigrams', 'top_bigrams']:
                json_results[key] = [[word, count] if isinstance(word, str) 
                                     else [list(word), count] 
                                     for word, count in value]
            elif key == 'gender_analysis':
                json_results[key] = {
                    k: v if not isinstance(v, list) else 
                    [[word, count] for word, count in v]
                    for k, v in value.items()
                }
            else:
                json_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    def run_analysis(self, start_year=1803, end_year=2005, sample_size=None):
        """Run complete analysis pipeline"""
        print(f"\n{'='*60}")
        print(f"Hansard NLP Analysis")
        print(f"Filter Level: {self.filter_level} ({self.filter_name})")
        print(f"Years: {start_year}-{end_year}")
        if sample_size:
            print(f"Sample size: {sample_size}")
        print(f"{'='*60}")
        
        # Load debates
        debates = self.load_debates(start_year, end_year, sample_size)
        
        if not debates:
            print("No debates loaded. Exiting.")
            return
        
        # Run analyses
        self.analyze_unigrams_bigrams(debates)
        self.perform_topic_modeling(debates)
        self.analyze_gender_language(debates)
        self.analyze_temporal_patterns(debates)
        
        # Create visualizations and save
        self.create_visualizations()
        self.save_results()
        
        print(f"\n{'='*60}")
        print(f"Analysis complete for {self.filter_name} filtering!")
        print(f"{'='*60}")
        
        return self.results

def run_all_levels(start_year, end_year, sample_size):
    """Run analysis at all filter levels for comparison"""
    
    all_results = {}
    summary_data = []
    
    # Determine which levels to run
    max_level = 7 if SPACY_AVAILABLE else 5
    
    for level in range(0, max_level + 1):
        print(f"\n\n{'#'*80}")
        print(f"RUNNING FILTER LEVEL {level}")
        print(f"{'#'*80}")
        
        analyzer = HansardAdvancedAnalyzer(filter_level=level)
        results = analyzer.run_analysis(start_year, end_year, sample_size)
        
        all_results[level] = results
        
        # Collect summary statistics
        if results and 'filtering_stats' in results:
            summary_data.append({
                'level': level,
                'name': analyzer.filter_name,
                'filtered_words': results['filtering_stats']['filtered_words'],
                'reduction_pct': results['filtering_stats']['reduction_pct'],
                'top_word': results['top_unigrams'][0] if results.get('top_unigrams') else ('', 0)
            })
    
    # Create comparison visualization
    create_comparison_plot(summary_data)
    
    # Save combined results
    output_path = Path("analysis/results_advanced") / "all_levels_comparison.json"
    with open(output_path, 'w') as f:
        json.dump({
            'summary': summary_data,
            'details': {str(k): v for k, v in all_results.items()}
        }, f, indent=2)
    
    print(f"\n\nComparison results saved to {output_path}")
    
    return all_results

def create_comparison_plot(summary_data):
    """Create comparison plot of all filter levels"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Filter Level Comparison', fontsize=16)
    
    levels = [d['level'] for d in summary_data]
    names = [d['name'] for d in summary_data]
    filtered_words = [d['filtered_words'] for d in summary_data]
    reduction_pcts = [d['reduction_pct'] for d in summary_data]
    
    # 1. Word count by level
    axes[0, 0].bar(levels, filtered_words, color='steelblue')
    axes[0, 0].set_xlabel('Filter Level')
    axes[0, 0].set_ylabel('Filtered Word Count')
    axes[0, 0].set_title('Words Remaining After Filtering')
    axes[0, 0].set_xticks(levels)
    axes[0, 0].set_xticklabels([f"{l}\n{n}" for l, n in zip(levels, names)], 
                               rotation=45, ha='right', fontsize=8)
    
    # 2. Reduction percentage
    axes[0, 1].plot(levels, reduction_pcts, marker='o', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Filter Level')
    axes[0, 1].set_ylabel('Reduction (%)')
    axes[0, 1].set_title('Filtering Reduction Percentage')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(levels)
    
    # 3. Top words comparison
    top_words_text = "\n".join([f"Level {d['level']} ({d['name']}): {d['top_word'][0]}" 
                                for d in summary_data if d.get('top_word')])
    axes[1, 0].text(0.1, 0.9, top_words_text, transform=axes[1, 0].transAxes,
                   fontsize=10, verticalalignment='top')
    axes[1, 0].set_title('Top Word at Each Filter Level')
    axes[1, 0].axis('off')
    
    # 4. Filter descriptions
    descriptions = [
        "Level 0: No filtering",
        "Level 1: Basic stop words",
        "Level 2: + Parliamentary terms",
        "Level 3: + Common verbs, vague words",
        "Level 4: + Modal verbs, quantifiers",
        "Level 5: TF-IDF scoring",
        "Level 6: Nouns only (POS)",
        "Level 7: Named entities"
    ]
    desc_text = "\n".join(descriptions[:len(summary_data)])
    axes[1, 1].text(0.1, 0.9, desc_text, transform=axes[1, 1].transAxes,
                   fontsize=9, verticalalignment='top')
    axes[1, 1].set_title('Filter Level Descriptions')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = Path("analysis/results_advanced") / "filter_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Hansard Analysis with Multiple Filter Levels')
    parser.add_argument('--years', type=str, help='Year range (e.g., 1920-1930)')
    parser.add_argument('--sample', type=int, help='Sample size for analysis')
    parser.add_argument('--filter-level', type=int, default=3, 
                       choices=[0, 1, 2, 3, 4, 5, 6, 7],
                       help='Filter level (0=none, 7=maximum)')
    parser.add_argument('--all-levels', action='store_true', 
                       help='Run all filter levels for comparison')
    parser.add_argument('--full', action='store_true', help='Analyze full corpus')
    
    args = parser.parse_args()
    
    # Determine year range
    if args.years:
        year_parts = args.years.split('-')
        start_year = int(year_parts[0])
        end_year = int(year_parts[1]) if len(year_parts) > 1 else start_year
    elif args.full:
        start_year = 1803
        end_year = 2005
    else:
        start_year = 1920
        end_year = 1935
    
    # Sample size
    sample_size = args.sample if not args.full else None
    
    # Run analysis
    if args.all_levels:
        run_all_levels(start_year, end_year, sample_size)
    else:
        analyzer = HansardAdvancedAnalyzer(filter_level=args.filter_level)
        analyzer.run_analysis(start_year, end_year, sample_size)

if __name__ == "__main__":
    main()