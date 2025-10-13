#!/usr/bin/env python3
"""
Unified Text Filtering for Hansard Parliamentary Debates

Single source of truth for text filtering across all analysis scripts.
Provides multiple filtering levels with documented justifications based on
corpus frequency analysis.

Filtering Levels:
    - minimal: Remove only artifacts and noise
    - basic: NLTK English stop words
    - parliamentary: + Parliamentary procedural terms
    - moderate: + Common verbs, vague words (recommended for most analyses)
    - aggressive: Maximum filtering for focused topic analysis

Special Modes:
    - tfidf: Use TF-IDF scores to identify distinctive words
    - pos_noun: Extract nouns only (requires spaCy)
    - pos_entity: Extract named entities only (requires spaCy)

Usage:
    from unified_text_filtering import HansardTextFilter

    filter = HansardTextFilter(level='moderate')
    cleaned_text = filter.filter_text(raw_text)
    bigrams = filter.extract_bigrams(cleaned_text)
"""

import re
from collections import Counter
from typing import List, Set, Tuple, Optional
import pandas as pd

# Try to import NLTK for standard stop words
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# Try to import spaCy for POS/NER filtering
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False


# Policy-relevant terms to always preserve (from hansard_nlp_analysis.py)
# These are substantive policy terms that should never be filtered
POLICY_TERMS = {
    # Economic
    'economy', 'economic', 'finance', 'financial', 'budget', 'tax', 'taxation',
    'revenue', 'spending', 'deficit', 'debt', 'inflation', 'unemployment',
    'employment', 'wages', 'salary', 'pension', 'insurance', 'banking',
    'investment', 'market', 'trade', 'export', 'import', 'tariff', 'industry',
    'industrial', 'manufacturing', 'production', 'business', 'commerce',
    'capitalism', 'socialism', 'nationalisation', 'privatisation', 'nationalization',
    'privatization',

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
    'soldier', 'weapon', 'nuclear', 'foreign', 'international',
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


class HansardTextFilter:
    """
    Unified text filtering for Hansard parliamentary debates.

    Provides consistent filtering across all analysis scripts with documented
    justifications based on corpus frequency analysis.
    """

    def __init__(self, level: str = 'moderate', preserve_policy_terms: bool = True):
        """
        Initialize filter with specified level.

        Args:
            level: Filtering level (minimal, basic, parliamentary, moderate, aggressive)
            preserve_policy_terms: Always keep policy-relevant terms regardless of level
        """
        self.level = level
        self.preserve_policy_terms = preserve_policy_terms
        self.stop_words = self._build_stop_words(level)
        self.tfidf_scores = None  # Set externally if using TF-IDF mode

    def _get_basic_stop_words(self) -> Set[str]:
        """
        Get basic English stop words from NLTK.

        Falls back to minimal set if NLTK not available.
        """
        if NLTK_AVAILABLE:
            base_stops = set(stopwords.words('english'))
        else:
            # Minimal fallback set
            base_stops = {
                'the', 'of', 'to', 'and', 'a', 'an', 'in', 'is', 'it', 'that',
                'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'at',
                'be', 'this', 'have', 'from', 'or', 'had', 'by', 'not', 'but',
                'what', 'all', 'were', 'we', 'when', 'there', 'can', 'said',
                'which', 'do', 'their', 'if', 'will', 'up', 'other', 'about',
                'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
                'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
                'very', 'been', 'am'
            }

        # Add missing auxiliaries that NLTK doesn't include
        # Based on analysis showing these appear in >70% of speeches with no semantic value
        missing_auxiliaries = {'could', 'might', 'shall', 'must', 'ought'}

        return base_stops | missing_auxiliaries

    def _get_parliamentary_stop_words(self) -> Set[str]:
        """
        Parliamentary procedural terms.

        Based on corpus frequency analysis showing these words appear in
        >80% of speeches but carry no topic information.

        Frequency data from enhanced_gender_corpus_analysis.py:
        - 'hon': 89% of speeches
        - 'honourable': 84% of speeches
        - 'member': 91% of speeches
        - 'house': 93% of speeches
        - etc.
        """
        basic = self._get_basic_stop_words()

        # Parliamentary terms with documented corpus frequencies
        parliamentary = {
            # Honorifics and titles (appear in 68-93% of speeches)
            'hon', 'honourable', 'right', 'gentleman', 'lady', 'member', 'members',
            'house', 'speaker', 'sir', 'madam', 'mr', 'mrs', 'ms',

            # Procedural terms (appear in 70-85% of speeches)
            'lord', 'lords', 'gallant', 'learned', 'friend', 'friends', 'noble',

            # Legislative process (appear in 60-80% of speeches)
            'bill', 'clause', 'amendment', 'committee', 'order', 'question',
            'division', 'reading', 'report', 'stage', 'moved', 'second', 'third',

            # Generic government terms (appear in 75-85% of speeches)
            'government', 'minister', 'secretary', 'state', 'department',

            # Debate structure
            'debate', 'discuss', 'speech', 'words', 'statement'
        }

        return basic | parliamentary

    def _get_moderate_stop_words(self) -> Set[str]:
        """
        Moderate filtering: parliamentary + common verbs and vague words.

        Recommended level for most analyses. Removes high-frequency words
        that appear in >60% of speeches but don't distinguish topics.
        """
        parliamentary = self._get_parliamentary_stop_words()

        # Common verbs appearing in >70% of speeches
        common_verbs = {
            'make', 'made', 'making', 'makes', 'take', 'took', 'taken', 'taking',
            'give', 'gave', 'given', 'giving', 'put', 'puts', 'putting',
            'get', 'got', 'getting', 'gets', 'come', 'came', 'coming', 'comes',
            'go', 'went', 'going', 'goes', 'gone', 'say', 'said', 'saying', 'says',
            'think', 'thought', 'thinking', 'thinks', 'know', 'knew', 'known', 'knowing',
            'see', 'saw', 'seeing', 'sees', 'seen', 'want', 'wanted', 'wanting', 'wants',
            'look', 'looked', 'looking', 'looks', 'find', 'found', 'finding', 'finds',
            'tell', 'told', 'telling', 'tells', 'ask', 'asked', 'asking', 'asks',
            'seem', 'seemed', 'seeming', 'seems', 'feel', 'felt', 'feeling', 'feels',
            'try', 'tried', 'trying', 'tries', 'leave', 'left', 'leaving', 'leaves',
            'call', 'called', 'calling', 'calls', 'keep', 'kept', 'keeping', 'keeps',
            'believe', 'believed', 'hope', 'hoped', 'wish', 'wished', 'need', 'needed',
            'use', 'used', 'using', 'uses', 'work', 'worked', 'working', 'works'
        }

        # Vague/generic words (>65% frequency)
        vague_words = {
            'thing', 'things', 'something', 'anything', 'nothing', 'everything',
            'way', 'ways', 'case', 'cases', 'matter', 'matters', 'fact', 'facts',
            'time', 'times', 'place', 'places', 'part', 'parts', 'kind', 'kinds',
            'point', 'points', 'view', 'views', 'position', 'number', 'numbers',
            'example', 'examples', 'word', 'words', 'subject', 'course', 'present',
            'regard', 'deal', 'side', 'end', 'result', 'means', 'moment'
        }

        # Temporal markers (65-80% frequency)
        temporal = {
            'today', 'yesterday', 'tomorrow', 'now', 'then', 'year', 'years',
            'day', 'days', 'week', 'weeks', 'month', 'months', 'last', 'next',
            'first', 'second'
        }

        return parliamentary | common_verbs | vague_words | temporal

    def _get_aggressive_stop_words(self) -> Set[str]:
        """
        Aggressive filtering for focused topic analysis.

        Adds discourse markers, quantifiers, and modal verbs appearing in
        >60% of speeches.
        """
        moderate = self._get_moderate_stop_words()

        # Discourse markers (>60% frequency)
        discourse = {
            'well', 'yes', 'no', 'indeed', 'perhaps', 'certainly', 'obviously',
            'clearly', 'surely', 'really', 'quite', 'sure', 'also', 'however',
            'therefore', 'whether', 'still', 'always', 'never', 'yet', 'already',
            'rather', 'even', 'far'
        }

        # Quantifiers and measurements
        quantifiers = {
            'much', 'many', 'more', 'most', 'less', 'least', 'few', 'fewer',
            'several', 'per', 'cent', 'percent', 'hundred', 'thousand', 'million'
        }

        # Common adjectives with little discriminative value
        adjectives = {
            'new', 'old', 'good', 'bad', 'great', 'small', 'large', 'important',
            'different', 'same', 'certain', 'possible', 'necessary', 'clear',
            'particular', 'general', 'special', 'full', 'long', 'short', 'high', 'low'
        }

        # Words identified as noise in previous analysis
        noise = {'out'}  # Appeared as #1 word but has no clear meaning

        return moderate | discourse | quantifiers | adjectives | noise

    def _get_ultra_stop_words(self) -> Set[str]:
        """
        Ultra filtering using collaborator's categorized stopword list.

        Uses hansard_stopwords.csv where words are marked as:
        - CONDITIONAL: Should be filtered (371 words)
        - KEEP: Content words (268 words)

        This is the most aggressive filtering, based on domain expertise.
        """
        aggressive = self._get_aggressive_stop_words()

        # Load collaborator's categorized stopwords
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.path_config import Paths

            stopwords_csv = Paths.DATA_DIR / 'word_lists' / 'hansard_stopwords.csv'
            df = pd.read_csv(stopwords_csv)

            # Get all CONDITIONAL words (marked for filtering)
            conditional_words = set(df[df['type'] == 'CONDITIONAL']['token'].str.lower())

            print(f"  Loaded {len(conditional_words)} CONDITIONAL words from collaborator's list")

            return aggressive | conditional_words

        except FileNotFoundError:
            print("  Warning: hansard_stopwords.csv not found, using aggressive")
            return aggressive
        except Exception as e:
            print(f"  Warning: Could not load stopwords CSV: {e}")
            return aggressive

    def _build_stop_words(self, level: str) -> Set[str]:
        """
        Build stop word set based on filtering level.

        Args:
            level: One of minimal, basic, parliamentary, moderate, aggressive

        Returns:
            Set of stop words for this level
        """
        if level == 'minimal':
            # Only remove artifacts and noise
            return {'nbsp', 'quot', 'amp', 'lt', 'gt', 'mdash', 'ndash'}
        elif level == 'basic':
            return self._get_basic_stop_words()
        elif level == 'parliamentary':
            return self._get_parliamentary_stop_words()
        elif level == 'moderate':
            return self._get_moderate_stop_words()
        elif level == 'aggressive':
            return self._get_aggressive_stop_words()
        elif level == 'ultra':
            return self._get_ultra_stop_words()
        else:
            raise ValueError(f"Unknown filtering level: {level}. "
                           f"Use: minimal, basic, parliamentary, moderate, aggressive, or ultra")

    def filter_text(self, text: str, min_word_length: int = 3) -> str:
        """
        Filter text according to current level.

        Args:
            text: Input text
            min_word_length: Minimum word length to keep

        Returns:
            Filtered text as string
        """
        if not text:
            return ""

        # Tokenize
        words = re.findall(r'\b[a-z]+\b', text.lower())

        # Filter
        filtered_words = []
        for word in words:
            # Keep if: policy term OR (not stop word AND meets length requirement)
            if self.preserve_policy_terms and word in POLICY_TERMS:
                filtered_words.append(word)
            elif word not in self.stop_words and len(word) >= min_word_length:
                filtered_words.append(word)

        return ' '.join(filtered_words)

    def filter_text_tfidf(self, text: str, min_score: float = 0.001,
                         min_word_length: int = 3) -> str:
        """
        Filter text using TF-IDF scores to keep only distinctive words.

        Requires self.tfidf_scores to be set externally (dict of word -> score).

        Args:
            text: Input text
            min_score: Minimum TF-IDF score to keep word
            min_word_length: Minimum word length

        Returns:
            Filtered text containing only high TF-IDF words
        """
        if self.tfidf_scores is None:
            raise ValueError("TF-IDF scores not set. Call set_tfidf_scores() first.")

        # First apply basic filtering
        basic_filtered = self.filter_text(text, min_word_length)
        words = basic_filtered.split()

        # Keep only words with high TF-IDF scores
        tfidf_filtered = []
        for word in words:
            if word in self.tfidf_scores and self.tfidf_scores[word] >= min_score:
                tfidf_filtered.append(word)
            elif self.preserve_policy_terms and word in POLICY_TERMS:
                tfidf_filtered.append(word)

        return ' '.join(tfidf_filtered)

    def filter_text_pos(self, text: str, pos_tags: List[str] = None,
                       min_word_length: int = 3) -> str:
        """
        Filter text by POS (part-of-speech) tags using spaCy.

        Args:
            text: Input text
            pos_tags: List of POS tags to keep (default: ['NOUN', 'PROPN'])
            min_word_length: Minimum word length

        Returns:
            Filtered text containing only specified POS tags
        """
        if not SPACY_AVAILABLE:
            raise RuntimeError("spaCy not available. Install with: pip install spacy && "
                             "python -m spacy download en_core_web_sm")

        if pos_tags is None:
            pos_tags = ['NOUN', 'PROPN']  # Nouns and proper nouns by default

        # Limit text length for spaCy processing
        text = text[:1000000]
        doc = nlp(text)

        filtered_words = []
        for token in doc:
            word = token.text.lower()
            if ((token.pos_ in pos_tags or word in POLICY_TERMS) and
                len(word) >= min_word_length):
                filtered_words.append(word)

        return ' '.join(filtered_words)

    def filter_text_entities(self, text: str, entity_types: List[str] = None,
                            min_word_length: int = 3) -> str:
        """
        Filter text to extract only named entities using spaCy.

        Args:
            text: Input text
            entity_types: List of entity types to keep (default: important types)
            min_word_length: Minimum word length

        Returns:
            Filtered text containing only named entities and policy terms
        """
        if not SPACY_AVAILABLE:
            raise RuntimeError("spaCy not available. Install with: pip install spacy && "
                             "python -m spacy download en_core_web_sm")

        if entity_types is None:
            # Default to important entity types
            entity_types = ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'LAW', 'NORP']

        # Limit text length for spaCy processing
        text = text[:1000000]
        doc = nlp(text)

        filtered_words = []

        # Add named entities
        for ent in doc.ents:
            if ent.label_ in entity_types:
                filtered_words.append(ent.text.lower())

        # Add policy terms that appear in text
        for token in doc:
            word = token.text.lower()
            if word in POLICY_TERMS and len(word) >= min_word_length:
                filtered_words.append(word)

        return ' '.join(filtered_words)

    def extract_bigrams(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract bigrams from filtered text.

        Args:
            text: Input text (should be pre-filtered)

        Returns:
            List of bigrams as tuples
        """
        words = text.split()
        bigrams = []

        for i in range(len(words) - 1):
            bigrams.append((words[i], words[i + 1]))

        return bigrams

    def set_tfidf_scores(self, tfidf_scores: dict):
        """
        Set TF-IDF scores for TF-IDF filtering mode.

        Args:
            tfidf_scores: Dict mapping word -> TF-IDF score
        """
        self.tfidf_scores = tfidf_scores

    def get_filtering_stats(self, original_text: str, filtered_text: str) -> dict:
        """
        Calculate filtering statistics.

        Args:
            original_text: Original unfiltered text
            filtered_text: Filtered text

        Returns:
            Dict with statistics
        """
        original_words = len(original_text.split())
        filtered_words = len(filtered_text.split())
        reduction_pct = ((original_words - filtered_words) / original_words * 100
                        if original_words > 0 else 0)

        return {
            'original_words': original_words,
            'filtered_words': filtered_words,
            'words_removed': original_words - filtered_words,
            'reduction_pct': reduction_pct,
            'level': self.level,
            'stop_words_count': len(self.stop_words),
            'policy_terms_preserved': len(POLICY_TERMS) if self.preserve_policy_terms else 0
        }


# Convenience function for quick filtering
def filter_hansard_text(text: str, level: str = 'moderate') -> str:
    """
    Quick convenience function for filtering Hansard text.

    Args:
        text: Input text
        level: Filtering level

    Returns:
        Filtered text
    """
    filter = HansardTextFilter(level=level)
    return filter.filter_text(text)
