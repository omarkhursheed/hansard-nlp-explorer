"""
Parliamentary Stop Words for Hansard Analysis

This module provides comprehensive lists of parliamentary procedural language
and titles that should be filtered out for meaningful content analysis.
"""

# Titles and honorifics
TITLES = {
    'hon', 'honourable', 'right', 'noble', 'lord', 'lords', 'lady', 'ladies',
    'sir', 'mr', 'mrs', 'ms', 'miss', 'dr', 'dame', 'baron', 'baroness',
    'viscount', 'viscountess', 'earl', 'countess', 'marquis', 'marquess',
    'duke', 'duchess', 'prince', 'princess', 'king', 'queen', 'majesty',
    'excellency', 'reverent', 'reverend', 'bishop', 'archbishop',
    'gentleman', 'gentlemen', 'member', 'members', 'friend', 'friends',
    'colleague', 'colleagues', 'secretary', 'minister', 'ministers',
    'prime', 'deputy', 'chancellor', 'speaker', 'chairman', 'chairwoman'
}

# Parliamentary procedural terms
PROCEDURAL = {
    'house', 'commons', 'parliament', 'chamber', 'bench', 'benches',
    'question', 'questions', 'answer', 'answers', 'asked', 'ask', 'asking',
    'said', 'say', 'saying', 'says', 'stated', 'statement', 'statements',
    'debate', 'debates', 'debating', 'debated', 'discussion', 'discussing',
    'bill', 'bills', 'act', 'acts', 'clause', 'clauses', 'amendment', 'amendments',
    'committee', 'committees', 'division', 'divisions', 'vote', 'votes', 'voting',
    'motion', 'motions', 'moved', 'move', 'moving', 'seconded', 'second',
    'order', 'orders', 'ordered', 'ordering', 'point', 'points',
    'hear', 'heard', 'hearing', 'rise', 'rising', 'rose', 'risen',
    'sitting', 'sittings', 'session', 'sessions', 'adjournment', 'adjourn',
    'leave', 'permission', 'beg', 'begs', 'begged', 'begging',
    'table', 'tabled', 'tabling', 'paper', 'papers', 'report', 'reports',
    'reading', 'readings', 'first', 'second', 'third', 'stage', 'stages',
    'ayes', 'noes', 'aye', 'no', 'yes', 'agreed', 'disagreed',
    'carried', 'negatived', 'withdrawn', 'withdraw', 'withdrawing',
    'proceed', 'proceedings', 'proceeding', 'business', 'matter', 'matters',
    'notice', 'notices', 'notified', 'notification', 'inform', 'informed',
    'wish', 'wishes', 'wished', 'wishing', 'hope', 'hopes', 'hoped',
    'think', 'thinks', 'thought', 'thinking', 'believe', 'believes', 'believed',
    'consider', 'considers', 'considered', 'considering', 'consideration',
    'understand', 'understands', 'understood', 'understanding',
    'aware', 'unaware', 'know', 'knows', 'knew', 'knowing', 'knowledge'
}

# Common parliamentary phrases (as bigrams/trigrams)
PHRASE_COMPONENTS = {
    'right hon', 'hon gentleman', 'hon friend', 'hon member', 'noble lord',
    'secretary state', 'hon members', 'learned friend', 'noble friend',
    'opposite side', 'front bench', 'back bench', 'dispatch box',
    'order paper', 'standing order', 'point order', 'hear hear',
    'prime minister', 'foreign secretary', 'home secretary', 'chancellor exchequer'
}

# Additional common but non-informative words in parliamentary context
COMMON_PARLIAMENTARY = {
    'would', 'could', 'should', 'shall', 'will', 'may', 'might', 'must',
    'can', 'cannot', 'able', 'unable', 'wish', 'want', 'wanted', 'wants',
    'great', 'good', 'better', 'best', 'bad', 'worse', 'worst',
    'very', 'much', 'many', 'more', 'most', 'less', 'least', 'few', 'fewer',
    'important', 'necessary', 'possible', 'impossible', 'certain', 'uncertain',
    'clear', 'unclear', 'obvious', 'indeed', 'however', 'therefore', 'thus',
    'given', 'giving', 'gave', 'give', 'take', 'takes', 'taking', 'took',
    'make', 'makes', 'making', 'made', 'done', 'doing', 'did', 'does',
    'put', 'puts', 'putting', 'bring', 'brings', 'bringing', 'brought',
    'come', 'comes', 'coming', 'came', 'gone', 'going', 'went', 'goes'
}

# Combine all stop words
PARLIAMENTARY_STOP_WORDS = TITLES | PROCEDURAL | COMMON_PARLIAMENTARY

# Function to get all stop words
def get_parliamentary_stop_words():
    """Return the complete set of parliamentary stop words."""
    return PARLIAMENTARY_STOP_WORDS

# Function to get phrase components for bigram filtering
def get_phrase_components():
    """Return components that form common parliamentary phrases."""
    return PHRASE_COMPONENTS

# Function to filter bigrams
def is_procedural_bigram(bigram):
    """
    Check if a bigram is procedural parliamentary language.
    
    Args:
        bigram: Tuple of two words
        
    Returns:
        Boolean indicating if bigram should be filtered
    """
    # Check if it's a known phrase
    bigram_str = ' '.join(bigram).lower()
    if bigram_str in PHRASE_COMPONENTS:
        return True
    
    # Check if both words are stop words
    word1, word2 = bigram[0].lower(), bigram[1].lower()
    if word1 in PARLIAMENTARY_STOP_WORDS and word2 in PARLIAMENTARY_STOP_WORDS:
        return True
    
    # Check specific patterns
    if word1 in TITLES or word2 in TITLES:
        return True
        
    return False

# Function to clean text
def clean_parliamentary_text(text, remove_stop_words=True, preserve_important_context=False):
    """
    Clean parliamentary text by removing procedural language.
    
    Args:
        text: Input text string
        remove_stop_words: Whether to remove stop words
        preserve_important_context: Keep some procedural terms for context
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    words = text.lower().split()
    
    if not remove_stop_words:
        return ' '.join(words)
    
    if preserve_important_context:
        # Keep some terms that might be important for certain analyses
        important_terms = {'government', 'opposition', 'policy', 'law', 'reform',
                          'war', 'peace', 'treaty', 'economy', 'budget', 'tax'}
        filtered_words = [w for w in words 
                          if w not in PARLIAMENTARY_STOP_WORDS or w in important_terms]
    else:
        filtered_words = [w for w in words if w not in PARLIAMENTARY_STOP_WORDS]
    
    return ' '.join(filtered_words)