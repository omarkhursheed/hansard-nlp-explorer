"""
Enhanced Parliamentary Stop Words for Hansard Analysis

Combines standard English stop words with parliamentary-specific terminology
for more effective filtering of procedural language.
"""

import string
from pathlib import Path

def get_english_stop_words():
    """Get standard English stop words from NLTK or use a comprehensive built-in list"""
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except:
        # Fallback to comprehensive built-in list if NLTK not available
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 
            'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
            'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 
            're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
            "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
            "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 
            'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
            'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
        }

# Parliamentary-specific titles and honorifics
PARLIAMENTARY_TITLES = {
    'hon', 'honourable', 'right', 'noble', 'lord', 'lords', 'lady', 'ladies',
    'sir', 'mr', 'mrs', 'ms', 'miss', 'dr', 'dame', 'baron', 'baroness',
    'viscount', 'viscountess', 'earl', 'countess', 'marquis', 'marquess',
    'duke', 'duchess', 'prince', 'princess', 'king', 'queen', 'majesty',
    'excellency', 'reverend', 'reverent', 'bishop', 'archbishop',
    'gentleman', 'gentlemen', 'member', 'members', 'friend', 'friends',
    'colleague', 'colleagues', 'secretary', 'minister', 'ministers',
    'prime', 'deputy', 'chancellor', 'speaker', 'chairman', 'chairwoman',
    'lordship', 'lordships', 'grace', 'graces', 'worship', 'esquire', 'esq'
}

# Parliamentary procedural terms
PARLIAMENTARY_PROCEDURAL = {
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
    'ayes', 'noes', 'aye', 'agreed', 'disagreed', 'negatived',
    'carried', 'withdrawn', 'withdraw', 'withdrawing', 'withdrawal',
    'proceed', 'proceedings', 'proceeding', 'business', 'matter', 'matters',
    'notice', 'notices', 'notified', 'notification', 'inform', 'informed',
    'wish', 'wishes', 'wished', 'wishing', 'hope', 'hopes', 'hoped',
    'think', 'thinks', 'thought', 'thinking', 'believe', 'believes', 'believed',
    'consider', 'considers', 'considered', 'considering', 'consideration',
    'understand', 'understands', 'understood', 'understanding',
    'aware', 'unaware', 'know', 'knows', 'knew', 'knowing', 'knowledge',
    'suppose', 'supposes', 'supposed', 'supposing', 'presume', 'presumed',
    'submit', 'submits', 'submitted', 'submitting', 'submission',
    'propose', 'proposes', 'proposed', 'proposing', 'proposal',
    'suggest', 'suggests', 'suggested', 'suggesting', 'suggestion',
    'refer', 'refers', 'referred', 'referring', 'reference',
    'mention', 'mentions', 'mentioned', 'mentioning',
    'speak', 'speaks', 'spoke', 'spoken', 'speaking',
    'tell', 'tells', 'told', 'telling', 'address', 'addressed', 'addressing'
}

# Document references and administrative terms
DOCUMENT_REFERENCES = {
    'deb', 'vol', 'volume', 'page', 'paragraph', 'section', 'subsection',
    'column', 'columns', 'line', 'lines', 'hansard', 'record', 'records',
    'document', 'documents', 'minute', 'minutes', 'note', 'notes',
    'memorandum', 'memoranda', 'correspondence', 'letter', 'letters',
    'communication', 'communications', 'dispatch', 'dispatches'
}

# Time references (often not meaningful in analysis)
TIME_REFERENCES = {
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr',
    'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
    'today', 'tomorrow', 'yesterday', 'morning', 'afternoon', 'evening', 'night',
    'oclock', "o'clock", 'clock', 'time', 'times', 'hour', 'hours', 'minute',
    'minutes', 'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
    'annual', 'annually', 'daily', 'weekly', 'monthly', 'yearly'
}

# Numbers and ordinals (often references)
NUMBERS_ORDINALS = {
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
    'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 
    'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion',
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth',
    'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth',
    'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth',
    '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th',
    '11th', '12th', '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th'
}

def get_comprehensive_stop_words():
    """
    Get comprehensive stop words list combining:
    - Standard English stop words
    - Parliamentary-specific terms
    - Document references
    - Time references
    """
    stop_words = get_english_stop_words()
    
    # Add all parliamentary-specific categories
    stop_words.update(PARLIAMENTARY_TITLES)
    stop_words.update(PARLIAMENTARY_PROCEDURAL)
    stop_words.update(DOCUMENT_REFERENCES)
    stop_words.update(TIME_REFERENCES)
    stop_words.update(NUMBERS_ORDINALS)
    
    return stop_words

def get_content_preserving_stop_words():
    """
    Get stop words that preserve important policy/content terms.
    This is less aggressive and keeps words like 'government', 'policy', etc.
    """
    stop_words = get_english_stop_words()
    
    # Add only the most procedural terms
    stop_words.update(PARLIAMENTARY_TITLES)
    stop_words.update(DOCUMENT_REFERENCES)
    stop_words.update(TIME_REFERENCES)
    
    # Add specific procedural terms but preserve content words
    procedural_only = {
        'house', 'commons', 'parliament', 'chamber', 'bench', 'benches',
        'question', 'questions', 'answer', 'answers', 'asked', 'ask',
        'said', 'say', 'saying', 'says', 'stated', 'statement',
        'debate', 'debates', 'debating', 'debated',
        'motion', 'motions', 'moved', 'move', 'moving',
        'order', 'orders', 'ordered', 'ordering', 'point', 'points',
        'hear', 'heard', 'hearing', 'rise', 'rising', 'rose',
        'sitting', 'sittings', 'session', 'sessions',
        'beg', 'begs', 'begged', 'begging', 'leave', 'permission',
        'table', 'tabled', 'tabling', 'reading', 'readings',
        'ayes', 'noes', 'aye', 'agreed', 'disagreed',
        'wish', 'wishes', 'wished', 'wishing'
    }
    stop_words.update(procedural_only)
    
    return stop_words

def filter_bigrams(bigram, stop_words=None):
    """
    Check if a bigram should be filtered out.
    
    Args:
        bigram: Tuple of two words
        stop_words: Set of stop words to use (if None, uses comprehensive list)
    
    Returns:
        Boolean indicating if bigram should be filtered
    """
    if stop_words is None:
        stop_words = get_comprehensive_stop_words()
    
    word1, word2 = bigram[0].lower(), bigram[1].lower()
    
    # Filter if both words are stop words
    if word1 in stop_words and word2 in stop_words:
        return True
    
    # Filter specific parliamentary phrases
    bigram_str = f"{word1} {word2}"
    parliamentary_phrases = {
        'right hon', 'hon gentleman', 'hon friend', 'hon member', 'noble lord',
        'hon lady', 'noble friend', 'learned friend', 'gallant gentleman',
        'right reverend', 'noble earl', 'noble viscount', 'noble marquess',
        'secretary state', 'prime minister', 'foreign secretary', 'home secretary',
        'chancellor exchequer', 'attorney general', 'solicitor general',
        'front bench', 'back bench', 'opposite side', 'dispatch box',
        'order paper', 'standing order', 'point order', 'hear hear',
        'question time', 'supplementary question', 'oral question', 'written question',
        'first reading', 'second reading', 'third reading', 'committee stage',
        'report stage', 'royal assent', 'money bill', 'private bill', 'public bill'
    }
    
    if bigram_str in parliamentary_phrases:
        return True
    
    return False

def clean_text_aggressive(text):
    """
    Aggressively clean text removing all parliamentary language.
    Best for topic modeling and content analysis.
    """
    if not text:
        return ""
    
    stop_words = get_comprehensive_stop_words()
    words = text.lower().split()
    
    # Filter words
    filtered = [w for w in words if w not in stop_words and len(w) > 2]
    
    return ' '.join(filtered)

def clean_text_moderate(text):
    """
    Moderately clean text preserving some important policy terms.
    Good for maintaining context while removing procedural language.
    """
    if not text:
        return ""
    
    stop_words = get_content_preserving_stop_words()
    words = text.lower().split()
    
    # Filter words but keep important terms
    important_terms = {
        'government', 'opposition', 'policy', 'law', 'reform', 'legislation',
        'economy', 'budget', 'tax', 'taxation', 'defence', 'defense',
        'education', 'health', 'welfare', 'social', 'public', 'private',
        'war', 'peace', 'treaty', 'foreign', 'domestic', 'international',
        'trade', 'industry', 'agriculture', 'employment', 'unemployment',
        'housing', 'transport', 'infrastructure', 'environment',
        'justice', 'crime', 'police', 'prison', 'court', 'legal',
        'constitutional', 'democracy', 'election', 'referendum',
        'scotland', 'wales', 'ireland', 'northern', 'england', 'britain', 'british',
        'europe', 'european', 'commonwealth', 'empire', 'colonial'
    }
    
    filtered = [w for w in words if (w not in stop_words or w in important_terms) and len(w) > 2]
    
    return ' '.join(filtered)