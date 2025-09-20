#!/usr/bin/env python3
"""
Comprehensive Hansard Corpus Analysis with Multi-Level Filtering

Creates clean visualizations and analysis across different filtering levels to reveal
both procedural and substantive parliamentary content.

Usage:
    python comprehensive_corpus_analysis.py --years 1920-1930 --sample 1000
    python comprehensive_corpus_analysis.py --full --sample 10000
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveCorpusAnalyzer:
    def __init__(self, data_dir="data/processed_fixed", output_dir="analysis/corpus_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load gender wordlists
        self.male_words = self._load_gender_wordlist("data/gender_wordlists/male_words.txt")
        self.female_words = self._load_gender_wordlist("data/gender_wordlists/female_words.txt")
        
        # Define filtering levels
        self.filtering_levels = {
            0: {"name": "NONE", "stop_words": set(), "min_len": 2},
            1: {"name": "BASIC", "stop_words": self._get_basic_stop_words(), "min_len": 3},
            2: {"name": "PARLIAMENTARY", "stop_words": self._get_parliamentary_stop_words(), "min_len": 3},
            3: {"name": "MODERATE", "stop_words": self._get_moderate_stop_words(), "min_len": 3},
            4: {"name": "AGGRESSIVE", "stop_words": self._get_aggressive_stop_words(), "min_len": 3},
            5: {"name": "TFIDF", "stop_words": self._get_basic_stop_words(), "min_len": 3, "use_tfidf": True}
        }
        
    def _load_gender_wordlist(self, filepath):
        """Load gender wordlist and convert to lowercase set"""
        try:
            with open(filepath, 'r') as f:
                return set(word.strip().lower() for word in f if word.strip())
        except FileNotFoundError:
            print(f"Warning: Gender wordlist not found at {filepath}")
            return set()
    
    def _get_basic_stop_words(self):
        """Basic English stop words including been, prepositions, and common variants"""
        return set(['the', 'of', 'to', 'and', 'a', 'an', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 
                   'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 
                   'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when',
                   'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if',
                   'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
                   'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very', 'what',
                   'know', 'just', 'first', 'get', 'over', 'think', 'also', 'its', 'our', 'work', 'life',
                   'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here',
                   'through', 'when', 'where', 'much', 'go', 'me', 'back', 'with', 'well', 'were',
                   # Add been and variants
                   'been', 'being', 'am', 'are', 'was', 'were', 'be', 'had', 'has', 'have', 'having',
                   # Add prepositions and common words
                   'upon', 'under', 'above', 'below', 'between', 'among', 'during', 'within', 'without',
                   'against', 'towards', 'across', 'around', 'beside', 'beyond', 'inside', 'outside',
                   'near', 'far', 'off', 'onto', 'into', 'unto', 'until', 'since', 'while', 'whilst',
                   # Common variants that appear frequently
                   'any', 'who', 'those', 'than', 'such', 'whether'])
    
    def _get_parliamentary_stop_words(self):
        """Parliamentary procedural terms"""
        basic = self._get_basic_stop_words()
        parliamentary = {'hon', 'right', 'gentleman', 'member', 'members', 'house', 'speaker', 'sir',
                        'lord', 'lords', 'gallant', 'learned', 'friend', 'friends', 'noble', 'bill',
                        'clause', 'amendment', 'committee', 'order', 'question', 'division', 'read',
                        'reading', 'report', 'stage', 'passed', 'carried', 'agreed', 'moved', 'second',
                        'third', 'standing', 'select', 'chair', 'chairman', 'motion', 'debate', 'discuss'}
        return basic | parliamentary
    
    def _get_moderate_stop_words(self):
        """Moderate filtering including common procedural terms and more variants"""
        parliamentary = self._get_parliamentary_stop_words()
        moderate = {'government', 'minister', 'secretary', 'state', 'department', 'office', 'time',
                   'year', 'years', 'day', 'way', 'case', 'point', 'matter', 'place', 'hand', 'part',
                   'number', 'great', 'public', 'present', 'general', 'particular', 'whole', 'certain',
                   'important', 'necessary', 'possible', 'clear', 'able', 'shall', 'must', 'may',
                   'might', 'could', 'should', 'would', 'will', 'going', 'come', 'came', 'take', 'taken',
                   'give', 'given', 'put', 'say', 'said', 'see', 'seen', 'come', 'came', 'went', 'gone',
                   # Additional common words that don't add substantive meaning
                   'words', 'does', 'did', 'done', 'doing', 'make', 'makes', 'making', 'made',
                   'get', 'gets', 'got', 'getting', 'find', 'found', 'look', 'looks', 'looked',
                   'want', 'wants', 'wanted', 'need', 'needs', 'needed', 'use', 'used', 'using'}
        return parliamentary | moderate
    
    def _get_aggressive_stop_words(self):
        """Aggressive filtering to focus on substantive content"""
        moderate = self._get_moderate_stop_words()
        aggressive = {'ask', 'asked', 'whether', 'does', 'think', 'believe', 'hope', 'wish', 'want',
                     'quite', 'rather', 'really', 'indeed', 'course', 'fact', 'certainly', 'surely',
                     'perhaps', 'probably', 'therefore', 'however', 'moreover', 'nevertheless',
                     'although', 'though', 'since', 'because', 'reason', 'reasons', 'answer',
                     'question', 'questions', 'information', 'statement', 'statements', 'reply',
                     'attention', 'consideration', 'regard', 'respect', 'view', 'views', 'opinion',
                     'opinions', 'position', 'situation', 'circumstances', 'conditions', 'basis',
                     'ground', 'grounds', 'purpose', 'purposes', 'object', 'objects', 'means',
                     'method', 'methods', 'system', 'systems', 'policy', 'policies', 'principle',
                     'principles', 'rules', 'rule', 'law', 'laws', 'act', 'acts', 'provision',
                     'provisions', 'section', 'sections', 'subsection'}
        return moderate | aggressive
    
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
                                'speakers': debate.get('metadata', {}).get('speakers', []),
                                'chamber': debate.get('metadata', {}).get('chamber', ''),
                                'word_count': debate.get('metadata', {}).get('word_count', 0),
                            }
                            if debate_data['text']:  # Only include non-empty texts
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
            random.seed(42)  # For reproducibility
            # Stratified sampling to maintain year distribution
            debates_by_year = {}
            for debate in debates:
                year = debate['year']
                if year not in debates_by_year:
                    debates_by_year[year] = []
                debates_by_year[year].append(debate)
            
            # Sample proportionally from each year
            sampled_debates = []
            total_debates = len(debates)
            for year, year_debates in debates_by_year.items():
                year_sample_size = max(1, int(len(year_debates) / total_debates * sample_size))
                sampled_debates.extend(random.sample(year_debates, min(year_sample_size, len(year_debates))))
            
            debates = sampled_debates[:sample_size]
            print(f"Sampled {len(debates)} debates from total corpus")
        
        print(f"\nTotal debates loaded: {len(debates)} from {len(years_processed)} years")
        return debates, years_processed
    
    def filter_text(self, text, level=1):
        """Filter text according to specified level"""
        filter_config = self.filtering_levels[level]
        
        # Tokenize
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Apply filtering
        filtered_words = []
        for word in words:
            if (len(word) >= filter_config['min_len'] and 
                word not in filter_config['stop_words']):
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def analyze_filtering_levels(self, debates):
        """Analyze all filtering levels and compare results"""
        print("\nAnalyzing all filtering levels...")
        
        all_results = {}
        
        for level in range(6):  # 0-5 filtering levels
            print(f"\n--- LEVEL {level}: {self.filtering_levels[level]['name']} ---")
            
            # Filter all texts
            if level == 5:  # TFIDF special case
                filtered_texts = [self.filter_text(d['text'], level=1) for d in debates]  # Use basic filtering first
                results = self._analyze_with_tfidf(debates, filtered_texts, level)
            else:
                filtered_texts = [self.filter_text(d['text'], level) for d in debates]
                results = self._analyze_filtered_texts(debates, filtered_texts, level)
            
            all_results[level] = results
            
            # Create individual level visualizations
            self._create_level_visualizations(results, level)
        
        # Create comparison visualizations
        self._create_comparison_visualizations(all_results)
        
        # Save comprehensive results
        output_path = self.output_dir / 'comprehensive_analysis_results.json'
        with open(output_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self._prepare_for_json(all_results)
            json.dump(json_results, f, indent=2)
        print(f"\nComprehensive results saved to {output_path}")
        
        return all_results
    
    def _analyze_filtered_texts(self, debates, filtered_texts, level):
        """Analyze filtered texts for a specific level"""
        # Combine all filtered text
        all_text = ' '.join(filtered_texts)
        words = all_text.split()
        
        # Count words
        word_freq = Counter(words)
        original_word_count = sum(len(d['text'].split()) for d in debates)
        filtered_word_count = len(words)
        
        results = {
            'filter_level': level,
            'filter_name': self.filtering_levels[level]['name'],
            'years_processed': sorted(list(set(d['year'] for d in debates))),
            'total_debates': len(debates),
            'filtering_stats': {
                'original_words': original_word_count,
                'filtered_words': filtered_word_count,
                'reduction_pct': ((original_word_count - filtered_word_count) / original_word_count * 100) if original_word_count > 0 else 0
            },
            'top_unigrams': word_freq.most_common(30)
        }
        
        # Bigram analysis
        bigrams = []
        for text in filtered_texts:
            text_words = text.split()
            bigrams.extend([(text_words[i], text_words[i+1]) for i in range(len(text_words)-1)])
        
        bigram_freq = Counter(bigrams)
        results['top_bigrams'] = bigram_freq.most_common(20)
        
        # Topic modeling if enough content
        if len(filtered_texts) > 100 and filtered_word_count > 1000:
            results['topics'] = self._perform_topic_modeling(filtered_texts)
        
        # Gender analysis (always on original text to preserve accuracy)
        results['gender_analysis'] = self._analyze_gender_language(debates)
        
        # Speaker gender analysis
        results['speaker_analysis'] = self._analyze_speaker_gender(debates)
        
        return results
    
    def _analyze_with_tfidf(self, debates, filtered_texts, level):
        """Special analysis using TF-IDF for content extraction"""
        # Use TF-IDF to identify most important terms
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=3,
            max_df=0.7,
            ngram_range=(1, 2)
        )
        
        # Filter out very short texts
        valid_texts = [t for t in filtered_texts if len(t.split()) > 10]
        
        if len(valid_texts) < 50:
            print("Not enough valid text for TF-IDF analysis")
            return self._analyze_filtered_texts(debates, filtered_texts, 1)  # Fallback to basic
        
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get mean TF-IDF scores
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        word_scores = list(zip(feature_names, mean_scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create filtered vocabulary based on TF-IDF scores
        top_tfidf_words = set([word for word, score in word_scores[:500]])
        
        # Re-filter texts using TF-IDF vocabulary
        tfidf_filtered_texts = []
        for text in filtered_texts:
            words = text.split()
            tfidf_words = [w for w in words if w in top_tfidf_words]
            tfidf_filtered_texts.append(' '.join(tfidf_words))
        
        # Analyze the TF-IDF filtered results
        return self._analyze_filtered_texts(debates, tfidf_filtered_texts, level)
    
    def _perform_topic_modeling(self, texts, n_topics=10):
        """Perform LDA topic modeling"""
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # Filter valid texts
        valid_texts = [t for t in texts if len(t.split()) > 5]
        
        if len(valid_texts) < 100:
            return []
        
        try:
            doc_term_matrix = vectorizer.fit_transform(valid_texts)
            
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10,
                learning_method='online'
            )
            lda.fit(doc_term_matrix)
            
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topic_weights = topic[top_indices].tolist()
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': topic_weights
                })
            
            return topics
        except Exception as e:
            print(f"Topic modeling failed: {e}")
            return []
    
    def _analyze_gender_language(self, debates):
        """Analyze gender-related language patterns"""
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
        
        return {
            'male_word_count': male_count,
            'female_word_count': female_count,
            'male_ratio': male_ratio,
            'female_ratio': female_ratio,
            'top_male_words': male_word_freq.most_common(10),
            'top_female_words': female_word_freq.most_common(10)
        }
    
    def _analyze_speaker_gender(self, debates):
        """Analyze speaker gender distribution using title-based heuristics"""
        male_speakers = set()
        female_speakers = set()
        all_speakers = set()
        
        male_titles = {'mr', 'sir', 'lord', 'earl', 'duke', 'baron', 'count', 'marquess', 'prince', 'viscount'}
        female_titles = {'mrs', 'miss', 'lady', 'dame', 'madam', 'viscountess', 'countess', 'baroness', 'duchess', 'marchioness', 'princess'}
        
        for debate in debates:
            for speaker in debate.get('speakers', []):
                speaker_lower = speaker.lower()
                all_speakers.add(speaker)
                
                # Check for gendered titles
                if any(title in speaker_lower for title in male_titles):
                    male_speakers.add(speaker)
                elif any(title in speaker_lower for title in female_titles):
                    female_speakers.add(speaker)
        
        total_identified = len(male_speakers) + len(female_speakers)
        
        return {
            'total_unique_speakers': len(all_speakers),
            'identified_speakers': total_identified,
            'male_speakers': len(male_speakers),
            'female_speakers': len(female_speakers),
            'male_percentage': (len(male_speakers) / total_identified * 100) if total_identified > 0 else 0,
            'female_percentage': (len(female_speakers) / total_identified * 100) if total_identified > 0 else 0,
            'unknown_percentage': ((len(all_speakers) - total_identified) / len(all_speakers) * 100) if len(all_speakers) > 0 else 0
        }
    
    def _create_level_visualizations(self, results, level):
        """Create visualizations for a specific filtering level"""
        level_dir = self.output_dir / f"level_{level}_plots"
        level_dir.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'{results["filter_name"]} Filtering (Level {level}) - {results["filtering_stats"]["reduction_pct"]:.1f}% Word Reduction', 
                    fontsize=16, fontweight='bold')
        
        # 1. Top Unigrams
        ax1 = plt.subplot(3, 3, 1)
        if results['top_unigrams']:
            words, counts = zip(*results['top_unigrams'][:25])
            colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
            bars = ax1.barh(range(len(words)), counts, color=colors)
            ax1.set_yticks(range(len(words)))
            ax1.set_yticklabels(words)
            ax1.set_xlabel('Frequency')
            ax1.set_title(f'Top 25 Words - {results["filter_name"]} Filtering (Level {level})')
            ax1.invert_yaxis()
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax1.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{count:,}', ha='left', va='center', fontsize=8)
        
        # 2. Top Bigrams
        ax2 = plt.subplot(3, 3, 2)
        if results['top_bigrams']:
            bigrams, counts = zip(*results['top_bigrams'][:15])
            bigram_labels = [' '.join(b) for b in bigrams]
            colors = plt.cm.plasma(np.linspace(0, 1, len(bigram_labels)))
            bars = ax2.barh(range(len(bigram_labels)), counts, color=colors)
            ax2.set_yticks(range(len(bigram_labels)))
            ax2.set_yticklabels(bigram_labels)
            ax2.set_xlabel('Frequency')
            ax2.set_title('Top 15 Bigrams')
            ax2.invert_yaxis()
        
        # 3. Gender Word Distribution
        ax3 = plt.subplot(3, 3, 3)
        if 'gender_analysis' in results:
            gender_data = results['gender_analysis']
            labels = ['Male Words', 'Female Words']
            sizes = [gender_data['male_word_count'], gender_data['female_word_count']]
            colors = ['#2E86AB', '#A23B72']
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Gender Word Distribution')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 4. Topic Modeling Overview (improved visualization)
        ax4 = plt.subplot(3, 3, 4)
        if 'topics' in results and results['topics']:
            # Create a heatmap of top words across topics
            topics = results['topics'][:8]  # Top 8 topics
            all_words = []
            topic_labels = []
            
            for i, topic in enumerate(topics):
                topic_labels.append(f"Topic {i+1}")
                all_words.extend(topic['words'][:5])  # Top 5 words per topic
            
            # Get unique words and create matrix
            unique_words = list(set(all_words))[:20]  # Limit to top 20 unique words
            topic_word_matrix = []
            
            for topic in topics:
                row = []
                topic_words = dict(zip(topic['words'], topic['weights']))
                for word in unique_words:
                    row.append(topic_words.get(word, 0))
                topic_word_matrix.append(row)
            
            if topic_word_matrix:
                im = ax4.imshow(topic_word_matrix, aspect='auto', cmap='YlOrRd')
                ax4.set_xticks(range(len(unique_words)))
                ax4.set_xticklabels(unique_words, rotation=45, ha='right')
                ax4.set_yticks(range(len(topic_labels)))
                ax4.set_yticklabels(topic_labels)
                ax4.set_title('Topic-Word Importance Heatmap')
                plt.colorbar(im, ax=ax4)
        else:
            ax4.text(0.5, 0.5, 'Topic modeling\nnot available\n(insufficient data)', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Topic Modeling Overview')
            ax4.axis('off')
        
        # 5. Speaker Gender Distribution
        ax5 = plt.subplot(3, 3, 5)
        if 'speaker_analysis' in results:
            speaker_data = results['speaker_analysis']
            labels = ['Male Speakers', 'Female Speakers', 'Unknown']
            sizes = [speaker_data['male_percentage'], speaker_data['female_percentage'], speaker_data['unknown_percentage']]
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            
            wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax5.set_title(f'Speaker Gender Distribution\n({speaker_data["total_unique_speakers"]:,} unique speakers)')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 6. Filtering Impact
        ax6 = plt.subplot(3, 3, 6)
        filtering_stats = results['filtering_stats']
        categories = ['Original', 'Filtered']
        values = [filtering_stats['original_words'], filtering_stats['filtered_words']]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax6.bar(categories, values, color=colors)
        ax6.set_ylabel('Word Count')
        ax6.set_title(f'Filtering Impact\n{filtering_stats["reduction_pct"]:.1f}% Reduction')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        # 7. Wordcloud (smaller)
        ax7 = plt.subplot(3, 3, 7)
        if results['top_unigrams']:
            word_freq = dict(results['top_unigrams'][:100])
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                colormap='viridis',
                max_words=50
            ).generate_from_frequencies(word_freq)
            
            ax7.imshow(wordcloud, interpolation='bilinear')
            ax7.axis('off')
            ax7.set_title('Word Cloud')
        
        # 8. Year Distribution (if multiple years)
        ax8 = plt.subplot(3, 3, 8)
        if len(results['years_processed']) > 1:
            year_counts = Counter()
            # This would need debate-year mapping, simplified for now
            ax8.text(0.5, 0.5, f'Years Analyzed:\n{results["years_processed"][0]}-{results["years_processed"][-1]}\n({len(results["years_processed"])} years)', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=12)
            ax8.set_title('Temporal Coverage')
            ax8.axis('off')
        else:
            ax8.text(0.5, 0.5, f'Single Year:\n{results["years_processed"][0]}', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=12)
            ax8.set_title('Temporal Coverage')
            ax8.axis('off')
        
        # 9. Statistics Summary
        ax9 = plt.subplot(3, 3, 9)
        stats_text = f"""
Debates: {results['total_debates']:,}
Years: {len(results['years_processed'])}
Original Words: {filtering_stats['original_words']:,}
Filtered Words: {filtering_stats['filtered_words']:,}
Reduction: {filtering_stats['reduction_pct']:.1f}%

Gender Analysis:
Male: {results['gender_analysis']['male_ratio']:.1%}
Female: {results['gender_analysis']['female_ratio']:.1%}
"""
        ax9.text(0.1, 0.5, stats_text.strip(), ha='left', va='center', 
                transform=ax9.transAxes, fontsize=10, fontfamily='monospace')
        ax9.set_title('Analysis Statistics')
        ax9.axis('off')
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        output_path = level_dir / f'comprehensive_analysis_level_{level}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Level {level} visualization saved to {output_path}")
        plt.close()
    
    def _create_comparison_visualizations(self, all_results):
        """Create comparison visualizations across filtering levels"""
        # Filter comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Filtering Levels Comparison', fontsize=16, fontweight='bold')
        
        # 1. Word Reduction Comparison
        levels = list(range(6))
        level_names = [all_results[l]['filter_name'] for l in levels]
        reductions = [all_results[l]['filtering_stats']['reduction_pct'] for l in levels]
        
        bars = axes[0, 0].bar(level_names, reductions, 
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
        axes[0, 0].set_ylabel('Word Reduction (%)')
        axes[0, 0].set_title('Filtering Effectiveness')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, reduction in zip(bars, reductions):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{reduction:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Top Words Evolution (show how top word changes with filtering)
        top_words_by_level = []
        top_counts_by_level = []
        for level in levels:
            if all_results[level]['top_unigrams']:
                top_word, count = all_results[level]['top_unigrams'][0]
                top_words_by_level.append(f"{top_word}\n({count:,})")
                top_counts_by_level.append(count)
        
        if top_words_by_level:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            bars = axes[0, 1].bar(range(len(top_words_by_level)), top_counts_by_level, color=colors)
            axes[0, 1].set_xticks(range(len(top_words_by_level)))
            axes[0, 1].set_xticklabels([all_results[l]['filter_name'] for l in range(len(top_words_by_level))], rotation=45)
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Top Word by Filtering Level')
        
        # 3. Gender Ratio Consistency
        male_ratios = [all_results[l]['gender_analysis']['male_ratio'] * 100 for l in levels]
        female_ratios = [all_results[l]['gender_analysis']['female_ratio'] * 100 for l in levels]
        
        x = np.arange(len(level_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, male_ratios, width, label='Male Words', color='#2E86AB', alpha=0.8)
        axes[1, 0].bar(x + width/2, female_ratios, width, label='Female Words', color='#A23B72', alpha=0.8)
        
        axes[1, 0].set_xlabel('Filtering Level')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].set_title('Gender Language Ratio Consistency')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(level_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 100)
        
        # 4. Vocabulary Size
        vocab_sizes = [len(all_results[l]['top_unigrams']) if all_results[l]['top_unigrams'] else 0 for l in levels]
        unique_vocab = [all_results[l]['filtering_stats']['filtered_words'] for l in levels]
        
        ax_twin = axes[1, 1].twinx()
        
        line1 = axes[1, 1].plot(level_names, vocab_sizes, 'o-', color='#FF6B6B', linewidth=2, markersize=8, label='Unique Words (Top)')
        line2 = ax_twin.plot(level_names, unique_vocab, 's--', color='#4ECDC4', linewidth=2, markersize=8, label='Total Filtered Words')
        
        axes[1, 1].set_xlabel('Filtering Level')
        axes[1, 1].set_ylabel('Unique Words (Top)', color='#FF6B6B')
        ax_twin.set_ylabel('Total Filtered Words', color='#4ECDC4')
        axes[1, 1].set_title('Vocabulary Evolution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Combine legends
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = self.output_dir / 'filtering_levels_comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Comparison visualization saved to {comparison_path}")
        plt.close()
    
    def _prepare_for_json(self, data):
        """Prepare data for JSON serialization by converting numpy types"""
        if isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return list(data)
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Hansard Corpus Analysis')
    parser.add_argument('--years', type=str, help='Year range (e.g., 1920-1930)')
    parser.add_argument('--sample', type=int, help='Sample size for analysis')
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
        # Default to a focused period for testing
        start_year = 1920
        end_year = 1935
    
    # Create analyzer and run
    analyzer = ComprehensiveCorpusAnalyzer()
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE HANSARD CORPUS ANALYSIS")
    print(f"Years: {start_year}-{end_year}")
    if args.sample:
        print(f"Sample size: {args.sample}")
    print(f"{'='*80}")
    
    # Load debates
    debates, years_processed = analyzer.load_debates(start_year, end_year, args.sample)
    
    if not debates:
        print("No debates loaded. Exiting.")
        return
    
    # Run comprehensive analysis
    results = analyzer.analyze_filtering_levels(debates)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"Results saved in: {analyzer.output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()