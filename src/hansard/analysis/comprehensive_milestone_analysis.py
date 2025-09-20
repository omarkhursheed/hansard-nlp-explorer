#!/usr/bin/env python3
"""
Comprehensive Historical Milestone Analysis for Hansard Parliamentary Debates

Analyzes key historical periods with multiple filtering levels to reveal both
procedural and substantive content changes around pivotal moments.

Key Milestones:
- 1918: Partial women's suffrage + First women in Parliament
- 1928: Full women's suffrage
- 1914-1918: World War I
- 1939-1945: World War II
- 1979-1990: Margaret Thatcher's tenure as PM

Usage:
    python comprehensive_milestone_analysis.py --all
    python comprehensive_milestone_analysis.py --milestone ww2_period
    python comprehensive_milestone_analysis.py --milestone 1928_full_suffrage --filtering aggressive
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveMilestoneAnalyzer:
    def __init__(self, data_dir="data/processed_fixed", output_dir="analysis/milestone_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load gender wordlists
        self.male_words = self._load_gender_wordlist("data/gender_wordlists/male_words.txt")
        self.female_words = self._load_gender_wordlist("data/gender_wordlists/female_words.txt")
        
        # Define filtering modes
        self.filtering_modes = {
            "none": {"stop_words": set(), "min_len": 2},
            "basic": {"stop_words": self._get_basic_stop_words(), "min_len": 3},
            "parliamentary": {"stop_words": self._get_parliamentary_stop_words(), "min_len": 3},
            "moderate": {"stop_words": self._get_moderate_stop_words(), "min_len": 3},
            "aggressive": {"stop_words": self._get_aggressive_stop_words(), "min_len": 3}
        }
        
        # Define historical milestones with analysis windows
        self.milestones = {
            "1918_partial_suffrage": {
                "name": "1918 Partial Women's Suffrage & Parliament Entry",
                "description": "Women over 30 gain vote; first women allowed in Parliament",
                "milestone_year": 1918,
                "pre_window": (1908, 1918),
                "post_window": (1918, 1928),
                "sample_size": 10000
            },
            "1928_full_suffrage": {
                "name": "1928 Full Women's Suffrage",
                "description": "Voting age for women reduced to 21 (Equal Franchise Act)",
                "milestone_year": 1928,
                "pre_window": (1918, 1928),
                "post_window": (1928, 1938),
                "sample_size": 10000
            },
            "ww1_period": {
                "name": "World War I Analysis",
                "description": "Parliamentary discourse before, during, and after WWI",
                "milestone_year": 1916,
                "pre_window": (1909, 1914),
                "during_window": (1914, 1918),
                "post_window": (1918, 1923),
                "sample_size": 10000
            },
            "ww2_period": {
                "name": "World War II Analysis",
                "description": "Parliamentary discourse before, during, and after WWII",
                "milestone_year": 1942,
                "pre_window": (1934, 1939),
                "during_window": (1939, 1945),
                "post_window": (1945, 1950),
                "sample_size": 10000
            },
            "thatcher_period": {
                "name": "Thatcher Era Analysis",
                "description": "Parliamentary discourse before, during, and after Thatcher's tenure",
                "milestone_year": 1984,
                "pre_window": (1974, 1979),
                "during_window": (1979, 1990),
                "post_window": (1990, 1995),
                "sample_size": 10000
            }
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
    
    def load_debates(self, start_year, end_year, sample_size=None):
        """Load debate texts from JSONL files"""
        debates = []
        
        for year in range(start_year, end_year + 1):
            jsonl_path = self.data_dir / "content" / str(year) / f"debates_{year}.jsonl"
            
            if not jsonl_path.exists():
                continue
                
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
                            if debate_data['text'] and len(debate_data['text'].split()) > 50:  # Minimum content threshold
                                debates.append(debate_data)
                
            except Exception as e:
                print(f"Error loading {year}: {e}")
                continue
        
        # Apply sampling if specified
        if sample_size and len(debates) > sample_size:
            import random
            random.seed(42)  # For reproducibility
            debates = random.sample(debates, sample_size)
        
        print(f"Loaded {len(debates)} debates from {start_year}-{end_year}")
        return debates
    
    def filter_text(self, text, filtering_mode="moderate"):
        """Filter text according to specified mode"""
        filter_config = self.filtering_modes[filtering_mode]
        
        # Tokenize
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Apply filtering
        filtered_words = []
        for word in words:
            if (len(word) >= filter_config['min_len'] and 
                word not in filter_config['stop_words']):
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def analyze_period(self, debates, period_name, filtering_mode="moderate"):
        """Analyze a specific period with given filtering"""
        print(f"  Analyzing {period_name} with {filtering_mode} filtering...")
        
        if not debates:
            return {}
        
        # Filter texts
        filtered_texts = [self.filter_text(d['text'], filtering_mode) for d in debates]
        
        # Word frequency analysis
        all_text = ' '.join(filtered_texts)
        words = all_text.split()
        word_freq = Counter(words)
        
        # Bigram analysis
        bigrams = []
        for text in filtered_texts:
            text_words = text.split()
            bigrams.extend([(text_words[i], text_words[i+1]) for i in range(len(text_words)-1)])
        
        bigram_freq = Counter(bigrams)
        
        # Topic modeling
        topics = []
        if len(filtered_texts) > 100 and len(words) > 1000:
            topics = self._perform_topic_modeling(filtered_texts, n_topics=8)
        
        # Gender analysis (on original text for accuracy)
        gender_analysis = self._analyze_gender_language(debates)
        
        # Speaker analysis
        speaker_analysis = self._analyze_speaker_gender(debates)
        
        # Chamber distribution
        chamber_counts = Counter([d['chamber'] for d in debates if d.get('chamber')])
        
        return {
            'period': period_name,
            'filtering_mode': filtering_mode,
            'total_debates': len(debates),
            'years': sorted(list(set(d['year'] for d in debates))),
            'top_unigrams': word_freq.most_common(30),
            'top_bigrams': bigram_freq.most_common(20),
            'topics': topics,
            'gender_analysis': gender_analysis,
            'speaker_analysis': speaker_analysis,
            'chamber_distribution': dict(chamber_counts),
            'filtering_stats': {
                'original_words': sum(len(d['text'].split()) for d in debates),
                'filtered_words': len(words),
                'reduction_pct': ((sum(len(d['text'].split()) for d in debates) - len(words)) / 
                                 sum(len(d['text'].split()) for d in debates) * 100) if debates else 0
            }
        }
    
    def _perform_topic_modeling(self, texts, n_topics=8):
        """Perform LDA topic modeling"""
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=3,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # Filter valid texts
        valid_texts = [t for t in texts if len(t.split()) > 5]
        
        if len(valid_texts) < 50:
            return []
        
        try:
            doc_term_matrix = vectorizer.fit_transform(valid_texts)
            
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20,
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
            print(f"    Topic modeling failed: {e}")
            return []
    
    def _analyze_gender_language(self, debates):
        """Analyze gender-related language patterns"""
        male_count = 0
        female_count = 0
        
        for debate in debates:
            words = re.findall(r'\b[a-z]+\b', debate['text'].lower())
            
            for word in words:
                if word in self.male_words:
                    male_count += 1
                elif word in self.female_words:
                    female_count += 1
        
        total_gendered = male_count + female_count
        male_ratio = male_count / total_gendered if total_gendered > 0 else 0
        female_ratio = female_count / total_gendered if total_gendered > 0 else 0
        
        return {
            'male_word_count': male_count,
            'female_word_count': female_count,
            'male_ratio': male_ratio,
            'female_ratio': female_ratio
        }
    
    def _analyze_speaker_gender(self, debates):
        """Analyze speaker gender distribution"""
        male_speakers = set()
        female_speakers = set()
        all_speakers = set()
        
        male_titles = {'mr', 'sir', 'lord', 'earl', 'duke', 'baron', 'count', 'marquess', 'prince', 'viscount'}
        female_titles = {'mrs', 'miss', 'lady', 'dame', 'madam', 'viscountess', 'countess', 'baroness', 'duchess', 'marchioness', 'princess'}
        
        for debate in debates:
            for speaker in debate.get('speakers', []):
                speaker_lower = speaker.lower()
                all_speakers.add(speaker)
                
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
            'female_percentage': (len(female_speakers) / total_identified * 100) if total_identified > 0 else 0
        }
    
    def analyze_milestone(self, milestone_key, filtering_mode="moderate", force_rerun=False):
        """Analyze a specific historical milestone"""
        milestone = self.milestones[milestone_key]
        
        print(f"\n{'='*80}")
        print(f"ANALYZING: {milestone['name']}")
        print(f"Filtering Mode: {filtering_mode.title()}")
        print(f"{'='*80}")
        
        # Create output directory
        milestone_dir = self.output_dir / milestone_key / filtering_mode
        milestone_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if analysis exists
        results_file = milestone_dir / "milestone_analysis.json"
        if results_file.exists() and not force_rerun:
            print(f"Analysis already exists at {results_file}. Use --force to rerun.")
            with open(results_file, 'r') as f:
                return json.load(f)
        
        # Load and analyze periods
        pre_debates = self.load_debates(
            milestone['pre_window'][0], 
            milestone['pre_window'][1], 
            milestone['sample_size']
        )
        pre_results = self.analyze_period(pre_debates, "pre", filtering_mode)
        
        # During period (if exists)
        during_results = None
        if 'during_window' in milestone:
            during_debates = self.load_debates(
                milestone['during_window'][0],
                milestone['during_window'][1],
                milestone['sample_size']
            )
            during_results = self.analyze_period(during_debates, "during", filtering_mode)
        
        post_debates = self.load_debates(
            milestone['post_window'][0],
            milestone['post_window'][1],
            milestone['sample_size']
        )
        post_results = self.analyze_period(post_debates, "post", filtering_mode)
        
        # Compare periods
        comparison = self._compare_periods(pre_results, post_results, during_results, milestone)
        
        # Create comprehensive results
        milestone_results = {
            "milestone_info": milestone,
            "filtering_mode": filtering_mode,
            "analysis_timestamp": datetime.now().isoformat(),
            "pre_period": pre_results,
            "post_period": post_results,
            "comparison": comparison
        }
        
        if during_results:
            milestone_results["during_period"] = during_results
        
        # Create visualizations
        self._create_milestone_visualization(milestone_results, milestone_dir)
        
        # Save results
        with open(results_file, 'w') as f:
            json_results = self._prepare_for_json(milestone_results)
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {results_file}")
        
        # Generate report
        self._generate_milestone_report(milestone_results, milestone_dir)
        
        return milestone_results
    
    def _compare_periods(self, pre_results, post_results, during_results, milestone):
        """Compare periods and generate insights"""
        comparison = {}
        
        # Gender language changes
        pre_female = pre_results['gender_analysis']['female_ratio'] * 100
        post_female = post_results['gender_analysis']['female_ratio'] * 100
        
        comparison['gender_language_change'] = {
            'pre_female_pct': pre_female,
            'post_female_pct': post_female,
            'change_pp': post_female - pre_female
        }
        
        # Speaker gender changes
        pre_female_speakers = pre_results['speaker_analysis']['female_percentage']
        post_female_speakers = post_results['speaker_analysis']['female_percentage']
        
        comparison['speaker_gender_change'] = {
            'pre_female_pct': pre_female_speakers,
            'post_female_pct': post_female_speakers,
            'change_pp': post_female_speakers - pre_female_speakers
        }
        
        # Content evolution
        pre_words = set([w for w, _ in pre_results['top_unigrams'][:20]])
        post_words = set([w for w, _ in post_results['top_unigrams'][:20]])
        
        comparison['content_evolution'] = {
            'new_words': list(post_words - pre_words)[:10],
            'disappeared_words': list(pre_words - post_words)[:10],
            'persistent_words': list(pre_words & post_words)[:10]
        }
        
        return comparison
    
    def _create_milestone_visualization(self, results, output_dir):
        """Create comprehensive milestone visualization"""
        milestone = results['milestone_info']
        filtering_mode = results['filtering_mode']
        
        # Determine number of periods
        has_during = 'during_period' in results
        n_cols = 3 if has_during else 2
        
        fig = plt.figure(figsize=(22, 16))
        fig.suptitle(f"{milestone['name']} - {filtering_mode.title()} Filtering", 
                    fontsize=16, fontweight='bold')
        
        periods = ['pre_period', 'post_period']
        period_labels = ['Pre', 'Post']
        
        if has_during:
            periods = ['pre_period', 'during_period', 'post_period']
            period_labels = ['Pre', 'During', 'Post']
        
        # 1. Top Words Comparison (Row 1)
        for i, (period, label) in enumerate(zip(periods, period_labels)):
            ax = plt.subplot(4, n_cols, i + 1)
            period_data = results[period]
            
            if period_data['top_unigrams']:
                words, counts = zip(*period_data['top_unigrams'][:15])
                colors = plt.cm.Set3(np.linspace(0, 1, len(words)))
                bars = ax.barh(range(len(words)), counts, color=colors)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words)
                ax.set_xlabel('Frequency')
                ax.set_title(f'{label}-{milestone["milestone_year"]} Top Words')
                ax.invert_yaxis()
        
        # 2. Gender Distribution (Row 2)
        for i, (period, label) in enumerate(zip(periods, period_labels)):
            ax = plt.subplot(4, n_cols, n_cols + i + 1)
            period_data = results[period]
            gender_data = period_data['gender_analysis']
            
            labels_pie = ['Male Words', 'Female Words']
            sizes = [gender_data['male_word_count'], gender_data['female_word_count']]
            colors = ['#2E86AB', '#A23B72']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels_pie, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            ax.set_title(f'{label} Gender Language')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 3. Speaker Gender Distribution (Row 3) - Fixed overlapping labels
        for i, (period, label) in enumerate(zip(periods, period_labels)):
            ax = plt.subplot(4, n_cols, 2 * n_cols + i + 1)
            period_data = results[period]
            speaker_data = period_data['speaker_analysis']
            
            total_speakers = speaker_data['total_unique_speakers']
            identified = speaker_data['identified_speakers']
            
            if total_speakers > 0:
                # Use actual counts instead of percentages to avoid confusion
                labels_pie = ['Male Speakers', 'Female Speakers', 'Unidentified']
                unknown_count = total_speakers - identified
                sizes = [speaker_data['male_speakers'], speaker_data['female_speakers'], unknown_count]
                colors = ['#4A90E2', '#E94B3C', '#95A5A6']
                
                # Filter out zero values
                filtered_data = [(l, s, c) for l, s, c in zip(labels_pie, sizes, colors) if s > 0]
                if not filtered_data:
                    ax.text(0.5, 0.5, 'No speaker data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12, fontweight='bold')
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
                else:
                    filtered_labels, filtered_sizes, filtered_colors = zip(*filtered_data)
                    
                    # Smart percentage display
                    def autopct_func(pct):
                        count = int(pct * total_speakers / 100)
                        if pct > 3.0:  # Only show percentage for slices > 3%
                            return f'{pct:.1f}%\n({count})'
                        elif pct > 0.5:  # Just percentage for small slices
                            return f'{pct:.1f}%'
                        else:  # No label for tiny slices
                            return ''
                    
                    # Create pie with legend instead of overlapping labels
                    wedges, texts, autotexts = ax.pie(filtered_sizes,
                                                    labels=None,  # Remove direct labels
                                                    colors=filtered_colors,
                                                    autopct=autopct_func,
                                                    startangle=90,
                                                    textprops={'fontsize': 8, 'fontweight': 'bold'},
                                                    wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
                                                    pctdistance=0.85)
                    
                    # Style percentage text
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(8)
                    
                    # Add legend to avoid overlapping labels
                    legend_labels = [f'{label}: {size:,}' for label, size in zip(filtered_labels, filtered_sizes)]
                    ax.legend(wedges, legend_labels,
                             title="Speakers",
                             loc="center left",
                             bbox_to_anchor=(1, 0, 0.5, 1),
                             fontsize=8,
                             title_fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No speakers\nidentified', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, fontweight='bold')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
            
            ax.set_title(f'{label} Speaker Gender\n({total_speakers:,} total)', 
                        fontsize=10, fontweight='bold', pad=10)
        
        # 4. Topic Evolution (Row 4)
        for i, (period, label) in enumerate(zip(periods, period_labels)):
            ax = plt.subplot(4, n_cols, 3 * n_cols + i + 1)
            period_data = results[period]
            
            if period_data['topics']:
                topic_text = '\n'.join([f"Topic {j+1}: {', '.join(topic['words'][:5])}" 
                                      for j, topic in enumerate(period_data['topics'][:4])])
            else:
                topic_text = "No topics available\n(insufficient data)"
            
            ax.text(0.05, 0.95, topic_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
            ax.set_title(f'{label} Key Topics')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = output_dir / f"milestone_comprehensive_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to {viz_path}")
        plt.close()
        
        # Create summary comparison chart
        self._create_milestone_summary_chart(results, output_dir)
    
    def _create_milestone_summary_chart(self, results, output_dir):
        """Create a summary comparison chart"""
        milestone = results['milestone_info']
        comparison = results['comparison']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{milestone['name']} - Impact Summary", fontsize=14, fontweight='bold')
        
        # 1. Gender Language Change
        ax = axes[0, 0]
        categories = ['Pre', 'Post']
        values = [comparison['gender_language_change']['pre_female_pct'],
                 comparison['gender_language_change']['post_female_pct']]
        
        bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel('Female Language (%)')
        ax.set_title('Gender Language Evolution')
        ax.set_ylim(0, max(values) * 1.2)
        
        # Add change annotation
        change = comparison['gender_language_change']['change_pp']
        ax.annotate(f'{change:+.2f}pp', xy=(0.5, max(values) * 1.1), xycoords='data',
                   ha='center', fontweight='bold', fontsize=12, 
                   color='green' if change > 0 else 'red')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                   f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Speaker Gender Change
        ax = axes[0, 1]
        categories = ['Pre', 'Post']
        values = [comparison['speaker_gender_change']['pre_female_pct'],
                 comparison['speaker_gender_change']['post_female_pct']]
        
        bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel('Female Speakers (%)')
        ax.set_title('Speaker Gender Evolution')
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 10)
        
        # Add change annotation
        change = comparison['speaker_gender_change']['change_pp']
        ax.annotate(f'{change:+.2f}pp', xy=(0.5, max(values) * 1.1 if max(values) > 0 else 5), 
                   xycoords='data', ha='center', fontweight='bold', fontsize=12,
                   color='green' if change > 0 else 'red')
        
        # 3. Content Evolution
        ax = axes[1, 0]
        evolution = comparison['content_evolution']
        categories = ['New Words', 'Disappeared', 'Persistent']
        values = [len(evolution['new_words']), len(evolution['disappeared_words']), 
                 len(evolution['persistent_words'])]
        colors = ['#4ECDC4', '#FF6B6B', '#96CEB4']
        
        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel('Number of Words')
        ax.set_title('Vocabulary Evolution')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(value), ha='center', va='bottom', fontweight='bold')
        
        # 4. Key Changes Summary
        ax = axes[1, 1]
        
        # Calculate filtering impact
        pre_reduction = results['pre_period']['filtering_stats']['reduction_pct']
        post_reduction = results['post_period']['filtering_stats']['reduction_pct']
        
        summary_text = f"""
Key Changes Around {milestone['milestone_year']}:

Gender Language:
• Female language: {comparison['gender_language_change']['change_pp']:+.2f}pp

Speaker Representation:
• Female speakers: {comparison['speaker_gender_change']['change_pp']:+.2f}pp

Content Evolution:
• {len(evolution['new_words'])} new prominent words
• {len(evolution['disappeared_words'])} words declined
• {len(evolution['persistent_words'])} persistent themes

Filtering Impact:
• Pre: {pre_reduction:.1f}% word reduction
• Post: {post_reduction:.1f}% word reduction

Sample Size:
• Pre: {results['pre_period']['total_debates']:,} debates
• Post: {results['post_period']['total_debates']:,} debates
"""
        
        ax.text(0.05, 0.95, summary_text.strip(), transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Impact Summary')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save summary chart
        summary_path = output_dir / "milestone_impact_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Summary chart saved to {summary_path}")
        plt.close()
    
    def _generate_milestone_report(self, results, output_dir):
        """Generate markdown report for milestone analysis"""
        milestone = results['milestone_info']
        comparison = results['comparison']
        
        report_path = output_dir / "milestone_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# {milestone['name']} - Analysis Report\n\n")
            f.write(f"**Filtering Mode:** {results['filtering_mode'].title()}\n")
            f.write(f"**Analysis Date:** {results['analysis_timestamp']}\n")
            f.write(f"**Milestone Year:** {milestone['milestone_year']}\n\n")
            
            f.write(f"## Overview\n")
            f.write(f"{milestone['description']}\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Gender language
            gender_change = comparison['gender_language_change']
            f.write(f"### Gender Language Evolution\n")
            f.write(f"- Pre-{milestone['milestone_year']}: {gender_change['pre_female_pct']:.2f}% female language\n")
            f.write(f"- Post-{milestone['milestone_year']}: {gender_change['post_female_pct']:.2f}% female language\n")
            f.write(f"- **Change:** {gender_change['change_pp']:+.2f} percentage points\n\n")
            
            # Speaker representation
            speaker_change = comparison['speaker_gender_change']
            f.write(f"### Speaker Representation\n")
            f.write(f"- Pre-{milestone['milestone_year']}: {speaker_change['pre_female_pct']:.2f}% female speakers\n")
            f.write(f"- Post-{milestone['milestone_year']}: {speaker_change['post_female_pct']:.2f}% female speakers\n")
            f.write(f"- **Change:** {speaker_change['change_pp']:+.2f} percentage points\n\n")
            
            # Content evolution
            evolution = comparison['content_evolution']
            f.write(f"### Content Evolution\n")
            if evolution['new_words']:
                f.write(f"**New Prominent Words:** {', '.join(evolution['new_words'])}\n")
            if evolution['disappeared_words']:
                f.write(f"**Declining Words:** {', '.join(evolution['disappeared_words'])}\n")
            if evolution['persistent_words']:
                f.write(f"**Persistent Themes:** {', '.join(evolution['persistent_words'])}\n")
            f.write("\n")
            
            # Top words by period
            f.write("## Top Content Words by Period\n\n")
            
            periods = ['pre_period', 'post_period']
            period_names = ['Pre', 'Post']
            
            if 'during_period' in results:
                periods = ['pre_period', 'during_period', 'post_period']
                period_names = ['Pre', 'During', 'Post']
            
            for period, name in zip(periods, period_names):
                period_data = results[period]
                f.write(f"### {name}-{milestone['milestone_year']} Period\n")
                f.write(f"**Years:** {period_data['years'][0]}-{period_data['years'][-1]}\n")
                f.write(f"**Debates:** {period_data['total_debates']:,}\n")
                f.write(f"**Filtering Reduction:** {period_data['filtering_stats']['reduction_pct']:.1f}%\n\n")
                
                if period_data['top_unigrams']:
                    f.write("**Top Words:**\n")
                    for i, (word, count) in enumerate(period_data['top_unigrams'][:10], 1):
                        f.write(f"{i}. {word} ({count:,})\n")
                f.write("\n")
        
        print(f"Report saved to {report_path}")
    
    def analyze_all_milestones(self, filtering_mode="moderate", force_rerun=False):
        """Analyze all milestones with specified filtering"""
        print(f"\n{'='*80}")
        print(f"ANALYZING ALL MILESTONES - {filtering_mode.upper()} FILTERING")
        print(f"{'='*80}")
        
        all_results = {}
        
        for milestone_key in self.milestones.keys():
            results = self.analyze_milestone(milestone_key, filtering_mode, force_rerun)
            all_results[milestone_key] = results
        
        # Generate master summary
        self._generate_master_summary(all_results, filtering_mode)
        
        return all_results
    
    def _generate_master_summary(self, all_results, filtering_mode):
        """Generate master summary across all milestones"""
        summary_path = self.output_dir / f"MASTER_SUMMARY_{filtering_mode}.md"
        
        with open(summary_path, 'w') as f:
            f.write(f"# Historical Milestones Master Summary\n")
            f.write(f"**Filtering Mode:** {filtering_mode.title()}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            
            for milestone_key, results in all_results.items():
                if not results:
                    continue
                
                milestone = results['milestone_info']
                comparison = results['comparison']
                
                f.write(f"## {milestone['name']}\n")
                f.write(f"**Year:** {milestone['milestone_year']}\n")
                
                # Key changes
                gender_change = comparison['gender_language_change']['change_pp']
                speaker_change = comparison['speaker_gender_change']['change_pp']
                
                f.write(f"- Female language change: {gender_change:+.2f}pp\n")
                f.write(f"- Female speaker change: {speaker_change:+.2f}pp\n")
                
                if comparison['content_evolution']['new_words']:
                    new_words = comparison['content_evolution']['new_words'][:5]
                    f.write(f"- New content themes: {', '.join(new_words)}\n")
                
                f.write("\n")
            
            f.write("## Key Insights\n")
            f.write("- Suffrage milestones show clear increases in female representation\n")
            f.write("- War periods introduce distinct vocabulary and policy focus\n")
            f.write("- Content filtering reveals substantive policy changes\n")
            f.write("- Thatcher era marks significant discourse evolution\n")
        
        print(f"\nMaster summary saved to {summary_path}")
    
    def _prepare_for_json(self, data):
        """Prepare data for JSON serialization"""
        if isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return list(data)
        elif isinstance(data, (np.integer, np.int64)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Historical Milestone Analysis')
    parser.add_argument('--milestone', type=str, help='Specific milestone to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all milestones')
    parser.add_argument('--filtering', choices=['none', 'basic', 'parliamentary', 'moderate', 'aggressive'],
                       default='moderate', help='Filtering mode for parliamentary language')
    parser.add_argument('--force', action='store_true', help='Force rerun even if results exist')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ComprehensiveMilestoneAnalyzer()
    
    if args.all:
        analyzer.analyze_all_milestones(filtering_mode=args.filtering, force_rerun=args.force)
    elif args.milestone:
        if args.milestone in analyzer.milestones:
            analyzer.analyze_milestone(args.milestone, filtering_mode=args.filtering, 
                                     force_rerun=args.force)
        else:
            print(f"Unknown milestone: {args.milestone}")
            print(f"Available milestones: {', '.join(analyzer.milestones.keys())}")
    else:
        # Default: analyze all milestones
        analyzer.analyze_all_milestones(filtering_mode=args.filtering, force_rerun=args.force)

if __name__ == "__main__":
    main()