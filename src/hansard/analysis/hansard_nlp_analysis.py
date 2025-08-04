#!/usr/bin/env python3
"""
Hansard NLP Analysis Script

Performs comprehensive NLP analysis on the Hansard parliamentary debates corpus:
1. Unigram and bigram frequency analysis  
2. Topic modeling with LDA
3. Gender analysis using UCLA NLP gender wordlists
4. Temporal analysis around women's suffrage (1928)

Usage:
    python hansard_nlp_analysis.py --years 1920-1930  # Specific range
    python hansard_nlp_analysis.py --sample 1000      # Sample analysis
    python hansard_nlp_analysis.py --full             # Full corpus
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

class HansardNLPAnalyzer:
    def __init__(self, data_dir="data/processed_fixed", output_dir="analysis/results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load gender wordlists
        self.male_words = self._load_gender_wordlist("data/gender_wordlists/male_words.txt")
        self.female_words = self._load_gender_wordlist("data/gender_wordlists/female_words.txt")
        
        # Analysis results storage
        self.results = {}
        
        # Speaker gender analysis patterns
        self.female_indicators = [
            'Mrs.', 'Miss', 'Lady', 'Dame', 'Madam', 'Viscountess', 'Countess', 
            'Baroness', 'Duchess', 'Marchioness', 'Princess'
        ]
        self.male_indicators = [
            'Mr.', 'Sir', 'Lord', 'Earl', 'Duke', 'Baron', 'Count', 'Marquess', 
            'Prince', 'Viscount'
        ]
        
        
    def _load_gender_wordlist(self, filepath):
        """Load gender wordlist and convert to lowercase set"""
        try:
            with open(filepath, 'r') as f:
                return set(word.strip().lower() for word in f if word.strip())
        except FileNotFoundError:
            print(f"Warning: Gender wordlist not found at {filepath}")
            return set()
    
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
                            if debate_data['text']:  # Only include non-empty texts
                                year_debates.append(debate_data)
                
                print(f"Loaded {len(year_debates)} debates from {year}")
                debates.extend(year_debates)
                years_processed.append(year)
                
            except Exception as e:
                print(f"Error loading {year}: {e}")
                continue
        
        if sample_size and len(debates) > sample_size:
            debates = np.random.choice(debates, sample_size, replace=False).tolist()
            print(f"Sampled {sample_size} debates from {len(debates)} total")
        
        if years_processed:
            print(f"Total debates loaded: {len(debates)} from years {min(years_processed)}-{max(years_processed)}")
        else:
            print(f"Total debates loaded: {len(debates)}")
        return debates
    
    def clean_text(self, text):
        """Basic text cleaning for NLP analysis"""
        # Remove HTML entities and normalize whitespace
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def analyze_unigrams_bigrams(self, texts, top_n=50):
        """Analyze most frequent unigrams and bigrams"""
        print("Analyzing unigrams and bigrams...")
        top_unigrams, top_bigrams = self._compute_unigrams_bigrams(texts, top_n)
        self.results['unigrams'] = top_unigrams
        self.results['bigrams'] = top_bigrams
        return top_unigrams, top_bigrams
    
    def topic_modeling(self, texts, n_topics=10, n_words=10):
        """Perform LDA topic modeling"""
        print(f"Performing topic modeling with {n_topics} topics...")
        
        # Clean and vectorize texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Use TF-IDF for better topic modeling
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=5,
            max_df=0.7,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        
        # LDA topic modeling
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100,
            learning_method='batch'
        )
        lda.fit(tfidf_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-n_words:][::-1]]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': topic[topic.argsort()[-n_words:][::-1]]
            })
        
        self.results['topics'] = topics
        return topics
    
    def gender_analysis(self, texts):
        """Analyze gender-related language using UCLA wordlists"""
        print("Performing gender analysis...")
        gender_stats = self._compute_gender_stats(texts)
        self.results['gender_analysis'] = gender_stats
        return gender_stats
    
    def _compute_gender_stats(self, texts):
        """Compute gender statistics without storing in self.results"""
        gender_stats = {
            'male_word_count': 0,
            'female_word_count': 0,
            'male_words_found': Counter(),
            'female_words_found': Counter(),
            'total_words': 0
        }
        
        for text in texts:
            words = self.clean_text(text).split()
            gender_stats['total_words'] += len(words)
            
            for word in words:
                if word in self.male_words:
                    gender_stats['male_word_count'] += 1
                    gender_stats['male_words_found'][word] += 1
                elif word in self.female_words:
                    gender_stats['female_word_count'] += 1
                    gender_stats['female_words_found'][word] += 1
        
        # Calculate ratios
        total_gendered = gender_stats['male_word_count'] + gender_stats['female_word_count']
        if total_gendered > 0:
            gender_stats['male_ratio'] = gender_stats['male_word_count'] / total_gendered
            gender_stats['female_ratio'] = gender_stats['female_word_count'] / total_gendered
        else:
            gender_stats['male_ratio'] = gender_stats['female_ratio'] = 0
            
        return gender_stats
    
    def _compute_unigrams_bigrams(self, texts, top_n=50):
        """Compute unigrams and bigrams without storing in self.results"""
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Unigrams
        unigram_vectorizer = CountVectorizer(
            stop_words='english',
            max_features=10000,
            min_df=5,
            token_pattern=r'\b[a-zA-Z]{3,}\b'  # Words with 3+ letters
        )
        unigram_matrix = unigram_vectorizer.fit_transform(cleaned_texts)
        unigram_counts = unigram_matrix.sum(axis=0).A1
        unigram_freq = dict(zip(unigram_vectorizer.get_feature_names_out(), unigram_counts))
        top_unigrams = Counter(unigram_freq).most_common(top_n)
        
        # Bigrams
        bigram_vectorizer = CountVectorizer(
            stop_words='english',
            ngram_range=(2, 2),
            max_features=5000,
            min_df=3,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        bigram_matrix = bigram_vectorizer.fit_transform(cleaned_texts)
        bigram_counts = bigram_matrix.sum(axis=0).A1
        bigram_freq = dict(zip(bigram_vectorizer.get_feature_names_out(), bigram_counts))
        top_bigrams = Counter(bigram_freq).most_common(top_n)
        
        return top_unigrams, top_bigrams
    
    def temporal_analysis(self, debates, split_year=1928):
        """Analyze differences before and after women's suffrage (1928)"""
        print(f"Performing temporal analysis with split year {split_year}...")
        
        pre_suffrage = [d for d in debates if d['year'] < split_year]
        post_suffrage = [d for d in debates if d['year'] >= split_year]
        
        print(f"Pre-suffrage debates: {len(pre_suffrage)} (years {min(d['year'] for d in pre_suffrage) if pre_suffrage else 'N/A'}-{split_year-1})")
        print(f"Post-suffrage debates: {len(post_suffrage)} (years {split_year}-{max(d['year'] for d in post_suffrage) if post_suffrage else 'N/A'})")
        
        temporal_results = {}
        
        # Gender analysis for each period
        if pre_suffrage:
            pre_texts = [d['text'] for d in pre_suffrage]
            temporal_results['pre_suffrage_gender'] = self._compute_gender_stats(pre_texts)
            temporal_results['pre_suffrage_unigrams'], temporal_results['pre_suffrage_bigrams'] = self._compute_unigrams_bigrams(pre_texts, top_n=25)
        
        if post_suffrage:
            post_texts = [d['text'] for d in post_suffrage]
            temporal_results['post_suffrage_gender'] = self._compute_gender_stats(post_texts)
            temporal_results['post_suffrage_unigrams'], temporal_results['post_suffrage_bigrams'] = self._compute_unigrams_bigrams(post_texts, top_n=25)
        
        self.results['temporal_analysis'] = temporal_results
        return temporal_results
    
    def generate_visualizations(self):
        """Generate individual visualization files"""
        print("Generating individual visualizations...")
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plt.style.use('default')
        
        # 1. Top Unigrams
        if 'unigrams' in self.results:
            plt.figure(figsize=(8, 6))
            words, counts = zip(*self.results['unigrams'][:15])
            counts = [int(c) for c in counts]
            plt.barh(range(len(words)), counts, color='steelblue')
            plt.yticks(range(len(words)), words)
            plt.title('Top 15 Most Frequent Words', fontweight='bold')
            plt.xlabel('Frequency')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(plots_dir / 'top_unigrams.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Top Bigrams
        if 'bigrams' in self.results:
            plt.figure(figsize=(8, 6))
            bigrams, counts = zip(*self.results['bigrams'][:12])
            counts = [int(c) for c in counts]
            plt.barh(range(len(bigrams)), counts, color='darkgreen')
            plt.yticks(range(len(bigrams)), [' '.join(bg.split()) for bg in bigrams])
            plt.title('Top 12 Most Frequent Phrases', fontweight='bold')
            plt.xlabel('Frequency')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(plots_dir / 'top_bigrams.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Chamber Distribution
        if 'corpus_overview' in self.results and self.results['corpus_overview']['chamber_distribution']:
            plt.figure(figsize=(6, 6))
            corpus = self.results['corpus_overview']
            chambers = list(corpus['chamber_distribution'].keys())
            counts = list(corpus['chamber_distribution'].values())
            plt.pie(counts, labels=chambers, autopct='%1.1f%%', startangle=90)
            plt.title(f'Chamber Distribution\n({corpus["total_debates"]:,} debates)', fontweight='bold')
            plt.tight_layout()
            plt.savefig(plots_dir / 'chamber_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Gender Word Distribution
        if 'gender_analysis' in self.results:
            plt.figure(figsize=(6, 6))
            gender_data = self.results['gender_analysis']
            categories = ['Male Words', 'Female Words']
            counts = [gender_data['male_word_count'], gender_data['female_word_count']]
            colors = ['lightblue', 'pink']
            plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors)
            plt.title('Gender Word Distribution', fontweight='bold')
            plt.tight_layout()
            plt.savefig(plots_dir / 'gender_word_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 5. Temporal Gender Comparison
        if 'temporal_analysis' in self.results:
            temp_data = self.results['temporal_analysis']
            periods = []
            male_ratios = []
            female_ratios = []
            
            if 'pre_suffrage_gender' in temp_data:
                periods.append('Pre-1928')
                male_ratios.append(temp_data['pre_suffrage_gender']['male_ratio'])
                female_ratios.append(temp_data['pre_suffrage_gender']['female_ratio'])
            
            if 'post_suffrage_gender' in temp_data:
                periods.append('Post-1928')
                male_ratios.append(temp_data['post_suffrage_gender']['male_ratio'])
                female_ratios.append(temp_data['post_suffrage_gender']['female_ratio'])
            
            if periods:
                plt.figure(figsize=(6, 6))
                x = np.arange(len(periods))
                width = 0.35
                plt.bar(x - width/2, male_ratios, width, label='Male', alpha=0.8, color='lightblue')
                plt.bar(x + width/2, female_ratios, width, label='Female', alpha=0.8, color='pink')
                plt.xlabel('Period')
                plt.ylabel('Ratio of Gendered Words')
                plt.title('Gender Language Before/After 1928 Suffrage', fontweight='bold')
                plt.xticks(x, periods)
                plt.legend()
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(plots_dir / 'temporal_gender_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        # 6. Speaker Gender Distribution
        if 'speaker_gender_analysis' in self.results:
            plt.figure(figsize=(6, 6))
            speakers = self.results['speaker_gender_analysis']
            gender_counts = [
                int(speakers['unique_gender_distribution'].get('male', 0)),
                int(speakers['unique_gender_distribution'].get('female', 0)),
                int(speakers['unique_gender_distribution'].get('unknown', 0))
            ]
            labels = ['Male Speakers', 'Female Speakers', 'Unknown']
            colors = ['lightblue', 'pink', 'lightgray']
            plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            plt.title(f'Speaker Gender Distribution\n({speakers["unique_speakers"]:,} unique speakers)', fontweight='bold')
            plt.tight_layout()
            plt.savefig(plots_dir / 'speaker_gender_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 7. Topic Modeling Visualizations
        if 'topics' in self.results and self.results['topics']:
            # Topic words visualization - just show the words without confusing weights
            plt.figure(figsize=(12, 8))
            n_topics = min(8, len(self.results['topics']))
            for i in range(n_topics):
                topic = self.results['topics'][i]
                words = topic.get('words', [])[:8]  # Top 8 words per topic
                
                plt.subplot(2, 4, i+1)
                # Use simple equal heights for clean visualization
                word_heights = list(range(len(words), 0, -1))  # Descending order
                colors = plt.cm.Set3(i)
                
                plt.barh(range(len(words)), word_heights, color=colors, alpha=0.7)
                plt.yticks(range(len(words)), words)
                plt.title(f'Topic {i+1}', fontweight='bold')
                plt.gca().invert_yaxis()
                plt.xlabel('Importance')
                plt.xticks([])  # Hide x-axis ticks since heights are just for ranking
            
            plt.suptitle('Topic Modeling Results - Top Words per Topic', fontweight='bold', fontsize=16)
            plt.tight_layout()
            plt.savefig(plots_dir / 'topic_modeling_overview.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Individual plots saved to {plots_dir}/")
    
    def save_results(self, filename='hansard_nlp_results.json'):
        """Save analysis results to JSON"""
        output_path = self.output_dir / filename
        
        # Convert Counter objects to regular dicts for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, Counter):
                        json_results[key][k] = dict(v.most_common(50))
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}")
    
    def analyze_corpus_overview(self, debates):
        """Analyze overall corpus statistics"""
        print("Analyzing corpus overview...")
        
        # Basic corpus stats
        total_debates = len(debates)
        years_span = max(d['year'] for d in debates) - min(d['year'] for d in debates) + 1
        chambers = Counter(d['chamber'] for d in debates if d['chamber'])
        
        # Word count statistics
        word_counts = [d['word_count'] for d in debates if d['word_count']]
        total_words = sum(word_counts)
        avg_words_per_debate = total_words / len(word_counts) if word_counts else 0
        
        # Speaker statistics
        all_speakers = []
        debates_with_speakers = 0
        for d in debates:
            speakers = d.get('speakers', [])
            if speakers:
                all_speakers.extend(speakers)
                debates_with_speakers += 1
        
        unique_speakers = len(set(all_speakers))
        speaker_coverage = (debates_with_speakers / total_debates) * 100 if total_debates > 0 else 0
        
        # Temporal distribution
        year_counts = Counter(d['year'] for d in debates)
        decades = Counter((d['year'] // 10) * 10 for d in debates)
        
        corpus_stats = {
            'total_debates': total_debates,
            'years_span': years_span,
            'year_range': f"{min(d['year'] for d in debates)}-{max(d['year'] for d in debates)}",
            'total_words': total_words,
            'avg_words_per_debate': avg_words_per_debate,
            'chamber_distribution': dict(chambers.most_common()),
            'unique_speakers': unique_speakers,
            'total_speaker_mentions': len(all_speakers),
            'debates_with_speakers': debates_with_speakers,
            'speaker_coverage_pct': speaker_coverage,
            'debates_per_decade': dict(decades.most_common()),
            'most_active_years': dict(year_counts.most_common(10)),
            'avg_debates_per_year': total_debates / years_span if years_span > 0 else 0
        }
        
        self.results['corpus_overview'] = corpus_stats
        return corpus_stats
    
    def classify_speaker_gender(self, speaker_name):
        """Classify speaker gender based on parliamentary titles only"""
        if pd.isna(speaker_name):
            return 'unknown'
        speaker_name = str(speaker_name).strip()
        speaker_lower = speaker_name.lower()
        
        # Check for female indicators (titles) - case insensitive with proper boundaries
        for indicator in self.female_indicators:
            # Use start/end of string or whitespace boundaries to handle periods properly
            pattern = r'(?:^|\s)' + re.escape(indicator.lower()) + r'(?:\s|$)'
            if re.search(pattern, speaker_lower):
                return 'female'
        
        # Check for male indicators - case insensitive with proper boundaries
        for indicator in self.male_indicators:
            pattern = r'(?:^|\s)' + re.escape(indicator.lower()) + r'(?:\s|$)'
            if re.search(pattern, speaker_lower):
                return 'male'
        
        return 'unknown'
    
    def analyze_speaker_gender(self, debates):
        """Analyze male/female speaker distribution for the specific debates in the sample"""
        print("Analyzing speaker gender distribution...")
        
        try:
            # Extract all speakers from the actual debate sample
            all_speakers = []
            debate_speakers = []
            
            for debate in debates:
                speakers = debate.get('speakers', [])
                year = debate['year']
                
                for speaker in speakers:
                    if speaker and speaker.strip():  # Skip empty speakers
                        all_speakers.append({
                            'speaker_name': speaker.strip(),
                            'year': year,
                            'decade': (year // 10) * 10
                        })
                        debate_speakers.append(speaker.strip())
            
            if not all_speakers:
                print("No speakers found in debate sample")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_speakers)
            
            # Apply gender classification
            df['gender'] = df['speaker_name'].apply(self.classify_speaker_gender)
            
            # Speaker mention statistics (including duplicates)
            gender_counts = df['gender'].value_counts()
            total_speaker_mentions = len(df)
            
            # Unique speaker statistics (deduplicated)
            unique_speakers_df = df.drop_duplicates('speaker_name')
            unique_gender_counts = unique_speakers_df['gender'].value_counts()
            unique_speakers_count = len(unique_speakers_df)
            
            # Temporal analysis by decade (unique speakers only)
            gender_by_decade = unique_speakers_df.groupby(['decade', 'gender']).size().unstack(fill_value=0)
            
            # Find first female speakers (from unique speakers)
            first_females = unique_speakers_df[unique_speakers_df['gender'] == 'female'].nsmallest(10, 'year')
            
            # Key historical periods (using unique speakers)
            suffrage_analysis = {}
            years_in_sample = sorted(df['year'].unique())
            
            if len(years_in_sample) > 1:
                if 1918 in years_in_sample or (min(years_in_sample) <= 1918 <= max(years_in_sample)):
                    pre_1918 = unique_speakers_df[unique_speakers_df['year'] < 1918]['gender'].value_counts()
                    post_1918 = unique_speakers_df[unique_speakers_df['year'] >= 1918]['gender'].value_counts()
                    suffrage_analysis['1918_split'] = {
                        'pre_1918': dict(pre_1918),
                        'post_1918': dict(post_1918)
                    }
                
                if 1928 in years_in_sample or (min(years_in_sample) <= 1928 <= max(years_in_sample)):
                    pre_1928 = unique_speakers_df[unique_speakers_df['year'] < 1928]['gender'].value_counts()
                    post_1928 = unique_speakers_df[unique_speakers_df['year'] >= 1928]['gender'].value_counts()
                    suffrage_analysis['1928_split'] = {
                        'pre_1928': dict(pre_1928),
                        'post_1928': dict(post_1928)
                    }
            
            speaker_gender_results = {
                'total_speaker_mentions': total_speaker_mentions,
                'unique_speakers': unique_speakers_count,
                'debates_analyzed': len(debates),
                'avg_speakers_per_debate': total_speaker_mentions / len(debates),
                'avg_unique_speakers_per_debate': unique_speakers_count / len(debates),
                
                # Mention-based statistics (with duplicates)
                'mention_gender_distribution': dict(gender_counts),
                'mention_gender_percentages': dict(gender_counts / total_speaker_mentions * 100),
                
                # Unique speaker statistics (deduplicated)
                'unique_gender_distribution': dict(unique_gender_counts),
                'unique_gender_percentages': dict(unique_gender_counts / unique_speakers_count * 100),
                
                'first_female_speakers': [
                    {'year': int(row['year']), 'name': row['speaker_name']} 
                    for _, row in first_females.iterrows()
                ],
                'gender_by_decade': gender_by_decade.to_dict(),
                'suffrage_analysis': suffrage_analysis,
                'years_analyzed': f"{min(years_in_sample)}-{max(years_in_sample)}"
            }
            
            self.results['speaker_gender_analysis'] = speaker_gender_results
            return speaker_gender_results
            
        except Exception as e:
            print(f"Error in speaker gender analysis: {e}")
            return None
    
    def run_full_analysis(self, start_year=1803, end_year=2005, sample_size=None):
        """Run complete NLP analysis pipeline with corpus overview"""
        print("=== Hansard NLP Analysis ===")
        print(f"Analyzing years {start_year}-{end_year}")
        if sample_size:
            print(f"Using sample size: {sample_size}")
        
        # Load data
        debates = self.load_debates(start_year, end_year, sample_size)
        if not debates:
            print("No debates loaded. Exiting.")
            return
        
        texts = [d['text'] for d in debates]
        
        # Run analyses - now including corpus overview and speaker gender
        self.analyze_corpus_overview(debates)
        self.analyze_unigrams_bigrams(texts)
        self.topic_modeling(texts, n_topics=8)
        self.gender_analysis(texts)
        self.temporal_analysis(debates)
        self.analyze_speaker_gender(debates)
        
        # Generate outputs
        self.generate_visualizations()
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print analysis summary with corpus overview"""
        print("\n=== ANALYSIS SUMMARY ===")
        
        # Corpus Overview
        if 'corpus_overview' in self.results:
            corpus = self.results['corpus_overview']
            print(f"\nCORPUS OVERVIEW:")
            print(f"  Total debates: {corpus['total_debates']:,}")
            print(f"  Time span: {corpus['year_range']} ({corpus['years_span']} years)")
            print(f"  Total words: {corpus['total_words']:,}")
            print(f"  Avg words per debate: {corpus['avg_words_per_debate']:,.0f}")
            print(f"  Unique speakers: {corpus['unique_speakers']:,}")
            print(f"  Speaker coverage: {corpus['speaker_coverage_pct']:.1f}% of debates")
            print(f"  Avg debates per year: {corpus['avg_debates_per_year']:.0f}")
            
            if corpus['chamber_distribution']:
                print(f"  Chamber distribution:")
                for chamber, count in corpus['chamber_distribution'].items():
                    pct = (count / corpus['total_debates']) * 100
                    print(f"     {chamber}: {count:,} ({pct:.1f}%)")
        
        if 'unigrams' in self.results:
            print(f"\nTOP 10 WORDS:")
            for word, count in self.results['unigrams'][:10]:
                print(f"  {word}: {count:,}")
        
        if 'topics' in self.results:
            print(f"\nTOPICS FOUND:")
            for topic in self.results['topics'][:5]:
                print(f"  Topic {topic['topic_id']}: {', '.join(topic['words'][:5])}")
        
        if 'gender_analysis' in self.results:
            gender = self.results['gender_analysis']
            print(f"\nGENDER ANALYSIS:")
            print(f"  Male words: {gender['male_word_count']:,} ({gender['male_ratio']:.1%})")
            print(f"  Female words: {gender['female_word_count']:,} ({gender['female_ratio']:.1%})")
        
        if 'temporal_analysis' in self.results:
            temp = self.results['temporal_analysis']
            print(f"\nTEMPORAL ANALYSIS (Pre/Post 1928):")
            if 'pre_suffrage_gender' in temp and 'post_suffrage_gender' in temp:
                pre_male = temp['pre_suffrage_gender']['male_ratio']
                post_male = temp['post_suffrage_gender']['male_ratio']
                change = ((post_male - pre_male) / pre_male * 100) if pre_male > 0 else 0
                print(f"  Male language ratio change: {change:+.1f}%")
        
        if 'speaker_gender_analysis' in self.results:
            speakers = self.results['speaker_gender_analysis']
            print(f"\nSPEAKER GENDER ANALYSIS:")
            print(f"  Debates analyzed: {speakers['debates_analyzed']:,}")
            print(f"  Total speaker mentions: {speakers['total_speaker_mentions']:,}")
            print(f"  Unique speakers: {speakers['unique_speakers']:,}")
            print(f"  Avg speakers per debate: {speakers['avg_speakers_per_debate']:.1f}")
            print(f"  Avg unique speakers per debate: {speakers['avg_unique_speakers_per_debate']:.1f}")
            
            if 'unique_gender_percentages' in speakers:
                female_pct = speakers['unique_gender_percentages'].get('female', 0)
                male_pct = speakers['unique_gender_percentages'].get('male', 0)
                unknown_pct = speakers['unique_gender_percentages'].get('unknown', 0)
                print(f"  Female speakers (unique): {speakers['unique_gender_distribution'].get('female', 0):,} ({female_pct:.1f}%)")
                print(f"  Male speakers (unique): {speakers['unique_gender_distribution'].get('male', 0):,} ({male_pct:.1f}%)")
                print(f"  Unknown gender (unique): {speakers['unique_gender_distribution'].get('unknown', 0):,} ({unknown_pct:.1f}%)")
            
            if 'first_female_speakers' in speakers and speakers['first_female_speakers']:
                first_female = speakers['first_female_speakers'][0]
                print(f"  First female speaker: {first_female['name']} ({first_female['year']})")
            
            # Show suffrage impact
            if 'suffrage_analysis' in speakers and '1928_split' in speakers['suffrage_analysis']:
                split_1928 = speakers['suffrage_analysis']['1928_split']
                pre_female = split_1928['pre_1928'].get('female', 0)
                post_female = split_1928['post_1928'].get('female', 0)
                pre_total = sum(split_1928['pre_1928'].values())
                post_total = sum(split_1928['post_1928'].values())
                
                if pre_total > 0 and post_total > 0:
                    pre_female_pct = (pre_female / pre_total) * 100
                    post_female_pct = (post_female / post_total) * 100
                    print(f"  Female speakers pre-1928: {pre_female_pct:.1f}% vs post-1928: {post_female_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Hansard NLP Analysis')
    parser.add_argument('--years', type=str, help='Year range (e.g., "1920-1935")')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--full', action='store_true', help='Analyze full corpus')
    parser.add_argument('--output', type=str, default='analysis/results', help='Output directory')
    
    args = parser.parse_args()
    
    # Parse year range
    if args.years:
        start_year, end_year = map(int, args.years.split('-'))
    elif args.full:
        start_year, end_year = 1803, 2005
    else:
        # Default: small test range
        start_year, end_year = 1920, 1935
        if not args.sample:
            args.sample = 500
    
    # Run analysis
    analyzer = HansardNLPAnalyzer(output_dir=args.output)
    analyzer.run_full_analysis(start_year, end_year, args.sample)


if __name__ == "__main__":
    main()