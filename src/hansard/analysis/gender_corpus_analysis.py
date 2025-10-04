#!/usr/bin/env python3
"""
Gender-Matched Hansard Corpus Analysis

Analyzes the gender-matched parliamentary dataset, focusing only on speeches
by confirmed MPs with known gender classifications.

Usage:
    python gender_corpus_analysis.py --years 1920-1930 --sample 1000
    python gender_corpus_analysis.py --full --sample 10000
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

# Import path configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_config import Paths

class GenderCorpusAnalyzer:
    def __init__(self):
        # Use centralized paths
        self.data_dir = Paths.GENDER_ENHANCED_DATA
        self.output_dir = Paths.CORPUS_RESULTS
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load gender wordlists using centralized paths
        self.male_words = self._load_gender_wordlist(Paths.MALE_WORDS)
        self.female_words = self._load_gender_wordlist(Paths.FEMALE_WORDS)

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
        """Basic English stop words"""
        return set(['the', 'of', 'to', 'and', 'a', 'an', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for',
                   'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from',
                   'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when',
                   'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if',
                   'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
                   'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very',
                   'been', 'being', 'am', 'are', 'was', 'were', 'be', 'had', 'has', 'have', 'having'])

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
        """Moderate filtering"""
        parliamentary = self._get_parliamentary_stop_words()
        moderate = {'government', 'minister', 'secretary', 'state', 'department', 'office', 'time',
                   'year', 'years', 'day', 'way', 'case', 'point', 'matter', 'place', 'hand', 'part',
                   'number', 'great', 'public', 'present', 'general', 'particular', 'whole', 'certain'}
        return parliamentary | moderate

    def _get_aggressive_stop_words(self):
        """Aggressive filtering"""
        moderate = self._get_moderate_stop_words()
        aggressive = {'think', 'know', 'say', 'well', 'good', 'right', 'may', 'now', 'see', 'come',
                     'take', 'make', 'give', 'look', 'want', 'go', 'get', 'put', 'let', 'find',
                     'made', 'given', 'taken', 'done', 'said', 'went', 'came', 'got', 'found'}
        return moderate | aggressive

    def load_gender_matched_data(self, year_range=None, sample_size=None):
        """Load gender-matched debate data from parquet files efficiently"""
        import glob
        import random

        # Use centralized path method to get files
        if year_range:
            start_year, end_year = map(int, year_range.split('-'))
            year_files = Paths.get_year_files(start_year, end_year)
        else:
            year_files = Paths.get_year_files()

        year_files = [str(f) for f in year_files]  # Convert Path objects to strings for compatibility

        if not year_files:
            print(f"No data files found in {self.data_dir}")
            return []

        print(f"Loading gender-matched data from {len(year_files)} year files...")

        # Collect speeches by gender
        male_speeches = []
        female_speeches = []
        debates_metadata = []

        total_debates = 0
        files_processed = 0

        for file_path in year_files:
            try:
                df = pd.read_parquet(file_path)
                files_processed += 1

                # Extract year from filename
                year_match = re.search(r'debates_(\d{4})_enhanced\.parquet', os.path.basename(file_path))
                year = int(year_match.group(1)) if year_match else 0

                # Process speeches from enhanced data structure
                for _, row in df.iterrows():
                    if 'speech_segments' in row and row['speech_segments'] is not None:
                        debate_male_texts = []
                        debate_female_texts = []

                        # Get speaker details for gender mapping
                        speaker_details = row.get('speaker_details', [])
                        speaker_gender_map = {}
                        if isinstance(speaker_details, (list, np.ndarray)):
                            for detail in speaker_details:
                                if isinstance(detail, dict):
                                    name = detail.get('original_name', '')
                                    gender = detail.get('gender', '').lower()
                                    if name and gender in ['m', 'f']:
                                        speaker_gender_map[name] = 'male' if gender == 'm' else 'female'

                        # Process speech segments
                        segments = row['speech_segments']
                        if isinstance(segments, (list, np.ndarray)):
                            for segment in segments:
                                if isinstance(segment, dict):
                                    speaker = segment.get('speaker', '')
                                    text = segment.get('text', '')

                                    # Map speaker to gender
                                    gender = speaker_gender_map.get(speaker)
                                    if not gender:
                                        # Try to match partial names
                                        for orig_name, g in speaker_gender_map.items():
                                            if speaker in orig_name or orig_name in speaker:
                                                gender = g
                                                break

                                    if gender == 'male' and text:
                                        debate_male_texts.append(text)
                                    elif gender == 'female' and text:
                                        debate_female_texts.append(text)

                        if debate_male_texts or debate_female_texts:
                            total_debates += 1

                            # Store metadata
                            debates_metadata.append({
                                'year': year,
                                'debate_id': row.get('debate_id', ''),
                                'title': row.get('title', ''),
                                'num_male_speeches': len(debate_male_texts),
                                'num_female_speeches': len(debate_female_texts)
                            })

                            # Collect speeches
                            male_speeches.extend(debate_male_texts)
                            female_speeches.extend(debate_female_texts)

                            # Early exit if we have enough samples
                            if sample_size and total_debates >= sample_size * 2:
                                break

                print(f"Processed {files_processed}/{len(year_files)} files, {total_debates} debates so far...")

                # Check if we have enough samples
                if sample_size and total_debates >= sample_size * 2:
                    break

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        # Sample if needed
        if sample_size and total_debates > sample_size:
            random.seed(42)
            sample_indices = random.sample(range(total_debates), sample_size)

            # Create sampled data
            sampled_metadata = [debates_metadata[i] for i in sample_indices]

            # For simplicity in sampling, we'll take proportional speeches
            male_sample_size = min(len(male_speeches), sample_size * 10)  # Approx 10 speeches per debate
            female_sample_size = min(len(female_speeches), sample_size * 2)  # Fewer female speeches historically

            male_speeches = random.sample(male_speeches, male_sample_size) if len(male_speeches) > male_sample_size else male_speeches
            female_speeches = random.sample(female_speeches, female_sample_size) if len(female_speeches) > female_sample_size else female_speeches

            debates_metadata = sampled_metadata
            total_debates = len(sampled_metadata)

        print(f"\nLoaded {total_debates} gender-matched debates")
        print(f"  - {len(male_speeches)} male MP speeches")
        print(f"  - {len(female_speeches)} female MP speeches")

        return {
            'debates_metadata': debates_metadata,
            'male_speeches': male_speeches,
            'female_speeches': female_speeches,
            'total_debates': total_debates
        }

    def filter_text(self, text, level=0):
        """Apply filtering to text based on level"""
        if not text:
            return ""

        # Convert to lowercase and split
        words = text.lower().split()

        # Get filtering parameters
        params = self.filtering_levels[level]
        stop_words = params['stop_words']
        min_len = params['min_len']

        # Filter words
        filtered = []
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w\s]', '', word)

            # Apply filters
            if word and len(word) >= min_len and word not in stop_words:
                filtered.append(word)

        return ' '.join(filtered)

    def analyze_gender_patterns(self, data, filtered_male_texts, filtered_female_texts):
        """Analyze gender-specific patterns in the corpus"""
        metadata = data['debates_metadata']

        results = {
            'total_debates': data['total_debates'],
            'debates_with_male_speakers': sum(1 for d in metadata if d['num_male_speeches'] > 0),
            'debates_with_female_speakers': sum(1 for d in metadata if d['num_female_speeches'] > 0),
            'total_male_speeches': sum(d['num_male_speeches'] for d in metadata),
            'total_female_speeches': sum(d['num_female_speeches'] for d in metadata)
        }

        # Analyze filtered word frequencies by gender
        if filtered_male_texts:
            male_words = ' '.join(filtered_male_texts).lower().split()
            male_word_freq = Counter(male_words)
            results['top_male_words'] = male_word_freq.most_common(30)
            results['unique_male_words'] = len(male_word_freq)

        if filtered_female_texts:
            female_words = ' '.join(filtered_female_texts).lower().split()
            female_word_freq = Counter(female_words)
            results['top_female_words'] = female_word_freq.most_common(30)
            results['unique_female_words'] = len(female_word_freq)

        # Calculate gender language usage
        if filtered_male_texts and self.male_words and self.female_words:
            male_text = ' '.join(filtered_male_texts).lower()
            male_words_set = set(male_text.split())
            results['male_using_male_words'] = len(male_words_set & self.male_words)
            results['male_using_female_words'] = len(male_words_set & self.female_words)

        if filtered_female_texts and self.male_words and self.female_words:
            female_text = ' '.join(filtered_female_texts).lower()
            female_words_set = set(female_text.split())
            results['female_using_male_words'] = len(female_words_set & self.male_words)
            results['female_using_female_words'] = len(female_words_set & self.female_words)

        return results

    def create_gender_visualizations(self, results, level):
        """Create visualizations specific to gender analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Gender participation over time
        ax = axes[0, 0]
        if 'temporal_gender_data' in results:
            temporal_data = results['temporal_gender_data']
            ax.plot(temporal_data['years'], temporal_data['male_participation'], label='Male MPs', color='blue')
            ax.plot(temporal_data['years'], temporal_data['female_participation'], label='Female MPs', color='red')
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Speeches')
            ax.set_title('MP Participation by Gender Over Time')
            ax.legend()

        # Plot 2: Top words by gender
        ax = axes[0, 1]
        if 'gender_patterns' in results and 'top_male_words' in results['gender_patterns']:
            male_words = results['gender_patterns']['top_male_words'][:15]
            female_words = results['gender_patterns'].get('top_female_words', [])[:15]

            y_pos = np.arange(len(male_words))
            ax.barh(y_pos, [count for _, count in male_words], color='blue', alpha=0.7, label='Male MPs')
            if female_words:
                ax.barh(y_pos + 0.4, [count for _, count in female_words], color='red', alpha=0.7, label='Female MPs')

            ax.set_yticks(y_pos + 0.2)
            ax.set_yticklabels([word for word, _ in male_words])
            ax.set_xlabel('Frequency')
            ax.set_title('Top Words by Gender')
            ax.legend()

        # Plot 3: Gender word usage
        ax = axes[1, 0]
        if 'gender_language_usage' in results:
            gender_usage = results['gender_language_usage']
            categories = ['Male-associated words', 'Female-associated words']
            male_usage = [gender_usage.get('male_using_male_words', 0), gender_usage.get('male_using_female_words', 0)]
            female_usage = [gender_usage.get('female_using_male_words', 0), gender_usage.get('female_using_female_words', 0)]

            x = np.arange(len(categories))
            width = 0.35
            ax.bar(x - width/2, male_usage, width, label='Male MPs', color='blue')
            ax.bar(x + width/2, female_usage, width, label='Female MPs', color='red')
            ax.set_xlabel('Word Category')
            ax.set_ylabel('Usage Frequency')
            ax.set_title('Gendered Language Usage Patterns')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()

        # Plot 4: Debate participation ratio
        ax = axes[1, 1]
        if 'gender_patterns' in results:
            gender_data = results['gender_patterns']
            sizes = [gender_data.get('total_male_speeches', 0), gender_data.get('total_female_speeches', 0)]
            labels = ['Male MPs', 'Female MPs']
            colors = ['blue', 'red']

            if sum(sizes) > 0:
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title('Overall Speech Distribution by Gender')

        plt.suptitle(f'Gender Analysis - Filter Level {level}: {self.filtering_levels[level]["name"]}', fontsize=16)
        plt.tight_layout()

        output_path = self.output_dir / f'gender_analysis_level_{level}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Gender visualization saved to {output_path}")

    def run_analysis(self, year_range=None, sample_size=None, full=False):
        """Run comprehensive gender-matched corpus analysis"""
        print("\n" + "="*80)
        print("GENDER-MATCHED HANSARD CORPUS ANALYSIS")
        print("="*80)

        # Load gender-matched data
        data = self.load_gender_matched_data(year_range, sample_size)

        if not data or data['total_debates'] == 0:
            print("No gender-matched debates found!")
            return None

        all_results = {}

        for level in range(6):  # 0-5 filtering levels
            print(f"\n--- LEVEL {level}: {self.filtering_levels[level]['name']} ---")

            # Filter texts by gender
            filtered_male_texts = [self.filter_text(text, level) for text in data['male_speeches']]
            filtered_female_texts = [self.filter_text(text, level) for text in data['female_speeches']]

            # Remove empty strings
            filtered_male_texts = [t for t in filtered_male_texts if t]
            filtered_female_texts = [t for t in filtered_female_texts if t]

            # Combined analysis
            all_filtered = filtered_male_texts + filtered_female_texts
            all_text = ' '.join(all_filtered)
            words = all_text.split()
            word_freq = Counter(words)

            results = {
                'filter_level': level,
                'filter_name': self.filtering_levels[level]['name'],
                'total_debates': data['total_debates'],
                'total_words': len(words),
                'unique_words': len(word_freq),
                'top_words': word_freq.most_common(50),
                'gender_patterns': self.analyze_gender_patterns(data, filtered_male_texts, filtered_female_texts)
            }

            # Topic modeling if enough content
            if len(all_filtered) > 100 and len(words) > 1000:
                try:
                    vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.8)
                    tfidf_matrix = vectorizer.fit_transform(all_filtered)

                    lda = LatentDirichletAllocation(n_components=5, random_state=42)
                    lda.fit(tfidf_matrix)

                    feature_names = vectorizer.get_feature_names_out()
                    topics = []
                    for topic_idx, topic in enumerate(lda.components_):
                        top_indices = topic.argsort()[-10:][::-1]
                        top_words = [feature_names[i] for i in top_indices]
                        topics.append(top_words)
                    results['topics'] = topics
                except Exception as e:
                    print(f"Topic modeling failed: {e}")

            all_results[level] = results

            # Create visualizations
            self.create_gender_visualizations(results, level)

        # Save comprehensive results
        output_path = self.output_dir / 'gender_corpus_analysis_results.json'
        with open(output_path, 'w') as f:
            # Convert for JSON serialization
            json_results = {}
            for level, data in all_results.items():
                json_results[str(level)] = {
                    k: v if not isinstance(v, (np.ndarray, np.int64, np.float64)) else
                    v.tolist() if isinstance(v, np.ndarray) else float(v)
                    for k, v in data.items()
                }
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to {output_path}")
        return all_results

def main():
    parser = argparse.ArgumentParser(description='Analyze gender-matched Hansard corpus')
    parser.add_argument('--years', type=str, help='Year range (e.g., 1920-1935)')
    parser.add_argument('--sample', type=int, help='Sample size for analysis')
    parser.add_argument('--full', action='store_true', help='Analyze full corpus')

    args = parser.parse_args()

    analyzer = GenderCorpusAnalyzer()
    results = analyzer.run_analysis(
        year_range=args.years,
        sample_size=args.sample,
        full=args.full
    )

    if results:
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("Check the 'analysis/gender_corpus_results' directory for outputs")
        print("="*80)

if __name__ == "__main__":
    main()