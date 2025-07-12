# Hansard NLP Explorer

A comprehensive tool for crawling, analyzing, and exploring UK Parliamentary debates from the Historic Hansard archive using natural language processing techniques.

## Features

- **Web Crawler**: Production-ready crawler for fetching Historic Hansard debates from the UK Parliament API
- **Data Processing**: Extract and process compressed HTML debate files  
- **NLP Analysis**: Topic modeling, named entity recognition, and sentiment analysis
- **Interactive Dashboard**: Streamlit-based web interface for exploring debates
- **Timeline Analysis**: Track political themes and discussions over time

## Project Structure

```
src/hansard/
├── crawler.py          # Web crawler for Historic Hansard API (v4.2)
├── data/hansard/       # Raw debate data (compressed HTML files)
│   └── 1864/          # Sample data from 1864
│       ├── feb/       # February debates
│       └── apr/       # April debates
├── parser.py          # [TODO] HTML parsing and text extraction
├── analysis/          # [TODO] NLP analysis modules
└── app.py            # [TODO] Streamlit dashboard
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate hansard

# Or install manually
pip install httpx tenacity polars spacy gensim scikit-learn streamlit
```

### 2. Crawl Data

```bash
# Crawl a single year
python src/hansard/crawler.py 1864

# Crawl a decade
python src/hansard/crawler.py 1860s

# Crawl a range with house filter
python src/hansard/crawler.py 1860 1869 --house commons --out data/hansard
```

### 3. Process and Analyze

```bash
# [TODO] Parse HTML files
python src/hansard/parser.py

# [TODO] Run analysis
python src/hansard/analysis/topics.py

# [TODO] Launch dashboard
streamlit run src/hansard/app.py
```

## Data Format

The crawler saves debates as:
- **Compressed HTML**: `{day}_{index}_{topic-slug}.html.gz`
- **Summary JSON**: `{day}_summary.json` with metadata
- **Directory structure**: `year/month/files`

## Dependencies

- **Core**: Python 3.12, httpx, tenacity, polars
- **NLP**: spaCy, gensim, scikit-learn  
- **UI**: Streamlit
- **Dev**: black, ruff, pytest

## Development Roadmap

- [x] Web crawler with rate limiting and error handling
- [x] Data collection for 1864 sample period
- [ ] HTML parser for text extraction
- [ ] Topic modeling pipeline
- [ ] Named entity recognition
- [ ] Timeline analysis tools
- [ ] Interactive Streamlit dashboard
- [ ] Search and filtering capabilities
- [ ] Export functionality

## API Structure

The Historic Hansard API follows a hierarchy:
```
/sittings/1860s → /sittings/1864 → /sittings/1864/feb → /sittings/1864/feb/15
```

## Contributing

1. Set up development environment with `conda env create -f environment.yml`
2. Run tests with `pytest`
3. Format code with `black` and `ruff`
4. Follow existing patterns in crawler.py for new modules
