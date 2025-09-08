# Quora NLP Analyzer

**Semantic Analysis and Best Answer Retrieval for Quora Questions Using NLP**

A comprehensive system that automates the retrieval and semantic analysis of questions and answers from Quora using advanced NLP techniques.

## 🎯 Project Overview

### Key Features:
- **Semantic Question Matching**: Find semantically similar questions using sentence embeddings
- **Answer Quality Ranking**: Intelligent ranking based on multiple quality indicators
- **Efficient Vector Search**: FAISS-powered semantic search for fast retrieval
- **Comprehensive Pipeline**: End-to-end processing from web scraping to search interface

## 🏗️ Architecture

```
Data Flow: Scraping → Preprocessing → Embeddings → Vector Search → Results
Storage: SQLite (structured data) + FAISS (vector embeddings)
Models: Sentence-BERT for semantic understanding
```

## 📁 Project Structure

```
quora_nlp_project/
├── src/
│   ├── scraper/          # Web scraping modules
│   ├── database/         # SQLite management
│   ├── preprocessing/    # Text cleaning & NLP
│   ├── embeddings/       # Embedding generation & FAISS
│   └── api/             # Query interface
├── scripts/             # Utility scripts
├── ui/                  # Streamlit interface
├── data/               # Data storage
├── notebooks/          # Analysis notebooks
└── logs/               # System logs
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or download the project
cd quora_nlp_project

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Quick setup (automated)
python scripts/quick_start.py
```

### 2. Initialize Database
```bash
python scripts/setup_database.py
```

### 3. Run Complete Pipeline
```bash
# Using default sample queries
python main.py --mode pipeline

# Using custom queries file
python main.py --mode pipeline --queries-file data/sample_queries/technology_queries.txt --topic Technology
```

### 4. Start Web Interface
```bash
streamlit run ui/streamlit_app.py
```

## 📝 Usage Examples

### Command Line Interface

#### 1. Scrape Specific Queries
```bash
python main.py --mode scrape --query "What is machine learning" --topic Technology
```

#### 2. Interactive Search
```bash
python main.py --mode search --query "How to learn programming"
```

#### 3. Demo Mode
```bash
python scripts/demo.py
```

### Python API Usage

```python
from src.embeddings.similarity_search import SemanticSearchEngine

# Initialize search engine
search_engine = SemanticSearchEngine()

# Find similar questions
similar_questions = search_engine.find_similar_questions(
    "How to learn Python programming", 
    top_k=5
)

# Find best answers
best_answers = search_engine.find_best_answers_for_question(
    "What is machine learning", 
    top_k=3
)

# Semantic answer search
semantic_results = search_engine.semantic_answer_search(
    "artificial intelligence career advice",
    top_k=5
)
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
embeddings:
  model_name: "all-MiniLM-L6-v2"  # or "all-mpnet-base-v2" for better accuracy
  batch_size: 32
  index_type: "flat"

scraping:
  delay_between_queries: 6
  headless_mode: false
  max_answers_per_question: 10

search:
  default_top_k: 5
  similarity_threshold: 0.3
```

## 🧠 NLP Techniques Used

### 1. Text Preprocessing
- **Tokenization**: Breaking text into tokens
- **Stopword Removal**: Filtering common words
- **Lemmatization**: Word normalization
- **Text Cleaning**: Removing artifacts and noise

### 2. Semantic Similarity
- **Sentence Embeddings**: Using Sentence-BERT models
- **Cosine Similarity**: Measuring semantic closeness
- **Vector Search**: FAISS for efficient similarity search

### 3. Answer Ranking
- **Quality Scoring**: Multi-factor quality assessment
- **Semantic Matching**: Question-answer relevance
- **Metadata Integration**: Length, credentials, engagement

### 4. Clustering & Analysis
- **Topic Detection**: Grouping similar questions
- **Duplicate Detection**: Finding repeated content
- **Sentiment Analysis**: Answer tone evaluation

## 📊 System Components

### Database Schema
- **Questions**: Core question data and metadata
- **Answers**: Answer content with quality scores
- **Embeddings**: Vector representations for semantic search
- **Similarity Mappings**: Precomputed similar question pairs

### Search Capabilities
- **Exact Question Matching**: Find identical questions
- **Semantic Question Search**: Find similar meaning questions
- **Answer Retrieval**: Get best answers for questions
- **Cross-Answer Search**: Find relevant answers across questions

## 🎯 Use Cases

### 1. Question Answering System
- Input: User question
- Output: Best matching answers with quality scores

### 2. Content Deduplication
- Identify duplicate or very similar questions
- Merge redundant content

### 3. Topic Analysis
- Group questions by semantic similarity
- Analyze trending topics and themes

### 4. Answer Quality Assessment
- Rank answers by estimated quality
- Identify high-quality contributors

## 🔍 Performance Metrics

The system tracks:
- **Similarity Scores**: Cosine similarity for semantic matching
- **Quality Scores**: Multi-factor answer quality assessment
- **Coverage**: Percentage of data with embeddings
- **Search Speed**: Average query response time

## 🛠️ Development

### Adding New Features

1. **New NLP Models**: Add to `src/embeddings/embedding_generator.py`
2. **Custom Ranking**: Extend `src/ranking/answer_ranker.py`
3. **Additional Scrapers**: Add to `src/scraper/`
4. **New Analysis**: Add to `src/analysis/`

### Testing
```bash
# Run tests
python -m pytest tests/

# Test specific module
python -m pytest tests/test_embeddings.py
```

## 📋 Requirements

See `requirements.txt` for complete list. Key dependencies:

- **Python 3.8+**
- **PyTorch** (for embeddings)
- **Sentence-Transformers** (semantic embeddings)
- **FAISS** (vector search)
- **Selenium** (web scraping)
- **NLTK/spaCy** (NLP processing)
- **SQLite** (database)
- **Streamlit** (web interface)

## 🚨 Important Notes

### Ethical Scraping
- Respects robots.txt and rate limits
- Adds delays between requests
- Only scrapes publicly available content

### Model Performance
- **all-MiniLM-L6-v2**: Fast, good performance (384 dim)
- **all-mpnet-base-v2**: Best accuracy (768 dim)
- **paraphrase-MiniLM-L6-v2**: Good for paraphrase detection

### System Requirements
- **RAM**: 4GB+ (8GB+ recommended for large datasets)
- **Storage**: 1GB+ for embeddings and indexes
- **GPU**: Optional but recommended for faster embedding generation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is for educational purposes as part of the NLP course curriculum.

---

**Built with ❤️ for Natural Language Processing**
