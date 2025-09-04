import os
import sqlite3
import logging

def setup_project_directories():
    """Create all necessary project directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/processed/embeddings",
        "data/databases",
        "data/raw/backup",
        "logs",
        "models/saved_models",
        "models/pretrained"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """
# Core dependencies
pandas>=1.5.0
numpy>=1.21.0
sqlite3

# Web scraping
selenium>=4.0.0
beautifulsoup4>=4.11.0
webdriver-manager>=3.8.0

# NLP and ML
nltk>=3.8
spacy>=3.4.0
sentence-transformers>=2.2.0
transformers>=4.20.0
torch>=1.12.0
scikit-learn>=1.1.0

# Vector search
faiss-cpu>=1.7.0  # Use faiss-gpu if you have CUDA

# Configuration and utilities
pyyaml>=6.0
tqdm>=4.64.0

# UI (optional)
streamlit>=1.22.0
flask>=2.2.0

# Development and testing
pytest>=7.0.0
jupyter>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())
    
    print("Created requirements.txt")

def main():
    """Setup the entire project structure"""
    print("Setting up Quora NLP project...")
    
    setup_project_directories()
    create_requirements_file()
    
    print("\nProject setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download spaCy model: python -m spacy download en_core_web_sm")
    print("3. Run the scraping pipeline: python scripts/run_pipeline.py")

if __name__ == "__main__":
    main()