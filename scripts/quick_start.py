import os
import subprocess
import sys

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'selenium', 'beautifulsoup4', 
        'nltk', 'sentence-transformers', 'faiss-cpu', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_nltk_data():
    """Download required NLTK data"""
    import nltk
    
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

def download_spacy_model():
    """Download spaCy model"""
    print("Downloading spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def main():
    """Quick start setup"""
    print("🚀 Quora NLP Project - Quick Start")
    print("=" * 40)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {missing}")
        response = input("Install missing packages? (y/n): ").lower()
        if response == 'y':
            install_dependencies()
        else:
            print("Please install dependencies manually: pip install -r requirements.txt")
            return
    
    # Download NLTK data
    try:
        download_nltk_data()
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
    
    # Download spaCy model
    try:
        download_spacy_model()
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")
        print("You can install it manually: python -m spacy download en_core_web_sm")
    
    # Setup project structure
    from scripts.setup_project import setup_project_directories
    setup_project_directories()
    
    # Initialize database
    from scripts.setup_database import main as setup_db
    setup_db()
    
    print("\n✅ Quick start completed!")
    print("\nNext steps:")
    print("1. Run demo scraping: python main.py --mode pipeline")
    print("2. Start Streamlit UI: streamlit run ui/streamlit_app.py")
    print("3. Interactive search: python scripts/demo.py")

if __name__ == "__main__":
    main()