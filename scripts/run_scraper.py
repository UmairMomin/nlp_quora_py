import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.scraper.quora_scraper import EnhancedQuoraScraper
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraping.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Sample queries for testing
    test_queries = [
        "What is machine learning",
        "How does artificial intelligence work",
        "Best programming languages for beginners",
        "What is data science",
        "How to learn Python programming"
    ]
    
    with EnhancedQuoraScraper() as scraper:
        scraper.setup_driver(headless=False)  # Set to True for headless mode
        
        results = scraper.scrape_multiple_queries(
            queries=test_queries,
            topic_category="Technology",
            delay_between_queries=7
        )
        
        print(f"\\nScraping Results:")
        print(f"Successful: {results['successful_scrapes']}")
        print(f"Failed: {results['failed_scrapes']}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Total Answers: {results['total_answers']}")
        
        # Show database stats
        stats = scraper.db_manager.get_database_stats()
        print(f"\\nDatabase Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()