import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(__file__))

from scripts.data_pipeline import DataPipeline
from src.embeddings.similarity_search import SemanticSearchEngine

def main():
    parser = argparse.ArgumentParser(description="Quora NLP Project")
    parser.add_argument("--mode", choices=['setup', 'scrape', 'search', 'pipeline'], 
                       required=True, help="Operation mode")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--queries-file", type=str, help="File containing queries to scrape")
    parser.add_argument("--topic", type=str, help="Topic category for scraped data")
    
    args = parser.parse_args()
    
    if args.mode == 'setup':
        # Setup project structure
        from scripts.setup_project import main as setup_main
        setup_main()
        
    elif args.mode == 'pipeline':
        # Run complete pipeline
        queries = []
        if args.queries_file and os.path.exists(args.queries_file):
            with open(args.queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
        else:
            # Use default queries
            queries = [
                "What is machine learning",
                "How to learn Python programming",
                "Best career advice for students"
            ]
        
        pipeline = DataPipeline()
        pipeline.run_complete_pipeline(queries, args.topic)
        
    elif args.mode == 'scrape':
        # Run only scraping
        if not args.query and not args.queries_file:
            print("Please provide either --query or --queries-file")
            return
        
        queries = []
        if args.queries_file:
            with open(args.queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
        elif args.query:
            queries = [args.query]
        
        pipeline = DataPipeline()
        pipeline.run_scraping_phase(queries, args.topic)
        
    elif args.mode == 'search':
        # Interactive search mode
        if not args.query:
            args.query = input("Enter your search query: ")
        
        search_engine = SemanticSearchEngine()
        
        print(f"\nSearching for: {args.query}")
        print("=" * 50)
        
        # Find similar questions
        similar_questions = search_engine.find_similar_questions(args.query, top_k=5)
        print("\nSimilar Questions:")
        for i, q in enumerate(similar_questions, 1):
            print(f"{i}. {q['question_text']}")
            print(f"   Similarity: {q['similarity_score']:.3f}, Answers: {q['total_answers']}")
        
        # Find best answers
        best_answers = search_engine.find_best_answers_for_question(args.query, top_k=3)
        print("\nBest Answers:")
        for i, ans in enumerate(best_answers, 1):
            print(f"{i}. {ans['answer_text'][:200]}...")
            print(f"   Quality Score: {ans['quality_score']:.3f}")

if __name__ == "__main__":
    main()