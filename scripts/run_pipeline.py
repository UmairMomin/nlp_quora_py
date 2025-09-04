import sys
import os
import csv
import logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.data_pipeline import DataPipeline

def load_sample_queries_from_csv(csv_path, limit: int):
    """Load sample queries from CSV file"""
    queries = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "question" in row:  # make sure 'question' column exists
                    queries.append(row["question"])
                if len(queries) >= limit:  # only take first N questions
                    break
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
    return queries

def main():
    """Run the complete data pipeline with sample queries"""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "quora_faq_questions.csv")

    # Load queries from CSV
    sample_queries = load_sample_queries_from_csv(csv_path, limit=200)


    # Initialize pipeline
    pipeline = DataPipeline()
    
    try:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            queries=sample_queries,
            topic_category="Mixed"
        )
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        
        for phase, phase_results in results.items():
            print(f"\n{phase.upper()}:")
            if isinstance(phase_results, dict):
                for key, value in phase_results.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {phase_results}")
        
        # Test the search functionality
        print("\n" + "="*60)
        print("TESTING SEARCH FUNCTIONALITY")
        print("="*60)
        
        test_search_pipeline(pipeline)
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        logging.error(f"Pipeline execution failed: {e}")

def test_search_pipeline(pipeline):
    """Test the search functionality"""
    from src.embeddings.similarity_search import SemanticSearchEngine
    
    # Initialize search engine
    search_engine = SemanticSearchEngine(
        db_path=pipeline.config['database']['sqlite_path'],
        model_name=pipeline.config['embeddings']['model_name']
    )
    
    # Test queries
    test_queries = [
        "How to learn programming",
        "What is AI",
        "Career advice for students"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        # Find similar questions
        similar_questions = search_engine.find_similar_questions(query, top_k=3)
        print("Similar Questions:")
        for i, q in enumerate(similar_questions, 1):
            print(f"  {i}. {q['question_text'][:80]}... (Score: {q['similarity_score']:.3f})")
        
        # Find best answers
        best_answers = search_engine.find_best_answers_for_question(query, top_k=2)
        print("\nBest Answers:")
        for i, ans in enumerate(best_answers, 1):
            print(f"  {i}. {ans['answer_text'][:100]}... (Quality: {ans['quality_score']:.3f})")

if __name__ == "__main__":
    main()