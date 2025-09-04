import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.embeddings.similarity_search import SemanticSearchEngine
from src.database.sqlite_manager import SQLiteManager

def main():
    print("🔍 Quora NLP Analyzer - Demo")
    print("=" * 40)
    
    # Check if system is ready
    db_manager = SQLiteManager()
    stats = db_manager.get_database_stats()
    
    print(f"Database Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if stats['total_questions'] == 0:
        print("\n❌ No data found in database!")
        print("Please run the scraping pipeline first:")
        print("  python main.py --mode pipeline")
        return
    
    if stats['question_embeddings'] == 0:
        print("\n⚠️ No embeddings found!")
        print("Please generate embeddings first:")
        print("  python main.py --mode pipeline")
        return
    
    # Interactive search
    print("\n" + "=" * 40)
    print("Starting interactive search...")
    print("(Type 'quit' to exit)")
    print("=" * 40)
    
    search_engine = SemanticSearchEngine()
    
    while True:
        query = input("\nEnter your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print(f"\n🔍 Searching for: {query}")
        print("-" * 40)
        
        # Find similar questions
        similar_questions = search_engine.find_similar_questions(query, top_k=3)
        
        if similar_questions:
            print("\n📋 Similar Questions:")
            for i, q in enumerate(similar_questions, 1):
                print(f"{i}. {q['question_text']}")
                print(f"   Similarity: {q['similarity_score']:.3f} | Answers: {q['total_answers']}")
        
        # Find best answers
        best_answers = search_engine.find_best_answers_for_question(query, top_k=2)
        
        if best_answers:
            print("\n💡 Best Answers:")
            for i, ans in enumerate(best_answers, 1):
                print(f"\n{i}. {ans['answer_text'][:200]}...")
                print(f"   Quality: {ans['quality_score']:.3f} | Length: {ans['answer_length']} words")
        
        print("\n" + "-" * 40)

if __name__ == "__main__":
    main()