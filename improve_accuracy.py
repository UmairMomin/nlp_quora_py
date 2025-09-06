#!/usr/bin/env python3
"""
Script to improve accuracy by regenerating embeddings and testing
"""

import sys
import os
import sqlite3

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from scripts.data_pipeline import DataPipeline
from src.embeddings.similarity_search import SemanticSearchEngine

def regenerate_embeddings():
    """Regenerate embeddings with improved model"""
    print("=" * 60)
    print("REGENERATING EMBEDDINGS WITH IMPROVED MODEL")
    print("=" * 60)
    
    try:
        pipeline = DataPipeline()
        
        # Run embedding generation phase
        print("Generating new embeddings...")
        results = pipeline.run_embedding_phase()
        print(f"Embedding generation results: {results}")
        
        # Build new indexes
        print("Building new FAISS indexes...")
        index_results = pipeline.run_index_building_phase()
        print(f"Index building results: {index_results}")
        
        return True
        
    except Exception as e:
        print(f"Error regenerating embeddings: {e}")
        return False

def test_improved_accuracy():
    """Test the improved accuracy"""
    print("=" * 60)
    print("TESTING IMPROVED ACCURACY")
    print("=" * 60)
    
    try:
        search_engine = SemanticSearchEngine()
        
        test_queries = [
            "What is machine learning?",
            "How to learn Python programming?",
            "Best career advice for students",
            "What are the benefits of artificial intelligence?",
            "How does quantum computing work?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 50)
            
            # Test similar questions
            similar_questions = search_engine.find_similar_questions(query, top_k=3)
            print("Similar Questions:")
            for i, q in enumerate(similar_questions, 1):
                print(f"  {i}. {q['question_text'][:80]}... (Score: {q['similarity_score']:.3f})")
            
            # Test best answers
            best_answers = search_engine.find_best_answers_for_question(query, top_k=2)
            print("Best Answers:")
            for i, ans in enumerate(best_answers, 1):
                print(f"  {i}. {ans['answer_text'][:100]}... (Quality: {ans['quality_score']:.3f})")
            
            print()
            
    except Exception as e:
        print(f"Error testing accuracy: {e}")

def update_existing_quality_scores():
    """Update quality scores for existing answers"""
    print("=" * 60)
    print("UPDATING QUALITY SCORES FOR EXISTING ANSWERS")
    print("=" * 60)
    
    try:
        from src.scraper.quora_scraper import EnhancedQuoraScraper
        
        scraper = EnhancedQuoraScraper()
        
        # Get all answers
        conn = sqlite3.connect('data/databases/quora_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, answer_text FROM answers')
        answers = cursor.fetchall()
        
        updated_count = 0
        for answer_id, answer_text in answers:
            # Create answer data structure
            answer_data = {
                'text': answer_text,
                'quality_indicators': {}  # We don't have original indicators
            }
            
            # Calculate new quality score
            new_score = scraper.estimate_answer_quality(answer_data)
            
            # Update database
            cursor.execute('UPDATE answers SET estimated_quality_score = ? WHERE id = ?', 
                         (new_score, answer_id))
            updated_count += 1
        
        conn.commit()
        conn.close()
        
        print(f"Updated quality scores for {updated_count} answers")
        return True
        
    except Exception as e:
        print(f"Error updating quality scores: {e}")
        return False

def main():
    """Main improvement function"""
    print("QUORA NLP PROJECT - ACCURACY IMPROVEMENT")
    print("=" * 60)
    
    # Step 1: Update quality scores
    print("Step 1: Updating quality scores...")
    update_existing_quality_scores()
    
    # Step 2: Regenerate embeddings
    print("\nStep 2: Regenerating embeddings...")
    if regenerate_embeddings():
        print("✅ Embeddings regenerated successfully")
    else:
        print("❌ Failed to regenerate embeddings")
        return
    
    # Step 3: Test improved accuracy
    print("\nStep 3: Testing improved accuracy...")
    test_improved_accuracy()
    
    print("\n" + "=" * 60)
    print("ACCURACY IMPROVEMENT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
