#!/usr/bin/env python3
"""
Script to analyze accuracy issues in the Quora NLP project
"""

import sqlite3
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.embeddings.similarity_search import SemanticSearchEngine

def analyze_database_quality():
    """Analyze the quality of data in the database"""
    print("=" * 60)
    print("DATABASE QUALITY ANALYSIS")
    print("=" * 60)
    
    conn = sqlite3.connect('data/databases/quora_data.db')
    cursor = conn.cursor()
    
    # Basic stats
    cursor.execute('SELECT COUNT(*) FROM questions')
    total_questions = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM answers')
    total_answers = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM question_embeddings')
    question_embeddings = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM answer_embeddings')
    answer_embeddings = cursor.fetchone()[0]
    
    print(f"Total Questions: {total_questions}")
    print(f"Total Answers: {total_answers}")
    print(f"Question Embeddings: {question_embeddings}")
    print(f"Answer Embeddings: {answer_embeddings}")
    print()
    
    # Quality score analysis
    cursor.execute('SELECT AVG(estimated_quality_score), MIN(estimated_quality_score), MAX(estimated_quality_score) FROM answers')
    avg_quality, min_quality, max_quality = cursor.fetchone()
    print(f"Answer Quality Scores:")
    print(f"  Average: {avg_quality:.3f}")
    print(f"  Min: {min_quality:.3f}")
    print(f"  Max: {max_quality:.3f}")
    print()
    
    # Sample questions and answers
    print("Sample Questions:")
    cursor.execute('SELECT question_text, total_answers FROM questions LIMIT 5')
    for i, (question, answers) in enumerate(cursor.fetchall(), 1):
        print(f"{i}. {question} (Answers: {answers})")
    print()
    
    print("Sample High Quality Answers:")
    cursor.execute('SELECT answer_text, estimated_quality_score FROM answers WHERE estimated_quality_score > 0.5 ORDER BY estimated_quality_score DESC LIMIT 3')
    for i, (answer, score) in enumerate(cursor.fetchall(), 1):
        print(f"{i}. {answer[:150]}... (Score: {score:.3f})")
    print()
    
    print("Sample Low Quality Answers:")
    cursor.execute('SELECT answer_text, estimated_quality_score FROM answers WHERE estimated_quality_score < 0.3 ORDER BY estimated_quality_score ASC LIMIT 3')
    for i, (answer, score) in enumerate(cursor.fetchall(), 1):
        print(f"{i}. {answer[:150]}... (Score: {score:.3f})")
    print()
    
    conn.close()

def test_search_accuracy():
    """Test search accuracy with sample queries"""
    print("=" * 60)
    print("SEARCH ACCURACY TEST")
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
            print("-" * 40)
            
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
        print(f"Error testing search: {e}")

def analyze_embedding_quality():
    """Analyze the quality of embeddings"""
    print("=" * 60)
    print("EMBEDDING QUALITY ANALYSIS")
    print("=" * 60)
    
    try:
        search_engine = SemanticSearchEngine()
        
        # Test with similar queries
        test_cases = [
            ("machine learning", "artificial intelligence"),
            ("python programming", "python coding"),
            ("career advice", "job guidance"),
            ("data science", "data analysis")
        ]
        
        for query1, query2 in test_cases:
            print(f"\nTesting similarity between: '{query1}' and '{query2}'")
            
            # Get embeddings for both queries
            cleaned1 = search_engine.text_cleaner.preprocess_for_embeddings(query1)
            cleaned2 = search_engine.text_cleaner.preprocess_for_embeddings(query2)
            
            print(f"  Cleaned query 1: '{cleaned1}'")
            print(f"  Cleaned query 2: '{cleaned2}'")
            
            # Generate embeddings
            emb1 = search_engine.embedding_generator.generate_single_embedding(cleaned1)
            emb2 = search_engine.embedding_generator.generate_single_embedding(cleaned2)
            
            # Calculate cosine similarity
            import numpy as np
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"  Cosine similarity: {similarity:.3f}")
            
    except Exception as e:
        print(f"Error analyzing embeddings: {e}")

def main():
    """Main analysis function"""
    print("QUORA NLP PROJECT - ACCURACY ANALYSIS")
    print("=" * 60)
    
    # Analyze database quality
    analyze_database_quality()
    
    # Test search accuracy
    test_search_accuracy()
    
    # Analyze embedding quality
    analyze_embedding_quality()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
