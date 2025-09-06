import numpy as np
import sqlite3
from typing import List, Dict, Tuple, Optional
import logging
from .embedding_generator import EmbeddingGenerator
from .faiss_manager import FAISSManager
from ..preprocessing.text_cleaner import TextCleaner

class SemanticSearchEngine:
    """Main interface for semantic search operations"""
    
    def __init__(self, db_path: str = "data/databases/quora_data.db",
                 model_name: str = 'all-MiniLM-L6-v2'):
        self.db_path = db_path
        self.embedding_generator = EmbeddingGenerator(model_name)
        self.faiss_manager = FAISSManager()
        self.text_cleaner = TextCleaner()
        self.logger = logging.getLogger(__name__)
        
        # Try to load existing indexes
        self.faiss_manager.load_indexes()
    
    def find_similar_questions(self, query_text: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict]:
        """Find questions similar to the query"""
        try:
            # Preprocess query
            cleaned_query = self.text_cleaner.preprocess_for_embeddings(query_text)
            
            if not cleaned_query or len(cleaned_query.strip()) < 3:
                self.logger.warning(f"Query too short or empty after preprocessing: '{query_text}' -> '{cleaned_query}'")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(cleaned_query)
            
            # Search for similar questions (get more than needed for filtering)
            similar_questions = self.faiss_manager.search_similar_questions(query_embedding, top_k * 2)
            
            # Get question details from database and filter by similarity
            results = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for question_id, similarity_score in similar_questions:
                    # Apply similarity threshold
                    if similarity_score < min_similarity:
                        continue
                        
                    cursor.execute("""
                        SELECT question_text, total_answers, topic_category, scraped_at
                        FROM questions WHERE id = ?
                    """, (question_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        results.append({
                            'question_id': question_id,
                            'question_text': row[0],
                            'total_answers': row[1],
                            'topic_category': row[2],
                            'scraped_at': row[3],
                            'similarity_score': similarity_score
                        })
                        
                        # Stop when we have enough results
                        if len(results) >= top_k:
                            break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding similar questions: {e}")
            return []
    
    def find_best_answers_for_question(self, question_text: str, top_k: int = 3, min_quality: float = 0.1) -> List[Dict]:
        """Find best answers for a given question"""
        try:
            # First, find the exact question or most similar question
            similar_questions = self.find_similar_questions(question_text, top_k=1, min_similarity=0.2)
            
            if not similar_questions:
                self.logger.warning("No similar questions found")
                return []
            
            best_match_question_id = similar_questions[0]['question_id']
            question_similarity = similar_questions[0]['similarity_score']
            
            # Get answers for this question with quality filtering
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, answer_text, estimated_quality_score, answer_length, sentiment_score
                    FROM answers 
                    WHERE question_id = ? 
                    AND estimated_quality_score >= ?
                    AND answer_length > 50
                    ORDER BY estimated_quality_score DESC, answer_length DESC
                    LIMIT ?
                """, (best_match_question_id, min_quality, top_k * 2))
                
                results = []
                for row in cursor.fetchall():
                    answer_text = row[1]
                    
                    # Additional quality checks
                    if self._is_quality_answer(answer_text):
                        results.append({
                            'answer_id': row[0],
                            'answer_text': answer_text,
                            'quality_score': row[2],
                            'answer_length': row[3],
                            'sentiment_score': row[4],
                            'question_id': best_match_question_id,
                            'question_match_score': question_similarity
                        })
                        
                        if len(results) >= top_k:
                            break
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error finding best answers: {e}")
            return []
    
    def _is_quality_answer(self, answer_text: str) -> bool:
        """Check if answer meets quality criteria"""
        if not answer_text or len(answer_text.strip()) < 50:
            return False
        
        # Check for navigation text
        nav_indicators = ['upvote', 'share', 'comment', 'follow', 'more answers', 'see more', 'continue reading']
        if any(indicator in answer_text.lower() for indicator in nav_indicators):
            return False
        
        # Check for meaningful content
        meaningful_indicators = ['.', '!', '?', 'because', 'however', 'therefore', 'example', 'for instance']
        if not any(indicator in answer_text for indicator in meaningful_indicators):
            return False
        
        return True
    
    def semantic_answer_search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search for answers semantically similar to query"""
        try:
            # Preprocess query
            cleaned_query = self.text_cleaner.preprocess_for_embeddings(query_text)
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(cleaned_query)
            
            # Search for similar answers
            similar_answers = self.faiss_manager.search_similar_answers(query_embedding, top_k)
            
            # Get answer details with corresponding questions
            results = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for answer_id, similarity_score in similar_answers:
                    cursor.execute("""
                        SELECT a.answer_text, a.estimated_quality_score, a.answer_length,
                               q.id as question_id, q.question_text
                        FROM answers a
                        JOIN questions q ON a.question_id = q.id
                        WHERE a.id = ?
                    """, (answer_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        results.append({
                            'answer_id': answer_id,
                            'answer_text': row[0],
                            'quality_score': row[1],
                            'answer_length': row[2],
                            'question_id': row[3],
                            'question_text': row[4],
                            'similarity_score': similarity_score
                        })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in semantic answer search: {e}")
            return []
    
    def rebuild_indexes_from_database(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Rebuild FAISS indexes from database embeddings"""
        try:
            # Get embeddings from database
            question_embeddings = self._get_question_embeddings_from_db(model_name)
            answer_embeddings = self._get_answer_embeddings_from_db(model_name)
            
            # Build indexes
            if question_embeddings:
                self.faiss_manager.build_question_index(question_embeddings)
            
            if answer_embeddings:
                self.faiss_manager.build_answer_index(answer_embeddings)
            
            # Save indexes
            self.faiss_manager.save_indexes()
            
            self.logger.info("Successfully rebuilt and saved FAISS indexes")
            
        except Exception as e:
            self.logger.error(f"Error rebuilding indexes: {e}")
    
    def _get_question_embeddings_from_db(self, model_name: str) -> List[Tuple[int, np.ndarray]]:
        """Get question embeddings from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT question_id, embedding_vector 
                FROM question_embeddings 
                WHERE model_name = ?
            """, (model_name,))
            
            embeddings = []
            for question_id, embedding_blob in cursor.fetchall():
                embedding = pickle.loads(embedding_blob)
                embeddings.append((question_id, embedding))
            
            return embeddings
    
    def _get_answer_embeddings_from_db(self, model_name: str) -> List[Tuple[int, np.ndarray]]:
        """Get answer embeddings from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT answer_id, embedding_vector 
                FROM answer_embeddings 
                WHERE model_name = ?
            """, (model_name,))
            
            embeddings = []
            for answer_id, embedding_blob in cursor.fetchall():
                embedding = pickle.loads(embedding_blob)
                embeddings.append((answer_id, embedding))
            
            return embeddings