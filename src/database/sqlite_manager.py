import sqlite3
import hashlib
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
import logging
from .schema import DatabaseSchema

class SQLiteManager:
    """Manages SQLite database operations for Quora NLP project"""
    
    def __init__(self, db_path: str = "data/databases/quora_data.db"):
        self.db_path = db_path
        self.setup_database()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize database with schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Create tables
            for sql in DatabaseSchema.get_create_tables_sql():
                conn.execute(sql)
            
            # Create indexes
            for sql in DatabaseSchema.get_indexes_sql():
                conn.execute(sql)
            
            conn.commit()
    
    def _generate_hash(self, text: str) -> str:
        """Generate MD5 hash for text deduplication"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def insert_question(self, question_text: str, topic_category: str = None) -> int:
        """Insert question and return question_id"""
        question_hash = self._generate_hash(question_text)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if question already exists
            cursor.execute(
                "SELECT id FROM questions WHERE question_hash = ?", 
                (question_hash,)
            )
            existing = cursor.fetchone()
            
            if existing:
                return existing[0]
            
            # Insert new question
            cursor.execute("""
                INSERT INTO questions (question_text, question_hash, topic_category)
                VALUES (?, ?, ?)
            """, (question_text, question_hash, topic_category))
            
            return cursor.lastrowid
    
    def insert_answer(self, question_id: int, answer_text: str) -> Optional[int]:
        """Insert answer for a question and return answer_id"""
        answer_hash = self._generate_hash(answer_text)
        answer_length = len(answer_text)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if this exact answer already exists for this question
            cursor.execute("""
                SELECT id FROM answers 
                WHERE question_id = ? AND answer_hash = ?
            """, (question_id, answer_hash))
            
            if cursor.fetchone():
                self.logger.info(f"Duplicate answer skipped for question_id {question_id}")
                return None
            
            # Insert new answer
            cursor.execute("""
                INSERT INTO answers (question_id, answer_text, answer_hash, answer_length)
                VALUES (?, ?, ?, ?)
            """, (question_id, answer_text, answer_hash, answer_length))
            
            answer_id = cursor.lastrowid
            
            # Update question's total_answers count
            cursor.execute("""
                UPDATE questions 
                SET total_answers = (
                    SELECT COUNT(*) FROM answers WHERE question_id = ?
                )
                WHERE id = ?
            """, (question_id, question_id))
            
            return answer_id
    
    def insert_question_embedding(self, question_id: int, embedding: np.ndarray, 
                                 model_name: str) -> bool:
        """Insert question embedding"""
        embedding_blob = pickle.dumps(embedding)
        embedding_dim = len(embedding)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO question_embeddings 
                    (question_id, embedding_vector, model_name, embedding_dim)
                    VALUES (?, ?, ?, ?)
                """, (question_id, embedding_blob, model_name, embedding_dim))
                
                # Mark question as having embedding generated
                cursor.execute("""
                    UPDATE questions SET embedding_generated = TRUE WHERE id = ?
                """, (question_id,))
                
                return True
            except Exception as e:
                self.logger.error(f"Error inserting question embedding: {e}")
                return False
    
    def insert_answer_embedding(self, answer_id: int, embedding: np.ndarray, 
                               model_name: str) -> bool:
        """Insert answer embedding"""
        embedding_blob = pickle.dumps(embedding)
        embedding_dim = len(embedding)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO answer_embeddings 
                    (answer_id, embedding_vector, model_name, embedding_dim)
                    VALUES (?, ?, ?, ?)
                """, (answer_id, embedding_blob, model_name, embedding_dim))
                
                # Mark answer as having embedding generated
                cursor.execute("""
                    UPDATE answers SET embedding_generated = TRUE WHERE id = ?
                """, (answer_id,))
                
                return True
            except Exception as e:
                self.logger.error(f"Error inserting answer embedding: {e}")
                return False
    
    def get_questions_without_embeddings(self, model_name: str) -> List[Tuple[int, str]]:
        """Get questions that don't have embeddings for the specified model"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT q.id, q.question_text 
                FROM questions q
                LEFT JOIN question_embeddings qe 
                ON q.id = qe.question_id AND qe.model_name = ?
                WHERE qe.id IS NULL
            """, (model_name,))
            return cursor.fetchall()
    
    def get_answers_without_embeddings(self, model_name: str) -> List[Tuple[int, str]]:
        """Get answers that don't have embeddings for the specified model"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.id, a.answer_text 
                FROM answers a
                LEFT JOIN answer_embeddings ae 
                ON a.id = ae.answer_id AND ae.model_name = ?
                WHERE ae.id IS NULL
            """, (model_name,))
            return cursor.fetchall()
    
    def get_question_with_answers(self, question_id: int) -> Dict:
        """Get question and all its answers"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get question
            cursor.execute("""
                SELECT id, question_text, topic_category, total_answers, scraped_at
                FROM questions WHERE id = ?
            """, (question_id,))
            question_row = cursor.fetchone()
            
            if not question_row:
                return None
            
            # Get answers
            cursor.execute("""
                SELECT id, answer_text, answer_length, estimated_quality_score, 
                       sentiment_score, scraped_at
                FROM answers WHERE question_id = ?
                ORDER BY estimated_quality_score DESC
            """, (question_id,))
            answers = cursor.fetchall()
            
            return {
                "question": {
                    "id": question_row[0],
                    "text": question_row[1],
                    "topic_category": question_row[2],
                    "total_answers": question_row[3],
                    "scraped_at": question_row[4]
                },
                "answers": [
                    {
                        "id": ans[0],
                        "text": ans[1],
                        "length": ans[2],
                        "quality_score": ans[3],
                        "sentiment_score": ans[4],
                        "scraped_at": ans[5]
                    } for ans in answers
                ]
            }
    
    def get_all_question_embeddings(self, model_name: str) -> List[Tuple[int, np.ndarray]]:
        """Get all question embeddings for FAISS index building"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT question_id, embedding_vector 
                FROM question_embeddings 
                WHERE model_name = ?
            """, (model_name,))
            
            results = []
            for question_id, embedding_blob in cursor.fetchall():
                embedding = pickle.loads(embedding_blob)
                results.append((question_id, embedding))
            
            return results
    
    def get_all_answer_embeddings(self, model_name: str) -> List[Tuple[int, np.ndarray]]:
        """Get all answer embeddings for FAISS index building"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT answer_id, embedding_vector 
                FROM answer_embeddings 
                WHERE model_name = ?
            """, (model_name,))
            
            results = []
            for answer_id, embedding_blob in cursor.fetchall():
                embedding = pickle.loads(embedding_blob)
                results.append((answer_id, embedding))
            
            return results
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Questions count
            cursor.execute("SELECT COUNT(*) FROM questions")
            stats['total_questions'] = cursor.fetchone()[0]
            
            # Answers count
            cursor.execute("SELECT COUNT(*) FROM answers")
            stats['total_answers'] = cursor.fetchone()[0]
            
            # Embeddings count
            cursor.execute("SELECT COUNT(*) FROM question_embeddings")
            stats['question_embeddings'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM answer_embeddings")
            stats['answer_embeddings'] = cursor.fetchone()[0]
            
            # Average answers per question
            cursor.execute("""
                SELECT AVG(total_answers) FROM questions WHERE total_answers > 0
            """)
            result = cursor.fetchone()[0]
            stats['avg_answers_per_question'] = round(result, 2) if result else 0
            
            return stats
    
    def search_questions_by_text(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Simple text search for questions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, question_text, total_answers, scraped_at
                FROM questions 
                WHERE question_text LIKE ?
                ORDER BY total_answers DESC
                LIMIT ?
            """, (f"%{search_term}%", limit))
            
            return [
                {
                    "id": row[0],
                    "text": row[1],
                    "total_answers": row[2],
                    "scraped_at": row[3]
                } for row in cursor.fetchall()
            ]