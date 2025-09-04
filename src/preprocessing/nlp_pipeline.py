import sqlite3
import pandas as pd
from typing import List, Dict, Tuple
import logging
from .text_cleaner import TextCleaner

class NLPPreprocessingPipeline:
    """Main preprocessing pipeline for Quora data"""
    
    def __init__(self, db_path: str = "data/databases/quora_data.db"):
        self.db_path = db_path
        self.text_cleaner = TextCleaner()
        self.logger = logging.getLogger(__name__)
    
    def preprocess_all_questions(self) -> int:
        """Preprocess all questions in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all questions
            cursor.execute("SELECT id, question_text FROM questions")
            questions = cursor.fetchall()
            
            processed_count = 0
            for question_id, question_text in questions:
                try:
                    # Clean text
                    cleaned_text = self.text_cleaner.advanced_clean(question_text)
                    
                    # Preprocess for embeddings
                    embedding_ready_text = self.text_cleaner.preprocess_for_embeddings(cleaned_text)
                    
                    # Update database with cleaned text (optional - store in separate column)
                    # For now, we'll keep original and generate embeddings from cleaned version
                    
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error preprocessing question {question_id}: {e}")
            
            self.logger.info(f"Preprocessed {processed_count} questions")
            return processed_count
    
    def preprocess_all_answers(self) -> int:
        """Preprocess all answers in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all answers
            cursor.execute("SELECT id, answer_text FROM answers")
            answers = cursor.fetchall()
            
            processed_count = 0
            for answer_id, answer_text in answers:
                try:
                    # Clean text
                    cleaned_text = self.text_cleaner.advanced_clean(answer_text)
                    
                    # Preprocess for embeddings
                    embedding_ready_text = self.text_cleaner.preprocess_for_embeddings(cleaned_text)
                    
                    # Calculate text statistics
                    word_count = len(cleaned_text.split())
                    sentence_count = len(self.text_cleaner.extract_sentences(cleaned_text))
                    
                    # Update answer with statistics
                    cursor.execute("""
                        UPDATE answers 
                        SET answer_length = ?
                        WHERE id = ?
                    """, (word_count, answer_id))
                    
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error preprocessing answer {answer_id}: {e}")
            
            conn.commit()
            self.logger.info(f"Preprocessed {processed_count} answers")
            return processed_count
    
    def get_processed_questions_for_embeddings(self) -> List[Tuple[int, str]]:
        """Get preprocessed questions ready for embedding generation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, question_text FROM questions")
            
            processed_questions = []
            for question_id, question_text in cursor.fetchall():
                cleaned_text = self.text_cleaner.preprocess_for_embeddings(question_text)
                if cleaned_text:
                    processed_questions.append((question_id, cleaned_text))
            
            return processed_questions
    
    def get_processed_answers_for_embeddings(self) -> List[Tuple[int, str]]:
        """Get preprocessed answers ready for embedding generation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, answer_text FROM answers")
            
            processed_answers = []
            for answer_id, answer_text in cursor.fetchall():
                cleaned_text = self.text_cleaner.preprocess_for_embeddings(answer_text)
                if cleaned_text:
                    processed_answers.append((answer_id, cleaned_text))
            
            return processed_answers
    
    def run_full_preprocessing(self) -> Dict[str, int]:
        """Run complete preprocessing pipeline"""
        self.logger.info("Starting full preprocessing pipeline...")
        
        results = {
            'questions_processed': self.preprocess_all_questions(),
            'answers_processed': self.preprocess_all_answers()
        }
        
        self.logger.info("Preprocessing pipeline completed!")
        return results