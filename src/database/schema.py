import sqlite3
from datetime import datetime

class DatabaseSchema:
    """Database schema definitions for Quora NLP project"""
    
    @staticmethod
    def get_create_tables_sql():
        """Returns SQL statements to create all necessary tables"""
        return [
            # Questions table
            """
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_text TEXT NOT NULL UNIQUE,
                question_hash TEXT NOT NULL UNIQUE,
                topic_category TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_answers INTEGER DEFAULT 0,
                embedding_generated BOOLEAN DEFAULT FALSE
            )
            """,
            
            # Answers table
            """
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                answer_text TEXT NOT NULL,
                answer_hash TEXT NOT NULL,
                answer_length INTEGER,
                estimated_quality_score REAL DEFAULT 0.0,
                sentiment_score REAL,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding_generated BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (question_id) REFERENCES questions (id),
                UNIQUE(question_id, answer_hash)
            )
            """,
            
            # Question embeddings table
            """
            CREATE TABLE IF NOT EXISTS question_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                embedding_vector BLOB NOT NULL,
                model_name TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES questions (id),
                UNIQUE(question_id, model_name)
            )
            """,
            
            # Answer embeddings table
            """
            CREATE TABLE IF NOT EXISTS answer_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                answer_id INTEGER NOT NULL,
                embedding_vector BLOB NOT NULL,
                model_name TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (answer_id) REFERENCES answers (id),
                UNIQUE(answer_id, model_name)
            )
            """,
            
            # Similar questions mapping table
            """
            CREATE TABLE IF NOT EXISTS similar_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id_1 INTEGER NOT NULL,
                question_id_2 INTEGER NOT NULL,
                similarity_score REAL NOT NULL,
                similarity_method TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id_1) REFERENCES questions (id),
                FOREIGN KEY (question_id_2) REFERENCES questions (id),
                UNIQUE(question_id_1, question_id_2, similarity_method)
            )
            """,
            
            # Answer rankings table
            """
            CREATE TABLE IF NOT EXISTS answer_rankings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                answer_id INTEGER NOT NULL,
                rank_position INTEGER NOT NULL,
                ranking_score REAL NOT NULL,
                ranking_method TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES questions (id),
                FOREIGN KEY (answer_id) REFERENCES answers (id),
                UNIQUE(question_id, answer_id, ranking_method)
            )
            """
        ]
    
    @staticmethod
    def get_indexes_sql():
        """Returns SQL statements to create indexes for better performance"""
        return [
            "CREATE INDEX IF NOT EXISTS idx_questions_hash ON questions(question_hash)",
            "CREATE INDEX IF NOT EXISTS idx_answers_question_id ON answers(question_id)",
            "CREATE INDEX IF NOT EXISTS idx_answers_hash ON answers(answer_hash)",
            "CREATE INDEX IF NOT EXISTS idx_question_embeddings_question_id ON question_embeddings(question_id)",
            "CREATE INDEX IF NOT EXISTS idx_answer_embeddings_answer_id ON answer_embeddings(answer_id)",
            "CREATE INDEX IF NOT EXISTS idx_similar_questions_q1 ON similar_questions(question_id_1)",
            "CREATE INDEX IF NOT EXISTS idx_similar_questions_q2 ON similar_questions(question_id_2)",
            "CREATE INDEX IF NOT EXISTS idx_answer_rankings_question_id ON answer_rankings(question_id)"
        ]