import os
import sys
import yaml
import logging
from datetime import datetime
import sqlite3

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.database.sqlite_manager import SQLiteManager
from src.scraper.quora_scraper import EnhancedQuoraScraper
from src.preprocessing.nlp_pipeline import NLPPreprocessingPipeline
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.faiss_manager import FAISSManager

class DataPipeline:
    """Complete data processing pipeline for Quora NLP project"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.db_manager = SQLiteManager(self.config['database']['sqlite_path'])
        self.scraper = None
        self.preprocessor = NLPPreprocessingPipeline(self.config['database']['sqlite_path'])
        self.embedding_generator = EmbeddingGenerator(self.config['embeddings']['model_name'])
        self.faiss_manager = FAISSManager()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default config if file doesn't exist
            return {
                'database': {'sqlite_path': 'data/databases/quora_data.db'},
                'embeddings': {'model_name': 'all-MiniLM-L6-v2', 'batch_size': 32},
                'scraping': {'delay_between_queries': 6, 'headless_mode': False},
                'search': {'default_top_k': 5},
                'logging': {'level': 'INFO', 'log_dir': 'logs/'}
            }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config['logging']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        log_level = getattr(logging, self.config['logging']['level'])
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        )
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=[file_handler, console_handler]
        )
    
    def run_scraping_phase(self, queries: list[str], topic_category: str = None) -> dict:
        """Phase 1: Scrape data from Quora"""
        self.logger.info("=" * 50)
        self.logger.info("STARTING SCRAPING PHASE")
        self.logger.info("=" * 50)
        
        self.scraper = EnhancedQuoraScraper(self.config['database']['sqlite_path'])
        self.scraper.setup_driver(headless=self.config['scraping']['headless_mode'])
        
        try:
            results = self.scraper.scrape_multiple_queries(
                queries=queries,
                topic_category=topic_category,
                delay_between_queries=self.config['scraping']['delay_between_queries']
            )
            
            self.logger.info(f"Scraping completed: {results}")
            return results
            
        finally:
            self.scraper.close()
    
    def run_preprocessing_phase(self) -> dict:
        """Phase 2: Preprocess scraped text data"""
        self.logger.info("=" * 50)
        self.logger.info("STARTING PREPROCESSING PHASE")
        self.logger.info("=" * 50)
        
        results = self.preprocessor.run_full_preprocessing()
        self.logger.info(f"Preprocessing completed: {results}")
        return results
    
    def run_embedding_phase(self) -> dict:
        """Phase 3: Generate embeddings for questions and answers"""
        self.logger.info("=" * 50)
        self.logger.info("STARTING EMBEDDING GENERATION PHASE")
        self.logger.info("=" * 50)
        
        model_name = self.config['embeddings']['model_name']
        batch_size = self.config['embeddings']['batch_size']
        
        results = {'questions_embedded': 0, 'answers_embedded': 0}
        
        # Generate question embeddings
        self.logger.info("Generating question embeddings...")
        question_data = self.preprocessor.get_processed_questions_for_embeddings()
        
        if question_data:
            question_ids, question_texts = zip(*question_data)
            question_embeddings = self.embedding_generator.generate_batch_embeddings(
                list(question_texts), batch_size
            )
            
            # Save embeddings to database
            for question_id, embedding in zip(question_ids, question_embeddings):
                self.db_manager.insert_question_embedding(question_id, embedding, model_name)
                results['questions_embedded'] += 1
        
        # Generate answer embeddings
        self.logger.info("Generating answer embeddings...")
        answer_data = self.preprocessor.get_processed_answers_for_embeddings()
        
        if answer_data:
            answer_ids, answer_texts = zip(*answer_data)
            answer_embeddings = self.embedding_generator.generate_batch_embeddings(
                list(answer_texts), batch_size
            )
            
            # Save embeddings to database
            for answer_id, embedding in zip(answer_ids, answer_embeddings):
                self.db_manager.insert_answer_embedding(answer_id, embedding, model_name)
                results['answers_embedded'] += 1
        
        self.logger.info(f"Embedding generation completed: {results}")
        return results
    
    def run_index_building_phase(self) -> dict:
        """Phase 4: Build FAISS indexes"""
        self.logger.info("=" * 50)
        self.logger.info("STARTING INDEX BUILDING PHASE")
        self.logger.info("=" * 50)
        
        model_name = self.config['embeddings']['model_name']
        index_type = self.config['embeddings']['index_type']
        
        # Get embeddings from database
        question_embeddings = self.db_manager.get_all_question_embeddings(model_name)
        answer_embeddings = self.db_manager.get_all_answer_embeddings(model_name)
        
        results = {}
        
        # Build question index
        if question_embeddings:
            success = self.faiss_manager.build_question_index(question_embeddings, index_type)
            results['question_index_built'] = success
            results['question_index_size'] = len(question_embeddings)
        
        # Build answer index
        if answer_embeddings:
            success = self.faiss_manager.build_answer_index(answer_embeddings, index_type)
            results['answer_index_built'] = success
            results['answer_index_size'] = len(answer_embeddings)
        
        # Save indexes
        self.faiss_manager.save_indexes()
        
        self.logger.info(f"Index building completed: {results}")
        return results
    
    def run_complete_pipeline(self, queries: list[str], topic_category: str = None) -> dict:
        """Run the complete data pipeline"""
        self.logger.info("STARTING COMPLETE DATA PIPELINE")
        self.logger.info(f"Processing {len(queries)} queries")
        
        pipeline_results = {}
        
        try:
            # Phase 1: Scraping
            scraping_results = self.run_scraping_phase(queries, topic_category)
            pipeline_results['scraping'] = scraping_results
            
            # Phase 2: Preprocessing
            preprocessing_results = self.run_preprocessing_phase()
            pipeline_results['preprocessing'] = preprocessing_results
            
            # Phase 3: Embedding Generation
            embedding_results = self.run_embedding_phase()
            pipeline_results['embeddings'] = embedding_results
            
            # Phase 4: Index Building
            index_results = self.run_index_building_phase()
            pipeline_results['indexing'] = index_results
            
            # Final statistics
            final_stats = self.db_manager.get_database_stats()
            pipeline_results['final_stats'] = final_stats
            
            self.logger.info("=" * 50)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 50)
            self.logger.info(f"Final Stats: {final_stats}")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def get_pipeline_status(self) -> dict:
        """Get current status of the pipeline"""
        stats = self.db_manager.get_database_stats()
        
        # Check if indexes exist
        index_stats = self.faiss_manager.get_index_stats()
        
        return {
            'database_stats': stats,
            'index_stats': index_stats,
            'embeddings_ready': stats.get('question_embeddings', 0) > 0,
            'indexes_ready': len(index_stats) > 0
        }