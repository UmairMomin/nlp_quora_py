import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Tuple, Dict
import logging
import os

class EmbeddingGenerator:
    """Generates embeddings using pre-trained models"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with embedding model
        Popular options:
        - 'all-MiniLM-L6-v2': Fast, good performance, 384 dimensions
        - 'all-mpnet-base-v2': Best performance, 768 dimensions  
        - 'paraphrase-MiniLM-L6-v2': Good for paraphrase detection
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self.logger = logging.getLogger(__name__)
        
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension by encoding a test sentence
            test_embedding = self.model.encode("test")
            self.embedding_dim = len(test_embedding)
            
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            self.logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)  # Convert to float32 for FAISS
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} texts...")
            
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'Unknown')
        }