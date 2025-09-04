import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
import logging

class FAISSManager:
    """Manages FAISS indexes for semantic search"""
    
    def __init__(self, index_dir: str = "data/databases/"):
        self.index_dir = index_dir
        self.question_index = None
        self.answer_index = None
        self.question_id_map = []  # Maps FAISS index to question_id
        self.answer_id_map = []    # Maps FAISS index to answer_id
        self.logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        os.makedirs(index_dir, exist_ok=True)
    
    def build_question_index(self, question_embeddings: List[Tuple[int, np.ndarray]], 
                           index_type: str = 'flat') -> bool:
        """Build FAISS index for questions"""
        try:
            if not question_embeddings:
                self.logger.warning("No question embeddings provided")
                return False
            
            # Extract embeddings and IDs
            question_ids, embeddings = zip(*question_embeddings)
            embeddings_array = np.vstack(embeddings)
            
            # Get embedding dimension
            embedding_dim = embeddings_array.shape[1]
            
            # Create FAISS index
            if index_type == 'flat':
                index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (cosine similarity)
            elif index_type == 'ivf':
                # More efficient for large datasets
                nlist = min(100, len(embeddings_array) // 10)  # Number of clusters
                quantizer = faiss.IndexFlatIP(embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
                index.train(embeddings_array)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add embeddings to index
            index.add(embeddings_array)
            
            self.question_index = index
            self.question_id_map = list(question_ids)
            
            self.logger.info(f"Built question index with {index.ntotal} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Error building question index: {e}")
            return False
    
    def build_answer_index(self, answer_embeddings: List[Tuple[int, np.ndarray]], 
                          index_type: str = 'flat') -> bool:
        """Build FAISS index for answers"""
        try:
            if not answer_embeddings:
                self.logger.warning("No answer embeddings provided")
                return False
            
            # Extract embeddings and IDs
            answer_ids, embeddings = zip(*answer_embeddings)
            embeddings_array = np.vstack(embeddings)
            
            # Get embedding dimension
            embedding_dim = embeddings_array.shape[1]
            
            # Create FAISS index
            if index_type == 'flat':
                index = faiss.IndexFlatIP(embedding_dim)
            elif index_type == 'ivf':
                nlist = min(100, len(embeddings_array) // 10)
                quantizer = faiss.IndexFlatIP(embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
                index.train(embeddings_array)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add embeddings to index
            index.add(embeddings_array)
            
            self.answer_index = index
            self.answer_id_map = list(answer_ids)
            
            self.logger.info(f"Built answer index with {index.ntotal} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Error building answer index: {e}")
            return False
    
    def search_similar_questions(self, query_embedding: np.ndarray, 
                               top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for similar questions"""
        if self.question_index is None:
            raise ValueError("Question index not built")
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.question_index.search(query_embedding, top_k)
        
        # Map back to question IDs
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.question_id_map):  # Valid index
                question_id = self.question_id_map[idx]
                results.append((question_id, float(similarity)))
        
        return results
    
    def search_similar_answers(self, query_embedding: np.ndarray, 
                             top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for similar answers"""
        if self.answer_index is None:
            raise ValueError("Answer index not built")
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.answer_index.search(query_embedding, top_k)
        
        # Map back to answer IDs
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.answer_id_map):  # Valid index
                answer_id = self.answer_id_map[idx]
                results.append((answer_id, float(similarity)))
        
        return results
    
    def save_indexes(self, question_index_path: str = None, answer_index_path: str = None):
        """Save FAISS indexes and ID mappings to disk"""
        if question_index_path is None:
            question_index_path = os.path.join(self.index_dir, "question_index.faiss")
        if answer_index_path is None:
            answer_index_path = os.path.join(self.index_dir, "answer_index.faiss")
        
        try:
            # Save question index
            if self.question_index is not None:
                faiss.write_index(self.question_index, question_index_path)
                
                # Save ID mapping
                with open(question_index_path.replace('.faiss', '_id_map.pkl'), 'wb') as f:
                    pickle.dump(self.question_id_map, f)
                
                self.logger.info(f"Saved question index to {question_index_path}")
            
            # Save answer index
            if self.answer_index is not None:
                faiss.write_index(self.answer_index, answer_index_path)
                
                # Save ID mapping
                with open(answer_index_path.replace('.faiss', '_id_map.pkl'), 'wb') as f:
                    pickle.dump(self.answer_id_map, f)
                
                self.logger.info(f"Saved answer index to {answer_index_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving indexes: {e}")
    
    def load_indexes(self, question_index_path: str = None, answer_index_path: str = None):
        """Load FAISS indexes and ID mappings from disk"""
        if question_index_path is None:
            question_index_path = os.path.join(self.index_dir, "question_index.faiss")
        if answer_index_path is None:
            answer_index_path = os.path.join(self.index_dir, "answer_index.faiss")
        
        try:
            # Load question index
            if os.path.exists(question_index_path):
                self.question_index = faiss.read_index(question_index_path)
                
                # Load ID mapping
                id_map_path = question_index_path.replace('.faiss', '_id_map.pkl')
                if os.path.exists(id_map_path):
                    with open(id_map_path, 'rb') as f:
                        self.question_id_map = pickle.load(f)
                
                self.logger.info(f"Loaded question index from {question_index_path}")
            
            # Load answer index
            if os.path.exists(answer_index_path):
                self.answer_index = faiss.read_index(answer_index_path)
                
                # Load ID mapping
                id_map_path = answer_index_path.replace('.faiss', '_id_map.pkl')
                if os.path.exists(id_map_path):
                    with open(id_map_path, 'rb') as f:
                        self.answer_id_map = pickle.load(f)
                
                self.logger.info(f"Loaded answer index from {answer_index_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading indexes: {e}")
    
    def get_index_stats(self) -> dict:
        """Get statistics about the indexes"""
        stats = {}
        
        if self.question_index:
            stats['question_index_size'] = self.question_index.ntotal
            stats['question_embedding_dim'] = self.question_index.d
        
        if self.answer_index:
            stats['answer_index_size'] = self.answer_index.ntotal
            stats['answer_embedding_dim'] = self.answer_index.d
        
        return stats
