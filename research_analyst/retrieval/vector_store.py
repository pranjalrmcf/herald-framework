"""
Vector store for the research analyst system.
Handles embedding generation and semantic search using FAISS.
"""

import numpy as np
from typing import List, Optional, Tuple
import pickle
from pathlib import Path

from research_analyst.core.models import DocumentChunk
from research_analyst.core.exceptions import EmbeddingError
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id


logger = get_logger()


class VectorStore:
    """Vector storage and semantic search using FAISS."""
    
    def __init__(self):
        """Initialize vector store."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Initialize FAISS index
        self.index = None
        self.chunk_map = {}  # Maps index ID to DocumentChunk
        self.embedding_dim = None
    
    def _init_embedding_model(self):
        """Initialize embedding model (sentence-transformers only)."""
        try:
            from sentence_transformers import SentenceTransformer
            # Use the model from settings
            model_name = self.settings.embedding_model
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_method = "sentence_transformers"
            self.logger.info(f"Initialized sentence-transformers: {model_name}")
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers not installed",
                details={"required": "pip install sentence-transformers"}
            )
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using sentence-transformers.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embedding: {str(e)}",
                details={"text_length": len(text)}
            )
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for multiple chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of chunks with embeddings added
        """
        self.logger.info(
            "Generating embeddings",
            num_chunks=len(chunks),
            method=self.embedding_method
        )
        
        # Batch embed for efficiency
        texts = [chunk.text for chunk in chunks]
        
        try:
            # Sentence transformers supports batching
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings = [emb.astype(np.float32) for emb in embeddings]
            
            # Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()
            
            self.logger.info(
                "Embeddings generated",
                num_embeddings=len(embeddings),
                embedding_dim=len(embeddings[0]) if embeddings else 0
            )
            
            return chunks
            
        except Exception as e:
            self.logger.error(
                "Batch embedding failed",
                error=str(e)
            )
            raise EmbeddingError(
                f"Batch embedding failed: {str(e)}",
                details={"num_texts": len(texts)}
            )
    
    def build_index(self, chunks: List[DocumentChunk]):
        """
        Build FAISS index from chunks.
        
        Args:
            chunks: List of chunks with embeddings
        """
        try:
            import faiss
        except ImportError:
            raise EmbeddingError(
                "FAISS not installed",
                details={"required": "pip install faiss-cpu"}
            )
        
        if not chunks:
            self.logger.warning("No chunks to index")
            return
        
        # Ensure chunks have embeddings
        chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
        
        if not chunks_with_embeddings:
            self.logger.warning("No chunks with embeddings")
            return
        
        self.logger.info(
            "Building FAISS index",
            num_chunks=len(chunks_with_embeddings)
        )
        
        # Convert embeddings to matrix
        embeddings = np.array(
            [chunk.embedding for chunk in chunks_with_embeddings],
            dtype=np.float32
        )
        
        # Set embedding dimension
        self.embedding_dim = embeddings.shape[1]
        
        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Create chunk map
        self.chunk_map = {i: chunk for i, chunk in enumerate(chunks_with_embeddings)}
        
        self.logger.info(
            "FAISS index built",
            num_vectors=self.index.ntotal,
            dimension=self.embedding_dim
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            self.logger.warning("Index is empty or not built")
            return []
        
        # Generate query embedding
        query_embedding = self.embed_text(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            chunk = self.chunk_map.get(idx)
            if chunk:
                # Convert L2 distance to similarity score (inverse distance)
                # Normalize to 0-1 range
                similarity = 1.0 / (1.0 + dist)
                results.append((chunk, similarity))
        
        self.logger.info(
            "Vector search completed",
            query_length=len(query),
            num_results=len(results)
        )
        
        return results
    
    def save_index(self, filepath: str):
        """
        Save FAISS index and chunk map to disk.
        
        Args:
            filepath: Path to save index
        """
        try:
            import faiss
        except ImportError:
            raise EmbeddingError("FAISS not installed")
        
        if self.index is None:
            self.logger.warning("No index to save")
            return
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = str(path.with_suffix('.faiss'))
        faiss.write_index(self.index, index_path)
        
        # Save chunk map
        map_path = str(path.with_suffix('.pkl'))
        with open(map_path, 'wb') as f:
            pickle.dump(self.chunk_map, f)
        
        self.logger.info(
            "Index saved",
            index_path=index_path,
            map_path=map_path
        )
    
    def load_index(self, filepath: str):
        """
        Load FAISS index and chunk map from disk.
        
        Args:
            filepath: Path to load index from
        """
        try:
            import faiss
        except ImportError:
            raise EmbeddingError("FAISS not installed")
        
        path = Path(filepath)
        index_path = str(path.with_suffix('.faiss'))
        map_path = str(path.with_suffix('.pkl'))
        
        if not Path(index_path).exists() or not Path(map_path).exists():
            raise EmbeddingError(
                f"Index files not found at {filepath}",
                details={"index_path": index_path, "map_path": map_path}
            )
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        self.embedding_dim = self.index.d
        
        # Load chunk map
        with open(map_path, 'rb') as f:
            self.chunk_map = pickle.load(f)
        
        self.logger.info(
            "Index loaded",
            num_vectors=self.index.ntotal,
            num_chunks=len(self.chunk_map)
        )