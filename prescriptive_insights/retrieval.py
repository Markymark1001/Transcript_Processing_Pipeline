"""
Retrieval utilities for Prescriptive Insights

Provides filtering, semantic ranking, and caching for efficient
retrieval of relevant chunks from the chunk index.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from functools import lru_cache
import config
from embedding_generator import EmbeddingGenerator
from .topic_registry import TopicRegistry, Topic

# In-memory cache for query results
_query_cache = {}

def load_chunk_index() -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Load the master chunk index and optional embeddings."""
    master_file = config.INSIGHTS_CHUNK_DIR / "master_chunks.parquet"
    
    if not master_file.exists():
        raise FileNotFoundError(f"Master chunk file not found: {master_file}. "
                              f"Run chunk_builder first to create chunks.")
    
    # Load dataframe
    df = pd.read_parquet(master_file)
    
    # Try to load embeddings if they exist
    embeddings = None
    if "embedding" in df.columns and not df["embedding"].isna().all():
        # Convert embedding lists to numpy array
        embedding_data = df["embedding"].dropna().tolist()
        if embedding_data:
            embeddings = np.array(embedding_data)
            print(f"Loaded {len(embeddings)} embeddings from {len(df)} chunks")
    
    return df, embeddings

def filter_chunks_by_topic(topic_id: str, 
                         df: pd.DataFrame, 
                         keywords: List[str] = None) -> pd.DataFrame:
    """Filter chunks by topic using deterministic keyword/entity matching."""
    # Initialize topic registry
    topic_registry = TopicRegistry()
    
    # Get topic definition
    try:
        topic = topic_registry.get_topic(topic_id)
    except ValueError:
        raise ValueError(f"Unknown topic ID: {topic_id}")
    
    # Use provided keywords or topic keywords
    filter_keywords = keywords or topic.keywords
    
    # Create mask for matching chunks
    mask = pd.Series(False, index=df.index)
    
    # Check matching_topics column first
    if "matching_topics" in df.columns:
        mask = df["matching_topics"].apply(
            lambda topics: topic_id in (topics.split(",") if isinstance(topics, str) else
                                 (topics if isinstance(topics, list) else []))
        )
    else:
        # Fallback to keyword matching
        mask = df["text"].str.lower().apply(
            lambda text: any(keyword.lower() in text for keyword in filter_keywords)
        )
    
    # Apply entity filtering if entities are available
    if topic.entities and "entities" in df.columns:
        entity_mask = df["entities"].apply(
            lambda entities: topic.matches_entities(entities if isinstance(entities, list) else [])
        )
        mask = mask | entity_mask
    
    return df[mask]

def semantic_rank(chunks: pd.DataFrame, 
                query_embeddings: np.ndarray = None,
                query_text: str = None,
                top_k: int = 100) -> List[Dict[str, Any]]:
    """Rank chunks by semantic similarity using cosine similarity."""
    if query_embeddings is None and query_text is None:
        raise ValueError("Either query_embeddings or query_text must be provided")
    
    # Check if chunks have embeddings
    if "embedding" not in chunks.columns:
        # Return chunks without semantic ranking
        return chunks.head(top_k).to_dict('records')
    
    # Get chunk embeddings
    chunk_embeddings_list = chunks["embedding"].dropna().tolist()
    if not chunk_embeddings_list:
        # No embeddings available, return top-k by importance
        return chunks.sort_values(
            "max_importance_score", ascending=False
        ).head(top_k).to_dict('records')
    
    chunk_embeddings = np.array(chunk_embeddings_list)
    
    # Generate query embeddings if needed
    if query_embeddings is None:
        try:
            embedding_generator = EmbeddingGenerator()
            query_emb = embedding_generator.get_transcript_embedding(query_text)
            if query_emb is None:
                # Fallback to importance-based ranking
                return chunks.sort_values(
                    "max_importance_score", ascending=False
                ).head(top_k).to_dict('records')
            query_embeddings = query_emb.reshape(1, -1)
        except Exception as e:
            print(f"Warning: Could not generate query embeddings: {e}")
            # Fallback to importance-based ranking
            return chunks.sort_values(
                "max_importance_score", ascending=False
            ).head(top_k).to_dict('records')
    
    # Calculate cosine similarity
    similarities = _cosine_similarity(query_embeddings, chunk_embeddings)[0]
    
    # Get indices of top-k most similar chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Create results
    results = []
    for idx in top_indices:
        chunk_data = chunks.iloc[idx].to_dict()
        chunk_data["semantic_similarity"] = float(similarities[idx])
        results.append(chunk_data)
    
    return results

@lru_cache(maxsize=config.INSIGHTS_CACHE_SIZE)
def cached_topic_filter(topic_id: str, keyword_hash: str) -> str:
    """Cache key for topic filtering results."""
    return f"{topic_id}_{keyword_hash}"

def get_keyword_hash(keywords: List[str]) -> str:
    """Generate hash for keywords list."""
    keywords_str = "|".join(sorted(keywords))
    return hashlib.md5(keywords_str.encode()).hexdigest()[:8]

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between matrices."""
    # Normalize vectors
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    
    # Calculate similarity
    return np.dot(a_norm, b_norm.T)

class RetrievalEngine:
    """High-level retrieval engine with caching and filtering."""
    
    def __init__(self, cache_size: int = None):
        """Initialize retrieval engine."""
        self.cache_size = cache_size or config.INSIGHTS_CACHE_SIZE
        self.topic_registry = TopicRegistry()
        self.embedding_generator = None
        
        # Load chunk index
        self.df, self.embeddings = load_chunk_index()
        
        # Initialize embedding generator lazily
        try:
            self.embedding_generator = EmbeddingGenerator()
        except Exception as e:
            print(f"Warning: Could not initialize embedding generator: {e}")
    
    def search_by_topic(self, 
                      topic_id: str, 
                      keywords: List[str] = None,
                      top_k: int = 100,
                      use_semantic: bool = True) -> List[Dict[str, Any]]:
        """Search chunks by topic with optional semantic ranking."""
        # Check cache first
        keyword_hash = get_keyword_hash(keywords or [])
        cache_key = cached_topic_filter(topic_id, keyword_hash)
        
        if cache_key in _query_cache:
            cached_results = _query_cache[cache_key]
            if len(cached_results) >= top_k:
                return cached_results[:top_k]
        
        # Filter by topic
        filtered_chunks = filter_chunks_by_topic(topic_id, self.df, keywords)
        
        # Apply semantic ranking if requested and available
        if use_semantic and self.embedding_generator:
            topic_prompt = self.topic_registry.get_topic_embedding_prompt(topic_id)
            results = semantic_rank(
                filtered_chunks, 
                query_text=topic_prompt,
                top_k=top_k
            )
        else:
            # Use importance-based ranking
            results = filtered_chunks.sort_values(
                "max_importance_score", ascending=False
            ).head(top_k).to_dict('records')
        
        # Cache results
        _query_cache[cache_key] = results
        
        return results
    
    def search_by_query(self, 
                      query_text: str,
                      top_k: int = 100) -> List[Dict[str, Any]]:
        """Search chunks by query text with semantic ranking."""
        if not self.embedding_generator:
            # Fallback to keyword search
            return self._keyword_search(query_text, top_k)
        
        # Use semantic ranking
        return semantic_rank(
            self.df,
            query_text=query_text,
            top_k=top_k
        )
    
    def _keyword_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback keyword-based search."""
        query_lower = query_text.lower()
        
        # Calculate simple keyword match score
        self.df["keyword_score"] = self.df["text"].str.lower().apply(
            lambda text: sum(1 for word in query_lower.split() if word in text)
        )
        
        # Sort by keyword score and importance
        results = self.df.sort_values(
            ["keyword_score", "max_importance_score"], 
            ascending=[False, False]
        ).head(top_k)
        
        # Clean up temporary column
        results = results.drop("keyword_score", axis=1)
        
        return results.to_dict('records')
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID."""
        mask = self.df["chunk_id"] == chunk_id
        chunk = self.df[mask]
        
        if len(chunk) == 0:
            return None
        
        return chunk.iloc[0].to_dict()
    
    def get_chunks_by_transcript(self, transcript_id: str) -> List[Dict[str, Any]]:
        """Get all chunks from a specific transcript."""
        mask = self.df["transcript_id"] == transcript_id
        chunks = self.df[mask].sort_values("chunk_index")
        
        return chunks.to_dict('records')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the chunk index."""
        stats = {
            "total_chunks": len(self.df),
            "total_transcripts": self.df["transcript_id"].nunique(),
            "avg_chunk_length": self.df["character_count"].mean(),
            "avg_statements_per_chunk": self.df["statement_count"].mean(),
            "topics_available": list(self.topic_registry.get_all_topics().keys())
        }
        
        # Topic distribution
        if "matching_topics" in self.df.columns:
            topic_counts = {}
            for topics in self.df["matching_topics"].dropna():
                for topic in topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
            stats["topic_distribution"] = topic_counts
        
        # Embedding availability
        if "embedding" in self.df.columns:
            stats["chunks_with_embeddings"] = self.df["embedding"].notna().sum()
            stats["embedding_coverage"] = stats["chunks_with_embeddings"] / stats["total_chunks"]
        
        return stats
    
    def clear_cache(self):
        """Clear the in-memory query cache."""
        global _query_cache
        _query_cache.clear()
        # Clear LRU cache
        cached_topic_filter.cache_clear()

# Convenience functions for backward compatibility
def search_chunks(topic_id: str = None,
                query_text: str = None,
                keywords: List[str] = None,
                top_k: int = 100,
                use_semantic: bool = True) -> List[Dict[str, Any]]:
    """Convenience function for searching chunks."""
    engine = RetrievalEngine()
    
    if topic_id:
        return engine.search_by_topic(
            topic_id=topic_id,
            keywords=keywords,
            top_k=top_k,
            use_semantic=use_semantic
        )
    elif query_text:
        return engine.search_by_query(query_text=query_text, top_k=top_k)
    else:
        raise ValueError("Either topic_id or query_text must be provided")