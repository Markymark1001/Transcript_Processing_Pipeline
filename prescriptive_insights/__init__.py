"""
Prescriptive Insights Package

This package provides utilities for chunking, indexing, and retrieving insights
from processed transcript data. It includes:

- ChunkBuilder: Creates rolling window chunks from transcript statements
- TopicRegistry: Defines topic categories and their associated keywords
- Retrieval: Provides filtering and semantic ranking utilities
"""

from .chunk_builder import ChunkBuilder
from .topic_registry import TopicRegistry, TOPICS
from .retrieval import (load_chunk_index, filter_chunks_by_topic, semantic_rank,
                        RetrievalEngine, search_chunks)
from .llm_client import LLMClient, create_llm_client, test_connection
from .insights_orchestrator import (
    InsightsRequest, SectionOutput, InsightsResponse,
    InsightsOrchestrator, build_persona_prompt, create_insights_orchestrator
)

__all__ = [
    "ChunkBuilder",
    "TopicRegistry",
    "TOPICS",
    "load_chunk_index",
    "filter_chunks_by_topic",
    "semantic_rank",
    "RetrievalEngine",
    "search_chunks",
    "LLMClient",
    "create_llm_client",
    "test_connection",
    "InsightsRequest",
    "SectionOutput",
    "InsightsResponse",
    "InsightsOrchestrator",
    "build_persona_prompt",
    "create_insights_orchestrator"
]

__version__ = "0.1.0"