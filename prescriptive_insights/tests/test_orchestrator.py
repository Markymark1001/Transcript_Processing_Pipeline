"""
Unit tests for Insights Orchestrator

Tests the orchestrator functionality with mocked LLM client and retrieval engine.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insights_orchestrator import (
    InsightsRequest, SectionOutput, InsightsResponse,
    InsightsOrchestrator, build_persona_prompt,
    TransportError, ContentError, CitationError
)
from llm_client import LLMClient, LLMClientError

class TestInsightsOrchestrator(unittest.TestCase):
    """Test cases for InsightsOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock retrieval engine
        self.mock_retrieval_engine = Mock()
        
        # Mock LLM client
        self.mock_llm_client = Mock(spec=LLMClient)
        self.mock_llm_client.health_check.return_value = {
            "status": "healthy",
            "host": "http://localhost:11434",
            "models": ["qwen3:latest"],
            "current_model_available": True
        }
        self.mock_llm_client.model = "qwen3:latest"
        
        # Sample chunks for testing
        self.sample_chunks = [
            {
                "chunk_id": "test_transcript_chunk_0",
                "text": "Keto diet can help improve insulin sensitivity and reduce inflammation.",
                "max_importance_score": 0.8,
                "transcript_id": "test_transcript",
                "matching_topics": ["ketosis", "metabolic_health"]
            },
            {
                "chunk_id": "test_transcript_chunk_1", 
                "text": "Intermittent fasting promotes autophagy and metabolic flexibility.",
                "max_importance_score": 0.7,
                "transcript_id": "test_transcript",
                "matching_topics": ["fasting", "metabolic_health"]
            }
        ]
        
        # Initialize orchestrator
        self.orchestrator = InsightsOrchestrator(
            retrieval_engine=self.mock_retrieval_engine,
            llm_client=self.mock_llm_client
        )
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        self.assertEqual(self.orchestrator.retrieval_engine, self.mock_retrieval_engine)
        self.assertEqual(self.orchestrator.llm_client, self.mock_llm_client)
        
        # Test with unhealthy LLM
        self.mock_llm_client.health_check.return_value = {"status": "unhealthy"}
        with self.assertRaises(TransportError):
            InsightsOrchestrator(self.mock_retrieval_engine, self.mock_llm_client)
    
    def test_build_persona_prompt(self):
        """Test persona prompt building."""
        base_prompt = build_persona_prompt()
        self.assertIn("board-certified metabolic health physician", base_prompt)
        self.assertIn("Evidence Summary", base_prompt)
        self.assertIn("Baseline Plan", base_prompt)
        
        # Test with custom additions
        custom_prompt = build_persona_prompt("Focus on diabetes management.")
        self.assertIn("Focus on diabetes management", custom_prompt)
    
    def test_retrieve_relevant_chunks(self):
        """Test chunk retrieval functionality."""
        # Set up mock
        self.mock_retrieval_engine.search_by_topic.return_value = self.sample_chunks
        
        # Create request
        request = InsightsRequest(
            topics=["ketosis", "fasting"],
            chunk_limit=10
        )
        
        # Test retrieval
        chunks = self.orchestrator._retrieve_relevant_chunks(request)
        
        # Verify
        self.assertEqual(len(chunks), 2)
        self.mock_retrieval_engine.search_by_topic.assert_called()
        
        # Test with keyword overrides
        request.keyword_overrides = {"ketosis": ["custom", "keywords"]}
        self.orchestrator._retrieve_relevant_chunks(request)
        
        # Verify keyword override was used
        call_args = self.mock_retrieval_engine.search_by_topic.call_args
        self.assertEqual(call_args[1]['keywords'], ["custom", "keywords"])
    
    def test_generate_chunk_summaries(self):
        """Test chunk summary generation."""
        # Set up mock
        self.mock_llm_client.generate.return_value = "Summary of [test_transcript_chunk_0] content."
        
        # Create request
        request = InsightsRequest()
        
        # Test summary generation
        summaries = self.orchestrator._generate_chunk_summaries(self.sample_chunks, request)
        
        # Verify
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["chunk_id"], "test_transcript_chunk_0")
        self.assertIn("Summary of", summaries[0]["summary"])
        
        # Test error handling
        self.mock_llm_client.generate.side_effect = LLMClientError("API error")
        summaries = self.orchestrator._generate_chunk_summaries(self.sample_chunks, request)
        
        # Should fallback to original text
        self.assertEqual(len(summaries), 2)
        self.assertIn("Original content", summaries[0]["summary"])
    
    def test_generate_prescriptive_synthesis(self):
        """Test prescriptive synthesis generation."""
        # Set up chunk summaries
        chunk_summaries = [
            {
                "chunk_id": "test_transcript_chunk_0",
                "summary": "Summary of keto benefits [test_transcript_chunk_0].",
                "original_chunk": self.sample_chunks[0]
            },
            {
                "chunk_id": "test_transcript_chunk_1",
                "summary": "Summary of fasting benefits [test_transcript_chunk_1].",
                "original_chunk": self.sample_chunks[1]
            }
        ]
        
        # Mock LLM response
        mock_response = """
        Evidence Summary:
        The evidence suggests keto and fasting have metabolic benefits [test_transcript_chunk_0] [test_transcript_chunk_1].
        
        Baseline Plan:
        Start with keto diet and intermittent fasting [test_transcript_chunk_0].
        
        Supplement Guidance:
        Consider electrolytes and MCT oil [test_transcript_chunk_1].
        """
        
        self.mock_llm_client.generate.return_value = mock_response
        
        # Create request
        request = InsightsRequest(
            topics=["ketosis", "fasting"],
            persona_toggle=True,
            include_supplements=True
        )
        
        # Test synthesis
        response = self.orchestrator._generate_prescriptive_synthesis(chunk_summaries, request)
        
        # Verify
        self.assertIsInstance(response, InsightsResponse)
        self.assertEqual(len(response.sections), 3)  # Evidence, Baseline, Supplements
        self.assertEqual(response.total_citations, 3)
        self.assertIn("test_transcript_chunk_0", response.chunks_used)
        
        # Verify LLM was called with correct parameters
        self.mock_llm_client.generate.assert_called_once()
        call_args = self.mock_llm_client.generate.call_args
        self.assertIn("Evidence Summary", call_args[1]['prompt'])
        self.assertIn("board-certified metabolic health physician", call_args[1]['system_prompt'])
    
    def test_parse_response_sections(self):
        """Test response parsing into sections."""
        response_text = """
        Evidence Summary:
        This is the evidence summary [chunk_1].
        
        Baseline Plan:
        This is the baseline plan [chunk_2].
        
        Supplement Guidance:
        This is the supplement guidance [chunk_3].
        """
        
        sections = self.orchestrator._parse_response_sections(response_text)
        
        # Verify
        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0].title, "Evidence Summary")
        self.assertEqual(sections[1].title, "Baseline Plan")
        self.assertEqual(sections[2].title, "Supplement Guidance")
        
        # Verify citations
        self.assertEqual(sections[0].citations, ["chunk_1"])
        self.assertEqual(sections[1].citations, ["chunk_2"])
        self.assertEqual(sections[2].citations, ["chunk_3"])
    
    def test_extract_citations(self):
        """Test citation extraction."""
        text = "This is a test with [chunk_1] and [another_chunk_2] citations."
        citations = self.orchestrator._extract_citations(text)
        
        self.assertEqual(citations, ["chunk_1", "another_chunk_2"])
    
    def test_validate_citations(self):
        """Test citation validation."""
        # Create response with invalid citations
        response = InsightsResponse(
            sections=[
                SectionOutput(
                    title="Test Section",
                    content="Content with [valid_chunk] and [invalid_chunk].",
                    citations=["valid_chunk", "invalid_chunk"],
                    chunk_ids=["valid_chunk", "invalid_chunk"]
                )
            ]
        )
        
        # Validate with only valid_chunk in source
        source_chunks = [{"chunk_id": "valid_chunk"}]
        
        self.orchestrator._validate_citations(response, source_chunks)
        
        # Verify invalid citation was removed
        self.assertEqual(response.sections[0].citations, ["valid_chunk"])
        self.assertEqual(response.sections[0].chunk_ids, ["valid_chunk"])
        self.assertEqual(response.total_citations, 1)
    
    @patch('insights_orchestrator.logger')
    def test_generate_insights_integration(self, mock_logger):
        """Test full insights generation integration."""
        # Set up mocks
        self.mock_retrieval_engine.search_by_topic.return_value = self.sample_chunks
        self.mock_llm_client.generate.return_value = "Test response [test_transcript_chunk_0]."
        
        # Create request
        request = InsightsRequest(
            topics=["ketosis"],
            chunk_limit=5,
            persona_toggle=True
        )
        
        # Generate insights
        response = self.orchestrator.generate_insights(request)
        
        # Verify
        self.assertIsInstance(response, InsightsResponse)
        self.assertIn("test_transcript_chunk_0", response.chunks_used)
        self.assertEqual(response.generation_metadata["topics_requested"], ["ketosis"])
        self.assertTrue(response.generation_metadata["persona_enabled"])
    
    def test_transport_error_handling(self):
        """Test handling of transport errors."""
        # Mock LLM client to raise connection error
        self.mock_llm_client.generate.side_effect = LLMClientError("Connection failed")
        
        # Create request
        request = InsightsRequest(topics=["ketosis"])
        
        # Set up retrieval to return chunks
        self.mock_retrieval_engine.search_by_topic.return_value = self.sample_chunks
        
        # Test error handling
        with self.assertRaises(TransportError):
            self.orchestrator.generate_insights(request)
    
    def test_content_error_handling(self):
        """Test handling of content generation errors."""
        # Mock LLM client to raise generation error
        self.mock_llm_client.generate.side_effect = LLMClientError("Generation failed")
        
        # Create request
        request = InsightsRequest(topics=["ketosis"])
        
        # Set up retrieval to return chunks
        self.mock_retrieval_engine.search_by_topic.return_value = self.sample_chunks
        
        # Test error handling
        with self.assertRaises(ContentError):
            self.orchestrator.generate_insights(request)
    
    def test_no_chunks_found_error(self):
        """Test error when no relevant chunks are found."""
        # Mock retrieval to return empty list
        self.mock_retrieval_engine.search_by_topic.return_value = []
        
        # Create request
        request = InsightsRequest(topics=["nonexistent_topic"])
        
        # Test error handling
        with self.assertRaises(ContentError):
            self.orchestrator.generate_insights(request)

class TestInsightsRequest(unittest.TestCase):
    """Test cases for InsightsRequest dataclass."""
    
    def test_default_values(self):
        """Test default values for InsightsRequest."""
        request = InsightsRequest()
        
        self.assertEqual(request.topics, [])
        self.assertTrue(request.persona_toggle)
        self.assertEqual(request.chunk_limit, 10)
        self.assertTrue(request.include_supplements)
        self.assertEqual(request.patient_context, "")
        self.assertEqual(request.custom_persona_additions, "")
        self.assertTrue(request.use_semantic_search)
        self.assertEqual(request.temperature, 0.7)
        self.assertEqual(request.max_tokens, 2048)
    
    def test_custom_values(self):
        """Test custom values for InsightsRequest."""
        request = InsightsRequest(
            topics=["ketosis", "fasting"],
            persona_toggle=False,
            chunk_limit=20,
            include_supplements=False,
            patient_context="Diabetic patient",
            custom_persona_additions="Focus on diabetes",
            temperature=0.5,
            max_tokens=1024
        )
        
        self.assertEqual(request.topics, ["ketosis", "fasting"])
        self.assertFalse(request.persona_toggle)
        self.assertEqual(request.chunk_limit, 20)
        self.assertFalse(request.include_supplements)
        self.assertEqual(request.patient_context, "Diabetic patient")
        self.assertEqual(request.custom_persona_additions, "Focus on diabetes")
        self.assertEqual(request.temperature, 0.5)
        self.assertEqual(request.max_tokens, 1024)

class TestSectionOutput(unittest.TestCase):
    """Test cases for SectionOutput dataclass."""
    
    def test_section_output_creation(self):
        """Test SectionOutput creation."""
        section = SectionOutput(
            title="Test Section",
            content="Test content with [chunk_1] citation.",
            citations=["chunk_1"],
            chunk_ids=["chunk_1"]
        )
        
        self.assertEqual(section.title, "Test Section")
        self.assertEqual(section.content, "Test content with [chunk_1] citation.")
        self.assertEqual(section.citations, ["chunk_1"])
        self.assertEqual(section.chunk_ids, ["chunk_1"])

class TestInsightsResponse(unittest.TestCase):
    """Test cases for InsightsResponse dataclass."""
    
    def test_insights_response_creation(self):
        """Test InsightsResponse creation."""
        sections = [
            SectionOutput(
                title="Section 1",
                content="Content 1 [chunk_1].",
                citations=["chunk_1"],
                chunk_ids=["chunk_1"]
            ),
            SectionOutput(
                title="Section 2", 
                content="Content 2 [chunk_2].",
                citations=["chunk_2"],
                chunk_ids=["chunk_2"]
            )
        ]
        
        response = InsightsResponse(
            sections=sections,
            summary="Test summary",
            total_citations=2,
            chunks_used=["chunk_1", "chunk_2"],
            generation_metadata={"model": "test_model"}
        )
        
        self.assertEqual(len(response.sections), 2)
        self.assertEqual(response.summary, "Test summary")
        self.assertEqual(response.total_citations, 2)
        self.assertEqual(response.chunks_used, ["chunk_1", "chunk_2"])
        self.assertEqual(response.generation_metadata["model"], "test_model")

if __name__ == "__main__":
    unittest.main()