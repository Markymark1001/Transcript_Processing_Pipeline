"""
Insights Orchestrator for Prescriptive Content Generation

Coordinates retrieval of relevant chunks and generates prescriptive insights
using LLM synthesis with medical authority and patient empathy personas.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import config
from .retrieval import RetrievalEngine
from .llm_client import LLMClient, LLMClientError, LLMGenerationError

# Configure logging
logger = logging.getLogger(__name__)

# Custom exceptions
class InsightsOrchestratorError(Exception):
    """Base exception for insights orchestrator errors."""
    pass

class TransportError(InsightsOrchestratorError):
    """Exception for transport/connectivity errors."""
    pass

class ContentError(InsightsOrchestratorError):
    """Exception for content generation errors."""
    pass

class CitationError(InsightsOrchestratorError):
    """Exception for citation validation errors."""
    pass

@dataclass
class InsightsRequest:
    """Request object for insights generation."""
    topics: List[str] = field(default_factory=list)
    persona_toggle: bool = True
    keyword_overrides: Dict[str, List[str]] = field(default_factory=dict)
    chunk_limit: int = 10
    include_supplements: bool = True
    patient_context: str = ""
    custom_persona_additions: str = ""
    use_semantic_search: bool = True
    temperature: float = 0.7
    max_tokens: int = 2048

@dataclass
class SectionOutput:
    """Output for a single section of insights."""
    title: str
    content: str
    citations: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)

@dataclass
class InsightsResponse:
    """Complete response for insights generation."""
    sections: List[SectionOutput] = field(default_factory=list)
    summary: str = ""
    total_citations: int = 0
    chunks_used: List[str] = field(default_factory=list)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)

def build_persona_prompt(custom_additions: str = "") -> str:
    """
    Build the system prompt with medical authority and patient empathy persona.
    
    Args:
        custom_additions: Additional persona text to append
        
    Returns:
        Complete persona system prompt
    """
    base_persona = config.INSIGHTS_DEFAULT_PERSONA
    
    # Add structured instructions
    structured_instructions = """
    
    Structure your response with the following sections:
    1. Evidence Summary - Briefly summarize the key evidence from the provided chunks
    2. Baseline Plan - Core recommendations for metabolic health
    3. Supplement Guidance - Evidence-based supplement recommendations (if applicable)
    4. Holistic Routines - Lifestyle and routine recommendations
    
    For each recommendation, cite the specific chunk IDs using the format [chunk_id].
    Maintain a balance between scientific precision and compassionate coaching.
    """
    
    persona_prompt = base_persona + structured_instructions
    
    if custom_additions.strip():
        persona_prompt += f"\n\n{custom_additions.strip()}"
    
    return persona_prompt

class InsightsOrchestrator:
    """Orchestrates the generation of prescriptive insights from retrieved chunks."""
    
    def __init__(self, 
                 retrieval_engine: RetrievalEngine,
                 llm_client: LLMClient):
        """Initialize orchestrator with required components."""
        self.retrieval_engine = retrieval_engine
        self.llm_client = llm_client
        
        # Check LLM health
        health = self.llm_client.health_check()
        if health["status"] != "healthy":
            raise TransportError(f"LLM service unhealthy: {health}")
        
        logger.info(f"Initialized InsightsOrchestrator with model: {self.llm_client.model}")
    
    def generate_insights(self, request: InsightsRequest) -> InsightsResponse:
        """
        Generate prescriptive insights based on the request.
        
        Args:
            request: InsightsRequest with all parameters
            
        Returns:
            InsightsResponse with generated content and metadata
        """
        try:
            # Step 1: Retrieve relevant chunks
            relevant_chunks = self._retrieve_relevant_chunks(request)
            
            if not relevant_chunks:
                raise ContentError("No relevant chunks found for the given topics")
            
            # Step 2: Generate evidence summaries per chunk
            chunk_summaries = self._generate_chunk_summaries(relevant_chunks, request)
            
            # Step 3: Generate prescriptive synthesis
            insights_response = self._generate_prescriptive_synthesis(
                chunk_summaries, request
            )
            
            # Step 4: Validate citations
            self._validate_citations(insights_response, relevant_chunks)
            
            # Add metadata
            insights_response.generation_metadata.update({
                "model": self.llm_client.model,
                "chunks_retrieved": len(relevant_chunks),
                "topics_requested": request.topics,
                "persona_enabled": request.persona_toggle
            })
            
            return insights_response
            
        except LLMClientError as e:
            raise TransportError(f"LLM transport error: {e}")
        except LLMGenerationError as e:
            raise ContentError(f"Content generation error: {e}")
        except Exception as e:
            raise InsightsOrchestratorError(f"Unexpected error: {e}")
    
    def _retrieve_relevant_chunks(self, request: InsightsRequest) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks based on topics and keywords."""
        all_chunks = []
        
        for topic in request.topics:
            # Get keywords for this topic (with overrides if provided)
            keywords = request.keyword_overrides.get(topic)
            
            # Search by topic
            topic_chunks = self.retrieval_engine.search_by_topic(
                topic_id=topic,
                keywords=keywords,
                top_k=request.chunk_limit,
                use_semantic=request.use_semantic_search
            )
            
            all_chunks.extend(topic_chunks)
        
        # Remove duplicates by chunk_id
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk["chunk_id"] not in seen_ids:
                seen_ids.add(chunk["chunk_id"])
                unique_chunks.append(chunk)
        
        # Sort by importance and limit
        unique_chunks.sort(
            key=lambda x: x.get("max_importance_score", 0), 
            reverse=True
        )
        
        return unique_chunks[:request.chunk_limit]
    
    def _generate_chunk_summaries(self, 
                                 chunks: List[Dict[str, Any]], 
                                 request: InsightsRequest) -> List[Dict[str, Any]]:
        """Generate evidence summaries for each chunk."""
        summaries = []
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["text"]
            
            # Build summary prompt
            summary_prompt = f"""
            Summarize the key medical insights from this transcript chunk:
            
            Chunk ID: {chunk_id}
            Content: {chunk_text}
            
            Focus on actionable metabolic health advice. Keep it concise but comprehensive.
            Include the chunk ID in your response as [{chunk_id}].
            """
            
            try:
                summary = self.llm_client.generate(
                    prompt=summary_prompt,
                    options={"temperature": 0.3, "max_tokens": 300}
                )
                
                summaries.append({
                    "chunk_id": chunk_id,
                    "summary": summary,
                    "original_chunk": chunk
                })
                
            except Exception as e:
                logger.warning(f"Failed to summarize chunk {chunk_id}: {e}")
                # Use original text as fallback
                summaries.append({
                    "chunk_id": chunk_id,
                    "summary": f"Original content [{chunk_id}]: {chunk_text[:200]}...",
                    "original_chunk": chunk
                })
        
        return summaries
    
    def _generate_prescriptive_synthesis(self, 
                                       chunk_summaries: List[Dict[str, Any]], 
                                       request: InsightsRequest) -> InsightsResponse:
        """Generate the final prescriptive synthesis."""
        # Combine all summaries
        combined_summaries = "\n\n".join([
            f"Chunk {s['chunk_id']}: {s['summary']}" 
            for s in chunk_summaries
        ])
        
        # Build system prompt
        system_prompt = ""
        if request.persona_toggle:
            system_prompt = build_persona_prompt(request.custom_persona_additions)
        
        # Build main prompt
        main_prompt = f"""
        Based on the following evidence summaries from medical transcripts, generate comprehensive prescriptive insights for metabolic health:
        
        PATIENT CONTEXT: {request.patient_context or "General metabolic health inquiry"}
        
        EVIDENCE SUMMARIES:
        {combined_summaries}
        
        Please provide:
        1. Evidence Summary - Brief synthesis of key findings
        2. Baseline Plan - Core metabolic health recommendations
        3. Supplement Guidance - Evidence-based supplement recommendations
        {"4. Holistic Routines - Lifestyle and routine recommendations" if request.include_supplements else ""}
        
        CRITICAL: Every recommendation MUST cite the source chunk ID using [chunk_id] format.
        Maintain medical authority while showing patient empathy.
        """
        
        # Generate response
        try:
            response_text = self.llm_client.generate(
                prompt=main_prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
            )
            
            # Parse response into sections
            sections = self._parse_response_sections(response_text)
            
            # Extract citations and chunk IDs
            all_citations = self._extract_citations(response_text)
            chunk_ids = list(set([cid for cid in all_citations if cid.startswith(tuple([s["chunk_id"] for s in chunk_summaries]))]))
            
            return InsightsResponse(
                sections=sections,
                summary=response_text,
                total_citations=len(all_citations),
                chunks_used=chunk_ids
            )
            
        except Exception as e:
            raise ContentError(f"Failed to generate prescriptive synthesis: {e}")
    
    def _parse_response_sections(self, response_text: str) -> List[SectionOutput]:
        """Parse the response text into structured sections."""
        sections = []
        
        # Define section patterns
        section_patterns = [
            (r'(?i)evidence summary[:\s]*\n', "Evidence Summary"),
            (r'(?i)baseline plan[:\s]*\n', "Baseline Plan"),
            (r'(?i)supplement guidance[:\s]*\n', "Supplement Guidance"),
            (r'(?i)holistic routines[:\s]*\n', "Holistic Routines")
        ]
        
        # Split response by sections
        current_section = None
        current_content = []
        
        lines = response_text.split('\n')
        
        for line in lines:
            matched_section = None
            
            for pattern, section_title in section_patterns:
                if re.match(pattern, line):
                    # Save previous section if exists
                    if current_section:
                        content = '\n'.join(current_content).strip()
                        if content:
                            citations = self._extract_citations(content)
                            sections.append(SectionOutput(
                                title=current_section,
                                content=content,
                                citations=citations,
                                chunk_ids=[c for c in citations if c.startswith(tuple(['chunk']))]
                            ))
                    
                    # Start new section
                    current_section = section_title
                    current_content = [line]
                    matched_section = section_title
                    break
            
            if not matched_section and current_section:
                current_content.append(line)
        
        # Add final section
        if current_section:
            content = '\n'.join(current_content).strip()
            if content:
                citations = self._extract_citations(content)
                sections.append(SectionOutput(
                    title=current_section,
                    content=content,
                    citations=citations,
                    chunk_ids=[c for c in citations if c.startswith(tuple(['chunk']))]
                ))
        
        # If no sections found, treat entire response as one section
        if not sections:
            citations = self._extract_citations(response_text)
            sections.append(SectionOutput(
                title="Complete Response",
                content=response_text,
                citations=citations,
                chunk_ids=[c for c in citations if c.startswith(tuple(['chunk']))]
            ))
        
        return sections
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract all chunk ID citations from text."""
        # Pattern matches [chunk_id] where chunk_id can contain letters, numbers, underscores, hyphens
        pattern = r'\[([a-zA-Z0-9_\-]+)\]'
        matches = re.findall(pattern, text)
        return matches
    
    def _validate_citations(self, 
                          response: InsightsResponse, 
                          source_chunks: List[Dict[str, Any]]):
        """Validate that all citations reference valid chunks."""
        valid_chunk_ids = {chunk["chunk_id"] for chunk in source_chunks}
        
        for section in response.sections:
            for citation in section.citations:
                if citation not in valid_chunk_ids:
                    logger.warning(f"Invalid citation found: [{citation}]")
                    # Note: We don't raise an error here to be more resilient,
                    # but in production you might want stricter validation
        
        # Filter citations to only valid ones
        for section in response.sections:
            section.citations = [c for c in section.citations if c in valid_chunk_ids]
            section.chunk_ids = [c for c in section.chunk_ids if c in valid_chunk_ids]
        
        # Update total citations count
        response.total_citations = sum(len(s.citations) for s in response.sections)

# Convenience function for quick usage
def create_insights_orchestrator(retrieval_engine: RetrievalEngine = None,
                               llm_client: LLMClient = None) -> InsightsOrchestrator:
    """Create an insights orchestrator with default components."""
    if retrieval_engine is None:
        retrieval_engine = RetrievalEngine()
    
    if llm_client is None:
        llm_client = LLMClient()
    
    return InsightsOrchestrator(retrieval_engine, llm_client)