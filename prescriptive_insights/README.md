# Prescriptive Insights Package

This package provides utilities for chunking, indexing, and retrieving insights from processed transcript data.

## Features

- **ChunkBuilder**: Creates rolling window chunks from transcript statements with configurable parameters
- **TopicRegistry**: Defines topic categories with keywords and embedding prompts
- **RetrievalEngine**: Provides filtering and semantic ranking utilities with caching
- **LLMClient**: Wrapper for Ollama HTTP API with health checks and error handling
- **InsightsOrchestrator**: Generates prescriptive insights with medical authority + patient empathy persona

## Quick Start

### Building Chunks

```bash
# Build chunks from processed transcripts
python -m prescriptive_insights.chunk_builder --rebuild

# With custom parameters
python -m prescriptive_insights.chunk_builder --window-size 6 --overlap 2 --max-chars 1500
```

### Using Retrieval

```python
from prescriptive_insights import RetrievalEngine, search_chunks

# Initialize retrieval engine
engine = RetrievalEngine()

# Search by topic
results = engine.search_by_topic('supplements', top_k=10)

# Use convenience function
results = search_chunks(topic_id='weight_management', top_k=5)
```

### Using the LLM Client

```python
from prescriptive_insights import LLMClient, create_llm_client, test_connection

# Test connection to Ollama
if test_connection():
    print("Ollama is running!")
    
    # Create client with default settings
    client = create_llm_client()
    
    # Generate text
    response = client.generate(
        prompt="Explain the benefits of keto diet for metabolic health",
        system_prompt="Act as a metabolic health physician."
    )
    print(response)
    
    # Check health and available models
    health = client.health_check()
    print(f"Available models: {health['models']}")
```

### Using the Insights Orchestrator

```python
from prescriptive_insights import (
    InsightsOrchestrator, InsightsRequest,
    create_insights_orchestrator
)

# Create orchestrator with default components
orchestrator = create_insights_orchestrator()

# Create request for insights
request = InsightsRequest(
    topics=["metabolic_health", "weight_management"],
    persona_toggle=True,
    chunk_limit=10,
    include_supplements=True,
    patient_context="45-year-old female with insulin resistance",
    custom_persona_additions="Focus on hormonal balance for women."
)

# Generate prescriptive insights
response = orchestrator.generate_insights(request)

# Access results
print(f"Generated {len(response.sections)} sections")
print(f"Total citations: {response.total_citations}")

for section in response.sections:
    print(f"\n=== {section.title} ===")
    print(section.content)
    print(f"Citations: {section.citations}")
```

## Available Topics

- `start_plan`: Getting started with keto, initial steps
- `supplements`: Nutritional supplements, vitamins, minerals
- `holistic_view`: Overall health perspective, long-term effects
- `metabolic_health`: Metabolism, insulin resistance, blood sugar
- `weight_management`: Weight loss, body composition, BMI
- `intermittent_fasting`: Fasting protocols, time-restricted eating
- `keto_foods`: Keto-friendly foods, recipes, meal planning
- `exercise_fitness`: Physical activity, fitness on keto

## Configuration

The package uses configuration from `config.py`:

- `INSIGHTS_CHUNK_DIR`: Output directory for chunks
- `INSIGHTS_CHUNK_MAX_CHARS`: Maximum characters per chunk (default: 1000)
- `INSIGHTS_CHUNK_WINDOW_SIZE`: Number of statements per chunk (default: 4)
- `INSIGHTS_CHUNK_OVERLAP`: Overlap between chunks (default: 1)
- `INSIGHTS_CACHE_SIZE`: Maximum cached queries (default: 100)

### Ollama Configuration

- `OLLAMA_HOST`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: Model to use for generation (default: `qwen3:latest`)
- `OLLAMA_TIMEOUT`: Request timeout in seconds (default: 120)
- `OLLAMA_MAX_RETRIES`: Maximum retry attempts (default: 3)
- `INSIGHTS_DEFAULT_PERSONA`: Default medical authority + empathy persona

## File Structure

```
insights_chunks/
├── master_chunks.parquet          # Master dataframe of all chunks
├── insights_manifest.json         # Manifest with file hashes/mtimes
└── {transcript_id}_chunks.jsonl # Per-transcript chunk files
```

## API Reference

### ChunkBuilder

```python
builder = ChunkBuilder(
    window_size=4,      # Statements per chunk
    overlap=1,          # Overlap between chunks
    max_chars=1000,     # Max characters per chunk
    use_embeddings=True   # Generate embeddings
)

stats = builder.build_chunks(force_rebuild=False)
```

### TopicRegistry

```python
from prescriptive_insights import TopicRegistry, TOPICS

registry = TopicRegistry()
topic = registry.get_topic('supplements')
matching = registry.find_matching_topics(text, entities)
```

### RetrievalEngine

```python
engine = RetrievalEngine()

# Topic-based search
results = engine.search_by_topic(
    topic_id='supplements',
    keywords=['vitamin', 'mineral'],  # Optional custom keywords
    top_k=100,
    use_semantic=True
)

# Query-based search
results = engine.search_by_query(
    query_text='how to start keto diet',
    top_k=50
)

# Get statistics
stats = engine.get_statistics()
```

### LLMClient

```python
from prescriptive_insights import LLMClient

client = LLMClient(
    host="http://localhost:11434",
    model="qwen3:latest",
    timeout=120,
    max_retries=3
)

# Health check
health = client.health_check()
if health["status"] == "healthy":
    print("LLM service is available")

# Generate text
response = client.generate(
    prompt="Your medical question here",
    system_prompt="Medical persona prompt",
    options={
        "temperature": 0.7,
        "max_tokens": 2048
    }
)

# Stream response
for chunk in client.generate(prompt="Question", stream=True):
    print(chunk, end="")
```

### InsightsOrchestrator

```python
from prescriptive_insights import InsightsOrchestrator, InsightsRequest

# Initialize orchestrator
orchestrator = InsightsOrchestrator(
    retrieval_engine=RetrievalEngine(),
    llm_client=LLMClient()
)

# Create detailed request
request = InsightsRequest(
    topics=["metabolic_health", "supplements"],
    persona_toggle=True,
    keyword_overrides={
        "metabolic_health": ["insulin", "glucose", "metabolism"]
    },
    chunk_limit=15,
    include_supplements=True,
    patient_context="Patient with pre-diabetes",
    custom_persona_additions="Emphasize lifestyle changes over medication",
    use_semantic_search=True,
    temperature=0.6,
    max_tokens=3000
)

# Generate insights
response = orchestrator.generate_insights(request)

# Access structured response
for section in response.sections:
    print(f"Section: {section.title}")
    print(f"Content: {section.content}")
    print(f"Citations: {section.citations}")
    print(f"Referenced chunks: {section.chunk_ids}")

# Access metadata
print(f"Model used: {response.generation_metadata['model']}")
print(f"Chunks retrieved: {response.generation_metadata['chunks_retrieved']}")
```

### Custom Persona Building

```python
from prescriptive_insights import build_persona_prompt

# Build custom persona
custom_persona = build_persona_prompt(
    "Focus specifically on athletes and performance optimization."
)

print(custom_persona)
```

## Running the Orchestrator CLI

```bash
# Generate insights for specific topics
python -c "
from prescriptive_insights import create_insights_orchestrator, InsightsRequest

orchestrator = create_insights_orchestrator()
request = InsightsRequest(
    topics=['metabolic_health', 'weight_management'],
    patient_context='Patient looking to start keto diet'
)

response = orchestrator.generate_insights(request)
for section in response.sections:
    print(f'=== {section.title} ===')
    print(section.content)
"
```

## Testing

```bash
# Run unit tests
python -m pytest prescriptive_insights/tests/

# Run specific test file
python -m unittest prescriptive_insights.tests.test_orchestrator
```

## Using the Streamlit Tab

The Prescriptive Insights package is integrated into the main application's Tab 4: Prescriptive Insights. This provides a user-friendly interface for generating personalized health plans without needing to use the CLI directly.

### Prerequisites

1. **Process Transcripts First**: Use Tab 1 (YouTube Extractor) and Tab 2 (Transcript Processor) to download and process transcripts before using the Prescriptive Insights tab.

2. **Build Chunk Index**: The first time you use the feature or after adding new transcripts, click "Build Chunk Index" in the tab. This creates the semantic search index used for retrieving relevant content.

3. **Start Ollama**: Ensure Ollama is running with `ollama serve` in a separate terminal before using the tab.

### Tab Workflow

1. **Index Status**: The tab shows the current chunk index statistics, including total chunks, source transcripts, and available topics.

2. **Topic Selection**: Choose from predefined health topics like metabolic health, weight management, supplements, etc.

3. **Persona Customization**: Toggle the medical/empathetic persona and adjust chunk limits for evidence retrieval.

4. **Advanced Options**:
   - Override default keywords for each topic
   - Provide patient context for personalized recommendations
   - Add custom persona instructions

5. **Generate Plan**: Click "Generate Prescriptive Plan" to:
   - Retrieve relevant chunks using semantic search
   - Generate evidence-based recommendations using the LLM
   - Include citations to source transcripts

6. **Review Results**:
   - View the generated prescriptive plan organized by topic
   - Download as Markdown
   - Examine evidence chunks and citations

### CLI Integration

The tab uses the same backend components as the CLI:

- Chunk building: `python -m prescriptive_insights.chunk_builder --rebuild`
- LLM connection: Tests Ollama availability automatically
- Retrieval engine: Uses semantic search and topic filtering
- Insights generation: Same orchestrator logic as CLI examples

For programmatic access or batch processing, refer to the CLI examples in the sections above.