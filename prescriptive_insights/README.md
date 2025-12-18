# Prescriptive Insights Package

This package provides utilities for chunking, indexing, and retrieving insights
from processed transcript data.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Features](#features)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)

## Prerequisites

Before using this package, ensure you have:

- **Python 3.8+** installed
- **Ollama** installed and running (required for LLM functionality)
  - Installation: <https://ollama.ai/download>
  - Start service: `ollama serve`
- **Processed transcript data** in the expected format
- **Sufficient RAM** (8GB+ recommended for large datasets)

## Installation

The package is included as part of the main project. To use it independently:

```bash
# Clone the repository
git clone https://github.com/yourusername/youtube-transcript-processor.git
cd youtube-transcript-processor

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Features

- **ChunkBuilder**: Creates rolling window chunks from transcript statements with
  configurable parameters
- **TopicRegistry**: Defines topic categories with keywords and embedding prompts
- **RetrievalEngine**: Provides filtering and semantic ranking utilities with caching
- **LLMClient**: Wrapper for Ollama HTTP API with health checks and error handling
- **InsightsOrchestrator**: Generates prescriptive insights with medical authority
  - patient empathy persona

## Quick Start

### Building Chunks

```bash
# Build chunks from processed transcripts
python -m prescriptive_insights.chunk_builder --rebuild

# With custom parameters
python -m prescriptive_insights.chunk_builder --window-size 6 --overlap 2 \
  --max-chars 1500
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

### Chunk Configuration

- `INSIGHTS_CHUNK_DIR`: Output directory for chunks (default: `insights_chunks/`)
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

### Performance Tuning

For optimal performance with large datasets:

```python
# In config.py or your script
INSIGHTS_CHUNK_WINDOW_SIZE = 6  # Larger windows for more context
INSIGHTS_CHUNK_MAX_CHARS = 1500  # Larger chunks for more detail
INSIGHTS_CACHE_SIZE = 500  # Larger cache for frequently accessed queries
```

## File Structure

```text
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

## Examples

### Example 1: Custom Topic Analysis

```python
from prescriptive_insights import RetrievalEngine, TopicRegistry

# Initialize components
engine = RetrievalEngine()
registry = TopicRegistry()

# Find topics in custom text
text = "The patient should consider intermittent fasting and keto diet for weight
loss"
matching_topics = registry.find_matching_topics(text, [])

# Search for relevant chunks
for topic_id in matching_topics:
    chunks = engine.search_by_topic(topic_id, top_k=5)
    print(f"Found {len(chunks)} chunks for topic {topic_id}")
```

### Example 2: Batch Processing Multiple Patients

```python
from prescriptive_insights import create_insights_orchestrator, InsightsRequest

orchestrator = create_insights_orchestrator()

# Define patient contexts
patients = [
    {"id": "patient1", "context": "45-year-old male with type 2 diabetes"},
    {"id": "patient2", "context": "30-year-old female looking to lose weight"},
    {"id": "patient3", "context": "60-year-old with metabolic syndrome"}
]

# Generate insights for each patient
for patient in patients:
    request = InsightsRequest(
        topics=["metabolic_health", "weight_management"],
        patient_context=patient["context"],
        chunk_limit=15
    )
    
    response = orchestrator.generate_insights(request)
    
    # Save results
    with open(f"insights_{patient['id']}.md", "w") as f:
        f.write(f"# Health Plan for {patient['id']}\n\n")
        for section in response.sections:
            f.write(f"## {section.title}\n\n")
            f.write(f"{section.content}\n\n")
```

### Example 3: Custom Persona Configuration

```python
from prescriptive_insights import build_persona_prompt, LLMClient

# Create a specialized persona
custom_persona = build_persona_prompt(
    "Focus specifically on athletes and performance optimization."
)

# Use with LLM client
client = LLMClient()
response = client.generate(
    prompt="Create a keto diet plan for an endurance athlete",
    system_prompt=custom_persona
)
print(response)
```

## Troubleshooting

### Common Issues

#### Ollama Connection Errors

**Problem**: "Ollama service is not running"

**Solutions**:

1. Check if Ollama is installed:

   ```bash
   which ollama
   ```

2. Start the Ollama service:

   ```bash
   ollama serve
   ```

3. Verify it's running:

   ```bash
   curl http://localhost:11434/api/tags
   ```

#### Memory Issues with Large Datasets

**Problem**: Out of memory errors when processing large transcript collections

**Solutions**:

1. Reduce chunk size:

   ```python
   builder = ChunkBuilder(max_chars=500)  # Smaller chunks
   ```

2. Process in batches:

   ```python
   # Process subsets of transcripts at a time
   for batch in transcript_batches:
       builder.process_batch(batch)
   ```

3. Increase system RAM or use a machine with more memory

#### Slow Performance

**Problem**: Retrieval or generation is taking too long

**Solutions**:

1. Enable caching:

   ```python
   engine = RetrievalEngine(cache_size=1000)  # Larger cache
   ```

2. Reduce semantic search for initial filtering:

   ```python
   results = engine.search_by_topic(
       topic_id='supplements',
       use_semantic=False,  # Faster keyword-only search
       top_k=50
   )
   ```

3. Use a smaller model for faster generation:

   ```python
   client = LLMClient(model="llama2:7b")  # Smaller model
   ```

#### Model Not Available

**Problem**: "Model not found" error from Ollama

**Solutions**:

1. List available models:

   ```bash
   ollama list
   ```

2. Pull the required model:

   ```bash
   ollama pull qwen3:latest
   ```

3. Update the model in your configuration:

   ```python
   client = LLMClient(model="your-available-model")
   ```

### Debug Mode

Enable verbose logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all operations will show detailed logs
orchestrator = create_insights_orchestrator()
```

## Testing

```bash
# Run all unit tests
python -m pytest prescriptive_insights/tests/

# Run with coverage
python -m pytest prescriptive_insights/tests/ --cov=prescriptive_insights

# Run specific test file
python -m unittest prescriptive_insights.tests.test_orchestrator

# Run with verbose output
python -m pytest prescriptive_insights/tests/ -v
```

### Test Coverage

The test suite covers:

- Chunk building and validation
- Topic registry functionality
- Retrieval engine operations
- LLM client connectivity
- Insights orchestration workflow

To add new tests, place them in the `prescriptive_insights/tests/` directory
and follow the naming convention `test_*.py`.

## Using the Streamlit Tab

The Prescriptive Insights package is integrated into the main application's
Tab 4: Prescriptive Insights. This provides a user-friendly interface for
generating personalized health plans without needing to use the CLI directly.

### Tab Prerequisites

1. **Process Transcripts First**: Use Tab 1 (YouTube Extractor) and Tab 2
   (Transcript Processor) to download and process transcripts before using the
   Prescriptive Insights tab.

2. **Build Chunk Index**: The first time you use the feature or after adding new
   transcripts, click "Build Chunk Index" in the tab. This creates the semantic
   search index used for retrieving relevant content.

3. **Start Ollama**: Ensure Ollama is running with `ollama serve` in a separate
   terminal before using the tab.

### Tab Workflow

1. **Index Status**: The tab shows the current chunk index statistics,
   including total chunks, source transcripts, and available topics.

2. **Topic Selection**: Choose from predefined health topics like metabolic
   health, weight management, supplements, etc.

3. **Persona Customization**: Toggle the medical/empathetic persona and adjust
   chunk limits for evidence retrieval.

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

For programmatic access or batch processing, refer to the CLI examples in the
sections above.
