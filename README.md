# Transcript Processing Pipeline

A powerful pipeline for processing transcript files using spaCy for text cleaning and statement extraction, combined with Hugging Face models for generating embeddings. This tool is designed to handle large volumes of transcripts (500+) and extract meaningful insights from raw conversation data.

## Features

- **Text Cleaning**: Removes timestamps, speaker labels, filler words, and normalizes whitespace
- **Statement Extraction**: Identifies important statements using NLP techniques
- **Entity Recognition**: Extracts named entities from transcripts
- **Embedding Generation**: Creates semantic embeddings using Hugging Face models
- **Batch Processing**: Efficiently handles large numbers of transcripts
- **Multiple Output Formats**: Supports JSONL, JSON, CSV, and Parquet formats
- **Hugging Face Integration**: Upload processed data to Hugging Face Hub
- **Sentiment Analysis**: Optional sentiment analysis for extracted statements
- **Importance Scoring**: Ranks statements by importance using multiple factors

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd transcript-processor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Quick Start

1. Create sample transcripts for testing:
```bash
python main.py --sample
```

2. Process the sample transcripts:
```bash
python main.py
```

3. View the results in the `output/` directory.

## Usage

### Basic Usage

Process all transcripts in the default directory (`data/transcripts/`):
```bash
python main.py
```

### Advanced Usage

Process transcripts from a specific directory:
```bash
python main.py --input-dir ./my_transcripts --output-file ./results.jsonl
```

Specify output format:
```bash
python main.py --format csv
```

Skip embedding generation (faster processing):
```bash
python main.py --no-embeddings
```

Upload results to Hugging Face:
```bash
python main.py --upload-to-hf --hf-token YOUR_HF_TOKEN --hf-repo your-username/transcript-dataset
```

Test embedding generation:
```bash
python main.py --test-embedding
```

## Configuration

The pipeline can be configured by modifying the [`config.py`](config.py:1) file:

### Text Processing Settings
- `SPACY_MODEL`: spaCy model to use (en_core_web_sm, en_core_web_md, en_core_web_lg)
- `REMOVE_FILLER_WORDS`: Whether to remove filler words like "um", "uh", etc.
- `NORMALIZE_WHITESPACE`: Normalize whitespace in cleaned text
- `MIN_STATEMENT_LENGTH`: Minimum length for a statement to be considered important

### Embedding Settings
- `HF_MODEL_NAME`: Hugging Face model for embeddings (default: sentence-transformers/all-MiniLM-L6-v2)
- `BATCH_SIZE`: Batch size for embedding generation
- `INCLUDE_EMBEDDINGS`: Whether to generate embeddings

### Output Settings
- `OUTPUT_FORMAT`: Output format (jsonl, json, csv, parquet)
- `MAX_STATEMENTS_PER_TRANSCRIPT`: Maximum number of statements to extract per transcript

## Input Formats

The pipeline supports various transcript formats:

- Plain text files (`.txt`)
- Markdown files (`.md`)
- Transcript files (`.transcript`)
- WebVTT files (`.vtt`)

Transcripts should contain conversation text with optional timestamps and speaker labels. The pipeline automatically detects and removes common timestamp and speaker formats.

## Output Structure

### JSONL Format
Each line contains a JSON object with the following structure:

```json
{
  "transcript_id": "sample1",
  "source_file": "data/transcripts/sample1.txt",
  "original_length": 1234,
  "cleaned_length": 987,
  "cleaned_text": "Cleaned transcript text...",
  "statements": [
    {
      "text": "Important statement from the transcript",
      "start_char": 100,
      "end_char": 150,
      "length": 50,
      "tokens": 12,
      "entities": [
        {
          "text": "Entity Name",
          "label": "PERSON",
          "start": 5,
          "end": 15
        }
      ],
      "sentiment": {
        "polarity": 0.5,
        "subjectivity": 0.8
      },
      "importance_score": 0.85,
      "embedding": [0.1, 0.2, 0.3, ...]
    }
  ],
  "statement_count": 5,
  "entity_count": 12,
  "transcript_embedding": [0.1, 0.2, 0.3, ...],
  "embedding_dim": 384
}
```

### CSV/Parquet Format
Flattened structure with top statements as separate columns:
- `transcript_id`: Unique identifier
- `cleaned_text`: Processed text
- `statement_1`, `statement_2`, etc.: Top statements
- `statement_1_importance`, etc.: Importance scores
- `has_embedding`: Whether embeddings were generated

## Processing Pipeline

1. **Text Cleaning**: Removes timestamps, speaker labels, filler words, and normalizes text
2. **Statement Extraction**: Identifies individual sentences and calculates importance scores
3. **Entity Recognition**: Extracts named entities (people, organizations, locations, etc.)
4. **Embedding Generation**: Creates semantic embeddings for full transcripts and statements
5. **Output Generation**: Saves results in specified format
6. **Optional Upload**: Uploads to Hugging Face Hub if requested

## Importance Scoring

Statements are ranked based on:
- Length (medium-length sentences preferred)
- Presence of named entities
- Grammatical structure (verbs + nouns)
- Question/exclamation markers
- Modal verbs and adjectives
- Sentiment analysis (if available)

## Hugging Face Integration

To upload processed data to Hugging Face:

1. Get your Hugging Face API token from https://huggingface.co/settings/tokens
2. Run the pipeline with the `--upload-to-hf` flag:
```bash
python main.py --upload-to-hf --hf-token YOUR_TOKEN --hf-repo your-username/dataset-name
```

The dataset will be available at `https://huggingface.co/datasets/your-username/dataset-name`

## Performance Considerations

- **Memory Usage**: Large transcripts are truncated to `MAX_TRANSCRIPT_LENGTH` (default: 100,000 characters)
- **Processing Speed**: Embedding generation is the most resource-intensive step
- **GPU Support**: Automatically uses GPU if available for embedding generation
- **Batch Processing**: Processes transcripts in batches to optimize memory usage

## Troubleshooting

### Common Issues

1. **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
2. **Out of memory errors**: Reduce `BATCH_SIZE` in config or use `--no-embeddings`
3. **Slow processing**: Consider using a smaller spaCy model or skip embeddings
4. **Hugging Face upload fails**: Check your token and repository permissions

### Performance Tips

- Use `en_core_web_sm` for faster processing
- Skip embeddings if not needed (`--no-embeddings`)
- Process smaller batches if memory is limited
- Use GPU for embedding generation if available

## Example Use Cases

- **Meeting Analysis**: Extract key decisions and action items from meeting transcripts
- **Interview Processing**: Identify important responses and themes in interview data
- **Customer Service**: Extract common issues and resolutions from support calls
- **Research**: Analyze qualitative data from focus groups or interviews
- **Content Analysis**: Process podcasts, webinars, or video transcripts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [spaCy](https://spacy.io/) for advanced NLP processing
- [Hugging Face](https://huggingface.co/) for transformer models and datasets
- [Sentence Transformers](https://www.sbert.net/) for sentence embeddings