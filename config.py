import os
from pathlib import Path

# Project configuration
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, TRANSCRIPTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# spaCy configuration
SPACY_MODEL = "en_core_web_sm"  # Can be changed to en_core_web_md or en_core_web_lg

# Hugging Face configuration
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # For embeddings
HF_REPO_NAME = "transcript-processor"  # Will be created on HF
HF_PRIVATE = False  # Set to True for private repo

# Processing configuration
BATCH_SIZE = 32
MAX_TRANSCRIPT_LENGTH = 100000  # Maximum characters per transcript
MIN_STATEMENT_LENGTH = 10  # Minimum characters for a statement to be considered important
MAX_STATEMENTS_PER_TRANSCRIPT = 50  # Limit to prevent too many statements

# Output configuration
OUTPUT_FORMAT = "jsonl"  # Can be "jsonl", "csv", or "parquet"
INCLUDE_EMBEDDINGS = True
INCLUDE_TIMESTAMPS = True  # If transcripts contain timestamp information

# Text cleaning configuration
REMOVE_FILLER_WORDS = True
FILLER_WORDS = [
    "um", "uh", "er", "ah", "like", "you know", "I mean", 
    "sort of", "kind of", "actually", "basically", "literally"
]

NORMALIZE_WHITESPACE = True
REMOVE_REPETITIONS = True
MIN_SENTIMENT_SCORE = 0.1  # Minimum sentiment score for important statements