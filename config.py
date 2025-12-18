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

# Prescriptive Insights configuration
PROCESSED_TRANSCRIPTS_DIR = PROJECT_DIR / "processed_transcripts"
INSIGHTS_CHUNK_DIR = PROJECT_DIR / "insights_chunks"
INSIGHTS_CHUNK_MAX_CHARS = 1000  # Maximum characters per chunk
INSIGHTS_CHUNK_WINDOW_SIZE = 4  # Number of statements per chunk
INSIGHTS_CHUNK_OVERLAP = 1  # Overlap between chunks
INSIGHTS_CACHE_SIZE = 100  # Maximum number of cached queries

# Ollama LLM configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

# Prescriptive Insights persona configuration
INSIGHTS_DEFAULT_PERSONA = "Act as a board-certified metabolic health physician, balancing scientific precision with compassionate coaching. Maintain patient empathy while citing transcript chunk IDs to justify each recommendation."

# Create directories if they don't exist
for dir_path in [INSIGHTS_CHUNK_DIR]:
    dir_path.mkdir(exist_ok=True)