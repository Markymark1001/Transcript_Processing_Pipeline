# YouTube ID & Transcript Processor

An integrated Streamlit application that combines YouTube video extraction with advanced spaCy-based transcript processing and analysis.

## Features

### YouTube ID Extractor
- Extract video IDs from YouTube URLs (single videos, playlists, or channels)
- Filter videos by upload date range
- UK date format support (DD/MM/YYYY)
- Manually select/deselect videos using checkboxes
- Export selected video IDs in a copy-friendly format
- Automatic subtitle downloading for selected videos

### Transcript Processing Pipeline
- Advanced text processing using spaCy
- Statement extraction with importance scoring
- Named entity recognition and extraction
- Sentiment analysis using TextBlob
- Embedding generation using Hugging Face transformers
- Batch processing capabilities
- Multiple output formats (JSONL, JSON, CSV, Parquet)

### Integration Features
- Seamless workflow from YouTube extraction to transcript analysis
- Three-tab interface for different functionalities
- Progress tracking for all operations
- Comprehensive error handling and reporting

## How to Use

### Quick Start (macOS)

1. Simply double-click the `run_app.command` file in Finder
2. The script will automatically:
   - Create a virtual environment if needed
   - Install all required dependencies
   - Launch the Streamlit application in your browser

### Manual Setup

If you prefer to run manually:

1. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run integrated_app.py
   ```

## Usage Instructions

### Tab 1: YouTube Extractor
1. **Enter YouTube URL**: Input the URL of a YouTube video, playlist, or channel in the text field
2. **Fetch Videos**: Click "Fetch Videos" button to extract metadata
3. **Filter by Date**: Use the date range picker to filter videos by upload date (displayed in UK format DD/MM/YYYY)
4. **Select Videos**: 
   - Use the checkboxes in the table to manually select/deselect videos
   - Use "Select All" or "Deselect All" buttons for bulk operations
5. **Download Subtitles**: Click "Download Subtitles as TXT" to get transcript files for selected videos

### Tab 2: Transcript Processor
1. **Download Subtitles First**: Use Tab 1 to download subtitles to the 'subtitles' directory
2. **Select Transcript Files**: Choose which TXT files to process using checkboxes
3. **Process Transcripts**: Click "Process Selected Transcripts" to run the spaCy pipeline
4. **View Results**: Check processing statistics and download processed files

### Tab 3: Analysis Results
1. **Select Processed File**: Choose a processed transcript to analyze
2. **View Analysis**: Examine extracted statements, entities, and sentiment
3. **Download Results**: Export processed data in JSON format

## Requirements

- Python 3.7+
- Dependencies (automatically installed by run_app.command):
  - streamlit
  - yt-dlp
  - pandas
  - spaCy
  - transformers
  - torch
  - datasets
  - huggingface_hub
  - tqdm
  - numpy
  - scikit-learn
  - textblob
  - selenium (for subtitle downloading)
  - requests (for subtitle downloading)
  - webdriver-manager

## Technical Details

### spaCy Integration
The application uses spaCy for advanced natural language processing:
- Text cleaning and normalization
- Statement extraction with importance scoring
- Named entity recognition (persons, organizations, locations, etc.)
- Part-of-speech tagging for linguistic analysis

### Embedding Generation
- Uses Hugging Face's sentence-transformers for semantic embeddings
- Model: sentence-transformers/all-MiniLM-L6-v2
- Enables semantic similarity search and clustering

### Hybrid Extraction Approach
The YouTube extractor uses a hybrid extraction method:
1. First performs a fast flat extraction to get the video list
2. Then fetches upload dates in small batches with delays to avoid rate limiting
3. Combines data for a complete view with sorting by upload date

## Output Files

### Subtitle Files
- Location: `subtitles/` directory
- Format: Plain text (.txt)
- Naming: Based on video titles

### Processed Transcripts
- Location: `processed_transcripts/` directory
- Format: JSON (.json)
- Content: Cleaned text, statements, entities, sentiment, embeddings

## Troubleshooting

- If the application doesn't start after double-clicking `run_app.command`, try running it from Terminal:
  ```
  cd /path/to/project
  streamlit run integrated_app.py
  ```
- Make sure you have an active internet connection for fetching YouTube metadata
- If you encounter rate limiting issues, the application will automatically retry with delays
- For large playlists, extraction process may take some time due to YouTube's rate limiting
- If spaCy model download fails, the application will attempt to download it automatically
- For transcript processing errors, check that subtitle files are properly formatted as plain text

## Development

### Project Structure
```
project/
├── integrated_app.py          # Main integrated application
├── text_processor.py          # spaCy text processing
├── transcript_processor.py     # Main transcript processing pipeline
├── embedding_generator.py     # Hugging Face embeddings
├── config.py                 # Configuration settings
├── requirements.txt           # All dependencies
├── README.md                # This file
├── subtitles/               # Downloaded subtitle files
└── processed_transcripts/     # Processed transcript data
```

### Extending the Application
The application is designed to be modular and extensible:
- Add new processing features to `text_processor.py`
- Extend analysis capabilities in `transcript_processor.py`
- Add new embedding models in `embedding_generator.py`
- Modify the UI in `integrated_app.py` for new tabs or features

## License

This project combines features from both YouTube ID Extractor and Transcript Processing Pipeline. Please refer to individual project licenses for specific components.