# YouTube ID & Transcript Processor

An integrated Streamlit application that combines YouTube video extraction with advanced spaCy-based transcript processing and analysis.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage Instructions](#usage-instructions)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before installing this application, ensure you have the following:

- **Python 3.8 or higher** (recommended: Python 3.9+)
- **pip** (Python package installer)
- **Git** (for cloning the repository)
- **Ollama** (required for Prescriptive Insights feature only)
  - Installation instructions: <https://ollama.ai/download>
  - After installation, run `ollama serve` to start the service
- **Sufficient disk space** for storing transcripts and processed data
- **Stable internet connection** for YouTube API access

### System Requirements

- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 2GB free space for transcripts and processed data
- **OS**: macOS 10.14+, Windows 10+, or Linux (Ubuntu 18.04+)

## Installation

### Option 1: Quick Start (macOS)

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/youtube-transcript-processor.git
   cd youtube-transcript-processor
   ```

2. Simply double-click the `run_app.command` file in Finder
3. The script will automatically:
   - Create a virtual environment if needed
   - Install all required dependencies
   - Launch the Streamlit application in your browser

### Option 2: Manual Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/youtube-transcript-processor.git
   cd youtube-transcript-processor
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download required spaCy model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. For Prescriptive Insights (Tab 4), ensure Ollama is running:

   ```bash
   ollama serve
   ```

   And build the chunk index if needed:

   ```bash
   python -m prescriptive_insights.chunk_builder --rebuild
   ```

6. Run the application:

   ```bash
   streamlit run integrated_app.py
   ```

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
- Four-tab interface for different functionalities
- Progress tracking for all operations
- Comprehensive error handling and reporting

### Prescriptive Insights

- AI-powered health plan generation using processed transcript data
- Topic-based retrieval of relevant content with semantic search
- Customizable persona with medical authority and patient empathy
- Evidence-based recommendations with citations to source transcripts
- Integration with Ollama for local LLM inference

## Quick Start

### For macOS Users

1. Simply double-click the `run_app.command` file in Finder
2. The script will automatically:
   - Create a virtual environment if needed
   - Install all required dependencies
   - Launch the Streamlit application in your browser

> **Tip:** For long batch downloads, see [Prevent macOS sleep during long downloads](#prevent-macos-sleep-during-long-downloads) to avoid interrupted transfers.

### For Windows/Linux Users

1. Clone and navigate to the project directory
2. Run the installation commands from the [Installation](#installation) section
3. Start the application:

   ```bash
   streamlit run integrated_app.py
   ```

## Examples

### Example 1: Processing a Single YouTube Video

```python
# Using the batch download script for a single video
python3 batch_download.py --video-urls "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --output-dir subtitles
```

### Example 2: Processing a Playlist

```python
# Using the batch download script for a playlist
python3 batch_download.py --video-urls "https://www.youtube.com/playlist?list=PLexample" --output-dir subtitles --min-delay 2.0 --max-delay 5.0
```

### Example 3: Using the Prescriptive Insights API

```python
from prescriptive_insights import create_insights_orchestrator, InsightsRequest

# Create orchestrator
orchestrator = create_insights_orchestrator()

# Generate insights for specific topics
request = InsightsRequest(
    topics=["metabolic_health", "weight_management"],
    patient_context="45-year-old female with insulin resistance"
)

response = orchestrator.generate_insights(request)
for section in response.sections:
    print(f"=== {section.title} ===")
    print(section.content)
```

### Batch Download (Command Line)

For downloading large numbers of videos (500+) with resume capability:

1. Use the batch download script:

   ```bash
   python3 batch_download.py --input-file video_urls.txt --output-dir subtitles
   ```

2. Or pass URLs directly:

   ```bash
   python3 batch_download.py --video-urls "url1,url2,url3" --output-dir subtitles
   ```

3. Configure delay and verbose output:

   ```bash
   python3 batch_download.py --input-file video_urls.txt --output-dir subtitles \
       --min-delay 2.0 --max-delay 5.0 --verbose-yt
   ```

**Available flags:**

- `--min-delay SECONDS`: Minimum delay between downloads (default: 1.0)
- `--max-delay SECONDS`: Maximum delay between downloads (default: 3.0)
- `--verbose-yt`: Show full yt-dlp stdout/stderr output for debugging

The batch download script includes:

- Resume capability (skips already downloaded files)
- Progress tracking with batch processing
- Error handling and reporting with detailed logs
- Configurable batch size and delays
- Automatic SRT to TXT conversion

#### Prevent macOS sleep during long downloads

macOS may put your Mac to sleep during long-running downloads, which can interrupt transfers. Resumable downloads (like those from `batch_download.py`) can pick up where they left off, but non-resumable transfers will fail completely if sleep occurs mid-download.

**GUI approach:**

- **System Settings → Lock Screen**: Set "Turn display off" to a longer interval or "Never"
- **System Settings → Battery → Options** (laptops): Enable "Prevent automatic sleeping when the display is off"

**CLI approach using `caffeinate`:**

Wrap your download command with `caffeinate` to prevent sleep while the process runs:

```bash
caffeinate -w $$ &
python3 batch_download.py --input-file video_urls.txt --output-dir subtitles
```

Or tie `caffeinate` directly to the download process:

```bash
caffeinate -w $(python3 batch_download.py --input-file video_urls.txt --output-dir subtitles & echo $!)
```

**Using `pmset` for persistent changes:**

```bash
# Check current sleep settings
pmset -g

# Disable sleep entirely (requires sudo)
sudo pmset -a sleep 0 disksleep 0

# Restore defaults when finished
sudo pmset -a sleep 1 disksleep 10
```

**Verify sleep is prevented:**

```bash
pmset -g assertions
```

Look for `PreventUserIdleSystemSleep` or `PreventSystemSleep` assertions to confirm sleep is blocked.

**Quick checklist for reliable long downloads:**

1. Plug into AC power (battery mode has more aggressive sleep)
2. Use a resumable download client like `batch_download.py`
3. Wrap the download with `caffeinate -w $$` or use GUI settings
4. Verify assertions with `pmset -g assertions`
5. Restore normal sleep settings when finished

**Caveats:**

- **Lid closure**: Closing the MacBook lid forces sleep regardless of `caffeinate` or `pmset` settings (unless using clamshell mode with external display)
- **App Nap**: macOS may throttle background apps; keep Terminal in the foreground or disable App Nap for Terminal in Get Info
- **Restore settings**: Remember to revert any `pmset` changes after downloads complete to preserve battery life

## Usage Instructions

### Tab 1: YouTube Extractor

1. **Enter YouTube URL**: Input the URL of a YouTube video, playlist, or channel in the text field
2. **Fetch Videos**: Click "Fetch Videos" button to extract metadata
3. **Filter by Date**: Use the date range picker to filter videos by upload date (displayed in UK format DD/MM/YYYY)
4. **Select Videos**:
   - Use the checkboxes in the table to manually select/deselect videos
   - Use "Select All" or "Deselect All" buttons for bulk operations
5. **Configure Download Settings**:
   - Adjust **Min Delay** and **Max Delay** sliders to control the wait time between downloads (helps avoid rate limiting)
6. **Download Subtitles**: Click "Download Subtitles as TXT" to get transcript files for selected videos
7. **Review Failed Downloads**: After downloading, a detailed log is available for download containing yt-dlp stdout/stderr for each failed video, useful for diagnosing issues

### Tab 2: Transcript Processor

1. **Download Subtitles First**: Use Tab 1 to download subtitles to the 'subtitles' directory
2. **Process All Transcripts**: Click "Run Transcript Pipeline" to process all discovered transcripts with the spaCy pipeline
3. **View Results**: Check processing statistics and download processed files

### Tab 3: Analysis Results

1. **Select Processed File**: Choose a processed transcript to analyze
2. **View Analysis**: Examine extracted statements, entities, and sentiment
3. **Download Results**: Export processed data in JSON format

### Tab 4: Prescriptive Insights

1. **Build Chunk Index**: Click "Build Chunk Index" if this is your first time using the feature or after adding new transcripts
2. **Ensure Ollama is Running**: Start Ollama with `ollama serve` in a separate terminal
3. **Select Topics**: Choose health topics of interest from the available categories
4. **Customize Persona**: Toggle medical/empathetic persona and adjust chunk limits
5. **Generate Plan**: Click "Generate Prescriptive Plan" to create evidence-based recommendations
6. **Review Evidence**: View citations and evidence chunks supporting the recommendations

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

### Starting the Application

- If the application doesn't start after double-clicking `run_app.command`, try running it from Terminal:

  ```bash
  cd /path/to/project
  streamlit run integrated_app.py
  ```

### Stopping Streamlit Applications

- To stop a running Streamlit app, use `Ctrl+C` in the terminal where it's running
- If Streamlit is stuck or running in the background, kill it with:

  ```bash
  pkill -f streamlit
  ```

- You can also kill a specific app:

  ```bash
  pkill -f "streamlit run integrated_app.py"
  ```

### Common Issues

- Make sure you have an active internet connection for fetching YouTube metadata
- If you encounter rate limiting issues, the application will automatically retry with delays
- For large playlists, extraction process may take some time due to YouTube's rate limiting
- If spaCy model download fails, the application will attempt to download it automatically
- For transcript processing errors, check that subtitle files are properly formatted as plain text

### Ollama Connectivity Issues

If you see "Ollama service is not running at <http://localhost:11434>":

1. **Check if Ollama is running**:

   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Verify Ollama process**:

   ```bash
   ps aux | grep ollama
   ```

3. **Restart Ollama if needed**:

   ```bash
   pkill ollama
   ollama serve
   ```

4. **Check for port conflicts**:

   ```bash
   lsof -i :11434
   ```

5. **Test model availability**:

   ```bash
   ollama list
   ```

If Ollama appears to be running but connections fail, try restarting both Ollama and your application.

### Interpreting the Detailed Download Log

When subtitle downloads fail, you can download a detailed log file containing diagnostics for each failed video. Here's how to interpret it:

1. **Exit Code**: A non-zero exit code indicates yt-dlp encountered an error
   - Exit code `1`: General error (video unavailable, no subtitles, etc.)
   - Exit code `2`: Network or connection issues

2. **YouTube Error Messages**: Look for lines containing `ERROR:` in the stderr output
   - "Video unavailable": The video may be private, deleted, or region-locked
   - "No subtitles": The video doesn't have English subtitles or auto-generated captions
   - "Sign in to confirm your age": Age-restricted content requires authentication

3. **Rate Limiting**: Messages like "HTTP Error 429" indicate YouTube is throttling requests
   - Increase `--min-delay` and `--max-delay` values
   - Wait before retrying

4. **Authentication Issues**: If you see "Sign in" or "cookies" errors:
   - Consider supplying cookies using yt-dlp's `--cookies` option
   - Export cookies from your browser using a browser extension
   - Example: `yt-dlp --cookies cookies.txt <url>`

## Development

### Project Structure

```text
project/
├── integrated_app.py          # Main integrated application
├── text_processor.py          # spaCy text processing
├── transcript_processor.py     # Main transcript processing pipeline
├── embedding_generator.py     # Hugging Face embeddings
├── config.py                 # Configuration settings
├── requirements.txt           # All dependencies
├── README.md                # This file
├── subtitles/               # Downloaded subtitle files
├── processed_transcripts/     # Processed transcript data
├── insights_chunks/          # Chunked data for Prescriptive Insights
└── prescriptive_insights/    # Prescriptive Insights package
    ├── __init__.py           # Package initialization
    ├── chunk_builder.py      # Build chunks from processed transcripts
    ├── retrieval.py          # Topic-based and semantic retrieval
    ├── llm_client.py         # Ollama LLM client
    ├── insights_orchestrator.py  # Main insights generation logic
    ├── topic_registry.py     # Topic definitions and keywords
    └── README.md            # Package documentation
```

### Extending the Application

The application is designed to be modular and extensible:

- Add new processing features to `text_processor.py`
- Extend analysis capabilities in `transcript_processor.py`
- Add new embedding models in `embedding_generator.py`
- Modify the UI in `integrated_app.py` for new tabs or features

## Contributing

We welcome contributions to improve this project! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Submit a pull request

### Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to new functions and classes
- Keep code DRY (Don't Repeat Yourself)

### Testing

Run tests before submitting:

```bash
python -m pytest prescriptive_insights/tests/
```

### Bug Reports

When reporting bugs, please include:

- Operating system and Python version
- Steps to reproduce the issue
- Expected vs. actual behavior
- Any error messages or tracebacks

## FAQ

### Q: Can I use this without Ollama?

A: Yes, all features except Prescriptive Insights (Tab 4) work without Ollama.

### Q: How do I increase the download speed?

A: You can adjust the delay settings in the batch download script, but be careful not to set them too low as YouTube may rate-limit your requests.

### Q: Can I process videos in languages other than English?

A: Currently, the NLP processing is optimized for English transcripts. Support for other languages is planned for future releases.

### Q: How much storage space do I need?

A: This depends on the number of videos you process. As a rough estimate:

- 1 hour of transcript text: ~50KB
- Processed transcript with embeddings: ~500KB
- Plan for at least 1GB for processing 1000+ videos

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This project incorporates several open-source components:

- **spaCy**: MIT License
- **Hugging Face Transformers**: Apache 2.0 License
- **yt-dlp**: Unlicense
- **Streamlit**: Apache 2.0 License

Please refer to the respective project licenses for specific component licensing information.
