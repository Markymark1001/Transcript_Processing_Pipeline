# How to Process Your Transcript Files

## Option 1: Copy Files to Accessible Location (Recommended)

1. **Create a new folder** where you have permissions:
   ```
   mkdir -p ~/Desktop/transcripts_to_process
   ```

2. **Copy your transcript files** from the restricted location:
   ```
   cp -r ~/Documents/subtitles-DrBoz/* ~/Desktop/transcripts_to_process/
   ```
   
   Or manually copy the files using Finder.

3. **Run the processing pipeline**:
   ```bash
   cd /Users/markmacmini/Documents/Kilo-Code
   python3 batch_processor.py --input-dir ~/Desktop/transcripts_to_process --output-file output/your_transcripts.jsonl
   ```

## Option 2: Use the Simple Script

1. **Copy files to the data folder**:
   ```
   cp -r ~/Documents/subtitles-DrBoz/* data/transcripts/
   ```

2. **Run the main script**:
   ```bash
   python3 main.py
   ```

## Option 3: Process from Any Location

If you copy the files to any location, just specify the path:

```bash
python3 batch_processor.py --input-dir /path/to/your/transcripts --output-file output/results.jsonl
```

## What the Pipeline Does

- ✅ **Cleans text**: Removes timestamps, speaker labels, filler words
- ✅ **Extracts statements**: Identifies important sentences with importance scores
- ✅ **Finds entities**: Detects people, organizations, dates, locations
- ✅ **Generates embeddings**: Creates semantic vectors for search and analysis
- ✅ **Saves results**: Outputs to JSONL format with all extracted data

## Expected Output

For each transcript, you'll get:
- Cleaned text (with filler words removed)
- Important statements ranked by importance
- Named entities (people, places, organizations)
- Semantic embeddings for similarity search
- Processing statistics and metadata

## Example Command

```bash
# After copying files to ~/Desktop/transcripts_to_process/
python3 batch_processor.py --input-dir ~/Desktop/transcripts_to_process --output-file output/drboz_results.jsonl
```

This will process all your transcript files and save the results to `output/drboz_results.jsonl`.