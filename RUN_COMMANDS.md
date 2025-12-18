# Exact Commands to Run - Step by Step

## Step 1: Create directory and copy files
```bash
mkdir -p ~/Desktop/transcripts_to_process
cp -r ~/Documents/subtitles-DrBoz/* ~/Desktop/transcripts_to_process/
```

## Step 2: Go to our project directory
```bash
cd /Users/markmacmini/Documents/Kilo-Code
```

## Step 3: Process the files
```bash
python3 batch_processor.py --input-dir ~/Desktop/transcripts_to_process --output-file output/drboz_results.jsonl
```

## Alternative: Use our custom script
```bash
cd /Users/markmacmini/Documents/Kilo-Code
python3 process_your_files.py
```

## If you want to use the main script (simpler)
```bash
cd /Users/markmacmini/Documents/Kilo-Code
cp -r ~/Desktop/transcripts_to_process/* data/transcripts/
python3 main.py
```

## Step 4: Start Ollama for Prescriptive Insights
```bash
# Start Ollama server (run in separate terminal and keep running)
ollama serve

# Pull a model if needed (only need to do this once)
ollama pull qwen3:latest
```

## Step 5: Build Chunk Index for Prescriptive Insights
```bash
cd /Users/markmacmini/Documents/Kilo-Code

# Build chunk index from processed transcripts
python -m prescriptive_insights.chunk_builder --rebuild

# Or with custom parameters
python -m prescriptive_insights.chunk_builder --window-size 6 --overlap 2 --max-chars 1500
```

## Step 6: Run the Integrated App
```bash
cd /Users/markmacmini/Documents/Kilo-Code
streamlit run integrated_app.py
```

## Troubleshooting:
- Make sure you're in the Kilo-Code directory when running python commands
- Use `python3` not `python`
- Check that the files were copied correctly with `ls ~/Desktop/transcripts_to_process`
- For Prescriptive Insights: Ensure Ollama is running with `ollama serve` before using Tab 4
- If chunk index is missing, rebuild it with `python -m prescriptive_insights.chunk_builder --rebuild`